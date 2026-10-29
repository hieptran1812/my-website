---
title: "The Fish Audio Family, Release by Release: From Fish Speech 1.0 to Drama 3"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of Fish Audio's text-to-speech models from the Bert-VITS2 era to October 2026: Fish Speech 1.0 through 1.5, Fish Agent, OpenAudio S1 and S1-mini, Fish Audio S2 Pro, the hosted S2.1 Pro and the Drama 3 MoE, with dates, specs, licenses and primary sources."
tags: ["fish-audio", "fish-speech", "text-to-speech", "speech-synthesis", "openaudio-s1", "fish-audio-s2", "drama-3", "open-weights", "model-family", "timeline", "tts"]
category: "machine-learning"
subcategory: "Text-to-Speech"
author: "Hiep Tran"
featured: false
readTime: 20
---

Most open TTS projects ship one good checkpoint and stall. Fish Audio is the exception that kept going. The team behind the `fishaudio` GitHub org started with community tooling (a diffusion singing-voice toolkit and a Bert-VITS2 fork), turned that into Fish Speech, an LLM-style text-to-speech model that went through five public versions in 2024, and then rebranded its research line twice: first as OpenAudio S1 in June 2025, then as Fish Audio S2 in March 2026. On October 13, 2026 it released Drama 3, a 35B mixture-of-experts speech model that activates 3B parameters per token. Along the way the training set grew from 150k hours to more than 10M hours, the backbone grew from a custom 24-layer transformer to a Qwen-initialized LLM, and the license changed four times.

That much churn makes the family hard to keep straight. Model names change (Fish Speech, OpenAudio, Fish Audio S, Drama), Hub repos get renamed (`fishaudio/openaudio-s1-mini` now redirects to `fishaudio/s1-mini`), and some of the best models never get open weights at all. This post is the map. It walks every release in date order, says what changed from the previous one, gives a spec table, names the license, and links a primary source for every claim. The timeline below is the mental model: a dense 2024 of small open checkpoints, then fewer, much larger releases with hosted-only models in between.

![Fish Audio release timeline from January 2023 to October 2026, open-weight releases in the top lane and hosted-only models in the bottom lane, Drama 3 highlighted as the newest open model](/imgs/blogs/fish-audio-model-family-timeline-1.webp)

Read the dates on the cards; the 2024 burst is stacked so the cards do not collide. One fact up front, because it is the question people ask most: **as of October 23, 2026, the newest open-weight Fish Audio model is Drama 3 (`fishaudio/drama-3`, released October 13, 2026), and the newest model on the hosted API is S2.1 Pro, which has no published weights.** The S2 Pro checkpoint from March 2026 remains supported in the same inference server.

A note on method and dates. I pulled the GitHub release list for `fishaudio/fish-speech`, the Hugging Face Hub API for every model under `fishaudio` (sorted by creation date, with each repo's `config.json` and file list), the README at each release tag, the arXiv reports, and the Fish Audio blog and API docs. Where a release note or blog post exists I use its date. Where only the Hub exists I write "Hub created", which is a lower bound on the public date. Parameter counts marked "about" are my own estimates from a released config and checkpoint size, and I say so each time. If you want the technique-level story of S2 rather than a catalog, read the companion deep-dive on [Fish Audio S2's Dual-AR and multi-reward RL](/blog/machine-learning/text-to-speech/fish-audio-s2-controllable-text-to-speech).

## The timeline at a glance

| Date | Release | Weights | One-line significance |
| --- | --- | --- | --- |
| 2023-01 (repo) | fish-diffusion | Code | Diffusion toolkit for TTS, singing synthesis and voice conversion |
| 2023-07 (repo) | Bert-VITS2 (fishaudio fork) | Code | VITS2 backbone with multilingual BERT; the community's pre-LLM workhorse |
| 2023-12-25 | Fish Speech v0.2.0 | Gated | First tagged release of the LLM-style architecture |
| 2024-04-30 | Fish Speech 1.0 | Open, CC-BY-NC-SA-4.0 | 150k hours, en/zh/ja; AR text-to-semantic model plus VQ-GAN |
| 2024-05-11 | Fish Speech 1.1 | Open, same | Adds a VITS decoder for better phone-level accuracy |
| 2024-07-18 | Fish Speech 1.2 (+ SFT) | Open, same | 300k hours; first Dual-AR checkpoint; Firefly GAN at 4 codebooks, 42 Hz |
| 2024-09-12 | Fish Speech 1.4 | Open, same | 700k hours, 8 languages; codec moves to 8 codebooks at 21 Hz |
| 2024-10-29 (Hub) | Fish Agent v0.1 3B | Open, same | Voice-to-voice model continued from Qwen-2.5-3B-Instruct |
| 2024-11-02 | Fish-Speech paper (arXiv 2411.01156) | n/a | Describes Dual-AR, GFSQ and Firefly-GAN |
| 2024-12-25 | Fish Speech 1.5 | Open, same | More than 1M hours, 13 languages; new 102k-token vocabulary |
| 2025-06-02 | OpenAudio S1 and S1-mini | S1 hosted; S1-mini open, same license | 4B flagship and 0.5B distilled; online RLHF with GRPO; more than 2M hours |
| 2026-03-09 | Fish Audio S2 Pro (arXiv 2603.08823) | Open, Fish Audio Research License | 10M+ hours, ~80 languages; Qwen3-4B slow AR; 10-codebook 44.1 kHz codec |
| 2026-06-23 | S2.1 Pro | Hosted only | Successor to S2 Pro on the API; 83 languages; free tier |
| 2026-10-13 | Fish Audio Drama 3 (arXiv 2610.18010) | Open, Fish Audio Community License | 35B MoE, 3B active; codebook-MTP decoder; multi-turn revision and infilling |

## Before Fish Speech: fish-diffusion and Bert-VITS2 (2023)

The org's first visible project was [fish-diffusion](https://github.com/fishaudio/fish-diffusion) (repo created January 9, 2023), which describes itself as "an easy to understand TTS / SVS / SVC framework", covering speech synthesis, singing synthesis and singing voice conversion. It mattered more as a community than as a model: it is where the people who later built Fish Speech learned what hobbyist voice cloners actually needed.

The second was the [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) fork (repo created July 29, 2023), "vits2 backbone with multilingual-bert". Bert-VITS2 became one of the most popular open voice-cloning stacks in Chinese-speaking communities in 2023; the fork sits at around 8.8k stars as I write. Its architecture is the classic one: a phoneme front end, a VITS-family acoustic model conditioned on BERT text features, and a waveform decoder trained end to end. Every Fish Speech README since v0.2.0 credits VITS2, Bert-VITS2, GPT VITS and (from 1.0) GPT-SoVITS, which tells you where the team came from. The [fish-speech repo](https://github.com/fishaudio/fish-speech) itself was created October 10, 2023, and its first tagged release, **v0.2.0 on December 25, 2023**, shipped "basic (arch) model implementation, inference acceleration, and pretrained model". The matching Hub repo, `fishaudio/speech-lm-v1`, was created December 7, 2023 and is gated, so I cannot report its specs.

## Fish Speech 1.0 (April 30, 2024)

Fish Speech 1.0 was the first release people outside the community noticed. The [v1.0.0 release](https://github.com/fishaudio/fish-speech/releases/tag/v1.0.0) points to `fishaudio/fish-speech-1`, a Hugging Face Space, and the fish.audio site, and states the models are under CC-BY-NC-SA-4.0. The [model card](https://huggingface.co/fishaudio/fish-speech-1) says it was "trained on 150k hours of English, Chinese, and Japanese audio data". The Hub repo was created April 19, 2024.

### What was new

The break from Bert-VITS2 was the move to a language-model formulation. Instead of predicting mel frames from phonemes, Fish Speech 1.0 tokenizes audio with a VQ-GAN and trains an autoregressive transformer to predict those tokens from text. The file list in the 1.0 repo makes the pieces concrete: `vq-gan-group-fsq-2x1024.pth` (a group finite-scalar-quantization tokenizer with two groups of 1,024 codes), `firefly-gan-base-generator.ckpt` (the waveform decoder), and the text-to-semantic models in two sizes, `text2semantic-pretrain-medium-2k-v1`, `text2semantic-sft-medium-v1-4k` and `text2semantic-sft-large-v1-4k`. The repo does not publish a config for these, so I cannot give parameter counts; the medium checkpoints are 0.78 GB and the large ones 2.05 GB on disk.

| Variant | Role | Checkpoint size |
| --- | --- | --- |
| text2semantic medium (pretrain, SFT) | Text to semantic tokens | 0.78 GB |
| text2semantic large (SFT) | Text to semantic tokens | 2.05 GB |
| VQ-GAN group FSQ 2x1024 | Audio tokenizer | 1.03 GB |
| Firefly GAN base | Token to waveform decoder | 0.15 GB |

License: weights CC-BY-NC-SA-4.0 (non-commercial, gated with a non-commercial checkbox), code BSD-3-Clause.

## Fish Speech 1.1 (May 11, 2024)

The [v1.1.0 release](https://github.com/fishaudio/fish-speech/releases/tag/v1.1.0) is one sentence: "we added the VITS decoder module, which provides better phone level accuracy and semantic similarity." The same `fish-speech-1` repo gained `vits_decoder_v1.1.ckpt` and v1.1 SFT checkpoints for both the medium and large text-to-semantic models. Point releases v1.1.1 (June 8) and v1.1.2 (July 2) were bug-fix and stability releases. Treat 1.1 as 1.0 with an optional, more accurate decoder, not a new model.

## Fish Speech 1.2 (July 18, 2024)

Fish Speech 1.2 is where the architecture that still defines the family first ships. The [v1.2 release](https://github.com/fishaudio/fish-speech/releases/tag/v1.2) rolls out "both 1.2 pretrain and SFT model, and also support auto-reranking for stable generation". The [model card](https://huggingface.co/fishaudio/fish-speech-1.2) doubles the data to "300k hours of English, Chinese, and Japanese audio".

### What changed versus 1.1

The released `config.json` names the model type `dual_ar`. A slow transformer with 24 layers (dim 1024, 16 heads, grouped-query attention with 2 KV heads) runs along time, and a fast transformer with 4 layers fills in the remaining codebooks for each frame. The tokenizer is now a single file, `firefly-gan-vq-fsq-4x1024-42hz-generator.pth`: 4 codebooks of 1,024 codes at about 42 frames per second. The VITS decoder is gone. By my count from the config (24 slow layers, 4 fast layers, a 32,000-token vocabulary), the model is about 0.5B parameters, which matches the 0.98 GB `model.pth` stored in 16-bit.

| Variant | Hub repo (created) | Slow AR | Fast AR | Codec |
| --- | --- | --- | --- | --- |
| 1.2 pretrain | `fish-speech-1.2` (2024-07-02) | 24 layers, dim 1024 | 4 layers | Firefly FSQ 4x1024, 42 Hz |
| 1.2 SFT | `fish-speech-1.2-sft` (2024-07-18) | same | same | same |

License: weights and code both CC-BY-NC-SA-4.0 per the README at the v1.2 tag. v1.2.1 (September 10) was "the final stable release before 1.4".

## Fish Speech 1.4 (September 12, 2024)

The [v1.4.0 release](https://github.com/fishaudio/fish-speech/releases/tag/v1.4.0) calls it "a leading TTS model trained on 700k hours of audio data in multiple languages" and gives the split: English and Chinese about 300k hours each, and German, Japanese, French, Spanish, Korean and Arabic about 20k hours each. That is the first jump from three languages to eight.

The backbone config is identical to 1.2 (24 slow layers, 4 fast layers, dim 1024). What changed is the codec: `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`, which halves the frame rate to about 21 Hz and doubles the codebooks to 8. Tokens per second of audio stay the same, but the slow model now takes half as many steps per second, so long utterances fit more comfortably in the 4,096-token context. The figure below shows that trade on one second of audio.

![Grid of acoustic tokens for one second of audio: Fish Speech 1.2 with 4 codebooks at about 42 frames, Fish Speech 1.4 and 1.5 with 8 codebooks at about 21 frames, and S2 Pro and Drama 3 with 10 codebooks at about 21.5 frames](/imgs/blogs/fish-audio-model-family-timeline-4.webp)

Notice that 1.2 and 1.4 cost the same 168 tokens per second; only the shape of the grid changed. The slow model predicts the first row, the fast model fills each column.

### The Fish-Speech paper (November 2, 2024)

The [Fish-Speech paper](https://arxiv.org/abs/2411.01156) (Liao, Li and others, submitted November 2, 2024) is the 1.4-era write-up; its only version reference is a link to "fish speech 1.4" samples. It describes the serial fast-slow Dual-AR design, grouped finite scalar quantization (GFSQ) with "near 100% codebook utilization", the Firefly-GAN vocoder, and an LLM front end that drops grapheme-to-phoneme conversion entirely. Training data is "about 720,000 hours". Training ran in three stages (pre-training, SFT, then DPO): the AR model on 8 H100s for a week and the vocoder on 8 RTX 4090s for another week. On its 300-sample voice-cloning test, Fish-Speech reports 6.89% WER (Whisper-medium) against 22.20% for CosyVoice and 13.98% for F5-TTS, but CosyVoice scores higher on both speaker-similarity metrics, so "best at intelligibility" is the honest summary. The paper also reports a real-time factor of about 1:15 on an RTX 4090 and 150 ms first-packet latency.

## Fish Agent v0.1 3B (Hub created October 29, 2024)

Fish Agent is the family's one detour into voice-to-voice. The [model card](https://huggingface.co/fishaudio/fish-agent-v0.1-3b) describes a "semantic-token-free" model that takes audio in and gives audio out without a separate Whisper-style recognizer, "a continue-pretrained version of Qwen-2.5-3B-Instruct for 200B voice & text tokens", with the same 700k-hour, 8-language audio mix as 1.4. Docs landed in the repo on November 5, 2024.

Its config is the first time the slow AR is borrowed from a general LLM instead of trained from scratch: 36 layers, dim 2048, a 151,936-token Qwen vocabulary, with the usual 4-layer fast AR on 8 codebooks. The checkpoint is 6.3 GB. That design choice, an LLM backbone plus a small acoustic head, is exactly what S2 would generalize sixteen months later. Fish Agent itself was never updated past v0.1, and the S2 release notes list it under "Removed legacy code". License: CC-BY-NC-SA-4.0.

## Fish Speech 1.5 (December 25, 2024)

The [v1.5.0 release](https://github.com/fishaudio/fish-speech/releases/tag/v1.5.0) is terse ("both inference and finetune are done"), but the [model card](https://huggingface.co/fishaudio/fish-speech-1.5) carries the facts: "more than 1 million hours of audio data", now 13 languages. English and Chinese have more than 300k hours each, Japanese more than 100k, German, French, Spanish, Korean, Arabic and Russian about 20k each, and Dutch, Italian, Polish and Portuguese under 10k each. The Hub repo was created November 24, 2024.

### What changed versus 1.4

Same Dual-AR shape (24 slow layers, 4 fast layers, 8 codebooks at 21 Hz, same Firefly codec file), with three config changes: the context doubled to 8,192 tokens, the tokenizer moved to a tiktoken vocabulary of 102,048 tokens, and the fast transformer got its own explicit dimensions. My estimate is about 0.6B parameters, consistent with the 1.28 GB checkpoint. The README at the tag claims CER and WER "around 2% for 5-minute English texts". This was the open model most people meant by "Fish Speech" through 2025: it is still the most-downloaded pre-S2 repo on the Hub by monthly count.

| Spec | 1.2 | 1.4 | 1.5 |
| --- | --- | --- | --- |
| Training audio | 300k h | 700k h | >1M h |
| Languages | 3 | 8 | 13 |
| Codec | 4x1024 @ 42 Hz | 8x1024 @ 21 Hz | 8x1024 @ 21 Hz |
| Vocabulary | 32,000 | 32,000 | 102,048 |
| Max sequence | 4,096 | 4,096 | 8,192 |

License: weights CC-BY-NC-SA-4.0, and from 1.5 the code moved to Apache-2.0. v1.5.1 (May 31, 2025) was "the last stable branch before the next model release".

## OpenAudio S1 and S1-mini (June 2, 2025)

In June 2025 the research line was rebranded OpenAudio. The README commit "Update for OpenAudio S1" landed June 2, 2025; the `s1-mini` Hub repo was created May 31, 2025. (The Fish Audio blog's [introducing S1](https://fish.audio/blog/introducing-s1/) page currently shows a November 20, 2025 date, which I read as a later republish, not the release.)

S1 came in two sizes, and only one was open:

| Variant | Params | Availability | Seed-TTS Eval (en) WER / CER | Speaker distance |
| --- | --- | --- | --- | --- |
| S1 | 4B | Hosted on fish.audio | 0.008 / 0.004 | 0.332 |
| S1-mini | 0.5B, distilled | Open, `fishaudio/s1-mini` | 0.011 / 0.005 | 0.380 |

Those numbers are from the README at launch, which says they use gpt-4o-transcribe for ASR, so do not compare them directly against papers that use Whisper. The README also claims the #1 spot on TTS-Arena2 at the time. The blog adds that S1 was trained on more than 2 million hours of audio, that the team annotated over 100k hours for instruction following, and that both sizes were trained with online RLHF using GRPO and in-house reward models. S1 introduced parenthesized emotion and tone markers like `(angry)`, `(sad)` and `(in a hurry)`. The model card lists the same 13 languages as 1.5. License for S1-mini: CC-BY-NC-SA-4.0, per the card metadata. The S2 report later reports S1 at 0.54% (zh) and 1.07% (en) Seed-TTS-Eval WER, but 17.00% on the hard Chinese set, which is the weakness S2 went after.

## Fish Audio S2 Pro (March 9, 2026)

S2 is the release that took the family from "good open TTS" to a serious competitor of closed systems. The [v2.0.0-beta release](https://github.com/fishaudio/fish-speech/releases/tag/v2.0.0-beta) (March 10), the [blog post](https://fish.audio/blog/fish-audio-open-sources-s2/) (March 9), the [S2 technical report](https://arxiv.org/abs/2603.08823) (arXiv, March 9) and the [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) Hub repo (created March 9) all appeared together.

![Four generations of the Fish Audio pipeline: Fish Speech 1.0 with an AR text-to-semantic model and Firefly GAN, Fish Speech 1.2 to 1.5 with a 24-layer Dual-AR, S1 and S2 Pro with a Qwen3-4B slow AR and a 10-codebook codec, and Drama 3 with a 35B MoE backbone and a codebook-MTP decoder](/imgs/blogs/fish-audio-model-family-timeline-2.webp)

The figure is the whole family in one picture: the shape (backbone, acoustic head, codec) is stable, and each generation swaps in a bigger part.

### What changed versus S1

The release notes call it a "new Dual-AR architecture with Qwen3 backbone, replacing Fish-Speech v1.5", and a "new ModifiedDAC audio codec (replaces Firefly/VQ-GAN)". Per the report, the slow AR is initialized from Qwen3-4B with 4,096 semantic tokens added to the vocabulary, and the fast AR is 4 transformer layers; the release notes and model card size it at 400M. The codec is DAC-based: 44.1 kHz audio, 10 RVQ codebooks, a 2048x downsampling ratio and about 21 frames per second, with a 446M-parameter tokenizer. Pre-training used "over 10 million hours of raw audio across approximately 80 languages and dialects", in two stages at 8,192 then 16,384 tokens, followed by SFT and a GRPO-style RL stage whose rewards cover intelligibility, acoustic preference and speaker similarity. The report notes that S2 prepends the reference audio to the system prompt (S1 appended it to the user turn) and moves control from global prompts to free-form inline tags such as `[whisper in small voice]` at word level; the card counts "15,000+ unique tags".

| Spec | S2 Pro |
| --- | --- |
| Slow AR | 4B (Qwen3-4B init) |
| Fast AR | 400M, 4 layers |
| Codec | ModifiedDAC, 10 codebooks, ~21 Hz, 44.1 kHz |
| Training audio | 10M+ hours, ~80 languages |
| Seed-TTS-Eval WER (zh / en / zh-hard) | 0.54 / 0.99 / 5.99 |
| EmergentTTS-Eval win rate vs gpt-4o-mini-tts | 81.88% |
| Streaming (one H200, SGLang) | RTF 0.195, ~100 ms to first audio |

The report also shows where S2 is not the best: on its long-audio Chinese test Qwen3-TTS has a higher speaker-similarity mean (0.574 against 0.557), and CosyVoice 3 edges it on the hard Chinese set (5.83 against 5.99). License: the S2 release switched the repo to the **Fish Audio Research License**: free for research and non-commercial use, commercial use needs a separate license. For the technique-level walkthrough of the RL and data pipeline, see the [S2 deep-dive](/blog/machine-learning/text-to-speech/fish-audio-s2-controllable-text-to-speech).

## S2.1 Pro (June 23, 2026, hosted only)

S2.1 Pro is a production update that did not come with weights. The [launch post](https://fish.audio/blog/s2-1-pro-free-api/) (June 23, 2026) reports a 61% win rate against S2 Pro in head-to-head listening tests, 83 languages from one model, and about 70 ms time to first audio at a single request (it gives about 90 ms for the standard API elsewhere on the page). It does not describe architecture changes; the engineering detail it gives is about FP8 GEMM and attention kernels, and it promises to open-source "infrastructure components", not the model. The [models overview](https://docs.fish.audio/developer-guide/models-pricing/models-overview) lists `s2.1-pro` as the recommended production model, `s2.1-pro-free` as the same model under fair-use limits with no latency guarantees, `s2-pro` as "open-source", and `s1` as deprecated, to be retired December 31, 2026. A July 27, 2026 [company post](https://fish.audio/blog/fish-audio-52m-seed-funding/) announcing a 52M seed round says the team shipped five audio models in its first year (four TTS, one speech-to-text) without naming them.

## Fish Audio Drama 3 (October 13, 2026)

Drama 3 is the latest model and the biggest architectural change since 1.2. The [v3.0.0-beta pre-release](https://github.com/fishaudio/fish-speech/releases/tag/v3.0.0-beta) and the [Drama 3 technical report](https://arxiv.org/abs/2610.18010) are both dated October 13, 2026; the [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) Hub repo was created October 1. Neither the report nor the card mentions a Drama 1 or 2; the only predecessor named is S2.

### What changed versus S2

The backbone is no longer dense. Drama 3 is initialized from the language model of Qwen3.5-35B-A3B: 40 layers in blocks of three Gated DeltaNet linear-attention layers plus one gated full-attention layer, 256 routed experts with top-8 routing and a shared expert, about 3B of 35B parameters active per token. S2's fast AR is replaced by a 13-layer **codebook-MTP decoder** (4 shared layers, then 9 per-codebook layers), and each of those layers attends causally over the whole sequence, reference audio included, instead of over one frame. The codec is reused unchanged from S2 (10 codebooks, about 21.5 Hz, 44.1 kHz). Pre-training covered "several hundred billion tokens" in about 80 languages and dialects over three stages, the last of which adds long-form synthesis, streaming interleave and speech infilling. RL uses GRPO with a single learned human-preference reward model, with no rule-based rewards and no KL penalty.

New capabilities, all in one model: multi-turn revision (a note such as `[happier versus previous]` makes the model redo a take), editing by infilling, scene dubbing with up to 16 speakers, and streaming text input.

| Spec | Drama 3 |
| --- | --- |
| Backbone | 35B MoE, ~3B active, hybrid linear attention |
| Audio decoder | 13-layer codebook-MTP |
| Codec | Same as S2: 10 codebooks, ~21.5 Hz, 44.1 kHz |
| Seed-TTS-Eval test-en WER / test-zh CER (final take) | 1.54 / 0.83 |
| Single-stream speed, one H200 | 271 frames/s (S2: 177) |
| Minimum GPU | 48 GB (FP8 experts) |

The Seed-TTS-Eval rows in the report compare Drama 3 only with its own SFT checkpoint, not with S2, so do not read them as a regression against S2's 0.99. Other systems beat it on some axes: OmniVoice and Qwen3-TTS have higher speaker similarity on LibriSpeech-PC cloning, and Step-Audio-EditX scores higher on the revision benchmark. The model card's serving table reports a 88.9 ms median time to first audio at concurrency 1 on one H200. There is also `fishaudio/drama-3-int4` (Hub created October 14), an unofficial W4A16 quantization credited to AK that cuts the download from 41.5 GB to 22.1 GB and runs on A100-class GPUs.

License: the release notes move the repo to the **Fish Audio Community License** and keep S2 Pro under the Research License. The Drama 3 card body agrees, although its YAML metadata still names the research license, so read the `LICENSE` file before you ship. The same release removed fine-tuning code from the repo.

## What came after, and what is latest as of October 2026

Nothing open has shipped after Drama 3. The `fishaudio` Hub org lists eleven model repos, and the two newest are `drama-3` and `drama-3-int4`. There is no S2.1 checkpoint, no S2-mini, and no Drama 3 mini. A separate `openaudio` Hub org holds only infrastructure artifacts (an FP8 Qwen3-Omni conversion and a Qwen3.5-35B-A3B tokenizer), not new TTS models. On the API side, S2.1 Pro remains the recommended hosted model and S1 retires on December 31, 2026. For how Drama 3 and S2 sit against the rest of the field, see [the 2026 audio model landscape](/blog/machine-learning/audio-generation/the-2026-audio-model-landscape).

## How the family evolved

### 1. Data grew 67x in two years

![Training audio hours by release on a log scale, from 150k hours for Fish Speech 1.0 to more than 10M hours for S2](/imgs/blogs/fish-audio-model-family-timeline-3.webp)

On a log axis the growth is nearly a straight line until S2's jump. Every '>' figure is a lower bound from the release's own card or report; Drama 3 reports tokens, not hours.

### 2. The backbone was borrowed, then scaled

Fish Speech 1.2 to 1.5 trained a 24-layer transformer from scratch. Fish Agent tried a Qwen-2.5-3B backbone, S1 and S2 standardized on a 4B LLM, and Drama 3 adopted a 35B MoE. The open backbone stayed near 0.5B for a year and then grew by roughly 70x in total parameters, while the active count per token only grew from about 0.5B to 3B.

![Parameter counts by release on a log scale, with S1 marked as hosted only and Drama 3 showing 35B total with 3B active](/imgs/blogs/fish-audio-model-family-timeline-5.webp)

### 3. The acoustic head became a first-class decoder

The fast AR started as a 4-layer helper that only saw one frame. In S2 it was sized at 400M; in Drama 3 it became 13 layers that attend over the whole sequence. Meanwhile the codec went from FSQ GAN tokens to a 10-codebook RVQ codec at 44.1 kHz.

### 4. Post-training went from DPO to RL with learned rewards

The 1.4-era paper ends with DPO. S1 introduced online RLHF with GRPO; S2 used multiple rewards (intelligibility, acoustic preference, speaker similarity); Drama 3 uses a single learned human-preference reward and no KL penalty.

### 5. Licenses got more specific, and the best model moved behind the API

CC-BY-NC-SA-4.0 covered weights from 1.0 through S1-mini. S2 introduced the Fish Audio Research License, Drama 3 the Community License. In parallel, the strongest model at each moment (S1 in 2025, S2.1 Pro in mid-2026) was hosted only.

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| Best open quality, multi-speaker scenes, editing | Drama 3 | Newest open model; revision, infilling, 16 speakers; needs 48 GB (or the unofficial int4 on an A100) |
| Strong open model on a single 24 GB-class GPU | S2 Pro | ~4.6B total; mature SGLang serving; Research License |
| Smallest open model, legacy pipelines | S1-mini or Fish Speech 1.5 | ~0.5B to 0.6B; CC-BY-NC-SA-4.0; much older quality |
| Production with an SLA | S2.1 Pro API | Recommended hosted model; no weights |
| Avoid | S1 on the API | Deprecated, retired December 31, 2026 |

The 24 GB note for S2 Pro is my inference from its two safetensors shards (about 9.1 GB together) plus the 1.9 GB codec, not a published requirement. Whichever you pick, none of the open weights since 1.0 allow commercial use without a separate agreement or a careful read of the Community License, so check the license file of the exact checkpoint you deploy.

## References

1. Fish Audio. [fish-speech GitHub releases](https://github.com/fishaudio/fish-speech/releases), v0.2.0 through v3.0.0-beta, accessed 2026-10-23.
2. Fish Audio. [fishaudio Hugging Face org](https://huggingface.co/fishaudio): model cards, `config.json` and file lists for fish-speech-1, 1.2, 1.2-sft, 1.4, 1.5, fish-agent-v0.1-3b, s1-mini, s2-pro, drama-3, drama-3-int4.
3. Liao, S., Li, T., et al. [Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis](https://arxiv.org/abs/2411.01156). arXiv 2411.01156, 2024.
4. Fish Audio. [Introducing S1](https://fish.audio/blog/introducing-s1/), and the fish-speech README at the OpenAudio S1 launch (June 2025).
5. Liao, S., Wang, Y., et al. [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823). arXiv 2603.08823, 2026.
6. Fish Audio. [Fish Audio Open-Sources S2](https://fish.audio/blog/fish-audio-open-sources-s2/), March 9, 2026.
7. Fish Audio. [S2.1 Pro: Free Text-to-Speech API for Developers](https://fish.audio/blog/s2-1-pro-free-api/), June 23, 2026.
8. Fish Audio. [Models overview](https://docs.fish.audio/developer-guide/models-pricing/models-overview), developer docs, accessed 2026-10-23.
9. Fish Audio. [5 Models, 22 People, 1 Year](https://fish.audio/blog/fish-audio-52m-seed-funding/), July 27, 2026.
10. Fish Audio. [Fish Audio Drama 3 Technical Report](https://arxiv.org/abs/2610.18010). arXiv 2610.18010, 2026.
11. Fish Audio. [fish-diffusion](https://github.com/fishaudio/fish-diffusion) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) repositories.
