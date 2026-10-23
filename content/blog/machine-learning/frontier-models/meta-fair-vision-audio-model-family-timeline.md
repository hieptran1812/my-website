---
title: "Meta FAIR Beyond Llama, Release by Release: DINO, SAM, V-JEPA, Seamless and the Audio Line"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of Meta FAIR's open-weight vision and audio models from wav2vec 2.0 to October 2026: DINO through DINOv3, SAM through SAM 3.1, Perception Encoder, V-JEPA, Sapiens, MMS, AudioCraft, Seamless, Omnilingual ASR and SAM Audio, with dates, sizes, licenses and primary sources."
tags: ["meta-fair", "dino", "segment-anything", "v-jepa", "perception-encoder", "seamless", "audiocraft", "open-weights", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 19
---

When people say "Meta's open models" they usually mean Llama. That line has its own catalog in the [Llama model family timeline](/blog/machine-learning/frontier-models/llama-model-family-timeline), and this post does not repeat it. The other half of Meta's open-weight output is arguably more influential in day-to-day engineering: the DINO backbones that sit under half the dense-prediction papers you read, the Segment Anything models that turned "click to get a mask" into a commodity, the V-JEPA video encoders, and an audio line that runs from wav2vec 2.0 through MusicGen and Seamless to an ASR system covering more than 1,600 languages.

That output is spread over a dozen GitHub repos and several hundred Hugging Face repos under the `facebook` org, with five different license families. This post is the map. It walks each significant open-weight release in date order, says what changed from its predecessor, gives a small spec table, names the license, and links a primary source. The timeline below is the mental model: four lanes that advance in bursts, with two speech generators that Meta announced and deliberately never released.

![Meta FAIR open-weight release timeline from 2020 to October 2026 in four lanes: self-supervised vision, segmentation, perception and video, audio and speech, with withheld models shown as hollow markers](/imgs/blogs/meta-fair-vision-audio-model-family-timeline-1.webp)

Notice how the segmentation and audio lanes thicken in late 2025: SAM 3, SAM 3D, Omnilingual ASR and SAM Audio all landed within five weeks. The question people ask most, answered up front: **as of October 23, 2026, the newest open releases in each lane are DINOv3 (August 2025, with satellite and canopy-height add-ons through March 2026), SAM 3.1 (March 27, 2026), Sapiens2 (April 24, 2026) and V-JEPA 2.1 (March 16, 2026) on the vision side, and SAM Audio (December 2025) plus Omnilingual ASR (November 2025) on the audio side.** The newest checkpoint on the `facebook` Hub org that fits this catalog is the MoE-ViE vision encoder family, Hub created August 2026.

A note on method. I pulled the README and LICENSE of each `facebookresearch` repo, the Hugging Face Hub API for every model named here (creation dates, license tags, and safetensors parameter totals where exposed), and the arXiv listing for each paper. Where an announcement or README gives a date I use it. Where only the Hub exists I write "Hub created", which is a lower bound on the public date. Repos created before March 2022 show a Hub creation date of March 2, 2022, which is the Hub migration date, not a release date, so I never use it.

## The timeline at a glance

| Date | Release | Weights | One-line significance |
| --- | --- | --- | --- |
| 2020-06-20 | wav2vec 2.0 | Open, fairseq (MIT code) | Self-supervised speech pretraining; 10 minutes of labels gets usable ASR |
| 2021-04-29 | DINO | Open, Apache-2.0 | Self-distillation with no labels; ViT attention maps segment objects |
| 2023-04-05 | Segment Anything (SAM) | Open, Apache-2.0 | Promptable masks; SA-1B with 11M images and 1.1B masks |
| 2023-04-14 | DINOv2 | Open, CC-BY-NC, then Apache-2.0 | 1.1B ViT-g on curated LVD-142M; frozen features that work everywhere |
| 2023-05-22 | MMS | Open, CC-BY-NC-4.0 | wav2vec 2.0 scaled to 1,000+ languages for ASR, TTS and language ID |
| 2023-06 | Voicebox | Withheld | Speech generation; announced, not released |
| 2023-06-08 to 08-02 | MusicGen, AudioGen, EnCodec (AudioCraft) | Open, CC-BY-NC-4.0 weights | Text-to-music and text-to-sound over a neural codec |
| 2023-08-22 | SeamlessM4T | Open, CC-BY-NC-4.0 | One model for speech and text translation in about 100 languages |
| 2023-11-29 (Hub) | SeamlessM4T v2, Expressive, Streaming | M4T v2 CC-BY-NC; Expressive under Seamless License | Prosody-preserving and low-latency translation |
| late 2023 | Audiobox | Demo only | Voicebox successor with watermarking; not released |
| 2024-02-15 | V-JEPA | Open, CC-BY-NC-4.0 | Predict masked video features, not pixels |
| 2024-07-29 | SAM 2 (2.1 on 2024-09-29) | Open, Apache-2.0 | SAM for video with a streaming memory |
| 2024-08-22 | Sapiens | Open, CC-BY-NC-4.0 | Human-centric ViTs, 0.3B to 2B, on 300M human images |
| 2025-02-07 | Audiobox Aesthetics | Open, CC-BY-4.0 | Automatic quality scores for speech, music and sound |
| 2025-04-17 | Perception Encoder | Open, Apache-2.0 | Contrastive image and video encoder; Core, Lang, Spatial variants |
| 2025-06-11 | V-JEPA 2 | Open, MIT | Video world model; action-conditioned V-JEPA 2-AC for robots |
| 2025-08-14 | DINOv3 | Gated, DINOv3 License | 7B ViT on LVD-1689M; distilled ViT and ConvNeXt family |
| 2025-11-12 | Omnilingual ASR | Open, Apache-2.0 | 1,600+ languages; wav2vec 2.0 encoders up to 7B |
| 2025-11-19 | SAM 3 and SAM 3D | Gated, SAM License | Segment by text concept; single-image 3D objects and bodies |
| 2025-12-16 | SAM Audio and PE-AV | Gated, SAM License (PE-AV Apache-2.0) | Separate any sound by text, visual or time prompt |
| 2026-03-16 | V-JEPA 2.1 | Open, MIT | Dense, temporally consistent features; 2B ViT-G |
| 2026-03-27 | SAM 3.1 | Gated, SAM License | Object Multiplex: shared-memory multi-object tracking |
| 2026-04-24 | Sapiens2 | Gated, Sapiens2 License | 0.1B to 5B on 1B human images |
| 2026-08 (Hub) | MoE-ViE | Open, CC-BY-NC-4.0 | Mixture-of-experts CLIP-style vision encoder |

## wav2vec 2.0 (June 20, 2020)

The oldest model in the catalog and still the root of the audio tree. wav2vec 2.0 masks spans of latent speech features and learns by contrasting the true quantized latent against distractors, then fine-tunes with CTC on a little labeled data. The weights shipped through fairseq (MIT license) and live on the Hub as `facebook/wav2vec2-base`, `facebook/wav2vec2-large-960h-lv60-self` and siblings, tagged Apache-2.0 there. Every later Meta speech encoder in this post, MMS and the Omnilingual ASR `W2V` checkpoints included, is a scaled wav2vec 2.0. For what goes wrong when you fine-tune one, see [Debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning).

Primary source: [arXiv 2006.11477](https://arxiv.org/abs/2006.11477).

## DINO (April 29, 2021)

DINO trains a student ViT to match a momentum teacher's output on different crops of the same image, with no labels and no negatives. The surprise was in the attention maps: the `[CLS]` token's attention in the last layer segments the foreground object without ever seeing a mask. The released checkpoints are small by today's standard.

| Checkpoint | Params |
| --- | --- |
| ViT-S/16, ViT-S/8 | 21M |
| ViT-B/16, ViT-B/8 | 85M |
| ResNet-50 | 23M |
| XCiT small / medium | 26M / 84M |

License: Apache-2.0, code and weights. Primary source: [arXiv 2104.14294](https://arxiv.org/abs/2104.14294), [facebookresearch/dino](https://github.com/facebookresearch/dino). For a gentle introduction see [ViT, SigLIP and DINO](/blog/machine-learning/computer-vision/vit-siglip-dino-explained), and for a 2026 result showing the `[CLS]` map is the wrong place to look for objects, see [Object-DINO](/blog/paper-reading/computer-vision/finding-distributed-object-centric-properties-in-self-supervised-transformers).

## Segment Anything (April 5, 2023)

SAM split segmentation into a heavy image encoder (run once per image) and a light prompt decoder (run per click or box), and trained it on SA-1B: 11 million images with 1.1 billion masks, largely produced by the model itself in a model-in-the-loop data engine. The point was not a better mask on a benchmark but a model you could prompt with points or boxes on anything.

| Checkpoint | Params (Hub safetensors) |
| --- | --- |
| `sam-vit-base` | 94M |
| `sam-vit-huge` | 641M |

License: Apache-2.0 for the model; SA-1B has its own research license. Primary source: [arXiv 2304.02643](https://arxiv.org/abs/2304.02643), [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).

## DINOv2 (April 14, 2023)

What changed versus DINO is mostly data and scale. DINOv2 trained a 1.1B ViT-g on LVD-142M (a curated, deduplicated 142M-image set), then distilled it into smaller ViTs. The resulting frozen features work with a linear head for classification, segmentation and depth, which is why DINOv2 became the default backbone for depth estimators and robotics policies. An October 2023 follow-up added "register" tokens that remove high-norm artifact patches from the feature maps.

| Checkpoint | Params |
| --- | --- |
| ViT-S/14 distilled | 21M |
| ViT-B/14 distilled | 86M |
| ViT-L/14 distilled | 300M |
| ViT-g/14 | 1,100M |

License: launched under CC-BY-NC. A commit on August 31, 2023 ("Update code and models license from CC-BY-NC to Apache 2.0") moved code and weights to Apache-2.0. That makes DINOv2 the one release here whose license got more permissive after launch. Primary source: [arXiv 2304.07193](https://arxiv.org/abs/2304.07193), [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).

## MMS: Massively Multilingual Speech (May 22, 2023)

MMS took wav2vec 2.0 and pretrained it on speech in over a thousand languages, then released ASR, TTS and language-identification models on top. `facebook/mms-1b-all` (965M parameters by Hub safetensors count) is one ASR model with per-language adapters. The TTS checkpoints followed on the Hub in August 2023 (`mms-tts-eng` Hub created August 24, 2023).

License: CC-BY-NC-4.0. Primary source: [arXiv 2305.13516](https://arxiv.org/abs/2305.13516).

## MusicGen, AudioGen and EnCodec: AudioCraft (June 8 to August 2, 2023)

MusicGen came first: a single autoregressive transformer that generates interleaved EnCodec tokens from a text prompt, optionally conditioned on a melody, trained on 20K hours of licensed music. On August 2, 2023 Meta bundled it with AudioGen (text-to-sound effects) and an improved EnCodec decoder as AudioCraft. EnCodec itself is older (paper October 2022); the 24 kHz checkpoint was Hub created June 12, 2023.

| Checkpoint | Params | Conditioning |
| --- | --- | --- |
| `musicgen-small` | 300M | Text |
| `musicgen-medium` | 1.5B | Text |
| `musicgen-melody` | 1.5B | Text + melody |
| `musicgen-large` | 3.3B | Text |
| `audiogen-medium` | Hub created July 26, 2023 | Text |

Stereo fine-tunes arrived later (`musicgen-stereo-large` Hub created October 23, 2023). License: code MIT, weights CC-BY-NC-4.0, which is the source of the conflicting press reports at launch. Primary sources: [arXiv 2306.05284](https://arxiv.org/abs/2306.05284), [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft). Deep-dives: [MusicLM and MusicGen](/blog/machine-learning/audio-generation/music-generation-musiclm-and-musicgen) and [EnCodec, DAC and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec).

Voicebox (announced June 2023, paper [arXiv 2306.15687](https://arxiv.org/abs/2306.15687)) and Audiobox (late 2023 demo): withheld, no weights, citing voice-cloning misuse risk.

## SeamlessM4T (August 22, 2023) and the Seamless suite (late 2023)

SeamlessM4T is one model that does speech-to-speech, speech-to-text, text-to-speech and text-to-text translation plus ASR in nearly 100 languages. Version 2 (`seamless-m4t-v2-large`, Hub created November 29, 2023, 2.31B parameters by safetensors count) swapped in a non-autoregressive text-to-unit decoder and became the base for two derived models: SeamlessExpressive, which carries the speaker's prosody and vocal style across languages, and SeamlessStreaming, which translates with about two seconds of latency. "Seamless" combines all three. In December 2023 Meta also open-sourced the w2v-BERT 2.0 speech encoder at the core of these models.

| Model | License |
| --- | --- |
| SeamlessM4T v1, v2; SeamlessStreaming | CC-BY-NC-4.0 |
| Seamless, SeamlessExpressive | Seamless License (gated) |
| w2v-BERT 2.0 encoder, code | MIT |

Primary sources: [arXiv 2308.11596](https://arxiv.org/abs/2308.11596), [arXiv 2312.05187](https://arxiv.org/abs/2312.05187), [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication).

## V-JEPA (February 15, 2024)

V-JEPA is Yann LeCun's joint-embedding predictive architecture applied to video: mask large spatio-temporal blocks and predict the target encoder's features for them, never the pixels. The bet is that discarding unpredictable pixel detail gives better features for understanding motion. License: CC-BY-NC-4.0 (the `jepa` repo LICENSE). Primary source: [arXiv 2404.08471](https://arxiv.org/abs/2404.08471), [facebookresearch/jepa](https://github.com/facebookresearch/jepa).

## SAM 2 (July 29, 2024) and SAM 2.1 (September 29, 2024)

What changed versus SAM: video. SAM 2 adds a streaming memory of past frames and predictions, so a click on frame 1 becomes a tracked mask through the clip, and it replaces the plain ViT with a hierarchical Hiera encoder that makes it far smaller than SAM's ViT-H. SAM 2.1 retrained the checkpoints and shipped the training code and a developer suite.

| Checkpoint (2.1) | Params |
| --- | --- |
| `sam2.1_hiera_tiny` | 38.9M |
| `sam2.1_hiera_small` | 46M |
| `sam2.1_hiera_base_plus` | 80.8M |
| `sam2.1_hiera_large` | 224.4M |

License: Apache-2.0 for checkpoints, demo and training code. Primary source: [arXiv 2408.00714](https://arxiv.org/abs/2408.00714), [facebookresearch/sam2](https://github.com/facebookresearch/sam2).

## Sapiens (August 22, 2024)

Sapiens is a bet on a narrow domain: ViTs pretrained on 300 million in-the-wild human images, then fine-tuned for 2D pose, body-part segmentation, depth and surface normals, at 0.3B, 0.6B, 1B and 2B parameters. License: CC-BY-NC-4.0. Primary source: [arXiv 2408.12569](https://arxiv.org/abs/2408.12569), [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens).

## Audiobox Aesthetics (February 7, 2025)

Not a generator but a judge: one model that predicts aesthetic quality scores for speech, music and sound alike. It is the open piece of the Audiobox effort and the one most useful for filtering training data. License: CC-BY-4.0, the only commercial-friendly Creative Commons license in this catalog. Primary source: [arXiv 2502.05139](https://arxiv.org/abs/2502.05139).

## Perception Encoder (April 17, 2025)

Perception Encoder (PE) is Meta's answer to SigLIP: a contrastively trained image and video encoder whose paper title states its finding, "the best visual embeddings are not at the output of the network". PE ships three heads off one recipe: PE Core (zero-shot classification and retrieval), PE Lang (aligned to an LLM, the vision tower of the Perception Language Model) and PE Spatial (dense tasks). The largest are G/14 at 448 pixels; Spatial comes down to T/16. License: Apache-2.0 for PE weights; the PLM models in the same repo are under the FAIR Research License. Primary source: [arXiv 2504.13181](https://arxiv.org/abs/2504.13181), [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models).

## V-JEPA 2 (June 11, 2025)

What changed versus V-JEPA: scale and a use. V-JEPA 2 scales the encoder to a 1B ViT-g and adds V-JEPA 2-AC, an action-conditioned variant that lets a robot plan by imagining futures in feature space.

| Checkpoint | Params |
| --- | --- |
| ViT-L/16 | 300M |
| ViT-H/16 | 600M |
| ViT-g/16 (256 and 384 px) | 1B |

License: MIT per the repo README, with some portions Apache-2.0; note the Hub tags disagree by repo (`vjepa2-vitl-fpc64-256` says MIT, `vjepa2-vitg-fpc64-256` says Apache-2.0). Either way, this is the most permissive video model Meta has shipped. Primary source: [arXiv 2506.09985](https://arxiv.org/abs/2506.09985), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).

## DINOv3 (August 14, 2025)

What changed versus DINOv2: a 6x larger teacher and a 12x larger dataset. DINOv3's 7B ViT trains on LVD-1689M, and a separate pair is trained on SAT-493M satellite imagery. The family is distilled down to ViT and ConvNeXt students so most users never touch the 7B.

| Checkpoint | Params | Data |
| --- | --- | --- |
| ViT-S/16, S+/16 distilled | 21M, 29M | LVD-1689M |
| ViT-B/16 distilled | 86M | LVD-1689M |
| ViT-L/16 distilled | 300M | LVD-1689M, SAT-493M |
| ViT-H+/16 distilled | 840M | LVD-1689M |
| ViT-7B/16 | 6,716M | LVD-1689M, SAT-493M |
| ConvNeXt T/S/B/L | 29M to 198M | LVD-1689M |

License: the custom **DINOv3 License**, royalty-free and redistributable but gated on the Hub, with trade-control and ITAR exclusions. This is the release where the DINO line left Apache-2.0. Follow-ups in the same repo: ConvNeXt distillation code (November 2025) and a Canopy Height Maps v2 model (March 10, 2026). Primary source: [arXiv 2508.10104](https://arxiv.org/abs/2508.10104), [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).

## Omnilingual ASR (November 12, 2025)

The descendant of MMS: speech recognition for more than 1,600 languages, released as three families of four sizes each. `W2V` checkpoints are bare wav2vec 2.0 encoders, `CTC` adds a CTC head, and `LLM` puts an autoregressive text decoder on top with optional language conditioning; a zero-shot `LLM-7B-ZS` variant handles unseen languages from a few examples.

| Family | Sizes (params) |
| --- | --- |
| W2V (SSL encoder) | 317M, 966M, 3.06B, 6.49B |
| CTC | 325M, 975M, 3.08B, 6.50B |
| LLM | 1.63B, 2.28B, 4.38B, 7.80B |

License: Apache-2.0 for code and models, with the training corpus released under CC-BY-4.0. After two years of CC-BY-NC audio models, this is the first speech release here under a fully permissive license. Primary source: [arXiv 2511.09690](https://arxiv.org/abs/2511.09690), [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr).

## SAM 3 and SAM 3D (November 19, 2025)

What changed versus SAM 2: the prompt. SAM 3 accepts a short noun phrase or an image exemplar and returns every matching instance, in images and video, alongside the old point and box prompts. It is a detector and a tracker sharing one vision encoder, 848M parameters per the README, trained with a data engine that annotated over 4 million unique concepts. SAM 3D shipped the same day as two models: SAM 3D Objects (single image to 3D shape and texture) and SAM 3D Body (human mesh recovery, with a variant on a DINOv3 backbone).

License: the custom **SAM License** for SAM 3, SAM 3D Objects and SAM 3D Body, gated on the Hub. This is where Segment Anything left Apache-2.0. Primary sources: [arXiv 2511.16719](https://arxiv.org/abs/2511.16719), [arXiv 2511.16624](https://arxiv.org/abs/2511.16624), [facebookresearch/sam3](https://github.com/facebookresearch/sam3).

## SAM Audio and PE-AV (December 16, 2025)

SAM Audio carries the Segment Anything interface into sound: isolate one source from a mixture by a text description, a visual prompt (click the thing making the noise in the video) or a time span. It ships in small, base and large, plus `-tv` variants and a separate Judge model that scores separation on precision, recall and faithfulness. Both lean on PE-AV, an audio-visual extension of Perception Encoder released the same day with a frame-level PE-A-Frame sibling. License: SAM License (gated) for SAM Audio; PE-AV sits in the Apache-2.0 perception_models repo. Primary source: [arXiv 2512.18099](https://arxiv.org/abs/2512.18099), [facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio).

## V-JEPA 2.1 and SAM 3.1 (March 2026)

Two incremental releases two weeks apart. **V-JEPA 2.1** (March 16, 2026) changes the recipe to get dense features: a predictive loss on visible as well as masked tokens, self-supervision at intermediate layers, and joint image and video tokenizers, with a new 2B ViT-G at the top. License stays MIT ([arXiv 2603.14482](https://arxiv.org/abs/2603.14482)). **SAM 3.1** (March 27, 2026) adds Object Multiplex, a shared memory for tracking many objects at once that the README describes as significantly faster without an accuracy loss; it stays under the SAM License. The same month brought EUPE (Efficient Universal Perception Encoder, Hub created March 23, 2026), small ViT and ConvNeXt encoders distilled from a large proxy teacher, under the FAIR Noncommercial Research License.

## Sapiens2 (April 24, 2026)

What changed versus Sapiens: 3.3x the human images (1 billion) and a range from 0.1B to 5B parameters, with pointmap and matting heads added to pose, segmentation and normals. The matting model followed on May 15, 2026.

| Backbone | Params |
| --- | --- |
| Sapiens2-0.1B | 0.114B |
| Sapiens2-0.4B | 0.398B |
| Sapiens2-0.8B | 0.818B |
| Sapiens2-1B (and 4K variant) | 1.462B (1.607B) |
| Sapiens2-5B | 5.071B |

License: the custom **Sapiens2 License**, which adds acceptable-use limits such as no unlicensed professional practice. Primary source: [arXiv 2604.21681](https://arxiv.org/abs/2604.21681), [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2).

## What is latest as of October 2026

The newest open checkpoint that fits this catalog is **MoE-ViE**, a mixture-of-experts CLIP-style vision encoder (`MoEViE-B16-224`, `L16-384`, `H14-448`, Hub created August 9 to 12, 2026, CC-BY-NC-4.0, [arXiv 2608.17402](https://arxiv.org/abs/2608.17402)). It studies fine-grained expert topologies for vision encoders and adds frame-level distillation for video. Lane by lane, the current heads are DINOv3 (self-supervised vision), SAM 3.1 (segmentation), Sapiens2 (humans), V-JEPA 2.1 (video), Omnilingual ASR (speech recognition) and SAM Audio (separation). There is no open successor to MusicGen or SeamlessM4T v2; those two lanes have been quiet since 2023.

## How the family evolved

![Lineage of Meta FAIR's vision and audio models: DINO to DINOv3 feeding SAM 3D Body, SAM to SAM 3.1, Perception Encoder to PE-AV to SAM Audio, V-JEPA to V-JEPA 2.1, wav2vec 2.0 to MMS to Omnilingual ASR, w2v-BERT 2.0 to Seamless, EnCodec to MusicGen](/imgs/blogs/meta-fair-vision-audio-model-family-timeline-2.webp)

Read the figure as seven parent-to-child lineages in two families, image and video on one side, speech and audio on the other. Most lines are a straight chain of successors; the cross-links are rare and telling, like SAM 3D Body shipping on a DINOv3 backbone and SAM Audio leaning on PE-AV.

### 1. Self-supervision is the spine

DINO, DINOv2, DINOv3, V-JEPA, V-JEPA 2 and 2.1, wav2vec 2.0 and the Omnilingual `W2V` encoders are all trained without labels. The supervised and promptable models (SAM, Sapiens, MMS, SAM 3D Body) sit on top of them. The practical lesson is that Meta's most reused releases are encoders, not end-task models.

### 2. Parameters grew about 80x at the top

![Flagship parameter counts on a log scale, from SAM 2.1 Hiera-L at 0.224B to DINOv3 ViT-7B at 6.72B and Omnilingual ASR LLM-7B at 7.8B](/imgs/blogs/meta-fair-vision-audio-model-family-timeline-3.webp)

The 2021 DINO flagship was an 85M ViT-B; the 2025 DINOv3 teacher is 6.7B, and Sapiens2 and Omnilingual ASR reach 5B and 7.8B. The counter-trend is in segmentation: SAM 2's large model (224M) is a third of SAM's ViT-H, because a video model has to run per frame.

### 3. Data grew faster than models

![Pretraining image sets as waffle grids, one cell per 10M images: SA-1B 11M, LVD-142M, Sapiens 300M, SAT-493M, Sapiens2 1B, LVD-1689M](/imgs/blogs/meta-fair-vision-audio-model-family-timeline-4.webp)

Measured against SA-1B, the later sets are 13x, 27x, 45x, 91x and about 154x larger. Meta's dataset names encode their size, which makes the trend easy to read: LVD-142M became LVD-1689M, a 12x jump, and the human-image set went from 300M to 1B. Curation, not raw scraping, is the recurring method: DINOv2's LVD-142M was retrieved and deduplicated against curated seeds, and SAM's masks came from a model-in-the-loop data engine.

### 4. Licenses went Apache, then non-commercial, then custom

![License class per release over time: Apache-2.0 for DINO, SAM, SAM 2, PE and Omnilingual ASR; CC-BY-NC for MMS, MusicGen, SeamlessM4T, V-JEPA and Sapiens; custom Meta licenses for DINOv3, SAM 3, SAM Audio and Sapiens2](/imgs/blogs/meta-fair-vision-audio-model-family-timeline-5.webp)

The pattern by year: 2021 to 2024 vision releases were Apache-2.0 (DINOv2 after its August 2023 switch), 2023 audio and 2024 video and human models were CC-BY-NC-4.0, and from August 2025 the flagships (DINOv3, SAM 3, SAM 3D, SAM Audio, Sapiens2) moved to bespoke Meta licenses that are royalty-free and commercial-capable but gated, with trade-control and acceptable-use clauses. Two releases run against the trend: V-JEPA 2 and 2.1 (MIT) and Omnilingual ASR (Apache-2.0).

### 5. Withholding is reserved for voice generation

Meta has released music, sound and translation models but withheld every model that can clone a voice from a short sample (Voicebox, Audiobox). SeamlessExpressive, which transfers vocal style, is the closest thing released, and it is the one Seamless model behind a gated custom license.

## Which variant to pick today

| Need | Pick | License note |
| --- | --- | --- |
| General frozen image features, permissive | DINOv2 ViT-L/14 or ViT-g/14 with registers | Apache-2.0 |
| Best frozen dense features | DINOv3 ViT-L/16 or ViT-H+/16 distilled | DINOv3 License, gated |
| Satellite or aerial imagery | DINOv3 SAT-493M ViT-L | DINOv3 License |
| Click or box to mask, commercial-simple | SAM 2.1 Hiera-B+ or L | Apache-2.0 |
| Segment by text concept, or many objects in video | SAM 3, or SAM 3.1 for multi-object tracking | SAM License |
| Image-text retrieval or a VLM vision tower | PE Core G/14 or PE Lang | Apache-2.0 |
| Video understanding or robot planning | V-JEPA 2.1 ViT-g or ViT-G | MIT |
| Human pose, parts, normals | Sapiens2 1B (5B if you can afford it) | Sapiens2 License |
| ASR in a long-tail language | Omnilingual ASR CTC-1B for speed, LLM-7B for accuracy | Apache-2.0 |
| Speech translation | SeamlessM4T v2 large | CC-BY-NC-4.0 |
| Music or sound effects | MusicGen / AudioGen | CC-BY-NC-4.0 weights |
| Isolate one sound in a mix | SAM Audio large | SAM License |
| Filter audio data by quality | Audiobox Aesthetics | CC-BY-4.0 |

The license column is not decoration. If your product is commercial and you cannot accept a gated custom license, the vision stack that remains is DINOv2, SAM 2.1, Perception Encoder and V-JEPA 2.1, and the speech stack is Omnilingual ASR plus wav2vec 2.0.

## What this catalog could not verify

- Exact public dates for SeamlessM4T v2, SeamlessExpressive and SeamlessStreaming. The Hub gives November 29, 2023 for `seamless-m4t-v2-large` and the paper is dated December 8, 2023; I did not fetch an announcement with a day.
- The exact Audiobox announcement day; secondary coverage places it in late 2023.
- V-JEPA 2's license: the repo says MIT, but one Hub repo is tagged Apache-2.0.
- SAM 3's parameter count: the README says 848M, the Hub safetensors total is 860M (the difference is plausibly non-encoder weights, unverified).
- Model cards for gated repos (DINOv3, SAM 3, SAM 3.1, SAM Audio) are not readable without accepting the license, so specs come from the GitHub READMEs instead.

## References

- wav2vec 2.0: [arXiv 2006.11477](https://arxiv.org/abs/2006.11477); [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
- DINO: [arXiv 2104.14294](https://arxiv.org/abs/2104.14294); [facebookresearch/dino](https://github.com/facebookresearch/dino)
- Segment Anything: [arXiv 2304.02643](https://arxiv.org/abs/2304.02643); [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- DINOv2: [arXiv 2304.07193](https://arxiv.org/abs/2304.07193); registers [arXiv 2309.16588](https://arxiv.org/abs/2309.16588); [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- MMS: [arXiv 2305.13516](https://arxiv.org/abs/2305.13516)
- MusicGen: [arXiv 2306.05284](https://arxiv.org/abs/2306.05284); AudioGen [arXiv 2209.15352](https://arxiv.org/abs/2209.15352); EnCodec [arXiv 2210.13438](https://arxiv.org/abs/2210.13438); [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- Voicebox: [arXiv 2306.15687](https://arxiv.org/abs/2306.15687)
- SeamlessM4T: [arXiv 2308.11596](https://arxiv.org/abs/2308.11596); Seamless [arXiv 2312.05187](https://arxiv.org/abs/2312.05187); [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication)
- V-JEPA: [arXiv 2404.08471](https://arxiv.org/abs/2404.08471); [facebookresearch/jepa](https://github.com/facebookresearch/jepa)
- SAM 2: [arXiv 2408.00714](https://arxiv.org/abs/2408.00714); [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Sapiens: [arXiv 2408.12569](https://arxiv.org/abs/2408.12569); Sapiens2 [arXiv 2604.21681](https://arxiv.org/abs/2604.21681)
- Audiobox Aesthetics: [arXiv 2502.05139](https://arxiv.org/abs/2502.05139)
- Perception Encoder: [arXiv 2504.13181](https://arxiv.org/abs/2504.13181); [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)
- V-JEPA 2: [arXiv 2506.09985](https://arxiv.org/abs/2506.09985); V-JEPA 2.1 [arXiv 2603.14482](https://arxiv.org/abs/2603.14482); [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- DINOv3: [arXiv 2508.10104](https://arxiv.org/abs/2508.10104); [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Omnilingual ASR: [arXiv 2511.09690](https://arxiv.org/abs/2511.09690); [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr)
- SAM 3: [arXiv 2511.16719](https://arxiv.org/abs/2511.16719); SAM 3D [arXiv 2511.16624](https://arxiv.org/abs/2511.16624); [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SAM Audio: [arXiv 2512.18099](https://arxiv.org/abs/2512.18099); [facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- EUPE: [arXiv 2603.22387](https://arxiv.org/abs/2603.22387); MoE-ViE [arXiv 2608.17402](https://arxiv.org/abs/2608.17402)
- Hugging Face Hub API, `facebook` org model listing (creation dates, license tags, safetensors totals), fetched October 23, 2026
