---
title: "The Mistral Family, Release by Release: From Mistral 7B to Timestral"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of every Mistral AI open-weight release from Mistral 7B in September 2023 to Timestral in October 2026, with dates, parameter counts, context windows, the license behind each checkpoint, and primary sources."
tags: ["mistral", "mixtral", "mistral-small", "devstral", "magistral", "voxtral", "open-weights", "model-family", "timeline", "licensing"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 20
---

Mistral AI made "open weights" a business model in Europe. Its first release, Mistral 7B, went out on September 27, 2023 under Apache 2.0 with the line "it can be used without restrictions." Three years later the `mistralai` Hugging Face organization holds 81 repositories, from 675B MoE language models to speech, a Lean 4 prover and time-series forecasters. The licenses changed more than the architectures did. Mistral moved from Apache 2.0 to two restrictive licenses in 2024, promised in January 2025 to "progressively move away from MRL-licensed models," and then introduced a Modified MIT license with a revenue exception for its largest dense models.

A Mistral checkpoint's name tells you its family and date stamp (`2407` is July 2024, `2512` is December 2025), but not whether you can ship it in a product. This post walks every open-weight release in date order: what changed, sizes and context, license, and a primary source. The timeline below is the mental model.

![Mistral release timeline from September 2023 to October 2026, with open-weight releases colored by license in the top lane and hosted-only models in the bottom lane](/imgs/blogs/mistral-model-family-timeline-1.webp)

Notice the shaded band from July to November 2024, when five of nine open releases shipped under the research license. The answer people ask for most: **as of October 23, 2026, the newest open-weight Mistral release is Timestral (October 15, 2026, four forecasting checkpoints from 5M to 209M parameters), and the newest open general-purpose LLM is Mistral Medium 3.5 (May 22, 2026, dense 128B, Modified MIT).** The newest Apache 2.0 general model is Mistral Small 4 (March 16, 2026).

Method: I pulled the Hugging Face Hub listing for `mistralai` sorted by creation date, the model card of each repository, and each announcement on `mistral.ai/news`, reading the `datePublished` field from the page. Dates are announcement dates. Where no announcement exists I write "Hub created", a lower bound on the public date. Benchmarks are vendor-reported.

## The timeline at a glance

| Date | Release | Sizes | License |
| --- | --- | --- | --- |
| 2023-09-27 | Mistral 7B v0.1 | 7.3B dense | Apache 2.0 |
| 2023-12-11 | Mixtral 8x7B; Mistral 7B Instruct v0.2 | 46.7B total, 12.9B active | Apache 2.0 |
| 2024-04-17 | Mixtral 8x22B | 141B total, 39B active | Apache 2.0 |
| 2024-05-22 (Hub) | Mistral 7B v0.3 | 7B | Apache 2.0 |
| 2024-05-29 | Codestral 22B | 22B dense | Non-Production (MNPL) |
| 2024-07-16 | Mathstral 7B; Codestral Mamba 7B | 7B each | Apache 2.0 |
| 2024-07-18 | Mistral NeMo 12B | 12B dense | Apache 2.0 |
| 2024-07-24 | Mistral Large 2 | 123B dense | Research (MRL) |
| 2024-09-17 | Pixtral 12B | 12B + 400M vision | Apache 2.0 |
| 2024-09-17 | Mistral Small 2409 | 22B dense | MRL |
| 2024-10-16 | Ministral 8B | 8B | MRL |
| 2024-11-18 | Pixtral Large; Mistral Large 2411 | 124B; 123B | MRL |
| 2025-01-30 | Mistral Small 3 | 24B dense | Apache 2.0 |
| 2025-03-17 | Mistral Small 3.1 | 24B + vision | Apache 2.0 |
| 2025-05-21 | Devstral Small 1.0 | 24B | Apache 2.0 |
| 2025-06-10 | Magistral Small 1.0 | 24B | Apache 2.0 |
| 2025-06-19 (Hub) | Mistral Small 3.2 | 24B + vision | Apache 2.0 |
| 2025-07-10 | Devstral Small 1.1 | 24B | Apache 2.0 |
| 2025-07-15 | Voxtral Mini; Voxtral Small | 3B; 24B | Apache 2.0 |
| 2025-07-18, 09-12 (Hub) | Magistral Small 1.1, 1.2 | 24B | Apache 2.0 |
| 2025-12-02 | Mistral Large 3; Ministral 3 | 675B total, 41B active; 3B to 14B | Apache 2.0 |
| 2025-12-09 | Devstral 2; Devstral Small 2 | 123B dense; 24B | Modified MIT; Apache 2.0 |
| 2026-02-04 | Voxtral Mini 4B Realtime | 4B | Apache 2.0 |
| 2026-03-16 | Mistral Small 4; Leanstral | 119B total, 6.5B active | Apache 2.0 |
| 2026-03-23 | Voxtral TTS | 4B | CC BY-NC 4.0 |
| 2026-05-22 | Mistral Medium 3.5 | 128B dense | Modified MIT |
| 2026-07-02 | Leanstral 1.5 | 119B total, 6.5B active | Apache 2.0 |
| 2026-08-04 | Shieldstral 1.0 3B | 3B | Apache 2.0 |
| 2026-10-08 (Hub) | Voxtral Realtime Arabic; LIDstral Arabic | ~4.4B; fastText stack | Apache 2.0 |
| 2026-10-15 | Timestral | 5M to 209M | Modified MIT |

Hosted-only siblings, one line each: Mistral Large (February 26, 2024, hosted only); Ministral 3B (October 16, 2024, hosted only); Codestral 25.01 (January 13, 2025, hosted only); Mistral Saba (February 17, 2025, hosted only); Mistral OCR (March 6, 2025, hosted only); Mistral Medium 3 (May 7, 2025, hosted only); Magistral Medium (June 10, 2025, hosted only); Devstral Medium (July 10, 2025, hosted only); Voxtral Mini Transcribe V2 (February 4, 2026, hosted only).

## Mistral 7B (September 27, 2023)

A 7.3B dense model that, by Mistral's own re-evaluation, "outperforms Llama 2 13B on all benchmarks" and on MMLU performed like a Llama 2 more than 3x its size. Architecture details are in arXiv 2310.06825.

| Variant | Context | License |
| --- | --- | --- |
| Mistral-7B-v0.1, Instruct-v0.1 | 8k | Apache 2.0 |

Source: [Announcing Mistral 7B](https://mistral.ai/news/announcing-mistral-7b).

## Mixtral 8x7B and Mistral 7B Instruct v0.2 (December 11, 2023)

The jump to sparse mixture-of-experts. Mixtral has 46.7B total parameters but "only uses 12.9B parameters per token," so it runs at roughly 13B-dense cost while matching or beating Llama 2 70B and GPT-3.5 on most of Mistral's benchmarks, with 6x faster inference than Llama 2 70B. Context went to 32k; the instruct model used SFT plus DPO. The same day Mistral shipped Mistral 7B Instruct v0.2 with a "32k context window (vs 8k context in v0.1)."

| Variant | Total / active | Context | License |
| --- | --- | --- | --- |
| Mixtral-8x7B v0.1 base, Instruct | 46.7B / 12.9B | 32k | Apache 2.0 |
| Mistral-7B-Instruct-v0.2 | 7B dense | 32k | Apache 2.0 |

Sources: [Mixtral of experts](https://mistral.ai/news/mixtral-of-experts), [La Plateforme](https://mistral.ai/news/la-plateforme).

## Mixtral 8x22B (April 17, 2024)

The scaled-up MoE: 141B total, 39B active, 64k context, stronger code and math, native French, German, Spanish and Italian. It was the last flagship-sized Mistral LLM under Apache 2.0 until December 2025. A follow-up, `Mistral-7B-v0.3` (Hub created May 22, 2024), is v0.2 with the vocabulary extended to 32,768.

| Variant | Total / active | Context | License |
| --- | --- | --- | --- |
| Mixtral-8x22B v0.1 base, Instruct | 141B / 39B | 64k | Apache 2.0 |

Source: [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b).

## Codestral 22B (May 29, 2024)

Mistral's "first-ever code model": 22B dense, 80+ programming languages, a 32k window. The license is the story. Codestral shipped under "the new Mistral AI Non-Production License," research and testing only, with commercial licenses on request. It was the first Mistral checkpoint you could not put in production for free.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Codestral-22B-v0.1 | 22B dense | 32k | MNPL-0.1 |

Source: [Codestral](https://mistral.ai/news/codestral).

## Mathstral 7B and Codestral Mamba 7B (July 16, 2024)

Two Apache 2.0 specialists. Mathstral "stands on the shoulders of Mistral 7B": 56.6% on MATH and 63.47% on MMLU, rising to 68.37% on MATH with majority voting and 74.59% with a reward model picking among 64 candidates. Codestral Mamba is an architecture experiment, a Mamba2 state-space model tested on in-context retrieval up to 256k tokens.

| Variant | Base | Headline | License |
| --- | --- | --- | --- |
| Mathstral-7B-v0.1 | Mistral 7B | MATH 56.6% | Apache 2.0 |
| Mamba-Codestral-7B-v0.1 | Mamba2 | 256k retrieval tested | Apache 2.0 |

Sources: [Mathstral](https://mistral.ai/news/mathstral), [Codestral Mamba](https://mistral.ai/news/codestral-mamba).

## Mistral NeMo 12B (July 18, 2024)

A 12B dense model "built in collaboration with NVIDIA": 128k context, quantization-aware training for lossless FP8 inference, and a tokenizer with 2^17 (about 128k) entries, the Tekken vocabulary later models reuse. Pixtral 12B's decoder is built on it.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Nemo-Base-2407, Instruct-2407 | 12B | 128k | Apache 2.0 |

Source: [Mistral NeMo](https://mistral.ai/news/mistral-nemo).

## Mistral Large 2 (July 24, 2024)

The first open-weight Mistral flagship, and the start of the research-license era. 123B dense, sized "for single-node inference," 128k context, 80+ coding languages,. The Mistral Research License "allows usage and modification for research and non-commercial usages"; commercial self-deployment needs a paid license. The Hub repository is gated.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Large-Instruct-2407 | 123B dense | 128k | MRL (gated) |

Source: [Large Enough](https://mistral.ai/news/mistral-large-2407).

## Pixtral 12B and Mistral Small 2409 (September 17, 2024)

One announcement, opposite licenses. Pixtral 12B, the first vision model, pairs a NeMo-based 12B decoder with a new 400M vision encoder that reads images at native resolution in 16x16 patches, multiple images in a 128k window, 52.5% on MMMU, Apache 2.0. Its weights hit the Hub on September 11, six days before the post; the page now says it "is no longer maintained." Mistral Small v24.09 is a 22B model under the MRL.

| Variant | Size | License |
| --- | --- | --- |
| Pixtral-12B-2409 | 12B + 400M encoder | Apache 2.0 |
| Mistral-Small-Instruct-2409 | 22B dense | MRL |

Sources: [Pixtral 12B](https://mistral.ai/news/pixtral-12b), [September 2024 release](https://mistral.ai/news/september-24-release).

## Ministral 8B (October 16, 2024)

On Mistral 7B's first anniversary came two edge models, Ministral 3B and 8B. Only one got weights: "The model weights for Ministral 8B Instruct are available for research use." The Hub agrees: `Ministral-8B-Instruct-2410` under the MRL, and no 2410 Ministral 3B repository.

| Variant | Weights | License |
| --- | --- | --- |
| Ministral-8B-Instruct-2410 | Open | MRL |

Source: [Un Ministral, des Ministraux](https://mistral.ai/news/ministraux).

## Pixtral Large and Mistral Large 2411 (November 18, 2024)

Pixtral Large is a 124B multimodal model built on Large 2 (123B decoder plus 1B vision encoder), 128k context fitting at least 30 high-resolution images, 69.4% on MathVista. Mistral Large 24.11 updated Large 2 with better long context, a new system prompt and more accurate function calling. Both MRL: the high-water mark of the restrictive era.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Pixtral-Large-Instruct-2411 | 123B + 1B encoder | 128k | MRL |
| Mistral-Large-Instruct-2411 | 123B dense | 128k | MRL |

Source: [Pixtral Large](https://mistral.ai/news/pixtral-large).

## Mistral Small 3 (January 30, 2025)

The license reversal. A 24B dense, latency-optimized model with "far fewer layers than competing models," over 81% on MMLU at 150 tokens per second, pitched against Llama 3.3 70B. Base and instruct went out under Apache 2.0 with a pledge: "We're renewing our commitment to using Apache 2.0 license for our general purpose models, as we progressively move away from MRL-licensed models." It used neither RL nor synthetic data.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Small-24B-Base-2501, Instruct-2501 | 24B | 32k | Apache 2.0 |

Source: [Mistral Small 3](https://mistral.ai/news/mistral-small-3).

## Mistral Small 3.1 (March 17, 2025)

Small 3 plus vision and a 128k window, still 24B and Apache 2.0. It became the base of most 2025 specialists: Devstral, Magistral 1.0 and 1.1, and Small 3.2 all name it as parent.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Small-3.1-24B Base-2503, Instruct-2503 | 24B + vision | 128k | Apache 2.0 |

Source: [Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1).

## Devstral Small 1.0 (May 21, 2025)

An agentic coding model built with All Hands AI, fine-tuned from Small 3.1, text-only, small enough for "a single RTX 4090 or a Mac with 32GB RAM." It scored 46.8% on SWE-Bench Verified, more than 6 points above the prior open best.

| Variant | Size | SWE-Bench Verified | License |
| --- | --- | --- | --- |
| Devstral-Small-2505 | 24B | 46.8% | Apache 2.0 |

Source: [Devstral](https://mistral.ai/news/devstral).

## Magistral Small 1.0 (June 10, 2025)

Mistral's "first reasoning model," a dual release. Magistral Small is Small 3.1 with SFT on Magistral Medium traces then RL: 70.7% on AIME 2024 (83.3% with majority voting at 64), against 73.6% and 90% for the hosted Medium. The card warns quality may degrade past 40k of its 128k window.

| Variant | Size | AIME24 (maj@64) | License |
| --- | --- | --- | --- |
| Magistral-Small-2506 | 24B | 70.7% (83.3%) | Apache 2.0 |

Source: [Magistral](https://mistral.ai/news/magistral).

## Mistral Small 3.2 (Hub created June 19, 2025)

A "minor update" of Small 3.1 with no announcement: better instruction following, fewer repetitive generations, a more robust function-calling template.

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Small-3.2-24B-Instruct-2506 | 24B + vision | 128k | Apache 2.0 |

Source: [model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506).

## Devstral Small 1.1 (July 10, 2025)

Same architecture, 53.6% on SWE-Bench Verified, 6.8 points over 1.0. The small model stays Apache 2.0; the stronger Devstral Medium stays on the API.

| Variant | Size | SWE-Bench Verified | License |
| --- | --- | --- | --- |
| Devstral-Small-2507 | 24B | 53.6% | Apache 2.0 |

Source: [Devstral Small 1.1](https://mistral.ai/news/devstral-2507).

## Voxtral Mini and Voxtral Small (July 15, 2025)

The first speech-understanding models: 3B Mini for edge, 24B Small for production. A 32k-token context covers up to 30 minutes of audio for transcription or 40 for understanding. Paper: arXiv 2507.13264.

| Variant | Size | License |
| --- | --- | --- |
| Voxtral-Mini-3B-2507 | 3B | Apache 2.0 |
| Voxtral-Small-24B-2507 | 24B | Apache 2.0 |

Source: [Voxtral](https://mistral.ai/news/voxtral).

## Magistral Small 1.1 and 1.2 (Hub created July 18 and September 12, 2025)

Two point releases without posts. 1.1 still sits on Small 3.1. 1.2 moves to Small 3.2 and adds a vision encoder, and its card now recommends the full 128k window.

| Variant | Base | New | License |
| --- | --- | --- | --- |
| Magistral-Small-2507 | Small 3.1 | Updated weights | Apache 2.0 |
| Magistral-Small-2509 | Small 3.2 | Vision input | Apache 2.0 |

Sources: [2507 card](https://huggingface.co/mistralai/Magistral-Small-2507), [2509 card](https://huggingface.co/mistralai/Magistral-Small-2509).

## Mistral 3: Large 3 and Ministral 3 (December 2, 2025)

The release that completed the return to Apache 2.0 at flagship scale. Mistral Large 3 is a granular MoE with 675B total and 41B active parameters (673B and 39B for the language model alone, so the rest is the vision encoder), trained from scratch on 3,000 H200s, 256k context, and "Mistral's first mixture-of-experts model since the seminal Mixtral series." Base, instruct, an NVFP4 checkpoint and an EAGLE head all shipped under Apache 2.0. A promised reasoning version is not on the Hub as of this writing.

Ministral 3 reboots the edge line: dense 3B, 8B and 14B, each in base, instruct and reasoning variants, with vision and 256k context. The 14B reasoning model scores 85% on AIME 2025. Paper: arXiv 2601.08584.

| Variant | Total / active | Context | License |
| --- | --- | --- | --- |
| Mistral-Large-3-675B Base, Instruct | 675B / 41B | 256k | Apache 2.0 |
| Ministral-3 3B, 8B, 14B (x3 variants) | 3B to 14B dense | 256k | Apache 2.0 |

Source: [Introducing Mistral 3](https://mistral.ai/news/mistral-3).

## Devstral 2 and Devstral Small 2 (December 9, 2025)

A week later the coding line split its licenses by size. Devstral 2, 123B dense with 256k context, reaches 72.2% on SWE-bench Verified under "a modified MIT license," its first appearance. Devstral Small 2 (24B) reaches 68.0% under Apache 2.0. The open-source Mistral Vibe CLI shipped alongside.

| Variant | Size | SWE-bench Verified | License |
| --- | --- | --- | --- |
| Devstral-2-123B-Instruct-2512 | 123B dense | 72.2% | Modified MIT |
| Devstral-Small-2-24B-Instruct-2512 | 24B | 68.0% | Apache 2.0 |

Source: [Devstral 2 and Mistral Vibe CLI](https://mistral.ai/news/devstral-2-vibe-cli).

## Voxtral Mini 4B Realtime (February 4, 2026)

Part of Voxtral Transcribe 2, whose batch model is hosted only. The realtime model is open: a 4B natively streaming speech-to-text model, 13 languages, delay configurable from 240ms to 2.4s, and at 480ms matching leading offline open models.

| Variant | Size | Latency | License |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 | 4B | 240ms to 2.4s | Apache 2.0 |

Source: [Voxtral Transcribe 2](https://mistral.ai/news/voxtral-transcribe-2).

## Mistral Small 4 and Leanstral (March 16, 2026)

Small 4 is the consolidation: "the first Mistral model to unify the capabilities of our flagship models, Magistral for reasoning, Pixtral for multimodal, and Devstral for agentic coding," with a `reasoning_effort` switch. It is no longer small: a 119B MoE with 256k context. The blog gives 6B active (8B with embedding and output layers); the card says 6.5B. Leanstral, the same day, is "the first open-source code agent designed for Lean 4," built in the Small 4 family: 128 experts, 4 active per token.

| Variant | Total / active | Context | License |
| --- | --- | --- | --- |
| Mistral-Small-4-119B-2603 | 119B / 6.5B | 256k | Apache 2.0 |
| Leanstral-2603 | 119B / 6.5B | 256k | Apache 2.0 |

Sources: [Mistral Small 4](https://mistral.ai/news/mistral-small-4), [Leanstral](https://mistral.ai/news/leanstral).

## Voxtral TTS (March 23, 2026)

The first text-to-speech model: 4B parameters with a 3.4B decoder backbone, 9 languages, zero-shot voice adaptation. Weights are open but CC BY-NC 4.0, inherited from the non-commercial datasets behind its bundled reference voices. Paper: arXiv 2603.25551.

| Variant | Size | License |
| --- | --- | --- |
| Voxtral-4B-TTS-2603 | 4B | CC BY-NC 4.0 |

Source: [Voxtral TTS](https://mistral.ai/news/voxtral-tts).

## Mistral Medium 3.5 (May 22, 2026)

The first Medium with weights (Medium 3 was hosted only). "Our first flagship merged model": dense 128B, 256k context, instruct, reasoning and coding in one set of weights. The card says it "supersedes all our previous coding models, namely Devstral," at 77.6% on SWE-Bench Verified. The Modified MIT allows commercial use "with exceptions for companies with large revenue."

| Variant | Size | Context | License |
| --- | --- | --- | --- |
| Mistral-Medium-3.5-128B (+ EAGLE) | 128B dense | 256k | Modified MIT |

Source: [Mistral Medium 3.5](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5).

## Leanstral 1.5 (July 2, 2026)

Same 119B MoE shape; Mistral reports it saturates miniF2F, solves 587 of 672 PutnamBench problems, and reaches 87% on FATE-H.

| Variant | Total / active | PutnamBench | License |
| --- | --- | --- | --- |
| Leanstral-1.5-119B-A6B | 119B / 6.5B | 587/672 | Apache 2.0 |

Source: [Leanstral 1.5](https://mistral.ai/news/leanstral-1-5).

## Shieldstral 1.0 3B (August 4, 2026)

A multimodal safety classifier built on Ministral-3-3B-Base with a Pixtral encoder. It scores content against a policy written in natural language, one forward pass, one continuous score. Report: arXiv 2607.25857.

| Variant | Size | License |
| --- | --- | --- |
| Shieldstral-1.0-3B | 3B | Apache 2.0 |

Source: [Shieldstral](https://mistral.ai/news/shieldstral).

## Arabic speech and language ID (Hub created October 8, 2026)

No announcement posts. `Voxtral-Mini-4B-Realtime-Arabic` is fine-tuned from Voxtral Realtime, about 4.4B parameters; its Hub metadata lacks a license tag but the card states Apache 2.0. `LIDstral-Arabic` is not neural at scale: a fastText classifier, a small MLP stacker and isotonic calibration that tell Modern Standard Arabic, dialects and other Arabic-script languages apart.

| Variant | Kind | License |
| --- | --- | --- |
| Voxtral-Mini-4B-Realtime-Arabic | Streaming ASR | Apache 2.0 |
| LIDstral-Arabic | Language ID | Apache 2.0 |

## Timestral (October 15, 2026)

Mistral's "first family of time-series foundation models," at 5M, 28M, 123M and 209M parameters. The 209M model is a 24-layer patch transformer with time and variate attention. It reads up to 32,768 history steps, forecasts up to 1,024 in one pass over 21 quantiles, and Mistral reports first place on GIFT-Eval, TIME and fev-bench.

| Variant | Size | Max history | License |
| --- | --- | --- | --- |
| Timestral 5M, 28M, 123M, 209M | 5M to 209M | 32,768 steps | Modified MIT |

Source: [Timestral](https://mistral.ai/news/timestral).

## What is latest as of October 2026

- **Newest open-weight release:** Timestral (October 15, 2026), Modified MIT.
- **Newest open general LLM:** Mistral Medium 3.5 (May 22, 2026), Modified MIT.
- **Newest Apache 2.0 general LLM:** Mistral Small 4 (March 16, 2026).
- **Largest open model:** Mistral Large 3 (December 2, 2025), 675B MoE, Apache 2.0.
- **Newest edge models:** Ministral 3 (December 2, 2025).

## How the family evolved

### 1. Total parameters grew more than 90x, active parameters did not

![Horizontal bars of total and active parameters for Mistral's main open LLMs, from 7.3B dense to 675B total with 41B active](/imgs/blogs/mistral-model-family-timeline-2.webp)

The longest bar is Large 3 at 675B, about 92x Mistral 7B. Look at the inner bars: Mixtral 8x7B ran 12.9B parameters per token, Large 3 runs 41B, Small 4 about 6.5B of 119B. Mistral alternates two bets: dense models sized for one node (Large 2, Devstral 2, Medium 3.5, all 123B to 128B) and MoE models that buy capacity without per-token compute. "Small 4" has five times Small 3's total parameters and earns its name only by its active count.

### 2. Context windows grew 32x in four steps

![Step chart of advertised context window for Mistral flagship open releases from 8k in 2023 to 256k from December 2025, with Small 3 at 32k as an outlier](/imgs/blogs/mistral-model-family-timeline-3.webp)

8k in September 2023, 32k three months later, 64k with Mixtral 8x22B, 128k from July 2024, 256k from Mistral 3 onward. Small 3 is the dip: shipped at 32k for latency, restored to 128k two months later. Advertised is not usable: Magistral 1.0 advertised 128k and recommended 40k.

### 3. The license story has three acts

![Grid of colored cells, one per open-weight release, grouped by year, showing 2024 split between Apache 2.0 and research licenses and 2025 almost entirely Apache 2.0](/imgs/blogs/mistral-model-family-timeline-4.webp)

Count colors per row. Act one, 2023 to early 2024: all Apache 2.0. Act two, May to November 2024: Codestral under the Non-Production License, then five releases under the Research License, with only the 7B and 12B models still Apache. Act three began with Small 3's pledge, and 2025 delivered: 13 of 14 releases Apache 2.0, including the 675B Large 3. The exception is the new Modified MIT, which then spread to Medium 3.5 and Timestral. It allows commercial use, unlike the MRL, but carves out large-revenue companies, so it is not Apache. Voxtral TTS's CC BY-NC is a data-provenance case, not a policy one.

### 4. Specialists branch off a shared base, then merge back

![Lineage tree showing Mathstral from Mistral 7B, Pixtral from NeMo and Large 2, Devstral and Magistral from Small 3.1 and 3.2, Shieldstral from Ministral 3, all converging into Small 4](/imgs/blogs/mistral-model-family-timeline-5.webp)

Read top to bottom. In 2025 the playbook was one strong 24B base (Small 3.1) and fine-tunes off it for code agents, reasoning and instruction following. In 2026 it reversed: Small 4 folds reasoning, vision and coding into one model, and Medium 3.5 "supersedes" Devstral. Specialists survive where the domain is far from chat: Lean proofs, safety, speech, time series.

### 5. The open line kept a tier below the hosted line

Each year had a hosted model above the open one (Mistral Large in 2024; Medium 3, Magistral Medium, Devstral Medium in 2025) until the first open Medium became the 2026 flagship.

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| General model, permissive license | Mistral Small 4 | Apache 2.0, reasoning switch, vision, 256k |
| Strongest open Mistral | Mistral Large 3 | 675B MoE, Apache 2.0, NVFP4 available |
| Best coding agent, revenue carve-out acceptable | Mistral Medium 3.5 | 77.6% SWE-Bench Verified |
| Coding agent on one consumer GPU | Devstral Small 2 | 24B, 68.0%, Apache 2.0 |
| Edge or on-device | Ministral 3 | 3B to 14B, Apache 2.0 |
| Streaming speech-to-text | Voxtral Mini 4B Realtime | Apache 2.0, sub-500ms |
| Commercial text-to-speech | Not Voxtral TTS weights | CC BY-NC 4.0; use the API |
| Lean 4 proofs | Leanstral 1.5 | Apache 2.0 |
| Moderation with custom policies | Shieldstral 1.0 3B | Apache 2.0 |
| Zero-shot forecasting | Timestral 209M | Check the Modified MIT terms |

Avoid for new work anything under the MRL (Large 2 and 2411, Pixtral Large, Small 2409, Ministral 8B) or the MNPL (Codestral 22B) unless you hold a commercial license. Each has a newer Apache 2.0 model in the same role.

## What this catalog could not verify

- **Public dates for card-only releases.** Mistral 7B v0.3, Small 3.2, Magistral 1.1 and 1.2, and the Arabic models have no announcement I found; their dates are Hub creation dates, a lower bound.
- **Small 4 and Leanstral active parameters.** The Small 4 blog says 6B (8B with embedding and output layers), the cards 6.5B, the Leanstral blog "120B-A6B." Tables use the card figure.
- **Timestral's date.** The page metadata reads July 15, 2026; the visible byline reads October 15 and the Hub repositories were created October 14. I use October 15.
- **Mistral Large 2's card** is gated, so its specs come from the announcement.

## References

- Mistral AI news, 2023 to 2024: [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b), [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts), [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b), [Codestral](https://mistral.ai/news/codestral), [Codestral Mamba](https://mistral.ai/news/codestral-mamba), [Mathstral](https://mistral.ai/news/mathstral), [NeMo](https://mistral.ai/news/mistral-nemo), [Large 2](https://mistral.ai/news/mistral-large-2407), [Pixtral 12B](https://mistral.ai/news/pixtral-12b), [Small 2409](https://mistral.ai/news/september-24-release), [Ministraux](https://mistral.ai/news/ministraux), [Pixtral Large](https://mistral.ai/news/pixtral-large).
- Mistral AI news, 2025 to 2026: [Small 3](https://mistral.ai/news/mistral-small-3), [Small 3.1](https://mistral.ai/news/mistral-small-3-1), [Devstral](https://mistral.ai/news/devstral), [Magistral](https://mistral.ai/news/magistral), [Devstral 1.1](https://mistral.ai/news/devstral-2507), [Voxtral](https://mistral.ai/news/voxtral), [Mistral 3](https://mistral.ai/news/mistral-3), [Devstral 2](https://mistral.ai/news/devstral-2-vibe-cli), [Voxtral Transcribe 2](https://mistral.ai/news/voxtral-transcribe-2), [Small 4](https://mistral.ai/news/mistral-small-4), [Leanstral](https://mistral.ai/news/leanstral), [Voxtral TTS](https://mistral.ai/news/voxtral-tts), [Medium 3.5](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5), [Leanstral 1.5](https://mistral.ai/news/leanstral-1-5), [Shieldstral](https://mistral.ai/news/shieldstral), [Timestral](https://mistral.ai/news/timestral).
- Hosted-only: [Mistral Large](https://mistral.ai/news/mistral-large), [Codestral 25.01](https://mistral.ai/news/codestral-2501), [Saba](https://mistral.ai/news/mistral-saba), [OCR](https://mistral.ai/news/mistral-ocr), [Medium 3](https://mistral.ai/news/mistral-medium-3).
- Hugging Face: the [mistralai organization](https://huggingface.co/mistralai), every model card named above, and the Hub API listing sorted by `createdAt`.
- Papers: [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) (Mistral 7B), [2507.13264](https://arxiv.org/abs/2507.13264) (Voxtral), [2601.08584](https://arxiv.org/abs/2601.08584) (Ministral 3), [2603.25551](https://arxiv.org/abs/2603.25551) (Voxtral TTS), [2607.25857](https://arxiv.org/abs/2607.25857) (Shieldstral).
