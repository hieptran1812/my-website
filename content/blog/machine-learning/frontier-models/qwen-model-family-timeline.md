---
title: "The Qwen Model Family, Release by Release: From Qwen-7B to Qwen3.8"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference catalog of every Qwen generation from August 2023 to October 2026: release dates, what changed each time, variant tables with sizes, context and licenses, and links to primary sources and our deep-dives."
tags: ["qwen", "qwen3", "qwen3-5", "large-language-model", "mixture-of-experts", "hybrid-attention", "open-weights", "model-family", "timeline", "alibaba"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 32
---

Ask five engineers which Qwen model to deploy and you will get five answers, and at least two of them will name a checkpoint that has already been superseded. That is not carelessness. Between August 2023 and October 2026 Alibaba's Qwen team shipped six major version numbers, three point releases, a dozen specialist lines (coder, math, vision, omni, audio, speech, guard, translation, image, embedding) and two separate "thinking" product lines that were later folded back into the main models. The model cards are excellent individually. Collectively they are a maze, because each card describes itself relative to the one before it and nobody keeps the whole map.

This post is that map. It walks the family in strict chronological order, from Qwen-7B to Qwen3.8, and for each generation it records the release date, what actually changed relative to the previous generation (architecture, data, context, post-training recipe, license), and a variant table. Every date and spec below was checked against a primary source: the arXiv technical report, the official Qwen blog, the QwenLM GitHub README news list, or the official Hugging Face model card. Where a primary source does not state a fact, the post says so instead of guessing. The timeline figure below is the mental model: a generation every six to eight months until Qwen3, then point releases every two to four months on a frozen architecture.

![Qwen release timeline from August 2023 to August 2026, transformer era in blue and hybrid-attention era in green](/imgs/blogs/qwen-model-family-timeline-1.webp)

The figure spaces milestones evenly for readability, so read the dates in the cards rather than the gaps between them. Where this blog already has a deep-dive on a release, it is linked inline. The comparison against Llama, Gemma and DeepSeek lives in [Modern LLM architectures](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek); this post stays inside one family.

## The release timeline at a glance

The table is the whole catalog in one screen. The rest of the post expands each row. Dates are the official announcement (blog post, GitHub news entry, or arXiv v1) unless marked "~", which means the earliest primary evidence is the date the Hugging Face repository was created; a repo can be created privately a few days before launch, so those dates are a floor.

| Date | Release | Headline change | Weights |
| --- | --- | --- | --- |
| 2023-08-03 | Qwen-7B / Qwen-7B-Chat | First open Qwen, 151K tiktoken vocab | Open (Tongyi Qianwen license) |
| 2023-08-24 | Qwen-VL | ViT-bigG + cross-attention adapter on Qwen-7B | Open |
| 2023-09-25 | Qwen-14B, Qwen-7B refresh | 7B retrained to 2.4T tokens, 8K context | Open |
| 2023-11-30 | Qwen-72B, Qwen-1.8B, Qwen-Audio | Full size ladder, first audio model | Open |
| 2024-02-04 | Qwen1.5 (0.5B to 72B) | 32K everywhere, native in `transformers` | Open |
| 2024-03-28 | Qwen1.5-MoE-A2.7B | First Qwen MoE, upcycled from Qwen-1.8B | Open |
| 2024-04 | Qwen1.5-32B, CodeQwen1.5, Qwen1.5-110B | GQA arrives, first 100B+ model | Open |
| 2024-06-07 | Qwen2 | GQA on every size, about 7T tokens, Apache 2.0 for most sizes | Open |
| 2024-07/08 | Qwen2-Audio, Qwen2-Math, Qwen2-VL | M-RoPE, dynamic resolution, GRPO in Math | Open (VL-72B API first) |
| 2024-09-19 | Qwen2.5 + Coder + Math | 18T tokens, 128K, multistage RL | Open (3B and 72B custom licenses) |
| 2024-11-15 | Qwen2.5-Turbo 1M | 1M-token context | API only |
| 2024-11-28 | QwQ-32B-Preview | First Qwen reasoning model | Open |
| 2024-12-25 | QVQ-72B-Preview | Visual reasoning | Open |
| 2025-01-26 | Qwen2.5-VL | Window-attention ViT, hour-long video, agent use | Open |
| 2025-01-27 | Qwen2.5-1M | First open 1M-context models | Open |
| 2025-01-28 | Qwen2.5-Max | Large MoE, 20T+ tokens | API only |
| 2025-03-06 | QwQ-32B | Outcome-based RL, R1-class reasoning at 32B | Open |
| 2025-03-27 | Qwen2.5-Omni-7B | Thinker-Talker, streaming speech out | Open |
| 2025-04-29 | Qwen3 | 36T tokens, 119 languages, hybrid thinking, 235B-A22B | Open (Apache 2.0) |
| ~2025-06-05 | Qwen3-Embedding / Reranker | Retrieval models on Qwen3 | Open |
| 2025-07-21 to 08-06 | Qwen3-2507 | Instruct / Thinking split, 256K | Open |
| 2025-07-22 | Qwen3-Coder-480B-A35B | Agentic coding MoE, 256K native | Open |
| 2025-07-24 | Qwen-MT | Translation across 92 languages | API only |
| 2025-08-04 | Qwen-Image | 20B MMDiT text-to-image | Open |
| 2025-09-11 | Qwen3-Next-80B-A3B | Hybrid Gated DeltaNet attention, 512 experts | Open |
| 2025-09-22 | Qwen3-Omni-30B-A3B | Omni-modal MoE | Open |
| ~2025-09-22 | Qwen3-VL (235B to 2B through Oct) | 256K multimodal context | Open |
| ~2025-09-23 | Qwen3Guard | Streaming safety classifier | Open |
| 2025-09 | Qwen3-Max | Over 1T parameters (as announced) | API only |
| ~2026-01-21 | Qwen3-TTS | 0.6B and 1.7B speech synthesis | Open |
| ~2026-01-28 | Qwen3-ASR | 0.6B and 1.7B recognizer + aligner | Open |
| 2026-01-30 | Qwen3-Coder-Next | 80B-A3B hybrid coder | Open |
| 2026-02-16 | Qwen3.5-397B-A17B | Hybrid architecture goes mainline, native multimodal, 201 languages | Open (Apache 2.0) |
| 2026-02-24 / 03-02 | Qwen3.5 mid and small sizes | 122B-A10B down to 0.8B | Open |
| 2026-04-16 / 04-22 | Qwen3.6 | Agentic coding focus, `preserve_thinking` | Open |
| 2026 (date n/d) | Qwen3.7-Max | Appears only as an API comparison column | API only |
| 2026-08-12 / 08-14 | Qwen3.8-2.4T-A95B, Qwen3.8-27B | First open Max-class model | Open (2.4T under a custom license) |
| ~2026-08-24 | Qwen3.8-Flash-Next | Preview of the Qwen4 architecture | Open (qwen-community-1.0) |

That is 38 rows. Qwen4 has not been released as of 2026-10-14; the only official mention of it is the Qwen3.8-Flash-Next card describing itself as a preview of the architecture that "will underpin Qwen4".

## 1. Qwen (August to November 2023): the first generation

**Released:** Qwen-7B and Qwen-7B-Chat on 2023-08-03, Qwen-14B on 2023-09-25, Qwen-72B and Qwen-1.8B on 2023-11-30 (all dates from the [QwenLM/Qwen README](https://github.com/QwenLM/Qwen) news list). Technical report: [arXiv 2309.16609](https://arxiv.org/abs/2309.16609), submitted 2023-09-28.

Qwen (the brand is Tongyi Qianwen) entered a market where Llama 2 had just reset expectations for open weights. The base architecture was deliberately conventional: a decoder-only transformer with RoPE, SwiGLU and RMSNorm, the Llama recipe. The differentiators were elsewhere.

- **Tokenizer.** A byte-level BPE built from tiktoken's `cl100k_base` and extended for Chinese and multilingual text, with numbers split into single digits. Vocabulary: 151,851 tokens. That vocabulary, give or take control tokens, would survive four generations, and it is one reason Qwen models are cheap to run on Chinese text: fewer tokens per character than a Llama tokenizer.
- **Context tricks instead of long-context training.** The models were trained at 2K to 8K and pushed further at inference with NTK-aware interpolation, LogN attention scaling and window attention. This was the 2023 state of the art and it would be replaced wholesale in Qwen2.
- **A refresh mid-generation.** On 2023-09-25 the 7B was retrained: pretraining went from 2.2T to 2.4T tokens and the context from 2,048 to 8,192 tokens. If you find an old Qwen-7B fine-tune with odd long-context behavior, check which revision it was built on.
- **Chat alignment** was SFT followed by RLHF, and the report also covers Code-Qwen, Code-Qwen-Chat and Math-Qwen-Chat plus tool use through ReAct-style prompting and a code interpreter.

| Model | Pretrain tokens | Max length | License | Released |
| --- | --- | --- | --- | --- |
| Qwen-1.8B (+Chat) | 2.2T | 32K | Tongyi Qianwen Research License | 2023-11-30 |
| Qwen-7B (+Chat) | 2.4T (after refresh) | 8K on the card, 32K in the README table | Tongyi Qianwen License | 2023-08-03 |
| Qwen-14B (+Chat) | 3.0T | 8K | Tongyi Qianwen License | 2023-09-25 |
| Qwen-72B (+Chat) | 3.0T | 32K | Tongyi Qianwen License | 2023-11-30 |

The Qwen-7B context discrepancy is in the official pages themselves (the model card says 8,192, the README table says 32K); the two numbers likely reflect the native training length versus the extended inference length. The Tongyi Qianwen License allowed commercial use after an application; the 1.8B shipped under a research-only variant.

### The first multimodal branches

| Model | Released | Architecture | Source |
| --- | --- | --- | --- |
| Qwen-VL / Qwen-VL-Chat | arXiv 2023-08-24 | OpenCLIP ViT-bigG (1.9B) + a 0.08B single-layer cross-attention adapter that compresses image features to 256 query tokens, on Qwen-7B; 9.6B total, 448px input | [arXiv 2308.12966](https://arxiv.org/abs/2308.12966) |
| Qwen-Audio / Qwen-Audio-Chat | arXiv 2023-11-14, weights 2023-11-30 | Whisper-large-v2 encoder on Qwen-7B, about 8B total, pretrained on 30+ audio tasks | [arXiv 2311.07919](https://arxiv.org/abs/2311.07919) |

Both established a pattern Qwen kept for two years: specialist models built by bolting a strong pretrained encoder onto the current Qwen LLM, released under the same family name a few weeks or months after the text model.

## 2. Qwen1.5 (February to April 2024): the cleanup release

**Released:** 2024-02-04 ([Qwen1.5 blog](https://qwenlm.github.io/blog/qwen1.5/)). No separate technical report.

Qwen1.5 is best read as a beta of Qwen2 shipped early. The architecture barely moved. What changed was everything around it:

- **32K context on every size**, trained rather than extrapolated.
- **Merged into Hugging Face `transformers` 4.37.** The Qwen 1 models required `trust_remote_code=True`; Qwen1.5 did not. That one change is a large part of why Qwen fine-tunes exploded in 2024.
- **Quantized releases from day one**: GPTQ Int4 and Int8, AWQ and GGUF, published by the Qwen team itself.
- **Preference alignment moved to DPO and PPO.** Qwen1.5-72B-Chat reported 8.61 on MT-Bench.

| Model | Released | Params | Context | Notes |
| --- | --- | --- | --- | --- |
| Qwen1.5-0.5B / 1.8B / 4B / 7B / 14B / 72B | 2024-02-04 | dense | 32K | Base + Chat; 0.5B under the research license, 7B and 72B under the Tongyi Qianwen license |
| Qwen1.5-MoE-A2.7B | 2024-03-28 | 14.3B total, 2.7B active | 32K | 60 routed experts (4 active) + 4 shared; upcycled from Qwen-1.8B ([blog](https://qwenlm.github.io/blog/qwen-moe/)) |
| Qwen1.5-32B | 2024-04-02 | dense | 32K | First Qwen with grouped-query attention (GQA) ([blog](https://qwenlm.github.io/blog/qwen1.5-32b/)) |
| CodeQwen1.5-7B | 2024-04-16 | dense | 64K | About 3T code tokens, 92 programming languages ([blog](https://qwenlm.github.io/blog/codeqwen1.5/)) |
| Qwen1.5-110B | 2024-04-25 | dense, GQA | 32K | First Qwen above 100B ([blog](https://qwenlm.github.io/blog/qwen1.5-110b/)) |

Two of these rows matter far more than their size suggests. The MoE-A2.7B was Qwen's first mixture of experts, and its shape (many fine-grained experts plus always-on shared experts, initialized by upcycling a dense model) is the direct ancestor of every Qwen MoE that followed. The 32B introduced GQA, which shrinks the KV cache by sharing key and value heads across groups of query heads. Qwen2 would make GQA universal.

## 3. Qwen2 (June to September 2024): the first "real" generation

**Released:** 2024-06-07 ([Qwen2 blog](https://qwenlm.github.io/blog/qwen2/)). Technical report: [arXiv 2407.10671](https://arxiv.org/abs/2407.10671), 2024-07-15.

Qwen2 is where the family stopped iterating on Qwen 1 and became its own design. Relative to Qwen1.5:

- **GQA on every size**, which cut long-context KV memory across the board.
- **About 7T pretraining tokens** for most sizes (12T for the 0.5B; the MoE got an extra 4.5T on top of its upcycled start), with heavier code and math mixing.
- **Proper long-context training.** Context went from 4K to 32K in the final pretraining phase, the RoPE base frequency moved from 10,000 to 1,000,000, and YaRN plus Dual Chunk Attention extended the larger models to 128K. This replaced the Qwen 1 inference-time tricks.
- **29 languages** explicitly targeted.
- **Post-training:** SFT on more than 500K examples, offline DPO, then online DPO against a reward model, with an "Online Merging Optimizer" to reduce the alignment tax. Data synthesis used rejection sampling, execution feedback and back-translation.
- **License:** Apache 2.0 for everything except the 72B, which stayed on the Tongyi Qianwen license.

| Model | Total (non-embedding) | Context | Tokens | License |
| --- | --- | --- | --- | --- |
| Qwen2-0.5B | 0.49B (0.35B), tied embeddings | 32K | 12T | Apache 2.0 |
| Qwen2-1.5B | 1.54B (1.31B), tied embeddings | 32K | 7T | Apache 2.0 |
| Qwen2-7B | 7.07B (5.98B) | 128K | 7T | Apache 2.0 |
| Qwen2-57B-A14B (MoE) | 57.41B total, 14B active | 64K | +4.5T | Apache 2.0 |
| Qwen2-72B | 72.71B (70.21B) | 128K | 7T | Tongyi Qianwen |

The MoE layout was 64 routed experts with 8 active plus 8 shared experts, 28 layers and 4 KV heads. Every size shipped as base and Instruct.

### Qwen2 specialists

| Model | Released | What it added | Source |
| --- | --- | --- | --- |
| Qwen2-Audio-7B | 2024-07-15 | Voice-chat and audio-analysis modes with no mode prompt; DPO; Apache 2.0 | [arXiv 2407.10759](https://arxiv.org/abs/2407.10759) |
| Qwen2-Math 1.5B / 7B / 72B | 2024-08-08 | Math reward model, rejection-sampling SFT, and GRPO; English only | [blog](https://qwenlm.github.io/blog/qwen2-math/) |
| Qwen2-VL 2B / 7B / 72B | 2024-08-29 | Naive Dynamic Resolution, M-RoPE (temporal, height, width), 20-minute-plus video | [blog](https://qwenlm.github.io/blog/qwen2-vl/), [arXiv 2409.12191](https://arxiv.org/abs/2409.12191) |

Qwen2-VL is the one to know. Naive Dynamic Resolution feeds an image at its native resolution as a variable number of visual tokens instead of resizing it to a fixed square, and M-RoPE gives each visual token a three-part position (time, row, column) so the same rotary scheme covers text, images and video. Both ideas carried into every later Qwen vision model. Our deep-dive: [Qwen2-VL: perception of the world at any resolution](/blog/machine-learning/frontier-models/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution). The 2B and 7B were Apache 2.0 at launch; the 72B launched API-only and its weights are now on Hugging Face under the Qwen license, but the exact date they went public is not stated in a primary source I could find.

Qwen2-Math is the first appearance of GRPO in a Qwen release, months before DeepSeek-R1 made the algorithm famous.

## 4. Qwen2.5 (September 2024 to March 2025): scale, specialists and reasoning

**Released:** 2024-09-19 ([Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/)). Technical report: [arXiv 2412.15115](https://arxiv.org/abs/2412.15115), 2024-12-19.

Qwen2.5 kept the Qwen2 architecture and spent the six months on data and post-training:

- **18T pretraining tokens**, up from about 7T. This is the single largest data jump in the family's history in relative terms (about 2.6x).
- **128K context** on the 7B and larger (32K native, YaRN factor 4 to 131,072), with 8K generation length. The 3B stayed at 32K.
- **New sizes at 3B, 14B and 32B**, filling the gaps that hobbyist and single-GPU users had been asking for. The 32B became the most fine-tuned Qwen of the era.
- **Post-training:** SFT on more than 1M samples followed by multistage reinforcement learning. The report abstract confirms "multistage RL" but does not name every algorithm in it.
- **License:** Apache 2.0 for every size except the 3B (Qwen Research license) and the 72B (Qwen license).

| Model | Context | License | Notes |
| --- | --- | --- | --- |
| Qwen2.5-0.5B / 1.5B | 32K | Apache 2.0 | Base + Instruct |
| Qwen2.5-3B | 32K | Qwen Research | Base + Instruct |
| Qwen2.5-7B / 14B / 32B | 128K | Apache 2.0 | Base + Instruct |
| Qwen2.5-72B | 128K | Qwen | 80 layers, 64 query / 8 KV heads |
| Qwen2.5-Turbo, Qwen2.5-Plus | n/d | API only | MoE models; sizes not disclosed |

### Coder and Math

| Model | Released | Facts | Source |
| --- | --- | --- | --- |
| Qwen2.5-Coder 1.5B / 7B | 2024-09-19 | 5.5T code-related tokens | [arXiv 2409.12186](https://arxiv.org/abs/2409.12186) |
| Qwen2.5-Coder 0.5B / 3B / 14B / 32B | 2024-11-12 | 0.5B and 3B at 32K, 14B and 32B at 128K; Apache 2.0 except the 3B | [blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/) |
| Qwen2.5-Math 1.5B / 7B / 72B | 2024-09-19 | Self-improvement pipeline: earlier Qwen math models generate pretraining data and an iterated math reward model guides SFT and RL | [arXiv 2409.12122](https://arxiv.org/abs/2409.12122) |

Qwen2.5-Coder-32B-Instruct was, for roughly half a year, the default open coding model for anyone with a single 80 GB GPU.

### The long tail of Qwen2.5

Qwen2.5 kept spawning releases for six months, and this is where the family's naming first got confusing.

| Release | Date | Key facts | Weights | Source |
| --- | --- | --- | --- | --- |
| Qwen2.5-Turbo (1M) | 2024-11-15 | Context raised from 128K to 1M; time to first token on 1M tokens dropped from 4.9 minutes to 68 seconds | API only | [blog](https://qwenlm.github.io/blog/qwen2.5-turbo/) |
| QwQ-32B-Preview | 2024-11-28 | First Qwen reasoning model, from Qwen2.5-32B, 32K context; AIME 50.0, MATH-500 90.6; known language-mixing and looping issues | Open, Apache 2.0 | [blog](https://qwenlm.github.io/blog/qwq-32b-preview/) |
| QVQ-72B-Preview | 2024-12-25 | Visual reasoning from Qwen2-VL-72B; MMMU 70.3 | Open, Qwen license | [blog](https://qwenlm.github.io/blog/qvq-72b-preview/) |
| Qwen2.5-VL 3B / 7B / 72B | 2025-01-26 | Window-attention ViT (only 4 full-attention layers), dynamic-FPS video training, absolute time encoding, hour-plus video, computer and phone agent use | Open (3B research, 7B Apache 2.0, 72B Qwen) | [blog](https://qwenlm.github.io/blog/qwen2.5-vl/), [arXiv 2502.13923](https://arxiv.org/abs/2502.13923) |
| Qwen2.5-7B / 14B-Instruct-1M | 2025-01-27 | First open 1M-context Qwen; trained to 256K and extrapolated to 1M; sparse-attention prefill 3.2x to 6.7x faster | Open, Apache 2.0 | [blog](https://qwenlm.github.io/blog/qwen2.5-1m/), [arXiv 2501.15383](https://arxiv.org/abs/2501.15383) |
| Qwen2.5-Max | 2025-01-28 | Large MoE, more than 20T tokens, SFT + RLHF; API id `qwen-max-2025-01-25`; parameter count not disclosed | API only | [blog](https://qwenlm.github.io/blog/qwen2.5-max/) |
| QwQ-32B | 2025-03-06 | Two-stage RL: outcome-based rewards (math answer checker, code execution server, no reward model), then general RL with a reward model; claims parity with DeepSeek-R1; 131,072 context | Open, Apache 2.0 | [blog](https://qwenlm.github.io/blog/qwq-32b/) |
| Qwen2.5-Omni-7B | 2025-03-27 | Thinker-Talker split: a "thinker" LLM produces text, a "talker" streams speech tokens from its hidden states; TMRoPE time-aligns audio and video | Open, Apache 2.0 (a later 3B under the research license) | [blog](https://qwenlm.github.io/blog/qwen2.5-omni/), [arXiv 2503.20215](https://arxiv.org/abs/2503.20215) |

QwQ-32B is the release that changed the family's direction. It showed that outcome-rewarded RL on a 32B dense base could reach the reasoning level of a 671B MoE, and its recipe (verifiable rewards first, general reward model second) became the reasoning stage inside Qwen3's post-training pipeline seven weeks later. Thinker-Talker from Qwen2.5-Omni likewise became the skeleton of Qwen3-Omni.

## 5. Qwen3 (April 2025): one model, two modes

**Released:** 2025-04-29 ([Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)). Technical report: [arXiv 2505.09388](https://arxiv.org/abs/2505.09388), 2025-05-14. Our deep-dive: [Qwen3 Technical Report: One Model, Two Minds](/blog/machine-learning/frontier-models/qwen3-technical-report).

Qwen3 is the biggest single-generation step in the catalog. Relative to Qwen2.5:

- **36T pretraining tokens** (double Qwen2.5) across **119 languages and dialects** (up from 29).
- **Hybrid thinking.** One checkpoint serves both a reasoning mode and a fast mode, switched by `enable_thinking` or the `/think` and `/no_think` soft switches, with a thinking budget that trades latency for accuracy. Before Qwen3 you needed QwQ for reasoning and Qwen2.5-Instruct for everything else.
- **A four-stage post-training pipeline** for the flagships (long chain-of-thought cold start, reasoning RL, thinking-mode fusion, general RL), then **strong-to-weak distillation** to produce the small models from the big ones instead of running the full pipeline at every size.
- **MoE redesign.** 128 experts with 8 active, and, unlike Qwen2's MoE, no shared experts. Load balancing moved to a global-batch auxiliary loss so experts could specialize.
- **Apache 2.0 for every size.** The custom-license carve-outs for 3B and 72B were gone.

| Model | Type | Layers | Context | License |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B / 1.7B | dense | 28 | 32K | Apache 2.0 |
| Qwen3-4B | dense | 36 | 32K | Apache 2.0 |
| Qwen3-8B / 14B / 32B | dense | 36 / 40 / 64 | 32K native, 128K with YaRN | Apache 2.0 |
| Qwen3-30B-A3B | MoE, 128 experts, 8 active | 48 | 32K native, 128K with YaRN | Apache 2.0 |
| Qwen3-235B-A22B | MoE, 128 experts, 8 active | 94 | 32K native, 128K with YaRN | Apache 2.0 |

The 30B-A3B became the most deployed Qwen3: 3B active parameters means it decodes at small-model speed while holding mid-size quality, which is exactly the trade a single consumer GPU wants. For a measured comparison against Gemma 3 and Llama 3 on equal hardware, see [our inference-engineering experiment](/blog/machine-learning/inference-engineering/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware).

## 6. The Qwen3 extensions (June 2025 to January 2026)

Qwen3's architecture stayed fixed for ten months while the team shipped specialists and refreshes on top of it. This section is long because the Qwen3 era is where most production deployments live today.

### Qwen3-2507: undoing hybrid thinking

**Released:** 2025-07-21 through 2025-08-06 ([Qwen3 GitHub news](https://github.com/QwenLM/Qwen3), [235B-Instruct-2507 card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)).

Three months after shipping hybrid thinking, Qwen split it back apart. The 2507 refresh released separate **Instruct-2507** (non-thinking) and **Thinking-2507** checkpoints, because one model serving both modes was measurably worse at each than two dedicated models. Native context rose to **256K** for all 2507 models, and on 2025-08-08 ultra-long input up to 1M tokens was enabled for the 235B and 30B.

| Date | Release |
| --- | --- |
| 2025-07-21 | Qwen3-235B-A22B-Instruct-2507 |
| 2025-07-25 | Qwen3-235B-A22B-Thinking-2507 |
| 2025-07-30 | Qwen3-30B-A3B-Instruct-2507 |
| 2025-07-31 | Qwen3-30B-A3B-Thinking-2507 |
| 2025-08-06 | Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507 |

This is a rare case of a lab publicly reversing a headline design choice within one generation, and it is worth remembering when someone tells you a single hybrid-reasoning model is strictly better.

### Retrieval, coding, translation and image

| Release | Date | Key facts | Weights | Source |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding / Reranker 0.6B / 4B / 8B | ~2025-06-05 | 32K context, 100+ languages | Open, Apache 2.0 | [arXiv 2506.05176](https://arxiv.org/abs/2506.05176) |
| Qwen3-Coder-480B-A35B-Instruct | 2025-07-22 | 480B total, 35B active, 62 layers, 256K native (1M with YaRN), trained for agentic coding with long-horizon RL | Open, Apache 2.0 | [blog](https://qwenlm.github.io/blog/qwen3-coder/), [card](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) |
| Qwen3-Coder-30B-A3B-Instruct | ~2025-07-31 | The single-GPU coder | Open, Apache 2.0 | [HF](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen-MT | 2025-07-24 | Translation across 92 languages, built on Qwen3 and RL-tuned | API only | [blog](https://qwenlm.github.io/blog/qwen-mt/) |
| Qwen-Image | 2025-08-04 | 20B MMDiT diffusion model, strong at rendering text (including Chinese) inside images | Open, Apache 2.0 | [arXiv 2508.02324](https://arxiv.org/abs/2508.02324) |
| Qwen-Image-Edit | ~2025-08-17 | Editing variant; later Edit-2509, Edit-2511 and Qwen-Image-2512 refreshes | Open | [HF](https://huggingface.co/Qwen/Qwen-Image-Edit) |

Deep-dives: [Qwen-MT: machine translation as a reinforcement-learned task](/blog/machine-learning/frontier-models/qwen-mt-multilingual-translation), [Qwen-Image technical report](/blog/machine-learning/frontier-models/qwen-image-technical-report), and the practitioner view in [Qwen-Image deep-dive: text rendering and editing](/blog/machine-learning/image-generation/qwen-image-deep-dive).

### Qwen3-Next: the architecture break

**Released:** 2025-09-11 ([Qwen3.8 repo news](https://github.com/QwenLM/Qwen3.8), [model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)). Our deep-dive: [Qwen3-Next: hybrid attention and an 80B model that thinks with 3B](/blog/machine-learning/frontier-models/qwen3-next-hybrid-attention-ultra-sparse-moe).

Qwen3-Next-80B-A3B (Instruct and Thinking) is the most important release in the catalog after Qwen3 itself, because it is where the family left the plain transformer.

- **Hybrid attention:** three Gated DeltaNet layers (a linear-attention recurrence with a gated delta-rule state update, constant memory per token) for every one gated full-attention layer. Only a quarter of the layers keep a growing KV cache. The DeltaNet mechanism is covered in our paper read of [Gated Delta Networks](/blog/paper-reading/large-language-model/gated-delta-networks), and the kernels Qwen open-sourced for it in [FlashQLA](/blog/machine-learning/open-source-library/flashqla-gated-deltanet-kernels).
- **Ultra-sparse MoE:** 512 experts, 10 routed plus 1 shared, so 80B total and 3B active, about 4% of the weights per token.
- **Multi-token prediction** for speculative decoding built into the model.
- **15T pretraining tokens**, 262,144 native context, extensible to about 1M.
- Per the model card, the base model beats Qwen3-32B-Base at about 10% of the training cost and runs about 10x the throughput above 32K context.

| Model | Params | Layers | Context | License |
| --- | --- | --- | --- | --- |
| Qwen3-Next-80B-A3B-Instruct / Thinking | 80B total, 3B active | 48 | 262K native, ~1M extended | Apache 2.0 |

It shipped as a "Next" side branch. Five months later it became the mainline.

### Omni, vision, guard and the API flagship

| Release | Date | Key facts | Weights | Source |
| --- | --- | --- | --- | --- |
| Qwen3-Omni-30B-A3B (Instruct, Thinking, Captioner) | 2025-09-22 | Thinker-Talker on MoE; 119 text languages, 19 speech-input and 10 speech-output languages | Open, Apache 2.0 | [arXiv 2509.17765](https://arxiv.org/abs/2509.17765) |
| Qwen3-VL-235B-A22B (Instruct, Thinking) | ~2025-09-22 | 256K multimodal context, OCR in 32 languages | Open, Apache 2.0 | [arXiv 2511.21631](https://arxiv.org/abs/2511.21631) |
| Qwen3-VL-30B-A3B | ~2025-09-30 | Mid-size MoE VL | Open | same |
| Qwen3-VL-4B / 8B | ~2025-10-11 | Dense VL | Open | same |
| Qwen3-VL-2B / 32B | ~2025-10-19 | Completes the VL ladder | Open | same |
| Qwen3Guard (Gen and Stream, 0.6B / 4B / 8B) | ~2025-09-23 | Safety classifier; the Stream variant scores each token as it is generated; 119 languages | Open, Apache 2.0 | [arXiv 2510.14276](https://arxiv.org/abs/2510.14276) |
| Qwen3-Max | 2025-09 | Announced as over 1T parameters, 36T tokens | API only | [qwen.ai blog](https://qwen.ai/blog?id=241398b9cd6353de490b0f82806c7848c5d2777d) |

Deep-dives: [Qwen3-Omni technical report](/blog/machine-learning/frontier-models/qwen3-omni-technical-report), [Qwen3Guard technical report](/blog/machine-learning/frontier-models/qwen3guard-technical-report), and the engineering view in [Guard models for LLMs: how Qwen3Guard moderates a token stream](/blog/machine-learning/frontier-models/guard-models-qwen3guard-streaming-moderation).

A caveat on Qwen3-Max: the official blog is a JavaScript-rendered page and the parameter and token counts above are as Alibaba announced them, not something a model card or report lets you check. Treat them as vendor claims.

### Speech and the hybrid coder (January 2026)

| Release | Date | Key facts | Weights | Source |
| --- | --- | --- | --- | --- |
| Qwen3-VL-Embedding / Reranker 2B / 8B | ~2026-01-07 | Multimodal retrieval | Open | [HF](https://huggingface.co/Qwen) |
| Qwen3-TTS (0.6B, 1.7B; CustomVoice, VoiceDesign, tokenizer) | ~2026-01-21 | 12 Hz speech tokenizer, 10 languages | Open, Apache 2.0 | [arXiv 2601.15621](https://arxiv.org/abs/2601.15621) |
| Qwen3-ASR 0.6B / 1.7B + ForcedAligner-0.6B | ~2026-01-28 | Built on Qwen3-Omni; 30 languages plus 22 Chinese dialects; an API-only Qwen3-ASR-Flash preceded it in September 2025 | Open, Apache 2.0 | [card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Qwen3-Coder-Next | 2026-01-30 | Qwen3-Next design at 80B total, 3B active; 48 layers, 512 experts (10 + 1 shared), 256K context, non-thinking only | Open, Apache 2.0 | [card](https://huggingface.co/Qwen/Qwen3-Coder-Next) |

Deep-dives: [Qwen3-ASR technical report](/blog/machine-learning/speech-to-text/qwen3-asr-technical-report) and [Qwen3-Coder-Next: scaling agentic training, not model size](/blog/machine-learning/frontier-models/qwen3-coder-next-technical-report). Qwen3-Coder-Next is the signal that Qwen had committed to the hybrid architecture: a production coder, not a research branch, built on the Qwen3-Next skeleton two weeks before Qwen3.5.

## 7. Qwen3.5 (February to March 2026): the hybrid becomes the mainline

**Released:** 2026-02-16, timed to Chinese New Year ([Qwen3.8 repo news](https://github.com/QwenLM/Qwen3.8), [397B card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), blog "Qwen3.5: Towards Native Multimodal Agents"). Our deep-dive: [Qwen3.5: the 397B flagship that unifies the whole Qwen stack](/blog/paper-reading/large-language-model/qwen3-5-397b-flagship-hybrid-moe).

Qwen3.5 merged three lines that had run in parallel through 2025: the Qwen3-Next hybrid architecture, the Qwen3-VL vision stack, and the agentic RL from Qwen3-Coder. Relative to Qwen3:

- **Architecture:** Qwen3-Next's 3:1 Gated DeltaNet to gated-attention layout, now on every model in the generation.
- **Native multimodality by early fusion.** Images and video tokens are trained alongside text from pretraining onward rather than bolted on by a separate VL model. Every Qwen3.5 model is image-text-to-text. The README states it "outperforms Qwen3-VL models", which effectively retired the separate VL line.
- **201 languages and dialects**, up from 119, with a new **248,320-entry vocabulary** (padded). This is the first tokenizer change since 2023.
- **Training:** multi-step MTP, and RL "scaled across million-agent environments" on an asynchronous RL framework.
- **Context:** 262,144 native, about 1.01M with YaRN.
- **License:** Apache 2.0.

| Model | Released | Type | Notes |
| --- | --- | --- | --- |
| Qwen3.5-397B-A17B | 2026-02-16 | MoE, 17B active | 60 layers = 15 x (3 DeltaNet + 1 gated attention), 512 experts, 10 routed + 1 shared; hosted as Qwen3.5-Plus |
| Qwen3.5-122B-A10B | 2026-02-24 | MoE, 10B active | |
| Qwen3.5-35B-A3B | 2026-02-24 | MoE, 3B active | Hosted as Qwen3.5-Flash |
| Qwen3.5-27B | 2026-02-24 | dense, 64 layers | |
| Qwen3.5-9B / 4B / 2B / 0.8B | 2026-03-02 | dense | 0.8B has 24 layers and 262K context with no YaRN extension listed |

The total pretraining token count for Qwen3.5 is not stated on the model cards. For what it takes to serve the 397B fast, see [580 tokens a second on a 397B model: inside the TokenSpeed Qwen3.5 record](/blog/machine-learning/inference-frameworks/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell).

## 8. Qwen3.6 (April 2026): the coding point release

**Released:** Qwen3.6-35B-A3B on 2026-04-16, Qwen3.6-27B on 2026-04-22 ([Qwen3.8 repo news](https://github.com/QwenLM/Qwen3.8), [35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [27B card](https://huggingface.co/Qwen/Qwen3.6-27B)).

Qwen3.6 keeps the Qwen3.5 architecture and changes the post-training target. The README frames it as prioritizing "stability and real-world utility" with two headline features:

- **Agentic coding:** front-end workflows and repository-level reasoning.
- **Thinking preservation** (`preserve_thinking`): reasoning from earlier turns is kept in context instead of being stripped, which matters for multi-step agent loops where the model's earlier plan is the most useful thing in the history.

| Model | Type | Layout | Context | License |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | MoE, 3B active | 40 layers, 256 experts, 8 routed + 1 shared | 262K native | Apache 2.0 |
| Qwen3.6-27B | dense VLM | 64 layers | 262K native | Apache 2.0 |

The 27B card gives a clean before-and-after against its predecessor: SWE-bench Verified 75.0 to 77.2 and Terminal-Bench 2.0 41.6 to 59.3 versus Qwen3.5-27B. That second jump is the one the release was built for. Note the 35B-A3B uses 256 experts, half of Qwen3.5's 512, so "same architecture" means same layer pattern, not identical hyperparameters.

## 9. Qwen3.7 (2026): API only

Qwen3.7 exists, but only as a hosted model. Its sole appearance in a Qwen primary source I could verify is a **Qwen3.7-Max** comparison column in the benchmark table of the [Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B). There are no open weights, no model card and no technical report. Its release date, parameter count and context length are not stated in any primary source I could access, so this catalog lists it and stops there.

## 10. Qwen3.8 (August 2026): the first open Max-class model

**Released:** Qwen3.8-2.4T-A95B on 2026-08-12 and Qwen3.8-27B on 2026-08-14 ([Qwen3.8 repo news](https://github.com/QwenLM/Qwen3.8)); Qwen3.8-Flash-Next repository created 2026-08-24 ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).

The README's opening line is the news: "For the first time, Qwen3.8 brings a Qwen-Max-class model to open release." Since Qwen2.5-Max, the Max tier had always been API-only. Qwen3.8 builds on the Qwen3.5 architecture and targets coding, professional work and long-horizon agentic tasks, adding a `reasoning_effort` control alongside `preserve_thinking`.

| Model | Params | Layout | Context | License |
| --- | --- | --- | --- | --- |
| Qwen3.8-2.4T-A95B | 2.4T total, 95B active | 92 layers = 23 x (3 DeltaNet + 1 gated attention), 512 experts, 10 routed + 1 shared | 262K native, ~1.01M extended | Custom "Qwen3.8-Max" license |
| Qwen3.8-27B | dense VLM | 64 layers | 262K native, ~1M extended | Apache 2.0 |
| Qwen3.8-Flash-Next | 125B total, 6B active, plus 51B n-gram embedding and a 4B MTP module | 48 layers = 12 x (3 DeltaNet + 1 Qwen Sparse Attention) | n/d on the card excerpt checked | qwen-community-1.0 |

Three details on the flagship that are easy to miss:

- **The open 2.4T is text-only and always thinks.** The card says multimodal inputs are not supported and thinking cannot be disabled. The hosted **Qwen3.8-Max** adds vision input, a non-thinking mode and 1M context by default. Open weights and the API product are not the same model surface.
- **It is not Apache 2.0.** It ships under a custom license; read its LICENSE file before building a commercial product on it. The 27B stays Apache 2.0.
- **Active ratio about 4%.** 95B of 2.4T, the same sparsity as Qwen3-Next, at thirty times the total size.

### Qwen3.8-Flash-Next: the Qwen4 preview

Flash-Next is described on its card as an "experimental preview of the architecture that will underpin Qwen4". It changes four things at once:

- **Qwen Sparse Attention (QSA)** replaces the gated full-attention layer. Instead of attending to every past token, it selects micro-blocks of the history through a small MQA indexer, so the one-in-four "full" layer is no longer quadratic.
- **Gated Residual:** widened residual streams with a data-dependent read gate and a per-branch scalar write gate.
- **N-gram embedding:** a 20M-entry table of bigrams and trigrams injected at layer 2, adding 51B parameters that cost lookups rather than FLOPs and can be offloaded. This is a new axis of parameter scaling, separate from experts.
- **Training recipe:** Muon and AdamW applied to different weight categories, and no batch-size warmup (training starts at the target batch size), justified by refitted scaling laws.

The technical report is a PDF in the [Qwen3.8-Flash-Next GitHub repo](https://github.com/QwenLM/Qwen3.8-Flash-Next); it is not on arXiv as of this writing. The hosted Qwen3.8-Flash is built on it with 1M context by default.

## How the family evolved

![Four-step evolution of the Qwen backbone from dense GQA transformer to fine-grained MoE to Gated DeltaNet hybrid to the Qwen4 preview, with active-parameter share falling from 100% to about 4%](/imgs/blogs/qwen-model-family-timeline-2.webp)

Read as one sequence, 38 releases reduce to five through-lines.

### 1. The backbone changed in four steps, and only twice in a big way

The figure above traces it. From Qwen to Qwen2.5 the backbone was a Llama-style dense transformer whose only material change was GQA (Qwen1.5-32B, then universal in Qwen2). Fine-grained MoE arrived as a side model in Qwen1.5 and became the flagship form in Qwen3. The first real break was Qwen3-Next's 3:1 Gated DeltaNet hybrid in September 2025, which became the mainline in Qwen3.5. The second is Flash-Next's QSA plus n-gram embeddings, previewed for Qwen4. Everything between those breaks was data and post-training.

| Era | Releases | Attention | FFN | KV cache growth |
| --- | --- | --- | --- | --- |
| Dense | Qwen, Qwen1.5, Qwen2.5 | full MHA, then GQA | dense SwiGLU | every layer |
| MoE | Qwen1.5-MoE, Qwen2-57B, Qwen3 | GQA | fine-grained experts | every layer |
| Hybrid | Qwen3-Next, Qwen3.5, Qwen3.6, Qwen3.8 | 3 DeltaNet : 1 gated attention | 256 to 512 experts | 1 layer in 4 |
| Qwen4 preview | Qwen3.8-Flash-Next | 3 DeltaNet : 1 sparse attention | experts + n-gram table | 1 layer in 4, sparse reads |

### 2. Sparsity went from 25% active to 4% active

Qwen2-57B-A14B activated about a quarter of its weights per token. Qwen3-235B-A22B activated about 9%. Qwen3-Next, Qwen3.5-397B-A17B (about 4.3%) and Qwen3.8-2.4T-A95B (about 4%) settled near one in twenty-five. The strategy is consistent: buy capacity with total parameters, keep per-token compute flat, and pay in memory. That is why Qwen MoE models are cheap per token and expensive to host, and why the expert-parallel serving problem dominates any Qwen3.5 deployment. Our [MoE serving guide](/blog/machine-learning/model-serving/serving-moe-models-at-scale) covers that cost.

### 3. Data grew 12x and context 32x

![Pretraining tokens per generation from 3T to 36T and native context from 8K to 262K](/imgs/blogs/qwen-model-family-timeline-3.webp)

Disclosed flagship pretraining went 3T (Qwen-14B and 72B), about 7T (Qwen2), 18T (Qwen2.5), 36T (Qwen3). Qwen1.5 and Qwen3.5 did not publish a figure; Qwen3-Next used a 15T subset. Native context went from 8K on the refreshed Qwen-7B to 32K (Qwen1.5 through Qwen3), then 256K (Qwen3-2507) and 262,144 (Qwen3.5 onward), with YaRN reaching about 1M. The context jump is the payoff of the hybrid architecture: a 256K window is affordable when three quarters of the layers keep a fixed-size state instead of a KV cache.

### 4. Post-training moved from preference tuning to agent RL

| Generation | Post-training recipe (as published) |
| --- | --- |
| Qwen | SFT + RLHF |
| Qwen1.5 | DPO + PPO |
| Qwen2 | SFT, offline DPO, online DPO with a reward model, online merging |
| Qwen2-Math, Qwen2.5 | GRPO (Math), multistage RL |
| QwQ-32B | Outcome-reward RL (verifiers, code execution), then general RL |
| Qwen3 | Four-stage pipeline + strong-to-weak distillation |
| Qwen3.5 onward | RL across "million-agent environments", asynchronous RL |

The reward signal moved from human preference to verifiable outcomes to environment feedback. Each step made a different benchmark family move: chat scores in Qwen1.5, math in Qwen2.5, reasoning in QwQ, then SWE-bench and Terminal-Bench from Qwen3-Coder onward.

### 5. Specialists were absorbed back into the mainline

In 2024 you picked among Qwen2.5, Qwen2.5-Coder, Qwen2.5-Math, Qwen2.5-VL, QwQ and Qwen2.5-Omni. By 2026 a single Qwen3.5 or Qwen3.6 checkpoint reasons, codes and reads images natively. What remains separate are the models whose output modality differs (Omni, TTS, ASR, Image), the classifiers (Guard, Embedding, Reranker), and the largest coder. Licensing followed a similar arc toward openness, Apache 2.0 nearly everywhere from Qwen3 on, with one notable exception: the first open Max-class model reintroduced a custom license at the very top.

## Which variant to pick today

As of October 2026, with open weights and self-hosting in mind:

| You need | Pick | Why |
| --- | --- | --- |
| Best open quality, budget no object | Qwen3.8-2.4T-A95B | Max-class, but text-only, always-thinking, custom license, multi-node |
| Strong general model on one node | Qwen3.5-397B-A17B or Qwen3.5-122B-A10B | Hybrid attention, native vision, Apache 2.0 |
| One 80 GB GPU, agentic coding | Qwen3.6-27B or Qwen3.8-27B (dense) | Highest coding scores at this size; 3.8 is newer |
| Fast and cheap per token | Qwen3.6-35B-A3B | 3B active, 262K context |
| Laptop, phone or edge | Qwen3.5-4B / 2B / 0.8B | Smallest hybrid models; 0.8B has no YaRN extension |
| Repository-scale coding agent | Qwen3-Coder-480B-A35B or Qwen3-Coder-Next | Dedicated agentic coders; Next is far cheaper to serve |
| Speech in or out | Qwen3-ASR, Qwen3-TTS, Qwen3-Omni-30B-A3B | Only lines with audio output or dedicated recognition |
| Text-to-image with legible text | Qwen-Image line | 20B MMDiT, Apache 2.0 |
| Safety filtering on a stream | Qwen3Guard-Stream | Per-token moderation |
| Embeddings and reranking | Qwen3-Embedding / Reranker (VL variants for images) | Same tokenizer family as the generators |
| Stuck on an older inference stack | Qwen3-30B-A3B-Instruct-2507 / Thinking-2507 | Plain transformer, no DeltaNet kernels needed |

The last row matters more than it looks. Qwen3.5 and later need inference engines with Gated DeltaNet kernels and hybrid KV-cache management. If your serving stack predates that support, the Qwen3-2507 checkpoints are the newest models it can run correctly. Our [hybrid KV cache manager](/blog/machine-learning/model-serving/hybrid-kv-cache-manager) post explains what the engine has to do differently.

## What this catalog could not verify

In the spirit of not inventing numbers, these are the gaps:

- **Qwen 1 commercial terms.** The Tongyi Qianwen License required an application for commercial use; the README states no user-count threshold.
- **Qwen1.5 licenses** for the 1.8B, 4B and 14B were not checked card by card.
- **Qwen2-VL-72B open-weights date.** It launched API-only; the date its weights went public is not in a primary source I found.
- **Qwen2-Math license** is not stated in its blog.
- **Qwen2.5-Max, Qwen2.5-Turbo, Qwen2.5-Plus sizes** are undisclosed.
- **Qwen3-Max specs** (over 1T parameters, 36T tokens) are vendor announcements on a JavaScript-rendered page, not checkable from a report or card.
- **Qwen3.5 and Qwen3.8 pretraining token counts** are not on the model cards.
- **Qwen3.6-Plus, and everything about Qwen3.7** beyond the Qwen3.7-Max benchmark column, appear only in secondary sources.
- **Dates marked "~"** are Hugging Face repository creation dates, a floor on the public release date.
- **Parameter counts on Hugging Face differ from model names**: the "397B" reports 403B in safetensors, the "27B" reports 28B, and Flash-Next reports 180B because the n-gram table and MTP module are included.

## References

Primary sources only, in catalog order.

**Qwen (2023)**
- QwenLM/Qwen README and release news: https://github.com/QwenLM/Qwen
- Qwen Technical Report, arXiv 2309.16609: https://arxiv.org/abs/2309.16609
- Qwen-7B model card: https://huggingface.co/Qwen/Qwen-7B
- Qwen-VL, arXiv 2308.12966: https://arxiv.org/abs/2308.12966
- Qwen-Audio, arXiv 2311.07919: https://arxiv.org/abs/2311.07919

**Qwen1.5 (2024)**
- Introducing Qwen1.5: https://qwenlm.github.io/blog/qwen1.5/
- Qwen1.5-MoE-A2.7B: https://qwenlm.github.io/blog/qwen-moe/
- Qwen1.5-32B: https://qwenlm.github.io/blog/qwen1.5-32b/
- CodeQwen1.5: https://qwenlm.github.io/blog/codeqwen1.5/
- Qwen1.5-110B: https://qwenlm.github.io/blog/qwen1.5-110b/

**Qwen2 (2024)**
- Hello Qwen2: https://qwenlm.github.io/blog/qwen2/
- Qwen2 Technical Report, arXiv 2407.10671: https://arxiv.org/abs/2407.10671
- Qwen2-Audio, arXiv 2407.10759: https://arxiv.org/abs/2407.10759
- Qwen2-Math: https://qwenlm.github.io/blog/qwen2-math/
- Qwen2-VL blog: https://qwenlm.github.io/blog/qwen2-vl/ and arXiv 2409.12191: https://arxiv.org/abs/2409.12191

**Qwen2.5 (2024 to 2025)**
- Qwen2.5 blog: https://qwenlm.github.io/blog/qwen2.5/
- Qwen2.5 Technical Report, arXiv 2412.15115: https://arxiv.org/abs/2412.15115
- Qwen2.5-Coder, arXiv 2409.12186: https://arxiv.org/abs/2409.12186 and family blog: https://qwenlm.github.io/blog/qwen2.5-coder-family/
- Qwen2.5-Math, arXiv 2409.12122: https://arxiv.org/abs/2409.12122
- Qwen2.5-Turbo: https://qwenlm.github.io/blog/qwen2.5-turbo/
- QwQ-32B-Preview: https://qwenlm.github.io/blog/qwq-32b-preview/
- QVQ-72B-Preview: https://qwenlm.github.io/blog/qvq-72b-preview/
- Qwen2.5-VL blog: https://qwenlm.github.io/blog/qwen2.5-vl/ and arXiv 2502.13923: https://arxiv.org/abs/2502.13923
- Qwen2.5-1M blog: https://qwenlm.github.io/blog/qwen2.5-1m/ and arXiv 2501.15383: https://arxiv.org/abs/2501.15383
- Qwen2.5-Max: https://qwenlm.github.io/blog/qwen2.5-max/
- QwQ-32B: https://qwenlm.github.io/blog/qwq-32b/
- Qwen2.5-Omni blog: https://qwenlm.github.io/blog/qwen2.5-omni/ and arXiv 2503.20215: https://arxiv.org/abs/2503.20215

**Qwen3 and extensions (2025 to 2026)**
- Qwen3 blog: https://qwenlm.github.io/blog/qwen3/
- Qwen3 Technical Report, arXiv 2505.09388: https://arxiv.org/abs/2505.09388
- QwenLM/Qwen3 release news (2507 refresh): https://github.com/QwenLM/Qwen3
- Qwen3-235B-A22B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507
- Qwen3-Embedding, arXiv 2506.05176: https://arxiv.org/abs/2506.05176
- Qwen3-Coder blog: https://qwenlm.github.io/blog/qwen3-coder/ and card: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct
- Qwen-MT: https://qwenlm.github.io/blog/qwen-mt/
- Qwen-Image, arXiv 2508.02324: https://arxiv.org/abs/2508.02324
- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Omni, arXiv 2509.17765: https://arxiv.org/abs/2509.17765
- Qwen3-VL, arXiv 2511.21631: https://arxiv.org/abs/2511.21631
- Qwen3Guard, arXiv 2510.14276: https://arxiv.org/abs/2510.14276
- Qwen3-Max announcement: https://qwen.ai/blog?id=241398b9cd6353de490b0f82806c7848c5d2777d
- Qwen3-TTS, arXiv 2601.15621: https://arxiv.org/abs/2601.15621
- Qwen3-ASR-1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-Coder-Next: https://huggingface.co/Qwen/Qwen3-Coder-Next

**Qwen3.5, Qwen3.6, Qwen3.8 (2026)**
- QwenLM/Qwen3.8 README and release news (covers Qwen3-Next, Qwen3.5, Qwen3.6, Qwen3.8): https://github.com/QwenLM/Qwen3.8
- Qwen3.5-397B-A17B: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6-27B: https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.8-2.4T-A95B (also the only primary mention of Qwen3.7-Max): https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-Flash-Next: https://huggingface.co/Qwen/Qwen3.8-Flash-Next and technical report repo: https://github.com/QwenLM/Qwen3.8-Flash-Next
