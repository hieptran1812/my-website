---
title: "ByteDance Seed, in order: every Doubao and Seed release from Skylark to Seed2.1"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological catalog of the ByteDance Seed and Doubao model family, from the 2023 Skylark chatbot to Seed2.1, with release dates, what changed each generation, variant tables, and a primary source for every entry."
tags:
  [
    "bytedance-seed",
    "doubao",
    "model-timeline",
    "large-language-model",
    "mixture-of-experts",
    "reasoning-models",
    "open-weights",
    "multimodal",
    "ai-agents",
    "foundation-models",
  ]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: true
readTime: 25
---

## Introduction: the same family, read front to back

ByteDance's model family is hard to hold in your head because it has two names and three release channels. The consumer product is **Doubao**, the chatbot app. The research lab is **ByteDance Seed**, which publishes the technical reports. The enterprise channel is **Volcano Engine** (BytePlus outside China), which sells the API. A single model often shows up under all three labels at once: the January 2025 flagship is "Doubao-1.5-pro" in the press, "Seed1.5" on the Seed portfolio page, and a dated model ID on Volcano Engine. Add the specialists (a vision model, a code model, a theorem prover, a diffusion LLM, an open 36B) and the media lines (Seedream, Seedance, Seed-TTS), and it is easy to lose track of what came before what.

This blog already has a map of the family organized **by use case**: [ByteDance Seed, mapped: the whole model family organized by what each model is for](/blog/machine-learning/frontier-models/bytedance-seed-model-universe-by-use-case). That post answers "which Seed model is for my problem." This one answers the other question: **in what order did these models arrive, and what changed each time?** Read them as a pair. The use-case map is the catalog index; this is the changelog. For the wider ByteDance research picture beyond Seed's flagship line, see the [ByteDance research model atlas](/blog/machine-learning/bytedance-research-model-atlas), and for the video stack in depth, [Seedance and the ByteDance video stack](/blog/machine-learning/video-generation/seedance-and-the-bytedance-video-stack).

The rules for this catalog are strict. Every release listed here actually shipped as of October 14, 2026. Every date and spec traces to a primary source (an arXiv report, a Seed blog post, the Seed model portfolio page, or an official Hugging Face or GitHub page), linked in the References section. Where ByteDance did not publish a number, the table says "not disclosed" instead of guessing. Where the only source for a fact is a third-party API listing or press report, the text says so. The release timeline below is the whole story in one picture; the rest of the post walks it generation by generation.

![Release timeline of the ByteDance Seed and Doubao model family from August 2023 to October 2026, with the flagship API line on top and open-weight and specialist releases below](/imgs/blogs/bytedance-seed-model-family-timeline-1.webp)

## The timeline at a glance

Before the detail, the skeleton. The flagship language line, the one that powers the Doubao app and the Volcano Engine API, has gone through these generations:

| Generation | Seed portfolio date | Headline change |
| --- | --- | --- |
| Skylark (Doubao app) | Aug 2023 (app debut) | First Doubao chatbot, internal model code-named Skylark |
| Doubao model family | May 15, 2024 | Public Volcano Engine API, aggressive pricing |
| Seed1.5 (Doubao-1.5-pro) | Jan 22, 2025 | Sparse MoE, 7x "performance leverage" claim |
| Seed1.5-Thinking | Apr 2025 (report Apr 10) | RL-trained reasoning, 20B active / 200B total |
| Seed1.6 | Jun 25, 2025 | 23B active / 230B total, 256K context, adaptive thinking |
| Seed1.8 | Dec 17, 2025 | Generalized agent: search, code, GUI in one model |
| Seed2.0 | Feb 14, 2026 | Pro / Lite / Mini tiers plus a Code model |
| Seed2.1 | Jun 23, 2026 | Pro / Turbo, aimed at long-horizon agent work |

The gaps between flagship releases from Doubao-1.5-pro onward are 3, 2, 6, 2 and 4 months. That is not a fixed cadence. The pattern is better described as two bursts (spring 2025 and winter 2025 into 2026) separated by a long summer in which the lab shipped specialists instead of a new flagship. The timeline figure is drawn in release order rather than to scale, precisely because 2025 is so crowded that a to-scale axis would squash a dozen releases into a smear.

Below the flagship line runs a second lane: specialist and open-weight releases. UI-TARS (January 2025), Seed1.5-VL (May 2025), Seed-Coder (May/June 2025), Seed-Prover (July 2025), Seed Diffusion Preview (July/August 2025), Seed-OSS-36B (August 2025) and Seed-Prover 1.5 (December 2025). Most of those branch off the flagship's research; a few are independent bets.

One naming note before going further. ByteDance shifted from "Doubao-N" naming to "SeedN" naming during 2025. The Seed portfolio page retroactively lists Doubao-1.5-pro as "Seed1.5." The API kept a hybrid: model IDs like `doubao-seed-1-6-...` and `doubao-seed-2-1-pro-260628`, where the trailing six digits are a YYMMDD snapshot date. When this post says "Seed1.6," it means the model the API calls Doubao-Seed-1.6.

## Origins: Skylark and the Doubao app (August 2023)

The family starts not with a paper but with a product. ByteDance launched the **Doubao** chatbot app in August 2023. The model underneath was internally code-named **Skylark** (Chinese: Yunque). According to the South China Morning Post's coverage of the May 2024 enterprise launch, the Skylark models had been tested internally and with selected partners for about a year before going public under the Doubao name.

There is almost nothing primary to cite about Skylark's architecture. ByteDance did not publish a technical report, parameter counts, or a context length for it. What matters historically is the strategic choice it represents: ByteDance built a consumer product first and treated the model as an internal component, the opposite of labs that led with a paper and an API. That product-first instinct shows up again and again in what follows.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Skylark (inside Doubao app) | Aug 2023 | not disclosed | not disclosed | text chat | closed |

## Generation 0: the Doubao model family on Volcano Engine (May 2024)

On **May 15, 2024**, at a Volcano Engine event in Beijing, ByteDance opened the Doubao models to enterprise customers through an API. The headline was price, not capability. Volcano Engine's president Tan Dai announced Doubao Pro at 0.0008 yuan per thousand input tokens, which Caixin reported as 99.8% cheaper than GPT-4 at the time. TechNode counted nine models in the launch lineup; SCMP described at least eight. The family spanned general-purpose chat tiers (Pro and Lite) plus role-play, speech synthesis, voice cloning, speech recognition, text-to-image and embedding models.

What changed versus Skylark: the model became a product you could buy by the token, priced to start a domestic price war. Architecture was still undisclosed.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Doubao-pro (tiered by context) | May 2024 | not disclosed | multiple tiers, sizes per third-party listings only | text | API |
| Doubao-lite | May 2024 | not disclosed | multiple tiers | text | API |
| Specialist APIs (role-play, TTS, voice clone, ASR, text-to-image, embedding) | May 2024 | not disclosed | n/a | mixed | API |

The speech work behind those TTS endpoints was published a few weeks later as **Seed-TTS** (arXiv, June 4, 2024): a family of autoregressive speech generators that clone a speaker from a short sample and control emotion, plus a non-autoregressive diffusion variant that needs no phoneme durations. It is the first Seed technical report in this catalog, and it set a pattern of publishing the research behind a product after the product was already live.

## Generation 1: Seed1.5, a.k.a. Doubao-1.5-pro (January 22, 2025)

**Doubao-1.5-pro** is the first release where ByteDance explained its architecture. The official Seed page, dated **January 22, 2025**, describes three things.

**A sparse Mixture-of-Experts model with a 7x "performance leverage."** The team trained an MoE model and a dense model on the same 9T-token data and reports that the MoE model, activating about one seventh of the dense model's parameters, matched or beat it. The industry norm they cite is roughly 3x. Total and active parameter counts for Doubao-1.5-pro itself were not published on that page. (Third-party docs repeat a "20B active equals 140B dense" figure; I could not find it on ByteDance's page, so treat it as unverified.)

**Training-inference co-design.** Inference splits prefill from decode and runs attention and FFN on separately scaled hardware pools, a PD-disaggregation setup built for serving cost rather than benchmark headlines. This matters for reading the whole family: from here on, almost every Seed release talks about serving cost alongside accuracy.

**No distillation from other models.** The page states twice that post-training, including the deep-thinking mode, used no data from other models. It also previews **Doubao-1.5-pro-AS1-Preview**, a deep-thinking variant, and a 2.4B-parameter **Doubao ViT** vision encoder.

The Seed model portfolio page later relabels this release as **Seed1.5**, which is why the reasoning and vision models that follow carry the "Seed1.5-" prefix.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Doubao-1.5-pro (Seed1.5) | Jan 22, 2025 | sparse MoE, counts not disclosed | not stated on official page | text | API |
| Doubao-1.5-pro-AS1-Preview | Jan 2025 (preview) | not disclosed | not stated | text, deep thinking | API |
| Doubao ViT encoder | Jan 2025 | 2.4B | n/a | vision encoder | not released |

### Alongside Seed1.5: UI-TARS (January 21, 2025)

One day earlier, a different ByteDance team published **UI-TARS** (arXiv, January 21, 2025), an end-to-end GUI agent that takes only screenshots and acts through keyboard and mouse. It is a native agent rather than a wrapper around a text LLM plus an accessibility tree, and it reported state-of-the-art results on more than ten GUI benchmarks. It is a separate research line rather than a branch of the Doubao MoE core, but it matters to this catalog because its ideas later land inside the flagship: Seed1.6 lists GUI interaction as a capability, and Seed1.8 makes the GUI agent one of its three pillars. The GitHub repo records **UI-TARS-1.5-7B** being open-sourced on April 16, 2025 (Apache-2.0) and **UI-TARS-2** announced on September 4, 2025. A UI-TARS-72B-DPO checkpoint appears in the repo's comparison tables.

| Variant | Released | Params | Modality | Weights |
| --- | --- | --- | --- | --- |
| UI-TARS (paper + checkpoints, incl. 72B-DPO) | Jan 21, 2025 | several sizes | screenshot in, actions out | open |
| UI-TARS-1.5-7B | Apr 16, 2025 | 7B | screenshot in, actions out | open (Apache-2.0) |
| UI-TARS-2 | Sep 4, 2025 | not stated in repo | GUI + game agent | report |

## Generation 1.5: Seed1.5-Thinking (April 2025)

**Seed1.5-Thinking** (arXiv, April 10, 2025) is the reasoning model of the Seed1.5 generation, and the first time ByteDance published parameter counts for a flagship: a Mixture-of-Experts model with **20B activated and 200B total parameters**. Headline scores in the abstract are 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, placing it in the same band as the frontier reasoning models of that spring.

What changed versus Doubao-1.5-pro is the post-training recipe. Seed1.5-Thinking is trained with large-scale reinforcement learning on verifiable and non-verifiable tasks, and the report sits on top of two RL algorithm papers the Seed team released that spring, DAPO and VAPO, which fix specific failure modes of PPO and GRPO on long chain-of-thought (entropy collapse, length bias, value-model instability). This blog's [Seed1.5-Thinking deep-dive](/blog/machine-learning/frontier-models/seed1-5-thinking-rl-reasoning-vapo-dapo) walks the recipe end to end, and [Beyond GRPO: DAPO, Dr. GRPO and GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) puts DAPO in the context of the other GRPO fixes.

The weights were not released; the model reached users through the Doubao app and the Volcano Engine API.

| Variant | Released | Total / active | Modality | Training | Weights |
| --- | --- | --- | --- | --- | --- |
| Seed1.5-Thinking | Apr 2025 (report Apr 10) | 200B / 20B MoE | text reasoning | large-scale RL (DAPO, VAPO lineage) | API |

### Alongside: Seed1.5-VL (May 13, 2025)

**Seed1.5-VL** (arXiv, May 11, 2025; Seed portfolio date May 13) is the vision-language member of the generation. It pairs a **532M-parameter vision encoder** with a **20B-active MoE LLM**, which is the same active size as Seed1.5-Thinking, and the report claims state-of-the-art results on 38 of 60 public benchmarks. The encoder handles native resolution rather than tiling to a fixed grid, and the model covers GUI and agent tasks as well as image and video understanding. Note the encoder: 532M, much smaller than the 2.4B Doubao ViT announced in January. This blog's [Seed1.5-VL deep-dive](/blog/machine-learning/frontier-models/seed1-5-vl-native-resolution-vision-language) covers the architecture and data mix.

| Variant | Released | Params | Modality | Weights |
| --- | --- | --- | --- | --- |
| Seed1.5-VL | May 13, 2025 (report May 11) | 532M ViT + 20B-active MoE LLM | image, video, GUI in; text out | API |

### Alongside: BAGEL (May 2025)

The same month, Seed released **BAGEL**, a unified understand-and-generate model built as a Mixture-of-Transformers: 14B total parameters, 7B active per token, open weights. It is a research model rather than a product, and it is initialized from Qwen2.5-7B-Instruct rather than the Doubao core, so it sits outside the flagship lineage. It belongs in this catalog because it is Seed's first serious open release in multimodal generation. See the [BAGEL paper reading](/blog/paper-reading/multimodal/bagel-unified-multimodal-mixture-of-transformers).

## Seed-Coder: the first open LLM from the core team (May/June 2025)

**Seed-Coder** (arXiv, June 4, 2025) is the first open-weight language model in this catalog that comes from the Seed LLM team itself. It is small on purpose: **8B parameters**, in Base, Instruct and Reasoning variants, licensed **MIT**.

What changed is the data pipeline, not the architecture. Seed-Coder's thesis is "let the code model curate data for itself": instead of hand-written filtering rules, LLMs score and filter the code corpus. The Instruct model is trained with SFT plus preference optimization; the Reasoning model with long-chain-of-thought RL. The [Seed-Coder deep-dive](/blog/machine-learning/frontier-models/seed-coder-model-centric-code-data-pipeline) explains why model-centric filtering beats rule-based filtering at this scale.

| Variant | Params | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| Seed-Coder-8B-Base | 8B dense | 32K | pretrained base | open (MIT) |
| Seed-Coder-8B-Instruct | 8B dense | 32K | SFT + preference optimization | open (MIT) |
| Seed-Coder-8B-Reasoning | 8B dense | 64K | long-CoT RL | open (MIT) |
| Seed-Coder-8B-Reasoning-bf16 | 8B dense | 64K | same, bf16 weights | open (MIT) |

## Generation 2: Seed1.6 (June 25, 2025)

**Seed1.6** is dated **June 25, 2025** on both the Seed portfolio page and the "Introduction to Techniques Used in Seed1.6" blog post. It is the generation where the flagship becomes natively multimodal and learns when not to think.

**Architecture and scale.** Sparse MoE, **23B active and 230B total parameters**, a 15% step up from Seed1.5-Thinking's 20B / 200B. The ratio of active to total stays at roughly one in ten.

**Pre-training in three continual stages.** A text-only stage (web, books, papers, code), then multimodal mixed continual training (MMCT), which mixes high-quality visual data in with more discipline, code and reasoning text, then long-context continual training (LongCT) that extends the maximum sequence from **32K to 256K**. Multimodality is no longer a separate VL model bolted on; it is folded into the flagship's own pre-training, and the post reports Seed1.6 matching or beating Seed1.5-VL on several visual tasks.

**Adaptive thinking (AdaCoT).** Seed1.6 exposes three modes through prompt configuration: FullCoT (always reason), NoCoT (answer directly) and AdaCoT (decide per prompt). AdaCoT is trained with an RL reward that penalizes overthinking, and the blog reports CoT trigger rates rising with difficulty: about 37% on MMLU, 70% on MMLU-Pro and 90 to 100% on AIME and BeyondAIME. This is the first time the family treats thinking tokens as a cost to be managed, an idea Seed-OSS turns into an explicit user-set budget two months later.

**Post-training.** Seed1.6-Thinking alternates multi-stage rejection fine-tuning with RL, each RL round starting where the last RFT round ended, with a multi-dimensional reward model choosing among candidates. Parallel decoding at inference adds about 8 points on BeyondAIME without retraining.

A **Seed1.6-Flash** tier also exists: the Seed blog's "Built on Seed1.6-Flash, Seed-1.6-Embedding Launched" post names it, and API model IDs such as `doubao-seed-1-6-flash-250828` show dated snapshots. The techniques post does not describe Flash's size.

| Variant | Released | Total / active | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Seed1.6 Base | Jun 25, 2025 | 230B / 23B MoE | 256K | text + vision | not released |
| Seed1.6 (AdaCoT) | Jun 25, 2025 | 230B / 23B MoE | 256K | text + vision, adaptive thinking | API |
| Seed1.6-Thinking | Jun 25, 2025 | 230B / 23B MoE | 256K | text + vision, always-on reasoning | API |
| Seed1.6-Flash | 2025 (dated snapshots) | not disclosed | 256K per third-party listings | text + vision | API |
| Seed-1.6-Embedding | 2025 | built on Seed1.6-Flash | not stated | embedding | API |

## The specialist summer of 2025

Between Seed1.6 in June and Seed1.8 in December, the flagship went quiet for six months, the longest gap in the catalog. The lab did not slow down; it shipped specialists instead.

### Seed-Prover (July 31, 2025)

**Seed-Prover** (arXiv, July 31, 2025) is a Lean 4 theorem prover trained with RL against the Lean verifier, using lemma-style whole-proof generation and iterative refinement. The abstract reports that it saturates miniF2F, exceeds 50% on PutnamBench, and fully proved 5 of the 6 IMO 2025 problems. It ships with **Seed-Geometry**, a geometry engine built because Lean's geometry support is thin. This blog's [Seed-Prover deep-dive](/blog/machine-learning/frontier-models/seed-prover-formal-theorem-proving-lean-imo) covers the search strategy.

### Seed Diffusion Preview (July 31, 2025)

**Seed Diffusion Preview** (Seed portfolio date July 31, 2025; arXiv August 4) is the family's most unusual entry: a discrete-state **diffusion language model** specialized for code, generating tokens in parallel rather than left to right. The report's headline is speed: **2,146 tokens per second on H20 GPUs**. It is labelled experimental and was not released as weights. The [Seed Diffusion paper reading](/blog/paper-reading/large-language-model/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference) explains why diffusion LMs trade quality for parallel decoding and how Seed narrowed that gap.

### Seed-OSS-36B (August 20, 2025)

**Seed-OSS-36B** (Hugging Face, created August 20, 2025) is the release most practitioners actually run. It is a **dense** 36B model, not an MoE, which is a deliberate break from the flagship: dense models are simpler to serve on commodity hardware and to fine-tune. The model card lists 64 layers, GQA with 80 query heads and 8 key-value heads, head size 128, hidden size 5120, a vocabulary of about 155K, RoPE base 1e7, and training on **only 12T tokens**. The context is **512K, trained natively**, double the flagship's 256K. License: **Apache-2.0**.

Its signature feature is the **thinking budget**: the user sets how many tokens the model may spend reasoning, in multiples of 512 that the model was trained on, and the model inserts progress tags reporting tokens used and remaining. A budget of 0 means answer directly. That is AdaCoT's "thinking is a cost" idea handed to the user as a dial. The release also includes a research-friendly base model trained without synthetic instruction data. See the [Seed-OSS-36B deep-dive](/blog/machine-learning/frontier-models/seed-oss-36b-open-long-context-thinking-budget).

| Variant | Params | Context | Notes | Weights |
| --- | --- | --- | --- | --- |
| Seed-OSS-36B-Base | 36B dense | 512K | pretrained with synthetic instruction data | open (Apache-2.0) |
| Seed-OSS-36B-Base-woSyn | 36B dense | 512K | pretrained without synthetic instruction data | open (Apache-2.0) |
| Seed-OSS-36B-Instruct | 36B dense | 512K | thinking budget, agent and tool use | open (Apache-2.0) |

### A note on Doubao-Seed-Code (November 2025)

Press coverage (KrASIA and others) reports that Volcano Engine launched **Doubao-Seed-Code**, an agentic-coding model with a 256K context, around November 11, 2025, with API ID `doubao-seed-code-preview-251028`. I could not find a Seed technical report or Seed blog post for it, so its specs here rest on secondary sources. It reads as the preview of the dedicated Code tier that ships with Seed2.0.

## Generation 3: Seed1.8, the generalized agent (December 17, 2025)

**Seed1.8** is dated **December 17, 2025** on the Seed portfolio page and December 18 on its release blog post ("Official Release of Seed1.8: A Generalized Agentic Model"). Its model card, "Seed1.8 Model Card: Towards Generalized Real-World Agency," followed on arXiv on March 21, 2026.

What changed versus Seed1.6 is the target, more than the architecture. Seed1.6 was a strong multimodal reasoner; Seed1.8 is pitched as an **agent**: one model that integrates a search agent, a code agent and a GUI agent behind a unified interface for search, code generation and execution, and GUI interaction, operating interfaces through native vision. The UI-TARS research line effectively merges into the flagship here. The release post compares itself against Seed1.5-VL for GUI, multimodal reasoning and long-video tasks, and highlights a cheaper image encoding (fewer tokens per image) and a configurable maximum number of video tokens.

Thinking control becomes a product feature. The release post says Seed1.8 supports three thinking modes that adjust to task complexity, and the model card describes configurable thinking modes designed for latency- and cost-aware deployment, crediting Seed1.6 as the predecessor that first introduced thinking modes.

Parameter counts and context window are **not disclosed** in either the release post or the model card abstract. Third-party API listings show 256K context and 64K maximum output; treat that as reseller data.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Seed1.8 | Dec 17, 2025 | not disclosed | 256K per third-party listings | text + image + video in; search, code, GUI agent | API |

### Alongside: Seed-Prover 1.5 (December 19, 2025)

Two days later, **Seed-Prover 1.5** (arXiv, December 19, 2025: "Mastering Undergraduate-Level Theorem Proving via Learning from Experience") moved the prover to large-scale agentic RL plus an efficient test-time-scaling workflow, claiming better results at a smaller compute budget than prior state of the art. Benchmark figures beyond the abstract circulate only in secondary reports, so I leave them out.

## Generation 4: Seed2.0 (February 14, 2026)

**Seed2.0**, branded **Doubao-Seed-2.0**, launched on **February 14, 2026** (Seed portfolio date; TechNode and Caixin covered it the same day). It is the first generation sold as a tiered family from day one rather than one flagship plus later spin-offs. Its formal model card, "Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Complexity," appeared on the Seed site on June 30, 2026.

**The tiers.** The official Seed2.0 page describes three general models: **Pro**, focused on long-chain reasoning and robustness in complex workflows; **Lite**, a production-grade balance of quality and speed; and **Mini**, optimized for inference throughput and deployment density. Launch coverage adds a fourth, a dedicated **Code** model that pairs with ByteDance's TRAE IDE. The official page I read does not list the Code model, so its specs rest on launch coverage.

**What changed.** The official page lists three shifts: a broad upgrade in multimodal understanding, stronger LLM and agent capability for long-horizon tasks, and a capability focus that moves "from competition-level reasoning to research-grade tasks." Read against the timeline, that last phrase is a statement of where the frontier moved: Seed1.5-Thinking's headline numbers were AIME and Codeforces; Seed2.0's are long-tail knowledge, complex instruction following and multi-step work.

**The late-April Lite update.** The Seed2.0 page notes that Lite was upgraded at the end of April 2026 (evaluation tables label snapshots 0215 and 0428). The update added audio input, making Seed2.0 Lite **the first omni-modal understanding model in the Seed foundation series**: video, image, audio and text in one model. It also improved agent, coding and GUI ability and added cross-modal audio-visual reasoning.

Parameter counts, architecture and context length are **not disclosed** on the official page. Third-party API listings give 256K context across tiers.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Seed2.0 Pro | Feb 14, 2026 | not disclosed | 256K per third-party listings | multimodal in, text out | API |
| Seed2.0 Lite | Feb 14, 2026; upgraded late Apr 2026 | not disclosed | 256K per third-party listings | video, image, audio, text in (after April update) | API |
| Seed2.0 Mini | Feb 2026 | not disclosed | 256K per third-party listings | multimodal in | API |
| Seed2.0 Code | Feb 14, 2026 (per launch coverage) | not disclosed | 256K per third-party listings | code, agentic | API (TRAE) |

## Generation 5: Seed2.1 (June 23, 2026)

![What ByteDance published per release: total params, active params, context and weights, with grey cells for figures that were never disclosed](/imgs/blogs/bytedance-seed-model-family-timeline-3.webp)

**Seed2.1** is the latest flagship generation as of this writing. The Seed portfolio dates it **June 23, 2026**, announced at Volcano Engine's FORCE conference; API model IDs carry the snapshot date 260628. There is no plain "Seed2.1": the family is two models, **Pro** and **Turbo**. Pro is the top-capability model and also powers the consumer Doubao 2.1 Pro assistant; Turbo is the faster, cheaper tier for high-volume enterprise traffic (half the Pro price per the launch rate card).

What changed versus Seed2.0, per the official Seed2.1 page: significantly stronger general-agent capability, much more stable delivery on coding tasks, and better handling of hour-long video and complex visual input (floor plans, design mockups) that can directly drive generation, editing and task execution. The page reports MMLongBench-128K scores of 78.3 (Pro) and 76.9 (Turbo), but does not state the context window.

As with every flagship after Seed1.6, parameter counts and architecture are **not disclosed**. The disclosure matrix above makes the pattern plain: the specs were public through Seed1.6, and from Seed1.8 on only the open Seed-OSS-36B keeps a full spec sheet. ByteDance's own benchmark claims against Western frontier models are vendor-reported and, at the time of writing, thinly corroborated by independent leaderboards.

| Variant | Released | Params | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- |
| Seed2.1 Pro | Jun 23, 2026 | not disclosed | not stated officially | text, image, long video in | API |
| Seed2.1 Turbo | Jun 23, 2026 | not disclosed | 262K per third-party listings | text, image, video in | API |

There is no Seed2.2, Seed2.5 or Seed3.0 as of October 14, 2026. The newest Seed releases after Seed2.1 are media and interaction models, covered next.

## The media lines, briefly

Seed's generative-media lines release on their own cadence. The use-case map and the [Seedance video-stack post](/blog/machine-learning/video-generation/seedance-and-the-bytedance-video-stack) cover them properly; here are the dated milestones only.

| Line | Release | Date | Source type |
| --- | --- | --- | --- |
| Seed-TTS | speech generation family | Jun 4, 2024 | arXiv report |
| Seedream 3.0 | bilingual text-to-image, up to 2K | Apr 15, 2025 | arXiv report |
| Seedance 1.0 | text/image-to-video | Jun 10, 2025 | arXiv report |
| Seedream 4.0 | one DiT for generation, editing, multi-image composition | Sep 24, 2025 | arXiv report |
| Seedance 2.0 | multimodal audio-video generation | launched Feb 2026; report Apr 15, 2026 | Seed publication |
| Seedream 5.0 Pro | image generation with design understanding | Jul 8, 2026 | Seed blog |
| Seed Audio 1.0 | speech-to-audio creation model | Jul 20, 2026 | Seed blog |
| Seedance 2.5 | video generation, one-take creation, flexible referencing | Jul 31, 2026 | Seed blog |
| SeedRealtime | full-duplex audio-visual LLM | Aug 5, 2026 | Seed blog |

SeedRealtime deserves a note because it blurs the line between media and the LLM family: it is a full-duplex audio-visual model meant for natural real-time interaction, the conversational counterpart to Seed2.0 Lite's omni-modal understanding.

## How the family evolved

![Lineage of the Seed language line: one MoE core per generation, specialists branching off it, open weights only for models far smaller than the core](/imgs/blogs/bytedance-seed-model-family-timeline-2.webp)

Read end to end, five through-lines run across the generations.

**1. The core is a roughly one-in-ten sparse MoE, and its size stopped being public.** Doubao-1.5-pro introduced sparse MoE with a 7x leverage claim. Seed1.5-Thinking disclosed 200B total / 20B active; Seed1.6 disclosed 230B / 23B. From Seed1.8 on, ByteDance stopped publishing parameter counts for the flagship. So the honest statement is: the last disclosed core was a 230B MoE with about 10% active, and nobody outside ByteDance knows what Seed2.x is. The lineage figure draws the core as one column for exactly that reason: it is the same design philosophy refreshed each generation, not a series of unrelated models.

**2. The thinking budget went from a training trick to a product dial.** Seed1.5-Thinking always thought. Seed1.6 learned when to think (AdaCoT). Seed-OSS-36B let the user set a token budget. Seed1.8 shipped configurable thinking modes for latency and cost. Seed2.0 split the family into tiers by how much reasoning you pay for. Over eighteen months, "how much does the model think" moved from the model's choice to the buyer's.

**3. Modalities folded inward.** In early 2025 vision was a separate model (Seed1.5-VL) and GUI agents were a separate team (UI-TARS). Seed1.6 folded vision into flagship pre-training. Seed1.8 folded the GUI agent, search agent and code agent into one model. Seed2.0 Lite added audio to become omni-modal. Each generation absorbed the previous generation's specialist.

**4. The target moved from contests to work.** Seed1.5-Thinking was sold on AIME and Codeforces. Seed1.8 was sold on agent benchmarks like BrowseComp and long-video tasks. Seed2.0's own page describes moving "from competition-level reasoning to research-grade tasks," and Seed2.1 leads with coding delivery stability and long-horizon agent work. The benchmarks in the launch posts are a good proxy for what ByteDance's paying customers ask for.

**5. Open weights are a side channel, never the flagship.** Every flagship from Skylark to Seed2.1 is API-only. Open releases are deliberately smaller or narrower: Seed-Coder-8B (MIT), Seed-OSS-36B (Apache-2.0, dense, not the MoE core), UI-TARS checkpoints, BAGEL. Some of them, like Seed-Coder and UI-TARS, do not branch from the MoE core at all. This is the opposite of DeepSeek or Qwen, where the open checkpoint is the flagship. If you want ByteDance's best model, you call an API; if you want ByteDance weights, you get a specialist.

## Which variant to pick today

As of October 2026, here is the short guide. For the fuller job-by-job version, see the [use-case map](/blog/machine-learning/frontier-models/bytedance-seed-model-universe-by-use-case).

| You need | Pick | Why |
| --- | --- | --- |
| Best ByteDance capability via API | Seed2.1 Pro | Latest flagship, strongest agent and coding claims |
| High-volume API traffic at lower cost | Seed2.1 Turbo, or Seed2.0 Mini for throughput-bound jobs | Cheaper tiers built for concurrency |
| Audio + video + image understanding in one call | Seed2.0 Lite (April 2026 update) | First omni-modal understanding model in the series |
| Open weights you can self-host and fine-tune | Seed-OSS-36B-Instruct | Apache-2.0, dense, 512K native context, thinking budget |
| A small open code model | Seed-Coder-8B (Instruct or Reasoning) | MIT, 8B, easy to serve on one GPU |
| Open GUI agent research | UI-TARS-1.5-7B | Apache-2.0, screenshot-in, action-out |
| Formal math | Seed-Prover 1.5 (research) | Lean proving with agentic RL |
| Open unified image understanding and generation | BAGEL | 14B / 7B active MoT, open |

Two cautions. First, the API model IDs are dated snapshots, and older snapshots get retired; pin the dated ID in production and watch the deprecation notices. Second, every flagship benchmark claim from Seed1.8 onward is vendor-reported. If a choice hinges on a few points of accuracy, run your own eval before switching.

## References

Primary sources for every release in this catalog, in timeline order.

- Doubao enterprise launch, May 15, 2024: [SCMP coverage](https://www.scmp.com/tech/big-tech/article/3262781/tiktok-owner-bytedance-launches-low-cost-doubao-ai-models-enterprises-initiating-price-war-crowded), [Caixin coverage](https://www.caixinglobal.com/2024-05-16/bytedance-enters-ai-arena-with-doubao-offering-ultra-low-cost-and-versatile-applications-102196732.html), [TechNode coverage](https://technode.com/2024/05/16/bytedance-surprises-ai-rivals-with-ultra-low-cost-doubao-model/) (no technical report exists for Skylark or the 2024 Doubao models).
- Seed-TTS: [arXiv:2406.02430](https://arxiv.org/abs/2406.02430).
- UI-TARS: [arXiv:2501.12326](https://arxiv.org/abs/2501.12326), [GitHub bytedance/UI-TARS](https://github.com/bytedance/UI-TARS).
- Doubao-1.5-pro (Seed1.5): [Seed official page](https://seed.bytedance.com/en/special/doubao_1_5_pro).
- Seed model portfolio (generation dates): [seed.bytedance.com/en/seed_model_portfolio](https://seed.bytedance.com/en/seed_model_portfolio).
- Seed1.5-Thinking: [arXiv:2504.13914](https://arxiv.org/abs/2504.13914).
- Seed1.5-VL: [arXiv:2505.07062](https://arxiv.org/abs/2505.07062).
- Seed-Coder: [arXiv:2506.03524](https://arxiv.org/abs/2506.03524), [Hugging Face Seed-Coder-8B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct).
- Seed1.6: [Introduction to Techniques Used in Seed1.6](https://seed.bytedance.com/en/blog/introduction-to-techniques-used-in-seed1-6), [Seed-1.6-Embedding launch](https://seed.bytedance.com/en/blog/built-on-seed1-6-flash-seed-1-6-embedding-launched).
- Seed-Prover: [arXiv:2507.23726](https://arxiv.org/abs/2507.23726).
- Seed Diffusion Preview: [arXiv:2508.02193](https://arxiv.org/abs/2508.02193).
- Seed-OSS-36B: [Hugging Face Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct).
- Doubao-Seed-Code (secondary only): [KrASIA](https://kr-asia.com/pulses/158276).
- Seed1.8: [Official release post](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model), [Seed1.8 Model Card, arXiv:2603.20633](https://arxiv.org/abs/2603.20633).
- Seed-Prover 1.5: [arXiv:2512.17260](https://arxiv.org/abs/2512.17260).
- Seed2.0: [Seed2.0 official page](https://seed.bytedance.com/en/seed2), [Seed2.0 Model Card](https://seed.bytedance.com/zh/public_papers/seed2-0-model-card-towards-intelligence-frontier-for-real-world-complexity), [TechNode launch coverage](https://technode.com/2026/02/14/bytedance-releases-doubao-seed-2-0/).
- Seed2.1: [Seed2.1 official page](https://seed.bytedance.com/en/seed2_1), [DataNorth launch coverage](https://datanorth.ai/news/bytedance-releases-seed-2-1-pro-and-seed-2-1-turbo).
- Media lines: [Seedream 3.0, arXiv:2504.11346](https://arxiv.org/abs/2504.11346), [Seedance 1.0, arXiv:2506.09113](https://arxiv.org/abs/2506.09113), [Seedream 4.0, arXiv:2509.20427](https://arxiv.org/abs/2509.20427), [Seed research and blog index](https://seed.bytedance.com/en/research) (Seedance 2.0 report, Seedream 5.0 Pro, Seed Audio 1.0, Seedance 2.5, SeedRealtime).
