---
title: "The Tencent Hunyuan Family, Release by Release: From HunyuanDiT to Hy4 Preview"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of Tencent Hunyuan's open-weight models across five lanes (LLM, image, video, 3D and world models, OCR and other) from May 2024 to October 2026, with dates, sizes, licenses, territory exclusions and primary sources."
tags: ["hunyuan", "tencent-hunyuan", "hy3", "hunyuanvideo", "hunyuan3d", "hunyuanimage", "open-weights", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 18
---

Tencent Hunyuan is not one model family. It is five that share a brand, a GitHub org (`Tencent-Hunyuan`) and, until mid-2026, a license. Between May 2024 and October 2026 the team published open weights for a 389B mixture-of-experts LLM and its successors, a text-to-image line that went from a 1.5B diffusion transformer to an 80B autoregressive MoE, a 13B video model and its lighter 8.3B rewrite, a 3D asset pipeline that turned into a world-model line, and a string of specialists: translation, OCR, motion, embodied agents. Most labs ship one flagship and a few distillations. Hunyuan shipped something open almost every month.

That pace makes the family hard to hold in your head, and the naming does not help. The brand shortened from "Hunyuan" to "HY" and then "Hy" (HY-MT1.5 in December 2025, Hy3 in April 2026), and some of the strongest models (Hunyuan-T1, TurboS, Hunyuan3D 3.0) never shipped weights at all. This post is the map: every open release by lane, in date order, with what changed, a spec table, the license, and a primary source. The figure below is the mental model: five lanes of open weights stacked over one thin lane of hosted-only models.

![Tencent Hunyuan release timeline from May 2024 to October 2026, five lanes of open-weight releases (LLM, image, video, 3D and world, other) above a muted lane of hosted-only models, with Hy3 and Hy4 preview highlighted](/imgs/blogs/hunyuan-model-family-timeline-1.webp)

Read it lane by lane. The image and video lanes front-load in 2024 and 2025; the LLM lane accelerates in 2026. One fact up front, because it is the question people ask most: **as of October 23, 2026, the newest open-weight Hunyuan flagship LLM is Hy4 preview (`tencent/Hy4-preview`, 770B total, 49B active, Hub created August 27, 2026, Apache 2.0), and the newest open Hunyuan model of any kind is Hy-Memory-30B-A3B (Hub created October 15, 2026).**

A note on method. I pulled the Hugging Face Hub API listing for the `tencent` and `Tencent-Hunyuan` organizations (with creation dates and safetensors parameter counts), the model cards and `LICENSE` files for each repo, the GitHub org's repo list and READMEs, and the arXiv metadata for each technical report. Where a README news line or a license "Release Date" exists, I use it. Where only the Hub exists I write "Hub created", a lower bound. Parameter counts labelled "Hub count" come from the Hub's safetensors total and include any vision tower or extra heads.

## The timeline at a glance

| Date | Release | Lane | License | One-line significance |
| --- | --- | --- | --- | --- |
| 2024-05-14 | HunyuanDiT (v1.1 Jun, v1.2 Jul 8) | Image | Hunyuan Community, no territory clause | 1.5B bilingual DiT, the first open Hunyuan weights |
| 2024-11-05 | Hunyuan-Large (A52B) | LLM | Hunyuan Community, EU excluded | 389B total, 52B active, 256K pretrain context |
| 2024-11-05 | Hunyuan3D 1.0 | 3D | Non-commercial | Text and image to 3D, built on HunyuanDiT |
| 2024-12-03 | HunyuanVideo | Video | Community, EU/UK/KR excluded | 13B text-to-video; the territory list widens here |
| 2025-01-21 | Hunyuan3D 2.0 | 3D | 3D 2.0 Community, 1M MAU cap | Shape DiT plus Paint texture model |
| 2025-01-24 | Hunyuan-7B-0124 | LLM | Tencent license (file not fetched) | Dense 7B, 256K, GQA |
| 2025-03-05 | HunyuanVideo-I2V | Video | Community | Image-to-video, 720p, 129 frames |
| 2025-03-18 | Hunyuan3D-2mv, 2mini | 3D | 3D 2.0 Community | Multi-view shape and a 0.6B mini |
| 2025-05 | HunyuanCustom, HunyuanVideo-Avatar | Video | Community | Subject-consistent and audio-driven video |
| 2025-06-13 | Hunyuan3D 2.1 | 3D | 3D 2.1 Community, 1M MAU | PBR materials, full training code |
| 2025-06-27 | Hunyuan-A13B | LLM | A13B Community | 80B total, 13B active, 256K |
| 2025-07-26 | HunyuanWorld 1.0 | World | World 1.0 Community, 1M MAU | Explorable 3D worlds from text or image |
| 2025-07-30 | Hunyuan 0.5B/1.8B/4B/7B | LLM | Community | Dense small models, 256K |
| 2025-08 | HunyuanVideo-Foley, GameCraft | Video | Community | Video-to-audio; interactive game video |
| 2025-09-01 | Hunyuan-MT-7B, MT-Chimera-7B | LLM | Community | Translation model plus ensemble model |
| 2025-09-08 | HunyuanImage 2.1 | Image | Community | 17B DiT, native 2K |
| 2025-09-28 | HunyuanImage 3.0 | Image | Community | 80B MoE, 13B active, autoregressive |
| 2025-11-21 | HunyuanVideo 1.5 | Video | Community | 8.3B, consumer GPUs, 1080p SR |
| 2025-11-25 | HunyuanOCR | Other | Community | About 1B end-to-end OCR VLM |
| 2025-12-17 | HY-World 1.5 (WorldPlay) | World | WorldPlay Community | Real-time interactive world model |
| 2025-12-30 | HY-MT1.5, HY-Motion 1.0 | LLM, Other | HY Community; Hunyuan Community | 1.8B/7B translation; 1.0B text-to-motion |
| 2026-01-26 | HunyuanImage 3.0-Instruct | Image | Community | Reasoning prompt enhancement, image-to-image |
| 2026-04-16 | HY-World 2.0 | World | World 2.0 Community | WorldMirror 2.0, Pano 2.0, WorldStereo 2.0 |
| 2026-04-23 | Hy3 preview | LLM | Hy Community, EU/UK/KR excluded | 295B total, 21B active |
| 2026-05-21 | Hy-MT2 | LLM | **Apache 2.0** | 1.8B, 7B, 30B-A3B translation |
| 2026-07 | Hy3, HunyuanOCR-1.5 | LLM, Other | **Apache 2.0**; Community | Hy3 drops all territory exclusions |
| 2026-08 (Hub 08-27) | Hy4 preview | LLM | **Apache 2.0** | 770B total, 49B active, 1M context |
| 2026-10-15 | Hy-Memory-30B-A3B | LLM | Apache 2.0 | Long-term-memory writer model |

Hosted only, no downloadable weights (one line each):

- **Hunyuan-T1**, March 2025: reasoning model built on TurboS, hosted only.
- **Hunyuan-TurboS**, report May 21, 2025 (arXiv 2505.15431): 560B total, 56B active hybrid Mamba-Transformer MoE, 16T tokens, 256K; hosted only, the GitHub repo holds the report and no weights.
- **Hunyuan3D 2.5**, report June 23, 2025 (arXiv 2506.16504): hosted only.
- **Hunyuan-Vision-1.5**, October 2025: Tencent Cloud API; the repo promises A56B and 4B checkpoints, none on the Hub as of this writing.
- **Hunyuan3D 3.0**: hosted only (no repo under either Hub org).

## LLM lane

### Hunyuan-Large (November 5, 2024)

The first Hunyuan LLM with open weights, and at release the largest open Transformer MoE. The README news line dates the open-sourcing of Hunyuan-A52B-Pretrain, -Instruct and -Instruct-FP8 to November 5, 2024, matching the license's release date; the technical report (arXiv 2411.02265) went up November 4. The abstract gives 389B total parameters, 52B activated, and up to 256K tokens. The weights were also distributed as zip files from Tencent's own CDN, and an instruct update shipped November 18.

| Variant | Total / active | Context | Notes |
| --- | --- | --- | --- |
| Hunyuan-A52B-Pretrain | 389B / 52B | 256K | Base |
| Hunyuan-A52B-Instruct (+FP8) | 389B / 52B | 128K | Updated 2024-11-18 |

License: Tencent Hunyuan Community License. Its "Territory" is "the worldwide territory, excluding the territory of the European Union", with a 100M monthly-active-user threshold above which you need a separate license. Source: [Hugging Face card](https://huggingface.co/tencent/Tencent-Hunyuan-Large).

### Hunyuan-7B-0124 (January 24, 2025)

A dense 7B (Pretrain and Instruct) with "better data allocation and training", 256K long-text capability and grouped-query attention, served through a vLLM fork and TensorRT-LLM. It matters mostly as the first small dense Hunyuan base. Source: [card](https://huggingface.co/tencent/Hunyuan-7B-Instruct-0124).

### Hunyuan-A13B (June 27, 2025)

The second MoE flagship went the other way on size: 80B total, 13B active, a fine-grained MoE with a native 256K window (shipped with `config.json` capped at 32K to save memory). The README's June 27 news line covers Pretrain, Instruct, FP8 and GPTQ-Int4 checkpoints plus a technical report PDF in the repo. Versus Hunyuan-Large it cut total parameters by about 5x and active by 4x, and added the reasoning-model tooling (tool parser, agent example) the market had moved to.

License: Tencent Hunyuan A13B Community License, release date June 27, 2025, and this one excludes the European Union, the United Kingdom and South Korea. Source: [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan-A13B).

![Two linear bar panels, total parameters on a 0 to 800B axis and active parameters on a 0 to 60B axis, for Hunyuan-Large, Hunyuan-A13B, Hy3 and Hy4 preview, with context length labels from 256K to 1M](/imgs/blogs/hunyuan-model-family-timeline-2.webp)

Each panel has its own linear scale, so compare bars within a panel. The bars make the flagship story plain: total size went down, back up, and then way up, while active parameters never left the 13B to 52B band. The ratio of active to total fell from 13.4% on Hunyuan-Large to 6.4% on Hy4 preview.

### Hunyuan dense 0.5B, 1.8B, 4B, 7B (July 30, 2025)

Eight checkpoints (Pretrain and Instruct at each size), trained with "strategies similar to the Hunyuan-A13B", each with a native 256K window. The 7B Instruct's Hub count is 7.50B. The license file says Tencent Hunyuan 7B, release date August 4, 2025, with the EU/UK/South Korea exclusion. Source: [Hunyuan-7B-Instruct card](https://huggingface.co/tencent/Hunyuan-7B-Instruct).

### Hunyuan-MT-7B and MT-Chimera-7B (September 1, 2025)

A translation model and what the card calls "the industry's first open-source translation ensemble model", Chimera, which merges several candidate translations into one. Both are 7B (Hub count 8.03B including embeddings). License: Tencent Hunyuan Community License, release date September 1, 2025, with the three-territory exclusion. Source: [card](https://huggingface.co/tencent/Hunyuan-MT-7B).

### HY-MT1.5 (December 30, 2025)

The first release under the "HY" prefix: a 1.8B and a 7B translation model covering mutual translation among 33 languages plus 5 ethnic and dialect variants, with terminology intervention, contextual and formatted translation. The 1.8B targets on-device use. Report: arXiv 2512.24092. Its license file is the first I found titled **Tencent HY Community License** (release date December 30, 2025): new name, same EU/UK/South Korea exclusion. Source: [card](https://huggingface.co/tencent/HY-MT1.5-7B).

### Hy3 preview (April 23, 2026)

The model the card calls "the first model trained on our rebuilt infrastructure". It is a 295B MoE with 21B active parameters, plus a 3.8B multi-token-prediction layer: 80 layers, 64 attention heads with 8 KV heads (GQA), 192 experts with top-8 routing, 256K context, a 120,832-token vocabulary. Hub created April 13; the README dates open-sourcing to April 23.

License: **Tencent Hy Community License** (the name HY-MT1.5 introduced), still excluding the EU, the UK and South Korea, still with the 100M MAU threshold. Source: [card](https://huggingface.co/tencent/Hy3-preview).

### Hy-MT2 (May 21, 2026)

Three "fast-thinking" translation models, 1.8B, 7B and 30B-A3B MoE, again across 33 languages, with an instruction-following benchmark (IFMTBench) released alongside. The report (arXiv 2605.22064) claims the 7B and 30B outperform DeepSeek-V4-Pro and Kimi K2.6 in fast-thinking mode; I have not reproduced that. A 1.25-bit quantization shrinks the 1.8B to 440 MB.

This is the first Hunyuan LLM I found under **Apache 2.0**, six weeks before the flagship followed. Source: [card](https://huggingface.co/tencent/Hy-MT2-30B-A3B).

### Hy3 (July 2026; Hub created July 2)

Same architecture as the preview (295B, 21B active, 256K), with scaled-up post-training and RL after feedback "from 50+ products". The card has no dated news line; the Hub repo was created July 2, and press coverage dates the announcement to July 6, 2026, which I take as approximate. The real change is the license: **Apache 2.0**, so the territory exclusions that had applied since December 2024 are gone for the flagship. Source: [card](https://huggingface.co/tencent/Hy3).

### Hy4 preview (Hub created August 27, 2026)

The current flagship. 770B total, 49B active, 78 layers (the first dense, the other 77 MoE with 256 routed experts plus 1 shared, top-8), one native MTP layer (10B total, 0.7B active) for speculative decoding, and a **1M-token context**. Architecturally it borrows from DeepSeek and GLM: gated DeepSeek Sparse Attention with IndexCache for cross-layer index reuse, and identity Hyper-Connections with 4 residual streams. Hub count is 780B including the MTP layer. The card is candid about known issues: it over-reasons and over-verifies. License: Apache 2.0. Source: [card](https://huggingface.co/tencent/Hy4-preview).

### Hy-Memory-30B-A3B (October 15, 2026)

A specialist: a 30B-total, roughly 3B-active model trained to run the write path inside memory frameworks (Mem0, MemOS, Hy-Memory), using on-policy distillation then RL with chunk-level credit (arXiv 2610.22196). Its tables compare against an "Hy3-A3B Base", which suggests a small Hy3-family base exists; I found no public repo for it. Apache 2.0. Source: [card](https://huggingface.co/tencent/Hy-Memory-30B-A3B).

## Image lane

### HunyuanDiT (May 14, 2024; v1.1 June, v1.2 July 8)

The first open Hunyuan weights. A 1.5B diffusion transformer with fine-grained Chinese understanding, conditioned by a bilingual CLIP and a 1.6B mT5, plus an optional 7.0B DialogGen model for multi-turn prompt rewriting (arXiv 2405.08748). Over the summer it gained v1.1 and v1.2 checkpoints, distilled variants, ControlNet (canny, depth, pose), LoRA, an IP-Adapter and the Hunyuan-Captioner. Its license (release date 2024/5/14) has the 100M MAU threshold but no territory clause at all. Source: [card](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

### HunyuanImage 2.1 (September 8, 2025)

A 17B single- and dual-stream DiT that only generates at 2K (2048x2048 at 1:1); 1K output produces artifacts. It pairs an MLLM text encoder with a character-aware multilingual encoder, uses a 32x-compression VAE aligned to DINOv2 features, and adds a refiner and a PromptEnhancer rewriter (arXiv 2509.04545). FP8 weights followed September 12, bringing 2K generation to 24 GB GPUs. Source: [card](https://huggingface.co/tencent/HunyuanImage-2.1).

### HunyuanImage 3.0 (September 28, 2025) and 3.0-Instruct (January 26, 2026)

The architectural break. Instead of a diffusion model conditioned on a frozen encoder, 3.0 is a native multimodal autoregressive MoE: 64 experts, over 80B total parameters, 13B active per token (arXiv 2509.23951), which the card calls the largest open image-generation MoE. vLLM acceleration landed October 30, 2025. The Instruct checkpoint (reasoning prompt enhancement and image-to-image) and an 8-step distilled variant followed January 26, 2026, though the Instruct Hub repo was created back in September. For the technique-level story, read [HunyuanImage and native multimodal generation](/blog/machine-learning/image-generation/hunyuanimage-and-native-multimodal-generation). Source: [GitHub](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0).

## Video lane

### HunyuanVideo (December 3, 2024)

A 13B-plus text-to-video model ("over 13 billion parameters, making it the largest among all open-source models", arXiv 2412.03603), with a CausalConv3D VAE compressing time, space and channels by 4, 8 and 16. It generates 720x1280 at 129 frames and needs 60 GB of GPU memory at that size; FP8 weights followed December 18. Its license introduced the territory list the family kept for 18 months: **the EU, the UK and South Korea**. For how it compares with Wan and CogVideoX, see [the open video frontier](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox). Source: [card](https://huggingface.co/tencent/HunyuanVideo).

### HunyuanVideo-I2V (March 5, 2025)

Image-to-video on the same backbone, up to 720p and 129 frames (5 seconds), with LoRA training for effects. Source: [card](https://huggingface.co/tencent/HunyuanVideo-I2V).

### HunyuanCustom and HunyuanVideo-Avatar (May 2025)

HunyuanCustom (Hub created May 8; arXiv 2505.04512) does subject-consistent customized video from image, audio and video conditions, with an editing checkpoint. HunyuanVideo-Avatar (Hub created May 26; arXiv 2505.20156) animates avatar images from audio for multiple characters, and its inference scripts load the `hunyuan-video-t2v-720p` checkpoint. HunyuanPortrait (Hub created May 27) is a sibling portrait-animation model. Sources: [Custom](https://huggingface.co/tencent/HunyuanCustom), [Avatar](https://huggingface.co/tencent/HunyuanVideo-Avatar).

### HunyuanVideo-Foley and Hunyuan-GameCraft (August 2025)

Foley (open-sourced August 28; arXiv 2508.16930) generates synchronized sound for video; an XL model followed September 29. GameCraft (weights August 14; arXiv 2506.17201) generates interactive game video from keyboard and mouse actions. Sources: [Foley](https://huggingface.co/tencent/HunyuanVideo-Foley), [GameCraft](https://huggingface.co/tencent/Hunyuan-GameCraft-1.0).

### HunyuanVideo 1.5 (November 21, 2025)

A rewrite for smaller hardware: an 8.3B DiT with a 3D causal VAE and a few-step super-resolution network up to 1080p (arXiv 2511.18870), with 480p and 720p T2V and I2V checkpoints plus CFG- and step-distilled variants (the 480p step-distilled I2V arrived December 5). It went from 13B to 8.3B and from 60 GB to consumer GPUs. Source: [card](https://huggingface.co/tencent/HunyuanVideo-1.5).

### OmniWeaving (April 3, 2026)

Built on HunyuanVideo 1.5 by the HunyuanVideo team: unified video generation with free-form composition and reasoning (arXiv 2603.24458), with the IntelligentVBench benchmark. Source: [card](https://huggingface.co/tencent/HY-OmniWeaving).

## 3D and world-model lane

### Hunyuan3D 1.0 (November 5, 2024)

A two-stage text- and image-to-3D framework that uses HunyuanDiT for the text-to-image step (arXiv 2411.02293), in lite and standard sizes. Its license is the odd one out: **Tencent Hunyuan Non-Commercial License**. Source: [card](https://huggingface.co/tencent/Hunyuan3D-1).

### Hunyuan3D 2.0, 2mv and 2mini (January 21 to April 2025)

A shape DiT plus a separate Paint texture model and a Delight model (arXiv 2501.12202), followed by multi-view (2mv) and a 0.6B mini shape model on March 18, Turbo variants and FlashVDM on March 19, and a turbo Paint model April 1. License: a 3D-specific community license with the three-territory exclusion and a much lower **1M MAU** threshold. Source: [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

### Hunyuan3D 2.1 (June 13, 2025)

"Fully open-sourced with new PBR model, VAE encoder, and all training code" (arXiv 2506.15442). The shape-plus-texture split stays; texturing moves to physically based materials. Same 1M MAU license pattern. Source: [card](https://huggingface.co/tencent/Hunyuan3D-2.1).

### HunyuanWorld 1.0, Voyager and WorldMirror (July to October 2025)

HunyuanWorld 1.0 (July 26; arXiv 2507.21809) generates explorable 3D worlds from text or an image, with a lite version for 4090-class GPUs on August 15. Voyager (September 2; arXiv 2506.04225) is an RGB-D video diffusion model for world-consistent exploration. HunyuanWorld 1.1, WorldMirror (October 22), is a roughly 1.2B feed-forward model that predicts cameras, depth and 3D from video or multi-view images (Hub count 1.26B). Each got its own community license. Sources: [World 1.0](https://huggingface.co/tencent/HunyuanWorld-1), [Mirror](https://huggingface.co/tencent/HunyuanWorld-Mirror).

### Hunyuan3D-Omni and Hunyuan3D-Part (September 2025)

Omni (3.3B, model table dated September 25; arXiv 2509.21245) adds multimodal controls (point cloud, voxel, skeleton, bounding box) to image-to-shape. Part (Hub created September 23) segments and generates parts. Source: [Omni card](https://huggingface.co/tencent/Hunyuan3D-Omni).

### HY-World 1.5, WorldPlay (December 17, 2025)

A real-time interactive world model with long-term geometric consistency. The flagship WorldPlay-8B is "based on HY Video"; a WAN-based 5B followed January 6, 2026, with RL post-training code (WorldCompass) on March 8. Source: [GitHub](https://github.com/Tencent-Hunyuan/HY-WorldPlay).

### HY-World 2.0 (April 16 to May 18, 2026)

A multi-modal world model to reconstruct, generate and simulate 3D worlds (arXiv 2604.14268), released in pieces: WorldMirror 2.0 weights April 16, HY-Pano 2.0 May 11, WorldStereo 2.0 and generation code May 18. HY-Pano 2.0's shipped code is HunyuanImage 3 model code, so the image lane now seeds the world lane. Source: [GitHub](https://github.com/Tencent-Hunyuan/HY-World-2.0).

![Lineage tree showing HunyuanDiT seeding Hunyuan3D 1.0, HunyuanVideo seeding I2V, Avatar and Custom, HunyuanVideo 1.5 seeding OmniWeaving and WorldPlay, HunyuanImage 3.0 seeding Instruct and HY-Pano 2.0, the 7B dense base seeding the MT line, and the flagship chain A13B to Hy3 to Hy4 preview](/imgs/blogs/hunyuan-model-family-timeline-4.webp)

The lineage shows why the lanes move together: a handful of backbones (DiT, the two HunyuanVideo generations, HunyuanImage 3.0) each seed three or four downstream releases.

## OCR and other lanes

### HunyuanOCR (November 25, 2025) and HunyuanOCR-1.5 (July 7, 2026)

An end-to-end OCR vision-language model covering document parsing, text spotting, information extraction and text-image translation, roughly 1B parameters (the current repo's Hub count is 1.12B). Version 1.5 (arXiv 2607.04884) keeps the backbone and adds DFlash speculative decoding, llama.cpp deployment and an "Agentic Data Flow" for training data; the 1.0 weights remain under `v1.0/`. License: Hunyuan Community License, release date November 25, 2025. Source: [card](https://huggingface.co/tencent/HunyuanOCR).

### HY-Motion 1.0 (December 30, 2025)

Flow-matching text-to-3D-human-motion (arXiv 2512.23464), in a 1.0B standard and 0.46B Lite. Source: [card](https://huggingface.co/tencent/HY-Motion-1.0).

### HY-Embodied (April 9 and July 15, 2026)

HY-Embodied-0.5 shipped a MoT-2B embodied model (arXiv 2604.07430), followed by VLA checkpoints in June and Hy-Embodied-VLM-1.0 on July 15, a 30B-total, roughly 3B-active MoE vision-language model for embodied agents under Apache 2.0. Source: [VLM-1.0 card](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0).

## What is latest as of October 2026

- **Flagship LLM:** Hy4 preview (770B, 49B active, 1M context, Apache 2.0). Hy3 is the stable 295B option.
- **Specialist LLMs:** Hy-MT2 for translation, Hy-Memory-30B-A3B for memory writing.
- **Image:** HunyuanImage 3.0-Instruct (and its distilled variant).
- **Video:** HunyuanVideo 1.5, with OmniWeaving on top.
- **3D and world:** Hunyuan3D 2.1 is the newest open asset model (2.5 and 3.0 are hosted); HY-World 2.0 for worlds.
- **OCR:** HunyuanOCR-1.5.

## How the family evolved

![License grid by release showing EU, UK and South Korea exclusion cells, MAU thresholds and commercial use from HunyuanDiT in May 2024 through the Apache 2.0 releases of 2026](/imgs/blogs/hunyuan-model-family-timeline-3.webp)

### 1. The license widened its exclusions, then dropped them

This is the single most useful fact for anyone deploying Hunyuan. HunyuanDiT (May 2024) had no territory clause. Hunyuan-Large (November 2024) excluded only the EU. HunyuanVideo (December 2024) added the UK and South Korea, and that three-territory list stayed on every community-licensed release, including Hy3 preview in April 2026. Then Hy-MT2 (May 2026), Hy3, Hy4 preview, Hy-Memory and Hy-Embodied-VLM-1.0 shipped under Apache 2.0. The grid makes the shape visible: the restricted cells fill up through 2025 and vanish in the 2026 rows. Note that the image, video, 3D and OCR lines are **still** under community licenses as far as I found.

### 2. The 3D lane ran on a stricter license

Hunyuan3D 1.0 was non-commercial. From 2.0 on the 3D and world licenses allow commercial use but cap it at 1M monthly active users, a hundredth of the 100M threshold on the LLM, image and video licenses.

### 3. The flagship LLM grew sparser, not just bigger

Active parameters stayed in a narrow band while total size swung from 389B to 80B to 770B. The architecture borrowed openly: GQA and MTP on Hy3, DeepSeek-style sparse attention and hyper-connections on Hy4 preview. Context went from 256K to 1M.

### 4. Image went from diffusion to autoregression

HunyuanDiT and HunyuanImage 2.1 are diffusion transformers; HunyuanImage 3.0 is an 80B MoE that generates in the same token stream as text. Video stayed diffusion, but got smaller (13B to 8.3B).

### 5. Backbones became platforms

Each big model fed the next lane: DiT into Hunyuan3D 1.0, HunyuanVideo into avatars, customization and world models, HunyuanImage 3.0 into panoramas for HY-World 2.0.

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| Open frontier LLM, any jurisdiction | Hy4 preview, or Hy3 for stability | Apache 2.0; 1M or 256K context |
| Single-GPU LLM | Hunyuan 7B or 4B dense | 256K context; check the EU/UK/KR clause |
| Translation, on device | Hy-MT2-1.8B | Apache 2.0; 440 MB at 1.25-bit |
| Text-to-image | HunyuanImage 3.0-Instruct | Reasoning prompt handling; needs serious hardware |
| Text-to-image, 24 GB GPU | HunyuanImage 2.1 FP8 | 2K output |
| Video on consumer GPUs | HunyuanVideo 1.5 | 8.3B, distilled variants |
| 3D assets | Hunyuan3D 2.1 | PBR textures, training code; 1M MAU cap |
| OCR | HunyuanOCR-1.5 | About 1B, llama.cpp support |

If you are in the EU, the UK or South Korea, only the Apache 2.0 models in this table are licensed for you at all.

## What this catalog could not verify

- **Hy3 and Hy4 preview release days.** Neither card has a dated news line; I give Hub creation dates (July 2 and August 27, 2026). The July 6 date for Hy3 comes from secondary press coverage.
- **Hunyuan-T1 specs and Hunyuan3D 3.0 details.** Hosted only; I found no primary model card, so I list them without numbers.
- **"Hy3-A3B Base".** Named in the Hy-Memory card's baselines; no public repo found.
- **Hunyuan-Vision-1.5 weights.** Promised in the repo; not on the Hub as of October 23, 2026.
- **Training-token counts** for most releases are not in the cards, so the charts use parameters and context only.

## References

- Hugging Face Hub: [tencent](https://huggingface.co/tencent) and [Tencent-Hunyuan](https://huggingface.co/Tencent-Hunyuan) organizations (model cards, LICENSE files, creation dates, safetensors counts).
- GitHub: [Tencent-Hunyuan org](https://github.com/Tencent-Hunyuan) READMEs and news logs.
- Hunyuan-Large, [arXiv 2411.02265](https://arxiv.org/abs/2411.02265).
- Hunyuan-TurboS, [arXiv 2505.15431](https://arxiv.org/abs/2505.15431).
- HY-MT1.5, [arXiv 2512.24092](https://arxiv.org/abs/2512.24092); Hy-MT2, [arXiv 2605.22064](https://arxiv.org/abs/2605.22064).
- Hy-Memory-30B-A3B, [arXiv 2610.22196](https://arxiv.org/abs/2610.22196).
- Hunyuan-DiT, [arXiv 2405.08748](https://arxiv.org/abs/2405.08748); HunyuanImage 2.1 PromptEnhancer, [arXiv 2509.04545](https://arxiv.org/abs/2509.04545); HunyuanImage 3.0, [arXiv 2509.23951](https://arxiv.org/abs/2509.23951).
- HunyuanVideo, [arXiv 2412.03603](https://arxiv.org/abs/2412.03603); HunyuanVideo 1.5, [arXiv 2511.18870](https://arxiv.org/abs/2511.18870); HunyuanCustom, [arXiv 2505.04512](https://arxiv.org/abs/2505.04512); HunyuanVideo-Avatar, [arXiv 2505.20156](https://arxiv.org/abs/2505.20156); HunyuanVideo-Foley, [arXiv 2508.16930](https://arxiv.org/abs/2508.16930); GameCraft, [arXiv 2506.17201](https://arxiv.org/abs/2506.17201); OmniWeaving, [arXiv 2603.24458](https://arxiv.org/abs/2603.24458).
- Hunyuan3D 1.0, [arXiv 2411.02293](https://arxiv.org/abs/2411.02293); 2.0, [arXiv 2501.12202](https://arxiv.org/abs/2501.12202); 2.1, [arXiv 2506.15442](https://arxiv.org/abs/2506.15442); 2.5, [arXiv 2506.16504](https://arxiv.org/abs/2506.16504); Omni, [arXiv 2509.21245](https://arxiv.org/abs/2509.21245).
- HunyuanWorld 1.0, [arXiv 2507.21809](https://arxiv.org/abs/2507.21809); Voyager, [arXiv 2506.04225](https://arxiv.org/abs/2506.04225); HY-World 2.0, [arXiv 2604.14268](https://arxiv.org/abs/2604.14268).
- HunyuanOCR-1.5, [arXiv 2607.04884](https://arxiv.org/abs/2607.04884); HY-Motion 1.0, [arXiv 2512.23464](https://arxiv.org/abs/2512.23464); HY-Embodied, [arXiv 2604.07430](https://arxiv.org/abs/2604.07430).
