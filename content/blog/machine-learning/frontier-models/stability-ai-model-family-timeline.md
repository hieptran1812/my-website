---
title: "The Stability AI Family, Release by Release: From Stable Diffusion 1.4 to Stable Audio 3"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of Stability AI's open-weight models from the August 2022 Stable Diffusion release to October 2026: SD 1.x and 2.x, SDXL and Turbo, Stable Video Diffusion, Stable Cascade, SD3 Medium, SD 3.5, the 3D line, Stable Audio Open and Stable Audio 3, and the StableLM language models, with dates, sizes, licenses and primary sources."
tags: ["stability-ai", "stable-diffusion", "sdxl", "sd3", "stable-video-diffusion", "stable-audio", "stablelm", "open-weights", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 20
---

Stability AI is the company that made open image generation a normal thing to run on a gaming GPU. On August 22, 2022 it announced the public release of Stable Diffusion, a latent diffusion model built on research from the CompVis group in Munich and Runway, with a 860M-parameter UNet that fit in under 10 GB of VRAM. Four years later the same org on Hugging Face holds more than a hundred repos across four modalities: images, video and 3D, audio, and language. The image line changed its backbone twice, the license changed at least four times, two whole generations of checkpoints quietly disappeared from the Hub, and the newest open release is not an image model at all. It is Stable Audio 3, from May 2026.

This post is the map. It walks every open-weight release in date order, says what changed from the one before, gives a spec table, names the license, and links a primary source. The timeline below is the mental model: a dense image lane from 2022 to late 2024, a 3D and video lane that kept shipping after the image line went quiet, an audio lane that ends at the right edge, and a short language lane that stops in 2024.

![Stability AI open-weight releases from August 2022 to October 2026 in four modality lanes, with hosted-only models in a separate bottom lane and Stable Audio 3 marked as the newest open release](/imgs/blogs/stability-ai-model-family-timeline-1.webp)

Read it as lanes, not as one line. **As of October 23, 2026, the newest open-weight Stability model is the Stable Audio 3 family (Small, Small SFX and Medium, announced May 20, 2026). The newest open image model is still Stable Diffusion 3.5 from October 2024.** I found no primary source for any newer open image model; the 2025 and 2026 image repos on the Hub are hardware ports (TensorRT, AMD GPU and NPU builds) of existing checkpoints.

A note on method. I pulled the Hugging Face Hub API listing for `stabilityai`, `CompVis` and `stable-diffusion-v1-5` sorted by creation date, each model's `cardData` license field and safetensors parameter total, the model cards themselves, the dated changelog in the `Stability-AI/generative-models` README, the arXiv reports, and Stability's own announcement posts. Stability's news pages print only month and day, so the year comes from the Hub creation date or the arXiv stamp next to it. Where only the Hub exists I write "Hub created", which is a lower bound on the public date. If you want the technique behind a release rather than its catalog entry, each section links the deeper post.

## The timeline at a glance

| Date | Release | Weights | One-line significance |
| --- | --- | --- | --- |
| 2021-12-20 | Latent Diffusion paper (arXiv 2112.10752) | n/a | Diffusion in a compressed autoencoder latent; the idea under SD 1 to SDXL |
| 2022-08-22 | Stable Diffusion 1.4 (with CompVis) | Open, CreativeML OpenRAIL-M | 860M UNet, 512px, runs on a consumer GPU |
| 2022-10 | Stable Diffusion 1.5 (released by Runway) | Open, same | 595k more aesthetic fine-tuning steps from v1.2 |
| 2022-11-24 | Stable Diffusion 2.0 | Open, OpenRAIL++-M | OpenCLIP text encoder, 768px, depth2img, 4x upscaler |
| 2022-12-07 | Stable Diffusion 2.1 | Open, same | Less aggressive NSFW filter, better anatomy |
| 2023-04 | StableLM Alpha 3B and 7B | Open, CC BY-SA-4.0 (base) | First Stability LLMs, 1.5T tokens |
| 2023-07-26 | SDXL 1.0 base and refiner | Open, OpenRAIL++-M | 3.5B base, 1024px native |
| 2023-11-21 | Stable Video Diffusion (SVD, SVD-XT) | Open, research only at launch | Image-to-video, 14 and 25 frames |
| 2023-11-28 | SDXL Turbo and SD-Turbo | Open, non-commercial at launch | Adversarial Diffusion Distillation, 1 to 4 steps |
| 2023-12 | Stable Zero123 (+ Zero123C) | Open, NC and Community | Novel views of an object from one image |
| 2024-01 to 04 | Stable Code 3B, Stable LM 2 1.6B and 12B | Open, Community License | 2T-token multilingual LLMs |
| 2024-02-12 | Stable Cascade | Open, non-commercial | Wuerstchen three-stage design, 24x24 latents |
| 2024-03-18 | SV3D | Open, NC and Community variants | Orbit video from one image, for 3D |
| 2024-06-05 | Stable Audio Open 1.0 | Open | 47-second text-to-audio from Freesound and FMA |
| 2024-06-12 | SD3 Medium | Open, NC research, relicensed 2024-07-05 | 2B MMDiT, rectified flow, 3 text encoders |
| 2024-08-01 | Stable Fast 3D | Open, Community License | Textured mesh in 0.5 s |
| 2024-10-22 / 29 | SD 3.5 Large, Large Turbo, Medium | Open, Community License | 8.1B and 2.5B MMDiT, QK norm |
| 2025-01-08 | SPAR3D | Open, Community License | Point cloud then mesh, editable |
| 2025-05-14 | Stable Audio Open Small | Open, Community License | 341M, on-device on Arm |
| 2025-05-20 | SV4D 2.0 | Open, research | 48 frames, 12 frames x 4 views |
| 2026-05-20 | Stable Audio 3 Small, Small SFX, Medium | Open, Stable Audio Community License | Up to 6 min 20 s, trained on licensed data |

Hosted-only siblings, one line each: SD3 Large and Large Turbo (2024, API at launch, hosted only), Stable Image Ultra and Stable Image Core (2024, hosted only), Stable Audio 1.0 and 2.0 (2023 to 2024, hosted only), and Stable Audio 3 Large (May 20, 2026, API and paid self-hosting only).

## Latent Diffusion: the paper underneath (December 20, 2021)

The family starts with "High-Resolution Image Synthesis with Latent Diffusion Models" ([arXiv 2112.10752](https://arxiv.org/abs/2112.10752)), from CompVis at LMU Munich and Runway. Its move was to run diffusion inside a pretrained autoencoder's latent space (downsampling factor 8) instead of pixel space, which cut training and sampling cost enough to make a public text-to-image model affordable. CompVis published class-conditional and small text-to-image LDM checkpoints on the Hub in July 2022 under Apache 2.0. Every Stability image model up to SDXL is this recipe scaled up. The mechanism is covered in depth in [Latent Diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).

## Stable Diffusion 1.x (August 22, 2022)

Stability's [public release post](https://stability.ai/news-updates/stable-diffusion-public-release) is dated August 22 and names Hugging Face and CoreWeave as collaborators; the weights live under `CompVis`. The [CompVis README](https://github.com/CompVis/stable-diffusion) gives the shape: an 860M UNet plus a 123M CLIP text encoder, running on a GPU with at least 10 GB VRAM. The v1 checkpoints are a training curriculum, not separate models: v1.1 trained 237k steps at 256px on LAION-2B-en then 194k at 512px, and v1.2 through v1.4 continued on aesthetic LAION subsets.

| Checkpoint | Hub repo created | Training (from model card) | License |
| --- | --- | --- | --- |
| v1.1 to v1.3 | 2022-08-09 to 08-19 | 256px then 512px on LAION subsets | CreativeML OpenRAIL-M |
| v1.4 | 2022-08-20 | v1.2 + 225k steps on laion-aesthetics v2 5+ | CreativeML OpenRAIL-M |
| v1.5 | October 2022 (Runway); re-hosted 2024-08-30 | v1.2 + 595k steps, 10% text dropout | CreativeML OpenRAIL-M |

**What changed in v1.5.** Only more fine-tuning: 595k steps on laion-aesthetics v2 5+ with 10% text-conditioning dropout for better classifier-free guidance, per the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). It was published by Runway, not Stability. The original `runwayml/stable-diffusion-v1-5` repo now redirects; the live copy sits in a `stable-diffusion-v1-5` org whose repos the Hub dates to August 30, 2024.

**License.** CreativeML OpenRAIL-M: weights redistributable and usable commercially, with a list of use restrictions that must be passed on to downstream users. This permissive start matters for the rest of the story.

## Stable Diffusion 2.0 and 2.1 (November 24 and December 7, 2022)

The [2.0 announcement](https://stability.ai/news-updates/stable-diffusion-v2-release) (November 24) changed the text encoder to OpenCLIP, trained by LAION with Stability's support, and shipped 512px and 768px text-to-image models, a depth-to-image model, a 4x upscaler and a new inpainting model. Training used an aesthetic subset of LAION-5B filtered with LAION's NSFW classifier. That filter removed many images of people, and the [2.1 release](https://stability.ai/news-updates/stablediffusion2-1-release7-dec-2022) on December 7 fine-tuned 2.0 with a less aggressive setting, claiming better anatomy, hands and art styles and support for non-standard aspect ratios.

| Model | Resolution | Text encoder | License |
| --- | --- | --- | --- |
| SD 2.0 / 2.0-base | 768px / 512px | OpenCLIP | CreativeML OpenRAIL++-M |
| SD 2.1 / 2.1-base | 768px / 512px | OpenCLIP | CreativeML OpenRAIL++-M |
| x4 upscaler | 4x, to 2048px and beyond | n/a | OpenRAIL++ (Hub created 2022-11-23, still live) |

**The surprise.** Neither generation is still where Stability put it. Today the Hub API returns 401 for `stabilityai/stable-diffusion-2`, `-2-1`, `-2-base` and `-2-1-base`, and the `Stability-AI/stablediffusion` GitHub repo returns Not Found. A Hugging Face forum thread shows users hitting the 404 by [November 14, 2025](https://discuss.huggingface.co/t/error-404-https-huggingface-co-stabilityai-stable-diffusion-2/170425). I found no official explanation. If you have a pipeline that pulls SD 2.x by repo ID, it is already broken.

## StableLM, Stable Code and Stable LM 2 (April 2023 to April 2024)

The language line deserves only a brief tour, because it stopped. StableLM Alpha 3B and 7B (Hub created April 11 to 19, 2023) were trained on about 1.5T tokens built on The Pile with a 4,096-token context; the base models were CC BY-SA-4.0 and the tuned ones CC BY-NC-SA-4.0. StableLM-3B-4E1T (Hub September 29, 2023) trained 1T tokens for 4 epochs on 256 A100s. In 2024 the line moved to the Community License: Stable Code 3B (2.7B parameters, 1.3T tokens, 16,384-token context, Hub January 9), Stable LM 2 1.6B (2T tokens, Hub January 18), and Stable LM 2 12B (12.1B, 2T tokens, Arcade100k tokenizer, Hub March 21; chat April 4). Japanese and Arabic variants followed through August 2024, and nothing since.

## SDXL 1.0 (July 26, 2023)

SDXL was the last and largest UNet. A research-only SDXL 0.9 appeared first (Hub created June 19, 2023; the generative-models changelog says June 22), and the [SDXL paper](https://arxiv.org/abs/2307.01952) followed on July 4. The [1.0 announcement](https://stability.ai/news-updates/stable-diffusion-sdxl-1-announcement) on July 26 described a 3.5B-parameter base model and a 6.6B-parameter ensemble pipeline with a refiner, generating natively at 1024x1024.

| Model | Parameters | Resolution | License |
| --- | --- | --- | --- |
| SDXL 0.9 base + refiner | not restated | 1024px | Research only |
| SDXL 1.0 base | 3.5B | 1024px | CreativeML OpenRAIL++-M |
| SDXL 1.0 base + refiner | 6.6B ensemble | 1024px | CreativeML OpenRAIL++-M |

**What changed versus 2.1.** A much larger UNet, native 1024px, and a two-stage base-then-refiner pipeline. The open license survived: OpenRAIL++-M, the same family as 2.x. SDXL 1.0 base is still the most-downloaded Stability repo in the Hub listing, at about 4.1M downloads in the trailing month when I pulled it. That number is a Hub counter, not a benchmark, but it says where the ecosystem settled.

![How the Stability image backbone changed twice: UNet latent diffusion from SD 1.x to SDXL with Turbo and SVD branches, the Wuerstchen three-stage Stable Cascade, then MMDiT rectified flow in SD3 and SD 3.5](/imgs/blogs/stability-ai-model-family-timeline-2.webp)

The lineage figure is the architectural summary of the whole image lane: one long UNet era with two offshoots, a short detour, then a transformer era. Notice that both offshoots of SDXL (Turbo and SVD) arrive in the same week of November 2023.

## Stable Video Diffusion (November 21, 2023)

The [SVD announcement](https://stability.ai/news-updates/stable-video-diffusion-open-ai-video-model) released two image-to-video models generating 14 and 25 frames at a user-chosen 3 to 30 fps; the [paper](https://arxiv.org/abs/2311.15127) followed on November 25. The SVD-XT card says it generates 25 frames at 576x1024 and was fine-tuned from the 14-frame model. SVD-XT 1.1 followed (Hub created February 2, 2024). At launch the release was explicitly "exclusively for research"; the current card has been rewritten to point commercial users at the Community License. The video mechanics are in [Latent Video Diffusion: SVD and AnimateDiff](/blog/machine-learning/video-generation/latent-video-diffusion-svd-and-animatediff).

## SDXL Turbo and SD-Turbo (November 28, 2023)

Turbo is a distillation, not a new model. The [announcement](https://stability.ai/news-updates/stability-ai-sdxl-turbo) and the [ADD paper](https://arxiv.org/abs/2311.17042) (same day) describe Adversarial Diffusion Distillation: score distillation from a teacher plus an adversarial loss, so one to four steps produce sharp images. Stability reports 207 ms for a 512x512 image on an A100 end to end, with the single UNet evaluation taking 67 ms, and that in blind tests one-step SDXL Turbo beat four-step LCM-XL. SD-Turbo is the same treatment applied to SD 2.1. The launch license was a non-commercial research license, the first time a flagship image checkpoint shipped without commercial rights.

## Stable Zero123 and the 3D line (December 2023 to November 2025)

The 3D and multi-view models are a lane of their own, and the one that kept shipping longest after the image line slowed.

| Release | Date | What it does | License |
| --- | --- | --- | --- |
| Stable Zero123 / Zero123C | Hub created 2023-12-08 | Novel views from one image; C trained only on CC-BY and CC0 objects | NC research / Community |
| TripoSR | Hub created 2024-02-29 | Fast single-image reconstruction | MIT |
| SV3D | 2024-03-18 ([arXiv 2403.12008](https://arxiv.org/abs/2403.12008)) | Orbit video from one image, built on SVD | NC and Community variants |
| SV4D | 2024-07-24 | Video to 4D, 5 frames x 8 views | Research |
| Stable Fast 3D | 2024-08-01 ([announcement](https://stability.ai/news-updates/introducing-stable-fast-3d)) | Textured, UV-unwrapped mesh in 0.5 s on a 7 GB GPU; retrained from TripoSR | Community |
| SPAR3D | 2025-01-08 ([arXiv 2501.04689](https://arxiv.org/abs/2501.04689)); Hub 2024-12-28 | Point cloud stage, then mesh; editable | Community |
| Stable Virtual Camera | Hub created 2025-02-19 | Novel-view video along a camera path | Non-commercial |
| SV4D 2.0 | 2025-05-20 | 48 frames, 12 video frames x 4 views, 576x576 | Research |
| SP4D | Hub created 2025-11-03 | Video-to-4D | Community |

The Zero123 split is a neat example of license-by-dataset: the variant trained with CC-BY-NC objects stays non-commercial, the one trained only on CC-BY and CC0 objects gets the commercial-capable license.

## Stable Cascade (February 12, 2024)

[Stable Cascade](https://stability.ai/news-updates/introducing-stable-cascade) is the detour in the lineage figure. Built on the [Wuerstchen](https://arxiv.org/abs/2306.00637) architecture, it splits generation into three stages: Stage C generates compact 24x24 latents for a 1024x1024 image, and Stages B and A decode them. The [model card](https://huggingface.co/stabilityai/stable-cascade) gives the sizes.

| Stage | Variants | Role |
| --- | --- | --- |
| Stage C | 1B and 3.6B (3.6B recommended) | Text-conditioned generation in the tiny latent |
| Stage B | 700M and 1.5B | Decompression toward the VQ latent |
| Stage A | 20M, fixed | VQ decoder to pixels |

The license is `stable-cascade-nc-community`: non-commercial only. Unlike SVD and Turbo, the current card still says so. It was a dead end architecturally; the next image model went to transformers.

## SD3 Medium (June 12, 2024)

The [SD3 paper](https://arxiv.org/abs/2403.03206), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", appeared March 5, 2024. It replaced the UNet with MMDiT, a transformer with separate weights for text and image tokens that attend jointly, trained with rectified flow instead of the classic noise schedule. The flagship SD3 Large went to the API first. The open release was [SD3 Medium](https://stability.ai/news-updates/stable-diffusion-3-medium) (Hub created May 30, announced June 12): a 2B-parameter model with three text encoders that can be used together or in subsets to trade quality for memory.

| Model | Parameters | Weights | License at launch |
| --- | --- | --- | --- |
| SD3 Medium | 2B | Open | Non-Commercial Research Community License |
| SD3 Large, Large Turbo | not disclosed | Hosted only | n/a |

**The license episode.** SD3 Medium shipped under a non-commercial research license with commercial use requiring a paid Stability license. On July 5, 2024 Stability published a [License Update](https://stability.ai/news-updates/license-update) introducing the **Stability AI Community License**: free for research, non-commercial and commercial use for individuals and organizations under USD 1 million in annual revenue, with an Enterprise license above that. It explicitly covered SD3 Medium, and the SD3 announcement now carries an "Edited on July 5th" note. The MMDiT recipe is unpacked in [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) and the training objective in [Flow Matching and Rectified Flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow).

## Stable Diffusion 3.5 Large, Large Turbo and Medium (October 22 and 29, 2024)

[SD 3.5](https://stability.ai/news-updates/introducing-stable-diffusion-3-5) is the last open image release in this catalog. Large and Large Turbo shipped October 22; Medium followed October 29.

| Model | Parameters | Resolution | Steps | License |
| --- | --- | --- | --- | --- |
| SD 3.5 Large | 8.1B | up to about 1 MP | standard | Community License |
| SD 3.5 Large Turbo | distilled from Large | up to about 1 MP | 4 | Community License |
| SD 3.5 Medium | 2.5B | 0.25 to 2 MP | standard | Community License |

**What changed versus SD3 Medium.** Query-Key normalization in the transformer blocks to stabilize training and ease fine-tuning, a much larger open checkpoint (8.1B, the first time Stability opened its largest image model at launch), and for Medium an "MMDiT-X" variant that, per its [model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), adds self-attention modules to the first 13 transformer layers for multi-resolution generation. ControlNets for Large (canny, depth, blur) followed on the Hub November 25, 2024.

![Parameter counts of Stability open image models to scale, from the 0.86B SD 1.x UNet to the 8.1B SD 3.5 Large](/imgs/blogs/stability-ai-model-family-timeline-3.webp)

The bars show roughly a 10x growth from the SD 1.x UNet to SD 3.5 Large, but read the footnote: SD 1.x is counted as the UNet alone and SDXL as Stability reported the pipeline, so the comparison is directional, not exact.

## Stable Audio Open 1.0 (June 5, 2024)

The audio lane opened with [Stable Audio Open](https://stability.ai/news-updates/introducing-stable-audio-open), announced June 5, 2024 (Hub created May 24), with the [paper](https://arxiv.org/abs/2407.14358) on July 19. It generates up to 47 seconds of audio from text and was trained on Freesound and Free Music Archive data. Stability later gave its size as 1.1B parameters. The earlier Stable Audio 1.0 and 2.0 products were hosted only. For the model design, see [Latent Diffusion for Music: Stable Audio](/blog/machine-learning/audio-generation/latent-diffusion-for-music-stable-audio) and the wider survey in [Diffusion for Audio](/blog/machine-learning/audio-generation/diffusion-for-audio).

## Stable Audio Open Small (May 14, 2025)

Built with Arm, [Stable Audio Open Small](https://stability.ai/news-updates/stability-ai-and-arm-release-stable-audio-open-small-enabling-real-world-deployment-for-on-device-audio-control) is a 341M-parameter model (versus 1.1B for the original) that produces up to 11 seconds of audio on a smartphone in under 8 seconds. License: Stability AI Community License. It is the first Stability audio model aimed at on-device inference.

## Stable Audio 3 (May 20, 2026)

The newest open release. Stability's announcement ("Meet Stable Audio 3") and [TechCrunch's report](https://techcrunch.com/2026/05/20/stability-ai-release-a-new-audio-model-that-can-create-six-minute-songs/) on May 20, 2026 describe four models; the Hub repos were created May 17.

| Model | Parameters (announced) | Max length | Weights |
| --- | --- | --- | --- |
| Stable Audio 3 Small SFX | 459M | not stated | Open |
| Stable Audio 3 Small (music) | 459M | up to 2 min | Open |
| Stable Audio 3 Medium | 1.4B | 6 min 20 s | Open |
| Stable Audio 3 Large | 2.7B | 6 min 20 s | API and paid self-hosting only |

**What changed versus Stable Audio Open.** Output length went from 47 seconds to six minutes and twenty seconds at the Medium tier, and Stability states every Stable Audio 3 model was trained on fully licensed data, after its earlier deals with Warner Music Group and Universal Music Group. Alongside the generators, the org published `SAME-S` and `SAME-L` autoencoder repos, base (pre-post-training) checkpoints for each size, and a `stable-audio-3-optimized` repo; several repos were updated as recently as October 6 to 10, 2026. The Hub's safetensors totals (2.31B for Medium, 568M for each Small) are larger than the announced sizes; my estimate is that the Hub figure counts bundled components such as the conditioner, but no source confirms that.

**License.** A new name: the cards say `stable-audio-community`, not the `stabilityai-ai-community` used by SD 3.5. Secondary coverage reports the same USD 1 million revenue threshold; I could not read the full license text, which is gated.

![Stability open audio models compared on parameters and maximum output length, from Stable Audio Open 1.0 at 1.1B and 47 seconds to Stable Audio 3 Medium at 1.4B and 380 seconds, with the hosted Large model shown greyed](/imgs/blogs/stability-ai-model-family-timeline-5.webp)

The two panels make the audio story in one glance: parameter counts stayed within a narrow band while maximum length grew about eightfold.

## What is latest as of October 2026

- **Newest open-weight release overall:** Stable Audio 3 Small, Small SFX and Medium (May 20, 2026), repos last modified October 2026.
- **Newest open image model:** SD 3.5 Large, Large Turbo and Medium (October 2024). Everything image-shaped on the Hub since then is a port: TensorRT builds (2025), AMD GPU builds (February and April 2025), and AMD NPU builds of SD-Turbo, SDXL, SD3 Medium and SD 3.5 Medium (February and May 2026).
- **Newest 3D:** SP4D (Hub created November 3, 2025).
- **Newest language model:** the Arabic Stable LM 2 variants (Hub created July and August 2024).
- **Hosted only:** Stable Audio 3 Large and the Stable Image API tiers.

## How the family evolved

![Stability license regimes over time: CreativeML OpenRAIL-M, OpenRAIL++-M, non-commercial research, and the Stability AI Community License, with each open release plotted on its launch license and SD3 Medium moving up in July 2024](/imgs/blogs/stability-ai-model-family-timeline-4.webp)

### 1. The license went permissive, then restrictive, then revenue-capped

The license figure is the most useful single view of the company's strategy. SD 1.x and 2.x and SDXL shipped under OpenRAIL variants that allowed commercial use. From November 2023 (SVD, SDXL Turbo) through June 2024 (Stable Cascade, SD3 Medium) new flagships launched non-commercial. The July 5, 2024 Community License reopened commercial use under a USD 1 million revenue cap, and every major release after it uses that license or a sibling of it. Stable Cascade never moved.

### 2. The image backbone changed twice

UNet latent diffusion carried SD 1.x through SDXL and the video and 3D offshoots. Stable Cascade tried an extreme-compression three-stage design. SD3 switched to MMDiT with rectified flow, and SD 3.5 refined it with QK normalization. Each switch made the previous generation's fine-tuning ecosystem (LoRAs, ControlNets) non-transferable, which is part of why SDXL stayed the most downloaded.

### 3. Distillation became a product line

SDXL Turbo (ADD, one to four steps), SD-Turbo and SD 3.5 Large Turbo (four steps) show a pattern: every flagship got a few-step distilled sibling within months or on the same day.

### 4. The center of gravity moved off images

Since October 2024 the open releases have been 3D, 4D and audio. Stable Audio 3 is the clearest bet: licensed training data, long outputs, and an open Medium model with a closed Large above it.

### 5. Old checkpoints are not guaranteed to stay

SD 1.5 moved orgs after Runway's repo disappeared. SD 2.x vanished from both the Hub and GitHub by late 2025. Mirror what you depend on.

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| Best open Stability image quality | SD 3.5 Large | 8.1B, Community License |
| Fast image generation | SD 3.5 Large Turbo, or SDXL Turbo for the ecosystem | 4 steps; Turbo has the widest tooling |
| Largest fine-tune and LoRA ecosystem | SDXL 1.0 base | OpenRAIL++-M, still the most downloaded |
| Lower VRAM image model | SD 3.5 Medium | 2.5B, 0.25 to 2 MP |
| Image to 3D asset | Stable Fast 3D or SPAR3D | Sub-second meshes, Community License |
| Image to video | SVD-XT 1.1 | The only open Stability video generator |
| Music or sound effects, local | Stable Audio 3 Medium; Small on CPU or phone | Licensed data, up to 6 min 20 s |
| Language model | Look elsewhere | The StableLM line has not shipped since 2024 |

## What this catalog could not verify

- The exact public date of SD 1.5 (October 2022 per secondary reports; Runway's original repo is gone).
- When and why the SD 2.x repos were removed; only the November 2025 forum report bounds it.
- The SD3 Large API launch date and the Stable Image Ultra and Core launch dates; listed as 2024 without a fetched primary source.
- The full text of the `stable-audio-community` license and whether it differs materially from the Stability AI Community License.
- Why Hub parameter totals for Stable Audio 3 exceed the announced sizes.

## References

- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), arXiv 2112.10752 (2021).
- Stability AI, [Stable Diffusion public release](https://stability.ai/news-updates/stable-diffusion-public-release) (Aug 22, 2022); [CompVis/stable-diffusion README](https://github.com/CompVis/stable-diffusion).
- [stable-diffusion-v1-5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- Stability AI, [Stable Diffusion 2.0](https://stability.ai/news-updates/stable-diffusion-v2-release) (Nov 24, 2022) and [2.1](https://stability.ai/news-updates/stablediffusion2-1-release7-dec-2022) (Dec 7, 2022).
- Podell et al., [SDXL](https://arxiv.org/abs/2307.01952), arXiv 2307.01952; Stability AI, [SDXL 1.0 announcement](https://stability.ai/news-updates/stable-diffusion-sdxl-1-announcement) (Jul 26, 2023).
- Blattmann et al., [Stable Video Diffusion](https://arxiv.org/abs/2311.15127), arXiv 2311.15127; [SVD announcement](https://stability.ai/news-updates/stable-video-diffusion-open-ai-video-model).
- Sauer et al., [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042), arXiv 2311.17042; [SDXL Turbo announcement](https://stability.ai/news-updates/stability-ai-sdxl-turbo).
- Pernias et al., [Wuerstchen](https://arxiv.org/abs/2306.00637), arXiv 2306.00637; [Stable Cascade announcement](https://stability.ai/news-updates/introducing-stable-cascade) and [model card](https://huggingface.co/stabilityai/stable-cascade).
- Esser et al., [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206), arXiv 2403.03206; [SD3 Medium announcement](https://stability.ai/news-updates/stable-diffusion-3-medium).
- Stability AI, [License Update](https://stability.ai/news-updates/license-update) (Jul 5, 2024).
- Stability AI, [Introducing Stable Diffusion 3.5](https://stability.ai/news-updates/introducing-stable-diffusion-3-5) (Oct 22, updated Oct 29, 2024).
- Voleti et al., [SV3D](https://arxiv.org/abs/2403.12008); [SF3D](https://arxiv.org/abs/2408.00653) and [Stable Fast 3D announcement](https://stability.ai/news-updates/introducing-stable-fast-3d); [SPAR3D](https://arxiv.org/abs/2501.04689); [generative-models changelog](https://github.com/Stability-AI/generative-models) (SV4D, SV4D 2.0).
- Evans et al., [Stable Audio Open](https://arxiv.org/abs/2407.14358); [Stable Audio Open announcement](https://stability.ai/news-updates/introducing-stable-audio-open); [Stable Audio Open Small announcement](https://stability.ai/news-updates/stability-ai-and-arm-release-stable-audio-open-small-enabling-real-world-deployment-for-on-device-audio-control).
- Stability AI, [Meet Stable Audio 3](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models) (May 20, 2026); [TechCrunch coverage](https://techcrunch.com/2026/05/20/stability-ai-release-a-new-audio-model-that-can-create-six-minute-songs/).
- Model cards: [StableLM Alpha](https://huggingface.co/stabilityai/stablelm-base-alpha-7b), [StableLM-3B-4E1T](https://huggingface.co/stabilityai/stablelm-3b-4e1t), [Stable Code 3B](https://huggingface.co/stabilityai/stable-code-3b), [Stable LM 2 1.6B](https://huggingface.co/stabilityai/stablelm-2-1_6b), [Stable LM 2 12B](https://huggingface.co/stabilityai/stablelm-2-12b), [Stable Zero123](https://huggingface.co/stabilityai/stable-zero123).
- [Hugging Face Hub API, stabilityai org listing](https://huggingface.co/api/models?author=stabilityai&sort=createdAt) (pulled October 23, 2026).
