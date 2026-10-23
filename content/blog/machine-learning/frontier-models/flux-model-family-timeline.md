---
title: "The FLUX Family, Release by Release: From FLUX.1 [schnell] to FLUX 3 Image Klein"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of Black Forest Labs' open-weight FLUX models from August 2024 to October 2026: FLUX.1 [schnell] and [dev], the Tools, Kontext [dev], Krea [dev], FLUX.2 [dev] and [klein], and the FLUX 3 Action and Image Klein checkpoints, with dates, block counts, sizes, licenses and primary sources."
tags: ["flux", "black-forest-labs", "flux-2", "flux-3", "flux-kontext", "rectified-flow", "text-to-image", "open-weights", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 18
---

Black Forest Labs is the rare image lab whose open checkpoints became the default for an entire ecosystem within weeks of launch. The company announced itself on August 1, 2024 with three models at once: a hosted flagship, a guidance-distilled open model for non-commercial use, and a step-distilled Apache 2.0 model anyone could ship. Two years later the family spans an image editor, a 32B frontier generator, sub-second distilled models, a robot action model and, as of October 15, 2026, a 5.6B open image model from the FLUX 3 generation.

That breadth makes the family hard to keep straight. The bracket tiers (`[pro]`, `[dev]`, `[schnell]`, `[klein]`) mean different things in different generations, licenses were renamed, and roughly half the named models never shipped weights. This post is the map. It walks every open-weight release in date order, says what changed from the previous one, gives a spec table, names the license, and links a primary source. The timeline below is the mental model: a steady drip of open checkpoints in the top lane, each paired with a hosted sibling in the bottom lane that stays one tier ahead.

![FLUX release timeline from August 2024 to October 2026 with open-weight releases in the top lane and hosted-only models in the bottom lane, FLUX 3 Image Klein highlighted as the newest open checkpoint](/imgs/blogs/flux-model-family-timeline-1.webp)

Read the dates on the cards; close releases are stacked so they do not collide. The answer people ask for most: **as of October 23, 2026, the newest open-weight FLUX image model is FLUX 3 Image Klein (`black-forest-labs/flux-3-image-klein`, announced October 15, 2026), and the newest FLUX.2 open checkpoint is FLUX.2 [klein] 9B-KV.** The flagship FLUX 3 Image and Video models are hosted or private-weight only.

A note on method. I pulled the Hugging Face Hub API listing for every repo under `black-forest-labs` (sorted by creation date), the model cards, the BFL blog index and each launch post, the reference code in `black-forest-labs/flux` and `black-forest-labs/flux2` (for block counts and encoders), and the Kontext paper on arXiv. Where a launch post exists I use its date. Where only the Hub exists I write "Hub created", which is a lower bound on the public date. Architecture numbers come from the reference code's config classes, not from marketing copy. For the technique-level story of the 12B transformer, read the companion [FLUX deep-dive](/blog/machine-learning/image-generation/flux-architecture-deep-dive).

## The timeline at a glance

| Date | Release | Weights | One-line significance |
| --- | --- | --- | --- |
| 2021-12-20 | Latent Diffusion paper (arXiv 2112.10752) | Prehistory | Rombach, Blattmann, Lorenz, Esser, Ommer: diffusion in a VAE latent |
| 2024-03 | Rectified flow transformers paper (arXiv 2403.03206) | Prehistory | Esser et al.: the MM-DiT and rectified-flow recipe behind SD3 |
| 2024-08-01 | FLUX.1 [schnell] | Open, Apache 2.0 | 12B, 1 to 4 steps, latent adversarial diffusion distillation |
| 2024-08-01 | FLUX.1 [dev] | Open, FLUX.1 [dev] Non-Commercial | 12B, guidance-distilled; became the community base model |
| 2024-11 (Hub 11-20) | FLUX.1 Tools [dev]: Fill, Depth, Canny, Redux | Open, same NC license | Inpainting, structural control, image variation; Depth and Canny also as LoRAs |
| 2025-01 (Hub) | ONNX builds of FLUX.1 | Open | TensorRT-ready exports made with NVIDIA |
| 2025-06-26 | FLUX.1 Kontext [dev] (paper arXiv 2506.15742) | Open, same NC license | 12B in-context editing by sequence concatenation |
| 2025-07-31 | FLUX.1 Krea [dev] | Open, same NC license | 12B aesthetic fine-tune with Krea, drop-in for [dev] |
| 2025-11-25 | FLUX.2 [dev] | Open, FLUX Non-Commercial | 32B, Mistral 24B text encoder, up to 10 references, new VAE |
| 2026-01-15 | FLUX.2 [klein] 4B and 9B (+ base) | 4B Apache 2.0; 9B NC | 4-step distilled generator plus editor; Qwen3 text encoders |
| 2026-03-09 (Hub) | FLUX.2 [klein] 9B-KV | Open, NC | KV cache for reference images, up to 2.5x faster multi-reference edits |
| 2026-04-06 (Hub) | FLUX.2 small decoder | Open, Apache 2.0 | Distilled VAE decoder, about 1.4x faster decode |
| 2026-09-23 | FLUX 3 Action | Open, FLUX Kommunity v1.0 | 7B world action model for robots |
| 2026-10-15 | FLUX 3 Image Klein (+ base, prompt upsampler) | Open, Kommunity; upsampler Apache 2.0 | 5.6B generator plus editor, 16-step distilled |

Hosted-only siblings, one line each:

- **FLUX.1 [pro]**, August 1, 2024: hosted only.
- **FLUX1.1 [pro]**, October 2, 2024: hosted only, launched with the BFL API.
- **FLUX1.1 [pro] Ultra and Raw modes**, November 6, 2024: hosted only.
- **FLUX.1 Fill, Depth and Canny [pro]**, November 2024: hosted only (Depth and Canny [pro] since deprecated).
- **FLUX.1 Kontext [pro] and [max]**, May 29, 2025: hosted only, launched with the BFL Playground.
- **FLUX.2 [pro] and [flex]**, November 25, 2025: hosted only.
- **FLUX.2 [max]**: hosted only; release date not verified from a BFL source (see the last section).
- **FLUX 3 Video and FLUX 3 Image**, announced July 23, 2026: Video in early access, Image "in the following weeks", both through APIs and private weight access, not public weights.

## Prehistory: from Latent Diffusion to rectified flow

The team did not start in 2024. The Latent Diffusion paper, submitted to arXiv on December 20, 2021, lists Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser and Björn Ommer; its core move (run diffusion in a compressed autoencoder latent instead of in pixels) became Stable Diffusion. In March 2024, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser, Kulal, Blattmann and colleagues) laid out the MM-DiT recipe: text and image tokens in separate weight streams that attend jointly, trained with a rectified-flow objective.

BFL's launch post claims exactly this lineage: VQGAN, Latent Diffusion, Stable Diffusion including SDXL and Stable Video Diffusion, Rectified Flow Transformers, and Adversarial Diffusion Distillation. Each of those shows up in FLUX. The Stability side of that history is cataloged in [the Stability AI model family timeline](/blog/machine-learning/frontier-models/stability-ai-model-family-timeline). The VAE latent comes from Latent Diffusion, the joint-attention blocks from the rectified-flow paper, and the 1-to-4-step [schnell] from adversarial distillation. For the architecture recipe in full, see [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).

## FLUX.1 [schnell] and [dev] (August 1, 2024)

The launch post announced Black Forest Labs, a 31 million US dollar seed round led by Andreessen Horowitz, and three models "scaled to 12B parameters". The architecture is described as "a hybrid of multimodal and parallel diffusion transformer blocks", trained with flow matching, using rotary positional embeddings and parallel attention layers.

The reference code makes that concrete. `FluxParams` for FLUX.1 sets `depth=19` double-stream blocks (separate weights for text and image tokens, joint attention) followed by `depth_single_blocks=38` single-stream blocks (one fused stream over the concatenated sequence), a hidden size of 3072 with 24 heads, 64 input channels (a 16-channel VAE latent packed 2x2), and rotary axes of `[16, 56, 56]`. Text arrives through two encoders: T5 v1.1 XXL at up to 512 tokens into the sequence, and CLIP ViT-L/14 as a pooled 768-dimensional vector into the modulation path.

| Variant | Params | Steps | Distillation | License |
| --- | --- | --- | --- | --- |
| FLUX.1 [schnell] | 12B | 1 to 4 | Latent adversarial diffusion distillation | Apache 2.0 |
| FLUX.1 [dev] | 12B | Not fixed by the card | Guidance distillation | FLUX.1 [dev] Non-Commercial |
| FLUX.1 [pro] | Not disclosed | n/a | n/a | Hosted only |

The split defined the family's business model for two years. [schnell] is the commercially free fast model; [dev] is the quality model whose weights are open but whose license forbids commercial use of the model itself (the card says generated outputs can be used commercially). [pro] is the API. Primary sources: [launch post](https://bfl.ai/announcements/24-08-01-bfl), [FLUX.1-dev card](https://huggingface.co/black-forest-labs/FLUX.1-dev), [FLUX.1-schnell card](https://huggingface.co/black-forest-labs/FLUX.1-schnell). Both Hub repos were created July 31, 2024.

## FLUX.1 Tools (November 2024)

Three months later BFL added four conditioning models, each in a [dev] open version and (except Redux) a [pro] API version:

| Tool | Input | Open [dev] form | Hub created |
| --- | --- | --- | --- |
| Fill | Image plus binary mask plus prompt | Full model | 2024-11-20 |
| Depth | Depth map plus prompt | Full model and LoRA | 2024-11-20 |
| Canny | Canny edges plus prompt | Full model and LoRA | 2024-11-20 |
| Redux | Image (adapter for variation and restyling) | Adapter | 2024-11-20 |

What changed versus [dev] is the input contract, not the backbone: Fill, Depth and Canny are fine-tuned [dev] variants that take extra conditioning channels, and the LoRA versions let you add structure control without a second 12B checkpoint. The [dev] tools ship under the same FLUX.1 [dev] license. The tools post now notes that Depth and Canny [pro] are deprecated in the API, while the open checkpoints "remain available for reference". In January 2025 BFL added ONNX exports of [dev], [schnell], Depth and Canny (Hub created January 18 and 28, 2025), the open side of the NVIDIA TensorRT collaboration announced January 2, 2025. Primary source: [Introducing FLUX.1 Tools](https://bfl.ai/blog/24-11-21-tools).

## FLUX.1 Kontext [dev] (June 26, 2025)

Kontext turned FLUX into an editor. The hosted Kontext [pro] and [max] launched May 29, 2025; the open [dev] followed on June 26, 2025 as a "12B parameter model that can run on consumer hardware". The Hub repo was created May 28, 2025, a month before it went public, which is why Hub dates alone mislead for this family.

The paper, "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space" (arXiv 2506.15742, submitted June 17, 2025, revised June 24), credits Black Forest Labs with 23 listed authors. The mechanism is "a simple sequence concatenation approach": the context image is encoded to latent tokens and appended to the sequence the transformer already processes, so local edits, global edits, character reference and style reference are all one forward pass with no new adapter. The paper introduced KontextBench, with 1026 image-prompt pairs across five task categories.

| Spec | FLUX.1 Kontext [dev] |
| --- | --- |
| Params | 12B (launch post) |
| Backbone | FLUX.1 transformer config (19 double, 38 single blocks) |
| Context | Reference image tokens concatenated to the sequence |
| Benchmark | KontextBench, 1026 pairs, 5 categories |
| License | FLUX.1 Non-Commercial, updated to require content filters or manual review |

The license story moves here. The same post launched a self-serve licensing portal where commercial rights to Kontext [dev], the Tools and FLUX.1 [dev] "can now be purchased with only a few clicks", and the [dev] license was amended to require content filtering. Primary sources: [Kontext [dev] post](https://bfl.ai/announcements/flux-1-kontext-dev), [paper](https://arxiv.org/abs/2506.15742).

## FLUX.1 Krea [dev] (July 31, 2025)

Krea [dev] is the open-weights version of Krea 1, built with Krea AI on a BFL base. The launch post frames it as an "opinionated" model that "overcomes the oversaturated 'AI look'". The model card calls it "a 12 billion parameter rectified flow transformer", trained with guidance distillation, and a "drop-in replacement in every system that supports the original FLUX.1 [dev]". The reference code confirms the same 19-plus-38 block config. License: FLUX.1 [dev] Non-Commercial, with commercial licenses sold through the portal. Hub created July 7, 2025. It is the last FLUX.1 checkpoint. Primary sources: [launch post](https://bfl.ai/announcements/flux-1-krea-dev), [model card](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev).

## FLUX.2 [dev] (November 25, 2025)

FLUX.2 is the first new architecture since launch. The open [dev] is a 32B rectified-flow transformer that "couples the Mistral-3 24B parameter vision-language model" with the generator, references up to 10 images at once, and outputs and edits at up to 4 megapixels. It also ships a VAE "re-trained from scratch", released separately under Apache 2.0.

What changed versus FLUX.1, from `Flux2Params` in the reference code:

| Spec | FLUX.1 | FLUX.2 [dev] |
| --- | --- | --- |
| Params | 12B | 32B |
| Double-stream blocks | 19 | 8 |
| Single-stream blocks | 38 | 48 |
| Hidden size / heads | 3072 / 24 | 6144 / 48 |
| MLP ratio | 4.0 | 3.0 |
| VAE latent channels | 16 | 32 |
| Text encoder | T5-XXL + CLIP-L | Mistral Small 3.2 24B (layers 10, 20, 30 stacked) |
| References | 1 (Kontext) | Up to 10 |

The block budget flipped: FLUX.2 keeps only eight double-stream blocks and puts the extra capacity into twice the width and more single-stream blocks. The CLIP pooled vector is gone; conditioning now comes from three stacked hidden layers of an LLM, which is why `context_in_dim` is 15360 (three stacked layers; 15360 / 3 = 5120 per layer, my arithmetic). License: the card calls it the FLUX Non-Commercial License. Hub created November 22, 2025; a BFL-produced NVFP4 build followed (Hub created December 31, 2025). A week later BFL announced a 300 million US dollar Series B (December 1, 2025). Primary sources: [FLUX.2 post](https://bfl.ai/blog/flux-2), [model card](https://huggingface.co/black-forest-labs/FLUX.2-dev).

## FLUX.2 [klein] (January 15, 2026)

The FLUX.2 post promised an Apache 2.0 [klein] "coming soon"; it arrived on January 15, 2026 as four checkpoints plus FP8 and NVFP4 builds of each:

| Checkpoint | Blocks (double + single) | Hidden | Text encoder | Steps | License | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| [klein] 4B | 5 + 20 | 3072 | Qwen3 4B | 4 (distilled) | Apache 2.0 | ~13 GB |
| [klein] base 4B | 5 + 20 | 3072 | Qwen3 4B | Undistilled | Apache 2.0 | Not stated |
| [klein] 9B | 8 + 24 | 4096 | Qwen3 8B | 4 (distilled) | FLUX Non-Commercial | ~29 GB |
| [klein] base 9B | 8 + 24 | 4096 | Qwen3 8B | Undistilled | FLUX Non-Commercial | Not stated |

All four reuse the FLUX.2 [dev] autoencoder. The launch post claims the 9B "matches or exceeds models 5x its size" and that images generate or edit "in under 0.5s on modern hardware" (charts measured on a GB200 in bf16); quantized builds run up to 1.6x faster with FP8 and up to 2.7x faster with NVFP4 on RTX 5080 and 5090. The post also renamed the "FLUX [dev] Non-Commercial License" to the "FLUX Non-Commercial License" with "no material changes".

Two follow-ups extended [klein] without a launch post. **[klein] 9B-KV** (Hub created March 9, 2026) caches the reference images' keys and values on the first denoising step and reuses them, which the card says speeds multi-reference editing "by up to 2.5 times". The **FLUX.2 small decoder** (Hub created April 6, 2026) is a distilled VAE decoder with about 28M parameters instead of about 50M, giving about 1.4x faster decoding and 1.4x less decode VRAM, under Apache 2.0. Primary sources: [klein post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence), [klein 4B card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

## FLUX 3 Action (September 23, 2026)

FLUX 3 itself was announced July 23, 2026 as "our new multimodal foundation model", learning jointly from images, video and audio and extending to action prediction. At launch the image and video models went to early access through APIs and private weights, and the post listed a planned open "FLUX 3 Dev" backbone without a date or license.

The first FLUX 3 open weights were not an image model. FLUX 3 Action, published September 23, 2026, is "an open weights 7B world action model": it takes camera frames, robot state and a text instruction, and returns the next chunk of actions plus the next video frames. The Hub collection holds a base checkpoint and two embodiment policies, `flux-3-action-so101` and `flux-3-action-droid` (all created September 22, 2026). The text encoder is an unmodified Qwen3-VL-4B-Instruct. It introduced a new license, the FLUX Kommunity License v1.0. Primary source: [FLUX 3 Action post](https://huggingface.co/blog/black-forest-labs/flux-3-action).

## FLUX 3 Image Klein (October 15, 2026)

FLUX 3 Image Klein is the second FLUX 3 open release and the first FLUX 3 image model with public weights. It is "a 5.6B open weights model for text-to-image generation and image editing", taking a prompt and up to 10 reference images.

![Bar chart of open-weight FLUX generator sizes in date order, rising from 12B to 32B for FLUX.2 dev then falling to 9B, 4B and 5.6B, bars colored by license](/imgs/blogs/flux-model-family-timeline-3.webp)

The chart shows the turn: after FLUX.2 [dev] reached 32B, every later open image checkpoint got smaller. The generator figures exclude text encoders, which matters most for FLUX.2 [dev], whose 24B encoder is nearly as large as the generator.

| Spec | FLUX 3 Image Klein |
| --- | --- |
| Generator params | 5.6B (11.2 GB BF16, 5.6 GB FP8) |
| Text encoder | Qwen3-VL-4B-Instruct (Apache 2.0) |
| Checkpoints | Distilled (16 steps, no guidance), base (50 steps, guidance 3.5); FP8, INT8, NVFP4 builds |
| Peak VRAM, B200, 1024x1024 | 31 GB BF16, 26 GB FP8, 25 GB NVFP4 |
| Latency (BFL's numbers) | Distilled, 16 steps: 0.55 s on GB300, 8.9 s on RTX 4070 |
| Versus [klein] 9B (BFL internal) | +91 Elo text-to-image, +70 Elo editing, 38% fewer params |
| License | FLUX Kommunity License v1.0 |

The architecture is not disclosed beyond needing "an implementation that supports flow-map sampling", a break from FLUX.1 and FLUX.2, whose configs are public. A companion 9B **prompt upsampler** (`flux-3-image-klein-pup`, a full fine-tune of Qwen3.5-9B, Apache 2.0) rewrites a short request into the long prompt the image model expects; the card is blunt that the image model "is not meant to run without a prompt upsampler in front of it". Hub created October 13 (model) and October 14, 2026 (upsampler, ComfyUI pack). Primary sources: [launch post](https://bfl.ai/blog/flux-3-image-klein), [model card](https://huggingface.co/black-forest-labs/flux-3-image-klein).

## Latest as of October 2026

| Question | Answer (October 23, 2026) |
| --- | --- |
| Newest open-weight image model | FLUX 3 Image Klein, 5.6B, October 15, 2026 |
| Newest open FLUX.2 checkpoint | [klein] 9B-KV (Hub March 9, 2026); small decoder (Hub April 6, 2026) |
| Largest open generator | FLUX.2 [dev], 32B |
| Newest open, commercially free without a revenue cap | FLUX.2 [klein] 4B (Apache 2.0) |
| Newest open model of any kind | FLUX 3 Image Klein; FLUX 3 Action (7B, robotics) a month earlier |
| Promised but not released | FLUX 3 Dev multimodal backbone |

## How the family evolved

### 1. Capacity moved from paired streams to the shared stream

![Stacked cell columns drawn to scale, one cell per transformer block, comparing FLUX.1, FLUX.2 dev, klein 9B and klein 4B double-stream and single-stream block counts and widths](/imgs/blogs/flux-model-family-timeline-2.webp)

Each column is one model, one cell per block, width proportional to hidden size. FLUX.1 spends a third of its depth on double-stream blocks (19 of 57). FLUX.2 [dev] spends a seventh (8 of 56) and doubles the width. The [klein] models sit in between (8 of 32 and 5 of 25 blocks are double-stream) and shrink width and depth.

### 2. The text side grew from encoders into an LLM

FLUX.1 read prompts through T5-XXL plus a CLIP pooled vector. FLUX.2 [dev] replaced both with a 24B Mistral vision-language model, [klein] with Qwen3 4B or 8B, and FLUX 3 Image Klein with Qwen3-VL-4B plus a separate 9B upsampler. By FLUX 3 the language side is a pipeline of its own.

### 3. Editing went from a separate model to the default

FLUX.1 needed separate checkpoints for inpainting (Fill), structure (Depth, Canny) and editing (Kontext). FLUX.2 [dev] and every [klein] do generation and multi-reference editing in one model, and 9B-KV optimizes exactly that path.

### 4. Distillation became the product line

[schnell] (adversarial, 1 to 4 steps) and [dev] (guidance) were both distilled from day one. [klein] made 4-step distillation the default and shipped undistilled base weights next to it for fine-tuning. FLUX 3 Image Klein repeats the pattern at 16 and 50 steps.

### 5. Licenses: Apache for the small, non-commercial for the flagship, then a revenue cap

![License grid with releases as rows and four license columns, showing Apache 2.0 for schnell, the VAE, klein 4B and the prompt upsampler, non-commercial for dev, Tools, Kontext, Krea, FLUX.2 dev and klein 9B, and FLUX Kommunity for FLUX 3](/imgs/blogs/flux-model-family-timeline-4.webp)

The grid shows the rule BFL has followed since 2024. The fast or small checkpoint ([schnell], [klein] 4B, VAE, decoder, upsampler) gets Apache 2.0. The best open checkpoint of each generation ([dev], Kontext, Krea, FLUX.2 [dev], [klein] 9B) stays non-commercial, with commercial rights sold through a portal since June 2025. FLUX 3 added a third tier: the Kommunity license allows free commercial use for individuals and companies with annual revenue under 5 million US dollars, and the card warns this "is not blanket permission for commercial fine-tuning or distribution".

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| Commercial product, no license negotiation | FLUX.2 [klein] 4B | Apache 2.0, 4 steps, ~13 GB VRAM, generation plus editing |
| Best open quality, research use | FLUX.2 [dev] | 32B, 10 references, 4 MP; budget for the 24B encoder |
| Small company, newest quality | FLUX 3 Image Klein | 5.6B, BFL reports +91 Elo over [klein] 9B; Kommunity license if revenue under 5 million US dollars |
| Fast multi-reference editing | [klein] 9B-KV | Up to 2.5x faster reference reuse; non-commercial |
| LoRA ecosystem and existing workflows | FLUX.1 [dev] or Krea [dev] | Largest fine-tune ecosystem; Krea is drop-in |
| Fully permissive and tiny steps on FLUX.1 tooling | FLUX.1 [schnell] | Apache 2.0, 1 to 4 steps |
| Robot policy research | FLUX 3 Action | The only open FLUX action model |

Treat every Elo and latency number above as BFL's own measurement; none is an independent benchmark.

## What this catalog could not verify

- **FLUX.1 Tools date.** The launch post's URL is dated `24-11-21` and the Hub repos were created November 20, 2024, but the BFL blog index today lists the post as November 1, 2024. I give "November 2024".
- **FLUX.2 [max].** It exists as a hosted model, but I found no BFL announcement with a date; third-party sources disagree (December 2025 vs March 2026).
- **FLUX 3 Image Klein architecture.** BFL discloses parameter count and sampling style but not block structure or autoencoder design, so it is absent from the block-stack figure.
- **FLUX 3 Dev.** Announced as planned on July 23, 2026; no weights, size or license as of this writing.
- **FLUX.1 [pro] size** has never been published.

## References

1. Black Forest Labs, "Announcing Black Forest Labs", August 1, 2024. https://bfl.ai/announcements/24-08-01-bfl
2. FLUX.1 [dev] and [schnell] model cards. https://huggingface.co/black-forest-labs/FLUX.1-dev, https://huggingface.co/black-forest-labs/FLUX.1-schnell
3. FLUX.1 reference code, `src/flux/util.py`. https://github.com/black-forest-labs/flux
4. Black Forest Labs, "Introducing FLUX.1 Tools". https://bfl.ai/blog/24-11-21-tools
5. Black Forest Labs, "FLUX.1 Kontext [dev]: Open Weights for Image Editing", June 26, 2025. https://bfl.ai/announcements/flux-1-kontext-dev
6. Black Forest Labs et al., "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space", arXiv 2506.15742. https://arxiv.org/abs/2506.15742
7. Black Forest Labs, "FLUX.1 Krea [dev]", July 31, 2025. https://bfl.ai/announcements/flux-1-krea-dev
8. Black Forest Labs, "FLUX.2: Frontier Visual Intelligence", November 25, 2025. https://bfl.ai/blog/flux-2
9. FLUX.2 reference code, `src/flux2/model.py`, `util.py`, `text_encoder.py`, `autoencoder.py`. https://github.com/black-forest-labs/flux2
10. Black Forest Labs, "FLUX.2 [klein]: Towards Interactive Visual Intelligence", January 15, 2026. https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
11. FLUX.2 [klein] 9B-KV and small decoder model cards. https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv, https://huggingface.co/black-forest-labs/FLUX.2-small-decoder
12. Black Forest Labs, "FLUX 3: Real World Models", July 23, 2026. https://bfl.ai/blog/flux-3
13. Black Forest Labs, "Introducing FLUX 3 Action", September 23, 2026. https://huggingface.co/blog/black-forest-labs/flux-3-action
14. Black Forest Labs, "FLUX 3 Image Klein", October 15, 2026. https://bfl.ai/blog/flux-3-image-klein
15. FLUX 3 Image Klein and prompt upsampler model cards. https://huggingface.co/black-forest-labs/flux-3-image-klein, https://huggingface.co/black-forest-labs/flux-3-image-klein-pup
16. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv 2112.10752. https://arxiv.org/abs/2112.10752
17. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", arXiv 2403.03206. https://arxiv.org/abs/2403.03206
18. Hugging Face Hub API, `black-forest-labs` organization listing (repo creation dates), retrieved October 23, 2026. https://huggingface.co/api/models?author=black-forest-labs
