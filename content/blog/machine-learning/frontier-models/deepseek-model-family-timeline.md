---
title: "The DeepSeek Model Family, in Order: Every Release from DeepSeek LLM to V4.1-Flash"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference catalog of every DeepSeek release from the 2023 DeepSeek LLM and Coder to the September 2026 V4.1-Flash, with dates, what changed each generation, variant tables, licenses and primary-source links."
tags: ["deepseek", "large-language-model", "mixture-of-experts", "model-family", "release-timeline", "reasoning", "open-weights", "frontier-models"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 32
---

DeepSeek is easy to misread as a company that ships one big model a year and a pile of side projects. Laid out in order, the pattern is different: four tracks run in parallel (a general line, a reasoning line, a code/math/prover line, and a vision/OCR line), and almost every idea that ends up in a flagship was first tried on a smaller specialist model. GRPO appeared in a 7B math model a year before R1 made it famous. Fine-grained experts appeared in a 16B MoE four months before V2. The timeline below puts every release in one picture.

![Swimlane timeline of DeepSeek releases from October 2023 to October 2026 in four tracks: general LLM, reasoning, code/math/prover, and vision/OCR](/imgs/blogs/deepseek-model-family-timeline-1.webp)

This post is a reference catalog, not a deep-dive. For each generation you get the release date, what changed against the previous one, a variant table and a primary-source link. Where this blog already has a close read of a release, it is linked inline, so you can go one level deeper from here. The cutoff is **14 October 2026**: the latest release that exists as of today is DeepSeek-V4.1-Flash (10 September 2026). V4.1-Pro has been announced as coming but does not exist yet, so it is not in the catalog.

**How dates are chosen.** DeepSeek often uploads weights to Hugging Face a few days before the announcement, and the arXiv report often lands weeks later. I use the public announcement date (API news page, GitHub release, or the day weights went public) as the release date, and give the paper date separately when they differ. Where only a month is verifiable, only a month is given.

## The timeline at a glance

| # | Release | Date | Track | Headline change | Weights |
|---|---|---|---|---|---|
| 1 | DeepSeek-Coder 1.3B to 33B | Nov 2023 | Code | 2T tokens, 87% code, 16K window, FIM | Open (DeepSeek License) |
| 2 | DeepSeek LLM 7B / 67B | 29 Nov 2023 | General | 2T bilingual tokens, scaling-law study | Open (DeepSeek License) |
| 3 | DeepSeekMoE 16B | Jan 2024 | General | Fine-grained + shared experts | Open (DeepSeek License) |
| 4 | DeepSeek-Coder 7B v1.5 | 25 Jan 2024 | Code | Coder continued from DeepSeek LLM 7B | Open |
| 5 | DeepSeekMath 7B | 5 Feb 2024 | Math | 120B math tokens, GRPO introduced | Open |
| 6 | DeepSeek-VL 1.3B / 7B | 8 Mar 2024 | Vision | Hybrid encoder, 1024x1024 input | Open |
| 7 | DeepSeek-V2 / V2-Lite | 7 May 2024 | General | MLA + DeepSeekMoE, 236B/21B, 128K | Open |
| 8 | DeepSeek-Prover V1 | 23 May 2024 | Prover | Large-scale synthetic Lean 4 data | Open (Aug 2024) |
| 9 | DeepSeek-Coder-V2 | 17 Jun 2024 | Code | V2 + 6T more tokens, 338 languages | Open |
| 10 | ESFT | 2 Jul 2024 | Research | Expert-specialized fine-tuning | Open |
| 11 | V2-Chat-0628 | Jul 2024 | General | Chat refresh | Open |
| 12 | DeepSeek-Prover-V1.5 | 15 Aug 2024 | Prover | RLPAF + RMaxTS | Open |
| 13 | DeepSeek-V2.5 | 5 Sep 2024 | General | Chat and Coder merged | Open |
| 14 | Janus 1.3B | 17 Oct 2024 | Multimodal | Decoupled encoders, understand + generate | Open |
| 15 | JanusFlow 1.3B | 12 Nov 2024 | Multimodal | Autoregression + rectified flow | Open |
| 16 | R1-Lite-Preview | 20 Nov 2024 | Reasoning | First visible-thinking model | Chat only |
| 17 | V2.5-1210 | 10 Dec 2024 | General | Final V2.5 refresh | Open |
| 18 | DeepSeek-VL2 | 13 Dec 2024 | Vision | MoE backbone, dynamic tiling | Open |
| 19 | DeepSeek-V3 | 26 Dec 2024 | General | 671B/37B, FP8, MTP, aux-loss-free | Open |
| 20 | DeepSeek-R1 / R1-Zero + distills | 20 Jan 2025 | Reasoning | Large-scale RL, MIT license | Open (MIT) |
| 21 | Janus-Pro 1B / 7B | 27 Jan 2025 | Multimodal | Scaled Janus | Open |
| 22 | DeepSeek-V3-0324 | 25 Mar 2025 | General | Post-training refresh, MIT | Open (MIT) |
| 23 | DeepSeek-Prover-V2 | 30 Apr 2025 | Prover | Subgoal decomposition RL, 671B | Open |
| 24 | DeepSeek-R1-0528 | 28 May 2025 | Reasoning | Deeper thinking, JSON + tools | Open (MIT) |
| 25 | DeepSeek-V3.1 | 21 Aug 2025 | General | Hybrid thinking, agent post-training | Open (MIT) |
| 26 | V3.1-Terminus | 22 Sep 2025 | General | Language consistency, agents | Open (MIT) |
| 27 | DeepSeek-V3.2-Exp | 29 Sep 2025 | General | DeepSeek Sparse Attention (DSA) | Open (MIT) |
| 28 | DeepSeek-OCR | Oct 2025 | OCR | Optical context compression | Open (MIT) |
| 29 | DeepSeekMath-V2 | 27 Nov 2025 | Math | Self-verifying proofs, 685B | Open (Apache-2.0) |
| 30 | DeepSeek-V3.2 / V3.2-Speciale | 1 Dec 2025 | General + Reasoning | DSA final, thinking in tool use | Open (MIT) |
| 31 | DeepSeek-OCR 2 | 27/28 Jan 2026 | OCR | DeepEncoder V2, visual causal flow | Open (Apache-2.0) |
| 32 | DeepSeek-V4 Preview (Pro, Flash) | 24 Apr 2026 | General | CSA/HCA attention, mHC, Muon, 1M | Open (MIT) |
| 33 | V4 DSpark checkpoints | 27 Jun 2026 | Inference | Speculative-decoding module attached | Open (MIT) |
| 34 | DeepSeek-V4-Flash-0731 | 31 Jul 2026 | General | Replaces V4-Flash preview, agents | Open (MIT) |
| 35 | DeepSeek-V4-Pro GA (0813) | 13 Aug 2026 | General | Official Pro, low/high/max effort | Open (MIT) |
| 36 | DeepSeek-V4-Flash-Vision-Exp | 21 Aug 2026 | Multimodal | First V4 image input | API, then open (MIT) |
| 37 | DeepSeek-V4.1-Flash | 10 Sep 2026 | General | Causal encoder-decoder MoE, native vision | Open (MIT) |

Thirty-seven entries. A few of them are refreshes (0628, 1210, 0324, Terminus) that I keep in the list because they changed something you would notice in production, usually the license or the API default. The rest of this post walks them generation by generation.

## Generation 0: the 2023 foundation (Coder and DeepSeek LLM)

DeepSeek's first public models were dense Llama-style transformers. Nothing about the architecture was new. What was new was the discipline: both models were trained from scratch on 2T tokens, both came with a careful data story, and the LLM paper is mostly a scaling-law methodology paper.

### DeepSeek-Coder (November 2023)

The first release was a code model. Weights went up on Hugging Face in late October 2023 and the family was announced in early November; the technical report followed on arXiv on 25 January 2024. The training mix was 87% code and 13% natural language (English and Chinese), 2T tokens from scratch, a 16K window, and a fill-in-the-middle objective so the models could infill, not just continue. Data was packed at repository level so a sample sees cross-file context. The close read of how that packing works is in [DeepSeek-Coder and Coder-V2](/blog/machine-learning/frontier-models/deepseek-coder-repo-packing-fim-continued-pretraining).

| Variant | Params | Context | Type | Weights |
|---|---|---|---|---|
| deepseek-coder-1.3b-base / -instruct | 1.3B dense | 16K | Code, base + instruct | Open |
| deepseek-coder-5.7bmqa-base | 5.7B dense (MQA) | 16K | Code, base | Open |
| deepseek-coder-6.7b-base / -instruct | 6.7B dense | 16K | Code, base + instruct | Open |
| deepseek-coder-33b-base / -instruct | 33B dense | 16K | Code, base + instruct | Open |
| deepseek-coder-7b-base-v1.5 / -instruct-v1.5 (Jan 2024) | 7B dense |  | Code, continued from DeepSeek LLM 7B | Open |

License: code under MIT, weights under the DeepSeek Model License, which permits commercial use with use-based restrictions. That two-license split holds for everything until R1.

Source: [arXiv 2401.14196](https://arxiv.org/abs/2401.14196), [GitHub deepseek-ai/DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder).

### DeepSeek LLM (29 November 2023)

Two sizes, 7B and 67B, each as Base and Chat. Both were trained on the same 2T-token English and Chinese corpus with a 4,096-token sequence length. The 7B uses multi-head attention; the 67B uses grouped-query attention. Chat models were made with SFT followed by DPO. The report, [arXiv 2401.02954](https://arxiv.org/abs/2401.02954) (5 January 2024), is subtitled "Scaling Open-Source Language Models with Longtermism", and its real contribution is a scaling-law study that fits optimal batch size and learning rate as functions of compute and re-derives the model/data split with a non-embedding FLOPs-per-token measure of model scale. That methodology is the subject of [DeepSeek LLM: the scaling-law methodology everyone skipped](/blog/machine-learning/frontier-models/deepseek-llm-scaling-laws-methodology).

| Variant | Params | Attention | Context | Tokens | Weights |
|---|---|---|---|---|---|
| deepseek-llm-7b-base / -chat | 7B dense | MHA | 4K | 2T | Open |
| deepseek-llm-67b-base / -chat | 67B dense | GQA | 4K | 2T | Open |

Source: [GitHub deepseek-ai/DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM).

**What this generation set up.** Two habits that never went away: train from scratch on your own corpus, and publish the methodology even when the model itself is ordinary. The 67B dense model also becomes the yardstick that V2 is measured against.

## Generation 1: the early-2024 specialists (MoE, Math, VL)

Between January and March 2024 DeepSeek shipped three small models that, in hindsight, are three of the four pillars of everything later: an MoE design, an RL algorithm, and a vision encoder strategy.

### DeepSeekMoE 16B (January 2024)

[arXiv 2401.06066](https://arxiv.org/abs/2401.06066) (11 January 2024) argues that conventional top-2 MoE (GShard style) produces experts that each learn a blurry mix of everything. The fix has two parts. **Fine-grained expert segmentation** splits each expert into several smaller ones and activates proportionally more of them, so the router can compose combinations instead of picking one generalist. **Shared expert isolation** reserves a few experts that every token uses, so common knowledge stops being duplicated across routed experts. The paper trained 2B, 16B and 145B versions; only the 16B was released.

| Variant | Total | Active | Context | Tokens | Weights |
|---|---|---|---|---|---|
| deepseek-moe-16b-base / -chat | 16.4B | about 2.8B | 4K | 2T | Open |

The released 16B matches DeepSeek LLM 7B at about 40% of the computation (the GitHub README gives 40.5%). The full lineage from this design to V3 is traced in [DeepSeek's MoE lineage](/blog/machine-learning/frontier-models/deepseek-moe-lineage-fine-grained-shared-experts).

Source: [GitHub deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE).

### DeepSeekMath 7B (5 February 2024)

[arXiv 2402.03300](https://arxiv.org/abs/2402.03300). DeepSeekMath continued pretraining from DeepSeek-Coder-Base-v1.5 7B on 120B math tokens mined from Common Crawl by an iterative fastText classifier. The base model is the less famous half. The famous half is **Group Relative Policy Optimization (GRPO)**: a PPO variant that drops the value network and computes the advantage of each sample relative to the other samples in its group. It cuts memory and turned out to be the RL workhorse of the whole family. The RL model reached 51.7% on MATH without tools or voting, 60.9% with 64-sample self-consistency.

| Variant | Params | Type | Weights |
|---|---|---|---|
| deepseek-math-7b-base | 7B dense | Math base | Open |
| deepseek-math-7b-instruct | 7B dense | Math SFT | Open |
| deepseek-math-7b-rl | 7B dense | Math, GRPO-trained | Open |

The data pipeline and the "RL sharpens rather than teaches" finding are covered in [DeepSeekMath: the data pipeline behind GRPO](/blog/machine-learning/frontier-models/deepseekmath-data-pipeline-and-grpo-origin); GRPO itself in [GRPO: group relative policy optimization](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization).

### DeepSeek-VL 1.3B / 7B (8 March 2024)

[arXiv 2403.05525](https://arxiv.org/abs/2403.05525). DeepSeek's first vision-language model paired a hybrid vision encoder (a low-resolution semantic encoder plus a high-resolution detail encoder) to handle images up to 1024x1024 at a fixed token budget, on top of DeepSeek LLM backbones.

| Variant | Params | Modality | Type | Weights |
|---|---|---|---|---|
| deepseek-vl-1.3b-base / -chat | 1.3B | Image + text in, text out | Base + chat | Open |
| deepseek-vl-7b-base / -chat | 7B | Image + text in, text out | Base + chat | Open |

## Generation 2: DeepSeek-V2 and the MLA breakthrough (May to December 2024)

### DeepSeek-V2 and V2-Lite (7 May 2024)

[arXiv 2405.04434](https://arxiv.org/abs/2405.04434) (submitted 7 May 2024). V2 is where DeepSeek stopped being "a good open Llama alternative" and started setting the agenda. Two ideas carry it:

- **Multi-head Latent Attention (MLA).** Instead of caching full keys and values per head, MLA caches a low-rank latent vector per token and reconstructs keys and values from it, with a small decoupled RoPE component so positional encoding survives the compression. The paper reports a **93.3% smaller KV cache** than DeepSeek 67B. The mechanism is unpacked in [Multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla), and its serving consequences in [MLA and attention variants at inference time](/blog/machine-learning/inference-engineering/mla-and-attention-variants-at-inference-time).
- **DeepSeekMoE at scale.** The 16B design scaled up to 236B total with 21B active.

Against DeepSeek 67B the paper claims 42.5% lower training cost and 5.76x maximum generation throughput. Pretraining used 8.1T tokens, and context went from 4K to 128K.

| Variant | Total | Active | Context | Tokens | Type | Weights |
|---|---|---|---|---|---|---|
| DeepSeek-V2 | 236B | 21B | 128K | 8.1T | Base | Open |
| DeepSeek-V2-Chat (SFT / RL) | 236B | 21B | 128K | 8.1T | Chat | Open |
| DeepSeek-V2-Lite | 15.7B | 2.4B | 32K | 5.7T | Base | Open (15 May 2024) |
| DeepSeek-V2-Lite-Chat | 15.7B | 2.4B | 32K | 5.7T | Chat | Open |
| DeepSeek-V2-Chat-0628 | 236B | 21B | 128K |  | Chat refresh | Open (Jul 2024) |

MLA plus sparse activation made V2 unusually cheap to serve, and that is the first time DeepSeek's architecture work showed up directly as an API price.

### DeepSeek-Coder-V2 (17 June 2024)

[arXiv 2406.11931](https://arxiv.org/abs/2406.11931). Coder-V2 is not trained from scratch: it continues pretraining from an intermediate V2 checkpoint on 6T additional tokens. Supported programming languages rose from 86 to 338, and context went from 16K to 128K. It was the first open model that was competitive with GPT-4 Turbo on code benchmarks. The continued-pretraining recipe is covered in the same [DeepSeek-Coder and Coder-V2](/blog/machine-learning/frontier-models/deepseek-coder-repo-packing-fim-continued-pretraining) post.

| Variant | Total | Active | Context | Type | Weights |
|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite-Base / -Instruct | 16B | 2.4B | 128K | Code base + instruct | Open |
| DeepSeek-Coder-V2-Base / -Instruct | 236B | 21B | 128K | Code base + instruct | Open |
| DeepSeek-Coder-V2-Instruct-0724 | 236B | 21B | 128K | Code instruct refresh | Open |

### DeepSeek-Prover V1 (23 May 2024) and V1.5 (15 August 2024)

The prover track works on Lean 4 formal proofs. **V1** ([arXiv 2405.14333](https://arxiv.org/abs/2405.14333)) is a data paper: translate informal competition problems into Lean, prove them with the model itself, keep what the proof assistant accepts, and iterate, producing a large synthetic corpus. **V1.5** ([arXiv 2408.08152](https://arxiv.org/abs/2408.08152)) adds RL from proof-assistant feedback (RLPAF) and **RMaxTS**, a Monte-Carlo tree search variant with an intrinsic-reward exploration bonus. It reaches 63.5% on miniF2F-test. Both are 7B models built on DeepSeekMath-Base; V1 weights were published alongside V1.5 in August 2024. Close read: [DeepSeek-Prover V1 and V1.5](/blog/machine-learning/frontier-models/deepseek-prover-synthetic-data-rmaxts).

| Variant | Params | Type | Weights |
|---|---|---|---|
| DeepSeek-Prover-V1 | 7B | Lean 4 prover (SFT on synthetic data) | Open (Aug 2024) |
| DeepSeek-Prover-V1.5-Base | 7B | Continued pretraining on math + Lean | Open |
| DeepSeek-Prover-V1.5-SFT | 7B | SFT | Open |
| DeepSeek-Prover-V1.5-RL | 7B | RLPAF-trained | Open |

### ESFT (2 July 2024)

A research release rather than a product model: [arXiv 2407.01906](https://arxiv.org/abs/2407.01906) shows that fine-tuning only the experts a task actually routes to (Expert-Specialized Fine-Tuning) matches full fine-tuning on fine-grained MoE models while forgetting less. The released checkpoints are 13 task-specific variants of V2-Lite (ESFT-gate-* and ESFT-token-* for intent, code, law, math, translation and summary, plus a vanilla baseline). It matters mainly because it shows that fine-grained experts really do specialize, which was the claim DeepSeekMoE was built on.

### DeepSeek-V2.5 (5 September 2024) and V2.5-1210 (10 December 2024)

V2.5 merged the two product lines: the [API announcement](https://api-docs.deepseek.com/news/news0905) describes it as a combination of DeepSeek-V2-0628 and DeepSeek-Coder-V2-0724, one model behind both the chat and coder endpoints. V2.5-1210 was the final V2-architecture refresh before V3, released two weeks before it. Same 236B/21B MoE, same 128K context, same DeepSeek Model License.

| Variant | Total | Active | Context | Weights |
|---|---|---|---|---|
| DeepSeek-V2.5 | 236B | 21B | 128K | Open |
| DeepSeek-V2.5-1210 | 236B | 21B | 128K | Open |

### Janus (17 October 2024) and JanusFlow (12 November 2024)

The unified multimodal line. [Janus](https://arxiv.org/abs/2410.13848) uses **two separate visual encoders**, one for understanding and one for generation (a VQ tokenizer), in front of a single autoregressive transformer, on the argument that the two tasks want different representations. [JanusFlow](https://arxiv.org/abs/2411.07975) keeps the understanding side and replaces image generation with rectified flow inside the same LLM. Both are 1.3B.

| Variant | Params | Modality | Weights |
|---|---|---|---|
| Janus-1.3B | 1.3B | Image understanding + image generation | Open |
| JanusFlow-1.3B | 1.3B | Image understanding + rectified-flow generation | Open |

### DeepSeek-R1-Lite-Preview (20 November 2024)

The first public sign of the reasoning line, and the only entry in this catalog that never got weights. The [announcement](https://api-docs.deepseek.com/news/news1120) claimed o1-preview-level performance on AIME and MATH and showed the full chain of thought in the chat UI. It was available only on chat.deepseek.com, with "open-source models and API coming soon". R1 two months later delivered on that.

### DeepSeek-VL2 (13 December 2024)

[arXiv 2412.10302](https://arxiv.org/abs/2412.10302). VL2 rebuilds the vision line on the V2 stack: a DeepSeekMoE language backbone with MLA, and **dynamic tiling** of high-resolution images so aspect ratios are preserved rather than squashed into one square. The three sizes differ mainly in the base LLM. The close read is [DeepSeek-VL and VL2](/blog/machine-learning/frontier-models/deepseek-vl-vl2-dynamic-tiling-moe).

| Variant | Active params | Base | Modality | Weights |
|---|---|---|---|---|
| deepseek-vl2-tiny | 1.0B | DeepSeekMoE (smallest) | Image + text in, text out | Open |
| deepseek-vl2-small | 2.8B | DeepSeekMoE | Image + text in, text out | Open |
| deepseek-vl2 | 4.5B | DeepSeekMoE-27B | Image + text in, text out | Open |

The model card states active parameters only; the base checkpoint is listed at 27B total.

## Generation 3: DeepSeek-V3 (26 December 2024)

[arXiv 2412.19437](https://arxiv.org/abs/2412.19437), [API announcement](https://api-docs.deepseek.com/news/news1226). V3 keeps V2's skeleton (MLA plus DeepSeekMoE) and scales it to 671B total parameters with 37B active, pretrained on 14.8T tokens. Four changes against V2 matter:

1. **FP8 mixed-precision training** at this scale, with fine-grained (tile- and block-wise) scaling to keep outliers under control. It was the first published proof that FP8 pretraining works for a frontier-class model.
2. **Auxiliary-loss-free load balancing.** Instead of an auxiliary loss that fights the language-modeling objective, a per-expert bias term is nudged up or down to keep experts balanced, and only affects routing, not the gating weights.
3. **Multi-token prediction (MTP)** as an extra training objective. The MTP module also doubles as a draft head for speculative decoding at inference.
4. **Co-designed infrastructure.** DualPipe pipeline parallelism, custom all-to-all kernels and memory tricks that let the full run fit in 2.788M H800 GPU hours.

The 2.788M GPU-hour figure, converted at a rental price, is where the widely quoted "trained for about 5.6M" number came from. The paper is explicit that it excludes research and ablation runs. That and the FP8/MTP/balancing details are in [How DeepSeek-V3 trained a 671B model](/blog/machine-learning/frontier-models/deepseek-v3-fp8-mtp-loss-free-balancing). In February 2025 DeepSeek open-sourced the infrastructure itself (FlashMLA, DeepEP, DeepGEMM, DualPipe, EPLB, 3FS) during "Open Source Week"; those repos are read as the V3 paper's footnotes in [Reading DeepSeek's open infrastructure](/blog/machine-learning/frontier-models/deepseek-open-infra-deepep-deepgemm-flashmla).

| Variant | Total | Active | Context | Tokens | Type | License | Weights |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3-Base | 671B | 37B | 128K | 14.8T | Base | DeepSeek Model License | Open |
| DeepSeek-V3 | 671B | 37B | 128K | 14.8T | Chat | DeepSeek Model License | Open |
| DeepSeek-V3-0324 (25 Mar 2025) | 671B | 37B | 128K | 14.8T | Chat refresh | **MIT** | Open |

Hugging Face lists the checkpoint at 685B because the repository also holds the 14B MTP module. V3-0324 is worth its own row for two reasons: the [announcement](https://api-docs.deepseek.com/news/news250325) reports much better reasoning, front-end coding and tool use from post-training alone, and it moved V3 to the MIT license, matching R1. From here on every general model is MIT.

## Generation 4: DeepSeek-R1 and the reasoning line (January to May 2025)

### DeepSeek-R1, R1-Zero and the distills (20 January 2025)

[arXiv 2501.12948](https://arxiv.org/abs/2501.12948) (v1 22 January 2025). R1 is a post-training result on V3-Base, and its headline is a negative one: you do not need human-written reasoning traces to get long chain-of-thought reasoning.

- **R1-Zero** applies large-scale GRPO directly to V3-Base with rule-based rewards (answer correctness plus a format reward), no SFT first. Reflection and long self-checking emerge on their own, along with poor readability and language mixing.
- **R1** fixes those problems with a small cold-start SFT set, then reasoning RL, then rejection sampling to build a broader SFT set, then a final RL stage covering helpfulness and harmlessness.
- **Distills.** Six dense models fine-tuned on about 800K samples generated by R1, with no RL on the students. The point the paper makes is that distillation from a strong reasoner beats running RL on the small model directly.

The close read is [DeepSeek-R1: incentivizing reasoning capability via RL](/blog/machine-learning/frontier-models/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning).

| Variant | Total | Active | Context | Base model | Type | Weights |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Zero | 671B | 37B | 128K | V3-Base | Pure-RL reasoning | Open (MIT) |
| DeepSeek-R1 | 671B | 37B | 128K | V3-Base | Reasoning | Open (MIT) |
| R1-Distill-Qwen-1.5B | 1.5B dense |  |  | Qwen2.5-Math-1.5B | Distilled reasoning | Open |
| R1-Distill-Qwen-7B | 7B dense |  |  | Qwen2.5-Math-7B | Distilled reasoning | Open |
| R1-Distill-Llama-8B | 8B dense |  |  | Llama-3.1-8B | Distilled reasoning | Open (Llama 3.1 license) |
| R1-Distill-Qwen-14B | 14B dense |  |  | Qwen2.5-14B | Distilled reasoning | Open |
| R1-Distill-Qwen-32B | 32B dense |  |  | Qwen2.5-32B | Distilled reasoning | Open |
| R1-Distill-Llama-70B | 70B dense |  |  | Llama-3.3-70B-Instruct | Distilled reasoning | Open (Llama 3.3 license) |

R1 was the first DeepSeek model under MIT, and the license explicitly allows distillation into other models. That single choice is a large part of why R1 traces became the default reasoning-SFT data for half the open ecosystem a few weeks later.

### Janus-Pro 1B / 7B (27 January 2025)

[arXiv 2501.17811](https://arxiv.org/abs/2501.17811). The same decoupled-encoder design as Janus, with an optimized training strategy, more data and a 7B size. It released one week after R1.

| Variant | Params | Modality | Weights |
|---|---|---|---|
| Janus-Pro-1B | 1B | Image understanding + generation | Open |
| Janus-Pro-7B | 7B | Image understanding + generation | Open |

### Native Sparse Attention (paper, 16 February 2025)

Not a model release, but it belongs on the timeline because it is the research ancestor of DSA. [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) proposes a hardware-aligned sparse attention that is trainable end to end, with compressed, selected and sliding-window branches. How NSA relates to the DSA that shipped in V3.2 is the subject of [Trainable sparse attention: NSA vs DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa).

### DeepSeek-Prover-V2 (30 April 2025)

[arXiv 2504.21801](https://arxiv.org/abs/2504.21801). The prover line jumps to the V3 backbone. V3 itself decomposes a theorem into subgoals and sketches the proof informally; a smaller prover closes the subgoals in Lean; the completed proofs are stitched back into chain-of-thought training data for cold-start, followed by RL. The 671B model reaches **88.9% on miniF2F-test** and solves **49 of 658** PutnamBench problems. The paper also introduces ProverBench, 325 formalized problems including 15 recent AIME problems, of which it solves 6.

| Variant | Total | Active | Base | Type | Weights |
|---|---|---|---|---|---|
| DeepSeek-Prover-V2-7B | 7B dense |  | Prover-V1.5-Base | Lean 4 prover | Open |
| DeepSeek-Prover-V2-671B | 671B | 37B | DeepSeek-V3-Base | Lean 4 prover | Open |

### DeepSeek-R1-0528 (28 May 2025)

A post-training refresh of R1 on the same base. The [announcement](https://api-docs.deepseek.com/news/news250528) lists better benchmark results, better front-end code, fewer hallucinations, and support for JSON output and function calling, which the original R1 lacked. The model thinks noticeably longer per problem than the January version. DeepSeek also distilled its chain of thought into a Qwen3 base.

| Variant | Total | Active | Context | Base | Weights |
|---|---|---|---|---|---|
| DeepSeek-R1-0528 | 671B | 37B | 128K | V3-Base | Open (MIT) |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense |  |  | Qwen3-8B-Base | Open (MIT) |

R1-0528 is the last standalone reasoning model. From V3.1 on, thinking is a mode of the main model, not a separate checkpoint.

## Generation 5: V3.1, V3.2 and sparse attention (August to December 2025)

### DeepSeek-V3.1 (21 August 2025) and V3.1-Terminus (22 September 2025)

V3.1 merges the two lines into one hybrid model: [the announcement](https://api-docs.deepseek.com/news/news250821) calls it "one model, two modes", with thinking and non-thinking switched by the chat template, both at 128K context. `deepseek-chat` and `deepseek-reasoner` became two modes of the same checkpoint. The base model got 840B tokens of extra long-context pretraining on top of V3: the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) says the 32K phase was extended tenfold to 630B tokens and the 128K phase 3.3x to 209B tokens. Weights and activations use the UE8M0 FP8 scale data format. Post-training focused on tool use and multi-step agents, which DeepSeek calls its "first step toward the agent era".

V3.1-Terminus ([announcement](https://api-docs.deepseek.com/news/news250922)) fixed the most complained-about V3.1 issues: Chinese/English mixing and stray characters, plus stronger code and search agents.

| Variant | Total | Active | Context | Modes | Weights |
|---|---|---|---|---|---|
| DeepSeek-V3.1-Base | 671B | 37B | 128K | Base | Open (MIT) |
| DeepSeek-V3.1 | 671B | 37B | 128K | Thinking + non-thinking | Open (MIT) |
| DeepSeek-V3.1-Terminus | 671B | 37B | 128K | Thinking + non-thinking | Open (MIT) |

### DeepSeek-V3.2-Exp (29 September 2025)

The experimental model that introduced **DeepSeek Sparse Attention (DSA)**. Built on V3.1-Terminus, it adds a lightweight "lightning indexer" that scores past tokens and a top-k selector, so each query attends to a small selected subset instead of the whole context. Quality is on par with Terminus; long-context cost drops enough that DeepSeek [cut API prices by more than 50%](https://api-docs.deepseek.com/news/news250929) the same day. The GPU kernels were released in TileLang and CUDA. Context stays at 128K.

| Variant | Total | Active | Attention | Weights |
|---|---|---|---|---|
| DeepSeek-V3.2-Exp-Base | 671B | 37B | MLA + DSA | Open (MIT) |
| DeepSeek-V3.2-Exp | 671B | 37B | MLA + DSA | Open (MIT) |

### DeepSeek-OCR (October 2025)

[arXiv 2510.18234](https://arxiv.org/abs/2510.18234) (21 October 2025; weights on Hugging Face from 17 October). The OCR model is really an experiment in **optical context compression**: render text as an image, encode it with a DeepEncoder into very few vision tokens, and decode with a small MoE (DeepSeek3B-MoE-A570M). At under 10x compression OCR precision is about 97%; at 20x it falls to about 60%. On OmniDocBench it beats GOT-OCR2.0 with 100 vision tokens per page. The interesting question it raises, whether long text context could be stored more cheaply as pixels, is explored in [DeepSeek-OCR: optical context compression](/blog/machine-learning/frontier-models/deepseek-ocr-optical-context-compression).

| Variant | Params | Modality | Weights |
|---|---|---|---|
| DeepSeek-OCR | about 3B (MoE decoder, about 570M active) plus encoder | Document image in, text out | Open (MIT) |

### DeepSeekMath-V2 (27 November 2025)

[arXiv 2511.22570](https://arxiv.org/abs/2511.22570). The math line moves from final-answer rewards to **self-verification**. A model trained only on correct final answers can still reason incorrectly, so DeepSeek first trains an LLM proof verifier, then trains the generator with the verifier as its reward model and rewards it for finding and fixing problems in its own proofs. With scaled test-time compute it reaches gold-level scores on IMO 2025 and CMO 2024 and 118/120 on Putnam 2024. Built on V3.2-Exp-Base, released under **Apache-2.0** (the only non-MIT large DeepSeek release). Close read: [Self-verifiable reasoning: DeepSeekMath-V2](/blog/machine-learning/frontier-models/self-verifiable-reasoning-deepseekmath-v2).

| Variant | Size | Base | Type | Weights |
|---|---|---|---|---|
| DeepSeek-Math-V2 | 685B checkpoint (V3.2-Exp architecture) | V3.2-Exp-Base | Natural-language theorem proving + verifier | Open (Apache-2.0) |

### DeepSeek-V3.2 and V3.2-Speciale (1 December 2025)

[arXiv 2512.02556](https://arxiv.org/abs/2512.02556), "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models" (submitted 2 December 2025). V3.2 is the finished version of V3.2-Exp. The report has three themes: DSA as the efficiency base, a much larger RL post-training budget, and an **agentic task synthesis pipeline** that generates large numbers of tool-use environments. The [announcement](https://api-docs.deepseek.com/news/news251201) describes V3.2 as the first DeepSeek model to integrate thinking directly into tool use, in both thinking and non-thinking modes. The report claims V3.2 performs comparably to GPT-5.

**V3.2-Speciale** is the same architecture with a high-compute reasoning-only post-training. It does not support tool calls. The report claims it surpasses GPT-5, matches Gemini-3.0-Pro on reasoning, and reached gold-medal performance at IMO 2025 and IOI 2025. On the API it ran on a temporary endpoint only until 15 December 2025, but the weights are open.

| Variant | Total | Active | Context | Type | Tool use | Weights |
|---|---|---|---|---|---|---|
| DeepSeek-V3.2 | 671B | 37B | 128K | Hybrid thinking, agentic | Yes, including while thinking | Open (MIT) |
| DeepSeek-V3.2-Speciale | 671B | 37B | 128K | Deep reasoning only | No | Open (MIT) |

### DeepSeek-OCR 2 (January 2026)

[arXiv 2601.20552](https://arxiv.org/abs/2601.20552), "DeepSeek-OCR 2: Visual Causal Flow" (submitted 28 January 2026; weights uploaded 27 January). The change against OCR 1 is the encoder. **DeepEncoder V2** reorders visual tokens by image semantics before the decoder reads them, instead of feeding them in fixed raster-scan order. The paper frames this as testing whether 2D understanding can come from two cascaded 1D causal reasoning structures. The model is 3B and released under Apache-2.0.

| Variant | Params | Modality | Weights |
|---|---|---|---|
| DeepSeek-OCR-2 | 3B | Document image in, text out | Open (Apache-2.0) |

## Generation 6: DeepSeek-V4 (April to August 2026)

### V4 Preview: V4-Pro and V4-Flash (24 April 2026)

[Announcement](https://api-docs.deepseek.com/news/news260424), [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (submitted 26 April 2026). The weights went up on 22 April and the preview was announced on the 24th. V4 is the first generation since V2 that replaces the attention mechanism rather than extending it, and its whole design goal is cheap native one-million-token context:

- **Hybrid attention.** Compressed Sparse Attention (CSA) interleaved with Heavily Compressed Attention (HCA), which replaces V3.2's MLA-plus-DSA stack.
- **Manifold-constrained hyper-connections (mHC)** in place of plain residual connections, to keep a very deep model's residual stream stable.
- **The Muon optimizer** at trillion-parameter scale.
- **FP4 + FP8 mixed precision** in the released instruct weights: MoE experts in FP4, most other parameters in FP8 (base checkpoints are FP8).
- **Specialist-then-distill post-training.** Domain experts trained with SFT and GRPO, then consolidated into one student by on-policy distillation.

The efficiency claim: at 1M tokens, V4-Pro needs about 27% of V3.2's single-token inference FLOPs and about 10% of its KV cache. Both models pretrain on more than 32T tokens, and 1M became the default context across DeepSeek's official services. Both support thinking and non-thinking modes. The architecture is read in detail in [DeepSeek-V4: how a 1.6T MoE reads a million tokens](/blog/machine-learning/frontier-models/deepseek-v4-million-token-context-moe).

| Variant | Total | Active | Context | Type | Precision | Weights |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro-Base | 1.6T | 49B | 1M | Base | FP8 | Open (MIT) |
| DeepSeek-V4-Pro (preview) | 1.6T | 49B | 1M | Instruct, non-think / high / max | FP4 + FP8 | Open (MIT) |
| DeepSeek-V4-Flash-Base | 284B | 13B | 1M | Base | FP8 | Open (MIT) |
| DeepSeek-V4-Flash (preview) | 284B | 13B | 1M | Instruct, non-think / high / max | FP4 + FP8 | Open (MIT) |

The legacy `deepseek-chat` and `deepseek-reasoner` API names were routed to V4-Flash and scheduled for retirement on 24 July 2026.

### DSpark checkpoints (27 June 2026)

DeepSeek-V4-Pro-DSpark and DeepSeek-V4-Flash-DSpark. The model cards are explicit that these are **not new models**: the same checkpoints with a DSpark speculative-decoding module attached, enabled in vLLM with `method: dspark` and in SGLang with a speculative-algorithm flag. The next day DeepSeek also published small draft heads for third-party models (DSpark, DFlash and EAGLE-3 drafters for Qwen3 4B/8B/14B and Gemma 4 12B) through its DeepSpec project. Every later V4 checkpoint ships with DSpark attached.

### V4-Flash-0731 (31 July 2026) and V4-Pro GA / V4-Pro-0813 (13 August 2026)

These two releases are where V4 left preview. Both model cards say they supersede the preview with stronger agentic and production performance, three reasoning-effort levels (low, high, max), and a recommended maximum output of 384K tokens at high and max effort. Both drop the Jinja chat template for a Python `encoding` folder. The [Pro GA announcement](https://api-docs.deepseek.com/news/news260813) also added native OpenAI Responses API support and off-peak pricing at 50% of peak.

The improvements are large on agent benchmarks. The Flash-0731 card lists Terminal Bench 2.1 at 82.7 against the preview's 61.8, and says it beats the V4-Pro preview on its listed benchmarks despite far fewer active parameters. The Pro-0813 card lists Terminal Bench 2.1 at 87.9 against 72.1 for the preview.

| Variant | Size on HF | Architecture | Reasoning effort | Weights |
|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 304B (incl. DSpark module) | V4-Flash + DSpark | low / high / max | Open (MIT) |
| DeepSeek-V4-Pro-0813 | 1.7T (incl. DSpark module) | V4-Pro + DSpark | low / high / max | Open (MIT) |

Neither card restates active parameters; they reuse the preview structure, so 49B (Pro) and 13B (Flash) active still apply to the backbone.

### DeepSeek-V4-Flash-Vision-Exp (21 August 2026)

DeepSeek's first experimental multimodal model in the V4 family, [announced](https://api-docs.deepseek.com/news/news260821) on the API on 21 August, with weights on Hugging Face from 31 August. It builds on V4-Flash with added visual modules and continued training. Text performance matches V4-Flash, and multimodal agent scores improve substantially. Images are billed at up to 384 tokens each, and a free Files API launched alongside it.

| Variant | Size on HF | Modality | Weights |
|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp | 305B | Image + text in, text out | Open (MIT) |

## Generation 7: DeepSeek-V4.1-Flash (10 September 2026)

[Announcement](https://api-docs.deepseek.com/news/news260910), [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [technical report (PDF on Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf). This is the latest release as of this writing, and it is a bigger change than the ".1" suggests.

- **Causal encoder-decoder (CED) architecture.** 40 transformer layers split into a 20-layer causal encoder and a 20-layer decoder. Input is processed with about 8B active parameters and output with about 16B: the model spends less compute reading the prompt than writing the answer, which fits long-context agent workloads where the input is mostly tool output.
- **MoE.** 552B backbone parameters, with 1 shared expert and 384 routed experts per MoE layer, 6 routed experts active per token. Hugging Face lists the full checkpoint at 763B including the vision encoder and other modules.
- **Native vision.** A DeepSeek-ViT encoder with a two-layer MLP projector, so image understanding is in the main model rather than a separate experimental checkpoint.
- **KV cache.** About a quarter of the HBM and an eighth of the SSD storage of V4.
- **Reasoning effort** is a continuous setting from 1 to 100 instead of three levels.

The announcement claims results ahead of V4-Pro. V4-Flash and V4-Flash-Vision-Exp were retired on the API the same day, and from 14 September `deepseek-v4-pro` requests route to V4.1-Flash "until V4.1-Pro launches". That is an announcement, not a release, so V4.1-Pro is not in this catalog.

| Variant | Backbone | Active (prefill / decode) | Context | Modality | Weights |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash-Base | 552B | 8B / 16B | up to 1M | Image + text in, text out | Evaluated in the card; no separate repo published |
| DeepSeek-V4.1-Flash | 552B | 8B / 16B | up to 1M | Image + text in, text out | Open (MIT) |

## How the family evolved

![Staircase of DeepSeek generations, each step adding one efficiency idea: dense LLM, DeepSeekMoE, MLA in V2, FP8/MTP/aux-loss-free in V3, GRPO-based RL in R1, DSA in V3.2, CSA/HCA/mHC/Muon in V4, causal encoder-decoder in V4.1-Flash](/imgs/blogs/deepseek-model-family-timeline-2.webp)

Read across three years, the catalog has five through-lines.

**1. Every generation keeps the previous stack and adds one efficiency idea.** DeepSeek LLM was a dense baseline. DeepSeekMoE added fine-grained and shared experts. V2 added MLA. V3 added FP8 training, MTP and bias-based balancing. V3.2 added DSA. V4 replaced attention with CSA/HCA and added mHC and Muon. V4.1-Flash split the model into an encoder and a decoder with different active sizes. The figure above is the whole architectural story, and almost every step is about cost per token, not raw capability. Capability came from scale and RL; architecture made the scale affordable.

**2. Specialists are the testbed.** GRPO was invented on a 7B math model in February 2024 and became the core of R1 in January 2025. Fine-grained experts shipped in a 16B model four months before V2. ESFT checked that experts really specialize. DeepSeek-OCR tested whether pixels compress text context. DeepSeekMath-V2's verifier-as-reward idea predates the generative reward models in V4's post-training. If you want to guess what the next flagship contains, read the latest specialist paper.

**3. Reasoning went from a separate model to a dial.** R1-Lite-Preview (chat only), R1 (separate checkpoint), R1-0528 (separate checkpoint with tools), V3.1 (a mode of the main model), V3.2 (thinking inside tool use), V4 (non-think / high / max), V4.1-Flash (effort 1 to 100). The reasoning line in the timeline figure goes quiet after R1-0528 because it was absorbed, not abandoned.

**4. Context length tracks the attention mechanism.** 4K with MHA/GQA, 128K once MLA made the KV cache cheap, still 128K through V3.2 while DSA cut the compute, then 1M native in V4 once CSA/HCA cut both. Context never got extended by brute force; each jump waited for an attention change that paid for it.

**5. The license loosened as the models got stronger.** Everything through V3 shipped under the DeepSeek Model License (commercial use allowed, use-based restrictions). R1 moved to MIT with distillation explicitly permitted, V3-0324 followed, and every general model since is MIT. The OCR 2 and Math-V2 releases use Apache-2.0. Few labs made their license more permissive while their models moved closer to the frontier.

A sixth, smaller pattern is the release cadence. DeepSeek ships a preview or "Exp" model, lets the community break it, and then ships a dated checkpoint (0324, 0528, 0731, 0813) or a renamed final (V3.2, V4-Pro GA). If you deploy DeepSeek, pin the dated checkpoint, not the family name.

## Which variant to pick today

![Decision tree for picking a DeepSeek open-weight model in October 2026 by deployment constraint: frontier self-hosting, cheap API agent, image input, single-GPU reasoning, formal proofs, document OCR](/imgs/blogs/deepseek-model-family-timeline-3.webp)

Pick by deployment constraint first and capability second. As of October 2026:

- **API default for agents and chat:** DeepSeek-V4.1-Flash. It is what the API routes to, it is cheaper than V4-Pro, and DeepSeek claims it beats V4-Pro. It also takes images.
- **Self-hosting the strongest open DeepSeek:** DeepSeek-V4-Pro-0813 (1.6T total, 49B active, MIT) if you have a multi-node cluster and want the most knowledge per token. V4.1-Flash is the better choice if its inference support is mature in your serving stack; check vLLM/SGLang support before committing, since a new encoder-decoder layout takes engines time.
- **One big node, no 1M-context need:** DeepSeek-V3.2 (671B/37B, 128K, MIT) is still a well-supported, well-understood model with mature kernels everywhere.
- **Single GPU or laptop reasoning:** DeepSeek-R1-0528-Qwen3-8B, or R1-Distill-Qwen-14B/32B if you have more memory. They are old but small, MIT-friendly (Llama-based distills carry the Llama license), and run on every engine.
- **Formal theorem proving in Lean 4:** DeepSeek-Prover-V2-7B for local work, Prover-V2-671B for best results.
- **Natural-language competition proofs:** DeepSeek-Math-V2 (Apache-2.0), if you can serve a 685B checkpoint.
- **Document OCR and page-to-text at volume:** DeepSeek-OCR 2 (3B, Apache-2.0).
- **Avoid for new work:** everything under the old DeepSeek Model License (V2.x, Coder-V2, V3 original, VL2, Janus) unless you have a specific reason. The newer MIT models are stronger at every size where both exist.

## References

Primary sources only. Dates as given on the source.

**2023**
- DeepSeek-Coder: [arXiv 2401.14196](https://arxiv.org/abs/2401.14196) · [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)
- DeepSeek LLM: [arXiv 2401.02954](https://arxiv.org/abs/2401.02954) · [GitHub](https://github.com/deepseek-ai/DeepSeek-LLM)

**2024**
- DeepSeekMoE: [arXiv 2401.06066](https://arxiv.org/abs/2401.06066) · [GitHub](https://github.com/deepseek-ai/DeepSeek-MoE)
- DeepSeekMath: [arXiv 2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-VL: [arXiv 2403.05525](https://arxiv.org/abs/2403.05525)
- DeepSeek-V2: [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) · [HF V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- DeepSeek-Prover V1: [arXiv 2405.14333](https://arxiv.org/abs/2405.14333)
- DeepSeek-Coder-V2: [arXiv 2406.11931](https://arxiv.org/abs/2406.11931) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)
- ESFT: [arXiv 2407.01906](https://arxiv.org/abs/2407.01906)
- DeepSeek-Prover-V1.5: [arXiv 2408.08152](https://arxiv.org/abs/2408.08152)
- DeepSeek-V2.5: [API news, 2024-09-05](https://api-docs.deepseek.com/news/news0905) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)
- Janus: [arXiv 2410.13848](https://arxiv.org/abs/2410.13848)
- JanusFlow: [arXiv 2411.07975](https://arxiv.org/abs/2411.07975)
- DeepSeek-R1-Lite-Preview: [API news, 2024-11-20](https://api-docs.deepseek.com/news/news1120)
- DeepSeek-V2.5-1210: [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210)
- DeepSeek-VL2: [arXiv 2412.10302](https://arxiv.org/abs/2412.10302) · [HF](https://huggingface.co/deepseek-ai/deepseek-vl2)
- DeepSeek-V3: [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) · [API news, 2024-12-26](https://api-docs.deepseek.com/news/news1226)

**2025**
- DeepSeek-R1: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- Janus-Pro: [arXiv 2501.17811](https://arxiv.org/abs/2501.17811)
- Native Sparse Attention: [arXiv 2502.11089](https://arxiv.org/abs/2502.11089)
- DeepSeek-V3-0324: [API news, 2025-03-25](https://api-docs.deepseek.com/news/news250325)
- DeepSeek-Prover-V2: [arXiv 2504.21801](https://arxiv.org/abs/2504.21801)
- DeepSeek-R1-0528: [API news, 2025-05-28](https://api-docs.deepseek.com/news/news250528) · [HF Qwen3-8B distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- DeepSeek-V3.1: [API news, 2025-08-21](https://api-docs.deepseek.com/news/news250821) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
- DeepSeek-V3.1-Terminus: [API news, 2025-09-22](https://api-docs.deepseek.com/news/news250922)
- DeepSeek-V3.2-Exp: [API news, 2025-09-29](https://api-docs.deepseek.com/news/news250929) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
- DeepSeek-OCR: [arXiv 2510.18234](https://arxiv.org/abs/2510.18234) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- DeepSeekMath-V2: [arXiv 2511.22570](https://arxiv.org/abs/2511.22570) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2)
- DeepSeek-V3.2 / Speciale: [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) · [API news, 2025-12-01](https://api-docs.deepseek.com/news/news251201) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)

**2026**
- DeepSeek-OCR 2: [arXiv 2601.20552](https://arxiv.org/abs/2601.20552) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- DeepSeek-V4 Preview: [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) · [API news, 2026-04-24](https://api-docs.deepseek.com/news/news260424) · [HF V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- V4 DSpark: [HF V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
- DeepSeek-V4-Flash-0731: [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- DeepSeek-V4-Pro GA: [API news, 2026-08-13](https://api-docs.deepseek.com/news/news260813) · [HF V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
- DeepSeek-V4-Flash-Vision-Exp: [API news, 2026-08-21](https://api-docs.deepseek.com/news/news260821) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
- DeepSeek-V4.1-Flash: [API news, 2026-09-10](https://api-docs.deepseek.com/news/news260910) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [Tech report PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

For the architecture comparison across families (Qwen, Llama, Gemma and DeepSeek side by side), see [Modern LLM architectures](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek).
