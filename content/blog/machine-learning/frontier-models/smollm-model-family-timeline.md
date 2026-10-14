---
title: "The SmolLM Family, Release by Release: From Cosmopedia to SmolLM3"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference catalog of Hugging Face's SmolLM family from February 2024 to October 2026: the open datasets that define it, SmolLM, SmolLM2, the SmolVLM, SmolDocling and SmolVLA side lines, and SmolLM3, with dates, specs, variant tables and primary sources."
tags: ["smollm", "smollm3", "smolvlm", "small-language-models", "hugging-face", "open-weights", "synthetic-data", "model-family", "timeline", "fineweb-edu"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 31
---

Most model families are defined by their weights. SmolLM is defined by its datasets. When Hugging Face's pretraining team (the `HuggingFaceTB` org) released SmolLM in July 2024, the three checkpoints were almost an afterthought next to the thing that made them good: SmolLM-Corpus, a fully open pretraining mix built from a synthetic textbook set (Cosmopedia), an educational-quality web filter (FineWeb-Edu) and a filtered code set (Python-Edu). Every later generation followed the same pattern. A new dataset ships, then a model trained on it ships, then the recipe, the intermediate checkpoints and the ablation configs ship too. If you have ever wanted to know exactly what went into a small model, this is the one family where you can look.

That openness has a cost for anyone trying to keep track. The family now spans three text generations, four vision-language releases, a document-conversion model co-built with IBM, a robotics policy, a training playbook and more than a dozen datasets, spread across four Hub organizations (`HuggingFaceTB`, `HuggingFaceM4`, `HuggingFaceFW`, `lerobot`) plus IBM's `ds4sd` and `docling-project`. This post is the map. It walks every release in chronological order, records what changed relative to the previous generation, gives a variant table for each, and links a primary source for every claim. The timeline below is the mental model: data first, then models, and a quieter year since SmolLM3.

![SmolLM family release timeline from Q1 2024 to Q4 2025, data releases in the top lane and model releases in the bottom lane, SmolLM3 highlighted as the latest model](/imgs/blogs/smollm-model-family-timeline-1.webp)

The figure groups releases by quarter so close-together cards do not collide; read the dates printed on the cards. One fact up front, because it is the question people ask most: **as of October 14, 2026, SmolLM3-3B (July 8, 2025) is the newest SmolLM model.** There is no SmolLM4, no SmolLM3.5 and no SmolVLM3. I searched the Hub API for every model, dataset and collection under the four orgs sorted by creation date, the `huggingface/smollm` GitHub repo, and the HF blog; the details are in the section on what came after SmolLM3. Anything that claims otherwise is extrapolating a version number.

A note on dates. Hugging Face often creates a Hub repo days or weeks before announcing it. Where a blog post exists, I use its date as the release date. Where only the Hub exists, I say "Hub created" and give the repository creation date from the Hub API, which is a lower bound on the public date, not the public date itself.

## The timeline at a glance

Before the detail, the whole family in one table. "Data" rows are datasets that a later model was trained on; "model" rows are checkpoints.

| Date | Release | Kind | One-line significance |
| --- | --- | --- | --- |
| 2024-03-20 (Hub 2024-02-18) | Cosmopedia v1 + cosmo-1b | Data + model | 25B tokens of synthetic textbooks from Mixtral-8x7B-Instruct |
| 2024-05-28 (Hub) / paper 2024-06-25 | FineWeb-Edu | Data | 1.3T educational tokens filtered out of the 15T-token FineWeb |
| 2024-07-16 | SmolLM (135M, 360M, 1.7B) + SmolLM-Corpus | Model + data | First family release; Cosmopedia v2 + FineWeb-Edu dedup + Python-Edu |
| mid-Aug 2024 (unverified) | SmolLM-Instruct v0.2 | Model | Better SFT mix; 360M v0.2 wins 63.3% vs v0.1 on AlpacaEval |
| 2024-10-31 (Hub) | SmolLM2 (135M, 360M, 1.7B) | Model | 2T / 4T / 11T tokens; multi-stage data curriculum; 8k context |
| 2024-11-17 / 2024-11-25 (Hub) | SmolTalk, FineMath | Data | The SFT set and the math pretraining set behind SmolLM2 |
| 2024-11-26 | SmolVLM (2B) | Model | Idefics3 recipe on a SmolLM2-1.7B backbone |
| 2025-01-23 | SmolVLM-256M and 500M | Model | Smallest VLMs in the line; 256M claimed to beat Idefics-80B |
| 2025-02-04 | SmolLM2 paper (arXiv 2502.02737) | Paper | Full data-centric recipe, plus Stack-Edu |
| 2025-02-20 | SmolVLM2 (2.2B, 500M, 256M) | Model | Video understanding |
| 2025-03-14 | SmolDocling-256M (with IBM) | Model | End-to-end document conversion to DocTags |
| 2025-04-07 | SmolVLM paper (arXiv 2504.05299) | Paper | Design study for small VLMs |
| 2025-05-21 | nanoVLM | Toolkit | nanoGPT-style VLM training repo |
| 2025-06-03 | SmolVLA (~450M) | Model | Vision-language-action policy for LeRobot |
| 2025-07-08 | SmolLM3-3B + SmolTalk2 | Model + data | Dual-mode reasoning, 64k trained / 128k with YaRN, fully open recipe |
| 2025-09-17 | granite-docling-258M (IBM) | Model | IBM's stated successor to SmolDocling |
| 2025-10-30 | The Smol Training Playbook | Book | The SmolLM3 training story, end to end |

The rest of this post goes row by row.

## Before SmolLM: Cosmopedia and cosmo-1b (February to March 2024)

The family's origin is a reproduction attempt. Microsoft's Phi-1.5 had shown in 2023 that a 1.3B model trained mostly on synthetic "textbook" text could punch far above its size, but the data was never released. Cosmopedia was Hugging Face's open answer. The [Cosmopedia blog post](https://huggingface.co/blog/cosmopedia) is dated March 20, 2024; the dataset repo was created on February 18, 2024.

What it is, per the blog and the [dataset card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia):

- **Size.** "Over 30 million files and 25 billion tokens" (the card lists 31.1M samples). Apache-2.0.
- **Generator.** Mixtral-8x7B-Instruct-v0.1, served through TGI on H100s, at more than 10k GPU-hours.
- **Seeds.** Curated educational sources (Stanford course outlines, Khan Academy, OpenStax, WikiHow) plus web seeds: RefinedWeb was clustered into 145 topics, of which 112 were kept, so that generated text covered the topics the web actually covers. AutoMathText seeded math; UltraChat and OpenHermes-2.5 seeded stories.

The key move was not "generate textbooks", which Phi had already done. It was **prompt diversity engineering**: the same seed is rewritten for different audiences (young children, high school students, college students, researchers) and different formats (textbook, blog post, story, WikiHow article), because the failure mode of synthetic data at scale is that a single prompt template produces 30 million near-duplicates. The [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) post goes deeper on why duplication, not quality, is the binding constraint for this kind of corpus.

Alongside the data came [cosmo-1b](https://huggingface.co/HuggingFaceTB/cosmo-1b), a proof that the corpus was useful:

| Attribute | cosmo-1b |
| --- | --- |
| Architecture | Llama-2 style, 2k sequence length, Mistral-7B tokenizer |
| Parameters | Sources disagree: the blog says about 1B, the model card says 1.8B, Hub metadata says 2B |
| Training data | ~30B-token mix (~25B Cosmopedia, ~5B AutoMathText, plus Stack Jupyter notebooks and UltraChat) |
| Tokens seen | 6 epochs, ~180B tokens |
| Compute | 160 H100s for ~15 hours |
| License | Apache-2.0 |
| Claim | Beats TinyLlama-1.1B on ARC-easy, ARC-challenge, OpenBookQA and MMLU; comparable to Qwen1.5-1B on ARC-c and OBQA; trails Phi-1.5 |

cosmo-1b is not formally a SmolLM, and it trailed Phi-1.5, so it never became a popular checkpoint. Its importance is that it validated the approach cheaply: 15 hours on 160 GPUs told the team that synthetic textbooks were worth scaling.

## FineWeb and FineWeb-Edu (April to June 2024)

The second ingredient arrived in spring 2024. [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (Hub created April 18, 2024) is 15T tokens of deduplicated, filtered English web text from 96 CommonCrawl snapshots. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (Hub created May 28, 2024) is the subset that matters for SmolLM: 1.3T tokens that a small classifier rated as educational. Both are ODC-By. The paper, ["The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"](https://arxiv.org/abs/2406.17557), went up on arXiv on June 25, 2024.

The FineWeb-Edu trick is a distillation of judgment. Llama-3-70B-Instruct annotated a few hundred thousand web pages on a 0 to 5 educational-value scale, a small regression head on a frozen embedding model was trained to imitate those scores, and the cheap classifier then scored all 15T tokens. Keeping pages with score 3 or higher gave 1.3T tokens, and the paper reports "dramatically better performance" on knowledge and reasoning benchmarks like MMLU and ARC than unfiltered FineWeb at the same token budget. The method, its calibration and its biases are covered in [classifier and perplexity-based quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering) and, at the scaling-law level, in [data quality is a scaling axis](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).

The reason this belongs in a model-family post: every SmolLM generation since has used FineWeb-Edu or its classifier, and the same "LLM labels a sample, a small classifier labels the corpus" pattern reappears in Python-Edu, Stack-Edu and FineMath. It is the family's signature technique.

| Dataset | Hub created | Size | License | Built by |
| --- | --- | --- | --- | --- |
| FineWeb | 2024-04-18 | 15T tokens, 96 CC snapshots | ODC-By | Heuristic filters + MinHash dedup per snapshot |
| FineWeb-Edu | 2024-05-28 | 1.3T tokens | ODC-By | Llama-3-70B annotations distilled into a classifier, threshold 3 |

## SmolLM (July 16, 2024)

The first family release came with the blog post ["SmolLM: blazingly fast and remarkably powerful"](https://huggingface.co/blog/smollm), dated July 16, 2024. Three sizes, all base plus instruct, all Apache-2.0 per the [model cards](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B).

### What was new

**The corpus was the product.** [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (ODC-By) combined three parts:

| Component | Tokens | What it is |
| --- | --- | --- |
| Cosmopedia v2 | 28B | Synthetic textbooks, stories and code textbooks generated by Mixtral-8x7B-Instruct |
| FineWeb-Edu (deduplicated) | 220B | The educational web subset, deduplicated |
| Python-Edu | 4B | Python files from The Stack filtered with an educational-quality classifier |

Cosmopedia v2 changed how topics were chosen. Instead of clustering the web, it started from a fixed list of 34k BISAC book-subject topics and retrieved matching web pages from two recent FineWeb snapshots (CC-MAIN-2024-10 and 2023-50) to use as seeds. The audience mix was about 40% middle school, 30% college and 30% mixed, and the set includes 1B tokens of code textbooks. The corpus card counts "over 39 million" documents.

**Architecture depended on size.** The 135M and 360M followed MobileLLM's argument that, at a tiny parameter budget, depth beats width and grouped-query attention is close to free. The 1.7B used a conventional Llama-style layout with full multi-head attention. All three use RoPE with θ = 10k, a 49,152-token vocabulary trained on the corpus itself, tied input and output embeddings, and a 2048-token context. Why tied embeddings and deep-and-thin shapes matter so much under 1B is the whole subject of [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design): at 135M, an untied 49k by 576 output matrix alone would be about 28M parameters, a fifth of the model.

| Model | Layers / hidden | Heads / KV heads | Context | Training tokens | Tokens per param |
| --- | --- | --- | --- | --- | --- |
| SmolLM-135M | 30 / 576 | 9 / 3 (GQA) | 2048 | 600B | ~4,400 |
| SmolLM-360M | 32 / 960 | 15 / 5 (GQA) | 2048 | 600B | ~1,700 |
| SmolLM-1.7B | 24 / 2048 | 32 / 32 (MHA) | 2048 | 1T | ~590 |

Variants per size: Base and Instruct. Modality: text only. Open weights.

**Post-training was light.** Instruct v0.1 did one epoch of SFT on the permissive subset of WebInstructSub plus StarCoder2-Self-OSS-Instruct, then one epoch of DPO (HelpSteer for the 135M and 1.7B, argilla/dpo-mix-7k for the 360M).

**Headline claims (from the blog).** The 135M beats MobileLLM-125M despite 600B training tokens versus MobileLLM's 1T. The 360M beats every model under 500M parameters. The 1.7B beats every model under 2B the team compared against, including Phi-1.5, MobileLLM-1.5B and Qwen2-1.5B, and reaches 24 pass@1 on HumanEval.

### Instruct v0.2 (likely mid-August 2024)

The instruct models were refreshed with a new SFT mix: a small synthetic everyday-conversations set generated with Llama-3.1, Magpie-Pro-300K-Filtered, StarCoder2-Self-OSS-Instruct and a slice of OpenHermes-2.5, trained with the alignment-handbook Zephyr-Gemma recipe for one epoch at learning rate 1e-3. The [1.7B-Instruct card](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct) reports that 360M-Instruct v0.2 wins 63.3% of AlpacaEval comparisons against v0.1. The everyday-conversations set was added because v0.1's SFT data contained almost no casual chat. **The exact release date of v0.2 is not stated in a primary source**; the earliest v0.2 GGUF on the Hub was created August 13, 2024, so mid-August is the best estimate.

## SmolLM2 (October 31, 2024; paper February 4, 2025)

SmolLM2 kept the same three sizes and roughly the same architecture and changed nearly everything about the data. The checkpoints were created on the Hub on October 30 and 31, 2024 (the [1.7B card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) is the canonical entry point). I found no dedicated launch blog post; secondary coverage dates the announcement to November 1, 2024, which I could not confirm from a primary source. The full recipe followed three months later in ["SmolLM2: When Smol Goes Big: Data-Centric Training of a Small Language Model"](https://arxiv.org/abs/2502.02737), arXiv 2502.02737, February 4, 2025.

### What changed versus SmolLM

**An order of magnitude more data.** The 1.7B went from 1T to about 11T tokens, the 360M from 600B to 4T and the 135M from 600B to 2T. This is deliberate over-training: the paper prices the 1.7B run at about $250k of compute (roughly 1e23 FLOPs on 256 H100s with nanotron), which is far past the compute-optimal point, because a small model is trained once and then served billions of times. The argument for spending training compute this way is laid out in [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws).

**A four-stage curriculum instead of one fixed mix (1.7B only).** The paper's main contribution is a manual, measurement-driven data schedule under a warmup-stable-decay learning-rate schedule:

| Stage | Tokens | Web | Code | Math | Other |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 to 6T | 90% (FineWeb-Edu 60 / DCLM 40) | 10% StarCoderData | 0 | |
| 2 | 6T to 8T | 75% | 20% | 5% OpenWebMath | |
| 3 | 8T to 10T | ~70% (web ratio flips to FineWeb-Edu 40 / DCLM 60) | Stack-Edu replaces StarCoderData | ~10% (OWM + InfiMM-WebMath) | |
| 4 (decay) | 10T to 11T | 58% | 24% Stack-Edu | 14% (FineMath 4+, InfiWebMath 3+) | 4% Cosmopedia v2 |

The pattern is the one [data mixing and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) describes: broad web first, then upweight the small, high-value domains (math, curated code) late, especially in the decay phase where the model is most sensitive to what it sees. The team rebalanced between stages by watching benchmarks rather than fitting a mixture law. The 360M and 135M used a single stage instead: DCLM filtered with the FineWeb-Edu classifier, plus Stack-Edu, InfiMM-WebMath, FineMath and Cosmopedia.

**Three new datasets, all released.**

| Dataset | Hub created | Size | Role |
| --- | --- | --- | --- |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 2024-11-25 | FineMath-3+: 34B tokens (21.4M docs); FineMath-4+: 9.6B tokens | Math pretraining, classifier-filtered web math that keeps step-by-step reasoning |
| Stack-Edu | 2025-03-18 (classifiers 2024-09-23) | ~125B tokens, 15 programming languages | Educational-quality code, FineWeb-Edu method applied to The Stack v2 |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 2024-11-17 | ~1M samples | SFT mix: Smol-Magpie-Ultra (400k from Llama-3.1-405B), Smol-constraints (36k), Smol-rewrite (50k), Smol-summarize (100k) plus public sets |

The team built FineMath because the existing open math web sets (OpenWebMath, InfiMM-WebMath) were small and skewed toward academic text; FineMath's classifier specifically favors pages with step-by-step worked solutions, the kind of reasoning that grade-school and competition math benchmarks reward.

**Longer context.** The 135M and 360M train at 8192 tokens. The 1.7B pretrains at 2k and is extended to 8k with about 75B extra tokens and RoPE θ raised to 130k (the paper's number). A later [SmolLM2-1.7B-Instruct-16k](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-16k) (Hub created February 21, 2025) fine-tuned on 15k long samples to reach 16k context; its card describes the RoPE base going from 100k to 500k, which does not match the paper's 130k, so treat the exact base as source-dependent.

**Post-training.** SFT on SmolTalk for two epochs at sequence length 8192 (the 135M and 360M use a reduced "smol-smoltalk" subset that drops content too hard for them), then DPO on UltraFeedback for two epochs with β = 0.5.

### Variants

| Model | Layers / hidden | Heads / KV heads | Context | Training tokens | Tokens per param |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M (Base, Instruct) | 30 / 576 | 9 / 3 (GQA) | 8192 | 2T | ~14,800 |
| SmolLM2-360M (Base, Instruct) | 32 / 960 | 15 / 5 (GQA) | 8192 | 4T | ~11,100 |
| SmolLM2-1.7B (Base, Instruct) | 24 / 2048, FFN 8192 | 32 / 32 (MHA) | 8192 (pretrain 2k) | ~11T | ~6,500 |
| SmolLM2-1.7B-Instruct-16k | same | same | 16k | + long-context SFT | |

Also published: SFT-only checkpoints, GGUF and MLX builds, and intermediate pretraining checkpoints for all three sizes (Hub, February 26, 2025). All Apache-2.0, text only, open weights. Vocabulary stayed at 49,152 with tied embeddings.

**Headline numbers (1.7B, from the card).** Base: HellaSwag 68.7, ARC 60.5, MMLU-Pro 19.4, GSM8K 31.0. Instruct: IFEval 56.7, MT-Bench 6.13, GSM8K 48.2. The card's comparison table also shows where SmolLM2 did not win: Qwen2.5-1.5B base scores 61.3 on GSM8K against SmolLM2's 31.0. Math remained the gap, which is the context for the reasoning focus of SmolLM3.

## SmolVLM (November 26, 2024)

The first side line put eyes on SmolLM2. The [SmolVLM blog post](https://huggingface.co/blog/smolvlm) is dated November 26, 2024, and was released by the `HuggingFaceM4` multimodal team.

**What it is.** SmolVLM reuses the Idefics3 recipe and swaps the language model: Idefics3 used Llama-3.1-8B, SmolVLM uses SmolLM2-1.7B. The vision encoder is SigLIP so400m. The efficiency lever is aggressive token compression: a pixel-shuffle step folds 3 by 3 neighborhoods of visual features into one token, a 9x reduction, so a 384 by 384 image patch costs 81 tokens. That matters doubly for a 1.7B backbone with a short native context, because every visual token displaces a text token.

| Variant | Notes |
| --- | --- |
| SmolVLM-Base | Pretrained on The Cauldron and Docmatix |
| SmolVLM-Synthetic | Base fine-tuned on synthetic data |
| SmolVLM-Instruct | The chat model |
| SmolVLM-Instruct-DPO | A LoRA adapter on top of Instruct |

All Apache-2.0, open weights, image and text input, text output, about 2B parameters total.

**Headline numbers (blog).** MMMU 38.8, MathVista 44.6, MMStar 42.1, DocVQA 81.6, TextVQA 72.7, at 5.02 GB minimum GPU RAM. Against Qwen2-VL-2B the blog reports 3.3 to 4.5x faster prefill and 7.5 to 16x faster generation. It also showed basic video ability, 27.14 on CinePile with up to 50 sampled frames, which foreshadowed SmolVLM2.

## SmolVLM-256M and SmolVLM-500M (January 23, 2025)

Two months later the team went smaller. The ["SmolVLM Grows Smaller"](https://huggingface.co/blog/smolervlm) post (January 23, 2025) introduced 256M and 500M models, which Hugging Face described as the smallest VLMs it had released.

**What changed versus SmolVLM 2B.**

- **A smaller vision encoder.** SigLIP base patch-16/512, about 93M parameters, replacing the 400M-class so400m. At 256M total, a 400M encoder would not fit.
- **More compression.** 4096 pixels per token.
- **Tokenizer fix.** Sub-image separator strings became single special tokens, which stabilized training at this size.
- **A document-heavy data mix.** About 41% document understanding and 14% captioning, with handwritten math (MathWriting) added.
- **A retrieval spin-off.** ColSmolVLM, a ColPali-style document retriever built on the small backbones, was announced alongside.

The language backbones are not named in that blog post. The SmolDocling model card's model tree lists SmolLM2-135M under SmolVLM-256M, which is consistent with the sizes, but I did not find a primary statement for the 500M's backbone. **Headline claim:** the 256M beats Idefics-80B, a model about 300x larger, on the team's evaluation suite.

| Variant | Params | Vision encoder | Hub created (Base / Instruct) | License |
| --- | --- | --- | --- | --- |
| SmolVLM-256M (Base, Instruct) | 256M | SigLIP base-patch16-512 (~93M) | 2025-01-10 / 2025-01-17 | Apache-2.0 |
| SmolVLM-500M (Base, Instruct) | 500M | SigLIP base-patch16-512 (~93M) | 2025-01-13 / 2025-01-20 | Apache-2.0 |

## SmolVLM2: video (February 20, 2025)

[SmolVLM2](https://huggingface.co/blog/smolvlm2) (blog February 20, 2025) extended the line to video at three sizes. The blog calls the 500M and 256M video models "the smallest video language models ever released" and claims the 2.2B "outperforms all existing 2B models" on Video-MME. The launch was framed around on-device apps: an iPhone app waitlist, a VLC media-player prototype that navigates video by semantic search, and a highlight-generator Space.

| Variant | Hub created | Modality | License |
| --- | --- | --- | --- |
| SmolVLM2-2.2B-Instruct | 2025-02-08 | Image, multi-image, video + text in; text out | Apache-2.0 |
| SmolVLM2-2.2B-Base | 2025-04-14 | same | Apache-2.0 |
| SmolVLM2-500M-Video-Instruct | 2025-02-11 | same | Apache-2.0 |
| SmolVLM2-256M-Video-Instruct | 2025-02-11 | same | Apache-2.0 |

The design study behind the whole VLM line, ["SmolVLM: Redefining small and efficient multimodal models"](https://arxiv.org/abs/2504.05299), followed on April 7, 2025. Its headline numbers: the 256M runs in under 1 GB of GPU memory and outperforms Idefics-80B, and the 2.2B rivals VLMs that need twice the memory. Its practical value is the list of small-scale design rules, for example that small models prefer a smaller vision encoder and more aggressive token compression than large ones, and that reusing text SFT data hurts small multimodal models more than it helps.

## SmolDocling (March 14, 2025, with IBM)

[SmolDocling](https://arxiv.org/abs/2503.11576) is the first family member built with an outside partner: IBM Research's Docling team, which maintains the open Docling document-conversion library. The paper, "SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion", went up on March 14, 2025.

**What it is.** A 256M model, based on SmolVLM-256M-Instruct per the [model card](https://huggingface.co/ds4sd/SmolDocling-256M-preview), that converts a page image straight into **DocTags**, a compact markup that records every element's type (text, table, code, formula, chart, picture caption), its bounding box and its content in reading order. Traditional pipelines chain a layout detector, an OCR engine, a table-structure model and a formula recognizer; SmolDocling replaces the chain with one forward pass. The paper claims it competes with VLMs up to 27x larger and contributes new datasets for charts, tables, equations and code.

| Attribute | SmolDocling-256M-preview |
| --- | --- |
| Base | SmolVLM-256M-Instruct |
| Output | DocTags (convertible to Markdown, HTML, JSON via Docling) |
| Developers (card) | Docling Team, IBM Research, with Hugging Face |
| License | Conflicting: card metadata says CDLA-Permissive-2.0, card summary says Apache-2.0 |
| Status | "preview"; repo has since moved to the `docling-project` org |

**Successor.** On September 17, 2025 IBM released [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), which the SmolDocling card calls "the successor to SmolDocling". It keeps the recipe but swaps the parts: a SigLIP2 base-patch16-512 encoder, a pixel-shuffle projector and a 165M Granite language model, Apache-2.0. Strictly, granite-docling is a Granite model, not a Smol model; it is listed here because it is where the SmolDocling line continued.

## nanoVLM (May 21, 2025)

[nanoVLM](https://huggingface.co/blog/nanovlm) is a training toolkit rather than a product model: a pure-PyTorch, nanoGPT-style repository for training a small VLM end to end. Its example checkpoint, nanoVLM-222M, pairs a SigLIP-base-patch16-224 encoder with SmolLM2-135M and trains in about 6 hours on one H100 on about 1.7M samples from The Cauldron. It belongs in the family because it is SmolLM2 used as a teaching backbone, and it is the cheapest way to understand what SmolVLM does internally.

## SmolVLA (June 3, 2025)

[SmolVLA](https://huggingface.co/blog/smolvla) (blog June 3, 2025; paper [arXiv 2506.01844](https://arxiv.org/abs/2506.01844), June 2, 2025) is the robotics branch, released through the LeRobot project as `lerobot/smolvla_base` (Apache-2.0).

**What it is.** A vision-language-action policy of about 450M parameters. The perception and language half is a SmolVLM2 backbone (SigLIP plus SmolLM2); the action half is a separate "action expert" of about 100M parameters trained with flow matching to output continuous action chunks. Three design choices keep it small and fast:

- The action expert reads features from only the **first half of the VLM's layers**, skipping the upper layers entirely, on the observation that mid-level features are what control needs.
- Cross-attention and self-attention layers alternate inside the action expert.
- Each camera frame is compressed to 64 visual tokens.

It also ships **asynchronous inference**: the robot keeps executing the current action chunk while the next one is being computed, which the blog reports gives about 30% faster task completion.

**Data and claims.** Pretrained on 487 community-contributed LeRobot datasets, about 10M frames in fewer than 30k episodes, which is small by VLA standards. Evaluated on LIBERO, Meta-World and real SO-100/SO-101 arms, and claimed comparable to VLAs about 10x larger, trainable on a single consumer GPU and runnable on a CPU. Through March and April 2026, task-specific fine-tunes kept appearing on the Hub (LIBERO, Meta-World, RoboTwin, RoboCasa, VLABench), but these are fine-tunes of the same base, not a SmolVLA 2. For the wider landscape of how VLAs are built and trained, see [vision-language-action models from RT-2 to OpenVLA](/blog/machine-learning/ai-agent/vision-language-action-models-rt2-to-openvla) and [training vision-language-action models](/blog/machine-learning/ai-agent/training-vision-language-action-models).

## SmolLM3 (July 8, 2025)

[SmolLM3](https://huggingface.co/blog/smollm3) (blog July 8, 2025; [model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), Apache-2.0) is the current flagship and the largest change since the family began. It is one size, 3B, and it was released as what the team called a fully open blueprint: weights, data mixtures, training configs, intermediate checkpoints and an engineering write-up.

### What changed versus SmolLM2

**Size and shape.** 3B instead of 1.7B, with grouped-query attention throughout (SmolLM2-1.7B had used full MHA):

| Attribute | SmolLM3-3B |
| --- | --- |
| Layers / hidden / FFN | 36 / 2048 / 11008 |
| Attention heads / KV heads | 16 / 4 (GQA, 4 groups) |
| Positional encoding | RoPE, removed in every 4th layer (NoPE, `no_rope_layer_interval: 4`) |
| Vocabulary | 128,256 (Llama-3.2 tokenizer) |
| Embeddings | Tied |
| Context | 64k trained (`max_position_embeddings` 65,536); 128k with YaRN extrapolation |
| Other | Intra-document attention masking; no weight decay on embeddings |

**NoPE layers.** Dropping rotary embeddings from one layer in four comes from the observation that layers without explicit position information generalize better beyond the training length, while the RoPE layers keep local order. It is a cheap hedge for long context, and the blog reports it did not hurt short-context performance in ablations. The extension machinery (θ scaling, YaRN) is explained in [long-context inference: RoPE scaling, sinks and the prefill cost curve](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve).

**Multilingual.** SmolLM2 was English. SmolLM3 targets English, French, Spanish, German, Italian and Portuguese, with smaller amounts of Arabic, Chinese and Russian in training. The switch to the 128k Llama-3.2 vocabulary is what makes that affordable.

**Pretraining.** 11.2T tokens (the card rounds to 11T) on 384 H100s for 24 days, in three stages:

| Stage | Tokens | Web | Code | Math |
| --- | --- | --- | --- | --- |
| 1 | 0 to 8T | 85% | 12% | 3% |
| 2 | 8T to 10T | 75% | 15% | 10% |
| 3 | 10T to 11.1T | 63% | 24% | 13% |

Same philosophy as SmolLM2: upweight code and math late.

**Mid-training, a new phase.** Two blocks sit between pretraining and post-training:

1. **Long context**: about 100B tokens, first 4k to 32k (RoPE θ 1.5M), then 32k to 64k (θ 5M).
2. **Reasoning**: a 35B-token set of reasoning traces (OpenThoughts3 plus NVIDIA Llama-Nemotron post-training data) for 4 epochs, about 140B tokens, to teach general reasoning before any task-specific SFT.

**Dual-mode reasoning.** SmolLM3 is a single model with two modes. Thinking is on by default; `/think` and `/no_think` in the system prompt (or `enable_thinking` in the chat template) toggle it. This mirrors the hybrid design Qwen3 introduced in April 2025 (see the [Qwen3 technical report](/blog/machine-learning/frontier-models/qwen3-technical-report) post), and Qwen3 is literally in the recipe: synthetic reasoning traces for SFT came from Qwen3-32B.

**Post-training recipe.**

| Step | Data | Detail |
| --- | --- | --- |
| SFT | 1.8B tokens (1B no-think, 0.8B think) | 4 epochs, ~8B tokens, best-fit-decreasing packing |
| APO | Tulu3 preference pairs (no-think); Qwen3-32B vs Qwen3-0.6B response pairs (think) | Anchored Preference Optimization, a DPO variant that anchors to the reference policy |
| Merge | 0.9 x APO model soup + 0.1 x mid-training checkpoint | MergeKit linear merge; recovers long-context RULER scores up to 128k that APO had eroded |

The merge step is the most interesting engineering detail. Preference optimization on short data degraded long-context ability, and rather than re-running long-context training, the team blended back a little of the mid-training checkpoint that still had it. Tool calling is supported through `xml_tools` and `python_tools` in the chat template.

**Headline numbers (blog and card).**

| Benchmark | SmolLM3 think | SmolLM3 no-think | Note |
| --- | --- | --- | --- |
| AIME 2025 | 36.7 | 9.3 | Qwen3-4B with thinking: 58.8 |
| GPQA Diamond | 41.7 | 35.7 | |
| LiveCodeBench | 30.0 | 15.2 | |
| IFEval | | 76.7 | |
| BFCL (tool calling) | | 92.3 | |

Base model: HellaSwag 76.15, GSM8K 67.63, MATH 46.10. The team's summary claim is that SmolLM3 beats Llama-3.2-3B and Qwen2.5-3B and is competitive with the 4B class (Qwen3-4B, Gemma3-4B). The table makes the honest version visible: with thinking on, it still trails Qwen3-4B substantially on competition math.

### Variants and companion releases

| Release | Hub created | What it is |
| --- | --- | --- |
| SmolLM3-3B-Base | 2025-06-19 | Base after pretraining + mid-training |
| SmolLM3-3B | 2025-07-08 | Dual-mode instruct model, also as ONNX |
| SmolLM3-3B-checkpoints | 2025-07-20 | Intermediate checkpoints |
| smollm3-configs, blueprint dataset | 2025-05-23 onward | Training configs and mixture definitions |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | 2025-07-10 | Mid-training (4.8M rows), SFT (25 datasets: 10 think, 15 no-think) and preference subsets |

All Apache-2.0, text only, open weights.

## The Smol Training Playbook (October 30, 2025)

The last big release was a book, not a model. [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingfacetb-smol-training-playbook.hf.space/) (October 30, 2025, by Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, Elie Bakouch, Ed Beeching and others) tells the SmolLM3 training story in four parts: a "training compass" on whether to train at all, pretraining, post-training and infrastructure. It is the most detailed public account of a modern small-model run, including the ablations that did not make the blog. (A widely repeated anecdote, that the run was restarted after about 1T tokens because of a tensor-parallel seeding bug, comes from secondary coverage; I did not locate it in the primary text, so treat it as unverified.)

Around it the data releases continued: [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) (Hub created July 28, 2025; paper [arXiv 2510.17269](https://arxiv.org/abs/2510.17269)), 24.3M vision-language samples from more than 200 sources, decontaminated against 66 benchmarks, and [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) (Hub created September 5, 2025, ODC-By), about 3T tokens of PDF text in 1,733 languages that the card says performs close to the SmolLM3 web mixture alone. Neither has yet been used to train a released Smol model.

## What came after SmolLM3 (through October 2026)

This is the section where it would be easy to invent things, so here is exactly what exists.

**No new SmolLM or SmolVLM generation has been released.** As of October 14, 2026:

- The newest `HuggingFaceTB` checkpoints are task fine-tunes (SmolLM3-3B-GSM8K-SFT, created April 3, 2026, and a Qwen3-1.7B GSM8K SFT), plus [nanowhale-100m](https://huggingface.co/HuggingFaceTB/nanowhale-100m) (April 24, 2026). nanowhale is not a SmolLM: its card calls it an educational miniature of the DeepSeek-V4 design, about 110M parameters with 8 layers, a small MoE (4 routed experts plus 1 shared, top-2), 2k context, fine-tuned on smol-smoltalk, and "not suitable for any production use".
- The SmolLM3 Hub collection was last updated October 9, 2025, and the `huggingface/smollm` README still tags SmolLM3 as new.
- `HuggingFaceM4` has released no smol model since SmolVLM2. SmolVLA has had fine-tunes, not a successor.
- The 2026 output of these orgs is data and tooling: FineTranslations (January 2026), FinePDFs/DCLM/FineWeb ablation mixes (February 2026), finephrase (February 2026) and a TRL distillation-trainer Space (April 2026). Hugging Face's "State of Open Models: Summer 2026" post (August 14, 2026) mentions no new smol release.

The caveat: the Hub API cannot see private or gated repositories, so work in progress is invisible by construction.

## How the family evolved

Five through-lines connect the releases. The lineage graph shows the first one, which is easy to miss when the releases are read one at a time.

![Lineage graph of the SmolLM family: datasets feed SmolLM, SmolLM2 and SmolLM3, and SmolLM2 becomes the language backbone of SmolVLM, which branches into SmolVLM2, SmolDocling, granite-docling and SmolVLA](/imgs/blogs/smollm-model-family-timeline-2.webp)

### 1. One text backbone feeds every side line

Every multimodal Smol model is SmolLM2 with something attached. SmolVLM 2B is SigLIP plus SmolLM2-1.7B. SmolVLM-256M sits on SmolLM2-135M. SmolDocling is SmolVLM-256M fine-tuned for DocTags, and SmolVLA is SmolVLM2 plus a flow-matching action expert. Even nanoVLM's teaching model uses SmolLM2-135M. This is why there is no "SmolVLM3" yet: the vision line has not been rebased on SmolLM3, whose smallest size is 3B, which is above the budget the VLM line was built for. If a successor appears, the first thing to check is which text backbone it uses.

### 2. Data is released first and is the real product

Look at the top lane of the timeline: Cosmopedia before SmolLM, FineMath and SmolTalk alongside SmolLM2, SmolTalk2 with SmolLM3. The models are, in a sense, the evaluation of the datasets. The same technique, an LLM grades a sample and a small classifier grades the corpus, runs through FineWeb-Edu, Python-Edu, Stack-Edu and FineMath. For anyone building their own pretraining mix, these datasets are more reusable than the checkpoints. The [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) post discusses the leakage risks of that pattern: a classifier trained to prefer "educational" text quietly encodes the annotator model's taste, and gains on knowledge benchmarks partly reflect that alignment.

### 3. Tokens per parameter kept climbing

![Bar chart on a log scale of training tokens for each SmolLM model, from 600B for SmolLM-135M to 11.2T for SmolLM3-3B, each paired with its far smaller Chinchilla-optimal budget](/imgs/blogs/smollm-model-family-timeline-3.webp)

Chinchilla's compute-optimal rule of thumb is about 20 training tokens per parameter (see [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)). SmolLM-1.7B used about 590. SmolLM2-135M used about 14,800, and SmolLM2-1.7B about 6,500. Every model in the family is trained 30x to 740x past the compute-optimal point. That is the correct choice for a model whose cost is dominated by inference on phones and laptops, and the SmolLM2 paper is one of the cleanest public demonstrations that small models keep improving far beyond Chinchilla: the 1.7B was still gaining at 11T tokens. The interesting inversion is SmolLM3, which spends fewer tokens per parameter (about 3,700) than SmolLM2-1.7B because it bought capability with size and a reasoning mid-training phase instead of more raw tokens.

### 4. From fixed mix to staged curriculum to mid-training

SmolLM used one mix for the whole run. SmolLM2-1.7B used four hand-tuned stages with math and curated code upweighted in the decay. SmolLM3 used three pretraining stages and then added a distinct mid-training phase for long context and reasoning. The direction is the same one the large labs took: the end of training is where the model's specialty is decided, and it gets its own data.

### 5. Post-training grew from "light SFT" to a frontier-style stack

v0.1 was one epoch of SFT and one of DPO. v0.2 fixed the conversational gap. SmolLM2 added a purpose-built million-sample SFT set. SmolLM3 has reasoning SFT from a larger teacher, APO with on-policy-style pairs, and checkpoint merging to repair what preference training broke. The gap between how a 3B open model and a frontier model are post-trained is now mostly scale, not technique. Distilling a larger reasoner into a small one, which is what SmolLM3's SFT data does with Qwen3-32B, is covered in [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning).

### Generation-over-generation summary

| | SmolLM (Jul 2024) | SmolLM2 (Oct 2024) | SmolLM3 (Jul 2025) |
| --- | --- | --- | --- |
| Sizes | 135M, 360M, 1.7B | 135M, 360M, 1.7B | 3B |
| Max training tokens | 1T | ~11T | 11.2T |
| Context | 2k | 8k (16k variant) | 64k trained, 128k YaRN |
| Attention | GQA small, MHA 1.7B | GQA small, MHA 1.7B | GQA + NoPE every 4th layer |
| Vocabulary | 49,152 | 49,152 | 128,256 |
| Languages | English | English | 6 primary + 3 minor |
| Data recipe | One fixed mix | 4-stage curriculum (1.7B) | 3 stages + mid-training |
| Post-training | SFT + DPO | SFT (SmolTalk) + DPO | SFT + APO + merging, dual mode |
| Openness | Weights + corpus | + datasets + paper + checkpoints | + configs + blueprint + playbook |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## Which variant to pick today

As of October 2026, here is the short guide. "Today" matters: these are 2024 and 2025 models, and the 3B-to-4B class has moved on, so the real competition for any of these is the current small checkpoints of Qwen, Gemma and Llama, not each other.

| Need | Pick | Why |
| --- | --- | --- |
| General small assistant, reasoning, tool calls, long docs | **SmolLM3-3B** | Only member with thinking mode, 64k to 128k context, tool calling and multilingual coverage |
| Sub-2B text model on a phone or laptop | **SmolLM2-1.7B-Instruct** | Strongest small member; 16k variant if you need longer inputs |
| Tiny text model: classification, extraction, routing, speculative-decoding draft | **SmolLM2-360M or 135M** | Trained on 4T and 2T tokens; GQA keeps KV cache tiny |
| Image or document Q&A on modest hardware | **SmolVLM-500M or SmolVLM2-2.2B** | 2.2B for quality, 500M when memory is the constraint |
| Video understanding on device | **SmolVLM2-500M or 256M Video-Instruct** | The smallest video-capable members |
| PDF or scan to Markdown/HTML | **granite-docling-258M** (or SmolDocling-256M-preview) | Granite-docling is the stated successor and has a clean Apache-2.0 license |
| Robot policy on a single GPU with LeRobot | **SmolVLA (`lerobot/smolvla_base`)** then fine-tune | Built for community-scale data and async inference |
| Learning how to pretrain or post-train | **SmolLM3 configs, checkpoints and the Playbook**; nanoVLM for VLMs | The most complete open recipe at this scale |
| Building your own pretraining mix | **FineWeb-Edu, FineMath, Stack-Edu, Cosmopedia v2** | Arguably the family's most durable contribution |

Two practical warnings. First, avoid the SmolLM v1 checkpoints for anything new: SmolLM2 is strictly better at the same sizes and the same license. Second, if you need hard competition math, the AIME table above is the honest answer: a 3B model with thinking still loses badly to the next size class.

If you want to place these models in the broader ecosystem, [A concise overview of leading Hugging Face AI organizations](/blog/machine-learning/open-source-library/best-organization-in-huggingface) explains where `HuggingFaceTB`, `HuggingFaceM4` and `HuggingFaceFW` sit, and [why data decides the model](/blog/machine-learning/training-data/why-data-decides-the-model) is the long version of through-line 2.

## References

Primary sources, in release order.

1. Cosmopedia blog, March 20, 2024: https://huggingface.co/blog/cosmopedia
2. Cosmopedia dataset: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
3. cosmo-1b model card: https://huggingface.co/HuggingFaceTB/cosmo-1b
4. FineWeb dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
5. FineWeb-Edu dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
6. Penedo et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", arXiv 2406.17557: https://arxiv.org/abs/2406.17557
7. SmolLM blog, July 16, 2024: https://huggingface.co/blog/smollm
8. SmolLM-1.7B model card: https://huggingface.co/HuggingFaceTB/SmolLM-1.7B
9. SmolLM-1.7B-Instruct (v0.2 notes): https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct
10. SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
11. SmolLM2-1.7B model card: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
12. SmolLM2-1.7B-Instruct-16k: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-16k
13. Ben Allal et al., "SmolLM2: When Smol Goes Big: Data-Centric Training of a Small Language Model", arXiv 2502.02737: https://arxiv.org/abs/2502.02737
14. FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
15. SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
16. SmolVLM blog, November 26, 2024: https://huggingface.co/blog/smolvlm
17. "SmolVLM Grows Smaller" (256M and 500M), January 23, 2025: https://huggingface.co/blog/smolervlm
18. SmolVLM2 blog, February 20, 2025: https://huggingface.co/blog/smolvlm2
19. Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models", arXiv 2504.05299: https://arxiv.org/abs/2504.05299
20. Nassar et al., "SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion", arXiv 2503.11576: https://arxiv.org/abs/2503.11576
21. SmolDocling-256M-preview model card: https://huggingface.co/ds4sd/SmolDocling-256M-preview
22. granite-docling-258M model card: https://huggingface.co/ibm-granite/granite-docling-258M
23. nanoVLM blog, May 21, 2025: https://huggingface.co/blog/nanovlm
24. SmolVLA blog, June 3, 2025: https://huggingface.co/blog/smolvla
25. Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics", arXiv 2506.01844: https://arxiv.org/abs/2506.01844
26. SmolLM3 blog, July 8, 2025: https://huggingface.co/blog/smollm3
27. SmolLM3-3B model card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
28. SmolTalk2: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
29. The Smol Training Playbook, October 30, 2025: https://huggingfacetb-smol-training-playbook.hf.space/
30. FineVision dataset: https://huggingface.co/datasets/HuggingFaceM4/FineVision
31. FineVision paper, arXiv 2510.17269: https://arxiv.org/abs/2510.17269
32. FinePDFs dataset: https://huggingface.co/datasets/HuggingFaceFW/finepdfs
33. nanowhale-100m model card: https://huggingface.co/HuggingFaceTB/nanowhale-100m
34. SmolLM GitHub repository: https://github.com/huggingface/smollm
