---
title: "The Gemma Family Timeline: Every Google Gemma Release From Gemma 1 to Gemma 4"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A verified, chronological reference to every Google Gemma release from February 2024 to October 2026: the four core generations, the two dozen specialized offshoots, what changed at each step, the variant tables, and which checkpoint to pick today."
tags:
  - gemma
  - google-deepmind
  - open-weights
  - model-family
  - sliding-window-attention
  - mixture-of-experts
  - multimodal
  - on-device-ai
  - llm-architecture
  - release-timeline
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: true
readTime: 38
image: "/imgs/blogs/gemma-model-family-timeline-1.webp"
---

> [!tldr]
> Google has shipped **four core Gemma generations** in 26 months: Gemma 1 (February 2024, 2B and 7B, 8K context), Gemma 2 (June 2024, 2B to 27B, distillation and 1:1 local/global attention), Gemma 3 (March 2025, 270M to 27B, multimodal, 128K context, 5:1 local/global), and Gemma 4 (April 2026, E2B to 31B dense plus a 26B MoE, thinking mode, 256K context, and the first **Apache 2.0** license). Around each core generation sits a ring of specialized offshoots: code, recurrent, vision-language, safety, medical, embedding, encoder-decoder, differentially private, translation, and function-calling models. Every one of them is an adaptation of a core checkpoint, so the generation it was built on sets its ceiling. If you are choosing today, start from Gemma 4 unless a specialized offshoot exists only on Gemma 3.

The hard part of working with Gemma is not any single model. It is the naming. There are at least twenty-five distinct things Google has called "Gemma something", released on a cadence of roughly one per month, and the names do not tell you which base they sit on. CodeGemma is a 2024 model built on Gemma 1. EmbeddingGemma is a 2025 model built on Gemma 3. EmbeddingGemma 2, released eight days ago as I write this, is built on Gemma 4. PaliGemma 2 has nothing to do with Gemma 2 except that its decoder happens to be Gemma 2. When an engineer on your team says "we tried Gemma and it was weak at long context", the first question is always *which* Gemma, and the honest answer usually turns out to be a 2024 checkpoint with an 8K window.

This post is the reference I wanted when sorting that out: every release in order, from the first one on 21 February 2024 to the latest that actually exists on 14 October 2026, with the date, what changed versus what came before, a variant table, and a primary-source link for each. Dates come from Google's own [Gemma release log](https://ai.google.dev/gemma/docs/releases), the technical reports on arXiv, and the official model cards on Hugging Face. Where two primary sources disagree (it happens more than you would think), I say so instead of picking one silently.

![Gemma release timeline from February 2024 to October 2026, with four core generations on a center lane and specialized offshoots colored by the generation they were built on](/imgs/blogs/gemma-model-family-timeline-1.webp)

The diagram above is the mental model: four core generations on the center lane, separated by gaps of roughly 4, 8 and 13 months, and every offshoot colored by the base it was built on. Three offshoots grew on Gemma 1, five on Gemma 2, nine on Gemma 3, and three so far on Gemma 4. The gaps between core releases are getting longer while the offshoot ring around each one gets denser. That pattern is the single most useful thing to know about the family, and we will come back to it in the synthesis.

If you want the architecture comparison against other families rather than the release history, the companion piece is [Modern LLM Architectures: Inside Qwen, Llama, Gemma, and DeepSeek](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek). For a measured comparison at equal hardware, see [Qwen3 versus Gemma 3 versus Llama 3.1 at equal hardware](/blog/machine-learning/inference-engineering/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware).

## The timeline at a glance

Before the generation-by-generation detail, here is the whole release log in one table. "Base" is the core generation the release was derived from. Dates are the release dates in Google's [Gemma release log](https://ai.google.dev/gemma/docs/releases) unless noted.

| Date | Release | Base | What it is |
| --- | --- | --- | --- |
| 21 Feb 2024 | **Gemma 1** (2B, 7B) | core | First open Gemma decoder, 8K context |
| 5 Apr 2024 | Gemma 1.1 | Gemma 1 | Refreshed instruction-tuned checkpoints |
| 9 Apr 2024 | CodeGemma | Gemma 1 | Code continued pretraining with fill-in-the-middle |
| 9 Apr 2024 | RecurrentGemma (2B) | Griffin | Linear recurrence plus local attention, fixed-size state |
| 3 May 2024 | CodeGemma 1.1 | Gemma 1 | Improved 2B and 7B-IT |
| 14 May 2024 | PaliGemma | Gemma 1 2B | 3B vision-language model (SigLIP-So400m + Gemma 2B) |
| 11 Jun 2024 | RecurrentGemma 9B | Griffin | Larger recurrent variant |
| 27 Jun 2024 | **Gemma 2** (9B, 27B) | core | Distillation, alternating local/global attention, soft-capping |
| 31 Jul 2024 | Gemma 2 2B | Gemma 2 | Distilled small model |
| 31 Jul 2024 | ShieldGemma | Gemma 2 | Text safety classifiers (2B, 9B, 27B) |
| 31 Jul 2024 | Gemma Scope | Gemma 2 | Sparse autoencoder interpretability suite |
| 12 Sep 2024 | DataGemma | Gemma 2 | Data Commons grounding (RIG and RAG variants) |
| 3 Oct 2024 | Gemma 2 JPN | Gemma 2 2B | Japanese-tuned 2B |
| 15 Oct 2024 | Gemma-APS | Gemma | Abstractive proposition segmentation (2B, 7B) |
| 5 Dec 2024 | PaliGemma 2 | Gemma 2 | 3B, 10B, 28B VLMs |
| 19 Feb 2025 | PaliGemma 2 mix | Gemma 2 | Multi-task fine-tuned PaliGemma 2 |
| 10 Mar 2025 | **Gemma 3** (1B, 4B, 12B, 27B) | core | Multimodal, 128K context, 5:1 local/global, QK-norm |
| 10 Mar 2025 | ShieldGemma 2 | Gemma 3 4B | Image safety classifier |
| 20 May 2025 | MedGemma (4B, 27B text) | Gemma 3 | Medical text and imaging |
| 20 May 2025 | Gemma 3n preview | Gemma 3n | Mobile-first architecture preview (Google I/O) |
| 26 Jun 2025 | Gemma 3n (E2B, E4B) | Gemma 3n | Full release: MatFormer, Per-Layer Embeddings, audio |
| 9 Jul 2025 | T5Gemma | Gemma 2 | Decoder-only to encoder-decoder adaptation |
| 9 Jul 2025 | MedGemma 27B multimodal | Gemma 3 27B | Multimodal 27B medical model |
| 14 Aug 2025 | Gemma 3 270M | Gemma 3 | Tiny fine-tuning base |
| 4 Sep 2025 | EmbeddingGemma (300M) | Gemma 3 | Text embedding model |
| 13 Sep 2025 | VaultGemma (1B) | Gemma 2 recipe | Trained from scratch with differential privacy |
| 18 Dec 2025 | FunctionGemma (270M) | Gemma 3 270M | Function-calling fine-tuning base |
| 18 Dec 2025 | T5Gemma 2 | Gemma 3 | Multimodal encoder-decoder (270M to 4B per side) |
| 19 Dec 2025 | Gemma Scope 2 | Gemma 3 | Interpretability suite for Gemma 3 |
| 13 Jan 2026 | MedGemma 1.5 (4B) | Gemma 3 4B | 3D CT/MRI, histopathology, EHR |
| 15 Jan 2026 | TranslateGemma (4B, 12B, 27B) | Gemma 3 | 55-language translation |
| 2 Apr 2026 | **Gemma 4** (E2B, E4B, 26B A4B, 31B) | core | Thinking mode, MoE, 256K context, Apache 2.0 |
| 16 Apr 2026 | Gemma 4 MTP drafters | Gemma 4 | Multi-token-prediction drafters for speculative decoding |
| 3 Jun 2026 | Gemma 4 12B Unified | Gemma 4 | Encoder-free multimodal 12B |
| 6 Oct 2026 | EmbeddingGemma 2 (740M) | Gemma 4 | Multimodal embedding model |

Two small date notes. The Gemma 4 announcement on [Google's blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) is dated 2 April 2026, while the release log dates the weights to 31 March 2026; I use the public announcement date throughout. Gemma 3n appeared twice: a preview at Google I/O on 20 May 2025 and the full release on 26 June 2025. The release log only records the June date.

What is *not* on this list matters too. There is no "Gemma 3.5", no "Gemma 4.1", and no Gemma 4 model above 31B. If you see a blog or a model hub repository claiming one of those, it is a community fine-tune or a fabrication.

## Gemma 1 (February 2024): the starting point

**Released:** 21 February 2024. **Primary source:** [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295) (arXiv 2403.08295, submitted 13 March 2024).

Gemma 1 was Google's answer to Llama 2 and Mistral 7B: two dense decoder-only models, a 2B and a 7B, with pretrained and instruction-tuned checkpoints for each. The architecture is a conventional 2024 transformer with three Gemini-flavored choices that every later generation inherits.

The first is the **vocabulary**. Gemma uses a 256,128-entry SentencePiece vocabulary, roughly eight times Llama 2's 32K. For a 2B model this is an enormous fraction of the parameter budget: with a model width of 2048, the embedding table alone is about 524M parameters, a quarter of the model. The payoff is multilingual and code tokenization efficiency; the cost is a cross-entropy layer that dominates training memory, which is why the [Unsloth fused cross-entropy post](/blog/machine-learning/open-source-library/unsloth-fused-cross-entropy) uses Gemma as its worst-case example. The broader tradeoff is covered in [Designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm).

The second is **head size 256**, wider than the usual 128, at both sizes. The third is the norm and activation stack: RMSNorm on the input of every sub-layer, GeGLU feed-forward, RoPE, and tied input/output embeddings. Gemma's RMSNorm stores its scale as an offset from one, a detail that has broken many hand-written loaders; see [Layer Normalization From the Inside Out](/blog/machine-learning/deep-learning/layer-normalization-inside-out) for the full trap.

### Gemma 1 variants

| Model | Params | Layers | d_model | Heads / KV heads | Pretraining tokens | Context | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 2B (PT, IT) | 2B | 18 | 2048 | 8 / 1 (multi-query) | 3T | 8,192 | Open (Gemma Terms of Use) |
| Gemma 7B (PT, IT) | 7B | 28 | 3072 | 16 / 16 (multi-head) | 6T | 8,192 | Open (Gemma Terms of Use) |
| Gemma 1.1 2B-IT, 7B-IT | same | same | same | same | same | 8,192 | Open (Gemma Terms of Use) |

All numbers are from the technical report's architecture table. Training ran on TPUv5e: the 2B used 512 chips and the 7B 4,096 chips.

Notice the attention split: the 2B uses **multi-query attention** (one shared KV head) while the 7B uses full **multi-head attention**. That makes the 7B unusually expensive to serve for its size, because its KV cache per token is 16 times wider than a single-KV-head design. [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) works through exactly why that matters. Gemma 2 fixed it.

**Gemma 1.1** (5 April 2024) refreshed only the instruction-tuned checkpoints with a new RLHF method. The paper's later revision reports its main instruction-tuned results on 1.1, and the architecture did not change.

**License.** Gemma 1 shipped under the custom **Gemma Terms of Use** plus a Prohibited Use Policy. Weights were downloadable and commercially usable, but this was not an OSI license, and that distinction stayed in place until Gemma 4.

## The Gemma 1 offshoots (April to June 2024)

Within four months Google built three very different things around Gemma 1. They tell you early on what the "Gemmaverse" strategy was going to be: one core model, many narrow adaptations.

### CodeGemma (9 April 2024; 1.1 on 3 May 2024)

**Primary source:** [CodeGemma: Open Code Models Based on Gemma](https://arxiv.org/abs/2406.11409) (arXiv 2406.11409).

CodeGemma is continued pretraining of the Gemma 1 checkpoints on roughly 500B tokens of mostly code, with fill-in-the-middle (FIM) training at an 80% rate in both prefix-suffix-middle and suffix-prefix-middle orderings. The FIM control tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|file_separator|>`) are what make the 2B usable as an IDE completion engine.

| Model | Base | Extra training | FIM rate | Role |
| --- | --- | --- | --- | --- |
| CodeGemma 2B (v1.0) | Gemma 2B | 500B tokens | 80% | Fast code completion |
| CodeGemma 2B (v1.1) | Gemma 2B | 1T tokens | 90% | Completion, recommended over v1.0 |
| CodeGemma 7B PT | Gemma 7B | 500B tokens | 80% | Completion and generation |
| CodeGemma 7B IT (v1.0, v1.1) | Gemma 7B | 500B tokens + instruction tuning | 80% | Code chat; v1.1 uses a new RL algorithm |

Context length is not stated separately in the report; it inherits Gemma 1's 8K. CodeGemma never got a Gemma 2 or Gemma 3 successor. Google's code story after 2024 runs through the general instruction models instead.

### RecurrentGemma (9 April 2024; 9B on 11 June 2024)

**Primary source:** [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://arxiv.org/abs/2404.07839) (arXiv 2404.07839).

RecurrentGemma is the one Gemma that is *not* a transformer. It uses the Griffin architecture: gated linear recurrences interleaved with local attention, which gives it a **fixed-size state** instead of a KV cache that grows with the sequence. For long generations on memory-constrained hardware that is a real advantage: memory stays flat no matter how many tokens you emit. The report claims quality comparable to similarly sized Gemma models despite fewer training tokens.

| Model | Architecture | Variants | Weights |
| --- | --- | --- | --- |
| RecurrentGemma 2B | Griffin (linear recurrence + local attention) | PT, IT | Open (Gemma Terms of Use) |
| RecurrentGemma 9B | Griffin | PT, IT | Open (Gemma Terms of Use) |

It also never got a successor. The idea it tested (bounded per-token memory) came back in a different form: Gemma 2 and 3 got most of the same benefit by making most of their layers sliding-window attention, which bounds those layers' KV cache while keeping a transformer.

### PaliGemma (14 May 2024)

**Primary source:** [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726) (arXiv 2407.07726).

PaliGemma combines the SigLIP-So400m vision encoder with the Gemma 2B decoder into a 3B vision-language model designed to be fine-tuned, not chatted with. Google shipped pretrained checkpoints at three input resolutions plus dozens of task-specific fine-tunes (VQA, captioning, OCR-VQA, referring segmentation, document VQA) to show transfer.

| Model | Vision encoder | Decoder | Resolutions | Variants |
| --- | --- | --- | --- | --- |
| PaliGemma 3B | SigLIP-So400m | Gemma 2B | 224, 448, 896 | PT, mix, many task FT |

The checkpoint names on Hugging Face (for example `paligemma-3b-pt-896`, `paligemma-3b-ft-docvqa-896`) carry the resolution, which is the knob that matters most: 896 is what you want for documents and small text, at a large token cost per image.

## Gemma 2 (June 2024): distillation and the first KV budget cut

**Released:** 27 June 2024 (9B, 27B); 2B on 31 July 2024. **Primary source:** [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) (arXiv 2408.00118).

Gemma 2 is where the family found its identity. Three changes do most of the work.

**Knowledge distillation for the small models.** The 2B and 9B were not trained with plain next-token prediction. They were trained to match a larger teacher's output distribution, which gives a much richer signal per token than a one-hot label. The 27B was trained from scratch on 13T tokens; the 9B on 8T and the 2B on 2T, both distilled. The report's claim is that the small models compete with models two to three times their size. This one decision is why "Gemma 2 9B" became the default small open model for much of late 2024. [Knowledge Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) uses Gemma 2 as its pretraining-scale distillation case study.

**Alternating local and global attention.** Every other layer is sliding-window attention with a 4,096-token window; the rest attend to the full 8,192 context. Half the layers therefore have a bounded KV cache. At an 8K context that saves little, but it established the pattern Gemma 3 would push much further.

**GQA everywhere, plus stability tricks.** All three sizes use grouped-query attention with two query heads per KV head, fixing Gemma 1 7B's expensive multi-head cache. Gemma 2 adds RMSNorm both before *and* after each sub-layer, and **logit soft-capping** (cap 50.0 on attention logits, 30.0 on final logits), which squashes logits through a scaled tanh to keep training stable. Soft-capping turned out to be awkward for fused attention kernels, and Gemma 3 dropped it.

### Gemma 2 variants

| Model | Layers | d_model | Heads / KV heads | Head size | Training tokens | Distilled | Context | Window | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 2 2B (PT, IT) | 26 | 2304 | 8 / 4 | 256 | 2T | Yes | 8,192 | 4,096 | Open (Gemma Terms) |
| Gemma 2 9B (PT, IT) | 42 | 3584 | 16 / 8 | 256 | 8T | Yes | 8,192 | 4,096 | Open (Gemma Terms) |
| Gemma 2 27B (PT, IT) | 46 | 4608 | 32 / 16 | 128 | 13T | No | 8,192 | 4,096 | Open (Gemma Terms) |

Post-training combined supervised fine-tuning on teacher-generated responses, on-policy distillation from the teacher, RLHF with a reward model an order of magnitude larger than the policy, and **model merging** (averaging runs with different hyperparameters). The vocabulary stayed at 256,128.

![Matrix comparing Gemma 1 through Gemma 4 on attention layout, local window, context length, vocabulary and license](/imgs/blogs/gemma-model-family-timeline-2.webp)

## How the attention stack evolved

The matrix above is the clearest way to see the family's architectural through-line. Read down any column and the values move in one direction; read down the license column and it changes exactly once.

| Generation | Attention layout | Local window | Context | Notable trick |
| --- | --- | --- | --- | --- |
| Gemma 1 | All global (2B MQA, 7B MHA) | none | 8K | Head size 256 |
| Gemma 2 | 1 local : 1 global, GQA | 4,096 | 8K | Logit soft-capping, pre+post norm |
| Gemma 3 | 5 local : 1 global, GQA | 1,024 | 128K (1B: 32K) | QK-norm replaces soft-capping; RoPE 1M global, 10K local |
| Gemma 4 | 5:1 (E2B 4:1), last layer always global | 512 (edge), 1,024 (others) | 128K edge, 256K others | Unified K=V and p-RoPE on global layers; KV sharing on edge models |

Here is why this sequence is the right one. KV cache memory per token is proportional to the number of layers that keep a full-length cache. In Gemma 1 that is every layer. In Gemma 2 it is half. In Gemma 3 it is one in six, and the other five keep only 1,024 tokens each. At a 128K context the local layers are almost free, so the long-context bill is the global layers alone. Gemma 4 then attacked the global layers directly: it reuses the keys as values (one tensor instead of two) and applies rotary embedding to only a quarter of the dimensions (p-RoPE with p = 0.25). The [Gemma 4 technical report](https://arxiv.org/abs/2607.02770) puts the global KV cache reduction from those two changes at up to 37.5%.

The serving consequence is that a Gemma 3 or Gemma 4 deployment has two kinds of cache with different lifetimes, and a server that does not know that will either waste memory or reject requests it could have served. [The Hybrid KV Cache Manager](/blog/machine-learning/model-serving/hybrid-kv-cache-manager) covers how vLLM handles it.

## The Gemma 2 offshoots (July 2024 to September 2025)

Gemma 2 had the longest productive life of any core generation as a base: offshoots were still appearing fifteen months after it shipped.

### ShieldGemma (31 July 2024)

**Primary source:** [ShieldGemma: Generative AI Content Moderation Based on Gemma](https://arxiv.org/abs/2407.21772) (arXiv 2407.21772).

ShieldGemma is a set of safety classifiers built on Gemma 2 that score both user prompts and model responses against four harm types: sexually explicit content, dangerous content, harassment, and hate speech. You give it a policy description and the text; it outputs the probability of a "Yes" token.

| Model | Base | Task | Weights |
| --- | --- | --- | --- |
| ShieldGemma 2B | Gemma 2 2B | Text prompt/response moderation | Open (Gemma Terms) |
| ShieldGemma 9B | Gemma 2 9B | Same | Open (Gemma Terms) |
| ShieldGemma 27B | Gemma 2 27B | Same | Open (Gemma Terms) |

For how this design compares to streaming moderators, see [Guard Models for LLMs: How Qwen3Guard Moderates a Token Stream](/blog/machine-learning/frontier-models/guard-models-qwen3guard-streaming-moderation), which places ShieldGemma in the "generative classifier" lineage alongside Llama Guard.

### Gemma Scope (31 July 2024)

Not a model you deploy but worth knowing: a suite of sparse autoencoders trained on Gemma 2's activations for interpretability research. Its successor, **Gemma Scope 2** (19 December 2025), covers Gemma 3. Both are listed in the [release log](https://ai.google.dev/gemma/docs/releases).

### DataGemma (12 September 2024)

**Primary source:** [Google blog: DataGemma](https://blog.google/technology/ai/google-datagemma-ai-llm/).

DataGemma grounds statistical answers in Google's Data Commons knowledge graph (over 240 billion data points from sources like the UN, WHO and census bureaus). It ships in two styles. **RIG** (Retrieval-Interleaved Generation) teaches the model to emit Data Commons queries inline while generating, so a number in the answer is fetched rather than recalled. **RAG** retrieves relevant tables before generation.

| Model | Base | Approach | Weights |
| --- | --- | --- | --- |
| `datagemma-rig-27b-it` | Gemma 2 27B | Retrieval-Interleaved Generation | Open (Gemma Terms) |
| `datagemma-rag-27b-it` | Gemma 2 27B | Retrieval-Augmented Generation | Open (Gemma Terms) |

One unresolved discrepancy: the release log entry reads "DataGemma in 2B size", but the only checkpoints on Google's Hugging Face organization are the two 27B models above. I list what is actually downloadable.

### Gemma 2 JPN and Gemma-APS (October 2024)

Two small releases from the [release log](https://ai.google.dev/gemma/docs/releases): **Gemma 2 JPN** (3 October 2024), a Japanese-tuned Gemma 2 2B, and **Gemma-APS** (15 October 2024) in 2B and 7B, a model for abstractive proposition segmentation (splitting text into atomic factual claims, useful as a preprocessing step for fact-checking and retrieval).

### PaliGemma 2 (5 December 2024; mix on 19 February 2025)

**Primary source:** [PaliGemma 2: A Family of Versatile VLMs for Transfer](https://arxiv.org/abs/2412.03555) (arXiv 2412.03555).

PaliGemma 2 keeps the SigLIP-So400m encoder and swaps the decoder for Gemma 2 at three sizes, which is where the odd parameter counts come from (the 400M encoder plus each decoder).

| Model | Vision encoder | Decoder | Resolutions | Variants | Weights |
| --- | --- | --- | --- | --- | --- |
| PaliGemma 2 3B | SigLIP-So400m/14 | Gemma 2 2B | 224, 448, 896 | PT, mix | Open (Gemma Terms) |
| PaliGemma 2 10B | SigLIP-So400m/14 | Gemma 2 9B | 224, 448, 896 | PT, mix | Open (Gemma Terms) |
| PaliGemma 2 28B | SigLIP-So400m/14 | Gemma 2 27B | 224, 448, 896 | PT, mix | Open (Gemma Terms) |

The "mix" checkpoints (February 2025) are fine-tuned on a mixture of tasks so they work out of the box for captioning, OCR, detection and segmentation prompts. Three months later Gemma 3 made every size above 1B natively multimodal, and the case for a separate PaliGemma line mostly disappeared for general use. It remains the better starting point when you want detection or segmentation outputs, since those were trained in.

### T5Gemma (9 July 2025)

**Primary source:** [Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](https://arxiv.org/abs/2504.06225) (arXiv 2504.06225).

T5Gemma takes pretrained decoder-only Gemma 2 weights and adapts them into encoder-decoder models, initializing both stacks from the decoder and continuing pretraining with either a UL2 or a PrefixLM objective. The argument is that encoder-decoders give a better quality/latency tradeoff for input-heavy tasks (summarization, translation, classification), and adaptation is far cheaper than pretraining from scratch.

| Configurations (encoder-decoder) | Objectives | Variants | Weights |
| --- | --- | --- | --- |
| 2B-2B, 9B-9B, 9B-2B (Gemma 2 based) | UL2, PrefixLM | PT, IT | Open (Gemma Terms) |
| S-S, B-B, L-L, ML-ML, XL-XL (T5-sized, newly pretrained) | UL2, PrefixLM | PT, IT | Open (Gemma Terms) |

The **9B-2B** "unbalanced" pairing is the interesting one: a large encoder to understand the input and a small decoder to write a short output, so decode cost stays at 2B.

### VaultGemma (13 September 2025)

**Primary source:** [VaultGemma: A Differentially Private Gemma Model](https://arxiv.org/abs/2510.15001) (arXiv 2510.15001) and [Google Research blog](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm).

VaultGemma is a 1B model trained *entirely* with differential privacy (DP-SGD in pretraining), using a Gemma 2 style architecture and the Gemma 2 data mixture. The privacy guarantee on the model card is epsilon at most 2.0 and delta at most 1.1e-10, and Google's memorization tests found no detectable exact or approximate memorization. The price is utility: Google's own framing is that it performs roughly like non-private models from about five years earlier.

| Model | Architecture | Context | Privacy | Variants | Weights |
| --- | --- | --- | --- | --- | --- |
| VaultGemma 1B | Gemma 2 style | 1,024 | DP-SGD, epsilon at most 2.0 | PT only | Open (Gemma Terms) |

It is a research artifact and a calibration point for DP scaling laws, not a production model, but it is the only openly released LLM of its kind at this scale.

## Gemma 3 (March 2025): multimodal and long context

**Released:** 10 March 2025. **Primary source:** [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (arXiv 2503.19786).

Gemma 3 is the biggest single jump in the family. Everything above 1B became multimodal, context went from 8K to 128K, and the size range spread from 1B to 27B (later 270M).

**What changed versus Gemma 2:**

- **Attention:** 5 local layers per global layer, window shortened to 1,024 tokens. Global layers use RoPE base 1M for long context; local layers keep 10K. Logit soft-capping is replaced by **QK-norm**, which is friendlier to fused attention kernels.
- **Context:** 128K for 4B, 12B and 27B; 32K for 1B.
- **Vision:** a 417M SigLIP-derived encoder at 896x896, producing 256 image tokens, with an optional **Pan and Scan** step at inference that tiles non-square or high-resolution images.
- **Tokenizer:** the Gemini 2.0 SentencePiece tokenizer, about 262K entries, with split digits and byte fallback.
- **Pretraining:** all sizes distilled (the report samples 256 teacher logits per token), on 2T to 14T tokens.
- **Post-training:** distillation from a large instruction-tuned teacher, then RL with improved versions of BOND, WARM and WARP, with rewards from weight-averaged reward models, code execution and ground-truth math checks.
- **Quantization-aware training:** official QAT checkpoints (about 5,000 steps) for int4 and fp8 targets, which is why Gemma 3 Q4_0 GGUFs lose so little quality.

The report's headline claims: Gemma 3 4B-IT is competitive with Gemma 2 27B-IT, and Gemma 3 27B-IT is comparable to Gemini 1.5 Pro on its benchmark suite.

### Gemma 3 variants

| Model | Params (embedding / non-embedding) | Vision encoder | Training tokens | Context | Modality | Variants | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 270M | ~170M / ~100M | none | n/s | 32K (per FunctionGemma card) | Text | PT, IT, QAT | Open (Gemma Terms) |
| Gemma 3 1B | 302M / 698M | none | 2T | 32K | Text | PT, IT, QAT | Open (Gemma Terms) |
| Gemma 3 4B | 675M / 3,209M | 417M | 4T | 128K | Text + image | PT, IT, QAT | Open (Gemma Terms) |
| Gemma 3 12B | 1,012M / 10,759M | 417M | 12T | 128K | Text + image | PT, IT, QAT | Open (Gemma Terms) |
| Gemma 3 27B | 1,416M / 25,600M | 417M | 14T | 128K | Text + image | PT, IT, QAT | Open (Gemma Terms) |

"n/s" means not stated in a primary source I could find. The 270M (14 August 2025, [developer blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)) is the oddest model in the family: about 170M of its 270M parameters are embeddings (the large vocabulary again), leaving roughly 100M in the transformer blocks. It is meant as a fine-tuning base for narrow tasks, and Google's figure is that the INT4 version used 0.75% of a Pixel 9 Pro battery for 25 conversations.

Look at the embedding column for the 1B: 302M of its 1B parameters are the embedding table. If you are choosing a sub-2B model for on-device use, that ratio is the hidden cost of Gemma's 262K vocabulary, and [Small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design) discusses exactly this tradeoff against Phi, MobileLLM and TinyLlama.

## Gemma 3n (May and June 2025): the mobile-first detour

**Preview:** 20 May 2025 at Google I/O. **Full release:** 26 June 2025. **Primary source:** [Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/).

Gemma 3n carries the Gemma 3 name but is a different architecture, designed for phones, and it is the bridge to Gemma 4's edge models. Google described it at I/O as the architecture that would also power the next Gemini Nano. Four ideas:

- **MatFormer** (nested transformer): the E2B model lives *inside* the E4B. You can extract the E2B, or use "Mix-n-Match" to build intermediate sizes by changing per-layer feed-forward widths and skipping layers.
- **Per-Layer Embeddings (PLE):** each layer gets its own small per-token embedding that can live in CPU memory, so only the core transformer weights must sit in accelerator memory. That is the meaning of the "E" in E2B: about 2B *effective* parameters in accelerator memory out of 5B raw (E4B: about 4B of 8B). Google quotes memory footprints as low as 2 GB and 3 GB.
- **KV cache sharing:** keys and values from a middle layer are reused by the top layers, which Google says roughly doubles prefill speed versus Gemma 3 4B.
- **New encoders:** a USM-based audio encoder (about 6 tokens per second of audio, speech recognition and translation, 30-second clips at launch) and a MobileNet-V5-300M vision encoder at 256, 512 or 768 pixels.

| Model | Raw params | Effective params | Inputs | Languages | Weights |
| --- | --- | --- | --- | --- | --- |
| Gemma 3n E2B (PT, IT) | 5B | ~2B | Text, image, audio, video | 140 text, 35 multimodal | Open (Gemma Terms) |
| Gemma 3n E4B (PT, IT) | 8B | ~4B | Text, image, audio, video | 140 text, 35 multimodal | Open (Gemma Terms) |

PLE and KV sharing both reappear in Gemma 4's E2B and E4B, so if you understood Gemma 3n, you already understand half of Gemma 4's edge design.

![Family tree showing each specialized Gemma model branching from the core generation it was built on, with RecurrentGemma as a separate Griffin side branch and Gemma 3n bridging to Gemma 4](/imgs/blogs/gemma-model-family-timeline-3.webp)

## The offshoot families

The tree above is the second mental model. Every specialized Gemma is a fine-tune, continued pretraining, or structural adaptation of a core checkpoint (RecurrentGemma is the one exception: a separate Griffin architecture that shares the brand and the tokenizer family). That has a practical consequence: an offshoot can only be as good as its base, so a 2024 offshoot like CodeGemma or ShieldGemma inherits Gemma 1's or Gemma 2's 8K context and older reasoning quality, no matter how good its specialization was. The Gemma 3 offshoots below are the largest cluster.

### ShieldGemma 2 (10 March 2025)

**Primary source:** [ShieldGemma 2: Robust and Tractable Image Content Moderation](https://arxiv.org/abs/2504.01081) (arXiv 2504.01081).

A 4B image-safety classifier built on Gemma 3 4B, covering sexually explicit content, violence and gore, and dangerous content, for both natural images (inputs to a VLM) and synthetic images (outputs of an image generator).

| Model | Base | Task | Weights |
| --- | --- | --- | --- |
| ShieldGemma 2 4B-IT | Gemma 3 4B | Image moderation, 3 harm categories | Open (Gemma Terms) |

Note that ShieldGemma 2 is not a text-moderation upgrade to ShieldGemma. For text moderation the newest Google option remains the 2024 Gemma 2 based ShieldGemma.

### MedGemma (20 May 2025; 27B multimodal on 9 July 2025; 1.5 on 13 January 2026)

**Primary source:** [MedGemma Technical Report](https://arxiv.org/abs/2507.05201) (arXiv 2507.05201) and the [MedGemma 1.5 model card](https://huggingface.co/google/medgemma-1.5-4b-it).

MedGemma adapts Gemma 3 4B and 27B to medical text and images, with a medically tuned SigLIP encoder (MedSigLIP). It ships under the Health AI Developer Foundations (HAI-DEF) terms rather than the plain Gemma terms.

| Model | Released | Base | Modality | Variants | Weights |
| --- | --- | --- | --- | --- | --- |
| MedGemma 4B | 20 May 2025 | Gemma 3 4B | Text + medical image | PT, IT | Open (HAI-DEF terms) |
| MedGemma 27B text | 20 May 2025 | Gemma 3 27B | Text | IT | Open (HAI-DEF terms) |
| MedGemma 27B multimodal | 9 Jul 2025 | Gemma 3 27B | Text + medical image | IT | Open (HAI-DEF terms) |
| MedGemma 1.5 4B | 13 Jan 2026 | Gemma 3 4B | Text + image, 3D CT/MRI volumes, whole-slide pathology | IT | Open (HAI-DEF terms) |

MedGemma 1.5 adds 3D CT and MRI interpretation, multi-patch histopathology, longitudinal chest X-ray comparison, bounding-box localization, lab-report extraction and EHR text interpretation. It is not a strict upgrade: its own card reports chest X-ray report generation dropping from 30.3 to 27.2 RadGraph F1 versus the tuned MedGemma 1 4B. If report generation is your task, benchmark both.

### EmbeddingGemma (4 September 2025)

**Primary source:** [EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354) (arXiv 2509.20354).

A 300M text embedding model built from Gemma 3 with encoder-decoder (T5Gemma-style) initialization. Output is 768-dimensional with Matryoshka truncation to 512, 256 or 128, input up to 2,048 tokens, trained on 100+ languages. The report claims the best MTEB results among models under 500M parameters across multilingual, English and code.

| Model | Params | Max input | Dims (Matryoshka) | Weights |
| --- | --- | --- | --- | --- |
| EmbeddingGemma 300M | 300M | 2,048 | 768, 512, 256, 128 | Open (Gemma Terms) |

### FunctionGemma (18 December 2025)

**Primary source:** [FunctionGemma model card](https://huggingface.co/google/functiongemma-270m-it).

Gemma 3 270M re-tuned with a different chat format as a base for *fine-tuning* into a narrow function-calling model. Google is explicit that it is not meant to be used as a dialogue model out of the box. Up to 32K input tokens.

| Model | Base | Context | Role | Weights |
| --- | --- | --- | --- | --- |
| FunctionGemma 270M-IT | Gemma 3 270M | 32K | Function-calling fine-tuning base | Open (Gemma Terms) |

### T5Gemma 2 (18 December 2025)

**Primary source:** [T5Gemma 2: Seeing, Reading, and Understanding Longer](https://arxiv.org/abs/2512.14856) (arXiv 2512.14856).

The encoder-decoder adaptation repeated on Gemma 3, now multimodal (SigLIP image tokens go into the encoder) and long-context (up to 128K input, 32K output). Two parameter-saving changes: word embeddings tied across encoder and decoder, and decoder self-attention merged with cross-attention into a single module.

| Model (encoder-decoder) | Base | Inputs | Context | Weights |
| --- | --- | --- | --- | --- |
| T5Gemma 2 270M-270M | Gemma 3 270M | Text + image | 128K in / 32K out | Open (Gemma Terms) |
| T5Gemma 2 1B-1B | Gemma 3 1B | Text + image | 128K in / 32K out | Open (Gemma Terms) |
| T5Gemma 2 4B-4B | Gemma 3 4B | Text + image | 128K in / 32K out | Open (Gemma Terms) |

### TranslateGemma (15 January 2026)

**Primary source:** [TranslateGemma Technical Report](https://arxiv.org/abs/2601.09012) (arXiv 2601.09012).

Gemma 3 fine-tuned for translation across 55 languages, with an SFT stage on 4.3B tokens followed by an RL stage on 10.2M tokens. It also translates text inside images (896x896, 256 tokens per image). The total input budget is only 2K tokens, so it is a sentence-and-paragraph translator, not a document translator.

| Model | Base | Languages | Inputs | Context | Weights |
| --- | --- | --- | --- | --- | --- |
| TranslateGemma 4B-IT | Gemma 3 4B | 55 | Text + image | 2K | Open (Gemma Terms) |
| TranslateGemma 12B-IT | Gemma 3 12B | 55 | Text + image | 2K | Open (Gemma Terms) |
| TranslateGemma 27B-IT | Gemma 3 27B | 55 | Text + image | 2K | Open (Gemma Terms) |

## Gemma 4 (April 2026): thinking, MoE and Apache 2.0

**Announced:** 2 April 2026 (release log: 31 March 2026). **Primary sources:** [Google blog: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), and the [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) (arXiv 2607.02770, submitted 2 July 2026).

Gemma 4 took 13 months, the longest gap in the family, and it changed more at once than any previous generation. Google says it is built from the same research as Gemini 3.

**What changed versus Gemma 3:**

- **License: Apache 2.0.** This is the change most teams will feel first. Every earlier core Gemma used the custom Gemma Terms of Use. Gemma 4 is the first core generation under an OSI-approved license, which removes the legal review step that kept Gemma out of some enterprise stacks.
- **A Mixture-of-Experts model.** The 26B A4B has 25.2B total parameters and activates about 3.8B per token, with 128 experts, 8 active plus 1 shared. (The technical report's Table 1 lists 24.5B total and 2.8B active, which conflicts with its own text and the model card; I use the model card.)
- **Thinking mode.** Put a `<|think|>` token in the system prompt and the model emits a reasoning trace in a thought channel before answering. Remove it to disable thinking.
- **Cheaper global attention.** Global layers use unified keys and values and p-RoPE (p = 0.25), for up to 37.5% less global KV cache. The final layer is always global.
- **Edge models inherit Gemma 3n.** E2B and E4B use Per-Layer Embeddings and KV cache sharing (20 of 35 layers for E2B, 18 of 42 for E4B), plus a 305M USM-based audio encoder that the report says is 55% smaller than Gemma 3n's.
- **New vision encoder.** A ViT with variable aspect ratio and a configurable soft-token budget (70, 140, 280, 560 or 1,120 tokens per image), 150M parameters for the edge models and 550M for the large ones. Video is processed as frames (up to 60 seconds at one frame per second).
- **Context:** 128K for E2B and E4B, 256K for the larger models. Over 140 languages. Training data cutoff January 2025.
- **Agentic features:** native function calling, structured JSON output, and system-instruction support.

### Gemma 4 variants

| Model | Released | Total params | Active / effective | Layers | Window | Context | Inputs | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B (PT, IT) | 2 Apr 2026 | 5.1B (with embeddings) | 2.3B effective | 35 | 512 | 128K | Text, image, audio | Open (Apache 2.0) |
| Gemma 4 E4B (PT, IT) | 2 Apr 2026 | 8B (with embeddings) | 4.5B effective | 42 | 512 | 128K | Text, image, audio | Open (Apache 2.0) |
| Gemma 4 26B A4B (PT, IT) | 2 Apr 2026 | 25.2B | 3.8B active (MoE 128 experts, 8+1 active) | 30 | 1,024 | 256K | Text, image | Open (Apache 2.0) |
| Gemma 4 31B (PT, IT) | 2 Apr 2026 | 30.7B | dense | 60 | 1,024 | 256K | Text, image | Open (Apache 2.0) |
| Gemma 4 12B Unified (PT, IT) | 3 Jun 2026 | 11.95B | dense | 48 | 1,024 | 256K | Text, image, audio (encoder-free) | Open (Apache 2.0) |

All specs are from the [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4). Audio clips are limited to 30 seconds. Google reports, among others, 89.2 on AIME 2026 and 80.0 on LiveCodeBench v6 for the 31B, against 20.8 and 29.1 for Gemma 3 27B in the same report; these are self-reported numbers, and the AIME jump in particular reflects the new reasoning post-training rather than the base model alone. At launch Google said the 31B ranked third and the 26B sixth among open models on the Arena text leaderboard.

The QAT story continues: official Q4_0 checkpoints for every size and a mixed int2/int4 "mobile" format for the edge models. The report's weight-memory table puts the 31B at 64 GB in bf16 and 19.2 GB at Q4_0, and the 26B A4B at 52 GB and 16.2 GB. That is the number that decides your GPU: the 26B A4B *computes* like a 4B but must *store* like a 26B.

### Gemma 4 MTP drafters (16 April 2026)

**Primary source:** [Gemma 4 31B-IT assistant model card](https://huggingface.co/google/gemma-4-31B-it-assistant) and the technical report.

Two weeks after launch Google released multi-token-prediction drafters ("assistant" checkpoints on Hugging Face) for speculative decoding. Each is a 4-layer transformer (three local, one global attention layer) that cross-attends to the target model's KV cache, so it needs no separate prefill. Google claims up to 3x decode speedup with identical output. For the edge models, the drafter replaces the full 262K vocabulary projection with a top-k over 4,096 token clusters.

| Drafter for | Drafter size | Weights |
| --- | --- | --- |
| Gemma 4 E2B | 76M | Open (Apache 2.0) |
| Gemma 4 E4B | 77M | Open (Apache 2.0) |
| Gemma 4 12B | 400M | Open (Apache 2.0) |
| Gemma 4 26B A4B | 430M | Open (Apache 2.0) |
| Gemma 4 31B | 500M | Open (Apache 2.0) |

The 12B drafter arrived with the 12B itself in June. How drafters behave under real batching is a separate question from the headline speedup; [Speculative decoding in production](/blog/machine-learning/speculative-decoding/speculative-decoding-in-production) covers the gap between "up to 3x" and what a loaded server sees.

### Gemma 4 12B Unified (3 June 2026)

**Primary source:** [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [Hugging Face: gemma-4-12B](https://huggingface.co/google/gemma-4-12B).

The most architecturally novel model in the family. The 12B has **no vision or audio encoder**. Image patches (48x48x3 RGB) go through a single 35M-parameter linear projection with 2D coordinate position embeddings; raw 16 kHz audio is cut into 40 ms chunks (640-dimensional vectors) and projected straight into the embedding space. Everything then flows through one decoder-only transformer. The practical benefits are that the whole model fine-tunes in one pass and there is no separate encoder to quantize or serve. It fills the gap between E4B and 26B A4B and, at 7.65 GB of Q4_0 weights in the report's table, is aimed at 16 GB laptops.

### EmbeddingGemma 2 (6 October 2026)

**Primary source:** [Hugging Face: google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and the [release log](https://ai.google.dev/gemma/docs/releases).

The newest Gemma as of this writing. A 740M multimodal embedding model built on the Gemma 4 architecture that maps text, code, images, video and audio into one 768-dimensional space. The parameters break down into a 270M text model, a 170M vision encoder and a 300M audio encoder, and the encoders are optional, so you can deploy text-only at 270M. Context grows to 8,192 tokens (four times EmbeddingGemma's 2,048), Matryoshka dimensions are 768, 512, 256 and 128, and the license is Apache 2.0. The card notes it is pretrained only, with no post-training safety tuning.

| Model | Params | Modalities | Max input | Dims | Weights |
| --- | --- | --- | --- | --- | --- |
| EmbeddingGemma 2 (text only) | 270M | Text, code | 8,192 | 768 to 128 | Open (Apache 2.0) |
| EmbeddingGemma 2 (full) | 740M | Text, code, image, video, audio | 8,192 | 768 to 128 | Open (Apache 2.0) |

Google reports MTEB Code rising from 68.76 to 78.68 versus the first EmbeddingGemma; that comparison comes from Google's launch materials as reported in coverage, and I could not open the launch post itself to confirm it.

## How the family evolved

Step back from the tables and five through-lines explain almost every decision.

**1. The KV cache is the design target.** Every generation's main architecture change reduces memory per token of context: Gemma 2 made half the layers sliding-window, Gemma 3 made five of six layers sliding-window with a quarter of the window, Gemma 4 made the remaining global layers cheaper with unified K=V and partial RoPE, and the edge models share KV across layers. Context length went 8K, 8K, 128K, 256K, and the cache did not grow anywhere near 32 times. The cost is that Gemma is a *hybrid-cache* model from Gemma 2 onward, and your serving stack must understand that.

**2. Small models are distilled from large ones.** From Gemma 2 on, the small sizes learn from a teacher's distribution instead of raw labels, Gemma 3 extended that to every size, and post-training also leans heavily on teacher distillation. This is why the small Gemmas punch above their parameter count and why the family's quality jumps tend to arrive at the small end first.

**3. One core, many adaptations.** Google ships one strong core per generation and lets narrow teams adapt it: code, vision, safety, medicine, embedding, translation, privacy, function calling, encoder-decoder. The offshoot count per base went 3, 5, 9, and Gemma 4 has three so far after six months. The corollary for users is in the tree figure: always ask what an offshoot was built on before comparing it to a newer core model.

**4. The edge is a first-class target.** The 2B in Gemma 1 was a small model; Gemma 3n and Gemma 4's E-series are *designed* for phones, with nested sub-models, per-layer embeddings held off the accelerator, smaller audio encoders, int2/int4 mobile quantization, and drafters with clustered vocabularies. Since Gemma 3, official QAT checkpoints have been part of every release rather than a community afterthought.

**5. Openness ratcheted up once.** Weights were open from day one, but under custom terms. Gemma 4 moved the core family and its newest offshoots (MTP drafters, 12B Unified, EmbeddingGemma 2) to Apache 2.0. The Gemma 3 based offshoots, including TranslateGemma, FunctionGemma and T5Gemma 2, remain under the Gemma terms, and MedGemma under HAI-DEF terms, so the license now depends on which base the model sits on.

## Which variant to pick today

As of October 2026, here is the short guide.

| You need | Pick | Why |
| --- | --- | --- |
| A general open model on one GPU | Gemma 4 31B (Q4_0 for 24 GB cards) | Strongest dense Gemma, 256K context, thinking mode, Apache 2.0 |
| High throughput per GPU-dollar | Gemma 4 26B A4B | About 3.8B active per token; but budget memory for all 25.2B |
| A laptop or 16 GB GPU | Gemma 4 12B Unified | Fills the mid gap; encoder-free audio and vision |
| A phone or browser | Gemma 4 E2B or E4B | PLE, KV sharing, audio input, mobile QAT format |
| Faster decode on any Gemma 4 | Pair with its MTP drafter | Up to 3x claimed, output unchanged |
| Retrieval embeddings | EmbeddingGemma 2 (text-only 270M if you need no images) | 8K input, multimodal option, Apache 2.0 |
| A tiny model to fine-tune for one task | Gemma 3 270M or FunctionGemma | Still the smallest Gemmas; no Gemma 4 equivalent yet |
| Medical imaging or text | MedGemma 1.5 4B or MedGemma 27B | Only Gemma-family medical models; check HAI-DEF terms |
| Translation of short texts | TranslateGemma | Purpose-trained, but only 2K tokens of input |
| Image safety classification | ShieldGemma 2 | Only official image moderator in the family |
| Text safety classification | ShieldGemma (Gemma 2 based) or a newer guard model | No Gemma 3 or 4 text guard yet |
| Detection or segmentation outputs | PaliGemma 2 mix | Trained to emit boxes and masks |

The default answer, in other words, is Gemma 4. The exceptions are the domains where the specialized offshoot still exists only on an older base, and in each of those it is worth re-checking the release log before starting: going by the family's history, a Gemma 4 based successor is likely, but it is not here until it is.

## References

Release log and announcements

- Google AI for Developers, [Gemma releases](https://ai.google.dev/gemma/docs/releases) (release dates for every entry above)
- Google, [Gemma 4: Byte for byte, the most capable open models](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) (2 April 2026)
- Google, [DataGemma announcement](https://blog.google/technology/ai/google-datagemma-ai-llm/) (12 September 2024)
- Google Developers Blog, [Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) (26 June 2025)
- Google Developers Blog, [Introducing Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/) (14 August 2025)
- Google Research, [VaultGemma: The world's most capable differentially private LLM](https://research.google/blog/vaultgemma-the-worlds-most-capable-differentially-private-llm)

Technical reports

- Gemma Team, [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295), arXiv 2403.08295
- [CodeGemma: Open Code Models Based on Gemma](https://arxiv.org/abs/2406.11409), arXiv 2406.11409
- [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://arxiv.org/abs/2404.07839), arXiv 2404.07839
- [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726), arXiv 2407.07726
- Gemma Team, [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118), arXiv 2408.00118
- [ShieldGemma: Generative AI Content Moderation Based on Gemma](https://arxiv.org/abs/2407.21772), arXiv 2407.21772
- [PaliGemma 2: A Family of Versatile VLMs for Transfer](https://arxiv.org/abs/2412.03555), arXiv 2412.03555
- Gemma Team, [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), arXiv 2503.19786
- [ShieldGemma 2: Robust and Tractable Image Content Moderation](https://arxiv.org/abs/2504.01081), arXiv 2504.01081
- [Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](https://arxiv.org/abs/2504.06225) (T5Gemma), arXiv 2504.06225
- [MedGemma Technical Report](https://arxiv.org/abs/2507.05201), arXiv 2507.05201
- [EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354), arXiv 2509.20354
- [VaultGemma: A Differentially Private Gemma Model](https://arxiv.org/abs/2510.15001), arXiv 2510.15001
- [T5Gemma 2: Seeing, Reading, and Understanding Longer](https://arxiv.org/abs/2512.14856), arXiv 2512.14856
- [TranslateGemma Technical Report](https://arxiv.org/abs/2601.09012), arXiv 2601.09012
- Gemma Team, [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770), arXiv 2607.02770

Model cards

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B), [google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant)
- [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it), [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it), [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- [google/t5gemma-2-1b-1b](https://huggingface.co/google/t5gemma-2-1b-1b), [google/vaultgemma-1b](https://huggingface.co/google/vaultgemma-1b), [google/paligemma2-3b-pt-448](https://huggingface.co/google/paligemma2-3b-pt-448)
