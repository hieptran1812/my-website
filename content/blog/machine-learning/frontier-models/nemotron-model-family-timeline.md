---
title: "The Nemotron Family Timeline: Every NVIDIA Nemotron Release from Nemotron-3 8B to Nemotron 3.5"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference catalog of NVIDIA's Nemotron models, from the 2023 Nemotron-3 8B to the 2026 Nemotron 3 Ultra and 3.5 Lightning: dates, variants, architecture changes, licenses and primary sources for each generation."
tags: ["nemotron", "nvidia", "llm", "model-family", "timeline", "mamba", "mixture-of-experts", "pruning", "distillation", "reasoning", "open-weights"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 33
---

## Why a Nemotron timeline needs to exist

NVIDIA's model naming is a trap. "Nemotron-3" is an 8B dense model from November 2023. "Nemotron 3" is a hybrid Mamba-Transformer mixture-of-experts family from December 2025 that reaches 550B parameters. They share a name and almost nothing else. In between sit Nemotron-4 (which is older than Nemotron 3), a run of models that are really pruned Llamas, a hybrid line called Nemotron-H, a "Nano 2" that has no "Nano 1" in the same architecture, and a 3.5 release that is the smallest member of the 3 family. If you join a conversation about "the latest Nemotron" without a map, you will be wrong about at least one of: the architecture, the license, or the year.

This post is that map. It walks every release in chronological order, from the first Nemotron weights in November 2023 to the latest release that exists as of October 14, 2026 (Nemotron 3.5 Lightning, August 2026). For each generation it gives the date, what changed against the previous one, a table of variants, and a primary-source link. Wherever this blog already has a deep-dive, the section links to it, so you can use this page as the index and go deep from there.

![Nemotron release timeline, Nov 2023 to Aug 2026, grouped into three architecture eras](/imgs/blogs/nemotron-model-family-timeline-1.webp)

The diagram above is the mental model: three architecture eras. In the first, NVIDIA trained its own dense Transformers from scratch (Nemotron-3 8B, Nemotron-4 15B and 340B). In the second, it stopped competing on pretraining and instead took Meta's Llama models and Mistral NeMo and made them smaller, faster and better aligned through pruning, distillation, neural architecture search and reward modeling. In the third, it went back to training its own models, but with a different architecture: first hybrid Mamba-2 plus a few attention layers (Nemotron-H, Nano 2), then hybrid Mamba plus mixture-of-experts (Nemotron 3). The eras overlap in time, which is part of why the naming confuses people.

A note on sourcing. Every date and spec below comes from an arXiv paper, an NVIDIA blog or developer post, or an official `huggingface.co/nvidia` model card. Where a source was silent or contradicted itself, the text says so instead of picking a number. Model cards in particular disagree with themselves on training-token counts, and I flag those cases rather than smoothing them over.

## How to read the names

Before the chronology, a decoder table. Most confusion comes from four naming collisions.

| Name you see | What it actually is | Year |
| --- | --- | --- |
| Nemotron-3 8B (hyphen) | NVIDIA's first own dense 8B, 4K context | 2023 |
| Nemotron-4 15B / 340B | Own dense Transformers, GQA, squared ReLU, 256K vocab | 2024 |
| Minitron | Pruned and distilled children of Nemotron-4 15B, Llama 3.1 8B and Mistral NeMo 12B | 2024 |
| Llama-3.1-Nemotron-70B | Meta's Llama 3.1 70B, re-aligned by NVIDIA with RLHF | 2024 |
| Llama-Nemotron Nano / Super / Ultra | Llama models shrunk with Puzzle NAS, plus a reasoning toggle | 2025 |
| Nemotron-H | Hybrid Mamba-2 / Transformer, trained from scratch | 2025 |
| Nemotron Nano 2 | Hybrid Mamba-2 dense 12B, compressed to 9B | 2025 |
| Nemotron 3 (space) Nano / Super / Ultra | Hybrid Mamba-2 plus MoE, up to 1M context | 2025 to 2026 |
| Nemotron 3.5 Lightning | Smallest member of the 3 family, 30B-A3B with MTP | 2026 |

Two rules of thumb fall out. If the name starts with `Llama-`, the weights are a Meta derivative and inherit the Llama community license on top of NVIDIA's. If the name contains `-H`, `Nano 2`, or a bare `3`/`3.5`, the model is NVIDIA-pretrained and hybrid Mamba. The "Nano / Super / Ultra" tiering appears twice (Llama-Nemotron in 2025, Nemotron 3 in 2025 to 2026) with the same meaning both times: edge or single small GPU, single data-center GPU or node, and multi-GPU data-center flagship.

## 1. Nemotron-3 8B (November 15, 2023)

The family starts small. On November 15, 2023, NVIDIA announced the Nemotron-3 8B family as part of its "AI Foundation Models" push, distributed through Azure AI's model catalog, Hugging Face and NGC. The pitch was enterprise chatbots and copilots built on production-ready weights you could customize, not a frontier model.

What it was: a GPT-style decoder-only Transformer with standard multi-head attention, trained on about 3.8T tokens covering 53 natural languages and 37 programming languages, with a 4,096-token context. The license was the NVIDIA AI Foundation Models Community License, which is more restrictive than what came later.

What made it interesting was not the base model but the alignment variants shipped around it. NVIDIA released the same base aligned four different ways, which reads in hindsight like a public comparison of alignment recipes.

| Variant | What it is | Context | Weights |
| --- | --- | --- | --- |
| Nemotron-3-8B-Base-4k | Pretrained base, about 3.8T tokens | 4K | Open (AI Foundation Models Community License) |
| Nemotron-3-8B-Chat-SFT | Supervised fine-tuning only | 4K | Open |
| Nemotron-3-8B-Chat-RLHF | SFT plus RLHF | 4K | Open |
| Nemotron-3-8B-Chat-SteerLM | SteerLM attribute-conditioned SFT | 4K | Open |
| Nemotron-3-8B-QA | Tuned for retrieval-augmented QA | 4K | Open |

The SteerLM variant is the one to remember. SteerLM conditions generation on attribute values (helpfulness, verbosity, and so on) at inference time and is trained with SFT instead of RL. It is the first appearance of a thread that runs through the whole family: NVIDIA invests heavily in reward and attribute data (HelpSteer, HelpSteer2, HelpSteer3) and in alignment methods that use it. The released layer count, hidden size and tokenizer were not stated on the cards I checked, so I leave them out.

Primary sources: [NVIDIA developer blog announcement](https://developer.nvidia.com/blog/nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/), [base model card](https://huggingface.co/nvidia/nemotron-3-8b-base-4k), [SteerLM post](https://developer.nvidia.com/blog/announcing-steerlm-a-simple-and-practical-technique-to-customize-llms-during-inference/).

## 2. Nemotron-4 15B (February 26, 2024)

Three months later NVIDIA published the Nemotron-4 15B technical report on arXiv (v1 on February 26, 2024). This is where the house architecture that defined Era 1 was written down.

| Item | Nemotron-4 15B |
| --- | --- |
| Layers / hidden size | 32 / 6144 |
| Attention | 48 query heads, 8 KV heads (grouped-query attention) |
| Activation | Squared ReLU |
| Positions | RoPE, no biases, untied embeddings, dropout 0 |
| Vocabulary | 256,000 (SentencePiece BPE) |
| Context | 4,096 |
| Parameters | 3.2B embedding + 12.5B non-embedding |
| Training tokens | 8T (70% English, 15% multilingual, 15% code) |
| Hardware | 384 DGX H100 nodes |

What changed against Nemotron-3 8B: roughly twice the training tokens (8T versus 3.8T), GQA instead of full multi-head attention (a direct KV-cache saving), a very large 256K vocabulary aimed at multilingual efficiency, and squared ReLU in the MLP. NVIDIA's claim was best-in-class multilingual performance at its size.

Two of these choices outlived the generation. The 256K vocabulary carried straight into the 340B model. The 15B itself became the parent of the Minitron family (section 4), which is arguably more important than the 15B as a product. I could not verify a public weights card for the 15B base, so treat it as a paper release plus a parent checkpoint.

Primary source: [arXiv 2402.16819](https://arxiv.org/abs/2402.16819).

## 3. Nemotron-4 340B Base, Instruct and Reward (June 14, 2024)

On June 14, 2024 NVIDIA released Nemotron-4 340B, and positioned it unusually: not as a chatbot, but as a pipeline for generating synthetic training data for other models. The technical report followed on arXiv (v1 June 17, 2024). This blog has a full deep-dive: [Nemotron-4 340B: How NVIDIA Aligned a Frontier Model on 98% Synthetic Data](/blog/machine-learning/frontier-models/nemotron-4-340b-synthetic-data-alignment).

Architecturally it is the 15B recipe scaled up.

| Item | Nemotron-4 340B |
| --- | --- |
| Layers / hidden size | 96 / 18432 |
| Attention | 96 heads, 8 KV heads (GQA) |
| Activation / positions | Squared ReLU / RoPE, untied embeddings |
| Vocabulary | 256K |
| Context | 4,096 |
| Parameters | 9.4B embedding + 331.6B non-embedding |
| Training tokens | 8T pretraining + 1T continued pretraining = 9T |
| Hardware | 6,144 H100 GPUs, Dec 2023 to May 2024 |
| Deployment | Fits one 8-GPU DGX H100 in FP8; BF16 needs 16 H100s |

The variants:

| Variant | Role | Context | Weights |
| --- | --- | --- | --- |
| Nemotron-4-340B-Base | Pretrained base, 9T tokens | 4K | Open (NVIDIA Open Model License) |
| Nemotron-4-340B-Instruct | Aligned chat model; Arena Hard 54.2, MT-Bench 8.22 | 4K | Open |
| Nemotron-4-340B-Reward | Multi-attribute reward model on HelpSteer2; RewardBench 92.0 | 4K | Open |

What changed: three things matter more than the size. First, the license. The 340B shipped under the NVIDIA Open Model License, far more permissive than the 2023 community license and explicit that you can use outputs to train other models. That license is the reason the "synthetic data generator" framing works at all. Second, the alignment recipe: code SFT (about 800K samples), general SFT (200K), DPO (160K), then three iterations of Reward-aware Preference Optimization (RPO, 300K), with over 98% of the alignment data synthetic and only around 20K human annotations. Third, the reward model ships as a product, scoring 92.0 on RewardBench, which made NVIDIA a reward-model vendor and not just a policy-model vendor.

The weakness was context. At 4K tokens it was already behind the 128K Llama 3.1 that Meta shipped a month later, and that gap is a big part of why Era 2 happened.

Primary sources: [NVIDIA blog](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/), [arXiv 2406.11704](https://arxiv.org/abs/2406.11704), [Base model card](https://huggingface.co/nvidia/Nemotron-4-340B-Base), [HelpSteer2 arXiv 2406.08673](https://arxiv.org/abs/2406.08673).

## 4. Minitron (July to August 2024)

Minitron is where NVIDIA found its comparative advantage. Instead of pretraining every size from scratch, it trains one big model and prunes and distills it into smaller ones. The deep-dive is [Minitron: How NVIDIA Builds a Whole LLM Family by Pruning and Distilling One Model](/blog/machine-learning/frontier-models/nvidia-minitron-pruning-distillation), and the broader technique is covered in [Pruning in LLMs](/blog/machine-learning/large-language-model/pruning-in-llm).

![Compression lineage: four parent families, three NVIDIA methods (Minitron, Puzzle, Elastic), most of the family's sizes](/imgs/blogs/nemotron-model-family-timeline-2.webp)

The figure above is the single most useful thing to know about the family's sizes: most of them were not trained from scratch. Minitron started it; Puzzle (section 7), MiniPuzzle (section 8) and Nemotron Elastic (section 12) continued it.

Three releases in about five weeks:

| Release | Date | Parent | How | Result |
| --- | --- | --- | --- | --- |
| Minitron 8B / 4B (paper) | Jul 19, 2024 (arXiv) | Nemotron-4 15B | Structured pruning (depth, width, heads, MLP) + distillation | Under 3% of the original data per model; up to 40x fewer tokens; Minitron-8B MMLU 64.5 |
| Llama-3.1-Minitron 4B Width / Depth | Aug 14, 2024 (blog); arXiv Aug 21 | Llama 3.1 8B | Width: hidden 4096 to 3072, MLP 14336 to 9216. Depth: 32 to 16 layers. 94B distillation tokens, teacher correction | About 1.8x (width) and 2.7x (depth) throughput vs the 8B on one H100 |
| Mistral-NeMo-Minitron 8B | Aug 21, 2024 | Mistral NeMo 12B | Width pruning: hidden 5120 to 4096, MLP 14336 to 11520; 380B distillation tokens | MMLU 69.5 |

What changed against the 340B era: NVIDIA started taking other people's base models as parents. Llama-3.1-Minitron is the first time a Meta model went through the pipeline, and "teacher correction" (fine-tune the teacher on the distillation data first, because the original pretraining data is not available) is the trick that made that possible. Everything in Era 2 follows from it. Licensing is the NVIDIA Open Model License on these cards, combined with the parent's license where one applies.

A December 17, 2024 follow-up shipped Mistral-NeMo-Minitron-128k-Instruct at 8B, 4B and 2B with 128K context, aimed at RTX AI PCs.

Primary sources: [arXiv 2407.14679](https://arxiv.org/abs/2407.14679), [arXiv 2408.11796](https://arxiv.org/abs/2408.11796), [Llama-3.1-Minitron blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/), [Mistral-NeMo-Minitron blog](https://blogs.nvidia.com/blog/mistral-nemo-minitron-8b-small-language-model/), [Minitron-8B-Base card](https://huggingface.co/nvidia/Minitron-8B-Base).

## 5. Llama-3.1-Nemotron 51B and 70B (September to October 2024)

Era 2 shows its full shape in autumn 2024, with two Llama-derived releases that do opposite things to the same parent.

**Llama-3.1-Nemotron-51B-Instruct (September 23, 2024)** takes Llama-3.1-70B-Instruct and makes it smaller. It uses neural architecture search with the Puzzle algorithm: each Transformer block gets a menu of cheaper alternatives (skip the attention, replace it with a linear layer, shrink the FFN), each alternative is trained by block-wise distillation, and a search picks the per-block combination that hits a throughput or memory target. The result keeps MMLU at 80.2 versus 81.66 for the parent and MT-Bench at 8.99 versus 8.93, while running about 2.2x faster and fitting one H100-80GB.

**Llama-3.1-Nemotron-70B-Instruct (October 2024)** keeps the size and improves the alignment. It starts from Llama-3.1-70B-Instruct and trains it with REINFORCE against a new reward model, using the HelpSteer2-Preference prompts. At release it led the three automatic alignment benchmarks NVIDIA reported: Arena Hard 85.0, AlpacaEval 2 LC 57.6, MT-Bench 8.98. It reached Chatbot Arena Elo 1267 as of October 24, 2024. The HelpSteer2-Preference paper (arXiv v1 October 2, 2024) is also the first matched head-to-head of Bradley-Terry versus regression-style reward modeling, and the reward model combines both. I could not find a dated NVIDIA post for the exact public release day, so I give it as October 2024.

| Variant | Date | Params | Context | Role | Weights |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-Nemotron-51B-Instruct | Sep 23, 2024 | 51B | Not verified | Puzzle NAS + distillation from Llama 3.1 70B | Open |
| Llama-3.1-Nemotron-70B-Instruct | Oct 2024 | 70B | 128K in, 4K out | REINFORCE alignment of Llama 3.1 70B | Open (NVIDIA Open Model License + Llama 3.1 license) |
| Llama-3.1-Nemotron-70B-Reward | Oct 2024 | 70B | 128K | Reward model; RewardBench 94.1, top of 140+ RMs as of Oct 1, 2024 | Open |

What changed: NVIDIA's flagship chat model was now somebody else's pretrain. The two levers it kept for itself were architecture search (51B) and alignment plus reward modeling (70B). Those two levers are exactly what the 2025 Llama-Nemotron family combined.

Primary sources: [51B developer blog](https://developer.nvidia.com/blog/advancing-the-accuracy-efficiency-frontier-with-llama-3-1-nemotron-51b/), [51B card](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct), [70B-Instruct card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF), [70B-Reward card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward), [HelpSteer2-Preference arXiv 2410.01257](https://arxiv.org/abs/2410.01257).

## 6. The late-2024 side releases: Nemotron-Mini, Hymba, Nemotron-CC

Three smaller releases in the second half of 2024 matter more for what they foreshadow than for their own use.

**Nemotron-Mini-4B-Instruct (August 2024).** Announced at Gamescom 2024 (around August 20) as NVIDIA's first on-device small language model for its ACE game-character stack. It is fine-tuned from Minitron-4B-Base, which was itself pruned from Nemotron-4 15B, so it sits at the end of the first lineage chain in the figure above. Hidden size 3072, 32 heads, GQA, 4K context, built for roleplay, RAG QA and function calling, quantized to INT4 at about 2GB of VRAM. License: NVIDIA Community Model License.

**Hymba-1.5B (November 20, 2024).** Not branded Nemotron, but it is the family's first published hybrid of attention and Mamba. Hymba runs attention heads and SSM heads in parallel inside each layer, adds learnable meta tokens, shares KV across layers and keeps only 3 full-attention layers. Against Llama-3.2-3B it claims +1.32% average accuracy with an 11.67x smaller cache and 3.49x throughput. Five months later Nemotron-H took the hybrid idea to 56B, with a different (sequential rather than parallel) layer layout.

**Nemotron-CC (December 3, 2024).** A 6.3T-token Common Crawl dataset built with classifier ensembling, synthetic rephrasing and fewer heuristic filters. The paper reports that an 8B model trained on 15T tokens (7.2T from Nemotron-CC) beats Llama 3.1 8B by 5 MMLU points. This is the start of NVIDIA releasing pretraining data alongside models, which becomes standard practice from Nano 2 onward.

| Release | Date | Size | What it foreshadowed | Weights / data |
| --- | --- | --- | --- | --- |
| Nemotron-Mini-4B-Instruct | Aug 2024 | 4B | On-device Nano tier | Open (Community Model License) |
| Hymba-1.5B Base / Instruct | Nov 20, 2024 | 1.5B | Hybrid attention + Mamba | Open (Open Model License) |
| Nemotron-CC | Dec 3, 2024 | 6.3T tokens | Open pretraining data | Open dataset |
| Mistral-NeMo-Minitron-128k-Instruct | Dec 17, 2024 | 8B / 4B / 2B | 128K context on small models | Open |

Primary sources: [Gamescom blog](https://blogs.nvidia.com/blog/gamescom-2024/), [Nemotron-Mini card](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct), [Hymba arXiv 2411.13676](https://arxiv.org/abs/2411.13676), [Nemotron-CC arXiv 2412.02595](https://arxiv.org/abs/2412.02595), [RTX small-model blog](https://developer.nvidia.com/blog/deploy-agents-assistants-and-avatars-on-nvidia-rtx-ai-pcs-with-new-small-language-models/).

## 7. Llama-Nemotron Nano, Super and Ultra (announced January 6, 2025; shipped March to July 2025)

At CES on January 6, 2025, NVIDIA announced two families with a shared tiering: Llama Nemotron for language and Cosmos Nemotron for vision-language, each in Nano (PC and edge), Super (single GPU) and Ultra (data center). Weights followed at GTC on March 18, 2025, by which time the pitch had become reasoning. The deep-dive is [Llama-Nemotron: How NVIDIA Makes Reasoning Models Fast with NAS, FFN Fusion, and a Reasoning Toggle](/blog/machine-learning/frontier-models/llama-nemotron-efficient-reasoning-models).

| Variant | Released | Parent | Params | Context | What is special | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-Nemotron-Nano-8B-v1 | Mar 18, 2025 | Llama-3.1-8B-Instruct | 8B | 128K | Reasoning toggle | Open |
| Llama-3.3-Nemotron-Super-49B-v1 | Mar 18, 2025 | Llama-3.3-70B-Instruct | 49B | 128K | Puzzle NAS, 40B distillation tokens, fits one H100-80GB | Open |
| Llama-3.1-Nemotron-Ultra-253B-v1 | Apr 7, 2025 | Llama-3.1-405B-Instruct | 253B | 128K | Puzzle NAS + FFN Fusion; 65B distillation + 88B continued-pretraining tokens; fits one 8xH100 node | Open |
| Llama-3.1-Nemotron-Nano-4B-v1.1 | May 20, 2025 | Llama-3.1-Minitron-4B-Width | 4B | 128K | Minitron child, post-trained for reasoning | Open |
| Llama-3.3-Nemotron-Super-49B-v1.5 | Jul 25, 2025 | Same as Super v1 | 49B | 128K | Reasoning on by default (`/no_think` to disable), RL + iterative DPO for tool calling | Open |

All five carry the NVIDIA Open Model License plus the relevant Llama 3.1 or 3.3 community license, and all are cleared for commercial use. The technical report (arXiv 2505.00949) went up on May 2, 2025, alongside the Llama-Nemotron-Post-Training-Dataset.

What changed against the 2024 Llama derivatives:

1. **A reasoning toggle.** A system prompt of `detailed thinking on` or `detailed thinking off` switches the same weights between long chain-of-thought and direct answers. The paper calls these the first open models with that dynamic toggle. It removes the need to serve a separate "thinking" model next to the chat model.
2. **FFN Fusion** on Ultra. After Puzzle removes many attention blocks, runs of consecutive FFNs are left with little dependency between them; FFN Fusion merges them into wider parallel FFNs, cutting sequential depth and latency.
3. **An RL stage for reasoning.** Distillation from strong reasoning teachers for SFT, then reinforcement learning with verifiable rewards on Ultra, and RLHF-style alignment for chat.

The weakness was structural: these models still had full-attention KV caches, so the long reasoning traces they were now trained to produce cost memory linearly in length. That is the problem the hybrid line, developed in parallel, was built to solve.

Primary sources: [CES 2025 blog](https://blogs.nvidia.com/blog/nemotron-model-families/), [arXiv 2505.00949](https://arxiv.org/abs/2505.00949), [Nano 8B card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1), [Super 49B card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1), [Ultra 253B card](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1), [Nano 4B v1.1 card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1), [Super v1.5 card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5).

## 8. Nemotron-H 8B, 47B and 56B (April 2025)

Nemotron-H is where Era 3 begins: NVIDIA back to pretraining its own models, with most attention layers replaced by Mamba-2. The paper went up on arXiv on April 4, 2025 and base weights followed on April 14. The deep-dive is [Nemotron-H: How NVIDIA Swaps Most Attention for Mamba-2](/blog/machine-learning/frontier-models/nemotron-h-hybrid-mamba-transformer), and the serving consequences are worked through in [Hybrid Models and the End of the KV-Cache Assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption).

| Variant | Layers | Attention layers | Training | Context | Weights |
| --- | --- | --- | --- | --- | --- |
| Nemotron-H-8B-Base-8K | 52 | 4 | 15T tokens | 8K | Research-only license |
| Nemotron-H-56B-Base-8K | 118 | 10 | 20T tokens, FP8 (first and last 4 layers in BF16) | 8K | Research-only license |
| Nemotron-H-47B-Base-8K | Compressed from 56B | Not verified | MiniPuzzle, 63B tokens | 8K | Research-only license |
| Nemotron-H-8B / 47B-Reasoning-128K (+ FP8) | As above | As above | Reasoning post-training (Jun 6, 2025) | 128K | Open research license |

What changed against Llama-Nemotron:

- **Architecture.** Roughly 8% of layers are self-attention; the rest alternate Mamba-2 and FFN. A Mamba-2 layer carries a fixed-size recurrent state instead of a KV cache that grows with sequence length, so long-context decode memory is nearly flat. NVIDIA reported accuracy matching or beating Qwen-2.5 and Llama-3.1 at comparable sizes, at up to 3x the inference speed.
- **Precision.** The 56B was pretrained in FP8, the first time a Nemotron flagship pretraining run was not BF16, and the paper reports FP8 matching BF16.
- **Compression.** MiniPuzzle (pruning plus distillation, a lighter cousin of Puzzle) shrank 56B to 47B in 63B tokens at similar accuracy and about 20% more speed.
- **License, backwards.** The base checkpoints shipped under a research and development license, not the commercial Open Model License. That is a step back from 2024, and the reason Nemotron-H base models never became popular production checkpoints despite the architecture.

The June 2025 reasoning variants made the argument concrete: NVIDIA's blog claims about 4x the throughput of Llama-Nemotron Super 49B v1 and 84% versus 46% on RULER at 128K.

Primary sources: [arXiv 2504.03624](https://arxiv.org/abs/2504.03624), [56B base card](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K), [8B base card](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K), [Nemotron-H Reasoning blog](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/).

## 9. Nemotron Nano 2 (August 18, 2025) and the Nano VL models

Nemotron Nano 2 is the hybrid architecture shipped as a commercial small model. NVIDIA-Nemotron-Nano-9B-v2 was released on August 18, 2025, with the paper on arXiv two days later.

The recipe stacks the two NVIDIA specialties. A 12B hybrid base (Mamba-2 and MLP layers with only 4 attention layers) was pretrained on 20T tokens with FP8. It was aligned, then compressed to 9B with the Minitron method so that 128K context runs on a single A10G (22 GiB, BF16). NVIDIA claims up to 6x the throughput of Qwen3-8B in an 8K-in, 16K-out reasoning setting.

| Variant | Params | Context | Role | Weights |
| --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-Nano-12B-v2-Base | 12B | 128K | Hybrid base, 20T tokens, FP8 | Open (Open Model License) |
| NVIDIA-Nemotron-Nano-9B-v2-Base | 9B | 128K | Minitron-compressed base | Open |
| NVIDIA-Nemotron-Nano-9B-v2 (+ FP8, NVFP4 builds) | 9B | 128K | Reasoning model with `/think`, `/no_think`, thinking budget | Open |

What changed against Nemotron-H: the commercial license came back (NVIDIA Open Model License), context went from 8K base to 128K, and reasoning control got a runtime thinking budget, not just an on/off switch. NVIDIA also released most of the data: Nemotron-Pretraining-Dataset-v1 (6.6T tokens), Nemotron-CC-v2, Nemotron-CC-Math-v1 (133B tokens), Nemotron-CC-Code-v1, and the Nemotron-Post-Training-Dataset-v1 (25.66M rows generated by DeepSeek-R1-0528 and Qwen3-235B, CC BY 4.0). From here on, an open Nemotron release usually means open weights plus open data.

**The vision-language branch.** The CES "Cosmos Nemotron" VLM line shipped under Nano VL names:

| Variant | Date | Backbone | Context | Inputs | Weights |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-Nemotron-Nano-VL-8B-V1 | Jun 3, 2025 | Llama-3.1-8B-Instruct + C-RADIOv2-H | 16K | Image + text | Open (Open Model License + Llama 3.1) |
| Nemotron Nano 12B v2 VL | Oct 28, 2025 | Nano-12B-v2 hybrid + C-RADIOv2-H | 128K | Up to 4 images, video, text | Open (BF16, FP8, NVFP4-QAD) |

The second one matters: it is the first time the hybrid Mamba backbone carried a vision model, and its paper (arXiv 2511.03929) leans on token reduction for long documents and video, which is exactly where a flat-memory backbone pays off.

Primary sources: [arXiv 2508.14444](https://arxiv.org/abs/2508.14444), [Nano-9B-v2 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2), [Nano 2 research page](https://research.nvidia.com/labs/adlr/NVIDIA-Nemotron-Nano-2), [Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1), [Nano VL 8B card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1), [Nano 12B v2 VL card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16), [Nano V2 VL arXiv 2511.03929](https://arxiv.org/abs/2511.03929).

## 10. Nemotron 3 Nano 30B-A3B (December 15, 2025)

Nemotron 3 is the current generation, and Nano was its first member, released December 15, 2025. The family paper, "NVIDIA Nemotron 3: Efficient and Open Intelligence," went up on arXiv on December 24, 2025, with per-model technical reports on NVIDIA's research site.

The architectural change is the biggest since 2023: the hybrid Mamba backbone gains mixture-of-experts. Nano has 52 layers (23 MoE, 23 Mamba-2, 6 GQA attention), 128 routed experts plus 1 shared expert with 6 active per token. The card lists about 3.5B active parameters out of 30B total; the "A3B" in the name rounds down.

| Variant | Total / active | Context | Precision builds | Reasoning control | Weights |
| --- | --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Nano-30B-A3B (Base, BF16, FP8) | 30B / about 3.5B | Up to 1M (HF default 256K) | BF16, FP8 | `enable_thinking`, `reasoning_budget` | Open (NVIDIA Nemotron Open Model License) |

What changed against Nano 2:

- **Dense to MoE.** Nano 2 is a 9B dense hybrid; Nemotron 3 Nano is a 30B sparse hybrid that activates about as many parameters as a 3B dense model. The Mamba layers fix decode memory, the experts fix per-token compute, and the two savings compound.
- **Context to 1M.** A flat-memory backbone is what makes a 1M-token window affordable on a model this size.
- **A new license name**, the NVIDIA Nemotron Open Model License.
- **RL at scale.** The family paper describes multi-environment RL post-training with reasoning-budget control. For RLHF judging NVIDIA released Qwen3-Nemotron-235B-A22B-GenRM the same day, an Apache 2.0 generative reward model fine-tuned from Qwen3-235B-A22B-Thinking-2507.

Pretraining tokens are one of the conflicts I flagged at the start. The card says about 25T tokens, while its own dataset table sums to a smaller unique-token figure. The likely reconciliation is multiple epochs over the curated data, but the card does not say so, so I report "about 25T, per the card."

Released data accompanying Nemotron 3 includes Nemotron-CC-v2.1 (2.5T English tokens), Nemotron-CC-Code-v1 (428B tokens), Nemotron-Pretraining-Code-v2, Nemotron-Pretraining-Specialized-v1, the Nemotron-Post-Training-v3 collection, and RL environments in NeMo Gym. The inference engineering series on this blog builds a serving engine for exactly this class of model; start with [Implementing a Two-Cache Engine: KV Blocks Plus Recurrent State](/blog/machine-learning/inference-engineering/implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state).

Primary sources: [arXiv 2512.20856](https://arxiv.org/abs/2512.20856), [Nemotron 3 research hub](https://research.nvidia.com/labs/nemotron/Nemotron-3), [Nano 30B-A3B card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [Qwen3-Nemotron GenRM card](https://huggingface.co/nvidia/Qwen3-Nemotron-235B-A22B-GenRM).

## 11. Nemotron 3 Super 120B-A12B (March 11, 2026)

Super shipped on March 11, 2026, and introduced the three techniques that separate the large Nemotron 3 models from Nano.

| Variant | Total / active | Context | Precision | Reasoning control | Weights |
| --- | --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Super-120B-A12B (Base, BF16, FP8, NVFP4, MTP checkpoint) | 120B / 12B | Up to 1M (HF default 256K) | Pretrained in NVFP4; BF16 / FP8 / NVFP4 builds | `enable_thinking`, `reasoning_budget`, low-effort mode | Open (NVIDIA Nemotron Open Model License) |

What changed against Nano:

1. **LatentMoE.** Tokens are projected into a smaller latent dimension, routed and processed by experts there, and projected back. The saved bandwidth and all-to-all communication are reinvested in more experts; NVIDIA says about 4x the experts at the same cost. This blog has a paper-reading post on the technique: [LatentMoE: Shrink the Expert Highway, Spend the Savings on More Experts](/blog/paper-reading/large-language-model/latentmoe), which also flags a caveat about the batch-1 latency regime.
2. **Multi-token prediction with shared weights.** MTP heads are trained with the model and double as a built-in speculative-decoding draft. NVIDIA reports up to 3x wall-clock speedup on code and tool-call generation. On why speculative decoding is harder on a recurrent backbone, see [Speculative Decoding and Rollback with Recurrent State](/blog/machine-learning/inference-engineering/speculative-decoding-and-rollback-with-recurrent-state).
3. **NVFP4 pretraining.** Super is the first Nemotron 3 model pretrained mostly in 4-bit floating point (NVFP4 for most linear layers' weights, activations and gradients, with some layers in BF16 or MXFP8). This continues the precision line from FP8 Nemotron-H.

The card says over 25T pretraining tokens, about 7M SFT samples, then RL across 21 environments with more than 1.2M rollouts. Languages: English, French, German, Italian, Japanese, Spanish and Chinese. NVIDIA claims more than 5x the throughput of the previous Super (Llama-Nemotron Super 49B), and that NVFP4 on B200 runs 4x faster than FP8 on H100. Those are vendor numbers; treat them as upper bounds on hardware NVIDIA chose.

Primary sources: [Super card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), [Super developer blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/).

## 12. Nemotron 3 Nano 4B and Nemotron 3 Nano Omni (March to April 2026)

Two Nano-tier releases filled in the small end of the 3 family in spring 2026.

**NVIDIA-Nemotron-3-Nano-4B (March 16, 2026)** is dense, not MoE: a Mamba-2 and MLP hybrid with 4 attention layers, compressed from Nano-9B-v2 with a new method, Nemotron Elastic, with 262K context. It is the third link in the Nano 2 chain shown in the lineage figure (12B base, 9B, 4B). License: NVIDIA Nemotron Open Model License, BF16 and FP8 builds.

**Nemotron 3 Nano Omni 30B-A3B-Reasoning (April 28, 2026)** puts vision and audio encoders on the Nano 30B-A3B backbone and accepts video, audio, images and text with 256K context. NVIDIA positions it as the successor to Nano 12B v2 VL. The paper is arXiv 2604.24954 (April 27, 2026). The encoders are reported as C-RADIOv4-H for vision and a Parakeet TDT encoder for audio; I verified those only through the model listing and blog summary, not the card body. For the Parakeet lineage, see [Canary and Parakeet: FastConformer ASR](/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr).

| Variant | Date | Params | Architecture | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Nano-4B | Mar 16, 2026 | 4B dense | Mamba-2 + MLP + 4 attention layers | 262K | Text | Open |
| Nemotron 3 Nano Omni 30B-A3B-Reasoning | Apr 28, 2026 | 30B / about 3B | Nano 3 MoE backbone + vision + audio encoders | 256K | Video, audio, image, text in; text out | Open (BF16, FP8, FP4) |

Primary sources: [Nano 4B card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), [Nano Omni arXiv 2604.24954](https://arxiv.org/abs/2604.24954), [Nano Omni developer blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/).

## 13. Nemotron 3 Ultra 550B-A55B (June 4, 2026)

Ultra completed the three-tier Nemotron 3 lineup on June 4, 2026.

| Variant | Total / active | Context | Precision | Reasoning control | Weights |
| --- | --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-3-Ultra-550B-A55B (Base, BF16, NVFP4) | 550B / 55B | Up to 1M | Pretrained in NVFP4; BF16 and NVFP4 builds | `enable_thinking`, `reasoning_budget`, medium-effort mode | Open (OpenMDW-1.1) |
| NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM | 550B / 55B | Not verified | Not verified | Generative reward model | Listed; card not checked |

Architecturally Ultra is Super scaled up: LatentMoE, Mamba-2 and MoE layers interleaved with select attention layers, and shared-weight MTP. Languages grew to ten (adding Hindi, Korean, Brazilian Portuguese and Chinese to the European languages and Japanese). Pretraining tokens are the second unresolved conflict: the card states "approximately 20T tokens" in two places and "14.8 trillion tokens" as the total dataset size in another. As with Nano, I report the card's training figure and the conflict.

What changed against Super, beyond scale, is the **license**. Ultra ships under OpenMDW-1.1, an open model license rather than NVIDIA's own Nemotron license. For a 550B flagship that is the most permissive license the family has used, and it continued into the 3.5 releases. It also brings Ultra-scale reasoning to a model that NVIDIA trained from scratch, rather than the Llama-405B-derived Ultra of 2025: the family's top tier is NVIDIA's own pretraining again for the first time since the 340B.

Primary sources: [Ultra card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [Nemotron 3 research hub](https://research.nvidia.com/labs/nemotron/Nemotron-3).

## 14. Nemotron 3.5 Lightning 30B-A3B (August 11, 2026)

The latest release in the family as of this writing is Nemotron 3.5 Lightning, published August 11, 2026. NVIDIA's blog calls it the smallest member of the Nemotron 3 family, which is the naming collision of the year: a ".5" that is not a new generation but a fast execution model inside the existing one.

| Variant | Total / active | Context | Architecture | Precision | Weights |
| --- | --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B (NVFP4 and other builds) | 30B / 3B | Up to 1M | Mamba-2 + MoE + attention hybrid with MTP | NVFP4 pretraining recipe; deployment quantization W4A16 experts, FP8 Mamba projections and KV cache | Open (OpenMDW-1.1) |

What changed against Nemotron 3 Nano, which has the same 30B-A3B shape: MTP layers (Nano has none), an NVFP4 pretraining recipe over 20T+ tokens (Nano was BF16), and the OpenMDW license. The card does not mention LatentMoE, so I do not claim it. Lightning ships with speculative-decoding drafters (DSpark and DFlash; see this blog's [DFlash post](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding)) and with NeMo Switchyard, a library that routes planning to frontier models and execution to Lightning. That pairing is the product thesis: a cheap, fast, long-context executor for agent loops, with a bigger model planning above it. NVIDIA claims "up to 4x" the output speed of similar-sized models.

**The 3.5 utility models.** NVIDIA's Nemotron product page also lists a set of 2026 non-LLM or guard models under the Nemotron name: Nemotron 3.5 Content Safety (4B, fine-tuned from Gemma-3-4B-it), a streaming Nemotron 3.5 ASR model, Nemotron Parse 2.0, Nemotron 3 Embed (1B and 8B), and several "Nemotron-Labs" research checkpoints. I verified these only from that one listing page, not from individual model cards, so treat the names as pointers rather than specs.

Primary sources: [Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4), [Lightning developer blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/), [NVIDIA Nemotron model listing](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/llm-info/).

As of October 14, 2026 there is no "Nemotron 4" in the new naming scheme, and I do not speculate about one.

## Side lines: reward models, research models and data

Around the main line runs a large set of Nemotron-branded models that are best understood as NVIDIA research artifacts. Most are fine-tunes of Qwen or DeepSeek distills, which surprises people who assume "Nemotron" means an NVIDIA backbone.

| Release | Date | Base | What it is | License |
| --- | --- | --- | --- | --- |
| OpenCodeReasoning-Nemotron 7B / 14B / 32B / 32B-IOI | Apr 2, 2025 (arXiv) | Qwen2.5-Instruct | SFT-only code reasoning distillation; 61.8% LiveCodeBench (32B) | Open |
| OpenMath-Nemotron | Apr 23, 2025 (arXiv) | Qwen2.5-Math | AIMO-2 winning solution; 540K problems, 3.2M CoT + 1.7M tool-integrated solutions | Open |
| Nemotron-Research-Tool-N1 7B / 14B | Apr 25, 2025 (arXiv) | Not stated in abstract | Rule-based RL for tool calling with a binary format-and-correctness reward | Open |
| AceReason-Nemotron-14B (and 1.1-7B) | May 2025; 1.1 Jun 16, 2025 | DeepSeek-R1-Distill-Qwen-14B | RL-only reasoning, math prompts then code prompts | Open Model License |
| Llama-3.3-Nemotron-70B-Reward (+ Multilingual) | Jun 27, 2025 | Llama-3.3-70B | Bradley-Terry RM on HelpSteer3; RM-Bench 79.9 | Open Model License + Llama 3.3 |
| OpenReasoning-Nemotron 1.5B / 7B / 14B / 32B | Jul 16, 2025 | Qwen2.5 | SFT on about 5M DeepSeek-R1-0528 traces, GenSelect "heavy" mode | CC-BY-4.0 |
| Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 2025 (sources conflict on the day) | Llama-3.1-8B-Instruct | Multilingual safety classifier, 23 categories | Open |
| Qwen3-Nemotron-235B-A22B-GenRM | Dec 15, 2025 | Qwen3-235B-A22B-Thinking-2507 | Generative RM used to judge Nemotron 3 RLHF | Apache 2.0 |

Three of these have deep coverage elsewhere on this blog. Tool-N1 is analyzed in [Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning](/blog/paper-reading/large-language-model/nemotron-research-tool-n1-exploring-tool-using-language-models-with-reinforced-reasoning). OpenMath-Nemotron's AIMO-2 win is dissected in [AIMO Progress Prize 2: Published Solutions](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions). And the synthetic-data pipelines that the 340B started, and that every Nemotron post-training dataset since has scaled, are the subject of [Synthetic Data Generation](/blog/machine-learning/training-data/synthetic-data-generation).

The data line is worth listing on its own, because for many teams it is more useful than the models:

| Dataset | Date | Size | Use |
| --- | --- | --- | --- |
| HelpSteer | Nov 2023 | 37K rated samples | Attribute-conditioned alignment (SteerLM) |
| HelpSteer2 | Jun 2024 | About 10K pairs | Reward modeling (340B Reward) |
| HelpSteer2-Preference | Oct 2024 | Preference labels on HelpSteer2 | 70B Reward / Instruct |
| Nemotron-CC | Dec 2024 | 6.3T tokens | Pretraining |
| Nemotron-Post-Training-Dataset-v1 | Jul 31, 2025 | 25.66M rows | SFT for reasoning, code, math, tools |
| Nemotron-Pretraining-Dataset-v1 / CC-Math-v1 | Aug 2025 | 6.6T / 133B tokens | Pretraining (Nano 2) |
| Nemotron-CC-v2.1, Pretraining-Code-v2, Post-Training-v3 | Dec 2025 | 2.5T English tokens (CC-v2.1) | Pretraining and post-training (Nemotron 3) |

## How the family evolved

![What changed per generation: architecture, context, precision, license and reasoning control](/imgs/blogs/nemotron-model-family-timeline-3.webp)

The matrix above compresses three years into one picture, with one highlighted cell per generation marking its biggest change. Read down the columns and five through-lines appear.

**1. Compression is the core competence, not pretraining.** From Minitron in July 2024 onward, nearly every Nemotron size is derived: Minitron pruning and distillation (Nemotron-4 15B, Llama 3.1 8B, Mistral NeMo 12B, Nano 12B v2), Puzzle NAS (51B, Super 49B, Ultra 253B), MiniPuzzle (Nemotron-H 47B) and Nemotron Elastic (Nano 4B). The economic logic is simple: one expensive pretraining run, many cheap children at 2 to 40x fewer tokens each. When NVIDIA did not have a competitive pretrain (2024 to early 2025), it applied the same toolkit to Meta's.

**2. The architecture moved toward flat memory, then sparse compute.** Era 1 was a conventional dense Transformer with GQA. Hymba and Nemotron-H replaced most attention with Mamba-2, so decode memory stops growing with sequence length. Nemotron 3 added MoE so per-token compute stays near a 3B, 12B or 55B dense model regardless of total size, then LatentMoE and MTP to recover bandwidth and decode speed. Context followed the architecture: 4K, then 128K, then 1M.

**3. Precision drops a notch every generation.** BF16 training with FP8 serving (340B), FP8 pretraining (Nemotron-H 56B, Nano 2), NVFP4 pretraining (Super, Ultra, Lightning). This is the hardware vendor's thesis made visible: each Nemotron generation is also a demonstration that the newest tensor-core format is safe to train in. Expect the release order of formats and models to stay coupled.

**4. Reward models and RL are first-class products.** SteerLM and HelpSteer in 2023, the 340B Reward in 2024, the 70B Reward that topped RewardBench, the HelpSteer3-based reward models and the Qwen3-Nemotron and Ultra GenRMs in 2025 to 2026. Reasoning control evolved from a binary "detailed thinking on/off" (March 2025) to `/think` with a thinking budget (Nano 2) to `reasoning_budget` plus effort modes (Nemotron 3). NVIDIA has been consistently more open than most labs about its alignment data and judges.

**5. Openness ratcheted up, with one detour.** Community license (2023), NVIDIA Open Model License (2024 to 2025), a research-only detour for Nemotron-H base models, the Nemotron Open Model License (Nano 3, Super), and OpenMDW-1.1 for Ultra and the 3.5 releases. Alongside the weights, the data releases grew from 37K rated samples to multi-trillion-token pretraining corpora. Of the large model families, Nemotron is one of the few where you can inspect a large share of both the pretraining and the post-training data.

## Which variant to pick today

As of October 2026, the old generations are mostly of historical interest. A short decision table:

| Your situation | Pick | Why |
| --- | --- | --- |
| Agent executor loop, long context, cost-sensitive | Nemotron 3.5 Lightning 30B-A3B | 3B active, MTP, 1M context, OpenMDW license; built to sit under a planner |
| General small reasoning model on one GPU | Nemotron 3 Nano 30B-A3B | Mature, well-supported, thinking budget control |
| Dense model for edge or a small GPU | Nemotron 3 Nano 4B (or Nano-9B-v2) | Dense hybrid, flat memory, no expert routing to serve |
| Strong open reasoning on one node | Nemotron 3 Super 120B-A12B | LatentMoE + MTP + NVFP4 builds; best quality per active parameter in the family below Ultra |
| Frontier-scale open model, permissive license | Nemotron 3 Ultra 550B-A55B | Largest Nemotron, OpenMDW-1.1, ten languages |
| Multimodal (video, audio, image) | Nemotron 3 Nano Omni 30B-A3B | Only Nemotron with audio and video inputs on the MoE backbone |
| Reward model or LLM judge | Ultra GenRM or Qwen3-Nemotron-235B GenRM; Llama-3.3-Nemotron-70B-Reward for a lighter scalar RM | Generative judges for RL, scalar RM for ranking |
| Your stack only serves standard Transformers | Llama-3.3-Nemotron-Super-49B-v1.5 | Plain attention, standard KV cache, still a good reasoner |
| Studying pruning or NAS | Minitron, Llama-3.1-Minitron, Llama-Nemotron papers | The clearest published recipes for compression at scale |

The last two rows deserve a caution. A hybrid Mamba MoE is not a drop-in replacement in a serving stack built for KV caches: you need an engine that manages recurrent state next to KV blocks, schedules prefill as a scan, and handles speculative decoding rollback. The [Batching and Scheduling Hybrid Models](/blog/machine-learning/inference-engineering/batching-and-scheduling-hybrid-models) post covers what changes. If your serving engine does not support the hybrid layers well, the older Llama-derived models are still the pragmatic choice despite worse long-context economics.

## References

Primary sources in chronological order.

1. NVIDIA, "NVIDIA AI Foundation Models: Build Custom Enterprise Chatbots and Co-Pilots with Production-Ready LLMs," Nov 15, 2023. [developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/)
2. Nemotron-3-8B-Base-4k model card. [huggingface.co/nvidia/nemotron-3-8b-base-4k](https://huggingface.co/nvidia/nemotron-3-8b-base-4k)
3. "HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM," Nov 2023. [arXiv 2311.09528](https://arxiv.org/abs/2311.09528)
4. "Nemotron-4 15B Technical Report," Feb 2024. [arXiv 2402.16819](https://arxiv.org/abs/2402.16819)
5. "HelpSteer2: Open-source dataset for training top-performing reward models," Jun 2024. [arXiv 2406.08673](https://arxiv.org/abs/2406.08673)
6. NVIDIA, "Nemotron-4 340B Technical Report," Jun 2024. [arXiv 2406.11704](https://arxiv.org/abs/2406.11704); [NVIDIA blog](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)
7. "Compact Language Models via Pruning and Knowledge Distillation," Jul 2024. [arXiv 2407.14679](https://arxiv.org/abs/2407.14679)
8. "LLM Pruning and Distillation in Practice: The Minitron Approach," Aug 2024. [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)
9. NVIDIA, "Lightweight Champ: NVIDIA Releases Small Language Model With State-of-the-Art Accuracy," Aug 21, 2024. [blogs.nvidia.com](https://blogs.nvidia.com/blog/mistral-nemo-minitron-8b-small-language-model/)
10. NVIDIA, "Advancing the Accuracy-Efficiency Frontier with Llama-3.1-Nemotron-51B," Sep 23, 2024. [developer.nvidia.com](https://developer.nvidia.com/blog/advancing-the-accuracy-efficiency-frontier-with-llama-3-1-nemotron-51b/)
11. "HelpSteer2-Preference: Complementing Ratings with Preferences," Oct 2024. [arXiv 2410.01257](https://arxiv.org/abs/2410.01257); [Llama-3.1-Nemotron-70B-Instruct card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)
12. "Hymba: A Hybrid-head Architecture for Small Language Models," Nov 2024. [arXiv 2411.13676](https://arxiv.org/abs/2411.13676)
13. "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset," Dec 2024. [arXiv 2412.02595](https://arxiv.org/abs/2412.02595)
14. NVIDIA, "NVIDIA Announces Nemotron Model Families to Advance Agentic AI," Jan 6, 2025. [blogs.nvidia.com](https://blogs.nvidia.com/blog/nemotron-model-families/)
15. NVIDIA, "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models," Apr 2025. [arXiv 2504.03624](https://arxiv.org/abs/2504.03624)
16. NVIDIA, "Llama-Nemotron: Efficient Reasoning Models," May 2025. [arXiv 2505.00949](https://arxiv.org/abs/2505.00949)
17. "Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning," Apr 2025. [arXiv 2505.00024](https://arxiv.org/abs/2505.00024)
18. "AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset" (OpenMath-Nemotron), Apr 2025. [arXiv 2504.16891](https://arxiv.org/abs/2504.16891)
19. "OpenCodeReasoning: Advancing Data Distillation for Competitive Coding," Apr 2025. [arXiv 2504.01943](https://arxiv.org/abs/2504.01943)
20. "AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning," May 2025. [arXiv 2505.16400](https://arxiv.org/abs/2505.16400)
21. NVIDIA, "NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model," Aug 2025. [arXiv 2508.14444](https://arxiv.org/abs/2508.14444); [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
22. NVIDIA, "NVIDIA Nemotron Nano V2 VL," Nov 2025. [arXiv 2511.03929](https://arxiv.org/abs/2511.03929)
23. NVIDIA, "NVIDIA Nemotron 3: Efficient and Open Intelligence," Dec 2025. [arXiv 2512.20856](https://arxiv.org/abs/2512.20856); [research hub](https://research.nvidia.com/labs/nemotron/Nemotron-3)
24. Nemotron 3 Nano 30B-A3B model card. [huggingface.co](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
25. Nemotron 3 Super 120B-A12B model card and blog, Mar 2026. [card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16); [blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)
26. Nemotron 3 Nano 4B model card, Mar 2026. [huggingface.co](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
27. "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence," Apr 2026. [arXiv 2604.24954](https://arxiv.org/abs/2604.24954)
28. Nemotron 3 Ultra 550B-A55B model card, Jun 2026. [huggingface.co](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
29. Nemotron 3.5 Lightning 30B-A3B model card and blog, Aug 2026. [card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4); [blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/)
30. NVIDIA Nemotron model listing. [nvidia.com](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/llm-info/)
