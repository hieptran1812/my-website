---
title: "The GLM Model Family, Release by Release: From Blank Infilling to GLM-5.3"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference catalog of every GLM release from Zhipu AI and Z.ai, March 2021 to August 2026: dates, what changed each generation, variant tables with sizes, context and licenses, and links to primary sources and our deep-dives."
tags: ["glm", "zhipu", "z-ai", "chatglm", "glm-4-5", "glm-5", "mixture-of-experts", "open-weights", "model-family", "timeline", "large-language-model"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 37
---

Ask someone what "GLM" is and the answer depends on the year they last looked. In 2021 it was a pretraining objective in an ACL paper. In 2022 it was a 130B bilingual model you had to apply for. In 2023 it was ChatGLM-6B, the model that ran on a gaming GPU in every Chinese university lab. In 2025 it was a 355B mixture-of-experts that sat next to DeepSeek and Kimi on agentic-coding leaderboards. In 2026 it is a 744B sparse-attention MoE with a 1M-token context, a separate natively multimodal Flash line, and, for the first time, a license that is not plain MIT. Same three letters, five different things, and the model cards each describe themselves only relative to the one before.

This post is the map. It walks the family in strict chronological order, from the 2021 GLM paper to GLM-5.3 and GLM-5.3-Flash in August 2026, and for each generation records the release date, what changed relative to the previous one (architecture, data, context, training recipe, license), and a variant table. Every date and spec below was checked against a primary source: an arXiv technical report, the Z.ai developer docs release notes, the official GitHub README news list, the official Hugging Face model card, or the `config.json` shipped with the weights. Where those sources disagree or stay silent, the post says so instead of guessing. The timeline figure below is the mental model: three eras, and a release cadence that went from one generation a year to one flagship every two months.

![GLM flagship text releases on a 2021 to 2026 axis, colored by era: research and dense models, the GLM-4 platform, and the frontier MoE line](/imgs/blogs/glm-model-family-timeline-1.webp)

The figure spaces cards for readability, so read the dates on the cards rather than the gaps between them. A note on scope: the *technique* story of this family (why blank infilling, why GLM-130B chose FP16, why GLM-4.5 went deep and narrow, how the agentic RL works) is already told in [The GLM Lineage: Five Years of Frontier-LLM Technique](/blog/machine-learning/frontier-models/glm-lineage-frontier-llm-technique). Read that for the "why". This post is the catalog that complements it: the "what, when, and which checkpoint". Where this blog has a deep-dive on a specific release, it is linked inline.

## The release timeline at a glance

The table is the whole catalog in one screen; the rest of the post expands each row. Dates are the official announcement (Z.ai release notes, GitHub news entry, official blog, or arXiv v1). Where the only primary evidence is the date the Hugging Face repository was created, the date is marked "~", because a repo is often created privately a few days before launch.

| # | Release | Date | Flagship size | Context | Open weights | License |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GLM (paper + checkpoints) | Mar 2021 (arXiv v1) | 10B dense | 1K | Yes | MIT code, research checkpoints |
| 2 | GLM-130B | Aug 4, 2022 | 130B dense | 2K | On application | Custom research license |
| 3 | ChatGLM-6B | Mar 14, 2023 | 6.2B dense | 2K | Yes | Custom model license |
| 4 | VisualGLM-6B | May 17, 2023 | 7.8B (6.2B LM + vision) | 2K | Yes | Custom model license |
| 5 | ChatGLM2-6B | Jun 25, 2023 | 6B dense | 32K (8K chat) | Yes | Custom model license |
| 6 | CogVLM | Oct 5, 2023 | 17B | n/a | Yes | Apache 2.0 code, model license |
| 7 | ChatGLM3-6B | Oct 27, 2023 | 6B dense | 8K / 32K / 128K | Yes | Custom model license |
| 8 | CogAgent | Dec 15, 2023 | 18B | n/a | Yes | Apache 2.0 code, model license |
| 9 | GLM-4 (API) | Jan 16, 2024 | Undisclosed | 128K | No | API |
| 10 | GLM-4-9B, GLM-4V-9B | Jun 5, 2024 | 9B / 13.9B | 128K / 1M / 8K | Yes | GLM-4 license |
| 11 | GLM-4-Voice | Oct 2024 | 9B | n/a | Yes | Not stated on card |
| 12 | GLM-Edge | ~Nov 2024 | 1.5B to 5B | n/a | Yes | GLM-4 license |
| 13 | GLM-4-0414 and GLM-Z1 | Apr 14, 2025 | 32B dense | 32K | Yes | MIT |
| 14 | GLM-4.1V-Thinking | Jul 1, 2025 | 9B (10.3B with ViT) | 64K | Yes | MIT |
| 15 | GLM-4.5, GLM-4.5-Air | Jul 28, 2025 | 355B / 32B active | 128K | Yes | MIT |
| 16 | GLM-4.5V | Aug 11, 2025 | 106B / 12B active | 64K | Yes | MIT |
| 17 | GLM-4.6 | Sep 30, 2025 | 355B / 32B active | 200K | Yes | MIT |
| 18 | GLM-4.6V, GLM-4.6V-Flash | ~Dec 2025 | 106B / 9B | 128K | Yes | MIT |
| 19 | GLM-4.7 | Dec 22, 2025 | 355B / 32B active | 200K | Yes | MIT |
| 20 | GLM-4.7-Flash | Jan 2026 | 30B / 3B active | 200K | Yes | MIT |
| 21 | GLM-OCR | Feb 3, 2026 | 0.9B | 128K | Yes | MIT |
| 22 | GLM-5 | Feb 12, 2026 | 744B / 40B active | 200K | Yes | MIT |
| 23 | GLM-5-Turbo | Mar 2026 | Undisclosed | 200K | No | API |
| 24 | GLM-5V-Turbo | Apr 2, 2026 | Undisclosed | 200K | No | API |
| 25 | GLM-5.1 | Apr 7, 2026 | 744B / 40B active | 200K | Yes | MIT |
| 26 | GLM-5.2 | Jun 16, 2026 | 744B / 40B active | 1M | Yes | MIT |
| 27 | GLM-5.3 | Aug 18, 2026 (weights Aug 28) | 744B / 40B active | 1M | Yes | GLM-5.3 License |
| 28 | GLM-5.3-Flash | Aug 26, 2026 | 320B / 18B active | 1M | Yes | MIT |

Three patterns jump out before we read a single model card. First, open weights were the default from 2023 onward, with the API-only exceptions clustered at the two "platform" moments (GLM-4 in January 2024 and the Turbo variants in spring 2026). Second, the license story is a slow liberalization (custom research license, then custom model license, then MIT from April 2025) followed by one deliberate step back with GLM-5.3. Third, the cadence: four years to go from GLM to GLM-4.5, then thirteen months to go from GLM-4.5 to GLM-5.3.

A note on parameter counts. Z.ai's model cards quote rounded "marketing" sizes (355B, 106B, 744B). The safetensors index on Hugging Face reports slightly larger totals because it includes the multi-token-prediction (MTP) layer and embeddings: 358.3B for GLM-4.5 and GLM-4.7, 356.8B for GLM-4.6, 753.9B for GLM-5 and GLM-5.1, 753.3B for GLM-5.2 and GLM-5.3, and 321.3B for GLM-5.3-Flash. Both numbers are "correct". This post quotes the model card number and gives the safetensors total where it matters for memory planning.

## 1. GLM (March 2021): the objective, not yet a product

**Released:** arXiv v1 in March 2021 ([2103.10360](https://arxiv.org/abs/2103.10360)); published at ACL 2022. **Org:** Tsinghua University KEG lab with BAAI; the commercial spin-out Zhipu AI came later.

GLM started as an answer to a 2021 problem: BERT was best at understanding, GPT at generation, T5 at conditional generation, and nobody had one objective that won all three. GLM's answer was *autoregressive blank infilling*: mask random spans, then regenerate them left to right, with 2D positional encoding (position in the corrupted text, position inside the span) and an attention mask that is bidirectional over the context and causal over the spans. The full mechanism, and why it still matters, is in our deep-dive [GLM: Autoregressive Blank Infilling](/blog/machine-learning/frontier-models/autoregressive-blank-infilling-glm). For the catalog, what matters is that this objective is the DNA every later release inherits by name, even after the architecture stopped using it.

The paper shipped a ladder of checkpoints, listed in the official [THUDM/GLM README](https://github.com/THUDM/GLM):

| Variant | Params | Language | Pretraining data | Objective |
| --- | --- | --- | --- | --- |
| GLM-Base | 110M | English | Wiki + Book | Token blanks |
| GLM-Large | 335M | English | Wiki + Book | Token blanks |
| GLM-Large-Chinese | 335M | Chinese | WuDaoCorpora | Token + sentence + document |
| GLM-Doc | 335M | English | Wiki + Book | Token + document |
| GLM-410M | 410M | English | Wiki + Book | Token + document |
| GLM-515M | 515M | English | Wiki + Book | Token + document |
| GLM-RoBERTa | 335M | English | RoBERTa corpus | Token blanks |
| GLM-2B | 2B | English | The Pile | Token + sentence + document |
| GLM-10B | 10B | English | The Pile | Token + sentence + document |
| GLM-10B-Chinese | 10B | Chinese | WuDaoCorpora | Token + sentence + document |

The `config.json` for `glm-10b` declares `max_sequence_length: 1024`. These are research checkpoints: encoder-friendly, short context, evaluated on SuperGLUE and summarization. Nobody deploys them today, but they established two habits that last the whole family: bilingual Chinese and English from day one, and releasing a full size ladder instead of one hero model.

## 2. GLM-130B (August 2022): the open 100B-scale bet

**Released:** August 4, 2022, on the Tsinghua KEG project page ([keg.cs.tsinghua.edu.cn/glm-130b](https://keg.cs.tsinghua.edu.cn/glm-130b/)); INT4 quantized weights August 24, 2022; paper October 2022 ([2210.02414](https://arxiv.org/abs/2210.02414)), accepted at ICLR 2023. **Repo:** [THUDM/GLM-130B](https://github.com/THUDM/GLM-130B).

What changed versus GLM: scale and intent. GLM-130B is a 130B dense bilingual model, pretrained on over 400 billion tokens split evenly between Chinese and English, trained on 96 DGX-A100 nodes (768 A100 40GB GPUs) between May 6 and July 3, 2022. It kept the GLM blank-infilling objective, mixed with a small multi-task instruction component. Its fame, though, is engineering: the LayerNorm variant bake-off (it settled on DeepNorm), embedding-gradient shrink to survive loss spikes, a deliberate FP16 choice over BF16 so it could run on more hardware, and INT4 weight quantization with almost no quality loss, which let the 130B model run on 4x RTX 3090 (24GB) or 8x RTX 2080 Ti (11GB). Every one of those decisions is unpacked in [Engineering GLM-130B](/blog/machine-learning/frontier-models/engineering-glm-130b).

| Variant | Params | Context | Data | Weights | Notes |
| --- | --- | --- | --- | --- | --- |
| GLM-130B | 130B dense | 2,048 | 400B+ tokens (zh/en 50:50) | Download after application | FP16 inference on 8x A100 40GB |
| GLM-130B INT4 | 130B (4-bit weights) | 2,048 | Same | Same | Runs on 4x RTX 3090 24GB |

GLM-130B was never a chat model. It was the base on which Zhipu built its first ChatGLM product (the hosted, closed "ChatGLM" of March 2023), and the proof that a Chinese academic lab could train and release a model in the GPT-3 size class.

## 3. ChatGLM-6B (March 2023): the model that made the name famous

**Released:** March 14, 2023 ([GLM-130B README news](https://github.com/THUDM/GLM-130B), [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)). The same day, Zhipu opened the hosted ChatGLM (built on GLM-130B) for invited testing. A v1.1 checkpoint followed on May 15, 2023, adding English instruction data to stop Chinese words leaking into English answers.

What changed versus GLM-130B: everything about the audience. ChatGLM-6B is a 6.2B dense GLM-architecture model trained on about 1T tokens of Chinese and English, then aligned with supervised fine-tuning, feedback bootstrap and RLHF, in the README's own words "technology similar to ChatGPT". With INT4 quantization it ran in 6GB of GPU memory, and it shipped a P-Tuning v2 recipe for cheap fine-tuning. That combination (useful Chinese chat, consumer-GPU footprint, a fine-tuning script) is why it became one of the most downloaded open models of 2023.

| Variant | Params | Context | Layers | Weights | Footprint |
| --- | --- | --- | --- | --- | --- |
| chatglm-6b | 6.2B | 2,048 | 28 | Yes | ~13GB FP16, ~6GB INT4 |
| chatglm-6b-int4 / int8 | 6.2B quantized | 2,048 | 28 | Yes | Consumer GPU |

**The first multimodal branch.** On May 17, 2023, the same team released [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B), ChatGLM-6B plus a visual bridge trained as a BLIP2-Qformer (7.8B parameters in total), the first image-understanding model to carry the GLM name. In the same window they shipped [WebGLM](https://github.com/THUDM/WebGLM) (June 14, 2023), a web-search-augmented QA system built on GLM. Both are side branches, but VisualGLM is the seed of the vision lane traced in the next figure.

## Three lines, one family

Before continuing chronologically, it helps to see the family's shape, because from mid-2023 on there are always at least three things shipping in parallel.

![GLM family as three lanes: the text flagship line, the vision line from VisualGLM through CogVLM and GLM-4V to GLM-4.6V and GLM-5V-Turbo, and standalone specialist models, with merge edges where vision models are built on text bases and the vision line rejoins the flagship in GLM-5.3-Flash](/imgs/blogs/glm-model-family-timeline-2.webp)

The text line is the spine. The vision line repeatedly borrows a text base (GLM-4-9B-0414 under GLM-4.1V-Thinking, GLM-4.5-Air under GLM-4.5V) and then, in August 2026, merges back in: GLM-5.3-Flash is the first GLM-5 model that is natively multimodal rather than a vision adapter on a text model. The specialist line (voice, edge, OCR, image generation) mostly stands alone. Keep these lanes in mind for the rest of the catalog.

| Lane | First release | Latest release | How it relates to the flagship |
| --- | --- | --- | --- |
| Text flagship | GLM (Mar 2021) | GLM-5.3 (Aug 2026) | The spine |
| Vision | VisualGLM-6B (May 2023) | GLM-5.3-Flash (Aug 2026) | Built on text bases, then merged |
| Specialist | GLM-4-Voice (Oct 2024) | GLM-OCR (Feb 2026) | Independent small models |

## 4. ChatGLM2-6B (June 2023): context and inference speed

**Released:** June 25, 2023 ([THUDM/ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)). ChatGLM2-6B-32K followed July 31, 2023. CodeGeeX2-6B, a code model continued from ChatGLM2-6B on 600B code tokens, shipped July 25, 2023.

What changed versus ChatGLM-6B, per the README: a fully upgraded base trained with GLM's hybrid objective plus human preference alignment; context extended from 2K to 32K for the base using FlashAttention, with 8K used during dialogue alignment; and Multi-Query Attention, which the README credits with 42% faster inference and lower memory. The `config.json` confirms `seq_length: 32768` and `multi_query_group_num: 2`, meaning grouped KV heads, a choice the family keeps all the way to GLM-4-9B. The headline benchmark gains the README reports over the first generation are MMLU +23%, C-Eval +33%, GSM8K +571% and BBH +60%.

| Variant | Params | Context | Attention | Weights | Notes |
| --- | --- | --- | --- | --- | --- |
| chatglm2-6b | 6B | 32K base (8K chat) | MQA, 2 KV groups | Yes | Free commercial use after registration |
| chatglm2-6b-32k | 6B | 32K | Same | Yes | Long-context fine-tune, Jul 31 |
| CodeGeeX2-6B | 6B | n/a | Same | Yes | Code continued pretraining, separate CodeGeeX family |

## 5. CogVLM and CogAgent (October to December 2023): the vision line gets serious

**Released:** CogVLM-17B on October 5, 2023 ([THUDM/CogVLM](https://github.com/THUDM/CogVLM), paper [2311.03079](https://arxiv.org/abs/2311.03079)); CogAgent on December 15, 2023 (paper [2312.08914](https://arxiv.org/abs/2312.08914), a CVPR 2024 highlight).

These are not "GLM" by name, and the first CogVLM used a Vicuna language backbone rather than a GLM one. They belong in this catalog because they come from the same Tsinghua and Zhipu team, they are the direct ancestors of GLM-4V, and the GLM-4.1V-Thinking model card explicitly names "CogVLM2 and the GLM-4V series" as its previous generation. CogVLM's idea was a *visual expert*: trainable QKV and FFN weights for image tokens inside every transformer layer, instead of a shallow adapter. CogAgent extended it to high-resolution GUI screenshots for computer-use agents.

| Variant | Params | Backbone | Date | Weights | Focus |
| --- | --- | --- | --- | --- | --- |
| CogVLM-17B | 17B (10B vision + 7B language) | Vicuna-7B | Oct 5, 2023 | Yes | Visual expert per layer |
| CogAgent-18B | 18B | CogVLM | Dec 15, 2023 | Yes | 1120x1120 GUI understanding |
| CogVLM2 (Llama3-chat-19B) | 19B | Llama 3 8B | May 20, 2024 | Yes | Next generation, Chinese and English |
| CogVLM2-Video | 19B | Llama 3 8B | ~Jul 2024 | Yes | Video understanding |
| cogagent-9b-20241220 | 9B | GLM-4V-9B | ~Dec 20, 2024 | Yes | GUI agent on a GLM base |

The last row is the tell: by December 2024 the agent model had moved onto a GLM-4V base. (The image and video generators CogView and CogVideoX are a separate lineage; we cover CogVideoX in [The Open Video Frontier](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox).)

## 6. ChatGLM3-6B (October 2023): tools enter the prompt format

**Released:** October 27, 2023 ([THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)); the Hugging Face repos were created October 25 to 26. ChatGLM3-6B-128K followed around January 30, 2024.

What changed versus ChatGLM2: a stronger base (ChatGLM3-6B-Base, released separately for the first time) and, more importantly, a new prompt format that natively supports function calling, a code interpreter, and agent tasks. That made ChatGLM3 the first open GLM you could wire into a tool loop without prompt hacking. The default chat model declares `seq_length: 8192`, with 32K and 128K long-context variants. How tool use was taught, and how it evolved into GLM-4 All Tools, is in [From ChatGLM to GLM-4 All Tools](/blog/machine-learning/frontier-models/chatglm-to-glm-4-all-tools).

| Variant | Params | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| chatglm3-6b-base | 6B | 8K | Base | Yes |
| chatglm3-6b | 6B | 8K | Chat with tools | Yes |
| chatglm3-6b-32k | 6B | 32K | Long-context chat | Yes |
| chatglm3-6b-128k | 6B | 128K | Long-context chat | Yes (~Jan 2024) |

ChatGLM3 is also the last release to carry the "ChatGLM" prefix on open weights. From GLM-4 on, the product name and model name converge on "GLM".

## 7. GLM-4 (January to October 2024): the platform generation

GLM-4 is the release where Zhipu stops being a lab that ships checkpoints and becomes a platform that ships an API, an agent product, and open small models on a schedule.

### GLM-4 on the API (January 16, 2024)

**Released:** January 16, 2024, at Zhipu's first DevDay in Beijing; the [ChatGLM technical report](https://arxiv.org/abs/2406.12793) refers to the model as GLM-4 (0116). Size undisclosed; API and chatglm.cn only.

What changed versus ChatGLM3: a 128K context, a claimed step toward GPT-4-class quality on English and Chinese benchmarks, and **GLM-4 All Tools**, an aligned mode that decides on its own when to call a browser, a Python interpreter, or a text-to-image model, and lets users build their own agents ("GLMs"). The technical report (arXiv v1 June 18, 2024) is the main primary source for this generation's data and recipe: pretraining on about ten trillion tokens, mostly Chinese and English, plus SFT and RLHF stages and the LongAlign long-context recipe. Zhipu's own evaluation suites (AgentBench, AlignBench, LongBench, NaturalCodeBench) were built in this era, and we explain why in [How Zhipu Measures Its Models](/blog/machine-learning/frontier-models/how-zhipu-measures-glm-benchmarks).

### GLM-4-9B open weights (June 5, 2024)

**Released:** June 5, 2024 ([zai-org/GLM-4 README news](https://github.com/zai-org/GLM-4)). The open members of GLM-4 are 9B models, trained with the same recipe at smaller scale.

| Variant | Params (safetensors) | Context | Modality | Weights | License |
| --- | --- | --- | --- | --- | --- |
| glm-4-9b | 9.4B | 8K | Text, base | Yes | GLM-4 license |
| glm-4-9b-chat | 9.4B | 128K | Text, chat with tools | Yes | GLM-4 license |
| glm-4-9b-chat-1m | 9.4B | 1M | Text, chat | Yes | GLM-4 license |
| glm-4v-9b | 13.9B | 8K | Image + text | Yes | GLM-4 license |

Two details matter for later. The 1M variant is the family's first million-token checkpoint, two years before the flagship gets there (figure 3 below makes that outlier visible). And `glm-4v-9b` processes images at 1120x1120, folding CogAgent's high-resolution lesson into the GLM name.

### The GLM-4 long tail (August to December 2024)

The rest of 2024 is specialists on the GLM-4-9B base, each with its own repo and most with a paper:

| Release | Date | Params | What it is | Source |
| --- | --- | --- | --- | --- |
| LongWriter-glm4-9b | ~Aug 2024 | 9B | 10,000-word-plus generation | [HF](https://huggingface.co/zai-org/LongWriter-glm4-9b) |
| LongCite-glm4-9b | ~Sep 2024 | 9B | Sentence-level citations in long-context QA | [HF](https://huggingface.co/zai-org/LongCite-glm4-9b) |
| GLM-4-Voice | Oct 2024 | 9B + tokenizer + decoder | End-to-end Chinese and English spoken chat | [2412.02612](https://arxiv.org/abs/2412.02612) |
| GLM-Edge | ~Nov 2024 | 1.5B, 4B chat; 2B, 5B vision | On-device chat and VLM | [HF org](https://huggingface.co/zai-org) |
| MathGLM-Vision | ~Dec 2024 | 9B, 19B | Multimodal math | [HF](https://huggingface.co/zai-org/MathGLM-Vision-19B) |

GLM-4-Voice deserves a line of its own: it is a 9B model with a 12.5Hz speech tokenizer and a flow-matching decoder, released with all three components, which made it one of the first fully open end-to-end speech chat stacks.

## 8. GLM-4-0414 and GLM-Z1 (April 2025): MIT, 32B, and reasoning

**Released:** April 14, 2025 ([zai-org/GLM-4 README news](https://github.com/zai-org/GLM-4), [GLM-4-32B-0414 card](https://huggingface.co/zai-org/GLM-4-32B-0414)). Hugging Face repos were created April 7 to 13.

What changed versus GLM-4-9B, per the model card: the open line scales to 32B, the base is pretrained on 15T tokens "including substantial reasoning-type synthetic data", and, crucially, the license switches to **MIT**. The release also introduces the family's first dedicated reasoning models under a new "Z1" name: GLM-Z1-32B was trained from GLM-4-32B-0414 through cold start, extended RL on math, code and logic, and general RL from pairwise ranking feedback. GLM-Z1-Rumination is a deep-research model trained with end-to-end RL that interleaves search tool calls with long thinking.

| Variant | Params (safetensors) | Context | Type | Weights | License |
| --- | --- | --- | --- | --- | --- |
| GLM-4-32B-Base-0414 | 32.6B | 32K | Base | Yes | MIT |
| GLM-4-32B-0414 | 32.6B | 32K | Chat, tools | Yes | MIT |
| GLM-Z1-32B-0414 | 32.6B | 32K | Reasoning | Yes | MIT |
| GLM-Z1-Rumination-32B-0414 | 32.6B | 32K | Deep research with search | Yes | MIT |
| GLM-4-9B-0414 | 9B | 32K | Chat | Yes | MIT |
| GLM-Z1-9B-0414 | 9B | 32K | Reasoning | Yes | MIT |

The 32K figure is the `max_position_embeddings` in the shipped configs; the cards describe YaRN-based extension for longer inputs. The 0414 release is the bridge: dense like everything before it, MIT and RL-heavy like everything after.

## 9. GLM-4.1V-Thinking (July 2025): RL comes to the vision line

**Released:** July 1, 2025 ([zai-org/GLM-V news](https://github.com/zai-org/GLM-V); the GLM-4 repo announced it July 2). Paper: [2507.01006](https://arxiv.org/abs/2507.01006), later retitled to cover GLM-4.5V as well.

What changed versus GLM-4V-9B and CogVLM2: a new base (GLM-4-9B-0414 plus a ViT encoder), a think-then-answer output format, and reinforcement learning with curriculum sampling over verifiable rewards across many visual domains. The card claims it matches or beats Qwen2.5-VL-72B on 18 benchmarks at 9B. Context rises to 64K, and the encoder supports images up to 4K resolution and arbitrary aspect ratios. The full recipe is in [GLM-4.1V-Thinking and GLM-4.5V](/blog/machine-learning/frontier-models/glm-4-1v-thinking-multimodal-reasoning-rl).

| Variant | Params (safetensors) | Context | Type | Weights | License |
| --- | --- | --- | --- | --- | --- |
| GLM-4.1V-9B-Base | 10.3B | 64K | Vision base | Yes | MIT |
| GLM-4.1V-9B-Thinking | 10.3B | 64K | Vision reasoning | Yes | MIT |

Two weeks later (July 16) the team open-sourced the VLM reward system used to train it, which is the more reusable artifact for anyone building their own visual RL.

## 10. GLM-4.5 (July 2025): the jump to frontier MoE

**Released:** July 28, 2025 ([z.ai/blog/glm-4.5](https://z.ai/blog/glm-4.5), [GLM-4.5 card](https://huggingface.co/zai-org/GLM-4.5)); technical report [2508.06471](https://arxiv.org/abs/2508.06471) in August 2025.

This is the single biggest discontinuity in the catalog. Everything before it is dense and 32B or smaller in the open line; GLM-4.5 is a 355B mixture-of-experts with 32B active, and a 106B / 12B-active Air sibling, both under MIT, both shipped with base checkpoints and FP8 builds. Both are *hybrid reasoning* models with a thinking mode and a non-thinking mode in one checkpoint. The card reports a combined score of 63.2 across 12 benchmarks, which it ranks third among all models at the time.

What changed, in one table:

| Dimension | GLM-4-32B-0414 | GLM-4.5 | Where to read more |
| --- | --- | --- | --- |
| Architecture | Dense 32B | MoE, 160 routed experts, 8 active, 92 layers | [GLM-4.5 Architecture](/blog/machine-learning/frontier-models/glm-4-5-architecture) |
| Shape choice | Conventional | Deeper and narrower than DeepSeek-V3, 96 attention heads | Same |
| Routing | n/a | Loss-free balance routing, sigmoid gates | Same |
| Extras | n/a | QK-Norm, one MTP layer for speculative decoding | Same |
| Optimizer | AdamW | Muon | Same |
| Pretraining | 15T tokens | 23T tokens, multi-stage | [Training GLM-4.5](/blog/machine-learning/frontier-models/training-glm-4-5-distillation-agentic-rl) |
| Post-training | SFT + RL per model | Train domain experts, distill into one hybrid model, agentic RL | Same |
| RL infrastructure | n/a | slime (open source) | [slime deep-dive](/blog/machine-learning/open-source-library/slime-rl-framework-deep-dive) |
| Context | 32K | 128K | Card |
| License | MIT | MIT | Card |

| Variant | Total / active | Context | Layers | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| GLM-4.5-Base | 355B / 32B | 128K | 92 | Base | Yes, MIT |
| GLM-4.5 | 355B / 32B | 128K | 92 | Hybrid reasoning | Yes, MIT (+ FP8) |
| GLM-4.5-Air-Base | 106B / 12B | 128K | 46 | Base | Yes, MIT |
| GLM-4.5-Air | 106B / 12B | 128K | 46 | Hybrid reasoning | Yes, MIT (+ FP8) |

### GLM-4.5V (August 11, 2025)

**Released:** August 11, 2025 ([zai-org/GLM-V news](https://github.com/zai-org/GLM-V), [card](https://huggingface.co/zai-org/GLM-4.5V)). GLM-4.5V puts the GLM-4.1V-Thinking recipe on top of GLM-4.5-Air: 106B total, 12B active, 64K context, MIT. The card claims state of the art at its scale on 42 public vision-language benchmarks and adds GUI-agent operation and a thinking-mode switch. Its safetensors total is 107.7B including the vision encoder.

## 11. GLM-4.6 (September 2025): the 200K coding point release

**Released:** September 30, 2025 ([Z.ai release notes](https://docs.z.ai/release-notes/new-released), [card](https://huggingface.co/zai-org/GLM-4.6)). No new paper; the card points back to the GLM-4.5 report.

What changed versus GLM-4.5: the same architecture (the shipped config still has 92 layers, 160 routed experts, 8 active), a context window expanded from 128K to 200K (`max_position_embeddings: 202752`), better coding in real agent harnesses (the card names Claude Code, Cline, Roo Code and Kilo Code), better front-end generation, and tool use during reasoning. This is the first of three "same weights shape, better post-training" releases that set the rhythm for the GLM-5 era.

| Variant | Total / active | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| GLM-4.6 | 355B / 32B | 200K | Hybrid reasoning | Yes, MIT (+ FP8) |

There is no GLM-4.6-Air. The Air-sized slot was filled two months later by the vision model.

### GLM-4.6V and GLM-4.6V-Flash (December 2025)

**Released:** December 2025; the Hugging Face repos were created December 7 and third-party listings date the launch December 8, but I could not retrieve a primary page with the exact day ([card](https://huggingface.co/zai-org/GLM-4.6V), [blog](https://z.ai/blog/glm-4.6v)). Two sizes: GLM-4.6V (106B, built on the GLM-4.5-Air MoE shape) and GLM-4.6V-Flash (9B, dense). Context goes to 128K, and the headline feature is **native multimodal function calling**: images, screenshots and document pages pass directly as tool inputs and tool outputs, without converting to text. The card is candid that pure-text QA "still has significant room for improvement".

| Variant | Params (safetensors) | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| GLM-4.6V | 107.7B (MoE, Air shape) | 128K | Vision, agentic | Yes, MIT |
| GLM-4.6V-Flash | 10.3B dense | 128K | Vision, local | Yes, MIT |

December 2025 also brought a burst of specialists: AutoGLM-Phone-9B (a phone-operating agent), GLM-ASR-Nano-2512 (speech recognition) and GLM-TTS. GLM-Image, an autoregressive-plus-diffusion image generator, followed on January 14, 2026 per the release notes.

## 12. GLM-4.7 and GLM-4.7-Flash (December 2025 to January 2026)

**Released:** GLM-4.7 on December 22, 2025 ([release notes](https://docs.z.ai/release-notes/new-released), [card](https://huggingface.co/zai-org/GLM-4.7), [blog](https://z.ai/blog/glm-4.7)). GLM-4.7-Flash weights on Hugging Face January 19, 2026, announced January 20 in China ([card](https://huggingface.co/zai-org/GLM-4.7-Flash)).

What changed versus GLM-4.6: again the same MoE shape (358.3B safetensors total, 92 layers, 160 experts), so the gains are post-training. The card reports SWE-bench 73.8% (+5.8 over GLM-4.6), SWE-bench Multilingual 66.7% (+12.9), Terminal Bench 2.0 41% (+16.5) and HLE 42.8% (+12.4). The behavioral change is in the thinking controls: GLM-4.7 extends the interleaved thinking introduced in GLM-4.5 with **Preserved Thinking** (reasoning carried across turns) and **Turn-level Thinking** (enable or disable reasoning per turn within one session).

| Variant | Total / active | Context | Layers | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| GLM-4.7 | 355B / 32B | 200K | 92 | Hybrid reasoning | Yes, MIT (+ FP8) |
| GLM-4.7-Flash | 30B / 3B (31.2B safetensors) | 200K | 47 | Hybrid reasoning, lightweight | Yes, MIT |

GLM-4.7-Flash is a new shape, not a distillation of an existing one: a `glm4_moe_lite` model with 64 routed experts and 4 active per token, marketed as the strongest model in the 30B-A3B class. It matters beyond its own use: the IndexCache paper used a DSA-converted GLM-4.7-Flash as its 30B testbed before scaling the idea to GLM-5, which we cover in [IndexCache](/blog/paper-reading/large-language-model/indexcache-cross-layer-index-reuse).

## 13. GLM-OCR (February 2026): the specialist that tops a leaderboard

**Released:** February 3, 2026 (announced and open-sourced that day per Chinese press; the Hugging Face repo was created January 30). Technical report [2603.10910](https://arxiv.org/abs/2603.10910) on March 12, 2026 ([zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)).

GLM-OCR is a 0.9B document-parsing model on the GLM-V encoder-decoder design: a CogViT visual encoder with a small GLM decoder, trained with a multi-token-prediction loss and full-task RL, and paired in its SDK with PP-DocLayoutV3 for layout analysis (Apache 2.0, the model itself is MIT). The card reports 1.86 PDF pages per second at single concurrency. The training recipe, including what is load-bearing and what is not, is in [GLM-OCR's Training Recipe](/blog/machine-learning/frontier-models/glm-ocr-training-recipe).

| Variant | Params | Context | Modality | Weights | License |
| --- | --- | --- | --- | --- | --- |
| GLM-OCR | 0.9B (1.3B safetensors incl. encoder) | 128K | Document image to text | Yes | MIT (layout model Apache 2.0) |

## 14. GLM-5 (February 2026): double the size, sparse attention

**Released:** February 12, 2026 per the [Z.ai release notes](https://docs.z.ai/release-notes/new-released); the Hugging Face repo is dated February 11 and most coverage uses February 11. Technical report: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763). Card: [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5).

What changed versus GLM-4.7, quoting the card: total parameters go from 355B (32B active) to **744B (40B active)**, pretraining data from 23T to **28.5T tokens**, and the attention integrates **DeepSeek Sparse Attention (DSA)** to cut long-context serving cost (our background on DSA is in [NSA vs DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa)). The shipped config confirms a new `glm_moe_dsa` model type: 78 layers (fewer and wider than GLM-4.5's 92), hidden size 6144, 256 routed experts with 8 active, and a 200K window. Post-training leans on slime, now described as asynchronous RL infrastructure, for "more fine-grained post-training iterations". The positioning moves from coding to "complex systems engineering and long-horizon agentic tasks".

| Dimension | GLM-4.7 | GLM-5 |
| --- | --- | --- |
| Total / active | 355B / 32B | 744B / 40B |
| Layers / hidden | 92 / 5120 | 78 / 6144 |
| Routed experts (active) | 160 (8) | 256 (8) |
| Attention | Full attention, GQA | DSA sparse attention |
| Pretraining tokens | 23T (GLM-4.5 base) | 28.5T |
| Context | 200K | 200K |
| BF16 weights on disk | ~717GB | ~1.5TB |
| License | MIT | MIT |

| Variant | Total / active | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| GLM-5 | 744B / 40B (753.9B safetensors) | 200K | Hybrid reasoning, text only | Yes, MIT (+ FP8) |
| GLM-5-Turbo | Undisclosed | 200K, 128K output | Speed-optimized variant | No, API only (Mar 2026) |

The BF16 footprint row is just 2 bytes per parameter on the safetensors totals, which is the number to plan GPU memory around. GLM-5-Turbo has a page in the [Z.ai docs](https://docs.z.ai/guides/llm/glm-5-turbo) with 200K context and 128K max output, but I could not find a primary source for its exact launch day; third-party gateways list March 15, 2026.

## 15. GLM-5V-Turbo and GLM-5.1 (April 2026)

### GLM-5V-Turbo (April 2, 2026)

**Released:** April 2, 2026 ([zai-org/GLM-V news](https://github.com/zai-org/GLM-V), [docs](https://docs.z.ai/guides/vlm/glm-5v-turbo)). API only, size undisclosed, 200K context. The docs describe it as combining vision and coding "with a smaller parameter size". It is the vision line's first GLM-5-generation model, and the first time the vision line ships without open weights.

### GLM-5.1 (April 7, 2026)

**Released:** April 7, 2026 per the [release notes](https://docs.z.ai/release-notes/new-released); weights on Hugging Face around April 8 ([card](https://huggingface.co/zai-org/GLM-5.1)). No new paper; the card points to the GLM-5 report.

What changed versus GLM-5: the same base architecture (identical config and 753.9B safetensors total), so this is a post-training release aimed at *staying power*. The card's framing is unusually specific: earlier models "exhaust their repertoire early" and plateau, while GLM-5.1 "sustains optimization over hundreds of rounds and thousands of tool calls". The release notes claim it can work independently for up to 8 hours in a single run. The card reports state of the art on SWE-Bench Pro and large gains over GLM-5 on NL2Repo and Terminal-Bench 2.0. Coverage at launch also noted Zhipu raised API prices alongside it, a first for a family that had competed mostly on price.

| Variant | Total / active | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| GLM-5.1 | 744B / 40B | 200K | Hybrid reasoning, long-horizon agentic | Yes, MIT (+ FP8) |

## 16. GLM-5.2 (June 2026): the 1M-token flagship

**Released:** June 16, 2026 ([release notes](https://docs.z.ai/release-notes/new-released), [card](https://huggingface.co/zai-org/GLM-5.2)).

What changed versus GLM-5.1, from the card: a "solid 1M-token context" (the config's `max_position_embeddings` goes from 202,752 to 1,048,576), and two architecture-level serving changes. **IndexShare** reuses the same sparse-attention indexer across every four sparse attention layers, cutting per-token FLOPs 2.9x at 1M context; the card links it to the IndexCache paper ([2603.12201](https://arxiv.org/abs/2603.12201)). And the MTP layer is improved so that speculative decoding acceptance length rises by up to 20%. The card also leans on its license line: "An MIT open-source license, no regional limits". Our teardown, which covers IndexShare, the layer-split serving topology and the RL stack in detail, is [GLM-5.2, Taken Apart](/blog/machine-learning/frontier-models/glm-5-2-architecture-techniques-evals).

| Variant | Total / active | Context | Type | Weights |
| --- | --- | --- | --- | --- |
| GLM-5.2 | 744B / 40B (753.3B safetensors) | 1M | Hybrid reasoning, long-horizon | Yes, MIT (+ FP8) |

Benchmarks in the card are evaluated at 400K context for SWE-Bench Pro, NL2Repo and DeepSWE, and at 1M for third-party long-horizon suites, which is itself a sign of where the target moved.

## 17. GLM-5.3 and GLM-5.3-Flash (August 2026): the split

August 2026 is the most interesting month in the catalog, because the family forks. One release keeps the GLM-5 base and changes the license; the other keeps MIT and changes the architecture.

### GLM-5.3 (August 18, 2026; weights August 28)

**Released:** August 18, 2026 per the [release notes](https://docs.z.ai/release-notes/new-released). Coverage reports a hosted launch through Z.ai's coding plan around August 14 and open weights about two weeks later, on August 28 US time. Card: [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3); docs: [docs.z.ai/guides/llm/glm-5.3](https://docs.z.ai/guides/llm/glm-5.3).

What changed versus GLM-5.2: the card is explicit that GLM-5.3 "uses the same base model as GLM-5.2, every gain comes from post-training". It claims a 50% improvement over GLM-5.2 on Z.ai's in-house Code Bench, open-source state of the art on Terminal Bench 3.0 and Agents' Last Exam, and an emergent cybersecurity capability: state of the art on CyberGym, with gains largest further up the exploitation chain. The docs list text-only input, a 1M context, 128K max output, and note that disabling reasoning is no longer supported.

The license is the real news. GLM-5.3 ships under a new **GLM-5.3 License**: MIT-style permissions, plus one condition. If the licensee operates a "Model as a Service" business and its group revenue exceeds 10 billion US dollars over any 12 months, it must pass a Z.AI security review before commercial use. For almost every reader this is MIT in practice. For hyperscalers it is not, and it is the first time since GLM-4-0414 that a flagship GLM is not under plain MIT. It reads as a direct response to the cyber capability the same card advertises.

| Variant | Total / active | Context | Type | Weights | License |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3 | 744B / 40B (753.3B safetensors) | 1M, 128K output | Reasoning, text only | Yes (+ BF16 repo) | GLM-5.3 License |

### GLM-5.3-Flash (August 26, 2026)

**Released:** August 26, 2026 ([release notes](https://docs.z.ai/release-notes/new-released), [zai-org/GLM-V news](https://github.com/zai-org/GLM-V), [card](https://huggingface.co/zai-org/GLM-5.3-Flash), [blog](https://z.ai/blog/glm-5.3-flash)).

Despite the name, this is not a smaller GLM-5.3. The card says it "starts from a newly trained base model, with its architecture and training recipe redesigned". It is **320B total, 18B active**, and it is the first *natively multimodal* model in the GLM-5 series. Three firsts for the family: a **hybrid attention** stack combining sparse and linear attention (to cut long-context compute and KV cache), **Manifold-Constrained Hyper-Connections (mHC)** for scaling efficiency, and a new 30T-token multimodal pretraining corpus. The config is a new `glm5_next_text` type with 45 layers, hidden size 4096, 288 routed experts and 8 active, and a 1M window. The card claims it beats GLM-5.2 across benchmarks at one tenth the price. It ships under MIT.

| Variant | Total / active | Context | Modality | Weights | License |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 320B / 18B (321.3B safetensors) | 1M | Native image + text | Yes (+ BF16 repo) | MIT |

On September 2, 2026, the GLM-V repository announced it would no longer be maintained and pointed GLM-5.3-Flash discussion to the main GLM-5 repo: the vision lane has formally merged into the flagship. As of October 14, 2026, GLM-5.3 and GLM-5.3-Flash are the latest GLM releases; the release notes list nothing after August 26.

## How the family evolved

Read the 28 rows end to end and five through-lines account for almost everything. (For the *techniques* behind each, the [lineage post](/blog/machine-learning/frontier-models/glm-lineage-frontier-llm-technique) is the companion.)

### 1. The backbone changed three times, and only one change was gradual

| Era | Backbone | Releases |
| --- | --- | --- |
| 2021 to 2022 | GLM blank-infilling transformer | GLM, GLM-130B |
| 2023 to Apr 2025 | Dense decoder-style GLM with MQA/GQA | ChatGLM-6B to GLM-4-32B-0414 |
| Jul 2025 to Jan 2026 | Deep-narrow MoE, 160 experts, full attention | GLM-4.5, 4.6, 4.7 |
| Feb to Aug 2026 | Wider MoE, 256 experts, DSA sparse attention | GLM-5, 5.1, 5.2, 5.3 |
| Aug 2026 | Hybrid sparse + linear attention MoE, mHC, native multimodal | GLM-5.3-Flash |

The first transition (blank infilling to dense chat models) was gradual and happened inside one product line. The second (dense to MoE) was abrupt: GLM-4.5 arrives fully formed with no small MoE preview. The third and fourth happen within six months of each other, and GLM-5.3-Flash looks like the preview of whatever comes next.

### 2. Sparsity went from 100% active to under 6% active

GLM-4.5 activates 32B of 355B (about 9%), GLM-4.5-Air 12B of 106B (11%), GLM-5 40B of 744B (5.4%), GLM-4.7-Flash 3B of 30B (10%), and GLM-5.3-Flash 18B of 320B (5.6%). The flagship's active count grew only from 32B to 40B while its total doubled. The family is buying capacity with experts and holding per-token cost nearly flat, which is the same trade the rest of the open frontier made.

### 3. Context grew 512x, in two big jumps

![Flagship GLM context window on a log scale from 2K in March 2023 to 1M in June 2026, with jumps at GLM-4 and GLM-5.2, an 18-month hold at 128K, and the GLM-4-9B-Chat-1M variant as an early outlier](/imgs/blogs/glm-model-family-timeline-3.webp)

The flagship line goes 2K (ChatGLM-6B) to 32K (ChatGLM2 base) to 8K (ChatGLM3's default chat) to 128K (GLM-4) to 200K (GLM-4.6) to 1M (GLM-5.2). From 2K to 1M is a factor of 512. Two observations from the chart. The flagship held 128K for 18 months, from GLM-4 in January 2024 to GLM-4.5 in July 2025, while a 9B variant already offered 1M in June 2024: the team knew how to train long context early but did not make it the flagship default until sparse attention (DSA plus IndexShare) made it affordable to *serve*. And ChatGLM3's 8K default is a step down from ChatGLM2's 32K base, a reminder that "context" on a card can mean the pretraining window, the alignment window or the serving default.

### 4. Post-training moved from preference tuning to long-horizon agent RL

ChatGLM-6B: SFT plus RLHF. ChatGLM3: tool-calling prompt format. GLM-4: All Tools and AgentTuning. GLM-Z1: cold start plus RL on verifiable math and code. GLM-4.1V: RL with curriculum sampling across visual domains. GLM-4.5: expert models distilled into one hybrid, then agentic RL on slime. GLM-5.1 to GLM-5.3: post-training only releases whose stated goal is staying productive over hours and thousands of tool calls. Four of the last six flagship releases (GLM-4.6, 4.7, 5.1 and 5.3) kept the previous base architecture and changed only post-training, which tells you where the team thinks the marginal gain lives.

### 5. Openness liberalized for four years, then took one hedged step back

| Period | License |
| --- | --- |
| 2022 | GLM-130B: download on application, research license |
| 2023 to 2024 | ChatGLM and GLM-4-9B: custom model licenses, commercial use after registration |
| Apr 2025 to Jun 2026 | MIT for every open flagship and nearly every specialist |
| Aug 2026 | GLM-5.3: MIT plus a security-review clause for 10B-dollar MaaS operators; GLM-5.3-Flash stays MIT |

The API-only releases (GLM-4 in 2024, GLM-5-Turbo and GLM-5V-Turbo in 2026) never replaced open ones; they sat alongside them. The GLM-5.3 clause is narrow, but it is the first time the family has written a capability concern into its license.

## Which variant to pick today

As of October 2026, with the trade-offs that matter:

| Need | Pick | Why | Watch out for |
| --- | --- | --- | --- |
| Best open model for coding and long agent runs | GLM-5.3 | Same base as 5.2, strongest post-training, 1M context | GLM-5.3 License if you run a large MaaS business; ~1.5TB BF16 weights |
| Same capability, plain MIT | GLM-5.2 | Last flagship under unrestricted MIT, 1M context | Weaker coding than 5.3 by Z.ai's numbers |
| Multimodal agent, or the best price per token | GLM-5.3-Flash | 18B active, native vision, hybrid attention for cheap long context | New architecture: check your serving engine supports `glm5_next_text` |
| Single-node deployment, text | GLM-4.7 or GLM-4.5-Air | 355B / 32B fits one 8-GPU node in FP8; Air is 106B / 12B | 200K and 128K context respectively |
| Laptop or single GPU | GLM-4.7-Flash | 30B total, 3B active, 200K context, MIT | Smaller knowledge capacity |
| Document OCR and parsing | GLM-OCR | 0.9B, MIT, strong document-parsing results | Pair with the layout model for full pages |
| Vision on modest hardware | GLM-4.6V-Flash | 9B dense, 128K, native multimodal tool calls | Card admits weaker pure-text QA |
| Research on the original objective | GLM-10B / GLM-130B | The blank-infilling models | Short context, dated quality |

If you need one rule: start with GLM-5.3-Flash unless you specifically need the strongest text-only coding agent (GLM-5.3) or must stay under plain MIT with flagship text capability (GLM-5.2). For serving details of the 1M-context flagships, the [GLM-5.2 teardown](/blog/machine-learning/frontier-models/glm-5-2-architecture-techniques-evals) covers the memory math. For how this family compares with its neighbors, see the sibling catalogs for [Qwen](/blog/machine-learning/frontier-models/qwen-model-family-timeline) and [Kimi](/blog/machine-learning/frontier-models/kimi-model-family-timeline).

## What this catalog could not verify

Honesty about the gaps, so nobody quotes a guess as a fact:

- **GLM-4 (January 2024) size and architecture.** Never disclosed; the technical report describes data and recipe but not parameter count.
- **GLM-4.6V exact launch day.** Hugging Face repos are dated December 7, 2025, third-party listings say December 8; the official blog page did not render for retrieval.
- **GLM-5-Turbo launch day and size.** The docs page exists (200K context, 128K output), but no primary source gives the day; March 15, 2026 appears only on third-party gateways. GLM-5V-Turbo's size is also undisclosed.
- **GLM-5 day.** Z.ai's release notes say February 12, 2026; the Hugging Face repo and most coverage say February 11. Both are plausibly correct across time zones.
- **GLM-5.3 launch versus weights.** The release notes say August 18; coverage reports a hosted launch around August 14 and weights on August 28. The table uses the release-notes date.
- **GLM-Edge and the 2024 long tail.** Dates for GLM-Edge, LongWriter, LongCite and MathGLM-Vision are Hugging Face repo-creation dates (marked "~"), not announcements.
- **Context for CogVLM and CogAgent** is not stated in a comparable way on their cards, so the summary table leaves it as n/a.

## References

Papers and technical reports:

1. Du et al., "GLM: General Language Model Pretraining with Autoregressive Blank Infilling", ACL 2022. [arXiv:2103.10360](https://arxiv.org/abs/2103.10360)
2. Zeng et al., "GLM-130B: An Open Bilingual Pre-trained Model", ICLR 2023. [arXiv:2210.02414](https://arxiv.org/abs/2210.02414)
3. Wang et al., "CogVLM: Visual Expert for Pretrained Language Models". [arXiv:2311.03079](https://arxiv.org/abs/2311.03079)
4. Hong et al., "CogAgent: A Visual Language Model for GUI Agents". [arXiv:2312.08914](https://arxiv.org/abs/2312.08914)
5. Team GLM, "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools". [arXiv:2406.12793](https://arxiv.org/abs/2406.12793)
6. Zeng et al., "GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot". [arXiv:2412.02612](https://arxiv.org/abs/2412.02612)
7. V Team, "GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning". [arXiv:2507.01006](https://arxiv.org/abs/2507.01006)
8. GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models". [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)
9. "GLM-5: from Vibe Coding to Agentic Engineering". [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
10. "GLM-OCR Technical Report". [arXiv:2603.10910](https://arxiv.org/abs/2603.10910)
11. "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" (IndexShare in GLM-5.2). [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)

Official pages, repositories and model cards:

12. GLM-130B project page, Tsinghua KEG. [keg.cs.tsinghua.edu.cn/glm-130b](https://keg.cs.tsinghua.edu.cn/glm-130b/)
13. GitHub: [THUDM/GLM](https://github.com/THUDM/GLM), [THUDM/GLM-130B](https://github.com/THUDM/GLM-130B), [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [THUDM/ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B), [THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3), [THUDM/CogVLM](https://github.com/THUDM/CogVLM), [zai-org/GLM-4](https://github.com/zai-org/GLM-4), [zai-org/GLM-V](https://github.com/zai-org/GLM-V), [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)
14. Z.ai developer docs release notes. [docs.z.ai/release-notes/new-released](https://docs.z.ai/release-notes/new-released)
15. Z.ai docs model pages: [GLM-4.5](https://docs.z.ai/guides/llm/glm-4.5), [GLM-4.6V](https://docs.z.ai/guides/vlm/glm-4.6v), [GLM-5-Turbo](https://docs.z.ai/guides/llm/glm-5-turbo), [GLM-5V-Turbo](https://docs.z.ai/guides/vlm/glm-5v-turbo), [GLM-5.1](https://docs.z.ai/guides/llm/glm-5.1), [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3)
16. Z.ai blog: [GLM-4.5](https://z.ai/blog/glm-4.5), [GLM-4.6V](https://z.ai/blog/glm-4.6v), [GLM-4.7](https://z.ai/blog/glm-4.7), [GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)
17. Hugging Face model cards (zai-org, formerly THUDM): [chatglm-6b](https://huggingface.co/zai-org/chatglm-6b), [chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b), [chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b), [glm-4-9b-chat](https://huggingface.co/zai-org/glm-4-9b-chat), [glm-4v-9b](https://huggingface.co/zai-org/glm-4v-9b), [GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414), [GLM-Z1-32B-0414](https://huggingface.co/zai-org/GLM-Z1-32B-0414), [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking), [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5), [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air), [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V), [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6), [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V), [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7), [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash), [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), [GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
