---
title: "The Llama Model Family Timeline: Every Release From LLaMA to Llama 4"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference to every Meta Llama release, from the research-only LLaMA of February 2023 through Llama 2, Code Llama, Llama Guard, Llama 3, 3.1, 3.2, 3.3 and the Llama 4 mixture-of-experts models, with what changed each time, the variant tables, primary sources, and what Meta shipped after Llama."
tags: ["llama", "meta", "large-language-model", "open-weights", "model-timeline", "mixture-of-experts", "llama-guard", "code-llama", "frontier-models"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 28
---

If you started working with open language models any time between 2023 and 2025, there is a good chance the first model you fine-tuned, quantized, or served was a Llama. For about two years the family was the default substrate of open-weight work: the architecture other labs copied, the checkpoint that llama.cpp was named after, the base under hundreds of community fine-tunes, and the reference point every new open model was benchmarked against. Then, in the space of a year, it went quiet. Llama 4 landed in April 2025 to a mixed reception, its largest model never shipped, and by April 2026 Meta's new frontier work was coming out under a different name entirely.

This post is a reference catalog of that arc. It walks every Llama generation in the order Meta released it, from the research-only LLaMA of February 2023 to Llama 4 Scout and Maverick in April 2025 and the safety releases that followed at LlamaCon. For each generation you get the release date, what changed versus the one before (architecture, data, context length, post-training recipe, license), a table of the variants, and a primary-source link. It closes with the through-lines that connect the generations, a short guide to which variant still makes sense today, and a brief note on what Meta shipped after it stopped shipping Llamas. The figure below is the whole story at one glance; the rest of the post is the detail behind each mark on it.

A note on sourcing before we start. Every date and specification here was checked against a Meta technical report, an official Meta blog post, or an official model card on Hugging Face or GitHub. Where a number is not stated in a primary source, the table says so rather than guessing. Llama versions that people speculated about but that do not exist (there is no Llama 4.1, and as of 14 October 2026 there is no Llama 5) are not in this catalog.

## The timeline at a glance

![Release timeline of the Meta Llama family from LLaMA in February 2023 to Llama 4 in April 2025, with the post-Llama Muse models on a separate track](/imgs/blogs/llama-model-family-timeline-1.webp)

Read the figure left to right and three phases stand out. The first, from February 2023 to December 2023, is the expansion phase: a research release, then a commercially licensed release with a chat model, then a code specialization, then a safety classifier. The second, from April 2024 to December 2024, is the consolidation phase: one architecture (dense decoder with grouped-query attention and a 128K vocabulary) stretched across sizes from 1B to 405B, a context window grown from 8K to 128K, vision bolted on through an adapter, and a final 70B refresh. The third phase is a single release, Llama 4 in April 2025, which changed almost everything at once (mixture of experts, native multimodality, a 10M-token context claim) and turned out to be the end of the line under the Llama name.

Here is the same story as a table, so you can scan dates without the figure.

| Release | Date | Headline sizes | Context | Weights |
| --- | --- | --- | --- | --- |
| LLaMA | 24 Feb 2023 | 7B, 13B, 33B, 65B | 2K | Research license, case by case |
| Llama 2 and Llama 2-Chat | 18 Jul 2023 | 7B, 13B, 70B | 4K | Open, Llama 2 Community License |
| Code Llama | 24 Aug 2023 (70B on 29 Jan 2024) | 7B, 13B, 34B, 70B | 16K trained, stable to 100K | Open, Llama 2 license |
| Llama Guard | 7 Dec 2023 | 7B | Not the point | Open |
| Llama 3 | 18 Apr 2024 | 8B, 70B | 8K | Open, Llama 3 Community License |
| Llama 3.1 | 23 Jul 2024 | 8B, 70B, 405B | 128K | Open, Llama 3.1 Community License |
| Llama 3.2 | 25 Sep 2024 | 1B, 3B, 11B Vision, 90B Vision | 128K | Open, Llama 3.2 Community License |
| Llama 3.3 | 6 Dec 2024 | 70B | 128K | Open, Llama 3.3 Community License |
| Llama 4 Scout, Maverick | 5 Apr 2025 | 109B total / 17B active, 400B total / 17B active | 10M, 1M | Open, Llama 4 Community License |
| Llama 4 Behemoth | Previewed 5 Apr 2025 | About 2T total / 288B active | Not stated | Never released |
| LlamaCon safety drop | 29 Apr 2025 | Llama Guard 4 12B, Prompt Guard 2 (22M, 86M) | Not the point | Open |

## LLaMA (February 2023): the research release that escaped

**Released:** 24 February 2023. **Primary source:** [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971), plus Meta's [launch post](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).

The first LLaMA (the capitalization changed with the second generation) was a research artifact with a specific argument attached. The Chinchilla scaling work had shown that for a fixed training budget you should train a smaller model on more tokens than people had been doing. The LLaMA paper pushed the argument one step further: if what you care about is the cost of *running* the model, you should train a small model far past the compute-optimal point, because inference cost is paid on every query forever while training cost is paid once. So LLaMA trained a 7B model on 1.0 trillion tokens and a 65B model on 1.4 trillion, all drawn from publicly available data (CommonCrawl, C4, GitHub, Wikipedia, books, arXiv, StackExchange). The headline result was that the 13B model beat the 175B GPT-3 on most of the paper's benchmarks.

The architecture was a decoder-only transformer with three changes from the original GPT recipe, each borrowed from an earlier model, and each of which became the default for open models for years afterwards:

- **Pre-normalization with RMSNorm**: normalize the input of each sublayer rather than the output, using RMSNorm rather than LayerNorm. This is the GPT-3 style of normalization, made cheaper. (If you want the long version of why this matters, see [layer normalization inside out](/blog/machine-learning/deep-learning/layer-normalization-inside-out).)
- **SwiGLU activations** in the feed-forward block, from PaLM.
- **Rotary position embeddings (RoPE)** instead of absolute positions, from GPT-NeoX.

The tokenizer was SentencePiece BPE with a 32K vocabulary, and the context window was 2,048 tokens. There was no instruction-tuned model; LLaMA was base weights only.

| Variant | Params | Training tokens | Context | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| LLaMA-7B | 6.7B | 1.0T | 2K | Base | Research license |
| LLaMA-13B | 13.0B | 1.0T | 2K | Base | Research license |
| LLaMA-33B | 32.5B | 1.4T | 2K | Base | Research license |
| LLaMA-65B | 65.2B | 1.4T | 2K | Base | Research license |

The license is the part of the story that changed the field. Access was granted to researchers case by case under a non-commercial license. Within about a week the weights had leaked publicly, and the community did what communities do: Alpaca, Vicuna, and a wave of instruction-tuned derivatives appeared within weeks, and Georgi Gerganov's llama.cpp made it possible to run the models on a laptop CPU. (Our guide to [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/inference-frameworks/running-llms-locally-llama-cpp-and-gguf) is a direct descendant of that moment.) The lesson Meta drew was that openness was going to happen anyway, and it might as well happen on Meta's terms.

## Llama 2 and Llama 2-Chat (July 2023): open for business

**Released:** 18 July 2023. **Primary source:** [Llama 2: Open Foundation and Fine-Tuned Chat Models (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288), Meta's [announcement](https://about.fb.com/news/2023/07/llama-2/), and the [model card](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md).

Llama 2 is the release that made the family the default. The architectural changes from LLaMA were modest. The real changes were three non-architectural ones: a commercial license, a properly aligned chat model, and an unusually detailed paper about how that chat model was made.

**What changed versus LLaMA:**

- **Data and context.** Pretraining grew to 2 trillion tokens, about 40% more than LLaMA, and the context window doubled to 4,096 tokens.
- **Grouped-query attention.** The 34B and 70B models used grouped-query attention (GQA), where several query heads share one key/value head. GQA shrinks the KV cache, which is the dominant memory cost at inference time for long sequences and large batches. (The mechanics are in our [KV cache](/blog/machine-learning/large-language-model/kv-cache) post.) The 7B and 13B models kept standard multi-head attention.
- **A chat model with a published recipe.** Llama 2-Chat was trained with supervised fine-tuning followed by iterative RLHF. The paper describes two separate reward models (one for helpfulness, one for safety), rejection sampling fine-tuning combined with PPO across several rounds, and a technique called Ghost Attention for keeping a system instruction in force across a multi-turn conversation. For many teams, this paper was the first readable end-to-end account of an RLHF pipeline at scale.
- **License.** The Llama 2 Community License allowed commercial use, with two notable strings attached: an acceptable use policy, and a clause requiring companies with more than 700 million monthly active users to request a separate license from Meta. That 700M clause stayed in every Llama license through Llama 4.

| Variant | Params | Attention | Context | Training tokens | Type | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 7B | MHA | 4K | 2T | Base, Chat | Open |
| Llama 2 13B | 13B | MHA | 4K | 2T | Base, Chat | Open |
| Llama 2 34B | 34B | GQA | 4K | 2T | Described in paper | Not released |
| Llama 2 70B | 70B | GQA | 4K | 2T | Base, Chat | Open |

The 34B model is worth a footnote. It appears in the paper's results but Meta withheld it, citing the time needed to red-team it adequately. Its slot was eventually filled by Code Llama 34B a month later.

## Code Llama (August 2023, 70B in January 2024): the first specialization

**Released:** 24 August 2023 (7B, 13B, 34B); 70B added on 29 January 2024. **Primary source:** [Code Llama: Open Foundation Models for Code (arXiv:2308.12950)](https://arxiv.org/abs/2308.12950) and Meta's [Code Llama post](https://ai.meta.com/blog/code-llama-large-language-model-coding/).

Code Llama established the pattern Meta would reuse: rather than train a code model from scratch, continue pretraining a Llama base on a code-heavy mix. Each Code Llama model started from the matching Llama 2 checkpoint and was trained on 500 billion tokens of code and code-related data (1 trillion for the 70B). Three ideas in the recipe are worth knowing:

- **Long context through RoPE base scaling.** Training used 16K-token sequences, and a long-context fine-tuning stage raised the RoPE base period so the model could extrapolate. Meta reported stable generation with inputs up to 100K tokens. This is the same idea, scaling the rotary frequency base rather than retraining positions, that later let Llama 3.1 reach 128K.
- **Fill-in-the-middle (infilling).** The 7B and 13B models (and, per the revised paper, the 70B) were trained to fill a gap given the code before and after it, which is what an IDE completion engine actually needs. If the data side of this interests you, our post on [DeepSeek Coder's repo packing and FIM continued pretraining](/blog/machine-learning/frontier-models/deepseek-coder-repo-packing-fim-continued-pretraining) covers the same technique from a different lab.
- **Specialization tree.** Each size came in three flavors: the base model, Code Llama - Python (a further 100 billion tokens of Python), and Code Llama - Instruct (instruction-tuned for natural-language requests).

| Variant | Params | Code tokens | Infilling | Flavors | Weights |
| --- | --- | --- | --- | --- | --- |
| Code Llama 7B | 7B | 500B | Yes | Base, Python, Instruct | Open |
| Code Llama 13B | 13B | 500B | Yes | Base, Python, Instruct | Open |
| Code Llama 34B | 34B | 500B | No | Base, Python, Instruct | Open |
| Code Llama 70B | 70B | 1T | Per revised paper | Base, Python, Instruct | Open (Jan 2024) |

Code Llama shipped under the same community license as Llama 2. It was the last code-specialized Llama; from Llama 3 on, Meta folded code into the main pretraining mix instead (Llama 3 had four times more code in its data than Llama 2).

## Llama Guard (December 2023): safety as a model

**Released:** 7 December 2023. **Primary source:** [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations (arXiv:2312.06674)](https://arxiv.org/abs/2312.06674).

Llama Guard launched as part of Purple Llama, Meta's umbrella project for open trust-and-safety tooling. The idea is simple and has aged well: rather than writing a content filter as a pile of rules, take a capable base model (Llama 2-7B), instruction-tune it on a labelled safety taxonomy, and ask it to classify either a user prompt or a model response as safe or unsafe, naming the violated category. Because the taxonomy is part of the prompt, you can adapt it to your own policy with few-shot examples or light fine-tuning.

| Generation | Released | Base | Size | Modality | Primary source |
| --- | --- | --- | --- | --- | --- |
| Llama Guard | 7 Dec 2023 | Llama 2-7B | 7B | Text | [arXiv:2312.06674](https://arxiv.org/abs/2312.06674) |
| Llama Guard 2 | 18 Apr 2024 (with Llama 3) | Llama 3 8B | 8B | Text, MLCommons taxonomy | [Llama 3 post](https://ai.meta.com/blog/meta-llama-3/) |
| Llama Guard 3 8B | 23 Jul 2024 (with Llama 3.1) | Llama 3.1 8B | 8B | Text, multilingual | [Llama 3.1 post](https://ai.meta.com/blog/meta-llama-3-1/) |
| Llama Guard 3 1B, 11B Vision | 25 Sep 2024 (with Llama 3.2) | Llama 3.2 | 1B, 11B | Text; text plus image | [Llama 3.2 post](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Llama Guard 4 | 29 Apr 2025 (LlamaCon) | Pruned Llama 4 Scout | 12B dense | Text plus multiple images | [model card](https://huggingface.co/meta-llama/Llama-Guard-4-12B) |

Each Llama generation after the first shipped a Guard model alongside it, so I track them here as one line rather than repeating the story in every section. Two details stand out. Llama Guard 3 1B was pruned and quantized from 2,858 MB down to 438 MB, which makes it cheap enough to run as a sidecar on every request. Llama Guard 4 was built by taking the Llama 4 Scout pretrained model, removing the routed experts and router layers, and keeping only the shared expert, which turned a 109B mixture-of-experts model into a 12B dense classifier with no further pretraining. It covers the 14 hazard categories S1 to S14 (the MLCommons taxonomy plus a Code Interpreter Abuse category). For how these classifiers compare with newer streaming guard models, see our post on [Qwen3Guard and streaming moderation](/blog/machine-learning/frontier-models/guard-models-qwen3guard-streaming-moderation), and for deploying them see [production LLM guardrails](/blog/machine-learning/large-language-model/production-llm-guardrails-models-evaluation-finetuning).

## Llama 3 (April 2024): same skeleton, much more data

**Released:** 18 April 2024. **Primary source:** Meta's [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/) and the [model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md). The full technical report came later with Llama 3.1.

Llama 3 is the clearest demonstration in the family that data, not architecture, was where the gains were. The network was still a dense decoder with RMSNorm, SwiGLU and RoPE. What changed:

- **Tokenizer.** The vocabulary grew from 32K to 128K tokens, using a tiktoken-based tokenizer. A bigger vocabulary means fewer tokens per sentence, which is effectively free context length and cheaper inference per word, at the cost of a bigger embedding matrix. (Our posts on [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) and [BPE](/blog/machine-learning/large-language-model/bpe-tokenizer) go into this tradeoff.)
- **GQA everywhere.** Both the 8B and the 70B used grouped-query attention, where Llama 2 had used it only on its large models.
- **Data, by a factor of seven.** Pretraining used over 15 trillion tokens from publicly available sources, about seven times Llama 2, with four times more code and over 5% high-quality non-English data spanning more than 30 languages.
- **Context.** Training sequences were 8,192 tokens, with a mask that stopped attention from crossing document boundaries inside a packed sequence.
- **Post-training.** The instruct models combined SFT, rejection sampling, PPO and DPO.

Meta also said in the launch post that its largest Llama 3 models, over 400B parameters, were still training. That model became Llama 3.1 405B three months later.

| Variant | Params | Attention | Context | Training tokens | Type | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3 8B | 8B | GQA | 8K | 15T+ | Base, Instruct | Open |
| Llama 3 70B | 70B | GQA | 8K | 15T+ | Base, Instruct | Open |

The launch also shipped Llama Guard 2, Code Shield (an inference-time filter for insecure generated code) and CyberSec Eval 2. If you want to feel what an 8B Llama 3 forward pass actually computes, our [forward pass by hand](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) post builds one from scratch, and the [Llama 3 8B on a single 4090 experiment](/blog/machine-learning/inference-engineering/experiment-llama-3-8b-on-a-single-4090) measures what it costs to serve.

## Llama 3.1 (July 2024): the 405B and the 128K window

**Released:** 23 July 2024. **Primary source:** [The Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783), submitted 31 July 2024, and Meta's [Llama 3.1 post](https://ai.meta.com/blog/meta-llama-3-1/).

Llama 3.1 is the release most people mean when they say "Llama 3". It shipped refreshed 8B and 70B models alongside a 405B flagship, and the accompanying paper is still one of the most complete public descriptions of training a frontier-scale model.

**What changed versus Llama 3:**

- **Scale.** The 405B was a dense transformer trained on more than 15 trillion tokens using over 16,000 H100 GPUs, the largest Llama run to that point. Meta wrote that it chose a dense model over mixture-of-experts specifically to maximize training stability. The 405B was then used as a teacher to improve the post-training of the smaller models.
- **Context.** All three sizes support 128K tokens, up from 8K, reached through a staged long-context extension during pretraining.
- **Multilinguality and tool use.** Official support for eight languages (English, German, French, Italian, Portuguese, Hindi, Spanish, Thai), plus built-in tool calling.
- **Post-training recipe.** Several rounds, each combining SFT, rejection sampling and DPO. Note what is missing compared with Llama 2 and Llama 3: PPO. Most of the SFT data was synthetic, generated and filtered by earlier models. (For why DPO displaced PPO across the industry around this time, see [fine-tuning with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) and the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide).)
- **FP8 serving.** Meta released the 405B quantized from BF16 to FP8 so that it fit on a single server node.
- **License.** The Llama 3.1 license explicitly allowed using Llama outputs, including the 405B's, to improve other models. That made the 405B a legitimate distillation and synthetic-data teacher for the whole open ecosystem.

| Variant | Params | Context | Languages | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B | 8B | 128K | 8 | Base, Instruct | Open |
| Llama 3.1 70B | 70B | 128K | 8 | Base, Instruct | Open |
| Llama 3.1 405B | 405B dense | 128K | 8 | Base, Instruct, FP8 Instruct | Open |

The safety drop with 3.1 was Llama Guard 3 8B and Prompt Guard, an 86M-parameter prompt-injection classifier. Meta also published a request for comment on Llama Stack, a standard set of interfaces for building agentic applications on Llama. The paper describes compositional image, video and speech extensions to Llama 3; the vision part shipped two months later as Llama 3.2, and the speech and video parts were never broadly released.

The 3.1 models are also where much of the open ecosystem branched off. NVIDIA's [Llama-Nemotron reasoning models](/blog/machine-learning/frontier-models/llama-nemotron-efficient-reasoning-models) start from Llama 3.1 and 3.3 checkpoints, and NVIDIA's [Minitron pruning and distillation work](/blog/machine-learning/frontier-models/nvidia-minitron-pruning-distillation) compresses Llama 3.1 8B.

## Llama 3.2 (September 2024): small models and vision

**Released:** 25 September 2024 at Meta Connect. **Primary source:** Meta's [Llama 3.2 post](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/). Quantized 1B and 3B variants followed on 24 October 2024 ([post](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)).

Llama 3.2 extended the family in both directions at once: down to phones and up to images. Neither direction involved training a new base model from scratch.

**The 1B and 3B text models** were made from Llama 3.1 8B by one-shot structured pruning, then recovered with knowledge distillation, using logits from Llama 3.1 8B and 70B as token-level targets during pretraining. Both keep the 128K context window. This prune-then-distill recipe is the same family of techniques covered in our [pruning in LLMs](/blog/machine-learning/large-language-model/pruning-in-llm) and [distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) posts. The October quantized versions used quantization-aware training with LoRA adaptors and, separately, SpinQuant, aimed squarely at on-device inference.

**The 11B and 90B Vision models** took the pretrained Llama 3.1 8B and 70B text models and attached a pretrained image encoder through a set of adapter weights: a series of cross-attention layers that feed image representations into the language model. During adapter training the image encoder was updated but the language-model weights were frozen, so the text abilities of 3.1 were preserved exactly. Post-training used SFT, rejection sampling and DPO. This "frozen LLM plus cross-attention adapter" design is worth remembering, because Llama 4 abandoned it.

| Variant | Params | Built from | Context | Modality | Type | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 1B | 1B | Pruned + distilled from 3.1 8B | 128K | Text | Base, Instruct, quantized Instruct | Open |
| Llama 3.2 3B | 3B | Pruned + distilled from 3.1 8B | 128K | Text | Base, Instruct, quantized Instruct | Open |
| Llama 3.2 11B Vision | 11B | 3.1 8B + vision adapter | 128K | Image + text in, text out | Base, Instruct | Open |
| Llama 3.2 90B Vision | 90B | 3.1 70B + vision adapter | 128K | Image + text in, text out | Base, Instruct | Open |

The release also made Llama Stack concrete, with a CLI, client SDKs in Python, Node, Kotlin and Swift, and distributions ranging from single-node and cloud to on-device iOS through PyTorch ExecuTorch. Licensing gained a wrinkle here: the Llama 3.2 acceptable use policy did not extend the rights for the multimodal models to individuals and companies based in the EU, a restriction that applied to the Vision models but not the text-only ones.

## Llama 3.3 (December 2024): the 70B refresh

**Released:** 6 December 2024. **Primary source:** the [Llama 3.3 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md) and the [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

Llama 3.3 was a single model with a single pitch: get close to 405B-class quality from a 70B that a team can actually afford to serve. Architecturally it is a Llama 3.1 70B (dense, GQA, 128K context, the same eight languages, a December 2023 knowledge cutoff); the improvement came from better post-training. Meta released only the instruction-tuned model, text in and text out, under the Llama 3.3 Community License.

| Variant | Params | Context | Languages | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| Llama 3.3 70B Instruct | 70B dense | 128K | 8 | Instruct only | Open |

Llama 3.3 70B was the last dense Llama, and for many teams it was the last Llama they upgraded to. It is still a sensible default for a dense open model in that size class, which is why it shows up in our [end-to-end chatbot serving guide](/blog/machine-learning/model-serving/serving-an-llm-chatbot-end-to-end) and in comparisons such as the [Qwen3 vs Gemma 3 vs Llama 3 equal-hardware experiment](/blog/machine-learning/inference-engineering/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware).

## Llama 4 (April 2025): mixture of experts and native multimodality

**Released:** 5 April 2025 (a Saturday). **Primary source:** Meta's [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) and the [Llama 4 Scout model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and [Maverick model card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E). There was no full technical report.

Where every release from Llama 2 to 3.3 had changed one or two levers, Llama 4 changed nearly all of them at once.

**What changed versus Llama 3.x:**

- **Mixture of experts.** The first Llama MoE models. Scout has 16 experts, 109B total parameters and 17B active per token. Maverick has 128 routed experts plus a shared expert, 400B total and 17B active; each token goes to the shared expert and one routed expert. Dense and MoE layers alternate. (Background: our [MoE architecture](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) post, and for comparison the very different [DeepSeekMoE fine-grained expert design](/blog/machine-learning/frontier-models/deepseek-moe-lineage-fine-grained-shared-experts).)
- **Native multimodality through early fusion.** Instead of Llama 3.2's frozen-LLM-plus-adapter, text and image tokens enter one backbone from the start of pretraining. The vision encoder is based on MetaCLIP.
- **iRoPE and long context.** Most layers use RoPE, but some interleaved attention layers use no positional embeddings at all, and attention is temperature-scaled at inference time. Meta pre-trained and post-trained at 256K and claimed a 10M-token context for Scout and 1M for Maverick.
- **Data and precision.** Over 30 trillion tokens in the pretraining mixture (the model card puts Scout at about 40T and Maverick at about 22T), covering 200 languages with roughly ten times the multilingual tokens of Llama 3. Training ran in FP8, and a hyperparameter technique called MetaP set per-layer learning rates and initialization scales. Officially supported languages rose to 12.
- **Post-training.** Lightweight SFT, then online RL, then lightweight DPO. Meta used Llama models as judges to remove more than 50% of the SFT data tagged as easy, and filtered RL prompts continuously to keep only medium-to-hard ones.
- **Codistillation.** Maverick was codistilled from Behemoth, the unreleased teacher.

| Variant | Total params | Active params | Experts | Context | Modality | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 4 Scout | 109B | 17B | 16 | 10M | Text + image in, text out | Open (BF16; Int4 fits one H100) |
| Llama 4 Maverick | 400B | 17B | 128 routed + 1 shared | 1M | Text + image in, text out | Open (BF16 and FP8; fits one H100 DGX host) |
| Llama 4 Behemoth | About 2T | 288B | 16 | Not stated | Text + image | Previewed, never released |

The model card lists about 7.38M H100 GPU hours of training (5.0M for Scout, 2.38M for Maverick), a knowledge cutoff of August 2024, and the Llama 4 Community License, which keeps the 700M monthly-active-user clause and adds a "Built with Llama" attribution requirement.

**The reception problem.** Llama 4 had a rough launch. The version of Maverick that topped the LMArena leaderboard at launch was an experimental chat-tuned variant rather than the released weights, which drew public criticism and an LMArena policy update. Independent users also reported long-context and coding results well below what the headline numbers suggested. The comparison of Llama 4's design choices with contemporaries is covered in our [modern LLM architectures comparison](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek).

**Behemoth's status.** Behemoth was announced as a preview that was "still training". In May 2025 [Axios, citing the Wall Street Journal,](https://www.axios.com/2025/05/15/meta-behemoth-llama-scaling-delays) reported its release had been delayed over internal doubts that it improved enough on earlier models, and in July 2025 [TechCrunch, citing the New York Times,](https://techcrunch.com/2025/07/14/meta-built-its-ai-reputation-on-openness-that-may-be-changing) reported that Meta's new Superintelligence Labs had discussed abandoning it in favor of a closed model; Meta replied that its open-source position was unchanged. As of 14 October 2026, Behemoth weights have never been released, and Meta has not published a formal cancellation that I could find. Treat it as a teacher that existed inside Meta, not as a model you can use.

## LlamaCon (April 2025): the last Llama-branded drop

**Released:** 29 April 2025. **Primary source:** Meta's [LlamaCon recap](https://ai.meta.com/blog/llamacon-llama-news) and the [Llama Guard 4 model card](https://huggingface.co/meta-llama/Llama-Guard-4-12B).

Meta's first developer conference for Llama shipped no new language model, but it did ship the last open Llama-branded weights:

| Release | Size | What it does | Weights |
| --- | --- | --- | --- |
| Llama Guard 4 | 12B dense (pruned from Scout) | Safety classifier for text and multi-image prompts, S1 to S14 hazards | Open |
| Llama Prompt Guard 2 | 86M and 22M | Prompt-injection and jailbreak detection; the 22M is English-focused and cheaper | Open |
| LlamaFirewall | Framework | Orchestrates guard models into an agent security layer | Open |
| Llama API | Service | Limited free preview, OpenAI-SDK compatible, serving Scout and Maverick | API |

## After Llama: the Muse models (2026)

This catalog stops at Llama 4 because that is where the Llama name stops. As of 14 October 2026 Meta has released no Llama 4.x point release and no Llama 5. What it has released, through Meta Superintelligence Labs, is a new family called Muse. These are not Llama models and I have not tried to fit them into the Llama tables, but anyone tracing Meta's lineage needs to know they exist:

| Release | Date | What it is | Weights | Primary source |
| --- | --- | --- | --- | --- |
| Muse Spark | 8 Apr 2026 | Natively multimodal reasoning model that replaced Llama behind Meta AI | Closed; private API preview | [Meta blog](https://ai.meta.com/blog/introducing-muse-spark-msl/) |
| Muse Spark 1.1 | 9 Jul 2026 | Upgrade, with a public preview of the Meta Model API | Closed | [Meta blog](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/) |
| Muse Glimmer | 10 Aug 2026 | 30B model for running agents locally, described as an open counterpart to Spark | Open, Apache 2.0 | [Meta blog](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) |
| Muse Wonder | 12 Oct 2026 | Meta's first frontier-class model, served through the Meta Model API | Closed | [Meta blog](https://research.meta.ai/blog/muse-wonder) |

Two things about this table matter for the Llama story. First, the flagship moved to closed weights, which is the opposite of the bet Meta made with Llama 2. Second, the one open Muse model so far ships under Apache 2.0 rather than a custom community license, so it drops the 700M-user clause and the acceptable use policy that every Llama carried. I have deliberately kept the Muse detail thin: these are recent releases, and architecture details beyond what is in the table were not verifiable from primary sources at the time of writing.

## How the family evolved

![Matrix of what changed per Llama generation across attention, context, vocabulary, pretraining data and post-training, with the biggest change in each row highlighted](/imgs/blogs/llama-model-family-timeline-2.webp)

Lay the generations side by side, as in the matrix above, and five through-lines come out.

**1. The decoder recipe was frozen early and changed one lever at a time.** RMSNorm pre-normalization, SwiGLU and RoPE arrived in LLaMA and never left. Each later generation changed one or two things: Llama 2 added GQA on large models and doubled context, Llama 3 swapped the tokenizer and put GQA everywhere, Llama 3.1 stretched context to 128K, Llama 3.2 bolted on vision. That discipline is a big part of why the family was so easy to build on: every inference engine that supported LLaMA supported Llama 3.3 with minimal changes. Llama 4 is the exception that proves the rule, changing MoE, multimodality, positional encoding and precision at once, and its rough launch is at least partly the cost of changing that many things in one step.

**2. Data grew roughly 40x while model sizes moved much less.** From 1.0T tokens for LLaMA-7B to about 40T for Llama 4 Scout is a factor of 40; the typical deployed size stayed in the 8B to 70B band throughout. The original LLaMA argument (overtrain small models because inference is the recurring cost) never went away. It just got more extreme.

**3. Context length went from 2K to 128K in four steps, then made a claim it could not back up.** 2K, 4K, 8K, 128K, each reached by training or RoPE base scaling rather than a new attention mechanism. Llama 4's jump to a 10M-token claim, on a model post-trained at 256K, is where the claimed window and the useful window visibly separated.

**4. Post-training got simpler and more synthetic.** Llama 2 had two reward models, rejection sampling and PPO. Llama 3 used SFT, rejection sampling, PPO and DPO. Llama 3.1 dropped PPO for rounds of SFT, rejection sampling and DPO on mostly synthetic data. Llama 4 trimmed SFT hard (removing over half the data as too easy) and moved the weight onto online RL. The direction matches the rest of the field: fewer human labels, more model-generated and model-filtered data, more RL on hard prompts.

**5. The license loosened, then the brand ended.** Research-only, then commercial with a 700M-user clause, then (3.1) explicit permission to train other models on Llama outputs, then an EU carve-out for multimodal weights. The strings never fully went away, and when Meta did return to open weights in 2026 it was under Apache 2.0 and a different name.

## Which variant to pick today

![Decision tree for choosing a Llama variant in October 2026 keyed on hardware and modality first, then on whether a Llama-licensed model is needed](/imgs/blogs/llama-model-family-timeline-3.webp)

The decision tree above runs in two steps. Step one is hardware and modality, because that narrows the field to one or two candidates almost immediately. Step two is whether you need a Llama-licensed model at all.

**Step 1: what hardware, what modality?**

- **Phone or edge, text only:** Llama 3.2 1B or 3B. Both keep the 128K window, and the official quantized versions exist for exactly this case.
- **A single GPU, text, general chat:** Llama 3.1 8B Instruct. It has the largest ecosystem of fine-tunes, adapters and serving recipes of any Llama.
- **Multi-GPU, the best dense text quality:** Llama 3.3 70B Instruct, the strongest dense Llama and much cheaper to serve than the 405B.
- **Images in, one H100 node:** Llama 4 Scout (Int4 on a single H100) or Maverick (FP8 on a single H100 DGX host). Do not plan around the 10M or 1M context claims without testing on your own long documents.
- **A safety classifier:** Llama Guard 4 12B if you need text plus images, Llama Guard 3 1B if you need something cheap enough to run on every text request.

**Step 2: do you need a Llama-licensed model at all?** If yes (because you have an existing Llama fine-tune, a customer requirement, or a toolchain built around it), ship the pick from step one. If not, compare it against the current generation of open models before committing, because the field has moved a long way since Llama 4 and a newer open model of the same size will often win. Our [modern LLM architectures comparison](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek) is a good starting point for that comparison. And remember the license terms travel with the weights: the 700M MAU clause, the acceptable use policy, the attribution requirement for Llama 4, and the EU restriction on the 3.2 Vision models.

One practical note on the 405B: it remains a useful *teacher* (the 3.1 license explicitly allows training on its outputs) but it is rarely the right model to serve in 2026, since Llama 3.3 70B gets close at a fraction of the cost.

## References

Primary sources, in release order:

1. Touvron et al., [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971), arXiv:2302.13971, February 2023. Meta, [Introducing LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).
2. Touvron et al., [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288), arXiv:2307.09288, July 2023. Meta, [Llama 2 announcement](https://about.fb.com/news/2023/07/llama-2/). [Llama 2 model card](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md).
3. Rozière et al., [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950), arXiv:2308.12950, August 2023. Meta, [Code Llama post](https://ai.meta.com/blog/code-llama-large-language-model-coding/).
4. Inan et al., [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674), arXiv:2312.06674, December 2023.
5. Meta, [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/), April 2024. [Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
6. Llama Team, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), arXiv:2407.21783, July 2024. Meta, [Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/).
7. Meta, [Llama 3.2: Revolutionizing edge AI and vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/), September 2024. Meta, [Introducing quantized Llama models](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/), October 2024.
8. [Llama 3.3 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md), December 2024. [Llama 3.3 70B Instruct on Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).
9. Meta, [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), April 2025. [Llama 4 Scout model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct), [Llama 4 Maverick model card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E).
10. Meta, [Everything we announced at LlamaCon](https://ai.meta.com/blog/llamacon-llama-news), April 2025. [Llama Guard 4 model card](https://huggingface.co/meta-llama/Llama-Guard-4-12B).

Reporting on Behemoth's status (secondary, cited because Meta published no primary statement):

11. Axios, [Meta delays "Behemoth" AI model release](https://www.axios.com/2025/05/15/meta-behemoth-llama-scaling-delays), May 2025.
12. TechCrunch, [Meta built its AI reputation on openness; that may be changing](https://techcrunch.com/2025/07/14/meta-built-its-ai-reputation-on-openness-that-may-be-changing), July 2025.

After Llama:

13. Meta, [Introducing Muse Spark](https://ai.meta.com/blog/introducing-muse-spark-msl/), April 2026; [Introducing Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/), July 2026; [Introducing Muse Glimmer](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model), August 2026; [Muse Wonder](https://research.meta.ai/blog/muse-wonder), October 2026.
