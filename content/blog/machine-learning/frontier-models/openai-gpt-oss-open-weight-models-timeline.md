---
title: "OpenAI's open-weight models, from GPT-2 to gpt-oss: a release timeline and architecture reference"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "Every OpenAI open-weight release in order, from the 2019 GPT-2 staged release to gpt-oss-120b, gpt-oss-20b, gpt-oss-safeguard and Privacy Filter, with verified specs, variant tables, and a deep look at how gpt-oss-120b is built and served."
category: "machine-learning"
subcategory: "Frontier Models"
tags:
  - gpt-oss
  - openai
  - open-weight-models
  - mixture-of-experts
  - mxfp4
  - harmony-format
  - attention-sinks
  - model-timeline
  - llm-inference
author: "Hiep Tran"
featured: false
readTime: 30
---

OpenAI is the lab whose name promised openness and whose open-weight language-model history is, for most of its existence, a single model: GPT-2. Between the last GPT-2 checkpoint in November 2019 and the first gpt-oss checkpoint in August 2025, OpenAI shipped no open-weight language model at all. It did ship important open weights in that window (CLIP, Whisper, a handful of diffusion and 3D research models), but none of them generated text. Then, in one release, it put out a 117B mixture-of-experts reasoning model that fits on a single 80 GB GPU, under Apache 2.0, and the open ecosystem has been running inference experiments on it ever since.

That shape matters for anyone choosing a model, because it means the "family" is short and unusually coherent. There is no gpt-oss 1, 1.5, 2, 2.5 lineage to untangle. There is a prehistory (GPT-2), a gap, one base architecture (gpt-oss), and a small set of things derived from or modeled on that architecture: gpt-oss-safeguard for policy-following moderation and Privacy Filter for PII redaction. Most of what a practitioner needs to know is in one model card, and most of what makes the models interesting is in how they are built to be cheap to serve.

![OpenAI's open-weight releases from 2019 to 2026: four GPT-2 stages, a six-year gap with only non-LLM releases, then the gpt-oss family](/imgs/blogs/openai-gpt-oss-open-weight-models-timeline-1.webp)

The diagram above is the mental model: four GPT-2 stages bunched into 2019, a six-year band where the only open OpenAI weights were perception and research models, then the gpt-oss family arriving in August 2025 and branching into specialized descendants over the following months. This post walks that timeline in order, gives each release a variant table and a primary source, spends most of its depth on gpt-oss-120b's architecture and serving story, and closes with the through-lines and a short "which one should I run" guide.

A note on method. Every date and spec below was checked against a primary source: the gpt-oss model card on arXiv, the model `config.json` files on Hugging Face, OpenAI's GitHub repositories, or OpenAI's own announcement pages. Where a fact rests only on secondary coverage, the text says so.

## The release timeline at a glance

Before the per-release sections, here is the whole catalog in one table. "Open weights" means the trained parameters are downloadable; it does not mean the training data or training code were released, and for every entry here they were not.

| Date | Release | Type | Sizes | License | Primary source |
| --- | --- | --- | --- | --- | --- |
| Feb 14, 2019 | GPT-2 (staged release begins) | Dense decoder LM | 124M published | Modified MIT | [OpenAI GPT-2 repo](https://github.com/openai/gpt-2) |
| May 2019 | GPT-2 medium | Dense decoder LM | 355M | Modified MIT | [GPT-2 repo](https://github.com/openai/gpt-2) |
| Aug 20, 2019 | GPT-2 large | Dense decoder LM | 774M | Modified MIT | [6-month follow-up](https://openai.com/blog/gpt-2-6-month-follow-up/) |
| Nov 5, 2019 | GPT-2 XL (full release) | Dense decoder LM | 1.5B | Modified MIT | [1.5B release post](https://openai.com/index/gpt-2-1-5b-release/) |
| Jan 2021 | CLIP (context only) | Image-text encoder | several | MIT | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) |
| Sep 2022 | Whisper (context only) | Speech encoder-decoder | 39M to 1.55B | MIT | [openai/whisper](https://github.com/openai/whisper) |
| Aug 5, 2025 | gpt-oss-120b, gpt-oss-20b | MoE reasoning LM | 116.83B / 20.91B total | Apache 2.0 | [arXiv 2508.10925](https://arxiv.org/abs/2508.10925) |
| Oct 29, 2025 | gpt-oss-safeguard-120b, -20b | Fine-tuned gpt-oss, safety classifier | same as gpt-oss | Apache 2.0 | [Technical report](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf) |
| Dec 2025 | circuit-sparsity (research) | Weight-sparse LM | 0.4B | see repo | [arXiv 2511.13653](https://arxiv.org/abs/2511.13653) |
| Apr 2026 | Privacy Filter | MoE token classifier (gpt-oss-like) | 1.5B total / 50M active | Apache 2.0 | [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) |

Three counts fall out of this table. The text-generating open-weight releases are GPT-2 (in four sizes) and gpt-oss (in two). The gpt-oss architecture has been reused twice: once as a direct fine-tune (safeguard), once as an architectural template for a much smaller non-generative model (Privacy Filter). And as of October 14, 2026, OpenAI has not released a successor to gpt-oss. Its 2026 release notes list closed products only; there is no gpt-oss-2, and this post will not invent one.

## Prehistory: GPT-2 and the staged release (2019)

### What shipped and when

GPT-2 was announced on February 14, 2019 with a now-famous refusal: OpenAI published the paper and a small checkpoint, and withheld the full 1.5B model on the grounds that it might be used to generate disinformation at scale. What followed was the first "staged release" of a language model, and it is the direct ancestor of every open-weight safety argument OpenAI made six years later about gpt-oss.

| Variant | Parameters | Released | Notes |
| --- | --- | --- | --- |
| GPT-2 small | 124M | Feb 2019 | Originally labeled "117M"; the repo README notes the original counts were wrong |
| GPT-2 medium | 355M | May 2019 | Originally labeled "345M" |
| GPT-2 large | 774M | Aug 20, 2019 | Announced in the 6-month follow-up post |
| GPT-2 XL | 1.5B | Nov 5, 2019 | "Final model release" of the staged process |

All four are dense, decoder-only transformers with a 1,024-token context and a byte-level BPE vocabulary of 50,257 tokens. They were base models only: no instruction tuning, no chat format, no RLHF. The license is a modified MIT. The [openai/gpt-2](https://github.com/openai/gpt-2) repository was archived on April 8, 2026, which is a small, fitting bookend.

### Why it matters for gpt-oss

The interesting part of GPT-2 for this catalog is not the architecture, which the field outgrew within a year, but the release logic. In November 2019 OpenAI said it had "seen no strong evidence of misuse" and released the 1.5B weights. The staged release was an experiment in the question "how do you decide an open model is safe enough?" and the answer was observational: release small, watch, release larger.

gpt-oss answered the same question differently, and more rigorously. Instead of watching for misuse after release, OpenAI tried to *manufacture* the worst case before release: it adversarially fine-tuned gpt-oss-120b toward biology and cybersecurity capability and measured whether even the attacked model crossed its internal "High" capability thresholds. We come back to this in the gpt-oss section, but the through-line is direct: both releases were gated by a safety argument about what an attacker could do with the weights, and both arguments were published.

## The gap: non-LLM open weights (2020 to 2024)

For context only, because they are not part of the language-model family: OpenAI did keep releasing open weights between 2019 and 2025, just not for text generation.

| Release | Date | What it is | License |
| --- | --- | --- | --- |
| CLIP | Jan 2021 | Contrastive image-text encoders; the backbone of a generation of text-to-image and VLM systems | MIT |
| Whisper | Sep 2022 | Encoder-decoder ASR, tiny (39M) through large (1,550M); large-v3 added Nov 2023, turbo (809M) Sep 2024 | MIT |
| Consistency models, Shap-E | 2023 | Diffusion and 3D research checkpoints on the OpenAI Hugging Face org | various |

Whisper in particular became infrastructure, and [this blog's Whisper post](/blog/machine-learning/signal-processing/whisper-under-the-hood) walks through it. But for six years, if you wanted an OpenAI model that wrote text, you called an API. The open-weight LLM ecosystem in that period was built by Meta, Mistral, Alibaba, DeepSeek and others, which is the backdrop gpt-oss landed into.

## gpt-oss-120b and gpt-oss-20b (August 5, 2025)

### What was released

On August 5, 2025, OpenAI released two open-weight reasoning models under Apache 2.0, its first open-weight language models since GPT-2. The [model card](https://arxiv.org/abs/2508.10925) (submitted to arXiv August 8) describes them as mixture-of-experts transformers trained with "large-scale distillation and reinforcement learning", optimized for agentic use (browsing, Python tool use, developer-defined function calls), and using a new rendered chat format called harmony. OpenAI released the weights, reference inference implementations, the tool environments, and the tokenizer. It did not release training data or training code.

| Variant | Layers | Total params | Active per token | Experts / top-k | Context | Checkpoint | Fits on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | 36 | 116.83B | 5.13B | 128 / 4 | 131,072 | 60.8 GiB | One 80 GB GPU (H100, MI300X) |
| gpt-oss-20b | 24 | 20.91B | 3.61B | 32 / 4 | 131,072 | 12.8 GiB | 16 GB of memory |

Both are text-only, both are reasoning models with a visible chain of thought, both support three reasoning effort levels (low, medium, high), and both have a knowledge cutoff of June 2024. There are no separate base, instruct, coder, or vision variants: each size ships as a single post-trained checkpoint. That is a deliberate contrast with families like Qwen or Llama, and it simplifies the choice to "which size".

OpenAI's headline framing was that gpt-oss-120b reaches near-parity with o4-mini on core reasoning benchmarks and gpt-oss-20b is comparable to o3-mini. The model card numbers at high reasoning effort:

| Benchmark (high effort) | gpt-oss-120b | gpt-oss-20b |
| --- | --- | --- |
| AIME 2024 (no tools) | 95.8 | 92.1 |
| AIME 2025 (no tools) | 92.5 | 91.7 |
| GPQA Diamond (no tools) | 80.1 | 71.5 |
| MMLU | 90.0 | 85.3 |
| Humanity's Last Exam (no tools) | 14.9 | 10.9 |
| SWE-bench Verified | 62.4 | 60.7 |
| Tau-Bench Retail | 67.8 | 54.8 |
| Codeforces Elo (no tools / with tools) | 2463 / 2622 | 2230 / 2516 |
| HealthBench / HealthBench Hard | 57.6 / 30.0 | 42.5 / 10.8 |

Read the gap between the two columns carefully, because it is the most useful thing in the table. On competition math the 20b is within a point of the 120b. On GPQA, Tau-Bench and HealthBench Hard, the gap opens to 9, 13 and 19 points. The pattern is the familiar one: knowledge-heavy and long-horizon agentic tasks scale with total parameters (the experts are where facts live), while short, verifiable reasoning tasks can be distilled into a small active path. If your workload is "solve this contained problem", the 20b is a bargain. If it is "know a lot of things and act over many steps", it is not.

### What changed versus GPT-2

Calling gpt-oss "the successor to GPT-2" is accurate as a matter of release history and meaningless as a matter of engineering. Almost every axis changed:

| Axis | GPT-2 XL (2019) | gpt-oss-120b (2025) |
| --- | --- | --- |
| FFN | Dense MLP | 128-expert MoE, top-4, SwiGLU |
| Attention | Full multi-head | Alternating banded (128) and dense, GQA 64Q/8KV, learned sinks |
| Positions | Learned absolute | RoPE with YaRN extension |
| Context | 1,024 | 131,072 |
| Vocabulary | 50,257 (byte BPE) | 201,088 (o200k_harmony) |
| Weights format | FP32 | MXFP4 experts, BF16 everything else |
| Training | Next-token pretraining only | Pretraining plus CoT RL similar to o3 |
| Interface | Raw text | harmony format, roles and channels |
| License | Modified MIT | Apache 2.0 |

## gpt-oss-120b: the architecture

This is the part worth reading slowly. The numbers below come from the model card and from the [config.json](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) on Hugging Face, which agree.

![gpt-oss-120b architecture: 18 pairs of banded and dense attention layers, each followed by a 128-expert top-4 MoE, with only expert weights in MXFP4](/imgs/blogs/openai-gpt-oss-open-weight-models-timeline-2.webp)

The figure shows the whole model as one repeated pair. Each pair is a banded-attention block followed by a dense-attention block, and every block ends in the same MoE feed-forward layer. Eighteen pairs make the 36 layers.

### The spec sheet

| Component | gpt-oss-120b | gpt-oss-20b |
| --- | --- | --- |
| Layers | 36 | 24 |
| Residual width | 2,880 | 2,880 |
| Query heads / KV heads / head dim | 64 / 8 / 64 | 64 / 8 / 64 |
| Attention pattern | Alternating banded (window 128) and dense | same |
| Attention sinks | Learned per-head bias in the softmax denominator | same |
| Positional encoding | RoPE, theta 150,000, YaRN factor 32 over 4,096 base | same |
| Max context | 131,072 tokens | same |
| Experts per MoE layer / active | 128 / 4 | 32 / 4 |
| Expert hidden size | 2,880 | 2,880 |
| Activation | SwiGLU, clamped (limit 7.0) | same |
| Normalization | RMSNorm, eps 1e-5 | same |
| Tokenizer | o200k_harmony, 201,088 tokens | same |
| Embeddings | Untied input/output | same |

Two things stand out immediately. First, the two models share everything except depth and expert count. The 20b is not a different design; it is the 120b with fewer layers (24 versus 36) and a quarter of the experts (32 versus 128). Second, the residual width of 2,880 is small for a 117B model. Llama-class dense models of a similar total size use residual widths roughly three times larger. gpt-oss puts almost all of its parameters into width *across experts* rather than width *of the stream*, which is what makes the active path so cheap.

### Where the parameters live

The model card's parameter breakdown is the single most revealing table about this architecture:

| Bucket | gpt-oss-120b | gpt-oss-20b |
| --- | --- | --- |
| MLP (experts + router) | 114.71B | 19.12B |
| Attention | 0.96B | 0.64B |
| Embed + unembed | 1.16B | 1.16B |
| **Total** | **116.83B** | **20.91B** |
| **Active per token** | **5.13B** | **3.61B** |

Over 98% of gpt-oss-120b is expert weights. Attention, the part that sees context, is under 1B parameters across all 36 layers. This is the most MoE-heavy design among widely used open models of its time, and it explains three later facts in one stroke: why quantizing only the experts was enough to fit the model on one GPU, why decode is bandwidth-bound on expert weights rather than on attention, and why the 20b, which keeps the same embedding table, has embeddings as a much larger share of its total (about 5.5%).

You can reproduce the active count roughly. Each MoE layer holds 128 experts; with top-4 routing, a token touches 4/128 of the expert weights, so the active expert parameters are about 114.71B times 4/128, or 3.58B. Add attention (0.96B) and the unembedding matrix used at the output (about 0.58B, half of the 1.16B embed+unembed bucket), and you land at about 5.1B, which matches the model card's 5.13B.

### Alternating banded and dense attention

Half the layers use a *banded* (sliding-window) attention mask with a window of 128 tokens: each token attends only to the previous 128 positions. The other half attend over the full context. The config lists the layer types in strict alternation, sliding then full, eighteen times.

The intuition is a division of labor. Local syntax, the next few clauses, the current line of code, all live within 128 tokens; a banded layer can handle that at a cost that does not grow with sequence length. Long-range retrieval ("the function defined 40k tokens ago") is the dense layers' job. Because information propagates through the residual stream, every banded layer still benefits from what the dense layer below it fetched. Gemma 2 and Gemma 3 use a similar local/global interleave (with much larger windows); gpt-oss pushes the local window down to 128, which is aggressive. The [modern LLM architectures](/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek) post compares these interleaving choices across families.

### Attention sinks

Each attention head has a learned scalar that enters the softmax denominator but has no value vector. The model card describes it as a bias that lets a head "pay no attention to any tokens". In equation form, for a head with sink logit $s$ and attention scores $z_j$ over keys $j$:

$$
a_j = \frac{\exp(z_j)}{\exp(s) + \sum_k \exp(z_k)}
$$

The weights $a_j$ no longer have to sum to 1. When a head has nothing useful to read, it can push mass into the sink instead of being forced to dump it onto some token.

Why this matters is clearest in the banded layers. The [StreamingLLM attention-sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks) work showed that trained transformers park excess attention on the first tokens of a sequence, and that evicting those tokens from a sliding window wrecks quality. A 128-token band evicts the first tokens almost immediately. A learned sink gives the head a dedicated, position-independent place to put "no-op" attention, so the band can slide without destabilizing the softmax. The [MiniMax sparse attention](/blog/machine-learning/frontier-models/minimax-sparse-attention) post notes the same GPT-OSS-style per-head sink logit in a later design, which is a sign the idea has started to propagate.

### Mixture of experts and the clamped SwiGLU

Every layer's feed-forward block is an MoE with a linear router that selects the top 4 of 128 experts (32 in the 20b). Each expert is a SwiGLU MLP with hidden size 2,880. The config sets a `swiglu_limit` of 7.0: the gate and linear branches are clamped before multiplication, which bounds activation magnitudes. That clamp is cheap insurance for a model whose expert weights are stored in 4-bit floating point, where an outlier activation multiplying a coarsely quantized weight is exactly how quality goes wrong. There is no shared expert, unlike the DeepSeek MoE lineage; every token's FFN compute is fully routed. The [MoE training and inference playbook](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) covers what that implies for expert-parallel serving.

### MXFP4: quantization designed in, not bolted on

MXFP4 is the OCP microscaling format: 4-bit floating-point values (E2M1) in blocks of 32 that share one 8-bit power-of-two scale. That is 4 bits per value plus 8/32 = 0.25 bits of scale overhead, or 4.25 bits per parameter, which is the number the model card quotes. In the checkpoint, each quantized tensor is stored as `blocks` (two FP4 values packed per `uint8`) and `scales`.

Only the MoE projection weights are MXFP4. The config's quantization section explicitly excludes attention, the router, embeddings, and the LM head, which stay in BF16. Since the experts are over 90% of the parameters, quantizing just them gets most of the saving while keeping the precision-sensitive parts intact. And because OpenAI ran all its evaluations on this quantized checkpoint, the benchmark table above *is* the MXFP4 model's quality, not a BF16 number you hope survives quantization.

Here is the size arithmetic, worth doing once because it shows the design was sized to a GPU:

```python
# gpt-oss-120b checkpoint size from the model card's parameter breakdown
GiB = 2**30

moe_params   = 114.71e9   # stored in MXFP4
attn_params  = 0.96e9     # BF16
embed_params = 1.16e9     # BF16 (embed + unembed, untied)

mxfp4_bits = 4 + 8 / 32   # 4-bit E2M1 value + one 8-bit scale per 32-value block
moe_bytes  = moe_params * mxfp4_bits / 8
bf16_bytes = (attn_params + embed_params) * 2

total = moe_bytes + bf16_bytes
print(f"MoE in MXFP4 : {moe_bytes / GiB:6.1f} GiB")
print(f"BF16 rest    : {bf16_bytes / GiB:6.1f} GiB")
print(f"Checkpoint   : {total / GiB:6.1f} GiB  (model card: 60.8 GiB)")

# Same weights all in BF16, for comparison
print(f"All-BF16     : {(moe_params + attn_params + embed_params) * 2 / GiB:6.1f} GiB")
```

This prints about 56.8 GiB for the experts, 3.9 GiB for the BF16 rest, and 60.7 GiB total, matching the model card's 60.8 GiB. The all-BF16 version would be about 218 GiB, three H100s' worth before any KV cache. MXFP4 is the difference between a multi-GPU deployment and a single card with roughly 15 GiB to spare for activations and cache. Whether your hardware executes MXFP4 natively or dequantizes on the fly is a separate question, covered in [FP8 and FP4 inference: what the hardware actually gives you](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you).

### The tokenizer and the harmony format

gpt-oss uses `o200k_harmony`, a 201,088-token BPE vocabulary that extends the o200k tokenizer used by OpenAI's GPT-4o-era models with special tokens for the harmony format. Harmony is not optional. The GitHub README says the models "should only be used with the harmony format" and will not work correctly otherwise.

Harmony gives each message a role, in a strict instruction hierarchy (system, developer, user, assistant, tool), and gives assistant output a *channel*:

| Channel | Carries | Show to end users? |
| --- | --- | --- |
| `analysis` | Chain-of-thought reasoning | No; OpenAI says the CoT is not safety-trained for display |
| `commentary` | Tool and function calls, preambles | Usually no |
| `final` | The answer | Yes |

Messages are delimited by special tokens such as `<|start|>`, `<|message|>` and `<|end|>`, and the system message carries the reasoning effort, the knowledge cutoff, and the list of valid channels. You rarely write these by hand. The [openai-harmony](https://github.com/openai/harmony) package (a Rust core with Python bindings) renders and parses them:

```python
from openai_harmony import (
    Conversation, DeveloperContent, HarmonyEncodingName, Message,
    ReasoningEffort, Role, SystemContent, load_harmony_encoding,
)

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
developer = DeveloperContent.new().with_instructions(
    "Answer in one sentence. Use metric units."
)

convo = Conversation.from_messages([
    Message.from_role_and_content(Role.SYSTEM, system),
    Message.from_role_and_content(Role.DEVELOPER, developer),
    Message.from_role_and_content(Role.USER, "How far is the Moon?"),
])

# Token ids to feed the model, ending with the assistant turn header
prefill_ids = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
print(len(prefill_ids), "prompt tokens")

# After generation, split the completion back into analysis / final messages:
# messages = enc.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
# answer = [m for m in messages if m.channel == "final"]
```

The practical rule: if you serve gpt-oss through vLLM, Ollama, LM Studio or Transformers' chat template, harmony is applied for you. If you call `model.generate` on raw strings, you will get a confused model, and the confusion looks like a quality bug rather than a format bug. The [tokenizer design playbook](/blog/machine-learning/large-language-model/task-specialized-tokenizer-design) discusses harmony's channel tokens as a case of task-specific vocabulary design.

### Reasoning effort as a dial

Reasoning effort is set in the system message (`Reasoning: low`, `medium` or `high`) and changes how long the analysis channel runs before the final answer. It is a latency and cost dial, not a different model. The model card's benchmark curves show accuracy rising with effort as the average CoT length grows, which is the usual test-time-compute trade. For serving, the consequence is that output length, and therefore your cost per request, depends on a field in the prompt. Budget for high effort if you expose it, and see [the cost model of inference](/blog/machine-learning/inference-engineering/the-cost-model-of-inference-dollars-per-million-tokens) for how output tokens dominate dollars per request.

### Training recipe

The model card is brief here, which is itself information:

- **Pretraining:** text-only, "trillions of tokens", focused on STEM, coding and general knowledge, with the CBRN pretraining filters reused from GPT-4o. Knowledge cutoff June 2024.
- **Compute:** gpt-oss-120b took 2.1 million H100-hours; gpt-oss-20b almost 10x fewer.
- **Post-training:** chain-of-thought RL "similar to OpenAI o3", plus training on the harmony format, tool use (browsing, Python, functions) and the three reasoning levels.

No data mixture, no token count, no learning-rate schedule. If you want training recipes, the DeepSeek and Kimi reports in this subcategory are far more detailed. gpt-oss is an artifact release, not a methods release.

### Safety: worst-case fine-tuning

The genuinely novel methodological contribution of the release is the companion paper [Estimating Worst-Case Frontier Risks of Open-Weight LLMs](https://arxiv.org/abs/2508.03153) (August 5, 2025). The logic: once weights are public, an attacker can fine-tune away refusals and train in capability, so the default model's safety evals are a lower bound on risk. OpenAI therefore built adversarially fine-tuned versions of gpt-oss-120b for biology/chemistry and for cyber, with its own RL infrastructure, and evaluated those.

The conclusion in the model card: the default model does not reach "High" capability in any of the three tracked Preparedness categories, and even the maliciously fine-tuned versions did not reach High in biology or cyber. OpenAI also observed that on most biology evals an existing open-weight model already performed near gpt-oss-120b, so the release added little marginal capability. The attacked checkpoints were not released. This is the GPT-2 staged-release question answered with an experiment instead of a waiting period.

## Why alternating banded attention matters for serving

The attention design looks like a modeling choice, but its largest consequence lands on whoever runs the model. KV cache per token per layer is $2 \times n_{kv} \times d_{head} \times b$ bytes: two tensors (K and V), 8 KV heads, head dimension 64, 2 bytes in BF16, for 2,048 bytes. A dense layer stores that for every token in context. A banded layer only needs the last 128.

![KV cache per 131K-token sequence in BF16: banded layers keep only 128 tokens, so half the layers cost almost nothing and the cache halves](/imgs/blogs/openai-gpt-oss-open-weight-models-timeline-3.webp)

```python
GiB, MiB = 2**30, 2**20
per_tok_layer = 2 * 8 * 64 * 2          # K+V, 8 KV heads, dim 64, BF16 = 2,048 bytes
ctx, window = 131_072, 128

for name, layers in [("gpt-oss-120b", 36), ("gpt-oss-20b", 24)]:
    dense = banded = layers // 2
    all_dense = layers * per_tok_layer * ctx
    actual = dense * per_tok_layer * ctx + banded * per_tok_layer * window
    print(f"{name}: all-dense {all_dense / GiB:.1f} GiB, "
          f"actual {actual / GiB:.2f} GiB "
          f"(banded layers: {banded * per_tok_layer * window / MiB:.1f} MiB)")
```

For gpt-oss-120b at the full 131K context, a hypothetical all-dense model would need 9.0 GiB of KV cache per sequence; the real model needs 4.5 GiB, because the 18 banded layers together hold only about 4.5 MiB. For the 20b it is 6.0 versus 3.0 GiB. On an 80 GB card holding the 60.8 GiB checkpoint, that factor of two is the difference between a handful of long-context users and roughly double that.

The catch is that the cache is no longer uniform. A serving engine that allocates the same block table for every layer either wastes the savings (allocating full-length cache for banded layers) or has to manage two kinds of layer with different eviction rules. That is exactly the problem described in [the hybrid KV cache manager](/blog/machine-learning/model-serving/hybrid-kv-cache-manager), which uses gpt-oss's banded layers as a motivating example, and the memory budgeting is worked through in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache). It also explains why quantizing the KV cache buys less on gpt-oss-20b than on dense models: half the cache is already tiny. The [KV-cache quantization](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) post cites a gain of only about 4% for gpt-oss-20b.

## How gpt-oss is served

OpenAI shipped three reference implementations in the [gpt-oss repo](https://github.com/openai/gpt-oss), and they are deliberately not production engines:

| Implementation | Purpose | Requirements |
| --- | --- | --- |
| PyTorch | Educational; upcasts weights to BF16 | At least 4x H100 for the 120b |
| Triton | Optimized single-GPU path with an MXFP4 MoE kernel, CUDA graphs and caching | One 80 GB GPU for the 120b |
| Metal | Apple Silicon | Not production-ready; matches PyTorch accuracy |

Production serving went to the ecosystem on day one: vLLM (a dedicated `gptoss` build at launch), Hugging Face Transformers, Ollama, LM Studio and llama.cpp all supported the models at or near release. The simplest production path is a single command:

```bash
# One 80 GB GPU (H100 / MI300X class). vLLM applies harmony and exposes
# an OpenAI-compatible server, including the Responses API shape.
vllm serve openai/gpt-oss-120b \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92

# Recommended sampling from the model repo: temperature 1.0, top_p 1.0
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "openai/gpt-oss-120b",
       "messages": [{"role": "user", "content": "Prove there are infinitely many primes."}],
       "reasoning_effort": "high",
       "temperature": 1.0, "top_p": 1.0}'
```

Three serving facts follow from the architecture and are worth internalizing:

1. **Decode is bandwidth-bound on expert weights.** With 5.1B active parameters in mostly 4.25-bit form, each decoded token reads only about 5 GB of weights (about 1.9 GB of MXFP4 experts plus about 3 GB of BF16 attention and unembedding), which is why the 120b decodes quickly for its size. But batching changes the picture: different tokens in a batch route to different experts, so at moderate batch sizes you end up reading most of the 60 GB anyway. The [skinny-matrix GEMM](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) post explains why.
2. **"3.6B active" does not mean "3.6B fast" on a laptop.** The 20b's total weights still have to sit in memory and be streamed; the [MoE on consumer hardware experiment](/blog/machine-learning/inference-engineering/experiment-a-moe-model-on-consumer-hardware) uses gpt-oss-20b to show where that intuition breaks.
3. **Hardware support for FP4 is uneven.** Blackwell executes FP4 natively; Hopper dequantizes MXFP4 to a wider type inside the kernel. Vendor numbers for gpt-oss-120b on B200 versus Hopper appear in the [quantization quality versus speed](/blog/machine-learning/inference-engineering/experiment-quantization-quality-vs-speed-across-models) post, with the honesty caveats such numbers deserve.

gpt-oss became a default benchmark target for inference work almost immediately, which is why it shows up across this blog: in [CUDA graphs for the decode loop](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop), in [LoRA swapping and cold starts](/blog/machine-learning/inference-engineering/serving-many-models-on-few-gpus-lora-swapping-and-cold-starts), in the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) as the MoE check, in [DFlash speculative decoding](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding) as a target model, in [TokenSpeed](/blog/machine-learning/inference-frameworks/tokenspeed-agentic-inference-engine), and in the [GPU selection guide](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency). On the training side, [the anatomy of an Unsloth speedup](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) covers fine-tuning gpt-oss-20b with SFT and GRPO on a single consumer GPU.

## gpt-oss-safeguard (October 29, 2025)

### What it is

On October 29, 2025, OpenAI released gpt-oss-safeguard-120b and gpt-oss-safeguard-20b as a "research preview": gpt-oss fine-tuned for safety classification, under the same Apache 2.0 license, with a [technical report](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf) and weights on [Hugging Face](https://huggingface.co/openai/gpt-oss-safeguard-120b). It was built with ROOST (Robust Open Online Safety Tools), with Discord and SafetyKit among the early partners, and is the open version of an approach OpenAI uses internally under the name Safety Reasoner.

| Variant | Base | Total / active | Runs on | Input | Output |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-safeguard-120b | gpt-oss-120b | 117B / 5.1B | Single H100 | Written policy + content (harmony format) | Label + reasoning |
| gpt-oss-safeguard-20b | gpt-oss-20b | 21B / 3.6B | 16 GB GPU | same | same |

The architecture is unchanged; this is a post-training release. Reasoning effort (low/medium/high) still applies.

### What changed versus gpt-oss

The change is in the *task contract*. A classic moderation classifier learns a policy from thousands of labeled examples, and changing the policy means relabeling and retraining. gpt-oss-safeguard instead takes the policy as a prompt at inference time, reasons over it in the analysis channel, and emits a decision plus its rationale. Change the policy text, and the classifier changes.

OpenAI's own numbers show where that helps. On an internal multi-policy accuracy eval (a label counts as correct only if it is right under every policy at once), safeguard-120b scored 46.3% and safeguard-20b 43.6%, versus 43.2% for gpt-5-thinking and 32.5% / 32.1% for the base gpt-oss models. On ToxicChat, gpt-5-thinking and the internal Safety Reasoner slightly beat both safeguard models. OpenAI is candid that a dedicated classifier trained on tens of thousands of high-quality labels can still outperform it on complex risks, and that reasoning over a policy per request is too slow and expensive for always-on filtering of all traffic. The recommended pattern is a cheap fast classifier first, safeguard on the ambiguous or high-stakes tail.

For a comparison with the other open approach to the same problem, a streaming guard model trained on a fixed taxonomy, see [Qwen3Guard and streaming moderation](/blog/machine-learning/frontier-models/guard-models-qwen3guard-streaming-moderation). The two designs sit at opposite ends of the latency versus policy-flexibility trade.

## Smaller releases after gpt-oss (December 2025 to April 2026)

Two later OpenAI open-weight releases are worth listing, with the caveat that neither is a general-purpose language model and neither is a gpt-oss successor.

### circuit-sparsity (December 2025)

An interpretability research release accompanying the paper [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653) (Leo Gao and colleagues, November 2025). The released model is a roughly 0.4B-parameter language model in which almost all weights are zero (the sparsest models in the paper keep about 1 in 1,000 nonzero), trained that way from scratch so its circuits are small enough to read. The paper says the authors release the weights and pruned circuits for the models used in its experiments; the Hugging Face repository [openai/circuit-sparsity](https://huggingface.co/openai/circuit-sparsity) was last updated December 12, 2025. The exact release day rests on secondary coverage. It is a research instrument, expensive per unit of capability by design, and not something to deploy.

### Privacy Filter (April 2026)

[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter), released around April 22, 2026 (the date comes from the Hugging Face update and same-week press coverage), is an open-weight PII detector under Apache 2.0, with code at [openai/privacy-filter](https://github.com/openai/privacy-filter). It is the most interesting architectural descendant of gpt-oss, because it shows the design scaled *down*:

| Property | Privacy Filter |
| --- | --- |
| Task | Bidirectional token classification: detect and mask 8 PII span categories (names, addresses, emails, phone numbers, account numbers, URLs, dates, secrets) |
| Blocks | 8 transformer blocks |
| Residual width | 640 |
| FFN | MoE, 128 experts, top-4 routing |
| Attention | Grouped-query attention |
| Parameters | 1.5B total, 50M active |
| Context | Up to 128K tokens (per release coverage) |
| Decoding | One forward pass, then constrained Viterbi span decoding |
| Training | Pretrained autoregressively with an architecture "similar" to gpt-oss, then converted to a token classifier |

The ratio is the story: 1.5B total and 50M active is a 30:1 sparsity ratio, even more extreme than gpt-oss-120b's roughly 23:1. A 50M-active model can run on CPUs and edge devices over long documents, which is the point: you redact PII *before* text leaves the machine. OpenAI reports an F1 of 96% on the PII-Masking-300k benchmark (94.04% precision, 98.04% recall); that is a vendor number on a public benchmark and should be re-measured on your own data, especially for identifiers in languages and formats the benchmark underrepresents.

## How the family evolved

Short as it is, the family has clear through-lines.

**1. The release question moved from "wait and see" to "attack it first".** GPT-2's staged release gated each size on observed misuse. gpt-oss gated release on adversarial fine-tuning run by OpenAI itself, with the method published. gpt-oss-safeguard then turned the same reasoning machinery outward, giving defenders an open model that applies their policies. Each release is shaped as much by the safety argument as by the capability.

**2. Sparsity is the organizing principle.** gpt-oss-120b keeps 5.13B of 116.83B parameters active (about 4.4%). Privacy Filter keeps 50M of 1.5B (about 3.3%). circuit-sparsity is a weight-sparse model on a different axis entirely. OpenAI's open releases consistently spend parameters on stored capacity and keep the per-token path small, which is exactly the trade that favors a single-GPU or on-device deployment.

**3. Quantization is part of the architecture, not a post-processing step.** GPT-2 shipped in FP32 and the community quantized it later. gpt-oss shipped in MXFP4 for the experts, with a clamped SwiGLU to keep activations bounded, and every reported eval ran on that format. The size of the checkpoint was a design target (one 80 GB card, one 16 GB card), and the format was chosen to hit it.

**4. Attention is engineered for the KV cache.** Alternating 128-token bands with dense layers halves long-context cache, and learned sinks make the tiny window stable. These are serving decisions expressed as model decisions, and they pushed inference engines to support non-uniform, hybrid KV caches.

**5. The interface became part of the model.** GPT-2 consumed raw text. gpt-oss is trained on harmony, with roles, an instruction hierarchy and channels that separate reasoning, tool calls and answers, and it misbehaves without it. Reasoning effort is a prompt field. The "API" of an open model now includes its chat format, and getting it wrong is a correctness bug.

## Which variant to pick today

| If you need | Pick | Why |
| --- | --- | --- |
| The strongest OpenAI open model, one datacenter GPU | gpt-oss-120b | o4-mini-class reasoning in 60.8 GiB; best on knowledge-heavy and agentic tasks |
| Local or edge reasoning, 16 GB of memory | gpt-oss-20b | Within a point of the 120b on competition math; weaker on GPQA, Tau-Bench, HealthBench Hard |
| A fine-tuning base on one consumer GPU | gpt-oss-20b | Small enough for LoRA SFT and GRPO on a single card |
| Moderation under a policy you write and revise | gpt-oss-safeguard-20b first, -120b for the hard tail | Policy at inference time with an explanation; put a cheap classifier in front for volume |
| PII redaction before data leaves the machine | Privacy Filter | 50M active, long context, token-level spans |
| Speech recognition | Whisper (large-v3 or turbo) | Not part of the LLM family, still OpenAI's open ASR |
| Studying circuits | circuit-sparsity | Research only |
| Vision input, or the very best open model regardless of lab | Not this family | gpt-oss is text-only; as of October 2026 several other labs' open models are larger and stronger |

Two cautions apply to every row. First, always serve through harmony (any mainstream engine does this for you); raw prompting is the most common way to make gpt-oss look worse than it is. Second, the chain of thought in the analysis channel is not intended for end users. If you build a product on gpt-oss, show the final channel and log the analysis.

## Facts this post could not verify

- **circuit-sparsity release day:** the Hugging Face repo shows an update on December 12, 2025, and coverage gives dates in mid-December; no official announcement page was found, so the post says "December 2025".
- **Privacy Filter exact date:** the official announcement page could not be fetched; the April 22, 2026 date comes from the Hugging Face repository update and press coverage dated April 22 to 23, 2026. The 128K context figure comes from coverage, not the model card.
- **gpt-oss-safeguard-20b January update:** the Hugging Face repo shows a later update dated January 14 with no year displayed and no announced change; it is not treated as a separate release.
- **CLIP and Whisper days of month** are given only as months, which is all this context-only table needs.

## References

1. OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", arXiv:2508.10925, August 2025. [arxiv.org/abs/2508.10925](https://arxiv.org/abs/2508.10925)
2. OpenAI, "Introducing gpt-oss", August 5, 2025. [openai.com/index/introducing-gpt-oss](https://openai.com/index/introducing-gpt-oss/)
3. OpenAI, gpt-oss reference implementations and tools. [github.com/openai/gpt-oss](https://github.com/openai/gpt-oss)
4. gpt-oss-120b weights and config. [huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
5. gpt-oss-20b weights and config. [huggingface.co/openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
6. OpenAI, harmony response format renderer. [github.com/openai/harmony](https://github.com/openai/harmony)
7. Wallace et al. (OpenAI), "Estimating Worst-Case Frontier Risks of Open-Weight LLMs", arXiv:2508.03153, August 2025. [arxiv.org/abs/2508.03153](https://arxiv.org/abs/2508.03153)
8. OpenAI, "Introducing gpt-oss-safeguard", October 29, 2025. [openai.com/index/introducing-gpt-oss-safeguard](https://openai.com/index/introducing-gpt-oss-safeguard/)
9. OpenAI, "Technical Report: Performance and baseline evaluations of gpt-oss-safeguard-120b and gpt-oss-safeguard-20b", October 2025. [PDF](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf)
10. gpt-oss-safeguard-120b weights. [huggingface.co/openai/gpt-oss-safeguard-120b](https://huggingface.co/openai/gpt-oss-safeguard-120b)
11. Gao et al. (OpenAI), "Weight-sparse transformers have interpretable circuits", arXiv:2511.13653, November 2025. [arxiv.org/abs/2511.13653](https://arxiv.org/abs/2511.13653)
12. OpenAI Privacy Filter model card and weights. [huggingface.co/openai/privacy-filter](https://huggingface.co/openai/privacy-filter)
13. OpenAI, "Introducing OpenAI Privacy Filter", April 2026. [openai.com/index/introducing-openai-privacy-filter](https://openai.com/index/introducing-openai-privacy-filter/)
14. OpenAI, GPT-2 code and model release notes. [github.com/openai/gpt-2](https://github.com/openai/gpt-2)
15. OpenAI, "GPT-2: 6-month follow-up", August 2019. [openai.com/blog/gpt-2-6-month-follow-up](https://openai.com/blog/gpt-2-6-month-follow-up/)
16. OpenAI, "GPT-2: 1.5B release", November 5, 2019. [openai.com/index/gpt-2-1-5b-release](https://openai.com/index/gpt-2-1-5b-release/)
17. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), arXiv:2103.00020. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
18. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), arXiv:2212.04356; code and changelog at [github.com/openai/whisper](https://github.com/openai/whisper)
19. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", arXiv:2309.17453. [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
