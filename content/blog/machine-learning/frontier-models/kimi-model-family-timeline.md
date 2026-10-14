---
title: "The Kimi Model Family: Every Moonshot AI Release from Kimi Chat to K3"
publishDate: "2026-10-14"
date: "2026-10-14"
category: "machine-learning"
subcategory: "Frontier Models"
tags:
  - kimi
  - moonshot-ai
  - model-timeline
  - mixture-of-experts
  - open-weights
  - linear-attention
  - reinforcement-learning
  - multimodal
  - agentic
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/kimi-model-family-timeline-1.webp"
description: "A dated, source-linked catalog of every Kimi release from Moonshot AI, from the 2023 long-context chatbot through k1.5, Moonlight, Kimi-VL, K2, Kimi Linear, K2.5, K2.6, K2.7 Code and the 2.8T-parameter K3, with variant tables, what changed each generation, and which checkpoint to pick today."
readTime: 28
---

> [!tldr]
> - **Two lines, one family.** Moonshot runs a flagship line (k1.5, K2, K2 Thinking, K2.5, K2.6, K2.7 Code, K3) and a stream of small open research models (Moonlight, Kimi-VL, Kimi-Audio, Kimi Linear). The small models are where techniques get proven; the flagships are where they ship.
> - **Closed to open in six months.** k1.5 (January 2025) never released weights. K2 (July 2025) put a 1T-parameter MoE on Hugging Face under a modified MIT license, and every flagship since has been open-weight.
> - **The scale ladder.** 1T total / 32B active from K2 through K2.7 Code, then 2.8T total / 104B active in K3. Context went 128K, then 256K (K2-0905 onward), then 1M (Kimi Linear, then K3).
> - **The architecture ladder.** Muon optimizer (Moonlight) became MuonClip (K2). Kimi Delta Attention (Kimi Linear) and Attention Residuals (a March 2026 paper) both landed in K3. INT4 QAT (K2 Thinking) became MXFP4 QAT (K3).
> - **Latest as of 2026-10-14:** Kimi K3, announced 2026-07-16 and open-weighted 2026-07-27. Nothing newer has shipped.

Moonshot AI is one of the hardest labs to keep a clean mental model of. It ships a flagship roughly every two to three months, and in between it drops small, oddly specific open models (a 16B MoE to prove an optimizer, a 48B hybrid to prove an attention variant, a 72B prover built on someone else's base) that only make sense once you see which flagship absorbed them. This post is the reference I wanted: every release in order, dated against a primary source, with a variant table per generation and a link to the deep-dive on this blog wherever one exists. The timeline below is the whole story in one picture; the rest of the post is the detail behind each box.

A note on method. Dates are the earliest of: the official blog post, the first public commit on the `moonshotai` (or `AI-MO`) Hugging Face org, or the arXiv v1 submission. Where only secondary press covers a fact (mostly the 2023 to 2024 chatbot era), I say so in the text instead of stating it as fact.

## The release timeline at a glance

![Swim-lane timeline of every Kimi release from Mooncake in June 2024 to K3 in July 2026, split into flagship, small open research models, specialists and agents, and systems papers](/imgs/blogs/kimi-model-family-timeline-1.webp)

The figure splits the family into four lanes, and the lanes are the right way to read Moonshot. The **flagship lane** is the one the press covers. The **small open research model** lane is where you see the next flagship coming: Moonlight in February 2025 predicted K2's optimizer, and Kimi Linear in October 2025 predicted K3's attention. The **specialist lane** (provers, a SWE model, a research agent) is where Moonshot tests post-training recipes on narrow domains. The **systems lane** holds the infrastructure and technique papers (Mooncake, MoBA, Attention Residuals) that are not models at all but explain why the models look the way they do.

Here is the same information as a flat list, which is what you want when you are looking for a date:

| Date | Release | Lane | Weights |
| --- | --- | --- | --- |
| Oct 2023 (press) | Kimi Chat | Product | Closed |
| Mar 2024 (press) | Kimi Chat 2M-character context | Product | Closed |
| 2024-06-24 | Mooncake serving paper | Systems | Code (kvcache-ai) |
| 2025-01-20 | Kimi k1.5 | Flagship | Closed |
| 2025-02-18 | MoBA | Technique | Code |
| 2025-02-22 | Moonlight-16B-A3B | Research model | Open, MIT |
| 2025-04-09 | Kimi-VL-A3B (Instruct, Thinking) | Research model | Open, MIT |
| 2025-04-15 | Kimina-Prover Preview | Specialist | Distills only |
| 2025-04-25 | Kimi-Audio-7B | Research model | Open, MIT |
| 2025-06-16 | Kimi-Dev-72B | Specialist | Open, MIT |
| 2025-06-20 | Kimi-Researcher | Agent | Closed (product) |
| 2025-06-21 | Kimi-VL-A3B-Thinking-2506 | Research model | Open, MIT |
| 2025-07-10 | Kimina-Prover-72B | Specialist | Open, MIT |
| 2025-07-11 | Kimi K2 (Base, Instruct) | Flagship | Open, modified MIT |
| 2025-09-03 | Kimi-K2-Instruct-0905 | Flagship | Open, modified MIT |
| 2025-10-30 | Kimi Linear 48B-A3B | Research model | Open, MIT |
| 2025-11-06 | Kimi K2 Thinking | Flagship | Open, modified MIT |
| 2026-01-27 | Kimi K2.5 | Flagship | Open, modified MIT |
| 2026-03-16 | Attention Residuals paper | Technique | Code |
| 2026-04-20 | Kimi K2.6 | Flagship | Open, modified MIT |
| 2026-06-11 | Kimi K2.7 Code | Flagship (coding) | Open, modified MIT |
| 2026-07-16 | Kimi K3 (weights 2026-07-27) | Flagship | Open, Kimi K3 License |

That is 22 entries. Roughly half are models you can download. The rest of this post walks them in order.

## Era 0: the long-context chatbot (2023 to mid-2024)

### Kimi Chat (October 2023) and the 2M-character upgrade (March 2024)

Moonshot's first product was not a model release in the open-weights sense. It was **Kimi Chat**, a consumer assistant whose selling point was context length: press coverage at launch described it as handling around 200,000 Chinese characters in a single conversation, which was well beyond what most chat products offered in late 2023. In March 2024, Moonshot announced an upgrade to roughly 2 million Chinese characters, which the South China Morning Post reported as starting in an invite-only beta.

I could not find a Moonshot-authored primary source (blog post, paper, or model card) for either the launch date or the context figures. Treat both numbers as press-reported, not verified. No parameter counts, architecture details or weights were ever published for the models behind Kimi Chat.

What matters for the rest of this timeline is the positioning. Moonshot built its brand on **long context first**, and the infrastructure paper that came next exists because serving long contexts to millions of users is an infrastructure problem before it is a modeling problem.

### Mooncake (June 2024)

**Mooncake** ([arXiv 2407.00079](https://arxiv.org/abs/2407.00079), v1 submitted 2024-06-24) is the serving architecture behind Kimi. It is a KVCache-centric, disaggregated design: prefill and decode run on separate pools, and the KV cache is treated as a first-class distributed resource that lives across GPU memory, CPU DRAM and SSD, rather than a per-request scratch buffer. The paper reports up to 525% higher throughput in simulated long-context scenarios and says the design let Kimi handle 75% more requests under real workloads.

There is no model here, but Mooncake is the reason Moonshot could afford to keep pushing context length on a consumer product. If you want the full mechanism, read the [Mooncake paper walkthrough](/blog/paper-reading/large-language-model/mooncake), and for the general pattern see [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).

## Era 1: the reasoning pivot and the research-model spring (January to June 2025)

This is the busiest stretch in the family's history. In six months Moonshot shipped a closed reasoning flagship, two architecture papers, and six open research or specialist models.

### Kimi k1.5 (January 2025)

**Released:** blog 2025-01-20; technical report [arXiv 2501.12599](https://arxiv.org/abs/2501.12599) v1 2025-01-22. Repo: [github.com/MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5) (README only, no weights).

**What changed:** k1.5 is Moonshot's answer to the o1-style reasoning wave, and it landed the same week as DeepSeek-R1. The recipe is RL with a deliberately simple setup: no Monte Carlo tree search, no learned value function, no process reward model. The two ideas that carry it are scaling the RL context window to **128K tokens** and using **partial rollouts** (reusing chunks of earlier trajectories so very long generations do not stall the RL loop). It is multimodal, trained jointly on text and vision data. The report also introduced **long2short** methods to distill long chain-of-thought behavior into a short-CoT model that answers with far fewer tokens.

The report gives AIME 2024 at 77.5 and MATH-500 at 96.2 for the long-CoT model.

| Variant | Params | Context | Modality | Type | Weights |
| --- | --- | --- | --- | --- | --- |
| k1.5 long-CoT | Not disclosed | 128K (RL context) | Text + vision | Reasoning | Closed |
| k1.5 short-CoT | Not disclosed | Not disclosed | Text + vision | Fast answer | Closed |

Parameter count was never disclosed. Whether k1.5 was ever served through a public API (rather than only inside the Kimi app) is something I could not verify from a primary source. Deep-dive: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/machine-learning/frontier-models/kimi-k1-5).

### MoBA (February 2025)

**Released:** [arXiv 2502.13189](https://arxiv.org/abs/2502.13189), 2025-02-18; code at [github.com/MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA).

**Mixture of Block Attention** applies the MoE routing idea to attention: the context is cut into blocks, and each query routes to a top-k subset of blocks instead of attending to everything. The paper presents it as having been deployed to serve Kimi's long-context requests. It is a technique, not a model, but it sits on the long-context through-line that Kimi Linear and K3 later finish. Walkthrough: [MoBA](/blog/paper-reading/large-language-model/moba).

### Moonlight (February 2025)

**Released:** Hugging Face 2025-02-22; blog 2025-02-23; paper "Muon is Scalable for LLM Training", [arXiv 2502.16982](https://arxiv.org/abs/2502.16982), v1 2025-02-24. Weights: [huggingface.co/moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B).

**What changed:** Moonlight is the first open-weight Moonshot language model, and it exists to prove an optimizer. The paper takes Muon (an orthogonalized-momentum optimizer that had only been shown on small models) and makes it work at scale, mainly by adding weight decay and matching per-parameter update RMS so Muon can reuse AdamW-tuned learning rates. The headline claim is roughly 2x compute efficiency over AdamW at matched quality.

The model itself is a DeepSeek-V3-style MoE trained on **5.7T tokens**. The marketing name says 16B total and 3B active; the model card gives **15.29B total and 2.24B activated**, excluding embeddings. The config has 64 routed experts with 6 active per token, and the paper notes that intermediate checkpoints were released too.

| Variant | Total / active | Context | Type | License |
| --- | --- | --- | --- | --- |
| Moonlight-16B-A3B | 15.29B / 2.24B (excl. embeddings) | 8K | Base | MIT |
| Moonlight-16B-A3B-Instruct | 15.29B / 2.24B | 8K | Instruct | MIT |

The 8K context is small by 2025 standards, and that is the tell: this is a research artifact for the optimizer, not a product. Five months later Muon, with one more fix, trained a trillion-parameter model. Walkthrough: [Muon and Moonlight](/blog/paper-reading/large-language-model/muon-moonlight).

### Kimi-VL (April 2025)

**Released:** Hugging Face 2025-04-09; technical report [arXiv 2504.07491](https://arxiv.org/abs/2504.07491), v1 2025-04-10. Weights: [huggingface.co/moonshotai/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct).

**What changed:** Kimi-VL is a small vision-language MoE built on the Moonlight language backbone. The language decoder activates about 2.8B parameters, and the vision side is **MoonViT**, a roughly 400M-parameter native-resolution vision encoder (released separately as MoonViT-SO-400M, MIT, on 2025-04-10). Context is **128K**, long enough for long documents and multi-image inputs. It shipped with two post-training variants on day one: an Instruct model and a Thinking model trained with long-CoT SFT plus RL, which is the k1.5 recipe applied to a small open VLM.

On 2025-06-21 Moonshot pushed **Kimi-VL-A3B-Thinking-2506**, a meaningful refresh: the card reports MathVision at 56.9 (up 20.1 points), adds video understanding (VideoMMMU 65.2), raises the per-image budget to 3.2M pixels (4x the original), and uses fewer thinking tokens.

| Variant | Total / active | Context | Modality | Type | License |
| --- | --- | --- | --- | --- | --- |
| Kimi-VL-A3B-Instruct | ~16B / ~3B (2.8B LLM + 0.4B ViT) | 128K | Image + text | Instruct | MIT |
| Kimi-VL-A3B-Thinking | ~16B / ~3B | 128K | Image + text | Thinking | MIT |
| Kimi-VL-A3B-Thinking-2506 | ~16B / ~3B | 128K (per original release) | Image + video + text | Thinking | MIT |
| MoonViT-SO-400M | ~400M | n/a | Vision encoder | Encoder | MIT |

The piece to remember is MoonViT. The same 400M encoder reappears as the vision tower of K2.5, K2.6 and K2.7 Code nine months later. Deep-dive: [Kimi-VL](/blog/machine-learning/frontier-models/kimi-vl).

### Kimina-Prover Preview (April 2025)

**Released:** [arXiv 2504.11354](https://arxiv.org/abs/2504.11354), 2025-04-15, with Project Numina. Code: [github.com/MoonshotAI/Kimina-Prover-Preview](https://github.com/MoonshotAI/Kimina-Prover-Preview). Weights live under the [AI-MO](https://huggingface.co/AI-MO) Hugging Face org.

**What changed:** Kimina-Prover is a Lean 4 theorem prover, RL-trained from Qwen2.5-72B, and the first time Moonshot applied its large-scale RL recipe to formal math. The Preview reported **80.7% on miniF2F at pass@8192**. The design point the paper argues for is "formal reasoning patterns": the model thinks in natural language, interleaved with Lean, before committing to a proof.

The 72B Preview weights were **not** released. What shipped in April were small distillations and an autoformalizer:

| Variant | Base | Released | Weights |
| --- | --- | --- | --- |
| Kimina-Prover-Preview (72B) | Qwen2.5-72B | Paper only | Not released |
| Kimina-Prover-Preview-Distill-1.5B | Qwen2.5 family | HF 2025-04-09 | Open |
| Kimina-Prover-Preview-Distill-7B | Qwen2.5-Math-7B-Instruct | HF 2025-04-10 | Open, Apache-2.0 |
| Kimina-Autoformalizer-7B | Qwen2.5 family | HF 2025-04-13 | Open |

The full 72B shipped in July (below). Deep-dive: [Kimina-Prover](/blog/machine-learning/frontier-models/kimina-prover).

### Kimi-Audio (April 2025)

**Released:** [arXiv 2504.18425](https://arxiv.org/abs/2504.18425) and Instruct weights 2025-04-25; base weights 2025-04-27. Code: [github.com/MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio).

**What changed:** Kimi-Audio is a universal audio foundation model (speech recognition, audio understanding, audio QA and speech conversation in one model). It is initialized from Qwen2.5-7B, uses a **12.5 Hz audio tokenizer** on the input side and a **flow-matching streaming detokenizer** on the output side, and was pretrained on more than **13 million hours** of audio. The architecture mixes discrete semantic tokens with continuous acoustic features in the input so the LLM sees both.

| Variant | Params | Modality | Type | License |
| --- | --- | --- | --- | --- |
| Kimi-Audio-7B | ~7B (Qwen2.5-7B init) | Audio + text in, audio + text out | Base | MIT (Qwen-derived code Apache-2.0) |
| Kimi-Audio-7B-Instruct | ~7B | Audio + text in, audio + text out | Instruct | MIT (Qwen-derived code Apache-2.0) |

There has been no Kimi-Audio 2 as of this writing. Audio is the one modality Moonshot has not folded into its flagships. Deep-dive: [Kimi-Audio](/blog/machine-learning/frontier-models/kimi-audio).

### Kimi-Dev-72B (June 2025)

**Released:** Hugging Face 2025-06-16; blog 2025-06-17. Weights: [huggingface.co/moonshotai/Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B). Later technical report: "Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents", [arXiv 2509.23045](https://arxiv.org/abs/2509.23045), 2025-09-27.

**What changed:** Kimi-Dev is a software-engineering specialist built on Qwen2.5-72B. It reported **60.4% on SWE-bench Verified**, which was the open-model state of the art at release. The interesting claim, made explicit in the September report, is that you do not need to train an agent end to end from the start: train the component skills (localize the bug, write the fix, write the test) in a structured "agentless" setup with RL on verifiable outcomes, and those skills transfer as a prior to agentic scaffolds later.

| Variant | Base | Params | Type | License |
| --- | --- | --- | --- | --- |
| Kimi-Dev-72B | Qwen2.5-72B | 72B dense | Coding / SWE | MIT |

Deep-dive: [Kimi-Dev](/blog/machine-learning/frontier-models/kimi-dev).

### Kimi-Researcher (June 2025)

**Released:** [moonshotai.github.io/Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/), 2025-06-20.

**What changed:** Kimi-Researcher is an autonomous research agent trained with end-to-end agentic RL on an internal k-series model, rather than by wiring a prompted model into a workflow. The page reports **26.9% Pass@1 on Humanity's Last Exam**, with the agent averaging about 23 reasoning steps and exploring 200+ URLs per task.

| Variant | Base | Weights |
| --- | --- | --- |
| Kimi-Researcher | Internal k-series model (undisclosed) | Product only |

The page said Moonshot would open-source the base model and the RL-trained agent model "in the following months". I could not confirm that this happened; no Kimi-Researcher repo exists on the `moonshotai` Hugging Face org as of 2026-10-14. What did happen is that its end-to-end agentic RL became the post-training spine of K2 Thinking. Deep-dive: [Kimi-Researcher](/blog/machine-learning/frontier-models/kimi-researcher).

### Kimina-Prover-72B (July 2025)

**Released:** the GitHub README announces it on 2025-07-10 (HF repo created 2025-07-09). Weights: [huggingface.co/AI-MO/Kimina-Prover-72B](https://huggingface.co/AI-MO/Kimina-Prover-72B).

The full prover, now with weights, reported **84.0% on miniF2F-test at pass@32**, a much stronger number at a much smaller sampling budget than the Preview's pass@8192. Around it came a family of small provers:

| Variant | Released (HF) | Type |
| --- | --- | --- |
| Kimina-Prover-72B | 2025-07-09/10 | RL prover, MIT, Qwen2.5-72B base |
| Kimina-Prover-Distill-8B, Distill-1.7B | 2025-07-04 | Distilled |
| Kimina-Prover-RL-1.7B | 2025-08-04 | RL |
| Kimina-Prover-Distill-0.6B | 2025-08-07 | Distilled |
| Kimina-Prover-RL-0.6B | 2025-08-12 | RL |

## Era 2: the open trillion-parameter flagship (July to November 2025)

### Kimi K2 (July 2025)

**Released:** blog and Hugging Face 2025-07-11. Technical report "Kimi K2: Open Agentic Intelligence", [arXiv 2507.20534](https://arxiv.org/abs/2507.20534), v1 2025-07-28. Weights: [huggingface.co/moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct).

**What changed versus k1.5:** almost everything. K2 is open-weight, it is a disclosed architecture, and it is a non-thinking "reflex" model aimed at agentic tool use rather than a reasoning model.

- **Architecture.** An ultra-sparse MoE with **1.04T total parameters and 32B activated**: 384 experts, 8 routed per token plus 1 shared expert, with MLA attention. The design is close to DeepSeek-V3 but sparser (more experts, fewer attention heads).
- **Optimizer.** **MuonClip**, which is Moonlight's Muon plus **QK-Clip**: when a head's max attention logit grows too large, the query and key projection weights are rescaled after the update. This is what let Moonshot report pretraining on **15.5T tokens with zero loss spikes**.
- **Data.** Rephrasing pipelines that rewrite high-quality knowledge and math text to get more learning signal per unique token, instead of repeating epochs.
- **Post-training.** A large-scale synthetic agentic-data pipeline (simulated tools and environments) plus joint RL that combines verifiable rewards with a self-critique rubric reward for non-verifiable tasks.
- **License.** Modified MIT: MIT, plus an attribution requirement that applies only to very large commercial deployments.

| Variant | Total / active | Context | Type | License |
| --- | --- | --- | --- | --- |
| Kimi-K2-Base | 1.04T / 32B | 128K | Base | Modified MIT |
| Kimi-K2-Instruct (later referred to as "0711") | 1.04T / 32B | 128K | Instruct, non-thinking | Modified MIT |

There is no repo named "Kimi-K2-Instruct-0711". That label appears only in the 0905 model card's comparison table, to distinguish the July checkpoint. Two deep-dives cover K2 from different angles: [Kimi K2: how a trillion-parameter MoE learns to use tools](/blog/machine-learning/frontier-models/kimi-k2) and the report-focused [Kimi K2: Open Agentic Intelligence](/blog/machine-learning/frontier-models/kimi-k2-open-agentic-intelligence).

### Kimi-K2-Instruct-0905 (September 2025)

**Released:** Hugging Face 2025-09-03; blog index 2025-09-05. Weights: [huggingface.co/moonshotai/Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905).

**What changed:** same 1T / 32B weights architecture, continued post-training. Context doubled from 128K to **256K**, and the card emphasizes better agentic coding and frontend generation. This is the first appearance of the 256K window that every flagship until K3 would use.

| Variant | Total / active | Context | Type | License |
| --- | --- | --- | --- | --- |
| Kimi-K2-Instruct-0905 | 1T / 32B | 256K | Instruct, non-thinking | Modified MIT |

Alongside it, Moonshot open-sourced two pieces of tooling worth knowing about: **checkpoint-engine** (2025-09-08), which updates the weights of a 1T-parameter model across an inference fleet in about 20 seconds (the piece that makes on-policy RL at this scale practical), and **K2-Vendor-Verifier** (2025-09-09), which checks whether third-party API providers serve K2 faithfully, especially its tool-calling.

### Kimi Linear (October 2025)

**Released:** [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) and Hugging Face, 2025-10-30. Weights: [huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct).

**What changed:** this is the research model that predicted K3. Kimi Linear introduces **Kimi Delta Attention (KDA)**, a linear-attention layer that refines Gated DeltaNet with a finer-grained (channel-wise) gate, and mixes it with full MLA attention at a **3:1 ratio** (three KDA layers per MLA layer). Trained on **5.7T tokens**, the 48B-total / 3B-active model reports matching or beating a full-attention baseline at equal training budget, while cutting KV cache by up to 75% and decoding up to 6x faster at 1M context.

| Variant | Total / active | Context | Type | License |
| --- | --- | --- | --- | --- |
| Kimi-Linear-48B-A3B-Base | 48B / 3B | 1M | Base | MIT |
| Kimi-Linear-48B-A3B-Instruct | 48B / 3B | 1M | Instruct | MIT |

Deep-dive: [Kimi Linear](/blog/machine-learning/frontier-models/kimi-linear). For the layer it builds on, see [Gated Delta Networks](/blog/paper-reading/large-language-model/gated-delta-networks); for the CUDA side, [FlashKDA: Kimi Delta Attention kernels](/blog/machine-learning/open-source-library/flashkda-kimi-delta-attention-cuda-kernels) (FlashKDA was open-sourced 2026-04-20).

### Kimi K2 Thinking (November 2025)

**Released:** blog and Hugging Face 2025-11-06. Weights: [huggingface.co/moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

**What changed:** K2 Thinking is the K2 base turned into a reasoning agent. It interleaves chain-of-thought with tool calls and the card claims it stays coherent across **200 to 300 consecutive tool calls** without human input. It reports **44.9 on Humanity's Last Exam with tools**. The engineering novelty is **native INT4**: quantization-aware training applied to the MoE weights during post-training, so the released INT4 checkpoint is the model as trained rather than a post-hoc compression, and generation runs roughly 2x faster.

| Variant | Total / active | Context | Type | Precision | License |
| --- | --- | --- | --- | --- | --- |
| Kimi-K2-Thinking | 1T / 32B | 256K | Thinking + tools | INT4 (MoE weights, QAT) | Modified MIT |

If you are tracing lineage: k1.5 gave the long-CoT RL recipe, Kimi-Researcher gave end-to-end agentic RL, K2 gave the base, and K2 Thinking is where those three meet. Deep-dive: [Kimi K2 Thinking](/blog/machine-learning/frontier-models/kimi-k2-thinking).

## Era 3: native multimodality and agent swarms (January to June 2026)

### Kimi K2.5 (January 2026)

**Released:** blog 2026-01-27; technical report "Kimi K2.5: Visual Agentic Intelligence", [arXiv 2602.02276](https://arxiv.org/abs/2602.02276), v1 2026-02-02. Weights: [huggingface.co/moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5).

**What changed:**

- **Native multimodality.** K2.5 continues pretraining from K2-Base on about **15T mixed vision and text tokens**, with the MoonViT 400M encoder from Kimi-VL as its vision tower. Vision is in pretraining, not bolted on in post-training.
- **One model, two modes.** Thinking and Instant modes in the same checkpoint, instead of separate Instruct and Thinking repos.
- **Agent Swarm.** The model learns to decompose a task and dispatch parallel sub-agents (the report calls the training method PARL, parallel-agent RL). The blog reports up to 4.5x lower wall-clock latency on wide search tasks versus a single agent, with up to 100 sub-agents and 1,500 coordinated steps.
- **Native INT4** carried over from K2 Thinking.

| Variant | Total / active | Layers / experts | Context | Modality | Modes | License |
| --- | --- | --- | --- | --- | --- | --- |
| Kimi-K2.5 | 1T / 32B | 61 layers, 384 experts, 8 active | 256K | Text + image + video | Thinking, Instant | Modified MIT |

Only the post-trained checkpoint was released; there is no separate K2.5 base model. Deep-dive: [Kimi K2.5](/blog/machine-learning/frontier-models/kimi-k2-5). Moonshot also released the [WorldVQA](/blog/paper-reading/multimodal/worldvqa) benchmark in the same window (2026-02-03) and a broader **Kimi Vendor Verifier** (2026-01-22).

### Attention Residuals (March 2026)

**Released:** [arXiv 2603.15031](https://arxiv.org/abs/2603.15031), 2026-03-16; code at [github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals).

Not a model, but it belongs on the timeline because K3 uses it. **AttnRes** replaces the fixed residual sum (each layer adds its output to a running stream with weight 1) with a learned softmax attention over the outputs of earlier layers, so a layer can choose which depths to read from. **Block AttnRes** groups layers into blocks to keep the cost manageable at large depth, and is the variant intended for large training runs. Walkthrough: [Attention Residuals](/blog/paper-reading/large-language-model/attention-residuals).

### Kimi K2.6 (April 2026)

**Released:** blog and Hugging Face 2026-04-20. Weights: [huggingface.co/moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6). Blog: [kimi.ai/blog/kimi-k2-6](https://www.kimi.ai/blog/kimi-k2-6).

**What changed:** the architecture is unchanged from K2.5 (1T / 32B, 256K, MoonViT 400M, native INT4, Thinking and Instant). The whole delta is post-training for long-horizon work. The Agent Swarm grows from 100 sub-agents and 1,500 steps to **300 sub-agents and 4,000 steps**. The blog's showcase runs are a 12+ hour autonomous Zig inference-optimization session with more than 4,000 tool calls and a 13-hour exchange-core engineering run. The card lists image and video input, with video chat marked as experimental and supported only on the official API.

| Variant | Total / active | Context | Modality | Modes | License |
| --- | --- | --- | --- | --- | --- |
| Kimi-K2.6 | 1T / 32B | 256K | Text + image + video (video experimental) | Thinking, Instant | Modified MIT |

Deep-dive: [Kimi K2.6](/blog/machine-learning/frontier-models/kimi-k2-6).

### Kimi K2.7 Code (June 2026)

**Released:** Hugging Face weights 2026-06-11/12. Weights: [huggingface.co/moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code). Pricing page: [platform.kimi.ai/docs/pricing/chat](https://platform.kimi.ai/docs/pricing/chat).

**What changed:** the card describes K2.7 Code as a coding-focused agentic model built on K2.6, tuned for real-world long-horizon coding tasks. Same 1T / 32B architecture, 256K context and 400M vision encoder. Thinking is always on: there is no Instant mode. The card's evaluations were run through the Kimi Code CLI at a 262,144-token context.

| Variant | Total / active | Context | Modality | Modes | License |
| --- | --- | --- | --- | --- | --- |
| Kimi-K2.7-Code | 1T / 32B | 256K | Text + image | Thinking only | Modified MIT |
| kimi-k2.7-code-highspeed (API) | Same model, faster serving tier | 256K | Same | Thinking only | API |

I found no dated Moonshot blog post for K2.7 Code, so the date above is the Hugging Face release. The HighSpeed API tier appears on the official pricing page, but its launch date (reported in secondary press as 2026-06-15) and the press claim of about 30% fewer thinking tokens versus K2.6 are not verified against a primary source. There is no deep-dive on this blog yet.

## Era 4: the 3-trillion-parameter hybrid (July 2026)

### Kimi K3 (July 2026)

**Released:** announced on the Kimi blog 2026-07-16 (at WAIC), with weights promised by July 27; the Hugging Face repo's first public commit is **2026-07-27**. Weights: [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3). Blog: [kimi.ai/blog/kimi-k3](https://www.kimi.ai/blog/kimi-k3). The technical report is a PDF in the [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) GitHub repo; I found no arXiv version.

**What changed versus K2.x:** this is the first new pretraining run since K2, and it changes nearly every axis at once.

- **Scale.** **2.8T total parameters, 104B activated.** Activated parameters triple versus K2's 32B.
- **Attention.** A hybrid stack of **93 layers: 69 KDA layers and 24 Gated MLA layers**, plus one dense layer. This is Kimi Linear's KDA at flagship scale, at roughly the same 3:1 linear-to-full ratio.
- **Residual stream.** **Attention Residuals** from the March paper.
- **MoE.** A **Stable LatentMoE** with **896 experts, 16 active, 2 shared**, routed in a 3,584-dimensional latent space rather than the full hidden size.
- **Context.** **1,048,576 tokens**, up from 256K.
- **Modality.** Text, image and video in one model, with a new 401M **MoonViT-V2** encoder.
- **Precision.** **MXFP4 weights and MXFP8 activations**, with QAT applied from the SFT stage onward. This is the INT4 idea from K2 Thinking moved to an industry microscaling format.
- **Modes.** Thinking is always on, with a `reasoning_effort` setting of low, high or max (default max).
- **Other.** 160K vocabulary and a SiTU-GLU activation. The blog claims about 2.5x better scaling efficiency than K2.
- **License.** A custom **Kimi K3 License**, not the modified MIT used from K2 through K2.7 Code. Read it before you deploy.

| Variant | Total / active | Experts | Context | Modality | Precision | License |
| --- | --- | --- | --- | --- | --- | --- |
| Kimi-K3 | 2.8T / 104B | 896, 16 active, 2 shared | 1M (1,048,576) | Text + image + video | MXFP4 W / MXFP8 A | Kimi K3 License |

There is no K3 base checkpoint. The pretraining token count is not stated on the model card or the blog, so I leave it out. Deep-dive: [Kimi K3](/blog/machine-learning/frontier-models/kimi-k3). For the MoE design it builds on, see [LatentMoE](/blog/paper-reading/large-language-model/latentmoe).

### After K3

As of 2026-10-14, nothing newer has shipped. The K3 repo has had maintenance commits since release (an encoding fix on 2026-08-20 and evaluation results on 2026-09-02), the Kimi blog has no model post after 2026-07-16, and the platform pricing page lists `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.7-code-highspeed` and `kimi-k2.6`. There is no K3.x, no K3 base, no Kimi-VL 2 and no Kimi-Audio 2. Moonshot did open-source more infrastructure after K3 (nano-kpu, minitriton and the MoonEP expert-parallel library in late July 2026), but no models.

## The flagship spec ladder in one table

![Stepped spec ladder of Kimi flagships from K2 to K3: 1T total and 32B active parameters for six K2 releases while context grows from 128K to 256K, then K3 jumps to 2.8T total, 104B active and 1M context](/imgs/blogs/kimi-model-family-timeline-3.webp)

Pulling only the flagships together makes the step changes obvious:

| Model | Date | Total / active | Context | Attention | Precision | Modality | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k1.5 | 2025-01 | Undisclosed | 128K (RL) | Undisclosed | n/a | Text + vision | Closed |
| K2 | 2025-07 | 1.04T / 32B | 128K | MLA | Standard | Text | Modified MIT |
| K2-0905 | 2025-09 | 1T / 32B | 256K | MLA | Standard | Text | Modified MIT |
| K2 Thinking | 2025-11 | 1T / 32B | 256K | MLA | INT4 QAT | Text | Modified MIT |
| K2.5 | 2026-01 | 1T / 32B | 256K | MLA | INT4 | Text + image + video | Modified MIT |
| K2.6 | 2026-04 | 1T / 32B | 256K | MLA | INT4 | Text + image + video | Modified MIT |
| K2.7 Code | 2026-06 | 1T / 32B | 256K | MLA | INT4 | Text + image | Modified MIT |
| K3 | 2026-07 | 2.8T / 104B | 1M | KDA + Gated MLA (69 : 24) | MXFP4 / MXFP8 QAT | Text + image + video | Kimi K3 License |


Notice how long the 1T / 32B shape held: six releases over twelve months on one pretraining base, with every gain coming from continued pretraining (K2.5's multimodal tokens) or post-training. Then K3 changed every row at once.

## How the family evolved

![Technique lineage graph: Moonlight's Muon feeds K2's MuonClip, Kimi-VL's MoonViT feeds K2.5, K2 Thinking's INT4 QAT feeds K2.5, K2.6 and K3's MXFP4, Kimi Linear's KDA and the Attention Residuals paper feed K3, and k1.5's long-CoT RL feeds K2 Thinking](/imgs/blogs/kimi-model-family-timeline-2.webp)

Five through-lines explain almost every release above.

**1. Small open models are the test bench.** This is the single most useful thing to know about Moonshot, and the lineage figure is built around it. Moonlight (16B, 8K context, a research toy) proved Muon could train an MoE; five months later MuonClip trained K2. Kimi-VL proved MoonViT; K2.5 used MoonViT. Kimi Linear (48B-A3B) proved KDA could match full attention; K3 is 69 KDA layers. When Moonshot releases a small, narrowly scoped open model with a technical report, read it as a preview of the next flagship.

**2. Long context was the brand before reasoning was.** Kimi Chat sold on 200K, then 2M characters. Mooncake and MoBA exist to serve that. The flagships went 128K to 256K and then, with KDA cutting KV cache by up to 75%, to 1M in K3. The architecture change in K3 is ultimately a long-context cost decision, and you can trace it back to the first product.

**3. Closed reasoning, then open everything.** k1.5 was a closed model announced with a paper, in the same week as an open DeepSeek-R1. From K2 on, every flagship shipped weights, and the license stayed a lightly modified MIT through K2.7 Code. K3 is the first break in that pattern: still open-weight, but under its own license. If the license matters to you, it is now a per-release check.

**4. RL moved from reasoning to agency.** k1.5's RL targeted single-shot math and code reasoning. Kimi-Researcher trained an agent end to end. K2 Thinking pushed that to hundreds of sequential tool calls. K2.5 and K2.6 made the policy itself orchestrate parallel sub-agents (100, then 300). K2.7 Code specialized the same machinery for multi-hour coding. The unit of training went from "one answer" to "one long task".

**5. Precision became a training decision.** K2 Thinking introduced INT4 QAT on the MoE weights so the shipped checkpoint was the trained model. K2.5 through K2.7 Code kept native INT4. K3 moved to MXFP4 weights and MXFP8 activations, with QAT from SFT onward. For a 2.8T model, this is what makes self-hosting conceivable at all: the low-bit checkpoint is the reference model, not a degraded copy.

## Which variant to pick today

As of 2026-10-14, this is the short version:

| You need | Pick | Why |
| --- | --- | --- |
| The strongest Kimi model, period | **Kimi K3** (API, or self-host if you have the cluster) | 2.8T / 104B, 1M context, native multimodal. Check the Kimi K3 License first. |
| Strong open model under a permissive license | **Kimi K2.6** | Same capability tier as K2.5 with better long-horizon agents, modified MIT, 1T / 32B is far easier to host than K3 |
| Long-horizon agentic coding | **Kimi K2.7 Code** | Purpose-built on K2.6, always-on thinking, modified MIT |
| A non-thinking, low-latency trillion-scale model | **K2.6 in Instant mode**, or Kimi-K2-Instruct-0905 if you need text-only | Instant mode skips the reasoning trace; 0905 is the last text-only flagship |
| Long-context experiments on one node | **Kimi Linear 48B-A3B** | 3B active, 1M context, MIT, and the cleanest way to study KDA before touching K3 |
| A small open VLM | **Kimi-VL-A3B-Thinking-2506** | ~3B active, video support, MIT |
| Open speech and audio | **Kimi-Audio-7B-Instruct** | The only Kimi audio model; MIT |
| Lean 4 theorem proving | **Kimina-Prover-72B**, or the 1.7B / 0.6B RL models for cheap search | 84.0% miniF2F at pass@32 for the 72B |
| Studying Muon | **Moonlight-16B-A3B** with its intermediate checkpoints | It is the reference artifact for Muon at scale |

Two things I would avoid. Do not build on k1.5 or Kimi-Researcher expecting weights; neither has any. And do not assume a quantized third-party K2.x or K3 endpoint is equivalent to Moonshot's own: that gap is exactly why Moonshot ships a vendor verifier.

## References

Primary sources, in release order.

- **Kimi Chat 2M-character upgrade** (secondary, press only): South China Morning Post, March 2024. [scmp.com](https://scmp.com/tech/big-tech/article/3256109/alibaba-backed-moonshot-ai-claims-breakthrough-expanded-chinese-character-prompt-kimi-chatbot)
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, arXiv 2407.00079 (2024-06-24). [arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079)
- **Kimi k1.5: Scaling Reinforcement Learning with LLMs**, arXiv 2501.12599 (2025-01-22). [arxiv.org/abs/2501.12599](https://arxiv.org/abs/2501.12599) · [GitHub](https://github.com/MoonshotAI/Kimi-k1.5)
- **MoBA: Mixture of Block Attention for Long-Context LLMs**, arXiv 2502.13189 (2025-02-18). [arxiv.org/abs/2502.13189](https://arxiv.org/abs/2502.13189) · [GitHub](https://github.com/MoonshotAI/MoBA)
- **Muon is Scalable for LLM Training** (Moonlight), arXiv 2502.16982 (2025-02-24). [arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982) · [Hugging Face](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
- **Kimi-VL Technical Report**, arXiv 2504.07491 (2025-04-10). [arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) · [Thinking-2506](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506)
- **Kimina-Prover Preview**, arXiv 2504.11354 (2025-04-15). [arxiv.org/abs/2504.11354](https://arxiv.org/abs/2504.11354) · [GitHub](https://github.com/MoonshotAI/Kimina-Prover-Preview)
- **Kimi-Audio Technical Report**, arXiv 2504.18425 (2025-04-25). [arxiv.org/abs/2504.18425](https://arxiv.org/abs/2504.18425) · [GitHub](https://github.com/MoonshotAI/Kimi-Audio)
- **Kimi-Dev-72B** model card (2025-06-16). [Hugging Face](https://huggingface.co/moonshotai/Kimi-Dev-72B) · report: **Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents**, arXiv 2509.23045 (2025-09-27). [arxiv.org/abs/2509.23045](https://arxiv.org/abs/2509.23045)
- **Kimi-Researcher** (2025-06-20). [moonshotai.github.io/Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/)
- **Kimina-Prover-72B** model card (2025-07-10). [Hugging Face](https://huggingface.co/AI-MO/Kimina-Prover-72B)
- **Kimi K2: Open Agentic Intelligence**, arXiv 2507.20534 (2025-07-28; weights 2025-07-11). [arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- **Kimi-K2-Instruct-0905** model card (2025-09-03). [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)
- **checkpoint-engine** (2025-09-08). [GitHub](https://github.com/MoonshotAI/checkpoint-engine)
- **Kimi Linear: An Expressive, Efficient Attention Architecture**, arXiv 2510.26692 (2025-10-30). [arxiv.org/abs/2510.26692](https://arxiv.org/abs/2510.26692) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- **Kimi K2 Thinking** model card (2025-11-06). [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- **Kimi K2.5: Visual Agentic Intelligence**, arXiv 2602.02276 (2026-02-02; blog 2026-01-27). [arxiv.org/abs/2602.02276](https://arxiv.org/abs/2602.02276) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.5)
- **Attention Residuals**, arXiv 2603.15031 (2026-03-16). [arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031) · [GitHub](https://github.com/MoonshotAI/Attention-Residuals)
- **Kimi K2.6** blog and model card (2026-04-20). [kimi.ai/blog/kimi-k2-6](https://www.kimi.ai/blog/kimi-k2-6) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.6)
- **Kimi K2.7 Code** model card (2026-06-11). [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2.7-Code) · [Pricing](https://platform.kimi.ai/docs/pricing/chat)
- **Kimi K3** blog (2026-07-16) and model card (2026-07-27). [kimi.ai/blog/kimi-k3](https://www.kimi.ai/blog/kimi-k3) · [Hugging Face](https://huggingface.co/moonshotai/Kimi-K3) · [GitHub](https://github.com/MoonshotAI/Kimi-K3)
- Moonshot research index: [kimi.ai/blog](https://www.kimi.ai/blog) · organizations: [huggingface.co/moonshotai](https://huggingface.co/moonshotai), [github.com/MoonshotAI](https://github.com/MoonshotAI)
