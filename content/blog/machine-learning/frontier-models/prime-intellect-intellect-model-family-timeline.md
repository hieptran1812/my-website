---
title: "The INTELLECT Model Family: Every Prime Intellect Release from INTELLECT-1 to INTELLECT-3.1"
date: "2026-10-14"
publishDate: "2026-10-14"
description: "A chronological reference to Prime Intellect's INTELLECT models, from the first decentralized 10B pretrain to a GLM-4.5-Air MoE post-trained with large-scale RL, with verified dates, specs, variant tables, and the infrastructure each release depended on."
tags:
  [
    "prime-intellect",
    "intellect-3",
    "intellect-2",
    "intellect-1",
    "decentralized-training",
    "diloco",
    "reinforcement-learning",
    "prime-rl",
    "mixture-of-experts",
    "model-timeline",
    "open-weights",
  ]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 31
---

Most model families on this blog are a story about one lab making its models bigger and smarter. The INTELLECT family is a different kind of story. Prime Intellect did not start with a model and then build tooling around it. It started with a thesis about *where* training can happen, namely on compute that is scattered across providers, countries and trust boundaries, and every INTELLECT release is that thesis being tested at the next layer of the training stack. INTELLECT-1 asked whether pretraining could survive the public internet. INTELLECT-2 asked the same question of reinforcement learning. INTELLECT-3 changed the question: it trained inside one cluster, but treated the RL *environments* as an open, community-sourced artifact.

That shift matters for how you read the family. If you compare INTELLECT-1 to INTELLECT-3 on benchmark scores alone, you are comparing a 10B dense model trained from scratch on 1T tokens to a 106B Mixture-of-Experts post-trained on top of someone else's base model. The numbers are not on the same axis. What *is* on the same axis is the infrastructure: each model is the receipt for a piece of the stack (OpenDiLoCo, PCCL, prime-rl, TOPLOC, SHARDCAST, SYNTHETIC-1/2, verifiers, the Environments Hub, Prime Sandboxes) that shipped first.

The diagram below is the mental model for the whole post: models on the top lane, the infrastructure and data releases that made them possible on the bottom lane.

![Two-lane timeline of Prime Intellect releases from July 2024 to early 2026: INTELLECT models on top, infrastructure and datasets below, with arrows from each enabler to the model that used it](/imgs/blogs/prime-intellect-intellect-model-family-timeline-1.webp)

A note on sourcing before we start. Every date and spec in this post was checked against a primary source: the arXiv technical reports, Prime Intellect's own blog, the INTELLECT-3 technical report PDF, and the model cards and `config.json` files on the [PrimeIntellect Hugging Face org](https://huggingface.co/PrimeIntellect). Where two primary sources disagree, I say so. Where a fact is not published (INTELLECT-3.1's benchmark scores, for example), I mark it as unpublished rather than fill the gap. I have not included any model that does not exist as of October 2026; there is no INTELLECT-4 at the time of writing.

## The timeline at a glance

Here is the whole family on one table. "Release" means the public weights date; for the two models that were trained in public, the run's announcement date is listed separately because the gap between the two is part of the story.

| Release | Date | What it is | Size | Base | Training regime | License |
| --- | --- | --- | --- | --- | --- | --- |
| OpenDiLoCo (framework) | Jul 10–11, 2024 | Open replication of DiLoCo on Hivemind | 1B-scale experiments | n/a | Low-communication data parallel | Open source |
| INTELLECT-1 run starts | Oct 11, 2024 | Public launch of the decentralized pretrain | 10B | From scratch | DiLoCo over the internet | n/a |
| **INTELLECT-1** | Nov 29, 2024 | First collaboratively pretrained 10B LLM | 10B dense | From scratch (Llama-3 style) | Pretrain 1T tokens + Arcee post-train | Apache 2.0 |
| **INTELLECT-MATH** | Jan 21, 2025 (weights Jan 17) | Math reasoning model, SFT then RL | 7B dense | Qwen 7B math base | SFT on verified QwQ traces + PRIME implicit-PRM RL | MIT |
| TOPLOC | Jan 28, 2025 | Locality-sensitive hashing for verifiable inference | n/a | n/a | Verification layer | Open source |
| SYNTHETIC-1 | Feb 6 / Feb 20, 2025 | 2M verified DeepSeek-R1 traces, crowd-generated | n/a | n/a | Distributed data generation | Open data |
| INTELLECT-2 run starts | Apr 15, 2025 | Public launch of the decentralized RL run | 32B | QwQ-32B | Async RL over permissionless workers | n/a |
| **INTELLECT-2** | May 11, 2025 | First 32B model trained through globally distributed RL | 32B dense | QwQ-32B | prime-rl, GRPO variant, TOPLOC, SHARDCAST | Apache 2.0 |
| PCCL | May 20, 2025 | Fault-tolerant collectives for the public internet | n/a | n/a | Communication library | Open source |
| SYNTHETIC-2 | Jun 23 / Jul 10, 2025 | 4M verified reasoning traces | n/a | n/a | Pipeline-parallel inference on consumer GPUs | Open data |
| Environments Hub | Aug 27, 2025 | Open registry of RL environments (verifiers modules) | n/a | n/a | Environment ecosystem | Open source |
| **INTELLECT-3** | Nov 26, 2025 | 106B-A12B MoE, SFT + large-scale RL | 106B total / 12B active | GLM-4.5-Air-Base | Two SFT stages + async RL on 512 H200 | MIT |
| **INTELLECT-3.1** | Early 2026 (HF repo created Jan 20, last updated Feb 18) | Continued RL on INTELLECT-3 | 106B total / 12B active | INTELLECT-3 (GLM-4.5-Air-Base lineage) | Additional RL on math, code, SWE, agentic | MIT |

Five models, seven infrastructure milestones. The bolded rows are the ones you can download and run; the rest are the scaffolding. The two lanes alternate with surprising regularity: almost every model release is preceded, by one to three months, by the infrastructure piece it leans on hardest.

## Before the models: OpenDiLoCo (July 2024)

The family's prehistory is a single paper. In July 2024 Prime Intellect published [OpenDiLoCo](https://arxiv.org/abs/2407.07852) (Jaghouar, Ong, Hagemann; arXiv 2407.07852, submitted July 10, 2024), an open-source reimplementation of DeepMind's DiLoCo algorithm on top of the Hivemind library. The [blog announcement](https://www.primeintellect.ai/blog/opendiloco) followed on July 11.

The idea behind DiLoCo is simple enough to state in one paragraph. Ordinary data-parallel training all-reduces gradients after *every* step, which is why it needs datacenter-grade interconnect (see [the interconnect physics post](/blog/machine-learning/distributed-training/the-interconnect-physics) for why that bandwidth is non-negotiable in the normal regime). DiLoCo instead lets each worker take many local optimizer steps with an inner optimizer (AdamW), then computes a *pseudo-gradient*, the difference between the weights before and after the local phase, and synchronizes only that. An outer optimizer (Nesterov momentum SGD) applies the averaged pseudo-gradient. Communication drops by roughly the number of inner steps.

Written out, with $H$ inner steps per round, worker $k$ starting from shared weights $\theta^{(t)}$ and ending its local phase at $\theta_k^{(t,H)}$:

$$
\Delta^{(t)} = \frac{1}{K} \sum_{k=1}^{K} \left( \theta^{(t)} - \theta_k^{(t,H)} \right), \qquad \theta^{(t+1)} = \text{OuterOpt}\left(\theta^{(t)}, \Delta^{(t)}\right)
$$

where $K$ is the number of workers and OuterOpt is Nesterov SGD. The all-reduce happens once per $H$ steps instead of once per step.

What OpenDiLoCo claimed, in its abstract: training across two continents and three countries at 90 to 95% compute utilization, scaling the method to three times the size of the original DiLoCo work (into the billion-parameter range), and all-reducing pseudo-gradients in FP16 without degradation. That last finding is the seed of INTELLECT-1's int8 all-reduce.

| OpenDiLoCo fact | Value | Source |
| --- | --- | --- |
| arXiv | 2407.07852, July 10, 2024 | [arXiv](https://arxiv.org/abs/2407.07852) |
| Substrate | Hivemind | arXiv abstract |
| Geography | 2 continents, 3 countries | arXiv abstract |
| Compute utilization | 90 to 95% | arXiv abstract |
| Precision finding | FP16 all-reduce of pseudo-gradients with no degradation | arXiv abstract |

## INTELLECT-1 (November 29, 2024): pretraining over the internet

### What it was

[INTELLECT-1](https://huggingface.co/PrimeIntellect/INTELLECT-1) is a 10B-parameter dense decoder trained from scratch on 1 trillion tokens by compute contributed from multiple independent providers. The run was [announced on October 11, 2024](https://www.primeintellect.ai/blog/intellect-1), the weights and the [release post](https://www.primeintellect.ai/blog/intellect-1-release) landed on November 29, 2024, and the [technical report](https://arxiv.org/abs/2412.01152) (arXiv 2412.01152) was submitted on December 2, 2024. Training took 42 days.

The architecture is deliberately unremarkable, a Llama-3-style dense transformer, because the point was the training system, not the network:

| Spec | INTELLECT-1 |
| --- | --- |
| Parameters | 10B, dense |
| Layers | 42 |
| Hidden size | 4,096 |
| Attention heads | 32 |
| Context length | 8,192 |
| Vocabulary | 128,256 (Llama-3 vocabulary size) |
| Training tokens | 1T |
| Data mix | 55% FineWeb-Edu, 20% Stack, 10% FineWeb, 10% DCLM-baseline, 5% OpenWebMath |
| Inner / outer optimizer | AdamW / Nesterov SGD (DiLoCo, 100 inner steps) |
| LR schedule | WSD, inner LR 7.5e-5, plus auxiliary max-z-loss |
| Peak compute | Up to 14 concurrent nodes, up to 112 H100s |
| Geography | 5 countries, 3 continents; 30 independent compute providers joined and left over the run |
| License | Apache 2.0 |

One small discrepancy in the primary sources: the release blog lists the code slice as "Stack v2", while the model card says "Stack V1". Both agree on the 20% share.

### What changed versus everything before it

Nothing in INTELLECT-1 is new as a *model*. Everything is new as a *system*. The [PRIME framework](https://github.com/PrimeIntellect-ai/prime) described in the report added three things on top of OpenDiLoCo:

1. **ElasticDeviceMesh.** A process-group manager that separates a fast *local* group (FSDP2 within a node, over NVLink) from a slow, dynamic *global* group (DiLoCo across the internet). Nodes can join and leave the global group without killing the run, which is the property you need when 30 providers come and go over 42 days. The [fault tolerance post](/blog/machine-learning/distributed-training/fault-tolerance-and-elastic-training) covers why elastic membership is so much harder than it sounds inside a single cluster; doing it across providers is harder still.
2. **Custom int8 all-reduce** of the pseudo-gradients. Combined with 100 inner steps, the report claims roughly a **400× reduction** in communication bandwidth versus standard data-parallel training at comparable quality.
3. **Live checkpoint recovery**, so a newly joining node pulls the current weights from peers instead of from shared storage.

The report gives compute utilization of 83 to 96% depending on geography (96% for a US-only configuration with a 103 s median sync, 85.6% transatlantic, 83% global) and model FLOPS utilization of 36.2 to 41.4%.

A quick sanity check on why 100 inner steps buys so much: if a synchronous all-reduce of 10B BF16 parameters moves on the order of 20 GB per step, then doing it once every 100 steps instead of every step, in int8 instead of BF16, cuts the bytes per step by $100 \times 2 = 200\times$ before any other trick. The reported 400× is the same order of magnitude, which is what you would expect once the remaining engineering overlaps and compresses the rest.

### Variants

| Variant | Type | Notes | Link |
| --- | --- | --- | --- |
| INTELLECT-1 | Base | 10B, 8K context | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-1) |
| INTELLECT-1-Instruct | Chat | Post-trained with Arcee AI: 16 SFT runs, 8 DPO runs, merged with MergeKit; EvolKit-75K among the data | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-1-Instruct) |
| INTELLECT-1-fp32 | Base, FP32 weights | Full-precision copy | [HF org](https://huggingface.co/PrimeIntellect) |
| INTELLECT-1-step-N | Intermediate checkpoints | Steps 15.5k through 88k published | [HF org](https://huggingface.co/PrimeIntellect) |

### How good was it

Honestly, not very, and the model card says so. On the base-model table, INTELLECT-1 scores 37.5 MMLU, 26.12 GPQA, 8.1 GSM8K, 52.13 ARC-C and 72.26 HellaSwag. That puts it ahead of MPT-7B and Falcon-7B and behind LLaMA2-7B on most columns. The Instruct model reaches 49.89 MMLU and 38.58 GSM8K, roughly LLaMA2-7B-Chat territory.

| Model | MMLU | GPQA | GSM8K | ARC-C | HellaSwag |
| --- | --- | --- | --- | --- | --- |
| INTELLECT-1 (base) | 37.5 | 26.12 | 8.1 | 52.13 | 72.26 |
| LLaMA2-7B | 45.3 | 25.89 | 13.5 | 54.10 | 78.64 |
| MPT-7B | 26.8 | 25.67 | 8.3 | 46.67 | 77.41 |
| INTELLECT-1-Instruct | 49.89 | 28.32 | 38.58 | 54.52 | 71.42 |
| LLaMA2-7B-Chat | 47.20 | 28.57 | 23.96 | 53.33 | 78.69 |

The right way to read this is as a feasibility result. A 10B model on 1T tokens in late 2024 was never going to be competitive with models trained on 15T tokens in a single cluster. The interesting number is not 37.5 MMLU; it is that the loss curve looked like a normal loss curve while nodes were joining and leaving across three continents.

## INTELLECT-MATH (January 21, 2025): the detour into reasoning

### What it was

[INTELLECT-MATH](https://huggingface.co/PrimeIntellect/INTELLECT-MATH) is the release most people forget, and it is the hinge of the family. It is a 7B math reasoning model, published on Hugging Face on January 17, 2025 and explained in the January 21 post ["Distributed Training in the Inference-Time-Compute Paradigm"](https://www.primeintellect.ai/blog/intellect-math). It was *not* trained on decentralized compute; the post treats distributed training as the next step, not as something this model did.

| Spec | INTELLECT-MATH |
| --- | --- |
| Parameters | 7B (Qwen2 architecture, 28 layers per `config.json`) |
| Base | Qwen 7B math base model (blog: "Qwen-7B-Math") |
| Context | 4,096 (SFT traces filtered to this length) |
| SFT data | NuminaMath questions with QwQ-32B responses filtered for correctness and length (~733k to 766k examples; the post gives both numbers) |
| RL | PRIME recipe (implicit process reward model), same RL data and setup as Eurus-2-7B-PRIME |
| License | MIT |

A naming trap: the RL recipe used here is **PRIME** from the PRIME-RL research group (the Eurus-2 authors, implicit process rewards), which is unrelated to Prime Intellect's own **prime-rl** framework that trains INTELLECT-2 and INTELLECT-3. Same letters, different projects.

### Why it matters in the lineage

The finding was about initialization: a better SFT stage made the RL stage both better and much cheaper. An early checkpoint (step 47) matched Eurus-2-7B-PRIME, which needed 592 RL steps, a roughly 10× reduction in RL compute. The model card table:

| Benchmark | INTELLECT-MATH (step 255) | INTELLECT-MATH (step 47) | Eurus-2-PRIME (step 592) | INTELLECT-MATH-SFT | Qwen-2.5-Math |
| --- | --- | --- | --- | --- | --- |
| MATH-500 | 82.0 | 81.6 | 79.2 | 72.8 | 79.8 |
| OlympiadBench | 49.5 | 46.7 | 42.1 | 39.1 | 40.7 |
| AIME 2024 | 26.7 | 26.7 | 26.7 | 16.6 | 13.3 |
| AMC | 60.2 | 57.8 | 57.8 | 45.8 | 50.6 |
| Average | 51.6 | 50.1 | 48.9 | 41.6 | 43.8 |

This is where the company's direction visibly turns. After DeepSeek-R1 (released the same week, see [the R1 deep-dive](/blog/machine-learning/frontier-models/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning)), the valuable compute was no longer only pretraining FLOPs; it was rollout generation for RL, which is embarrassingly parallel and tolerant of slow links. That is exactly the shape of workload a decentralized network is good at. The [inference-time scaling post](/blog/machine-learning/scaling-laws/reasoning-models-rl-test-time-scaling) covers why that shift happened industry-wide.

| Variant | Notes | Link |
| --- | --- | --- |
| INTELLECT-MATH | Final RL model | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-MATH) |
| INTELLECT-MATH-SFT | SFT-only starting point | [HF org](https://huggingface.co/PrimeIntellect) |
| INTELLECT-MATH-step-47 | Early RL checkpoint that matched Eurus-2-PRIME | [HF org](https://huggingface.co/PrimeIntellect) |

## The bridge: TOPLOC and SYNTHETIC-1 (January to February 2025)

Two infrastructure releases sit between INTELLECT-MATH and INTELLECT-2, and INTELLECT-2 cannot be understood without both.

**TOPLOC** ([blog, Jan 28, 2025](https://www.primeintellect.ai/blog/toploc)) is a locality-sensitive hashing scheme for verifiable inference. The problem it solves: if you outsource rollout generation to untrusted machines, how do you know the worker actually ran the model you sent it, at the precision you asked for, instead of returning cheap garbage? TOPLOC commits to a compact hash of intermediate activations that a verifier can check far more cheaply than re-running generation. By the SYNTHETIC-2 run, Prime Intellect reported TOPLOC v2 verification as about 25× cheaper than re-running inference.

**SYNTHETIC-1** ([announced Feb 6](https://www.primeintellect.ai/blog/synthetic-1), [released Feb 20, 2025](https://www.primeintellect.ai/blog/synthetic-1-release)) is roughly 2 million reasoning traces generated by DeepSeek-R1 (671B) on community-contributed 8×H200 nodes, each verified. The task mix: 777k math, 144k algorithmic coding, 70k real-world software engineering, 313k open-ended STEM QA and 61k synthetic code understanding. It shipped with a 900k-sample SFT subset, an 11.5k-pair preference set, and a demonstration model, SYNTHETIC-1-SFT-7B. Its RL-ready tasks became part of INTELLECT-2's training pool.

| Release | Date | What it contributed to INTELLECT-2 |
| --- | --- | --- |
| TOPLOC | Jan 28, 2025 | Verification of rollouts from untrusted inference workers |
| SYNTHETIC-1 | Feb 20, 2025 | Verifiable math and coding tasks for the RL dataset |
| Planetary-scale inference post | Apr 28, 2025 | Pipeline-parallel inference across consumer GPUs (later used for SYNTHETIC-2) |

## INTELLECT-2 (May 11, 2025): RL over permissionless compute

### What it was

[INTELLECT-2](https://huggingface.co/PrimeIntellect/INTELLECT-2) is a 32B reasoning model produced by running reinforcement learning on top of [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B), with rollouts generated by a permissionless, heterogeneous swarm of inference workers. The run was [announced April 15, 2025](https://www.primeintellect.ai/blog/intellect-2), the weights and [release post](https://www.primeintellect.ai/blog/intellect-2-release) came on May 11, 2025, and the [technical report](https://arxiv.org/abs/2505.07291) (arXiv 2505.07291) was submitted May 12, 2025.

| Spec | INTELLECT-2 |
| --- | --- |
| Parameters | 32B dense (Qwen2 architecture, 64 layers) |
| Base | QwQ-32B |
| Max positions | 40,960 (`config.json`) |
| RL data | 285k verifiable math and coding tasks from NuminaMath-1.5, DeepScaleR and SYNTHETIC-1 |
| Algorithm | GRPO variant: two-sided token-ratio clipping, offline + online difficulty filtering, aggressive gradient clipping |
| Asynchrony | Two-step asynchronous RL: weight broadcast overlaps with inference and training |
| Length control | Binary task reward plus a length reward targeting thinking budgets of 2,000 to 10,000 tokens |
| Trainer | prime-rl, PyTorch FSDP2 |
| Inference | vLLM on untrusted workers |
| Weight distribution | SHARDCAST, an HTTP tree-topology broadcast |
| Rollout verification | TOPLOC |
| License | Apache 2.0 |

### What changed versus INTELLECT-1

The thing being decentralized moved from *gradients* to *rollouts*. In INTELLECT-1 every participant was a trainer holding a full replica (FSDP-sharded within its node). In INTELLECT-2 the trainer is a comparatively small trusted cluster; the swarm only does inference. That reshapes the communication pattern completely:

- **Downstream**, the trainer must push fresh 32B weights to every inference worker each policy step. SHARDCAST does this as a tree of HTTP relays rather than having every worker hit the trainer directly.
- **Upstream**, workers return rollouts plus TOPLOC proofs. Verifiers check the proofs and the task rewards before anything reaches the trainer.
- **In between**, the two-step asynchrony means the trainer is allowed to learn from rollouts generated by a policy up to two steps stale. That is the price of not stalling the whole network every time new weights propagate.

![Three-column comparison of what each INTELLECT generation distributed: pseudo-gradients for INTELLECT-1, weights out and rollouts back for INTELLECT-2, and environments from an open hub for INTELLECT-3](/imgs/blogs/prime-intellect-intellect-model-family-timeline-2.webp)

The two-sided clipping change is worth a sentence. Standard PPO/GRPO clipping bounds the probability ratio only on the side that matches the advantage sign; with large negative-advantage tokens the unclipped side can produce gradient spikes, and asynchrony makes large ratios more likely. Clipping both sides is a cheap stabilizer. (The GRPO family's origin is covered in [the DeepSeekMath post](/blog/machine-learning/frontier-models/deepseekmath-data-pipeline-and-grpo-origin).)

### Results

From the model card:

| Model | AIME24 | AIME25 | LiveCodeBench v5 | GPQA-Diamond | IFEval |
| --- | --- | --- | --- | --- | --- |
| **INTELLECT-2** | **78.8** | 64.9 | **67.8** | 66.8 | 81.5 |
| QwQ-32B | 76.6 | 64.8 | 66.1 | 66.3 | 83.4 |
| Qwen-R1-Distill-32B | 69.9 | 58.4 | 55.1 | 65.2 | 72.0 |
| DeepSeek-R1 | 78.6 | 65.1 | 64.1 | 71.6 | 82.7 |

Read it honestly: INTELLECT-2 is QwQ-32B plus a couple of points on AIME24 and LiveCodeBench, with IFEval slightly *worse*. QwQ-32B was already heavily RL-trained, so there was not much headroom. The model card also admits that length control did not fully take: it recommends appending "Think for 10000 tokens before giving a response." and notes that other budgets make little difference. As with INTELLECT-1, the result is the system, and it was the first time a 32B RL run had been executed this way.

| Variant | Notes | Link |
| --- | --- | --- |
| INTELLECT-2 | 32B reasoning model, BF16 | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-2) |
| Community quantizations | 13 quantized derivatives listed on the model page (third-party) | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-2) |

## The second bridge: PCCL, SYNTHETIC-2 and the Environments Hub (May to August 2025)

Between INTELLECT-2 and INTELLECT-3 came three more pieces, and they point in two directions at once.

**PCCL** ([blog, May 20, 2025](https://www.primeintellect.ai/blog/pccl); [arXiv 2505.14065](https://arxiv.org/abs/2505.14065)), the Prime Collective Communications Library, is the low-level answer to INTELLECT-1's pain. NCCL assumes stable, fast links; PCCL assumes peers can die at any moment. It supports fault-tolerant retries of collectives with the surviving peers, dynamic membership where new peers wait to be accepted and then resume at the same point in the training loop, shared-state hashing so drifting peers get a retransmission, and bandwidth-aware topology optimization. The post reports up to 45 Gbit/s across Western European datacenters and 25 Gbit/s intercontinental across North America and Western Europe, and notes that INTELLECT-1 had needed a VPN workaround that cost throughput. It is built to support DiLoCo, including an async variant with one-step delayed updates.

**SYNTHETIC-2** ([announced Jun 23](https://www.primeintellect.ai/blog/synthetic-2), [released Jul 10, 2025](https://www.primeintellect.ai/blog/synthetic-2-release)) produced four million verified reasoning traces. DeepSeek-R1-0528 generated the SFT subset; smaller Qwen3 models computed pass rates to difficulty-label the RL subset. Large models were split across consumer GPUs with pipeline parallelism, and over 1,250 GPUs (many RTX 3090 and 4090 cards) joined over three days. INTELLECT-3's code environment "makes heavy use" of SYNTHETIC-2.

**Environments Hub** ([blog, Aug 27, 2025](https://www.primeintellect.ai/blog/environments)) is the conceptual turn. An RL environment in the verifiers library is an installable Python module: a dataset, a rollout loop (single-turn, multi-turn or tool-calling), a rubric of reward functions, and a `load_environment` entry point. The Hub is a versioned public registry of those modules. The INTELLECT-3 report argues that environments should play the role datasets play in pretraining: decoupled from training code, pinnable, shareable, ablatable.

| Release | Date | Points toward |
| --- | --- | --- |
| PCCL | May 20, 2025 | Future decentralized pretraining (not used by INTELLECT-3) |
| SYNTHETIC-2 | Jul 10, 2025 | INTELLECT-3 code environment, SFT-style trace data |
| Environments Hub | Aug 27, 2025 | INTELLECT-3's entire environment mix |
| INTELLECT-3-Base | Sep 9, 2025 (HF) | A clone of GLM-4.5-Air-Base with a custom chat template |

That last row is a small tell. On September 9, 2025 Prime Intellect published [INTELLECT-3-Base](https://huggingface.co/PrimeIntellect/INTELLECT-3-Base), which its card describes as a clone of GLM-4.5-Air-Base with a chat template adapted from Qwen3-Coder. It is not a separately trained model, so I don't count it as a release, but it dates the decision: by early September the next model would be a post-train of Zhipu's open MoE base, not a new pretrain.

## INTELLECT-3 (November 26, 2025): large-scale RL on an open MoE base

### What it was

[INTELLECT-3](https://huggingface.co/PrimeIntellect/INTELLECT-3) is a 106B-total, 12B-active Mixture-of-Experts reasoning model, post-trained from Z.ai's [GLM-4.5-Air-Base](https://huggingface.co/zai-org/GLM-4.5-Air-Base) with two SFT stages followed by large-scale reinforcement learning. The [announcement](https://www.primeintellect.ai/blog/intellect-3) is dated November 26, 2025, the Hugging Face repos were created the same day, and the [technical report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) is a 27-page PDF hosted by Prime Intellect (not on arXiv at the time of writing).

Because the network is GLM-4.5-Air, the architecture comes straight from the base model's `config.json`:

| Spec | INTELLECT-3 |
| --- | --- |
| Parameters | 106B total, 12B active (HF metadata rounds to 107B) |
| Architecture | `Glm4MoeForCausalLM` (GLM-4.5-Air) |
| Layers | 46 (first layer dense, rest MoE) |
| Experts | 128 routed + 1 shared, 8 routed active per token |
| Attention | 96 query heads, 8 KV heads, head dim 128, partial RoPE (factor 0.5) |
| Hidden size | 4,096 |
| MTP | 1 next-token prediction layer (inherited) |
| Max positions | 131,072 |
| Trained context | SFT stage 1 at 65K, agentic SFT at 98K (context parallel), RL at 65,536 max |
| Vocabulary | 151,552 |
| License | MIT |

If you want the reasons behind that shape (deep-narrow, unusually many heads, loss-free routing, MTP, Muon), the [GLM-4.5 architecture post](/blog/machine-learning/frontier-models/glm-4-5-architecture) covers it, and [the GLM-4.5 training post](/blog/machine-learning/frontier-models/training-glm-4-5-distillation-agentic-rl) covers Z.ai's own post-training, which is the thing INTELLECT-3 is competing against. The broader [GLM lineage](/blog/machine-learning/frontier-models/glm-lineage-frontier-llm-technique) explains where the Air base came from.

### What changed versus INTELLECT-2

![INTELLECT-3 training pipeline from GLM-4.5-Air-Base through general SFT at 65K, agentic SFT at 98K and large-scale RL at 65K, fed by Environments Hub modules, ending in INTELLECT-3 and then INTELLECT-3.1](/imgs/blogs/prime-intellect-intellect-model-family-timeline-3.webp)

Almost everything about the *training topology* reversed. INTELLECT-3 was not trained over the internet. Both stages, including ablations, ran on a single 512-GPU H200 cluster (64 nodes) over about two months. What remained open and distributed was the *inputs*: the environments came from the Hub, and the code-execution substrate (Prime Sandboxes) was purpose-built to run thousands of untrusted programs per training step.

**SFT, stage 1 (general reasoning).** A full epoch over a large mix dominated by NVIDIA's Nemotron-Post-Training-Dataset-v1 (math, code, science, tool splits) and AM-DeepSeek-R1-0528-Distilled (chat and instruction following), both containing R1-0528 reasoning traces. Roughly 33M tokens per step at 65K context, Muon optimizer with LR 5e-5 and weight decay 0.01, FSDP world size 64 with DP replicate 8 across all 512 GPUs. The report's data table lists about 78.1B math tokens, 94.3B code tokens and 32B science tokens among the sources.

**SFT, stage 2 (agentic).** A smaller set (SWE-Swiss, Toucan Tool, and traces synthesized from Hub environments with R1-0528), two epochs, 800 steps, Muon with a linearly decayed LR, and context parallelism to train at 98K tokens. Its stated second purpose is pushing effective context beyond 65K.

**RL.** The settings the report gives:

| RL setting | Value |
| --- | --- |
| Batch | 256 prompts × 16 rollouts |
| Max context | 65,536 |
| Max off-policy steps | 8 (rollouts from older policies are discarded) |
| Optimizer | Muon, LR 1e-6 |
| Nodes | 60 × 8 H200: 16 trainer nodes, 44 inference nodes (about 1:3) |
| Step time | about 1,500 s at 65K with in-flight weight updates; more than 2× slower without them |
| Objective | Masked token-level importance sampling (IcePop-style), mask outside $[\alpha, \beta] = [0.5, 5]$ |
| Advantage | Reward minus group mean (no std normalization in the formula given) |
| Filtering | Online difficulty filtering; prompts with pass rate 1 moved to an easy pool and not resampled |

Two of these deserve unpacking.

**Masking instead of clipping.** The trainer and the inference engine compute token probabilities with different kernels and precisions, so even with identical weights the ratio $\pi_{\text{train}} / \pi_{\text{infer}}$ drifts away from 1. The report says this mismatch could crash runs "multiple days into the experiments." Its fix is to zero out (mask) tokens whose ratio leaves $[0.5, 5]$, and to mask whole rollouts if any token's ratio falls below $10^{-5}$. The report positions this as similar in spirit to MiniMax's CISPO (see [the MiniMax-M1 post](/blog/machine-learning/frontier-models/minimax-m1-cispo-test-time-compute)) but with masking rather than clipping to avoid noisy updates from extreme ratios. An early ablation in the report shows GSPO's reward collapsing under 8-step asynchrony while CISPO (their algorithm at the time) held.

**In-flight weight updates.** With rollouts tens of thousands of tokens long, waiting for every sequence in a batch to finish before swapping weights leaves inference GPUs idle. prime-rl instead uses continuous batching and interrupts generation to load new weights mid-sequence, so a single trajectory can be produced by several consecutive policies. `max_off_policy_steps = 8` bounds how stale any token can be.

The environment mix, all published on the Hub:

| Environment | Size | Verification |
| --- | --- | --- |
| `i3-math` | 21.2K problems (Skywork-OR1, AceReason-Math, DAPO, ORZ-Hard) | math-verify, plus CompassVerifier-7B as LLM judge on rule-based negatives |
| `i3-code` | 8.6K problems, heavy use of SYNTHETIC-2 | Up to 15 tests per problem in Prime Sandboxes, 4,000+ concurrent sandboxes |
| `i3-science` | 29.3K problems from MegaScience | math-verify plus LLM judge |
| `i3-logic` | 11.6K problems, 29 task types from SynLogic | Task-specific verifiers |
| `deepdive` | 1K SFT + 2.2K RL questions from Z.ai's DeepDive | Search/click/open/finish tools, binary reward |
| `deepswe`, `mini-swe-agent-plus` | R2E-Gym, SWE-smith, Multi-SWE-bench formats | Repository test suite, up to 200 turns, 20,000+ prebuilt repo images |

Every environment is difficulty-annotated by the solve rate of a small Qwen3-4B model, and samples that are too easy are filtered out at various stages. Failed sandboxes mask the completion rather than assigning a zero reward, which is a detail that matters: a sandbox infrastructure failure should not teach the policy that its correct code was wrong.

The scaling details are instructive for anyone running long-context RL. The report says FSDP with aggressive activation checkpointing and FlashAttention 3 reached 48K sequence length; offloading the checkpointed activations (about 18 GB at 48K for 46 layers of hidden size 4,096) to CPU got it to 72K on the same hardware with no MFU loss. Ring-attention context parallelism reached 256K at a CP degree of 2, but it halved the data-parallel degree and showed accuracy degradation, so they ruled it out for the production RL run. For the general picture of those trade-offs, see [sequence and context parallelism](/blog/machine-learning/distributed-training/sequence-and-context-parallelism) and [the ZeRO/FSDP memory model](/blog/machine-learning/distributed-training/zero-and-fsdp-the-memory-model).

### Results

From Table 2 of the technical report (all evaluated by Prime Intellect through its own Environments Hub evaluation implementations, against API services where needed; two cells carry asterisks in the report's table):

| Model | AIME24 | AIME25 | LCB v6 | GPQA | HLE | MMLU-Pro |
| --- | --- | --- | --- | --- | --- | --- |
| **INTELLECT-3** | **90.8** | **88.0** | **69.3** | 74.4 | 14.6 | 81.9 |
| GLM-4.5-Air | 84.6 | 82.0 | 61.5 | 73.3 | 13.3 | 73.9 |
| GLM-4.5 | 85.8 | 83.3 | 64.5 | 77.0 | 14.8 | 83.5 |
| GLM-4.6 | 92.0 | 90.3 | 73.0 | 78.8 | 13.3 | 83.1 |
| DeepSeek R1 0528 | 83.2 | 73.4 | 62.5 | 77.5 | 15.9 | 75.3 |
| DeepSeek V3.2 | 88.1 | 84.7 | 71.6 | 81.4 | 17.9 | 84.6 |
| GPT-OSS 120B | 75.8 | 77.7 | 69.9 | 70.0 | 10.6 | 67.1 |

The model card also lists MATH-500 at 98.1. The apples-to-apples comparison is the GLM-4.5-Air row: same base, Z.ai's own post-training versus Prime Intellect's. INTELLECT-3 beats it on every column, by 6 points on AIME24, 6 on AIME25, 7.8 on LiveCodeBench v6 and 8 on MMLU-Pro. It does not beat GLM-4.6 (a larger model), and the report shows reasoning scores still trending up at the end of RL, which the authors read as headroom.

| Variant | Notes | Link |
| --- | --- | --- |
| INTELLECT-3 | BF16, needs 2× H200 per the card | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-3) |
| INTELLECT-3-FP8 | FP8 quantization, fits on a single H200 | [HF](https://huggingface.co/PrimeIntellect/INTELLECT-3-FP8) |
| INTELLECT-3-RL (dataset) | RL data release | [HF org](https://huggingface.co/PrimeIntellect) |

Serving notes from the report and card: the model always reasons (there is no reasoning-effort switch; a `<|think|>` token is appended by the chat template), and vLLM should use the `qwen3_coder` tool-call parser and the `deepseek_r1` reasoning parser.

```bash
# Serve INTELLECT-3 (BF16) on two H200s, per the model card
vllm serve PrimeIntellect/INTELLECT-3 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1

# The FP8 checkpoint fits on one H200
vllm serve PrimeIntellect/INTELLECT-3-FP8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1
```

## INTELLECT-3.1 (early 2026): more RL on the same base

### What it was

[INTELLECT-3.1](https://huggingface.co/PrimeIntellect/INTELLECT-3.1) is, in the model card's words, "a continued training of INTELLECT-3 with additional reinforcement learning on math, coding, software engineering, and agentic tasks." Same 106B-A12B GLM-4.5-Air architecture (the `config.json` is identical in shape: 46 layers, 128 routed experts, 8 active, 131,072 positions), same prime-rl and verifiers stack, same MIT license.

This is the release with the thinnest primary-source record, so here is exactly what is and is not known:

| Fact | Status |
| --- | --- |
| Exists on Hugging Face | Yes, public, MIT |
| Repo created | January 20, 2026 (Hugging Face API) |
| Last updated | February 18, 2026 (Hugging Face API) |
| Public announcement date | **Not found.** No dedicated blog post; Prime Intellect's blog index has no INTELLECT-3.1 entry. The card links back to the INTELLECT-3 blog and technical report. |
| Benchmark scores | **Not published** on the card |
| Training compute, steps, new environments | **Not published** |
| Serving | 2× H200 with vLLM, same parsers as INTELLECT-3 |

So the honest summary is: INTELLECT-3.1 is a quiet point release, a longer RL run on the INTELLECT-3 checkpoint, consistent with the INTELLECT-3 report's own observation that benchmark curves had not plateaued. Treat any third-party benchmark numbers you see for it with care, because Prime Intellect has not published its own.

### What came after (and what did not)

As of October 14, 2026 there is no INTELLECT-4 and no INTELLECT-3.x beyond 3.1 on the PrimeIntellect Hugging Face org or the company blog. The 2026 blog is about the platform: Lab (Feb 10, 2026), verifiers v1 (Jul 12, 2026), prime-rl performance work at 1T-parameter scale (Jun 22, 2026), a GLM-5.2 RL weight-transfer post (Sep 3, 2026), Prime Sandboxes on microVMs (Sep 23, 2026) and Prime Inference (Oct 2, 2026). The company has been training *other people's* models on its stack rather than shipping new INTELLECT checkpoints. If an INTELLECT-4 lands, it will most likely be visible first on the Hugging Face org.

## Side-by-side: the four generations

| | INTELLECT-1 | INTELLECT-MATH | INTELLECT-2 | INTELLECT-3 / 3.1 |
| --- | --- | --- | --- | --- |
| Date | Nov 2024 | Jan 2025 | May 2025 | Nov 2025 / early 2026 |
| Stage trained | Pretrain from scratch | SFT + RL | RL only | SFT (2 stages) + RL |
| Base | None | Qwen 7B math | QwQ-32B | GLM-4.5-Air-Base |
| Params | 10B dense | 7B dense | 32B dense | 106B MoE, 12B active |
| Context | 8K | 4K | 40,960 positions | 131,072 positions, RL at 65K |
| Compute topology | Up to 14 nodes over the internet, 3 continents | Single cluster | Trusted trainer + permissionless inference swarm | Single 512×H200 cluster |
| What was decentralized | Gradient sync | Nothing | Rollouts | Environments (sourced from open Hub) |
| Key infra | PRIME, ElasticDeviceMesh, int8 all-reduce | PRIME implicit-PRM recipe | prime-rl, TOPLOC, SHARDCAST | prime-rl (in-flight updates, Muon), verifiers, Hub, Sandboxes |
| Headline number | 400× less communication | 10× fewer RL steps to match Eurus-2 | AIME24 78.8 (QwQ 76.6) | AIME24 90.8 (GLM-4.5-Air 84.6) |
| License | Apache 2.0 | MIT | Apache 2.0 | MIT |

## How the family evolved

Five through-lines run across the generations.

**1. The decentralized unit kept moving up the stack.** Pretraining over the internet (INTELLECT-1) requires syncing the most bytes and gets the least benefit from slow, heterogeneous compute. RL rollouts (INTELLECT-2) are almost the opposite: inference-heavy, verifiable, tolerant of staleness. Environments (INTELLECT-3) are not compute at all; they are the training *signal*, and they decentralize through a package registry rather than a network protocol. The figure in the INTELLECT-2 section shows the three shapes. Each move followed the economics: put the distributed part where communication is cheapest relative to value.

**2. From from-scratch to post-train of the best available open base.** INTELLECT-1 trained every weight. INTELLECT-2 inherited QwQ-32B. INTELLECT-3 inherited GLM-4.5-Air-Base and its architecture wholesale, including Muon-friendly weights; the report cites prior work that post-training with Muon works best when the model was pretrained with Muon, which is a reason to keep Z.ai's optimizer choice rather than switch to AdamW. The lesson: a small lab gets more frontier capability per GPU-hour by post-training a strong open base than by pretraining a weak one.

**3. Asynchrony became a first-class training parameter.** INTELLECT-1 tolerated 100 local steps between syncs. INTELLECT-2 tolerated rollouts two policy steps stale. INTELLECT-3 tolerated up to 8, with in-flight weight swaps inside a single trajectory. Each increase came with a stabilizer: Nesterov outer momentum, two-sided clipping, then token-level importance masking. If you are building RL infrastructure, "how stale can my data be, and what corrects for it" is the design question this family answers most concretely.

**4. Infrastructure first, model second, always open.** Every model release was preceded by its enabling tool, and the tools were released as open source before the model existed. The models are, in effect, the integration tests. That is why the family's benchmarks were modest until INTELLECT-3: the earlier releases were optimizing for proving the system worked, not for leaderboard position.

**5. Verification moved from proofs about compute to proofs about answers.** INTELLECT-2 needed TOPLOC because it did not trust the *workers*. INTELLECT-3 trusts its own cluster but invests heavily in verifying the *outputs*: rule-based math checking backed by an LLM judge, sandboxed test execution with failure masking, repository test suites for SWE. The trust boundary moved from the hardware to the reward.

## Which variant to pick today

A short, opinionated guide as of October 2026:

| You want | Pick | Why |
| --- | --- | --- |
| The strongest INTELLECT model for reasoning, code or agentic tool use | **INTELLECT-3.1** if you are comfortable with no published evals; otherwise **INTELLECT-3** | 3.1 is more RL on the same checkpoint; 3 has the documented numbers |
| To run on a single 141 GB GPU | **INTELLECT-3-FP8** | Card states it fits one H200 |
| A small, inspectable reasoning model for RL experiments | **INTELLECT-MATH** (7B) | Cheap, well documented SFT-to-RL ablation, MIT |
| A 32B dense reasoner | Probably not INTELLECT-2 in 2026 | It was QwQ-32B plus a few points; newer 32B-class open reasoners have moved on |
| A research artifact on decentralized pretraining | **INTELLECT-1** plus its intermediate checkpoints | The checkpoint series is the interesting part, not the final scores |
| To reproduce the recipe rather than use the weights | prime-rl + verifiers + the `i3-*` environments | The environments are the most reusable artifact of the family |

One caution on INTELLECT-3 versus its base: if you already serve Z.ai's own GLM-4.5-Air, INTELLECT-3 is a drop-in architecture (same `glm4_moe` config, same vocabulary) with better reasoning scores per the report, but it always thinks, so expect longer outputs and plan your latency budget accordingly. Newer GLM releases (see [GLM-5.2](/blog/machine-learning/frontier-models/glm-5-2-architecture-techniques-evals)) are a different and larger class of model.

## Facts I could not verify

- **INTELLECT-3.1 public release date, benchmarks and training details.** Only Hugging Face repo timestamps (created Jan 20, 2026; updated Feb 18, 2026) are available; no announcement post or eval table exists in primary sources.
- **INTELLECT-1 code data version.** The release blog says Stack v2, the model card says Stack V1.
- **INTELLECT-MATH exact base checkpoint.** The blog says "Qwen-7B-Math"; the `config.json` confirms a Qwen2-architecture 7B with 4K positions, consistent with Qwen2.5-Math-7B, but the card does not name the exact repo.
- **INTELLECT-2 trained context length.** Only the inherited `max_position_embeddings` of 40,960 is published.

## References

**Technical reports**

- Jaghouar, Ong, Hagemann. [OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training](https://arxiv.org/abs/2407.07852). arXiv 2407.07852, July 2024.
- Prime Intellect. [INTELLECT-1 Technical Report](https://arxiv.org/abs/2412.01152). arXiv 2412.01152, December 2024.
- Prime Intellect. [INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning](https://arxiv.org/abs/2505.07291). arXiv 2505.07291, May 2025.
- Prime Intellect. [PCCL technical report](https://arxiv.org/abs/2505.14065). arXiv 2505.14065, May 2025.
- Prime Intellect Team. [INTELLECT-3: Technical Report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf). November 2025.

**Official blog posts**

- [OpenDiLoCo](https://www.primeintellect.ai/blog/opendiloco) (Jul 11, 2024)
- [INTELLECT-1 launch](https://www.primeintellect.ai/blog/intellect-1) (Oct 11, 2024) and [INTELLECT-1 release](https://www.primeintellect.ai/blog/intellect-1-release) (Nov 29, 2024)
- INTELLECT-MATH: [Distributed Training in the Inference-Time-Compute Paradigm](https://www.primeintellect.ai/blog/intellect-math) (Jan 21, 2025)
- [TOPLOC](https://www.primeintellect.ai/blog/toploc) (Jan 28, 2025)
- [SYNTHETIC-1](https://www.primeintellect.ai/blog/synthetic-1) (Feb 6, 2025) and [SYNTHETIC-1 release](https://www.primeintellect.ai/blog/synthetic-1-release) (Feb 20, 2025)
- [INTELLECT-2 launch](https://www.primeintellect.ai/blog/intellect-2) (Apr 15, 2025) and [INTELLECT-2 release](https://www.primeintellect.ai/blog/intellect-2-release) (May 11, 2025)
- [PCCL](https://www.primeintellect.ai/blog/pccl) (May 20, 2025)
- [SYNTHETIC-2](https://www.primeintellect.ai/blog/synthetic-2) (Jun 23, 2025) and [SYNTHETIC-2 release](https://www.primeintellect.ai/blog/synthetic-2-release) (Jul 10, 2025)
- [Environments Hub](https://www.primeintellect.ai/blog/environments) (Aug 27, 2025)
- [INTELLECT-3](https://www.primeintellect.ai/blog/intellect-3) (Nov 26, 2025)
- [Blog index](https://www.primeintellect.ai/blog) (for post-2025 platform releases)

**Model cards and code**

- [PrimeIntellect Hugging Face organization](https://huggingface.co/PrimeIntellect)
- [INTELLECT-1](https://huggingface.co/PrimeIntellect/INTELLECT-1), [INTELLECT-1-Instruct](https://huggingface.co/PrimeIntellect/INTELLECT-1-Instruct)
- [INTELLECT-MATH](https://huggingface.co/PrimeIntellect/INTELLECT-MATH)
- [INTELLECT-2](https://huggingface.co/PrimeIntellect/INTELLECT-2)
- [INTELLECT-3-Base](https://huggingface.co/PrimeIntellect/INTELLECT-3-Base), [INTELLECT-3](https://huggingface.co/PrimeIntellect/INTELLECT-3), [INTELLECT-3-FP8](https://huggingface.co/PrimeIntellect/INTELLECT-3-FP8), [INTELLECT-3.1](https://huggingface.co/PrimeIntellect/INTELLECT-3.1)
- [prime (INTELLECT-1 framework)](https://github.com/PrimeIntellect-ai/prime), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- [GLM-4.5-Air-Base](https://huggingface.co/zai-org/GLM-4.5-Air-Base) (INTELLECT-3's base) and [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) (INTELLECT-2's base)
