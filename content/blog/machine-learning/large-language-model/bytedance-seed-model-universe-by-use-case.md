---
title: "ByteDance Seed, mapped: the whole model family organized by what each model is for"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A field guide to ByteDance Seed's sprawling model lineup — eight use-case families, the release cadence, what is open versus API-only, and the shared RL-and-infrastructure platform that ties it all together."
tags:
  [
    "bytedance-seed",
    "model-survey",
    "large-language-model",
    "multimodal",
    "video-generation",
    "reinforcement-learning",
    "open-weights",
    "ai-agents",
    "ai-for-science",
    "foundation-models",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 25
---

## Introduction: a lab that ships too fast to track

If you have tried to keep a mental index of ByteDance's AI research over the last two years, you have probably given up at least once. The names blur together — Seed1.5-Thinking, Seed1.5-VL, Seedance, Seedream, Seed-Coder, Seed-OSS, Seed-Prover, Seed-TTS, Seed3D, BAGEL, UI-TARS, AHN, Keel, Protenix — and they arrive faster than anyone can read the reports. The HuggingFace org alone lists dozens of repositories; the official research page adds dozens more papers that never shipped weights. It is genuinely hard to answer a simple question: *which of these is the one I should care about for my problem?*

This article is the map I wish I had when I started reading these reports. It does not try to re-derive every technical contribution — there are dedicated deep-dives for that, including ones on this blog for [Seed1.5-Thinking](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo), [Seed1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language), and [Seed Diffusion](/blog/paper-reading/large-language-model/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference). Instead it organizes the whole lineup by **what each model is for**, so you can navigate it the way you would navigate a product catalog rather than a chronological feed.

![A grid of eight use-case families covering ByteDance Seed's model lineup, from reasoning and multimodal understanding to generative media, code, agents, speech, architecture research, and AI for science](/imgs/blogs/bytedance-seed-model-universe-by-use-case-1.png)

The diagram above is the mental model: ByteDance Seed is not one product line, it is **eight problem domains served by a common platform**. Reasoning and core LLMs sit next to vision-language models, which sit next to video and image generation, which sit next to code models, GUI agents, speech, architecture research, and — increasingly — AI for science. The first thing this map buys you is the realization that "Seed" is a research organization the size of a frontier lab, deliberately spreading bets across every modality rather than concentrating on one flagship.

The second thing it buys you is a filter. Most of these models will never be relevant to your work; two or three will be exactly what you need. Let us walk the eight families, then step back and look at the three cross-cutting patterns that actually explain the lineup: the **release cadence**, the **open-versus-closed split**, and the **shared platform** underneath everything.

One caveat before we start. ByteDance Seed publishes in two registers that are easy to confuse. There are the *product* models — Doubao, Jimeng, Seedance — which most users meet through an app and which rarely come with a paper or weights. And there are the *research* releases — the ones with arXiv reports, HuggingFace repositories, and benchmark tables — which is what this map is built from. Many of the research releases quietly power the products (Seed-TTS is the voice in Doubao; Seedance is the engine in Jimeng), so the two registers are the same models wearing different hats. When I say a model is "closed," I mean its weights are not downloadable, not that it is invisible — you can usually call it through Volcano Engine or meet it inside a consumer app.

## The eight families, one at a time

A use-case taxonomy is only useful if each bucket is sharp. Here is what each family actually contains, what the headline model in it does, and when you would reach for it.

### Reasoning and core LLM

This is the center of gravity. [Seed1.5-Thinking](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo) is the flagship reasoning model: a Mixture-of-Experts with **20B active / 200B total** parameters, trained with a heavily engineered reinforcement-learning recipe that reaches 86.7 on AIME 2024 and a class-leading 39.9 on ARC-AGI while staying far smaller than DeepSeek-R1's 37B-active/671B-total. The interesting part is not the model, it is the *method*: two paired RL algorithms (VAPO, value-based; DAPO, value-free) that make long chain-of-thought RL stable enough to run at scale.

**Seed-OSS-36B** is the open-weights workhorse: a 36B dense model with a native **512K context**, trained on only 12T tokens, and shipped under Apache-2.0. Its memorable trick is a user-controllable **thinking budget** — you tell it how many tokens to spend reasoning (512, 1K, 2K, … 16K) and trade latency for accuracy at inference time. That single knob is more practically useful than it sounds: most production traffic is easy questions that do not need 8,000 tokens of deliberation, and being able to cap the reasoning length per request is the difference between a model that is affordable to serve and one that is not. It reports MMLU-Pro 82.7, AIME 2024 91.7, and LiveCodeBench v6 67.4, which puts it in the conversation with much larger open models.

**Seed2.0 Pro** (February 2026) is the closed flagship, positioned against GPT-5.2 and Gemini 3 Pro, reporting AIME 2025 of 98.3, Codeforces around 3020, GPQA Diamond 88.9, and SWE-Bench Verified of 76.5 — at roughly an order of magnitude lower token cost than the competition, with API pricing around $0.47 in / $2.37 out per million tokens. It ships as a family — Pro, Lite, Mini, and a dedicated Code model — so you can match the variant to the latency and cost envelope of your workload. If you want to *run* a Seed reasoning model yourself, you want Seed-OSS; if you want the frontier and you are fine with an API, you want Seed2.0. And if your problem is multilingual translation specifically, there is even a specialist: **Seed-X-7B**, a 7B open model covering 28 languages that rivals Gemini-2.5 and Claude-3.5 on translation quality.

### Multimodal understanding

[Seed1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language) is the vision-language flagship: a 532M-parameter from-scratch vision encoder (Seed-ViT) feeding a 20B-active MoE language model, claiming state of the art on **38 of 60** public benchmarks. Its distinguishing choices are native-resolution image encoding with 2D rotary position embeddings and a dynamic frame-resolution scheme for video, plus unusually strong GUI-agent performance inherited from the UI-TARS data lineage.

**BAGEL-7B-MoT** is the open unified model: 14B-total / 7B-active, built as a *Mixture-of-Transformers* with one expert for understanding and one for generation. It both reads and *draws* — image editing, future-frame prediction, simple world navigation — and ships Apache-2.0. What makes BAGEL interesting beyond the benchmark numbers is that the dual-expert design produces *emergent* abilities as the interleaved text-image-video pretraining data scales: capabilities like free-form editing and 3D manipulation that were never explicitly supervised. **Tarsier2** rounds out the family as a 7B open video-language model for detailed captioning that tops nine video benchmarks. The split here is the recurring Seed pattern: the closed flagship (Seed1.5-VL) pushes the benchmark frontier, the open model (BAGEL) gives the community something to build on. If you want to *understand* images and screens at the frontier, reach for Seed1.5-VL; if you want a self-hostable model that both understands and generates, BAGEL is the one to download.

### Generative media

This is the family most people actually interact with, through ByteDance's consumer apps. **Seedance** is the video generator — Seedance 1.0 produces 5-second 1080p clips in about 41 seconds on a single L20, and **Seedance 1.5 pro** adds native audio-visual joint generation through a dual-branch diffusion transformer (~4.5B params) with parallel video and audio branches and a cross-modal joint module, giving it millisecond-accurate, dialect-aware lip-sync and cinematic camera control in a single forward pass rather than a generate-then-dub pipeline. The reason that matters: bolting audio onto a finished video never quite syncs, whereas generating both jointly bakes the alignment into the model. **Seedream** is the image generator (Seedream 4.0 runs native 1K–4K and is more than 10× faster than 3.0 through distillation; Seedream 4.5 pushed text rendering further). **Seed3D 2.0** generates *simulation-ready* 3D assets — coarse-to-fine geometry, a unified physically-based-rendering material model, training-free articulation — that you can drop into a physics engine like Isaac Sim, not just static meshes; that "simulation-ready" qualifier is the whole point, because a mesh you cannot simulate is a screenshot, not an asset. Around these sit a research constellation — OmniHuman for audio-driven human video, SeedVR for video restoration, Goku and VINCIE for flow-based and in-context generation. Almost all of the headline media models are closed and ship as products (Jimeng, Doubao); the research variants are where the open weights live. This is the family where the open-versus-closed line is sharpest, because these models *are* the consumer business.

### Code and developer tools

**Seed-Coder** is a family of 8B open code models (Base, Instruct, Reasoning) whose real contribution is a *model-centric data pipeline*: instead of hand-written heuristics, LLMs score and filter code pretraining data at billion-sample scale — "let the code model curate its own diet." The insight is that the rules people write to filter code data (file length, comment ratio, license heuristics) are crude proxies for "is this good code to learn from," and a capable model is a better judge of that than a regex. The payoff is a model that is SOTA among ~8B open code models and competitive with several larger ones. [Seed Diffusion](/blog/paper-reading/large-language-model/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference) takes a different bet entirely: a discrete *diffusion* language model for code that decodes in parallel rather than left-to-right, hitting 2,146 tokens/second on an H20 — about 5.4× faster than an autoregressive model of similar quality, through two-stage diffusion training, constrained-order learning, and on-policy refinement. One family optimizes the *data* that goes into a conventional model; the other optimizes the *decoding* mechanism itself. Both are bets that the bottleneck in code generation is not raw model quality but the economics of producing tokens.

### Agents and computer-use

**UI-TARS** is the native GUI agent: it looks at pixels and emits actions, with no dependence on accessibility trees. That choice matters — accessibility-tree agents break the moment they meet a canvas, a game, or a custom-rendered UI, whereas a pixel-native agent degrades gracefully because it sees what a human sees. **UI-TARS-2** generalizes it into an all-in-one agent across GUI, games, code, and tool use, trained with stabilized multi-turn RL in a unified sandbox with file systems and terminals, fed by a data flywheel of its own rollouts. It beats Claude and OpenAI agents on Online-Mind2Web (88.2), OSWorld (47.5), WindowsAgentArena (50.6), and AndroidWorld (73.3). This family matters beyond itself: the UI-TARS data lineage is what makes Seed1.5-VL such a strong GUI grounder, scoring ScreenSpot-V2 95.2 versus OpenAI CUA's 87.9. Agents are where the vision and reasoning tracks converge — the perception model grounds the clicks, the reasoning recipe trains the long-horizon policy.

### Speech

**Seed-TTS** is a family of large-scale text-to-speech models — a multi-billion-parameter autoregressive flagship plus a fully diffusion non-autoregressive variant (Seed-TTS_DiT) that drops pre-estimated phoneme durations. Dropping the duration model matters more than it sounds: traditional TTS pipelines predict how long each phoneme should last and then synthesize against that grid, which is brittle for expressive or code-switched speech, whereas the diffusion variant learns timing end-to-end. It does zero-shot speaker cloning, emotion control, and in-context speech editing at near human-parity CMOS, and it powers Doubao's voice, with self-distillation for speaker factorization and an RL stage for robustness and speaker similarity. A full-duplex speech LLM now drives real-time conversation — the kind where you can interrupt the model mid-sentence and it stops and listens — with a reported +12% fluency. This family is closed: it is a product capability, not a research giveaway, because voice is one of the surfaces ByteDance competes on directly.

### Architecture and efficiency research

This is the family that does not ship a product at all — it ships *ideas the rest of the lineup reuses*. **Keel** ("Post-LayerNorm Is Back") shows a Highway-style residual that trains Post-LN transformers past 1,000 layers with no special tricks, reopening a design axis — Post-LN was abandoned years ago because it was unstable at depth — that the field had written off. **AHN** (Artificial Hippocampus Networks) bolts a learnable RNN/SSM memory onto a frozen pretrained transformer so out-of-window tokens get compressed into fixed-size long-term memory; it borrows the cognitive-science Multi-Store Model directly, pairing exact local attention (short-term memory) with a compressed global memory (long-term). Trained by self-distillation with the base model frozen and only ~0.4% new parameters, Qwen2.5-3B + AHN cuts FLOPs 40.5% and KV cache 74% while *improving* long-context scores — and the adapters are on HuggingFace, so you can try it on your own model. **MegaScale** is the training-infrastructure work that makes everything else possible at 10,000+ GPUs, with a serving counterpart (MegaScale-Infer) for disaggregated MoE inference. These are the seeds (pun intended) that grow into the next generation of foundation models: AHN is how the next long-context model gets cheaper, Keel is how the next model gets deeper, MegaScale is how either one gets trained at all.

### AI for Science

The newest and fastest-growing family. **Protenix-v1** is an open AlphaFold3-style biomolecular structure predictor — the first fully open-source model to exceed AF3 under matched data, size, and inference budget, and it shows inference-time scaling (more samples, better predictions). It adds protein-template integration and RNA MSA on top of the AF3 recipe and ships a `PXMeter` evaluation toolkit over 6,000+ complexes, which matters because reproducible evaluation has been the weak point of the whole structure-prediction field. **Seed-Prover** is the formal-math system that solved **5 of 6** IMO 2025 problems with Lean-checked whole-proof reasoning and self-summarization, with a companion geometry engine (Seed-Geometry) that cracked one problem in about two seconds; a later Seed-Prover 1.5 reports 88% on PutnamBench. Science is where Seed is planting its most ambitious flags, and notably, it is doing so with open weights — Protenix is Apache-2.0 and Seed-Prover's code is public, which is the opposite of the closed posture it takes on consumer media.

Here is the same eight-family view as a quick-reference table, with the one model you should remember per family:

| Family | Remember this model | What it is | Open? |
|---|---|---|---|
| Reasoning & core LLM | Seed-OSS-36B / Seed2.0 Pro | 512K-context open LLM / closed frontier | Open / Closed |
| Multimodal understanding | Seed1.5-VL | 20B-active native-resolution VLM, SOTA 38/60 | Report-only |
| Generative media | Seedance 1.5 pro | Native audio-visual video generation | Closed (product) |
| Code & developer | Seed-Coder | 8B open code LLM, model-curated data | Open (MIT) |
| Agents & computer-use | UI-TARS-2 | Native multi-turn GUI/computer agent | Open-ish |
| Speech | Seed-TTS | Zero-shot cloning, emotion, editing | Closed (product) |
| Architecture research | AHN / Keel | Long-context memory / deep Post-LN | Open / Research |
| AI for Science | Protenix-v1 | Open AF3-beating structure prediction | Open (Apache-2.0) |

## The release cadence tells you the strategy

A list of models is a snapshot. The *order* in which they shipped is the story.

![A timeline of ByteDance Seed releases from mid-2024 through 2026, starting with Seed-TTS and accelerating across reasoning, vision, video, code, agents, and science](/imgs/blogs/bytedance-seed-model-universe-by-use-case-2.png)

Read the timeline left to right and a pattern jumps out. Seed started narrow — **Seed-TTS** (speech) in mid-2024, then **Seed-Music** — the kind of bounded, product-adjacent capability you ship when you are still building the platform. Then, in the first half of 2025, the floodgates open: **UI-TARS** (January), **Seed1.5-Thinking** (April), **Seed1.5-VL** (May), **BAGEL** (May), **Seed-Coder** and **Seedance 1.0** (June), **Seed-Prover** (July), **Seed-OSS-36B** (August). That is eight major releases across six distinct domains in eight months. By 2026 the cadence has not slowed — it has moved up-market, into flagship **Seed2.0** and science (**Protenix-v1**) in February and **Seed3D 2.0** in April.

The strategic read: ByteDance spent 2024 building shared infrastructure and a reinforcement-learning stack, and 2025 *cashing it in* across every modality at once. You do not get eight domain launches in eight months by staffing eight independent teams from scratch; you get it by building a platform once and forking it. That is the thesis the rest of this article defends.

There is a second pattern in the cadence worth naming: the *move up-market over time*. The early releases are capabilities (speech, music); the middle releases are foundation models (reasoning, vision, code); the latest releases are either flagships that compete head-to-head with the frontier (Seed2.0) or ambitious science bets (Protenix, Seed-Prover). A lab that ships in this order is one that has stopped playing fast-follower and started playing for the frontier. The competitive implication for everyone else is uncomfortable: Seed's marginal cost of entering a new modality is now low, because the expensive part — the training stack, the data engines, the serving infrastructure — is already paid for. When the platform is the asset, each new model is cheap, and the cadence compounds.

| Period | Center of gravity | Representative releases |
|---|---|---|
| 2024 | Speech & media foundations | Seed-TTS, Seed-Music |
| 2025 H1 | Reasoning, vision, agents | UI-TARS, Seed1.5-Thinking, Seed1.5-VL, BAGEL |
| 2025 H2 | Code, video, open weights, math | Seed-Coder, Seedance, Seed-OSS-36B, Seed-Prover |
| 2026 | Flagship, 3D, science | Seed2.0 Pro, Seed3D 2.0, Protenix-v1 |

## Open versus closed: what you can actually run

The single most practical question about any model is "can I download the weights?" Seed's answer is more interesting than a blanket yes or no — it is a deliberate, domain-by-domain split.

![A matrix splitting Seed models by domain into an open-weights column and a report-or-API-only column, showing open workhorses and closed flagships](/imgs/blogs/bytedance-seed-model-universe-by-use-case-3.png)

The matrix makes the policy legible. The **open-weights column is real and broad** — Seed-OSS-36B and Seed-X-7B in language, BAGEL and Tarsier2 in multimodal, Seed-Coder and the earlier UI-TARS in code and agents, Protenix and Seed-Prover in science, AHN adapters and Depth Anything 3 in architecture. These are mostly the *smaller, more reusable workhorses*, often Apache-2.0 or MIT. The **report-only / API column** holds the *flagships and the consumer-facing generators* — Seed1.5-Thinking, Seed2.0 Pro, Seed1.5-VL, the entire Seedance/Seedream/Seed3D media stack, and Seed-TTS.

It is worth placing this against the rest of the field. DeepSeek open-sources almost everything, including its frontier weights, and monetizes through a cheap API. OpenAI and Anthropic open-source essentially nothing. Seed sits deliberately in between: open at the workhorse tier, closed at the flagship and product tier. That hybrid posture is the most common end-state for a lab attached to a large consumer company, because the company has two things it must protect — the consumer products that generate revenue and the frontier models that cost a fortune to train — but also two things it gains from openness — research mindshare and an ecosystem of developers fluent in its stack. The matrix above is the visible footprint of that calculation.

The logic is the logic every large lab eventually converges on. Open-source the model that builds goodwill, recruits researchers, and seeds an ecosystem you can later monetize through serving; keep closed the model that is either a direct product (video, image, voice — these *are* the business) or a frontier asset whose training cost you want to recoup through API revenue. The notable exception is **science**, which is entirely open — a signal that Seed treats AI-for-science as reputation-building and field-advancing rather than as a near-term product.

The licensing details reinforce the read. The open models are not released under restrictive research-only terms; they are genuinely permissive — Seed-OSS-36B and BAGEL under Apache-2.0, Seed-Coder under MIT, Protenix under Apache-2.0 — which means commercial use is allowed. That is a recruiting and ecosystem play: a startup that builds its product on Seed-Coder is a startup whose engineers know the Seed stack, file issues against it, and are a phone call away from a hire. Meanwhile the closed models are exactly the ones where ByteDance has a consumer distribution advantage it does not want to hand to competitors — there is no business reason to open-source the video model that powers Jimeng when Jimeng's entire moat is that video model. The split is not ideology; it is a clear-eyed read of where weights are an asset versus a liability. The one thing to watch is that "report-only" is a real third category: several models (Seed1.5-Thinking, Seed1.5-VL) ship a detailed technical report and sometimes a code repository but never the weights, so you can learn the method without running the model.

| If you need to… | Reach for | License |
|---|---|---|
| Run a strong general LLM on your own GPUs | Seed-OSS-36B (512K context) | Apache-2.0 |
| Self-host a unified understand-and-generate model | BAGEL-7B-MoT | Apache-2.0 |
| Self-host a code model | Seed-Coder-8B | MIT |
| Predict biomolecular structure | Protenix-v1 | Apache-2.0 |
| Best multimodal reasoning, API ok | Seed1.5-VL / Seed2.0 (Volcano Engine) | Closed |
| Generate video or images | Seedance / Seedream (Jimeng, API) | Closed |

## One platform, many models

So how does a single organization ship across eight domains without spreading itself into mediocrity? The answer is that it is not really eight teams building eight things — it is one platform with eight output ports.

![A four-layer stack with products on top, foundation models below, a shared training stack of RL and data engines under that, and infrastructure at the base](/imgs/blogs/bytedance-seed-model-universe-by-use-case-4.png)

Read the stack bottom-up. At the **base sits infrastructure**: MegaScale and its successors train at 10,000+ GPUs with 55%+ MFU and fault tolerance, ByteCheckpoint handles elastic checkpointing, and a HybridFlow/Ray programming layer composes the parallelism. Above it sits the **training stack** that is genuinely Seed's competitive moat: the VAPO and DAPO reinforcement-learning algorithms, the two-tier verifiers and generative reward models, and the model-centric data engines that curate pretraining corpora. On top of *that* sit the **foundation models** — Seed2.0, Seed1.5-VL, Seedance, Seed-Coder, and the rest — each one a fork of the shared base specialized to a modality. And at the **top sit the products** that hundreds of millions of people use: Doubao (the assistant), Jimeng (creative generation), Coze (agent building), all served through Volcano Engine.

This is why the cadence works. When your RL stack is a reusable asset rather than a per-project effort, adding a new modality is a matter of swapping the data and the reward signal, not reinventing the training loop. The clearest evidence is in how the *reinforcement-learning recipe itself* travels across the lineup.

It is worth being concrete about what lives in that training-stack layer, because it is the least visible and most important part. MegaScale is the published system for training at 10,000+ GPUs — 175B parameters across 12,288 GPUs at 55.2% model FLOPs utilization, with 3D-parallel communication overlap and the kind of fault tolerance you need when a run spans weeks and a node *will* fail. ByteCheckpoint makes checkpointing elastic, so you can change the parallelism configuration mid-run without losing state. HybridFlow on Ray is the programming abstraction that lets researchers express an RL loop — actor, critic, reward model, rollout engine — without hand-wiring the distributed plumbing. And the verifiers and reward models are their own research output: a two-tier verifiable-reward system (a fast rule-based judge plus a slower reasoning-based judge that hits 99.3% test accuracy) for math and code, and generative reward models for the open-ended tasks where there is no ground truth to check against. None of this is glamorous, and none of it ships as a product, but it is the reason one organization can credibly field models in eight domains. The models are the visible output; the platform is the actual company. It is also the part that is hardest for a competitor to copy: a benchmark-topping checkpoint can be matched in a quarter, but a training stack that lets a few hundred researchers ship across every modality at once is years of compounded investment. When you read a Seed report and the model impresses you, remember that the model is the easy half — the platform underneath is the part that should worry the competition.

## The RL backbone is the throughline

If you only remember one cross-cutting fact about ByteDance Seed, make it this: the reinforcement-learning recipe built to make a text reasoning model think is the same recipe — adapted, not rebuilt — that post-trains the vision model, the GUI agent, and the theorem prover.

![A graph showing the Seed-Verifier feeding the VAPO and DAPO recipe, which fans out to post-train the reasoning, vision, agent, and prover models, three of which converge into the Seed2.0 flagship](/imgs/blogs/bytedance-seed-model-universe-by-use-case-5.png)

The graph traces the propagation. A high-accuracy **Seed-Verifier** (99.3% test accuracy on its reasoning judgments) supplies the reward signal to the **VAPO + DAPO recipe**. That recipe was developed for [Seed1.5-Thinking](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo), the 20B-active text reasoner. But the same machinery shows up downstream: [Seed1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language) post-trains with a *hybrid* loop that mixes RLHF and verifiable-reward RL in a single PPO batch; **UI-TARS-2** uses stabilized multi-turn RL for long-horizon GUI tasks; **Seed-Prover** uses RL against Lean's proof checker. Three of those four — the reasoner, the VL model, and the agent — then converge into **Seed2.0 Pro**, the flagship that reports AIME 2025 of 98.3.

The two algorithms in that recipe are worth a sentence each, because they encode the bet. DAPO is the value-free member — it drops the critic and normalizes advantages within a group of samples, in the GRPO family, but adds four stabilizers (decoupled "clip-higher" bounds, dynamic sampling, token-level loss, and soft overlong-length shaping) that keep long chain-of-thought RL from collapsing. VAPO is the value-based member — it keeps a critic for finer per-token credit assignment and neutralizes the critic's bias with value-pretraining and length-adaptive advantage estimation, reaching a higher ceiling at the cost of more machinery. Having both means Seed can pick the right tool per task: value-free where rollouts are cheap and rewards are dense, value-based where credit assignment over a long trajectory is the bottleneck. That optionality is itself a platform feature.

This is the deepest reason the lineup is coherent rather than scattershot. Each model looks like a different product, but under the hood they share a training philosophy: define a verifiable or learned reward, stabilize long-trajectory RL with the VAPO/DAPO toolkit, and let the policy explore. It is the same bet DeepSeek made with [GRPO and R1](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) and that the broader field has been [stress-testing all year](/blog/paper-reading/large-language-model/part-i-tricks-or-traps-a-deep-dive-into-rl-for-llm-reasoning) — Seed's distinction is the breadth of modalities it pushes the bet across.

## A practitioner's guide: which Seed model for which job

Maps are for navigation, so here is the navigation. If you are evaluating Seed models for actual work, start from the task, not the model name.

- **"I need a self-hostable general-purpose LLM with long context."** Seed-OSS-36B. 512K native context, Apache-2.0, and the thinking-budget knob lets you tune latency per request. It is the most directly useful Seed release for most engineering teams.
- **"I need frontier reasoning and I can call an API."** Seed2.0 Pro via Volcano Engine. If you want the open approximation, Seed-OSS in its high thinking-budget mode.
- **"I'm building a document/chart/screen understanding pipeline."** Seed1.5-VL — its OCR, chart, and grounding numbers (DocVQA 96.9, InfographicVQA 91.2) are best-in-class, and the native-resolution encoder means you do not lose detail to tiling, which is exactly the failure mode that kills fixed-resolution VLMs on dense documents and small UI text.
- **"I'm building a computer-use or GUI agent."** UI-TARS-2 for the agent itself; Seed1.5-VL if you need the perception model that grounds clicks to pixels.
- **"I need a code model I can fine-tune."** Seed-Coder-8B (MIT). If decode latency is your bottleneck and you can tolerate a research preview, watch Seed Diffusion.
- **"I'm generating video, image, or 3D."** Seedance, Seedream, Seed3D — all via API/product; there are no open weights at the frontier here.
- **"I'm doing structural biology."** Protenix-v1, full open-source, beats AlphaFold3 under matched budgets.
- **"I'm doing architecture research and want a long-context trick."** AHN adapters drop onto frozen models; Keel if you are scaling depth.

A few rules of thumb cut across all of these. First, if a model has a dedicated technical report, read it before you commit — Seed's reports are unusually candid about limitations (Seed1.5-VL openly catalogs its counting and spatial-reasoning failures; Seed1.5-Thinking flags its weak factuality), and those limitation sections are where you find out whether the model will survive contact with your actual data. Second, prefer the open model unless you specifically need the frontier or the modality is closed-only; the operational freedom of running weights you control — no rate limits, no API deprecations, no data leaving your boundary — is worth a few benchmark points for most teams. Third, treat the "report-only" models as design references even when you cannot run them: the value of the Seed1.5-Thinking report is not the checkpoint you cannot download, it is the VAPO/DAPO recipe you can reimplement. The most useful thing about a lab that publishes this much is that you can steal the method even when you cannot have the model.

| Constraint | Open option | Closed/frontier option |
|---|---|---|
| General LLM | Seed-OSS-36B | Seed2.0 Pro |
| Multimodal understanding | BAGEL-7B-MoT | Seed1.5-VL |
| Code | Seed-Coder-8B | Seed Diffusion (preview) |
| Agent | UI-TARS (v1) | UI-TARS-2 stack |
| Science | Protenix-v1, Seed-Prover | — |
| Media generation | SeedVR, VINCIE (research) | Seedance, Seedream, Seed3D |

## The research constellation: everything else

The eight families above are the headline acts, but a map of Seed that stopped there would be lying by omission. A large fraction of the org's output is smaller, sharper research — models that solve one problem extremely well, often with open weights, and that frequently feed back into the flagships. If you are trying to build something specific, the right tool is often one of these rather than a flagship.

- **Seed-Music** — unified controllable music generation (autoregressive LM plus diffusion) with vocal synthesis and lyric/melody post-production editing.
- **OmniHuman-1 / 1.5** — single image plus audio or video into realistic full-body human video with lip-sync and gestures; the technology behind a lot of ByteDance's avatar features.
- **Goku** — an 8B rectified-flow (flow-based) image and video generative foundation model, more research-oriented than the product Seedance line.
- **SeedVR / SeedVR2** — diffusion-transformer video *restoration*; SeedVR2 does one-step, arbitrary-resolution restoration and was a CVPR 2025 highlight. Open weights.
- **VINCIE-3B** — in-context image editing learned purely from video via next-image and segmentation proxy tasks; SOTA multi-turn editing, open weights.
- **Depth Anything 3** — the latest in the widely-used monocular depth and geometry foundation line; open and a de-facto standard for depth estimation.
- **Seed-X-7B** — the 28-language open translation model mentioned earlier; if translation is your whole problem, this beats reaching for a general LLM.
- **BFS-Prover and Seed-Geometry** — the best-first-search prover that preceded Seed-Prover and the geometry engine that complements it.
- **CryoFM** — a cryo-EM foundation model in the AI-for-science collection.
- **MegaScale-Infer** — the disaggregated expert-parallelism serving system that is the inference-side counterpart to the MegaScale training infrastructure.

These are not afterthoughts. SeedVR cleans up generated video; VINCIE and the editing models become features in Jimeng; Depth Anything feeds spatial understanding into Seed1.5-VL; MegaScale-Infer is how the flagships get served cheaply. The constellation is the connective tissue that makes the platform thesis real.

| Niche model | Use it when you need | Open? |
|---|---|---|
| Seed-X-7B | Multilingual translation, 28 languages | Open |
| Depth Anything 3 | Monocular depth / geometry | Open |
| SeedVR2 | Video restoration / upscaling | Open |
| VINCIE-3B | Multi-turn in-context image editing | Open |
| Seed-Music | Controllable music + vocals | Closed |
| OmniHuman | Audio-driven human avatar video | Closed |

## Summary

- **ByteDance Seed is eight use-case families, not one product line:** reasoning/core LLM, multimodal understanding, generative media, code, agents, speech, architecture research, and AI for science.
- **The release cadence reveals the strategy:** a narrow 2024 (speech/media), an explosive 2025 (eight domains in eight months), and a 2026 move into flagships and science. That velocity is only possible because of a shared platform.
- **Open versus closed is deliberate and domain-specific:** smaller workhorses (Seed-OSS, BAGEL, Seed-Coder, Protenix) ship open; flagships and consumer generators (Seed2.0, Seed1.5-VL, Seedance, Seed-TTS) stay closed. Science is the all-open outlier.
- **One platform underpins everything:** MegaScale infrastructure, the VAPO/DAPO RL stack with verifiers and reward models, and model-centric data engines — forked per modality into foundation models, then served as Doubao, Jimeng, and Coze.
- **The RL backbone is the throughline:** the reinforcement-learning recipe built for text reasoning is reused — adapted, not rebuilt — to post-train vision, agents, and provers, three of which converge into Seed2.0.
- **For practitioners, the single most useful open release is Seed-OSS-36B;** the single most useful closed one is whatever Volcano Engine flagship matches your modality. Start from the task, find the family, then pick the model — and if it has a dedicated deep-dive, read that before you commit.
