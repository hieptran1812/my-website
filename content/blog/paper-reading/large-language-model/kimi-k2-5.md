---
title: "Kimi K2.5: Visual Agentic Intelligence"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - kimi-k2-5
  - multimodal
  - mixture-of-experts
  - agentic-ai
  - parallel-agents
  - reinforcement-learning
  - vision-language-model
  - moonshot-ai
description: "A principal-engineer walkthrough of Kimi K2.5: how Moonshot fuses vision and text in one trillion-parameter MoE, then trains a self-directed Agent Swarm with PARL to parallelize agentic work up to 4.5x."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-k2-5-1.png"
readTime: 31
---

The frontier agentic models we shipped against over the last year — GPT-5.2, Claude Opus 4.5, Gemini 3 Pro, and Moonshot's own Kimi K2-Thinking — got very good at long-horizon, multi-step tool use. You hand them a task, they plan, they call tools, they iterate, they finish. That loop works. But two things about it have started to grate on anyone who actually deploys these systems. First, real agentic work is rarely pure text. The agent is staring at a screenshot, scrubbing through a video, reading a 100-page PDF, or clicking through a GUI it has never seen. Second, that loop is fundamentally *serial*: one agent, one step at a time. When the task is "go read these forty sources and reconcile them," a single agent is forced to walk them one by one, and you eat the latency of the sum.

Kimi K2.5, from Moonshot AI's Kimi Team (a 324-plus-author technical report posted to arXiv as 2602.02276 on February 2, 2026, with the model itself released January 27), is a direct answer to both grievances. It is an open-source roughly one-trillion-parameter Mixture-of-Experts model that is *natively* multimodal — vision and text are co-optimized in pre-training, not stitched together afterward — and it introduces **Agent Swarm**, a self-directed parallel orchestration framework in which the model itself decomposes a task into concurrent sub-agents. The swarm is trained with a recipe the paper calls **PARL** (Parallel-Agent Reinforcement Learning). The headline claim is that this cuts inference latency by up to **4.5x** over single-agent baselines on wide-search tasks while *improving* quality, and pushes the model to state-of-the-art-class numbers on coding, vision, reasoning, and agentic search.

![Kimi K2.5 as one native-multimodal MoE stack](/imgs/blogs/kimi-k2-5-1.png)

The diagram above is the mental model: Kimi K2.5 is a single stack, not two models taped together. Native-resolution pixels enter through a ~400M-parameter vision encoder called MoonViT-3D, get bridged by an MLP projector into the same 1.04-trillion-parameter MoE backbone that Kimi K2 already uses for text, and the whole thing is wrapped at the top by Agent Swarm orchestration that can spawn up to 100 sub-agents and coordinate up to 1,500 tool calls. The thesis the paper is willing to die on is that, at this scale, "the trade-off between vision and text capabilities disappears — they improve in unison." That is a strong claim, and the paper backs it with an ablation showing that *vision* RL improves *text* benchmarks. We will get to that.

> [!tldr] TL;DR
> - **What it claims:** A single 1.04T / 32B-active MoE, natively multimodal (joint text-vision pre-training, zero-vision SFT, joint text-vision RL), plus an Agent Swarm framework trained by PARL that parallelizes agentic work up to 4.5x while improving quality.
> - **Why it matters:** It is open-source (post-trained checkpoint on Hugging Face) and argues that vision and text are synergistic at scale, not a budget you trade off — backed by a cross-modal-transfer ablation where vision RL lifts text scores by +1.7 to +2.2 points.
> - **Most surprising finding:** Post-training activates *visual* agentic skills using **only text SFT data** ("zero-vision SFT"), proxying all image manipulation through programmatic IPython operations — and it generalizes better than hand-built visual trajectories.
> - **Where it fails:** Hard perception is a ceiling (ZeroBench ~9%, ~11% with tools); it trails Claude Opus 4.5 and GPT-5.2 on raw SWE-Bench Verified (76.8 vs 80.9 / 80.0) and posts the lowest Terminal-Bench 2.0 (50.8) of the compared models; PARL needs careful reward annealing to avoid collapse.

## Context: what came before

To place K2.5 correctly you have to hold two lineages in your head at once. The first is Moonshot's text-and-agent line: Kimi K2 established the 1.04T / 32B MoE backbone trained on 15 trillion high-quality text tokens with the MuonClip optimizer, and [Kimi K2 Thinking](/blog/paper-reading/large-language-model/kimi-k2-thinking) layered long-horizon reasoning and tool use on top. The second is Moonshot's vision line, [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl), a MoE vision-language model that used a MoonViT encoder with a contrastive plus caption objective. K2.5 is what happens when you take the K2 text backbone and fuse the vision line directly into it rather than training a separate VLM.

The gap K2.5 fills is two-sided. On the modeling side, most "multimodal" frontier models bolt a vision encoder onto a frozen or lightly-tuned language model late in training; the vision and text capabilities then live in tension, and teams report having to trade one against the other. K2.5's bet is that if you fuse early — continual pre-training over ~15 trillion additional vision-text tokens from a near-end K2 checkpoint — the two modalities co-adapt and the trade-off evaporates. On the agentic side, the gap is parallelism. Every frontier agent we have shipped against is serial by construction: the policy emits one action, observes, emits the next. That is fine for inherently sequential tasks but catastrophic for tasks that are *embarrassingly parallel*, like wide search across many independent sources. K2.5's Agent Swarm is the attempt to give the model the ability to schedule its own concurrency, with no hand-crafted workflow and no predefined roles — the orchestrator decides, at inference time, how to decompose.

It is worth naming the contemporaries explicitly, because the paper benchmarks against all of them with thinking modes enabled: GPT-5.2, Claude Opus 4.5 (the tables abbreviate it "Claude 4.5"), Gemini 3 Pro, DeepSeek V3.2, and Qwen3-VL. K2.5 is not claiming to dominate every cell. It is claiming to be the best *open* model in this tier and to win decisively on the axes it was built for — agentic search and tool-augmented reasoning — while being competitive on the rest. If you have read the [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2) and [Kimi k1.5](/blog/paper-reading/reinforcement-learning/kimi-k1-5) reports, much of the optimizer and RL machinery will feel familiar; K2.5 is the multimodal-plus-parallel evolution of that program.

## Contributions

The report makes five contributions that I would actually defend in a design review:

1. **Native joint text-vision pre-training.** Rather than a post-hoc adapter, K2.5 continues a near-end Kimi K2 checkpoint over ~15T additional vision-text tokens at 4K sequence length, co-optimizing both modalities. The ablation supporting "early fusion, low vision ratio" is concrete and useful (Section on ablations).
2. **Zero-vision SFT.** A post-training scheme that activates visual *and* agentic capabilities using only text SFT data, with all image manipulation proxied through programmatic IPython operations. The claim is that this generalizes better than manually-designed visual trajectories — a genuinely counterintuitive result.
3. **Joint text-vision RL with cross-modal transfer.** A single RL stage spanning text and vision tasks, using a token-level log-ratio-clipped objective. The headline evidence is that vision RL *improves text* benchmarks (MMLU-Pro +1.7, GPQA-Diamond +2.1, LongBench v2 +2.2).
4. **Agent Swarm + PARL.** A trainable orchestrator that dynamically decomposes tasks into heterogeneous parallel sub-agents (up to 100, up to 1,500 tool calls), with only the orchestrator RL-updated and sub-agents frozen, trained by a composite reward (PARL) that explicitly fights serial collapse and spurious parallelism, regulated by a new **Critical Steps** resource metric.
5. **Token-efficient RL ("Toggle").** A two-phase RL algorithm that alternates a budget-constrained phase with a full-scaling phase, cutting output tokens by ~25–30% with negligible performance loss.

The rest of this article is the method behind those five, the experiments that test them, and an honest accounting of where I think the claims are load-bearing versus decorative.

## Method

K2.5 is best understood as three stacked systems: a native-multimodal backbone, a five-stage training recipe that produces it, and an inference-time orchestration layer (Agent Swarm) that is itself trained by RL. I will take them in that order.

### The backbone: 1.04T MoE inherited from Kimi K2

The language backbone is, deliberately, *not* new. It is the Kimi K2 architecture: a Mixture-of-Experts transformer with **1.04 trillion** total parameters and **32 billion** activated per token. The MoE feed-forward layer has **384 experts** with **8 activated per token** — the report describes this as a "sparsity of 48," i.e., total experts divided by active experts. There is one shared expert. The GitHub README spec table (medium confidence, not all repeated verbatim in the PDF body) lists 61 layers including one dense layer, an MoE hidden dimension of 2,048 per expert, SwiGLU activation, and Multi-head Latent Attention (MLA) with 64 heads and an attention hidden dimension of 7,168. The vocabulary is 160K per the README. Context length is **256K tokens**, extended via YaRN interpolation during mid-training.

The optimizer is **MuonClip** with **QK-Clip** for stability — the same token-efficient optimizer line K2 introduced, and which Moonshot's [Muon is Scalable / Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) work established at scale. The detailed MLA dimension breakdown and the full MuonClip description are deferred in the PDF to the original Kimi K2 report; K2.5 treats them as a known-good substrate. If you want the attention internals, that lineage — including the block-attention experiments in [MoBA](/blog/paper-reading/large-language-model/moba) and the linear-attention direction in [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) — is the place to look. What matters for K2.5 is that the *text* model is a fixed, strong starting point, and everything interesting happens by (a) adding eyes and (b) adding parallel orchestration.

Here is the parameter accounting that you actually need when reasoning about serving cost. Because only 8 of 384 experts fire per token, the active parameter count is ~32B, not ~1.04T — that is the number that drives FLOPs per token, while the full 1.04T drives memory.

```python
def moe_active_params(
    total_experts: int = 384,
    active_experts: int = 8,
    shared_experts: int = 1,
    moe_hidden_per_expert: int = 2048,
    d_model: int = 7168,
    layers: int = 61,
) -> dict:
    """Rough active-vs-total accounting for the K2.5 MoE FFN.

    Each token routes to `active_experts` of `total_experts`, plus the
    always-on shared expert. Sparsity = total / active = 48.
    """
    sparsity = total_experts / active_experts          # 384 / 8 = 48
    # Per-expert FFN params: two SwiGLU projections (gate+up, down).
    per_expert = 3 * d_model * moe_hidden_per_expert
    active_ffn = (active_experts + shared_experts) * per_expert * layers
    total_ffn = (total_experts + shared_experts) * per_expert * layers
    return {
        "sparsity": sparsity,
        "active_ffn_billions": round(active_ffn / 1e9, 1),
        "total_ffn_billions": round(total_ffn / 1e9, 1),
        "fraction_active": round(active_ffn / total_ffn, 3),
    }

TAKEAWAY = """
The point: you pay ~1/48 of the FFN FLOPs per token but must hold
all 384 experts resident in memory. That gap is why MoE serving is a
memory-bound, not compute-bound, problem.
"""
```

The numbers in that snippet are illustrative of the *structure*, not exact total counts (the paper's 1.04T includes attention and embeddings too), but the load-bearing intuition is correct: sparsity 48 means roughly 1/48th of the FFN compute per token at the cost of holding all experts in memory. That memory-bound character is exactly why the README claims native INT4 support (same method as Kimi K2-Thinking), though the INT4/QAT procedure is not described in the PDF body, so I treat it as a README-level claim.

### Vision encoder: MoonViT-3D

The eyes are **MoonViT-3D**, a roughly 400M-parameter native-resolution Vision Transformer initialized and continual-pretrained from **SigLIP-SO-400M**. Three design choices matter:

- **NaViT patch packing.** It ingests variable native resolution by packing patches, so a 4K screenshot and a small thumbnail both go in without forced resizing that would destroy fine text. This is what makes the model usable on documents and GUIs where pixel-accurate reading matters.
- **4x4 temporal compression for video.** Four consecutive frames are grouped and patch-level temporally averaged through a shared encoder, so a video does not blow up the token budget by a factor of (frame count). Image and video paths share parameters entirely — there is no separate video encoder.
- **Caption loss only.** Unlike [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl), which used a contrastive objective, MoonViT-3D is trained with a cross-entropy caption loss only. Text targets are image alt-texts, synthetic captions of images and videos, grounding bounding boxes, and OCR text.

The vision-to-language bridge is a plain **MLP projector**. The full multimodal stack is therefore three components: MoonViT-3D → MLP projector → Kimi K2 MoE LLM. The projector is small and is the last thing trained in the alignment phase, which is the natural seam: you align the encoder once, then cheaply re-align the projector to the giant LLM.

### The five-stage training recipe

![Five-stage recipe from K2 checkpoint to agentic model](/imgs/blogs/kimi-k2-5-2.png)

The pipeline above is the spine of the whole paper. Five stages, each with a job: align the eyes, fuse the modalities, stretch the context, activate the skills, then optimize with RL. Let me define each.

#### Stage 1: ViT training (align the eyes)

This is a two-stage alignment. **Stage 1** updates MoonViT-3D to align with **Moonlight-16B-A3B** via the caption loss, consuming roughly **1T tokens** but at very few FLOPs (the encoder is small and the LLM here is the lightweight Moonlight, not the 1T model). **Stage 2** is very short and updates *only the MLP projector* to bridge the now-aligned ViT to the full 1T LLM. Aligning against a small LLM first, then re-projecting onto the big one, is a cost trick: you spend the expensive caption-alignment compute against a cheap decoder, then pay a tiny projector-only fitting cost against the expensive one.

#### Stage 2: Joint text-vision pre-training

This is the heart of the "native multimodal" claim. K2.5 continues from a **near-end Kimi K2 checkpoint** — not a fresh model, and not the fully-converged K2 — over **~15T additional vision-text tokens** at **4K** sequence length. The data recipe extends K2's distribution with new unique tokens, adjusts proportions with **increased weight on coding-related content**, and caps the maximum number of epochs per data source. The "near-end" detail is subtle and important: starting from a checkpoint that has not fully converged on text leaves headroom for the joint optimization to co-adapt both modalities, rather than fighting an already-saturated text model.

#### Stage 3: Long-context mid-training

High-quality text plus multimodal data, with long-context activation. The context window is extended *sequentially* via **YaRN** up to 256K. The README ablation table cites stage token figures around 500B then 200B at sequence lengths 32,768 then 262,144 (README-sourced, medium confidence). The pattern — train short, then progressively extend with YaRN rather than training at full length from the start — is the standard cost-efficient way to get a long-context model without paying quadratic attention cost across the entire pre-training run.

#### Stage 4: Zero-vision SFT

This is the single most surprising design decision in the paper, and it deserves a worked explanation. K2.5's SFT follows the Kimi K2 SFT pipeline: it synthesizes high-quality candidate responses from K2, K2-Thinking, and proprietary in-house expert models, with domain-specific generation pipelines, human annotation, prompt engineering, and multi-stage verification. The twist is **zero-vision SFT**: the SFT data is *text-only*, and "all image manipulations are proxied through programmatic operations in IPython."

What does that mean concretely? Instead of building thousands of hand-curated trajectories where the model "looks at" an image and reasons about it, you teach the model to manipulate images *as code*. If the task needs a crop, the trajectory contains a Python call that crops; if it needs OCR, the trajectory calls OCR; if it needs to measure a bounding box, it computes one. The model learns the *agentic loop over visual artifacts* — load, transform, inspect, decide — in pure text, and because the underlying perception is already in the weights from joint pre-training, the loop transfers to real images at inference time.

```python
def zero_vision_sft_trajectory(task: str):
    """A text-only SFT trajectory that activates a *visual* agentic skill.

    No image tokens appear in the SFT data. Every visual operation is a
    programmatic IPython call; perception lives in the pre-trained weights.
    The model learns the load -> transform -> inspect -> decide loop here,
    then applies it to real pixels at inference time.
    """
    plan = "Find the total in the bottom-right cell of the invoice table."
    code = [
        "img = load('invoice.png')",            # proxied I/O, no pixels in SFT
        "tbl = detect_table(img)",              # programmatic, deterministic
        "cell = tbl.cell(row=-1, col=-1)",      # spatial reasoning as code
        "crop = img.crop(cell.bbox)",           # the 'manipulation' is a call
        "text = ocr(crop)",                     # OCR proxied through IPython
        "answer = parse_currency(text)",        # decide
    ]
    # The supervised target is the reasoning + this code, NOT a description
    # of the picture. The paper reports this generalizes BETTER than
    # manually-designed visual trajectories.
    return {"plan": plan, "actions": code, "supervision": "text-only"}
```

The reported result — that this generalizes *better* than manually-designed visual trajectories — is the kind of finding I would want replicated before fully trusting, but the mechanism is plausible: programmatic proxies are diverse, compositional, and verifiable, whereas hand-built visual trajectories are expensive, narrow, and biased toward whatever scenarios the annotators happened to construct. The figure embed just above this paragraph (the five-stage pipeline) shows where this sits: right before RL, as the skill-activation step.

#### Stage 5: Joint text-vision RL

A *single* RL stage spans both text and vision tasks. The objective is a token-level clipped policy optimization with a log-ratio penalty. Writing $\theta$ for the policy parameters, $\pi_\theta$ for the current policy, $\pi_{\text{old}}$ for the behavior policy, $x$ for a prompt drawn from dataset $D$, $y^j$ for the $j$-th sampled response of $N$, $y_i^j$ for its $i$-th token, $r(x, y^j)$for the reward, and $\bar{r}(x)$ for the per-prompt baseline:

$$
L_{\text{RL}}(\theta) = \mathbb{E}_{x \sim D}\!\left[\frac{1}{N}\sum_j \sum_i \text{Clip}\!\left(\frac{\pi_\theta(y_i^j \mid x, y_{0:i}^j)}{\pi_{\text{old}}(\cdot)}, \alpha, \beta\right)\big(r(x, y^j) - \bar{r}(x)\big) - \tau\big(\log\tfrac{\pi_\theta}{\pi_{\text{old}}}\big)^2\right]
$$

The clip bounds the *log-ratio* between $\alpha$ and $\beta$ rather than applying sign-dependent clipping (the PPO/GRPO style), which the paper argues better mitigates off-policy divergence; the trailing $-\tau(\log \pi_\theta / \pi_{\text{old}})^2$ term is an explicit penalty that keeps the policy from straying too far from the behavior policy. If you have read the [GRPO/DAPO/GSPO lineage](/blog/paper-reading/large-language-model/beyond-grpo-dapo-dr-grpo-gspo), this is recognizably in that family — a leaner, advantage-style objective with a stability term, applied jointly across modalities so that the gradient from a vision task and a text task flow into the same weights.

That joint flow is the mechanism behind the cross-modal transfer result. Because text and vision share the backbone *and* share an RL stage, improvements in visual grounding can sharpen text reasoning, and the ablations claim exactly that.

### Agent Swarm: self-directed parallelism

Now the inference-time layer. Everything above produces a strong single agent. Agent Swarm is the system that lets that single agent become a *fleet* it commands itself.

![Agent Swarm fans a task out and back in](/imgs/blogs/kimi-k2-5-3.png)

The graph above shows the topology. An incoming task hits a single **orchestrator** — the only trainable component. The orchestrator decomposes the task into heterogeneous, parallelizable subtasks and spawns **dynamically instantiated, frozen sub-agents** to execute them concurrently, up to **100** sub-agents and **1,500** coordinated tool calls. Critically, there are *no predefined roles and no hand-crafted workflow*: the orchestrator decides the decomposition at runtime. The sub-agents run in their own contexts, and their results converge at a single merge where the orchestrator synthesizes the final answer.

The training asymmetry is the cleverest part: **only the orchestrator is RL-updated; sub-agent trajectories are excluded from the objective.** This keeps the credit-assignment problem tractable. You are not trying to backprop through a hundred concurrent rollouts of varying length; you are training one policy — "how should I decompose and dispatch?" — and treating the frozen sub-agents as a fixed (if stochastic) environment. This is the same instinct that makes [Kimi-Researcher](/blog/paper-reading/ai-agent/kimi-researcher)'s end-to-end agent RL tractable, pushed into a parallel setting.

#### Why parallelism: the Critical Steps metric

The reason a single agent is slow on wide search is not that it does more *work*; it is that it does the work *serially*. The right resource model is not total steps but the **critical path**, analogous to the critical path in a computation graph.

![Serial single agent vs parallel Agent Swarm](/imgs/blogs/kimi-k2-5-4.png)

The before/after above makes the latency argument precise. A single agent's wall-clock latency is the *sum* of its steps. The swarm's latency is the orchestration overhead plus the *slowest sub-agent per stage*. The paper formalizes this as the **Critical Steps** metric:

$$
\text{CriticalSteps} = \sum_{t=1}^{T} \Big( S_{\text{main}}(t) + \max_i S_{\text{sub},i}(t) \Big)
$$

where $S_{\text{main}}(t)$ is the orchestrator's own steps at stage $t$ and $\max_i S_{\text{sub},i}(t)$ is the slowest sub-agent at that stage. The genius of using this as the *resource constraint during training and evaluation* — instead of total steps — is that spawning a sub-agent only helps your budget if it shortens the critical path. Spawn ten useless sub-agents and your critical steps barely move, because the max over a stage is dominated by whatever you would have had to do anyway. Spawn the *right* parallel decomposition and your critical steps drop sharply. This is how the paper incentivizes *meaningful* parallelism rather than padding the rollout with busywork.

```python
def critical_steps(stages):
    """Latency-oriented budget: orchestration overhead + slowest sub-agent
    per stage, NOT the total number of steps across all agents.

    Spawning more sub-agents only reduces this if it shortens the longest
    chain. Useless parallelism does not pay off, which is the whole point.
    """
    total = 0
    for stage in stages:
        s_main = stage["orchestrator_steps"]
        subs = stage["sub_agent_steps"]            # list, one per sub-agent
        s_slowest = max(subs) if subs else 0       # the max is what bites
        total += s_main + s_slowest
    return total


def serial_steps(stages):
    """Contrast: the naive accounting a single agent pays (sum, not max)."""
    return sum(s["orchestrator_steps"] + sum(s["sub_agent_steps"])
               for s in stages)   # sum, not max -> grows with fan-out
```

The difference between `max` and `sum` in those two functions is the entire latency win. A single agent pays `sum`; a well-orchestrated swarm pays `max`. On a 10-way independent search that is, in the best case, a 10x reduction in the dominant term — the paper reports up to **4.5x** measured end-to-end, with the blog citing an "80% reduction in end-to-end runtime."

#### PARL: rewarding parallelism without reward hacking

Telling a model "be parallel" naively produces two pathologies. The first is **serial collapse**: the orchestrator learns that a single agent is safe and just defaults to it, never using its parallel capacity. The second is **spurious parallelism**: the orchestrator games the parallelism reward by spawning useless sub-agents that never finish anything. PARL is the reward design that threads between them.

![The PARL composite reward, decomposed](/imgs/blogs/kimi-k2-5-6.png)

The tree above decomposes the composite reward:

$$
r_{\text{PARL}}(x, y) = \lambda_1 \cdot r_{\text{parallel}} + \lambda_2 \cdot r_{\text{finish}} + r_{\text{perf}}(x, y)
$$

- $r_{\text{parallel}}$ is an **instantiation reward**: it pays the orchestrator for actually spawning parallel sub-agents, directly countering *serial collapse*.
- $r_{\text{finish}}$ is a **sub-agent completion-rate reward**: it pays only when spawned sub-agents actually finish their work, which kills the *spurious parallelism* exploit — you cannot farm reward by launching agents that go nowhere.
- $r_{\text{perf}}(x, y)$ is the **task-level outcome reward**: success plus quality, the thing you actually care about.

The crucial mechanism is that **$\lambda_1$ and $\lambda_2$ are annealed to zero** over training. Early on, the shaping terms bootstrap the parallel behavior — they teach the orchestrator that parallelism is allowed and useful. As training proceeds and parallelism becomes habitual, the shaping is withdrawn, so the policy ends up optimizing *true task performance* rather than the scaffolding that taught it parallelism. The paper reports that during training, **average parallelism increases together with training accuracy** as RL FLOPs grow — which is the dynamic you want: the model is not trading accuracy for parallelism, it is discovering that the right parallelism *causes* accuracy.

I want to flag the cost honestly: that annealing schedule is a tuning knob, and the paper notes the reward shaping is needed to avoid collapse on both sides. This is the part of the recipe I would expect to be most finicky to reproduce.

#### Token-efficient RL: the Toggle algorithm

A separate but related efficiency trick is the **Toggle** algorithm for token-efficient RL. It alternates two phases. In the **budget-constrained phase**, when the mean accuracy on a prompt exceeds a threshold $\lambda$, it enforces a token budget set to the $\rho$-th percentile of the lengths of *correct* responses — i.e., "you have been getting this right, now get it right *concisely*." In the **full-scaling phase**, it lifts the budget and lets the model use as many tokens as it needs to push accuracy on the hard cases. Alternating these two yields a model that is terse where it can be and verbose where it must be. The reported effect: **~25–30% fewer output tokens with negligible performance impact.** For anyone paying per output token in production, a 25–30% reduction at iso-quality is a real line item, not a rounding error.

### Infrastructure (briefly, because it constrains everything)

Training ran on **NVIDIA H800** clusters with **8x400 Gbps RoCE** interconnects. The parallelism is **16-way Pipeline Parallelism** (with virtual stages) × **16-way Expert Parallelism** × **ZeRO-1 Data Parallelism**, with the node count required to be a multiple of 32. Expert-parallel all-to-all communication is overlapped with compute under interleaved 1F1B scheduling. Memory is managed by selective recomputation (LayerNorm, SwiGLU, MLA up-projections), with insensitive activations compressed to **FP8-E4M3** and the remainder offloaded to CPU with overlapped streaming. For multimodal training specifically, a **Decoupled Encoder Process (DEP)** replicates the vision encoder on all GPUs (recomputed in the backward pass) and reaches **90% efficiency** relative to text-only training. If you want the deep version of why the all-to-all overlap and ZeRO sharding matter at this scale, the [DeepSpeed ZeRO / 3D parallelism](/blog/paper-reading/large-language-model/deepspeed-zero-3d-parallelism-deep-dive) and [Mooncake](/blog/paper-reading/large-language-model/mooncake) writeups are the natural companions. One honest gap: the report gives the cluster *type* but not total GPU-hours, GPU count, or wall-clock training time.

The table below consolidates the architecture into one reference.

| Component | Value | Confidence |
|---|---|---|
| Architecture | MoE transformer, native multimodal | High (PDF) |
| Total / active params | 1.04T total, 32B active | High (PDF) |
| Experts | 384 total, 8 active (sparsity 48) | High (PDF) |
| Shared experts | 1 | Medium (README) |
| Layers | 61 incl. 1 dense | Medium (README) |
| Attention | MLA, 64 heads, hidden dim 7,168 | Medium (README) |
| MoE hidden / expert | 2,048 | Medium (README) |
| Activation | SwiGLU | High |
| Vocabulary | 160K | Medium (README) |
| Context length | 256K (YaRN-extended) | High |
| Optimizer | MuonClip + QK-Clip | High |
| Vision encoder | MoonViT-3D, ~400M, from SigLIP-SO-400M | High |
| Video compression | 4x4 temporal, shared params | High |
| Quantization | native INT4 (README; not in PDF body) | Low |

## Experiments

The evaluation is broad: reasoning and knowledge, vision, coding, agentic search, computer use, long context, and internal productivity benchmarks. Competitors run with thinking modes enabled, and an asterisk in the source marks a third-party re-evaluation under K2.5's test conditions. I will lead with the matrix that captures the shape of the result, then give the full tables.

![Where K2.5 leads, ties, and trails frontier models](/imgs/blogs/kimi-k2-5-5.png)

The matrix above is the honest one-screen summary: K2.5 leads on tool-augmented reasoning (HLE with tools) and agentic search (BrowseComp with context management), wins big where its computer-use and multimodality compound (OSWorld-Verified), but *trails* the closed frontier on raw coding (SWE-Bench Verified) and posts the lowest Terminal-Bench 2.0. That is not a model that wins everywhere; it is a model that wins where it was built to.

### Reasoning and knowledge

| Benchmark | Metric | K2.5 | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | DeepSeek V3.2 |
|---|---|---|---|---|---|---|
| HLE-Full | acc | 30.1 | 34.5 | 30.8 | 37.5 | 25.1* |
| HLE w/ tools | acc | **50.2** | 45.5 | 43.2 | 45.8 | 40.8* |
| AIME 2025 | acc | 96.1 | 100.0 | 92.8 | 95.0 | 93.1 |
| HMMT 2025 | acc | 95.4 | 99.4 | 92.9* | 97.3* | 92.5 |
| GPQA-Diamond | acc | 87.6 | 92.4 | 87.0 | 91.9 | 82.4 |
| MMLU-Pro | acc | 87.1 | 86.7* | 89.3* | 90.1 | 85.0 |

The pattern here is instructive. On *raw* reasoning (HLE-Full, AIME, HMMT, GPQA), K2.5 is strong but a step behind GPT-5.2 and Gemini 3 Pro — AIME 96.1 versus GPT-5.2's perfect 100.0, HLE-Full 30.1 versus Gemini's 37.5. But the moment you add *tools*, K2.5 jumps to the front: HLE with tools 50.2, ahead of every competitor by 4.4 to 9.4 points. That is the signature of a model tuned for *agentic* reasoning rather than closed-book recall — it is better at *using* its environment than at memorizing.

### Vision

| Benchmark | Metric | K2.5 | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | Qwen3-VL |
|---|---|---|---|---|---|---|
| MMMU-Pro | acc | 78.5 | 79.5 | 74.0 | 81.0 | 69.3 |
| MathVision | acc | 84.2 | 83.0 | 77.1* | 86.1 | 74.6 |
| OmniDocBench 1.5 | score | **88.8** | 85.7 | 87.7* | 88.5 | 82.0* |
| VideoMMMU | acc | 86.6 | 85.9 | 84.4* | 87.6 | 80.0 |
| VideoMME | acc | 87.4 (README) | — | — | — | — |
| LongVideoBench | acc | **79.8** | 76.5 | 67.2 | 77.7* | 65.6* |

K2.5 leads on document understanding (OmniDocBench 1.5, 88.8) and long-video understanding (LongVideoBench, 79.8), which is exactly where the NaViT native-resolution packing and 4x4 temporal compression should pay off — fine document text survives, and long videos fit in budget. It is roughly tied or slightly behind on the harder visual-reasoning benchmarks (MMMU-Pro, MathVision) against Gemini 3 Pro. The cross-modal transfer story (next section) is the more interesting vision result than any single cell here.

### Coding

| Benchmark | Metric | K2.5 | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | DeepSeek |
|---|---|---|---|---|---|---|
| SWE-Bench Verified | resolved % | 76.8 | 80.0 | 80.9 | 76.2 | 73.1 |
| SWE-Bench Multilingual | resolved % | 73.0 | 72.0 | 77.5 | 65.0 | 70.2 |
| Terminal-Bench 2.0 | acc | 50.8 | 54.0 | 59.3 | 54.2 | 46.4 |
| LiveCodeBench v6 | acc | 85.0 | — | 82.2* | 87.4* | 83.3 |

Coding is K2.5's most honest weakness against the frontier. SWE-Bench Verified 76.8 trails Claude Opus 4.5 (80.9) and GPT-5.2 (80.0), and Terminal-Bench 2.0 at 50.8 is the *lowest* among the compared models. It does lead SWE-Bench Multilingual against GPT-5.2 and Gemini, and LiveCodeBench v6 at 85.0 is competitive. If your primary use case is autonomous repo-level software engineering, the closed frontier still has an edge here. The [Kimi-Dev](/blog/paper-reading/large-language-model/kimi-dev) line is Moonshot's dedicated SWE-agent effort, and it is worth reading alongside this if coding is your axis.

### Agentic search

| Benchmark | Metric | K2.5 | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | DeepSeek |
|---|---|---|---|---|---|---|
| BrowseComp | acc | 60.6 | — | 37.0 | 37.8 | 51.4 |
| BrowseComp (w/ ctx mgmt) | acc | 74.9 | 65.8 | 57.8 | 59.2 | 67.6 |
| **BrowseComp (Agent Swarm)** | acc | **78.4** | — | — | — | — |
| WideSearch | item-f1 | 72.7 | — | 76.2* | 57.0 | 32.5* |
| **WideSearch (Agent Swarm)** | item-f1 | **79.0** | — | — | — | — |
| DeepSearchQA | acc | **77.1** | 71.3* | 76.1* | 63.2* | 60.9* |

This is the table the paper was built to produce. Even the *single-agent* K2.5 leads BrowseComp (60.6 versus Claude's 37.0 and DeepSeek's 51.4). Add context management and it climbs to 74.9. Turn on **Agent Swarm** and it hits **78.4** — the parallelism is not just a latency trick, it raises the *quality* ceiling, because wide search genuinely benefits from many concurrent independent lookups that a serial agent would truncate or skip under budget. WideSearch tells the same story: single-agent 72.7 (behind Claude's 76.2), Agent Swarm 79.0 (ahead of everyone). The Swarm earns its keep on exactly the task class it was designed for.

### Computer use and long context

| Benchmark | Metric | K2.5 | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | Qwen3-VL |
|---|---|---|---|---|---|---|
| OSWorld-Verified | success % | 63.3 | 8.6* | 66.3 | 20.7* | 38.1 |
| WebArena | success % | 58.9 | — | 63.4* | — | 26.4* |

| Long context | Metric | K2.5 |
|---|---|---|
| LongBench v2 | acc | 61.0 (README) |
| AA-LCR | acc | 70.0 (README) |

On OSWorld-Verified, K2.5 (63.3) is second only to Claude Opus 4.5 (66.3) and crushes the third-party-evaluated GPT-5.2 and Gemini scores — though I would note those competitor numbers carry asterisks, meaning they were re-evaluated under K2.5's conditions, which is a place to be cautious about apples-to-apples.

### What the ablations actually prove

Three ablations carry real weight. **First, vision injection timing.** With a fixed vision-text token budget, injecting vision *early* with a *low* vision ratio wins across the board:

| Timing | Vision:Text | Vision Knowledge | Vision Reasoning | Text Knowledge |
|---|---|---|---|---|
| Early | 10% : 90% | **25.8** | **43.8** | **45.5** |
| Mid | 20% : 80% | 25.0 | 40.7 | 43.9 |
| Late | 50% : 50% | 24.2 | 39.0 | 43.1 |

Early-and-light beats late-and-heavy on *every* metric, including text. This is direct evidence for the "fuse early" design choice and against the bolt-on-late convention.

**Second, cross-modal transfer.** This is the paper's most quotable result: training *vision* RL improves *text* benchmarks.

| Benchmark | Before Vision-RL | After Vision-RL | Δ |
|---|---|---|---|
| MMLU-Pro | 84.7 | 86.4 | +1.7 |
| GPQA-Diamond | 84.3 | 86.4 | +2.1 |
| LongBench v2 | 56.7 | 58.9 | +2.2 |

If this replicates, it is a genuine argument against the "vision is a tax on text" worldview. The shared backbone plus joint RL lets visual grounding sharpen textual reasoning.

**Third, Agent Swarm versus single-agent**, which I gave above (BrowseComp 78.4 vs 60.6, an in-house Swarm Bench 58.3 vs 41.6). The Toggle ablation (~25–30% token reduction, negligible loss) and the PARL dynamics (accuracy and parallelism rise together) round out the set.

### What is load-bearing, and what might not transfer

The load-bearing results are the *agentic* ones: agentic search with Agent Swarm, tool-augmented reasoning (HLE with tools), and computer use. These are where K2.5 leads, and they are the cleanest demonstrations of the paper's thesis. The cross-modal-transfer ablation is load-bearing for the *narrative* but is a small effect (+1.7 to +2.2) on three benchmarks — I would want it shown on more before treating it as a law.

What might not transfer: the Agent Swarm gains are concentrated in *wide-search* tasks with many independent sub-problems. If your workload is inherently serial — a single deep chain of dependent reasoning — the swarm has nothing to parallelize, and you are back to single-agent latency. The 4.5x is a best-case on parallelizable work, not a universal speedup. Similarly, the competitor numbers with asterisks (third-party re-evaluation) are the ones I would independently verify before quoting them in a head-to-head, especially the suspiciously low OSWorld GPT-5.2 (8.6*) and Gemini (20.7*) figures, which suggest harness or scaffolding mismatches rather than raw capability gaps.

## Critique

**What is strong.** The framing is honest and the design is coherent. Native fusion is justified by the early-injection ablation, not just asserted. Agent Swarm solves a *real* latency problem (serial execution on parallelizable tasks) with a principled resource metric (Critical Steps) and a reward (PARL) that explicitly names and fights its two failure modes. The training asymmetry — only the orchestrator is RL-updated, sub-agents frozen — is the right call for tractable credit assignment, and it is the detail I would steal first. Zero-vision SFT is genuinely novel and, if it holds, a large practical win because text SFT data is far cheaper and more scalable than curated visual trajectories.

**What is weak or unfalsifiable.** The cross-modal-transfer claim rests on three benchmarks and a 1.7–2.2 point delta; that is suggestive, not settled, and it is the claim doing the most narrative work. The PARL reward needs $\lambda_1, \lambda_2$ annealed to zero with shaping tuned to avoid collapse on both sides — that is a fragile-sounding recipe, and the report does not give an ablation isolating *how* fragile (e.g., what happens at different anneal schedules). The strongest competitor numbers I distrust are the asterisked third-party re-evaluations; OSWorld GPT-5.2 at 8.6 is almost certainly a harness artifact, and presenting it next to K2.5's 63.3 flatters the comparison.

**The missing ablation.** The one I most want is an *isolation* of Agent Swarm's quality gain from its latency gain. The paper shows BrowseComp 78.4 (Swarm) vs 60.6 (single), but a single agent given the *same total compute budget* (same number of tool calls, just serial) is the control I need to know whether the swarm wins because of parallelism *or* simply because it does more total work. Without that control, "parallelism improves quality" and "more compute improves quality" are confounded.

**What would change my mind.** If an independent group reproduced the cross-modal-transfer effect at the same or larger magnitude on a fresh benchmark suite, and if a compute-matched single-agent control still lost to Agent Swarm on BrowseComp, I would upgrade my read of the paper from "strong engineering with a good agentic story" to "a genuine advance in how to structure multimodal agents." Conversely, if the swarm's quality edge vanishes under a compute-matched control, the contribution narrows to a (still valuable) latency optimization.

## What I'd build with this

1. **A compute-matched Swarm control harness.** Before deploying Agent Swarm anywhere, build the evaluation the paper omits: single-agent with the *same* tool-call budget, serial, versus the swarm. If the swarm only wins on latency, you deploy it on latency-sensitive wide-search and skip it elsewhere. This is a weekend of harness work that de-risks the whole adoption decision.

2. **Zero-vision SFT for *your* visual domain.** The idea — proxy all visual manipulation through programmatic IPython operations and supervise on text-only trajectories — is domain-agnostic. For a document-processing or GUI-automation product, I would generate text-only trajectories over my own tools (table extraction, chart parsing, screenshot diffing) and see whether the "activate visual skills with text data" claim holds on my distribution. The payoff is escaping expensive visual-trajectory annotation.

3. **Critical-Steps-budgeted agents on top of any model.** The Critical Steps metric does not require K2.5; it is a generic resource model for any orchestrator-plus-sub-agent system. I would adopt it as the *latency SLO* for an agent platform, constraining and rewarding by critical path rather than total steps, regardless of the underlying model.

4. **Toggle-style token budgets in production RL.** The two-phase budget-then-scale algorithm (~25–30% token reduction at iso-quality) is a clean, portable RL trick. For any RLHF/RLVR pipeline where output cost matters, the rule "enforce the $\rho$-th percentile of correct-response length once mean accuracy crosses $\lambda$" is worth trying directly.

5. **A WorldVQA-style atomic-knowledge probe for regressions.** Moonshot released [WorldVQA](/blog/paper-reading/multimodal/worldvqa) alongside K2.5; an atomic multimodal-knowledge benchmark is exactly the kind of fast, cheap regression gate I would put in CI for a multimodal agent to catch perception degradations early.

## When to reach for Kimi K2.5 (and when not to)

Reach for K2.5 when your workload is **multimodal and agentic at the same time** — an agent that reads screenshots, scrubs video, parses 100-page documents, and *acts* — and especially when the task is **wide and parallelizable**: many independent lookups, multi-source reconciliation, broad search. That is the sweet spot where Agent Swarm's up-to-4.5x latency reduction and its quality ceiling (BrowseComp 78.4, WideSearch 79.0) both kick in, and where being open-source lets you self-host the orchestrator and tune PARL to your own tools. It is also a strong default when **tool-augmented reasoning** matters more than closed-book recall: HLE-with-tools 50.2 leads the field even though raw HLE trails.

Do *not* reach for it as your first choice when the task is **inherently serial deep reasoning with no parallel structure** — the swarm has nothing to decompose, and on raw reasoning (AIME 96.1, GPQA 87.6) GPT-5.2 and Gemini 3 Pro are a step ahead. Do not reach for it for **frontier autonomous software engineering** as the single deciding factor: SWE-Bench Verified 76.8 and Terminal-Bench 2.0 50.8 trail Claude Opus 4.5 and GPT-5.2, so if repo-level coding is the whole job, the closed frontier still edges it. And be clear-eyed about **hard perception**: ZeroBench at ~9% (~11% with tools) means the *perceptual* ceiling on genuinely difficult vision is real, and K2.5 is not a substitute for a specialized vision model on that axis. Match the tool to the shape of the work — multimodal-and-parallel is where this one is built to win.

## References

- **Kimi K2.5: Visual Agentic Intelligence** — arXiv abstract: [https://arxiv.org/abs/2602.02276](https://arxiv.org/abs/2602.02276)
- **GitHub (tech report + checkpoint):** [https://github.com/MoonshotAI/Kimi-K2.5](https://github.com/MoonshotAI/Kimi-K2.5)
- **WorldVQA benchmark (released alongside):** [https://github.com/MoonshotAI/WorldVQA](https://github.com/MoonshotAI/WorldVQA)
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2)
- [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking)
- [Kimi-VL: A Mixture-of-Experts Vision-Language Model](/blog/paper-reading/multimodal/kimi-vl)
- [Kimi-Researcher: End-to-End RL for Autonomous Research](/blog/paper-reading/ai-agent/kimi-researcher)
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight)
