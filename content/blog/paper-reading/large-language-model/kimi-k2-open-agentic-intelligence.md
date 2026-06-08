---
title: "Kimi K2: Open Agentic Intelligence — a 1T-parameter MoE trained without a single loss spike"
date: "2026-06-05"
publishDate: "2026-06-05"
description: "A close read of the Kimi K2 technical report: the MuonClip optimizer and QK-Clip, an ultra-sparse 1.04T-parameter MoE, rephrasing for token utility, a large-scale agentic data-synthesis pipeline, and joint RL with verifiable rewards plus a self-critique rubric."
tags: ["kimi-k2", "large-language-model", "mixture-of-experts", "agentic", "reinforcement-learning", "muon", "muonclip", "qk-clip", "post-training", "tool-use", "moonshot", "paper-reading"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: false
readTime: 31
---

In the last two years the open-weight frontier has been a race between two scaling axes: how many parameters you can afford to *store*, and how many you can afford to *activate*. DeepSeek-V3 pushed the second one hard — 671B total parameters, only 37B active per token — and showed that an ultra-sparse Mixture-of-Experts (MoE) can match dense flagships at a fraction of the inference FLOPs. Kimi K2, from Moonshot AI, takes the same idea and pushes it further on *both* axes while adding a third constraint that rarely gets headline space: **token efficiency**, the amount of learning signal you extract per training token, which becomes the binding constraint once you have already scraped most of the high-quality text on the internet.

The Kimi K2 technical report ([arXiv:2507.20534](https://arxiv.org/pdf/2507.20534)) is the Moonshot team's argument that the next generation of foundation models will be defined less by raw scale and more by *agentic intelligence* — the ability to perceive, plan, call tools, and correct mistakes inside real environments — and that getting there requires solving problems in both pre-training (stable optimization, token utility) and post-training (synthesizing agentic data at scale, RL that works beyond verifiable tasks).

![The Kimi K2 recipe, end to end](/imgs/blogs/kimi-k2-open-agentic-intelligence-1.png)

The diagram above is the mental model for the whole report: a token-efficient pre-training stage (MuonClip optimizer + MLA-based ultra-sparse MoE on 15.5 trillion tokens), an agentic supervised fine-tuning stage that learns tool use from synthesized trajectories, and a joint reinforcement-learning stage that combines verifiable rewards with a self-critique rubric reward. The output is Kimi-K2-Instruct: 1.04 trillion total parameters, 32 billion activated, and — at release — the top open-source model on the LMSYS Arena and state of the art among open *non-thinking* models on agentic and software-engineering benchmarks. This post walks the report section by section, then takes the senior-engineer lens to what is genuinely load-bearing and what is likely to not transfer.

> **TL;DR**
> - **What it claims.** A 1.04T-parameter / 32B-activated MoE, pre-trained on 15.5T tokens with **zero loss spikes**, that sets a new open-source bar for agentic and software-engineering tasks without any extended "thinking" — 65.8 on SWE-bench Verified (single-attempt agentic), 66.1 on τ²-Bench, 76.5 on AceBench, 53.7 on LiveCodeBench v6, 49.5 on AIME 2025.
> - **Why it matters.** It is a complete, reproducible recipe for *stable* large-scale Muon training (the MuonClip optimizer with QK-Clip) plus the first detailed public account of an industrial **agentic data-synthesis** pipeline and a **self-critique rubric reward** that extends RL past verifiable tasks.
> - **Most surprising finding.** Rephrasing a knowledge corpus *once* and training for a single epoch beats training on the raw text for ten epochs (28.94 vs 23.76 SimpleQA), and **doubling attention heads** — the DeepSeek-V3 default — buys only 0.5–1.2% validation loss while costing 83% more inference FLOPs at 128k context, so K2 *halves* heads to 64.
> - **Where it fails.** On hard reasoning or under-specified tool definitions the model over-generates and truncates; one-shot project generation lags the same model inside an agentic loop; and safety under the *Crescendo* multi-turn jailbreak degrades sharply (down to ~56–65% pass rate).
> - **What you get.** Open base and instruct checkpoints on [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct), plus the [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) used for fast RL weight updates.

## Context: what came before

Three lines of work set up this report.

**The token-efficiency wall.** Chinchilla-style scaling told us how to trade parameters against tokens for a fixed compute budget, but it assumed an effectively unlimited supply of training tokens. By 2025 that assumption broke: the stock of high-quality human text is finite, and simply repeating it causes overfitting. The report frames token efficiency — performance gained per token consumed — as a first-class scaling coefficient. Two levers raise it: a more sample-efficient optimizer (more learning per gradient step) and synthetic data that increases *token utility* (more signal per token) without the overfitting that naive repetition causes.

**Muon vs AdamW.** Muon ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)) is an optimizer for hidden-layer weight matrices that orthogonalizes the momentum update via a Newton–Schulz iteration. At a fixed compute budget and model size — and therefore the same number of training tokens — Muon substantially outperforms AdamW, which is exactly the token-efficiency win the report is chasing. The catch: scaling Muon up surfaces a training instability — *exploding attention logits* — far more frequently than AdamW does. Existing fixes are insufficient. Logit soft-capping clips the attention logits directly but lets the underlying query–key dot products keep growing. Query-Key Normalization (QK-Norm) does not apply to Multi-head Latent Attention (MLA), because the full key matrices are never materialized at inference. K2's first contribution, MuonClip, is the fix.

**The agentic-intelligence shift.** The report opens by positioning the field's move from static imitation learning toward models that *learn through interaction* — acquiring skills beyond their training distribution by exploring environments and acting on feedback. Tool use, multi-step planning, and long-horizon execution are rare in natural data and expensive to collect, so they must be synthesized. This is the gap K2 claims to fill on the post-training side: a pipeline that manufactures high-fidelity, verifiably-correct agentic trajectories at scale, paired with an RL framework that does not require every task to be machine-checkable.

## Contributions

In the authors' framing, tightened:

1. **MuonClip** — a single optimizer that fuses the token-efficient Muon update (with weight decay and consistent update-RMS scaling) with a novel **QK-Clip** mechanism that bounds attention-logit growth. It is what made a 15.5T-token run on a trillion-parameter model possible with *no loss spikes*.
2. **A large-scale agentic data-synthesis pipeline** that systematically generates tool-use demonstrations across simulated and real environments — constructing diverse tools, agents, tasks, and trajectories that are verifiably correct, at the scale of tens of thousands of tools.
3. **A general reinforcement-learning framework** that combines verifiable rewards (RLVR) with a **self-critique rubric reward**: the model learns not only from externally checkable tasks but also from judging its own outputs against rubrics, extending alignment from narrow verifiable domains to open-ended ones.

## MuonClip: stable training with QK-Clip {#muonclip}

The failure mode is concrete. When you scale Muon, the maximum attention logit — the largest pre-softmax score $\frac{1}{\sqrt d}\,Q_i K_j^\top$ in a batch — drifts upward over training. In a mid-scale 9B-activated / 53B-total MoE trained with vanilla Muon, the report shows max logits blowing past **1000**, which precedes loss spikes and occasional divergence. Once logits are that large, softmax saturates, gradients through attention become pathological, and the run is one bad batch away from blowing up.

QK-Clip attacks the cause rather than the symptom: it rescales the query and key *projection weights* after the optimizer step, but only for the heads that are actually exploding, and only by exactly enough to pull their max logit back under a threshold $\tau$.

![QK-Clip mechanism](/imgs/blogs/kimi-k2-open-agentic-intelligence-2.png)

Let the input to a layer be $\mathbf{X}$. For head $h$, the projections are $\mathbf{Q}^h = \mathbf{X}\mathbf{W}_q^h$, $\mathbf{K}^h = \mathbf{X}\mathbf{W}_k^h$, $\mathbf{V}^h = \mathbf{X}\mathbf{W}_v^h$, and the output is $\mathbf{O}^h = \mathrm{softmax}\!\left(\frac{1}{\sqrt d}\mathbf{Q}^h{\mathbf{K}^h}^\top\right)\mathbf{V}^h$. Define the per-head **max logit** over a training batch $B$:

$$
S^h_{\max} = \frac{1}{\sqrt d}\,\max_{\mathbf{X}\in B}\ \max_{i,j}\ \mathbf{Q}^h_i\,{\mathbf{K}^h_j}^\top .
$$

The core idea is to rescale $\mathbf{W}_q^h, \mathbf{W}_k^h$ whenever $S^h_{\max}$ exceeds the target threshold $\tau$. Crucially, this does **not** change the current step's forward/backward computation — the max logit is only used as a *guiding signal* to decide how hard to clamp weight growth for the next step. A naive implementation clips all heads together with $\gamma = \min(1, \tau/S_{\max})$ where $S_{\max} = \max_h S^h_{\max}$, splitting the scale across queries and keys with a balancing exponent $\alpha = 0.5$:

$$
\mathbf{W}_q^h \leftarrow \gamma^{\alpha}\,\mathbf{W}_q^h, \qquad \mathbf{W}_k^h \leftarrow \gamma^{1-\alpha}\,\mathbf{W}_k^h .
$$

But in practice only a small subset of heads explode, so K2 applies a **per-head** factor $\gamma_h = \min(1, \tau/S^h_{\max})$ to minimize intervention. For MLA, the clip touches only the *unshared* components: the head-specific $\mathbf{q}^C, \mathbf{k}^C$ are each scaled by $\sqrt{\gamma_h}$, the head-specific rotary query $\mathbf{q}^R$ is scaled by $\gamma_h$, and the *shared* rotary key $\mathbf{k}^R$ is left untouched (scaling it would leak across heads).

![Vanilla Muon diverges; MuonClip self-stabilizes](/imgs/blogs/kimi-k2-open-agentic-intelligence-3.png)

The full optimizer interleaves a standard Muon step with the QK-Clip pass (notation: $M_0 = 0$; $G_t$ is the gradient of $W_t$; $\mu$ is momentum; $\eta, \lambda$ are the learning rate and weight decay):

```python
for W, G, M in params:                       # 1. Muon step, over every weight matrix
    M = mu * M + G
    O = newton_schulz(M) * sqrt(max(n, m)) * 0.2   # match Adam update RMS
    W = W - eta * (O + lam * W)               # decoupled weight decay

for h in all_heads(all_layers):              # 2. QK-Clip (attention only)
    s_max = forward_max_logit[h]             # already computed in forward
    if s_max > tau:                          # tau = 100 for K2
        gamma = tau / s_max
        W_qc[h] *= gamma ** 0.5              # MLA: unshared query/key
        W_kc[h] *= gamma ** 0.5
        W_qr[h] *= gamma                     # head-specific rotary query
        # W_kr (shared rotary key) is deliberately NOT scaled
```

Reading the Muon step itself is worth it, because the constants carry weight. Line 1 accumulates momentum $M_t = \mu M_{t-1} + G_t$ exactly as SGD-with-momentum would. Line 2 is what makes Muon *Muon*: a Newton–Schulz iteration orthogonalizes the momentum matrix (drives its singular values toward 1), and the result is rescaled by $\sqrt{\max(n,m)}\cdot 0.2$ so the update's root-mean-square matches what Adam would have produced — the "consistent update-RMS scaling" that lets you reuse Adam-tuned learning rates and weight-decay values instead of re-tuning the whole schedule from scratch. Line 3 applies *decoupled* weight decay ($\lambda W_{t-1}$ folded into the update, not the gradient), held at $\lambda = 0.1$ throughout. The QK-Clip pass then runs only over attention heads and reuses the max logit already computed during the forward pass, so it adds no extra matmuls — just a per-head comparison and a few scalar multiplies on weights you already hold in memory.

Two empirical results make the case. First, the mechanism does not degrade the model: ablations confirm the loss trajectory matches plain Muon, so QK-Clip preserves Muon's optimization characteristics while adding stability. Second — and this is the headline figure of the whole report — training Kimi K2 with $\tau = 100$ produced a **per-step loss curve with no spikes across all 15.5T tokens**. The max logits shoot up to the capped value of 100 early, then *decay on their own* to a comfortable operating range after roughly 30% of training, at which point the cap stops binding without any change to $\tau$. The model learns to keep its own logits small; QK-Clip just holds the line until it does.

The takeaway for practitioners: if you want Muon's token-efficiency at scale, you need *something* to bound attention logits, and QK-Clip is the only published option that works for MLA (where QK-Norm cannot). It is cheap — a handful of per-head scalar multiplies per step on weights you already have — and it is surgical, touching only the heads that misbehave.

## Model architecture {#model-architecture}

K2 follows the DeepSeek-V3 design — Multi-head Latent Attention for the attention mechanism, ultra-sparse MoE for the feed-forward — but re-derives two key dimensions from its own scaling-law analysis. The model hidden dimension is 7168 and the MoE expert hidden dimension is 2048.

![Inside a Kimi K2 block: MLA + sparse MoE](/imgs/blogs/kimi-k2-open-agentic-intelligence-4.png)

Each block routes a token through MLA attention, then through a router that activates **8 of 384 experts** plus **1 shared expert** that always runs. The two deliberate departures from DeepSeek-V3 are *more total experts* (higher sparsity) and *fewer attention heads*.

| Hyperparameter            | DeepSeek-V3 | Kimi K2 | Δ        |
|---------------------------|-------------|---------|----------|
| Layers                    | 61          | 61      | =        |
| Total parameters          | 671B        | 1.04T   | ↑ 54%    |
| Activated parameters      | 37B         | 32.6B   | ↓ 13%    |
| Experts (total)           | 256         | 384     | ↑ 50%    |
| Experts active per token  | 8           | 8       | =        |
| Shared experts            | 1           | 1       | =        |
| Attention heads           | 128         | 64      | ↓ 50%    |
| Dense (non-MoE) layers    | 3           | 1       | ↓ 67%    |
| Expert grouping           | yes         | no      | —        |

The arithmetic is worth pausing on: K2 has **54% more total parameters** but **13% fewer activated parameters** than DeepSeek-V3. It is a bigger model that is *cheaper* to run per token. That is the whole point of pushing sparsity.

Two choices on that table deserve a sentence each. **MLA (Multi-head Latent Attention)** compresses keys and values into a low-rank latent vector that is cached in place of full per-head K/V — this is what makes the KV cache affordable at 128k context, and (as noted above) it is also exactly why QK-Norm doesn't apply and QK-Clip had to be invented in the first place. The **shared expert** that always runs gives every token a common "backbone" FFN so the 8 routed experts can specialize on the residual; combined with *dropping* expert grouping (DeepSeek-V3 grouped experts to bound cross-node routing traffic), K2 trades a little routing locality for simpler, better-balanced expert parallelism at EP = 16 — which is only affordable because the 64-head attention is cheap enough to overlap with EP communication.

### Sparsity scaling law

The report develops a sparsity scaling law for Muon-trained MoEs, where *sparsity* is the ratio of total experts to activated experts. Holding the number of activated parameters fixed (constant FLOPs) and varying total experts, more experts consistently lowers both training and validation loss.

![Sparsity scaling: more experts, fewer FLOPs to a target loss](/imgs/blogs/kimi-k2-open-agentic-intelligence-5.png)

Concretely, to reach the same validation loss of 1.5, a sparsity of 48 needs **1.69×, 1.39×, and 1.15× fewer FLOPs** than sparsities of 8, 16, and 32 respectively. The gain is real but comes with infrastructure complexity (more experts means more expert-parallel communication and balancing), so K2 settles on sparsity 48 — 8 active out of 384 — as the cost/performance knee.

### Why only 64 attention heads

DeepSeek-V3 sets the number of attention heads to roughly twice the layer count to better use memory bandwidth. But head count scales inference cost badly at long context: at a 128k sequence length, going from 64 to 128 heads (with the expert count fixed at 384) inflates inference FLOPs by **83%**. For an agentic model that lives at long context, that is the wrong trade. Controlled experiments comparing heads-equal-to-layers against double-heads show doubling buys only **0.5–1.2%** validation-loss improvement across compute budgets. Given that sparsity 48 already delivers strong performance, K2 keeps **64 heads** and spends the saved FLOPs on context length instead. This is the kind of decision that only shows up when you take serving seriously during architecture search.

## Pre-training data: token utility via rephrasing {#pre-training-data}

The K2 corpus is 15.5T tokens of curated data across four domains — Web Text, Code, Mathematics, and Knowledge — with processing pipelines inherited from Kimi K1.5. The new idea over K1.5 is a controlled **synthetic rephrasing** strategy that increases token utility without the overfitting that comes from repeating the same tokens.

The intuition: a single epoch on knowledge-dense text under-absorbs it, but multi-epoch repetition yields diminishing returns and overfits. Rephrasing the text into varied styles and perspectives lets the model see the same *facts* many times in different *surface forms*, which absorbs the knowledge without memorizing the string.

![Chunk-wise autoregressive rephrasing](/imgs/blogs/kimi-k2-open-agentic-intelligence-6.png)

The knowledge-rephrasing pipeline has three parts. **Style- and perspective-diverse prompting** (inspired by WRAP) generates faithful rewrites in different voices while preserving factual integrity. **Chunk-wise autoregressive generation** splits long documents into segments, rephrases each with the preceding context as conditioning, then stitches them back together — this preserves global coherence and dodges the implicit output-length limits that make LLMs drop content when asked to rewrite a long passage in one shot. **Fidelity verification** compares each rewrite's semantics against the source as an up-front quality gate. For mathematics, high-quality documents are rewritten into a "learning-note" style (following SwallowMath), and high-quality non-English math material is translated into English to broaden coverage.

The ablation is the convincing part. On an early K2 checkpoint, three training strategies on a knowledge corpus give:

| Strategy                          | # Rephrasings | # Epochs | SimpleQA Accuracy |
|-----------------------------------|---------------|----------|-------------------|
| Raw wiki-text, repeated           | 0             | 10       | 23.76             |
| Rephrased once, repeated          | 1             | 10       | 27.39             |
| Rephrased 10×, single pass        | 10            | 1        | **28.94**         |

Rephrasing ten times and training for a *single* epoch beats ten epochs on the raw text by **5.18 points** of SimpleQA accuracy. The report extends this to other large corpora and reports similar gains, with each corpus rephrased at most twice. It is careful to flag the open risk: synthetic data at scale can drift on factual accuracy and introduce hallucinations, and generalizing the approach across domains remains an active question.

## Training infrastructure and recipe {#training-recipe}

K2 trains on a cluster of **NVIDIA H800** GPUs — each node has 2 TB of RAM and 8 GPUs joined by NVLink/NVSwitch, with 8×400 Gbps RoCE between nodes. The parallelism strategy is built for *flexibility under changing resource availability*: it runs on any node count that is a multiple of 32, combining **16-way Pipeline Parallelism** (with virtual stages), **16-way Expert Parallelism**, and **ZeRO-1 Data Parallelism**.

Storing parameters in BF16 with an FP32 gradient-accumulation buffer costs ~6 TB of GPU memory, distributed over a model-parallel group of 256 GPUs. Three engineering choices keep activation memory in budget:

- **EP communication overlap with interleaved 1F1B.** By increasing warm-up micro-batches, all-to-all expert-parallel communication overlaps with computation under the standard interleaved 1F1B schedule. DualPipe would double the memory for parameters and gradients — prohibitive past a trillion parameters — so K2 avoids it. To pay for the extra pipeline stages, weight-gradient computation is decoupled from each micro-batch's backward pass and run in parallel with the corresponding pipeline communication, so PP comms overlap fully except during warm-up. Because K2's 64 heads make attention cheap, the smallest feasible EP size (EP = 16) is enough to fully overlap, which also relaxes expert-balance constraints.
- **Selective recomputation.** Cheap, high-footprint stages (LayerNorm, SwiGLU, MLA up-projections) and MoE down-projections are recomputed during training to cut activation memory and prevent crashes from expert imbalance early on.
- **Activation CPU offload.** Remaining activations stream to CPU RAM via a copy engine, overlapping offload/onload with both compute and communication kernels; FP8 is used only for *storage* of insensitive activations (MoE up-projections and SwiGLU inputs), not for computation, after preliminary FP8-compute experiments showed degradation risk.

The recipe itself uses a **WSD (Warmup-Stable-Decay)** learning-rate schedule on a 4,096-token context window, then anneals and grows context:

![Pre-training schedule: WSD then long-context](/imgs/blogs/kimi-k2-open-agentic-intelligence-7.png)

The first 10T tokens run at a constant LR of 2e-4 (after a 500-step warmup); the next 5.5T tokens decay it cosine-style from 2e-4 to 2e-5. Weight decay is held at 0.1 throughout and the global batch size is fixed at 67M tokens. An annealing phase then decays the LR from 2e-5 to 7e-6 while training 400B tokens at 4k context, followed by 60B tokens at 32k context. Finally, the context window is extended to 128k using **YaRN**. This is a textbook "spend most of your tokens cheap at short context, then buy long context with a small tail" strategy — and it is only feasible at this token count *because* the loss curve is spike-free.

## Post-training I: SFT and agentic data synthesis {#post-training-i}

Post-training opens with supervised fine-tuning, and the report recommends Muon here too — a Muon-pre-trained checkpoint fine-tunes best under Muon. The SFT set spans many domains, built with per-domain data-generation pipelines that mix human annotation, prompt engineering, and verification, with K1.5 and other domain-expert models generating candidate responses that LLM or human judges then filter. The interesting part is the agentic slice.

Teaching a model to autonomously use unfamiliar tools, interact with environments, and self-correct is hard because real environments are costly, private, and hard to scale. K2's answer is to *simulate* most of it and ground a fraction in real execution.

![Synthesizing agentic tool-use trajectories](/imgs/blogs/kimi-k2-open-agentic-intelligence-8.png)

The pipeline has three stages. **Tool spec generation** builds a repository from two sources: 3,000+ real Model-Context-Protocol (MCP) tools fetched from GitHub, plus 20,000+ synthetic tools evolved through a hierarchical domain process (key categories → application domains → specialized tools with clear interfaces and semantics). A t-SNE of the two sets shows they cover complementary regions of tool space. **Agent and task generation** synthesizes thousands of distinct agents by combining different system prompts and tool subsets, then pairs each with rubric-tagged tasks that range from simple to complex. **Trajectory generation** runs a multi-agent loop:

- a **user simulator** (LLM personas with distinct communication styles) drives multi-turn dialogues;
- a **tool-execution environment** — a tool simulator functionally equivalent to a world model — executes calls, updates persistent state, and injects controlled stochasticity to produce successes, partial failures, and edge cases;
- an **LLM judge** scores each trajectory against the task rubric; only trajectories that clear the success criteria are kept.

To counter the inherent fidelity gap of simulation, the pipeline is *hybrid*: scenarios where authenticity is critical — coding and software engineering — run in **real execution sandboxes** with actual development environments and ground-truth signals (test pass rates). The software-engineering environment is built from real GitHub pull requests and issues with executable unit tests, on a Kubernetes-backed sandbox that supports **over 10,000 concurrent instances**. The combination implements large-scale rejection sampling: generate broadly in simulation, keep only what passes a rubric or a real test.

The token format for tool calling (Appendix B of the report) is worth noting if you plan to serve K2. Tools are declared in a dedicated message, the assistant emits a tool-invocation section, and a tool-result message carries the execution output. K2 trains on tools declared in *TypeScript* (concise, expressive types) but also includes JSON-declared tools so third-party OpenAI-compatible frameworks work without extra adaptation:

```typescript
// Tool declaration (TypeScript form, the K2-native format).
namespace functions {
  // Get weather for a location and date.
  type get_weather = (args: {
    location: string;          // City and country, e.g. "Beijing, China"
    date?: string;             // Date to query, format "%Y-%m-%d"
  }) => any;

  // Simple calculator.
  type Calculator = (args: {
    expr: string;              // Arithmetic expression in JavaScript
  }) => any;
}
```

## Post-training II: reinforcement learning {#post-training-ii}

RL is treated as the more token-efficient, more generalizable continuation of SFT. K2 scales it on two fronts — task diversity and training FLOPs — using a Gym-like extensible framework. The framework's central trick is to make as many tasks as possible *rewardable*: verifiable rewards where ground truth exists, and a self-critique reward where it does not.

![Two reward sources in K2's RL](/imgs/blogs/kimi-k2-open-agentic-intelligence-9.png)

### The verifiable-rewards gym

For checkable tasks, K2 builds reward sources by domain:

- **Math, STEM, and logic.** High-quality QA pairs from expert annotations, internal extraction pipelines, and open datasets, with a tagging system to deliberately raise coverage of under-represented areas; structured-data and logic-puzzle formats (multi-hop tabular reasoning, 24-game, Sudoku, riddles, cryptarithms, Morse decoding) fill out the set. Difficulty is calibrated using the SFT model's pass@k so prompts are neither trivial nor impossible — *moderate difficulty* gives the most learning signal.
- **Complex instruction following.** A hybrid verifier pairs deterministic code-interpreter checks (length, style, format constraints) with LLM-as-judge for nuanced constraints, plus a dedicated **hack-check** layer that catches the model claiming compliance it did not actually deliver. Instructions are generated three ways: expert-crafted prompts/rubrics, agentic instruction augmentation (inspired by AutoIF), and a fine-tuned model that probes specific failure modes.
- **Faithfulness.** A sentence-level judge (inspired by FACTS Grounding) flags claims unsupported by the provided context and serves as a reward to improve grounded factuality — essential for multi-turn tool use and long reasoning chains.
- **Coding and software engineering.** Competitive-programming problems with their judges and human-written unit tests, plus the PR/issue sandbox environment from SFT supplying objective test-pass signals.
- **Safety.** A human-curated set of seed prompts is expanded by an automated attack→target→judge pipeline that simulates jailbreaks (role-play, literary framing, academic discourse) and labels each interaction pass/fail with a task-specific rubric.

### Beyond verification: the self-critique rubric reward

The harder problem is open-ended tasks — creative writing, nuanced helpfulness, depth of reasoning — where there is no ground truth. K2's **self-critique rubric reward** has the actor generate responses to general prompts and the K2 *critic* rank them via pairwise comparison against a combination of rubrics: **core rubrics** encoding the assistant's intended values, **prescriptive rubrics** designed to kill specific reward-hacking behaviors, and **human-annotated rubrics** for particular contexts.

The clever part is **closed-loop critic refinement**. During RL, on-policy rollouts judged by the *verifiable* rewards are continuously used to update the critic. This distills objective performance signals from RLVR into the subjective evaluator — the critic's judgments on non-verifiable tasks stay calibrated against verifiable ground truth, in lockstep with the policy's evolution. Subjective evaluation is thereby anchored in verifiable data.

### The RL algorithm and its guardrails

K2 adopts the policy-optimization objective from K1.5. For each problem $x$, sample $K$ responses $\{y_1,\dots,y_K\}$ from the previous policy $\pi_{\text{old}}$ and optimize $\pi_\theta$ against:

$$
L_{\text{RL}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}}\left[\frac{1}{K}\sum_{i=1}^{K}\left(r(x,y_i) - \bar r(x) - \tau\log\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^2\right],
$$

where $\bar r(x) = \frac1K\sum_i r(x,y_i)$ is the mean reward of the sampled responses and $\tau > 0$ regularizes toward stable learning. (This $\tau$ is the RL regularizer, distinct from the QK-Clip threshold.) The optimizer is again Muon. Scaling this objective across many domains at once needs three additions, each fixing a real pathology:

- **Budget control.** RL tends to inflate response length, which helps hard reasoning but wastes inference on everything else. K2 enforces a *per-sample maximum token budget* set by task type; over-budget responses are truncated and penalized, pushing the model toward concise solutions.
- **PTX loss.** To stop the model forgetting high-quality SFT data during joint RL, a curated set of hand-selected examples is folded into the objective via an auxiliary PTX loss — leveraging the premium data without overfitting to the narrow RL task set.
- **Temperature decay.** Creative and reasoning tasks benefit from high exploration early (high sampling temperature) but high temperature late hurts reliability, so the schedule decays temperature to shift from exploration to exploitation.

```python
for x in batch:                              # one RL iteration; objective from K1.5, optimizer = Muon
    ys     = pi_old.sample(x, K=K, temperature=temp_schedule(step))
    ys     = [truncate_and_penalize(y, budget_for(task_of(x))) for y in ys]
    r      = [reward(x, y) for y in ys]        # RLVR or self-critique rubric
    r_bar  = mean(r)
    # squared-loss surrogate, advantage = r - r_bar, KL-style reg to pi_old
    loss   = mean((r[i] - r_bar - tau * logratio(pi_theta, pi_old, ys[i], x))**2
                  for i in range(K))
    loss  += ptx_weight * ptx_loss(pi_theta, premium_sft_data)   # anti-forgetting
    muon_step(pi_theta, grad(loss))
```

## RL infrastructure {#rl-infrastructure}

Synchronous RL at trillion-parameter scale is mostly a systems problem. K2 uses a **colocated architecture**: training and inference engines live on the *same* workers and hand GPU memory back and forth — when one engine works, the other offloads. A centralized controller calls the inference engine to generate rollouts, then tells the training engine to learn from them and ship updated weights back. As the model grows, two costs dominate: the latency of switching engines, and failure recovery.

![Checkpoint engine: 1T params updated in under 30 seconds](/imgs/blogs/kimi-k2-open-agentic-intelligence-10.png)

The weight-update path is the interesting bit. After a rollout, training-engine parameters are offloaded to DRAM, so bringing the training engine up is a simple host-to-device transfer. Refreshing the *inference* engine is harder: it needs the updated weights under a different sharding scheme. A network filesystem can't keep up — the aggregate bandwidth needed reaches petabytes per second — so K2 introduces a distributed **checkpoint engine**, co-located on the training nodes. Each checkpoint-engine worker grabs a local copy of the parameters, **broadcasts the full parameter set** to all checkpoint workers, and each inference worker then pulls just the shard it needs. Updates are pipelined parameter-by-parameter to keep the memory footprint small. Broadcasting the whole set (rather than transferring only what each shard needs) moves more data but fully decouples the training and inference engines, simplifying maintenance. The result: a **full parameter update for the 1T model in under 30 seconds** — negligible against a typical RL iteration. The [checkpoint engine is open-sourced](https://github.com/MoonshotAI/checkpoint-engine).

For long-horizon agentic rollouts, two more tricks keep GPUs busy: heavy environments (VMs, code interpreters) are deployed as separately-scalable services, and many concurrent rollouts amortize the latency of slow interactions. To stop a single long-tail trajectory from blocking the whole batch, **partial rollout** pauses unfinished long-tail tasks and resumes them in the next RL iteration. The whole thing is wrapped in an OpenAI-Gym-style interface for adding new environments.

## Experiments and results {#experiments}

Every result below is reported in the model's **non-thinking** configuration — no extended chain-of-thought, no test-time scaling — to isolate the base capability. Output length is capped at 8,192 tokens everywhere except SWE-bench Verified Agentless (16,384). High-variance benchmarks use repeated sampling averaged as Avg@k.

### Kimi-K2-Instruct vs the field

A condensed slice of the headline table (Table 3 in the report), against the strongest open and proprietary non-thinking baselines. **Bold** = global best in the row; numbers are Pass@1 / Acc / Avg@k as appropriate.

| Benchmark                                  | Kimi-K2-Inst | DeepSeek-V3-0324 | Qwen3-235B-A22B | Claude Sonnet 4 | Claude Opus 4 | GPT-4.1 | Gemini 2.5 Flash |
|--------------------------------------------|--------------|------------------|-----------------|-----------------|---------------|---------|------------------|
| LiveCodeBench v6 (Pass@1)                  | **53.7**     | 46.9             | 37.0            | 48.5            | 47.4          | 44.7    | 44.7             |
| OJBench (Pass@1)                           | **27.1**     | 24.0             | 11.3            | 15.3            | 19.6          | 19.5    | 19.5             |
| SWE-bench Verified (agentic, single)       | 65.8         | 38.8             | 34.4            | 72.7\*          | 72.5\*        | 54.6    | —                |
| SWE-bench Verified (agentic, multi)        | 71.6         | —                | —               | 80.2\*          | 79.4\*        | —       | —                |
| SWE-bench Multilingual (Pass@1)            | 47.3         | 25.8             | 20.9            | 51.0            | —             | 31.5    | —                |
| Tau2-Bench (weighted avg)                  | **66.1**     | —                | —               | —               | —             | —       | —                |
| AceBench (Acc)                             | 76.5         | 72.7             | 70.5            | 76.2            | 75.6          | **80.1**| 74.5             |
| AIME 2025 (Avg@64)                         | **49.5**     | 46.7             | 24.7            | 33.1            | 33.9          | 37.0    | 46.6             |
| MATH-500 (Acc)                             | **97.4**     | 94.0             | 91.2            | 94.0            | 94.4          | 92.4    | 95.4             |
| GPQA-Diamond (Avg@8)                       | **75.1**     | 68.4             | 62.9            | 70.0            | 74.9\*        | 66.3    | 68.2             |
| MMLU (EM)                                  | 89.5         | 89.4             | 87.0            | 91.5            | **92.9**      | 90.4    | 90.1             |
| SimpleQA (Correct)                         | 31.0         | 27.7             | 13.2            | 15.9            | 22.8          | **42.3**| 23.3             |

What's load-bearing here. Among **open** models, K2 is the clear leader on agentic and coding tasks — it roughly *doubles* DeepSeek-V3-0324 and Qwen3-235B on SWE-bench Verified and beats every open peer on LiveCodeBench, OJBench, AIME, MATH-500, and GPQA-Diamond. Against the proprietary frontier it closes most of the SWE-bench gap to Claude 4 (the strongest agentic coders) and leads several reasoning rows outright, while staying behind on raw knowledge breadth (MMLU, SimpleQA, where GPT-4.1 leads). The honest caveat: the strongest SWE-bench numbers for Claude carry a `*` (taken from vendor reports under their own harness), so cross-model SWE-bench comparisons mix evaluation setups — read them as "same ballpark," not a clean ranking. The agentic strength is the real story: 66.1 on τ²-Bench and 76.5 on AceBench put K2 at or near the top of multi-turn tool orchestration. On the LMSYS Arena (July 2025) it ranked **#1 open-source and #5 overall** on 3,000+ blind votes.

### Kimi-K2-Base

The base model is strong before any alignment (Table 4 condensed), evaluated against open base models under identical settings:

| Benchmark (shots)        | Kimi-K2-Base | DeepSeek-V3-Base | Llama4-Maverick-Base | Qwen2.5-72B-Base |
|--------------------------|--------------|------------------|----------------------|------------------|
| Architecture / Activated | MoE / 32B    | MoE / 37B        | MoE / 17B            | Dense / 72B      |
| MMLU (5-shot, EM)        | **87.79**    | 87.10            | 84.87                | 86.08            |
| MMLU-Pro (5-shot)        | **69.17**    | 60.59            | 63.47                | 62.80            |
| SimpleQA (5-shot)        | **35.25**    | 26.49            | 23.74                | 10.31            |
| GSM8k (8-shot)           | **92.12**    | 91.66            | 86.35                | 90.37            |
| MATH (4-shot)            | **70.22**    | 61.70            | 63.02                | 62.68            |
| LiveCodeBench v6 (1-shot)| **26.29**    | 24.57            | 25.14                | 22.29            |
| C-Eval (5-shot)          | **92.50**    | 90.04            | 80.91                | 90.86            |
| CMMLU (5-shot)           | **90.90**    | 88.84            | 81.24                | 90.55            |

K2-Base leads 10 of 12 English benchmarks and all three Chinese ones. The SimpleQA jump (35.25 vs DeepSeek-V3-Base's 26.49) is the clearest single signal that the rephrasing strategy paid off on factual knowledge absorption.

### Safety

Red-teaming uses Promptfoo to generate adversarial prompts across harmful/criminal/misinformation/privacy/security plugins crossed with attack strategies (basic, prompt injection, iterative jailbreak, Crescendo). The pattern in the results (Table 6) is more interesting than any single number:

- **Encoding transforms barely dent it.** Under the Base64 strategy, pass rates approach or hit 100% across categories — encoding the prompt doesn't smuggle much past the model.
- **Crescendo is the soft spot.** The multi-turn *Crescendo* strategy drops pass rates the most — e.g. Harmful/Crescendo at 64.71, Criminal/Crescendo at 56.06 — confirming that gradual, multi-turn escalation is a stronger attack than single-shot or naive jailbreaks. Notably, K2 holds up *better* than DeepSeek-V3/R1 and Qwen3 on some complex cases (Harmful/Iterative-Jailbreak 92.16) even without targeted optimization for these scenarios.

The report is candid that the eval involves human review (some subjectivity) and that several plugins target tool-calling/API-misuse behaviors that are more meaningful for agent models than base LLMs.

## Reproduction and serving notes

A few things in the report are directly actionable if you intend to fine-tune or serve K2 rather than just read about it.

**Fine-tuning.** The report is explicit that a Muon-pre-trained checkpoint fine-tunes best *under Muon* — reaching for AdamW in post-training leaves performance on the table. If you adopt the open checkpoint, budget for a Muon implementation with the consistent-RMS scaling described above, and keep QK-Clip wired in for any continued pre-training (it is cheap insurance even when your logits look tame at first — recall that K2's own logits only self-stabilized after ~30% of training). Weight decay 0.1, the WSD-style learning-rate shape, and a large token batch are reasonable defaults to inherit; the SFT recipe itself is standard supervised fine-tuning on filtered, rubric-judged responses, so the leverage is in the *data*, not an exotic loss.

**Serving long context.** The 128k window is reached by YaRN extension applied *after* a 32k training stage, not by training at 128k directly. In practice the long-context behavior is an interpolation of the 32k regime; if you serve at the full 128k you are relying on YaRN's frequency scaling holding up, so validate on your own long-context retrieval set rather than assuming the benchmark MRCR (55.0) and DROP (93.5) numbers transfer to your distribution. The 64-head MLA design is what keeps that 128k KV cache affordable on inference hardware in the first place.

**Tool calling.** K2's native tool-declaration format is TypeScript (concise, expressive types), but the training mix deliberately includes JSON-declared tools so OpenAI-compatible frameworks work without extra adaptation. If you see degraded tool use, check that your tool schemas are well-specified — the report's own limitations section calls out that *unclear tool definitions* trigger over-generation and truncated outputs, and that performance can even *drop* on some tasks when tool use is unnecessarily enabled.

**Reading the eval numbers correctly.** Every headline number is *non-thinking*: no extended chain-of-thought, no test-time scaling, so do not compare them against another model's "thinking" scores. Output is capped at 8,192 tokens (16,384 for SWE-bench Verified Agentless), high-variance benchmarks are reported as Avg@k (e.g. AIME at Avg@64, GPQA at Avg@8), and long-context tasks truncate inputs to the 128k window. SWE-bench Verified is reported in two modes — *Agentless* (a single patch, no tools) and *Agentic* (bash/editor tools, in single-attempt and multi-attempt-with-internal-verifier variants). When you compare K2 to a frontier model, match the mode: the 65.8 single-attempt agentic, 71.6 multi-attempt agentic, and 51.8 agentless numbers are three different settings, and several baseline cells are vendor-reported under a different harness. The cleanest apples-to-apples claim the report makes is the open-vs-open one, where K2 leads decisively.

## Critique: the senior-engineer lens

**What's strong.** Three things will outlive this particular checkpoint. First, **QK-Clip is a genuinely reusable result** — it's the missing piece that makes Muon viable at scale with MLA, it's cheap, and the "logits self-stabilize after 30% of training" observation suggests the instability is a transient of early training, not a permanent tax. Second, the **agentic data-synthesis pipeline** is the most detailed public description of how to manufacture tool-use data at industrial scale, and the hybrid simulated/real split is a pragmatic answer to the fidelity-vs-cost dilemma. Third, the **self-critique rubric reward with closed-loop critic refinement** is a principled attack on the "you can't RL what you can't verify" problem — grounding the subjective critic in verifiable signals is the right instinct.

**What's weak or unfalsifiable.** The headline cross-model comparisons are muddier than the bold numbers suggest: several frontier baselines are starred (vendor-reported, different harness), so any "K2 beats / trails model X by N points" on SWE-bench is partly a harness-difference claim. The rephrasing ablation, while convincing, is run on *one* early checkpoint and one knowledge corpus at SimpleQA — the report itself flags that scaling synthetic data to diverse domains without accumulating factual drift is unsolved, so the strategy's generality is asserted, not demonstrated. And the safety story is structurally limited: an automated red-team with human-in-the-loop judging measures what the attack pipeline thought to test, and the sharp Crescendo degradation hints that multi-turn adversarial robustness was not a first-class training objective.

**What ablation is missing.** The two most consequential architecture decisions — sparsity 48 and 64 heads — are each justified by a scaling-law curve, but there is no end-to-end ablation of the *full* trillion-parameter model with the alternative choices (you can't afford to train it twice, which is fair, but it means the small-scale law is doing all the load-bearing work). There is no isolation of how much of the agentic win comes from synthetic SFT data versus the RL stage versus the real-execution sandboxes. And MuonClip's $\tau = 100$ is presented as robust but never swept at the full scale — we see it works, not how sensitive the run is to it.

**What would change my mind.** If a from-scratch reproduction at, say, 50–100B activated parameters showed that QK-Clip can be replaced by plain logit soft-capping plus a smaller learning rate with no loss of token efficiency, the central optimizer contribution would shrink to "a nice convenience." Conversely, if independent SWE-bench harness-normalized evaluations (same scaffold for every model) preserved K2's open-source lead, the agentic claims would harden from "strong, with caveats" to "decisively ahead among open weights."

## What I'd build with this

1. **QK-Clip as a drop-in for any Muon run.** If you are experimenting with Muon on a dense model or a non-MLA MoE, port the per-head clip: track per-head max logits in the forward pass, and after each step scale the offending heads' $W_q, W_k$ by $\sqrt{\tau/S^h_{\max}}$. It's ~20 lines and removes the most common reason Muon runs blow up.
2. **A rephrasing pipeline for your own scarce corpus.** For any domain where you have a small high-quality corpus (internal docs, a niche codebase, regulatory text), the chunk-wise autoregressive rewrite with fidelity verification is a direct way to multiply token utility without renting more data. Start with "rephrase once, train one epoch" and measure on a held-out factual QA set before scaling rephrasings.
3. **A self-critique reward grounded in your verifiable tasks.** If your product has *some* checkable tasks and many subjective ones, train a critic on the checkable rollouts and reuse it as a pairwise judge for the subjective ones, refreshing it on-policy. The closed loop is the part most teams skip.
4. **An agentic-trajectory factory.** Stand up a tool simulator (a stateful world model) plus an LLM user-simulator and an LLM rubric judge, and generate multi-turn tool-use data for *your* tool set — then ground the highest-stakes slice (anything touching real systems) in a sandboxed real-execution environment with objective pass/fail.
5. **A follow-up experiment.** Sweep $\tau$ at a mid scale (say 9B-activated, the size the report used to demonstrate the instability) and plot max-logit decay against final loss. If the "self-stabilizes after ~30%" behavior holds across $\tau \in \{50, 100, 200\}$, you get a principled rule for setting the threshold instead of a single magic number.

## References

1. Kimi Team. *Kimi K2: Open Agentic Intelligence.* [arXiv:2507.20534](https://arxiv.org/pdf/2507.20534).
2. Open checkpoints: [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) (base and instruct).
3. Checkpoint engine for fast RL weight updates: [MoonshotAI/checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine).
4. Keller Jordan et al. *Muon: An optimizer for hidden layers in neural networks.* [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/); Jingyuan Liu et al. *Muon is scalable for LLM training.* [arXiv:2502.16982](https://arxiv.org/abs/2502.16982).
5. DeepSeek-AI. *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).

Related reading on this blog: [MoE architecture, training & inference](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies), [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek), [DeepSeek-V3](/blog/paper-reading/large-language-model/deepseek-v3-2), and [Group Sequence Policy Optimization](/blog/paper-reading/reinforcement-learning/group-sequence-policy-optimization).
