---
title: "Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - kimi-k2-thinking
  - mixture-of-experts
  - reasoning-models
  - agentic-tool-use
  - int4-quantization
  - test-time-scaling
  - moonshot-ai
  - open-weights
description: "A principal-engineer walkthrough of Kimi K2 Thinking: how Moonshot turned a trillion-parameter MoE into a long-horizon reasoning agent with native INT4, 200-300 step tool loops, and a heavy parallel mode."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-k2-thinking-1.png"
readTime: 32
---

There is a specific failure mode that kills most "agentic" language models in production, and it has nothing to do with their reasoning ceiling. You hand the model a research task that requires forty tool calls — search, read, cross-check, compute, search again — and somewhere around step thirty-five it forgets what it was doing. The chain of thought drifts. It re-searches a query it already answered, contradicts a fact it established ten steps ago, or simply declares victory on a sub-goal it never finished. The model was never *wrong* at any single step; it lost the thread across steps. This is the long-horizon coherence problem, and it is the wall that Kimi K2 Thinking is built to break through.

Kimi K2 Thinking is Moonshot AI's open-weight reasoning model. It is not a fresh trillion-token training run — it is the existing Kimi K2 base model (a strong *non-thinking* agentic MoE) extended into a long-horizon reasoning agent that interleaves step-by-step chain-of-thought with tool calls, and sustains that interleaving across **200 to 300 sequential tool invocations** where prior models degraded after 30 to 50. It ships under a Modified MIT License with native **INT4 quantization-aware training** that roughly doubles generation speed, a **256K** context window, and a "heavy mode" that runs eight reasoning trajectories in parallel and reflectively aggregates them. On the benchmarks Moonshot reports, it sets open-source state of the art on Humanity's Last Exam with tools, BrowseComp, Seal-0, and Frames.

![The K2 Thinking interleaved reason-and-act loop](/imgs/blogs/kimi-k2-thinking-1.png)

The diagram above is the mental model: K2 Thinking is a cycle, not a pipeline. Every step the model emits private reasoning (with a per-step token budget of 24K to 48K tokens), then *decides* whether to answer or call a tool. If it calls a tool — python for computation, web search for retrieval — the observation is fed back into context and the loop repeats. The entire capability story of this release is about making that loop stay coherent for hundreds of iterations without the inference cost exploding. Everything else — the INT4 quantization, the heavy mode, the inherited high-sparsity MoE — exists to make long, deep, tool-augmented reasoning *affordable and stable* at trillion-parameter scale.

One sourcing note up front, because it shapes how much you should trust each number in this article. **There is no dedicated arXiv technical report for K2 Thinking.** The only formal Moonshot paper, arXiv 2507.20534 ("Kimi K2: Open Agentic Intelligence"), explicitly describes the *non-thinking* base model — its own abstract says it achieves state of the art "among open-source non-thinking models." The Thinking variant is documented in a blog post and a HuggingFace model card. So in this article, the *architecture and pretraining* (MoE, MuonClip, MLA, the 15.5T-token recipe) come from a real peer-style report and are high-confidence; every *Thinking-specific* fact (INT4 QAT, 256K context, the reasoning and tool-use benchmarks, heavy mode, the 200-300 step claim) comes from the blog and model card and is medium-confidence — no independent replication exists yet. I will flag the seam wherever it matters.

> [!tldr] TL;DR
> - **What it claims:** A 1T-parameter / 32B-active open-weight MoE that reasons step-by-step while calling tools, stays coherent across 200-300 sequential tool calls, runs in native INT4 for ~2x faster generation, and sets open-source SOTA on agentic-search and tool-augmented reasoning benchmarks (HLE with tools 44.9, BrowseComp 60.2, Seal-0 56.3, Frames 87.0).
> - **Why it matters:** It is the first *open-weight* model to credibly contest closed frontier reasoning agents (GPT-5, Claude Sonnet 4.5 Thinking, Grok-4) on long-horizon agentic tasks — and it does so at INT4 serving cost, not BF16.
> - **Most surprising finding:** The INT4 quantization is applied *during* post-training via QAT and the published benchmark numbers are the INT4 numbers — Moonshot calls it "lossless." The headline scores are not a BF16 model later squeezed down; they are native INT4.
> - **Where it fails:** It trails GPT-5 High on pure no-tools math (AIME25 94.5 vs 94.6) and Claude Sonnet 4.5 on SWE-bench Verified (71.3 vs 77.2). Its strongest results lean on tool access and an 8x-cost heavy mode; the raw no-tools deltas versus the frontier are much smaller.

## Context: what came before

To understand what K2 Thinking *adds*, you have to know what base K2 already was. Kimi K2 (the July 2025 model, arXiv 2507.20534) was Moonshot's bet that you could build a frontier-class *agentic* model — one good at tool use, coding, and multi-step task execution — without it being a "reasoning model" in the o1/R1 sense. It was deliberately a non-thinking model: it produced answers directly, used tools, but did not emit long deliberate chains of thought before acting. Its claim to fame was being state of the art *among open-source non-thinking models*, and it got there through three architectural and training bets that K2 Thinking inherits wholesale.

The first bet was **high-sparsity MoE**. Base K2 uses 384 experts with 8 activated per token — a sparsity of 48 — versus DeepSeek-V3's 256 experts with 8 activated. The report's argument is empirical: increasing the total expert count (raising sparsity) "consistently lowers both training and validation loss," and at a fixed validation loss of 1.5, sparsity 48 cuts the FLOPs needed by **1.69x** compared to sparsity 8. They paired this with a deliberately *small* attention configuration — 64 heads, half of DeepSeek-V3's 128 — because doubling heads buys only 0.5% to 1.2% quality but costs +83% inference FLOPs at 128K context. That is a model designed from the start to be cheap at inference, which matters enormously once you start doing 300-step tool loops.

The second bet was **MuonClip**, the optimizer. Muon is token-efficient but, at trillion scale, it makes attention logits explode and the loss spikes. Moonshot's fix is QK-Clip: per attention head, compute the maximum softmax input logit $S_{max}$; if $S_{max}$ exceeds a threshold $\tau = 100$, rescale that head's query and key projection weights by $\gamma = \min(1, \tau / S_{max})$. Because "only a small subset of heads exhibit exploding logits," the clipping is per-head and selective, and the headline result is a 15.5-trillion-token pretraining run with **zero loss spikes**. The third bet was the **agentic post-training pipeline** — large-scale synthesis of tool-use trajectories (from 3,000+ real MCP tools and 20,000+ synthetic ones) followed by a joint RL stage that mixes verifiable rewards with a self-critique rubric.

So the gap K2 Thinking fills is precise. Base K2 had the architecture, the optimizer, and the agentic data, but it was a *non-thinking* model that, like everyone else's agents, lost coherence after a few dozen tool steps. The Thinking release asks: can we take that exact checkpoint, teach it to interleave long chain-of-thought with tool calls, push the coherent horizon from ~50 steps to ~300, and do it without making inference so expensive that test-time scaling becomes impractical? The answer they ship combines long-CoT/agentic post-training with native INT4 — and that combination is the genuinely new contribution.

![What changes from base K2 to K2 Thinking](/imgs/blogs/kimi-k2-thinking-3.png)

The before-and-after above isolates the three deltas. Base K2 degrades after 30-50 tool steps, has no interleaved long chain-of-thought, and serves at 128K context in BF16. K2 Thinking stays coherent across 200-300 steps, reasons-then-acts with a per-step CoT budget, and serves at 256K context in INT4 at roughly 2x the speed. Note what is *not* on the "after" side: a new pretraining run. The trillion-parameter foundation is the same — that is the whole point of building Thinking *on* K2.

## Contributions

Distilling the dossier, here is what K2 Thinking actually contributes that base K2 did not:

1. **A long-horizon reasoning agent that interleaves CoT with tool calls and stays coherent across 200-300 sequential tool invocations** — a roughly 6x increase over the 30-50 step coherence horizon of prior models, including base K2 itself.
2. **Native INT4 quantization-aware training applied during post-training**, weight-only on the MoE components, yielding ~2x faster generation while keeping benchmark scores at SOTA. The published numbers *are* the INT4 numbers; Moonshot describes the quantization as lossless.
3. **A heavy mode for parallel test-time scaling** — roll out eight trajectories simultaneously, then reflectively aggregate them into a final answer — which powers the top benchmark columns (HLE 51.0, AIME25 100.0).
4. **A clean three-regime evaluation protocol** (no tools / with tools / heavy) that, by reporting the same benchmark across all three, lets you read off how much of the gain comes from raw reasoning versus tool use versus parallel aggregation.
5. **A 256K context window** (up from base K2's 128K), giving the long tool loops room to retain their full trajectory in context.

The first four are the substance. Everything downstream of architecture — the MoE design, MuonClip, the agentic data synthesis — is *inherited* and the article will treat it as foundation, not as a Thinking-specific contribution.

## Method

### Shared K2 architecture

K2 Thinking and base K2 share an identical backbone; the only architectural deltas the model card lists are the 256K context and native INT4. Let me lay out the structure precisely, because the numbers explain why this model can afford to think for 300 steps.

![K2 Thinking architecture (1T sparse MoE, INT4)](/imgs/blogs/kimi-k2-thinking-2.png)

The stack above reads bottom to top as a forward pass. Tokens enter through a 160K-vocabulary embedding. Each of the **61 layers** (1 dense layer plus 60 sparse MoE layers) applies **Multi-head Latent Attention (MLA)** with **64 heads** over a hidden dimension of **7,168**, followed by a **Mixture-of-Experts feed-forward network** using **SwiGLU** activations. The MoE bank holds **384 experts** of hidden dimension 2,048 each; a router selects **8** per token plus **1 shared expert** that always fires. That is where the trillion parameters live, and it is why only **32B activate per token**: $8/384$ routed experts plus the shared one is a sparsity of **48**.

Here is the relationship that makes the whole thing tractable. Total capacity scales with the *number* of experts, but per-token compute scales with the number *activated*. Writing $E$ for total experts, $k$ for activated experts, and $d$ for the per-expert hidden dimension, the parameter count of the MoE bank is roughly proportional to $E \cdot d$ while the per-token FLOPs are proportional to $k \cdot d$. K2 pushes $E$ to 384 while holding $k$ at 8, so it buys representational capacity at a sparsity of $E/k = 48$ without paying for it on every forward pass. The report's ablation backs this directly: at a fixed validation loss of 1.5, sparsity 48 needs 1.69x fewer FLOPs than sparsity 8.

| Spec | Value | Why it matters for Thinking |
|---|---|---|
| Total parameters | 1T (1.04T) | Capacity for broad knowledge and tool competence |
| Activated per token | 32B | Per-step inference stays cheap across 300 steps |
| Layers | 61 (1 dense + 60 MoE) | Depth for multi-step reasoning |
| Attention | MLA, 64 heads, dim 7,168 | 64 heads (not 128) saves +83% FLOPs at 128K ctx |
| Experts | 384 total / 8 active / 1 shared | Sparsity 48; capacity without compute |
| MoE hidden dim | 2,048 per expert | — |
| Vocabulary | 160K | — |
| Context length | **256K** | Holds full 300-step trajectory in context |
| Activation | SwiGLU | — |
| Quantization | **Native INT4 (weight-only, MoE) via QAT** | ~2x faster generation, lossless per Moonshot |

The single most important design choice for a *thinking* model is the combination of high sparsity and the small attention configuration. A 300-step tool loop with a 24K-token reasoning budget per step can accumulate an enormous context. If your per-token cost were proportional to total parameters, test-time scaling would be a non-starter. Because K2 activates only 32B of its 1T parameters and uses 64 rather than 128 attention heads, the cost of *thinking longer* is bounded by activated compute, not total compute. The architecture was, in retrospect, built for this.

Let me make the sparsity argument concrete with a back-of-envelope worked example, because it is the load-bearing economic decision behind the whole release. Suppose you have a fixed activated-parameter budget — say the 32B that K2 settles on — and you can spend it either as *few large experts* or *many small experts*. With 8 activated experts of hidden dimension 2,048, each token's MoE FLOPs are proportional to $k \cdot d = 8 \cdot 2048$. Now hold that activated cost fixed and grow the *total* expert pool from 256 to 384. Per-token compute does not move — you still route to 8 experts — but the model's representational capacity grows by 50%, and the report's measurement is that this *lowers* both training and validation loss. The FLOP saving is the inverse statement: to *reach* a target validation loss of 1.5, the sparsity-48 configuration needs 1.69x fewer total FLOPs than sparsity-8 would. For a thinking model that generates millions of tokens per task, a 1.69x training-efficiency multiplier compounds into a very different cost curve than a dense or low-sparsity model would face.

The attention-head choice tells the same story from the inference side. At 128K context, the KV-cache and attention compute scale with the number of heads. Doubling from 64 to 128 heads buys 0.5% to 1.2% quality — a rounding error on most benchmarks — at the price of +83% inference FLOPs in the attention path. For a model whose entire value proposition is *running for 300 steps*, paying 83% more attention compute for sub-1.5% quality is exactly the wrong trade, and K2 declines it. This is the kind of decision that looks unremarkable in a spec table and turns out to be the difference between a model you can serve in a long agentic loop and one you cannot.

### Native INT4 quantization-aware training

This is the technical core of the Thinking release, and it is worth being precise about what it is and is not. Most "quantized" model releases you see are *post-training quantization* (PTQ): you train in BF16, then squeeze the weights down to INT8 or INT4 afterward, and you eat some accuracy loss. K2 Thinking does the opposite. It applies **quantization-aware training during the post-training phase** — the model learns *with* the INT4 constraint in the loop, so the weights settle into values that quantize cleanly. The quantization is **weight-only** (activations stay higher precision) and **applied to the MoE components**, which is where the overwhelming majority of the trillion parameters sit.

The payoff Moonshot reports is roughly **2x generation speed** — lower latency and lower GPU memory — with benchmark scores maintained at SOTA, which they describe as "lossless." The critical consequence: the published benchmark numbers are *native-INT4* numbers, not a post-hoc lossy compression of a BF16 model. When you read HLE 44.9 or BrowseComp 60.2 later in this article, you are reading the INT4 model's scores.

Why does QAT matter so much for a *thinking* model specifically? Because test-time scaling multiplies the cost of every token you generate. A model that produces a 200-token answer pays the quantization tax once; a model that reasons for 300 steps at 24K tokens per step generates millions of tokens per task. Halving the per-token generation cost is the difference between "long-horizon reasoning is a research demo" and "long-horizon reasoning is something you can actually serve." Here is a minimal sketch of what weight-only INT4 QAT looks like in a forward pass, using the standard straight-through estimator so gradients can flow through the (non-differentiable) rounding:

```python
import torch
import torch.nn as nn

def quantize_int4_weight_only(weight, group_size=128):
    """Symmetric weight-only INT4 quantization, per output-channel groups.
    Returns dequantized weights so the matmul runs in higher precision math
    but on values constrained to the INT4 grid (4 bits -> 16 levels)."""
    out_features, in_features = weight.shape
    w = weight.reshape(out_features, in_features // group_size, group_size)
    # per-group scale: map the group's max-abs onto the signed INT4 range [-8, 7]
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0
    q = torch.clamp(torch.round(w / scale), -8, 7)   # the INT4 grid
    deq = (q * scale).reshape(out_features, in_features)
    return deq, scale

class QATExpertLinear(nn.Module):
    """One expert's linear projection, trained quantization-aware.
    The forward uses a straight-through estimator: quantize on the
    forward pass, pass gradients through the round() unchanged."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        deq, _ = quantize_int4_weight_only(self.weight)
        # straight-through: forward sees quantized weights, backward sees real ones
        w_eff = self.weight + (deq - self.weight).detach()
        return torch.nn.functional.linear(x, w_eff)
```

The straight-through trick on the second-to-last line is the whole idea: the forward pass uses `deq` (the INT4-constrained weights) so the model *learns under the deployment constraint*, but the gradient flows to the full-precision `self.weight` because the `(deq - self.weight).detach()` term has zero gradient. After enough post-training steps, the full-precision weights settle into a configuration whose INT4 projection barely changes the loss — which is what "lossless" quantization means in practice. The dossier does not report the exact QAT schedule, group size, or an INT4-vs-BF16 ablation, so treat the code above as the *standard* recipe this description implies, not a verbatim reproduction of Moonshot's internals.

To see why weight-only INT4 is the right knob to turn, compare the precision options on the two axes that actually constrain a long-horizon agent: memory footprint and generation throughput. A trillion parameters stored in different precisions occupies wildly different amounts of GPU memory, and memory bandwidth — not compute — is the bottleneck during autoregressive generation, where you stream every weight once per token.

| Weight precision | Bytes/param | ~1T-param weight footprint | Relative gen. speed | Where it lands |
|---|---|---|---|---|
| BF16 | 2 | ~2 TB | 1.0x (baseline) | Accurate but bandwidth-bound |
| FP8 | 1 | ~1 TB | ~1.5x | Common training-storage choice |
| INT4 (PTQ, post-hoc) | 0.5 | ~0.5 TB | ~2x | Fast, but accuracy usually drops |
| **INT4 (QAT, native)** | **0.5** | **~0.5 TB** | **~2x** | **Fast and lossless per Moonshot** |

The point of the table is the last two rows. Post-training INT4 and QAT INT4 occupy the *same* memory and run at the *same* ~2x speed — the difference is entirely in accuracy. PTQ takes a BF16 model and rounds its weights after the fact, so the model never had a chance to adapt; QAT folds the rounding into training so the model compensates. Moonshot's claim is that the QAT version closes the accuracy gap to zero. If true, you get the throughput and memory of INT4 with the benchmark scores of a full-precision model — and for a workload that streams every weight millions of times per task, halving the bytes-per-weight roughly halves the bandwidth cost of generation. That is the mechanism behind the "~2x faster generation" number.

There is a subtlety worth flagging: the quantization is **weight-only**, not weight-and-activation. Activations stay at higher precision, which preserves the dynamic range needed for the long reasoning traces and avoids the activation-outlier problems that plague aggressive activation quantization in large models. It is also confined to the **MoE components** — the experts hold the overwhelming bulk of the parameters, so quantizing them captures almost all of the memory savings while leaving the comparatively tiny attention and embedding weights alone. This is a targeted, surgical quantization, not a blunt cast-everything-to-INT4, and that targeting is part of why "lossless" is even plausible.

### Agentic test-time scaling

The reason-and-act loop from Figure 1 is the behavioral core. Each step, the model generates reasoning tokens (the "thinking" trace), then decides among three actions: answer, call python, or call web search. Tool observations are appended to context, and the loop continues. The headline claim is *coherence across length*: where earlier agents "degrade after 30-50 steps," K2 Thinking sustains performance across **200-300 consecutive tool invocations**.

The evaluation budgets make the scale concrete. Agentic-search evaluations use a **300-step maximum with a 24K-token reasoning budget per step**. The HLE-with-tools evaluation uses **120 steps with a 48K-token budget per step**. Multiply those out and you see why INT4 is not optional: a single HLE-with-tools episode can generate on the order of millions of tokens of reasoning across its 120 steps. Here is a pseudocode skeleton of the loop that those budgets parameterize:

```python
def k2_thinking_agentic_loop(task, tools, max_steps=300, step_token_budget=24_000):
    """Interleave chain-of-thought with tool calls until the model answers
    or the step budget is exhausted. Tool results re-enter context each step."""
    context = [system_prompt(), user_message(task)]
    for step in range(max_steps):
        # 1. Reason: emit a private chain-of-thought under a per-step token cap
        thought = model.generate(context, max_tokens=step_token_budget, mode="think")
        context.append(assistant_thinking(thought))

        # 2. Decide: answer, or pick a tool with arguments
        action = model.decide_action(context, available=tools)
        if action.kind == "final_answer":
            return action.answer                      # terminates the loop

        # 3. Act: execute python or web search, observe the result
        observation = tools[action.name].run(action.arguments)
        context.append(tool_observation(action.name, observation))

        # 4. Coherence is the hard part: the 256K window must still hold
        #    the early trajectory so step 250 remembers step 3's finding.
        context = compact_if_needed(context, window=256_000)

    return model.generate(context, mode="answer")     # forced answer at budget
```

The genuinely hard engineering is in step 4. With a 256K context window, the model must keep the *early* trajectory legible at step 250 so it does not re-derive or contradict what it established at step 3. This is exactly the long-horizon coherence problem from the introduction, and the combination of the 256K window plus the long-CoT agentic post-training is Moonshot's answer to it. The dossier does not disclose the long-CoT SFT mixture or the precise RL recipe used to instill this behavior — those are explicitly *not stated* — so we know the *what* (300-step coherence) and the *budgets*, but not the full *how*.

It is worth doing the arithmetic on these budgets, because they explain why this capability was infeasible before native INT4. Consider the two evaluation harnesses the dossier specifies:

| Eval harness | Max steps | Token budget/step | Worst-case reasoning tokens |
|---|---|---|---|
| Agentic search | 300 | 24K | 300 × 24K = 7.2M |
| HLE with tools | 120 | 48K | 120 × 48K = 5.76M |

A single worst-case agentic-search episode can generate on the order of **7.2 million reasoning tokens** across its 300 steps — not counting the tool observations that re-enter context. At BF16 generation speed, an evaluation suite of a few hundred such episodes would be punishingly slow and expensive; at INT4's ~2x throughput, the same suite is twice as cheap to run, and more importantly, *serving* such an agent to real users becomes economically defensible. This is the concrete sense in which INT4 and long-horizon reasoning are co-designed: the quantization is not a separate optimization bolted on for marketing, it is the enabling condition for the headline behavior. Take the INT4 away and the 300-step loop is a research curiosity; keep it and the loop is a product.

The 256K context window is the other enabling condition, and the math there is just as direct. A 300-step loop where each step contributes thousands of tokens of reasoning plus a tool observation would overflow base K2's 128K window well before step 300 — you would be forced to truncate or summarize the early trajectory, which is precisely how coherence is lost. Doubling the window to 256K roughly doubles how much of the trajectory can stay verbatim in context, which is why the context-length bump is not a cosmetic spec change but a direct enabler of the coherence claim. The `compact_if_needed` call in the pseudocode is where a deployment makes the hard choices about *what* to drop when even 256K is not enough; the dossier does not specify Moonshot's compaction strategy, so that line is a placeholder for an engineering decision you would have to make yourself.

### Heavy mode: parallel test-time scaling

Sequential depth (one long trajectory) is one axis of test-time scaling; parallel width is the other. Heavy mode exploits width. Quoting the description: K2 Thinking Heavy Mode "rolls out eight trajectories simultaneously, then reflectively aggregates all outputs to generate the final result." Eight independent reasoning runs, then a reflective aggregation step that reconciles them into one answer. This is the engine behind the top benchmark columns — HLE 51.0 and AIME25 100.0 are heavy-mode numbers.

The mechanism is intuitive: independent trajectories explore different solution paths and make uncorrelated errors, so aggregating eight of them is a form of self-consistency with an explicit reflective reconciliation rather than a naive majority vote. The cost is equally intuitive: heavy mode multiplies inference compute by roughly 8x. That is the tradeoff the caution-colored node in the next figure encodes — heavy mode buys the last few benchmark points at eight times the cost, which is fine for a leaderboard and rarely justified in production.

### Three operation regimes

The cleanest thing about this release's evaluation is that it reports many benchmarks in all three regimes, which turns the benchmark table into an *ablation* of where the capability comes from.

![Three operation regimes of K2 Thinking](/imgs/blogs/kimi-k2-thinking-6.png)

The tree above shows the three regimes and a representative score for each. **No tools** is raw reasoning — HLE 23.9, AIME25 94.5. **With tools** adds python or search — HLE jumps to 44.9, and agentic-search benchmarks like BrowseComp (60.2) become possible at all. **Heavy mode** adds 8x parallel rollouts on top — HLE 51.0, AIME25 100.0. Read the gaps and you learn the decomposition: on HLE, tools add ~21 points (23.9 to 44.9) and heavy adds another ~6 (44.9 to 51.0); on AIME25, the model is already near-saturated without tools (94.5), so tools and heavy mostly close the last few points to 100.0. That is an honest way to report — it makes clear that the headline gains on knowledge-and-search tasks are *tool-driven*, while the math gains are *reasoning-driven* with tools as polish.

### How Thinking is produced from K2

Pulling the method together, here is the production lineage — and the most important thing to internalize is that there is no second pretraining run.

![From base K2 pretraining to K2 Thinking](/imgs/blogs/kimi-k2-thinking-4.png)

The pipeline above starts with the base K2 pretraining (MuonClip optimizer, 15.5T tokens, zero loss spike), continues through SFT and agentic data synthesis (3,000+ real MCP tools, 20,000+ synthetic), then the joint RL stage (verifiable rewards plus self-critique rubric). The two Thinking-specific additions bolt onto the *end*: the long-CoT/agentic-RL that instills 300-step coherence, and the **INT4 QAT** applied during post-training. The output is K2 Thinking — same trillion-parameter foundation, 256K context, ~2x faster serving. This is why I keep stressing the lineage: K2 Thinking's reproducibility floor is base K2's, and everything genuinely new lives in the last two boxes, only one of which (INT4 QAT) is described in any detail.

For completeness, the inherited pretraining recipe (from the base K2 report, high-confidence): the learning-rate schedule held a constant **2e-4 for the first 10T tokens**, then cosine-decayed **2e-4 to 2e-5 over the next 5.5T tokens**. Long-context annealing ran **400B tokens at 4K sequence length, then +60B tokens at 32K**. The data spanned four domains — Web Text, Code, Mathematics, Knowledge — with a "learning-note" rewriting style for math and a chunk-wise autoregressive rephrasing scheme for knowledge that, on SimpleQA, beat naive multi-epoch repetition. None of that is Thinking-specific, but it is the foundation the Thinking behavior is fine-tuned on top of.

### Training infrastructure (inherited)

The serving and training footprint is worth surfacing because it sets the bar for who can actually run this model, and because the INT4 story is partly a *response* to that footprint. From the base K2 report (high-confidence): training ran on **NVIDIA H800 GPUs** with an **8 × 400 Gbps RoCE** node interconnect. The parallelism strategy was **16-way Pipeline Parallel × 16-way Expert Parallel × ZeRO-1 Data Parallel**, with roughly **6 TB of GPU memory** for parameters and gradients spread across a **256-GPU model-parallel group**. To fit, they leaned on selective recomputation, FP8 storage, and CPU offload for activations. The RL stage *colocated* train and inference engines on the same workers, and a dedicated checkpoint engine updated parameters in **under 30 seconds** — which matters for RL, where you want the policy and the rollout engine to stay in lockstep without a long stall every update.

| Infra dimension | Base K2 value | Implication for Thinking |
|---|---|---|
| Training hardware | NVIDIA H800 GPUs | H-class GPUs assumed for both train and serve |
| Interconnect | 8 × 400 Gbps RoCE | Expert-parallel routing needs fast all-to-all |
| Parallelism | 16 PP × 16 EP × ZeRO-1 DP | MoE bank sharded across experts |
| Model-parallel group | ~6 TB GPU mem / 256 GPUs | Large footprint; INT4 cuts the serving share |
| RL checkpoint update | < 30 seconds | Keeps policy and rollout engine synced |
| Recommended inference | vLLM, SGLang, KTransformers | Where you would deploy the INT4 weights |

The thread that ties this back to Thinking: a trillion-parameter model is heavy to serve, and INT4 is precisely the lever that makes the *inference* footprint manageable even though the *training* footprint stays large. The recommended inference engines — **vLLM, SGLang, KTransformers** — are the runtimes where the released INT4 weights are meant to run, and they are where you would build the 300-step agentic loop from Figure 1 in practice. The GPU-hours and total training cost for the Thinking variant are *not stated* anywhere in the dossier, so this is the footprint of the foundation, not a measured cost of the Thinking post-training on top of it.

## Experiments

The results are reported across reasoning, general knowledge, agentic search, and coding, with competitors GPT-5 (High), Claude Sonnet 4.5 (Thinking), the prior non-thinking K2 0905, DeepSeek-V3.2, and Grok-4. All K2 Thinking numbers are native-INT4. An asterisk in the original tables marks results the K2 team re-tested under identical conditions. Let me start with the headline matrix.

![K2 Thinking vs frontier reasoning models](/imgs/blogs/kimi-k2-thinking-5.png)

The matrix above is the story in one image: K2 Thinking (green column) wins the agentic-search and tool-augmented reasoning rows — HLE with tools (44.9 vs GPT-5's 41.7), HLE heavy (51.0 vs GPT-5's 42.0), BrowseComp (60.2 vs GPT-5's 54.9 and Claude's 24.1), Seal-0 (56.3), Frames (87.0) — and loses (caution yellow) on SWE-bench Verified (71.3 vs Claude's 77.2). The pattern is consistent: K2 Thinking is at or above the frontier when the task rewards long, tool-augmented, search-heavy reasoning, and slightly behind on pure software engineering.

Here are the full reasoning numbers in all three regimes:

| Benchmark | Setting | K2 Thinking | Best competitor cited |
|---|---|---|---|
| Humanity's Last Exam | no tools | 23.9 | GPT-5 26.3; Grok-4 25.4; DeepSeek-V3.2 19.8 |
| Humanity's Last Exam | with tools | **44.9** | GPT-5 41.7; Grok-4 41.0; Claude 32.0 |
| Humanity's Last Exam | heavy | **51.0** | GPT-5 42.0; Grok-4 50.7 |
| AIME 2025 | no tools | 94.5 | GPT-5 94.6; Grok-4 91.7 |
| AIME 2025 | with python | 99.1 | Claude 100.0; GPT-5 99.6 |
| AIME 2025 | heavy | 100.0 | GPT-5 100.0; Grok-4 100.0 (tie) |
| HMMT 2025 | no tools | 89.4 | GPT-5 93.3; Grok-4 90.0 |
| HMMT 2025 | with python | 95.1 | GPT-5 96.7; Grok-4 93.9 |
| HMMT 2025 | heavy | 97.5 | GPT-5 100.0; Grok-4 96.7 |
| GPQA | — | 84.5 | not reported |
| IMO-AnswerBench | — | 78.6 | not reported |

Read the HLE rows carefully because they carry the headline. With no tools, K2 Thinking (23.9) trails GPT-5 (26.3) and Grok-4 (25.4) — its raw reasoning is competitive but not leading. Add tools and it jumps to 44.9, beating every listed competitor including GPT-5 (41.7). Heavy mode pushes it to 51.0, edging past Grok-4's 50.7 for the top spot. The math story is the inverse: K2 Thinking is strong but a hair behind on raw AIME25 (94.5 vs GPT-5's 94.6) and HMMT, and tools/heavy mostly close gaps rather than open leads.

The agentic-search results (all with tools) are where K2 Thinking is genuinely dominant:

| Benchmark | K2 Thinking | GPT-5 (High) | Claude 4.5 | K2 0905 | DeepSeek-V3.2 |
|---|---|---|---|---|---|
| BrowseComp | **60.2** | 54.9 | 24.1 | 7.4 | 40.1 |
| BrowseComp-ZH | 62.3 | 63.0 | 42.4 | 22.2 | 47.9 |
| Seal-0 | **56.3** | 51.4 | 53.4 | 25.2 | 38.5 |
| FinSearchComp-T3 | 47.4 | 48.5 | 44.0 | 10.4 | 27.0 |
| Frames | **87.0** | 86.0 | 85.0 | 58.1 | 80.2 |

The BrowseComp gap is the most striking number in the entire release: 60.2 versus Claude Sonnet 4.5's 24.1 and the prior non-thinking K2 0905's 7.4. That ~53-point jump over its own predecessor is the clearest single measurement of what the Thinking treatment buys — base K2 simply could not sustain the multi-step browsing required, and Thinking can. The general and coding numbers round out the picture:

| Benchmark | K2 Thinking | GPT-5 (High) | Claude 4.5 | K2 0905 | DeepSeek-V3.2 |
|---|---|---|---|---|---|
| MMLU-Pro | 84.6 | 87.1 | 87.5 | 81.9 | 85.0 |
| MMLU-Redux | 94.4 | 95.3 | 95.6 | 92.7 | 93.7 |
| Longform Writing | 73.8 | 71.4 | 79.8 | 62.8 | 72.5 |
| HealthBench | 58.0 | 67.2 | 44.2 | 43.8 | 46.9 |
| SWE-bench Verified | 71.3 | 74.9 | 77.2 | 69.2 | 67.8 |
| SWE-bench Multilingual | 61.1 | 55.3 | 68.0 | 55.9 | 57.9 |
| Multi-SWE-bench | 41.9 | 39.3 | 44.3 | 33.5 | 30.6 |
| LiveCodeBench V6 | 83.1 | 87.0 | 64.0 | 56.1 | 74.1 |
| SciCode | 44.8 | 42.9 | 44.7 | 30.7 | 37.7 |
| Terminal-Bench | 47.1 | 43.8 | 51.0 | 44.5 | 37.7 |

On general knowledge, K2 Thinking trails the closed frontier on MMLU-Pro/Redux but wins HealthBench (58.0, well above GPT-5's 67.2 — wait, GPT-5 leads here; K2 leads Claude's 44.2 and DeepSeek's 46.9 decisively). On coding it is mixed: it leads LiveCodeBench V6 (83.1) over Claude (64.0), wins SWE-bench Multilingual and Multi-SWE-bench against GPT-5, but loses SWE-bench Verified to both GPT-5 and Claude.

What is load-bearing here, and what might not transfer? The load-bearing claim is the agentic-search dominance, and it is load-bearing in a specific way: it depends on the model actually *executing* 100-300 tool calls under the stated budgets. If you deploy K2 Thinking with a tight step cap, a small token budget per step, or without web/python tools, you collapse it back toward the no-tools column — where it trails GPT-5. The BrowseComp 60.2 is not a property of the weights alone; it is a property of the weights *plus the 300-step / 24K-token harness*. The heavy-mode numbers transfer even less cleanly to production: an 8x inference cost is rarely worth ~6 benchmark points. And every one of these numbers comes from Moonshot's own evaluation with no independent replication captured in the dossier — the asterisked competitor results were re-run by the K2 team, which is good practice, but the K2 numbers themselves were not third-party verified.

## Critique

**What's strong.** Three things genuinely impress me. First, the INT4 QAT framing is the right one: by quantizing *during* post-training and reporting the INT4 model as the headline model, Moonshot sidesteps the usual "but the quantized version is worse" asterisk. If the lossless claim holds up under independent testing, this is the correct way to ship a model that has to do millions of tokens of reasoning per task. Second, the three-regime evaluation is honest engineering communication — reporting no-tools / with-tools / heavy for the same benchmark lets a careful reader decompose the gains, and most labs would have just shown you the best column. Third, the BrowseComp 60.2-vs-7.4 jump over base K2 is a clean, large, hard-to-fake measurement of the actual contribution.

**What's weak or unfalsifiable.** The reproducibility floor is the central weakness, and it is structural: there is no technical report. The Thinking-specific long-CoT SFT data, the RL recipe that instills 300-step coherence, the GPU-hours, and even the INT4-vs-BF16 ablation are all *not stated*. "Lossless INT4" is an extraordinary claim presented without the one ablation that would substantiate it — a side-by-side of the BF16 and INT4 checkpoints on the same benchmarks. As written, "lossless" is not falsifiable from the published material; you have to take it on faith or wait for the community to quantize-and-test independently. The "200-300 sequential tool calls" coherence claim is similarly under-specified: we are told the eval *budgets* (300 steps, 24K tokens) but not a coherence-vs-step-count curve showing *where* and *how gracefully* performance degrades, which is exactly the measurement that would prove the headline.

**What ablation is missing.** The single most important missing experiment is INT4 vs BF16 on the full benchmark suite — without it, the load-bearing "lossless" and "native INT4 numbers" claims are unverified. A close second is a coherence-degradation curve: plot benchmark score against the step cap (say, 30 / 50 / 100 / 200 / 300 steps) for the same task, so we can *see* that base K2 falls off at 30-50 and Thinking holds to 300. Third, an honest accounting of heavy mode's cost-benefit: a Pareto plot of accuracy versus inference FLOPs across no-tools / tools / heavy would let practitioners decide whether the 8x is ever worth it for them.

**What would change my mind.** If an independent group quantizes the released BF16 weights (or trains a BF16 control) and reproduces the INT4 benchmark numbers within noise, the lossless claim graduates from marketing to fact and this becomes one of the most important open-weight releases of the year. Conversely, if independent BrowseComp evaluation under a *standardized, lower* step budget shows the 60.2 collapsing toward the 40s, then the headline is a harness artifact and my assessment drops accordingly.

## What I'd build with this

1. **A cost-bounded long-horizon research agent.** The natural application is exactly what the benchmarks measure: deep-research agents that browse, read, cross-check, and compute over dozens of steps. Because the model is open-weight and INT4, you can self-host the agent loop and tune the step budget to your latency/quality envelope rather than paying per-token to a closed API. I would wire the loop from Figure 1 directly, with `compact_if_needed` tuned to keep the early trajectory salient.

2. **A heavy-mode-as-a-dial product feature.** Expose the no-tools / with-tools / heavy regimes as an explicit user-facing "effort" control. Most queries get with-tools; flagged hard queries get heavy mode's 8x rollout. The three-regime structure is practically a product spec — surface it.

3. **An INT4-QAT fine-tuning recipe for domain agents.** Since the quantization is baked in during post-training, the interesting extension is domain-specific continued post-training that *keeps* the INT4 constraint — fine-tune a legal or biomedical agent that inherits the 2x serving speed. This requires Moonshot to release (or the community to reverse-engineer) the QAT schedule, which is currently not stated.

4. **A coherence-probe harness.** Build the missing ablation as a reusable tool: a benchmark that runs the same agentic task at step caps of 30/50/100/200/300 and reports a degradation curve. This would both validate the 300-step claim and serve as a regression test for any fine-tune that risks eroding long-horizon coherence.

5. **A tool-call-budget autoscaler.** Because per-step cost is bounded by 32B activated parameters, you can afford a controller that *dynamically* extends the step budget when the model's confidence is low and truncates it when the answer stabilizes — turning the fixed 300-step cap into an adaptive one, which is where most of the real-world cost savings would come from.

## When to reach for K2 Thinking (and when not to)

Reach for K2 Thinking when your task is **long-horizon, tool-augmented, and search-heavy**, and when **self-hosting an open-weight model at INT4 serving cost** is a hard requirement rather than a nice-to-have. Deep research, multi-step web investigation, financial-search workflows, and agentic pipelines that genuinely execute dozens to hundreds of tool calls are its sweet spot — that is where the BrowseComp 60.2, Seal-0 56.3, and Frames 87.0 numbers come from, and where it beats closed frontier models. If you are building such an agent and you need the weights in your own infrastructure (for data-residency, cost, or customization reasons), there is currently no better open option on these benchmarks.

Do *not* reach for it as a drop-in "smartest model" without the agent harness around it. Stripped of tools, its no-tools HLE (23.9) trails GPT-5 (26.3), and its raw AIME25 (94.5) is a hair behind. If your workload is short-form Q&A, single-shot generation, or pure software engineering, the closed frontier (Claude on SWE-bench Verified, GPT-5 on MMLU-Pro) still leads, and you would be paying a 1T-parameter serving footprint — INT4 helps, but you still need H-class GPUs — for capability you are not using. And treat heavy mode as a leaderboard tool, not a default: the 8x inference cost rarely justifies the marginal points in production. Finally, until an independent group validates the lossless-INT4 and 300-step-coherence claims, weight your confidence accordingly — the *architecture* is well-documented in the base K2 report, but the *Thinking-specific* story rests on a blog post and a model card, with the key ablations not yet stated.

## References

- **Kimi K2 Thinking — official blog (primary for Thinking):** https://moonshotai.github.io/Kimi-K2/thinking.html
- **Kimi K2 Thinking — HuggingFace model card:** https://huggingface.co/moonshotai/Kimi-K2-Thinking
- **Kimi K2: Open Agentic Intelligence — base technical report (arXiv abstract, shared architecture + pretraining):** https://arxiv.org/abs/2507.20534
- **GitHub repository:** https://github.com/MoonshotAI/Kimi-K2

Related reading on this blog:

- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the non-thinking base model and the architecture, MuonClip, and agentic post-training that Thinking inherits.
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) — the optimizer lineage behind MuonClip and QK-Clip.
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — Moonshot's earlier RL-for-reasoning work that sets up the long-CoT direction.
- [Kimi K2.5: Visual Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2-5) — the multimodal continuation of the K2 line.
