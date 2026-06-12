---
title: "MiniMax-M2: Why They Walked Back Linear Attention for Agents"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - minimax
  - full-attention
  - linear-attention
  - agentic-llm
  - mixture-of-experts
  - interleaved-thinking
  - speculative-decoding
  - prefix-caching
  - tool-use
  - inference-optimization
description: "A deep read of MiniMax-M2: why the team that bet on lightning linear attention publicly reverted to full attention for their agentic model, and the interleaved-thinking serving contract that makes or breaks it."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/minimax-m2-full-attention-agentic-1.png"
readTime: 30
---

The most honest engineering document any frontier lab published in 2025 was a blog post titled, almost apologetically, *"Why did M2 end up as a full attention model?"* It was written by the same team that had spent a year and two flagship models — [MiniMax-01](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) and [MiniMax-M1](/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute) — championing lightning linear attention as the way to escape the quadratic-attention wall. With MiniMax-M2, they threw that bet out and went back to full softmax attention on every single layer. The post is a public negative result, and it is more instructive than most positive ones, because it is a detailed account of where a beautiful idea met a production inference stack and lost.

MiniMax-M2 (released October 23, 2025, [announcement](https://www.minimax.io/news/minimax-m2), open weights under MIT) is a compact, sparse mixture-of-experts built for coding and agentic workflows. It is much smaller in *activated* terms than its predecessors — roughly 10 billion active parameters out of 230 billion total, versus M1's 45.9B active — and it is full-attention everywhere. The two facts are connected: M2 is co-designed around the inference pattern of an agent loop, where a model is called dozens of times in a single task, and both the attention choice and the sparsity are in service of serving that loop fast and cheap. If you want the whole MiniMax lineage in one place, the [combined overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) is the hub; this post is about the reversal and the agentic design.

![MiniMax-M2 at a glance: a compact, sparse, full-attention MoE for agentic serving](/imgs/blogs/minimax-m2-full-attention-agentic-1.png)

The diagram above is the mental model, and it reads as three connected bets. The left column is *sparsity*: 230B total but only ~10B active, with 256 thin experts and top-8 routing. The middle column is the *reversal*: full softmax attention on every one of 62 layers, with grouped-query attention and per-layer QK-norm. The right column is *serving*: FP8 weights, three multi-token-prediction heads for speculative decoding, and the resulting ~100 tokens/second at a tenth of a frontier model's price. Every one of these choices points the same direction — toward an architecture you can run an agent on, cheaply, at high concurrency.

> [!tldr] TL;DR
> - **What it claims:** A compact full-attention MoE (~10B active / 230B total) tuned for agentic and coding workloads, competitive with frontier models on agent benchmarks at roughly 8% of the price.
> - **Why it matters:** MiniMax *publicly reversed* its signature linear-attention bet, arguing that on a real inference stack — prefix caching, speculative decoding, low-precision state — full attention is simply less fragile.
> - **Most surprising finding:** M2's interleaved thinking imposes a hard serving contract: prior-turn `<think>` blocks must be passed back verbatim, or agentic scores collapse (τ²-Bench 87 → 64, BrowseComp 44.0 → 31.4).
> - **Where it's soft:** The M2 post-training/RL recipe is unpublished (the "Forge" framework and 140K-task details belong to the later M2.1, not M2), and the launch "intelligence index" of 61 was later re-baselined to 36 under a harder methodology.

## Context: the agentic serving workload

To understand why M2 looks the way it does, you have to understand the workload it is built for, because it is genuinely different from the workload that justified M1. A reasoning model like M1 is called once and thinks for a long time — a single prompt, one very long generation. An *agent* is called many times in a single task: it plans, calls a tool, reads the result, plans again, calls another tool, and so on for dozens of turns. Each of those turns re-sends a growing conversation history (the prior reasoning, the tool calls, the tool results) and generates a relatively short continuation. The bottleneck is not one long generation; it is *many medium-length forward passes over a long, mostly-repeated prefix*.

That workload rewards a completely different set of optimizations than long-form reasoning does. The single most valuable one is **prefix caching**: because each agent turn re-sends almost the same prefix as the last, a serving stack that can cache the attention state for the unchanged prefix and only compute the new tokens turns an expensive re-processing into a cheap append. The second is **speculative decoding**: generate several tokens with a cheap draft and verify them in one pass of the big model, which works beautifully for the short, often-predictable continuations an agent produces. The third is raw **per-call latency and cost**, because an agent that makes 40 model calls per task multiplies whatever you pay per call by 40.

Contrast this with the reasoning workload that justified M1. There, the win condition is a single long, high-quality chain of thought — one call, one big generation, and the only thing that matters is that the model can think for a long time without the per-token cost exploding. Prefix caching is nearly irrelevant (there is one prefix, used once); speculative decoding helps but is secondary; what dominates is the cost curve of long generation, which is exactly what lightning attention flattened. The two workloads pull architecture in opposite directions. A reasoning model wants cheap long *output* and tolerates a complex recurrent state because it generates one sequence end to end. An agent wants cheap *re-reading* of a long shared prefix and short outputs, which is precisely what a flat, cacheable attention state delivers and a recurrent state does not. MiniMax built M1 for the first workload and M2 for the second, and the architecture diverged because the workloads diverged — which is the cleanest possible illustration that there is no single "best" attention mechanism, only a best one *for a given serving pattern*.

Here is the uncomfortable thing the MiniMax team discovered: lightning linear attention, the architecture that made M1's long *reasoning* cheap, is hostile to exactly these agentic optimizations. The rest of this post is the anatomy of that discovery and what they built instead.

It helps to put numbers on the agent-loop economics, because they are not intuitive. Suppose a coding agent works on a task for 30 turns. By the later turns the conversation history — prior reasoning, tool calls, file contents, test output — might be 50,000 tokens, while each new turn generates maybe 500 tokens of plan-and-tool-call. Without prefix caching, every turn re-processes the entire 50,000-token prefix from scratch, so the task pays roughly $30 \times 50{,}000 = 1.5$ million tokens of *prefill* just to re-read what it already knew, dwarfing the $30 \times 500 = 15{,}000$ tokens of actual generation. With prefix caching, the unchanged prefix is computed once and appended to cheaply, collapsing that 1.5 million-token prefill toward the 50,000 tokens of genuinely new content across the whole task. Prefix caching is not a nice-to-have for agents; it is a 10×-or-more cost reducer, and any architecture that breaks it is starting the serving-cost race 10× behind. That single multiplier is most of why M2's attention reversal pays for itself, even though full attention costs more per token in isolation.

## What M2 is

Before the argument, the spec, because the numbers are the argument's evidence. M2's configuration is public (the weights ship with a readable `config.json`), so unlike the training recipe, the architecture is fully verifiable:

| Property | MiniMax-M2 | (vs M1) |
| --- | --- | --- |
| Total / activated params | ~230B / ~10B | 456B / 45.9B |
| Routed experts / top-k | 256 / top-8 | 32 / top-2 |
| Layers / hidden size | 62 / 3072 | 80 / 6144 |
| Attention | full GQA, every layer | 7:1 lightning hybrid |
| Query / KV heads | 48 / 8 (GQA), head dim 128 | — |
| Normalization | per-layer QK-norm, RMSNorm | RMSNorm |
| RoPE base / rotary dim | 5,000,000 / 64 | 10,000,000 / partial |
| Context | 196,608 trained (~192K), 128K eval | 1M |
| Quantization / decoding | FP8 (e4m3), MTP × 3 modules | bf16 |

Two reversals jump out of that table. The attention reversal is the headline — `attn_type_list` in the config is all `1`s across all 62 layers, meaning full softmax attention everywhere, no lightning layers at all. The MoE reversal is quieter but just as deliberate: M2 swings from M1's 32 fat experts (top-2) all the way to 256 thin experts (top-8), which is the *DeepSeek* granularity, the exact design MiniMax-01 had defined itself against. Both reversals serve the same master — cheap, fast agentic serving — and the next sections trace why.

The design philosophy MiniMax states for M2 is unusually pointed: it is "born for agents and code," and the team frames it with a dogfooding principle — "to create a model that meets our requirements, we must first be able to use it ourselves." That orientation explains the otherwise-surprising willingness to walk back two signature architectural bets in one model. If your north star were maximizing benchmark scores, you would keep whatever squeezed out the last point of reasoning accuracy. If your north star is *being usable as the engine of an agent product* — fast, cheap, high-concurrency, integrable — you optimize the serving stack and accept a few points of peak capability in return. M2 is the second kind of model, and almost every choice that looks like a regression from M1 (smaller active count, full attention's higher per-token FLOPs, the lower headline reasoning scores) is a deliberate trade in favor of the agent-serving objective. Reading the model as "M1 but worse" misses the point; it is "M1 re-pointed at a different goal," and on that goal it is a clear step forward.

You can confirm the attention reversal directly from the shipped config, which is the kind of verification the closed training recipe does not allow:

```python
import json

cfg = json.load(open("config.json"))
## attn_type_list encodes the per-layer attention kind: 1 = full softmax attention.
## M1's hybrid would show a 7:1 pattern of lightning vs softmax; M2 is all ones.
assert all(t == 1 for t in cfg["attn_type_list"]), "expected full attention on every layer"
print(cfg["num_hidden_layers"], "layers, all full attention")   # 62 layers, all full attention
print(cfg["num_local_experts"], cfg["num_experts_per_tok"])      # 256 experts, top-8
print(cfg["num_mtp_modules"])                                    # 3 multi-token-prediction heads
```

## The reversal: why full attention won

The official rationale post is blunt in a way papers rarely are. Its thesis, in MiniMax's own words: "in a real-world, industrial-grade system, the truth is that efficient attention still has some way to go before it can definitively beat full attention," and "there's a vast difference between the promise on paper and its payoff in production." The matrix below collects the specific failures.

![Why M2 reverted to full attention: linear loses across the production inference stack](/imgs/blogs/minimax-m2-full-attention-agentic-2.png)

Read the rows as a checklist of where linear attention's paper advantage evaporates. On **compute**, linear attention is $O(n \cdot d^2)$ versus softmax's $O(n^2 \cdot d)$ — a real asymptotic win — but the MiniMax team found its kernels are *memory-bound* even during training, so the theoretical FLOP savings do not convert to wall-clock savings on a real GPU. On the **efficiency crossover**, the point where linear actually becomes faster than full attention sits "at a few thousand tokens — which isn't particularly long," so for the medium-length forward passes an agent makes, full attention is often already in its favor. On **low-precision state**, the linear-attention recurrent state is sensitive to the low-precision storage that production serving depends on. And then the two that matter most for agents: linear attention **breaks prefix caching** and **breaks speculative decoding**, the precise optimizations the agentic workload is built around. The last row is the deepest: at larger scale, the hybrid showed "clear deficits in complex, multi-hop reasoning tasks" — the recall tax of MiniMax-01 coming due, where one softmax layer per eight was enough for needle-retrieval but not for chaining facts through reasoning.

The "memory-bound" claim deserves unpacking because it is the least intuitive and the most important. GPU kernels fall into two regimes: *compute-bound*, where the chip's arithmetic units are the bottleneck, and *memory-bound*, where they sit idle waiting for data to arrive from memory. A FLOP count only predicts wall-clock time in the compute-bound regime. Linear attention's per-step update is a small matrix operation against a $d \times d$ state — few FLOPs, but it has to read and write that state from memory every step, and at the batch sizes and sequence lengths of real serving, those memory accesses dominate. So the kernel runs at low arithmetic intensity: the GPU's expensive tensor cores are starved, waiting on memory bandwidth, while the FLOP count says the work should be cheap. Full attention, by contrast, is a big dense matmul that keeps the tensor cores busy — it does *more* FLOPs but at high utilization, so on wall-clock it can finish first. This is the gap between "fewer FLOPs" and "less time," and it is exactly the gap a complexity table hides. MiniMax found it the hard way, in production, which is the only place it shows up.

There is one more experiment in the post worth surfacing because it is a specific, hard-won negative result. MiniMax tried a **sliding-window attention** (SWA) variant — local softmax attention with a bounded window, a different way to get linear cost. It "performed extremely poorly on agent tasks and complex long-context evaluations," and "performance degraded noticeably as context length grew — which is unacceptable in agentic scenarios." The diagnosis is subtle and important: global attention patterns like *retrieval heads* and *induction heads* are established early in pretraining, and once a model has been pretrained to rely on them, you cannot reliably restore them with continued pretraining (CPT) on a windowed architecture. The capability is baked in at pretraining time or not at all. This is a deep point about architecture search: some properties are *pretraining-time decisions* that no amount of fine-tuning can retrofit, and the global attention patterns an agent needs to track state across a long task appear to be one of them. You cannot bolt long-range capability onto a windowed model after the fact; you have to pretrain for it, which makes the windowed-attention shortcut a dead end for this workload rather than a tunable knob.

### The efficiency crossover, made concrete

The "few thousand tokens" crossover deserves its own picture, because it is the quiet center of the whole argument.

![The efficiency crossover: linear attention's theoretical win is memory-bound and arrives only past a few thousand tokens](/imgs/blogs/minimax-m2-full-attention-agentic-3.png)

The before-and-after frames the gap between theory and practice. On paper, linear attention has fewer FLOPs, full stop. In practice, those FLOPs are spread across a memory-bound kernel that cannot saturate the GPU's compute units, so the chip sits idle waiting on memory while a compute-bound full-attention kernel runs at high utilization. The crossover where linear's lower FLOP count finally wins on wall-clock is only a few thousand tokens out — and crucially, a lot of agentic serving happens *below* that crossover, in the regime where full attention is simply faster anyway. Add the sliding-window result (a different linear variant that failed on agents) and the conclusion writes itself: for this workload, full attention is the *least fragile* choice across tasks, model sizes, and inference stacks. It is not that linear attention is wrong in some absolute sense; it is that its wins are conditional and its failures are unconditional, and on a production stack you optimize for the floor, not the ceiling.

### Why full attention serves agents fast

The counterintuitive part is that reverting to the *more expensive* attention makes serving *cheaper*, and the figure explains why.

![Full attention keeps prefix caching and speculative decoding working, enabling fast agentic serving](/imgs/blogs/minimax-m2-full-attention-agentic-4.png)

Full attention preserves the two optimizations that dominate agentic serving cost. **Prefix caching** works because a standard attention KV cache is a flat, appendable structure — you can cache the prefix's keys and values and reuse them across turns, paying only for the new tokens. A linear-attention recurrent state is harder to cache and reuse across the partial, branching prefixes an agent produces: the state at position $n$ is a single compressed summary that has already folded in everything before it, so you cannot cleanly "rewind" to a shared prefix and branch, the way an agent that explores several tool calls from the same point needs to. The flatness of the KV cache — often seen as a memory liability — turns out to be a *cacheability asset* for the branching, re-reading access pattern of agents. (For the mechanics of KV-cache reuse and prefix caching in depth, the blog's [KV-cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) post goes through it.) **Speculative decoding** works because M2 ships three multi-token-prediction (MTP) heads: the model predicts several future tokens at once, a cheap draft proposes a span, and one verification pass accepts or rejects it. Both optimizations compound with FP8 weights to produce the headline serving numbers — around 100 tokens/second at high concurrency. The lesson, stated generally, is that *architecture choices should be evaluated on the full inference stack, not on a complexity table*: an attention pattern that breaks prefix caching and speculative decoding can be slower in production than a "more expensive" pattern that keeps them, because those two optimizations are worth more than the asymptotic FLOP difference for this workload.

## The architecture in detail

The attention reversal is the headline, but the MoE reversal is the part that makes the serving math close.

![The MoE granularity flip: from M1's 32 fat experts to M2's 256 thin experts](/imgs/blogs/minimax-m2-full-attention-agentic-5.png)

M1 (inheriting MiniMax-01) used 32 fat experts with top-2 routing, activating 45.9B parameters per token. M2 flips to 256 thin experts with top-8 routing, activating only ~10B. The granularity flip is what lets M2 be *sparse* in a way M1 was not — more, smaller experts mean each token touches a smaller fraction of the total parameters, so the activated count drops by more than 4× even though the total parameter count only roughly halved. For an agentic workload where you are paying per forward pass, dropping activated parameters from 45.9B to 10B is the difference between an expensive model and a cheap one. The router carries a small auxiliary-loss coefficient (0.001) to keep the 256 experts balanced.

Why does fine-grained sparsity help serving specifically? The activated-parameter count is what determines the compute and memory traffic of a single forward pass — and in autoregressive decoding, where you generate one token at a time, that per-pass cost is paid on every token. A model with 10B active parameters streams roughly a fifth as much weight data per decode step as a 45.9B-active model, which directly sets the decode throughput on a memory-bandwidth-bound serving setup. The total parameter count (230B) determines how much *knowledge and capacity* the model can hold and how much GPU memory the weights occupy, but it does not gate per-token decode speed the way the active count does. So M2's design splits the two concerns cleanly: keep a large total for capacity, shrink the active fraction for speed. The 256-thin-expert granularity is the mechanism — you cannot get to 10B active with 32 fat experts and top-2 without making each expert tiny, so you need many small experts to route sparsely while keeping each expert big enough to be useful. This is the same logic that drove DeepSeek's design, and M2 adopting it is MiniMax conceding that for a serving-first model, the DeepSeek granularity was right all along.

Combined with full attention (prefix-cache and speculative-decode friendly), FP8 weights, and the MTP heads, the architecture is a coherent argument that *10B active parameters served fast beats 45.9B served cleverly* when the workload is long agent loops.

A sketch of how MTP feeds speculative decoding, the serving-side payoff of the architecture:

```python
import torch

def speculative_step(model, prefix_kv, last_token):
    # M2 ships 3 MTP heads: from one forward pass, predict the next 3 tokens cheaply.
    draft = model.mtp_heads(prefix_kv, last_token)        # [3] proposed tokens, one cheap pass
    # The main head verifies all 3 in a single batched forward; accept the longest matching run.
    verified = model.verify(prefix_kv, draft)             # full-attention verify reuses the KV cache
    return verified                                       # 1..3 tokens committed per big-model pass
```

## Interleaved thinking and the serving contract

M2's most operationally consequential property is not in the weights at all — it is in how you must call it. M2 is an *interleaved thinking* model: it wraps reasoning in `<think>...</think>` blocks and alternates reasoning with tool calls across a multi-turn loop.

![The M2 interleaved thinking loop: plan, act, observe, with prior thinking carried forward](/imgs/blogs/minimax-m2-full-attention-agentic-6.png)

The loop in the figure is a plan-act-reflect cycle: the model thinks (plans), calls a tool, observes the result, and thinks again — carrying its prior reasoning forward into the next turn. The official framing calls this "alternating between explicit reasoning and tool use, while carrying that reasoning forward between steps," and the benefit is that the model's evolving understanding of the task persists across turns instead of being reconstructed from scratch each time. This is what keeps an agent grounded over a long task: by turn 15, the model still has its turn-3 reasoning about why it ruled out an approach.

The failure mode this prevents has a name worth knowing: *state drift*. When prior reasoning is discarded between turns, the model has to re-infer its own plan from the visible artifacts alone — the tool calls it made and the results it got — without the reasoning that connected them. Each re-inference is slightly different, so the agent's understanding of the task quietly drifts from turn to turn: it re-litigates decisions it already made, forgets why it rejected an approach and tries it again, and loses the thread of a multi-step plan. MiniMax's own framing is precise about the consequences of dropping the thinking: "cumulative understanding breaks down, state drift increases, self-correction weakens, and planning degrades." The interleaved-thinking design is, at bottom, a bet that an agent's *reasoning* is part of its state, not a disposable byproduct of producing an action — and that you must persist it with the same fidelity you persist the actions and observations. Most agent frameworks treat the chain of thought as scratch work to be thrown away once an action is emitted; M2 treats it as load-bearing memory, and the ablation is the proof that for this model it is.

The catch — and it is a hard serving contract, not a suggestion — is that you must pass the prior-turn `<think>` blocks back into the conversation *verbatim*. The docs are explicit: "Do not remove the `<think>...</think>` part, otherwise the model's performance will be negatively affected." And there is a trap baked into the ecosystem: the OpenAI Chat Completions API does not support passing reasoning content back in subsequent requests, so a naive OpenAI-shaped harness *silently strips* the thinking and quietly degrades M2. The fix is to preserve the assistant turn intact:

```python
def record_assistant_turn(messages, reasoning, visible_text, tool_calls):
    # WRONG: strips the model's prior reasoning, the way the OpenAI Chat
    # Completions schema does. M2 degrades hard (tau2-bench 87 -> 64).
    messages.append({"role": "assistant", "content": visible_text})

    # RIGHT: pass the full assistant turn back verbatim, <think> blocks included.
    messages.append({
        "role": "assistant",
        "content": f"<think>{reasoning}</think>{visible_text}",
        "tool_calls": tool_calls,        # M2 uses an XML-style tool-call format
    })
    # Dropping the <think> span between turns collapses planning and self-correction.
```

How much does it matter? The ablation is brutal.

![Keep versus drop the think blocks: agentic scores collapse when prior reasoning is stripped](/imgs/blogs/minimax-m2-full-attention-agentic-7.png)

Dropping prior-turn thinking versus keeping it: τ²-Bench falls from **87 to 64** (a 35.9% relative drop), BrowseComp from **44.0 to 31.4** (40.1%), GAIA from 75.7 to 67.9, SWE-bench Verified from 69.4 to 67.2. These are not rounding errors; on the agentic benchmarks that M2 is built to win, stripping the reasoning between turns costs a third to nearly half of the score. If you integrate exactly one M2-specific thing, it is this: audit your agent framework for where it discards reasoning between turns, because the most popular API shape discards it by default.

The tool-call format itself is XML-style rather than JSON-in-a-string. Tool calls are wrapped in a `minimax:tool_call` block containing an `invoke` with named `parameter` children, tools are declared as JSON Schema inside a `tools` block, and tool results return via a `role: "tool"` message. MiniMax recommends serving M2 on vLLM or SGLang, both of which parse this format; the practical point for an integrator is that M2 is not a drop-in for an OpenAI-tool-calling harness without adapting both the tool format and the reasoning-preservation behavior.

The choice of an XML-style format over JSON is not arbitrary, and it interacts with the interleaved-thinking design. JSON tool calls require the model to emit syntactically perfect JSON — balanced braces, correctly escaped strings — in a single uninterrupted span, which is fragile when the model is also weaving reasoning around the call. An XML-ish format with named parameter tags is more robust to the model thinking mid-structure and is easier to parse incrementally as it streams, which matters when you are interleaving `<think>` blocks with tool invocations in the same generation. The deeper point for anyone building on M2 is that the *output contract* is part of the model, not just the weights: the format, the reasoning-preservation rule, and the recommended serving engines together define how you must integrate it, and getting any one of them wrong silently degrades the model. This is a recurring theme in 2025-era agentic models — the integration surface is as load-bearing as the architecture, and a model that is excellent in a chat box can underperform badly in an agent harness that violates its contract.

## Economics

The whole point of ~10B active parameters is what it does to the bill.

![Agentic economics: M2's ten-billion active parameters undercut Claude Sonnet 4.5's price](/imgs/blogs/minimax-m2-full-attention-agentic-8.png)

M2 is priced at **$0.30 per 1M input tokens and $1.20 per 1M output tokens** — roughly 8% of Claude Sonnet 4.5's price — and serves at around 100 tokens/second (Artificial Analysis independently measured 111), which is about double Sonnet's measured speed. The matrix makes the trade legible: M2 wins decisively on every cost-and-speed axis and loses the quality axis (SWE-bench Verified 69.4 vs Sonnet's 77.2). That is the honest shape of the value proposition — not "as good as Sonnet," but "most of the way there at a tenth of the cost and twice the speed," which for an agentic workload that makes dozens of calls per task can be the better deal even at lower per-call quality.

One honest caveat from Artificial Analysis: M2 is *verbose*, spending around 120M tokens to complete their full evaluation suite (tied for the highest measured, against a median nearer 43M). Verbosity partially eats the per-token price advantage, because you pay for all those tokens — so the effective cost-per-task gap is narrower than the headline 12× price ratio suggests. Walk the arithmetic: if M2 spends roughly 2.5× the median model's tokens on a task, then its 12× lower output price translates to only about a 5× lower cost-per-task, not 12×. That is still a large advantage, but it is the right number to plan a budget around, and it is the number the per-token sticker price hides. The general lesson for evaluating any cheap-but-verbose model is to measure *cost per completed task*, integrating over the tokens the model actually spends, rather than comparing per-token prices — a chatty model at a low per-token rate can cost more per task than a terse model at a higher rate. Cheap tokens that you spend more of are not as cheap as they look.

## Experiments

M2's benchmark profile is exactly what you would predict from its design: strong on agentic and coding tasks, competitive but not leading on pure reasoning.

| Benchmark | MiniMax-M2 | Claude Sonnet 4.5 | GPT-5 (thinking) | DeepSeek-V3.2 |
| --- | --- | --- | --- | --- |
| SWE-bench Verified | 69.4 | 77.2 | 74.9 | 67.8 |
| Terminal-Bench | 46.3 | 50.0 | 43.8 | 37.7 |
| BrowseComp | 44.0 | 19.6 | 54.9 | 40.1 |
| τ²-Bench | 77.2 | — | — | — |
| GAIA (text) | 75.7 | — | — | — |
| AIME 2025 | 78.0 | 88.0 | 94.0 | 88.0 |
| GPQA-Diamond | 78.0 | 83.0 | 85.0 | 80.0 |
| LiveCodeBench | 83.0 | 71.0 | 85.0 | 79.0 |

The agentic and coding rows are where M2 earns its keep — BrowseComp 44.0 beats Sonnet's 19.6 by a wide margin, LiveCodeBench 83.0 beats Sonnet, and SWE-bench Verified 69.4 is within striking distance of the frontier while costing a fraction. The pure-reasoning rows (AIME, GPQA) trail the frontier, which is consistent with a model that traded some raw reasoning headroom for serving efficiency.

It is worth being explicit about what that trade buys and costs, because it is the central design decision and it is not free. M2 gives up some peak reasoning capability — the AIME 2025 gap to GPT-5 (78 vs 94) is real, and a 10B-active model simply has less to bring to a hard novel math problem than a much larger one. What it buys is the ability to run an agent loop *many times* per task at low latency and cost, which is a different axis of usefulness entirely. For a coding agent that makes 40 calls to fix a bug, a model that is 90% as good per call but 10× cheaper and 2× faster can complete more tasks per dollar and per minute than a marginally smarter, much pricier model — because the agent's quality comes as much from being able to *iterate* (try, observe, correct) as from any single call being brilliant. M2 is a bet that, for agents, throughput-weighted capability beats peak per-call capability. Whether that bet is right depends entirely on your workload: for one-shot hard-reasoning queries, the frontier models win; for long iterative agent loops, M2's economics can dominate. The benchmark table read in isolation makes M2 look like a second-tier model; read through the lens of cost-and-speed-per-task, it looks like a specialist that wins its niche. At launch, Artificial Analysis ranked M2 as the **#1 open-weights model** with an Intelligence Index of 61 (top-5 globally), which is the framing MiniMax cites — though one honest correction for the record: AA later moved to a harder v4.0 methodology under which M2 shows 36 (rank ~#33), a different and non-comparable index, not a regression.

The open-weights angle is a meaningful part of M2's significance, separate from the raw scores. At launch it was the highest-scoring open-weights model on the Artificial Analysis index, which matters because the agentic-coding niche it targets is exactly where teams most want to self-host — you do not want your coding agent's entire codebase flowing through a third-party API on every turn, and you want to control latency and cost rather than rent them. A capable, cheap-to-serve, MIT-licensed model that you can run on your own hardware is a different product from an equally-capable closed API, and M2's whole design — small active count, full-attention serving friendliness, FP8 weights — is what makes self-hosting it actually practical rather than a research curiosity. The benchmark ranking is the headline; the deployability is the substance.

A word on what is *not* in the report: M2's post-training recipe. The launch materials do not publish the RL details. The family's later writing confirms M2 "largely continued to rely on CISPO" (the objective from [M1](/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute)), but the granular reward-design and data details — the in-house "Forge" RL framework, the 140K augmented tasks — belong to the December 2025 **M2.1** writeup and should not be attributed to M2 itself. For M2, the architecture is fully documented and the training is essentially a black box.

## The arc: a bet and its retreat

Step back and the lineage tells a single story about conviction and evidence.

![The attention bet across the MiniMax lineage: committed for 01 and M1, reversed for M2](/imgs/blogs/minimax-m2-full-attention-agentic-9.png)

MiniMax bet hard on lightning linear attention for MiniMax-01 (January 2025) and doubled down for M1 (June 2025), where the hybrid's cheap long generation was the whole economic case for scaling test-time compute. Then, for the agentic M2 (October 2025), they reversed it — and the rationale in the figure is the crux: linear attention loses on the *production stack*, not on the benchmark. Three releases, two architectures, one honest about-face in the middle. This is not a story of a failed idea. It is a story of an idea whose domain of validity turned out to be narrower than hoped: linear attention is genuinely good for long-form reasoning where you generate one long sequence, and genuinely bad for agentic serving where you make many medium passes over a cached prefix. The intellectual honesty of publishing that distinction — rather than quietly shipping M2 and hoping nobody asked — is what makes this corpus worth reading as a body of work rather than a sequence of press releases.

## Critique

What is strong here is unusual: a frontier lab published a *negative result about its own prior bet*, with specifics. The "why full attention" post names the exact failure modes (memory-bound kernels, the few-thousand-token crossover, broken prefix caching and speculative decoding, multi-hop deficits at scale, the SWA retrieval-head problem), and each is a concrete, checkable claim rather than a vibe. The interleaved-thinking ablation is similarly concrete and immediately actionable — the keep-versus-drop numbers are the kind of thing that saves an integrator weeks of confusion. The architecture is fully open and verifiable from the config, which is more than most "open" releases offer.

What is soft is the training side and the index. M2's RL recipe is unpublished, so the most interesting question — *how* do you post-train a small full-attention model to be this good at agents — is unanswered for M2 specifically; you have to wait for the M2.1 writeup, and then carefully not back-port its details onto M2. The launch "intelligence index of 61, top-5 globally" framing aged badly when AA re-baselined to a harder methodology that puts M2 at 36; both numbers are real, but quoting the launch one without the re-baseline is misleading. And the verbosity caveat means the headline price advantage is softer than it looks once you account for how many tokens M2 actually spends.

There is also a subtler gap in the reversal argument itself, worth naming because it is easy to over-read. The "why full attention" post is an account of MiniMax's *own* experiments — their hybrid at their ratio (7:1), their sliding-window variant, their serving stack. It is a strong existence proof that *this team, with these constraints, found full attention less fragile*. It is not a proof that linear attention is unviable for agents in general, and the post is careful not to claim that. The broken prefix caching and speculative decoding are properties of *current* serving implementations, not laws of nature — there is active research on cacheable and speculatively-decodable linear-attention variants, and the memory-bound-kernel problem is partly an artifact of kernels that have had far less engineering investment than FlashAttention. So the honest reading is narrower than the headline: full attention won *for MiniMax, in 2025, on the available tooling*. That is a real and useful result, but it is a snapshot of an engineering trade-off, not a closed verdict on the architecture.

**What would change my mind** about "full attention won": a published, scale-matched experiment showing a *less aggressive* hybrid — say 3:1 or 1:1 lightning-to-softmax, rather than M1's 7:1 — that closes the multi-hop-reasoning gap while keeping prefix caching and speculative decoding workable through engineering rather than abandoning them. M2's argument is really two claims bundled: "7:1 was too aggressive for reasoning" and "the serving ecosystem isn't ready for linear attention." The first is probably about the ratio; the second is about tooling that could mature. If someone demonstrated a hybrid that a serving stack could prefix-cache and speculatively decode — the hard engineering MiniMax decided was not worth it — the reversal would look less like "linear attention is wrong" and more like "linear attention was early," which is a meaningfully different conclusion.

## What I'd build with this

1. **An agent gateway that enforces the reasoning-preservation contract.** A middleware that intercepts assistant turns and guarantees `<think>` blocks survive into the next request — refusing to silently strip them through an OpenAI-shaped API — would turn M2's 35-40% agentic degradation into a non-issue, and the same pattern protects any reasoning model with persistent thinking state.

2. **A prefix-cache-and-speculative-decode benchmark for attention variants.** The MiniMax reversal rests on claims about how attention patterns interact with these two optimizations. A reusable harness that measures prefix-cache hit rate and speculative-decode acceptance for any attention variant would let the field settle the "is linear attention serveable" question with numbers instead of blog posts.

3. **A verbosity-aware cost model for agentic workloads.** Because M2 is cheap-per-token but verbose, the real metric is cost-per-completed-task, not cost-per-token. A small evaluation harness that tracks tokens-to-completion across models would price agent deployments correctly, where the per-token sticker price misleads.

4. **A migration adapter from OpenAI-tool-calling to M2's XML format.** M2's XML tool-call format plus the reasoning-preservation requirement make it a non-drop-in for the dominant harness shape. A clean adapter (tool-schema translation plus think-block preservation) would remove the main friction to actually using M2 in an existing agent stack.

5. **A self-hosted coding-agent reference stack on M2.** The pieces — vLLM or SGLang serving with prefix caching and MTP speculative decoding enabled, FP8 weights, the reasoning-preservation middleware, and the XML tool adapter — are individually documented but not assembled anywhere as a turnkey reference. Putting them together as a reproducible "run a capable coding agent on your own GPUs" stack would be the most direct way to test M2's central claim: that a small, full-attention, serving-optimized model is the right engine for an agent you actually own and operate, not just a benchmark entry. The whole design only pays off if the serving stack is set up to exploit it, and most of M2's apparent advantages evaporate on a naive deployment that disables prefix caching or strips the think blocks.

## References

- MiniMax-M2 — [announcement](https://www.minimax.io/news/minimax-m2) · [why full attention](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) · [why interleaved thinking](https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2) · [model card](https://huggingface.co/MiniMaxAI/MiniMax-M2) · [config.json](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)
- Sibling MiniMax reads on this blog: [the combined overview](/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo) · [MiniMax-01 foundation](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) · [MiniMax-M1 and CISPO](/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute)
- Related: [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) · [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) · [Fine-tuning tool-calling LLMs](/blog/machine-learning/training-techniques/fine-tuning-tool-calling-llms-when-how) · [Serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems)
