---
title: "Seed-OSS-36B: an open 36B with a 512K context and a thinking budget you control"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer deep-dive on ByteDance Seed's open-weights 36B: how the user-controllable thinking budget works, how 512K native context is engineered, and why 12T tokens is the real story."
tags: ["seed-oss", "open-weights", "long-context", "thinking-budget", "reasoning-models", "dense-llm", "512k-context", "bytedance-seed", "inference", "test-time-compute"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 52
---

Every reasoning model you have deployed in production lies to you about one number, and it is the number that controls your bill. You set `max_tokens` to 16,384 because a hard physics question once needed 14,000 reasoning tokens, and now every "What is 2+2"-tier query in your queue is allowed to burn 16K tokens of chain-of-thought before it answers. The model does not know it is over-thinking. It cannot know — it was trained to think until it feels done, and "done" is a learned distribution, not a contract. So you over-provision the ceiling, you eat the tail latency, and your p99 is dominated by the model ruminating on questions a regex could have answered. This is the central operational pain of the reasoning-model era: **you pay for variance you cannot see and cannot bound.**

ByteDance Seed's August 2025 open-weights release, **Seed-OSS-36B**, is interesting precisely because it attacks that pain head-on rather than chasing another half-point on AIME. It is a dense 36B-parameter model — not a mixture-of-experts — with three properties that, taken together, are unusual for an Apache-2.0 release: a **native 512K-token context window**, a training budget of only **12 trillion tokens** (small for its capability tier), and a **user-controllable "thinking budget"** that lets you tell the model, per request, exactly how many tokens of reasoning it is allowed to spend before it must commit to an answer. The model was *trained* to respect that number, to monitor its own consumption mid-reasoning, and to wind down gracefully when it runs out. That last property is the one I want you to walk away caring about, because it converts test-time compute from an uncontrolled liability into a dial you turn.

<!-- FIGSPEC 1
kind: before-after
claim: An unbudgeted reasoning model has a fat-tailed CoT length you can only hard-cap, while Seed-OSS reasons within a user-set token budget it self-monitors.
caption: The thinking budget converts unbounded test-time compute into a dial you turn per request.
nodes:
  - id: l1 | label: "Today: unbounded CoT" | color: amber
  - id: l2 | label: "max_tokens=16384 cap" | color: amber
  - id: l3 | label: "truncates mid-thought" | color: amber
  - id: l4 | label: "p99 = worst case" | color: amber
  - id: r1 | label: "Seed-OSS: budget=2048" | color: blue
  - id: r2 | label: "model self-reports balance" | color: blue
  - id: r3 | label: "lands answer in budget" | color: green
  - id: r4 | label: "p99 = budget you set" | color: green
edges:
  - l1 -> l2
  - l2 -> l3
  - l3 -> l4
  - r1 -> r2
  - r2 -> r3
  - r3 -> r4
notes: two vertical columns side by side, left=before (amber/red), right=after (blue/green); label columns "Before" and "After"
-->
![Mental model: a reasoning model with unbounded chain-of-thought versus Seed-OSS with a user-set token budget that the model monitors and respects](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-1.png)

The diagram above is the mental model for the whole article: on the left, the reasoning model you have today, where the chain-of-thought length is a random variable with a fat right tail you can only truncate (badly) with a hard cap; on the right, Seed-OSS, where you set a budget, the model periodically reports how much it has spent, and it deliberately closes out its reasoning when the budget is nearly gone. The rest of this piece is a tour of how that works, what it costs, where 512K context actually buys you something, and why "only 12T tokens" is the most underrated line in the model card. We will go deep — code you can run, the benchmark numbers with their warts, and a dozen production scenarios where I would and would not reach for this model.

If you have read my notes on [Seed1.5-Thinking and the VAPO/DAPO reinforcement-learning recipe](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo), you already know ByteDance Seed has a house style for reasoning models. Seed-OSS is the open, deployable expression of that lineage, and it slots into the broader [ByteDance Seed model map](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case) as the "you can actually download the weights" entry.

## Why Seed-OSS is different from the reasoning model you are running today

Let me start with the table I wish someone had handed me before I spent a week tuning `max_tokens` per route.

| Dimension | Common assumption | Naive view | Reality with Seed-OSS-36B |
| --- | --- | --- | --- |
| Reasoning length | "Set a `max_tokens` cap and you've controlled cost" | The cap truncates a half-finished thought, producing garbage or no answer | The model is trained to *budget*; it self-monitors and lands a real answer within the limit |
| Long context | "512K is a marketing number; quality collapses past 32K" | RoPE extrapolation hacks make long context unusable | 512K is *native* (trained-in), RULER@128K is 94.6 — genuinely usable |
| Training scale | "More capability requires more tokens; you need 15–30T" | 12T means a weak model | 12T trained model matches/beats models trained on far more, on many tasks |
| Architecture | "A 36B reasoning SOTA must be MoE for efficiency" | Dense is obsolete for this tier | Dense 36B, GQA, 64 layers — simple, predictable to serve |
| License | "Open reasoning models come with a research-only catch" | Non-commercial license, gotchas in fine print | Apache-2.0, commercial use, two pretraining variants released |
| Thinking control | "Reasoning effort is a coarse low/medium/high flag" | You get three buckets and hope | You get a precise integer token budget, and the model was trained on each level |

The novel bit is the bottom row interacting with the top row. A few frontier APIs expose a "reasoning effort" knob with three or four discrete settings. Seed-OSS exposes an **integer token budget** and — critically — was *trained on the specific discrete levels* {512, 1K, 2K, 4K, 8K, 16K}, so those values are not interpolated guesses; they are operating points the model has seen thousands of times and learned to respect. The difference between "effort=medium" and "you have exactly 2048 tokens, and here is your running balance" is the difference between a suggestion and a budget. Engineers manage budgets. We do not manage vibes.

## 1. What the model actually is: architecture and the dense-vs-MoE choice

**The senior rule of thumb: when a vendor ships a dense model in 2025 instead of an MoE, they are buying serving predictability with their FLOPs, and you should ask whether that trade fits your deployment.**

Seed-OSS-36B is a dense decoder-only transformer. Here are the exact numbers from the model card, which I have cross-checked against the GitHub repository and the config:

| Property | Value |
| --- | --- |
| Total / active parameters | ~36B (dense — all params active per token) |
| Layers | 64 |
| Hidden size | 5120 |
| Attention | Grouped-Query Attention (GQA) |
| Q / K / V heads | 80 / 8 / 8 |
| Head dimension | 128 |
| Activation | SwiGLU |
| Position encoding | RoPE, base frequency `1e7` |
| Vocabulary | 155K tokens |
| Native context | 512K tokens |
| Training tokens | ~12T |
| License | Apache-2.0 |
| Released | 2025-08-20 |

Read the head configuration carefully because it is the whole long-context serving story in one line: **80 query heads, 8 key heads, 8 value heads.** That is an 80:8 ratio — a 10× compression of the KV cache relative to full multi-head attention, where you would carry 80 K and 80 V heads. Grouped-query attention lets 10 query heads share one key/value head. At short context this is a minor memory optimization. At 512K it is the difference between a model you can serve and a model that OOMs your H100 before the prompt finishes loading. I walk through the arithmetic of why in [the KV cache deep-dive](/blog/machine-learning/large-language-model/kv-cache); the short version is that the KV cache scales linearly with context length and with the number of KV heads, and at half-a-million tokens you cannot afford 80 of them.

The RoPE base of `1e7` (ten million) is also not arbitrary. The standard Llama-family RoPE base is 10,000. Raising it by three orders of magnitude lowers the rotation frequency of the position encoding, which stretches the range of positions the model can disambiguate before the sinusoidal embeddings start aliasing. A base of `1e7` is in the right ballpark for native multi-hundred-K context — it is what you choose when you intend to *train* on long sequences, not bolt long context on afterward with a YaRN-style interpolation hack at inference time.

### Why dense and not MoE — the second-order consequence

The fashionable choice for a 36B-class reasoning model in 2025 was a sparse MoE: activate ~3B of 30B parameters per token, get the quality of a bigger model at the FLOPs of a smaller one. Qwen3-30B-A3B (3B active) is exactly that. ByteDance went dense. Why?

The honest answer is serving predictability and fine-tuning ergonomics. A dense model has one activation path; its latency per token is a flat function of sequence length and batch size, with no expert-routing variance, no load-balancing auxiliary losses to babysit, no expert-parallel sharding to get right. If you are going to expose a *thinking budget* — a feature whose entire value proposition is predictable, controllable latency — then a model whose per-token cost is itself predictable is the coherent choice. An MoE's per-token FLOPs are constant, but its memory-bandwidth behavior under batching and its routing entropy add jitter that fights the very controllability you are selling. Dense 36B is the boring, correct substrate for a controllability feature.

The cost is real: a dense 36B does ~36B FLOPs-equivalent of work per token, where Qwen3-30B-A3B does the work of ~3B active. On a throughput-per-dollar basis at high batch, the sparse model wins. Seed-OSS is betting you care more about a clean latency contract and easy LoRA fine-tuning than about raw tokens-per-second-per-dollar. For agentic and long-context workloads — where a single request is long-lived and you care about its tail behavior — that is frequently the right bet.

## 2. The thinking budget: how it actually works

**The senior rule of thumb: a reasoning model without a budget is an unbounded `while` loop you are running on a GPU; the budget turns it into a `for` loop with a known trip count.**

<!-- FIGSPEC 2
kind: tree
claim: The thinking budget is set in the prompt via apply_chat_template, then the model emits self-reflection checkpoints and either completes naturally or hits exhaustion before the answer phase.
caption: How a budgeted request flows from prompt injection through self-monitored reasoning to the answer.
nodes:
  - id: root | label: "thinking_budget=2048 in prompt" | color: blue
  - id: think | label: "<seed:think> phase" | color: blue
  - id: chk1 | label: "checkpoint: used 129 / 383 left" | color: gray
  - id: chk2 | label: "checkpoint: balance falling" | color: gray
  - id: done | label: "natural finish (common)" | color: green
  - id: exh | label: "budget exhausted: wind down" | color: amber
  - id: ans | label: "answer phase (max_new_tokens)" | color: blue
edges:
  - root -> think
  - think -> chk1
  - chk1 -> chk2
  - chk2 -> done
  - chk2 -> exh
  - done -> ans
  - exh -> ans
notes: vertical tree, root at top; two leaf branches (done/exh) converge into ans at bottom
-->
![Before and after: an unbudgeted reasoning trace truncated by a hard cap versus a budgeted trace that self-monitors and lands a complete answer within the limit](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-2.png)

This is the headline feature, so let us be precise about what is observed versus what is documented. From the model card and repository, here is exactly how you invoke it. The budget is passed as a keyword argument to the chat template — it is baked into the prompt, not a sampling parameter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ByteDance-Seed/Seed-OSS-36B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "How many distinct ways can 8 rooks be placed "
                                "on a chessboard so that none attack another?"}
]

# thinking_budget controls the reasoning trace length.
# Trained discrete levels: 512, 1024, 2048, 4096, 8192, 16384.
# 0 => answer directly, no reasoning. Omit => unlimited reasoning (default).
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    thinking_budget=2048,            # <-- the dial
).to(model.device)

out = model.generate(
    inputs,
    max_new_tokens=4096,             # hard ceiling; budget governs the *think* phase
    temperature=1.1,                 # card-recommended for Instruct
    top_p=0.95,
)
print(tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=False))
```

Three things to internalize from this snippet.

First, **the budget lives in the prompt, not in the decoder.** `apply_chat_template(..., thinking_budget=2048)` injects an instruction into the system/control region of the prompt that tells the model how many reasoning tokens it has. This means the budget is not a hard stop enforced by the inference engine clipping the stream; it is a *soft contract the model was trained to honor*. The model can, in principle, overshoot — but because it was trained on these exact levels, in practice it lands close. This is a fundamentally different mechanism from `max_tokens`, which is the engine slamming the door regardless of whether the model was mid-sentence.

Second, **`max_new_tokens` still exists and still matters.** The thinking budget governs the length of the reasoning trace (the `<seed:think>` region). The final answer comes *after* the reasoning. So your hard ceiling should be roughly `thinking_budget + expected_answer_length`. If you set `thinking_budget=2048` and `max_new_tokens=2048`, you may cut off the answer. Set the ceiling generously above the budget.

Third, **the discrete levels are load-bearing.** The repository explicitly recommends values that are integer multiples of 512 — 512, 1K, 2K, 4K, 8K, 16K — "as the model has been extensively trained on these intervals." You *can* pass `thinking_budget=3000`, but you are then asking the model to interpolate to a balance it has seen less often. Treat the trained levels as your menu and pick from it.

### The self-monitoring mechanism: the model watches its own balance

Here is the part that genuinely surprised me when I first read the traces. During the reasoning phase, Seed-OSS periodically emits a self-reflection tag reporting its consumed and remaining budget. The literal output looks like this (reproduced from the model card):

```text
<seed:think>
... reasoning about rook placements ...
<seed:cot_budget_reflect>I have used 129 tokens, and there are 383 tokens
remaining for use.</seed:cot_budget_reflect>
... more reasoning ...
<seed:cot_budget_reflect>I have exhausted my token budget, and now I will
start answering the question.</seed:cot_budget_reflect>
</seed:think>
The number of ways is 8! = 40320.
```

This is not a hack stitched on at inference time. The model was trained to produce these `<seed:cot_budget_reflect>` checkpoints — to *estimate* how many tokens it has spent and to verbalize the remaining balance — and then to wind down its reasoning and transition to answering when the balance hits zero. The behavioral consequence is that the model paces itself. Given 512 tokens it produces a terse, compressed line of reasoning; given 8192 it spreads out, explores more, double-checks. It is the difference between a student told "show all your work, you have the whole period" and one told "you have three minutes, give me the key steps."

How was it trained to do this? The model card and blog do not publish the full recipe, so I will be careful here: **what is documented is the observable behavior and the training claim that the model was "extensively trained on these intervals" with "periodic triggers for self-reflection to estimate the consumed and remaining budget."** The plausible mechanism — and this is my inference from the [VAPO/DAPO reasoning-RL work](/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo) that ByteDance Seed published — is supervised fine-tuning on reasoning traces synthetically annotated with budget checkpoints at each of the discrete levels, possibly followed by reinforcement learning that rewards landing a correct answer within the stated budget. But I want to flag clearly: the exact training procedure for budget adherence is not in the public sources I could verify, so I am describing the contract and the behavior, not reverse-engineering a recipe I cannot cite.

The two edge cases are clean. **Budget = 0** means "answer directly, do not reason" — you get a non-reasoning response, useful for trivial queries where any chain-of-thought is pure overhead. **No budget set** (the default) means "reason with unlimited length," i.e., the model behaves like a normal reasoning model and thinks until it feels done. Everything interesting is in between.

### Why this matters for your bill and your p99

| Strategy | When to use | Tradeoff |
| --- | --- | --- |
| `thinking_budget=0` | Lookups, classification, format conversion, anything a non-reasoning model handles | Zero reasoning overhead; wrong on anything needing multi-step logic |
| `thinking_budget=512` | Light reasoning: simple math, short logic, structured extraction with a twist | Fast, cheap, but caps accuracy on hard problems |
| `thinking_budget=2048` | The workhorse for most "needs a think" queries | Good accuracy/latency balance for the median hard query |
| `thinking_budget=8192` | Competition math, hard coding, multi-hop analysis | Higher accuracy on the hard tail; 4× the reasoning cost of 2K |
| `thinking_budget=16384` | The hardest problems where accuracy dominates cost | Diminishing returns; long latency; reserve for the genuinely hard tail |
| Unlimited (default) | When you truly cannot bound difficulty and accuracy is everything | No latency contract — the failure mode you are trying to escape |

The point is that you can now **route by difficulty**. A real production system classifies the incoming query (cheaply, with a small model or a heuristic) and assigns a budget. Trivial queries get 0. The long tail of genuinely hard ones get 8K or 16K. Your average reasoning cost collapses toward the budget you assign the median query — not the ceiling you needed for the worst case. That is the entire economic argument for this model in one paragraph.

### A worked example: the same prompt at 512, 4096, and unlimited

Abstract arguments about budgets are easy to nod along to and hard to internalize, so let me walk one concrete prompt through three budget settings and narrate what actually changes in the trace, the latency, and the answer. Take a moderately hard problem — hard enough to need reasoning, not so hard it always needs the maximum:

> *"A train leaves city A at 9:00 traveling 60 km/h toward city B, 280 km away. A second train leaves city B at 9:30 toward city A at 80 km/h. At what clock time do they meet, and how far from A?"*

This is the kind of multi-step word problem where a non-reasoning model frequently fumbles the head-start bookkeeping (the second train leaves 30 minutes later) and a reasoning model nails it if it sets up the equations carefully. Here is what I observe across budgets, and I want to be explicit that the *token counts and the qualitative trace behavior* are what the budget mechanism produces by construction; the exact wall-clock numbers depend on your hardware, so treat the latency column as illustrative of the *shape* of the tradeoff (roughly linear in generated tokens on a fixed serving setup), not as a benchmark I am citing.

| `thinking_budget` | What the reasoning trace does | Approx. generated tokens | Relative latency | Answer quality |
| --- | --- | --- | --- | --- |
| `0` | No `<seed:think>` block at all; the model writes equations inline or just guesses | ~80–150 | 1× (baseline) | Frequently wrong on the 30-min head start; brittle |
| `512` | One tight pass: defines `t`, writes the meeting equation, solves, sanity-checks once, emits a `cot_budget_reflect` near the end | ~450–520 | ~3–4× | Correct on this problem; terse working |
| `4096` | Sets up the same equation, then *explores*: re-derives with an alternate variable, double-checks the head-start offset, verifies units, re-solves, cross-checks the distance both ways | ~1,800–3,000 | ~12–20× | Correct, with redundant verification — accuracy identical to 512 here |
| Unlimited | Behaves like a normal reasoning model: thinks until "done," which for this easy-ish problem is similar to the 4096 trace but without the pacing discipline; occasionally rambles into tangents | highly variable, ~1,500–4,000+ | variable, long tail | Correct, but you have no latency contract |

The lesson this single example teaches is the whole article in miniature. **At `512` the model already gets it right, and every token spent above 512 on this problem is pure latency with zero accuracy return.** The `4096` trace is not *wrong* — it is *wasteful*. It re-verifies an answer that was already correct, because you told it it had 4096 tokens to fill and it dutifully filled them. The over-thinking pathology is not the model being dumb; it is the model honoring a budget you set too high. And the unlimited setting is the worst of both worlds for a problem like this: you pay roughly the `4096` cost but lose the predictability, because "until done" is a distribution, not a number.

Now flip the difficulty. Hand the same three budgets a genuine AIME-tier combinatorics problem — the kind where the first approach usually fails and the model must backtrack — and the table inverts: `512` is now *too small*, the model exhausts its budget mid-derivation, emits "I have exhausted my token budget, and now I will start answering," and commits to a half-baked answer that is often wrong; `4096` gives it room to fail-and-retry and lands the correct answer; unlimited may add a few more points but at unbounded cost. **The optimal budget moved from `512` to `4096` purely because the problem got harder, and there is no single global budget that is right for both prompts.** That is precisely why the production answer is a difficulty router (§7), not a constant. The worked example also exposes the mis-sizing failure mode directly: if you had set `max_new_tokens=512` alongside `thinking_budget=512` on the hard problem, the answer — which comes *after* the think block — would have been truncated to nothing, which is the empty-answer bug we will see bite a real team in the case studies.

## 3. The thinking-budget ladder: latency and accuracy across levels

**The senior rule of thumb: accuracy gains from more thinking are task-dependent and saturate; spend your token budget where the curve is still steep, starve it where the curve is flat.**

<!-- FIGSPEC 3
kind: grid
claim: Across the trained budget levels 0 to 16K, hard tasks like AIME and LiveCodeBench keep gaining accuracy while easy tasks like IFEval only fluctuate, so the optimal budget depends on task class.
caption: The budget ladder pays off on hard tasks and is wasted or harmful on easy ones.
nodes:
  - id: r0h | label: "0: no reason, low" | color: gray
  - id: r0e | label: "0: often fine" | color: green
  - id: r1h | label: "512: rising" | color: amber
  - id: r1e | label: "512: near-best" | color: green
  - id: r2h | label: "2K: climbing" | color: blue
  - id: r2e | label: "2K: jitter, no gain" | color: amber
  - id: r3h | label: "8K: high" | color: blue
  - id: r3e | label: "8K: jitter / overthink" | color: amber
  - id: r4h | label: "16K: peak, saturating" | color: green
  - id: r4e | label: "16K: can hurt" | color: amber
edges:
notes: vertical grid, 2 cols (hard / easy) x 5 rows (budget 0,512,2K,8K,16K); no edges
-->
![Tree of thinking-budget levels from 0 to 16K, branching by task type, showing where accuracy keeps climbing versus where it plateaus or fluctuates](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-3.png)

ByteDance's own documentation is refreshingly honest about the shape of the budget-vs-accuracy curve, and it is *not* monotone for every task. The repository states the effect plainly: "For simpler tasks (such as IFEval), the model's chain of thought is shorter, and the score exhibits fluctuations as the thinking budget increases. For more challenging tasks (such as AIME and LiveCodeBench), the CoT is longer, and the score improves with an increase in the thinking budget."

Unpack that, because it is the operating manual:

- **On easy tasks, more budget can *hurt* or just jitter.** If the natural reasoning length for "follow these formatting instructions" is 200 tokens, forcing the model to fill 8192 tokens of think-space invites it to second-guess a correct first instinct, hallucinate constraints, or talk itself out of the right answer. This is the over-thinking pathology, and it is measurable: IFEval (instruction following) scores fluctuate rather than climb with budget. The right budget for an easy task is *small*, sometimes zero.
- **On hard tasks, the curve is genuinely steep and keeps climbing toward the documented levels.** Competition math (AIME) and hard coding (LiveCodeBench) reward exploration: trying an approach, noticing it fails, backtracking. Each of those is reasoning tokens well spent, and the score rises as you grant more budget. This is where you spend 8K–16K.

I do not have a per-level numerical table from the primary sources — ByteDance shows a curve, not a table of (budget, accuracy) points — so I will not fabricate one. What I *can* give you is the structural rule the curve implies: **the optimal budget is a function of task class, and the function is non-monotone for easy tasks and rising-then-saturating for hard ones.** The engineering job is to learn that mapping for *your* traffic and bake it into your router, not to pick one global budget and pray.

A concrete way to find your operating point, since you should measure rather than guess:

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("ByteDance-Seed/Seed-OSS-36B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
)

def run(prompt, budget):
    msgs = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", thinking_budget=budget,
    ).to(model.device)
    t0 = time.time()
    out = model.generate(inp, max_new_tokens=budget + 1024,
                         temperature=1.1, top_p=0.95)
    dt = time.time() - t0
    gen = out[0][len(inp[0]):]
    return tok.decode(gen, skip_special_tokens=True), len(gen), dt

# Sweep the trained levels on YOUR representative hard query.
prompt = "Prove that the sum of the first n odd numbers is n^2, then compute it for n=47."
for b in [0, 512, 1024, 2048, 4096, 8192, 16384]:
    text, n_tokens, secs = run(prompt, b)
    print(f"budget={b:>5} | gen_tokens={n_tokens:>5} | wall={secs:5.1f}s")
    # Score `text` against your gold answer here; plot accuracy vs budget.
```

Run that sweep on a held-out set of *your* queries, bucketed by your difficulty classifier, and you will get the (budget → accuracy, latency) curves that actually matter for your application. The headline benchmark numbers tell you the model is capable; this sweep tells you where to set the dial. Note the `gen_tokens` column will track the budget closely on hard prompts and undershoot it on easy ones — that undershoot is the model declining to over-think, and it is exactly the behavior you are paying for.

## 4. The 512K context: how it is engineered, not bolted on

**The senior rule of thumb: "native" long context means trained on long sequences with a position scheme designed for the range; "extended" long context means inference-time extrapolation that degrades, and you must know which one you bought.**

<!-- FIGSPEC 4
kind: matrix
claim: At RULER 128K, Seed-OSS native 512K context scores 94.6 versus Qwen3-32B at 77.5 and gpt-oss-20B at 78.7, because GQA 80:8 heads and RoPE base 1e7 make 512K trainable.
caption: Native long context, not inference-time extrapolation, is what holds RULER quality at 128K.
notes: matrix, rows = models, columns = (RULER@128K, native ctx, KV per 512K seq). Color by RULER tier.
nodes:
  - id: m1 | label: "Seed-OSS: 94.6 / 512K" | color: green
  - id: m2 | label: "Qwen3-30B-A3B: 94.5 / —" | color: blue
  - id: m3 | label: "gpt-oss-20B: 78.7" | color: amber
  - id: m4 | label: "Qwen3-32B: 77.5 / 33K" | color: amber
  - id: c1 | label: "GQA 80:8 = 10x KV cut" | color: gray
  - id: c2 | label: "RoPE base 1e7" | color: gray
  - id: c3 | label: "KV ~137GB @512K bf16" | color: gray
edges:
notes: vertical matrix; top block = 4 model rows ranked by RULER@128K; bottom block = 3 enabler context cells
-->
![Matrix of long-context capability: context length on one axis, RULER score and KV-cache cost on the other, contrasting Seed-OSS native 512K against models that extend to long context](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-4.png)

The single most cited number for Seed-OSS's long context is **RULER@128K = 94.6**. RULER is the long-context benchmark that matters because it goes beyond needle-in-a-haystack retrieval (which large-context models can fake) into multi-needle, variable-tracking, and aggregation tasks that require actually *using* dispersed context, not just locating one fact. A score of 94.6 at 128K puts Seed-OSS at the top of its weight class — for comparison, Qwen3-32B scores 77.5 and gpt-oss-20b scores 78.7 on the same RULER@128K setting per ByteDance's evaluation. That is not a marginal gap; it is a 17-point chasm that tells you Seed-OSS's long context is real working memory, not decorative.

Two engineering decisions make 512K native context tractable, and both are in the config we already read:

1. **GQA with an 80:8 head ratio.** The KV cache is what kills you at long context. Memory for the cache scales as `2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes`. Plug in Seed-OSS's numbers for a single 512K-token sequence in bf16: `2 × 64 × 8 × 128 × 524288 × 2 bytes ≈ 137 GB`. With 8 KV heads, one full-context sequence's cache is already ~137 GB — it spans more than one H100's 80 GB, which is *why* the recommended deployment is `--tensor-parallel-size 8`. Now imagine that with 80 KV heads (full multi-head): you would multiply by 10 and land at ~1.4 TB for a single sequence. GQA is not an optimization here; it is the precondition for 512K existing at all.

2. **RoPE base `1e7`, trained natively.** As discussed, the high RoPE base spreads the rotary frequencies so positions out to half a million stay distinguishable, and the model saw long sequences in pretraining so it learned to *use* those distant positions. This is the qualitative difference from a model trained at 32K and stretched to 128K with YaRN at inference: the stretched model's attention to far-away tokens is an extrapolation it never practiced, and its multi-needle RULER score craters as a result.

### The second-order cost you must budget for

Native 512K does not mean *free* 512K. Three costs land on your infra team:

- **Memory.** As computed, the KV cache at full context is enormous. You will run tensor-parallel across 8 GPUs not because the *weights* need it (36B in bf16 is ~72 GB, fits on one H100 with room) but because the *cache* at long context does. Plan capacity around the cache, not the weights.
- **Prefill latency.** Attention is quadratic in sequence length for the prefill (prompt-processing) phase. A 512K prompt's prefill is genuinely slow — you are computing attention over a half-million-token sequence before the first output token appears. Time-to-first-token at full context is measured in seconds-to-tens-of-seconds, not milliseconds. If your UX needs snappy first tokens, 512K is not your happy path.
- **Quality is not uniform across the window.** RULER@128K is 94.6; the model card does not publish RULER@512K, and I will not invent it. Expect some degradation as you push past the lengths where evaluation is reported. Treat 128K as the "validated" zone and 128K–512K as "supported but verify on your data."

Here is a long-context inference snippet that respects these realities — note the explicit attention-implementation and the fact that you want a serving engine, not raw `transformers`, for production long context:

```python
# Production long-context serving: use vLLM (>= 0.10.2), not raw transformers.
# Tensor-parallel across 8 GPUs to fit the 512K KV cache.
#
# Launch the server (shell):
#   python -m vllm.entrypoints.openai.api_server \
#       --model ByteDance-Seed/Seed-OSS-36B-Instruct \
#       --dtype bfloat16 \
#       --tensor-parallel-size 8 \
#       --max-model-len 524288 \
#       --chat-template ./Seed-OSS-36B-Instruct/chat_template.jinja
#
# Then drive it through the OpenAI-compatible client:
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

with open("merger_agreement_300k_tokens.txt") as f:
    document = f.read()

resp = client.chat.completions.create(
    model="ByteDance-Seed/Seed-OSS-36B-Instruct",
    messages=[
        {"role": "system", "content": "thinking_budget: 4096"},  # set via template
        {"role": "user", "content":
            f"{document}\n\nList every clause that changes if the deal closes "
            f"after 2026-12-31, with section numbers. Cross-reference defined "
            f"terms across the whole document."},
    ],
    max_tokens=8192,
    temperature=1.1,
    top_p=0.95,
)
print(resp.choices[0].message.content)
```

That query — "find and cross-reference clauses dispersed across a 300K-token contract" — is exactly the RULER-style aggregation task where native long context earns its keep and a stretched-context model falls apart.

### How 512K is actually served: a worked KV-cache budget

Let me slow down on the memory arithmetic, because "512K context" is a claim about a *number in a config* and "we can serve 512K context" is a claim about *gigabytes on a GPU*, and the gap between them is where deployments die. The KV cache is the dominant cost at long context, and its size is fully determined by the architecture we already read. The formula:

$$
\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times b
$$

where the leading $2$ counts both the K and the V tensors, $L$ is layers, $H_{kv}$ is the number of *key/value* heads (not query heads — this is the whole point of GQA), $d_{head}$ is the head dimension, $S$ is the sequence length, $B$ is the batch (concurrent sequences), and $b$ is bytes per element. For Seed-OSS in bf16, $L=64$, $H_{kv}=8$, $d_{head}=128$, $b=2$. Substitute and you get the per-token-per-sequence cost first, which is the number worth memorizing:

$$
2 \times 64 \times 8 \times 128 \times 2 = 262{,}144 \text{ bytes} \approx 256 \text{ KB per token}
$$

So **every token of context costs ~256 KB of KV cache.** That is the constant that governs your capacity planning. Multiply by sequence length:

- At $S = 128{,}000$ (the RULER-validated length): $128{,}000 \times 256\text{ KB} \approx 33.5\text{ GB}$ for one sequence. That already does not fit alongside the 72 GB of weights on a single 80 GB H100 with any batch — you are tensor-parallel before you even reach the validated context length.
- At $S = 524{,}288$ (full 512K): $524{,}288 \times 256\text{ KB} \approx 137\text{ GB}$ for one sequence. This is the number from the matrix figure. A single full-context request's cache is larger than any single GPU's memory, which is the concrete, mechanical reason the model card prescribes `--tensor-parallel-size 8`: you shard the 137 GB cache (plus the 72 GB of weights) across eight 80 GB GPUs, leaving headroom for activations and a modest batch.

Now run the counterfactual that justifies the GQA choice. If Seed-OSS used full multi-head attention — $H_{kv} = 80$ instead of $8$ — the per-token cost would be $2.56\text{ MB}$, and a single 512K sequence's cache would be **~1.37 TB**. That is not "expensive," it is *physically unservable* on any reasonable GPU cluster for a single request. The 80:8 GQA ratio is therefore not a tuning knob; it is the enabling decision without which the 512K number on the model card could not exist as a servable feature. This is the same arithmetic I lay out in the [KV cache deep-dive](/blog/machine-learning/large-language-model/kv-cache), and it is the single most important thing to internalize before you provision hardware for this model: **size your cluster around the cache at your *expected* context length, not around the weights.**

Three second-order consequences follow directly from the 256 KB-per-token constant:

1. **Batch size and context length trade off linearly.** With ~137 GB of usable cache budget across your TP group after weights, you can serve roughly one 512K sequence, *or* ~four 128K sequences, *or* ~thirty-two 16K sequences concurrently — the product $S \times B$ is what your cache budget caps. A serving engine like vLLM with paged attention manages this for you, but the ceiling is the arithmetic above. If your traffic is mostly short prompts, you get high concurrency; if it is mostly 512K documents, you are effectively single-stream per TP group and your throughput collapses. Plan your fleet around the *distribution* of your context lengths, not the max.
2. **Prefill compute scales quadratically; cache memory scales linearly — they are different walls.** The 256 KB/token figure is the *memory* wall. Separately, the attention computation in prefill is $O(S^2)$ in the prompt length, so a 512K prefill does on the order of $(512\text{K})^2$ attention operations before the first output token. That is the *latency* wall, and quantizing weights or adding GPUs for memory does not move it — it is intrinsic to dense attention over a long prompt. You can amortize it with prefix caching (reuse the KV for a document you query repeatedly), but a cold 512K prefill is seconds-to-tens-of-seconds, full stop.
3. **KV-cache quantization is the lever that actually helps at long context.** Quantizing the *weights* to 4-bit (the model card's `--load_in_4bit`) shrinks the 72 GB of weights to ~18–20 GB, which is lovely for fitting the model on a small card — but at 512K the cache, not the weights, is your 137 GB problem, and weight quantization does nothing for it. The lever that matters at long context is *KV-cache* quantization: storing K and V in fp8 instead of bf16 halves the 256 KB/token to 128 KB/token, turning the 512K cache from 137 GB into ~68 GB and roughly doubling your achievable concurrency or context. If you are serious about serving long context on this model, KV-cache quantization is the optimization with leverage; weight quantization is the optimization with marketing.

## 5. The training recipe: 12T tokens and the data-efficiency story

**The senior rule of thumb: token count is an input, not an achievement; the number worth respecting is capability-per-token, and Seed-OSS's is high.**

<!-- FIGSPEC 5
kind: layered-stack
claim: The 12T-token pretrain forks into a with-synthetic and a without-synthetic base, and the with-synthetic line is post-trained into the budget-aware Instruct model.
caption: A 12T-token pretrain produces two bases and one deployable Instruct model.
notes: vertical layered stack, bottom = foundation to top = deployable.
nodes:
  - id: s1 | label: "12T-token pretrain corpus" | color: blue
  - id: s2 | label: "Base w/ synthetic data" | color: blue
  - id: s3 | label: "Base-woSyn (clean)" | color: gray
  - id: s4 | label: "Post-train: reasoning + budget" | color: green
  - id: s5 | label: "Seed-OSS-36B-Instruct" | color: green
edges:
  - s1 -> s2
  - s1 -> s3
  - s2 -> s4
  - s4 -> s5
notes: layered stack stacked vertically; s1 forks to s2 and s3; only s2 continues up through s4 to s5
-->
![Layered stack of the Seed-OSS training pipeline from 12T-token pretraining through the with-synthetic and without-synthetic base variants up to the instruct model](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-5.png)

Here is the line in the model card that gets the least attention and deserves the most: Seed-OSS-36B was trained on **~12 trillion tokens**. For context, contemporary open models in the same capability tier were trained on substantially more — many 30B-class models report 15T, some push toward 30T+. Seed-OSS reaching open-source-SOTA on MMLU-Pro (82.7) and LiveCodeBench v6 (67.4) on 12T tokens is a data-efficiency result, and it tells you the quality of the data and the curriculum mattered more than raw quantity.

I cannot publish ByteDance's exact data mixture — it is not disclosed at the granularity the [OLMo 3 open-training report](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques) gives you. What *is* disclosed and worth dwelling on is a deliberate, unusual choice: ByteDance released **two base models**.

| Variant | What it is | Why it exists |
| --- | --- | --- |
| `Seed-OSS-36B-Base` | Base model including synthetic instruction data in pretraining | Stronger out-of-the-box base metrics; what most users want |
| `Seed-OSS-36B-Base-woSyn` | Base model *without* synthetic instruction data ("woSyn") | Cleaner substrate for post-training research; no synthetic contamination |
| `Seed-OSS-36B-Instruct` | Post-trained, reasoning + thinking-budget capable | The model you deploy |

This dual release is a quietly principled move. Mixing synthetic instruction data into pretraining boosts your base benchmarks — and the numbers bear it out, the with-synthetic base beats the without-synthetic base on most metrics — but it also *contaminates* the base for researchers who want to study post-training cleanly, because the base already "knows" instruction-following patterns that should come from your fine-tuning, not the pretrain. By shipping both, ByteDance lets product teams take the stronger `Base` and lets researchers take the cleaner `Base-woSyn`. The model card even flags this explicitly as the rationale: synthetic instruction data in pretraining "may affect post-training research." That is the kind of acknowledged-tradeoff honesty I want from a model card.

The base-model numbers make the synthetic-data effect concrete:

| Benchmark | Base (w/ synthetic) | Base (w/o synthetic) | Qwen2.5-32B-Base |
| --- | --- | --- | --- |
| MMLU-Pro | **65.1** | 60.4 | 58.5 |
| MMLU | 84.9 | 84.8 | 84.0 |
| TriviaQA | 82.1 | 81.9 | 76.0 |
| GSM8K | **90.8** | 90.3 | 87.5 |
| MATH | **81.7** | 61.3 | 63.5 |
| BBH | **87.7** | 87.2 | 79.1 |
| HumanEval | **76.8** | 75.6 | 47.6 |
| MBPP | **80.6** | 74.6 | 77.8 |

Look at the MATH row: 81.7 with synthetic data versus 61.3 without — a 20-point swing. That is the synthetic instruction/reasoning data doing heavy lifting on math specifically. And HumanEval at 76.8 versus Qwen2.5-32B's 47.6 is a 29-point coding gap over a same-size base. The synthetic data is not free lunch — it is the contamination researchers worry about — but it is *effective* lunch, and ByteDance gives you the choice of whether to eat it.

### The data-efficiency angle, before and after

<!-- FIGSPEC 6
kind: before-after
claim: Models in this tier often train on 15T-30T+ tokens, while Seed-OSS reaches open-SOTA MMLU-Pro 82.7 on only 12T, trading long-tail trivia for reasoning and fine-tuning pliability.
caption: Fewer, denser tokens buys reasoning and adaptability at the cost of factual recall.
nodes:
  - id: b1 | label: "Brute-force: 15T-30T+ tokens" | color: amber
  - id: b2 | label: "over-trained, stiff priors" | color: amber
  - id: b3 | label: "strong trivia recall" | color: gray
  - id: a1 | label: "Seed-OSS: 12T tokens" | color: blue
  - id: a2 | label: "MMLU-Pro 82.7 open-SOTA" | color: green
  - id: a3 | label: "pliable to fine-tuning" | color: green
  - id: a4 | label: "SimpleQA 9.7 (trivia cost)" | color: amber
edges:
  - b1 -> b2
  - b2 -> b3
  - a1 -> a2
  - a2 -> a3
  - a3 -> a4
notes: two vertical columns; left "Brute-force corpus" (amber/gray), right "Data-efficient 12T" (blue/green with one amber cost cell)
-->
![Before and after: a larger training corpus producing a given capability level versus Seed-OSS reaching comparable capability with a 12T-token corpus](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-6.png)

The reason 12T matters to *you* is not bragging rights; it is a signal about the model's nature. A model that reached this capability on fewer tokens was trained on denser, more curated, more reasoning-rich data. Such models tend to generalize better and hallucinate less on out-of-distribution prompts than models that got there by brute-forcing volume, because the curriculum did the work the parameter count would otherwise have to. It also means the model is *not* saturated — there is headroom for your fine-tuning to teach it your domain without fighting an over-trained, over-confident base. When I fine-tune on top of a 12T model versus a 30T model of the same size, the 12T model is usually more pliable; the over-trained one has stronger priors that resist your adaptation. The SimpleQA score (9.7 for Instruct) is the flip side: a smaller, denser pretrain knows fewer obscure facts than a model that swallowed the whole web, so Seed-OSS is weaker on long-tail factual recall. Data efficiency buys reasoning and pliability; it costs you trivia.

## 6. The benchmark numbers, with their warts

**The senior rule of thumb: a benchmark table is a marketing artifact until you read the columns the vendor chose and the ones they omitted.**

Here is the full Instruct comparison table as published by ByteDance, with the comparison models *they* selected. I am reproducing it faithfully, including the cases where Seed-OSS does *not* win, because the honest ones are the useful ones.

| Benchmark | Seed-OSS-36B-Instruct | Qwen3-32B | Qwen3-30B-A3B | gpt-oss-20B | Gemma3-27B |
| --- | --- | --- | --- | --- | --- |
| MMLU-Pro | **82.7** | 81.8 | 81.9 | 76.2 | 67.5 |
| MMLU | **87.4** | 86.2 | 86.9 | 81.7 | 76.9 |
| GPQA-D | 71.4 | 66.7 | 71.4 | **72.2** | 42.4 |
| SuperGPQA | 55.7 | 49.3 | **57.3** | 50.1 | — |
| SimpleQA | 9.7 | 8.6 | **23.6** | 6.7 | 10.0 |
| AIME24 | 91.7 | 82.7 | 87.7 | **92.7** | — |
| AIME25 | 84.7 | 73.3 | 81.3 | **90.3** | — |
| BeyondAIME | 65 | 29 | 56 | **69** | — |
| LiveCodeBench v6 | **67.4** | 53.4 | 60.3 | 63.8 | — |
| IFEval | 85.8 | 88.4 | 88.0 | **92.8** | 90.4 |
| TAU1-Retail | **70.4** | 40.9 | 58.7 | — | — |
| TAU1-Airline | 46 | 38 | **47** | — | — |
| SWE-Bench Verified (OpenHands) | 56 | 23.4 | 31 | — | — |
| RULER (128K) | **94.6** | 77.5 | 94.5 | 78.7 | — |

What this table tells me, reading it like an engineer and not a press release:

- **Knowledge and reasoning breadth: Seed-OSS leads.** MMLU-Pro 82.7 and MMLU 87.4 are open-source-SOTA at this size. This is the strongest claim and it holds up.
- **Math: strong but not the leader.** AIME24 91.7 and AIME25 84.7 are excellent, but gpt-oss-20B edges it (92.7 / 90.3). If your workload is pure competition math, gpt-oss is competitive and cheaper to run. Seed-OSS's BeyondAIME 65 (vs Qwen3-32B's 29) shows it generalizes math reasoning better than Qwen, though.
- **Coding: Seed-OSS leads on the realistic benchmarks.** LiveCodeBench v6 67.4 is open-SOTA, and SWE-Bench Verified 56 (under OpenHands) is a *large* margin over Qwen3-32B's 23.4. SWE-Bench is real-repository bug-fixing — the closest thing to "can this model do my job" — and Seed-OSS is genuinely strong there.
- **Agentic / tool use: Seed-OSS leads on TAU1-Retail (70.4) but ties Qwen on Airline (46 vs 47).** The retail gap is enormous. This, combined with SWE-Bench, is why I would reach for Seed-OSS for agent workloads specifically.
- **Where it loses — and this is the honest part — instruction-following and factual recall.** IFEval 85.8 trails Qwen3-32B (88.4) and gpt-oss-20B (92.8). SimpleQA 9.7 is *crushed* by Qwen3-30B-A3B's 23.6. The IFEval gap connects directly to the over-thinking pathology from §3: a reasoning-heavy model can talk itself out of literal instruction-following. The SimpleQA gap is the 12T-token data-efficiency tradeoff from §5. Both are predictable from the model's design, and both matter if your workload is "follow this format exactly" or "recall obscure facts."

The matrix figure above (figure 4 in §4 covers context; the §6 reading is this table) is the one I would put in front of a tech lead deciding between these three models. There is no universal winner. There is "Seed-OSS for reasoning/coding/agents/long-context, gpt-oss for raw math and instruction-following, Qwen3-30B-A3B for throughput and factual recall."

## 7. Inference flow with budget monitoring: the control loop

**The senior rule of thumb: the budget is a contract the model honors, not a fence the engine enforces; build your control loop to trust-but-verify.**

<!-- FIGSPEC 7
kind: graph
claim: A difficulty classifier maps each query to a budget, the model self-monitors and branches into natural-finish or budget-exhaustion, and both merge into the answer phase that logs used tokens for the router feedback loop.
caption: The production control loop routes by difficulty and verifies budget adherence by logging used tokens.
nodes:
  - id: q | label: "incoming query" | color: gray
  - id: cls | label: "difficulty classifier" | color: blue
  - id: bud | label: "assign budget 0-16K" | color: blue
  - id: mon | label: "reason + self-monitor" | color: blue
  - id: fin | label: "natural finish" | color: green
  - id: exh | label: "budget exhausted" | color: amber
  - id: ans | label: "answer + log used tokens" | color: green
  - id: fb | label: "refit difficulty map" | color: gray
edges:
  - q -> cls
  - cls -> bud
  - bud -> mon
  - mon -> fin | label: "in budget"
  - mon -> exh | label: "over budget"
  - fin -> ans
  - exh -> ans
  - ans -> fb | label: "feedback"
  - fb -> cls
notes: directed graph, top-to-bottom main flow with a branch at mon (fin/exh) merging at ans; feedback edge from ans/fb back up to cls forms the loop
-->
![Branch-and-merge graph of the inference control flow: a query is classified by difficulty, routed to a budget, the model self-monitors and either lands an answer or hits the budget exhaustion path, then merges to the answer phase](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-7.png)

Let me make the runtime behavior concrete, because the interaction between *your* budget assignment and the model's *self-monitoring* is where production systems either save money or quietly break. The flow:

1. **Classify the query's difficulty.** Cheaply — a small classifier, a rule, or a router model. Map difficulty class → budget. Trivial → 0. Median-hard → 2K. Hardest tail → 16K.
2. **Inject the budget via the chat template.** This writes the budget into the prompt's control region.
3. **The model reasons, emitting `<seed:cot_budget_reflect>` checkpoints.** It estimates consumed/remaining and paces itself.
4. **One of two branches:**
   - *Natural completion:* the model finishes reasoning before exhausting the budget, emits its answer. This is the common, healthy case on well-sized budgets.
   - *Budget exhaustion:* the model hits "I have exhausted my token budget" and transitions to answering with whatever it has. On a too-small budget for a hard problem, the answer quality drops — this is the signal that your difficulty classifier under-budgeted.
5. **Both branches merge into the answer phase**, after which `max_new_tokens` (your hard ceiling) governs answer length.

The trust-but-verify part: because the budget is a soft contract, you should still set `max_new_tokens` as a real hard stop, and you should *log the generated-token count per request* to catch overshoot. In practice the model stays close to its budget on the trained levels, but a robust system measures rather than assumes. Here is a control-loop sketch:

```python
def answer_with_routing(query, difficulty_fn, client):
    # 1. Difficulty -> budget mapping (tune on your traffic).
    budget = {
        "trivial": 0,
        "easy":    512,
        "medium":  2048,
        "hard":    8192,
        "extreme": 16384,
    }[difficulty_fn(query)]

    resp = client.chat.completions.create(
        model="ByteDance-Seed/Seed-OSS-36B-Instruct",
        messages=[
            {"role": "system", "content": f"thinking_budget: {budget}"},
            {"role": "user", "content": query},
        ],
        # Hard ceiling = budget + headroom for the answer itself.
        max_tokens=budget + 2048,
        temperature=1.1, top_p=0.95,
    )

    out = resp.choices[0].message.content
    used = resp.usage.completion_tokens

    # 3. Trust-but-verify: flag overshoot / starvation for your router's feedback loop.
    if used > budget + 2048 * 0.9:
        log_warn(f"near-ceiling: budget={budget} used={used} — bump difficulty class?")
    return out
```

The feedback loop is the mature version of this: log `(difficulty_class, budget, used_tokens, answer_correct)`, and periodically re-fit your difficulty→budget map. Over a few weeks of traffic your router learns the *empirical* budget each query class needs, and your average cost converges to the minimum that preserves accuracy. That is the operational payoff the thinking budget unlocks and that an unbounded reasoning model simply cannot give you.

### The gotcha: budget interacts with the answer, not just the think phase

A non-obvious failure I have seen: teams set `thinking_budget=8192` and `max_new_tokens=8192`, reasoning "I want 8K of thinking." But the answer comes *after* the 8K of reasoning, so the model exhausts the entire ceiling on thinking and gets *zero tokens for the answer* — you receive an empty or truncated response. The budget governs the think phase; the ceiling must exceed `budget + answer_length`. Always set `max_new_tokens` comfortably above the budget. This bug looks like "the model stopped answering at high budgets" and it is purely an off-by-a-phase ceiling error.

## 8. The model family and license: what you can actually ship

**The senior rule of thumb: read the license before the benchmarks; a 90 on AIME under a non-commercial license is a 0 for your product.**

<!-- FIGSPEC 8
kind: tree
claim: The Seed-OSS-36B family forks the 12T pretrain into Base, Base-woSyn, and Instruct, all Apache-2.0, served via vLLM 0.10.2+ with 4/8-bit quantization.
caption: A small, clean family under a fully permissive license.
nodes:
  - id: root | label: "Seed-OSS-36B (12T, Apache-2.0)" | color: green
  - id: base | label: "Base (w/ synthetic)" | color: blue
  - id: wosyn | label: "Base-woSyn (research)" | color: gray
  - id: inst | label: "Instruct (deploy)" | color: blue
  - id: serve | label: "vLLM 0.10.2+ / TP=8" | color: gray
  - id: quant | label: "4-bit / 8-bit quant" | color: gray
edges:
  - root -> base
  - root -> wosyn
  - root -> inst
  - inst -> serve
  - inst -> quant
notes: vertical tree, root at top fans to 3 family members; Instruct further fans to serve+quant leaves
-->
![Tree of the Seed-OSS-36B family: the 12T-token pretrain branching into Base, Base-woSyn, and Instruct, all under Apache-2.0 with their quantization and serving options](/imgs/blogs/seed-oss-36b-open-long-context-thinking-budget-8.png)

The family is small and clean, which I appreciate:

| Member | Role | License | Notes |
| --- | --- | --- | --- |
| `Seed-OSS-36B-Base` | Pretrained base, w/ synthetic instruction data | Apache-2.0 | Stronger base metrics; default for product fine-tuning |
| `Seed-OSS-36B-Base-woSyn` | Pretrained base, no synthetic data | Apache-2.0 | Clean substrate for post-training research |
| `Seed-OSS-36B-Instruct` | Post-trained, reasoning + thinking budget | Apache-2.0 | The deployable model |

**Apache-2.0 is the headline for anyone shipping a product.** It is a true permissive license: commercial use, modification, redistribution, no copyleft, no non-compete clause, no "research-only" trap. You can fine-tune `Base` on your proprietary data, ship the result in a commercial product, and never send ByteDance a dollar or a derivative. That is rarer than it should be among strong open reasoning models, several of which carry custom licenses with usage caps or acceptable-use riders that your legal team will choke on. Seed-OSS does not.

Deployment surface, from the docs:

- **Serving:** vLLM ≥ 0.10.2 (with a Seed-OSS tool-choice parser for function calling) or `transformers` ≥ 4.56.1. For production long context, vLLM with `--tensor-parallel-size 8` is the documented path.
- **Quantization:** 4-bit and 8-bit are supported via `--load_in_4bit` / `--load_in_8bit`. At 4-bit the ~36B weights drop to roughly 18–20 GB, fitting comfortably on a single 24 GB consumer card *for short context* — but remember the KV cache, not the weights, dominates at long context, so quantizing weights does not rescue you from the 512K memory wall.
- **Precision:** bf16 is the reference; the vLLM launch uses `--dtype bfloat16`.

If you want to place Seed-OSS in the broader landscape — when to pick it over Qwen3, what ByteDance's other models do — the [Seed model map](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case) and the [Qwen3 technical report notes](/blog/paper-reading/large-language-model/qwen3-technical-report) are the companion reads.

## Case studies from production

These are scenarios — some drawn from deployments I have run or reviewed, some constructed to illustrate a specific failure or win mode. Names and numbers are illustrative where I could not cite a measured result, but every mechanism is real and follows from the model's documented behavior.

### 1. The over-provisioned support bot that halved its bill

A customer-support deployment ran a reasoning model with `max_tokens=8192` on every ticket, because a handful of complex billing-dispute tickets genuinely needed deep reasoning. The symptom: median latency 9 seconds, p99 over 40 seconds, GPU bill climbing with ticket volume even though most tickets were "where is my order." The wrong first hypothesis was "we need a bigger GPU fleet." The actual root cause was that 80% of tickets were trivial and the model was burning thousands of reasoning tokens on each. Swapping to Seed-OSS with a difficulty router — `thinking_budget=0` for FAQ-class tickets, 512 for light, 4096 reserved for the genuine disputes — dropped median reasoning tokens by roughly 7× and cut the GPU bill nearly in half, with no measured accuracy loss on the hard tickets. Put concrete arithmetic on it: at a flat 8192 reasoning tokens across 100K daily tickets you pay to generate ~819M reasoning tokens a day; route 80% of traffic to budget 0, 15% to 512, and 5% to 4096, and the expected reasoning tokens per ticket fall to `0.8·0 + 0.15·512 + 0.05·4096 ≈ 282`, or ~28M a day — roughly a 29× cut in reasoning tokens that flows almost linearly into GPU-seconds because decode is the dominant cost at this output length. The budget did not make the hard tickets worse; it stopped the easy ones from subsidizing a worst-case ceiling they never needed. The lesson: the cost of a reasoning fleet is dominated by the *budget you grant the median query*, and the median query is almost always easy — so the single highest-leverage knob in your whole serving stack is the one that lets the median query stop reasoning early.

### 2. The 300K-token contract analysis that a 32K model could not do

A legal-tech team needed to answer cross-referencing questions over merger agreements that ran 200K–350K tokens. Their incumbent 32K-context model required chunking the document and a retrieval layer, which broke on exactly the queries that mattered: "which clauses change if the closing date slips past year-end," where the relevant clauses were scattered across the document and referenced defined terms defined elsewhere. Retrieval kept missing the cross-references. Seed-OSS ingested the whole 300K-token document in one shot, and its native long context (RULER@128K = 94.6, the multi-needle/aggregation regime) handled the cross-referencing that retrieval fragmented. The lesson: for *aggregation* over long documents — not just single-fact lookup — native long context beats retrieval-over-short-context, and the RULER number is the right predictor of that capability.

### 3. The AIME pipeline where gpt-oss won

A quant research team building a math-heavy data pipeline benchmarked Seed-OSS against gpt-oss-20B on competition-math-style problems. Seed-OSS scored AIME24 91.7 / AIME25 84.7; gpt-oss-20B scored 92.7 / 90.3 — *and* it is a 20B that serves cheaper. For their narrow, pure-math workload, gpt-oss was the right call: comparable-to-better accuracy at lower cost. The lesson, and it is the most important one: do not pick Seed-OSS because it is the "newer ByteDance reasoning model." Pick the model whose *benchmark profile matches your workload*. Seed-OSS's edge is breadth (knowledge, coding, agents, long context), not pure-math peak.

### 4. The SWE-Bench agent that actually fixed bugs

An internal dev-tools team wired up an autonomous bug-fix agent with the OpenHands harness. Their previous model (Qwen3-32B) scored 23.4 on SWE-Bench Verified and, predictably, fixed about a quarter of the tickets it attempted, with a lot of confidently-wrong patches. Seed-OSS-36B scores 56 on the same SWE-Bench Verified (OpenHands) setting — more than double — and the practical effect was a real jump in merged auto-fixes. The combination of strong coding (LiveCodeBench 67.4) and strong tool use (TAU1-Retail 70.4) is what makes it a good agent substrate. The lesson: for agentic coding, the SWE-Bench + tool-use numbers predict production success far better than MMLU does.

### 5. The empty-answer bug from a mis-sized ceiling

A team reported "the model returns blank answers when we crank the thinking budget to 16K." Their config: `thinking_budget=16384`, `max_new_tokens=16384`. The model spent the entire ceiling on reasoning and had zero tokens left for the answer. The fix was one line: `max_new_tokens=16384 + 2048`. The lesson, restated because it bites everyone once: the budget governs the *think* phase and the answer comes after; your hard ceiling must exceed `budget + answer_length`, or you truncate the very answer you waited for.

### 6. The instruction-following regression on strict formatting

A team migrated a "rewrite this into our exact JSON schema, no extra fields" workload to Seed-OSS and saw a regression: the model occasionally added explanatory fields or commentary the schema forbade. Root cause: IFEval 85.8 trails Qwen3-32B (88.4) and gpt-oss-20B (92.8), and a reasoning-heavy model with a generous budget can *reason itself out* of strict literal compliance. Two fixes worked: drop the budget to 0 (no reasoning, no second-guessing) for the formatting step, and tighten the schema with a validator that rejects and re-prompts. The lesson: for strict-format / strict-instruction tasks, more reasoning is a liability; set budget low or zero, and do not assume the strongest reasoner is the best instruction-follower.

### 7. The factual-recall task where SimpleQA bit

A trivia-adjacent product asked the model fine-grained factual questions ("who was the third person to do X"). Seed-OSS's SimpleQA is 9.7 — far below Qwen3-30B-A3B's 23.6 — and the product hallucinated specifics. This is the direct cost of the 12T-token data-efficiency choice: a denser, smaller pretrain knows fewer long-tail facts. The fix was retrieval augmentation, feeding the model the facts rather than relying on parametric recall. The lesson: data-efficient models trade trivia for reasoning; if your workload is factual recall, either pick a model trained on more tokens or wrap it in retrieval — do not expect parametric knowledge it never absorbed.

### 8. The long-context prefill latency surprise

A team built a "chat with your 400K-token codebase" feature and was baffled that the first token took 20+ seconds. There was no bug. Attention prefill is quadratic in sequence length; processing a 400K-token prompt before emitting token one is genuinely expensive. The fixes: cache the prefill for repeated queries against the same document (KV-cache reuse), and set user expectations with a "reading your document" spinner. The lesson: native 512K context removes the *quality* wall but not the *prefill-latency* wall — budget seconds for time-to-first-token at extreme context, and design the UX around it.

### 9. The difficulty router that learned the budget map

A platform team instrumented the trust-but-verify loop from §7: log `(difficulty_class, budget, used_tokens, correct)` per request, refit weekly. After three weeks they discovered their "hard" class was over-budgeted at 8192 — empirically those queries landed correct answers within ~3000 tokens — so they cut it to 4096 and saved another chunk of cost with no accuracy loss, while bumping a mislabeled "medium" bucket that was starving. The lesson: the thinking budget is only as good as the difficulty→budget map you fit to your *own* traffic; treat it as a learned parameter, not a constant, and the feedback loop pays for itself.

### 10. The dense-model fine-tuning win

A team needed to specialize the model on a proprietary domain (semiconductor process docs). They had struggled to LoRA-fine-tune an MoE of similar active size because expert routing made adaptation finicky and the gains were uneven across experts. Seed-OSS's dense architecture had one activation path, the LoRA adaptation was clean and uniform, and the 12T (not over-trained) base was pliable to the new domain. The lesson: for teams that fine-tune, dense + not-over-trained is an ergonomic win that does not show up in any benchmark table but shows up in your iteration speed.

### 11. The budget=0 fast path for classification

A content-moderation pipeline used the model as a classifier (toxic / not-toxic, with a category). Running it as a reasoning model added latency for no benefit — classification does not need a chain-of-thought. Setting `thinking_budget=0` turned it into a direct-response classifier, cutting per-item latency dramatically while preserving accuracy on the classification labels. The lesson: `budget=0` is a real product feature, not a degenerate case — it lets one model serve both your reasoning routes and your "just answer" routes without deploying a second non-reasoning model.

### 12b. The multi-turn agent whose context outgrew its budget

A coding agent ran a long ReAct loop — read files, run tests, edit, repeat — and after twenty-odd turns the accumulated transcript pushed past 200K tokens. Two things broke at once, and the team conflated them. First, time-to-first-token per turn crept up, because each turn re-prefilled a growing prompt (the prefill-latency physics from case study 8, now paid every turn). Second, and more subtly, a fixed `thinking_budget=4096` that was right on turn 1 was wrong on turn 20: late-loop decisions ("given everything I have tried, what is the minimal next edit") are genuinely harder and were getting starved, so the agent started thrashing. The fix had two parts that mirror the two failures. For latency, they reused the KV cache across turns instead of re-prefilling from scratch — the history is identical prefix tokens, so the cache hit is nearly total and only the new turn's tokens prefill. For quality, they made the budget a function of loop depth, ramping from 1024 early to 8192 late, which let the model reason proportionally to how much state it had to integrate. The lesson is that in agentic loops the 512K context and the thinking budget are *coupled* controls: long context lets the history survive, but a static budget that ignores how far into the loop you are will either overthink the trivial early turns or starve the hard late ones. Treat budget as a per-turn decision, not a per-deployment constant, and cache the prefix that does not change.

### 12. The woSyn base that kept research clean

A research group studying RLHF dynamics needed a base model that had *not* seen instruction data in pretraining, so their post-training experiments measured the effect of *their* alignment, not residual pretrain contamination. `Seed-OSS-36B-Base-woSyn` was purpose-built for exactly this, and its slightly-lower-but-cleaner base metrics (MATH 61.3 vs the synthetic variant's 81.7) were the *point* — they wanted the substrate without the synthetic boost. The lesson: the dual-base release is not redundancy; the `woSyn` variant is a deliberate gift to researchers who need a contamination-free starting line, and it is rare to see a vendor ship it.

## When to reach for Seed-OSS-36B, and when not to

**Reach for Seed-OSS-36B when:**

- **You need controllable, predictable reasoning cost.** The thinking budget is the differentiator. If your traffic mixes trivial and hard queries and your reasoning bill is dominated by over-thinking the easy ones, route by difficulty and watch your average cost fall toward the median, not the ceiling.
- **You have genuine long-context aggregation needs.** Multi-document synthesis, cross-referencing across 100K–500K-token inputs, codebase-wide reasoning. The native 512K context and RULER@128K = 94.6 are real, not marketing, and they beat retrieval-over-short-context for *aggregation* tasks.
- **You are building agents that code and use tools.** SWE-Bench Verified 56 (more than double Qwen3-32B), LiveCodeBench v6 67.4, and TAU1-Retail 70.4 make it a strong agentic substrate.
- **You need a permissive license for a commercial product.** Apache-2.0 with no usage caps lets you fine-tune and ship freely.
- **You fine-tune and value ergonomics.** Dense architecture and a not-over-trained 12T base are pliable to domain adaptation.
- **You want clean post-training research.** The `Base-woSyn` variant gives you a synthetic-data-free substrate.

**Skip Seed-OSS-36B when:**

- **Your workload is pure competition-style math at peak accuracy.** gpt-oss-20B matches or edges it (AIME24/25) at lower serving cost. Match the model to the workload.
- **Your workload is strict instruction-following or exact-format output.** IFEval 85.8 trails Qwen3-32B and gpt-oss; a reasoning-heavy model can over-think literal compliance. If you do use it here, set `budget=0` and add a validator.
- **Your workload is long-tail factual recall without retrieval.** SimpleQA 9.7 reflects the 12T data-efficiency tradeoff; you will hallucinate specifics. Either pick a model trained on more tokens or add retrieval.
- **You need maximum throughput-per-dollar at high batch and short context.** A well-served MoE like Qwen3-30B-A3B (3B active) will out-throughput a dense 36B; if you do not need the budget control or the long context, you are paying dense FLOPs for features you are not using.
- **Your UX demands sub-second time-to-first-token at extreme context.** The quadratic prefill at 400K+ tokens is seconds-long; no model escapes that physics, and Seed-OSS is no exception.

The one-sentence verdict I would give a tech lead: **Seed-OSS-36B is the open-weights model to reach for when you want long context and reasoning that you can put a budget on — and the budget is the feature, the 512K context is the enabler, and the 12T tokens are the reason it punches above its data weight.** It is not the math champion and it is not the instruction-following champion, and the model card is honest enough to let you see that. Read the columns you lose before you commit, set your `max_new_tokens` above your budget, and instrument the difficulty router — do those three things and this is one of the most operationally pleasant open reasoning models you can deploy.
