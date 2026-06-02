---
title: "Speculative Decoding: A Mental Model, Case Studies, and Production Best Practices"
date: "2026-05-08"
publishDate: "2026-05-08"
description: "A senior-engineer's mental model for speculative decoding — why it works, when it pays for itself, when it burns tokens, and how the major implementations behave under real production load."
tags:
  [
    "llm",
    "speculative-decoding",
    "inference",
    "optimization",
    "deep-learning",
    "transformer",
    "draft-model",
    "serving",
    "vllm",
    "sglang",
    "eagle",
    "medusa",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 51
---

If you have ever stared at `nvidia-smi` while a 70B model decoded one token at a time and wondered why an $40,000 GPU was sitting at 8% utilization, this article is for you. That idle silicon is not a bug, and it is not your code. It is the load-bearing assumption underneath every speculative decoding system shipped in the last three years: **single-token decoding does not stress the compute units of a modern GPU; it stresses the wires that carry weights from HBM to the SMs**. Speculative decoding is what you do once you decide to stop wasting those FLOPs.

This is not an introduction to the technique. It is the mental model I wish I had been given on day one of trying to make speculative decoding pay for itself in a production serving stack — together with the math you actually need, the eight production incidents I learned the hard way, and a flowchart for when to reach for it and when to leave it alone.

## Why your single-GPU LLM is bored, not busy

Before you can reason about any decoding-side optimization, you have to internalize one number: the **arithmetic intensity** of a single decoding step. For a transformer with weights of size $W$ bytes generating one token, you read roughly $W$ bytes of weights from HBM and perform roughly $2W$ FLOPs of matmul work. The arithmetic intensity is therefore $2W / W = 2$ FLOP/byte. That number does not change with the size of the model.

Now compare that to the GPU's intrinsic ratio. An H100 SXM has 989 TFLOP/s of FP16 compute and 3.35 TB/s of HBM3 bandwidth. Its **balance point** — the FLOP/byte ratio at which compute and bandwidth are simultaneously saturated — is $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOP/byte. For prefill on a 4096-token prompt the effective intensity climbs into the low hundreds and you sit comfortably near the FLOP roof. For single-token decode you sit at intensity 2. You are using less than 1% of the hardware's compute throughput, gated entirely on how fast the chip can drag weights across the bus.

| Phase | Tokens / step | Arithmetic intensity | Bottleneck | Typical SM utilization |
|---|---|---|---|---|
| Prefill (4k prompt) | 4096 | ~120 FLOP/B | FLOP roof | 80–95% |
| Decode (no spec) | 1 | ~2 FLOP/B | HBM bandwidth | 5–15% |
| Decode (γ=4 spec, batch 1) | up to 5 verified per pass | ~10 FLOP/B | HBM bandwidth | 15–35% |
| Decode (batch 64) | 64 | ~120 FLOP/B | FLOP roof | 70–85% |

![Why decoding is bored, not busy](/imgs/blogs/speculative-decoding-2.png)

The figure above makes the assumption-vs-reality concrete. Prefill is the bullet-train: a few hundred FLOPs per byte means each weight matrix you read does a useful amount of arithmetic. Decode is the elevator: you pay the full cost of moving every weight through the chip, then do almost nothing with it. The whole game of inference optimization at small batch is to **amortize the bandwidth cost across more useful work per weight read**. KV-cache paging packs more sequences into HBM. Quantization shrinks $W$ so the read is cheaper. Speculative decoding takes the third path: keep the weights the same, but get *more verified tokens out of each forward pass*.

> If your decoding loop is bandwidth-bound, every optimization that does not reduce reads from HBM or increase tokens per read is theater.

This is why the very same speculative decoding system that produces a 2.7× speedup on a developer laptop running batch-1 decode regresses to a slowdown on a production endpoint serving batch-64. The hardware is not the same workload at all.

## The mental model: gambling with free tokens

![The speculative decoding bargain](/imgs/blogs/speculative-decoding-1.png)

The diagram above is the mental model: **a small drafter proposes K tokens, the target model verifies all K of them in a single forward pass, and the system commits the longest accepted prefix**. The rest of this article is a tour of that diagram. Three pieces deserve naming up front:

1. **The drafter** — any cheap distribution $q(x \mid x_{<i})$ over next tokens. It can be an entire smaller LLM (Llama-3.2-1B drafting for Llama-3.1-70B), a few extra heads on the target (Medusa, EAGLE), an n-gram lookup over the prompt, or a retrieval over a corpus. The only requirement is that you can sample from it cheaply — much cheaper than running the target.
2. **The verification step** — one forward pass through the target model over the *entire drafted suffix*. Because the target is autoregressive, it produces a logit vector at every position simultaneously. With one extra all-batch pass you get $K$ candidate logits for the price of one decoding step.
3. **Modified rejection sampling** — the per-position decision rule that decides which drafted tokens survive. It is constructed so that the surviving sequence is distributed exactly as the target would have sampled it, end to end. No quality regression. No "approximate" decoding.

This third point is the part that surprises people. Speculative decoding does not change what your model says; it only changes how fast it says it. If your unit tests pass at temperature 0 with greedy decoding and target = drafter, they will pass with speculative decoding too — bit-identical when α equals 1, distributionally identical otherwise. (We will see in §4 exactly why.)

The bargain is straightforward. You pay for $K$ extra drafter forward passes. You save by collapsing up to $K+1$ target forward passes into one. If the drafter agrees with the target most of the time and is significantly cheaper, the trade is profitable. If either condition fails, you get a slowdown.

> Speculative decoding is a gamble paid for in drafter compute. The house edge is the gap between drafter cost and target cost; the win rate is the acceptance rate α.

## The math you actually need

Let $\alpha$ be the per-token **acceptance rate** — the probability that the target accepts a drafter's proposal at any given position, conditioned on all earlier proposals at this step having been accepted. Let $\gamma$ be the **draft length** (the number of tokens the drafter proposes per verification call). Let $c = T_d / T_t$ be the cost ratio of a drafter forward pass to a target forward pass; for a 1B drafter on a 70B target with the same hardware and similar architecture, $c$ is typically 0.05–0.10.

The **expected number of tokens accepted per verification call** is the expected number of consecutive successes in a Bernoulli$(\alpha)$ chain, capped at $\gamma$:

$$E[k] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

This counts the $K$ accepted drafts plus the one bonus token the target always emits "for free" in the same verify pass (sampled from the residual at the rejection point, or from the $\gamma+1$-th position if all drafts were accepted). The expected speedup over baseline decoding is then:

$$S(\alpha, \gamma, c) = \frac{E[k]}{1 + c \gamma} = \frac{1}{1 + c \gamma} \cdot \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

The numerator grows with $\gamma$ but is capped by $1/(1-\alpha)$. The denominator grows linearly in $\gamma$. Differentiate, and you find a sweet spot $\gamma^\star$ that depends only weakly on $c$ and almost entirely on $\alpha$.

![Speedup as a function of acceptance rate and draft length](/imgs/blogs/speculative-decoding-3.png)

The matrix above shows the closed form for $c = 0.07$. At $\alpha = 0.95$, doubling $\gamma$ from 4 to 8 still buys you 60% more tokens per call — go big. At $\alpha = 0.5$, you peak at $\gamma=4$ and get worse from there. The cliff in the bottom-left ($\gamma=8, \alpha=0.5$) is the regime where most novice speculative-decoding deployments live, and it is exactly the regime where they fail to ship: the math is screaming "you are wasting drafter compute on tokens that will be rejected", but the operator only sees an unexplained 1.4× speedup that does not justify the engineering cost.

```python
import numpy as np
def speedup(alpha, gamma, c=0.07):
    E_k = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return E_k / (1 + c * gamma)

for a in [0.5, 0.7, 0.85, 0.95]:
    for g in [2, 4, 6, 8]:
        print(f"alpha={a}, gamma={g}: {speedup(a, g):.2f}x")
```

Three corollaries fall out of this formula and they are the only three you need to remember:

- **High α dominates everything.** A drafter that lifts α from 0.7 to 0.85 is worth more than a drafter that halves $c$. This is why every modern technique (EAGLE, Medusa, MTP, ReDrafter) is fundamentally an *acceptance-rate* play, not a draft-cost play.
- **Optimal γ scales with $\log(1/\alpha)$, not with model size.** At α = 0.6 you want γ ≈ 3. At α = 0.9 you want γ ≈ 7. Don't blindly copy γ from a paper that benchmarked on a different drafter.
- **The speedup ceiling is $1/(1-\alpha)$.** No matter how cheap your drafter, no matter how long your draft, you cannot do better than this. At α = 0.8 the ceiling is 5×; at α = 0.5 the ceiling is 2×. If the marketing deck promised you 4× and your α is 0.7, somebody is benchmarking on cherry-picked prompts.

## Anatomy of a verification step

Now zoom into a single position. The drafter proposes token $x$ from its distribution $q(\cdot)$. The target, in the verification pass, computes its own distribution $p(\cdot)$ at the same position. We want to decide whether to keep $x$ — and whatever we do, the marginal distribution of the surviving token must be exactly $p$.

![Modified rejection sampling at one position](/imgs/blogs/speculative-decoding-4.png)

The decision is **modified rejection sampling**:

1. Draw $r \sim U(0,1)$.
2. If $r \leq p(x)/q(x)$, **accept** $x$ and move to position $i+1$.
3. Otherwise, **reject** $x$, stop verifying any further drafted tokens at this step, and resample one token $y$ from the **residual distribution** $p_\text{res}(\cdot) \propto \max(0, p(\cdot) - q(\cdot))$.

Two non-obvious facts about this rule:

**Fact 1: it preserves the target distribution exactly.** The probability that the surviving token is $z$ equals $q(z) \cdot \min(1, p(z)/q(z)) + (1 - A) \cdot p_\text{res}(z)$, where $A = \sum_x q(x)\min(1, p(x)/q(x))$ is the acceptance probability. Algebra grinds this down to $p(z)$. The proof is short and lives in the Leviathan et al. (2023) paper if you want to redo it on a napkin.

**Fact 2: rejection at position $i$ kills positions $i+1, \dots, \gamma$.** Once the residual sample replaces a drafted token, the prefix has diverged from what the drafter assumed when proposing later tokens. The target's distribution at position $i+1$ given the new token is *different from what we computed in the verify pass*. This is the hidden cost of speculation — every rejection is "throw away the suffix and take the bonus token". It is also why high α matters quadratically: a 0.9-vs-0.7 drafter does not just accept more tokens at each position, it gets to deeper positions before its suffix is cut off.

A common implementation mistake is to skip the residual resample and just take the target's *argmax* at the rejection position. This biases the decoded distribution — the bias is small at temperature 0 but compounds over thousands of tokens at temperature 1 and shows up as subtle quality regressions (more repetition, lower diversity, occasionally a coherence loss). The vLLM, SGLang, and TensorRT-LLM implementations all do the proper residual resample. If you are building your own, the test that catches this bug is a histogram of generated token IDs over 100k tokens with target = drafter and temperature 1: it should match the target's no-spec histogram to within Monte Carlo error. If it doesn't, you have a residual bug.

```python
def verify_one_position(p, q, x, rng):
    """Returns (accepted_token, accepted_bool)."""
    r = rng.uniform(0, 1)
    if r <= p[x] / max(q[x], 1e-9):
        return x, True
    residual = np.maximum(0.0, p - q)
    residual = residual / residual.sum()
    return rng.choice(len(p), p=residual), False
```

For temperature-0 (greedy) decoding the rule degenerates: accept iff $\arg\max q = \arg\max p$ at this position. For temperature $T$ both $p$ and $q$ must be evaluated at temperature $T$ — a very common bug is to draft at $T=0$ and verify at $T=1$, which makes the ratio $p(x)/q(x)$ collapse for any $x$ that the drafter would not have argmaxed, and α tanks. Drafter and target temperatures must match. (Case study #4 below is exactly this.)

## The drafter zoo

Almost every speculative decoding paper since 2023 is, at its core, a different way of producing $q$. The taxonomy I keep in my head has two axes: **independent vs coupled** with the target, and **per-token (chain) vs multi-head (tree)** in how it produces candidates per call.

![The drafter zoo](/imgs/blogs/speculative-decoding-5.png)

| Family | Example systems | α (typical) | $c$ (drafter cost) | When to reach for it |
|---|---|---|---|---|
| Independent draft model | vLLM `speculative_model`, original DeepMind 2023 | 0.65–0.80 | 0.05–0.10 | When you have a small model from the same family already trained (Llama-3.2-1B for Llama-3-70B). Decoupled lifecycle, simple to debug. |
| n-gram / prompt lookup | REST, vLLM `prompt_lookup_decoding` | 0.30–0.90 (depends on input) | ~0 | Repetitive workloads — code completion, retrieval-augmented generation, agents copying context. Free if you have it. |
| Self-speculation heads (chain) | Medusa | 0.55–0.75 | ~0.01 | Latency-critical batch-1 serving where you control the target weights. Cheap to add post-hoc. |
| Tree drafter (coupled) | EAGLE, EAGLE-2, EAGLE-3, ReDrafter | 0.75–0.90 | 0.02–0.05 | Highest acceptance per drafter compute. Best general-purpose choice in 2025–2026 if you can train it. |
| MTP heads (multi-token prediction) | DeepSeek-V3, Qwen2.5-MoE | 0.70–0.85 | ~0.02 | When MTP is part of the pretraining recipe; reusing the heads at inference is essentially free. |

A few notes on the rows worth their own paragraph.

**Independent draft models** look like the simplest option — pick any smaller model from the same family and point your serving stack at it. The trap is that "same family" is a stronger condition than people realize. Llama-3.1-8B drafting for Llama-3.1-70B works because they were co-trained on the same corpus with related architectures. Llama-3.1-8B drafting for Llama-3.3-70B-Instruct gets you α ≈ 0.55 instead of 0.78 — same architecture, same family name, different post-training mix. The drafter and target must agree about token boundaries, system prompt habits, and instruct formatting. A drafter trained on raw web data and a target heavily RLHF'd will disagree the most exactly where it matters: the polite, refusal-ish, structured-output continuations a chat target prefers.

**n-gram lookup** is the cheapest drafter on earth and shockingly competitive in narrow workloads. The idea is: when generating, look up the last 2–4 tokens in the *prompt itself*, find any matching span, and propose its continuation as the draft. For long-context summarization (the prompt contains the source text and the model is paraphrasing it) you hit α ≈ 0.7. For code completion of a function whose tokens are all in scope above, α ≈ 0.85. For free-form chat, α ≈ 0.15 and you are wasting cycles. The vLLM flag is `--num-speculative-tokens 5 --speculative-model "[ngram]"`; the cost is essentially zero memory and zero FLOPs.

**EAGLE-family drafters** are the state of the art. EAGLE adds a small autoregressive head on top of the target's hidden states, predicting next-token *features* rather than tokens directly, then decoding the features through the target's own LM head. EAGLE-2 makes the head produce a *tree* of candidates with learned probabilities so you can fan out a few likely branches and verify them all in one target pass. EAGLE-3 extends the feature concat to multiple layers. The acceptance rates in the 0.85–0.90 range you read about in benchmarks are real, but they are conditional on (a) training the EAGLE head on data drawn from the *finetuned* target, not the base model, and (b) keeping the EAGLE head's KV cache aligned with the target's KV cache, which is a non-trivial implementation detail.

**MTP heads** are the most under-appreciated entry. DeepSeek-V3's pretraining recipe uses a multi-token-prediction objective (predict the next $D$ tokens, weighted by depth) as a training-time auxiliary loss. The clever bit at inference time is that the same heads, having been optimized to predict $D$ steps ahead, double as a coupled drafter for free. No second training run, no separate model, no extra parameters at deployment. This is a glimpse of where the field is going: training-time choices that subsidize inference-time speculation.

## Tree vs chain speculation

So far we have mostly imagined the drafter as proposing a single linear sequence of $\gamma$ tokens. That is the **chain** strategy, and it is what the original DeepMind paper does. The problem is that as soon as α drops below 1, every rejection wastes the rest of the chain. If $\alpha = 0.7$ and $\gamma = 4$, the probability that the entire chain is accepted is $0.7^4 \approx 0.24$ — three times out of four, you have wasted at least one drafter forward pass.

The fix is to draft a **tree** of continuations and verify all of them in a single masked attention pass.

![Chain vs tree speculation](/imgs/blogs/speculative-decoding-6.png)

A tree of width $W$ and depth $D$ proposes $W^D$ candidate paths. The verification pass evaluates them simultaneously with a custom attention mask that lets each path attend only to its own ancestors — the **tree attention** kernel. The target produces $W^D$ candidate logit vectors per call, but the cost is roughly $W^D$ tokens in the verify pass, not $W^D$ separate calls.

The acceptance math becomes more nuanced. Now you accept the *deepest path through the tree that survives modified rejection sampling at every node*. Because the residual sample at any rejection node still buys you a free bonus token, and because multiple paths give you more chances to follow a high-likelihood continuation, expected accepted-tokens-per-call rises substantially. EAGLE-2 reports 1.5–2× improvement over chain at the same drafter cost.

The catch is **KV-cache pressure**. A tree of 16 candidates means up to 16× the KV-cache fan-out during the verify pass. PagedAttention copes with this gracefully — each tree branch uses its own block list — but on stacks without paging, you can blow your KV budget at long context. The fix is **adaptive tree topology**: shrink width and depth as context grows, or as α drops below a threshold. SGLang's RadixAttention plus tree-spec scheduler does this dynamically; vLLM's tree-spec is opt-in and has fewer adaptive controls.

> Trees give you more shots on goal per verify pass. Chains give you cleaner failure modes. Pick chains when you don't have tree-attention kernels; pick trees when you have α > 0.7 and KV headroom.

## Batching and the throughput cliff

Here is the part of speculative decoding that gets glossed over in every paper but is the single most common reason production deployments roll the feature back: **speculative decoding is a latency optimization, and at high concurrency it becomes a throughput regression**.

![The throughput cliff](/imgs/blogs/speculative-decoding-7.png)

The figure above shows the regime change. At batch 1, decoding is bandwidth-bound; the target's verify pass costs essentially the same as a single-token decode would, so every accepted draft is a free token. At batch 4, you are still well below the compute roof; spec wins big. At batch 16, you are within 30% of the compute roof; spec wins marginally. At batch 64, the target's verify pass on a $\gamma$-token suffix costs $\gamma+1$ times more than a single-token decode would have, because you are now compute-bound and FLOPs scale linearly with sequence length in the pass.

The break-even point depends on three knobs that interact:

1. **Model size** — larger models reach the compute roof at higher concurrency. A 7B on an H100 is compute-bound at batch ≈ 32; a 70B on the same H100 is bandwidth-bound until batch ≈ 128.
2. **Sequence length** — long contexts widen the verify pass and bring you to the compute roof faster.
3. **Acceptance rate** — at high α you are wasting fewer FLOPs in the verify pass, so the cliff arrives later.

Concretely, on an H100 SXM5 serving Llama-3.1-70B at 4k context:

| Batch | Mode | Tokens / sec | Verdict |
|---|---|---|---|
| 1 | no spec | 42 | baseline |
| 1 | EAGLE-2 γ=5 | 118 | **2.8× win** |
| 4 | no spec | 160 | baseline |
| 4 | EAGLE-2 γ=5 | 360 | **2.2× win** |
| 16 | no spec | 540 | baseline |
| 16 | EAGLE-2 γ=4 | 720 | 1.3× marginal |
| 64 | no spec | 1640 | baseline |
| 64 | EAGLE-2 γ=4 | 1480 | **0.9× LOSS** |

The takeaway is operational, not theoretical: if your endpoint serves a wide concurrency range, **you must be able to disable speculation per-batch**. vLLM and SGLang both expose this; TensorRT-LLM bakes the choice into the engine. Static "spec on" or "spec off" deployments leave money on the floor in both directions.

## Continuous batching meets speculation

Modern LLM servers do not process requests in lockstep. They use **continuous batching** (a.k.a. iteration-level scheduling): every step, the scheduler decides which sequences to advance, splices new prefills into the batch, evicts finished sequences, and runs one forward pass on the resulting heterogeneous batch. PagedAttention makes this affordable because each sequence's KV blocks are independently managed.

Speculative decoding adds a second dimension of irregularity: **per-sequence accept lengths differ by step**. Sequence A advances 5 tokens this step; sequence B advances 1 because its first draft was rejected; sequence C just emitted EOS. The scheduler must commit ragged advances to the KV cache, recompute next-step batch composition, and possibly re-trigger the drafter on a different prefix length per sequence — all without losing the win that motivated speculation in the first place.

![Continuous batching with ragged accept lengths](/imgs/blogs/speculative-decoding-8.png)

Three implementation choices materially affect throughput here:

**Where the drafter runs.** The drafter forward pass can be (a) batched across all sequences in lockstep, (b) per-sequence on its own stream, or (c) tree-fused with the target's verify pass. Option (a) is the simplest and what vLLM's classic spec uses; it loses efficiency when a few sequences have just rejected and need a different prefix than the rest. Option (b) parallelizes across sequences but adds GPU stream management overhead. Option (c), used by EAGLE and ReDrafter, runs the drafter heads as part of the target's forward graph — no separate process, no separate kernel launch. This is where TensorRT-LLM's ReDrafter integration earns its 30% latency win over vLLM's classic spec at small batches.

**How rejected suffixes are handled.** When a verify pass rejects a draft and resamples a residual, the *next* spec step needs a fresh draft starting from the residual token. Naive implementations stall the sequence for one decoding step (no spec) before resuming. Smart implementations (vLLM 0.6+, SGLang 0.4+) keep the sequence in the spec path and call the drafter on the residual prefix — the only cost is one extra small-batch drafter call.

**Whether the batch waits for the slowest sequence.** A single ragged accept (one sequence committing 1 token while four others commit 5) does not hurt throughput because they all share the same verify pass. But a *failure to disable spec for a high-concurrency sub-batch* does — case study #7 below is exactly this.

## Tuning knobs that actually matter

After you have picked your drafter, three knobs determine 90% of the speedup. The rest is noise.

![Tuning surface: gamma vs tree width W](/imgs/blogs/speculative-decoding-9.png)

The matrix above shows real-world tuning behavior on a 70B target with an EAGLE-2 drafter, normalized speedup against no-spec at batch 1. The basin of high speedup is narrow: $\gamma \in [4, 6]$ and tree width $W \in [2, 4]$. Outside that basin, KV pressure (top-right) or rising drafter cost (bottom-right) dominate. If you tune $\gamma$ once at deployment time and never touch it again, you should land in this basin — but **the basin shifts with workload**. On long-context tasks the optimum is $(\gamma=4, W=2)$. On short, repetitive tasks $(\gamma=8, W=4)$ wins. Adaptive selection from a small per-batch decision matrix is worth the engineering.

The three knobs:

**1. Draft length γ.** Start at $\gamma = \lceil \log_{0.5}(1 - \alpha) \rceil + 1$. For α = 0.7, γ ≈ 4. For α = 0.9, γ ≈ 7. Walk it ±2 in benchmarks and pick the peak. Do not believe the paper's recommendation if your α is different from theirs.

**2. Tree topology (width × depth).** If you have tree-attention kernels and KV headroom, set initial $(W, D) = (2, 4)$ and ramp depth before width. Wider trees stress the verify pass more than deep trees because attention is quadratic in suffix length.

**3. Drafter temperature alignment.** Both drafter and target sample at the same temperature. If your target serves a mix of temperatures (chat at 0.7, code at 0.0, JSON at 0.0), you may want temperature-conditional drafter heads or simply disable spec for the hottest temperatures where α is naturally lower. The sampling code in vLLM and SGLang both support this, but you have to look for the flag.

```bash
## vLLM 0.6+ with EAGLE-2 drafter
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
  --speculative-model yuhuili/EAGLE-LLaMA3.1-Instruct-70B \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --use-v2-block-manager
```

```bash
## SGLang with tree-spec
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path yuhuili/EAGLE-LLaMA3.1-Instruct-70B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 64
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)
out = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=256))
```

A non-obvious knob worth naming: **the drafter's KV cache must be reused across spec steps**. The drafter advances one token per draft step, exactly like a normal decoding loop, and its KV cache should grow accordingly. Bug-prone implementations re-prefill the drafter from scratch every spec call, which silently triples drafter cost. The clue in your traces is a drafter latency that scales linearly with sequence position rather than staying constant.

## Measuring acceptance rate and drafter cost in production

You cannot tune what you do not measure. The single most common mistake teams make when adopting speculative decoding is shipping it without instrumentation, then debating tuning constants without data. Two numbers are non-negotiable for any spec-decode deployment: **per-sequence rolling α** and **end-to-end speedup vs no-spec**. Anything else is downstream.

**Per-sequence α.** vLLM exposes `metrics.spec_decoding.acceptance_rate` per request and a rolling average over the last $N$ verify passes. Export it as a histogram into Prometheus, bucketed by `(model, temperature, prompt_class)`. The histogram, not the mean, is what you look at — α distributions for production traffic are bimodal more often than you'd think. A mean of 0.72 that decomposes into 80% of requests at α = 0.85 and 20% at α = 0.20 tells you to disable spec for the second cohort, not to retune γ for the average.

**End-to-end speedup.** Maintain a small canary endpoint with spec disabled and route ~5% of traffic to it. Measure p50/p99 tokens/sec and wall-clock latency on both paths, weighted by request class. The gap is your real speedup; it will be smaller than your benchmark numbers because real traffic mixes high-α and low-α requests. If the gap is below 1.3× p50, your tuning has work left.

**Drafter cost ratio $c$.** Measure separately. On most stacks the drafter and target run on the same GPU but with different stream priorities. Profile a single representative batch with NVIDIA Nsight Systems and look at the kernel timeline: drafter forward passes appear as small clusters of GEMMs between target verify passes. Sum drafter kernel time, divide by target verify time, and you have $c$ for that batch. Repeat for batch 1, 4, 16, 64; $c$ at high batch is dominated by drafter cost-per-token (which decreases with batch) but bounded below by drafter prefill cost (which is fixed-per-step). If your measured $c$ at batch 1 is much larger than the architectural ratio (say 0.30 instead of 0.07 for a 7B drafter / 70B target), you have a kernel inefficiency to debug — usually a missing CUDA graph capture on the drafter.

```python
## Minimal Prometheus exporter for spec-decode metrics
from prometheus_client import Histogram, Counter
acceptance = Histogram(
    "spec_decode_alpha",
    "Per-request acceptance rate",
    buckets=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    labelnames=["model", "temperature_bucket"],
)
verify_lat = Histogram(
    "spec_decode_verify_ms",
    "Wall-clock per verify pass",
    buckets=[5, 10, 20, 40, 80, 160, 320],
    labelnames=["model", "batch_bucket"],
)

def on_request_done(req):
    a = req.spec_metrics.accepted_tokens / max(1, req.spec_metrics.total_drafted)
    acceptance.labels(req.model, _bucket_temp(req.temperature)).observe(a)
    verify_lat.labels(req.model, _bucket_batch(req.batch_at_finish)).observe(
        req.spec_metrics.verify_wall_ms_avg
    )
```

Three diagnostic questions every dashboard should answer at a glance:

1. **What fraction of verify passes accepted zero tokens?** This is the "wasted spec" rate. Above 10% means your drafter is meaningfully out of distribution; investigate before tuning γ.
2. **What is the spec speedup ratio at p99 batch?** Anything below 1.0 means spec is hurting tail latency; you need a per-batch-size disable rule.
3. **What is the per-temperature α?** A single-number α hides the temperature cliff; if α at $T=0.0$ is 0.91 and α at $T=1.0$ is 0.42, you want temperature-conditional spec.

## Training your own drafter

Once you outgrow off-the-shelf drafters, you train your own. The two recipes that matter in 2026 are **EAGLE-style head training** and **distilled small drafters**, and they have different operational profiles.

### EAGLE head training

The EAGLE drafter is a small transformer (typically 1–2 layers) that takes the target's hidden states at the LM-head input and produces feature predictions for the next $\gamma$ positions, decoded through the target's own LM head. The training data is generated by running the target on a representative corpus and recording, for every position, the (input hidden state, target logits) pair. The loss is feature-MSE plus a weighted cross-entropy through the LM head — the cross-entropy is what aligns the drafter's distribution with the target's, and it is what gives EAGLE its high α.

Practical recipe (rough numbers for a 7B target):

| Item | Value |
|---|---|
| Drafter parameters | 0.3–0.5B (1–2 layers, hidden = target hidden) |
| Training data | 50M–200M tokens of target-generated continuations |
| Hardware | 8× A100 / H100 for ~24 hours |
| Loss | $\mathcal{L} = \lambda_{\text{feat}} \|\hat{h} - h\|^2 + \lambda_{\text{ce}} \text{CE}(\hat{p}, p)$ |
| Typical $\lambda$ | $\lambda_{\text{feat}} = 0.1$, $\lambda_{\text{ce}} = 1.0$ |

Three implementation traps cost most teams a week each:

**Trap 1: training on base-model continuations when serving the instruct model.** EAGLE heads must be trained on continuations from the *exact target* you serve, including any RLHF or DPO post-training. A head trained on Llama-3.1-8B base and deployed on Llama-3.1-8B-Instruct gets you 0.55 acceptance instead of 0.85.

**Trap 2: KV-cache misalignment between drafter and target.** EAGLE shares the target's KV cache positions but maintains its own KV for the drafter's transformer layers. If the page indexing logic is wrong by even one token, the drafter is conditioning on a stale prefix and α silently halves. The smoke test is to run greedy decoding (T=0) with target = drafter (using the EAGLE head's own logits as both q and p) and confirm the head argmax sequence matches the target argmax sequence for the first 50 tokens. If it diverges, you have a KV alignment bug.

**Trap 3: drafter saturation on long context.** A 1-layer EAGLE head trained on 4k context degrades sharply when serving 32k context — long-range dependencies require more drafter depth. The fix is mixed-context training: sample 70% short, 20% medium, 10% long context during EAGLE training.

### Distilled small drafters

When you can't or won't share the target's hidden states, the alternative is a fully-decoupled small model trained by **knowledge distillation** from the target. The teacher is the production target running over a curated prompt distribution; the student is the drafter you will serve. The distillation loss is the standard temperature-softened cross-entropy:

$$\mathcal{L}_\text{distill} = \tau^2 \cdot \text{KL}\big(p_\text{teacher}(\cdot/\tau) \,\|\, p_\text{student}(\cdot/\tau)\big)$$

with $\tau \in [2, 4]$. The point of distillation here is not to get the student to match the teacher's argmax (it's small; it can't), but to match the teacher's *mass distribution* — that is what α actually depends on. A student that argmaxes correctly 80% of the time but has wildly wrong tail probabilities will get an α of 0.55 because of the modified rejection sampling rule. A student that argmaxes correctly only 70% of the time but has well-calibrated tails will get α of 0.75.

This is also the right framing for explaining why **independent drafters from the same family work pre-distilled**: Llama-3.2-1B was trained on the same data distribution as Llama-3.1-70B, so its mass distribution is already similar in the regions both models cover. Distillation explicitly aligns the tails further. See the [distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) deep-dive for the full training recipe; the spec-decoding-specific considerations are: (1) match temperature during distillation to the temperature you will serve at, and (2) ensure the distillation corpus covers the production prompt distribution, not just generic web text.

## Verifying quality preservation

The strongest claim in this article is that speculative decoding preserves the target distribution exactly. That claim is only as good as your testing. Three tests every spec deployment should run on every release.

**Test 1: distributional equivalence at temperature 0.** With $T = 0$ and a fixed seed, the target's argmax sequence should be bit-identical with and without spec. If they differ, you have a logit-numerical bug (most often, the drafter producing logits in a different precision than the target and the comparison ratio being off in the last decimal). Run on 1000 prompts × 256 tokens each, compare full sequences, expect exact match. Anything less is a bug.

**Test 2: distributional equivalence at temperature 1.** Statistical, not exact. Sample $N \geq 10^5$ tokens from the target with no spec, fix a histogram over the top-1024 token IDs. Sample $N$ tokens with spec on, same prompts, same RNG seed for the rejection sampling decisions but different draws (it cannot be bit-identical when α < 1 because the rejection draws inject randomness). The two histograms should pass a chi-squared goodness-of-fit test at p > 0.01 over the 1024 buckets. If they don't, you have a residual sampling bug.

**Test 3: downstream task equivalence.** The strongest test, run on every release: compare downstream eval scores (MMLU, HumanEval, MT-Bench, your task-specific eval) with spec on vs off, with the same sampling parameters. Expect equality within run-to-run noise (typically ±0.3 points on a 100-point benchmark). A 1-point regression is suspicious. A 3-point regression is a bug. (We have caught two real spec-decode bugs in production this way: a residual normalization off-by-one and a tree attention mask mistake.)

The first two tests are unit-test material; run them in CI on every spec-decode change. The third is too expensive for CI but should run on every deployment candidate.

> Speculative decoding is a load-bearing optimization in 2026, and load-bearing optimizations need their own test suite. If you cannot prove distributional equivalence, you cannot safely ship spec.

## Speculation in disaggregated and tensor-parallel deployments

Two production realities complicate the simple drafter-and-target picture: **disaggregated prefill/decode** and **tensor-parallel target models**.

### Disaggregated prefill / decode

State-of-the-art LLM serving stacks (vLLM disaggregated, NVIDIA Dynamo, TogetherAI's stack) split prefill and decode onto different GPU pools. Prefill is FLOP-bound; you want fewer, bigger, faster GPUs. Decode is bandwidth-bound; you want more, smaller, KV-cache-rich GPUs. The two pools communicate by transferring KV blocks over NVLink or NIC. Speculative decoding lives entirely in the decode pool — but the drafter's model weights, KV cache, and stream allocation all have to be planned around that placement.

Three patterns work in disaggregated stacks:

1. **Drafter on every decode node.** The simplest. Drafter weights are replicated; each decode node runs target verify and drafter forward on the same GPU. KV bandwidth is local. Cost: drafter weights take HBM that could have been KV cache.
2. **Centralized drafter pool.** Drafter weights live on a small dedicated GPU pool (often L4 or A10G class). Each draft request goes over the network. Drafter cost amortizes across many decode nodes, but per-step latency rises by the network round trip — usually a regression unless the drafter is large.
3. **Drafter as a fused block in the target's verify pass.** EAGLE/ReDrafter/MTP pattern. The drafter is part of the target's forward graph; no separate process; KV layout is shared. This is the only pattern that scales well in disaggregated stacks.

If you are designing a fresh disaggregated deployment in 2026, the answer is pattern 3. Pattern 1 is fine for small fleets; pattern 2 only works at extreme drafter sizes (≥ 7B drafting for ≥ 405B targets) where centralization actually amortizes.

### Tensor-parallel targets

When the target spans multiple GPUs via tensor parallelism (TP), the drafter has a placement choice: **TP-replicated** (drafter on each rank, identical weights) or **TP-sharded** (drafter sharded across the same ranks as the target). vLLM's `--speculative-draft-tensor-parallel-size 1` flag picks the first; no flag picks the second. The trade is compute vs communication.

- **TP-replicated drafter.** Each rank runs the full drafter. No NCCL traffic for the drafter forward pass. Costs HBM equal to drafter parameters × number of ranks. At TP=8 with a 1B drafter, you are paying 8 GB of HBM for drafter replication. On 80 GB H100s with a 70B target taking ~140 GB of weights split across 8 ranks (~17.5 GB/rank), this is fine.
- **TP-sharded drafter.** Drafter weights split across ranks like the target. NCCL traffic during drafter forward, just like the target. HBM cost is amortized. Latency is worse because all-reduce overhead is amplified for small drafters where the all-reduce is most of the wall time.

The rule of thumb: **TP-replicate drafters smaller than 3B; TP-shard drafters larger than 7B; benchmark in between**. The numbers shift on different interconnects (NVLink 4 vs PCIe Gen5) so always profile.

A subtle gotcha: when the drafter is TP-replicated, the rejection sampling RNG must be **synchronized across ranks** so all ranks make the same accept/reject decisions. Otherwise different ranks commit different prefixes, KV diverges, and the model produces gibberish. The implementations get this right, but if you write your own, this is the bug you will hit on day one.

## Decoding strategies and how they interact with speculation

Speculative decoding does not exist in isolation; it composes with the rest of your sampling pipeline. Some compositions are clean; some are subtly wrong; one is actively broken.

**Greedy (T=0).** Cleanest possible composition. Drafter picks argmax, target accepts iff argmax matches. α equals the per-token argmax-agreement rate, which is high for distilled drafters (0.85–0.95). If you serve only greedy decoding, you can use a simpler accept rule (string compare on argmax), which is what most code-completion tools do.

**Top-k / top-p (nucleus).** Both target and drafter must apply identical top-k / top-p filtering before computing $p$ and $q$. The most common bug here is filtering only at the target side; the drafter's $q(x)$ then has support outside the target's filtered set, and any accepted token outside the target's nucleus violates the distribution preservation guarantee. The vLLM and SGLang fixes are to apply filtering symmetrically.

**Repetition penalty / presence penalty.** These are post-logits adjustments computed from the *generation history*. They are not a problem at the verify pass (the target sees the same history) but they are a problem at the drafter (the drafter sees the same history but applies a different LM, so its $q$ is computed under different conditions). The clean fix: apply repetition penalty only to the target $p$, not to the drafter $q$. Most implementations get this wrong by default; check before relying on rep penalty + spec.

**Beam search.** Spec decode is fundamentally incompatible with beam search at the level the original paper formulates it — beam search maintains $B$ candidate sequences and picks the global best, while spec decode commits a single accepted prefix per step. There are research papers on tree-decoding-as-beam-search (HotSpec, BiLD-beam) that try to reconcile the two; in production, just use top-p with high $p$ instead of beam search, and you'll get spec-decode-compatible quality with much higher throughput.

**Constrained decoding (JSON / regex / grammar).** Spec preserves the target distribution but the *grammar enforcer* often runs after sampling, masking out grammar-violating tokens. If the drafter doesn't apply the same mask, every drafter proposal that violates the grammar gets rejected — α collapses on structured outputs. Outlines (the constrained-decoding library) gained spec-aware mode in late 2024 that propagates the grammar mask to the drafter; without that, spec decode on JSON workloads gets α ≈ 0.4 instead of 0.8.

| Strategy | Spec compatibility | Tweaks required |
|---|---|---|
| Greedy (T=0) | clean | none |
| Top-k / top-p | clean | apply filter symmetrically |
| Repetition penalty | clean if separated | apply to target only |
| Beam search | broken | use top-p instead |
| Constrained / JSON | needs grammar-aware drafter | propagate mask to drafter |
| Min-p | clean | apply filter symmetrically |

## Speculation and quantization

Most production targets in 2026 are quantized — INT8 weight-only, FP8 weights+activations, INT4 weight-only with FP16 KV. Speculative decoding interacts with quantization in three places, and each one has its own failure mode.

**Drafter quantization.** The drafter can be quantized independently of the target. INT4 drafters are common because the drafter is bandwidth-bound just like the target was, and INT4 reduces drafter wall time by ~3×. The watch-out: aggressive quantization (INT3, ternary) drops α non-trivially because it shifts the drafter's mass distribution. INT4 typically loses 1–3 points of α; INT3 loses 5–10. Always re-measure α after quantizing the drafter.

**Target quantization affects α.** A drafter that hit α = 0.83 against an FP16 target may only hit α = 0.78 against an INT8 quantized version of the same target — the quantization shifts the target's mass distribution slightly, and the drafter was trained against the FP16 distribution. The fix is to retrain the drafter (or EAGLE head) against the quantized target. EAGLE training data should be regenerated whenever the target's serving precision changes.

**Logit comparison precision.** The accept-or-reject test compares $p(x)/q(x)$ to a uniform draw. If $p$ and $q$ are computed in different precisions (target in FP16, drafter in BF16), the ratio at small $p, q$ values can be off by 5–10% — small probabilities get magnified by the division. The fix is to compute both in the same numerical precision (typically FP32) at the comparison point. vLLM and SGLang do this; if you write your own, do not skip it.

A compounding effect worth knowing about: **FP8 KV cache + spec decode + tree spec is the cliff stack**. FP8 KV reduces precision in attention outputs; tree spec re-uses those attention outputs for many candidate paths; spec compares logits across positions. Errors compound. We have seen α drop from 0.85 (FP16 KV) to 0.62 (FP8 KV with naive quantization) on the same target and drafter. The fix is per-channel FP8 scaling rather than per-tensor, which reduces the precision loss enough that α only drops ~3 points instead of 23.

## Profiling, traces, and what to look for

Most spec-decode bugs hide in the GPU timeline, not in the Python layer. Three traces every spec-decode operator should learn to read.

**Nsight Systems timeline.** Run a representative single-stream workload with `nsys profile -t cuda,nvtx,osrt --output spec_trace`. The timeline should show, for each verify pass: a target forward pass (large NCCL all-reduces if TP > 1), interspersed with small drafter forward passes between verifies. The ratio of drafter to target wall time on the timeline is your measured $c$. A correctly-tuned spec decoder shows a clean rhythm: verify, K small drafts, verify, K small drafts. A misconfigured one shows long gaps between drafts (CPU-side scheduling overhead) or oversized drafter passes (drafter not using a CUDA graph).

**Trace anomalies that mean a bug:**
- Drafter passes growing linearly with sequence position → KV cache not reused, drafter re-prefilling every step. Fix: enable drafter KV state.
- Verify pass occasionally taking 2–5× longer than peers → tree spec falling back to chain because the tree pruner blew a budget. Fix: reduce tree width or add adaptive pruning.
- All-reduces longer than expected on the drafter → drafter sharded across too many ranks for its size. Fix: TP-replicate.
- Long Python gaps between verify passes → CPU-side scheduling bottleneck. Fix: increase async draft generation, batch the per-sequence drafter calls.

**Per-step accept distribution.** Log the number of accepted tokens per verify pass for 10,000 passes and plot the histogram. The shape tells you everything:
- Bell-shaped around $E[k]$: healthy spec decode.
- Bimodal at 0 and γ: drafter is either confidently right or completely wrong; α is decent on average but you have two prompt populations. Investigate per-class α.
- Mass at 0: drafter is consistently rejected. α is broken; check temperature, KV alignment, distribution shift since training.
- Mass at γ: α is high. Increase γ.

The 30 minutes you spend learning to read these traces will save weeks of "spec decode doesn't seem to be helping us" debugging.

## Speculation in agentic and tool-using workloads

The fastest-growing serving workloads in 2026 are not chat — they are **agents**: LLMs in a loop with tools, calling models repeatedly with growing context. Spec decode behaves differently here in three important ways, and most teams miss the implications.

**Acceptance is bimodal in agent workloads.** The agent's outputs are often a mix of high-α structured tokens (function call signatures, JSON keys, expected fields) and lower-α generative tokens (natural language explanations, reasoning chains). In a single response you might see α = 0.95 for the first 30 tokens (boilerplate) and α = 0.50 for the next 200 (reasoning). Constant-γ spec misses this; **adaptive γ that grows during structured passages and shrinks during reasoning** is worth implementing if you serve agentic workloads at scale.

**Tool-call boilerplate is a free lunch.** Function call openings (`{"name": "get_weather", "arguments": {`) are perfectly predicted by n-gram drafters because the schema is in the prompt. Adding a cheap n-gram drafter alongside an EAGLE drafter, with the system picking whichever has higher confidence at each position, lifts effective α by 5–10 points on tool-heavy workloads.

**Reasoning chain length amplifies spec wins.** A "reasoning model" output is 2,000–10,000 tokens of mostly autoregressive thought. At 2× spec speedup, that's 5,000 tokens of saved wall time per request. The economics shift: the engineering cost of training an EAGLE head dominates over a few weeks of agent traffic, vs over months for chat traffic. Agent workloads are the highest-leverage place to invest in spec-decode infrastructure in 2026.

## Memory layout: how spec decode reshapes KV usage

KV cache management changes meaningfully under spec decode because (a) every verify pass writes $\gamma+1$ KV positions instead of 1, and (b) on rejection, $\gamma - k$ of those positions must be discarded. The naive implementation writes all $\gamma+1$ KV positions on every verify pass, then "rolls back" the rejected suffix by overwriting on the next step. PagedAttention does this cleanly because rolling back is just freeing pages; on contiguous KV stacks, rollback creates fragmentation.

A subtle implementation choice: should the drafter's KV cache be **persistent across spec steps** or **rebuilt from the target's KV**? Persistent is faster; rebuilt is simpler and avoids drift. EAGLE chose persistent and pays a small alignment-bug tax. vLLM's classic spec chose rebuilt and pays a small latency tax. Both are reasonable; the lesson is that the choice is load-bearing and worth understanding before changing.

When you tree-spec, the KV layout becomes a tree-shaped fan-out during the verify pass. A tree of 16 candidate paths writes 16 candidate KV positions per layer, then prunes 15 of them after the rejection sampling decisions. PagedAttention treats this as 16 page allocations and 15 frees per step — high allocation pressure. The vLLM scheduler pre-allocates a "tree spec scratch space" of fixed size to avoid the alloc/free thrashing. SGLang's RadixAttention can keep the unaccepted branches around as cached prefixes for future requests, which is occasionally a win on agent workloads where similar tool calls recur.

## Case studies from production

Eleven incidents I have either debugged personally or watched a colleague debug. Each one is concrete: real model, real flag, real symptom — and each one teaches a lesson that does not appear in the original Leviathan paper.

### 1. The drafter that beat the target

A team replaced a 7B drafter with a 13B drafter for a 70B target, hoping the larger drafter would have higher α. The α did rise from 0.78 to 0.84. The end-to-end latency *got worse* by 22%.

The math explains it. With $T_d = 0.07 T_t$ at 7B, $E[k]/(1 + 0.07 \cdot 5) = 3.7 / 1.35 \approx 2.74$. With $T_d = 0.18 T_t$ at 13B, $E[k]/(1 + 0.18 \cdot 5) = 4.0 / 1.90 \approx 2.10$. The drafter cost dominated the acceptance gain. The fix was reverting to 7B and instead spending the engineering budget on EAGLE-2 over 7B, which lifted α to 0.86 at $T_d = 0.04 T_t$.

The deeper diagnosis was instructive. The team had assumed that "larger drafter = higher α" was a monotonic relationship, but the EAGLE-2 result on the smaller 7B base shows that the *training procedure* of the drafter matters far more than its parameter count. EAGLE training over a 7B base gives you a drafter that has been *explicitly aligned to the target's distribution*, while a vanilla 13B from the same family is only implicitly aligned by sharing pretraining data. Aligned-and-small dominates large-and-unaligned almost always.

The lesson: **drafter cost matters as much as α**. Always compute $E[k]/(1 + c \gamma)$ end-to-end before committing to a drafter swap. And before scaling up the drafter, ask whether you can lift α with training alignment instead — it's almost always the better lever.

### 2. EAGLE-2 on a long-context summarizer that blew KV at 32k

EAGLE-2 with tree depth 5 and width 4 is wonderful at 2k context. The same configuration at 32k context started OOMing in the middle of long summarization requests. The cause: tree spec with $W=4, D=5$ multiplies KV pressure during the verify pass by up to 1024 candidate paths (capped by tree pruning), and at 32k the per-sequence KV is already 6 GB.

The fix was a length-conditional adaptation in the scheduler: if context length > 8k, drop to $W=2, D=3$; if > 24k, drop to chain ($W=1, D=4$). End-to-end speedup went from "great below 8k, OOM above 16k" to "2.5× below 8k, 1.6× above 16k, never OOM". The lesson: **tree-spec parameters must be a function of context length, not a deployment constant**.

### 3. n-gram lookup wins on code completion

A code-completion endpoint running Code-Llama-34B was paying for vLLM with an EAGLE-2 drafter. Latency p50 was 38 ms/token. We tried `--speculative-model "[ngram]" --num-speculative-tokens 6` and the p50 dropped to 21 ms/token. The α was 0.88 because most code-completion suffixes are token-for-token reproductions of context already in the prompt (function bodies repeat their signatures, imports repeat their names, type annotations repeat).

The cost of the n-gram drafter is essentially zero — a small lookup table built once per request from the prompt. The lesson: **for repetitive workloads, the cheapest drafter wins, and "cheap" beats "smart" when α is already high**.

### 4. The temperature mismatch that killed acceptance

A team ran a chat endpoint at temperature 0.8. The drafter was being run at temperature 0 because they wanted "deterministic drafts". Acceptance plummeted to 0.41 — the target's distribution at $T = 0.8$ is much wider than the drafter's argmax, so even drafted tokens that had decent target probability often failed the $r \leq p(x)/q(x)$ test (because $q(x) = 1$ for the argmax made the ratio always equal to $p(x)$, and $p(x)$ at $T=0.8$ was rarely close to 1).

The fix was to set drafter temperature to 0.8 too. α rose to 0.74 and end-to-end speedup recovered to 2.1×. The lesson: **drafter and target sampling temperatures must match exactly**. Any tooling that sets them independently is broken.

### 5. Medusa heads went stale after fine-tuning

A team trained Medusa heads on a base 13B model, then fine-tuned the target on instruction data. They re-deployed without retraining the Medusa heads. Acceptance dropped from 0.78 to 0.41 over the course of one week as more instruction-following requests hit the endpoint. The base-model heads were proposing pre-RLHF continuations that the post-RLHF target consistently rejected.

The fix was to retrain the Medusa heads on data drawn from the fine-tuned target. The lesson: **coupled drafters must be retrained whenever the target's distribution moves, including post-training adjustments**. If you cannot retrain, use a decoupled drafter that does not depend on target weights.

### 6. DeepSeek-V3 MTP at production batch

Reusing MTP heads as a drafter at inference time is essentially free at single batch. At batch 64, however, the MTP heads add real FLOPs to every forward pass — they are not trivial heads. The team measured a 1.4× speedup at batch 1 collapsing to 0.95× at batch 64. They added a `--mtp-spec-batch-threshold=8` flag that disables MTP-as-drafter above the threshold; throughput at batch 64 went from 0.95× to 1.0× (no spec) and the whole thing became unconditionally non-regressive across batch sizes.

The lesson: **even free drafters can hurt at high batch**. The correct answer is rarely "spec everywhere".

### 7. vLLM at batch 64 stalling on the slowest sequence

A reported throughput regression in vLLM 0.5 with spec-decode enabled at high concurrency turned out to be a scheduler issue: when one sequence in the batch had a very low α (say, generating a JSON schema while the rest of the batch was free-form chat), its repeated 1-token accepts slowed down the entire batch's spec step cadence. Each verify pass cost the same regardless of which sequences accepted what, but the per-sequence speedup averaged across the batch was being dragged down to 1.1× by the JSON sequence.

The fix in vLLM 0.6 was to allow per-sequence drafter disable: any sequence with rolling-α below a threshold gets spec turned off for it within the same batch. The lesson: **per-sequence spec opt-out is essential at high concurrency with mixed workloads**.

### 8. TensorRT-LLM ReDrafter on H100

A team migrated a Llama-3.1-70B endpoint from vLLM (with EAGLE-2 drafter) to TensorRT-LLM with ReDrafter, hoping for a kernel-fusion latency win on H100. They got it — single-stream p50 dropped from 18 ms/token to 11 ms/token, a 1.6× wall-clock improvement on top of the EAGLE-2 speedup. But the engine build time tripled (45 minutes vs 15) and the TRT-LLM engine was not portable across GPU SKUs, requiring per-SKU rebuilds.

The lesson: **kernel-fused spec on TRT-LLM is the latency-optimal solution for fixed hardware fleets**. For heterogeneous fleets or rapid iteration, vLLM's flexibility wins despite the latency tax.

### 9. The DeepSeek-V3 MTP retrofit on a non-MTP base

A team running Qwen2.5-72B wanted DeepSeek-V3-style MTP gains without retraining from scratch. They tried a "retrofit": train MTP heads on top of a frozen Qwen2.5 base using a small continued-pretraining run. After 50B tokens of MTP-loss training, the MTP heads achieved α = 0.71 — meaningful but well below the α = 0.83 a co-trained MTP head reaches.

The diagnosis was that retrofit MTP heads cannot fully recover the *shared representation* benefit that MTP gets when the heads are part of pretraining from token zero. The base model's hidden states had been optimized for next-token prediction; making them simultaneously useful for $D$-token prediction required either deeper MTP heads (which raised $c$) or partial unfreezing of the base (which risks regressing the target).

The lesson: **MTP works best when planned at pretraining time**. Retrofit MTP is a viable last-resort drafter when no smaller family member exists, but EAGLE-style heads usually beat it for the same training compute spent. Wait for native-MTP open-weights releases before betting your latency on this approach.

### 10. Drafter degradation under concept drift

A code-completion endpoint deployed in early 2025 had its drafter (a 1B distilled student of the production target) trained on a snapshot of the company's codebase as of 2024. Six months in, α had drifted from 0.84 to 0.71. The codebase had moved on — new internal libraries, new naming conventions, new function signatures — and the drafter's distribution lagged. Telemetry caught it because the per-class α dashboard had a clear monotonic decline for the "internal-library" prompt class.

The fix was to add the drafter to the regular model retraining cadence: every two months, regenerate distillation data from the latest target on the latest codebase, and refresh the drafter weights. The α drift effectively stopped — modest periodic retrains held α at 0.82 ± 0.02. The cost was nontrivial (one weekly small-scale GPU job) but the inference savings far exceeded it.

The lesson: **drafters drift, especially when the target's environment drifts**. A drafter is not a model artifact you train once and serve forever; it is a continuously-aligned auxiliary model with its own MLOps lifecycle. Build the retraining pipeline before you ship the first drafter, not after the first regression.

### 11. The hidden cost of long γ on tail latency

A team optimizing for throughput pushed γ from 5 to 8 because their α was high (~0.88) and the math suggested γ = 8 was optimal. Mean tokens/sec went up 9%. p99 latency went up 23%. The mean improved; the tail regressed.

The diagnosis: at γ = 8, the verify pass costs more wall-clock per call (more tokens through the target). When the rare low-α request hits the system (α ≈ 0.5 instead of 0.88), it now does 8-token verifies that mostly reject, instead of 5-token verifies that mostly reject — the per-step cost rose without the per-step benefit. The mean is dominated by the high-α common case; the tail is dominated by the low-α uncommon case.

The fix was per-sequence γ that adapts to rolling α: high-α sequences run at γ = 8, low-α sequences fall back to γ = 4 or chain. p99 dropped back below baseline-spec; mean stayed at the higher level.

The lesson: **mean and tail metrics under spec decode optimize at different points**. If you have an SLA on p99, tune γ on p99, not mean.

## A common pitfalls checklist

After eight case studies, the pattern of mistakes is clearer than any one of them. Before deploying speculative decoding, walk this checklist; each item is a real bug we have caught in production.

**Pre-deployment checks:**

- [ ] Drafter and target tokenizers are *exactly* the same (vocabulary, BPE merges, special tokens). A single mismatched special token can cap α at 0.5.
- [ ] Drafter and target are evaluated at the *same temperature*. If you serve mixed temperatures, plan for temperature-conditional spec.
- [ ] Drafter has been trained or distilled against the *production target weights*, not the base model.
- [ ] Sampling parameters (top-k, top-p, min-p, repetition penalty) are applied to both drafter and target, or only to the target — never asymmetrically in a way that affects $q$ or $p$ at the comparison point.
- [ ] Tree-spec parameters $(W, D)$ are scaled with context length, not held constant.
- [ ] At least one canary endpoint runs without spec for ground-truth comparison.

**Runtime checks (hooks for your observability stack):**

- [ ] Per-sequence rolling α is exported as a histogram by `(model, temperature, prompt_class)`.
- [ ] Verify pass wall-clock time is exported by `(model, batch_bucket)`.
- [ ] Drafter wall-clock time is exported separately so $c$ is observable.
- [ ] Per-batch-size enable/disable rule is in place; spec is disabled above the empirically-measured cliff.
- [ ] Per-sequence opt-out is in place for sequences with rolling α below threshold.

**Post-incident checks (for after the cliff bites you):**

- [ ] Distributional equivalence test (T=1, chi-squared on histogram) has been run since the last spec-decode change.
- [ ] Downstream eval (MMLU / HumanEval / your task eval) has been run with spec on vs off, agreement within ±0.5 points.
- [ ] No KV alignment bug between drafter and target (smoke test: T=0 with target = drafter, argmax sequences match for 50 tokens).

If you can check every box, you have a production-ready speculative decoding deployment. If you can check half of them, you have what most teams ship.

## Reach for speculative decoding when…

A senior engineer's checklist. If you can answer "yes" to most of these, deploy spec decode and tune the basin we walked through above.

- **Your target endpoint runs at low-to-moderate concurrency.** Batch ≤ 16 for 70B-class models on H100; batch ≤ 64 for 7–13B models. Above this, decoding is already compute-bound and spec adds work without buying you anything.
- **Your latency p99 matters more than your throughput.** Spec is a latency optimization first and a throughput optimization second. Interactive chat, code completion, voice agents, anything with a human in the loop — these are where spec earns its keep.
- **You have a drafter with α ≥ 0.7 on your real workload.** Measure on a representative sample of production prompts, not a benchmark. If α is below 0.7, spend the engineering budget on a better drafter (EAGLE / Medusa / MTP) before tuning γ.
- **Your workload has high token reuse.** Code completion, RAG, agents, summarization — anything where the output recapitulates input tokens — gets free α from n-gram drafters.
- **You can disable spec per-batch and per-sequence at runtime.** This is not optional in production. Static configurations leave performance on the floor in both directions.

## Skip speculative decoding when…

- **You serve at high concurrency with no headroom on the compute roof.** The cliff is real. Measure first.
- **Your acceptance rate is structurally below 0.5.** Strong RLHF, wide temperature, multilingual mix, or domain-mismatched drafters all push α down. Below 0.5, spec is rarely worth the engineering cost.
- **Your latency budget is dominated by network or tokenizer overhead.** A 200 ms TLS handshake makes a 30 ms/token to 12 ms/token speedup invisible. Profile the whole request path before optimizing decoding.
- **You cannot retrain coupled drafters when the target moves.** Medusa, EAGLE, MTP all need retraining after fine-tuning, RLHF, or distillation rounds. If your model lifecycle does not budget this, use a decoupled drafter or skip spec entirely.
- **You are debugging quality regressions.** Spec decode preserves the target distribution exactly, but only if every implementation detail is correct. If you are chasing a quality issue, turn spec off as the first sanity check.

> The honest truth about speculative decoding in 2026: it is a 1.5–3× latency win for interactive workloads, gated on getting α right and on staying out of the compute-bound regime. Everything else is detail.

## A worked example: end-to-end speedup on a real workload

Tying everything in this article together with a single concrete deployment example. The setup: an interactive chat endpoint serving Llama-3.1-70B-Instruct on an 8× H100 SXM5 node, average concurrency around 12, p99 concurrency around 28, average prompt length 1,800 tokens, average output length 350 tokens, temperature 0.7.

Step 1 — measure the baseline. No spec decode. p50 latency per token is 28 ms. p99 latency per token is 41 ms. Tokens/sec averaged over the full SLA window is 880. We want to halve p99.

Step 2 — pick a drafter. Llama-3.2-1B-Instruct is the obvious independent choice. We benchmark it on 1,000 representative prompts and measure α = 0.71 — usable but not great. We then train an EAGLE-2 head over Llama-3.1-70B-Instruct using 80M tokens of self-generated continuations; α rises to 0.86 with a $c$ of 0.04. The training cost was 18 hours on the same 8× H100 node.

Step 3 — pick γ. From the formula and our α = 0.86, γ\* ≈ 6. We benchmark γ ∈ {4, 5, 6, 7, 8} and the empirical peak at our typical concurrency is γ = 5 (the verify pass is starting to creep into the compute-bound regime at γ = 7). We deploy γ = 5 for the bulk of traffic and γ = 7 for batch-1 traffic from a low-traffic premium endpoint.

Step 4 — set up disable rules. Per-batch: disable spec when batch ≥ 24 (empirically the cliff). Per-sequence: disable spec when rolling α drops below 0.5 (catches JSON / tool-call sequences with weird tokenization).

Result: p50 latency drops from 28 ms to 11 ms (2.5× speedup, slightly above the closed-form prediction because of n-gram-style boilerplate gains). p99 latency drops from 41 ms to 19 ms (2.2× speedup). Tokens/sec climbs to 1,920 averaged over the SLA window. Engineering cost: 4 weeks (training pipeline, observability, integration tests, canary roll-out). Ongoing cost: one biweekly EAGLE retrain to keep α stable. Win: real, durable, paid for in a quarter of operational savings.

## What's next: the frontier of speculative decoding

The field is moving in three directions, and they will reshape the operating points described above.

**Direction 1: training-time speculation.** DeepSeek-V3's MTP heads are the first widely-deployed example, but they will not be the last. Once a pretraining recipe explicitly produces multi-token-prediction-capable weights, the inference-time drafter is essentially free, and the engineering cost of speculation collapses to "pass a flag at serve time". Expect every major open-weights release in 2026–2027 to ship MTP heads or equivalent.

**Direction 2: drafter-aware quantization and decoding.** As we saw in the quantization section, FP8 KV + tree spec compounds errors. New techniques (per-channel FP8, Hadamard-rotated quantization, drafter-conditioned activation rescaling) reduce the compounding. The end state is a serving stack where the drafter, the verifier, the KV cache precision, and the rejection sampling rule are co-designed rather than independent layers. The vLLM v2 architecture is starting to look like this; SGLang's RadixAttention is going further.

**Direction 3: speculative reasoning.** Chain-of-thought reasoning produces long, mostly autoregressive outputs that are exactly the workload spec decode helps the most. But reasoning chains have structure — premises, intermediate conclusions, retractions — and uniform γ wastes opportunities. Research papers in 2025 (Speculative Streaming, BiLD-CoT, ReST-MCTS) propose context-conditional γ that grows during expected-easy passages and shrinks during expected-hard ones. The α gains are 5–10 points, which compounds with the latency-shape gains for reasoning workloads.

A speculation worth indulging: the long-run end-state for inference acceleration is not "speculative decoding", "kv-cache reuse", and "quantization" as separate techniques. It is a single co-designed inference stack where the model architecture, the sampling rule, the cache layout, and the precision of every tensor are all optimized jointly for the bandwidth-bound regime that single-token decoding lives in. Speculative decoding will not be a separate feature in 2030; it will be how transformers are run.

## Further reading

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2023) — the original paper, still the cleanest derivation of modified rejection sampling.
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024).
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (2024).
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) — see section on multi-token prediction.
- vLLM speculative decoding docs and blog posts on the v2 block manager.
- Sibling deep-dives on this blog: [KV cache management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management), [optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide), [vLLM internals](/blog/machine-learning/large-language-model/vllm-inference), [SGLang internals](/blog/machine-learning/large-language-model/sglang-inference), and [distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) for drafter training.
