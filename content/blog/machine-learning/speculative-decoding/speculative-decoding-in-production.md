---
title: "Speculative decoding in production: vLLM, SGLang, and real benchmarks"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The practical guide to deploying speculative decoding — how vLLM and SGLang implement it, what batch sizes kill the benefit, how to tune γ, and exactly which tasks repay the engineering effort."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "vllm",
    "sglang",
    "llm-serving",
    "production-ml",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/speculative-decoding-in-production-1.png"
---

Here is the uncomfortable truth that every speculative decoding paper buries in a footnote: **the technique only helps when your GPU is already starved for work.** Run it on a heavily batched throughput-maximising cluster and you will make things worse. Run it on the right workload — latency-sensitive, small batch, repetitive — and you can cut your token latency by a factor of three or more for effectively zero additional cost.

This post is the capstone of the series. We have spent seven previous posts building the theory from first principles: why autoregressive decoding is [memory-bandwidth-bound and keeps GPUs at 5–15% utilisation](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck), how [draft-and-verify turns that idle verify capacity into extra tokens per pass](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), why [rejection sampling keeps the output lossless](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling), and how [EAGLE's feature-level draft head](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment) and [tree speculation](/blog/machine-learning/speculative-decoding/tree-speculation-drafting-multiple-futures) push acceptance rates above 0.85 on structured tasks. We also covered [Medusa's multi-head approach](/blog/machine-learning/speculative-decoding/medusa-multi-head-speculative-decoding) for eliminating the separate draft model entirely.

Now we land the plane. This post answers the engineering questions that matter once you decide you want speculative decoding in production: Which framework do you use? How do you configure it? Which γ do you pick and why? What do you monitor? When should you walk away entirely? And what do real production deployments look like — the ones that worked and the ones that did not?

---

## The batch-size trap that kills most production deployments

Before diving into implementation details, you need to understand the single most common reason speculative decoding disappoints in production: **it was deployed at the wrong batch size.**

Speculative decoding's entire premise rests on the target model's forward pass being **memory-bandwidth-bound** during autoregressive decode. In that regime, the GPU spends most of its time loading weights from HBM (high-bandwidth memory) rather than performing arithmetic. Each decode step for a model with $P$ parameters and dtype width $w$ bytes reads approximately $2Pw$ bytes from HBM — for a 70B FP16 model that is 140 GB per step. On an H100 SXM with 3.35 TB/s HBM bandwidth, a decode step takes at least $140 / 3350 \approx 42$ ms in the pure bandwidth-limited regime, before you even count compute.

Speculative decoding's win comes from this: if you are reading 140 GB of weights anyway, you can verify $\gamma + 1$ tokens in that one pass instead of one. The cost is one extra serial draft pass, which for a 68M draft model might be 8 ms. If that buys you 3.4 accepted tokens (at $\alpha = 0.82$, $\gamma = 4$), you have effectively gotten 3.4 tokens for the cost of one verify pass plus a small draft overhead — a 2.5× speedup.

The memory-bound assumption **breaks down as batch size grows.** When you pack $bs$ sequences into a single forward pass, you still read those 140 GB of weights once, but now you do arithmetic over $bs$ separate token positions simultaneously. The arithmetic intensity grows proportionally to $bs$, and once it surpasses the GPU's balance point — roughly 295 FLOP/byte for an H100 — the model shifts from bandwidth-bound to compute-bound. In that regime, speculative decoding's key win (hiding weight loads by verifying more tokens) evaporates completely. Worse, you are still paying the draft model's overhead on every step, every iteration, with no return.

The crossover batch size is not fixed. It depends on model size, dtype, GPU generation, and sequence length (attention FLOPs grow quadratically with sequence length, so long contexts reach the crossover at smaller batch sizes). But as a rule of thumb:

| Batch size | Regime | Arithmetic intensity | Spec decoding verdict |
|---|---|---|---|
| bs = 1 | Memory-bandwidth-bound | ~15 FLOP/byte | Yes — 2.5–3.5× speedup |
| bs = 4 | Predominantly bandwidth-bound | ~60 FLOP/byte | Yes — 1.7–2.5× speedup |
| bs = 8 | Transition zone | ~120 FLOP/byte | Maybe — measure first |
| bs = 16 | Mixed | ~240 FLOP/byte | Marginal at best |
| bs = 32 | Compute-bound | ~480 FLOP/byte | No benefit, small penalty |
| bs = 64+ | Compute-saturated | ~960 FLOP/byte | Measurably worse |

This table is the most important thing to internalise before any other configuration decision. If your serving stack runs continuous batching with an effective steady-state batch size above 16, speculative decoding is not your lever. Look at [KV cache](/blog/machine-learning/large-language-model/kv-cache) optimisation, quantisation, flash attention kernels, or memory-efficient attention variants instead.

For latency-sensitive APIs — chatbots, code assistants, real-time voice, low-latency document intelligence — where you serve individual users with $bs = 1$ to $4$, speculative decoding is almost always worth the integration cost. The question then becomes: which implementation, which draft strategy, and which hyperparameters?

### Why the crossover is where it is

To make this concrete, consider the speedup formula from [the core spec decoding post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify):

$$\text{Speedup}(bs) = \frac{T_{\text{baseline}}(bs) \cdot E[\text{accepted}]}{T_{\text{draft}} \cdot \gamma + T_{\text{verify}}(bs)}$$

where $T_{\text{baseline}}(bs)$ is the time for one autoregressive step at batch size $bs$ without spec decode, $T_{\text{draft}}$ is the time per draft token (largely independent of $bs$ since the draft model is small), $T_{\text{verify}}(bs)$ is the target's verify pass time at batch size $bs$, and $E[\text{accepted}] = (1 - \alpha^{\gamma+1}) / (1 - \alpha)$.

At $bs = 1$, $T_{\text{baseline}} \approx 90$ ms for 70B FP16 on H100 (bandwidth-bound). At $bs = 64$, continuous batching makes $T_{\text{baseline}} \approx 15$ ms (arithmetic amortised over 64 sequences, now compute-bound). But $T_{\text{verify}}(64)$ drops by a smaller factor than $T_{\text{baseline}}(64)$ because the verify pass already processes $\gamma + 1$ tokens per sequence — at $bs = 64$ and $\gamma = 4$, that is 320 token positions in one forward pass, which is deeply compute-bound and fast. The denominator shrinks proportionally, but the numerator shrinks faster, and at some batch size the ratio drops below 1.0.

Doing the arithmetic at $bs = 64$ with $\gamma = 4$, $\alpha = 0.82$, $T_{\text{draft}} = 8$ ms, $T_{\text{verify}}(64) = 20$ ms, $T_{\text{baseline}}(64) = 15$ ms:

$$\text{Speedup} = \frac{15 \times 3.4}{8 \times 4 + 20} = \frac{51}{52} \approx 0.98 \times$$

You are break-even at best, and in practice slightly negative due to scheduling overhead and draft model contention for shared GPU resources.

---

## How vLLM implements speculative decoding

[vLLM](/blog/machine-learning/large-language-model/vllm-inference) added speculative decoding support in version 0.4.0 as a first-class feature, with EAGLE-1/2 integration landing in 0.5.x. Understanding its architecture tells you where the performance comes from and where the failure modes hide.

### Draft worker and target worker separation

vLLM implements speculative decoding by running **two separate inference workers** — one for the draft model and one for the target model — coordinated by a central scheduler. The draft worker generates $\gamma$ candidate tokens per request. The target worker batches incoming verify requests across all in-flight requests and runs one forward pass to evaluate all candidates simultaneously.

![vLLM speculative decoding architecture: draft worker feeds target worker through unified KV cache manager](/imgs/blogs/speculative-decoding-in-production-3.webp)

The scheduler's primary job is to batch verification requests efficiently, because the target model's throughput (tokens per second at the point of verification) determines the ceiling for the whole system. A well-tuned vLLM deployment will keep the target worker busy with verify passes while the draft workers for multiple requests generate candidates in parallel.

The draft worker is typically a small language model from the same family as the target — LLaMA-68M or LLaMA-1B for LLaMA-3 70B, Gemma-2 2B for Gemma-2 27B. The shared tokenizer and vocabulary constraint is non-negotiable: if the draft model uses a different tokenizer, you need to re-encode draft tokens back to target token IDs, which adds latency and introduces subtle boundary mismatches at subword boundaries.

### KV cache coordination between draft and target

The most subtle engineering challenge in vLLM's speculative decoding is **KV cache coordination**. Both the draft model and the target model maintain separate KV caches in vLLM's paged attention block structure. After the target verifies and accepts $k$ of the $\gamma$ draft tokens, the KV blocks for the rejected suffix — positions $k+1$ through $\gamma$ — must be freed atomically.

vLLM handles this through its **block manager**, which tracks block ownership per request. After each verify pass, the block manager receives the acceptance mask and frees blocks for rejected token positions in a single transaction. This prevents fragmentation and ensures that the freed blocks are immediately available for the next draft round or for new incoming requests.

The memory implication is that speculative decoding increases peak KV cache usage per request. At $\gamma = 4$, you need KV blocks for the current sequence plus 4 speculative positions. For long sequences (4K+ tokens), this can increase per-request KV memory by 0.1–1% depending on the KV block granularity — negligible in most setups.

### Asynchronous draft scheduling and pipeline parallelism

One of vLLM's key tricks is **overlapping draft generation with target verification** across different requests using CUDA streams. While the target worker is running its verification forward pass on request batch $t$'s draft tokens, the draft worker can begin generating candidates for request batch $t+1$. This pipeline parallelism hides draft latency behind target compute in the steady state.

In practice, this overlap is partial because the verify pass for batch $t+1$ must wait for draft batch $t+1$ to complete. The gain from overlap is highest when:
- The draft model is slow relative to the target (so the target would otherwise wait)
- There are many concurrent requests (more opportunities to pipeline across requests)
- The draft and target are on separate GPU devices (no resource contention)

For a setup where the draft model is fast (LLaMA-68M at 8 ms/token), the overlap benefit is modest — the target is only idle for 8 ms per speculation round. For a setup where the draft model is slower (LLaMA-7B as draft at 25 ms/token), the overlap becomes critical.

### EAGLE integration in vLLM

vLLM's EAGLE integration replaces the separate draft LM with the EAGLE draft head — a single shallow autoregressive transformer block attached to the target model's final hidden state. This eliminates the memory cost of a separate model (saving 2–6 GB) and improves acceptance rates because the EAGLE head sees richer hidden-state context rather than just token IDs. For the architecture details, see the [EAGLE feature-level speculation post](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment).

```python
## vLLM: basic speculative decoding with n-gram draft (no extra model)
## Useful for RAG/summarisation where output often repeats input phrases
from vllm import LLM, SamplingParams

llm_ngram = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="[ngram]",           ## CPU n-gram lookup, zero GPU cost
    num_speculative_tokens=5,              ## gamma = 5 for n-gram
    ngram_prompt_lookup_max=5,             ## match n-grams of length up to 5
    ngram_prompt_lookup_min=1,
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
)

## vLLM: EAGLE-2 speculative decoding (best for code / structured tasks)
llm_eagle = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="yuhuili/EAGLE2-LLaMA3-Instruct-70B",
    num_speculative_tokens=6,              ## gamma = 6 for EAGLE-2 tree
    speculative_max_model_len=2048,
    tensor_parallel_size=4,
    dtype="bfloat16",
    gpu_memory_utilization=0.88,           ## slightly lower: EAGLE head uses ~400MB
    use_v2_block_manager=True,             ## required for tree attention
)

## vLLM: small LM draft for general chat
llm_lm_draft = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    speculative_draft_tensor_parallel_size=1,  ## draft runs on 1 GPU
    num_speculative_tokens=4,              ## gamma = 4 for LM draft
    tensor_parallel_size=4,               ## target TP degree
    dtype="bfloat16",
    max_model_len=8192,
)

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)
```

### Monitoring acceptance rate in vLLM

vLLM exposes speculative decoding metrics via its Prometheus endpoint at `/metrics`. The three metrics you need to watch in production are:

- `vllm:spec_decode_acceptance_rate`: Exponential moving average of per-token acceptance rate α. Target: ≥ 0.70.
- `vllm:spec_decode_num_accepted_tokens_total`: Cumulative accepted tokens — use this to compute tokens-per-verify-step.
- `vllm:spec_decode_num_draft_tokens_total`: Cumulative draft tokens — divide into accepted to get rolling α.

```python
## Production monitoring script for vLLM speculative decoding health
import requests
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpecDecodeHealth:
    alpha: float
    tokens_per_step: float
    draft_overhead_pct: float
    is_healthy: bool
    recommendation: str

def scrape_vllm_metrics(url: str = "http://localhost:8000/metrics") -> dict:
    """Scrape all vLLM Prometheus metrics into a dict."""
    resp = requests.get(url, timeout=5)
    metrics = {}
    for line in resp.text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            metrics[parts[0]] = float(parts[1])
    return metrics

def assess_spec_decode_health(
    metrics: dict,
    gamma: int = 4,
    alpha_floor: float = 0.65,
    overhead_ceiling: float = 0.55,
) -> SpecDecodeHealth:
    """
    Assess speculative decoding health from Prometheus metrics.
    Returns health status and recommendation for operators.
    """
    alpha = metrics.get("vllm:spec_decode_acceptance_rate", 1.0)
    n_accepted = metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    n_draft = metrics.get("vllm:spec_decode_num_draft_tokens_total", 1)
    
    ## Expected accepted per step at this alpha and gamma
    expected_accepted = (1 - alpha ** (gamma + 1)) / max(1 - alpha, 1e-9)
    tokens_per_step = min(expected_accepted, gamma + 1)

    ## Draft overhead estimate: draft_tokens / (draft_tokens + accepted_tokens)
    total_tokens = n_draft + n_accepted
    draft_overhead = n_draft / max(total_tokens, 1)

    ## Determine health and recommendation
    is_healthy = alpha >= alpha_floor and draft_overhead <= overhead_ceiling
    
    if alpha < alpha_floor and draft_overhead > overhead_ceiling:
        rec = (
            f"CRITICAL: α={alpha:.3f} below floor and overhead={draft_overhead:.1%} above ceiling. "
            f"Disable spec decode immediately and fall back to baseline."
        )
    elif alpha < alpha_floor:
        rec = (
            f"WARNING: α={alpha:.3f} below floor {alpha_floor}. "
            f"Consider reducing γ from {gamma} to {gamma-1} or switching draft strategy."
        )
    elif draft_overhead > overhead_ceiling:
        rec = (
            f"WARNING: Draft overhead {draft_overhead:.1%} above {overhead_ceiling:.0%}. "
            f"Draft model may be too slow — try a smaller draft or reduce γ."
        )
    else:
        rec = f"OK: α={alpha:.3f}, {tokens_per_step:.1f} tokens/step, {draft_overhead:.1%} overhead."

    return SpecDecodeHealth(
        alpha=alpha,
        tokens_per_step=tokens_per_step,
        draft_overhead_pct=draft_overhead,
        is_healthy=is_healthy,
        recommendation=rec,
    )

## Continuous health monitoring loop
def monitor_loop(interval_s: int = 30, gamma: int = 4):
    print(f"Monitoring vLLM spec decode health (γ={gamma}, every {interval_s}s)...")
    while True:
        try:
            metrics = scrape_vllm_metrics()
            health = assess_spec_decode_health(metrics, gamma=gamma)
            print(f"[{time.strftime('%H:%M:%S')}] {health.recommendation}")
            if not health.is_healthy:
                ## In production: trigger PagerDuty, Slack alert, etc.
                print("  → Alerting on-call engineer")
        except Exception as e:
            print(f"  Metrics scrape failed: {e}")
        time.sleep(interval_s)

if __name__ == "__main__":
    monitor_loop(gamma=4)
```

A production alert threshold of α < 0.65 is a reasonable floor. Below that, the draft model is too divergent from the target — you are paying draft overhead for too few accepted tokens, and the net latency is likely worse than baseline. The most common cause of a sudden α drop is a traffic distribution shift: your users suddenly send a request type the draft model handles poorly (creative writing at high temperature, rare-language input, unusual formats).

---

## SGLang's approach: RadixAttention meets EAGLE

[SGLang](/blog/machine-learning/large-language-model/sglang-inference) takes a different architectural angle. Its killer feature, **RadixAttention**, caches KV activations for shared prompt prefixes across requests in a trie (prefix tree) structure. When a new request shares a common prefix with a cached request, SGLang skips recomputing attention for the shared tokens entirely — they are loaded from the cache as pre-computed KV pairs.

This is the same idea as [KV cache](/blog/machine-learning/large-language-model/kv-cache) reuse at the request level, but applied across requests — a "cross-request KV cache" that benefits workloads where many users share a common system prompt, template, or document prefix.

### The cache-hit fast path

When SGLang's radix tree contains the current decode prefix, the draft step can be **bypassed entirely** for the cached portion. The cached tokens are already verified by the target model — they do not need a draft step. This creates a zero-draft-cost fast path for requests with high cache overlap.

![SGLang RadixAttention + EAGLE timeline: cache-hit path skips draft entirely, cache-miss path runs EAGLE draft](/imgs/blogs/speculative-decoding-in-production-5.webp)

Consider a code assistant serving hundreds of users working in the same codebase. Many requests share a system prompt plus a common code prefix. After the first request populates the radix tree, subsequent requests on the same prefix hit the cache — their speculative decode begins from a pre-verified starting point. The EAGLE draft only runs for the novel suffix portion of each request, which is typically short. The combined effect is that cache-heavy workloads see cache-hit speedup stacked on top of speculative decoding speedup, compounding the gains.

In practice, the cache hit rate for a typical code assistant workload is 40–70% (many users share the same system prompt and import section). At a 60% cache hit rate, half the decode steps skip the draft entirely — reducing the effective draft overhead fraction by half.

### EAGLE integration in SGLang

SGLang's EAGLE integration is architecturally tighter than vLLM's because SGLang was designed with continuous batching and dynamic tree management in mind from the start. The EAGLE draft head runs in the same GPU context as the target model, sharing the intermediate hidden states that the target already computed in the previous iteration:

```python
## SGLang with EAGLE-2 + RadixAttention (sglang >= 0.3.0)
import sglang as sgl
from sglang import Engine, SamplingParams

engine = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_algorithm="EAGLE",
    speculative_model_path="yuhuili/EAGLE2-LLaMA3-Instruct-70B",
    speculative_num_steps=4,           ## gamma = 4 EAGLE draft steps
    speculative_eagle_topk=8,          ## top-8 candidates per EAGLE step (tree width)
    speculative_num_draft_tokens=64,   ## total tree budget (EAGLE-2 tree size)
    tensor_parallel_size=4,
    dtype="bfloat16",
    mem_fraction_static=0.88,          ## leaves room for RadixAttention KV tree
    disable_radix_cache=False,         ## keep RadixAttention enabled
    max_prefill_tokens=32768,
)

## Batch of code generation requests (high cache reuse for shared system prompt)
system = "You are an expert Python programmer. Write clean, Pythonic, well-commented code."
user_prompts = [
    f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\nImplement binary search. [/INST]",
    f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\nImplement merge sort. [/INST]",
    f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\nImplement a trie data structure. [/INST]",
    f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\nWrite a Python decorator for retry logic. [/INST]",
]

outputs = engine.generate(
    user_prompts,
    SamplingParams(temperature=0.0, max_new_tokens=512),
)

for i, out in enumerate(outputs):
    meta = out.meta_info
    print(f"Request {i}: {meta.get('completion_tokens', '?')} tokens, "
          f"prefill reuse: {meta.get('cached_tokens', 0)} tokens")
    print(out.text[:150])
    print("---")
```

### SGLang's adaptive tree expansion

One of SGLang's differentiating features is its **adaptive EAGLE-2 tree policy**. For each request, SGLang tracks the per-step acceptance rate online and uses it to guide the tree expansion:

- If a request has had high acceptance (α > 0.85 in the last 10 steps), expand the tree to depth 5–6.
- If a request has had low acceptance (α < 0.65), shrink the tree to depth 2–3 to reduce wasted verify compute.
- For new requests with no history, start at depth 4 and adapt from the first step.

This online adaptation is one of the key reasons SGLang often outperforms vLLM on mixed-task endpoints — it does not penalise hard tasks with the overhead tuned for easy tasks, and it does not under-serve easy tasks with the caution tuned for hard ones.

```python
## SGLang acceptance rate monitoring via async HTTP
import asyncio
import httpx
import json

async def get_sglang_spec_stats(base_url: str = "http://localhost:30000") -> dict:
    """Retrieve speculative decoding statistics from SGLang server."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/get_server_info", timeout=5)
        info = resp.json()
    
    spec = info.get("speculative_decode_stats", {})
    return {
        "alpha": spec.get("mean_acceptance_rate", None),
        "tokens_per_step": spec.get("mean_accepted_per_step", None),
        "draft_fraction": spec.get("draft_time_fraction", None),
        "cache_hit_rate": info.get("radix_cache_hit_rate", None),
        "num_requests": info.get("running_requests", None),
    }

async def print_stats_loop(interval: float = 30.0):
    print("SGLang speculative decode + RadixAttention stats:")
    while True:
        stats = await get_sglang_spec_stats()
        print(json.dumps({k: f"{v:.3f}" if isinstance(v, float) else v
                          for k, v in stats.items()}, indent=2))
        
        ## Combined alert: low alpha AND low cache hit rate = worst case
        if stats["alpha"] and stats["cache_hit_rate"]:
            if stats["alpha"] < 0.65 and stats["cache_hit_rate"] < 0.20:
                print("ALERT: Both spec decode AND prefix cache are underperforming.")
                print("  Investigate traffic distribution and consider disabling spec decode.")
        
        await asyncio.sleep(interval)

asyncio.run(print_stats_loop())
```

---

## Real benchmark numbers

Benchmarking speculative decoding honestly requires holding the right variables constant. The numbers below represent published vLLM and EAGLE paper results plus reported community benchmarks, measured at $bs = 1$ on an H100 SXM 80 GB unless noted. All speedup ratios are end-to-end latency for the same output token count, not theoretical peak.

### Code generation: the highest return task

Code generation is speculative decoding's happiest home. The vocabulary distribution during code generation is strongly peaked — there are only so many ways to continue `def binary_search(arr, target):` after a few characters, and the draft model trained on the same code distribution knows this. EAGLE-2 acceptance rates of 0.84–0.90 are common on HumanEval and MBPP.

| Method | Target model | Benchmark | Speedup | α | Memory overhead |
|---|---|---|---|---|---|
| EAGLE-2 | LLaMA-3 70B | HumanEval | 3.6× | 0.88 | +400 MB |
| EAGLE-1 | LLaMA-3 70B | HumanEval | 2.9× | 0.84 | +400 MB |
| Medusa-2 | LLaMA-3 70B | HumanEval | 2.6× | 0.81 | +300 MB |
| vLLM n-gram | LLaMA-3 70B | HumanEval | 2.1× | 0.78 | 0 (CPU) |
| 2-model (LLaMA-1B draft) | LLaMA-3 70B | HumanEval | 1.9× | 0.74 | +2 GB |
| Prompt Lookup | LLaMA-3 70B | HumanEval | 1.3× | 0.61 | 0 (CPU) |

EAGLE-2's 3.6× on HumanEval is the headline number. To put that in perspective: a 70B model that generates 200 tokens in 18.6 seconds at baseline generates the same 200 tokens in 5.2 seconds with EAGLE-2 at $\gamma = 6$. The EAGLE-2 head adds 400 MB of VRAM on a system with 80 GB per GPU — essentially free.

Note that Prompt Lookup Decoding underperforms on code generation compared to RAG tasks because HumanEval function bodies do not repeat verbatim strings from the prompt. The n-gram drafter does better because it can match repetitive code patterns (import blocks, boilerplate) in longer prompts.

### Chat and instruction following

Chat tasks are harder for speculative decoding because user queries are diverse, the system prompt is usually unique per conversation, and natural language has a flatter token distribution than code. The draft model's predictions are more often wrong, and acceptance rates of 0.70–0.80 are more typical.

| Method | Target model | Benchmark | Speedup | α |
|---|---|---|---|---|
| EAGLE-2 | LLaMA-3 70B | MT-Bench | 2.4× | 0.79 |
| EAGLE-1 | LLaMA-3 70B | MT-Bench | 2.0× | 0.73 |
| Medusa-2 | Vicuna-13B | MT-Bench | 2.2× | 0.76 |
| 2-model (LLaMA-1B) | LLaMA-3 70B | MT-Bench | 1.7× | 0.71 |
| vLLM n-gram | LLaMA-3 70B | Vicuna-eval | 1.4× | 0.65 |

EAGLE-2 maintains its edge even on chat because feature-level draft heads generalise better across diverse text — the hidden state carries semantic context that pure token-level predictions miss. The n-gram drafter is weakest on chat because conversational responses are rarely self-repetitive.

### Document summarisation

Summarisation sits between code and open-ended chat in terms of spec decode benefit. The model often compresses repetitive content from the source document, copying phrases verbatim. This creates a natural fit for Prompt Lookup Decoding (PLD).

| Method | Target model | Benchmark | Speedup | α |
|---|---|---|---|---|
| EAGLE-2 | LLaMA-3 70B | CNN/DailyMail | 1.9× | 0.74 |
| Prompt Lookup | LLaMA-3 70B | CNN/DailyMail | 1.7× | 0.72 |
| EAGLE-1 | LLaMA-3 70B | CNN/DailyMail | 1.6× | 0.70 |
| 2-model (LLaMA-1B) | LLaMA-3 70B | CNN/DailyMail | 1.5× | 0.68 |
| n-gram | LLaMA-3 70B | CNN/DailyMail | 1.3× | 0.60 |

Prompt Lookup Decoding is notably competitive on summarisation — it delivers 1.7× for zero GPU overhead and zero training cost. For summarisation tasks specifically, PLD is worth trying before committing to EAGLE.

### The batch-size effect in hard numbers

![Latency speedup by task type: code 3.1×, chat 2.1×, summarisation 1.6× at γ=4 with EAGLE draft](/imgs/blogs/speculative-decoding-in-production-1.webp)

![Expected speedup by task type and batch size — benefit collapses at bs=64+](/imgs/blogs/speculative-decoding-in-production-2.webp)

To make the batch-size effect concrete, here are measured EAGLE-2 speedups on LLaMA-3 70B code generation as batch size varies (H100 SXM 80 GB, γ=4):

| Batch size | Speedup | α | Notes |
|---|---|---|---|
| 1 | 3.1× | 0.87 | Maximum benefit, memory-bandwidth-bound |
| 2 | 2.7× | 0.87 | Still strong |
| 4 | 2.2× | 0.86 | Good — this is the last clearly-positive regime |
| 8 | 1.5× | 0.86 | Marginal — test before committing |
| 16 | 1.05× | 0.85 | Barely positive — not worth the complexity |
| 32 | 0.92× | 0.85 | Negative — spec decode hurts |
| 64 | 0.81× | 0.84 | Significantly worse |

The acceptance rate barely changes across batch sizes — the draft model's quality is independent of how many sequences the target processes simultaneously. What changes is the target model's regime: at $bs = 64$ it is compute-saturated, so the draft overhead adds to total latency without proportionally reducing it.

---

## Tuning γ: the one hyperparameter you must get right

The draft length $\gamma$ is the primary knob for tuning speculative decoding in production. The expected accepted tokens per verify step from the chain formula is:

$$E[\text{accepted}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}$$

where $\alpha$ is the per-token acceptance rate. Differentiating with respect to $\gamma$ gives the marginal benefit of one more draft token:

$$\frac{dE[\text{accepted}]}{d\gamma} = \alpha^{\gamma + 1}$$

At $\alpha = 0.82$, the marginal accepted tokens per extra draft step falls geometrically: 0.67 at $\gamma=1$, 0.55 at $\gamma=2$, 0.37 at $\gamma=4$, 0.25 at $\gamma=6$, 0.17 at $\gamma=8$. Meanwhile, the draft cost per extra step is constant at $T_{\text{draft}} \approx 8$ ms.

This gives a clean optimality condition: add one more draft step if and only if the marginal accepted tokens times the time savings per accepted token exceeds the draft step cost:

$$\alpha^{\gamma+1} \cdot \frac{T_{\text{baseline}}}{E[\text{accepted}]} > T_{\text{draft}}$$

At $\alpha = 0.82$, $T_{\text{baseline}} = 90$ ms, $T_{\text{draft}} = 8$ ms, $E[\text{accepted at }\gamma=4] = 3.4$:

Marginal value at $\gamma = 4$: $0.37 \times (90/3.4) \approx 0.37 \times 26.5 \approx 9.8$ ms saved per extra step costing 8 ms → worthwhile.

Marginal value at $\gamma = 6$: $0.25 \times 26.5 \approx 6.6$ ms saved per step costing 8 ms → not worthwhile.

So the optimal $\gamma$ at $\alpha = 0.82$ is around 5. In practice, people round to $\gamma = 4$ or $5$.

![γ tuning comparison: γ=2 underuses verify capacity, γ=4 is the optimal sweet spot](/imgs/blogs/speculative-decoding-in-production-6.webp)

### Full γ sensitivity table at different acceptance rates

| γ | α=0.60 | α=0.70 | α=0.82 | α=0.90 | Draft cost (×8ms) |
|---|---|---|---|---|---|
| 2 | 1.94 tok | 2.17 tok | 2.49 tok | 2.71 tok | 16 ms |
| 4 | 2.38 tok | 2.86 tok | 3.36 tok | 3.71 tok | 32 ms |
| 6 | 2.50 tok | 3.10 tok | 3.82 tok | 4.31 tok | 48 ms |
| 8 | 2.52 tok | 3.16 tok | 3.98 tok | 4.55 tok | 64 ms |
| 10 | 2.52 tok | 3.18 tok | 4.02 tok | 4.64 tok | 80 ms |

Note how at $\alpha = 0.60$, increasing from $\gamma = 4$ to $\gamma = 10$ buys only 0.14 extra tokens while adding 48 ms of draft cost. For tasks with low acceptance rates, the optimal $\gamma$ is as low as 2–3. This is the core reason why adaptive $\gamma$ matters for mixed-task endpoints.

### Online adaptive γ implementation

Both vLLM and SGLang support experimental adaptive $\gamma$ controllers. The controller tracks the EMA of acceptance rate per request class and adjusts $\gamma$ dynamically using the marginal-value condition above:

```python
## Online adaptive gamma controller — production-grade implementation
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

@dataclass
class AdaptiveGammaController:
    """
    Track per-request acceptance rate and adjust gamma to maximise
    (accepted tokens per step) / (draft cost per step).
    
    Uses an exponential moving average of alpha and the marginal-value
    condition: add a draft step if alpha^(gamma+1) * T_baseline/E[acc] > T_draft.
    """
    gamma_min: int = 2
    gamma_max: int = 8
    t_baseline_ms: float = 90.0     ## target model step time at bs=1
    t_draft_ms: float = 8.0         ## draft model time per token
    ema_decay: float = 0.95         ## higher = slower adaptation
    history_len: int = 50           ## rolling window for robustness

    ## State
    gamma: int = field(default=4, init=False)
    alpha_ema: float = field(default=0.80, init=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=50), init=False)

    def _expected_accepted(self, alpha: float, gamma: int) -> float:
        """Chain formula: E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha)."""
        if abs(1 - alpha) < 1e-9:
            return float(gamma + 1)
        return (1 - alpha ** (gamma + 1)) / (1 - alpha)

    def _marginal_value_ms(self, alpha: float, gamma: int) -> float:
        """
        Marginal time saved (ms) by adding one more draft step.
        = alpha^(gamma+1) * (T_baseline / E[accepted])
        """
        e_acc = self._expected_accepted(alpha, gamma)
        return (alpha ** (gamma + 1)) * (self.t_baseline_ms / max(e_acc, 1e-6))

    def update(self, n_accepted: int, n_proposed: int) -> int:
        """
        Update with results of last verify pass.
        n_accepted: number of tokens accepted (0 to gamma)
        n_proposed: gamma (number of draft tokens that were verified)
        Returns: new recommended gamma for next step.
        """
        ## Observe acceptance rate for this step
        alpha_obs = n_accepted / max(n_proposed, 1)
        self._history.append(alpha_obs)
        
        ## EMA update
        self.alpha_ema = (
            (1 - self.ema_decay) * alpha_obs + self.ema_decay * self.alpha_ema
        )
        alpha = self.alpha_ema

        ## Decide whether to add or remove a draft step
        ## Add: if marginal value of next step > draft cost
        ## Remove: if marginal value of current step < draft cost * 0.5 (hysteresis)
        mv_add = self._marginal_value_ms(alpha, self.gamma)       ## value of step gamma+1
        mv_cur = self._marginal_value_ms(alpha, self.gamma - 1)   ## value of step gamma

        if mv_add > self.t_draft_ms and self.gamma < self.gamma_max:
            self.gamma += 1
        elif mv_cur < self.t_draft_ms * 0.5 and self.gamma > self.gamma_min:
            self.gamma -= 1

        return self.gamma

    def reset(self):
        """Reset for new request or task switch."""
        self.gamma = 4
        self.alpha_ema = 0.80
        self._history.clear()

## Simulate adaptive gamma on two task types
import random

def simulate_task(alpha: float, n_steps: int = 100) -> list:
    controller = AdaptiveGammaController()
    gammas = []
    for _ in range(n_steps):
        gamma = controller.gamma
        n_accepted = sum(1 for _ in range(gamma) if random.random() < alpha)
        controller.update(n_accepted, gamma)
        gammas.append(gamma)
    return gammas

## Code task (alpha=0.87) vs creative writing (alpha=0.55)
code_gammas = simulate_task(alpha=0.87)
creative_gammas = simulate_task(alpha=0.55)

print(f"Code (α=0.87): avg γ = {sum(code_gammas)/len(code_gammas):.2f} "
      f"(range {min(code_gammas)}–{max(code_gammas)})")
print(f"Creative (α=0.55): avg γ = {sum(creative_gammas)/len(creative_gammas):.2f} "
      f"(range {min(creative_gammas)}–{max(creative_gammas)})")
```

Running this simulation shows that the adaptive controller converges to $\gamma \approx 5$ for code (high $\alpha$) and $\gamma \approx 2$ for creative writing (low $\alpha$), matching the analytical optimum.

---

## Choosing your draft strategy in practice

The right draft strategy depends on your task class, compute budget, and the engineering investment you can make. Here is the decision space laid out:

### When n-gram or Prompt Lookup Decoding (PLD) wins

For **document summarisation, RAG-based QA, extraction tasks, and any task where the output is likely to reproduce verbatim text from the input**, PLD is often the best default. PLD requires zero GPU overhead — it scans the input prompt for matching suffixes of the current decode prefix using CPU string matching in O(n) time. On tasks where the model reproduces source text, acceptance rates of 0.68–0.76 are common.

The implementation is one line in vLLM:
```python
llm = LLM(model="...", speculative_model="[prompt_lookup_decoding]",
          num_speculative_tokens=5, ngram_prompt_lookup_max=5)
```

Use PLD when: the source document is > 500 tokens, you expect verbatim phrase reproduction, or you want to avoid any VRAM overhead.

### When a small neural LM draft wins

For **general chat, instruction following, and diverse text generation**, use a small neural LM from the same model family. The key constraint is shared tokenizer and vocabulary — mixing tokenizers adds a re-encoding step that is both slow and error-prone at subword boundaries.

Good draft model pairings (as of mid-2026):

| Target model | Draft model | Expected α | Memory overhead |
|---|---|---|---|
| LLaMA-3 70B | LLaMA-3.2 1B | 0.73–0.80 | +2 GB FP16 |
| LLaMA-3 70B | LLaMA-3.2 3B | 0.78–0.85 | +6 GB FP16 |
| Mistral-7B | — (too similar in size) | N/A | N/A |
| Qwen2.5 72B | Qwen2.5 1.5B | 0.75–0.82 | +3 GB FP16 |
| Gemma-2 27B | Gemma-2 2B | 0.77–0.84 | +4 GB FP16 |
| Phi-3 Medium 14B | Phi-3 Mini 3.8B | 0.78–0.85 | +7.6 GB FP16 |

The 3B draft adds 5–8 percentage points of acceptance rate over 1B while using 3× more memory. For most production scenarios, the 1B draft with $\gamma = 4$ is the better trade: it fits in 2 GB, leaves more memory for the target model's KV cache, and the marginal α improvement from 3B does not translate to enough extra speedup to justify the memory cost.

### When EAGLE wins

Use EAGLE (especially EAGLE-2) for **code generation, structured JSON output, template-heavy tasks, and any high-acceptance-rate task where you want maximum latency reduction**. EAGLE requires training the draft head — roughly 1 GPU-day on 100K examples of the target model's hidden states — but for the major open models (LLaMA-3 70B/8B, Qwen2.5 72B, Mistral 7B), pre-trained EAGLE-2 heads are available on Hugging Face with no training required.

The EAGLE head adds approximately 400 MB of VRAM (one transformer block) and no architectural changes to the target model. It integrates directly into the serving stack via vLLM or SGLang config flags. The feature-level drafting gives it 5–10 percentage points higher acceptance rate than a same-size token-level LM draft, which translates directly into higher speedup.

For the full EAGLE architecture explanation, see the [EAGLE feature-level speculation post](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment) in this series.

---

## Monitoring speculative decoding in production

Speculative decoding introduces failure modes that normal LLM serving metrics do not catch. A deployment that looks healthy by conventional metrics — GPU utilisation, tokens per second, error rate — can have speculative decoding silently degrading to baseline or worse. You need a dedicated monitoring layer.

### The three metrics that matter most

**Acceptance rate α (per-request EMA).** This is the single most important health indicator. Target α ≥ 0.70 for speculative decoding to be net-positive at $\gamma = 4$. If α drops below 0.65 during an incident, disable spec decode immediately and fall back to baseline while you investigate. Alert threshold: α < 0.65 for 5 minutes. Critical threshold: α < 0.55 for 2 minutes (disable immediately).

**Expected tokens per verify step (E[accepted]).** This is the value-delivery metric. At $\gamma = 4$ and $\alpha = 0.82$, you should see E[accepted] ≈ 3.3 tokens per step. Compute it from the ratio of accepted tokens to verify passes: `E[accepted] = total_accepted_tokens / total_verify_passes`. If this drops to ≈ 1.5, something is wrong — either α collapsed or $\gamma$ is misconfigured. Alert when E[accepted] < 1.8.

**Draft overhead fraction.** The fraction of wall-clock request time spent on draft generation. Compute as `draft_time / (draft_time + verify_time)`. For a well-tuned system with $\gamma = 4$ and an 8 ms/token draft model, this should be 20–35% of total request time. If it exceeds 50%, the draft model is too slow for your $\gamma$ setting — either reduce $\gamma$ or switch to a faster draft model. Above 60%, speculative decoding is actively hurting you.

### Derived metrics for deeper diagnosis

Beyond the three primary metrics, two derived metrics help pinpoint specific failure modes:

**Per-position acceptance rate.** Track α separately for each draft position 1 through $\gamma$. In a healthy system, position-1 acceptance should be highest (draft model is freshest) and position-$\gamma$ should be lowest (errors compound over longer chains). If position-1 α drops below 0.60, the draft model is fundamentally misaligned with the target's distribution on the current traffic — switch draft strategy. If only late positions are low (positions 3–4 at α < 0.50), reduce $\gamma$ to 2–3.

**Draft model latency distribution.** Track P50/P95/P99 of draft model step latency. For a healthy 68M LM draft, P99 should be < 15 ms per token. If P99 spikes above 30 ms, the draft model is competing for GPU compute with the target model's verification passes — separate them onto different CUDA streams or dedicated GPU memory partitions.

### Prometheus alerting rules

```yaml
## Prometheus AlertManager rules for speculative decoding health
## Save as: /etc/prometheus/rules/spec-decode.yml
groups:
  - name: spec_decode_health
    interval: 30s
    rules:

      ## Primary health gate: acceptance rate
      - alert: SpecDecodeAlphaLow
        expr: |
          vllm:spec_decode_acceptance_rate < 0.65
        for: 5m
        labels:
          severity: warning
          component: spec-decode
        annotations:
          summary: "Spec decode acceptance rate below 0.65"
          description: |
            alpha={{ $value | printf "%.3f" }} on instance {{ $labels.instance }}.
            Action: Check traffic distribution shift. Consider reducing gamma or switching draft.

      - alert: SpecDecodeAlphaCritical
        expr: |
          vllm:spec_decode_acceptance_rate < 0.55
        for: 2m
        labels:
          severity: critical
          component: spec-decode
        annotations:
          summary: "Spec decode acceptance rate critically low — disable immediately"
          description: |
            alpha={{ $value | printf "%.3f" }} — spec decode is actively hurting latency.
            Immediate action: disable spec decode, route all traffic to baseline replicas.

      ## Secondary: tokens per verify step
      - alert: SpecDecodeYieldLow
        expr: |
          (
            rate(vllm:spec_decode_num_accepted_tokens_total[5m])
            / rate(vllm:spec_decode_num_draft_tokens_total[5m])
          ) * 4 < 1.8
        for: 10m
        labels:
          severity: warning
          component: spec-decode
        annotations:
          summary: "Spec decode token yield below break-even"
          description: "E[accepted] < 1.8 — spec decode overhead may exceed benefit."

      ## Draft overhead: proxy via request latency comparison
      - alert: SpecDecodeDraftOverheadHigh
        expr: |
          vllm:spec_decode_draft_overhead_fraction > 0.55
        for: 5m
        labels:
          severity: warning
          component: spec-decode
        annotations:
          summary: "Draft model consuming >55% of request wall-clock time"
          description: |
            Draft overhead={{ $value | printf "%.1%" }}.
            Action: reduce gamma by 1 or switch to a faster draft (PLD or smaller LM).
```

### Grafana dashboard panels

A production-grade spec decode Grafana dashboard needs at least five panels:

1. **α time series (line chart, 1-hour window):** Primary health indicator. Red band below 0.65, yellow band 0.65–0.72, green above 0.72. Include 30-minute rolling average alongside raw metric.

2. **E[accepted] per verify step (gauge + sparkline):** Target range 2.5–4.5 at $\gamma = 4$. Instant gauge for current value, sparkline for 1-hour trend. Red threshold at 1.8.

3. **Speedup vs baseline (derived metric):** Compute as `baseline_latency_p50 / spec_decode_latency_p50`. Both metrics must be scraped from canary traffic split. Drop below 1.2× triggers warning.

4. **Draft overhead fraction (stacked bar chart):** Shows the composition of request time as draft vs verify vs other (KV management, scheduling). Changes in the ratio surface configuration drift.

5. **Per-position acceptance heatmap:** Rows = draft positions 1–$\gamma$, columns = time buckets. Dark blue = high acceptance, light = low. A sudden stripe of low acceptance at position 1 = draft model quality problem. Stripes only at positions 3–4 = $\gamma$ too high.

### Detecting traffic distribution shifts

The most common production incident is a traffic distribution shift that drops α from 0.80 to 0.58 without any deployment change. Causes include:

- **Time-of-day effects:** Business chat traffic during working hours (α ≈ 0.75) shifts to creative writing at night (α ≈ 0.55).
- **Feature launches:** A new product feature drives new request types the draft model has never seen.
- **Prompt template changes:** A marketing team changes the system prompt — the draft model's distribution is calibrated to the old prompt.
- **Language shifts:** Internationalisation launches drive non-English traffic the draft model predicts poorly.

Detection: alert when the 15-minute rolling mean of α drops more than 0.10 below the 24-hour mean. This catches distribution shifts faster than absolute thresholds while being robust to normal variance.

Response playbook:
1. Check traffic composition (request type distribution, language distribution, temperature settings).
2. If traffic mix has shifted, consider routing affected request types to baseline replicas.
3. If the shift is permanent, retrain the draft model on a sample of the new traffic distribution (for EAGLE: 1–2 GPU-days; for small LM: may need to fine-tune or swap to a different pre-trained model).
4. If the shift is temporary (flash sale, viral moment), reduce $\gamma$ adaptively and restore when traffic normalises.

---

## Deep configuration guide: every flag that matters

This section covers every configuration parameter that affects speculative decoding behaviour in vLLM and SGLang, with practical guidance on tuning each one.

### vLLM speculative decoding flags

```python
## Complete vLLM spec decode configuration reference (vllm >= 0.5.0)
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",

    ## --- Draft model selection ---
    ## Option A: separate LM draft
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    ## Option B: n-gram / PLD (no GPU cost)
    ## speculative_model="[ngram]",
    ## Option C: EAGLE-2 head
    ## speculative_model="yuhuili/EAGLE2-LLaMA3-Instruct-70B",

    ## --- Draft length γ ---
    num_speculative_tokens=4,
    ## Rule: start at 4, increase to 6 if alpha > 0.85, decrease to 2 if alpha < 0.68

    ## --- Draft model parallelism ---
    ## For small LM draft: usually 1 GPU is enough
    speculative_draft_tensor_parallel_size=1,

    ## --- Target parallelism ---
    tensor_parallel_size=4,

    ## --- Memory and compute ---
    dtype="bfloat16",                    ## bf16 for H100/A100; fp16 acceptable
    gpu_memory_utilization=0.88,         ## lower if EAGLE head causes OOM
    max_model_len=8192,

    ## --- Tree attention (for EAGLE / Medusa) ---
    use_v2_block_manager=True,           ## required for tree attention

    ## --- PLD-specific flags ---
    ## ngram_prompt_lookup_max=5,        ## max n-gram length to match in prompt
    ## ngram_prompt_lookup_min=1,        ## min n-gram length

    ## --- EAGLE-specific max tree size ---
    ## speculative_max_model_len=2048,   ## cap verify pass length for EAGLE

    ## --- Disable for batch throughput workloads ---
    ## speculative_model=None,           ## disables spec decode entirely
)
```

### Key flag interactions

**`num_speculative_tokens` × `gpu_memory_utilization`:** Increasing $\gamma$ requires slightly more KV cache per request (to hold the $\gamma$ speculative positions). At $\gamma = 8$, you may need to reduce `gpu_memory_utilization` by 1–2% to avoid OOM on long sequences. For EAGLE, this is more pronounced because the tree can have up to `speculative_num_draft_tokens` nodes, each requiring its own KV entries.

**`speculative_draft_tensor_parallel_size` vs `tensor_parallel_size`:** The draft model runs on fewer GPUs than the target. If `speculative_draft_tensor_parallel_size=1` and `tensor_parallel_size=4`, the draft worker occupies one GPU and the target worker uses all 4. There is no free lunch — if the draft model is large (7B), running it on 1 GPU at 32 ms/token may be too slow. Profile your specific draft model size against your GPU memory budget.

**`use_v2_block_manager`:** Required for tree attention (EAGLE, Medusa). Without it, vLLM falls back to linear spec decode even if you specify an EAGLE draft model. If you see E[accepted] tracking linearly with $\gamma$ (no tree branching benefit), this flag is likely missing.

### SGLang configuration flags

```python
## Complete SGLang EAGLE configuration reference (sglang >= 0.3.0)
import sglang as sgl

engine = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3-70B-Instruct",

    ## --- Speculative algorithm ---
    speculative_algorithm="EAGLE",       ## or "EAGLE2" for EAGLE-2 variant

    ## --- EAGLE draft model ---
    speculative_model_path="yuhuili/EAGLE2-LLaMA3-Instruct-70B",

    ## --- Draft parameters ---
    speculative_num_steps=4,             ## gamma: number of EAGLE autoregressive steps
    speculative_eagle_topk=8,            ## top-K candidates per EAGLE step (tree width)
    speculative_num_draft_tokens=64,     ## total tree node budget

    ## --- Target parallelism ---
    tensor_parallel_size=4,
    dtype="bfloat16",
    mem_fraction_static=0.88,

    ## --- RadixAttention ---
    disable_radix_cache=False,           ## keep prefix cache ON (default)
    radix_cache_size_mb=4096,            ## 4 GB for radix KV cache

    ## --- Prefill configuration ---
    max_prefill_tokens=32768,
    chunked_prefill_size=8192,           ## chunk long prefills to avoid OOM

    ## --- Adaptive gamma (experimental) ---
    ## spec_decode_adaptive_gamma=True,  ## enable online gamma adaptation
    ## spec_decode_alpha_ema_decay=0.95, ## EMA decay for acceptance rate tracking
)
```

### What `speculative_eagle_topk` and `speculative_num_draft_tokens` control

These two parameters define the shape of the EAGLE-2 candidate tree. `speculative_eagle_topk=8` means the EAGLE head proposes the top-8 most likely tokens at each draft step, creating up to 8 branches at each node. `speculative_num_draft_tokens=64` caps the total number of nodes in the tree across all steps.

With $k = 8$ and budget = 64, the tree layout is approximately: step 1 → 8 nodes, step 2 → 16 nodes (top 2 per step-1 node), step 3 → 16 nodes, step 4 → 24 nodes — exhausting the 64 budget. This produces a tree that is wide at the top (high-confidence positions) and narrows towards depth.

The verify cost scales linearly with the tree size. A 64-node tree processes 64 candidate tokens in one target forward pass. At $bs = 1$ on H100, this is approximately 1.5× the cost of verifying a linear 4-token chain, but the expected accepted tokens for a 64-node tree at $\alpha = 0.85$ are approximately 4.8 vs 3.4 for the linear chain — a better tokens-per-verify-cost ratio.

---

## vLLM vs SGLang: which framework to pick

Both frameworks offer solid speculative decoding support. The choice comes down to workload characteristics and operational preferences.

![vLLM speculative decoding architecture with draft worker, target worker, and shared KV cache manager](/imgs/blogs/speculative-decoding-in-production-3.webp)

![Implementation feature comparison across vLLM, SGLang, TGI, and HuggingFace Assisted Decoding](/imgs/blogs/speculative-decoding-in-production-8.webp)

**Choose vLLM if:**
- You need the widest model and hardware support — vLLM supports more model architectures, quantization backends (GPTQ, AWQ, FP8), and hardware (AMD ROCm, Intel Gaudi).
- You are integrating with an existing OpenAI-compatible API layer (vLLM's OpenAI API emulation is more battle-tested).
- You need tensor parallelism across multiple GPUs where the draft and target share the same TP group.
- Your team is already familiar with vLLM's codebase and needs to customise the scheduler or block manager.
- You want the widest community support and upstream issue response time.

**Choose SGLang if:**
- Your workload has high prompt prefix sharing (code assistants, RAG, multi-turn chat with fixed system prompts) — RadixAttention stacks with spec decode for compounded wins.
- You want EAGLE-2 with online adaptive $\gamma$ tuning — SGLang's integration is tighter and the adaptive controller is more mature.
- You are building stateful multi-request workflows using SGLang's programming model (parallel decoding, constrained generation, beam search over generations).
- Raw TTFT latency is your primary metric and you are serving a single-model setup on Hopper-generation GPUs (where SGLang often shows better raw throughput).

**Choose TGI if:**
- You are constrained to Hugging Face's ecosystem (model hub, inference endpoints).
- You only need basic two-model speculative decoding without EAGLE or tree attention.
- You want managed hosting without operating your own inference infrastructure.

**HuggingFace Assisted Decoding** (`model.generate(assistant_model=...)`) is useful for single-machine experimentation. It does not support batching, async scheduling, multi-GPU serving, or adaptive $\gamma$ — it is not production-grade.

---

## When NOT to use speculative decoding

This is worth stating explicitly because every speculative decoding paper leads with speedup numbers and buries the contraindications.

**Do not use speculative decoding if:**

**Your batch size is above 16 at steady state.** Throughput-first workloads — offline batch transcription, bulk translation, nightly report generation, dataset annotation — should use continuous batching with the largest batch size your GPU memory allows. Spec decode adds draft overhead with no latency benefit at high batch. Use baseline decode and maximise GPU utilisation through batching.

**Your output is short on average (< 50 tokens).** The fixed cost of draft model initialisation and the per-step overhead are amortised over output length. For 20-token outputs, you pay the first draft step's overhead on a large fraction of the request — the amortisation ratio is poor. Below 50 tokens, the speedup often drops to 1.1× or less.

**Your acceptance rate α is below 0.65 in practice.** Some tasks are inherently hard to predict: creative writing at high temperature (α ≈ 0.50–0.60), code generation in rare programming languages (α ≈ 0.55–0.65), multi-language mixing, or heavily instruction-tuned tasks where the target's distribution is very different from its base-model weights. If your offline test shows α < 0.65, baseline decode is faster. Measure before deploying.

**You are already compute-bound for other reasons.** If your GPU runs at > 80% SM utilisation due to prefill batching, spec decode's draft worker competes for the same SM resources. Profile with `nvidia-smi dmon` or DCGM to verify your decode utilisation before enabling.

**Your output distribution changes dramatically between requests.** If you serve a mix of code generation ($\alpha = 0.88$) and creative writing ($\alpha = 0.55$) behind a single endpoint with fixed $\gamma$, the code requests are well-served but the creative writing requests are actively hurt. Use adaptive $\gamma$ or route request types to separate replicas.

**Your system has tight memory constraints.** A separate 1B draft model adds 2 GB. The EAGLE-2 head adds 400 MB. If your KV cache is already at > 90% utilisation, the added memory pressure will cause more KV evictions, hurting throughput more than spec decode helps latency. Profile your memory budget before committing.

---

## The production decision flowchart

![Three-question production decision tree: latency-sensitive? small batch? repetitive task?](/imgs/blogs/speculative-decoding-in-production-7.webp)

The three questions you need to answer before enabling speculative decoding:

**Question 1: Is this latency-sensitive?** If you are optimising for tokens per second at scale (throughput) rather than time-to-first-token or inter-token latency (TTFT / TBT), skip speculative decoding. Maximise batch size instead. If you have a latency SLO — users are waiting for tokens to appear — continue.

**Question 2: What is your effective batch size?** Run your production traffic and measure the average number of concurrent decode sequences in flight during a typical busy period. Use vLLM's `num_running_seqs` metric or equivalent. If the steady-state average is above 8, spec decode is unlikely to help. If it is 1–4, it almost certainly will.

**Question 3: What is your task's acceptance rate?** The fastest way to measure this is to run a 200-request sample of production traffic through vLLM or SGLang with spec decode enabled (γ=4, any draft strategy) and observe the acceptance rate metric. If α > 0.72, proceed to choose your optimal draft strategy and γ. If α < 0.65, skip spec decode for this task class and invest elsewhere.

---

## Regime transitions and the math behind the crossover

![Throughput vs batch size regime transitions: spec decode wins at bs<8, breaks even at bs=16, fails at bs=64](/imgs/blogs/speculative-decoding-in-production-4.webp)

Understanding the batch-size crossover mathematically prevents you from being surprised by it in production. The key insight is that both $T_{\text{baseline}}$ and $T_{\text{verify}}$ are functions of batch size, but they are not the same function:

At small batch sizes, both are bandwidth-limited: $T_{\text{baseline}}(bs) \approx T_{\text{verify}}(bs) \approx T_{\text{BW}}$. The speedup is:

$$\text{Speedup}(bs \ll bs_c) \approx \frac{T_{\text{BW}} \cdot E[\text{accepted}]}{T_{\text{draft}} \cdot \gamma + T_{\text{BW}}} = \frac{E[\text{accepted}]}{1 + T_{\text{draft}} \cdot \gamma / T_{\text{BW}}}$$

Since $T_{\text{draft}} \cdot \gamma / T_{\text{BW}}$ is typically 0.3–0.5 for a 68M draft with $\gamma = 4$ and 70B target at $bs = 1$, the speedup is roughly $E[\text{accepted}] / 1.4 \approx 3.4 / 1.4 \approx 2.4\times$.

At large batch sizes, the target is compute-limited: $T_{\text{baseline}}(bs) \approx T_{\text{compute}} / bs$ (time per step drops with batch size due to arithmetic amortisation). But the draft model, being tiny, saturates arithmetic much earlier — its time per token is nearly constant for $bs > 1$. The speedup becomes:

$$\text{Speedup}(bs \gg bs_c) \approx \frac{(T_{\text{compute}} / bs) \cdot E[\text{accepted}]}{T_{\text{draft}} \cdot \gamma + T_{\text{compute}} / bs}$$

As $bs \to \infty$, $T_{\text{compute}} / bs \to 0$, and speedup $\to 0 / T_{\text{draft}} \cdot \gamma = 0$. The draft overhead dominates.

The crossover batch size $bs_c$ where speedup = 1.0× solves:

$$\frac{(T_{\text{compute}} / bs_c) \cdot E[\text{accepted}]}{T_{\text{draft}} \cdot \gamma + T_{\text{compute}} / bs_c} = 1$$

For LLaMA-3 70B FP16 on H100 with $T_{\text{compute}} = 1,200$ ms (total compute time for 70B weights at peak arithmetic), $E[\text{accepted}] = 3.4$, $T_{\text{draft}} \cdot \gamma = 32$ ms:

$$bs_c = \frac{T_{\text{compute}} \cdot (E[\text{accepted}] - 1)}{T_{\text{draft}} \cdot \gamma} = \frac{1200 \times 2.4}{32} \approx 90$$

So the crossover is around $bs = 90$ in theory — but in practice, KV cache pressure, scheduling overhead, and the verify pass processing $\gamma+1$ tokens instead of 1 shift the effective crossover down to $bs \approx 24–32$ for well-tuned deployments. The table above showing negative speedup at $bs = 32$ is consistent with this analysis.

---

## Canary deployment and rollback procedures

Never enable speculative decoding on 100% of traffic immediately. A canary deployment pattern prevents production incidents:

**Week 1 — Offline validation:**
1. Collect 500 production requests into a replay dataset.
2. Run all 500 requests through baseline and spec-decode replicas, measure latency distributions and output quality (use LLM-as-judge or task-specific metrics).
3. Verify that α > 0.70 on your traffic mix and that output quality is identical (spec decode is lossless, but verify against your specific sampling parameters).
4. Verify that P99 TTFT improves by at least 30% (otherwise the complexity is not worth it).

**Day 1 — 5% canary:**
- Route 5% of production traffic to spec-decode replicas.
- Monitor α, E[accepted], TTFT/TBT distributions, error rates.
- Watch for any increase in timeout errors (sometimes a collapsed α leads to very long requests if output diverges).

**Days 2–7 — Progressive rollout:**
- If Day 1 is clean, promote to 25% → 50% → 100% over 5 days.
- Keep baseline replicas in standby for 1 week after 100% rollout.

**Rollback trigger:**
- P99 TTFT on spec-decode replicas > 1.3× baseline for > 10 minutes.
- α < 0.60 for > 5 minutes (traffic distribution shift).
- Any increase in error rate > 0.5 percentage points.

The rollback is a simple load balancer weight change — route all traffic back to baseline replicas while you investigate.

---

## Case studies

### Case study 1: code assistant serving LLaMA-3 70B at bs=1

**Setup.** A startup builds a VS Code extension that calls a self-hosted LLaMA-3-70B-Instruct endpoint for inline code completion. The backend runs on 4× H100 80 GB SXM, vLLM 0.5.0, tensor-parallel degree 4. Traffic is mostly single-turn requests (user selects a code snippet, asks for completion or refactoring), average output length 150 tokens, average concurrent sequences 1.2.

**Problem.** Baseline TTFT is 2.1 seconds for a 200-token completion, which feels sluggish in an interactive editor. Users expect < 800 ms for inline completions to feel natural. The product team's retention data shows a sharp drop in DAU when P50 TTFT exceeds 1.5 seconds.

**Solution.** Enable EAGLE-2 with $\gamma = 5$. Code generation on LLaMA-3-70B with EAGLE-2 achieves $\alpha \approx 0.87$ on HumanEval-style prompts. Using the speedup formula: $E[\text{accepted}] = (1 - 0.87^6) / (1 - 0.87) = 4.07$ tokens per verify step. Draft cost: $5 \times 8$ ms = 40 ms. Verify pass: 105 ms (slightly longer than baseline due to tree attention masking overhead). Total per spec-decode iteration: 145 ms generating 4.07 tokens on average = 35.6 ms/token effective. Baseline: 90 ms/token. Effective speedup: 2.5×.

**Result.** P50 TTFT drops from 2.1 s to 870 ms. P99 drops from 4.1 s to 1.8 s. The engineering cost was two days to configure vLLM with the pre-trained EAGLE-2 head and validate output quality against a test suite of 200 code generation prompts. Acceptance rate remained stable at 0.86–0.88 across two weeks of production traffic. A 48-hour canary at 10% traffic showed no regression in output quality (code correctness rate unchanged on unit-testable completions).

**Key finding.** For code generation at $bs = 1$, EAGLE-2 with $\gamma = 5$ is essentially free speedup. The only cost is the EAGLE-2 head's 400 MB memory footprint, which is negligible on 80 GB GPUs. The product team shipped the feature, user retention improved 8%, and the team did not need to add GPU capacity.

---

### Case study 2: chat API serving Mistral-7B at bs=4

**Setup.** A mid-size SaaS company serves a customer support chatbot using Mistral-7B-Instruct-v0.3 on 2× A100 40 GB. The model fits comfortably on 2 GPUs with tensor-parallel degree 2. Average batch size is 3.8 (multiple simultaneous users), average output length 80 tokens. The team runs 4 replicas total to handle traffic spikes.

**Hypothesis.** The team hears about speculative decoding and wants to cut GPU cost by doubling efficiency (running 2 replicas instead of 4). They plan to use a small Mistral-family draft model.

**Experiment.** They enable vLLM speculative decoding with LLaMA-3.2-1B as draft (same BPE tokenizer family, close enough), $\gamma = 4$.

**Result.** The measured acceptance rate on support chat traffic is $\alpha = 0.63$. At $\alpha = 0.63$ and $\gamma = 4$: $E[\text{accepted}] = (1 - 0.63^5) / (1 - 0.63) = 2.42$. Draft cost: $4 \times 6$ ms = 24 ms. Verify pass: 35 ms (A100, smaller model). Total: 59 ms for 2.42 tokens = 24.4 ms/token. Baseline on Mistral-7B at $bs = 4$: 28 ms/token. Speedup: 1.15×.

The 15% speedup does not justify the added engineering complexity. Acceptance rate is too low because Mistral-7B customer support chat has highly variable language, domain-specific terminology, and the 1B draft model was not trained on support domain data.

**Better approach.** Switch to Prompt Lookup Decoding with $\gamma = 4$, since support tickets frequently quote product documentation and error messages verbatim. PLD achieves $\alpha = 0.69$ with zero GPU overhead. Net speedup: 1.28×. Still not enough to cut from 4 replicas to 2, but enough to improve P99 latency from 4.2 s to 3.3 s, improving user experience without additional cost. The team keeps all 4 replicas but upgrades the hardware utilisation metrics.

**Key finding.** Speculative decoding requires $\alpha > 0.70$ to deliver compelling speedup at $bs = 4$. Measure your task's actual acceptance rate before committing to the integration. Do not assume chat traffic will behave like HumanEval benchmarks.

---

### Case study 3: document summarisation with Prompt Lookup Decoding

**Setup.** A legal tech company summarises contracts using LLaMA-3-70B on 4× A100 80 GB. Contracts are 5,000–20,000 words and the summaries reference specific clauses, dollar amounts, and party names verbatim. Average batch size 2, average output length 400 tokens. The team's P95 wall-clock time per summary is 68 seconds — unacceptably slow for their UI, which shows a live spinner.

**Goal.** Reduce P95 time per summary to under 30 seconds without increasing GPU cost.

**Solution.** Enable PLD with $\gamma = 5$. Legal contracts are highly self-referential — summaries copy exact clause numbers, party names, dates, and defined terms from the source. PLD's suffix-matching finds these continuations reliably at zero GPU cost.

**Measured results.** $\alpha = 0.74$ on a 50-document test set (weighted average — opening sentences are novel with $\alpha \approx 0.55$, clause-referencing sections are high with $\alpha \approx 0.85$). $E[\text{accepted}]$ at $\gamma = 5$: $(1 - 0.74^6) / (1 - 0.74) = 3.47$. Verify pass time increases by ~12% due to processing up to 6 tokens instead of 1 (more KV positions per pass). Net effective speedup: $3.47 / (1 + 0.12) \approx 3.1\times$ for the high-$\alpha$ sections, with the overall weighted average at approximately 2.1×.

**Result.** P95 time per summary drops from 68 s to 33 s — close enough to the 30-second target that a follow-up optimisation (enabling FP8 quantisation on the target) closed the gap. PLD required zero model training, zero GPU memory overhead, and one config line in vLLM. Total implementation time: 4 hours including offline validation on the 50-document test set.

**Key finding.** PLD is severely underrated for long-context tasks with repetitive output. It requires no GPU memory, no draft model training, and delivers meaningful speedup on the right task class. Evaluate PLD before spending engineering time on EAGLE for summarisation workloads.

---

### Case study 4: batch throughput workload — where speculative decoding failed

**Setup.** A research lab runs nightly batch inference to re-annotate 500,000 rows in a dataset using LLaMA-3-70B on 8× H100 80 GB. They use vLLM with continuous batching (effective batch size 48 at steady state, maximum 100 output tokens per row). A team member proposes enabling EAGLE-2 to speed up the nightly job from 6 hours to 4 hours.

**Experiment.** Enable EAGLE-2 with $\gamma = 4$ and run the full 500K-row job.

**Result.** The nightly job takes 6.9 hours — 15% slower than baseline. Root cause: at $bs = 48$, the target model runs at 82% SM utilisation, deeply compute-bound. The EAGLE draft head's overhead — running sequentially before each verify pass — adds 32 ms per iteration ($\gamma = 4$, 8 ms/token). The per-step time in compute-bound mode is 18 ms at $bs = 48$, so the draft overhead is 178% of the baseline step time. The verify pass, which processes $\gamma + 1 = 5$ token positions simultaneously, is faster than 5 baseline steps (saving $5 \times 18 - 22 = 68$ ms per verify pass) — but the 32 ms draft overhead plus scheduling overhead exceeds this savings.

**Fix.** Disable EAGLE-2. Increase batch size to 64 and enable FP8 quantisation (H100 supports FP8 GEMM natively at 2× throughput). Combined change: nightly job drops from 6 hours to 4.7 hours — 22% improvement without spec decode.

**Key finding.** Speculative decoding and throughput maximisation are fundamentally opposed strategies. Never enable spec decode on batch throughput workloads without measuring at the actual production batch size first. The improvement a researcher sees at $bs = 1$ on their workstation completely misleads the decision for $bs = 48$ in production.

---

### Case study 5: mixed-task endpoint — adaptive γ as the solution

**Setup.** An enterprise AI company serves a unified LLaMA-3-70B endpoint behind a single load balancer. The endpoint handles three request types: structured JSON generation (40% of traffic, $\alpha = 0.88$), general chat (45%, $\alpha = 0.71$), and creative writing at temperature 1.1 (15%, $\alpha = 0.54$). The team enables SGLang with EAGLE-2 and fixed $\gamma = 5$.

**Problem.** Creative writing requests at $\gamma = 5$ and $\alpha = 0.54$ have $E[\text{accepted}] = 1.91$ draft tokens per verify pass but pay $5 \times 8 = 40$ ms of draft overhead per iteration. Their effective "speedup" is:

$$\frac{90 \times 1.91}{40 + 95} = \frac{171.9}{135} \approx 0.87 \times$$

P99 TBT for creative writing requests spikes to 2.1× baseline. Users notice the creative writing endpoint feels slower than before, and support tickets increase.

**Root cause.** Fixed $\gamma = 5$ is optimal for structured JSON ($\alpha = 0.88$) but is a clear liability for creative writing ($\alpha = 0.54$). The draft model is generating 5 candidates per step for a task where it is only right 54% of the time.

**Solution.** Enable SGLang's adaptive $\gamma$ controller with the marginal-value threshold. The controller detects low acceptance rate on creative writing requests within 3–5 steps and reduces $\gamma$ to 2 for those requests. For structured JSON, $\gamma$ stays at 5–6. For chat, it settles around $\gamma = 4$.

**Result after adaptive γ.** Creative writing P99 TBT drops from 2.1× to 1.02× baseline (essentially neutral — spec decode is correctly identified as not helpful for this task class and backs off). Structured JSON maintains 3.3× speedup. Chat holds at 2.1×. The unified endpoint now handles mixed traffic without degrading any request class.

**Key finding.** Mixed-workload endpoints need either request-type routing to separate replicas (cleanest operationally) or adaptive $\gamma$ (lower engineering effort). Fixed $\gamma$ for mixed tasks will always over-serve one class and under-serve another — and the under-served class may be actively hurt.

---

## Pulling it all together: the 2026 production checklist

Speculative decoding is a mature production technique. The remaining engineering work for most teams is:

**Step 1 — Measure α on your traffic (30 min).**
Enable vLLM or SGLang with spec decode, $\gamma = 4$, any draft strategy. Run 200 production requests. Read the acceptance rate metric. Gate on α ≥ 0.70 before proceeding.

A quick way to baseline α without touching production: replay recent request logs through a shadow replica with spec decode enabled. This gives you a realistic α estimate without the risk of user-facing latency regression during measurement.

If you do not have a shadow replica, route 1% of traffic with a separate load balancer rule for 2 hours — 200 requests at typical chatbot traffic levels arrive in well under 2 hours. The measurement cost is negligible and the information is invaluable.

**Step 2 — Validate your batch size regime (15 min).**
Check vLLM's batch-size histogram under production load. If steady-state $bs > 16$, stop here — spec decode is not your lever.

**Step 3 — Pick the right draft strategy.**
- Code / structured output → EAGLE-2 (pre-trained heads available).
- RAG / summarisation / extraction → PLD (zero cost, one config line).
- General chat / diverse text → small LM draft (same model family, 1B scale).
- Highly repetitive output → n-gram drafter (zero cost, simpler than PLD).

**Step 4 — Set γ = 4 as default, enable adaptive γ for mixed traffic.**
Adaptive $\gamma$ is experimental in both frameworks but provides meaningful protection against the mixed-task penalty.

**Step 5 — Deploy via canary at 5% traffic for 24 hours.**
Alert on α < 0.65, P99 TTFT > 1.3× baseline, any error rate increase.

**Step 6 — Alert on α < 0.65 in production.**
This is the leading indicator of spec decode becoming net-negative. Alert before it causes user-facing latency regression.

**Step 7 — Revisit every 90 days.**
Traffic distributions change, new draft model checkpoints are released, and new EAGLE heads become available for your target model family. Speculative decoding configuration is not set-and-forget. Check at least quarterly whether a newer EAGLE-2 head or a different draft model size would improve your production α. A 5-point improvement in α (e.g., from 0.78 to 0.83) translates to roughly 0.3 more accepted tokens per step — compounding across millions of requests, this is significant.

The full theoretical foundation for each of these decisions lives in the preceding posts:

- [Why LLMs are slow: the autoregressive bottleneck](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck)
- [The core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify)
- [Token acceptance: rejection sampling explained](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling)
- [Medusa: multi-head speculative decoding](/blog/machine-learning/speculative-decoding/medusa-multi-head-speculative-decoding)
- [EAGLE: feature-level speculative decoding](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment)
- [Tree speculation: drafting multiple futures](/blog/machine-learning/speculative-decoding/tree-speculation-drafting-multiple-futures)

For the serving-stack perspective beyond spec decode, see [vLLM serving](/blog/machine-learning/large-language-model/vllm-inference), [SGLang inference](/blog/machine-learning/large-language-model/sglang-inference), [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques), and the [speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding).

The one-line summary: **if you are latency-sensitive, your batch size is below 8, and your task acceptance rate exceeds 0.72, enable EAGLE-2 with $\gamma = 4$ in vLLM or SGLang today.** The theory says it should work. The benchmarks confirm it does. The engineering work is three config lines and one afternoon of offline validation.

Everything else in this post is about the cases where it does not work. Knowing those cases — and having the monitoring to detect them quickly — is what separates a reliable production deployment from a speculative decoding experiment that silently degrades user experience at 3 AM on a Tuesday.
