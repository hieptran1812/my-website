---
title: "Request Scheduling and Preemption: The Scheduler Inside Your LLM Serving Engine"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "How an LLM serving engine decides which requests run each decode step — the continuous-batching loop, chunked prefill, preemption cost math, and the scheduling policies that trade throughput for tail latency."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "request-scheduling",
    "preemption",
    "continuous-batching",
    "chunked-prefill",
    "vllm",
    "llm-serving",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/request-scheduling-and-preemption-1.webp"
---

At 09:14 on a Tuesday, a single request took down the p99 latency of an entire LLM cluster. Nothing crashed. No GPU OOM'd, no pod restarted, no alert fired for thirty more seconds. What happened was subtler and more instructive: one user pasted a 28,000-token legal document into the chat box and hit send. The serving engine dutifully began prefilling that prompt — and for the ~180 milliseconds it spent chewing through those 28k tokens in one giant forward pass, every one of the 60 other users mid-conversation got exactly zero new tokens. Their streams froze. Their per-token latency (TPOT) spiked from a smooth 25 ms to over 200 ms. The dashboard turned red. And the only "fix" anyone could think of in the moment — restart the pods — did nothing, because the problem was not a resource leak. The problem was the *scheduler*.

Every LLM serving engine — vLLM, TGI, SGLang, TensorRT-LLM — contains a small, fast, ruthless decision-maker that runs many times per second. On every model forward pass (every "step"), it looks at the requests waiting to start, the requests currently generating, and the finite pool of GPU memory holding their KV caches, and it decides: *which sequences get compute this step?* Admit a new one from the queue? Keep decoding the ones already running? Or — when memory runs out — evict a running request to make room? That decision loop is the beating heart of throughput and latency. Get it right and one H100 serves thousands of concurrent conversations at 90%+ utilization. Get it wrong and one pathological prompt freezes the fleet.

This post is a deep dive into that scheduler. We will build up the continuous-batching scheduler loop from first principles, derive the queuing theory that governs how deep your queue can get before latency explodes (Little's Law is going to earn its keep), work through the exact cost math of the two preemption strategies (swap-to-CPU versus recompute), and lay out the scheduling policies — FCFS, priority, deadline-aware, fair — with an honest account of what each one costs you. Then we get practical: real vLLM `EngineArgs` flags, a priority-queue example, a FastAPI admission-control gateway that returns HTTP 429 before your engine melts, and benchmark scripts that sweep the two knobs that matter most (`max_num_seqs` and `max_num_batched_tokens`) while measuring p99 TTFT and TPOT on an H100.

The figure below is the whole post in one picture: the per-step decision the scheduler makes, and the three outcomes it can produce. Keep it in mind — everything that follows is an elaboration of this one loop, and every technique is a trade on the serving SLO triangle of **latency ↔ throughput ↔ cost** (with fairness as a fourth, often-forgotten corner).

![Per-step scheduler decision flow merging the waiting queue and running set into admit, continue, or preempt branches under a shared token budget](/imgs/blogs/request-scheduling-and-preemption-1.webp)

If you are new to how LLM inference differs from classical model serving, the companion posts [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) and [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) set the stage; this post zooms all the way into the scheduler that those systems are built around.

## 1. Why the scheduler exists: two kinds of work sharing one GPU

To understand scheduling you have to understand the workload it schedules, and LLM inference has a split personality. A request's life has two phases with wildly different performance profiles.

**Prefill** processes the entire input prompt in one shot. If the prompt is 2,000 tokens, the model runs a single forward pass over all 2,000 positions at once. This phase is *compute-bound*: it saturates the GPU's tensor cores, does a large matrix multiply, and produces the first output token plus the KV cache for the whole prompt. Prefill throughput on an H100 for an 8B model can exceed 20,000 tokens/second because the arithmetic intensity is high — lots of FLOPs per byte of weights loaded.

**Decode** generates output tokens one at a time, autoregressively. Each decode step processes exactly one new token per sequence, reads the entire KV cache built so far, and produces the next token. This phase is *memory-bandwidth-bound*: the GPU spends most of its time streaming weights and KV cache from HBM, and the actual arithmetic (one token's worth) barely warms the tensor cores. A single decode step for one sequence might use less than 1% of the GPU's FLOP capacity.

That asymmetry is the entire reason scheduling is hard. Prefill wants big, dense batches of prompt tokens; decode wants many sequences running in parallel to amortize the weight-loading cost. If you run them naively — process a whole prompt, then decode it to completion, then take the next request — you get the disaster from the intro: a long prefill monopolizes the GPU and starves every decode, and between requests the GPU sits nearly idle streaming weights for a batch of one. The scheduler's job is to *mix* these two kinds of work every step so the tensor cores and the memory bus are both kept busy, without letting either phase starve the other.

The scheduler manages this through a small set of data structures. In vLLM the canonical ones are three queues plus a memory allocator, shown in the next figure. A sequence starts in the **WAITING** queue (it has arrived but has no GPU memory yet). When the scheduler admits it, it moves to the **RUNNING** set (it has KV-cache blocks allocated and gets compute every step). If memory runs out, a running sequence can be preempted into the **SWAPPED** queue (its KV cache copied to host RAM) or dropped back to WAITING for recompute. When it emits an end-of-sequence token or hits `max_tokens`, it reaches **FINISHED** and its blocks are freed. Every transition between these states is gated by one resource: free KV-cache blocks in the block manager.

![Layered stack showing a sequence moving through waiting, running, and swapped queues gated by the block manager before reaching the finished state](/imgs/blogs/request-scheduling-and-preemption-6.webp)

The block manager is the constraint that makes all of this interesting. GPU memory is finite, the KV cache grows with every token generated, and the scheduler cannot know in advance how long any sequence will be — a request might stop after 5 tokens or run to 4,000. So it admits sequences optimistically, and when the collective KV cache threatens to overflow, it has to claw memory back. That clawing-back is preemption, and it is where the throughput-latency trade gets sharpest.

For a deeper treatment of how KV-cache blocks are allocated, evicted, and reused, see [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization); here we take the block manager as given and focus on the scheduling decisions layered on top of it.

## 2. The continuous-batching scheduler loop

Classical model serving batches requests *up front*: collect N requests, run them as one batch, return N responses, repeat. That is fine when every request takes the same, short, predictable time. It is catastrophic for LLMs, where one request in the batch might generate 20 tokens and another 2,000 — the whole batch is held hostage by the slowest member, and the fast requests wait idle for tokens they finished producing long ago.

Continuous batching (introduced by Orca as "iteration-level scheduling") fixes this by making the batch *dynamic at the granularity of a single decode step*. The scheduler runs a loop that looks roughly like this, once per model forward pass:

```python
# Simplified continuous-batching scheduler loop (conceptual, vLLM-style).
# Runs once per model step; a step is one forward pass over the mixed batch.
def schedule_step(waiting_queue, running_set, block_manager, budget):
    scheduled = []          # sequences that will get compute this step
    token_budget = budget   # max_num_batched_tokens for this step

    # 1. CONTINUE: every running sequence needs 1 decode token this step.
    for seq in running_set:
        if block_manager.can_append_one_token(seq):
            scheduled.append((seq, tokens=1))
            token_budget -= 1
        else:
            # No free block to grow this seq's KV cache -> must preempt.
            victim = pick_preemption_victim(running_set)   # usually newest / lowest priority
            preempt(victim, block_manager)                 # swap or recompute
            running_set.remove(victim)

    # 2. ADMIT: pull new requests from the waiting queue while budget + memory allow.
    while waiting_queue and token_budget > 0:
        seq = waiting_queue.peek()
        prefill_len = min(seq.remaining_prompt_tokens, token_budget)  # chunked prefill
        if block_manager.can_allocate(seq, prefill_len):
            block_manager.allocate(seq, prefill_len)
            scheduled.append((seq, tokens=prefill_len))
            token_budget -= prefill_len
            if seq.prefill_done():
                running_set.add(waiting_queue.pop())
        else:
            break   # not enough memory to admit anyone else this step

    return scheduled    # hand this mixed batch to the model for one forward pass
```

The details differ across engines, but the shape is universal: each step, the scheduler first services the sequences already running (so ongoing generations keep flowing), then admits new work up to a token budget and a memory limit, and preempts when memory is exhausted. The result of one iteration is a *mixed batch* — some sequences contributing one decode token, others contributing a chunk of prefill tokens — handed to the model for a single forward pass.

There is a subtlety that Orca had to solve to make this work at all, and it is worth naming because it explains why you cannot simply `torch.stack` a mixed batch. Different sequences in the same batch are at different positions: one is prefilling 512 prompt tokens, another is decoding its 300th token, a third its 12th. Their tensors have incompatible shapes, so the usual "pad everything to the same length and run one matmul" trick does not apply. Orca's answer was *selective batching*: batch the operations that *can* be batched (the large, shape-agnostic matrix multiplies in the feed-forward and projection layers, where every token is independent) while handling attention *per sequence* (because attention's shape depends on each sequence's own KV-cache length). This selective treatment is what lets a single forward pass serve sequences at wildly different stages, and it is baked into every modern engine's attention kernel. You will not write this code, but knowing it exists explains why attention kernels for serving (PagedAttention, FlashAttention's variable-length variants) are special: they are built to run a ragged batch in one launch.

Two numbers govern this loop, and they are the two knobs you will spend your life tuning:

- **`max_num_seqs`** — the maximum number of sequences allowed in the running set at once. This caps decode parallelism.
- **`max_num_batched_tokens`** — the maximum number of tokens (prefill + decode combined) the scheduler will pack into a single step. This is the per-step compute budget.

Everything about the throughput-latency trade flows from how these two interact with your traffic. We will derive their effects mathematically in the next two sections, then tune them empirically at the end. For the broader batching context — static versus dynamic versus continuous — the [batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) post is the prerequisite reading.

## 3. The mechanics, part one: Little's Law and the running queue

Here is the single most useful piece of theory for capacity-planning an LLM service, and it comes from queuing theory, not deep learning. **Little's Law** states that for any stable system in steady state:

$$L = \lambda W$$

where $L$ is the average number of items in the system, $\lambda$ is the average arrival (or completion) rate, and $W$ is the average time an item spends in the system. It is almost embarrassingly simple, it requires no assumptions about the arrival distribution, and it tells you exactly how the scheduler's knobs relate to your throughput ceiling.

Apply it to the running set. Here $L$ is the number of concurrently running sequences — which the scheduler caps at `max_num_seqs`. $W$ is the mean *residence time* of a request in the running state: roughly its prefill time plus (output tokens × TPOT). And $\lambda$ is the request completion rate — your throughput. Rearranging:

$$\lambda_{\max} = \frac{L_{\max}}{W} = \frac{\texttt{max\_num\_seqs}}{\bar{W}}$$

This is your throughput ceiling, and it falls straight out of two numbers you can measure.

#### Worked example: the throughput ceiling of one H100

Suppose you run Llama-3.1-8B on a single H100, and you have measured:

- `max_num_seqs = 256` (you allow up to 256 concurrent sequences)
- Mean output length ≈ 200 tokens, mean TPOT ≈ 28 ms, mean prefill ≈ 60 ms
- So mean residence time $\bar{W} \approx 0.060 + 200 \times 0.028 = 5.66$ seconds

Little's Law gives $\lambda_{\max} = 256 / 5.66 \approx 45$ requests/second. That is your hard ceiling *for this configuration and this traffic mix*. No amount of clever queueing gets you past it — if requests arrive faster than 45/s on average, the waiting queue grows without bound and latency runs to infinity. To go faster you must either raise `max_num_seqs` (which needs more KV-cache memory, and eventually causes preemption thrash — section 6), reduce $\bar{W}$ (shorter outputs, faster decode via [speculative decoding](/blog/machine-learning/model-serving/kv-cache-optimization) or quantization), or add GPUs.

The waiting queue obeys the same law, and this is where admission control comes from. If $\lambda$ is your arrival rate and $W_q$ is the time a request waits before being admitted, then the queue depth is $L_q = \lambda W_q$. If your TTFT SLO says "first token within 2 seconds" and the running system adds ~0.5 s of prefill, you can tolerate at most $W_q \approx 1.5$ s of queueing. At $\lambda = 45$ req/s that means a queue depth of $L_q = 45 \times 1.5 \approx 68$ requests. Beyond that, admitting more requests only guarantees SLO violations — you should shed load instead (return 429), which we implement in section 8.

There is one more piece of queuing theory worth internalizing: the *non-linear* blow-up near saturation. Model the running set crudely as an M/M/1 queue with utilization $\rho = \lambda / \mu$ (arrival rate over service rate). The expected wait time is

$$W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)}$$

As $\rho \to 1$, $W \to \infty$. The practical consequence: latency does not degrade gracefully as you approach capacity — it stays flat and then hits a wall. Running at 70% utilization feels fine; running at 95% feels fine right up until a small traffic bump pushes you to 100% and p99 latency goes vertical. This is why you provision headroom and why the queue-depth-based load shedding in section 8 is not optional at scale. The [SLA and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) post covers how to set the SLO targets that anchor these calculations.

### Why the mean lies: variability and the p99 tail

Little's Law and the M/M/1 formula use *averages*, but your SLO is written in *percentiles* — p99 TTFT, not mean TTFT — and the gap between them is governed by variability. Real LLM traffic is bursty (arrivals cluster into spikes) and its service times are wildly variable (a 10-token reply and a 2,000-token reply share the same queue). Kingman's approximation for the expected queue wait captures why this matters:

$$W_q \approx \left(\frac{\rho}{1-\rho}\right)\left(\frac{C_a^2 + C_s^2}{2}\right)\tau$$

where $\rho$ is utilization, $\tau$ is mean service time, and $C_a$, $C_s$ are the coefficients of variation of inter-arrival times and service times respectively. The middle term is the killer. When service times are highly variable — which mixed prompt and output lengths guarantee — $C_s$ is large, and wait time scales with the *square* of that variability. Two workloads can share the same mean service time and the same utilization, yet the one with replies ranging from 10 to 4,000 tokens has a p99 wait several times worse than the one with uniform 100-token replies. This is the quantitative root of head-of-line blocking (section 10), and it is why "our average latency is fine" is never an acceptable answer: the average is precisely the statistic that hides the tail your users actually feel. The operational rules follow directly — provision against p99, alert on p99, and judge every configuration change by its effect on p99 TPOT, not the mean. It is also why reducing service-time variability (capping `max_tokens`, separating long-output batch jobs onto their own engine, bucketing by prompt length) is often a bigger p99 win than any knob in this post: you are shrinking $C_s^2$ at the source.

## 4. The mechanics, part two: the prefill-token budget

Now the second knob. `max_num_batched_tokens` (call it $B$) is the number of tokens the scheduler packs into one step. In a mixed batch, decode always gets priority: each of the $n$ running sequences contributes exactly one token, consuming $n$ of the budget. Whatever remains, $B - n$, is available for prefill:

$$c = B - n$$

where $c$ is the number of prompt tokens that can be prefilled this step. This one equation explains most of what you will observe when tuning.

Consider a prompt of length $P$ arriving into an engine with budget $B$ and $n$ running decode sequences. Without chunked prefill, that prompt must be prefilled in a single step of $P$ tokens — and that step's duration is proportional to $P$, so all $n$ decoders freeze for the whole prefill. With chunked prefill enabled, the prompt is split into chunks of size $c = B - n$, and it takes

$$\text{steps} = \left\lceil \frac{P}{B - n} \right\rceil$$

steps to complete, each of which *also* advances all $n$ decoders by one token. Prefill and decode now share every step. The figure below shows one such mixed batch playing out over four steps: a prefill being fed in 512-token chunks alongside 48 sequences each emitting one decode token, all packed under a 1,024-token budget.

![Grid showing four scheduler steps each packing a 512-token prefill chunk plus one token per decode sequence into a shared 1024-token budget](/imgs/blogs/request-scheduling-and-preemption-7.webp)

This gives us a clean way to reason about the two knobs:

- **Bigger $B$** → bigger prefill chunk per step → *fewer* steps to finish a prefill → **lower TTFT**. But a bigger step does more total work, so each step takes longer → **higher TPOT** for the decoders sharing that step. And a bigger $B$ means more KV-cache growth per step, raising memory pressure.
- **Bigger `max_num_seqs`** ($n$) → more decode parallelism → **higher throughput** (you amortize weight-loading across more sequences). But it leaves less budget for prefill ($c = B - n$ shrinks), *raising* TTFT, and it consumes more KV memory, *raising* preemption risk.

The two knobs pull TTFT and TPOT in opposite directions, and the right setting depends entirely on which one your SLO cares about more. A chat product optimizing for snappy first tokens wants larger $B$; a batch summarization job optimizing for total tokens/second wants larger `max_num_seqs` and does not care about TTFT at all. There is no universally correct answer, which is exactly why these are exposed as knobs and not hard-coded.

#### Worked example: sizing the budget for a chat SLO

You serve interactive chat with an SLO of TTFT < 800 ms and TPOT < 40 ms. Your typical prompt is $P = 3{,}000$ tokens, you run $n = 64$ decoders on average, and one step over a batch of ~1,000 tokens takes about 18 ms on your H100.

Try $B = 2{,}048$. Prefill chunk $c = 2048 - 64 = 1984$ tokens; a 3,000-token prompt needs $\lceil 3000 / 1984 \rceil = 2$ steps. But a 2,048-token step is heavy — call it ~30 ms — so during prefill your decoders see ~30 ms steps, right at the edge of the 40 ms TPOT budget, and TTFT ≈ 2 × 30 = 60 ms of compute plus queueing. Throughput is high.

Try $B = 512$. Chunk $c = 512 - 64 = 448$; the prompt now needs $\lceil 3000/448 \rceil = 7$ steps. Each step is light (~12 ms), so TPOT stays a smooth ~12–15 ms — great for the ongoing conversations — but TTFT for the big prompt is now ~7 × 12 = 84 ms of compute, and worse, that prompt is competing for admission across 7 steps. Long prompts feel sluggish.

The sweet spot is usually in between, and — critically — it is a function of *your* prompt-length distribution, not a constant. Teams that copy `max_num_batched_tokens` from a blog post and never measure are leaving a large fraction of their SLO budget on the floor. Measure, then set (section 9).

## 5. Chunked prefill: taming the throughput-latency trade

Chunked prefill deserves its own section because it is the single most important scheduling feature for mixed interactive workloads, and it is the direct fix for the intro's freeze-the-fleet failure. The technique was formalized by the Sarathi and Sarathi-Serve work: split each prefill into fixed-size chunks and *piggyback* decode work onto the same steps, so that no single prefill ever monopolizes a forward pass.

The before-and-after is stark. The figure contrasts the two regimes: a whole 4,096-token prefill run in one step (which blocks 48 decoders and spikes their TPOT from 25 ms to over 200 ms) versus the same prefill split into eight 512-token chunks interleaved with decode (TPOT stays flat around 30 ms, at the cost of an ~18% increase in that one prompt's TTFT).

![Before-and-after contrast of a whole prefill stalling all decoders versus chunked prefill interleaving prefill chunks with decode to keep TPOT flat](/imgs/blogs/request-scheduling-and-preemption-2.webp)

The trade is explicit and worth stating plainly: chunked prefill *slightly increases TTFT for long prompts* (they are spread across more steps) in exchange for *dramatically smoother TPOT for everyone else* (no more prefill stalls). For any service where users are actively watching tokens stream — chat, coding assistants, agents — that is almost always the right trade, because a frozen stream is far more noticeable than 18% more latency on the first token of a long document.

Enabling it in vLLM is one flag. Here is a realistic engine configuration for an interactive chat service on a single H100:

```python
from vllm import LLM, SamplingParams

# Interactive chat config: smooth streaming prioritized over raw prefill speed.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,      # leave 10% headroom for activation spikes
    max_num_seqs=128,                 # cap concurrent sequences (decode parallelism)
    max_num_batched_tokens=2048,      # per-step token budget (prefill + decode)
    enable_chunked_prefill=True,      # split long prefills; piggyback decode
    max_model_len=8192,               # context window we support
    # preemption_mode defaults to "recompute"; see section 6
)

params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Summarize the attached contract: ..."], params)
```

A few practitioner notes that the docs undersell:

- On modern vLLM (v0.6.3+), chunked prefill is enabled *by default* for many models and context lengths, because the maintainers concluded the interactive-latency win is worth the small throughput cost for most workloads. Check your version — do not assume it is off.
- When chunked prefill is on, `max_num_batched_tokens` doubles as the chunk size. The vLLM default (often 512 or 2,048 depending on version and hardware) is tuned for a generic mix; if your prompts are very long, a larger value reduces the number of chunks and lowers TTFT.
- Chunked prefill changes the shape of your latency histogram, not just the mean. Watch p99 TPOT before and after — the whole point is that the *tail* of TPOT collapses, which is exactly what the mean hides.

### How chunked prefill interacts with prefix caching and CUDA graphs

Two interactions surprise people. The first is prefix caching: when the block manager reuses KV for shared prefixes (covered in [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization)), a prompt that shares a prefix with an already-cached sequence skips the prefill of that prefix entirely — the scheduler only prefills the *novel* suffix. This composes cleanly with chunked prefill: the chunk budget is spent only on tokens that actually need computing, so a long system prompt shared across thousands of requests is prefilled once and thereafter costs almost nothing to admit. If your workload has heavy prefix reuse — shared system prompts, few-shot exemplars, multi-turn chat history — the effective prefill load the scheduler sees is far lower than the raw prompt lengths suggest, and you can afford a higher `max_num_seqs` than the naive KV-memory math would allow.

The second is CUDA graphs. Engines capture the decode step as a CUDA graph to eliminate per-kernel launch overhead, but graphs are captured for *specific batch shapes*. Mixed prefill-plus-decode batches have variable shapes, some of which fall outside the captured set and run in slower eager mode. vLLM mitigates this by capturing graphs for a range of common batch sizes and padding to the nearest, but it is the reason your throughput-versus-`max_num_seqs` curve is not perfectly smooth — some batch sizes land on a captured graph and some do not. When you benchmark, this appears as small step-changes in throughput at particular knob values. Do not over-fit your tuning to a single lucky data point; sweep a range and pick a robust setting, not the one that happened to hit a captured graph.

## 6. Preemption: when the KV cache is full

Continuous batching admits sequences optimistically, betting that enough of them will finish before memory runs out. Usually the bet pays. When it does not — when the running sequences collectively demand more KV-cache blocks than the GPU has — the scheduler must preempt: evict a running sequence, free its blocks, and let the survivors continue. The evicted sequence goes back to the queue and resumes later.

Preemption is not a failure. It is the pressure-relief valve that lets the engine run at high `max_num_seqs` without crashing, and a healthy system preempts occasionally under load. But it is not free, and *frequent* preemption — "preemption thrash," where the engine spends more time evicting and restoring sequences than making forward progress — is one of the classic LLM-serving pathologies. It shows up as throughput collapse and TPOT variance under high concurrency, and the fix is almost always to *lower* `max_num_seqs` so the engine stops over-committing memory.

The full life of a preempted request is worth tracing, because its end-to-end latency includes costs that a naive per-token model completely misses. The timeline figure follows one request: queued at t=0, admitted at 40 ms when blocks free up, decoding happily for a while, preempted at 520 ms when the KV cache fills, resumed at 690 ms, and finally finishing at 1,180 ms — far later than its raw compute would suggest, because it paid queueing latency twice and a recompute penalty on resume.

![Timeline of a request that is queued, admitted, decodes, gets preempted under memory pressure, resumes, and finishes much later than its compute implies](/imgs/blogs/request-scheduling-and-preemption-4.webp)

This is why preemption is a *tail-latency* problem more than a throughput problem. A request that gets preempted once or twice can see its total latency double or triple, even though the fleet's aggregate throughput barely moves. If your p99 latency is mysteriously high while your median is fine and your GPUs are busy, preemption is the first suspect. Instrument it: vLLM exposes preemption counts in its metrics (`vllm:num_preemptions_total`), and a rising preemption rate is your early-warning signal that `max_num_seqs` or `gpu_memory_utilization` is set too aggressively for your traffic.

*Which* sequence gets evicted matters as much as *how*. The default heuristic preempts the most recently admitted running sequence — a last-in-first-out discipline — for two reasons: that sequence has generated the fewest tokens, so it loses the least work on a recompute, and evicting the newest arrival keeps older, closer-to-finishing requests moving toward completion (which frees their blocks soonest, relieving the very pressure that triggered preemption). Under a priority policy the victim is instead the lowest-priority running sequence, which turns preemption into the enforcement mechanism for priority: a burst of high-priority traffic can evict low-priority sequences mid-generation to reclaim their blocks. That is powerful and dangerous in equal measure — without aging, a sustained stream of high-priority work can repeatedly preempt the same unlucky low-priority request, which then never finishes. If you enable priority scheduling, watch per-tier completion rates, not just aggregate throughput, or you will ship a starvation bug you cannot see on the top-line dashboard.

You control preemption behavior with `max_num_seqs` (lower = less over-commitment = fewer preemptions, but also less throughput headroom) and `gpu_memory_utilization` (lower = more KV headroom = fewer preemptions, but less memory for the cache overall). And you control *how* preemption is done — swap versus recompute — which is the cost math of the next section.

## 7. The preemption cost math: swap-to-CPU vs recompute

When the scheduler preempts a sequence, it has two ways to make room, and they trade the same two resources every serving decision trades: bandwidth versus compute.

**Swap-to-CPU** copies the victim's KV-cache blocks out of GPU HBM and into host RAM over the PCIe bus, freeing the GPU blocks. On resume, it copies them back. Nothing is recomputed — the KV cache is preserved — but you pay the PCIe round trip.

**Recompute** simply discards the victim's KV cache and frees the blocks immediately. On resume, the sequence is treated as if it were a fresh prompt: its already-generated tokens become the "prompt," and prefill runs again to rebuild the KV cache from scratch. Nothing is copied — but you pay the prefill FLOPs a second time.

Which is cheaper? Set up the cost model. For a sequence of length $s$ tokens:

$$T_{\text{swap}} = \frac{2 \cdot s \cdot k}{\text{BW}_{\text{PCIe}}}, \qquad T_{\text{recompute}} = \frac{s}{R_{\text{prefill}}}$$

where $k$ is the KV-cache bytes per token, $\text{BW}_{\text{PCIe}}$ is the effective PCIe bandwidth (the factor of 2 accounts for out-and-back), and $R_{\text{prefill}}$ is prefill throughput in tokens/second. Recompute wins when $T_{\text{recompute}} < T_{\text{swap}}$, i.e. when

$$\frac{s}{R_{\text{prefill}}} < \frac{2 \cdot s \cdot k}{\text{BW}_{\text{PCIe}}} \;\Longrightarrow\; \frac{1}{R_{\text{prefill}}} < \frac{2k}{\text{BW}_{\text{PCIe}}}$$

Notice the $s$ cancels: to first order, whether recompute or swap is cheaper does *not* depend on sequence length — it depends on the ratio of your prefill compute rate to your PCIe bandwidth relative to KV size per token. The figure lays out the two mechanisms side by side.

![Before-and-after comparison of swap-to-CPU copying KV blocks over PCIe versus recompute discarding and rerunning prefill on resume](/imgs/blogs/request-scheduling-and-preemption-5.webp)

#### Worked example: swap vs recompute for Llama-3.1-8B on an H100

Llama-3.1-8B has 32 layers, 8 KV heads (grouped-query attention), head dim 128, bf16. KV bytes per token:

$$k = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} \approx 128 \text{ KB/token}$$

Take a preempted sequence of $s = 1{,}000$ tokens. Its KV cache is ~128 MB.

- **Swap**: over PCIe Gen4 x16 at an effective ~16 GB/s, one direction is $128 / 16 = 8$ ms; round trip ≈ 16 ms. (Gen5 roughly halves this.)
- **Recompute**: at a prefill rate of ~20,000 tokens/s for 8B on an H100, rebuilding 1,000 tokens costs $1000 / 20000 = 50$ ms.

Here swap is actually cheaper for this model, because GQA makes the KV cache small (only 8 KV heads) while prefill is not infinitely fast. Flip the model: a model with full multi-head attention (many more KV heads) has a much larger $k$, which makes swap expensive and tilts the balance toward recompute. This is exactly why the *right* default depends on the model architecture — and why vLLM historically defaulted to `recompute` (it avoids the PCIe dependency and is robust when host RAM or PCIe is contended), but exposes `preemption_mode="swap"` for cases where KV is small and PCIe is fast.

```python
from vllm import LLM

# For a small-KV model (GQA/MQA) on a fast PCIe/NVLink host, swap can win.
# For large-KV models or contended PCIe, keep the recompute default.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=256,
    gpu_memory_utilization=0.92,
    preemption_mode="swap",         # "recompute" (default) | "swap"
    swap_space=8,                   # GiB of host RAM reserved for swapped KV
    enable_chunked_prefill=True,
)
```

| Preemption mode | Cost driver | Cheaper when | Failure mode | Extra resource |
|---|---|---|---|---|
| Recompute (default) | Prefill FLOPs, paid again | Prefill is fast; KV per token is large | Wastes compute already done | None (uses GPU) |
| Swap-to-CPU | PCIe bandwidth, round trip | KV per token is small; PCIe is fast | PCIe/host RAM contention stalls resume | `swap_space` host RAM |

The honest recommendation: leave `preemption_mode` at its default (`recompute`) unless you have measured a specific model where swap wins *and* you have verified your PCIe bus is not already a bottleneck. Swap trades a GPU-local operation for a cross-bus dependency, and cross-bus dependencies are where tail latency goes to hide.

## 8. Scheduling policies: FCFS, priority, deadline-aware, fair

Everything so far assumed the scheduler admits requests in arrival order — first-come-first-served (FCFS), which is vLLM's default. FCFS is simple, it is work-conserving, and it maximizes raw throughput because it never idles a resource waiting for a "better" request. But it has one glaring weakness that the intro already exposed: **head-of-line blocking**. One giant prompt at the front of the queue can hold up everyone behind it, and one greedy tenant can monopolize capacity while others starve.

The alternatives trade a little of FCFS's throughput for control over *who* gets served *when*. The matrix figure compares the four policies across the axes that matter — p99 TTFT, throughput, starvation risk, and the situation each one fits.

![Matrix comparing FCFS, priority, SLO-aware, and fair scheduling policies across p99 TTFT, throughput, starvation risk, and best-fit situation](/imgs/blogs/request-scheduling-and-preemption-3.webp)

**Priority scheduling** orders the queue by an explicit priority number instead of arrival time. This is the right tool when you have tiers — paid versus free users, interactive versus batch jobs, a latency-critical health check versus a bulk backfill. vLLM supports it directly:

```python
from vllm import LLM, SamplingParams

# Enable priority scheduling; lower priority value = scheduled sooner.
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    scheduling_policy="priority",    # "fcfs" (default) | "priority"
    max_num_seqs=256,
    enable_chunked_prefill=True,
)

params = SamplingParams(temperature=0.0, max_tokens=256)

# priority is passed per request; paid tier jumps ahead of free tier.
paid_req  = llm.generate(["urgent user query"],  params, priority=0)
free_req  = llm.generate(["background summary"], params, priority=10)
```

The danger with pure priority is **starvation**: if high-priority traffic never lets up, low-priority requests can wait forever. Production priority schedulers guard against this with *aging* — a request's effective priority slowly improves the longer it waits, guaranteeing it eventually runs. If you roll your own priority layer in a gateway (common when the engine's built-in support is too coarse), build aging in from day one; it is the difference between "the free tier is slow under load" and "the free tier is down."

#### Worked example: priority with aging prevents starvation

You run two tiers on one engine: paid (priority 0) and free (priority 10). At peak, paid traffic alone consumes 90% of capacity. Under pure priority, the free tier's requests sit at the back of the queue and — because paid work keeps arriving — some never get admitted; their effective wait diverges and they time out. Add linear aging: a request's effective priority improves by 1 for every 2 seconds it has waited. A free-tier request that has waited 20 seconds now has effective priority $10 - 20/2 = 0$, tying it with fresh paid work, and after 22 seconds it *beats* fresh paid work and is guaranteed admission. The knob is the aging rate: too slow and free requests still starve under sustained load; too fast and you erode the paid tier's latency advantage that the tiering existed to provide. A reasonable starting point is to set the aging rate so that a request reaches top priority just inside its timeout budget — if free-tier requests time out at 30 seconds, age them to top priority by ~25 seconds, leaving a margin. Then confirm with the per-tier completion-rate metric from section 6: paid p99 should stay low while free-tier *completion rate* stays at 100%, even if free-tier latency is high. Starvation is a completion-rate failure, not a latency one, and only the per-tier view catches it.

**SLO-aware / deadline-aware scheduling** goes further: each request carries a deadline (derived from its SLO), and the scheduler orders by earliest deadline first (EDF), preempting to keep in-flight requests on track to meet their targets. This is the approach high-concurrency production systems converge on when they must guarantee, say, "95% of interactive requests get first token within 500 ms" while still draining batch work in the background. It extracts the most SLO-compliance per GPU, at the cost of a more complex scheduler and a modest throughput hit from the reordering and preemption. This is the frontier of production LLM scheduling and an active research area.

**Fair scheduling** (weighted fair queuing, as covered in depth in the fairness-focused posts of this series) targets multi-tenant environments where the goal is not speed but *isolation*: no tenant should be able to degrade another, regardless of how much they submit. Each tenant gets a weighted share of capacity, and a tenant that floods the queue only slows itself. Fairness costs the most throughput of the four (the scheduler sometimes idles capacity a greedy tenant could have used) but it is the only policy that makes a shared cluster's behavior *predictable per tenant*.

| Policy | Ordering key | Throughput | Tail latency control | Starvation guard | Use when |
|---|---|---|---|---|---|
| FCFS | Arrival time | Highest | Weak (HOL blocking) | N/A | Uniform prompts, single tenant |
| Priority | Priority number | ~5% lower | Strong for top tier | Needs aging | Tiered users/jobs |
| SLO / deadline (EDF) | Deadline | ~8% lower | Strongest | Bounded by deadline | Hard latency SLOs |
| Fair (WFQ) | Weighted share | ~10% lower | Per-tenant | Built-in | Multi-tenant isolation |

(The percentages are illustrative of the general ordering, not universal constants — the throughput cost of a policy depends heavily on your traffic skew.)

## 9. Admission control: shedding load before the engine melts

Little's Law told us there is a hard throughput ceiling and a bounded queue depth beyond which admitting more work only guarantees SLO violations. Admission control is the mechanism that enforces that bound: when the system is saturated, *reject* new requests fast (HTTP 429, "Too Many Requests") instead of accepting them into a queue where they will time out anyway. A fast rejection lets the client retry, back off, or fail over — all far better outcomes than a request that sits in a queue for 30 seconds and then returns a token stream the user has already given up on.

The cleanest place to implement admission control is a thin gateway in front of the engine, because it keeps the policy out of the hot inference path and lets you shed load using cheap, gateway-local signals: the number of in-flight requests and the engine's reported queue depth. Here is a FastAPI gateway that fronts a vLLM OpenAI-compatible server and sheds load based on a concurrency limit derived from Little's Law:

```python
import asyncio
import time
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()
ENGINE = "http://localhost:8000"          # vLLM OpenAI-compatible server

# From Little's Law: max in-flight = throughput_ceiling * max_acceptable_latency.
# e.g. 45 req/s * 1.5 s tolerable queue wait ~= 68; add headroom, cap at running + queue.
MAX_INFLIGHT = 68
_inflight = 0
_lock = asyncio.Lock()

@app.middleware("http")
async def admission_control(request: Request, call_next):
    global _inflight
    if request.url.path.startswith("/v1/"):
        async with _lock:
            if _inflight >= MAX_INFLIGHT:
                # Shed load: reject fast with a Retry-After hint for backoff.
                return JSONResponse(
                    status_code=429,
                    content={"error": "server at capacity, retry shortly"},
                    headers={"Retry-After": "1"},
                )
            _inflight += 1
        try:
            return await call_next(request)
        finally:
            async with _lock:
                _inflight -= 1
    return await call_next(request)

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{ENGINE}/v1/chat/completions", content=body,
                              headers={"content-type": "application/json"})
    return Response(content=r.content, status_code=r.status_code,
                    media_type="application/json")
```

Two refinements make this production-grade. First, prefer a *dynamic* limit over a static `MAX_INFLIGHT`: scrape the engine's `vllm:num_requests_waiting` and `vllm:num_requests_running` from its Prometheus endpoint and shed when the waiting queue exceeds your Little's-Law bound, so the gateway adapts as residence time drifts. Second, distinguish load-shedding (429, transient, retry) from rate-limiting (per-tenant quotas, covered by token buckets) — they are different policies serving different goals, and conflating them produces confusing client behavior. The [SLA and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) post details which engine metrics to scrape and how to alert on them.

#### Worked example: choosing the shed threshold

Suppose measurement gives you $\lambda_{\max} = 45$ req/s and your TTFT SLO tolerates 1.5 s of queue wait. Little's Law says the queue can hold $45 \times 1.5 \approx 68$ requests before the request at the back will miss SLO. So set the shed threshold at ~68 in-flight (running + waiting). If traffic spikes to 90 req/s, the gateway now rejects roughly half of arrivals immediately — and critically, the requests it *does* admit still meet SLO, whereas an unbounded queue would have caused *100%* of requests to miss SLO as the backlog grew. Shedding load is how you keep the requests you accept fast. It feels counterintuitive to reject users on purpose, but the alternative is a system where everyone is slow and no one is served well.

One trap to avoid: load shedding without coordinated client backoff creates a *retry storm*. If every rejected client immediately retries, your shed 429s become new arrivals within milliseconds, and the offered load *increases* exactly when the system is already saturated — a positive feedback loop that turns a brief overload into a sustained outage. The fixes are standard but non-negotiable at scale. Emit a `Retry-After` header and honor it on the client. Use exponential backoff *with jitter* so retries spread out in time instead of synchronizing into a thundering herd. Cap the retry count so a doomed request gives up rather than hammering forever. And on the server side, prefer shedding the *newest* arrivals (they have waited least and lost least) over killing in-flight work you have already invested compute in — the same last-in-first-out logic that governs preemption victim selection in section 6. Load shedding and retry policy are two halves of one mechanism; shipping the first without the second is how a load-shedding feature designed to prevent outages ends up causing one.

## 10. Head-of-line blocking and the fairness corner

It is worth pulling head-of-line (HOL) blocking out into its own discussion, because it is the failure mode that most often surprises teams and it sits at the intersection of everything above. HOL blocking is what happens when one expensive item at the front of a FIFO queue delays every cheaper item behind it. In LLM serving it takes two forms.

The *prefill* form is the intro's disaster: one 28k-token prompt occupies a whole step (or, with chunked prefill, a run of steps with big chunks) and stalls decode for everyone. Chunked prefill is the direct mitigation — it caps how much a single prefill can hog per step — which is why enabling it is the first thing to do when a few long prompts are wrecking your TPOT tail.

The *queue* form is subtler: even with chunked prefill, if a giant prompt sits at the head of an FCFS waiting queue and the engine cannot admit it (not enough free blocks to even start), it can block admission of small requests behind it that *would* fit. Some schedulers mitigate this by allowing small requests to skip ahead of a blocked large one; priority and deadline-aware policies sidestep it by not using arrival order at all. If you observe that TTFT p99 is dominated by a handful of very long prompts while short prompts occasionally wait behind them, this is your signal to move off pure FCFS.

This is where the fourth corner of the trade — **fairness** — matters. The classic serving triangle is latency, throughput, and cost, but in any multi-tenant or multi-workload system, fairness is a first-class concern, and it is fundamentally at odds with pure throughput maximization. A throughput-optimal scheduler is happy to starve a tenant if doing so keeps the tensor cores full; a fair scheduler deliberately leaves some throughput on the table to guarantee isolation. There is no setting that maximizes throughput, minimizes tail latency, and guarantees fairness simultaneously — you pick a point in that space, and the scheduling policy is how you encode the choice. Naming the trade explicitly, and choosing deliberately rather than by default, is the mark of a team that has been paged by the alternative.

## 11. Tuning the two knobs in practice

We have derived how `max_num_seqs` and `max_num_batched_tokens` affect throughput, TTFT, TPOT, and preemption. Now let us set them empirically, because the theory tells you the *shape* of the trade but only measurement tells you the *numbers* for your model, hardware, and traffic.

The method is a sweep: fix everything else, vary one knob, and measure p99 TTFT, p99 TPOT, and throughput at your target load. Here is a benchmark harness that sweeps `max_num_seqs` against a fixed request trace and reports the percentiles, using vLLM's async engine so it exercises the real scheduler under concurrency:

```python
import asyncio, time, numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def run_trace(max_num_seqs, prompts, arrival_rate):
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
    ))
    ttfts, tpots = [], []

    async def one(prompt, rid):
        params = SamplingParams(temperature=0.0, max_tokens=256)
        t0 = time.perf_counter()
        first_tok_t, n_tok, last_t = None, 0, t0
        async for out in engine.generate(prompt, params, request_id=str(rid)):
            now = time.perf_counter()
            if first_tok_t is None:
                first_tok_t = now
            n_tok, last_t = len(out.outputs[0].token_ids), now
        ttfts.append((first_tok_t - t0) * 1000)                      # ms
        if n_tok > 1:
            tpots.append((last_t - first_tok_t) / (n_tok - 1) * 1000)  # ms/token

    tasks = []
    for rid, prompt in enumerate(prompts):
        tasks.append(asyncio.create_task(one(prompt, rid)))
        await asyncio.sleep(1.0 / arrival_rate)   # open-loop Poisson-ish arrivals
    t_start = time.perf_counter()
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start

    return {
        "max_num_seqs": max_num_seqs,
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "p99_tpot_ms": float(np.percentile(tpots, 99)),
        "throughput_rps": len(prompts) / wall,
    }

async def main(prompts):
    for mns in [16, 32, 64, 128, 256]:
        print(await run_trace(mns, prompts, arrival_rate=40))

# asyncio.run(main(load_your_prompt_trace()))
```

Run this against a *representative* trace — real prompt-length and output-length distributions, at your real arrival rate — not a synthetic uniform load, because the whole point of the sweep is to find where *your* traffic hits the preemption wall. The result you are looking for is the classic knee-shaped curve: throughput rises with `max_num_seqs` up to the point where KV-cache pressure triggers preemption thrash, then throughput *falls* and TTFT explodes as the engine spends its cycles evicting and restoring. The matrix figure summarizes the shape of what you will find on an H100.

![Matrix showing how max_num_seqs and max_num_batched_tokens settings trade p99 TTFT, TPOT, throughput, and the failure mode each extreme hits](/imgs/blogs/request-scheduling-and-preemption-8.webp)

For the second knob, an A/B on chunked-prefill chunk size (`max_num_batched_tokens`) is the complement: hold `max_num_seqs` fixed, sweep the budget, and watch TTFT fall and p99 TPOT rise as the budget grows.

```python
# A/B the token budget: bigger budget -> lower TTFT, higher (spikier) TPOT.
import asyncio
async def ab_budget(prompts):
    results = []
    for budget in [512, 2048, 8192]:
        # (reuse run_trace, overriding max_num_batched_tokens=budget)
        results.append(await run_trace_with_budget(budget, prompts, arrival_rate=40))
    for r in results:
        print(f"budget={r['budget']:>5}  "
              f"p99_TTFT={r['p99_ttft_ms']:.0f}ms  "
              f"p99_TPOT={r['p99_tpot_ms']:.0f}ms  "
              f"tput={r['throughput_rps']:.1f} rps")
# Expected shape:
#   budget=  512  p99_TTFT=610ms  p99_TPOT=24ms  tput=3.2 rps-equivalent
#   budget= 2048  p99_TTFT=310ms  p99_TPOT=31ms  tput=3.8 rps-equivalent
#   budget= 8192  p99_TTFT=210ms  p99_TPOT=48ms  tput=3.6 rps-equivalent
```

The named-hardware before-and-after below shows what disciplined tuning buys, measured on a single H100 80GB serving Llama-3.1-8B with a chat-like trace (median prompt ~1.2k tokens, median output ~200 tokens). "Before" is a common misconfiguration — chunked prefill off, `max_num_seqs` cranked to the max in a throughput-chasing reflex. "After" is the tuned config from this post.

| Metric (H100 80GB, Llama-3.1-8B, chat trace) | Before (untuned) | After (tuned) | Change |
|---|---|---|---|
| Config | chunked prefill off, `max_num_seqs`=512 | chunked prefill on, `max_num_seqs`=128, budget=2048 | — |
| p99 TTFT | 1,850 ms | 420 ms | −77% |
| p99 TPOT | 190 ms | 34 ms | −82% |
| Throughput | 3,100 tok/s | 4,050 tok/s | +31% |
| Preemptions / min | ~140 | ~6 | −96% |
| GPU utilization | 71% (thrashing) | 93% | +22 pts |

The counterintuitive result — *lowering* `max_num_seqs` from 512 to 128 *raised* throughput by 31% — is the preemption wall in action. At 512 the engine over-committed KV memory, thrashed on preemption, and burned cycles moving sequences around instead of decoding them. At 128 it stopped over-committing, preemptions collapsed, and every cycle went to useful work. This is the most common tuning mistake in LLM serving, and it is why "just turn the concurrency up" is bad advice. (Numbers here are representative of the effect and its direction; measure your own trace — the exact figures depend on prompt/output distributions and model.)

#### Worked example: reading the knee in a sweep

Say your `max_num_seqs` sweep on the H100 above produces this throughput series: 16 → 1.4k tok/s, 32 → 2.4k, 64 → 3.6k, 128 → 4.05k, 256 → 3.7k, 512 → 3.1k. Read it the way you would read a Little's-Law curve hitting a resource wall. From 16 to 128, throughput climbs because you are amortizing weight-loading across more decoders — the memory bus was underutilized and more concurrency fills it. At 128 you hit the knee: KV-cache blocks are now the binding constraint, not compute. Past 128, adding sequences does not add useful work — it adds *preemptions*, and each preemption is negative work (evict, later restore or recompute). By 512 the engine spends a third of its cycles shuffling sequences instead of decoding them, and throughput has fallen 24% below the peak. The correct operating point is *just below* the knee — here, 128 — not at it and certainly not past it, because you want headroom for the traffic bursts that the M/M/1 blow-up (section 3) warns are lurking. A useful sanity check: at your chosen `max_num_seqs`, `vllm:gpu_cache_usage_perc` under peak load should sit around 85–90%, not pinned at 100%. If it is pinned, you are on the wrong side of the knee and a burst will tip you into thrash. Re-run the sweep whenever your prompt-length or output-length distribution shifts materially — the knee moves with the workload, and last quarter's optimal `max_num_seqs` can be this quarter's thrash.

## 12. Prefill/decode interference and the disaggregation alternative

Everything so far runs prefill and decode on the *same* GPU, interleaved by the scheduler into mixed batches. That co-location is exactly what makes chunked prefill and the token-budget math necessary — the two phases interfere because they compete for the same forward pass. There is a fundamentally different architecture that removes the interference at the root: **prefill/decode (PD) disaggregation**, where prefill runs on one pool of GPUs and decode on another, and the KV cache is handed off between them over a fast interconnect.

The motivation is the phase asymmetry from section 1. Prefill is compute-bound and wants large dense batches at high FLOP utilization; decode is memory-bandwidth-bound and wants high concurrency at high bandwidth utilization. When both share a GPU, every knob is a compromise — the token budget that is right for prefill is wrong for decode, and vice versa. Disaggregation lets each pool be tuned for its own phase, and a prefill spike on the prefill pool can no longer stall decode on the decode pool, because they are not on the same device. You have replaced *temporal* interference (fighting over steps) with a clean *spatial* separation.

The cost is a KV-cache transfer. When a prefill worker finishes a prompt, it must ship that prompt's entire KV cache — potentially gigabytes for a long prompt on a large model — to the decode worker that will generate the response, typically over NVLink or InfiniBand using NCCL peer-to-peer transfer. That transfer is pure overhead, and it only amortizes when you have enough traffic to keep both pools saturated and a network fast enough to move KV cheaply. The consensus from teams who have deployed PD disaggregation in production is a rule of thumb: it is worth the substantial operational complexity above roughly 100 QPS on large models with a fast fabric, and it is a distraction below that. For the co-located case that covers the overwhelming majority of deployments, the mixed-batch scheduling in this post *is* the answer and chunked prefill *is* the interference mitigation — disaggregation is the tool you reach for only when a single scheduler on a single device can no longer serve both phases well. The transfer mechanics, the interconnect requirements, and the exact QPS threshold are a topic in their own right, covered by the large-scale-serving posts later in this series.

## 13. Observing the scheduler in production

You cannot tune what you cannot see, and the scheduler's health lives in a handful of metrics that vLLM — and every serious engine — exports to Prometheus. Four matter most:

- `vllm:num_requests_running` — sequences in the running set. Compare it to `max_num_seqs`; if it sits pinned at the cap, you are concurrency-limited and raising the cap (memory permitting) will help.
- `vllm:num_requests_waiting` — queue depth. This is the $L_q$ from Little's Law; a persistently rising value means arrivals exceed capacity, and it is the trigger for load shedding.
- `vllm:num_preemptions_total` — cumulative preemptions. Its *rate* is your thrash detector.
- `vllm:gpu_cache_usage_perc` — fraction of KV-cache blocks in use. Sustained values near 100% are the precursor to preemption.

Wire them into alerts that map one-to-one onto the failure modes from this post. A few PromQL expressions that have earned their place on a serving dashboard:

```promql
# Preemption thrash: more than ~5 preemptions/sec sustained -> lower max_num_seqs.
rate(vllm:num_preemptions_total[1m]) > 5

# Queue building faster than it drains -> arrivals exceed capacity, shed load.
vllm:num_requests_waiting > 68            # your Little's-Law-derived bound

# KV cache saturated -> preemption imminent, expect tail-latency growth.
avg_over_time(vllm:gpu_cache_usage_perc[2m]) > 0.95

# p99 TTFT SLO breach, straight from the engine's histogram.
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 0.8

# p99 TPOT (inter-token latency) SLO breach — the metric chunked prefill protects.
histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m])) > 0.05
```

The discipline that separates teams who tune well from teams who tune by folklore: every knob change is a hypothesis, and these metrics are how you falsify it. Lowered `max_num_seqs` to fix thrash? The preemption rate should fall and throughput should *rise* — if it does not, your diagnosis was wrong and you should revert. Raised `max_num_batched_tokens` to cut TTFT? The TTFT histogram should shift left and the TPOT histogram should shift right — confirm both moved, because the second is the price you paid for the first, and a change that only moved one is not the change you thought it was. Treat the dashboard as the scoreboard for the theory in sections 3 and 4; the [SLA and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) post goes deeper on histogram buckets, exemplars, and SLO burn-rate alerting.

## 14. Scheduling across serving engines

The concepts in this post are universal, but the knobs and defaults differ across engines, and knowing the mapping saves hours when you switch stacks. vLLM's scheduler — three queues, a PagedAttention block manager, recompute-default preemption, FCFS or priority policy, chunked prefill — is the reference implementation this post is built around. The others implement the same ideas under different names:

- **Text Generation Inference (TGI)** calls its continuous batching "in-flight batching" and exposes the token budget as `--max-batch-prefill-tokens`, with concurrency bounded by `--max-batch-total-tokens` and `--max-concurrent-requests`. It has a waiting-queue admission stage and its own prefill/decode interleaving; the same intuition transfers directly, only the flag names change. TGI's own deep dive lives in [text generation inference deep dive](/blog/machine-learning/model-serving/text-generation-inference-deep-dive).
- **TensorRT-LLM** (served via Triton's TensorRT-LLM backend) popularized the term "in-flight batching" and pairs it with a paged KV cache. Its batch scheduler is configurable through `max_num_sequences` and `kv_cache_free_gpu_mem_fraction`, and it leans on the TensorRT compiler to fuse each mixed-batch step into a small number of highly optimized kernels.
- **SGLang** adds RadixAttention — a prefix-sharing KV scheme — and a scheduler that is prefix-cache-aware, so it will preferentially co-schedule requests that share a prefix to maximize reuse. This is scheduling *in service of the memory system*, a twist the plain co-located mixed-batch model does not have out of the box.

| Engine | Continuous-batching term | Token-budget knob | Concurrency knob | Preemption |
|---|---|---|---|---|
| vLLM | Iteration-level scheduling | `max_num_batched_tokens` | `max_num_seqs` | Recompute (default) / swap |
| TGI | In-flight batching | `--max-batch-prefill-tokens` | `--max-concurrent-requests` | Queue-based backpressure |
| TensorRT-LLM | In-flight batching | batch-scheduler config | `max_num_sequences` | Paged KV eviction |
| SGLang | Continuous batching | `--chunked-prefill-size` | `--max-running-requests` | Recompute + RadixAttention |

The practical takeaway: when you evaluate or migrate engines — a decision covered in [choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack) — do not re-learn scheduling from scratch. Map the four concepts (token budget, concurrency cap, preemption mode, policy) onto the new engine's flags and your intuition carries over intact. Only the defaults and the edge-case behaviors differ, and those are precisely what the benchmark sweep from section 11 is designed to expose.

## 15. Case studies and benchmarks

Three systems anchor the ideas in this post, and reading their papers is the fastest way to go deeper. All figures below are as reported by the respective authors; treat cross-system comparisons as directional, since hardware and workloads differ.

**Orca (Yu et al., OSDI 2022) — iteration-level scheduling.** Orca introduced the core idea that makes everything else possible: scheduling at the granularity of a single iteration (one decode step) rather than one request. Before Orca, serving systems batched at request granularity and suffered the "convoy" problem where a batch waits for its slowest member. Orca's iteration-level scheduling — later popularized as *continuous batching* — lets finished sequences leave and new ones join the batch every step, plus *selective batching* to handle the fact that different sequences are at different positions. Orca reported throughput improvements of up to ~36× over then-current systems (FasterTransformer) at comparable latency. This is the paper that turned LLM serving from "batch inference" into "streaming inference."

**vLLM / PagedAttention (Kwon et al., SOSP 2023) — the memory manager and its scheduler.** vLLM's contribution was to eliminate KV-cache fragmentation with PagedAttention (non-contiguous, block-based KV storage), which in turn let the scheduler run at far higher concurrency without wasting memory on internal fragmentation. Critically for this post, vLLM's scheduler is where preemption-by-recompute-or-swap was made a first-class, tunable behavior: when blocks run out, it preempts, and it defaults to recompute because it is robust and avoids the PCIe dependency. vLLM reported 2–4× throughput gains over Orca-style systems at the same latency, driven almost entirely by being able to pack more sequences into the same memory. The scheduler internals covered here map directly onto vLLM's `Scheduler` class and its three-queue structure.

**Sarathi-Serve (Agrawal et al., OSDI 2024) — chunked prefills and stall-free batching.** Sarathi-Serve tackled exactly the intro's problem: prefill stalling decode. Its two techniques — *chunked prefills* (split a long prompt into fixed-size chunks) and *stall-free batching* (piggyback decode onto the chunk steps so decode never freezes) — are the direct ancestors of `enable_chunked_prefill`. The paper showed that this decouples the TTFT/TPOT trade, letting operators hold TPOT (they use the term "TBT," time-between-tokens) nearly flat while accepting a small, bounded TTFT increase — and reported serving-capacity improvements of up to ~2.6× under tight latency SLOs on models like Mistral-7B and Llama-70B versus vLLM's then-current default. If you read one paper from this list to understand modern scheduling, read this one.

The reported headline numbers, side by side, make the progression concrete:

| System (year) | Key scheduling idea | Reported gain (as published) | Baseline compared against |
|---|---|---|---|
| Orca (OSDI 2022) | Iteration-level scheduling + selective batching | up to ~36× throughput at similar latency | FasterTransformer (request-level batching) |
| vLLM / PagedAttention (SOSP 2023) | Paged KV + high-concurrency scheduler with preemption | 2–4× throughput at same latency | Orca-style continuous batching |
| Sarathi-Serve (OSDI 2024) | Chunked prefills + stall-free batching | up to ~2.6× serving capacity under tight SLOs | vLLM default of that period |

These multiply rather than compete — vLLM built on Orca's iteration-level scheduling, and Sarathi-Serve's chunked prefill is now a feature *inside* vLLM. Treat the figures as directional (each paper used its own hardware, models, and traffic, so they are not apples-to-apples), but the direction is unmistakable.

The through-line across all three: each advance moved a decision from *coarse* (per request, per batch) to *fine* (per iteration, per token, per chunk), and each fine-grained decision let the scheduler keep the GPU busier without sacrificing latency. It is the same lesson the operating-systems community learned decades ago — finer-grained preemptive scheduling beats coarse batch processing for interactive workloads — arriving now in the GPU-serving world with token-level granularity and a KV-cache memory wall as the twist. That is the arc of LLM serving, and the scheduler is where it plays out.

## When to use this (and when not to)

Scheduling is not a feature you bolt on; it is inherent in any LLM serving engine. But *how much scheduler tuning is worth your time* varies enormously.

**Invest in scheduling when:**

- You serve interactive traffic (chat, coding assistants, agents) where TPOT smoothness is user-visible. Enable chunked prefill, full stop — it is the highest-leverage single change for these workloads.
- You run at meaningful concurrency (dozens to thousands of simultaneous sequences). The `max_num_seqs` sweep and the preemption-wall discovery routinely find 20–40% throughput on the table.
- You have mixed prompt lengths, especially any long prompts sharing an engine with short interactive ones. Chunked prefill and a non-FCFS policy prevent the long prompts from wrecking the short ones' tail latency.
- You have multiple tenants or tiers. Priority or fair scheduling is the difference between graceful degradation and a noisy-neighbor incident.
- You are near capacity. Admission control (429 load shedding) keeps the requests you accept within SLO instead of letting an unbounded queue take everyone down.

**Do not over-invest when:**

- You run a single low-QPS internal service (a few requests per second, one tenant). FCFS with defaults is fine; spending a week tuning knobs to shave 10 ms off p99 that no one measures is misallocated effort. Ship it and move on.
- Your workload is pure offline batch inference with no latency SLO. Crank `max_num_seqs` for maximum throughput, ignore TTFT entirely, and do not bother with chunked prefill's latency-smoothing (it can even cost you a little throughput). The scheduling considerations here are about *interactive* serving.
- Your prompts are short and uniform (e.g., classification, embedding-style tasks). HOL blocking and prefill stalls barely occur; the exotic policies buy you little over FCFS.
- You have not yet measured. The cardinal sin is tuning by folklore. Every recommendation in this post is conditional on your traffic; the sweep script is more valuable than any specific number.

A blunt heuristic: if you cannot state your p99 TTFT SLO, your p99 TPOT SLO, and your arrival rate, you are not ready to tune the scheduler — you are ready to *measure*, which is section 9 and 11's job.

## Key takeaways

- **The scheduler runs every step and makes three decisions**: admit new work from the waiting queue, continue the running sequences, or preempt when KV-cache memory is exhausted. Throughput and latency both live or die on this loop.
- **Little's Law is your capacity planner.** $\lambda_{\max} = \texttt{max\_num\_seqs} / \bar{W}$ gives your throughput ceiling; $L_q = \lambda W_q$ gives the queue depth beyond which you must shed load. Latency blows up non-linearly near saturation, so provision headroom.
- **`max_num_batched_tokens` and `max_num_seqs` are the two knobs.** The budget trades TTFT against TPOT; the sequence cap trades throughput against preemption risk. They pull latency in opposite directions — set them from your SLO, not a blog post's constant.
- **Chunked prefill is the default win for interactive serving.** It stops one long prompt from freezing every decoder, trading a small, bounded TTFT increase for a dramatic collapse of the TPOT tail. Turn it on.
- **Preemption is a tail-latency problem.** Occasional preemption is healthy; frequent preemption ("thrash") collapses throughput. If p99 is high while median and GPU utilization look fine, suspect preemption and *lower* `max_num_seqs`.
- **Recompute vs swap: the `s` cancels.** Which preemption mode is cheaper depends on your prefill rate versus PCIe bandwidth relative to KV size per token, not on sequence length. Leave it at the `recompute` default unless you have measured a specific win for swap.
- **Policy encodes priorities.** FCFS maximizes throughput but suffers head-of-line blocking; priority, deadline-aware, and fair scheduling trade a little throughput for tier control, SLO compliance, and multi-tenant isolation respectively. Add aging to any priority scheme.
- **Admission control keeps accepted requests fast.** Shedding load with a fast 429 above a Little's-Law-derived threshold is how you meet SLO under overload — an unbounded queue makes *everyone* slow.
- **Lowering concurrency can raise throughput.** The preemption wall means cranking `max_num_seqs` past your memory budget is counterproductive. Sweep it against a representative trace and find the knee, then operate just below it with headroom for bursts.
- **The mean lies; provision against p99.** Kingman's formula shows wait time scales with the *square* of service-time variability, so mixed prompt and output lengths blow up the tail even at modest utilization. Reducing variability at the source (cap `max_tokens`, split batch jobs onto their own engine) is often a bigger p99 win than any single knob.
- **Instrument the scheduler, then treat every knob change as a falsifiable hypothesis.** Running-set size, queue depth, preemption rate, and KV-cache usage are the four vital signs; a change that does not move them the way your theory predicted is a change you should revert.
- **Load shedding needs matching client backoff.** A 429 without `Retry-After`, jittered exponential backoff, and a retry cap turns a brief overload into a self-sustaining retry storm — the two are one mechanism, not two features.

## Further reading

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022. The origin of iteration-level (continuous) scheduling and selective batching.
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** — Kwon et al., SOSP 2023. The vLLM paper; PagedAttention plus the preempting scheduler.
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** — Agrawal et al., OSDI 2024 (and the earlier SARATHI, 2023). Chunked prefills and stall-free batching, the basis of `enable_chunked_prefill`.
- **vLLM documentation** — the [official docs](https://docs.vllm.ai) for `EngineArgs`, chunked prefill, scheduling policy, and preemption mode; the source of truth for current defaults, which change across versions.
- **Within this series**: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle, [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) for the memory manager the scheduler sits on, [batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) for the batching taxonomy and Little's Law, [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) for the block manager internals, and [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) for the TTFT/TPOT/p99 targets that anchor every tuning decision here.
