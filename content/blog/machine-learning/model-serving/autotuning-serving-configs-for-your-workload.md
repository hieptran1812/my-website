---
title: "Autotuning Serving Configs for Your Workload: Stop Copying Defaults, Search the Space"
date: "2026-07-07"
publishDate: "2026-07-07"
description: "A data-driven method to find the vLLM serving config that maximizes goodput under your p99 SLO — the knobs that matter, the search methodology, and a full autotuning harness you can run."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "vllm",
    "autotuning",
    "performance-tuning",
    "benchmarking",
    "slo",
    "pareto-optimization",
    "optuna",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/autotuning-serving-configs-for-your-workload-1.webp"
---

The incident that made me stop trusting defaults started with a copy-paste. A teammate had stood up a new vLLM deployment for a customer-support chat feature, lifted the launch command straight out of a popular tuning blog — `--max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90` — and shipped it. Load tests at 5 requests per second looked fine. Then marketing ran a campaign, real traffic climbed to 25 requests per second, and the p99 time-to-first-token blew past 900 ms against a 500 ms SLO. Half the sessions were technically "served," but they were served too slowly to count. Effective throughput of requests that actually met the SLO — the number the business cares about — had collapsed to nearly zero, even though the GPU was only 58% utilized. We were paying for an H100 and getting the goodput of a laptop.

The fix was not a bigger GPU. The fix was a two-hour sweep. We replayed a captured trace of our own traffic against forty candidate configs, measured p99 TTFT, p99 inter-token latency, and SLO-meeting throughput for each, and picked the feasible config with the highest goodput. The winner looked nothing like the blog's numbers: `--max-num-seqs 96 --max-num-batched-tokens 2048`, chunked prefill on, prefix caching on. Same hardware, same model, same SLO — 38% more goodput and a p99 TTFT of 420 ms, comfortably inside budget. The default was not "safe." It was just someone else's answer to someone else's workload.

That is the thesis of this post. Every serving config is a point in a high-dimensional trade-off space, and the *optimal* point depends on three things the blog author never knew: your input and output length distribution, your arrival pattern, and which side of the latency-throughput-cost triangle your SLO leans toward. Copying a config is copying a coordinate in someone else's space and hoping it lands well in yours. It usually does not. The disciplined alternative is to treat config selection as what it actually is — a constrained optimization problem — and *search* it. The autotuning method has five layers, shown in the figure below: state the objective, enumerate the search space, pick a search strategy, evaluate every candidate on a replayed trace, and select the feasible config with maximum goodput.

![The autotuning method drawn as five layers: a constrained objective at the top, then the search space, then the search strategy, then trace-replay evaluation, then the chosen SLO-feasible max-goodput config at the bottom.](/imgs/blogs/autotuning-serving-configs-for-your-workload-1.webp)

By the end of this post you will be able to: name every serving knob that moves the trade and predict which way it moves it; write the objective down as a formula (maximize goodput subject to a p99 constraint); choose grid, random, coordinate-descent, or Bayesian search based on how many knobs you are tuning; build a Pareto frontier from a sweep and read the right config off it; and run a full autotuning harness in Python that launches vLLM per config, replays a real trace, and hands you the winning flags. This is the recurring spine of the whole series — model, packaging, runtime, server, infrastructure, observability, scale, all balanced on the SLO triangle of latency, throughput, and cost — reduced to a single, answerable question: *for my model, my workload, my GPU, and my SLO, which config wins?* Let us make it answerable.

## 1. The knobs that matter (and which way each one moves the trade)

Before you can search a space you have to know its dimensions. vLLM exposes dozens of flags; only about a dozen meaningfully move the latency-throughput-memory trade for a fixed model on a fixed GPU. Here is the working set, grouped by what they control, with the direction each one pushes.

**Concurrency and batch shape.** `--max-num-seqs` caps how many sequences the scheduler will run concurrently in a single decode batch. Raising it packs more requests into each GPU step, which raises throughput and GPU utilization — up to the point where the KV cache can no longer hold that many active sequences, at which point vLLM starts preempting and recomputing, and your tail latency detonates. `--max-num-batched-tokens` caps the total number of tokens processed in one scheduler step (prefill tokens plus decode tokens). It is the single most important throughput lever for prefill-heavy workloads: a bigger token budget lets vLLM stuff more prompt tokens into each forward pass, raising prefill throughput, but it also lengthens each step, which is exactly the step that a decode-phase request is waiting behind — so raising it can hurt inter-token latency for concurrent generations.

**Memory budget.** `--gpu-memory-utilization` (default 0.90) sets the fraction of each GPU's VRAM that vLLM is allowed to claim for the KV cache after loading weights and reserving activation scratch. Raise it and you get a larger KV pool, which directly means more concurrent sequences or longer contexts before eviction. Raise it too far and you leave no headroom for activation spikes, CUDA-graph memory, or fragmentation, and the server OOMs under load — usually at peak traffic, which is the worst possible time. `--kv-cache-dtype` lets you store the KV cache in FP8 (`fp8`, `fp8_e5m2`, or `fp8_e4m3`) instead of the model's native FP16/BF16. FP8 halves the bytes per cached token, roughly *doubling* the number of tokens the same pool can hold, at the cost of a small accuracy hit (typically well under a percent of perplexity on most models). `--block-size` sets the granularity of the paged KV allocator (commonly 16 tokens per block); smaller blocks reduce internal fragmentation for short sequences but add scheduling overhead.

**Prefill scheduling.** `--enable-chunked-prefill` splits a long prefill into fixed-size chunks that are interleaved with ongoing decode steps, so a single 8k-token prompt no longer monopolizes the GPU and stalls everyone else's token stream. It trades a slightly higher TTFT on the chunked request for a much steadier TPOT across the batch. The chunk size is governed by `--max-num-batched-tokens` when chunked prefill is on — the two knobs are the *same* knob wearing two hats, which is why they must be tuned together. `--max-model-len` caps the maximum sequence length; setting it lower than the model's trained context frees KV memory (each slot reserves fewer blocks) and is one of the highest-leverage, most-overlooked knobs when your real prompts are short.

**Reuse and parallelism.** `--enable-prefix-caching` hashes and reuses the KV blocks of shared prompt prefixes across requests, so a common system prompt is computed once and reused, cutting TTFT dramatically on cache hits. Its value is entirely workload-dependent: if your requests share long prefixes (system prompts, few-shot examples, RAG boilerplate) it is a massive win; if every prompt is unique it is pure overhead. `--tensor-parallel-size` shards the model across GPUs; beyond fitting a large model, TP frees per-GPU VRAM for a bigger KV pool, but at low batch sizes the extra GPUs sit under-utilized and you have simply lit money on fire. Two more matter at the margins: the attention backend (selected via `VLLM_ATTENTION_BACKEND` — `FLASH_ATTN`, `FLASHINFER`, `XFORMERS`), which changes the kernel and its throughput/latency profile depending on batch and sequence shape; and CUDA-graph capture sizes (disabled entirely by `--enforce-eager`, tuned via the compilation config's capture-size list), which cut per-step launch overhead for the batch sizes you actually run. Finally, `--scheduling-policy` chooses `fcfs` (default) or `priority`, which changes *which* requests wait when the system is saturated — a fairness/tail knob rather than a throughput knob.

The matrix below is the cheat sheet: for each knob, what raising it buys, what lowering it buys, its primary failure mode, and the neighbor knob it is coupled to. Keep it open while you design a search space — the last column is why you cannot tune these one at a time.

![A matrix of six serving knobs with columns for the effect of raising it, the effect of lowering it, the main trade-off, and the neighboring knob it interacts with, showing that each knob raises one metric while lowering another.](/imgs/blogs/autotuning-serving-configs-for-your-workload-2.webp)

Notice the pattern in that last column. `max-num-seqs` is coupled to `gpu-memory-util` because the KV pool sizes the achievable batch. `max-num-batched-tokens` is coupled to `chunked-prefill` because chunk size *is* the token budget. `kv-cache-dtype fp8` is coupled to `gpu-memory-util` because halving bytes-per-token changes what utilization level is safe. There is no column where a knob stands alone. That is the mathematical heart of why defaults fail and why you must search rather than reason your way to an answer, which we will make precise in Section 3.

## 2. Why the optimal config is workload-specific

The reason a copied config rarely wins is not that the person who published it was careless. It is that the optimum genuinely moves when the workload moves, and their workload was not yours. Three properties of your traffic each shift the sweet spot, often in opposite directions.

**Input and output length distribution.** A workload of 2000-token RAG prompts producing 100-token answers is *prefill-bound*: most of the GPU work is the one-shot prompt encoding, so a large `max-num-batched-tokens` and chunked prefill dominate the tuning. A workload of 40-token chat turns producing 800-token replies is *decode-bound*: the prompt is cheap, generation is the long pole, and `max-num-seqs` (how many generations run in parallel) is what matters, while a huge token budget just adds step latency for no prefill benefit. The same two knobs that are optimal for the first workload are actively harmful for the second. There is no length-agnostic best config because the ratio of prefill FLOPs to decode FLOPs — which decides whether you are compute-bound or memory-bandwidth-bound — is set by the length distribution, not by the flags.

**Arrival pattern.** A steady 20 requests-per-second stream and a bursty pattern that idles then spikes to 60 have different optimal batch shapes even at the same *average* rate. The steady stream can afford a modest `max-num-seqs` because the queue never gets deep; the bursty one needs either more headroom to absorb the spike without preemption, or a priority scheduling policy so that latency-sensitive requests are not stuck behind a burst of batch traffic. Little's Law makes this concrete: for average concurrency $N$, throughput $X$, and mean residence time $W$, ${N = X \cdot W}$. If your arrivals are bursty, the *instantaneous* $X$ spikes, so to hold $W$ (your latency) constant you must allow $N$ (your batch) to spike too — which means provisioning KV pool for the peak, not the average. Tune to the average and the peak preempts; tune to the peak and you waste memory the rest of the day.

**SLO priorities.** Two teams running the identical model on identical hardware will land on different configs if one has a 200 ms TTFT SLO for an interactive assistant and the other has a 5-second budget for an async summarization job. The interactive team must keep batches small and prefill chunked to protect the tail; the batch team should crank `max-num-seqs` and `max-num-batched-tokens` to the memory limit and harvest throughput, because a 4-second p99 is fine. The SLO is not a constraint you bolt on after tuning — it *is* the objective, and it moves the optimum as much as the hardware does. The figure below contrasts the two paths: blindly copying a default that ignores all three properties, versus searching your own replayed trace and landing on a feasible, higher-goodput config on the very same GPU.

![A before-and-after figure contrasting copying a blog default, which misses the p99 SLO and yields near-zero goodput, against searching your own replayed trace to find a feasible config with 38 percent higher goodput on the same GPU.](/imgs/blogs/autotuning-serving-configs-for-your-workload-3.webp)

The practical consequence is uncomfortable but freeing: there is no config you can carry from one deployment to the next and trust. What you *can* carry is the method. The knobs are the same everywhere; the objective is the same everywhere; only the numbers change, and the numbers are exactly what a search recovers cheaply. Stop shopping for a config and start shopping for a search harness.

## 3. The mechanics: goodput, constrained optimization, and the non-separable space

Let us write the objective down precisely, because a vague objective produces a vague sweep. The quantity you actually want to maximize is **goodput**: the rate of requests that are served *and* meet their latency targets. A request that arrives, gets served, but violates the SLO contributes to raw throughput and to your GPU bill, but contributes *nothing* to the business — it is a slow answer the user may have already abandoned. Formalizing goodput is the contribution of the DistServe work (Zhong et al., OSDI 2024), which defines it as the maximum request rate a system can sustain while meeting SLO-attainment goals on both TTFT and TPOT.

Write a config as a vector $c = (c_1, c_2, \dots, c_k)$ over the $k$ knobs (max-num-seqs, max-num-batched-tokens, and so on), drawn from the discrete search space $\mathcal{C} = \mathcal{C}_1 \times \mathcal{C}_2 \times \cdots \times \mathcal{C}_k$. Let $X(c)$ be the sustained throughput at config $c$ under your offered load, and let $p_{99}^{\text{TTFT}}(c)$ and $p_{99}^{\text{TPOT}}(c)$ be the measured tail latencies. The autotuning problem is the constrained optimization:

$$
c^\* = \arg\max_{c \in \mathcal{C}} \; X(c) \quad \text{subject to} \quad p_{99}^{\text{TTFT}}(c) \le \tau_{\text{ttft}}, \;\; p_{99}^{\text{TPOT}}(c) \le \tau_{\text{tpot}}
$$

Equivalently, if you fold the constraint into the metric, you maximize goodput $G(c)$ — throughput counted only over SLO-meeting requests — and the feasible-set restriction becomes automatic. Either framing gives the same winner. What matters is that the SLO is a *hard constraint*, not a term you trade against throughput: a config that is 10% faster in throughput but violates the p99 is not "a little better on one axis," it is *infeasible*, and it scores zero. This is why "highest tokens/sec" benchmarks are misleading for production — they optimize $X$ with the constraint deleted.

**Why the space is not separable.** The seductive shortcut is to tune each knob alone: sweep max-num-seqs holding everything else fixed, lock in the best, sweep the next knob, and so on. This works if and only if the objective is *separable* — if $G(c_1, \dots, c_k)$ decomposes as $\sum_i g_i(c_i)$, so that the best value of $c_i$ does not depend on the others. It does not. The KV memory budget ties max-num-seqs, max-model-len, and gpu-memory-utilization together through one equation. The bytes of KV cache per token are

$$
\text{bytes/token} = 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot b
$$

where $L$ is layers, $h_{kv}$ is key/value heads (small under grouped-query attention), $d_{head}$ is head dimension, $b$ is bytes per element, and the leading ${2}$ counts keys and values. For Llama-3-8B ($L{=}32$, $h_{kv}{=}8$, $d_{head}{=}128$, $b{=}2$) that is $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes — 128 KiB of KV per token, or 2 MiB per 16-token block. The total KV pool is fixed by memory: $\text{pool} = (\text{VRAM} \cdot \text{util}) - \text{weights} - \text{activations}$. On an 80 GB H100 with the 8B model in BF16 (~16 GB of weights) at util 0.90, that leaves roughly 52 GB, or about 400k tokens of KV. Now the coupling is undeniable: the *maximum feasible* max-num-seqs is that token budget divided by the average sequence length, which max-model-len bounds and kv-cache-dtype halves. Raising gpu-memory-utilization raises the budget; switching to FP8 KV doubles the token capacity; lowering max-model-len raises how many sequences fit. You literally cannot pick the best max-num-seqs without knowing the other three. In optimization terms, the mixed partials $\partial^2 G / \partial c_i \partial c_j$ are non-zero, coordinate-wise ascent can stall in a ridge, and you must search the *joint* space, at least over the coupled clusters.

#### Worked example: computing the feasible max-num-seqs from the KV budget

Make that coupling numeric. Take Llama-3-8B in BF16 on an H100 80GB. Weights are about 16 GB; reserve roughly 4 GB for activation scratch, CUDA-graph buffers, and fragmentation headroom. The KV pool is whatever gpu-memory-utilization leaves after those two: at util 0.90 the server may claim 72 GB, so the pool is about ${72 - 16 - 4 = 52}$ GB. Each token costs 128 KiB of KV in BF16 (the 131,072 bytes we computed above), so one gigabyte of pool holds 8,192 tokens and 52 GB holds roughly 426k tokens. The maximum number of sequences you can run concurrently is that token capacity divided by the KV footprint of an average active sequence — its prompt length plus the tokens it has generated so far. The table makes the four-way link between utilization, KV dtype, context length, and concurrency impossible to miss:

| KV dtype | gpu-mem-util | KV pool | token capacity | max-num-seqs @ 512-tok avg | @ 2048-tok avg |
|---|---|---|---|---|---|
| BF16 | 0.85 | ~48 GB | ~393k | ~768 | ~192 |
| BF16 | 0.90 | ~52 GB | ~426k | ~832 | ~208 |
| FP8 e4m3 | 0.90 | ~52 GB | ~852k | ~1660 | ~416 |
| FP8 e4m3 | 0.95 | ~56 GB | ~918k | ~1790 | ~448 |

Read across the rows and the non-separability is arithmetic, not opinion. At BF16 and util 0.90, a workload whose sequences average 2,048 tokens of context can sustain about 208 concurrent sequences before the pool is exhausted — so a copied `--max-num-seqs 256` is already over the ceiling, and the scheduler will preempt and recompute the moment the batch fills. Switch the KV cache to FP8 and the same pool holds twice the tokens, lifting the ceiling to about 416 and making 256 comfortable. Halve the average context — shorter prompts, or a lower `--max-model-len` — and the ceiling doubles again. You cannot name the best max-num-seqs on its own: its feasible range is a function of three other knobs, and every one of them shifts the row you are reading. This is why the harness in Section 8 computes a feasibility bound before it sweeps; there is no point spending GPU-minutes on configs the memory arithmetic already rules out.

**The Pareto frontier.** When you sweep configs and plot each as a point in (latency, throughput) space, most points are *dominated*: some other config is at least as fast on latency and at least as high on throughput. A config $c$ dominates $c'$ when $X(c) \ge X(c')$ and $\text{latency}(c) \le \text{latency}(c')$ with at least one inequality strict. The **Pareto frontier** is the set of non-dominated configs — the ones where you cannot improve throughput without giving up latency. Every config worth considering lives on that frontier; everything behind it is strictly worse and should be discarded. The tuning decision reduces to a one-dimensional choice *along* the frontier: walk it from the low-latency end toward higher throughput, and stop at the last point still inside your p99 SLO. That point is $c^\*$. We will build and read this frontier in Section 6; for now, hold onto the definition, because it turns a scary $k$-dimensional search into "find the frontier, then slide to the SLO line."

## 4. The search methodology: objective, space, strategy, evaluation

With the objective pinned down, the search itself is a loop. For each candidate config the harness boots a fresh vLLM instance, replays a representative trace against it, measures p99 latencies and goodput, checks feasibility against the SLO, keeps the config if it is both feasible and the best goodput so far, discards it otherwise, and returns the best after the sweep completes. The figure shows the branch that makes it a *search* and not just a benchmark: the SLO-feasibility test, which routes every candidate to either "keep as best" or "discard."

![A flow graph of the autotuning loop: define the search space, pull the next config, launch vLLM and replay the trace, measure p99 and goodput, then branch on whether the SLO is feasible, keeping the config if feasible and higher-goodput or discarding it, and returning the best config after the sweep.](/imgs/blogs/autotuning-serving-configs-for-your-workload-4.webp)

Four design decisions turn that loop from a toy into something trustworthy.

**Define the objective as code, not vibes.** Encode the SLO as concrete thresholds — `TTFT_P99_MS = 500`, `TPOT_P99_MS = 50` — and a scalar you maximize, goodput at your target load. If a config is infeasible, it scores below every feasible config, no exceptions. Do not average TTFT and throughput into a single mushy score; that hides SLO violations behind a good mean.

**Define the search space deliberately.** List the knobs, and for each, a small set of candidate values that brackets the plausible range. For a chat workload: `max_num_seqs ∈ {32, 64, 96, 128, 192, 256}`, `max_num_batched_tokens ∈ {1024, 2048, 4096, 8192}`, `enable_chunked_prefill ∈ {true, false}`, `enable_prefix_caching ∈ {true, false}`. That is already $6 \times 4 \times 2 \times 2 = 96$ configs — and it grows multiplicatively with every knob you add. The size of the full space, $|\mathcal{C}| = \prod_i |\mathcal{C}_i|$, is why the search *strategy* matters: at six knobs with five values each you are looking at $5^6 \approx 15{,}600$ configs, and at ten knobs the grid is astronomically large. Prune ruthlessly: fix knobs whose value you already know (tensor-parallel-size is set by the model fitting, not by tuning), and only sweep what genuinely trades.

**Pick a search strategy** — grid, random, coordinate descent, or Bayesian — sized to the space. We give the decision rule in Section 5.

**Evaluate on a replayed trace, never on fixed prompts.** This is the failure that quietly ruins most tuning efforts. If you benchmark with 100 identical 512-token prompts, you will measure a config's behavior on a distribution that does not exist in production, and you will tune to it. Capture a real trace — timestamps, prompt lengths, output lengths — from your production logs, and replay it with its real arrival timing and its real length distribution. vLLM's own `benchmark_serving.py` supports this directly with `--dataset-name sharegpt` (or your own dataset) and `--request-rate` to control arrival; NVIDIA's GenAI-Perf does the equivalent with configurable input/output length distributions. The point is that the evaluation must exercise the same prefill/decode ratio and the same burstiness the config will face in production, or the number you optimize is fiction.

Here is the launch-and-measure primitive the harness is built on. A short bash wrapper starts a vLLM server with a config and points a benchmark at it, using the `--goodput` flag so the tool reports SLO-meeting throughput directly:

```bash
#!/usr/bin/env bash
# launch_and_measure.sh — boot vLLM with a config, replay a trace, print goodput.
set -euo pipefail

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PORT=8000

# $1..$4 are the config values under test for this trial.
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --max-num-seqs "$1" \
  --max-num-batched-tokens "$2" \
  --gpu-memory-utilization "$3" \
  ${4:+--enable-chunked-prefill} \
  --disable-log-requests \
  --port "$PORT" &
SERVER_PID=$!

# Block until the server is serving, then replay a captured trace.
until curl -sf "http://localhost:$PORT/health" >/dev/null; do sleep 2; done

python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model "$MODEL" \
  --dataset-name sharegpt \
  --dataset-path ./traces/support_chat_trace.json \
  --request-rate 25 \
  --num-prompts 2000 \
  --goodput ttft:500 tpot:50 \
  --percentile-metrics ttft,tpot,itl \
  --save-result --result-filename "./results/trial.json"

kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
```

The `--goodput ttft:500 tpot:50` argument tells the benchmark to count only requests whose TTFT is under 500 ms and whose per-output-token latency is under 50 ms — exactly the SLO-meeting throughput we defined as the objective. That is the measurement primitive; the harness in Section 8 wraps it in a proper Python search loop that manages the process lifecycle, parses the JSON, and selects the winner.

## 5. Choosing the search strategy by knob count

The right way to walk the space depends almost entirely on how big the space is, which is set by how many knobs you sweep. The decision tree below encodes the rule I use.

![A decision tree that selects a search strategy by knob count: one to three knobs go to full grid search, four to six knobs go to coordinate descent or random sampling, and seven or more knobs go to Bayesian optimization with Optuna.](/imgs/blogs/autotuning-serving-configs-for-your-workload-5.webp)

**One to three knobs: grid search.** If you are tuning max-num-seqs and max-num-batched-tokens (the two that matter most, for most people), the full grid is small — a few dozen configs — and grid search is not just adequate, it is *ideal*. It is exhaustive, so it cannot miss the optimum; it is trivially parallelizable across machines; and it produces the complete Pareto frontier as a free byproduct, which is worth as much as the winning config because it shows you the *shape* of the trade and tells you how much headroom you would buy by relaxing the SLO. Do not reach for anything clever when the grid is cheap. Each trial costs a few minutes of GPU time; forty trials is a coffee break.

**Four to six knobs: coordinate descent or random search.** Once the grid crosses a few hundred configs, exhaustive becomes expensive. Two cheap approximations work well. Coordinate descent sweeps one knob at a time, holding the others at their current best, and iterates a couple of passes — cheap and interpretable, but it can stall on the coupled ridges we discussed, so always run at least two full passes and sweep coupled knobs (batched-tokens with chunked-prefill) *jointly* rather than separately. Random search — sampling, say, 60 random configs from the space — is a shockingly strong baseline; the classic result from hyperparameter optimization (Bergstra and Bengio, 2012) is that random search beats grid when only a few dimensions actually matter, because it does not waste trials on the flat ones. For serving, where two or three knobs dominate, random search finds a near-optimal config in far fewer trials than the grid would need.

**Seven or more knobs: Bayesian optimization.** When you genuinely have a large, coupled space — you are jointly tuning batch shape, memory, prefill scheduling, KV dtype, block size, and attention backend — the number of configs explodes and each trial is expensive (minutes of GPU time to boot and replay). This is exactly the regime where Bayesian optimization pays off: it builds a probabilistic surrogate model of goodput as a function of config, and uses it to propose the *next* config that best balances exploring uncertain regions and exploiting promising ones, so it converges to a good config in tens of trials instead of thousands. Optuna's Tree-structured Parzen Estimator (Akiba et al., KDD 2019) is the practical default, and it handles the mix of integer, categorical, and boolean knobs that a serving config is made of. We build the Optuna version in Section 8.

The meta-rule: start with the smallest strategy that fits your space, and only escalate when the grid is genuinely too big. Most teams over-engineer this. If two knobs move 90% of your goodput — and they usually do — a grid over those two, holding the rest at sane defaults, is the whole job.

## 6. Building and reading the Pareto frontier

A sweep produces a table of (config, latency, throughput, goodput, feasible?) rows. The single most useful thing you can do with that table is plot it as a throughput-versus-latency scatter and find the frontier. The figure below is that plot rendered as a grid: throughput increases left to right, p99 latency increases top to bottom, and each swept config is placed by where it landed.

![A grid laying out swept configs on a throughput-versus-latency plane: configs in the top and middle rows sit on or behind the Pareto frontier, the chosen config is the highest-throughput frontier point inside the SLO, and the bottom row of configs violates the p99 SLO.](/imgs/blogs/autotuning-serving-configs-for-your-workload-6.webp)

Read it as follows. The bottom row — configs F, G, H — all have p99 latency above the 500 ms SLO line; they are infeasible and score zero no matter how much raw throughput they post (config H does 34 req/s, the most of anyone, and it is *worthless* because every one of those requests is late). The frontier runs through configs B, D, and E: each is non-dominated, meaning nothing beats it on both axes at once. Configs A and C are *dominated* — A sits behind B (B is faster in latency terms relative to its throughput), C is behind D — so they are never the right answer; some frontier config is strictly better. The chosen config is E: it is on the frontier, it is inside the SLO (460 ms < 500 ms), and it has the highest throughput of any feasible frontier point (30 req/s). Walk the frontier from the fast end and stop at the last point under the line — that is the algorithm, and it is one line of code once you have the frontier.

Here is the same sweep as a table, so you can trace the frontier logic row by row rather than eyeballing the scatter:

| config | p99 TTFT (ms) | throughput (req/s) | feasible? | on frontier? | note |
|---|---|---|---|---|---|
| A | 300 | 16 | yes | no | dominated by B — B is higher throughput at barely more latency |
| B | 320 | 22 | yes | yes | fast end of the frontier |
| C | 410 | 24 | yes | no | dominated by D |
| D | 430 | 27 | yes | yes | frontier |
| E | 460 | 30 | yes | yes | **chosen** — highest-throughput feasible frontier point |
| F | 540 | 31 | no | — | 40 ms over the 500 ms SLO |
| G | 610 | 33 | no | — | infeasible |
| H | 680 | 34 | no | — | highest raw throughput, zero goodput |

The algorithm reads straight off the table: discard every infeasible row (F, G, H), discard every dominated row (A, C), and among what remains take the highest throughput — that is E at 30 req/s. Notice that H, the config a throughput benchmark would crown, sits at the very bottom of the goodput ranking despite topping the raw-throughput column, because all 34 of its req/s are late. The table is also the artifact you archive: six months from now it is the record of exactly which points you tried and why the winner won, which is precisely what you diff against when you re-tune.

Two things this picture teaches that a single winning config does not. First, *how close to the edge you are*: E is at 460 ms against a 500 ms budget, so you have almost no margin — a traffic uptick or a longer-prompt day will push it over, and you should either provision more capacity or pick a slightly more conservative frontier point. Second, *the value of the SLO*: if you could negotiate the SLO up to 700 ms, config H becomes feasible and you jump from 30 to 34 req/s — a 13% goodput gain for 200 ms of latency budget. The frontier turns "what config?" into a business conversation about how much latency a little more throughput is worth. Always build the frontier, not just the winner.

## 7. The interactions you cannot ignore

Section 3 proved the space is non-separable in the abstract; this section names the specific couplings that bite in practice, so you know which knobs to sweep *jointly*. The matrix below catalogs the five interaction pairs that cause the most self-inflicted incidents.

![A matrix of five coupled knob pairs, each with why the two knobs share a resource, what goes wrong if you tune one of them alone, the resulting symptom, and the joint fix that resolves it.](/imgs/blogs/autotuning-serving-configs-for-your-workload-7.webp)

Before the prose walks each pair, here is the whole set as a lookup table — the shared resource that creates the coupling, the failure you get from tuning one knob blind, and the joint move that fixes it:

| coupled pair | shared resource | tune-one-alone failure | symptom | joint fix |
|---|---|---|---|---|
| max-num-batched-tokens × chunked-prefill | one scheduler step's token budget | huge unchunked prefill step | decode stalls, TPOT jitter | chunk on, sweep chunk size 1024–4096 |
| gpu-mem-util × max-num-seqs | KV pool size | admit more seqs than the pool holds | preemption/recompute or OOM | cap seqs at the KV-budget ceiling |
| max-num-seqs × max-model-len | KV blocks reserved per sequence | reserve context you never use | starved batch, low throughput | set max-model-len to the p99 real length |
| prefix-caching × trace mix | shared prompt prefixes | benchmark on unique prompts | wrong "it doesn't help" verdict | gate on hit rate from a real trace |
| tensor-parallel-size × batch size | per-GPU occupancy | TP with a small batch | idle shards, 2× cost/token | use TP only when memory-bound or the model won't fit |

Every row is one resource that two knobs draw from the same pool — step time, KV bytes, GPU occupancy — which is exactly why moving one without the other backfires. The prose below unpacks each row.

**max-num-batched-tokens × chunked-prefill.** When chunked prefill is on, the token budget *is* the chunk size, so these are one knob. Tune batched-tokens alone with chunking off and you will get great TTFT but murderous TPOT jitter, because a big prompt runs as one giant step that stalls every concurrent decode. The symptom is decode stalls — users watching their token stream freeze for a second whenever someone else submits a long prompt. The fix is to co-sweep: turn chunking on and search chunk sizes (2048, 4096) that keep each step short enough to protect decode while still amortizing prefill.

**gpu-memory-util × max-num-seqs.** The KV pool, sized by utilization, sets the ceiling on concurrent sequences. Raise max-num-seqs without accounting for the pool and the scheduler will admit more sequences than the cache can hold, forcing preemption and recompute, or worse, an OOM crash at peak load. The joint fix is to compute the feasible max-num-seqs from the KV budget equation in Section 3 and cap the sweep there, or to raise utilization and max-num-seqs together and watch for the preemption counter climbing.

**max-num-seqs × max-model-len.** KV blocks consumed scale as (sequence length × concurrent sequences), so context length and batch size draw from the same pool. Set max-model-len to the full trained context "to be safe" while running short chat prompts, and you reserve blocks for lengths you never see, starving the batch and dropping throughput. The fix is to budget max-model-len to your real p99 prompt-plus-output length, which frees blocks for more concurrency.

**prefix-caching × trace mix.** Prefix caching only pays off when requests actually share prefixes, and whether they do is a property of your workload, not a flag. Enable it and benchmark on a cold, all-unique synthetic mix and you will measure pure hashing overhead and conclude, wrongly, that it does not help — then disable it and lose a huge win on your real, prefix-heavy traffic. The fix is to gate the decision on measured hit rate from a *real* trace, never a synthetic one.

**tensor-parallel-size × batch size.** TP frees per-GPU VRAM for a bigger KV pool, but it only earns its keep if the batch is large enough to keep all the shards busy. TP=2 at low batch leaves both GPUs half-idle and doubles your cost per token for nothing. Match TP to the load: use it when the model does not fit or when you are memory-bound at high concurrency, not as a reflexive "more GPUs = faster."

The operational takeaway is a search-space design rule: identify the coupled clusters, and sweep each cluster jointly even when you use a cheap strategy for the rest. Coordinate descent over *clusters* is fine; coordinate descent over individual coupled knobs is how you land in a ridge and declare victory 20% short of optimal.

## 8. A full autotuning harness in Python

Now we assemble the pieces. This section gives you eight runnable artifacts: a grid-search harness that launches vLLM per config and selects the winner, an Optuna Bayesian version, a coordinate-descent quick-tune, a Pareto-frontier plotter, a config-diff reporter for the before/after write-up, a random-search harness, a KV-budget feasibility calculator that bounds the space before you spend a single GPU-minute, and a trace-capture utility that turns production logs into the replay dataset the rest of them assume. Adapt the model, trace path, and SLO thresholds to yours.

### 8.1 The grid-search harness

The core harness defines the space, iterates configs, and for each one launches a vLLM server as a subprocess, waits for health, replays the trace with `benchmark_serving.py`, parses the JSON result for p99 metrics and goodput, records the row, and tears the server down. At the end it filters to feasible configs and returns the max-goodput winner.

```python
# autotune.py — grid-search the vLLM serving config for a fixed workload/SLO.
import itertools, json, subprocess, time, signal, os
from dataclasses import dataclass, asdict
import requests

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TRACE = "./traces/support_chat_trace.json"
PORT = 8000
TTFT_SLO_MS, TPOT_SLO_MS = 500.0, 50.0   # the hard constraints
TARGET_RATE = 25                          # offered load, req/s

# The search space: only the knobs that trade for THIS workload.
SPACE = {
    "max_num_seqs":            [32, 64, 96, 128, 192, 256],
    "max_num_batched_tokens":  [1024, 2048, 4096, 8192],
    "enable_chunked_prefill":  [True, False],
    "enable_prefix_caching":   [True, False],
}

@dataclass
class Result:
    config: dict
    ttft_p99_ms: float
    tpot_p99_ms: float
    throughput: float      # completed req/s
    goodput: float         # SLO-meeting req/s
    feasible: bool

def launch_server(cfg: dict) -> subprocess.Popen:
    args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", str(PORT), "--disable-log-requests",
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]),
        "--gpu-memory-utilization", "0.90",
    ]
    if cfg["enable_chunked_prefill"]:
        args.append("--enable-chunked-prefill")
    if cfg["enable_prefix_caching"]:
        args.append("--enable-prefix-caching")
    # New process group so we can kill the whole server tree cleanly.
    return subprocess.Popen(args, preexec_fn=os.setsid)

def wait_healthy(timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False

def replay_and_measure(out_file: str) -> dict:
    subprocess.run([
        "python", "benchmarks/benchmark_serving.py",
        "--backend", "vllm", "--model", MODEL,
        "--dataset-name", "sharegpt", "--dataset-path", TRACE,
        "--request-rate", str(TARGET_RATE), "--num-prompts", "2000",
        "--goodput", f"ttft:{int(TTFT_SLO_MS)}", f"tpot:{int(TPOT_SLO_MS)}",
        "--percentile-metrics", "ttft,tpot,itl",
        "--save-result", "--result-filename", out_file,
    ], check=True)
    with open(out_file) as fh:
        return json.load(fh)

def evaluate(cfg: dict) -> Result:
    server = launch_server(cfg)
    try:
        if not wait_healthy():
            # Could not even boot (usually OOM): infeasible by construction.
            return Result(cfg, float("inf"), float("inf"), 0.0, 0.0, False)
        raw = replay_and_measure(f"./results/{hash(frozenset(cfg.items()))}.json")
    finally:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait()

    ttft = raw["p99_ttft_ms"]
    tpot = raw["p99_tpot_ms"]
    thr  = raw["request_throughput"]
    good = raw.get("request_goodput", thr if (ttft <= TTFT_SLO_MS and tpot <= TPOT_SLO_MS) else 0.0)
    feasible = ttft <= TTFT_SLO_MS and tpot <= TPOT_SLO_MS
    return Result(cfg, ttft, tpot, thr, good, feasible)

def grid_search() -> list[Result]:
    keys = list(SPACE)
    results = []
    for combo in itertools.product(*SPACE.values()):
        cfg = dict(zip(keys, combo))
        # Skip incoherent points: batched_tokens is the chunk size only when chunking is on.
        if not cfg["enable_chunked_prefill"] and cfg["max_num_batched_tokens"] < 8192:
            continue
        r = evaluate(cfg)
        results.append(r)
        print(f"{cfg} -> ttft={r.ttft_p99_ms:.0f} tpot={r.tpot_p99_ms:.1f} "
              f"good={r.goodput:.1f} {'OK' if r.feasible else 'MISS'}")
    return results

def select_best(results: list[Result]) -> Result | None:
    feasible = [r for r in results if r.feasible]
    return max(feasible, key=lambda r: r.goodput) if feasible else None

if __name__ == "__main__":
    all_results = grid_search()
    best = select_best(all_results)
    json.dump([asdict(r) for r in all_results], open("sweep.json", "w"), indent=2)
    if best:
        print("\nWINNER:", json.dumps(best.config), f"goodput={best.goodput:.1f} req/s")
    else:
        print("\nNo feasible config — relax the SLO or add capacity.")
```

Three things worth calling out. The `finally` block guarantees the server is killed even if the benchmark crashes, which matters when you are running hundreds of trials unattended — a leaked server holds the GPU and every subsequent trial OOMs. The `wait_healthy` timeout doubling as an OOM detector is deliberate: a config that cannot even boot is infeasible, and treating boot failure as goodput zero keeps the search from crashing. And the coherence skip prunes nonsensical grid points (a small token budget only means "chunk size" when chunking is on), which trims wasted trials.

### 8.2 The Optuna Bayesian version

When the space is large, replace the grid with Optuna. The objective function suggests a config, evaluates it with the exact same `evaluate()` primitive, and returns goodput — with infeasible configs penalized so the sampler learns to avoid the SLO-violating region.

```python
# autotune_optuna.py — Bayesian search over a large, coupled config space.
import optuna
from autotune import evaluate, TTFT_SLO_MS   # reuse the launch/measure primitive

def objective(trial: optuna.Trial) -> float:
    cfg = {
        "max_num_seqs":           trial.suggest_int("max_num_seqs", 16, 256, step=16),
        "max_num_batched_tokens": trial.suggest_categorical(
                                      "max_num_batched_tokens", [1024, 2048, 4096, 8192]),
        "enable_chunked_prefill": trial.suggest_categorical("chunked", [True, False]),
        "enable_prefix_caching":  trial.suggest_categorical("prefix", [True, False]),
        # extra knobs worth it only once the space is large:
        "gpu_memory_utilization": trial.suggest_float("gpu_mem", 0.80, 0.95, step=0.01),
        "kv_cache_dtype":         trial.suggest_categorical("kv_dtype", ["auto", "fp8"]),
    }
    r = evaluate(cfg)
    # Record the raw metrics for later frontier plotting and constraint handling.
    trial.set_user_attr("ttft_p99_ms", r.ttft_p99_ms)
    trial.set_user_attr("tpot_p99_ms", r.tpot_p99_ms)
    trial.set_user_attr("feasible", r.feasible)
    # Constraint for the sampler: <= 0 means satisfied. TTFT headroom, normalized.
    trial.set_user_attr("constraint", (r.ttft_p99_ms - TTFT_SLO_MS) / TTFT_SLO_MS)
    # Infeasible configs score below any feasible one so the TPE learns the boundary.
    return r.goodput if r.feasible else -1.0

def constraints(trial):
    return (trial.user_attrs["constraint"],)

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(constraints_func=constraints, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40)      # ~40 boots instead of thousands
    best = study.best_trial
    print("WINNER:", best.params, f"goodput={best.value:.1f} req/s")
    # Persist every trial for the Pareto plot in 8.4.
    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)
```

The TPE sampler with `constraints_func` is the clean way to handle the SLO: rather than only penalizing the return value, you hand Optuna the constraint violation directly, and it prioritizes proposals that are both high-goodput *and* likely feasible. Forty trials over a six-knob space routinely lands within a percent or two of what an exhaustive grid would find, at a fraction of the GPU time — which is the entire reason to reach for Bayesian search when the grid is too big.

### 8.3 The coordinate-descent quick-tune

Sometimes you do not want a full study — you want a good config in an hour, starting from your current one. Coordinate descent sweeps one knob at a time, keeps the best feasible value, and iterates. The catch, per Section 7, is the coupling, so this version sweeps the two coupled knobs (batched-tokens with chunked-prefill) as a *pair*.

```python
# autotune_coord.py — cheap coordinate descent from a starting config.
from autotune import evaluate, SPACE

def quick_tune(start: dict, passes: int = 2) -> dict:
    best_cfg = dict(start)
    best = evaluate(best_cfg)
    for p in range(passes):
        improved = False
        # 1) Sweep the coupled (batched_tokens, chunked_prefill) pair JOINTLY.
        for bt in SPACE["max_num_batched_tokens"]:
            for chunk in [True, False]:
                cand = {**best_cfg, "max_num_batched_tokens": bt,
                        "enable_chunked_prefill": chunk}
                r = evaluate(cand)
                if r.feasible and r.goodput > best.goodput:
                    best, best_cfg, improved = r, cand, True
        # 2) Sweep the remaining independent-ish knobs one at a time.
        for knob in ["max_num_seqs", "enable_prefix_caching"]:
            for val in SPACE[knob]:
                cand = {**best_cfg, knob: val}
                r = evaluate(cand)
                if r.feasible and r.goodput > best.goodput:
                    best, best_cfg, improved = r, cand, True
        print(f"pass {p}: goodput={best.goodput:.1f} req/s  cfg={best_cfg}")
        if not improved:
            break     # converged — no knob move helped this pass
    return best_cfg

if __name__ == "__main__":
    seed = {"max_num_seqs": 256, "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": False, "enable_prefix_caching": False}
    print("TUNED:", quick_tune(seed))
```

Coordinate descent is the right tool when you already have a working config and want to nudge it — say after a model swap — without a full re-sweep. Run at least two passes so a knob you moved early can be revisited after later knobs changed the landscape, and never trust a single pass, because the first pass optimizes each knob against the *starting* values of the others, which the coupling then invalidates.

### 8.4 The Pareto-frontier plotter

Turn the recorded sweep into the picture from Section 6: a scatter of every config in (latency, throughput) space, the non-dominated frontier drawn as a line, the SLO cutoff as a vertical rule, and the chosen config highlighted.

```python
# plot_pareto.py — build and draw the throughput-vs-latency Pareto frontier.
import json
import matplotlib.pyplot as plt

def pareto_front(points):
    """points: list of (latency_ms, throughput). Return the non-dominated subset."""
    ordered = sorted(points, key=lambda p: (p[0], -p[1]))   # latency asc
    front, best_thr = [], -1.0
    for lat, thr in ordered:
        if thr > best_thr:            # nothing seen so far is faster AND higher
            front.append((lat, thr))
            best_thr = thr
    return front

def main(sweep="sweep.json", ttft_slo=500.0):
    rows = json.load(open(sweep))
    pts = [(r["ttft_p99_ms"], r["throughput"], r["feasible"]) for r in rows
           if r["ttft_p99_ms"] != float("inf")]
    front = pareto_front([(lat, thr) for lat, thr, _ in pts])

    feas = [(l, t) for l, t, ok in pts if ok]
    infeas = [(l, t) for l, t, ok in pts if not ok]
    # Chosen config: highest-throughput frontier point still inside the SLO.
    inside = [(l, t) for l, t in front if l <= ttft_slo]
    chosen = max(inside, key=lambda p: p[1]) if inside else None

    plt.figure(figsize=(8, 5))
    if infeas: plt.scatter(*zip(*infeas), c="#d64545", marker="x", label="SLO miss")
    if feas:   plt.scatter(*zip(*feas), c="#4a7", label="feasible")
    plt.plot(*zip(*front), c="#333", lw=1.5, label="Pareto frontier")
    plt.axvline(ttft_slo, ls="--", c="#888", label=f"p99 TTFT SLO = {ttft_slo:.0f} ms")
    if chosen:
        plt.scatter([chosen[0]], [chosen[1]], s=220, edgecolors="k",
                    facecolors="none", linewidths=2, label="chosen")
    plt.xlabel("p99 TTFT (ms)"); plt.ylabel("throughput (req/s)")
    plt.legend(); plt.tight_layout(); plt.savefig("pareto.png", dpi=150)

if __name__ == "__main__":
    main()
```

The `pareto_front` function is the whole idea in eight lines: sort by latency ascending, walk left to right, and keep a point only if its throughput beats everything seen so far — anything that does not is dominated by something faster. The chosen config falls out as the highest-throughput frontier point left of the SLO line. Save this plot alongside every tuning run; it is the artifact you show when someone asks "why this config?"

### 8.5 The config-diff reporter

Finally, the write-up. When you hand a tuned config to the team, show the *diff* from the default and the metrics it moved, so the change is legible and reversible.

```python
# config_diff.py — render a default-vs-tuned diff for the changelog.
def diff_report(default: dict, tuned: dict, before: dict, after: dict) -> str:
    lines = ["Config changes:"]
    for k in sorted(set(default) | set(tuned)):
        d, t = default.get(k, "—"), tuned.get(k, "—")
        mark = "  " if d == t else "* "
        lines.append(f"  {mark}{k}: {d}  ->  {t}")
    lines.append("\nMetrics (replayed trace @ 25 req/s):")
    for m in ["ttft_p99_ms", "tpot_p99_ms", "goodput"]:
        lines.append(f"    {m}: {before[m]}  ->  {after[m]}")
    return "\n".join(lines)

if __name__ == "__main__":
    default = {"max_num_seqs": 256, "max_num_batched_tokens": 8192,
               "enable_chunked_prefill": False, "enable_prefix_caching": False}
    tuned   = {"max_num_seqs": 96, "max_num_batched_tokens": 2048,
               "enable_chunked_prefill": True, "enable_prefix_caching": True}
    before  = {"ttft_p99_ms": 780, "tpot_p99_ms": 42, "goodput": 12.0}
    after   = {"ttft_p99_ms": 430, "tpot_p99_ms": 28, "goodput": 27.0}
    print(diff_report(default, tuned, before, after))
```

This prints a compact, greppable record — four flags changed, three metrics moved — that goes straight into the PR description and the runbook. When traffic shifts in three months and someone asks whether the tuning still holds, this is the baseline they diff against.

### 8.6 A random-search harness

For four-to-six-knob spaces, random search is the cheap, strong baseline from Section 5: sample configs uniformly, evaluate each with the same primitive, and keep the best feasible one. It reuses everything from 8.1.

```python
# autotune_random.py — random search over the config space.
import random, json
from dataclasses import asdict
from autotune import evaluate, SPACE, select_best

def random_search(n_trials: int = 60, seed: int = 0) -> list:
    rng = random.Random(seed)
    results = []
    for i in range(n_trials):
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        # Respect the same coherence rule as the grid: a small token budget
        # is only meaningful as a chunk size when chunked prefill is on.
        if not cfg["enable_chunked_prefill"] and cfg["max_num_batched_tokens"] < 8192:
            cfg["max_num_batched_tokens"] = 8192
        r = evaluate(cfg)
        results.append(r)
        best = select_best(results)
        tag = "OK" if r.feasible else "MISS"
        sofar = f"{best.goodput:.1f}" if best else "none"
        print(f"trial {i:02d}: good={r.goodput:.1f} {tag} | best so far={sofar} req/s")
    return results

if __name__ == "__main__":
    results = random_search(n_trials=60)
    json.dump([asdict(r) for r in results], open("sweep_random.json", "w"), indent=2)
    best = select_best(results)
    if best:
        print("\nWINNER:", best.config, f"goodput={best.goodput:.1f} req/s")
    else:
        print("\nno feasible config")
```

Sixty random draws over a four-knob space cover the important corners without the multiplicative blow-up of the full grid, and because the two dominant knobs (max-num-seqs and the token budget) are sampled independently on every trial, random search spends most of its budget varying exactly the dimensions that move goodput — the Bergstra-and-Bengio result made operational. Feed the resulting `sweep_random.json` to the Pareto plotter in 8.4 and you get the frontier for free, just sparser than the grid's. Bump `n_trials` if the frontier still looks ragged near the SLO line.

### 8.7 A KV-budget feasibility calculator

Section 3 showed that the memory arithmetic rules configs out before you ever boot them. Encode it once and you can prune the search space — and hand the sweep a hard ceiling on max-num-seqs — for free.

```python
# kv_budget.py — bound max-num-seqs from the KV-cache memory equation.
GIB = 1024 ** 3

def bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes   # 2 = keys + values

def kv_pool_bytes(vram_gib, util, weight_gib, reserve_gib=4.0):
    return (vram_gib * util - weight_gib - reserve_gib) * GIB

def max_feasible_seqs(vram_gib, util, weight_gib, model, avg_ctx_tokens, dtype_bytes):
    bpt  = bytes_per_token(**model, dtype_bytes=dtype_bytes)
    pool = kv_pool_bytes(vram_gib, util, weight_gib)
    token_capacity = pool / bpt
    return int(token_capacity // avg_ctx_tokens), int(token_capacity)

LLAMA3_8B = dict(layers=32, kv_heads=8, head_dim=128)   # GQA: 8 KV heads

if __name__ == "__main__":
    for dtype, b in [("bf16", 2), ("fp8", 1)]:
        for util in (0.85, 0.90, 0.95):
            for ctx in (512, 2048):
                seqs, cap = max_feasible_seqs(
                    vram_gib=80, util=util, weight_gib=16,
                    model=LLAMA3_8B, avg_ctx_tokens=ctx, dtype_bytes=b)
                print(f"{dtype} util={util} ctx={ctx:>4}: "
                      f"pool={cap:>9,} tok -> max_num_seqs <= {seqs}")
```

Run it before the sweep and clamp `SPACE["max_num_seqs"]` to the value it returns for your real average context and target utilization. A grid that never proposes a max-num-seqs above the memory ceiling wastes zero trials on configs that would only preempt or OOM — often trimming the grid by a third. It is also the fastest triage when a config mysteriously OOMs: plug in the numbers and the equation usually shows the pool was a handful of sequences short of the batch you asked for.

### 8.8 Capturing a representative trace

Every harness above replays a trace, and the whole method is only as honest as that trace. This utility turns a day of production request logs into the ShareGPT-shaped dataset `benchmark_serving.py` expects, preserving the real prompt-length distribution and — critically — the real inter-arrival timing.

```python
# capture_trace.py — turn production request logs into a replayable trace.
import json, sys
from transformers import AutoTokenizer

# Input: JSONL, one request per line, e.g.
#   {"ts": 1719800000.12, "prompt": "...", "completion": "..."}
def build_trace(log_path, out_path, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    rows.sort(key=lambda r: r["ts"])
    t0 = rows[0]["ts"]
    trace = []
    for r in rows:
        p_ids = tok(r["prompt"]).input_ids
        c_ids = tok(r.get("completion", "")).input_ids
        trace.append({
            "prompt":     r["prompt"],
            "prompt_len": len(p_ids),
            "output_len": max(len(c_ids), 1),    # replay generates this many tokens
            "arrival_s":  round(r["ts"] - t0, 4),  # relative arrival keeps burstiness
        })
    json.dump(trace, open(out_path, "w"))
    lens = sorted(t["prompt_len"] for t in trace)
    outs = sorted(t["output_len"] for t in trace)
    q = lambda xs, p: xs[int(p * (len(xs) - 1))]
    print(f"{len(trace)} requests over {trace[-1]['arrival_s']:.0f}s")
    print(f"prompt_len   p50={q(lens,.5)}  p99={q(lens,.99)}")
    print(f"output_len   p50={q(outs,.5)}  p99={q(outs,.99)}")

if __name__ == "__main__":
    build_trace(sys.argv[1], sys.argv[2])
```

Two properties make this trace trustworthy where a synthetic one lies. First, the length pairs are drawn from real requests, so the prefill-to-decode FLOP ratio the sweep sees is the ratio production sees — the single thing that decides whether you are prefill- or decode-bound. Second, the relative arrival times preserve burstiness, so a config is tested against the same queue-depth spikes it will actually face; replaying at a flat rate would hide exactly the preemption behavior you are trying to tune away. The printed p50/p99 lengths double as your re-tuning tripwire, discussed later in this post: when next quarter's capture shows the p99 prompt length drifting 20% past this one, the tuned config's shelf life has expired.

## 9. Worked examples and measured results

Numbers make the method concrete. Both examples below use Llama-3-8B-Instruct on a single H100 80GB, replaying a captured support-chat trace (median prompt 180 tokens, median output 210 tokens, bursty arrivals around a 25 req/s mean) with a 500 ms p99 TTFT and 50 ms p99 TPOT SLO. Treat the exact figures as representative of the *shape* of the result — your absolute numbers will differ — but the direction and the ratios are what generalize.

#### Worked example: tuning max-num-seqs × max-num-batched-tokens for a chat workload

The two-knob grid is the highest-value sweep most teams will ever run, so start there. Holding chunked prefill and prefix caching on, sweep `max_num_seqs ∈ {64, 96, 128, 192, 256}` against `max_num_batched_tokens ∈ {2048, 8192}`. The default config the team copied — 256 sequences, 8192 token budget, chunking off — is the row to beat. The sweep results:

| max-num-seqs | max-batched-tokens | p99 TTFT (ms) | p99 TPOT (ms) | throughput (req/s) | goodput (req/s) | feasible? |
|---|---|---|---|---|---|---|
| 256 (default) | 8192 | 780 | 42 | 28 | 12 | no (TTFT) |
| 256 | 2048 | 610 | 31 | 27 | 18 | no (TTFT) |
| 192 | 2048 | 520 | 30 | 26 | 22 | no (TTFT) |
| 128 | 2048 | 470 | 29 | 25 | 25 | yes |
| 96 | 2048 | 430 | 28 | 24 | 27 | yes |
| 64 | 2048 | 360 | 26 | 20 | 20 | yes |
| 96 | 8192 | 690 | 40 | 27 | 14 | no (TTFT) |

Read the goodput column, not the throughput column. The default posts the *highest* raw throughput (28 req/s) and the *lowest* goodput (12 req/s) — it is fast at completing requests and terrible at completing them on time, because 256 concurrent sequences plus an 8192-token unchunked prefill drives the tail past 500 ms. Dropping the token budget to 2048 helps, but the winner is `max_num_seqs=96`: it sacrifices 4 req/s of raw throughput to bring p99 TTFT down to 430 ms, and because nearly every request now meets the SLO, goodput jumps to 27 req/s — a 2.25× improvement over the default on the metric that matters. Push max-num-seqs lower to 64 and goodput falls again, because now the batch is too small to keep the GPU busy: you have over-corrected. The optimum is a genuine interior peak, invisible unless you sweep.

#### Worked example: a chunked-prefill chunk-size sweep

Chunked prefill trades TTFT for TPOT stability, and the chunk size (the token budget when chunking is on) sets the exchange rate. Fixing `max_num_seqs=96`, sweep the chunk size and watch both tails move in opposite directions:

| chunk size (max-batched-tokens) | p99 TTFT (ms) | p99 TPOT (ms) | TPOT jitter (stdev, ms) | goodput (req/s) |
|---|---|---|---|---|
| chunking off (8192 one-shot) | 690 | 40 | 14.0 | 14 |
| 4096 | 500 | 33 | 8.5 | 23 |
| 2048 | 430 | 28 | 4.2 | 27 |
| 1024 | 410 | 27 | 3.8 | 26 |
| 512 | 460 | 29 | 3.5 | 24 |

Turning chunking off entirely gives the worst of both worlds here: a 690 ms TTFT (the whole prompt blocks one step) *and* 40 ms TPOT with heavy jitter (concurrent decodes stall behind that step). Shrinking the chunk to 2048 tokens cuts TTFT to 430 ms and TPOT to 28 ms with a fourth of the jitter, because each prefill step is now short enough to interleave cleanly with decode — this is exactly the "stall-free batching" that Sarathi-Serve (Agrawal et al., OSDI 2024) formalizes. But go too small (512) and TTFT *rises* again to 460 ms, because a long prompt now needs many chunks and many scheduler steps to finish its prefill, adding per-step overhead. The sweet spot is 2048; the curve is U-shaped in TTFT, and only a sweep finds the bottom.

#### Worked example: the gpu-memory-utilization OOM boundary on a RAG workload

The first two examples held gpu-memory-utilization at 0.90 on a short chat context, where the KV pool was never the binding constraint. Switch to a RAG workload — median prompt 1,800 tokens, output 150 — with `max_num_seqs=192`, `max_model_len=4096`, chunk on at 2048, and sweep the memory knob alone. The working set at full batch is roughly ${192 \times 1{,}950 \approx 375\text{k}}$ tokens of KV, which is close to the BF16 pool, so the memory knob now bites at both ends:

| gpu-mem-util | KV pool | preemptions/min | p99 TTFT (ms) | goodput (req/s) | status |
|---|---|---|---|---|---|
| 0.80 | ~360k tok | 140 | 910 | 9 | pool < working set: heavy preempt |
| 0.85 | ~393k tok | 35 | 560 | 19 | fits, no burst headroom |
| 0.90 | ~426k tok | 2 | 440 | 26 | healthy |
| 0.93 | ~446k tok | 0 | 435 | 26 | healthy, thin activation headroom |
| 0.96 | — | — | — | 0 | OOM crash on a prefill burst |

This is the Section 7 coupling turned into a measurement. Below 0.85 the KV pool is smaller than the batch's working set, so the scheduler preempts and recomputes constantly, p99 TTFT more than doubles, and goodput collapses to 9 req/s. Above 0.93 the pool is ample but there is no memory left for the activation spike a burst of concurrent prefills produces, and the server OOMs and drops every in-flight request — the worst failure of all, because it takes healthy requests down with it. The feasible band is narrow (0.90–0.93) and invisible without the sweep: the KV-budget calculator in Section 8.7 predicts its lower edge, and only a real burst in the trace exposes its upper edge. Switching to FP8 KV would double the pool and widen this band dramatically — which is the joint fix, and exactly why kv-cache-dtype and gpu-memory-utilization must be tuned together rather than one at a time.

#### Worked example: prefix caching depends entirely on the trace

The last coupling worth measuring is prefix caching against the trace mix. Take the tuned chat config (96 / 2048, chunk on) and toggle `enable_prefix_caching` on two different traces: the real support-chat trace, whose requests all share a 1,200-token system prompt and few-shot preamble, and a synthetic trace of all-unique prompts drawn from the same length distribution.

| trace | prefix caching | cache hit rate | p99 TTFT (ms) | goodput (req/s) |
|---|---|---|---|---|
| support-chat (shared 1.2k-tok preamble) | off | — | 430 | 27 |
| support-chat | on | 68% | 250 | 34 |
| all-unique synthetic | off | — | 300 | 30 |
| all-unique synthetic | on | 2% | 315 | 29 |

The identical flag is a 26% goodput win on one trace and a small loss on the other. On the real support-chat traffic the shared preamble is computed once and its KV blocks are reused across 68% of requests, so TTFT nearly halves (430 to 250 ms) and goodput climbs from 27 to 34 req/s. On the all-unique synthetic trace the hit rate is a rounding error, the prefix hashing is pure overhead, and goodput slips from 30 to 29. A team that benchmarked prefix caching on the synthetic trace would disable it and silently forfeit the biggest single win available on their real workload. This is the sharpest form of the post's thesis: a flag is neither good nor bad in the abstract; it is good or bad on your trace, and only your trace can tell you which.

Putting these worked examples together, and returning to the two-knob chat sweep that produced the winner, the tuned config measured against the copied default on the same H100 and the same trace is:

| metric | default (256 / 8192, chunk off) | autotuned (96 / 2048, chunk on, prefix on) | change |
|---|---|---|---|
| p99 TTFT | 780 ms | 430 ms | −45% |
| p99 TPOT | 42 ms | 28 ms | −33% |
| raw throughput | 28 req/s | 24 req/s | −14% |
| **goodput (SLO-meeting)** | **12 req/s** | **27 req/s** | **+125%** |
| GPU utilization | 58% | 91% | +33 pts |
| SLO compliance | ~43% | ~98% | — |

![A before-and-after comparison on one H100 80GB: the default config at max-num-seqs 256 and batched-tokens 8192 versus the autotuned config at 96 and 2048, showing p99 TTFT dropping from 780 to 430 ms, TPOT from 42 to 28 ms, and goodput and GPU utilization more than doubling.](/imgs/blogs/autotuning-serving-configs-for-your-workload-8.webp)

The autotuned config posts *lower* raw throughput and more than *double* the goodput, on identical hardware, by changing four flags. That inversion — worse on the vanity metric, dramatically better on the business metric — is the entire argument for tuning goodput under a constraint rather than chasing tokens per second. It also shows up as GPU utilization climbing from 58% to 91%: the tuned config is not idling the GPU waiting on a pathological batch shape; it is keeping it busy with work that actually lands inside the SLO.

## Case studies

Four real, published efforts anchor this method; treat the numbers as reported by their authors and the takeaways as the transferable part.

**vLLM's own auto-tuning helper.** The vLLM repository ships an auto-tuning script under `benchmarks/auto_tune` that automates exactly the loop in this post: it grid-searches `max_num_seqs` and `max_num_batched_tokens`, launches the server per config, runs the serving benchmark, and reports the highest-throughput configuration whose measured latency stays under a target you specify. It exists precisely because the maintainers know the defaults are a starting point, not an answer — the tool's premise is that the right config is workload-specific and must be searched. If you take one thing from this section, it is that the people who *wrote* vLLM ship a config search harness in the box; use it or the one in Section 8, but do not ship the defaults untested.

**NVIDIA GenAI-Perf concurrency sweeps.** NVIDIA's GenAI-Perf (part of the Triton/perf-analyzer tooling) is built around sweeping concurrency and input/output length distributions to trace out the latency-throughput curve for a given model and server. Its standard workflow — fix the model, sweep the concurrency, plot TTFT and throughput at each level — *is* Pareto-frontier construction under a different name, and its configurable synthetic length distributions let you approximate your workload when you cannot capture a real trace. The lesson to carry over: benchmark across a *range* of load, not a single point, because the config that wins at 10 req/s is often not the one that wins at 40, and the frontier is what shows you the crossover.

**DistServe and the goodput objective.** DistServe (Zhong et al., OSDI 2024) is the work that made "goodput under joint TTFT and TPOT SLOs" the standard objective for LLM serving, and it did so to justify prefill/decode disaggregation — but the framing is what matters for tuning even without disaggregation. By reporting goodput rather than throughput, DistServe shows configurations that look worse on raw tokens/sec but serve substantially more requests *within SLO*, which is the same inversion our measurement table shows. If your benchmarks still report only throughput, you are optimizing the wrong number; adopt goodput and half the confusing results resolve themselves.

**Sarathi-Serve and chunked-prefill sizing.** Sarathi-Serve (Agrawal et al., OSDI 2024) introduced chunked prefills and stall-free batching, and its central experiment is a chunk-size sweep showing the TTFT-versus-TPOT exchange we reproduced in the second worked example. Its reported gains — large improvements in serving capacity under tight tail-latency SLOs — come precisely from *tuning the chunk size to the workload* rather than accepting a fixed prefill policy. The transferable result: chunked prefill is not a boolean you flip on; the chunk size is a tuned knob, and the right value depends on your prompt-length distribution.

Read the four together and one thread runs through all of them: the people closest to these systems — the framework maintainers, the benchmark authors, and the researchers who defined the objective — do not ship a magic default and walk away. They ship a *search*, a *sweep*, or a *goodput metric*, because every one of them learned that the right config is a property of the workload, not of the framework. The published numbers differ (treat each as approximate and specific to that team's model, hardware, and traffic), but the method they encode is identical to the one in this post, and it is the method — not their numbers — that transfers to your deployment.

## Avoiding overfitting and re-tuning cadence

A tuned config is a fit to a traffic distribution, and like any fit it can overfit. Three disciplines keep the tuning honest.

**Tune on a representative trace, validate on a held-out one.** Capture two traces from different days or different hours, tune on the first, and confirm the winner still holds on the second. If the optimal config swings wildly between the two, your trace was too short or too unusual to tune on, and you should widen the capture window. A config that only wins on the exact two hours you sampled is overfit and will regress the moment traffic drifts.

**Do not tune to a stale mix.** Traffic changes: a new feature ships, a customer onboards, prompt templates get longer, a caching layer changes the prefix distribution. Each of these moves the optimum. The config that was optimal for last quarter's 180-token prompts is not optimal for this quarter's 600-token RAG prompts. Treat the tuned config as having a shelf life, not as a permanent truth.

**Set a re-tuning cadence, and trigger it on drift.** Re-run the sweep on a schedule (quarterly is a reasonable default for stable products) *and* on triggers: a model version change, a major feature launch, a hardware change, or a monitored shift in the input/output length distribution or arrival pattern. The cheapest trigger is a dashboard: track p99 TTFT, p99 TPOT, goodput, and the median prompt/output lengths over time, and re-tune when the length distribution moves by more than, say, 20% or when goodput starts sliding at constant load. Do not re-tune weekly out of anxiety — each sweep costs GPU hours and a config change is a production change with its own risk — but do not let a config calcify for a year either. The re-tune is cheap; a quarter of silently degraded goodput is not.

## When to use this (and when not to)

Autotuning is not free, and it is not always worth it. Be honest about the stakes.

**Do the full search when:** the deployment is large enough that goodput translates to real money (you are running more than a handful of GPUs, or you are capacity-constrained and every percent of goodput defers a hardware purchase); the SLO is tight enough that the default plausibly violates it (interactive, sub-second TTFT); the workload is stable enough to be worth fitting (you are not re-architecting the feature every week); or you have already been paged for a latency incident, in which case the search pays for itself the first time. For a serious production LLM service, the two-knob grid alone is close to mandatory — it is a one-hour investment that routinely recovers 30-100% goodput.

**Skip it, or do the cheap version, when:** the stakes are low (an internal tool serving a few requests a minute on a single spare GPU — the default is fine, spend your time elsewhere); the SLO is loose (a batch job with a multi-second budget will meet its SLO under almost any config, so tuning throughput to the memory limit needs at most a five-minute coordinate-descent nudge, not a Bayesian study); or the workload is too new to characterize (you cannot tune to a trace you do not have — ship a sane default, collect a week of real traffic, *then* tune). And never let tuning become a substitute for capacity: if no config in the space is feasible at your load, the answer is more GPUs or a smaller/quantized model or prefill-decode disaggregation, not another sweep. The search finds the best point in the space you have; it cannot conjure a point that is not there.

The decision is a cost-benefit one, not a dogma. The method scales down as gracefully as it scales up: two knobs and a grid for the common case, a full Optuna study for the large coupled space, and nothing at all when the default already clears a loose SLO on a lightly loaded box.

## Key takeaways

- **Optimize goodput under a hard SLO constraint, not raw throughput.** A config that is faster on tokens/sec but misses the p99 is infeasible and scores zero. The default config in our worked example had the highest throughput and the lowest goodput.
- **The optimum is workload-specific.** Input/output length distribution, arrival pattern, and SLO priorities each move the sweet spot, often in opposite directions. A copied config is a coordinate from someone else's space; it rarely lands well in yours.
- **The space is not separable.** KV memory ties max-num-seqs, max-model-len, gpu-memory-utilization, and kv-cache-dtype together; batched-tokens *is* the chunk size. Sweep coupled knobs jointly or you stall short of the optimum.
- **Size the search strategy to the space.** Grid for one to three knobs (it is exhaustive and gives you the frontier for free), coordinate descent or random for four to six, Bayesian/Optuna for seven or more.
- **Always build the Pareto frontier, not just the winner.** It shows how close to the SLO edge you are and what a relaxed SLO would buy — turning "what config?" into a business conversation.
- **Evaluate on a replayed real trace, never on fixed prompts.** The prefill/decode ratio and burstiness of your traffic are what set the optimum; benchmark on a synthetic distribution and you tune to a fiction.
- **Two knobs move most of the goodput.** For most teams, a grid over max-num-seqs and max-num-batched-tokens, holding the rest at sane defaults, is the whole job.
- **A tuned config has a shelf life.** Re-tune on a cadence and on drift triggers (model change, feature launch, a 20% shift in length distribution), and never let tuning substitute for capacity when no config is feasible.

## Further reading

- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023** — the vLLM paper; the KV-cache mechanics that make max-num-seqs and gpu-memory-utilization the levers they are.
- **Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI 2024** — the formal definition of goodput under joint TTFT/TPOT SLOs that this post optimizes.
- **Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024** — chunked prefill and stall-free batching; the source for the chunk-size sweep.
- **Bergstra and Bengio, "Random Search for Hyper-Parameter Optimization," JMLR 2012** — why random search beats grid when only a few dimensions matter, which is the serving-config case.
- **Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," KDD 2019** — the TPE sampler and define-by-run search spaces used in the Bayesian harness.
- **vLLM documentation — optimization and tuning guide, and the `benchmarks/auto_tune` and `benchmarks/benchmark_serving.py` scripts** — the official knobs, the `--goodput` benchmark flag, and the in-box auto-tuning helper.
- **NVIDIA GenAI-Perf documentation** — concurrency and length-distribution sweeps for building the latency-throughput curve when you cannot capture a real trace.
- Within this series: [load testing and capacity planning](/blog/machine-learning/model-serving/load-testing-and-capacity-planning), [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption), [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management), [profiling LLM serving with Nsight](/blog/machine-learning/model-serving/profiling-llm-serving-with-nsight), [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization), and the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive).
