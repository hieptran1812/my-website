---
title: "The scheduler as a policy problem: FCFS, fair share, and why goodput is the only score that counts"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Your step loop makes three policy decisions every twenty milliseconds and nobody ever told you which ones. Learn to name the three levers, define goodput precisely enough to optimize it, build a pluggable scheduling policy in nanoserve, and prove with a simulator you can run that the batch which maximizes tokens per second can satisfy exactly zero users."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "scheduling",
    "batching",
    "latency",
    "throughput",
    "goodput",
    "preemption",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
---

Two posts ago you wrote a `step()` loop. It gathers the running requests, runs one fused forward pass, appends a token to each, retires whoever hit their stop condition, and pulls new work off a queue. It is about sixty lines and it works. What nobody tells you is that those sixty lines contain three policy decisions, they get made every twenty milliseconds, and right now you are making all three by accident.

Here is how you find out. Your service is fine at 40 requests per second. At 55 it is still fine — better, even, tokens per second is up 30% and the dashboard is green. At 58 the p99 time-to-first-token goes from 400 ms to 14 seconds, the support queue fills with "it just spins", and the tokens-per-second number on your dashboard is *higher than it has ever been*. Nothing crashed. No GPU is idle. Utilization reads 99%. The engine is working perfectly and every user is having a bad time.

![A single scheduler iteration branching into an admission decision a token budget split and a victim choice which merge into one step plan](/imgs/blogs/the-scheduler-as-a-policy-problem-1.webp)

That gap — between the machine being busy and the users being served — is the whole subject of this post. The scheduler is where you close it, and closing it means treating scheduling as what it actually is: a policy problem with a stated objective, not a queue you drain in order. By the end you will have a new file in `nanoserve` — `nanoserve/policy.py`, containing a `SchedulingPolicy` interface plus four implementations — a one-line way to swap between them without touching `step()`, and a CPU-only simulator that runs the same arrival stream through all four so you can see the difference on your laptop tonight.

The usual promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), restated because this post is full of numbers: **I have no GPU and I have run none of this.** Every figure below is either derived from arithmetic I show you in full, cited from a public source with a link, or framed as something you should expect when you run the provided script yourself. The results table carries a `Source` column and the simulator outputs are described as shapes to reproduce, never as measurements I took.

---

## 1. Three decisions, ten thousand times a minute

Strip `step()` down to its decisions and there are exactly three.

**Lever 1 — admission.** Some set of requests is waiting. Zero or more of them join the running set this iteration. Choosing *which* is the classic queueing decision, and it is the one that determines time-to-first-token (TTFT: the delay from a request arriving to its first token reaching the client).

**Lever 2 — the token budget split.** A step has a maximum number of tokens it will process, `max_num_batched_tokens` in vLLM's vocabulary. Every running request in decode wants exactly one of those tokens. Every request still in prefill wants as many as you will give it. The split between "new prefill work" and "keep the decoders moving" is a policy decision made every step, and it is the one that determines inter-token latency (ITL: the gap between consecutive tokens of one request). If you built chunked prefill in [the previous post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop), you have already implemented the mechanism for this lever without necessarily choosing a policy for it.

**Lever 3 — the victim.** Blocks run out. Some running request must give theirs back. Which one, and by which exit — swapped to host memory or dropped and recomputed — is [the eviction and preemption question](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping), and it determines whether your p99 is bad or catastrophic.

That is the entire decision space. It is worth sitting with how small it is, because the flag surface of a production engine is not small at all, and almost all of it collapses into these three:

| Knob you have seen                | Lever it pulls | What it actually says                              |
| --------------------------------- | -------------- | -------------------------------------------------- |
| `--max-num-seqs`                  | 1              | admit nobody past this concurrency                  |
| `--max-num-batched-tokens`        | 2              | this much token work per step, total                |
| `--long-prefill-token-threshold`  | 2              | cap how much of a step one prefill may eat          |
| `--scheduling-policy` fcfs / priority | 1          | the order the waiting queue is scanned in           |
| `--swap-space`                    | 3              | which exit a victim takes                           |
| `--gpu-memory-utilization`        | 1 and 3        | how big the pool is, so how often lever 3 fires     |

The vLLM team's [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) describes the same shape in production form: a waiting queue, a running queue, a KV-cache manager, and a policy that is "FCFS or priority". Their V1 rewrite made the budget lever explicit — per [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27), the V1 scheduler represents every decision as a plain dictionary `{request_id: num_tokens}`, which erases the prefill/decode distinction entirely. A request being prefilled at 512 tokens this step and a request decoding at 1 token this step are the same kind of entry. That is the right data structure, and we will copy it.

So the question this post answers is not "how do I schedule" — you already do. It is: **by what rule, and what does that rule cost?**

---

## 2. What are you actually optimizing?

Before comparing policies you need a scoreboard, and the obvious one is wrong.

### The metrics, defined once

Using the definitions the vLLM anatomy post uses, so we are all talking about the same thing:

- **TTFT** — time from request submission to its first output token.
- **ITL** — the interval between two consecutive output tokens of a single request.
- **TPOT** — time per output token, the average ITL over a request's whole output.
- **Throughput** — tokens per second summed over all requests, or requests per second completed.

An SLO (service level objective) is a promise about the first three. A realistic one for a chat product is *TTFT under 1 second and ITL under 50 ms* — that specific pair is the SLO the vLLM team used to evaluate their [MoRIIO KV connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07), so it is a defensible industry target rather than one I made up.

**Goodput** is the metric that combines them:

$$G = \frac{1}{T}\Big|\big\{\,r \in \text{completed} : \text{TTFT}_r \le \tau_{\text{ttft}} \;\wedge\; \text{ITL}^{p95}_r \le \tau_{\text{itl}} \,\big\}\Big|$$

In words: the number of requests per second that completed *and* met every part of their SLO. A request that finishes in 90 seconds when you promised 5 contributes tokens to your throughput and zero to your goodput. It is worse than useless — the user closed the tab at second 12 and you spent 78 seconds of GPU on output nobody read.

Two authoring notes on that formula, because they matter in practice. First, the per-request percentile: taking $\text{ITL}^{p95}$ *within* a request rather than a global histogram is deliberate. A global ITL histogram will look wonderful while one unlucky request stutters for a minute, because that request's samples are a rounding error among everyone else's. Second, a strict $\max$ instead of p95 is defensible but brutal — a single 200 ms hiccup from one preemption fails a request that was otherwise perfect. Pick p95 or p99 per request and write the choice down.

### The two ceilings, derived

Now the part that explains the dashboard-is-green failure. Model the decode step time for a batch of $N$ requests. During decode the engine must read, from HBM: the full weight tensor once, and every running request's KV cache once.

$$t_{\text{step}}(N) \;=\; \underbrace{\frac{W}{B}}_{\text{weights}} \;+\; \underbrace{\frac{\sum_{i=1}^{N} c_i \cdot k}{B}}_{\text{KV}} \;+\; \underbrace{\frac{P}{R_{\text{pf}}}}_{\text{prefill work}}$$

where $W$ is weight bytes, $B$ is HBM bandwidth, $c_i$ is request $i$'s context length, $k$ is KV bytes per token, $P$ is prefill tokens scheduled this step and $R_{\text{pf}}$ is prefill throughput. This is a first-order model — it ignores kernel launch overhead, activation traffic and the fact that a small batch does not saturate bandwidth — and it is accurate enough to predict the cliff, which is all we need.

Fill it in for the series spine: Llama-3.1-8B in bf16 on one A100 80GB SXM.

- Weights: $8.03 \times 10^9$ parameters $\times\, 2$ bytes $= 16.06$ GB. NVIDIA's A100 datasheet lists 2.0 TB/s of HBM2e bandwidth on the 80GB SXM part, so the weight read alone costs ${16.06 \times 10^9 / 2.0 \times 10^{12}} = 8.03$ ms. **That is the floor.** No decode step on this GPU is faster than 8 ms, at any batch size, ever.
- KV per token: $2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dims} \times 2 \text{ bytes} = 131{,}072$ bytes $=128$ KiB.
- A request holding 2,048 tokens of context therefore costs $2048 \times 131072 = 256$ MiB, and reading it costs $268.4 \times 10^6 / 2.0 \times 10^{12} = 0.134$ ms per step.

So at a uniform 2k context: $t_{\text{step}}(N) = 8.03 + 0.134N$ ms.

Two constraints bound $N$, and they are completely different in character.

**The capacity ceiling.** The pool is finite. An 80 GB A100 is 74.5 GiB; weights take 14.96 GiB; leave 3.5 GiB for activations and workspace; 56 GiB remains for KV. At the vLLM default block size of 16 tokens, one block holds $16 \times 128\text{ KiB} = 2$ MiB, so the pool is 28,672 blocks or 458,752 tokens. At 2k context that is ${458752/2048 = 224}$ concurrent requests. Past that, `allocate()` returns nothing and lever 3 fires.

**The latency ceiling.** ITL equals the step time, because in continuous batching every running request gets exactly one token per step. So an ITL target $\tau$ implies

$$N_{\max} \;=\; \frac{\tau - W/B}{\bar{c}\,k/B}$$

At $\tau = 20$ ms: $N_{\max} = (20 - 8.03)/0.134 = 89$ requests.

Here is the punchline. Throughput is $N/t_{\text{step}}(N)$, which **increases monotonically in $N$** and saturates at $1/0.134 = 7{,}463$ tok/s. Goodput is $N/(L \cdot t_{\text{step}})$ up to $N_{\max}$ and then falls off a cliff to *zero*, because at $N_{\max}+1$ every single request misses the ITL target simultaneously. There is no gentle degradation. One is a ramp; the other is a ramp with a trapdoor at the end.

![Two column comparison of a batch filled to the memory ceiling against a batch capped at the latency ceiling with step time tokens per second and requests inside the target](/imgs/blogs/the-scheduler-as-a-policy-problem-2.webp)

#### Worked example: the operator who filled the batch

Same GPU, same model, same 2k contexts, two admission policies.

| Policy                          | $N$ | $t_{\text{step}}$ | Throughput | Requests inside ITL 20 ms | Source  |
| ------------------------------- | --- | ----------------- | ---------- | ------------------------- | ------- |
| Admit until blocks run out      | 224 | 38.1 ms           | 5,879 tok/s | 0 of 224                 | derived |
| Admit until ITL budget runs out | 89  | 19.9 ms           | 4,457 tok/s | 89 of 89                 | derived |

The first row is 32% more tokens per second and a goodput of zero. The second row throws away a quarter of the machine's peak token rate and serves everybody. If your only dashboard is tokens per second, the first configuration looks like a promotion.

The capacity and latency ceilings also trade places depending on context length, which is why one tuning never survives a workload change. Same GPU, same model, three context regimes:

| Mean context | KV per request | Capacity ceiling | Latency ceiling at ITL 50 ms | Binding constraint | Source  |
| ------------ | -------------- | ---------------- | ---------------------------- | ------------------ | ------- |
| 2k tokens    | 256 MiB        | 224              | 313                          | capacity           | derived |
| 8k tokens    | 1 GiB          | 56               | 78                           | capacity           | derived |
| 80k tokens   | 10 GiB         | 5                | 7                            | capacity, barely   | derived |

At 80k context — which is exactly where agentic workloads live, as section 11 shows — five requests fill an 80 GB A100 and the latency ceiling is right behind at seven. Both constraints bite at once, and the scheduler has almost no room to be clever. Remember that when you are tempted to solve an agentic-serving problem with a better queue discipline.

---

## 3. FCFS: the honest baseline that blocks its own queue

First-come-first-served is the default in every engine, including vLLM's, and it deserves respect. It has one property no other policy has for free: **no starvation, provably**. A request's wait is bounded by the total service demand of the requests ahead of it, which is finite. Nothing you do to the arrival stream can leave a request in the queue forever. That property is worth more than it sounds when you are on call.

It also has one specific failure, and it is the one that will bite you first.

### Head-of-line blocking, at the block level

Look at the shape of the admission loop, because the bug is a single keyword.

```python
for r in sorted(waiting, key=lambda r: r.arrival):
    need = blocks_for(r.prompt_len)
    if free_blocks - need < watermark:
        break          # <-- FCFS: stop scanning. Order is preserved.
    admit(r)
```

That `break` is the FCFS invariant made executable. It says: if the request at the head does not fit, nobody behind it runs, because running them would violate arrival order.

Now the scenario. The engine is in steady state with 100 free blocks. At the head of the queue sits a document-summarization request with an 8,192-token prompt, which needs 512 blocks. Behind it are three chat turns with 200-token prompts, needing 13 blocks each — 39 blocks between them, which would fit four times over in the 100 blocks sitting idle right now.

FCFS runs the loop, hits `break`, and does nothing. The 100 blocks stay free. Blocks come back only as running requests finish; if completions return an average of 0.7 blocks per step in this steady state, reaching 512 free takes roughly $(512-100)/0.7 \approx 590$ steps, and at 20 ms per step that is **11.8 seconds**. Three requests that could have started producing tokens at step 1 instead have a TTFT of about twelve seconds, and the reason is not that the GPU was busy. The GPU had room. The policy refused to use it.

![A left to right sequence showing a queue head that needs 512 blocks stalling three small requests for 590 steps before it is admitted](/imgs/blogs/the-scheduler-as-a-policy-problem-3.webp)

### The one-word fix and what it costs

Change `break` to `continue` and the three chat turns start immediately. This is usually called skip-ahead or backfill, and every batch scheduler in HPC has it. It is also the moment you give up the starvation guarantee: with a steady stream of small requests, the 8k prompt can be skipped forever. Its wait is no longer bounded by anything.

That is the trade in its purest form. The FCFS invariant is bought with idle memory; skip-ahead spends the memory and sells the invariant. Neither is wrong. What is wrong is not knowing which one your `for` loop implements — and if you have never looked, it is whichever one you typed first.

The durable answer is to keep skip-ahead but bound how long anyone can be skipped, which is aging, which is the next section.

---

## 4. Priority, and the starvation you must design against

Priority scheduling replaces "sort by arrival" with "sort by tier, then arrival". Interactive traffic goes in tier 0, batch summarization in tier 2, and the queue drains accordingly. It is the single most requested scheduling feature in any serving system, and it introduces a failure mode that queueing theory quantified seventy years ago.

### How bad can starvation get? Exactly this bad

For a non-preemptive priority queue with classes $1 \dots n$ (class 1 most urgent), the classic waiting-time result — Cobham's formula, from the 1954 priority-queue analysis that every queueing text reproduces — gives the mean wait for class $k$ as

$$W_k \;=\; \frac{W_0}{(1-\sigma_{k-1})(1-\sigma_k)}, \qquad \sigma_k = \sum_{j \le k} \rho_j$$

where $\rho_j$ is the utilization contributed by class $j$ and $W_0$ is the mean residual work in the system. The structure is what matters. Class 1's wait depends only on $\rho_1$. Class 2's wait has $(1-\sigma_1)$ in the denominator, so it degrades as the *high* class gets busier, not as its own class does.

Put numbers on it. Suppose high-priority traffic occupies 60% of the machine ($\rho_1 = 0.6$), low-priority 30% ($\rho_2 = 0.3$), total utilization 90%, and mean residual work $W_0 = 1$ s.

- $W_1 = 1/((1)(1-0.6)) = 2.5$ s
- $W_2 = 1/((1-0.6)(1-0.9)) = 25$ s

Tenfold, at a utilization that any operator would call comfortable. Now let high-priority traffic grow to $\rho_1 = 0.7$ while low-priority stays at 0.3. Total utilization is 1.0, and the denominator becomes $(0.3)(0) = 0$: the low class's wait diverges. The high class is still being served in about 3.3 s and its dashboard looks *fine*. Starvation does not announce itself in the aggregate metrics; it announces itself in a support ticket from the one team whose jobs are tier 2.

### Aging: bound the wait, keep the tiers

The fix is standard and one line: let waiting time buy priority.

```python
def effective_priority(r, now, bump_after=2.0):
    """Every `bump_after` seconds of queueing promotes the request one tier."""
    gained = int((now - r.arrival) // bump_after)
    return max(r.priority - gained, 0)
```

Now derive what that guarantees. A request submitted at tier $P$ reaches tier 0 after $P \cdot t_{\text{bump}}$ seconds of waiting, and from then on it is ordered by arrival within the most urgent tier — which is exactly FCFS, whose bounded-wait property we established in section 3. So the total wait is at most $P \cdot t_{\text{bump}}$ plus the FCFS wait it would have had as a tier-0 request. With three tiers and a 2-second bump, no request ever waits more than 6 seconds longer than an equivalent urgent one. **Starvation is now a tunable number instead of a possibility**, and that is the whole point.

The cost is that aging erodes the priority guarantee it protects. If tier 2 keeps promoting itself into tier 0, tier 0 is no longer a fast lane; it is a slightly-less-slow lane. Set `bump_after` from the SLO you promised tier 2, not from a feeling.

---

## 5. Shortest job first, and the length you cannot know

Everything above is about *ordering*. Classical scheduling theory has a definitive answer to ordering, and it is one of the cleanest results in computer science: **shortest remaining processing time is optimal for mean response time.** Not "good in practice" — provably optimal, for any arrival sequence, as Schrage established in 1968. If you know how long each job will take, run the shortest remaining one and you cannot do better on average latency.

Here is the same worked example that the animated figure below runs, done by hand.

#### Worked example: four requests, three policies, identical throughput

Four requests arrive at $t=0$. Request R1 needs 40 steps of service; R2, R3 and R4 need 5 each. Total work is 55 steps under any policy, because the engine is work-conserving — it never idles while there is work. So **throughput is identical in all three cases**. Only latency moves.

- **FCFS, unlucky order (R1 first).** Completions at steps 40, 45, 50, 55. Mean flow time $(40+45+50+55)/4 = 47.5$ steps.
- **Fair share (all four progress at a quarter rate).** Each short request accumulates its 5 units of service by step 20, so R2, R3 and R4 all finish at 20. R1 has received 5 units by then, has 35 left, and runs alone from step 20 to step 55. Mean $(20+20+20+55)/4 = 28.75$ steps.
- **SJF (short ones first).** Completions at 5, 10, 15, 55. Mean $(5+10+15+55)/4 = 21.25$ steps.

Now add an SLO: a request must complete within 25 steps.

| Policy      | Mean flow time | Makespan  | Requests inside SLO | Goodput share | Source  |
| ----------- | -------------- | --------- | ------------------- | ------------- | ------- |
| FCFS        | 47.5 steps     | 55 steps  | 0 of 4              | 0%            | derived |
| Fair share  | 28.75 steps    | 55 steps  | 3 of 4              | 75%           | derived |
| SJF         | 21.25 steps    | 55 steps  | 3 of 4              | 75%           | derived |

Same machine. Same work. Same finishing time for the batch as a whole. Zero satisfied users versus three, decided entirely by the order.

<figure class="blog-anim">
<svg viewBox="0 0 700 340" role="img" aria-label="Four requests scheduled two ways on one shared clock; under first-come-first-served every bar finishes past the deadline line while under fair share three of them finish well before it" style="width:100%;height:auto;max-width:820px">
<style>
.s1-row{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.2}
.s1-bar{fill:var(--accent,#6366f1);opacity:.9;transform-box:fill-box;transform-origin:left center}
.s1-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.s1-lb{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
.s1-nt{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.s1-mid{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.s1-slo{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:6 5;opacity:.75}
.s1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes s1-a1{0%{transform:scaleX(.004)}72.7%,100%{transform:scaleX(1)}}
@keyframes s1-a2{0%{transform:scaleX(.004)}81.8%,100%{transform:scaleX(1)}}
@keyframes s1-a3{0%{transform:scaleX(.004)}90.9%,100%{transform:scaleX(1)}}
@keyframes s1-a4{0%{transform:scaleX(.004)}100%{transform:scaleX(1)}}
@keyframes s1-b2{0%{transform:scaleX(.011)}36.4%,100%{transform:scaleX(1)}}
@keyframes s1-fade{0%,93%{opacity:.9}99%,100%{opacity:0}}
.s1-1{animation:s1-a1 13s linear infinite,s1-fade 13s linear infinite}
.s1-2{animation:s1-a2 13s linear infinite,s1-fade 13s linear infinite}
.s1-3{animation:s1-a3 13s linear infinite,s1-fade 13s linear infinite}
.s1-4{animation:s1-a4 13s linear infinite,s1-fade 13s linear infinite}
.s1-s{animation:s1-b2 13s linear infinite,s1-fade 13s linear infinite}
@media (prefers-reduced-motion:reduce){.s1-1,.s1-2,.s1-3,.s1-4,.s1-s{animation:none;transform:scaleX(1);opacity:.9}}
</style>
<text class="s1-hd" x="20" y="22">FCFS — mean latency 47.5 steps, 0 of 4 inside the deadline</text>
<rect class="s1-row" x="100" y="34" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="60" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="86" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="112" width="560" height="20" rx="5"/>
<rect class="s1-bar s1-1" x="100" y="34" width="407" height="20" rx="5"/>
<rect class="s1-bar s1-2" x="100" y="60" width="458" height="20" rx="5"/>
<rect class="s1-bar s1-3" x="100" y="86" width="509" height="20" rx="5"/>
<rect class="s1-bar s1-4" x="100" y="112" width="560" height="20" rx="5"/>
<text class="s1-lb" x="92" y="48">R1 · 40 steps</text>
<text class="s1-lb" x="92" y="74">R2 · 5 steps</text>
<text class="s1-lb" x="92" y="100">R3 · 5 steps</text>
<text class="s1-lb" x="92" y="126">R4 · 5 steps</text>
<line class="s1-slo" x1="354" y1="28" x2="354" y2="140"/>
<text class="s1-nt" x="20" y="156">Three five-step requests spend their whole life queued behind one forty-step request.</text>
<text class="s1-mid" x="354" y="178">deadline · 25 steps</text>
<text class="s1-hd" x="20" y="204">Fair share — mean latency 28.75 steps, 3 of 4 inside the deadline</text>
<rect class="s1-row" x="100" y="216" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="242" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="268" width="560" height="20" rx="5"/>
<rect class="s1-row" x="100" y="294" width="560" height="20" rx="5"/>
<rect class="s1-bar s1-4" x="100" y="216" width="560" height="20" rx="5"/>
<rect class="s1-bar s1-s" x="100" y="242" width="204" height="20" rx="5"/>
<rect class="s1-bar s1-s" x="100" y="268" width="204" height="20" rx="5"/>
<rect class="s1-bar s1-s" x="100" y="294" width="204" height="20" rx="5"/>
<text class="s1-lb" x="92" y="230">R1 · 40 steps</text>
<text class="s1-lb" x="92" y="256">R2 · 5 steps</text>
<text class="s1-lb" x="92" y="282">R3 · 5 steps</text>
<text class="s1-lb" x="92" y="308">R4 · 5 steps</text>
<line class="s1-slo" x1="354" y1="210" x2="354" y2="322"/>
<line class="s1-ax" x1="100" y1="330" x2="660" y2="330"/>
<text class="s1-nt" x="100" y="326">0</text>
<text class="s1-nt" x="600" y="326">55 steps</text>
</svg>
<figcaption>Both policies finish the same forty-five steps of work in the same fifty-five steps of wall clock — the throughput is identical. Only the order changes, and with it three of the four requests move from outside the deadline to inside it.</figcaption>
</figure>

### The twist that makes LLM serving different

Every classical result above assumes you know the job size. In an LLM engine you do not, and you fundamentally cannot, because **the job size is the output length and the output length is decided by the model, one token at a time, until it emits EOS**. A 40-token prompt can produce 12 tokens or 12,000. The same prompt at temperature 0.7 can produce both, on different runs. This is not a measurement problem you can engineer around; it is the nature of the workload. The one input you would need to run the optimal policy is generated by the process you are scheduling.

Three responses exist, and all three have real costs.

**Predict the length.** A body of work does exactly this — the $S^3$ system (Jin et al., NeurIPS 2023, [arXiv:2306.06000](https://arxiv.org/abs/2306.06000)) predicts an output-length bucket to pack the batch better, and [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792) (Fu et al., 2024) argues for predicting the *relative order* of lengths rather than absolute values, which is a much easier learning problem and exactly what a scheduler needs.

But quantify the residual error before you trust it. Suppose output lengths are lognormal with $\sigma = 1$ — pull your own $\sigma$ from your logs, this is only a plausible shape. Then the ratio of the 90th percentile to the median is $e^{1.2816\sigma} = 3.6$. A predictor that nails the conditional median exactly is still off by a factor of 3.6 at the 90th percentile, because that spread is *in the distribution*, not in the model. And the loss is asymmetric: an underestimate means the request outlives its allocation and gets preempted (expensive); an overestimate means the scheduler reserves blocks it did not need and admits fewer requests (also expensive, quietly).

**Learn the length by running it.** This is multi-level feedback queueing, the answer operating systems settled on in the 1960s. Everyone enters the top queue with a small quantum; a job that exceeds its quantum is demoted to a slower queue with a bigger one. Short jobs finish before they are ever demoted, so you approximate SJF with no oracle at all. FastServe ([arXiv:2305.05920](https://arxiv.org/abs/2305.05920)) applies this idea to LLM serving with a skip-join variant that seeds a request into the queue matching its prompt length rather than demoting it through every level.

The cost is the reason this is not the universal answer. **A demotion is a preemption, and in an LLM engine a preemption costs the entire KV cache.** In an operating system a context switch is a register save — call it a microsecond. Here, if you take the recompute exit, it costs a re-prefill. With geometric quanta of 32, 64, 128, ... a 2,048-token request is demoted six times, and re-prefilling its context at each demotion sums to about 3,840 tokens of extra prefill work.

The way to price that is *not* against the victim's own lifetime — it is against the batch. Prefill throughput on an A100 at a plausible 40% model FLOPs utilization is $124.8 \times 10^{12} / (2 \times 8.03 \times 10^9) = 7{,}771$ tokens/s, so a single 2,048-token recompute adds ${2048/7771 = 264}$ ms to whatever step it lands in. Every one of the other 88 running requests takes a 264 ms hiccup — **13× their entire 20 ms ITL budget** — because one request got demoted. That is the LLM-specific constant factor, and it is why the theoretically optimal policy is often the practically worst one. Chunked prefill from [the previous post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) softens this: cap the recompute at 512 tokens per step and the hiccup drops to 66 ms. Still three ITL budgets.

**Stop needing the length.** Fair share and SLO-aware scheduling do not require job sizes at all, and the worked example above already showed fair share landing at 28.75 steps versus SJF's optimal 21.25 — about 60% of the way to an oracle policy, with no oracle. For most services that is the correct engineering trade, and it is where the next two sections go.

---

## 6. Fair share: nobody's batch job eats the cluster

Multi-tenancy changes the question from "which request" to "whose request". One customer submitting a thousand-document batch job should not push every other customer's chat traffic into the tail, and no per-request policy prevents that — the batch customer's requests are individually perfectly ordinary.

The classical answer is fair queueing: track how much service each tenant has received and serve whoever is furthest behind. The LLM adaptation has one wrinkle, which is that a "request" is a meaningless unit of service. One request can be 200 tokens or 200,000. **You must account in tokens, not requests**, and you should weight input and output tokens differently because they cost differently — a prefill token is compute-bound, a decode token is memory-bound, and on the numbers above a decode token on a full batch costs far more wall-clock per token than a prefill token does.

That is essentially the Virtual Token Counter scheme from [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) (Sheng et al., OSDI 2024), which defines fairness over a weighted count of input and output tokens served per client and shows that counting requests instead fails exactly as you would expect.

### The one subtlety that makes it correct

A naive counter has a bug that shows up on day two. Tenant B is idle for an hour while tenant A works. B's counter stays at zero while A's climbs to millions. B then submits one request and, being infinitely far behind, **monopolizes the engine until its counter catches up**. Idleness has become bankable credit, which is the opposite of fairness.

The fix: when a tenant transitions from having no work to having work, lift its counter to the minimum among currently active tenants. It rejoins as an equal, not as a creditor. This one rule is the difference between a fair-share scheduler and an outage.

### The burst bound you can promise a customer

If you want a contractual guarantee rather than a best effort, layer a token bucket on top. A tenant gets a bucket with refill rate $r$ tokens/s and capacity $C$ tokens; serving $n$ tokens draws $n$ from the bucket. Over any window of length $T$, the tenant can be served at most $C + rT$ tokens — that is the whole guarantee, and it follows immediately from the refill rule.

Feasibility is then a one-line admission check on your sales process: with $K$ tenants each guaranteed $r$, you need $K \cdot r \le S$ where $S$ is the engine's sustainable token rate. From section 2, the A100 at the latency ceiling sustains 4,457 tok/s, so twenty tenants each guaranteed 200 tok/s consume 4,000 tok/s and leave 457 tok/s of slack for bursts. Sell a twenty-first tenant and the guarantee is arithmetic fiction. It is worth knowing that number before the contract is signed rather than after.

---

## 7. SLO-aware: schedule to the deadline, refuse the rest

The last policy stops optimizing a proxy and optimizes the thing itself. Two changes from everything above.

**Order by deadline, not by arrival or size.** For TTFT, a request that arrived at $t_a$ has a deadline $t_a + \tau_{\text{ttft}}$, and the queue is sorted by that. This is earliest-deadline-first, which Liu and Layland's 1973 result showed is optimal for uniprocessor real-time scheduling in the sense that if *any* policy can meet all deadlines, EDF does. Note the condition carefully: EDF is optimal *when the set is feasible*. When it is not, EDF is notoriously bad — it thrashes, missing deadlines it could have met by giving up on one it could not. Which brings us to the second change.

**Admit only what you can serve.** This is the part most schedulers are missing. Section 2 gave you $t_{\text{step}}(N)$, an executable model of what a step will cost. So before admitting request $N+1$, ask whether $t_{\text{step}}(N+1)$ still fits inside the ITL target, and if it does not, **leave the request in the queue even though the GPU has room.** That is the schedulability test, and it is where "goodput not throughput" stops being a slogan and becomes eleven lines of Python.

This feels wrong the first time you write it. You are declining work with free memory and idle bandwidth on the table. But the derivation in section 2 says the alternative is not "slightly slower for everyone" — it is "SLO violated for everyone", including the requests already halfway through their output. Admitting one request past the ceiling does not degrade one request; it degrades $N+1$ of them. Refusing is the cheaper option by a factor of the batch size, and the deeper version of this argument — with backpressure, queue depth limits and 429s — is [admission control and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse).

The public evidence that this reframing pays is the MoRIIO evaluation in [the vLLM MoRIIO KV connector post](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07). Their setup, quoted with it because a number without a setup is not evidence: Qwen3-235B-A22B-FP8 on 8×MI300X split 4 prefill + 4 decode, input 2,000 tokens and output 1,000 tokens, 8 requests per second, with the SLO defined as TTFT under 1 s and ITL under 50 ms. Under that SLO the disaggregated write-mode configuration met the SLO for **73 of 100 requests**, read mode for 70, and the collocated baseline for **30** — reported as 2.5× higher goodput. The mechanism there is prefill/decode disaggregation rather than a queue discipline, but the framing is the point: they counted requests that met a stated SLO, not tokens.

---

## 8. The seam: a pluggable policy in nanoserve

Enough theory. The engineering goal is that swapping FCFS for fair share is a constructor argument, not a rewrite, which means all three levers must go behind one interface.

![One step loop calling a single scheduling interface that branches to four policy implementations which all merge back into one step plan](/imgs/blogs/the-scheduler-as-a-policy-problem-4.webp)

The interface is deliberately small: two decisions and one accounting hook. Anything larger and policies start reaching into engine internals; anything smaller and stateful policies like fair share cannot work.

```python
# nanoserve/policy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

BLOCK_SIZE = 16  # tokens per KV block, matching nanoserve/blocks.py


@dataclass
class Request:
    rid: str
    tenant: str = "anon"
    priority: int = 0                 # 0 is the most urgent tier
    arrival: float = 0.0              # seconds, when it entered `waiting`
    prompt_len: int = 0
    prefilled: int = 0                # prompt tokens processed so far
    generated: int = 0                # output tokens emitted so far
    last_token_at: float | None = None
    preemptions: int = 0

    @property
    def ctx(self) -> int:
        """Tokens currently occupying KV blocks."""
        return self.prefilled + self.generated

    @property
    def blocks_held(self) -> int:
        return math.ceil(self.ctx / BLOCK_SIZE)

    def blocks_for(self, new_tokens: int) -> int:
        """Extra blocks needed to append `new_tokens` more tokens."""
        return math.ceil((self.ctx + new_tokens) / BLOCK_SIZE) - self.blocks_held


@dataclass
class SchedulerState:
    now: float
    free_blocks: int
    token_budget: int                 # tokens left in this step's budget
    running: list[Request]
    waiting: list[Request]


@dataclass
class Admission:
    req: Request
    tokens: int                       # prefill tokens to schedule this step
    blocks: int                       # blocks to reserve now


@runtime_checkable
class SchedulingPolicy(Protocol):
    name: str

    def select_admits(self, st: SchedulerState) -> list[Admission]:
        """Lever 1 and 2: who joins `running`, and how much budget each gets.
        Must respect st.token_budget and st.free_blocks."""

    def select_victim(self, st: SchedulerState) -> Request | None:
        """Lever 3: who gives their blocks back. None means 'stall the step'."""

    def on_step(self, st: SchedulerState, served: dict[str, int]) -> None:
        """Called after each step with {rid: tokens served} so stateful
        policies can do their accounting."""
```

FCFS becomes the reference implementation, and every other policy subclasses it by overriding one method.

```python
class FCFS:
    name = "fcfs"

    def __init__(self, watermark: int = 256, max_running: int = 256):
        self.watermark = watermark        # blocks reserved for decode growth
        self.max_running = max_running

    def _order(self, st: SchedulerState) -> list[Request]:
        return sorted(st.waiting, key=lambda r: r.arrival)

    def select_admits(self, st: SchedulerState) -> list[Admission]:
        admits: list[Admission] = []
        free, budget = st.free_blocks, st.token_budget
        slots = self.max_running - len(st.running)
        for r in self._order(st):
            if slots <= 0 or budget <= 0:
                break
            take = min(budget, r.prompt_len - r.prefilled)
            need = r.blocks_for(take)
            if take <= 0 or free - need < self.watermark:
                break   # head-of-line blocking: `continue` here is skip-ahead
            admits.append(Admission(r, take, need))
            free -= need
            budget -= take
            slots -= 1
        return admits

    def select_victim(self, st: SchedulerState) -> Request | None:
        # Last admitted, first preempted. The newest request has the least
        # accumulated work to lose, and evicting it guarantees the oldest
        # request in the running set makes progress. Section 12 shows what
        # happens when you pick the other end.
        if not st.running:
            return None
        return max(st.running, key=lambda r: r.arrival)

    def on_step(self, st: SchedulerState, served: dict[str, int]) -> None:
        pass
```

Priority with aging is now four lines, because ordering is the only thing that changes.

```python
class PriorityAging(FCFS):
    name = "priority+aging"

    def __init__(self, bump_after: float = 2.0, **kw):
        super().__init__(**kw)
        self.bump_after = bump_after   # seconds of waiting per tier gained

    def _order(self, st):
        def key(r):
            aged = max(r.priority - int((st.now - r.arrival) // self.bump_after), 0)
            return (aged, r.arrival)
        return sorted(st.waiting, key=key)
```

Fair share needs state, which is what `on_step` is for. Note the counter-lifting rule from section 6 — without it this class is a denial-of-service vector against yourself.

```python
class FairShare(FCFS):
    name = "fair-share"

    def __init__(self, weights=None, w_in=1.0, w_out=2.0, **kw):
        super().__init__(**kw)
        self.weights = weights or {}       # tenant -> share weight
        self.counter: dict[str, float] = {}
        self.w_in, self.w_out = w_in, w_out

    def _w(self, tenant: str) -> float:
        return self.weights.get(tenant, 1.0)

    def _virtual_time(self, tenant: str) -> float:
        return self.counter.get(tenant, 0.0) / self._w(tenant)

    def _order(self, st):
        # An idle tenant must not bank credit: on becoming active, lift its
        # counter to the minimum among tenants already being served.
        active = {r.tenant for r in st.running}
        if active:
            floor = min(self.counter.get(t, 0.0) for t in active)
            for r in st.waiting:
                self.counter[r.tenant] = max(self.counter.get(r.tenant, 0.0), floor)
        return sorted(st.waiting,
                      key=lambda r: (self._virtual_time(r.tenant), r.arrival))

    def on_step(self, st, served):
        by_rid = {r.rid: r for r in st.running}
        for rid, tokens in served.items():
            r = by_rid.get(rid)
            if r is None:
                continue
            weight = self.w_out if r.prefilled >= r.prompt_len else self.w_in
            self.counter[r.tenant] = self.counter.get(r.tenant, 0.0) + weight * tokens

    def select_victim(self, st):
        # Take blocks from the tenant that is furthest ahead on service.
        if not st.running:
            return None
        return max(st.running,
                   key=lambda r: (self._virtual_time(r.tenant), r.arrival))
```

For hard guarantees rather than proportional ones, the token bucket from section 6 drops in beside it:

```python
@dataclass
class TokenBucket:
    rate: float                  # tokens per second this tenant is entitled to
    burst: float                 # tokens it may bank while idle
    level: float = 0.0
    stamp: float = 0.0

    def take(self, now: float, tokens: float) -> bool:
        self.level = min(self.burst, self.level + (now - self.stamp) * self.rate)
        self.stamp = now
        if self.level < tokens:
            return False
        self.level -= tokens
        return True
```

The SLO-aware policy is where the section 2 derivation becomes executable. Note that `step_seconds` is the same model that produced the tables above, so the numbers in this file and the numbers in this post cannot drift apart.

```python
# Step-time model from section 2, in seconds. Constants are Llama-3.1-8B in
# bf16 on one A100 80GB SXM. Re-derive them for your model and your GPU.
WEIGHT_BYTES = 16.06e9
KV_BYTES_PER_TOKEN = 131_072          # 2 * 32 layers * 8 kv heads * 128 dims * 2 B
HBM_BYTES_PER_S = 2.0e12              # A100 80GB SXM datasheet figure
PREFILL_TOKENS_PER_S = 7_771          # 124.8 TFLOP/s effective / (2 * 8.03e9)


def step_seconds(running, prefill_tokens: int = 0) -> float:
    weights = WEIGHT_BYTES / HBM_BYTES_PER_S
    kv = sum(r.ctx for r in running) * KV_BYTES_PER_TOKEN / HBM_BYTES_PER_S
    prefill = prefill_tokens / PREFILL_TOKENS_PER_S
    return weights + kv + prefill


class SLOAware(FCFS):
    name = "slo-aware"

    def __init__(self, ttft_slo=1.0, itl_slo=0.050, guard=0.85, **kw):
        super().__init__(**kw)
        self.ttft_slo, self.itl_slo, self.guard = ttft_slo, itl_slo, guard

    def _order(self, st):
        return sorted(st.waiting, key=lambda r: r.arrival + self.ttft_slo)  # EDF

    def select_admits(self, st):
        admits, free, budget = [], st.free_blocks, st.token_budget
        cohort = list(st.running)
        for r in self._order(st):
            take = min(budget, r.prompt_len - r.prefilled)
            need = r.blocks_for(take)
            if take <= 0 or free - need < self.watermark:
                break
            # Schedulability test: would admitting this request push the step
            # time past the ITL target for everyone already running?
            if step_seconds(cohort + [r], take) > self.guard * self.itl_slo:
                break
            admits.append(Admission(r, take, need))
            cohort.append(r)
            free -= need
            budget -= take
        return admits

    def select_victim(self, st):
        # Preempt whoever has the most slack against their next-token deadline.
        if not st.running:
            return None
        def slack(r):
            due = (r.last_token_at or r.arrival) + self.itl_slo
            return due - st.now
        return max(st.running, key=slack)
```

And now the payoff: here is `step()` with the policy wired in, and it is the *last* time this loop changes in this series. Every policy above, and every policy you invent later, plugs into these three call sites.

```python
# nanoserve/engine.py  (excerpt — the loop from the continuous-batching post)
def step(self) -> StepOutput:
    st = SchedulerState(
        now=time.monotonic(),
        free_blocks=self.pool.free(),
        token_budget=self.max_num_batched_tokens,
        running=self.running,
        waiting=self.waiting,
    )

    # LEVERS 1 and 2: who joins, and how much of the budget each gets.
    for a in self.policy.select_admits(st):
        self.pool.reserve(a.req, a.blocks)
        a.req.prefilled += a.tokens
        st.token_budget -= a.tokens
        self.waiting.remove(a.req)
        self.running.append(a.req)

    # LEVER 3: decoders need room to grow. The drain rate is len(running) /
    # BLOCK_SIZE blocks per step, so reserve at least that much or preempt.
    need = len(self.running) // BLOCK_SIZE + 1
    while self.pool.free() < need:
        victim = self.policy.select_victim(st)
        if victim is None:
            break
        self.preempt(victim)                  # swap or drop, per the eviction post
        self.metrics.preemptions += 1

    out = self.model_step(self.running)       # one fused prefill + decode forward
    self.policy.on_step(st, out.served)       # {rid: tokens served this step}
    return out
```

Read what did *not* happen there. `step()` contains no `sorted()`, no priority comparison, no notion of a tenant, and no SLO. It knows how to reserve blocks, run a forward pass and preempt. Everything that constitutes a *decision* lives on the other side of three method calls. That is the whole design, and it is what lets the next section run four policies over one arrival stream without a single `if policy == ...`.

---

## 9. Simulate it, because you cannot afford to guess

You now have four policies and no way to compare them. Running an A/B on production traffic to find out whether fair share beats FCFS is an expensive way to learn something a laptop can tell you in ten seconds — and if the answer is "FCFS collapses at 58 rps", you really do not want to learn it in production.

Build a simulator. The key insight is that you already have the only hard part: `step_seconds()` is a model of the engine's cost, and the policies are already pure functions of `SchedulerState`. So the simulator is the same loop as `step()`, with the forward pass replaced by arithmetic. It runs in pure Python with no GPU, no torch, and no dependencies.

```python
# sim.py — a step-driven scheduler simulator. Pure stdlib, no GPU.
# Usage: python sim.py --policy fcfs --rate 6 --seconds 120 --seed 0
import argparse, math, random, statistics
from nanoserve.policy import (
    Request, SchedulerState, FCFS, PriorityAging, FairShare, SLOAware,
    step_seconds, BLOCK_SIZE,
)

POOL_BLOCKS = 28_672            # 56 GiB of KV on an 80 GB A100, 2 MiB per block
TTFT_SLO, ITL_SLO = 1.0, 0.050
TENANTS = ["alpha", "beta", "gamma"]


def workload(rng, rate, seconds):
    """Open-loop Poisson arrivals with lognormal prompt and output lengths.
    Open-loop matters: a closed-loop generator with N virtual users throttles
    itself when the server slows down, which hides exactly the collapse we
    are trying to observe."""
    t, out, i = 0.0, [], 0
    while t < seconds:
        t += rng.expovariate(rate)
        heavy = rng.random() < 0.15                    # 15% long-context traffic
        mu = math.log(6000) if heavy else math.log(400)
        prompt = int(rng.lognormvariate(mu, 0.6))
        gen = int(rng.lognormvariate(math.log(120), 1.0))
        r = Request(
            rid=f"r{i}",
            tenant=rng.choice(TENANTS),
            priority=0 if rng.random() < 0.10 else 1,  # 10% urgent traffic
            arrival=t,
            prompt_len=max(16, min(prompt, 32_000)),
        )
        out.append((t, r, max(8, min(gen, 4096))))
        i += 1
    return out
```

The loop itself. It is deliberately the same shape as `engine.step()` so that a bug in one is visible in the other.

```python
def simulate(policy, arrivals, budget=8192):
    now, pending = 0.0, list(arrivals)
    waiting, running, done = [], [], []
    free = POOL_BLOCKS
    target, ttft, itl = {}, {}, {}

    def grow(r, n):
        """Reserve blocks for n more tokens. Returns False if the pool is dry."""
        nonlocal free
        need = r.blocks_for(n)
        if need > free:
            return False
        free -= need
        return True

    while pending or waiting or running:
        while pending and pending[0][0] <= now:
            _, r, gen = pending.pop(0)
            target[r.rid] = gen
            waiting.append(r)
        if not running and not waiting and pending:
            now = pending[0][0]           # engine idle: jump to the next arrival
            continue

        st = SchedulerState(now, free, budget, running, waiting)
        used = 0
        for a in policy.select_admits(st):
            free -= a.blocks
            a.req.prefilled += a.tokens
            used += a.tokens
            waiting.remove(a.req)
            running.append(a.req)

        if not running:
            # Nothing running and nothing admittable means the head of the queue
            # does not fit in an empty pool. A real engine rejects it at the
            # front door; here we drop it rather than spin forever.
            if waiting:
                waiting.pop(0)
            continue

        # Spend the rest of the budget continuing in-flight prefills.
        for r in running:
            room = budget - used
            if room <= 0:
                break
            if r.prefilled < r.prompt_len:
                take = min(room, r.prompt_len - r.prefilled)
                if not grow(r, take):
                    break
                r.prefilled += take
                used += take

        # Decode: one token each, preempting when the pool cannot cover it.
        served = {}
        for r in [x for x in running if x.prefilled >= x.prompt_len]:
            if r not in running:
                continue
            while not grow(r, 1):
                victim = policy.select_victim(
                    SchedulerState(now, free, 0, running, waiting))
                if victim is None or victim is r:
                    break
                free += victim.blocks_held
                victim.preemptions += 1
                # Recompute on resume: the emitted tokens are known, their KV
                # is not, so the prompt to re-prefill grows by what was lost.
                target[victim.rid] -= victim.generated
                victim.prompt_len += victim.generated
                victim.generated = 0
                victim.prefilled = 0
                running.remove(victim)
                waiting.append(victim)
            else:
                r.generated += 1
                served[r.rid] = 1

        dt = step_seconds(running, used)
        now += dt
        for r in list(running):
            if r.generated >= 1 and r.rid not in ttft:
                ttft[r.rid] = now - r.arrival
            if r.generated >= 2:
                itl.setdefault(r.rid, []).append(dt)
            r.last_token_at = now
            if r.generated >= target[r.rid]:
                free += r.blocks_held
                running.remove(r)
                done.append(r)
        policy.on_step(st, served)

    return done, ttft, itl, now
```

And the scoreboard, which is the only part that encodes the argument of this post: it reports throughput and goodput side by side so you can watch them diverge.

```python
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def report(policy, done, ttft, itl, wall):
    ttfts = [ttft[r.rid] for r in done if r.rid in ttft]
    met = sum(1 for r in done
              if ttft.get(r.rid, 9e9) <= TTFT_SLO
              and pct(itl.get(r.rid, [0.0]), 0.95) <= ITL_SLO)
    print(f"{policy.name:<15} done={len(done):<5} "
          f"ttft_mean={statistics.mean(ttfts) * 1e3:8.1f} ms  "
          f"ttft_p99={pct(ttfts, 0.99) * 1e3:8.1f} ms  "
          f"throughput={len(done) / wall:5.2f} rps  "
          f"goodput={met / wall:5.2f} rps  "
          f"preempts={sum(r.preemptions for r in done)}")


POLICIES = {"fcfs": FCFS, "priority": PriorityAging,
            "fair": FairShare, "slo": SLOAware}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="fcfs", choices=sorted(POLICIES))
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arrivals = workload(random.Random(a.seed), a.rate, a.seconds)
    pol = POLICIES[a.policy]()
    report(pol, *simulate(pol, arrivals))
```

Run all four on one arrival stream:

```bash
for p in fcfs priority fair slo; do python sim.py --policy "$p" --rate 6 --seed 0; done
```

### What you should see, and what you should not believe

**I have not run this.** What follows is the shape of the result the model predicts, and the honest instruction is to run it yourself and check whether your output has that shape — not whether it matches numbers I have printed.

The absolute values are meaningless anyway: they are the output of a cost model, not of a GPU. What is *not* meaningless is the ordering between columns on one seed, because every policy sees the identical arrival stream. Specifically, sweep `--rate` from well below capacity to well above and watch for these:

| Prediction to check                                                                 | Why the model implies it                            | Source            |
| ----------------------------------------------------------------------------------- | --------------------------------------------------- | ----------------- |
| At low rate, all four policies report nearly identical numbers                       | no queue forms, so ordering never matters            | reproduce: sim.py |
| As rate rises, throughput keeps climbing for FCFS while goodput turns over and falls  | the two ceilings of section 2                        | reproduce: sim.py |
| FCFS mean TTFT stays acceptable while its p99 blows out first                        | head-of-line blocking hits the unlucky minority      | reproduce: sim.py |
| Fair share moves p99 TTFT far more than it moves the mean                            | it reorders the tail, not the median                 | reproduce: sim.py |
| SLO-aware completes *fewer* requests but more of them meet the SLO                   | the schedulability test refuses admissible work      | reproduce: sim.py |
| Preemption counts rise superlinearly past the capacity ceiling                        | every preempt re-prefills, stealing budget           | reproduce: sim.py |

If your run contradicts one of those, the interesting possibility is that your workload parameters put you in a different regime — try shrinking `POOL_BLOCKS` until the capacity ceiling binds, or raising the heavy-request fraction above 0.15. That exploration is the point of having a simulator.

Two things this simulator deliberately does not model, so you do not over-trust it: the GPU's actual kernel behaviour at small batch (where launch overhead dominates and the linear model is optimistic), and any prefix-cache hits, which in the agentic workload of section 11 change everything.

---

## 10. Reading the comparison honestly

![A five row by four column comparison of scheduling policies against mean latency tail risk oracle requirement and extra preemptions](/imgs/blogs/the-scheduler-as-a-policy-problem-5.webp)

The matrix above compresses the argument, and three cells deserve a sentence each.

**Nobody dominates.** SJF-family policies win mean latency and lose the tail; fair share wins the tail and gives up some mean; SLO-aware wins goodput and gives up completions. Any vendor claiming a policy that wins every column is measuring one workload.

**"Needs a length oracle" is the column that eliminates candidates.** In production, a policy that degrades gracefully when its inputs are wrong beats a policy that is optimal when they are right. Length prediction is wrong by a factor of 3.6 at p90 for a lognormal with $\sigma = 1$, as derived in section 5, and there is no version of your service where that error goes away.

**"Extra preemptions" is a hidden multiplier on everything else.** A policy that preempts more is not slightly worse; it is worse by the recompute cost of the whole batch's stalled step, which section 5 put at 264 ms for one 2,048-token victim against a 20 ms ITL budget.

My default recommendation, stated plainly: **fair share on the admission lever, last-admitted-first on the victim lever, with an SLO-based schedulability cap on top.** Fair share needs no oracle and bounds per-tenant tail; last-admitted-first minimizes lost work and guarantees the oldest request progresses; the cap is what stops the whole thing collapsing under overload. Priority tiers only if you have a real business reason, and only with aging.

---

## 11. Workload shape beats policy cleverness

Now the part that reorders everything above.

Everything so far treated requests as independent arrivals. That is a chat workload: short prompt, long-ish output, no relationship between consecutive requests. An agentic workload — a coding assistant, a tool-using research agent — is a different animal, and the vLLM team published enough of its shape to reason about.

From [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06), measured over 610 Codex and SWE-bench agentic traces:

- input-to-output token ratio of **131 to 1**
- median of **33 turns** per session
- roughly **2,242 tokens of context growth per turn**
- about **80,000 tokens of context at turn 30**
- inter-turn delay with a median of **5.2 seconds** and a P99 of **81.4 seconds**

![A left to right sequence of an agentic session growing from two thousand to eighty thousand tokens of context across turns with idle gaps between them](/imgs/blogs/the-scheduler-as-a-policy-problem-6.webp)

Read those five numbers as scheduling requirements and almost everything above changes.

**The 131:1 ratio means this is a prefill machine.** Lever 2, the budget split, dominates lever 1. Optimizing decode ordering on a workload that spends 99% of its tokens in prefill is optimizing the 1%.

**33 turns with 5.2-second gaps means the same session comes back — and the scheduling question becomes eviction, not ordering.** Between turns the session is not running. Its KV cache is 10 GiB at turn 30 (80,000 tokens × 128 KiB, from section 2's arithmetic). Do you keep it or drop it?

#### Worked example: how long to hold an idle session's cache

Two costs to compare, and they are in different units, which is why this decision is usually made by vibes.

*Recompute cost.* Re-prefilling 80,000 tokens at 7,771 tokens/s is ${81920/7771 = 10.5}$ seconds of the entire GPU. That is the cost of dropping the cache and paying for it on the next turn.

*Retention cost.* Holding 10 GiB of a 56 GiB pool is a fraction $f = 10/56 = 0.179$ of the machine's admission capacity, denied to everyone else for the duration of the idle gap. If the pool is not saturated, this cost is **zero** — you are holding memory nobody wanted. If the pool is saturated, holding fraction $f$ for $T_{\text{idle}}$ seconds costs the engine $f \cdot T_{\text{idle}}$ GPU-seconds of foregone work.

Break-even is where those meet:

$$T^{*}_{\text{idle}} \;=\; \frac{T_{\text{prefill}}}{f} \;=\; \frac{10.5\ \text{s}}{0.179} \;=\; 59\ \text{s}$$

So: **hold an idle 80k-token session for about a minute, then evict it.** Compare against the trace: the median inter-turn gap is 5.2 s, which is eleven times under the threshold — hold, without hesitation. The P99 gap is 81.4 s, which is past it — those sessions should be evicted and re-prefilled. A fixed session TTL of roughly 60 seconds captures the right behaviour for this workload, and now it is a derived number rather than a guess. Re-derive it for your own model and pool: the formula is $T^* = T_{\text{prefill}} / f$ and both terms are things you know.

Note the conditional that makes the whole calculation honest — retention costs nothing while the pool has slack. So the correct implementation is not a fixed TTL at all; it is *evict idle sessions only under memory pressure, oldest-idle first*, with the TTL as a backstop. Which is to say: the answer to an agentic scheduling problem turned out to live in the eviction policy, not the queue discipline.

**Session affinity is the cluster-level version of the same decision.** All of this is worthless if turn 31 lands on a different replica from turn 30, because that replica has none of the 80,000 tokens cached. This is why the [vLLM Router](https://vllm.ai/blog/2025-12-13-vllm-router-release) (2025-12-13) offers consistent hashing on a session or user key alongside the conventional Power-of-Two and Round Robin strategies: hashing pins a session to a replica specifically to maximize KV reuse, accepting *worse* instantaneous load balance in exchange for cache hits. That is the same trade as section 7's schedulability test, one layer up — refusing the locally optimal placement because the globally optimal one is elsewhere. The vLLM team reports that configuration, on Llama 3.1 8B with 8 prefill and 8 decode pods, delivering 25% higher throughput than llm-d and roughly 1,200 ms faster TTFT.

The general lesson is worth stating bluntly, because it will save you weeks: **before tuning a queue discipline, characterize your traffic.** Turn count, context growth per turn, idle-gap distribution and prefix overlap will tell you more about what to build than any amount of policy theory. A chat service and an agentic service running the same model on the same GPU want genuinely different schedulers.

---

## 12. Stress tests: three ways a policy goes wrong in production

A policy is defined as much by its failure modes as by its steady state. Here are the three that show up.

### Preemption thrash from a bad victim rule

Pick the wrong victim and the engine can do nearly zero useful work at full utilization. The bad rule is seductive: *"preempt the request closest to finishing, it will be back soon"* or its cousin *"preempt the biggest one, it frees the most blocks"*. Both preferentially destroy the most accumulated work in the system.

The pathological case is easy to construct. If the victim is always the most-progressed request, then whenever the pool is tight, the request nearest completion is repeatedly reset. Under sustained pressure it never finishes. Meanwhile the requests that displaced it grow, become the most-progressed, and get preempted in turn. Nobody finishes. Utilization is 100%, throughput is near zero, and this is the livelock condition that [the eviction and preemption post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) describes in detail.

Quantify the safe operating region. Let $p$ be preemptions per step and $c$ the average recompute size in tokens. Added time per step is $p \cdot c / R_{\text{pf}}$ seconds. To keep that under 10% of a 20 ms step:

$$\frac{p \cdot c}{7771} < 0.002 \quad\Longrightarrow\quad p \cdot c < 15.5\ \text{tokens per step}$$

With $c = 2048$, that is $p < 0.0076$ — **fewer than one preemption per 130 steps.** If your engine preempts more often than that, more than a tenth of the machine is spent redoing work it already did, and no scheduling policy will save you; you need less load or more memory. Put preemptions-per-step on the dashboard next to tokens-per-second and treat 0.8% as the red line.

Two rules keep you inside it. Prefer last-admitted-first-preempted, because the newest request has the least work to lose. And never preempt a request holding shared prefix blocks that others are reading — copy-on-write refcounts make that eviction illegal anyway, and forcing it corrupts every reader.

### Priority inversion

A high-priority request cannot proceed because a low-priority request is holding what it needs. In an inference engine this happens in two concrete ways.

The first is memory. The high-priority request needs blocks; the only free blocks would come from preempting a low-priority request that holds a *shared prefix* several other requests are reading. The refcount is not zero, so it cannot be evicted, and the urgent request waits on the unimportant one. The fix is the classical one, priority inheritance: temporarily raise the holder's priority so the scheduler works to retire it rather than leaving it parked.

The second is subtler and specific to LLM engines: **the step budget itself is a shared resource.** A low-priority request in the middle of a 32,000-token chunked prefill consumes budget every step. Even if the high-priority request is admitted instantly — TTFT looks great — its ITL is degraded by the prefill work sharing every step with it. Priority on the admission lever did nothing for it, because the damage is happening on the budget lever. If your tiers are meant to mean something, they must apply to the token budget too: cap what fraction of `max_num_batched_tokens` low-priority prefill may take.

### Everyone claims to be urgent

The most reliable finding in the operation of any priority system: given a free `priority` field, every caller sets it to the maximum. Within two quarters, 90% of your traffic is tier 0, tier 0 is the new baseline, and you have a strictly worse FCFS with extra code.

This is not a technical failure, it is an incentive failure, and the fix is to make priority scarce. **Priority must be a budget, not a claim.** Give each tenant a bucket of high-priority tokens per hour — the `TokenBucket` from section 8 does this directly — and when the bucket empties, their traffic is served at the normal tier. Suddenly "is this request urgent" is a question the caller has a reason to answer accurately.

The rate limit also restores the guarantee that makes tiers work at all. Section 4's Cobham formula blows up as $\sigma_1 \to 1$; capping the urgent class at $\rho_1 \le 0.5$ by admission keeps the low-priority wait at $W_0/((0.5)(0.5-\rho_2))$, which is finite and computable for any $\rho_2 < 0.5$. **A priority tier without a rate limit is not a scheduling policy, it is a promise you have no mechanism to keep.**

---

## 13. The same three levers, one layer up

![A layered stack from a session hashing router down through the engine scheduler token budget and block allocator converging on goodput](/imgs/blogs/the-scheduler-as-a-policy-problem-7.webp)

Zoom out and the structure repeats, which is the most useful thing to know when you cannot fix a problem at the layer where you found it.

At the **cluster** layer, the router admits (which replica gets this request), splits (prefill pods versus decode pods, which is exactly a token-budget split expressed in hardware) and preempts (drain a replica, evict a session). Consistent hashing on a session key is an admission decision made for cache affinity rather than for load, per the vLLM Router post above.

At the **engine** layer, it is everything this post described.

At the **allocator** layer, block allocation admits (reserve or refuse), splits (watermarks reserving blocks for decode growth versus new prefills) and preempts (evict a cached block, choose a victim).

They differ only in time constant — seconds at the router, milliseconds at the scheduler, microseconds at the allocator — and that is what tells you where a fix belongs. A problem whose timescale is "this session over ten minutes" cannot be fixed by a policy that re-decides every 20 ms. Session affinity is a routing problem. Head-of-line blocking is a scheduler problem. Fragmentation is an allocator problem. Solving one at the wrong layer produces a policy that oscillates, which is worse than either fix alone.

---

## 14. The numbers, with provenance

Every quantitative claim in this post, with where it came from.

| Quantity                                              | Value                       | Source                            |
| ----------------------------------------------------- | --------------------------- | --------------------------------- |
| KV bytes per token, Llama-3.1-8B bf16                 | 128 KiB                     | derived                           |
| Weight-read floor per decode step, A100 80GB          | 8.03 ms                     | derived                           |
| KV read cost per request at 2k context                | 0.134 ms                    | derived                           |
| KV pool after weights and workspace, 80 GB A100       | 56 GiB = 28,672 blocks      | derived                           |
| Capacity ceiling at 2k / 8k / 80k context             | 224 / 56 / 5 requests       | derived                           |
| Latency ceiling at ITL 20 ms, 2k context              | 89 requests                 | derived                           |
| Throughput at capacity vs latency ceiling             | 5,879 vs 4,457 tok/s        | derived                           |
| Prefill rate at 40% MFU on A100                       | 7,771 tok/s                 | derived                           |
| Recompute stall from one 2,048-token victim           | 264 ms                      | derived                           |
| Safe preemption rate at 10% overhead                  | under 1 per 130 steps       | derived                           |
| Mean flow time, four-job example (FCFS / fair / SJF)  | 47.5 / 28.75 / 21.25 steps  | derived                           |
| Low-priority mean wait at 0.6/0.3 class utilizations  | 25 s versus 2.5 s           | derived from Cobham's formula     |
| Lognormal p90-to-p50 length ratio at sigma 1          | 3.6×                        | derived                           |
| Session KV break-even idle time at 80k context        | 59 s                        | derived                           |
| Agentic input-to-output token ratio                   | 131:1                       | cited: vLLM Mooncake Store post   |
| Agentic median turns / context at turn 30             | 33 / about 80,000 tokens    | cited: vLLM Mooncake Store post   |
| Agentic inter-turn delay, median and P99              | 5.2 s / 81.4 s              | cited: vLLM Mooncake Store post   |
| SLO-meeting requests, MoRIIO write vs collocated      | 73/100 vs 30/100            | cited: vLLM MoRIIO post           |
| Goodput uplift from PD disaggregation, that setup     | 2.5×                        | cited: vLLM MoRIIO post           |
| Router throughput versus llm-d, 8P8D Llama 3.1 8B     | +25%, TTFT ~1,200 ms faster | cited: vLLM Router post           |
| Relative policy ordering on p99 TTFT and goodput      | see section 9 predictions   | reproduce: sim.py                 |

### How to measure this honestly on real hardware

The simulator tells you about policies. Only a real load test tells you about your service, and scheduling experiments have specific ways of lying to you.

**Use an open-loop load generator.** A closed-loop generator with $N$ virtual users sends a new request only when the last one returns, so when the server slows down, the offered load falls with it. That is a negative feedback loop that makes latency collapse invisible — the exact failure this post is about. Drive Poisson arrivals at a fixed rate and let the queue grow if it wants to.

**Measure at steady state, not from cold.** The first thirty seconds of any run include cold prefix caches, an empty block pool, CUDA graph capture and torch.compile warmup. Discard them explicitly rather than averaging them in.

**Report per-request percentiles, then percentiles across requests.** Compute p95 ITL *within* each request, then take the distribution of those. A global ITL histogram will hide one starved request among ten thousand happy samples, and the starved one is the support ticket.

**Instrument the three levers directly.** Your scheduler dashboard needs exactly three counters, and almost nobody has them: admissions per step, the prefill/decode split of the token budget per step, and preemptions per step. With those three plus TTFT and ITL you can diagnose almost any scheduling pathology from the graphs alone. Without them you are guessing from tokens-per-second, which is precisely the metric that lied at the start of this post.

**Sweep the rate; the interesting number is where goodput turns over.** A single load point tells you nothing. Run a rate sweep and plot throughput and goodput on the same axes. The point where they diverge is your real capacity, and it will be lower than the number in your capacity plan. Setting up that measurement properly is its own discipline — see [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

---

## 15. Case studies and public numbers

**vLLM's scheduler, as documented.** The [anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) describes exactly the structure this post builds toward: a waiting queue, a running queue, a KV-cache manager with a free-block pool, and a policy that is FCFS or priority. It also documents the block size default of 16 tokens and the per-block KV formula that section 2 uses. Notably, the production engine ships the *simple* policies — the sophistication is in the memory manager, not the queue discipline. That is a deliberate choice and a defensible one.

**The V1 scheduler's data structure.** Per [the V1 release post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27), representing every scheduling decision as `{request_id: num_tokens}` unifies chunked prefill, prefix caching and speculative decoding under one policy, and the post reports up to 1.7× throughput versus V0 on Llama 3.1 8B and 3.3 70B with ShareGPT traffic. The lesson for your own engine is the representation: once a decision is "how many tokens for this request", prefill and decode stop being separate code paths.

**Goodput as the reported metric.** The [MoRIIO connector post](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07) is the cleanest public example of reporting SLO-meeting request counts rather than tokens per second: 73/100 for write mode, 70/100 for read mode, 30/100 for the collocated baseline, on Qwen3-235B-A22B-FP8 across 8×MI300X split 4+4, ISL 2,000 / OSL 1,000 at 8 req/s, against a TTFT-under-1-second and ITL-under-50-millisecond SLO. Their own listed caveats matter: single node only, and prefix caching must be disabled with `--no-enable-prefix-caching`, which for the agentic workload of section 11 would be disqualifying.

**Scheduling for cache affinity at the cluster layer.** The [vLLM Router release](https://vllm.ai/blog/2025-12-13-vllm-router-release) (2025-12-13) offers consistent hashing on a session key alongside Power-of-Two and Round Robin, with reported gains of 25% throughput over llm-d and about 1,200 ms lower TTFT on Llama 3.1 8B with 8 prefill and 8 decode pods.

**The academic line.** Schrage's 1968 optimality result for shortest-remaining-processing-time, Liu and Layland's 1973 EDF result, and Cobham's 1954 priority-queue waiting times are all still exactly correct — they simply assume an input, job size, that LLM inference cannot supply. The LLM-specific literature is largely about working around that: $S^3$ ([arXiv:2306.06000](https://arxiv.org/abs/2306.06000)) predicting length buckets, FastServe ([arXiv:2305.05920](https://arxiv.org/abs/2305.05920)) using skip-join MLFQ to learn length by running, learning-to-rank scheduling ([arXiv:2408.15792](https://arxiv.org/abs/2408.15792)) predicting order instead of magnitude, and VTC ([arXiv:2401.00588](https://arxiv.org/abs/2401.00588)) sidestepping the question entirely with token-weighted fairness.

---

## 16. When to reach for this (and when not to)

**Write your own policy when** you have multi-tenancy with real fairness obligations, tiered SLOs that differ by more than 2×, or a workload with session structure that no off-the-shelf policy models. These are business constraints that a general-purpose engine cannot know about, and they are exactly the case where a hundred lines of policy code beats a hundred flags.

**Do not write your own policy when** you are single-tenant with homogeneous traffic. Run vLLM with FCFS, set `--max-num-seqs` from the latency-ceiling calculation in section 2, and spend your effort on prefix caching and quantization, which will move your numbers far more. A clever scheduler on a workload with no queue is zero improvement over a dumb one.

**Do not write your own engine to get a custom policy.** If you need policy control on top of a production engine, the right moves are, in order: tune the flags; use the engine's priority support; put the policy in a router in front of several engine replicas, where you can implement fair share and session affinity without touching engine internals at all. Reach for your own `step()` loop when you are learning how this works — which is the point of `nanoserve` — or when you have a genuinely unusual constraint, not because the queue discipline annoys you.

**Reach for goodput as your primary metric immediately, regardless.** This costs nothing, requires no code change to the engine, and it is the single highest-leverage change in this post. Write down a TTFT target and an ITL target, count requests that meet both, and put that number on the dashboard next to tokens per second. Everything else here is optional; that is not.

---

## 17. Key takeaways

1. **A scheduler has exactly three levers**: which waiting requests to admit, how to split the token budget between prefill and decode, and whom to preempt. Every serving flag you have ever tuned pulls one of them.
2. **Throughput and goodput diverge, and the divergence is a cliff, not a slope.** Filling the batch to the memory ceiling can buy 32% more tokens per second while satisfying zero requests, because ITL equals step time and step time grows with the batch.
3. **There are two ceilings — capacity and latency — and which one binds depends on context length.** At 2k context on an A100 the pool binds at 224 requests; at 80k it binds at five, with the latency ceiling right behind.
4. **FCFS never starves anyone and blocks its own queue.** The `break` in the admission loop is the invariant; changing it to `continue` buys throughput and sells the guarantee. Know which one you typed.
5. **Priority without aging is starvation with extra steps.** Cobham's formula puts the low class at ten times the high class's wait at 90% utilization, and at infinity as total utilization approaches one. Aging turns that into a number you choose.
6. **Shortest-job-first is provably optimal and unavailable**, because output length is unknown until EOS. Predicting it leaves a 3.6× p90 error for a lognormal with unit sigma, and learning it by demotion costs a full KV preemption per demotion.
7. **A preemption in an LLM engine costs the whole batch, not the victim.** One 2,048-token recompute adds 264 ms to a step whose ITL budget is 20 ms. Keep preemptions under one per 130 steps or you are spending a tenth of the machine on repeated work.
8. **Fair share gets 60% of the way to the optimal policy with no oracle**, provided idle tenants cannot bank credit. That one rule is the difference between fairness and a self-inflicted outage.
9. **Workload shape beats policy cleverness.** For an agentic trace at 131:1 input-to-output with 33 turns, the winning decision is how long to hold an idle session's cache — about 59 seconds at 80k context by the break-even derivation — not how to order the queue.
10. **The three levers repeat at every layer**, from the router down to the block allocator, distinguished only by time constant. Fix the problem at the layer whose time constant matches it.

---

## Further reading

- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the production scheduler's structure, block sizes and metric definitions.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — why every scheduling decision became `{request_id: num_tokens}`.
- [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) — the agentic trace statistics used throughout section 11.
- [MoRIIO KV connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) — an evaluation reported in SLO-meeting requests rather than tokens per second.
- [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) — the Virtual Token Counter scheme and why counting requests fails.
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) — skip-join MLFQ as a way to learn job size by running it.
- [Writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) — the `step()` loop this post makes policy-driven.
- [Eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — what lever 3 actually costs, and the thrash spiral in detail.
- [Admission control, backpressure and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) — what to do at the front door when the schedulability test says no.
- [Request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) — the same territory from the operator's side rather than the engine author's.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — where this piece sits in the finished engine.
