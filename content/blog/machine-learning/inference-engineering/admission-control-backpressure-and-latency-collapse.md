---
title: "Admission control, backpressure and latency collapse: the curve that explains every outage"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive the utilization knee from first principles, watch an LLM server thrash instead of queue, and build the three bounds — queue caps, memory-aware admission, and real backpressure — that keep nanoserve honest under 3x overload."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "admission-control",
    "backpressure",
    "queueing-theory",
    "batching",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
---

The dashboard is the same shape every time. For six weeks p99 latency sits flat at 1.4 seconds. Traffic grows maybe 4% a week — nothing dramatic, nobody files a ticket. Then on a Tuesday afternoon p99 goes to 40 seconds in under two minutes, the queue depth graph turns into a straight diagonal line to the top of the chart, and GPU utilization reads a comfortable 94%. Someone says the thing that always gets said: "but the GPU isn't even maxed out."

That sentence is the whole problem. GPU utilization at 94% is not headroom. It is the reason. What broke was not a bug that appeared on Tuesday; it was a curve you have been sliding along since the day you deployed, and the curve is vertical near the right edge. Figure 1 is the curve — the mean queue wait as a function of utilization — and the only thing that changed on Tuesday was that your operating point crossed the knee.

![A left-to-right sequence showing how mean queue wait grows from one service time at fifty percent utilization to ninety-nine service times at ninety-nine percent](/imgs/blogs/admission-control-backpressure-and-latency-collapse-1.webp)

This post is about the fact that an LLM server does not merely queue when you overload it. It *thrashes*. A web server that is over capacity has a queue that grows; the work per request stays constant, and when the burst ends the queue drains at a predictable rate. An LLM server that is over capacity starts preempting requests to reclaim KV blocks, and every preemption throws away computed tokens that must be recomputed, which adds load, which forces more preemption. Goodput does not plateau. It *falls* as offered load rises. That is a different failure — it has a name in the networking literature, congestion collapse, and it does not fix itself.

By the end you will have a new file in `nanoserve` — `nanoserve/admit.py` — plus a load-gate in `nanoserve/gate.py`, an open-loop load generator in `bench/loadgen.py`, and a pure-Python simulator in `nanoserve/sim_admit.py` that reproduces the collapse on a laptop with no GPU at all. You will have three bounds, each derived rather than guessed: a queue cap that comes from your SLO divided by your service rate, an admission predicate that comes from the KV memory math of [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), and a backpressure contract that stops your own clients from doubling the arrival rate exactly when you can least afford it. And you will have a rule for which optimizations must be switched *off* when the server is busy — because some of the things that make you fast at low load make you slower at high load, and speculative decoding is the canonical example.

If you have not read [the scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) yet, read it first: the scheduler decides *who runs next among the requests you already accepted*. This post is the layer above — it decides *which requests you accept at all*. A perfect scheduler cannot save you from a bad admission decision, because by the time the scheduler sees a request, the request is already holding memory.

## 1. The knee, derived rather than asserted

Let us define the two quantities everything rests on, in the plainest way possible.

**Arrival rate** $\lambda$ is how many requests per second show up. **Service rate** $\mu$ is how many requests per second the server can finish when it is working flat out. **Utilization** is their ratio:

$$\rho = \frac{\lambda}{\mu}$$

If $\rho = 0.5$ you are using half your capacity. If $\rho \ge 1$ the queue grows without bound forever, which is obvious. The non-obvious part — the part that ruins Tuesdays — is what happens *below* 1.

Take the simplest queueing model that has the right shape: a single server, Poisson arrivals at rate $\lambda$, exponentially distributed service times with mean $1/\mu$, one infinite FIFO queue. This is M/M/1. Its state is just the number of requests in the system, $n$, and transitions are births at rate $\lambda$ and deaths at rate $\mu$. In steady state, the flow across the boundary between state $n$ and state $n+1$ must balance:

$$\lambda P_n = \mu P_{n+1} \quad \Rightarrow \quad P_{n+1} = \rho P_n \quad \Rightarrow \quad P_n = (1-\rho)\rho^n$$

The normalization $(1-\rho)$ comes from $\sum_n \rho^n = 1/(1-\rho)$, which converges only when $\rho \lt 1$ — the stability condition falls out of the algebra rather than being imposed. Now take the expectation:

$$L = \mathbb{E}[n] = \sum_{n=0}^{\infty} n(1-\rho)\rho^n = \frac{\rho}{1-\rho}$$

Little's law ($L = \lambda W$) converts a population into a time. With mean service time $S = 1/\mu$:

$$W = \frac{L}{\lambda} = \frac{1}{\mu - \lambda} = \frac{S}{1-\rho}, \qquad W_q = W - S = S\cdot\frac{\rho}{1-\rho}$$

That last expression is the entire post in one line. **Mean time spent waiting, measured in units of service time, is $\rho/(1-\rho)$.**

| $\rho$ | $W_q / S$ (queue wait in service times) | Throughput relative to $\rho=0.5$ |
| ------ | -------------------------------------- | --------------------------------- |
| 0.50   | 1.0                                    | 1.00×                             |
| 0.70   | 2.3                                    | 1.40×                             |
| 0.80   | 4.0                                    | 1.60×                             |
| 0.90   | 9.0                                    | 1.80×                             |
| 0.95   | 19.0                                   | 1.90×                             |
| 0.99   | 99.0                                   | 1.98×                             |

Source for every row: `derived` from $W_q/S = \rho/(1-\rho)$.

Read the two columns together, because that comparison is the whole economic argument. Moving from $\rho = 0.5$ to $\rho = 0.99$ buys you 98% more throughput and costs you 9,800% more queue wait. Nobody would sign that trade if it were written on a purchase order. It gets signed constantly because it is never written down — it happens gradually, 4% a week, and each individual week looks free.

<figure class="blog-anim">
<svg viewBox="0 0 660 320" role="img" aria-label="An operating point creeps rightward along a utilization curve while the queue-wait bar below stays flat and then explodes past the knee" style="width:100%;height:auto;max-width:820px">
<style>
.k-axis{stroke:var(--border,#d1d5db);stroke-width:2;fill:none}
.k-curve{stroke:var(--text-secondary,#6b7280);stroke-width:2.5;fill:none;stroke-linejoin:round}
.k-zone{fill:#dc2626;opacity:.10}
.k-knee{stroke:#dc2626;stroke-width:2;stroke-dasharray:5 5}
.k-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.k-hd{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.k-warn{font:700 13px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
.k-dot{fill:var(--accent,#6366f1)}
.k-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.k-bar{fill:var(--accent,#6366f1)}
@keyframes k-creep{0%,4%{transform:translate(0px,0px)}14%{transform:translate(108px,-44px)}24%{transform:translate(189px,-77px)}34%{transform:translate(270px,-110px)}44%{transform:translate(324px,-132px)}54%{transform:translate(378px,-154px)}64%{transform:translate(432px,-176px)}74%{transform:translate(486px,-198px)}84%{transform:translate(513px,-209px)}94%,100%{transform:translate(535px,-218px)}}
@keyframes k-grow{0%,4%{transform:scaleX(.010)}14%{transform:scaleX(.013)}24%{transform:scaleX(.015)}34%{transform:scaleX(.020)}44%{transform:scaleX(.025)}54%{transform:scaleX(.033)}64%{transform:scaleX(.050)}74%{transform:scaleX(.100)}84%{transform:scaleX(.200)}94%,100%{transform:scaleX(1)}}
.k-move{animation:k-creep 9s ease-in-out infinite alternate}
.k-fill{animation:k-grow 9s ease-in-out infinite alternate;transform-box:fill-box;transform-origin:left center}
@media (prefers-reduced-motion:reduce){.k-move{animation:none;transform:translate(535px,-218px)}.k-fill{animation:none;transform:scaleX(1)}}
</style>
<rect class="k-zone" x="546" y="30" width="60" height="230"/>
<path class="k-axis" d="M60 30 L60 260 L610 260"/>
<path class="k-curve" d="M60 260 L168 216 L249 183 L330 150 L384 128 L438 106 L492 84 L546 62 L573 51 L595 42"/>
<path class="k-knee" d="M546 40 L546 260"/>
<text class="k-hd" x="16" y="24">latency</text>
<text class="k-lbl" x="330" y="278">utilization rho = arrival rate / service rate</text>
<text class="k-lbl" x="60" y="278">0</text>
<text class="k-lbl" x="330" y="248">0.5</text>
<text class="k-lbl" x="546" y="278">0.9</text>
<text class="k-warn" x="576" y="24">knee</text>
<circle class="k-dot k-move" cx="60" cy="260" r="8"/>
<rect class="k-track" x="60" y="292" width="540" height="16" rx="4"/>
<rect class="k-bar k-fill" x="60" y="292" width="540" height="16" rx="4"/>
<text class="k-hd" x="16" y="305">wait</text>
</svg>
<figcaption>The operating point creeps right as offered load rises; the queue-wait bar below barely moves until the point crosses the knee, then it fills the track.</figcaption>
</figure>

### 1.1 Your server is not M/M/1, and it does not matter

Every assumption in M/M/1 is false for an LLM server, and you should know exactly which ones and why the conclusion survives anyway.

**Service times are not exponential.** An LLM request's service time is roughly proportional to its output length, and output-length distributions in real chat traffic are long-tailed but not exponential — they have a spike at the `max_tokens` cap and another at short refusals.

**There is not one server.** Continuous batching means the "server" processes dozens of requests concurrently, and its per-request rate *depends on how many it is processing*. This is the LLM-specific twist, and section 2 shows it makes things worse, not better.

**Arrivals are not Poisson.** Real traffic is burstier than Poisson: retries cluster, cron jobs fire on the minute, an agent framework fans out eight tool calls at once. Agentic traffic in particular is spiky — the Mooncake Store write-up on [vLLM's blog](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) reports, for 610 Codex and SWE-bench agent traces, a median of 33 turns per session with a median inter-turn delay of 5.2 s but a p99 of 81.4 s. That is not a smooth arrival process; that is long silences punctuated by bursts.

So why keep the formula? Because the $\rho/(1-\rho)$ shape is not an artifact of exponential assumptions. Kingman's approximation generalizes it to arbitrary distributions:

$$W_q \approx \left(\frac{\rho}{1-\rho}\right)\left(\frac{c_a^2 + c_s^2}{2}\right)S$$

where $c_a$ and $c_s$ are the coefficients of variation of interarrival and service times. Changing the distributions changes the *multiplier* in the middle bracket. It does not touch $\rho/(1-\rho)$. Burstier traffic and more variable output lengths make every row of that table worse by a constant factor; they never remove the pole at $\rho = 1$.

That is the honest claim to carry forward: **the constant is wrong, the shape is right, and the shape is what kills you.**

#### Worked example: what $\mu$ actually is for one A100

Numbers below are `derived` from the KV math in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) plus NVIDIA's published A100 80GB SXM specs (2,039 GB/s HBM2e bandwidth, 312 TFLOP/s dense bf16 — see the [A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/)). Run the harness from [the naive decode loop and your first baseline](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) to get your own; these are the arithmetic, not a measurement.

Serving Llama-3.1-8B in bf16, chat traffic with 2,048-token prompts and 300 output tokens on average:

- KV bytes per token: $2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dims} \times 2 \text{ bytes} = 131{,}072$ B = 128 KiB.
- A 16-token block is therefore 2 MiB. (Reassuringly, vLLM's KV-offloading post reports a physical block size of 2 MB for Llama-3.1-8B after their KV-layout change — [vLLM, 2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector).)
- Weights take 16 GB; leave ~6 GB for activations, workspace and CUDA context; the KV pool gets ~58 GiB, which is $58 \times 1024 / 2 = 29{,}696$ blocks, or 475,136 tokens.
- Decode step time: the weight read alone is $16 \text{ GB} / 2039 \text{ GB/s} = 7.85$ ms. Call that $t_0$.
- Each additional request in the batch adds its own KV read: at a mean live context of ~2,200 tokens that is $2200 \times 128 \text{ KiB} = 275$ MiB, or ${0.14}$ ms at 2,039 GB/s, plus ~${0.05}$ ms of matmul ($2 \times 8\times10^9$ FLOPs at 312 TFLOP/s). Call the marginal cost $\beta \approx 0.19$ ms per request.

So the step-time model is $t(B) = t_0 + \beta B = 7.85 + 0.19B$ ms.

Memory caps the batch. A request that reaches 2,348 tokens holds 147 blocks; averaged over its life it holds about 138, so the pool supports roughly $29{,}696/138 \approx 215$ concurrent requests — and if you admit on *worst case* (prompt plus `max_tokens = 512`, i.e. 160 blocks) the cap is $\lfloor 29{,}696/160 \rfloor = 185$. Take 185 as the operating batch. Then:

- Step time: $t(185) = 7.85 + 0.19 \times 185 = 43.0$ ms.
- Residency: 300 output tokens × 43.0 ms = 12.9 s per request.
- By Little's law, $\mu = B / R = 185 / 12.9 = 14.3$ req/s.

**Call it $\mu \approx 14$ req/s.** That single number is the input to every bound in this post. Notice how much structure went into it: the model architecture, the dtype, the GPU's bandwidth, the workload's prompt and output lengths, and the admission policy itself. Change any one and $\mu$ moves. This is why "set `max_num_seqs` to 256 because that is what the tutorial said" is not capacity planning.

## 2. Why an LLM server thrashes instead of queueing

Here is the part that makes LLM serving different, and it is worth being precise because the difference is not one of degree.

In a stateless web service, an admitted request holds a thread and a few kilobytes for a duration that is essentially independent of how many other requests are in flight. Overload produces a longer queue and constant per-request work. The system is *stable in the bad case*: it is slow, but it is doing useful work at its maximum rate, and when the burst ends it drains.

In an LLM server, an admitted request holds KV blocks for the *entire* generation — hundreds of decode steps, seconds of wall-clock. Two consequences follow, and they compound.

### 2.1 Service time is load-dependent, so capacity is self-referential

Residency $R$ is output length $O$ times step time, and step time depends on batch size $B$. Little's law says $B = \lambda R$. Substitute:

$$B = \lambda \cdot O \cdot t(B) = \lambda O (t_0 + \beta B)$$

Solve for $B$:

$$B(1 - \lambda O \beta) = \lambda O t_0 \quad \Rightarrow \quad B = \frac{\lambda O t_0}{1 - \lambda O \beta}$$

Look at what happened. The equilibrium batch size has a pole at $\lambda O \beta = 1$, that is at

$$\mu_{\max} = \frac{1}{O\beta}$$

and if you define $\rho = \lambda / \mu_{\max} = \lambda O \beta$, the equilibrium batch is $B = \rho t_0 / (\beta(1-\rho))$ — **the same $\rho/(1-\rho)$ pole, derived from nothing but batching physics.** No Poisson assumption, no exponential service times. The knee is structural.

With our numbers ($O = 300$, $\beta = 0.19$ ms), $\mu_{\max} = 1/(300 \times 0.19 \times 10^{-3}) = 17.5$ req/s. At $\lambda = 12$ req/s, $\rho = 0.686$ and $B = 12 \times 300 \times 7.85\text{ms} / (1 - 0.686) = 28.3 \text{s} / 0.314 = 90$. At $\lambda = 16$ req/s, $\rho = 0.914$ and $B = 37.7/0.086 = 438$. The batch the system *wants* more than doubles for a 33% increase in arrival rate.

### 2.2 Memory binds before the pole does, and that is the whole problem

The pool holds ~215 concurrent requests at typical context. The equilibrium batch at $\lambda = 16$ wants 438. The system cannot have 438. What happens instead is not "the extra requests wait politely" — it is that the scheduler admits requests it cannot sustain, the pool runs dry mid-generation, and preemption starts. That mechanism is the subject of [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping); what matters here is its feedback structure.

![A branching diagram showing over-admission raising KV pressure, preemption splitting into recompute and longer residency, and both merging back into higher effective load](/imgs/blogs/admission-control-backpressure-and-latency-collapse-2.webp)

Trace the loop in figure 2. A preemption with the recompute policy discards $n$ computed tokens and re-runs them later. Let $p$ be the mean number of preemptions a request suffers and $\bar{n}$ the mean tokens discarded per preemption. The *effective* work per request in token-equivalents is

$$W_{\text{eff}} = O + p \cdot \bar{n}$$

and because $\mu_{\max} = 1/(W_{\text{eff}}\beta)$ once you account for recompute, effective utilization becomes

$$\rho_{\text{eff}} = \lambda \beta (O + p\bar{n})$$

Now the sting: $p$ is itself an increasing function of $\rho_{\text{eff}}$ — the fuller the pool, the more often admission overruns it. Write $p = f(\rho_{\text{eff}})$ and you have a fixed-point equation

$$\rho_{\text{eff}} = \lambda\beta\big(O + \bar{n}\,f(\rho_{\text{eff}})\big)$$

A stable fixed point exists only if the loop gain is under one:

$$\lambda\beta\bar{n}\,\frac{df}{d\rho_{\text{eff}}} \lt 1$$

Above that threshold there is no stable solution: every increment of load produces more preemption, which produces more load. That is the formal difference between the two failure modes:

- **A queue that grows.** $\rho > 1$ with constant work per request. Backlog grows linearly at $(\lambda - \mu)$ per second, goodput stays pinned at $\mu$, and the burst's end is the beginning of recovery.
- **A system that thrashes.** Loop gain above one. Goodput *decreases* as offered load increases, so the system does not recover when the burst ends — it recovers only when load drops well *below* the level that started it. This is hysteresis, and it is why "traffic went back to normal but the service stayed down."

The networking community named this in the 1980s. John Nagle's [RFC 896](https://www.rfc-editor.org/rfc/rfc896) (1984) describes congestion collapse in a packet network where retransmissions of already-transmitted data consume the capacity needed to make progress; Van Jacobson and Michael Karels' *Congestion Avoidance and Control* (1988) is the classic follow-up. Swap "retransmitted packets" for "recomputed prefill tokens" and the mechanism is identical. Your engine is a congested network whose packets happen to be tokens.

### 2.3 The observable signature

You can tell the two apart on your dashboards, and knowing which you are in changes what you do:

| Signal                    | Growing queue           | Thrash                                        | Source                                    |
| ------------------------- | ----------------------- | --------------------------------------------- | ----------------------------------------- |
| Output tok/s              | flat at capacity        | falling while load rises                      | derived from the loop-gain condition       |
| Preemptions per step      | ~0                      | rising, often several per step                | derived                                   |
| Mean batch size           | pinned at the cap       | oscillating (fills, collapses, refills)       | derived                                   |
| Recovery after burst ends | starts immediately      | needs load below the onset level (hysteresis) | derived                                   |
| GPU utilization           | high                    | high — it is busy recomputing                 | derived                                   |

That last row deserves emphasis. During thrash the GPU is nearly 100% busy doing work that will be thrown away. Utilization is not a measure of usefulness. The metric that distinguishes them is **goodput**: completed requests per second that met their SLO. Section 7 makes that measurable.

## 3. Lever one: bound the queue, and derive the bound

![A layered stack from gateway concurrency down through the queue cap and KV predicate to the scheduler, block pool and GPU step](/imgs/blogs/admission-control-backpressure-and-latency-collapse-3.webp)

Figure 3 is the shape of the fix: three bounds at three layers, each computed from the layer beneath it. Start at the queue.

An unbounded queue is a device for converting a capacity problem into a latency problem and then lying to you about both. It accepts work it cannot do, reports success at accept time, and delivers a timeout ten seconds later. The fix is a bound — but the interesting question is *what number*, and there is a correct answer.

### 3.1 The queue cap is an SLO divided by a service rate

A request sitting at position $k$ in a FIFO queue served at rate $\mu$ will wait approximately $k/\mu$ seconds before it starts. If your TTFT SLO allows a queue-wait budget of $T_q$ seconds, then any request that arrives when the queue is deeper than

$$K = \mu \cdot T_q$$

is *guaranteed* to miss its SLO. Not likely to. Guaranteed to, by arithmetic, before you have executed one instruction on its behalf. Admitting it can only make things worse for everyone behind it.

With $\mu \approx 14$ req/s and a 2-second queue-wait budget, $K = 28$. Twenty-eight. Not 1,000, not "unbounded because we do not want to drop traffic". If you have ever set a queue depth by feel, this is the formula you were reaching for.

Three refinements make it usable:

**$\mu$ is not constant.** It depends on the mix (a batch of 8k-token RAG prompts has a different $\mu$ than short chats). Estimate it online from a moving average of completions per second rather than hardcoding it.

**FIFO is the wrong discipline under overload.** In a queue that is over budget, FIFO serves the requests that have waited longest — which are exactly the ones whose clients have most likely given up. LIFO serves the freshest, whose clients are still listening. Ben Maurer's [*Fail at Scale*](https://queue.acm.org/detail.cfm?id=2839461) (ACM Queue, 2015) describes exactly this "adaptive LIFO" at Facebook: normal operation is FIFO, and the queue switches to LIFO when it is deep. It sounds unfair, and it is, but it converts a period where *everyone* fails into a period where *some* succeed.

**Queue length is a proxy for queue time, and a bad one when $\mu$ moves.** Which brings us to the better bound.

### 3.2 CoDel: cap the time, not the length

Kathleen Nichols and Van Jacobson's *Controlled Delay* algorithm ([ACM Queue, 2012](https://queue.acm.org/detail.cfm?id=2209336)) solves the same problem for network buffers: distinguish a *good* queue (a burst passing through) from a *bad* queue (a standing backlog) by tracking the **minimum** sojourn time over a sliding window. A burst has moments where the queue empties, so the minimum stays low. A standing queue never empties, so the minimum stays high. When the minimum exceeds a target for longer than an interval, start dropping. *Fail at Scale* reports Facebook using an adapted CoDel in front of their services for precisely this reason.

Ported to `nanoserve`, this is thirty lines and it is strictly better than a fixed length cap because it adapts to $\mu$ automatically:

```python
# nanoserve/admit.py
import time
from dataclasses import dataclass, field


@dataclass
class CoDelQueue:
    """Time-based overload detector for the waiting queue.

    target:   acceptable standing queue delay (seconds)
    interval: window over which we require at least one low-delay moment
    """
    target: float = 0.020
    interval: float = 0.100
    _first_above: float = 0.0     # when delay first exceeded target
    _dropping: bool = False
    _drop_next: float = 0.0
    _drop_count: int = 0

    def should_drop(self, sojourn: float, now: float) -> bool:
        """Call once per dequeue with the head request's time in queue."""
        if sojourn < self.target:
            # the queue emptied out below target: this is a good queue
            self._first_above = 0.0
            self._dropping = False
            self._drop_count = 0
            return False

        if self._first_above == 0.0:
            self._first_above = now + self.interval
            return False

        if not self._dropping and now >= self._first_above:
            self._dropping = True
            self._drop_count = 1
            self._drop_next = now + self.interval
            return True

        if self._dropping and now >= self._drop_next:
            self._drop_count += 1
            # drop rate rises as sqrt(count): gentle at first, firm if it persists
            self._drop_next = now + self.interval / (self._drop_count ** 0.5)
            return True

        return False
```

The control law is the clever part. The first drop happens only after the queue has been above target for a full interval — a burst that clears on its own is never punished. If the standing queue persists, drops accelerate as $\sqrt{\text{count}}$, which in the original derivation is the shape that drives a TCP-like control loop back to target without oscillating. For us, "drop" means "return 429 to the request at the head" rather than "discard a packet", and the accelerating rate is what makes the shedding *proportional to the excess* instead of all-or-nothing.

### 3.3 Why fast rejection beats slow acceptance

This is worth deriving because it is counterintuitive to product owners, who hear "reject" as "lose a customer".

Consider a request with a client-side timeout $T_c$. If you admit it and it does not finish by $T_c$, the client hangs up. The work you did is entirely wasted — but you did not know that while doing it, so you did it at full cost. Define the wasted-work fraction $w$ as the share of GPU time spent on requests that were abandoned. If a fraction $q$ of admitted requests time out, and a timed-out request consumed on average a fraction $\gamma$ of a full request's work before being abandoned, then

$$w = \frac{q\gamma}{q\gamma + (1-q)}$$

At $q = 0.5$ and $\gamma = 0.8$ — half the traffic times out, having burned 80% of a full generation each — the waste fraction is $0.4/(0.4+0.5) = 44\%$. Nearly half your GPU is producing tokens nobody will read. And that waste raises $\rho$ for the survivors, which raises $q$, which raises the waste. It is the same feedback loop as section 2, wearing a different hat.

Rejecting the same request at admission costs one hash lookup. **The 429 is not the failure; the timeout is the failure. The 429 is what you do instead of failing.** And it is a *better* failure: it is fast, it is explicit, it carries a `Retry-After` the client can act on, and it leaves the capacity intact for the requests you did accept.

There is a second-order benefit that matters more than it sounds. A server that sheds cleanly has a *stable* p99 for the requests it accepts. A server that never sheds has an unbounded p99 for everyone. Given a choice between "97% of users get 1.4 s and 3% get a 429 with a retry hint" and "100% of users get somewhere between 1.4 s and 90 s", every product owner who has seen both picks the first — but they have to see the second described in those terms first.

## 4. Lever two: memory-aware admission — the single most valuable idea here

If you implement one thing from this post, implement this. A queue cap protects latency. A memory-aware admission predicate protects the engine from itself.

The rule: **never admit a request whose worst-case KV footprint cannot fit alongside the requests already running.** Not "probably fit". Not "fit on average". Fit in the worst case, or be explicit and quantified about the risk you are taking instead.

![A decision tree branching on queue wait, worst-case KV fit and running-set headroom into admit, enqueue or reject outcomes](/imgs/blogs/admission-control-backpressure-and-latency-collapse-4.webp)

Figure 4 shows the ordering, and the ordering matters: the two cheap checks come first, and all three complete before any prefill runs. A request that will be rejected should cost you a few microseconds of Python, never a GPU kernel.

### 4.1 The predicate

A request's worst-case token count is `len(prompt_ids) + max_tokens`. Its worst-case block count is that divided by the block size, rounded up. The running set's future demand is the sum of what each running request may still need before it finishes. Formally, for candidate $c$ and running set $\mathcal{R}$:

$$\text{admit}(c) \iff \Big\lceil \frac{p_c + m_c}{S_b} \Big\rceil \;+\; \sum_{r \in \mathcal{R}} \Big( \Big\lceil \frac{p_r + m_r}{S_b} \Big\rceil - |\text{blocks}(r)| \Big) \;\le\; F$$

where $S_b$ is the block size, $m$ is `max_tokens`, and $F$ is the free block count. The middle term is the key one and it is the term everybody forgets: **the requests you already admitted have not finished allocating.** A request 40 tokens into a 512-token generation still owes the pool 30 more blocks. Counting only current occupancy is how you end up at 97% pool usage with 180 requests each about to ask for one more block.

```python
# nanoserve/admit.py (continued)
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Capacity:
    block_size: int = 16
    total_blocks: int = 29_696        # 58 GiB / 2 MiB for Llama-3.1-8B bf16
    max_running: int = 185            # hard cap on the running set
    reserve_blocks: int = 256         # headroom for prefix-cache CoW and fragmentation


def worst_case_blocks(prompt_len: int, max_tokens: int, block_size: int) -> int:
    """Blocks this request could ever hold, assuming it generates to the cap."""
    return math.ceil((prompt_len + max_tokens) / block_size)


def outstanding_blocks(req, block_size: int) -> int:
    """Blocks this RUNNING request has not allocated yet but may still need."""
    worst = worst_case_blocks(req.prompt_len, req.max_tokens, block_size)
    return max(0, worst - len(req.block_ids))


class MemoryAdmission:
    def __init__(self, pool, cap: Capacity):
        self.pool = pool
        self.cap = cap

    def free_after_commitments(self, running) -> int:
        committed = sum(outstanding_blocks(r, self.cap.block_size) for r in running)
        return self.pool.num_available() - committed - self.cap.reserve_blocks

    def can_admit(self, req, running) -> tuple[bool, str]:
        if len(running) >= self.cap.max_running:
            return False, "running_set_full"
        need = worst_case_blocks(req.prompt_len, req.max_tokens, self.cap.block_size)
        if need > self.cap.total_blocks - self.cap.reserve_blocks:
            # this request can never fit, on an empty server, ever
            return False, "request_exceeds_capacity"
        if need > self.free_after_commitments(running):
            return False, "insufficient_kv"
        return True, "ok"

    def can_admit_budgeted(self, req, running, budget_tokens: int) -> tuple[bool, str]:
        """Same predicate, but against a token budget instead of raw max_tokens.

        budget_tokens comes from OutputLengthModel below; passing max_tokens
        reproduces the strict worst-case behaviour of can_admit exactly.
        """
        shadow = type(req)(**{**req.__dict__, "max_tokens": budget_tokens})
        return self.can_admit(shadow, running)
```

Note the third branch. `request_exceeds_capacity` is a *permanent* rejection — a 128k-context request on a pool that holds 475k tokens total is fine, but a 128k-context request with `max_tokens=32768` on a 24 GB RTX 4090 is not, ever, and the correct response is an immediate 4xx explaining the limit rather than a queue slot it will never leave. Distinguishing "no room right now" (retryable, 429) from "no room ever" (not retryable, 400 or 413) is a small change that saves your on-call from a lot of confused pages.

### 4.2 Worst case is conservative — how conservative, and what to do about it

`max_tokens` is usually a lie. Clients set it to 4096 out of habit and generate 180 tokens. Admitting on worst case means reserving 4096 tokens of KV for a request that will use 180, which is a 22× over-reservation and leaves your GPU mostly idle. So the honest framing is a spectrum, not a rule:

| Admission predicate           | Concurrency reached | Preemption rate | Failure mode                          | Source                     |
| ----------------------------- | ------------------- | --------------- | ------------------------------------- | -------------------------- |
| Worst case (prompt + max)     | lowest              | zero by construction | wasted capacity, low utilization | derived from §4.1          |
| p95 of observed output length | moderate            | ~5% of requests | occasional preemption, recoverable    | derived; needs a histogram |
| p50 of observed output length | high                | ~50% of requests | thrash risk under load                | derived; see §2.2          |
| None (admit and hope)         | unbounded           | unbounded       | congestion collapse                   | derived; see §2.2          |

The middle rows require you to *measure the output-length distribution of your own traffic*, which almost nobody does and which takes ten lines. Keep a rolling histogram of completed generation lengths per route or per model, and admit against a chosen quantile:

```python
# nanoserve/admit.py (continued)
import bisect
from collections import deque


class OutputLengthModel:
    """Rolling empirical quantiles of realized output length."""

    def __init__(self, window: int = 4096, quantile: float = 0.95):
        self.window = deque(maxlen=window)
        self.quantile = quantile
        self._sorted: list[int] = []
        self._dirty = True

    def observe(self, output_len: int) -> None:
        self.window.append(output_len)
        self._dirty = True

    def budget(self, max_tokens: int) -> int:
        """Token budget to admit against: the quantile, capped by max_tokens."""
        if len(self.window) < 256:          # not enough data: stay conservative
            return max_tokens
        if self._dirty:
            self._sorted = sorted(self.window)
            self._dirty = False
        idx = min(len(self._sorted) - 1,
                  int(self.quantile * len(self._sorted)))
        return min(max_tokens, self._sorted[idx])
```

Then `worst_case_blocks(req.prompt_len, model.budget(req.max_tokens), block_size)` replaces the raw `max_tokens`. The tail beyond the quantile is not ignored — it becomes preemption, handled by the machinery in [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — but now the preemption rate is a *number you chose* (5% at the p95) rather than an emergent property of overload. That is the entire difference between an engineering decision and an outage.

One caveat that bites in production: this model must be **per route**, not global. A summarization endpoint whose p95 output is 900 tokens and a classification endpoint whose p95 is 6 tokens share nothing. A single blended histogram gives you a p95 that is wrong for both.

#### Worked example: how much does the quantile buy?

All rows `derived` from the running example (Llama-3.1-8B bf16, A100 80GB, 2,048-token prompts, `max_tokens=512`, 16-token blocks, 29,696 blocks).

| Budget used for admission | Blocks per request | Concurrency | Step time $t(B)$ | Throughput $B/(O \cdot t)$ |
| ------------------------- | ------------------ | ----------- | ---------------- | -------------------------- |
| `max_tokens` = 512        | 160                | 185         | 43.0 ms          | 14.3 req/s                 |
| p95 output = 380          | 152                | 195         | 44.9 ms          | 14.5 req/s                 |
| p50 output = 240          | 143                | 207         | 47.2 ms          | 14.6 req/s                 |

The arithmetic for the first row: $\lceil (2048+512)/16 \rceil = 160$ blocks; $\lfloor 29696/160 \rfloor = 185$; $t = 7.85 + 0.19(185) = 43.0$ ms; $185/(300 \times 0.043) = 14.3$ req/s.

Now look at what that table says, because it is not what people expect. Dropping from worst case to the median budget raises concurrency by 12% and throughput by **2%**. The reason is $t(B) = t_0 + \beta B$: at batch 185 the marginal request already costs $\beta$ and adds nothing to weight-read amortization, so more concurrency buys almost nothing in throughput while adding real preemption risk. Past the point where $\beta B \gg t_0$, aggressive admission is a bad trade — you are paying in tail latency and thrash risk for a rounding error in throughput.

This is the argument for being *conservative* in a way that most engines are not by default, and it only becomes visible once you write the step-time model down.

## 5. Lever three: backpressure that actually reaches the caller

Shedding load is useless if the client immediately resends. Worse than useless — it converts a capacity shortfall into a retry storm.

### 5.1 Retry amplification, derived

Suppose each logical request will be attempted up to $R$ times, and each attempt succeeds with probability $s$. The expected number of attempts is

$$\mathbb{E}[\text{attempts}] = \sum_{k=0}^{R} (1-s)^k = \frac{1 - (1-s)^{R+1}}{s}$$

For $R$ large this tends to ${1/s}$. So the arrival rate your server actually sees is

$$\lambda_{\text{eff}} \approx \frac{\lambda}{s}$$

At a 30% shed rate ($s = 0.7$), $\lambda_{\text{eff}} = 1.43\lambda$. At a 50% shed rate, $\lambda_{\text{eff}} = 2\lambda$. **Your shedding made the overload worse**, because $s$ falls as $\lambda_{\text{eff}}$ rises and you are back in a positive feedback loop — the exact structure of section 2.2, now living in your client library instead of your block pool.

The mitigation the industry converged on is a **retry budget**: cap retries as a fraction of successful requests rather than per-request. Google's SRE book chapter on [handling overload](https://sre.google/sre-book/handling-overload/) describes a per-client budget (their example limits retries to around 10% of requests) plus adaptive client-side throttling, where a client that is seeing rejections starts rejecting locally without sending anything. That second part is the one people skip and the one that actually breaks the loop: if the backend is shedding, the client must shed too, or the backend's shedding decision has no effect on the arrival rate.

### 5.2 The response contract

Three things must be true of a rejection for it to function as backpressure rather than noise:

1. **It is fast.** Rejection cost must be orders of magnitude below service cost, or shedding does not free capacity. A queue-cap check and a block-count comparison qualify. Running prefill and *then* deciding does not.
2. **It is machine-readable.** `429 Too Many Requests` with `Retry-After`, plus a body naming the reason (`insufficient_kv` vs `queue_full` vs `request_exceeds_capacity`) so the caller can distinguish "back off" from "your request is impossible".
3. **The retry hint is jittered.** A fixed `Retry-After: 2` on a thousand rejected clients produces a thousand simultaneous retries two seconds later. You have built a metronome for your own outage. Full jitter — a uniform draw over $[0, \text{base}]$ with exponential backoff on the base — is the standard fix.

```python
# nanoserve/api.py
import asyncio
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nanoserve.admit import AdmissionController

app = FastAPI()
engine = None            # set at startup
admission: AdmissionController = None

# Bound concurrency at the edge too: 185 running + 28 queued = 213.
# The gateway must never hold more in flight than the engine can name.
GATEWAY_LIMIT = 213
_inflight = asyncio.Semaphore(GATEWAY_LIMIT)


def _retry_after(base: float) -> str:
    """Full jitter: uniform in [0, base]. Never a constant."""
    return f"{max(0.05, random.uniform(0.0, base)):.2f}"


def _reject(reason: str, base: float, status: int = 429) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers={"Retry-After": _retry_after(base)},
        content={"error": {"type": "server_overloaded", "reason": reason}},
    )


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    # 1. Edge concurrency: refuse instantly rather than parking a coroutine.
    if _inflight.locked() and _inflight._value <= 0:
        return _reject("gateway_saturated", base=1.0)

    async with _inflight:
        prompt_ids = engine.tokenizer.encode_chat(body["messages"])
        decision = admission.try_admit(
            prompt_len=len(prompt_ids),
            max_tokens=body.get("max_tokens", 512),
            deadline_s=body.get("timeout", 30.0),
        )
        if not decision.accepted:
            status = 413 if decision.reason == "request_exceeds_capacity" else 429
            return _reject(decision.reason, base=decision.retry_after_hint, status=status)

        return await engine.stream(decision.request)
```

Two details worth stating explicitly.

**The gateway limit is derived, not chosen.** 185 running plus 28 queued is 213. Holding more connections than that in the gateway means holding requests in a place with no admission logic, no SLO accounting and no visibility — which is to say, you have rebuilt the unbounded queue one layer up. Every hop in your stack has a queue whether you configured it or not; the ones you did not configure are unbounded by default.

**Admission must happen before the first token is streamed.** Once you have written `HTTP/1.1 200 OK` and flushed one SSE frame, you have no way to say 429. Streaming responses make admission a one-shot decision taken at the door — which is exactly why the predicate has to be based on worst case rather than "we will see how it goes".

### 5.3 The full controller

Composing the three levers:

```python
# nanoserve/admit.py (continued)
import time
from dataclasses import dataclass


@dataclass
class Decision:
    accepted: bool
    reason: str = "ok"
    retry_after_hint: float = 1.0
    request: object | None = None


class AdmissionController:
    def __init__(self, pool, cap, engine, *, queue_wait_budget_s: float = 2.0):
        self.mem = MemoryAdmission(pool, cap)
        self.engine = engine
        self.codel = CoDelQueue(target=queue_wait_budget_s / 4, interval=0.1)
        self.budget = queue_wait_budget_s
        self.olen = OutputLengthModel(quantile=0.95)
        self.mu_ewma = 1.0                     # completions/sec, EWMA
        self.stats = {"admitted": 0, "queued": 0, "shed_queue": 0,
                      "shed_kv": 0, "shed_permanent": 0}

    # ---- online service-rate estimate ------------------------------------

    def observe_completion(self, output_len: int, now: float) -> None:
        self.olen.observe(output_len)
        dt = now - getattr(self, "_last_completion", now - 1.0)
        self._last_completion = now
        inst = 1.0 / max(dt, 1e-6)
        self.mu_ewma = 0.99 * self.mu_ewma + 0.01 * inst

    def queue_cap(self) -> int:
        """K = mu * T_q, recomputed continuously. Never a hardcoded constant."""
        return max(1, int(self.mu_ewma * self.budget))

    # ---- the decision -----------------------------------------------------

    def try_admit(self, prompt_len: int, max_tokens: int, deadline_s: float) -> Decision:
        now = time.monotonic()
        req = self.engine.make_request(prompt_len, max_tokens, deadline_s, now)

        budget_tokens = self.olen.budget(max_tokens)
        ok, reason = self.mem.can_admit_budgeted(req, self.engine.running, budget_tokens)
        if not ok and reason == "request_exceeds_capacity":
            self.stats["shed_permanent"] += 1
            return Decision(False, reason, retry_after_hint=0.0)

        # Free path: room right now, run it immediately.
        if ok and not self.engine.waiting:
            self.engine.admit(req)
            self.stats["admitted"] += 1
            return Decision(True, request=req)

        # Otherwise it must queue. Is the queue already past its budget?
        depth = len(self.engine.waiting)
        cap = self.queue_cap()
        projected_wait = depth / max(self.mu_ewma, 1e-6)
        if depth >= cap or projected_wait > deadline_s:
            self.stats["shed_queue"] += 1
            return Decision(False, "queue_full", retry_after_hint=projected_wait)

        head_sojourn = now - self.engine.waiting[0].enqueued_at if depth else 0.0
        if self.codel.should_drop(head_sojourn, now):
            self.stats["shed_queue"] += 1
            return Decision(False, "standing_queue", retry_after_hint=self.budget)

        if not ok:
            self.stats["shed_kv"] += 1
            return Decision(False, "insufficient_kv", retry_after_hint=self.budget)

        req.enqueued_at = now
        self.engine.waiting.append(req)
        self.stats["queued"] += 1
        return Decision(True, request=req)
```

And the engine-side hook, which is a two-line change to the `step()` loop from [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop): before promoting anything from `waiting` to `running`, re-run the predicate, because the pool state has changed since the request was enqueued.

```python
# nanoserve/engine.py (continued)
    def _fill_from_waiting(self, admission) -> None:
        """Promote waiting requests while the predicate holds. Called each step."""
        now = time.monotonic()
        while self.waiting:
            head = self.waiting[0]

            # Deadline check: never start a request that has already lost.
            if now - head.enqueued_at > head.deadline_s:
                self.waiting.popleft()
                head.fail("deadline_exceeded_in_queue")
                self.metrics["expired_in_queue"] += 1
                continue

            ok, _ = admission.mem.can_admit_budgeted(
                head, self.running, admission.olen.budget(head.max_tokens)
            )
            if not ok:
                break                      # head-of-line: nothing behind it fits either

            self.waiting.popleft()
            self.admit(head)
```

The `deadline_exceeded_in_queue` counter is one of the most useful metrics you will ever export. It is the count of requests you accepted and then failed to start in time — pure, quantified, self-inflicted waste. If it is above zero, your queue cap is too large. That is not a heuristic; it is the definition.

## 6. Degradation under load: which optimizations must be turned off

![A two-column comparison of a light-load feature configuration against a heavy-load configuration with speculative decoding disabled and chunk sizes reduced](/imgs/blogs/admission-control-backpressure-and-latency-collapse-5.webp)

Figure 5 is the uncomfortable one, because the left column is what your benchmark ran and the right column is what production needs.

Here is the general law, and once you see it you will find it everywhere: **any technique that reduces latency by consuming spare capacity must be gated on the existence of spare capacity.** At low load a GPU running batch-1 decode is severely memory-bound — it reads 16 GB of weights to compute a handful of FLOPs, so the tensor cores are idle and extra arithmetic is genuinely free. At high load the same GPU is compute-bound, and every extra FLOP comes directly out of someone else's tokens.

### 6.1 Speculative decoding: the canonical example

Speculative decoding drafts $k$ tokens cheaply and verifies them in one forward pass, so a step processes $k+1$ positions per sequence instead of one. At batch 1 that is nearly free. At batch 185 it multiplies the compute term by $k+1$.

Derive the break-even. The decode step is roughly $\max(\text{weight bytes}/\text{BW},\ \text{FLOPs}/F)$. With $P$ parameters, batch $B$, and $k+1$ verified positions per sequence, the compute term is ${2P B (k+1)/F}$ and the memory term is $2P/\text{BW}$ (bf16 weights). They cross at

$$B^\star = \frac{F}{\text{BW} \cdot (k+1)}$$

For an A100 80GB SXM (312 TFLOP/s dense bf16, 2,039 GB/s, per NVIDIA's [datasheet](https://www.nvidia.com/en-us/data-center/a100/)):

- With $k = 0$ (no speculation): $B^\star = 312{\times}10^{12} / (2.039{\times}10^{12} \times 1) \approx 153$.
- With $k = 4$: $B^\star = 312{\times}10^{12} / (2.039{\times}10^{12} \times 5) \approx 31$.

Both `derived`. So on this machine, speculative decoding with 4 draft tokens stops being free somewhere around batch 31 — and our operating batch is 185. Past that point, drafting does not fill idle tensor cores; it competes for busy ones, and the accepted-token yield has to beat a $(k+1)\times$ compute tax that it usually cannot.

That derivation predicts a sign flip, and the sign flip has been reported publicly. The vLLM team's [speculative decoding post](https://vllm.ai/blog/2024-10-17-spec-decode) (2024-10-17) measured up to 1.5× speedup with a draft model (Llama-3-70B, 4×H100, ShareGPT, QPS=1) and up to 2.8× with prompt-lookup n-gram drafting on CNN/DailyMail summarization, where prompt overlap is high — and then reported that **at high QPS the same configurations become a 1.4× slowdown (draft model, ShareGPT) and a 1.8× slowdown (n-gram, CNN/DailyMail)**. Same code, same flags, opposite sign, and the only thing that changed was load. Cite that number with its setup, because "speculative decoding is 2.8× faster" without "at QPS=1 on summarization" is how a configuration flag becomes an outage.

A useful framing: [speculative decoding's core idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) is trading throughput for latency. If you are latency-bound you want that trade. If you are throughput-bound you emphatically do not, and no amount of draft-model tuning changes the sign.

### 6.2 The load-gate table

| Technique                     | At low load                        | At high load                          | Gate on                          | Source                                    |
| ----------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------- | ----------------------------------------- |
| Speculative decoding, $k=4$   | large win, free FLOPs below batch 31 | up to 1.8× slowdown reported          | batch size / running count       | derived + cited: vLLM 2024-10-17          |
| Large prefill chunk (8k)      | best TTFT                          | starves decoders, TPOT spikes         | queue depth                      | derived; see chunked-prefill post         |
| Small prefill chunk (512)     | slightly worse TTFT                | protects TPOT for the running set     | always on above target load      | derived                                   |
| Prefix caching                | win when prefixes repeat           | still a win; near-zero cost on a miss | leave on                         | cited: vLLM V1, 2025-01-27                |
| Large `max_num_seqs`          | irrelevant                         | invites over-admission and preemption | KV predicate, not a fixed number | derived; see §4                           |
| CUDA graph capture, many sizes | lower CPU overhead                | memory cost, capture-size mismatch    | leave on, cap the size list      | derived                                   |
| Best-of-$n$ / multi-sample     | cheap when idle                    | multiplies KV and compute by $n$      | disable above target load        | derived                                   |

Prefix caching deserves its exception. The vLLM team reports their V1 implementation has "zero overhead" prefix caching with under a 1% throughput decrease even at a 0% hit rate, and enables it by default ([vLLM V1 alpha](https://vllm.ai/blog/2025-01-27-v1-alpha-release), 2025-01-27). A technique whose worst case is a rounding error and whose best case is skipping an entire prefill does not need a gate. Techniques with a $(k+1)\times$ worst case do.

And the general point about admission policy being workload-specific has a nice public data point: vLLM's PegaFlow external KV cache service ships an optional TinyLFU admission filter for its cache tiers and leaves it **disabled by default**, on the stated grounds that the best admission policy depends on workload shape ([vLLM, 2026-05-18](https://vllm.ai/blog/2026-05-18-pegaflow)). If the people who wrote the cache will not pick a universal admission policy for you, you should be suspicious of anyone who does — including this post. The *structure* here (bound the queue by time, bound admission by worst-case memory, push back to the caller) generalizes. The *constants* do not.

### 6.3 Implementing the gate, with hysteresis

```python
# nanoserve/gate.py
from dataclasses import dataclass


@dataclass
class LoadGate:
    """Switches expensive-when-busy features off, with hysteresis.

    Two thresholds, not one: turning a feature off at 0.75 and back on at
    0.55 stops the gate from flapping every step when load sits at 0.75.
    """
    off_above: float = 0.75          # utilization at which we degrade
    on_below: float = 0.55           # utilization at which we restore
    _degraded: bool = False

    def update(self, running: int, max_running: int, queue_depth: int, queue_cap: int) -> bool:
        rho = max(running / max(max_running, 1), queue_depth / max(queue_cap, 1))
        if self._degraded and rho < self.on_below:
            self._degraded = False
        elif not self._degraded and rho > self.off_above:
            self._degraded = True
        return self._degraded

    def config(self) -> dict:
        if self._degraded:
            return {"speculative_tokens": 0,     # disable drafting entirely
                    "prefill_chunk": 1024,       # protect TPOT for the running set
                    "max_best_of": 1,
                    "prefix_caching": True}      # cheap on a miss: always on
        return {"speculative_tokens": 4,
                "prefill_chunk": 8192,
                "max_best_of": 4,
                "prefix_caching": True}
```

Hysteresis is not a nicety. A single-threshold gate at exactly the load where you spend most of your time will toggle every step, and toggling speculative decoding every step means invalidating CUDA graphs and draft-model state repeatedly — you pay both configurations' costs and get neither's benefit. Two thresholds, twenty lines apart, and the problem disappears.

## 7. Measuring the right thing: open loop, closed loop, and the collapse you cannot see

Everything above is theory until you can reproduce the collapse on demand. And here is the trap that has hidden latency collapse from more teams than any other single mistake: **the standard way to load-test a server makes collapse impossible to observe.**

![A two-row comparison of closed-loop and open-loop load generation across offered load, visibility of the knee, queue growth and intended use](/imgs/blogs/admission-control-backpressure-and-latency-collapse-6.webp)

### 7.1 Why closed-loop testing lies

A closed-loop load generator runs $N$ worker threads. Each sends a request, waits for the response, and sends the next. Offered load is therefore

$$\lambda = \frac{N}{R + Z}$$

where $R$ is the response time and $Z$ is think time. Read that equation again with the server's perspective in mind. **When the server slows down, $R$ rises, and $\lambda$ falls.** The load generator is a negative feedback controller that holds the server at exactly the load it can handle. You cannot drive $\rho$ above 1; you cannot make the queue diverge; you cannot see the pole in $\rho/(1-\rho)$ because the generator retreats from it.

What you see instead is a beautiful, reassuring, linear graph: double $N$, latency roughly doubles, throughput plateaus smoothly. Every closed-loop test of a broken server looks like a healthy server with more users. Gil Tene named the general form of this measurement error **coordinated omission**: the load generator's schedule is coordinated with the server's stalls, so the samples that would have shown the stall are never taken.

Figure 6 sums up the difference. Closed loop is the right tool for questions about the server's internals — maximum throughput, kernel efficiency, batch-size scaling — because it keeps the pipeline saturated. It is the wrong tool for every question about behavior under overload, which is to say for every question this post is about.

An open-loop generator issues requests on a schedule that does not depend on the server. Exponential inter-arrival times give Poisson arrivals at rate $\lambda$; the generator keeps issuing at $\lambda$ whether the server is answering in 200 ms or 200 seconds. Now overload is expressible, and so is shedding: a 429 is a *result*, recorded and counted, not an error that aborts the run.

### 7.2 The load generator

```python
# bench/loadgen.py — open-loop Poisson arrivals, SLO goodput accounting
import argparse, asyncio, json, random, time
import aiohttp

SLO_TTFT = 2.0        # seconds
SLO_TPOT = 0.050      # seconds between output tokens
SLO_E2E = 30.0        # hard client deadline


class Result:
    __slots__ = ("scheduled", "ttft", "tpot", "e2e", "status", "out_tokens")

    def __init__(self, scheduled):
        self.scheduled = scheduled
        self.ttft = self.tpot = self.e2e = float("inf")
        self.status = 0
        self.out_tokens = 0

    def met_slo(self) -> bool:
        return (self.status == 200 and self.ttft <= SLO_TTFT
                and self.tpot <= SLO_TPOT and self.e2e <= SLO_E2E)


async def one_request(session, url, payload, scheduled, results):
    r = Result(scheduled)
    # CRITICAL: measure from the SCHEDULED time, not from send time.
    # Timing from send time is exactly the coordinated-omission bug.
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=SLO_E2E)) as resp:
            r.status = resp.status
            if resp.status != 200:
                r.e2e = time.monotonic() - scheduled
                results.append(r)
                return
            first = None
            last = None
            async for raw in resp.content:
                if not raw.startswith(b"data:"):
                    continue
                now = time.monotonic()
                if first is None:
                    first = now
                    r.ttft = now - scheduled       # includes queue time
                last = now
                r.out_tokens += 1
            r.e2e = (last or time.monotonic()) - scheduled
            if r.out_tokens > 1 and first is not None:
                r.tpot = (last - first) / (r.out_tokens - 1)
    except asyncio.TimeoutError:
        r.status = 599
        r.e2e = time.monotonic() - scheduled
    results.append(r)


async def run(url, qps, duration, prompt_tokens, max_tokens, seed=0):
    rng = random.Random(seed)
    results, tasks = [], []
    payload = {"messages": [{"role": "user", "content": "x " * prompt_tokens}],
               "max_tokens": max_tokens, "stream": True}
    connector = aiohttp.TCPConnector(limit=0)      # never throttle at the client
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.monotonic()
        next_at = t0
        while next_at - t0 < duration:
            # Exponential gaps => Poisson process. NOT a fixed 1/qps sleep:
            # a fixed gap is far less bursty than real traffic and hides the tail.
            next_at += rng.expovariate(qps)
            delay = next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                one_request(session, url, payload, next_at, results)))
        await asyncio.gather(*tasks)
    return results


def report(results, duration, qps):
    n = len(results)
    good = sum(1 for r in results if r.met_slo())
    shed = sum(1 for r in results if r.status == 429)
    timed_out = sum(1 for r in results if r.status == 599)
    ttfts = sorted(r.ttft for r in results if r.status == 200)
    p = lambda q: ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))] if ttfts else float("inf")
    print(json.dumps({
        "offered_qps": qps,
        "issued": n,
        "goodput_rps": round(good / duration, 2),
        "shed_pct": round(100 * shed / max(n, 1), 1),
        "timeout_pct": round(100 * timed_out / max(n, 1), 1),
        "ttft_p50": round(p(0.50), 3),
        "ttft_p99": round(p(0.99), 3),
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    res = asyncio.run(run(a.url, a.qps, a.duration, a.prompt_tokens, a.max_tokens))
    report(res, a.duration, a.qps)
```

Four details in there are the difference between a useful harness and a decorative one.

**Timing starts at `scheduled`, not at send.** If the event loop is backed up, the send itself is delayed, and timing from the send hides exactly the delay you are hunting. This one line is the fix for coordinated omission.

**`expovariate`, not a fixed sleep.** A fixed $1/\lambda$ gap is a much smoother arrival process than reality; it understates queueing because it never delivers two requests at once. Poisson is the conservative default; if you have real traces, replay them.

**`TCPConnector(limit=0)`.** The default connection-pool limit in most HTTP clients silently converts your open-loop generator back into a closed-loop one at $N = \text{limit}$. Check this in whatever tool you use. It is the single most common way an "open-loop" test quietly is not one.

**429s are counted, not raised.** Shed requests are a first-class outcome. A run at 3× capacity where 66% are shed and 34% meet SLO is a *success*; the same run where 100% are accepted and 8% meet SLO is a failure. Only goodput accounting can tell those apart, which is why goodput and not throughput is the objective — the argument made at length in [the scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem).

### 7.3 The number to report

Goodput needs an explicit SLO or it means nothing. Say it out loud: *completed requests per second with TTFT under 2 s and TPOT under 50 ms*. Publishing "goodput" without the predicate is like publishing a p99 without saying of what.

There is a good public example of this discipline. The MoRIIO KV-connector write-up on [vLLM's blog](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07) reports 2.5× higher goodput than a collocated baseline for Qwen3-235B-A22B-FP8 on 8×MI300X split 4 prefill + 4 decode, ISL 2000 / OSL 1000, at 8 req/s — and states the SLO used ($\text{TTFT} \lt 1\text{ s}$ and $\text{ITL} \lt 50\text{ ms}$) plus the raw counts: 73 of 100 requests meeting SLO in write mode, 70 in read mode, 30 for the baseline. That is the right shape for a claim: a number, an SLO, a setup, and the denominator.

Here is what the model in this post predicts for our running example. Every row is `derived` from $\mu = 14$ req/s, a 30 s client deadline, and a 300 s run; the point is the shape, and you should reproduce it with `loadgen.py` against your own server and compare.

| Offered $\lambda$ (req/s) | $\rho$ | Goodput with admission control | Shed rate | Goodput with no admission control | Source           |
| ------------------------- | ------ | ------------------------------ | --------- | --------------------------------- | ---------------- |
| 7                         | 0.50   | 7.0 req/s                      | 0%        | 7.0 req/s                         | derived (model)  |
| 11                        | 0.79   | 11.0 req/s                     | 0%        | 11.0 req/s                        | derived (model)  |
| 14                        | 1.00   | 14.0 req/s                     | 0%        | 14.0 req/s (marginal)             | derived (model)  |
| 21                        | 1.50   | 14.0 req/s                     | 33%       | 2.8 req/s                         | derived (model)  |
| 28                        | 2.00   | 14.0 req/s                     | 50%       | 1.4 req/s                         | derived (model)  |
| 42                        | 3.00   | 14.0 req/s                     | 67%       | 0.7 req/s                         | derived (model)  |

The last column's arithmetic, for the $\lambda = 21$ row: the backlog grows at ${21 - 14 = 7}$ req/s, so the queue wait crosses the 30 s client deadline once the backlog reaches $30 \times 14 = 420$ requests, which happens at $t = 420/7 = 60$ s. After that every request times out. Over a 300 s run you complete $14 \times 60 = 840$ requests inside SLO, an average of 2.8 req/s.

And this model is *optimistic*, because it assumes $\mu$ stays at 14 during overload. Section 2 showed it does not — preemption drives it down. The real uncontrolled column is worse than the table says.

That is the punchline of the whole post, in one row: at 3× offered load, admission control delivers **20× the goodput** of no admission control, and it does so by refusing two-thirds of the traffic. Refusing work is how you get more work done.

## 8. Stress tests: burst, drain, and the retry storm

![A left-to-right sequence showing a queue filling in thirty seconds during a burst and taking two hundred seconds to drain afterward](/imgs/blogs/admission-control-backpressure-and-latency-collapse-7.webp)

Figure 7 is the asymmetry every on-call engineer eventually learns the hard way, and it has a one-line derivation.

### 8.1 Fill fast, drain slow

During a burst at rate $\lambda_b > \mu$, the backlog grows at $\lambda_b - \mu$ per second. After the burst, with baseline load $\lambda_0 \lt \mu$, it drains at $\mu - \lambda_0$ per second. The ratio of drain time to burst time is therefore

$$\frac{T_{\text{drain}}}{T_{\text{burst}}} = \frac{\lambda_b - \mu}{\mu - \lambda_0}$$

#### Worked example: a 30-second burst at 3× capacity

All `derived` from $\mu = 14$ req/s and a baseline of $\lambda_0 = 9.8$ req/s ($\rho = 0.70$).

- Burst rate $\lambda_b = 3\mu = 42$ req/s for 30 s.
- Backlog accumulated: $(42 - 14) \times 30 = 840$ requests.
- Drain rate: ${14 - 9.8 = 4.2}$ req/s.
- Drain time: ${840 / 4.2 = 200}$ s.
- Ratio: $200/30 = 6.7\times$.

A burst that lasted half a minute owns your latency for the next three and a half minutes. Nobody's incident timeline expects this — the traffic graph went back to normal at 12:04 and the latency graph did not recover until 12:07, so everyone goes looking for a second cause. There is no second cause. It is $\frac{\lambda_b - \mu}{\mu - \lambda_0}$.

Two consequences worth internalizing:

**Running at higher baseline utilization makes recovery superlinearly worse.** At $\rho_0 = 0.7$ the drain rate is $0.3\mu$; at $\rho_0 = 0.9$ it is $0.1\mu$, so the same burst takes three times as long to clear. Headroom is not idle capacity — it is recovery speed. That reframing is the one to bring to the capacity-planning meeting where someone proposes running at 90%.

**A queue cap bounds your recovery time, which is its underrated benefit.** With a cap of $K$, the backlog can never exceed $K$, so drain time is at most $K/(\mu - \lambda_0)$. For $K = 28$ and our numbers: 6.7 seconds, no matter how large or long the burst. The cap does not merely protect latency during the burst; it converts an unbounded recovery into a bounded one. **Bounded queue, bounded recovery.** That is the sentence to remember.

### 8.2 The retry storm

Now add clients that retry. Recall $\lambda_{\text{eff}} = \lambda \cdot \mathbb{E}[\text{attempts}]$ with $\mathbb{E}[\text{attempts}] = (1-(1-s)^{R+1})/s$ and $s = \mu/\lambda_{\text{eff}}$ once you are shedding. Solve the fixed point for $R = 3$ retries, $\mu = 14$, and an offered load of just $\lambda = 15$ req/s — a 7% overload:

$$\frac{\mu}{\lambda} = 1 - \left(1 - \frac{\mu}{\lambda_{\text{eff}}}\right)^{4} \;\Rightarrow\; \frac{14}{15} = 1 - \left(1 - \frac{14}{\lambda_{\text{eff}}}\right)^4$$

$(1 - 14/\lambda_{\text{eff}})^4 = 1/15 = 0.0667$, so $1 - 14/\lambda_{\text{eff}} = 0.508$, giving $\lambda_{\text{eff}} = 28.5$ req/s. `derived`.

**A 7% overload became a 103% overload.** The success rate at that fixed point is $14/28.5 = 49\%$ — half of all attempts rejected, when the underlying demand exceeded capacity by seven percent. Your clients did that, using retry logic that every code review would approve.

The fix is not "fewer retries" — retries are genuinely valuable against transient failures. The fix is a **retry budget** that caps retries as a fraction of successes rather than per-request, exactly as described in Google's SRE book chapter on [handling overload](https://sre.google/sre-book/handling-overload/). With a 10% budget, $\lambda_{\text{eff}} \le 1.1\lambda = 16.5$ req/s, so $s = 14/16.5 = 85\%$ instead of 49%. Same demand, same server, one bounded counter in the client.

```python
# nanoserve/client.py — a retry budget, not a retry count
import random
import time
from collections import deque


class RetryBudget:
    """Allows retries only while they stay under a fraction of successes.

    ratio=0.1 means: at most one retry per ten successful requests, measured
    over a sliding window. Under sustained overload the budget empties and
    retries stop entirely, which is the behaviour you want.
    """

    def __init__(self, ratio: float = 0.10, window_s: float = 10.0, min_per_sec: float = 1.0):
        self.ratio = ratio
        self.window_s = window_s
        self.min_per_sec = min_per_sec
        self._successes: deque[float] = deque()
        self._retries: deque[float] = deque()

    def _prune(self, now: float) -> None:
        for dq in (self._successes, self._retries):
            while dq and now - dq[0] > self.window_s:
                dq.popleft()

    def record_success(self) -> None:
        self._successes.append(time.monotonic())

    def try_consume(self) -> bool:
        now = time.monotonic()
        self._prune(now)
        allowance = self.ratio * len(self._successes) + self.min_per_sec * self.window_s
        if len(self._retries) >= allowance:
            return False                      # budget exhausted: fail fast, do not resend
        self._retries.append(now)
        return True


def backoff_delay(attempt: int, base: float = 0.2, cap: float = 8.0) -> float:
    """Full jitter. A deterministic backoff synchronises every client you own."""
    return random.uniform(0.0, min(cap, base * (2 ** attempt)))
```

The `min_per_sec` floor exists so a client that has had zero successes (because the backend just came back from a cold start) can still make a first attempt. Without it, a budget that empties completely never refills — the client has locked itself out of the recovery it is waiting for.

### 8.3 The simulator: reproduce all of it without a GPU

You should not need an A100 to watch congestion collapse. This is a discrete-event simulator over the same step-time model, with a block pool, worst-case admission, a queue cap and preemption. Run it, change one parameter, watch the goodput column turn over.

```python
# nanoserve/sim_admit.py — pure Python, no torch, no GPU
import heapq
import math
import random
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    max_tokens: int
    output: int                    # realized length, unknown to the scheduler
    produced: int = 0
    blocks: int = 0
    start: float | None = None
    done: float | None = None
    preemptions: int = 0


def simulate(*, qps, duration=300.0, mu_blocks=29_696, block=16,
             t0=7.85e-3, beta=0.19e-3, max_running=185,
             queue_cap=None, worst_case_admit=True, deadline=30.0, seed=0):
    """queue_cap=None means an unbounded queue (the uncontrolled baseline)."""
    rng = random.Random(seed)
    waiting, running = [], []
    free = mu_blocks
    completed_in_slo = shed = expired = preempted = 0
    t, rid = 0.0, 0
    next_arrival = rng.expovariate(qps)

    def worst_blocks(r):
        return math.ceil((r.prompt + r.max_tokens) / block)

    def live_blocks(r):
        return math.ceil((r.prompt + r.produced) / block)

    while t < duration:
        # --- arrivals up to the next step boundary -------------------------
        step_dt = t0 + beta * len(running)
        while next_arrival <= t + step_dt:
            rid += 1
            out = min(512, max(1, int(rng.lognormvariate(5.4, 0.55))))
            r = Req(rid, next_arrival, 2048, 512, out)
            if queue_cap is not None and len(waiting) >= queue_cap:
                shed += 1
            else:
                waiting.append(r)
            next_arrival += rng.expovariate(qps)

        # --- admission ------------------------------------------------------
        while waiting and len(running) < max_running:
            head = waiting[0]
            if t - head.arrival > deadline:
                waiting.pop(0); expired += 1; continue
            need = worst_blocks(head) if worst_case_admit else math.ceil(head.prompt / block)
            if need > free:
                break
            waiting.pop(0)
            free -= need if worst_case_admit else need
            head.blocks = need
            head.start = t
            running.append(head)

        # --- one decode step -------------------------------------------------
        t += step_dt
        finished = []
        for r in running:
            r.produced += 1
            if not worst_case_admit:
                want = live_blocks(r)
                if want > r.blocks:
                    if free <= 0:
                        # no block available: preempt this request (recompute policy)
                        finished.append((r, "preempt")); continue
                    free -= 1; r.blocks += 1
            if r.produced >= r.output:
                finished.append((r, "done"))

        for r, why in finished:
            running.remove(r)
            free += r.blocks
            if why == "done":
                r.done = t
                if (r.start - r.arrival) <= 2.0 and (r.done - r.arrival) <= deadline:
                    completed_in_slo += 1
            else:
                preempted += 1
                r.preemptions += 1
                r.produced = 0                 # recompute: all progress is lost
                r.blocks = 0
                waiting.insert(0, r)

    return {
        "qps": qps,
        "goodput_rps": round(completed_in_slo / duration, 2),
        "shed_pct": round(100 * shed / max(rid, 1), 1),
        "expired_pct": round(100 * expired / max(rid, 1), 1),
        "preemptions": preempted,
        "queue_depth_end": len(waiting),
    }


if __name__ == "__main__":
    for q in (7, 11, 14, 21, 28, 42):
        print("capped  ", simulate(qps=q, queue_cap=28, worst_case_admit=True))
        print("uncapped", simulate(qps=q, queue_cap=None, worst_case_admit=False))
```

Run it and watch two things. In the capped configuration, `goodput_rps` rises to roughly $\mu$ and then flattens while `shed_pct` climbs — the system trades acceptance for stability. In the uncapped configuration, `goodput_rps` rises, peaks, and then **falls**, while `preemptions` climbs into the thousands and `queue_depth_end` grows without bound. That downturn is congestion collapse, reproduced on a laptop in a few seconds of CPU time.

The `r.produced = 0` line is where the whole feedback loop lives. Change it to `r.produced = r.produced // 2` (a swap-based policy that preserves half the progress, in the spirit of [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping)) and watch how much later the collapse arrives. The recompute policy is not merely more expensive per preemption; it raises the loop gain, and loop gain is what determines whether there is a stable operating point at all.

### 8.4 A stress checklist

Before you believe a serving deployment is production-ready, run these five and write down what happened:

1. **Sustained 1.5× capacity for 10 minutes.** Expected: goodput flat at $\mu$, shed rate ~33%, p99 TTFT stable for accepted requests. Failure signature: goodput falling, preemption counter climbing.
2. **A 30-second 3× burst on a 0.7 baseline.** Expected: drain time bounded by $K/(\mu-\lambda_0)$. Failure signature: recovery taking minutes, which means your queue is effectively unbounded somewhere (check the gateway, the ingress, the client pool).
3. **Retry storm.** Point a client with 3 retries and no budget at a server at 1.05× capacity. Expected with a budget: success rate stays high. Without: watch $\lambda_{\text{eff}}$ roughly double, exactly as §8.2 derives.
4. **A single pathological request.** One 100k-token prompt with `max_tokens=8192` arriving into a busy server. Expected: rejected by the predicate as `request_exceeds_capacity` or admitted only when it genuinely fits. Failure signature: it is admitted, evicts a dozen requests, and stalls everyone.
5. **Cold start under load.** Restart the server while the load generator runs at 0.8 capacity. Expected: shedding during warmup, then convergence. Failure signature: an unbounded queue accumulates during model load and the server is instantly at 5× capacity the moment it accepts its first request — a self-inflicted burst that has taken down more deployments than any organic traffic spike.

Test 5 is the one everybody skips and the one that catches the most bugs.

## 9. Case studies and public numbers

Four public results that make the argument concretely, each cited with its setup. None of these are measurements I made; every one is a claim by its authors on their hardware, and the setup is part of the claim.

**Speculative decoding flips sign with load.** The vLLM team's [*How Speculative Decoding Boosts vLLM Performance by up to 2.8×*](https://vllm.ai/blog/2024-10-17-spec-decode) (2024-10-17) reports up to 1.5× with a draft model (Llama-3-70B, 4×H100, ShareGPT, QPS=1) and up to 2.8× with prompt-lookup n-gram drafting on CNN/DailyMail summarization — and then, in the same post, that at high QPS these become a **1.4× slowdown** (draft, ShareGPT) and a **1.8× slowdown** (n-gram, CNN/DailyMail). This is the clearest published statement that a latency optimization can be a throughput liability, and it validates the roofline derivation in §6.1: past $B^\star \approx 31$ for $k=4$, the extra verification FLOPs are no longer free.

**Goodput under an explicit SLO, not throughput.** The [MoRIIO KV connector post](https://vllm.ai/blog/2026-04-07-moriio-kv-connector) (2026-04-07) reports 2.5× higher goodput than a collocated baseline for Qwen3-235B-A22B-FP8 on 8×MI300X (4 prefill + 4 decode, ISL 2000 / OSL 1000, 8 req/s), with the SLO stated as TTFT under 1 s and ITL under 50 ms, and the counts given as 73/100 and 70/100 meeting SLO versus 30/100 for the baseline. Note also its acknowledged constraints — single-node only, and prefix caching must be disabled with `--no-enable-prefix-caching`. Constraints are part of a result.

**Adaptive LIFO and CoDel in front of a real service.** Ben Maurer's [*Fail at Scale*](https://queue.acm.org/detail.cfm?id=2839461) (ACM Queue, 2015) describes Facebook's use of a controlled-delay queue plus adaptive LIFO: FIFO in normal operation, LIFO when the queue is deep, on the reasoning that under overload the oldest request is the one most likely already abandoned. The underlying CoDel algorithm is Nichols and Jacobson's [*Controlling Queue Delay*](https://queue.acm.org/detail.cfm?id=2209336) (ACM Queue, 2012). Both are worth reading in full; the §3.2 implementation is a direct port.

**Retry budgets and client-side throttling.** Google's SRE book chapter on [handling overload](https://sre.google/sre-book/handling-overload/) describes per-client retry budgets (their example caps retries at around 10% of requests) and adaptive throttling, where a client experiencing rejections begins rejecting locally. The §8.2 fixed-point calculation is why: without a budget, a 7% overload amplifies to 103%.

**And one negative result worth citing.** vLLM's [PegaFlow](https://vllm.ai/blog/2026-05-18-pegaflow) post (2026-05-18) ships a TinyLFU admission filter for its KV cache tiers and leaves it **off by default**, stating that the best admission policy depends on workload shape. Their headline numbers are strong (Qwen3-8B single host: +56% throughput at 11.97 vs 7.68 req/s, −36% TTFT, hit rate 52.35% vs 11.77%), which makes the restraint more instructive, not less: a team with those results still would not pick a default admission policy for other people's traffic.

## 10. When to reach for this (and when not to)

**Do this always, even at toy scale.** The memory-aware predicate in §4.1 is thirty lines and it is the difference between a server that degrades and a server that dies. There is no scale small enough to skip it — a single-GPU dev box with two users can be OOM'd by one 100k-token request with a large `max_tokens`.

**Do this if you run anything user-facing.** The queue cap, the deadline check, the 429 with jittered `Retry-After`. All of it is cheap, none of it needs tuning beyond the two formulas ($K = \mu T_q$ and the worst-case block count), and every piece pays for itself the first time you have a burst.

**Consider CoDel over a fixed cap when $\mu$ varies a lot.** If your traffic mixes 200-token chats and 32k-token RAG requests, $\mu$ swings by an order of magnitude and a fixed queue length is wrong in both directions. If your traffic is homogeneous, the fixed cap is simpler and fine.

**Skip the load gate if you never run above 60% utilization.** If you have provisioned generously and $\rho$ stays low, gating speculative decoding buys nothing and adds a failure mode (a gate stuck in the wrong state is worse than no gate). Revisit when your utilization dashboards start touching 0.7.

**And use vLLM instead of your own engine if you are shipping a product.** vLLM, SGLang and TGI have all of this — `--max-num-seqs`, `--max-num-batched-tokens`, scheduler policies, chunked prefill, prefix caching, preemption, metrics — plus years of edge cases you have not thought of, and a router with KV-affinity routing ([vLLM Router](https://vllm.ai/blog/2025-12-13-vllm-router-release), 2025-12-13). Write `nanoserve` to understand *why* `--max-num-seqs` matters and what number to set it to. Then set it, in vLLM, and go build the product. That distinction is the whole thesis of [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and it is worth restating here because this is the post where the temptation to roll your own is strongest: admission control looks easy, and the easy version is the one that thrashes.

**One case where you genuinely do need your own layer:** when admission must consider information the engine cannot see — per-tenant quotas, business priority, a paid tier that must never be shed, a fairness guarantee across customers. That logic belongs in a gateway in front of vLLM, using vLLM's own metrics endpoint to estimate $\mu$ and pool pressure. Build that. Do not rebuild the block allocator underneath it.

For the broader operational picture — autoscaling, multi-replica routing, capacity planning across a fleet — see [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management), which covers the layer above a single engine.

## 11. Key takeaways

1. **Mean queue wait is $\rho/(1-\rho)$ service times.** At 90% utilization you wait 9× the service time; at 99%, 99×. The last 10% of capacity costs more than the first 90%, and no amount of GPU headroom in the dashboard changes that.
2. **The knee is structural, not statistical.** Substituting Little's law into the batch step-time model $t(B) = t_0 + \beta B$ produces the same $\rho/(1-\rho)$ pole with no queueing-theory assumptions at all.
3. **An LLM server thrashes rather than queues.** Preemption recomputes discarded tokens, which raises load, which causes more preemption. Goodput *falls* as offered load rises, and recovery needs load below the level that started it. That is congestion collapse, described for packet networks in RFC 896 forty years ago.
4. **The queue cap is $K = \mu \cdot T_q$.** Service rate times queue-wait budget. Fourteen requests per second times a two-second budget is 28 — not 1,000, and not "unbounded because dropping traffic feels bad".
5. **Never admit a request whose worst-case KV cannot fit alongside the running set's outstanding commitments.** Counting current occupancy instead of future commitments is how you reach 97% pool usage with 180 requests each about to ask for one more block.
6. **Aggressive admission buys almost nothing.** Past the point where $\beta B \gg t_0$, going from a worst-case to a median token budget raised concurrency 12% and throughput 2% in the worked example, in exchange for real preemption risk. Conservative admission is the better trade, and you can only see that once the step-time model is written down.
7. **Reject fast or waste GPU time.** A request that times out after consuming 80% of a generation is pure waste; at a 50% timeout rate that waste was 44% of the GPU. The 429 is not the failure — the timeout is.
8. **Techniques that spend spare capacity must be gated on spare capacity existing.** Speculative decoding at $k=4$ stops being free around batch 31 on an A100, and vLLM measured up to a 1.8× slowdown at high QPS. Load-gate it, with hysteresis.
9. **Closed-loop load testing cannot show you collapse.** Fixed-concurrency clients throttle themselves the instant the server slows. Use Poisson arrivals, time from the scheduled instant rather than the send, and count 429s as outcomes.
10. **A bounded queue means bounded recovery.** A burst fills at $\lambda_b - \mu$ and drains at $\mu - \lambda_0$; a 30-second 3× burst took 200 seconds to clear in the worked example. With a queue cap of 28, it takes 6.7 seconds regardless of the burst.
11. **Retries amplify overload multiplicatively.** Three retries with no budget turned a 7% overload into a 103% overload at the fixed point. A 10% retry budget takes the success rate from 49% to 85% with no server change at all.

## 12. Further reading

- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the scheduler, waiting and running queues, and the metric definitions (TTFT, ITL, TPOT) this post assumes.
- [How Speculative Decoding Boosts vLLM Performance by up to 2.8×](https://vllm.ai/blog/2024-10-17-spec-decode) — including the high-QPS slowdown that motivates load gating.
- [Handling Overload](https://sre.google/sre-book/handling-overload/), Google SRE book — retry budgets, client-side adaptive throttling, and load shedding as a first-class outcome.
- [Fail at Scale](https://queue.acm.org/detail.cfm?id=2839461), Ben Maurer, ACM Queue 2015 — adaptive LIFO and controlled delay in front of a production service.
- [Controlling Queue Delay](https://queue.acm.org/detail.cfm?id=2209336), Nichols and Jacobson, ACM Queue 2012 — the CoDel algorithm ported in §3.2.
- [RFC 896: Congestion Control in IP/TCP Internetworks](https://www.rfc-editor.org/rfc/rfc896), John Nagle, 1984 — the original description of congestion collapse.
- [The scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) — who runs next among the requests you accepted, and why goodput is the objective.
- [Eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) — what happens when admission control fails and the pool runs dry mid-generation.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — where the 128 KiB per token and the 29,696 blocks come from.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where every lever in this series gets assembled into one decision procedure.


