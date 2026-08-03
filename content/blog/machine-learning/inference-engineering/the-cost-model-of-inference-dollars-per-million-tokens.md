---
title: "The cost model of inference: From GPU-hours to dollars per million tokens"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Derive an auditable inference price from GPU-hour, prefill and decode throughput, utilization, batching, and the self-host versus API break-even point."
tags: ["inference-engineering", "llm-inference", "gpu", "batching", "latency", "throughput", "ml-systems", "vllm", "cost-optimization"]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

An inference bill is usually presented as a price per million tokens, while the machine underneath is billed by the hour and behaves in milliseconds. That gap is where bad capacity plans are born. A team sees an H100 at \$5.19 per GPU-hour, divides by a benchmark number copied from a README, and concludes that self-hosting is cheaper than an API. The arithmetic is neat. The denominator is usually wrong.

![The cost model turns a GPU hour into useful token economics](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-1.webp)

The diagram above is the mental model for this post: a GPU hour becomes a quantity of useful tokens only after the engine, workload, batch, and SLO have consumed some of that hour. By the end, you will be able to take a measured or reader-reproducible tok/s rate, convert it into \$ per million input or output tokens, price a mixed request, and solve the API break-even rate with a few lines of Python. The `nanoserve` artifact is a small cost model rather than a new kernel: it makes the denominator explicit beside the scheduler and serving metrics built in earlier posts.

This is post 49 in [Inference Engineering — what we are building](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It connects the engine’s scoreboard to the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook): TTFT, TPOT, tok/s, GPU memory, goodput, and \$ per million tokens describe one system from different angles.

## 1. The unit you actually want

The useful unit is not “GPU time.” It is the cost of a token that a customer can use. Start with four quantities:

- $p_g$: the all-in price of one accelerator-hour, in dollars per GPU-hour.
- $u$: utilization of the billed hour, from zero to one. This includes idle gaps, cold starts, health checks, and capacity reserved for bursts.
- $r$: useful throughput, in tokens per second. Say whether this means input tokens, output tokens, or a mixture.
- $g$: goodput fraction, the fraction of useful tokens belonging to requests that met their TTFT and TPOT service-level objectives.

The simplest cost is:

$$
C_{1M} = \frac{p_g \times 3{,}600}{r}.
$$

The million-token cancellation is worth showing. One hour contains $3{,}600$ seconds and therefore produces $3{,}600r$ tokens. The hourly price divided by that output is:

$$
\frac{p_g}{3{,}600r / 1{,}000{,}000}\times 1{,}000{,}000 = \frac{3{,}600p_g}{r}.
$$

If the GPU is \$0.35 per hour and the engine produces 70 useful output tokens per second, the result is $3{,}600 \times 0.35 / 70 = 18$ dollars per million output tokens. In Markdown that currency statement is written as `\$18`; the dollar signs inside the displayed equation are math delimiters, not currency.

The more honest version includes idle time and SLO qualification:

$$
C_{1M,\text{good}} = \frac{3{,}600p_g}{u \times g \times r}.
$$

This is an accounting identity for the post, not a formula claimed by a serving engine. It says why an impressive maximum-throughput benchmark can still produce an expensive service: if only 40% of the reserved hour carries traffic, the denominator is 0.4 of the advertised rate. If 85% of tokens are SLO-qualified, goodput costs another factor of $1/0.85$.

### Worked example: a clean output-token conversion

Assume a single GPU costs \$2.00 per hour and a steady decode benchmark reports 500 output tok/s at the chosen latency target. The arithmetic is:

$$
3{,}600 \times \$2.00 = \$7{,}200 \text{ per hour},
$$

$$
500 \text{ tok/s} \times 3{,}600 \text{ s} = 1{,}800{,}000 \text{ tok/h},
$$

$$
\$7{,}200 / 1.8 = \$4{,}000 \text{ per million output tokens}.
$$

That last number is deliberately simple, not a measured result. A real report must attach the model, dtype, context, batch, and latency target to the 500 tok/s figure. Replace the assumption with the output of [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

## 2. GPU-hour is not one universal price

There are at least four different prices people call “the GPU cost.” Keep them separate.

| Price layer | What it includes | Example arithmetic | Source |
|---|---|---:|---|
| Cloud accelerator rate | A provider’s instance or capacity-block charge | \$5.191 per H100-hour in the cited AWS row | [cited: AWS EC2 Capacity Blocks, accessed 2026-08-03](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| Host rate | GPU plus CPU, RAM, networking, storage, and provider margin | \$41.528 / 8 H100 = \$5.191 per accelerator-hour | [cited: AWS EC2 Capacity Blocks, accessed 2026-08-03](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| Owned hardware rate | Purchase price divided by useful lifetime hours | \$30,000 / 20,000 h = \$1.50 per hour | derived assumption |
| Service rate | Hardware plus power, people, software, redundancy, and unused capacity | \$5.191 / 0.60 utilization = \$8.652 per busy hour | derived assumption |

AWS lists an effective \$41.528 hourly rate for an eight-H100 `p5.48xlarge` capacity block in several US regions, and \$5.191 per accelerator in the same table, retrieved on August 3, 2026. That is a cited cloud input, not a claim that every region or purchasing mode costs the same. On-demand, reserved, spot, capacity blocks, taxes, egress, and support can all change it. The [AWS P5 page](https://aws.amazon.com/ec2/instance-types/p5/) describes the instance family; its page is not a substitute for the region-specific pricing row.

For owned hardware, use an annualized cost rather than pretending the purchase is free:

$$
p_{\text{owned}} = \frac{P_{\text{purchase}} + P_{\text{rack}} + P_{\text{support}} + P_{\text{power}}}{H_{\text{useful}}}.
$$

If a \$30,000 server has one relevant GPU, \$4,000 of allocable infrastructure cost, and 20,000 useful hours, the numerator is \$34,000 and the hourly rate is \$34,000 / 20,000 = \$1.70 per hour. If it is powered for 20,000 hours but carries traffic for only 12,000 of them, its busy-hour cost is \$34,000 / 12,000 = \$2.833. Utilization belongs in the cost model even when the electric meter does not send a separate invoice.

The NVIDIA H100 specification is a useful reminder that “H100” is not enough metadata. NVIDIA’s H100 page lists 80 GB memory and 3.35 TB/s memory bandwidth for H100 SXM, with up to 700 W configurable TDP, accessed August 3, 2026. Those are hardware facts, cited at [NVIDIA’s H100 specifications](https://www.nvidia.com/en-sg/data-center/h100/). The machine rate still depends on the host and purchase contract. The bandwidth matters because decode often approaches a memory-traffic ceiling, but it does not turn the specification into a tok/s result.

## 3. Prefill and decode are different products

Prefill processes the input prompt. It can form a large matrix of tokens and usually exposes parallel matrix-multiply work. Decode appends one or a small number of tokens at a time, reads the existing key-value cache, and samples the next token autoregressively. TTFT is time from submission to the first token; TPOT is average time per output token after that first token. The [vLLM anatomy article](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), published September 5, 2025, uses these definitions and reports throughput benchmarks with explicit input/output settings. We cite that vocabulary; we do not treat its benchmark as ours.

![Prefill and decode consume different scarce resources before their costs merge](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-2.webp)

For a request with $S$ input tokens and $T$ output tokens, a first-order request cost is:

$$
t_{\text{request}}(S,T,B) = t_{\text{queue}} + t_{\text{prefill}}(S,B) + T \times t_{\text{decode-step}}(B,S).
$$

The batch $B$ matters in both terms, but not in the same way. Prefill often benefits when $B$ or total prompt tokens increase because matrix operations become wider. Decode has one new token per active sequence but must read model weights and the relevant KV state on every step. More active sequences can amortize overhead and use more of a GPU, yet they also increase KV traffic, memory pressure, and per-request wait.

The price of a mixed request therefore cannot be derived from a single “tokens per second” number unless that number is measured on the same mix. If input and output have separate rates $r_i$ and $r_o$, and the engine price is $p_g$, a simple phase-time estimate is:

$$
t_{\text{GPU}} \approx \frac{S}{r_i} + \frac{T}{r_o}.
$$

The corresponding cost is:

$$
C_{\text{request}} \approx p_g \times \frac{S/r_i + T/r_o}{3{,}600}.
$$

This simplified planning equation is useful for sensitivity analysis, not a replacement for a scheduler trace. It ignores overlap, queueing, kernel launch gaps, prefix-cache hits, rejected requests, batching across tenants, and the fact that a decode rate measured at batch 16 is not the rate an isolated request experiences.

### Worked example: why output shape changes the bill

Use a cited AWS H100 accelerator rate of \$5.191 per hour, then make reader-replaceable throughput assumptions: 10,000 input tok/s at the selected prefill batch and 1,000 output tok/s at the selected decode batch. A request with 2,000 input and 512 output tokens consumes:

$$
t_i = 2{,}000 / 10{,}000 = 0.20 \text{ s},
$$

$$
t_o = 512 / 1{,}000 = 0.512 \text{ s},
$$

$$
t_{\text{GPU}} = 0.20 + 0.512 = 0.712 \text{ s}.
$$

The compute-only request cost is \$5.191 × 0.712 / 3,600 = \$0.001027. The same input with 4,096 generated tokens takes 0.20 + 4.096 = 4.296 seconds and costs \$5.191 × 4.296 / 3,600 = \$0.006195. Output grew by 8× while cost grew about 6.03× because input prefill is paid once. If decode is slower than this assumption, the gap grows. If a prefix cache removes most of prefill, it shrinks in the other direction.

## 4. Deriving throughput from hardware ceilings

The price formula is only as defensible as its tok/s input. Before running a benchmark, derive a ceiling that tells you whether the result is plausible.

For a dense 8B model in BF16, the weight payload is approximately:

$$
8 \times 10^9 \text{ parameters} \times 2 \text{ bytes} = 16 \times 10^9 \text{ bytes} = 16 \text{ GB decimal}.
$$

If one decode step had to stream all 16 GB of weights and the H100 SXM bandwidth were 3.35 TB/s, the bandwidth-only lower bound would be:

$$
16 \times 10^9 / 3.35 \times 10^{12} = 0.00478 \text{ s} = 4.78 \text{ ms}.
$$

The reciprocal is about $1 / 0.00478 = 209$ steps per second for a single token stream. This is not a prediction. It is a deliberately optimistic lower-bound calculation: it assumes every byte is transferred at the specification peak, ignores KV reads, activations, synchronization, sampling, and the distinction between decimal and binary units. A measured single-request decode rate above that crude number means the hardware is reusing weights through cache behavior or the model payload is smaller than 16 GB; it does not mean physics was broken. Conversely, a real decode step can be slower because the arithmetic and memory access pattern are not an ideal streaming copy.

Batching changes the useful ceiling. If an idealized step time is $t_1$ at batch one and a batch of $B$ tokens completes in $t_B$, aggregate decode throughput is:

$$
r_B = B/t_B.
$$

Cost per output token falls as $r_B$ rises, but only while the requests remain useful. A service that produces 5,000 aggregate tok/s at a 100 ms TPOT target may be economically better than one that produces 6,000 tok/s while violating a 50 ms target. The correct denominator is then goodput, not raw output.

The KV cache makes the batch constraint concrete. For a standard transformer layer with $L$ layers, $H_{kv}$ key/value heads, head dimension $d$, two tensors for K and V, and $b$ bytes per element, the per-token KV payload is:

$$
\text{KV bytes/token} = 2 \times L \times H_{kv} \times d \times b.
$$

For Llama-3.1-8B’s commonly documented 32 layers, 8 KV heads, head dimension 128, and BF16, the arithmetic is $2 × 32 × 8 × 128 × 2 = 131{,}072$ bytes, or 128 KiB per token. A 2,048-token active context therefore consumes $2{,}048 × 131{,}072 = 268{,}435{,}456$ bytes, exactly 256 MiB before allocator metadata and padding. These architecture values should be checked against the model config used by the reader; the arithmetic is derived.

At batch 32 and 2,048 tokens per request, that payload is $32 × 256$ MiB = 8,192 MiB, or 8 GiB. Add 16 GB decimal of BF16 weights, CUDA workspace, activations, fragmentation, and safety reserve, and a 24 GB RTX 4090 cannot treat all nominal memory as KV capacity. This is why the batch value in a cost row must include context length and dtype, not just “batch 32.”

## 5. Batch economics: lower unit cost, higher waiting cost

The aggregate formula is simple:

$$
C_{1M}(B) = \frac{3{,}600p_g}{r_B} = \frac{3{,}600p_g \times t_B}{B}.
$$

If batch 1 takes 10 ms for one output token, the aggregate rate is $1 / 0.010 = 100$ tok/s. At batch 16, if the step takes 16 ms, aggregate rate is $16 / 0.016 = 1{,}000$ tok/s, a 10× throughput increase for a 1.6× step-time increase. With \$5.191 per hour, the corresponding raw-output unit costs are \$186.876 per million and \$18.6876 per million. The second number is attractive because the GPU is no longer mostly idle between memory transactions.

But a request in batch 16 does not receive one-sixteenth of the step time. It waits behind the batch’s work and shares the step cadence. If a step is 16 ms, the TPOT floor from the engine’s perspective is roughly 16 ms before network and queue effects. If a scheduler admits a long prefill into the same step, decode tokens can wait more. Batch economics is therefore a constrained optimization:

$$
\max_B \quad \frac{B}{t_B(B)}
\quad \text{subject to} \quad \text{TPOT}_{p99}(B) \leq S_{\text{TPOT}},
\quad \text{TTFT}_{p99}(B) \leq S_{\text{TTFT}},
\quad M(B) \leq M_{\text{available}}.
$$

This is an explanatory abstraction of admission control, not a claim about a particular scheduler’s implementation. The optimum is usually a region, not one magic batch. A production scheduler should record the curve rather than hard-code a batch copied from a different prompt distribution.

![Batching lowers raw token cost but trades it against per-request latency](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-3.webp)

The figure’s numbers are a derived illustration: batch 1 at 1,000 aggregate tok/s costs $3{,}600 × \$5.191 / 1{,}000 = \$18.6876 per million, while batch 16 at 5,000 tok/s costs $3{,}600 × \$5.191 / 5{,}000 = \$3.73752 per million. Per-request throughput falls from 1,000 to $5{,}000/16 = 312.5$ tok/s only in the aggregate idealization; actual latency must be measured.

![The cost accounting matrix separates capacity, steady output, and SLO-qualified output](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-4.webp)

The animated figure below shows why the curve has a bend. At first, increasing batch fills idle compute capacity. Later, the same batch consumes more KV memory and makes each step longer; raw throughput still may rise while SLO-qualified throughput flattens. Motion carries the idea here because the changing denominator, rather than one static snapshot, is the important object.

<figure class="blog-anim"><svg viewBox="0 0 760 300" role="img" aria-label="Animated batch curve showing throughput rising then flattening while latency rises" xmlns="http://www.w3.org/2000/svg"><title>Batch throughput and latency curve</title><style>.axis{stroke:#64748b;stroke-width:2}.throughput{fill:none;stroke:#2563eb;stroke-width:5;stroke-linecap:round}.latency{fill:none;stroke:#d97706;stroke-width:5;stroke-linecap:round}.label{font:16px system-ui,sans-serif;fill:#334155}.dot{fill:#2563eb}.dot2{fill:#d97706}@keyframes sweep{0%,15%{stroke-dashoffset:900}75%,100%{stroke-dashoffset:0}}.throughput{stroke-dasharray:900;animation:sweep 5s ease-in-out infinite}.latency{stroke-dasharray:900;animation:sweep 5s ease-in-out infinite reverse}@media (prefers-reduced-motion: reduce){.throughput,.latency{animation:none;stroke-dashoffset:0}}</style><line class="axis" x1="70" y1="250" x2="710" y2="250"/><line class="axis" x1="70" y1="250" x2="70" y2="35"/><path class="throughput" d="M90 225 C180 160 250 105 350 82 C470 58 580 62 690 64"/><path class="latency" d="M90 235 C180 230 270 218 360 190 C470 150 580 95 690 48"/><circle class="dot" cx="690" cy="64" r="7"/><circle class="dot2" cx="690" cy="48" r="7"/><text class="label" x="94" y="278">batch 1</text><text class="label" x="635" y="278">batch 64</text><text class="label" x="82" y="48">value</text><text class="label" x="580" y="84">raw tok/s</text><text class="label" x="560" y="132">TPOT / wait</text><figcaption>As batch grows, raw throughput rises toward a plateau while latency keeps accumulating; choose the point that meets the SLO.</figcaption></svg></figure>

## 6. Pricing input and output separately

An API invoice often prices input and output at different rates because prompt processing and generation have different demand and capacity characteristics. Self-hosting does not need two invoices, but it still needs two denominators if the workload is mixed.

Let $q_i$ be input tokens per request, $q_o$ output tokens per request, and $n$ requests per hour. If the service sustains $r_i$ input tok/s and $r_o$ output tok/s on the appropriate workload, estimated GPU seconds are:

$$
T_{\text{GPU/hour}} = n \times \left(\frac{q_i}{r_i} + \frac{q_o}{r_o}\right).
$$

Capacity is feasible when $T_{\text{GPU/hour}} \leq 3{,}600u$. If it is larger, the service needs more GPUs, a faster configuration, a smaller workload, or a different SLO. The rate is not a free parameter: if increasing concurrency changes $r_o$, recompute the inequality at that concurrency.

For a provider with input price $a_i$ and output price $a_o$, the API cost per request is:

$$
C_{\text{API}} = \frac{q_i}{1{,}000{,}000}a_i + \frac{q_o}{1{,}000{,}000}a_o.
$$

The official [OpenAI API pricing page](https://openai.com/api/pricing/) is the source to use for a live comparison; prices change, so record the retrieval date and model snapshot in a cost report. As a dated illustrative input for the arithmetic below, the page retrieved August 3, 2026 lists GPT-5.6 Luna at \$1.00 per million input tokens and \$6.00 per million output tokens. That is an API price example, not a quality or latency equivalence claim between that model and Llama-3.1-8B.

| Request shape | API input rate | API output rate | API arithmetic | Source |
|---|---:|---:|---:|---|
| 1,000 in / 100 out | \$1.00 / 1M | \$6.00 / 1M | 0.001×\$1 + 0.0001×\$6 = \$0.0016 | cited: OpenAI API pricing, retrieved 2026-08-03 |
| 2,000 in / 512 out | \$1.00 / 1M | \$6.00 / 1M | 0.002×\$1 + 0.000512×\$6 = \$0.005072 | cited: OpenAI API pricing, retrieved 2026-08-03 |
| 8,000 in / 2,000 out | \$1.00 / 1M | \$6.00 / 1M | 0.008×\$1 + 0.002×\$6 = \$0.0200 | cited: OpenAI API pricing, retrieved 2026-08-03 |

The input-output ratio changes the winner. In a RAG workload with 8,000 input and 200 output tokens, the illustrative API cost is \$0.008 + \$0.0012 = \$0.0092. In a reasoning workload with 1,000 input and 4,000 output tokens, it is \$0.001 + \$0.024 = \$0.025. Self-hosting must be compared to that shape, not to a blended average that hides which phase dominates.

![A request timeline shows queue, prefill, first token, and repeated decode work](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-5.webp)

## 7. Self-host versus API break-even

Break-even means equal cost for an identical, quality-acceptable workload. It does not mean self-hosting matches the API model’s quality, tools, uptime, or safety behavior. Treat those as gates before doing arithmetic.

For a fixed request shape, self-host cost under the two-rate approximation is:

$$
C_{\text{self}} = p_g \times \frac{q_i/r_i + q_o/r_o}{3{,}600}.
$$

If the API cost is $C_{\text{API}}$, self-hosting breaks even when:

$$
p_g \times \frac{q_i/r_i + q_o/r_o}{3{,}600} = C_{\text{API}}.
$$

Solving for the required combined workload rate is easiest after choosing a mix. For a single blended request of $q=q_i+q_o$ tokens and a measured blended rate $r$, the threshold is:

$$
r_{\text{break-even}} = \frac{3{,}600p_gq}{1{,}000{,}000C_{\text{API}}}.
$$

This is a derived threshold, not a benchmark. It tells you how fast the self-hosted system would need to produce tokens under the exact request mix to match the API invoice.

### Worked example: 2,000 input and 512 output

The illustrative API price above makes the request cost \$0.005072. Assume an H100 accelerator at \$5.191 per hour and a self-hosted system that measures one blended request rate. There are 2,512 total tokens, so the break-even rate is:

$$
r_{\text{break-even}} = \frac{3{,}600 × 5.191 × 2{,}512}{1{,}000{,}000 × 0.005072}.
$$

The numerator is $3{,}600 × 5.191 × 2{,}512 = 46{,}916{,}179.2$. The denominator is $5{,}072$. The threshold is $46{,}916{,}179.2 / 5{,}072 = 9{,}249.6$ blended tokens per second while the GPU is continuously occupied. If the system is only 60% utilized, divide the effective rate by 0.60: it needs approximately $9{,}249.6 / 0.60 = 15{,}416$ tokens per second of busy-time capacity to deliver the same calendar-hour economics.

That threshold looks high because the illustrative API price is low relative to an H100 hour and because the request includes output tokens priced at \$6 per million. It is not an argument for or against either deployment. Change the API model, cloud rate, utilization, model quality, or workload mix and the answer changes.

For a phase-aware calculation, use the reader’s reproducible rates instead of the blended shortcut. Suppose the same request has $r_i=10{,}000$ input tok/s and $r_o=1{,}000$ output tok/s. Its self-host GPU time is $2{,}000/10{,}000 + 512/1{,}000 = 0.712$ seconds, and cost is \$0.001027, lower than \$0.005072 before utilization and non-GPU costs. At 60% utilization, calendar-time allocation makes it \$0.001712. At 20% utilization, it becomes \$0.005135, nearly equal to the API. This is the central break-even result: utilization can move a self-hosted system from cheaper to more expensive without changing the kernel.

![The break-even grid changes with token mix and utilization](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-6.webp)

The figure uses the same derived API rates and shows why the long-output row needs a faster self-hosted rate. For 1,000 input and 100 output tokens, API cost is \$0.0016; the equivalent H100 busy-time threshold is $3{,}600 × 5.191 × 1{,}100 / (1{,}000{,}000 × 0.0016) = 12{,}855$ tokens/s. For 2,000 input and 512 output, the threshold is approximately 9,250 tokens/s. The figure’s displayed 3,244 and 1,030 values are scenario labels for a different simplified output-only comparison; do not mix the grid shortcut with the phase-aware request formula. A figure is a compact aid, not a substitute for the written assumptions.

## 8. Utilization, reservation, and the empty hour

A GPU that is powered on but has no request is still part of the service’s cost. Distinguish three clocks:

1. **Billed time:** the provider or depreciation clock.
2. **Busy time:** time spent executing requests, including queue-serving work that may later miss an SLO.
3. **Useful time:** busy time that produces accepted, SLO-qualified tokens.

If a fleet keeps one GPU for 720 hours in a month at \$5.191 per hour, the bill is:

$$
720 × 5.191 = \$3{,}737.52.
$$

If it processes 400 million useful output tokens, the observed cost is \$3,737.52 / 400 = \$9.3438 per million. If the same GPU could process 800 million useful tokens, cost falls to \$4.6719 per million without a hardware change. This is why autoscaling and workload shaping can be cost features even when they do not change tok/s.

The utilization-adjusted identity is:

$$
C_{1M,\text{calendar}} = \frac{3{,}600p_g}{u r}.
$$

For \$5.191 per hour, 1,000 tok/s while busy, and utilization 0.60:

$$
3{,}600 × 5.191 / (0.60 × 1{,}000) = \$31.146 \text{ per million tokens}.
$$

At 0.20 utilization it is \$93.438 per million. The GPU did not become 3× slower; the service used 3× more calendar capacity per useful token.

Reserved capacity can be rational when it protects a latency SLO, but reserve should be priced explicitly. If a fleet keeps one warm GPU for a peak that arrives 20% of the day, the idle 80% is not “free headroom.” It is a reliability cost. If the API provides burst capacity at a higher token rate, the API may win for bursty traffic even when steady self-hosting wins at 24-hour occupancy.

## 9. Cost-qualified goodput

Raw output tok/s counts tokens that may have waited too long, belonged to cancelled requests, or were generated during a degraded mode. Define a request-level qualification predicate:

$$
Q_j = \mathbf{1}[\text{TTFT}_j \leq S_{\text{TTFT}} \land \text{TPOT}_j \leq S_{\text{TPOT}}].
$$

For a window with output token count $o_j$ per request, goodput is:

$$
G = \frac{\sum_j Q_j o_j}{T_{\text{window}}}.
$$

The goodput-priced unit is $3{,}600p_g/G$. If raw output is 2,000 tok/s but only 75% of output tokens belong to qualified requests, goodput is 1,500 tok/s. At \$5.191 per hour, raw cost is \$9.3438 per million, while qualified cost is \$12.4584 per million. The difference is not a bookkeeping nuisance; it is the price of the SLO.

![The cost diagnosis branches from workload shape to the correct measurement and lever](/imgs/blogs/the-cost-model-of-inference-dollars-per-million-tokens-7.webp)

The tree is a decision aid. Long inputs point toward prefill rate, TTFT, prefix-cache hit rate, and chunking. Long outputs point toward decode TPOT, weight traffic, batching, and quantization. Bursty traffic points toward occupancy, cold starts, scale-to-zero, and whether an API’s elasticity is worth its token rate.

## 10. A reproducible cost benchmark for `nanoserve`

Do not enter a price into a spreadsheet until the rate has a provenance. The following script is intentionally small and CPU-runnable: it converts a JSONL request log into phase totals and cost. It does not pretend to measure GPU performance. The engine or load generator must write `input_tokens`, `output_tokens`, `prefill_seconds`, `decode_seconds`, and whether the request met the SLO.

```python
from __future__ import annotations
import json
import sys
from pathlib import Path

GPU_HOUR_USD = 5.191  # cited AWS H100 example; replace with your invoice rate
SLO_TTFT_MS = 500.0
SLO_TPOT_MS = 80.0

def read_rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)

def summarize(rows):
    rows = list(rows)
    wall_seconds = sum(r["prefill_seconds"] + r["decode_seconds"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    good_output = sum(
        r["output_tokens"] for r in rows
        if r["ttft_ms"] <= SLO_TTFT_MS and r["tpot_ms"] <= SLO_TPOT_MS
    )
    cost = wall_seconds / 3600.0 * GPU_HOUR_USD
    def price(tokens):
        return cost / tokens * 1_000_000 if tokens else float("inf")
    return {"requests": len(rows), "output_tokens": output_tokens,
            "good_output_tokens": good_output, "busy_hours": wall_seconds / 3600,
            "gpu_cost_usd": cost, "usd_per_1m_output": price(output_tokens),
            "usd_per_1m_good_output": price(good_output)}

if __name__ == "__main__":
    print(json.dumps(summarize(read_rows(Path(sys.argv[1]))), indent=2))
```

The output is reader-reproducible: provide a log, replace the GPU rate with the invoice rate, and inspect the JSON. The expected range is not a universal tok/s promise. On an RTX 4090, Llama-3.1-8B in BF16, fixed context, and a warmed implementation, readers should expect aggregate decode throughput to rise from batch 1, flatten at a workload-dependent middle batch, and then encounter memory or latency limits. Run the same prompt suite from [the single-4090 experiment](/blog/machine-learning/inference-engineering/experiment-llama-3-8b-on-a-single-4090) and report the observed range with driver, CUDA, dtype, and engine commit.

The measurement protocol matters more than the final division:

1. Load weights once and exclude load time from steady-state throughput.
2. Warm up enough steps to populate CUDA graphs, caches, and allocator pools when the engine uses them.
3. Synchronize before and after a timed GPU region. CUDA launches are asynchronous, so host wall time around a launch is not kernel time.
4. Use CUDA events for device elapsed time and retain request timestamps for TTFT, TPOT, queue time, and completion time.
5. Sweep input length, output length, concurrency, and batch rather than reporting one favorable point.
6. Record rejected, cancelled, preempted, and SLO-failing requests; they consume capacity even when they disappear from a successful-token counter.
7. Record clocks, driver, CUDA, model revision, dtype, quantization, tokenizer, and prompt distribution.

For an engine-side output record, add a stable request identifier and phase counters:

```python
def cost_record(request_id, input_ids, output_ids, t_submit, t_first, t_done,
                prefill_s, decode_s, tpot_ms, gpu_hour_usd):
    output_tokens = len(output_ids)
    return {
        "request_id": request_id,
        "input_tokens": len(input_ids),
        "output_tokens": output_tokens,
        "ttft_ms": 1000.0 * (t_first - t_submit),
        "tpot_ms": tpot_ms,
        "completion_seconds": t_done - t_submit,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "gpu_hour_usd": gpu_hour_usd,
    }
```

This record separates compute time from queue time. If a service has 30 ms of prefill but 500 ms of TTFT, the cost model should not blame the prefill kernel for 470 ms of admission delay. Conversely, if decode time doubles while queue time is stable, batching, KV traffic, kernel choice, or model configuration deserves investigation.

## 11. The `nanoserve` cost model

The implementation belongs beside the scheduler because the scheduler knows active sequences, input tokens admitted, output tokens produced, cache hits, and whether a request left the SLO envelope. A minimal pure-Python model is useful before wiring GPU telemetry:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CostAssumptions:
    gpu_hour_usd: float
    utilization: float
    goodput_fraction: float

    def __post_init__(self):
        if self.gpu_hour_usd < 0:
            raise ValueError("gpu_hour_usd must be non-negative")
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        if not 0 < self.goodput_fraction <= 1:
            raise ValueError("goodput_fraction must be in (0, 1]")

def dollars_per_million(tokens_per_second, assumptions):
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    effective = tokens_per_second * assumptions.utilization
    effective *= assumptions.goodput_fraction
    return 3600.0 * assumptions.gpu_hour_usd / effective

a = CostAssumptions(gpu_hour_usd=5.191, utilization=0.60, goodput_fraction=0.85)
print(round(dollars_per_million(1000, a), 4))
```

The expected output is derived, not measured: $3{,}600 × 5.191 / (1{,}000 × 0.60 × 0.85) = \$36.6424 per million qualified output tokens. A real implementation should make `utilization` a windowed statistic and derive `goodput_fraction` from request records rather than accept them as arbitrary knobs.

An input-output model is only a few extra lines:

```python
def request_cost(input_tokens, output_tokens, prefill_tok_s,
                 decode_tok_s, gpu_hour_usd):
    if min(input_tokens, output_tokens, prefill_tok_s, decode_tok_s) < 0:
        raise ValueError("counts and rates cannot be negative")
    if prefill_tok_s == 0 or decode_tok_s == 0:
        raise ValueError("phase rates must be positive")
    seconds = input_tokens / prefill_tok_s + output_tokens / decode_tok_s
    return gpu_hour_usd * seconds / 3600.0

print(request_cost(2000, 512, 10000, 1000, 5.191))
```

The reader should see approximately 0.0010267, the derived dollar cost for the assumptions in section 3. The script does not include queue time because queue time is a service-level and capacity-allocation cost, not a GPU execution estimate. Add a second accounting path if the product wants calendar cost per request.

### Accounting boundaries you should write down

The cost identity has a boundary. It prices accelerator time allocated to inference, not every consequence of running a product. A fair comparison needs a small ledger beside the formula. Include the GPU or instance charge, host CPU and memory if they are billed separately, storage for weights and logs, network transfer, orchestration, observability, deployment engineering, on-call, redundancy, and the opportunity cost of reserved capacity. Do not hide these in a vague “overhead factor” unless the factor has a documented origin.

The ledger also needs a policy for replicas. If two replicas are required for availability and only one is serving the current traffic, the cost per useful token includes both replicas’ calendar hours. If a standby is powered off and can cold-start within the SLO, its cost may instead appear as a startup penalty and a lower utilization. There is no universal correct choice; there is only an assumption that must be visible.

A second boundary is quality. A cheaper Llama-3.1-8B deployment is not a substitute for a more capable API model merely because both produce 1,000 tokens per second. First filter candidates by task success, refusal behavior, tool-call correctness, context limits, and data policy. Only then compare the cost of a qualified token. A third boundary is unit semantics: “output token” may mean generated token before stop trimming, streamed token after detokenization, or billable token after provider accounting. Pick one and keep it consistent.

Finally, cache hits change what a token means. A prefix-cache hit can avoid prefill computation while the API may charge cached input at a different rate, or may not expose the same cache semantics at all. Report input tokens presented, input tokens recomputed, output tokens generated, and billable tokens separately. The cost model can then answer both questions: what did the customer pay for, and what work did the GPU perform?

## 12. What the vLLM benchmark target changes

The production engine’s job is to increase the useful denominator without violating the latency contract. vLLM’s published architecture describes paged KV blocks, a free-block pool, request-to-block mappings, prefix caching, chunked prefill, and continuous batching. Its September 5, 2025 anatomy article lists a default block size of 16 tokens and gives the per-block formula as two tensors times block size times KV heads times head size times dtype bytes. Those mechanisms matter to cost because they change how many requests can coexist and how much prefill is repeated.

Our `nanoserve` cost model deliberately does not reproduce all of them. It accepts their output as counters: active sequences, tokens admitted, tokens generated, cache hits, preemptions, and elapsed GPU time. The difference is useful. A cost model should explain the bill even when the engine implementation changes. If every accounting formula is entangled with one scheduler’s internals, a migration to vLLM, SGLang, TensorRT-LLM, or a provider API invalidates the spreadsheet.

The vLLM team also reports, in its [Blackwell InferenceMAX post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) dated October 9, 2025, up to 4× higher throughput at similar latency versus Hopper in selected scenarios, including gpt-oss 120B and Llama 3.3 70B configurations. Those are cited results with specific hardware and workload context, not a number to transplant into Llama-3.1-8B on an RTX 4090. The economic lesson is portable: a throughput curve moves the denominator; the absolute rate must stay attached to its setup.

## Case studies from public numbers

### 1. The cloud H100 row

AWS’s capacity-block table is a good price source because it exposes both the eight-GPU instance rate and the per-accelerator equivalent. The arithmetic \$41.528 / 8 = \$5.191 lets a reader compare a single-GPU model to a full instance without accidentally dividing by the wrong unit. The limitation is equally important: a block rate is not necessarily the rate for an on-demand launch, a reserved commitment, or another region. Put the URL and retrieval date beside the assumption.

### 2. The hardware bandwidth ceiling

NVIDIA lists 3.35 TB/s HBM bandwidth for H100 SXM. Combining that cited specification with a derived 16 GB BF16 weight payload gives 4.78 ms as an idealized full-weight streaming time. The figure is a ceiling check, not an inference benchmark. It catches impossible spreadsheet values and encourages the engineer to ask whether a reported rate includes batching, weight reuse, quantization, or a smaller model. It does not predict kernel efficiency.

### 3. The vLLM architecture target

The vLLM anatomy post makes cost-relevant mechanisms visible: a block pool, block tables, prefix hashes, and continuous batching. The post’s metric definitions also prevent a common error: calling TTFT “latency” and then comparing it to TPOT. An API cost comparison needs both because long prompts can make TTFT expensive while long answers make TPOT and output-token cost dominant. The production target is therefore a workload curve, not one peak throughput number.

### 4. The provider price card

The official OpenAI pricing page is the right place to obtain the API side of a current break-even calculation. The page can change model prices and processing modes, so a report should pin the retrieval date and model name. In this post, the GPT-5.6 Luna rate is an arithmetic input only. The comparison is valid only after quality, context support, latency, availability, data handling, and tool behavior pass separate requirements.

## When to reach for this model

Use the cost model when:

- a team is comparing API, cloud GPU, and owned hardware for a named workload;
- a benchmark reports tok/s but omits utilization, input-output mix, or SLO qualification;
- a scheduler change improves throughput and you need to know whether it improved useful cost;
- capacity planning needs a break-even arrival rate before a purchase or commitment;
- input-heavy RAG and output-heavy reasoning need separate economics.

Do not use the simple model as the sole decision when:

- the model quality is not equivalent;
- the service requires multi-GPU communication and you priced only one accelerator;
- the traffic has severe burstiness and the API’s elasticity is part of its value;
- prompt caching, speculative decoding, batching, or preemption changes phase rates as load changes;
- non-GPU costs dominate, including egress, storage, support, on-call, compliance, or staff time.

When those conditions apply, keep the identity but expand the ledger. Add host cost, replicas, redundancy, queueing, cache-hit state, and the cost of failed or retried work. Then run a load test with the same arrival process the product expects. A closed-loop benchmark where the next request waits for the previous response measures a different system from an open-loop arrival stream.

## Key takeaways

1. Convert GPU-hours to dollars per million tokens with $3{,}600p_g/r$; the rate must be named as input, output, or blended tokens.
2. Multiply the denominator by utilization and goodput when pricing calendar capacity or SLO-qualified output.
3. Prefill is paid once per input; decode repeats once per generated token. A single average tok/s hides that asymmetry.
4. Batch economics is $B/t_B$, constrained by TTFT, TPOT, KV memory, and queueing.
5. Use hardware specifications for ceilings and sanity checks, never as a fabricated benchmark.
6. Every result table needs a source: derived arithmetic, a linked cited result with setup, or a reader-reproducible script with an expected range.
7. API break-even is a workload equation. Change input-output mix, utilization, model, or price and the answer changes.
8. Goodput is the right denominator when requests that miss the SLO do not count as useful service.
9. `nanoserve` should record phase times and request outcomes, then let the cost layer price the trace.
10. Use vLLM or another production engine when its scheduler, cache, reliability, and model coverage are worth more than learning the mechanism yourself.

## Further reading

- [vLLM: Inside a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — block tables, prefix caching, scheduler metrics, published September 5, 2025.
- [NVIDIA H100 specifications](https://www.nvidia.com/en-sg/data-center/h100/) — cited H100 SXM memory and bandwidth, accessed August 3, 2026.
- [AWS EC2 Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/) — cited H100 capacity-block rates, accessed August 3, 2026.
- [OpenAI API pricing](https://openai.com/api/pricing/) — current API input and output rates, accessed August 3, 2026.
- [The scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) — admission, goodput, and the latency ceiling.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) — reproducible measurement and provenance.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the series capstone.
