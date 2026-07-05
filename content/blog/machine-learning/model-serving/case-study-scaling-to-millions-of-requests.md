---
title: "Scaling to Millions of Requests: A War-Story Case Study"
date: "2026-07-05"
publishDate: "2026-07-05"
description: "One LLM service, nine scaling walls, from a single-GPU Flask prototype to 30 million requests a day — the symptom, the governing equation, and the one config diff that broke each ceiling."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "llm-serving",
    "vllm",
    "scaling",
    "kv-cache",
    "tensor-parallelism",
    "prefill-decode-disaggregation",
    "cost-optimization",
    "case-study",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/case-study-scaling-to-millions-of-requests-1.webp"
---

The demo worked. That was the problem.

A team had wrapped a fine-tuned 8B chat model in forty lines of Flask, called Hugging Face `model.generate()` inside the request handler, and shipped it to a single A100. It answered questions. Leadership loved it. Traffic was a trickle — a few thousand requests a day from an internal beta — and the p99 latency looked fine because there was never more than one person typing at once. Then the feature got a launch date, a marketing push, and a projected load of "a few million requests a day, growing." The same forty lines of Flask, unchanged, would have fallen over at roughly fifty thousand requests a day. Everything that happened next was the story of finding out *why*, one wall at a time.

This post is that story told as a sequence of scaling walls. At each wall, three things happen: a symptom appears at a specific scale, a diagnosis names the bottleneck with an equation, and a fix — usually one technique, sometimes one config flag — breaks the ceiling and reveals the next one. The numbers throughout are illustrative and composite: they are stitched together from public benchmarks, from the vLLM and DistServe papers, from DeepSeek's and other operators' disclosures, and from the shape of systems I have watched scale. Treat them as a plausible trajectory, not a measurement of any one company's stack. The trajectory is what matters, because the *ordering* of the walls is the real lesson. You do not disaggregate prefill and decode on day one. You earn your way there.

![Timeline of nine scaling stages from a single-GPU Flask prototype at fifty thousand requests a day to thirty million requests a day with quality gates, each stage broken by one named technique.](/imgs/blogs/case-study-scaling-to-millions-of-requests-1.webp)

Everything here is a trade on one triangle: **latency, throughput, cost**. You never get all three for free. Every wall is a moment where one corner of that triangle became the binding constraint, and every fix is a deliberate move that spent one corner to buy another. If you have not yet internalized that triangle, start with [what model serving actually is](/blog/machine-learning/model-serving/what-is-model-serving) and come back — the rest of this post assumes it. By the end you will be able to look at a serving system at any scale, name which wall it is about to hit, and know which technique buys you the next order of magnitude.

Here is the map. We climb from Stage 0 (one GPU, one request at a time) through eight walls: the throughput wall, the memory wall, the tail-latency-under-burst wall, the single-GPU-capacity wall, the cost wall, the latency-at-scale wall, the reliability wall, and the quality wall. Each has its own section. Let us start at the bottom.

## Stage 0 — The naive baseline: one GPU, one request at a time

The starting point is honest and worth respecting: it works, and it is simple. A Flask app, a model loaded once at startup, and a handler that tokenizes, generates, and returns text.

```python
# baseline_server.py — the prototype that leadership loved
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
)

@app.post("/generate")
def generate():
    prompt = request.json["prompt"]
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    # One request occupies the whole GPU for the whole generation.
    out = model.generate(**inputs, max_new_tokens=256, do_sample=True)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:])
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)  # single worker, single request in flight
```

Let us be precise about what this does at the hardware level, because the bottleneck is not obvious until you count. Text generation is autoregressive: the model produces one token, appends it to the sequence, and runs a full forward pass to produce the next. For an 8B model on an A100, a single decode step — one forward pass for one sequence — takes on the order of 20 to 25 milliseconds. That is dominated not by arithmetic but by memory bandwidth: to produce one token for one sequence, the GPU must stream all 16 GB of BF16 weights from HBM into the compute units. The A100's HBM delivers about 2 TB/s, so reading 16 GB takes roughly 8 ms of pure bandwidth time, and the rest is kernel launch overhead, attention, and sampling. The arithmetic intensity is terrible: you move 16 GB to do a few hundred megaflops of useful work. The GPU's tensor cores sit almost entirely idle.

The clean way to see this is the **roofline model**. A kernel is compute-bound if its arithmetic intensity — floating-point operations per byte of memory traffic — exceeds the ratio of the GPU's peak FLOPs to its peak bandwidth; below that ratio it is memory-bound and the tensor cores starve. An A100 does roughly 312 TFLOP/s of BF16 and moves about 2 TB/s, so its crossover intensity is about 156 FLOPs per byte. A single-sequence decode step does roughly two FLOPs per weight parameter it reads (one multiply, one add), which is an arithmetic intensity of about 2 — nearly two orders of magnitude below the crossover. That gap *is* the 8% utilization. You cannot close it by working the GPU harder on one sequence, because there is no more arithmetic to do per byte moved; you can only close it by moving those same bytes on behalf of *more sequences at once*. Hold that sentence — it is the reason every technique in the next three stages exists.

Concretely: one sequence, decoding at roughly 40 tokens per second, drives GPU compute utilization to about 8%. A 256-token response takes about 6.4 seconds of pure decode, plus prefill. Because Flask processes one request at a time in this configuration, a second concurrent user waits in line behind the first. Two users see the second one's p99 double. Ten users and the tail latency is measured in tens of seconds. The system tops out around fifty thousand requests a day before the queue behind that single serial worker grows without bound — the classic signature of an unstable queue, which we will make rigorous in Stage 3.

There is a second, subtler cost hiding in the baseline: it wastes the *prefill* phase too. Prefill is compute-bound and parallel — it can saturate the tensor cores on a single request — but Flask runs it serially, so while one request prefills a long prompt, no other request can decode, and vice versa. The two phases have complementary appetites (prefill wants FLOPs, decode wants bandwidth) and the naive server lets neither help the other. Interleaving them is worth throughput on its own, and it is exactly what the continuous-batching scheduler does for free once you switch engines.

The diagnosis is stark and it is the thesis of the whole first half of this post: **a single decoding sequence cannot saturate a modern GPU, because decode is memory-bound, not compute-bound.** You are paying for a Ferrari and driving it in first gear. The fix is not a faster GPU. It is putting more sequences through the same forward pass. That is the throughput wall.

## Stage 1 — The throughput wall: continuous batching and PagedAttention

The symptom at fifty thousand requests a day is a queue that never drains. Add users and latency climbs linearly because requests are serialized. The instinct is to add GPUs, and you can — but at 8% utilization you would be buying eleven times more hardware than the work requires. The right move is to make the one GPU you have do eleven times more work first.

The mechanism is **batching**, and specifically **continuous batching** (also called iteration-level or in-flight batching). The insight that makes it work is the same one that made the baseline slow: since decode is memory-bound, the weights are already being streamed from HBM on every step. If you process 40 sequences in that same forward pass instead of 1, you stream the weights *once* and amortize them across all 40. The arithmetic goes up 40x; the memory traffic for weights stays roughly flat. You convert an idle memory-bound kernel into a busy compute-bound one, and throughput rises almost linearly with batch size until the tensor cores finally become the constraint.

### The mechanics: Little's Law sets the ceiling

The governing law here is **Little's Law**, the most useful equation in all of serving. For any stable system,

$$L = \lambda \cdot W$$

where $L$ is the average number of requests resident in the system, $\lambda$ is the arrival rate, and $W$ is the average time each request spends inside. Rearranged, the maximum sustainable arrival rate is

$$\lambda_{\max} = \frac{L_{\max}}{W}$$

For an LLM server, $L_{\max}$ is the largest batch of sequences you can keep in flight at once, and $W$ is the average end-to-end generation time. Serial Flask pins $L_{\max}=1$: with $W \approx 6.5$ s per request, $\lambda_{\max} \approx 0.15$ req/s, about thirteen thousand requests a day of *steady* capacity (bursts eat the rest). Continuous batching raises $L_{\max}$ to the number of sequences the GPU's memory and compute can co-resident — call it 48 — and the same equation gives roughly 7 req/s, comfortably into the hundreds of thousands per day. The technique did not make any single request faster. It made the *system* hold more of them at once. That is the trade: identical per-request latency, dramatically higher throughput.

Continuous batching adds one more trick over naive static batching. Static batching waits for a full batch, runs it to completion, then starts the next — so a batch is only as fast as its slowest, longest-generating member, and short requests wait for long ones. Continuous batching instead makes scheduling decisions *every decode step*: the moment any sequence emits its end-of-sequence token, it leaves the batch and a waiting request takes its slot on the very next iteration. No sequence waits for another to finish. This is the algorithm vLLM's scheduler runs, and it is why continuous batching beats static batching by another large factor on real, ragged traffic.

It helps to name the three regimes precisely, because teams conflate them and then wonder why their "batching" barely helped. *Static batching* groups requests that arrive together and is what a naive `generate()` on a padded tensor does — it helps only if requests are homogeneous and arrive in lockstep, which production traffic never is. *Dynamic batching* (Triton's `dynamic_batching`, TorchServe's batching) waits a small window to gather arrivals into one batch before running it — better, but still batch-at-a-time, so a long generation blocks the batch behind it. *Continuous* (iteration-level) batching is the one that matters for LLMs, because generation length is unbounded and wildly variable: only by re-forming the batch every step can you keep the GPU full without long requests holding short ones hostage. The scheduler enforces this with a *token budget* — a cap on total tokens (prefill plus decode) admitted into any one forward pass — and greedily fills that budget each iteration from the waiting queue and the running set. That token budget is the single knob that trades throughput against tail latency, and it reappears at every later wall.

The second half of vLLM's throughput win is **PagedAttention**, which solves a memory problem that would otherwise cap your batch far below what compute allows. We give it its own section next, because it *is* the memory wall. For now, the switch itself:

```python
# stage1_vllm.py — the continuous-batching switch
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server  # OpenAI-compatible

# One engine, one GPU — but iteration-level batching underneath.
llm_args = dict(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,   # give the KV cache 90% of the GPU
    max_num_seqs=256,              # upper bound on in-flight sequences
    max_model_len=8192,
)
# In practice you launch the server, not the library, so the scheduler runs:
#   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
#     --dtype bfloat16 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 256 --max-model-len 8192
```

![Before-and-after comparison of serial Flask generation versus vLLM continuous batching, showing GPU utilization rising from eight to seventy-one percent and throughput multiplying fifty-seven times on the same hardware.](/imgs/blogs/case-study-scaling-to-millions-of-requests-2.webp)

The figure above is the before-and-after that matters most in the whole journey, because it is the cheapest win: no new hardware, one framework swap. On the left, the serial baseline idles at 8% utilization and roughly 40 tokens per second. On the right, with dozens of sequences in flight, the same A100 runs at about 71% utilization and roughly 2,300 tokens per second aggregate. That is the move from a memory-bound single stream to a nearly compute-bound batch.

#### Worked example: the throughput math before and after batching

Take a concrete decode step for the 8B model on one A100. Streaming 16 GB of weights at 2 TB/s costs about 8 ms; attention over the KV cache and kernel overhead push a single-sequence step to about 24 ms, giving roughly 42 tokens per second for one sequence. Now batch 48 sequences. The weight read is shared, so it still costs about 8 ms. The per-sequence attention and the sampling work grow with the batch, and the tensor cores now have real work, so the step time rises — but only to about 21 ms per step for the whole batch of 48, because most of that step was already bandwidth-bound and is now overlapped with compute. Aggregate throughput is $48 \text{ seqs} \times (1 \text{ token} / 0.021 \text{ s}) \approx 2{,}290$ tokens per second. Against the 42 tokens per second baseline, that is about **55x** more throughput from the same silicon. The per-token latency each user experiences barely changed; the number of users the box can serve concurrently went up by a factor of nearly sixty. Daily capacity climbs from fifty thousand to about four hundred thousand requests. New ceiling reached; new wall ahead.

The reasoning here is developed in full in [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention); if you serve LLMs and remember one technique from this entire post, make it this one. It is the single highest-leverage change available, and it is nearly free.

## Stage 2 — The memory wall: FP8 quantization and KV-cache management

Continuous batching promised that throughput rises with batch size until compute saturates. In practice you hit a different ceiling first, and it is made of memory. At four hundred thousand requests a day the traffic is bursty; during peaks the scheduler wants to admit 90 or 100 concurrent sequences, but the engine starts *preempting* — evicting half-finished sequences to disk or recomputing them — because there is nowhere to put their KV cache. The symptom is a throughput that plateaus well below the compute limit and a p99 that spikes whenever the cache fills. The GPU is not compute-starved; it is out of room.

### The mechanics: the KV-cache memory equation

Every token a model attends to must have its key and value vectors kept in memory for the rest of the sequence — that is the **KV cache**, and it is the dominant consumer of GPU memory in LLM serving after the weights themselves. The bytes it costs per token are exactly:

$$\text{bytes/token} = 2 \cdot L \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot b$$

The factor of 2 is for keys and values; $L$ is the number of transformer layers; $n_{\text{kv}}$ is the number of key/value heads (smaller than the number of query heads under grouped-query attention); $d_{\text{head}}$ is the per-head dimension; and $b$ is bytes per element. For Llama-3-8B — $L=32$, $n_{\text{kv}}=8$, $d_{\text{head}}=128$, and $b=2$ for BF16 — that is $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes, or 128 KiB per token. A single 2,048-token conversation therefore holds about 256 MB of KV cache. On a 40 GB A100 with 16 GB of BF16 weights, roughly 24 GB is left, so the concurrent-sequence budget is about $24 \text{ GB} / 256 \text{ MB} \approx 90$ sequences at 2K context. That is the wall: not compute, but the arithmetic of how many conversations' worth of keys and values fit alongside the weights.

The $n_{\text{kv}}$ term in that equation is worth a beat, because it is the single biggest architectural lever on KV size and it is already baked into modern models. Original multi-head attention sets $n_{\text{kv}}$ equal to the number of query heads — for an 8B model that would be 32 KV heads, quadrupling the cache. Grouped-query attention (GQA) shares each key/value head across a group of query heads, dropping $n_{\text{kv}}$ to 8 here; multi-query attention (MQA) takes it to the limit of 1. Llama-3, Mistral, and essentially every recent model ship GQA precisely because the KV cache, not the parameter count, is what bounds serving batch size. When you pick a model to serve, its $n_{\text{kv}}$ is a serving-cost decision as much as a quality one — a model with 4x the KV heads needs 4x the memory per token and serves a quarter the concurrency on the same GPU.

Two techniques attack it. The first, **PagedAttention**, attacks *waste*. Classical serving reserves a contiguous KV region sized to `max_model_len` for every sequence, so a request that generates 200 tokens but could have generated 8,192 ties up the full reservation — internal fragmentation that can waste more than half the cache. PagedAttention borrows the operating system's idea of paging: it chops the KV cache into fixed 16-token blocks and hands them out on demand, so a sequence holds only the blocks it actually uses, plus at most one partially filled block. Fragmentation drops from tens of percent to under 4%, and the effective batch you can hold roughly doubles or triples on the same memory. This is on by default in vLLM; the deep mechanics are in [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).

The second technique attacks the *weights*, and it is the one that broke this wall for the team in the story: **FP8 quantization**. Store the weights in 8-bit floating point instead of 16-bit and the weight footprint halves, from 16 GB to 8 GB. On an H100 (or with Marlin/Machete kernels on Ada and Hopper) FP8 also runs the matmuls faster. Crucially, the 8 GB you free does not vanish — it becomes KV headroom. Accuracy loss for FP8 on an 8B model is typically under a tenth of a percent of perplexity, well inside the noise of a chat product.

```python
# stage2_fp8.py — FP8 weights + KV budget on one A100
#   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
#     --quantization fp8 \                 # 16 GB -> 8 GB weights
#     --kv-cache-dtype fp8_e5m2 \          # optional: halve KV too
#     --gpu-memory-utilization 0.92 \
#     --max-num-seqs 256 \
#     --enable-chunked-prefill \           # (Stage 3 — foreshadowing)
#     --max-model-len 8192
from vllm import EngineArgs
args = EngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="fp8",
    kv_cache_dtype="fp8_e5m2",
    gpu_memory_utilization=0.92,
    max_num_seqs=256,
)
```

![Grid comparing GPU memory budget in BF16 versus FP8 on a forty-gigabyte A100, showing weights halving from sixteen to eight gigabytes, KV budget rising, concurrent sequences going from ninety to a hundred twenty-five, and throughput increasing one point eight times.](/imgs/blogs/case-study-scaling-to-millions-of-requests-3.webp)

The grid above tells the whole story of this wall in three rows. Halving the weights frees 8 GB; that 8 GB becomes KV cache; the concurrent-sequence budget rises from about 90 to about 125; and because more sequences share each weight read, aggregate throughput rises from roughly 2,300 to roughly 4,300 tokens per second on the same A100. You can push further by quantizing the KV cache itself to FP8, which halves the per-token cost from 128 KiB to 64 KiB and roughly doubles the sequence budget again — at the price of a small, measurable quality cost that you must gate (Stage 8). The choice among GPTQ, AWQ, FP8, and SmoothQuant, and when each pays, is its own decision covered in [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving); FP8 is the right default on Hopper hardware.

#### Worked example: the concurrent-sequence budget

Suppose your average conversation is 3,000 tokens of context by the time it finishes generating. At 128 KiB/token that is 384 MB of KV per sequence. In BF16, with 24 GB of KV budget, you hold $24{,}576 / 384 \approx 64$ sequences. Switch weights to FP8: budget rises to 32 GB, giving $32{,}768 / 384 \approx 85$ sequences. Now also quantize the KV cache to FP8, halving per-sequence cost to 192 MB: $32{,}768 / 192 \approx 170$ sequences. You have roughly tripled concurrency — and therefore throughput, and therefore requests per dollar — without touching latency or buying a single new GPU. Daily capacity climbs to about nine hundred thousand requests. The catch, and the reason Stage 8 exists, is that KV quantization is the first change so far that can *silently* degrade output quality. Everything up to here was free; this one has to be measured.

## Stage 3 — The tail-latency-under-burst wall: chunked prefill and admission control

Now the service is near a million requests a day and mostly healthy — until a burst hits. Marketing tweets a link, or a batch job fires ten thousand requests in a minute, and the p99 **time to first token** (TTFT — the delay from request arrival to the first streamed token) jumps from a comfortable 300 ms to eight seconds, even though average throughput looks fine. Two distinct failure modes are hiding inside that one symptom, and they need two different fixes.

The first failure mode is **head-of-line blocking from prefill**. Generation has two phases with opposite cost profiles. *Prefill* processes the entire prompt in one big parallel forward pass — it is compute-heavy and its cost grows with prompt length. *Decode* produces one token per step and is memory-bound, as we established. When a request arrives with a 6,000-token prompt, its prefill can occupy the GPU for a long, indivisible chunk, and every other user's decode steps stall behind it. One long prompt spikes everyone's TTFT. The fix is **chunked prefill**: split that 6,000-token prefill into, say, twelve 512-token chunks and interleave them with ongoing decode steps in the same batch. No single scheduling iteration is monopolized by one giant prefill, so decode keeps flowing and TTFT stays bounded. It costs a hair of raw throughput — you give up a little prefill efficiency — to buy a dramatically tighter tail. In vLLM it is one flag, `--enable-chunked-prefill`, which is why it appeared in the Stage 2 config already.

The second failure mode is **an unbounded queue**. This is where Little's Law becomes a threat instead of a tool. If the arrival rate $\lambda$ exceeds the service rate $\mu$ even briefly, queueing theory is unambiguous about what happens to waiting time. For an M/M/1 queue with utilization $\rho = \lambda/\mu$, the expected time in system is

$$W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)}$$

As $\rho \to 1$, $W \to \infty$. The knee is vicious: at $\rho = 0.9$, waiting time is already ten times the service time; at $\rho = 0.99$, a hundred times. A burst that drives you past $\rho = 1$ for thirty seconds produces a queue that takes minutes to drain, and every request in it blows its SLA. You cannot batch your way out of this, because the problem is that you accepted more work than you can do. The only correct response is **admission control**: cap the queue depth and, when it is full, reject new work fast with an HTTP 429 rather than accepting it into a queue it will die waiting in. A fast rejection that a client can retry with backoff is a far better experience than a request that hangs for two minutes and then times out.

There is a failure mode that makes this worse than the raw queueing math suggests: the **retry storm**. When requests start timing out, clients retry — and naive clients retry immediately, so a system already at $\rho > 1$ suddenly sees its effective arrival rate *double* from retries of work it has not finished. This is a positive feedback loop that turns a brief overload into a sustained outage; it is the mechanism behind a surprising number of "the whole service fell over for an hour after a thirty-second blip" postmortems. Admission control breaks the loop at the entry point, but the client side matters too: retries must use exponential backoff with jitter, and ideally a circuit breaker that stops retrying entirely once failures cross a threshold. Distinguish the two entry-point defenses, because teams conflate them. **Rate limiting** is a policy tool — it enforces per-tenant fairness and quota (token buckets, weighted fair queuing) regardless of system load. **Load shedding** is a survival tool — it drops work only when the system itself is saturated, based on a real-time signal like queue depth. You want both: rate limiting so one abusive tenant cannot starve the rest, and load shedding so a legitimate global burst cannot drive everyone past the queueing knee. The admission queue in the figure below is where load shedding lives; the gateway above it is where rate limiting lives.

![Graph of a request path with admission control and chunked prefill: an API gateway feeds a depth-capped admission queue that either sheds excess load with HTTP 429 or admits requests to a scheduler, which interleaves chunked prefill and decode steps before streaming tokens at a held tail latency.](/imgs/blogs/case-study-scaling-to-millions-of-requests-4.webp)

The path above shows both fixes working together. The gateway feeds a bounded admission queue; overflow is shed immediately as 429; admitted work flows to a scheduler that interleaves 512-token prefill chunks with one-token-per-iteration decode; and the result is a p99 TTFT held at roughly 340 ms even under a 5,000 RPS burst. The third leg is **autoscaling**: admission control protects you in the seconds before more capacity arrives, and a horizontal autoscaler that watches queue depth or RPS (not GPU utilization, which is a lagging and misleading signal for LLMs) spins up replicas to raise $\mu$ so you are not shedding for long. A Kubernetes HPA driven by a custom `queue_depth` metric, or a KEDA `ScaledObject` on a Prometheus query, is the standard shape here.

```yaml
# stage3_hpa.yaml — scale on queue depth, not GPU util
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-8b
  minReplicas: 2
  maxReplicas: 16
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting   # queue depth per replica
        target:
          type: AverageValue
          averageValue: "8"                 # scale out above 8 queued/replica
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30        # react fast to bursts
      policies:
        - type: Pods
          value: 4
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300       # scale in slowly to avoid flapping
```

The lesson of this wall is that throughput and tail latency are different problems that fail at different times. Continuous batching gave you throughput; it did nothing for the tail under burst. Chunked prefill and admission control are what hold the tail. Together they push daily capacity past two million requests while keeping p99 TTFT under half a second. The full treatment of scheduling, preemption, and SLO-aware admission lives in [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management).

## Stage 4 — The single-GPU-capacity wall: tensor parallelism, then multi-node

Two forces converge to push the service off a single GPU. The first is simply traffic: at two million requests a day, one A100 replica plus autoscaling copies of it works, but the model has also become the product's bottleneck on *quality*. Users want better answers, and the 8B model, however well tuned, has a ceiling. The team decides to upgrade to a 70B model. And a 70B model in BF16 needs about 140 GB just for weights — it does not fit on one 80 GB GPU at all, let alone with room for a KV cache. This is the capacity wall, and the answer is **model parallelism**: split one model across multiple GPUs.

### The mechanics: the interconnect-bandwidth argument

The first tool is **tensor parallelism** (TP): shard each weight matrix across $N$ GPUs so each holds $1/N$ of every layer, run the matmuls in parallel, and combine partial results with an **all-reduce** collective every layer. TP works beautifully — *if the GPUs can talk fast enough*. The all-reduce moves the layer's activations across the interconnect twice per layer, and it is on the critical path of every single forward pass. Inside one node, GPUs are connected by NVLink at roughly 900 GB/s; the all-reduce cost is a few milliseconds and TP=4 or TP=8 pays for itself immediately. The moment you cross a *node boundary*, the interconnect drops to InfiniBand at roughly 400 Gb/s — about 50 GB/s, nearly twenty times slower than NVLink. Now the all-reduce dominates, and adding GPUs can make you *slower*.

Put numbers on it. A ring all-reduce of a message of size $M$ bytes across $N$ GPUs moves about $2M(N-1)/N$ bytes per GPU over the link, so its time is roughly that divided by the link bandwidth $B$. For one transformer layer's activations at batch $b$, sequence position, and hidden size $h$ in BF16, $M \approx 2bh$ bytes, and there are two all-reduces per layer times $L$ layers on the critical path. With TP=4 on NVLink at 900 GB/s the per-layer collective is on the order of microseconds and sums to a few milliseconds across all layers — negligible against the tens of milliseconds of useful compute. Run that same TP=4 *across* a node boundary on 50 GB/s InfiniBand and every collective is roughly eighteen times slower; the all-reduces now cost more than the compute they coordinate, and your "4 GPUs" deliver less throughput than one. That single ratio — NVLink bandwidth over InfiniBand bandwidth — is why the rule below is close to a law rather than a heuristic.

The rule that falls out is: **keep tensor parallelism inside the NVLink domain, and use pipeline parallelism across nodes.** Pipeline parallelism (PP) splits the model by *layers* — GPUs 0–3 hold the first half of the layers, GPUs 4–7 the second half — and passes activations forward only at the pipeline boundary, a tiny transfer compared to TP's per-layer all-reduce. PP tolerates the slow inter-node link because it barely uses it, at the cost of a "bubble" of idle time while the pipeline fills and drains. For a 70B model you typically run TP=4 or TP=8 within a node and, only if the model still does not fit or you need more aggregate memory, add PP across nodes. Mixture-of-experts models like DeepSeek-V3 add a third axis, **expert parallelism** (EP), sharding the experts themselves across GPUs. The full derivation of when each axis pays is in [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving); the [multi-node 100B-plus serving guide](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) covers the cluster mechanics.

![Matrix of parallelism strategies for the capacity wall, rating single-GPU, tensor-parallel, tensor-plus-pipeline, and tensor-plus-expert configurations by whether the model fits, the interconnect needed, the latency added, and when to use each.](/imgs/blogs/case-study-scaling-to-millions-of-requests-5.webp)

The matrix above is the decision table. Single GPU: nothing beats it when the model fits, because there is zero communication overhead. TP=4 inside a node: the 70B fits, NVLink keeps the all-reduce cheap (about 8 ms added), use it. TP=4 plus PP=2 across two nodes: needed when one node is not enough, but InfiniBand adds a pipeline bubble of roughly 15 ms — pay it only when forced. TP=8 plus EP for a 671B MoE: the frontier case, worth it only above a couple hundred billion parameters. The launch is a single command; vLLM handles the sharding and the collectives:

```bash
# stage4_tp.sh — 70B on one 4xH100 node with tensor parallelism
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \          # shard every layer across 4 GPUs, NVLink all-reduce
  --quantization fp8 \                # 140 GB BF16 -> ~70 GB FP8 -> fits with KV headroom
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 128

# If a single node is not enough (e.g. serving 405B), add pipeline parallelism
# across nodes with Ray as the distributed backend:
#   ray start --head                                   # on node 0
#   ray start --address=<head>:6379                    # on node 1
#   vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct \
#     --tensor-parallel-size 8 --pipeline-parallel-size 2 \
#     --quantization fp8 --distributed-executor-backend ray
```

#### Worked example: when a second node stops helping

You are serving the 70B at TP=4 on one node and want more throughput, so you consider TP=8 across two nodes. Naively that doubles the compute, so you expect roughly 2x. Now account for the all-reduce. At TP=4 on NVLink the collectives cost, say, 3 ms across all 80 layers against 30 ms of compute — a 10% tax, so effective throughput is 0.9 of the compute peak. Cross to TP=8 spanning the InfiniBand link and the collective volume grows while the bandwidth drops roughly eighteenfold; the collectives now cost on the order of 40 ms against 15 ms of compute per GPU-share, so the forward pass is *collective-bound* and each step takes about 55 ms instead of 33 ms. You added GPUs and made the step 1.7x *slower*. The eight GPUs deliver less aggregate throughput than the four did — a negative return on a hardware doubling. The fix is not more TP; it is TP=4 within each node plus data-parallel *replication* across nodes (two independent TP=4 replicas behind the router), or pipeline parallelism if a single replica genuinely cannot hold the model. This is the concrete version of the rule: parallelism that crosses the slow link must be the kind that barely uses it.

The honest cost of this wall shows up on the scorecard: moving to a 70B model raises the cost per token roughly elevenfold, from about \$0.16 to about \$1.90 per million tokens, because you are now burning four to eight H100s per replica for a model with nearly nine times the parameters. That is the trade — you spent cost to buy quality — and it is the moment the *next* wall, the cost wall, becomes the one everyone in the room is staring at. Daily capacity, meanwhile, has climbed to about five million requests as the autoscaler fans out these bigger replicas.

## Stage 5 — The cost wall: spot fleets, prefix caching, right-sizing

At five million requests a day on a 70B model, the monthly GPU bill has become a line item that finance asks about by name. Nothing is *broken* — latency is fine, throughput is fine — but the cost per token is roughly twelve times what it was two stages ago, and the business needs it back down before it scales the traffic another 6x. This is the cost wall, and unlike the earlier walls it is not one bottleneck but a portfolio of three moves, each attacking a different term in the cost equation.

### The mechanics: the cost-per-token formula

Cost per token is almost embarrassingly simple, which is why it is so useful:

$$c_{\text{token}} = \frac{P_{\text{gpu}} \cdot N_{\text{gpu}}}{\text{goodput} \cdot 3600}$$

where $P_{\text{gpu}}$ is the hourly price of one GPU, $N_{\text{gpu}}$ is how many you use, and *goodput* is the *useful* tokens per second you actually deliver to users (not peak throughput — the tokens that made it out the door within SLA). Every cost lever is an attack on one of these three terms.

**Attack $P_{\text{gpu}}$ with spot instances.** Cloud spot (preemptible) GPUs run at roughly 35–40% of on-demand price — an H100 that costs about \$4/hr on demand is about \$1.60/hr spot. The catch is that the provider can reclaim them with a two-minute warning. For *stateless* serving this is manageable: run a mixed fleet where a baseline of on-demand replicas guarantees capacity and a much larger spot fleet handles the bulk, drain a preempted replica gracefully on the termination signal (finish in-flight requests, deregister from the load balancer, let the router route around it), and let the autoscaler replace it. A 70/30 spot/on-demand split cuts the blended GPU price by roughly half.

**Attack goodput with prefix caching.** Many requests share a prefix — the same long system prompt, the same few-shot examples, the same RAG context reused across a conversation's turns. Prefill for that shared prefix is pure repeated work. **Prefix caching** (RadixAttention in SGLang, automatic prefix caching in vLLM) stores the KV cache of common prefixes and reuses it, so a request that matches a cached prefix skips prefill for those tokens entirely. On workloads with heavy prompt sharing this cuts prefill compute by 30–50%, which raises goodput for the same hardware — a direct reduction in cost per token. It is one flag, `--enable-prefix-caching`, and the mechanism is detailed in [KV-cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).

**Attack $N_{\text{gpu}}$ with right-sizing.** The 70B model is worth it for hard queries, but a large fraction of production traffic is easy — greetings, short factual lookups, follow-ups — that the 8B model answers just as well. Route by difficulty: a cheap classifier or a length/complexity heuristic sends easy queries to the 8B fleet and only the hard ones to the 70B fleet. If 60% of traffic can be served by the model that costs an eighth as much, the blended cost per token drops sharply without any user noticing. This is the serving-side twin of the [cost-optimization-at-llm-scale](/blog/machine-learning/model-serving/cost-optimization-at-llm-scale) playbook, which goes deep on batch-versus-online routing and token cost modeling.

Spot is the highest-return lever but also the one that most often gets teams into trouble, so it earns a paragraph of operational detail. The contract is that the cloud can reclaim the instance with a short warning — typically two minutes on the major clouds, delivered as a metadata signal or a Kubernetes node-termination event. Your serving process must catch that signal and *drain gracefully*: deregister from the load balancer immediately so no new requests arrive, stop admitting to the local queue, let in-flight generations finish (or checkpoint their KV to the decode pool if you have disaggregated), then exit before the hard kill. Get draining right and a reclamation is invisible to users; get it wrong and every reclamation drops a batch of in-flight requests. The fleet shape that survives this is a three-tier blend: a small floor of **reserved** instances (committed one-to-three-year capacity at the deepest discount) for guaranteed baseline, a layer of **on-demand** for predictable diurnal peaks, and a large, cheap **spot** layer for the bulk — with the autoscaler biased to add spot first and fall back to on-demand only when spot is unavailable. The blended price lands near spot's, with on-demand and reserved as the insurance that you never drop below your SLA floor when a whole spot pool evaporates at once.

Combined, these three moves took the 70B tier from about \$1.90 to about \$0.78 per million tokens — spot halved the price term, prefix caching lifted goodput by a third, and right-sizing shrank how many 70B GPUs the traffic actually needed. Daily capacity, freed from the cost constraint, grows to about twelve million requests. But squeezing cost this hard surfaced a subtler problem: at this fleet size and this level of sharing, the *tail* latency at scale started to fray, because prefill and decode were still fighting over the same GPUs. That is the next wall.

## Stage 6 — The latency-at-scale wall: prefill/decode disaggregation and cache-aware routing

By twelve million requests a day the fleet is dozens of GPUs, spot-heavy, prefix-cached, and cost-efficient. The remaining problem is the tail, and its root cause is architectural rather than a matter of tuning. Prefill and decode have *opposite* hardware profiles — prefill is compute-bound and bursty, decode is memory-bound and steady — yet in every stage so far they have run on the *same* GPUs, interleaved by the scheduler. Chunked prefill (Stage 3) softened the interference, but it did not remove it: on a busy replica, a wave of long prefills still steals cycles from decode, and decode-heavy moments leave prefill capacity idle. At small scale that averages out. At large scale, with strict p99 targets, the interference *is* the tail.

The fix is **prefill/decode disaggregation** (PD disaggregation): stop running the two phases on the same GPUs. Stand up a *prefill pool* of compute-optimized GPUs that only does prefill, and a *decode pool* of memory-optimized GPUs that only does decode. A request prefills on the prefill pool, and its resulting KV cache is transferred over a fast interconnect (NVLink within a rack, RDMA over InfiniBand across racks) to a decode-pool GPU that streams the rest of the tokens. Each pool now runs a homogeneous, predictable workload you can size and tune independently: the prefill pool saturates on compute, the decode pool packs sequences densely for memory-bound throughput, and neither one's spikes bleed into the other's tail. This is the architecture behind DistServe and Microsoft's Splitwise, and it is deployed in production by operators like Tencent and Xiaomi; the full mechanics are in [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).

The obvious objection is that you have added a network hop to the critical path: the KV cache now has to *move* between prefill and decode before the first token can stream. Whether that hop is affordable is a bandwidth calculation. The KV cache handed off is the full prompt's worth — for a 70B model with $L=80$ layers, $n_{\text{kv}}=8$ KV heads, and $d_{\text{head}}=128$ in FP8, that is $2 \cdot 80 \cdot 8 \cdot 128 \cdot 1 = 163{,}840$ bytes per token, about 160 KiB. A 2,000-token prompt hands off roughly 320 MB. Over 400 Gb/s InfiniBand with RDMA that transfer takes about 6–7 ms; over NVLink within a rack, under a millisecond. Set that against the prefill it replaces on the decode GPU — hundreds of milliseconds of compute for a long prompt — and the hop is cheap. The transfer only becomes a problem for very long contexts on a slow fabric, which is why serious disaggregated deployments insist on RDMA and co-locate the pools in the same rack when they can. The equation to keep is that the handoff cost scales with *prompt length*, while the win scales with *prefill compute avoided on the decode pool* — and for any prompt long enough to matter, the second dominates.

The second half of this stage is a smarter router. Once you have pools and a prefix cache, routing stops being "pick the least-loaded replica" and becomes **cache-aware routing**: hash the request's prefix, and route it to the pool or replica that already holds that prefix's KV cache, so the hit rate on the prefix cache stays high instead of being scattered across replicas by a naive load balancer. A cache-aware router turns a 15% incidental hit rate into a 38% deliberate one, and every hit is a prefill skipped — lower TTFT and lower cost at once. This router is the beginning of a real **control plane**, the subject of [LLM control planes: AIBrix and KServe](/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve).

![Graph of prefill/decode disaggregation with cache-aware routing: a cache-aware router checks a prefix cache and routes cache misses to a compute-bound prefill pool, whose KV cache transfers over RDMA to a memory-bound decode pool that streams tokens at a p99 time-to-first-token of two hundred forty milliseconds.](/imgs/blogs/case-study-scaling-to-millions-of-requests-6.webp)

The path above is the mature request flow. The cache-aware router hashes the prefix and does a lookup; on a hit (38% of the time) it skips prefill entirely and hands the request straight to the decode pool; on a miss it routes to the prefill pool, which does the compute-heavy prefill and ships the KV cache to the decode pool over RDMA at tens of gigabytes per second; the decode pool streams tokens over SSE. The two pools are sized independently — here 8 compute-bound H100s for prefill feeding 16 memory-bound H100s for decode — because their bottlenecks are different. The payoff is a p99 TTFT cut to about 240 ms even as aggregate traffic hits thirty million requests a day, and, because prefill and decode GPUs each run dense and homogeneous, a further drop in cost per token to about \$0.52.

In practice the disaggregated launch is two vLLM roles plus a router that knows about the split. The exact flags are still evolving across vLLM releases, so treat this as a shape rather than a copy-paste, but the structure is stable:

```bash
# stage6_disagg.sh — prefill role, decode role, cache-aware router (illustrative)
# Prefill workers: compute-optimized, produce KV and push it to the decode pool.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 --quantization fp8 \
  --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0}' \
  --enable-prefix-caching --port 8100    # prefill pool endpoint

# Decode workers: memory-optimized, consume KV and stream tokens.
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 --quantization fp8 \
  --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1}' \
  --enable-prefix-caching --port 8200    # decode pool endpoint
```

```yaml
# stage6_router.yaml — cache-aware routing policy (control-plane sketch)
router:
  prefill_endpoints: ["http://prefill-0:8100", "http://prefill-1:8100"]
  decode_endpoints:  ["http://decode-0:8200",  "http://decode-1:8200", "http://decode-2:8200"]
  routing:
    strategy: prefix-hash          # route by hashed prompt prefix, not round-robin
    prefix_block_tokens: 512       # granularity of the prefix cache key
    on_cache_hit: skip_prefill     # matched prefix -> straight to a decode worker
    decode_lb: least-outstanding   # among decode workers, pick fewest in-flight
    slo:
      ttft_ms: 300                 # priority class target; shed/deprioritize above it
```

The equation that justifies disaggregation is the one about *matching resources to bottlenecks*. Prefill throughput scales with FLOPs; decode throughput scales with memory bandwidth and KV capacity. When both run on one GPU, you buy a compromise SKU and under-utilize half of it at any instant. Split them and you buy compute-optimized GPUs for prefill and memory-optimized configurations for decode, right-sizing each to its actual constraint. This is the same "match the resource to the bottleneck" logic as Stage 4's interconnect argument, applied to the phase structure of generation instead of the layer structure of the model.

## Stage 7 — The reliability wall: observability, SLO-aware scheduling, the runbook

Thirty million requests a day across dozens of spot GPUs in two pools with a stateful prefix cache is a lot of moving parts, and the wall you hit here is not performance — it is *knowing what is happening*. The symptom is an incident you cannot diagnose: p99 TTFT crept from 240 ms to 900 ms over an hour, users are complaining, and no single dashboard says why. Was it a spot reclamation that shrank the decode pool? A prefix-cache hit-rate collapse after a deploy changed the system prompt? A single straggler GPU with a failing NVLink? A slow KV transfer saturating the RDMA fabric? Without instrumentation you are guessing, and guessing at 3 a.m. across a distributed system is how a one-hour blip becomes a four-hour outage.

The fix is **observability designed in, not bolted on**, organized around the metrics that actually predict LLM-serving pain. The four that matter most: TTFT and TPOT (time per output token) percentiles, because they are what users feel; queue depth and admission-reject rate, because they are the leading indicator of the Stage 3 instability; prefix-cache hit rate, because a drop in it silently raises cost and latency; and KV-cache utilization and preemption rate per pool, because preemption is the Stage 2 memory wall reappearing under load. Export these from every replica, scrape them with Prometheus, and alert on **burn rate** against an explicit error budget rather than on raw thresholds — a burn-rate alert fires when you are consuming your monthly SLO budget too fast, which catches slow degradations that a static threshold misses.

```yaml
# stage7_alerts.yaml — Prometheus burn-rate alerts on the SLOs that matter
groups:
  - name: llm-serving-slo
    rules:
      # Fast burn: TTFT SLO (p99 < 500ms) being violated at >14x budget rate.
      - alert: TTFTSLOFastBurn
        expr: |
          histogram_quantile(0.99,
            sum by (le, pool) (rate(vllm_time_to_first_token_seconds_bucket[5m]))
          ) > 0.5
          and
          histogram_quantile(0.99,
            sum by (le, pool) (rate(vllm_time_to_first_token_seconds_bucket[1h]))
          ) > 0.5
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "p99 TTFT over 500ms on {{ $labels.pool }} — runbook: TTFT-burn"
      # Leading indicator: prefix-cache hit rate collapsed after a deploy.
      - alert: PrefixCacheHitRateDrop
        expr: |
          sum(rate(vllm_prefix_cache_hits_total[10m]))
            / sum(rate(vllm_prefix_cache_queries_total[10m])) < 0.20
        for: 10m
        labels: { severity: ticket }
        annotations:
          summary: "Prefix-cache hit rate < 20% (was ~38%) — check system-prompt change"
      # Memory wall resurfacing: sequences being preempted under load.
      - alert: KVPreemptionSpike
        expr: sum by (pool) (rate(vllm_num_preemptions_total[5m])) > 1
        for: 5m
        labels: { severity: page }
        annotations:
          summary: "KV preemptions on {{ $labels.pool }} — reduce max-num-seqs or scale out"
```

Metrics tell you *that* something degraded; **distributed tracing** tells you *where*. In a disaggregated stack a single request touches the gateway, the router, a cache lookup, a prefill GPU, a KV transfer, and a decode GPU — six hops, any of which can be the one adding latency. Propagate a trace context (OpenTelemetry, W3C `traceparent`) from the gateway through every hop and emit a span at each, and a slow request becomes a waterfall you can read: the 900 ms TTFT was 700 ms of it sitting in the prefill queue, which points straight at an undersized prefill pool, not a slow network or a cache miss. Without tracing you would have stared at six green per-component dashboards and concluded, wrongly, that nothing was broken. Sample traces at a low rate in steady state and turn sampling up during an incident; the storage cost of tracing every request at thirty million a day is not worth it, but a 1% sample plus on-demand boost catches the patterns.

The second half of this wall is **SLO-aware scheduling**. Not all traffic is equal — an interactive chat user needs a fast first token, while a background summarization job cares only about total throughput. An SLO-aware scheduler assigns priorities and can preempt a batch job's decode slots to protect an interactive request's TTFT, so your p99 for the traffic that *has* a tight SLO stays green even when the cheap-and-patient traffic surges. This is the architecture Kimi and other high-concurrency operators use to co-schedule latency-critical inference with throughput-oriented work, covered in [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management).

![Stack diagram of the final production serving architecture in six layers, from an admission-controlling API gateway at the top through a cache-aware router, disaggregated prefill and decode pools, vLLM engines, a Kubernetes GPU fleet, down to an observability and SLO-gate layer at the bottom.](/imgs/blogs/case-study-scaling-to-millions-of-requests-7.webp)

The stack above is the architecture that emerged from the whole journey, and every layer is the residue of a wall. The admission-controlling gateway is Stage 3. The cache-aware router is Stage 6. The disaggregated pools are Stage 6. The vLLM engines with FP8, paged KV, and chunked prefill are Stages 1–3. The Kubernetes GPU fleet with spot-and-on-demand mixing is Stages 4–5. And the observability layer with SLO gates watching the whole stack is this stage. Nobody designed this stack up front; it accreted, one wall at a time, and that is exactly why the *ordering* is the lesson. When an incident does happen, a written runbook — the exact queries to run, the exact scale-out commands, the exact rollback — is what turns the alert into a fifteen-minute fix; that discipline is the subject of the [troubleshooting-and-runbook guide](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook).

## Stage 8 — The quality wall: regression gates so speed didn't cost accuracy

The last wall is the quietest and the most dangerous, because it does not page you. Somewhere across the previous stages you made changes that trade accuracy for speed and cost: FP8 weights, FP8 KV cache, aggressive prefix caching, speculative decoding if you added it, a routing layer that sends "easy" queries to a smaller model. Each one is individually justified and individually tiny. But they compound, and none of them show up on a latency dashboard. The failure mode is a model that is 40x cheaper to serve and subtly, measurably worse — and you find out from a slow bleed of user trust rather than an alert.

The fix is a **regression gate** in the deployment pipeline: before any serving-configuration change reaches production, run the candidate against a fixed evaluation set and block the deploy if quality drops beyond a threshold. The evaluation set is a curated bank of representative prompts with either reference answers or an automated judge, and the gate checks that the quantized-and-cached candidate scores within, say, one point of the reference configuration on your task metrics. This is the serving-side analogue of a unit test: it does not tell you the model is *good*, it tells you the deploy did not make it *worse*.

```python
# stage8_gate.py — block a serving-config change that regresses quality
import json, sys

THRESHOLD = 1.0  # max allowed drop, in points, vs the reference config

def evaluate(endpoint: str, eval_set: list[dict]) -> float:
    # Run the candidate serving config against a fixed prompt bank; score with
    # an automated judge or exact-match/rouge against reference answers.
    scores = [score_one(endpoint, item) for item in eval_set]
    return sum(scores) / len(scores)

def main():
    eval_set = json.load(open("eval_bank.json"))         # frozen, versioned
    reference = evaluate("http://reference-bf16:8000", eval_set)
    candidate = evaluate("http://candidate-fp8:8000", eval_set)
    drop = reference - candidate
    print(f"reference={reference:.2f} candidate={candidate:.2f} drop={drop:.2f}")
    if drop > THRESHOLD:
        print("REGRESSION GATE FAILED — blocking deploy")
        sys.exit(1)     # CI fails; the config change does not ship
    print("quality gate passed")

if __name__ == "__main__":
    main()
```

Run this gate in CI on every change to the serving configuration, and run a lighter continuous version in production — sample real traffic through both the current and a shadow configuration, compare, and alert on divergence. The gate is what lets you keep optimizing aggressively *because* you have a safety net: you can try FP8 KV cache, or a more aggressive quantization, or a new draft model for speculative decoding, and ship it the instant the gate says quality held. Without the gate, every optimization is a gamble on user trust. With it, optimization becomes routine.

Two traps make quality gates lie, and both are worth knowing before you trust one. The first is the **eval-set-as-training-target** problem: if the same fixed bank gates every deploy for a year, teams start (consciously or not) tuning to it, and it stops predicting real-world quality. Refresh it periodically with freshly sampled production prompts, and keep a held-out slice you never inspect. The second is **judge bias**: an LLM-as-judge is cheap and scalable but has known systematic biases — it prefers longer answers, prefers responses that match its own style, and can be swayed by position when comparing two candidates. Mitigate with pairwise comparison plus position-swapping, calibrate the judge against a small human-labeled set, and treat a judge-only score as a screen, not a verdict, for anything user-facing. The point of the gate is not to certify that the model is good; it is to catch the specific, silent regression where a serving optimization made it *worse* — and for that narrow job, a well-built gate is the difference between shipping speedups confidently and shipping them on faith. The measurement discipline behind these gates — building the eval bank, choosing metrics, avoiding judge bias — is its own craft, adjacent to the load-and-quality evaluation covered across the reliability track of this series.

## The final architecture and the aggregate numbers

Step back and look at what the journey produced. The prototype was one A100 running Flask, serving fifty thousand requests a day at 1.8 seconds p99 TTFT and about \$17 per million tokens. The final system is a Kubernetes fleet of 24 to 64 H100s, spot-heavy, split into disaggregated prefill and decode pools behind a cache-aware SLO-aware router, running a 70B model in FP8 with paged KV and chunked prefill, watched by burn-rate alerts and protected by regression gates. It serves thirty million-plus requests a day at about 260 ms p99 TTFT and about \$0.50 per million tokens.

![Matrix scorecard of six scaling stages against four metrics — daily scale, p99 time to first token, cost per million tokens, and GPU count — showing scale growing from fifty thousand to thirty million-plus requests a day while p99 latency falls and cost drops from seventeen dollars forty to fifty cents per million tokens.](/imgs/blogs/case-study-scaling-to-millions-of-requests-8.webp)

The scorecard above is the whole story in one frame, and the honest bump in it is the point. Read down the cost column and you see it drop from \$17.40 to \$0.30 to \$0.16 as batching and quantization did their work on the 8B model — and then *jump* to \$1.90 at Stage 4 when the model upgraded to 70B for quality. The later stages claw that back to \$0.50. That bump is not a mistake in the story; it is the story. Scaling is not monotonic cost reduction. You spend a corner of the triangle to buy quality, then earn it back with the next set of techniques. Here is the same data as a table you can copy:

| Stage | Scale (req/day) | Aggregate throughput | p99 TTFT | Cost / 1M tok | GPUs | Model |
|---|---|---|---|---|---|---|
| 0 — Flask serial | 50K | ~40 tok/s | 1.8 s | \$17.40 | 1 | 8B |
| 1 — Continuous batching | 400K | ~2,300 tok/s | 900 ms | \$0.30 | 1 | 8B |
| 2 — FP8 + KV mgmt | 900K | ~4,300 tok/s | 820 ms | \$0.16 | 1 | 8B |
| 3 — Chunked prefill + admission | 2M | ~4,000 tok/s | 340 ms | \$0.17 | 2–8 | 8B |
| 4 — TP + 70B upgrade | 5M | ~2,340 tok/s/replica | 520 ms | \$1.90 | 8–32 | 70B |
| 5 — Spot + prefix cache | 12M | ~3,100 tok/s/replica | 520 ms | \$0.78 | 8–40 | 70B |
| 6 — PD disaggregation | 30M | ~4,600 tok/s/replica | 240 ms | \$0.52 | 24–64 | 70B |
| 7 — Observability + SLO sched | 30M+ | ~4,600 tok/s/replica | 260 ms | \$0.50 | 24–64 | 70B |
| 8 — Regression gates | 30M+ | ~4,600 tok/s/replica | 260 ms | \$0.50 | 24–64 | 70B |

*All figures illustrative and composite; the trajectory is representative, the exact numbers are not from any single production system.*

The aggregate wins: daily scale grew about 600x, cost per token on the 8B tier fell roughly 100x (\$17.40 to \$0.16) and on the upgraded 70B tier fell about 3.8x from its post-upgrade peak (\$1.90 to \$0.50), and p99 TTFT dropped from 1.8 s to about 0.26 s *while* the model got nine times larger and the traffic grew three orders of magnitude. Every one of those wins came from a single named technique applied at the right moment. Here is the techniques-applied summary — the second reference table, the one that maps each wall to its governing law and its fix:

| Wall | Symptom at scale | Governing law | The fix | What it bought |
|---|---|---|---|---|
| Throughput | queue never drains, GPU 8% util | Little's Law, $\lambda_{\max}=L_{\max}/W$ | continuous batching (vLLM) | ~55x throughput |
| Memory | throughput plateaus, preemption spikes | KV bytes/token equation | FP8 weights + PagedAttention | ~1.8x concurrency |
| Tail-under-burst | p99 TTFT spikes 300ms→8s | M/M/1: $W=1/\mu(1-\rho)$ | chunked prefill + admission control | bounded tail |
| Capacity | 70B does not fit one GPU | interconnect bandwidth (NVLink vs IB) | tensor then pipeline parallelism | bigger model fits |
| Cost | GPU bill dominates | $c=P_{\text{gpu}}N_{\text{gpu}}/(\text{goodput}\cdot 3600)$ | spot + prefix cache + right-size | ~2.4x cheaper |
| Latency-at-scale | prefill/decode interference | match resource to bottleneck | PD disaggregation + cache-aware routing | p99 halved |
| Reliability | undiagnosable degradations | error-budget burn rate | observability + SLO scheduling | fast diagnosis |
| Quality | silent accuracy drift | eval score vs reference | regression gates in CI | trust preserved |

## Named-hardware scorecard

The abstract stages ran on specific silicon, and the choice of GPU per stage was itself a decision on the triangle. The 8B tier lived on A100 40GB because it fit and the price was right; the 70B tier moved to H100 80GB because FP8 throughput and NVLink bandwidth both matter more at that size. The table below grounds the journey in named hardware.

| Stage / tier | GPU | HBM | HBM bandwidth | Intra-node link | Role | Spot? |
|---|---|---|---|---|---|---|
| 0–3 (8B) | A100 40GB | 40 GB | ~2.0 TB/s | NVLink 600 GB/s | single-replica serving | on-demand |
| 4 (70B) | H100 80GB | 80 GB | ~3.35 TB/s | NVLink 900 GB/s | TP=4 replica | on-demand baseline |
| 5–8 prefill pool | H100 80GB | 80 GB | ~3.35 TB/s | NVLink 900 GB/s | compute-bound prefill | 70% spot |
| 5–8 decode pool | H100 80GB | 80 GB | ~3.35 TB/s | NVLink 900 GB/s | memory-bound decode | 70% spot |

The hardware ecosystem — H100 versus H200 versus B200, AMD MI300X, and heterogeneous clusters — is a decision in its own right, but the principle that carried through every stage is the one worth keeping: **decode is bounded by HBM bandwidth, prefill by FLOPs, and multi-GPU by interconnect.** Pick the GPU, and the sharding, to match whichever of those is your binding constraint at that stage.

#### Worked example: cost per million tokens at the final stage

Take the decode-heavy final configuration and put real numbers through the cost formula. A decode-pool H100 on spot costs about \$1.60/hr. A serving unit that delivers the product's output — accounting for the blended prefill-plus-decode GPUs behind each stream of tokens — works out to roughly 5 H100-equivalents feeding an effective goodput of about 4,600 output tokens per second (after prefix-cache skips, chunked-prefill overhead, and the utilization you actually sustain under SLA). Then

$$c_{\text{token}} = \frac{1.60 \times 5}{4600 \times 3600} \approx \$0.48\ \text{per million tokens}.$$

Round to about \$0.50, matching the scorecard. Now see what each lever contributed. Start from the Stage 4 on-demand figure of \$1.90. Spot cuts the price term $P_{\text{gpu}}$ by roughly 0.42x. Prefix caching and disaggregation together raise effective goodput by roughly 1.6x (a 38% prefill-skip rate plus denser decode packing). Right-sizing trims the GPU count behind interactive traffic. Multiply through — $1.90 \times 0.42 / 1.6 \approx \$0.50$ — and the three moves account for the whole 3.8x reduction. That is the discipline of the cost wall: it is not one clever trick but a product of independent multipliers, each traceable to one term in one equation.

## Case studies

The composite trajectory above is stitched from real, public results. Anchoring the illustrative numbers to their sources is worth doing, both to be honest about what is measured versus modeled and because the primary sources are where the depth is.

**vLLM and PagedAttention (Kwon et al., SOSP 2023).** The original PagedAttention paper reports 2–4x higher throughput than prior systems at the same latency, driven by cutting KV-cache waste from 60–80% down to under 4% through paging. The Stage 1 and Stage 2 numbers in this post — the continuous-batching throughput jump and the memory-headroom win — trace directly to this work and to the vLLM benchmarks that followed. The "up to 24x over Hugging Face Transformers" figure often quoted is against the naive `generate()`-in-a-loop baseline of Stage 0, which is exactly why the baseline in this story is so slow: it is that same naive path.

**DistServe (Zhong et al., OSDI 2024) and Splitwise (Patel et al., ISCA 2024).** Both papers make the prefill/decode disaggregation case with measurements. DistServe reports serving up to several times more requests within tight latency SLOs by placing prefill and decode on separate resources and tuning each independently; Splitwise, from Microsoft, characterizes the opposite hardware profiles of the two phases and shows cost and power wins from splitting them onto different machine pools. The Stage 6 p99 improvement and the pool-sizing logic come from this line of work. The key measured insight — that prefill is compute-bound and decode is memory-bound, so co-locating them wastes one or the other — is the load-bearing claim of that stage.

**DeepSeek-V3 / R1 inference disclosures (2025).** DeepSeek published unusually detailed notes on serving their 671B-parameter mixture-of-experts model, including large-scale expert parallelism, prefill/decode separation, and aggressive FP8 use, reporting cost-per-token figures far below what a dense model of comparable quality would cost. This grounds the Stage 4 expert-parallelism row and the general claim that MoE plus EP plus disaggregation is the frontier cost structure. The specifics of MoE routing and expert parallelism are their own deep dive in [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale).

**Large chat services (public postmortems and engineering blogs).** Operators of large public chat products have described, in blog posts and talks, the same arc this post narrates: an early scramble to add continuous batching, a memory crisis solved by quantization and paging, a tail-latency fight won by admission control and disaggregation, and a cost program built on spot fleets and caching. The exact numbers differ by model and traffic, but the *ordering of the walls* is strikingly consistent across independent operators — which is the strongest evidence that the ordering is not incidental but structural.

The honest caveat on all of it: the scorecard's specific dollar figures are illustrative, assembled to be internally consistent with the equations in this post and directionally consistent with these public reports. Your mileage will vary with model, sequence lengths, prompt-sharing rate, cloud pricing, and SLA. What will not vary is which wall comes next.

## When to use this (and when not to)

The single most important lesson of this entire journey is about *ordering*, and it cuts against the instinct of every engineer who has read about the fancy techniques and wants to build them on day one. Do not.

**Do not pre-optimize for scale you do not have.** If you serve ten thousand requests a day, the Stage 0 prototype is *correct*. Flask and `generate()` are fine. Building prefill/decode disaggregation for ten thousand requests a day is not sophistication; it is waste — you will spend months constructing and operating a control plane, a KV-transfer fabric, and two GPU pools to solve a tail-latency-at-scale problem you do not have, while the actual bottleneck (you have not shipped the feature yet) goes unaddressed. Every wall in this post is worth breaking *only when you are hitting it*.

**The ordering is close to universal, so use it as a checklist.** Continuous batching before quantization before admission control before parallelism before cost work before disaggregation before observability-at-scale before quality gates. This order is not arbitrary — each fix is cheap relative to the next and each reveals the wall that the next one solves. If you find yourself reaching for disaggregation (Stage 6) while still running static batching (pre-Stage 1), stop: you have skipped the cheap 55x win to chase a 2x one. Climb the walls in order.

**Where the ordering bends.** If your model does not fit one GPU from the start — you are serving a 70B or a large MoE as your *first* product — then Stage 4's parallelism is not optional and moves to the front. If your workload has near-zero prompt sharing (every request is unique, no system prompt, no RAG), skip prefix caching; it will not pay. If you are on-premises with fixed, owned hardware, the spot-instance lever of Stage 5 disappears and the cost wall is fought entirely with quantization, batching, and utilization. And if you serve a strict-latency, low-throughput workload — a handful of requests per second with a tight p99 — several of these throughput-oriented techniques (large batches, disaggregation) actively hurt you, and you should optimize for single-stream latency instead.

**When *not* to self-host at all.** If your traffic is genuinely small and spiky, a hosted API priced per token may beat any self-hosted stack on total cost of ownership, because you pay nothing when idle and inherit someone else's optimization. The self-hosting journey in this post pays off when you have *sustained, high* traffic on an *open* model where the per-token economics and the control justify running the infrastructure. Below that threshold, the right answer to "how do I scale my serving stack" is "don't, yet."

## Key takeaways

- **Scaling is a sequence of walls, not a single problem.** At each order of magnitude a different corner of the latency-throughput-cost triangle becomes binding. Name the binding constraint before you reach for a technique.
- **Continuous batching is the highest-leverage change and it is nearly free.** Because decode is memory-bound, one sequence cannot saturate a GPU; batching dozens of them amortizes the weight read and buys roughly an order of magnitude of throughput with no new hardware.
- **The KV cache, not the weights, usually caps your batch.** Quantize weights to FP8 to hand the freed memory to the cache, and use PagedAttention to stop wasting it on fragmentation. Memory headroom is throughput.
- **Throughput and tail latency fail at different times.** Batching fixes throughput and does nothing for the burst tail. Chunked prefill plus a bounded admission queue is what holds p99 when a burst arrives — because an unbounded queue at $\rho>1$ diverges.
- **Keep tensor parallelism inside NVLink; go to pipeline parallelism across nodes.** The all-reduce on TP's critical path is cheap on NVLink and ruinous over InfiniBand. Match the parallelism axis to the interconnect.
- **Cost per token is a product of three independent levers.** Cheaper GPUs (spot), higher goodput (prefix caching), and fewer GPUs (right-sizing) multiply. Attack all three; each traces to one term in $c = P_{\text{gpu}}N_{\text{gpu}}/(\text{goodput}\cdot 3600)$.
- **Disaggregate prefill and decode only at scale.** Their opposite hardware profiles waste a co-located GPU, but the KV-transfer fabric and control plane are real cost — worth it above tens of QPS with strict p99, not before.
- **Observability and quality gates are what let you keep optimizing.** Burn-rate alerts turn silent degradations into pages; regression gates turn every risky optimization into a safe, routine deploy. Without them, aggressive optimization is a gamble on user trust.
- **The ordering is the lesson.** Do not build Stage 6 while running Stage 0. Climb the walls in order, and only when you are actually hitting them.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 — the vLLM paper; the origin of continuous batching plus paging.
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI 2024 — the measured case for Stage 6.
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024 — the hardware-profile argument for disaggregation.
- vLLM documentation — production flags for FP8, chunked prefill, prefix caching, tensor parallelism, and disaggregated serving.
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the Stage 1 and Stage 2 mechanics in full.
- [Prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) and [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving) — Stages 4 and 6.
- [High-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management), [cost optimization at LLM scale](/blog/machine-learning/model-serving/cost-optimization-at-llm-scale), and the [troubleshooting-and-runbook guide](/blog/machine-learning/model-serving/troubleshooting-llm-serving-runbook) — Stages 3, 5, and 7.
