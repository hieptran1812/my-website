---
title: "What inference engineering is: turning weights into tokens under a budget"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "The opening post of a 65-part series that builds an LLM inference engine from scratch: the six-layer map, the two-phase workload, the arithmetic that sets your token ceiling, and the scoreboard you will be judged on."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "latency",
    "throughput",
    "batching",
    "pytorch",
    "gpu",
    "ml-systems",
    "decoding",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 44
---

There is a moment that happens to almost every engineer who ships an LLM feature. The demo works. One user, one prompt, tokens streaming out at a pleasant reading pace. Then you put it behind a real endpoint, twenty people open the tab, and the same GPU that felt fast is now producing four tokens per second per user, the p99 time-to-first-token is eleven seconds, and `nvidia-smi` insists the card is at 96% utilization — as if it were working hard the whole time. Nothing is broken. No exception was thrown. The system is simply doing exactly what you asked it to, and what you asked it to do was wasteful in a way that is completely invisible from the Python.

That invisibility is the entire problem. In training you can see the loss curve. In web services you can see the request log. In LLM inference the interesting quantities — how many bytes crossed the memory bus to produce one token, how much of the KV cache is reserved but unused, how long a request sat in a queue before any GPU work started, how many tokens of a batch were padding — are not printed anywhere by default. You have to know they exist before you can go looking for them.

**Inference engineering is the craft of making those quantities visible and then controlling them.** It is not "using vLLM well", although knowing vLLM is useful. It is not "profiling", although you will profile constantly. It is the discipline of turning a directory of weight files into a stream of tokens that meets a latency target, fits in a memory budget, and costs an amount of money you can defend in a planning meeting — and of understanding every layer between the weights and the user well enough that you could, if you had to, build the thing yourself.

This series takes that "if you had to" literally. Over 65 posts we build a toy inference engine called **`nanoserve`**: it loads a checkpoint, runs a forward pass, keeps its own KV cache, pages that cache into blocks, schedules a continuously churning batch of requests, samples with real samplers and grammar masks, gets its hot loops rewritten in CUDA and Triton, shards itself across GPUs, and finally speaks an OpenAI-compatible HTTP API. In the capstone we point a load generator at both `nanoserve` and vLLM and publish an honest gap table — what we are within 20% of, what we are five times behind on, and exactly why. This first post is the map. Figure 1 is the spine of everything that follows.

![A stacked diagram showing six layers from model weights at the bottom through kernels, engine, decoding layer, API and finally product economics at the top](/imgs/blogs/what-inference-engineering-is-1.webp)

By the end of this post you will be able to: state the decode-time floor for any model on any GPU from two numbers you can look up; compute how many concurrent users fit in a given amount of VRAM; explain why a "tokens per second" figure with no batch size attached is not a measurement; and read the rest of the series knowing which layer each post is operating on.

## 1. Three crafts that keep getting confused

Three job descriptions overlap here, and conflating them is why teams end up optimizing the wrong layer for a quarter.

**Model serving** is the operational craft: you take an engine somebody else wrote — vLLM, TensorRT-LLM, SGLang, TGI, Ollama — and you make it run reliably. Container images, GPU node pools, autoscaling, health checks, canary rollouts, quota. The unit of work is a *deployment*. The skill is knowing which flags exist and what they do. This repo already has a large body of work on exactly that, starting with [what model serving is](/blog/machine-learning/model-serving/what-is-model-serving) and [why LLM serving is different from serving any other model](/blog/machine-learning/model-serving/why-llm-serving-is-different). If your job is to keep an endpoint up, that is the series to read.

**Performance engineering** is the measurement craft: you have a running system that is too slow, and you find out why. Nsight Systems timelines, kernel-level counters, occupancy, launch overhead, host-device synchronization, the roofline. The unit of work is a *bottleneck*. The skill is being able to prove where the time went instead of guessing. Also well covered here — [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is the prerequisite for taking any number in this series seriously.

**Inference engineering** is the constructive craft in between: given a checkpoint and a latency, memory and cost budget, *design the thing that turns one into the other*. The unit of work is a *component* — an allocator, a scheduler policy, a sampling kernel, a cache layout, a streaming protocol. The skill is knowing what each component is allowed to cost and what happens to the other components when it costs more.

| | Model serving | Performance engineering | Inference engineering |
| --- | --- | --- | --- |
| Unit of work | A deployment | A bottleneck | A component |
| Typical artifact | A Helm chart, a flag set | A profile, a flame graph | An allocator, a scheduler |
| Core question | Is it up and scaled? | Where did the time go? | What should this layer cost? |
| Fails as | Outage, cold start, quota | A regression nobody catches | A design that cannot meet the SLO at any flag setting |
| Reads the source of | Kubernetes, the engine's docs | The kernel timeline | The engine itself |
| This series | Links out to it | Uses it as the instrument | Is this |

The reason the distinction matters practically: **there are failures no flag can fix.** If your workload is 200-token prompts and 2,000-token outputs, you are almost entirely decode-bound, and no amount of tuning `--max-num-batched-tokens` will help you until you understand that a decode step's cost is set by how many bytes of weights the GPU must read, not by how many FLOPs it must do. If your workload is 100k-token RAG prompts and 50-token answers, you are almost entirely prefill-bound, and the KV cache optimizations everyone talks about barely move you. Choosing the right lever requires a model of where the cost lives. That model is what the next four sections build.

## 2. The layer map: six layers, six ways to lose

Figure 1 is the frame the whole series hangs on. Six layers between a directory of `.safetensors` files and a line item on a P&L. Each layer owns a specific decision, and each layer has a characteristic way of costing you when the decision is wrong.

**Layer 1 — Weights.** What is on disk and what lands in VRAM. Format (safetensors, GGUF, AWQ, GPTQ), dtype, sharding, memory-mapped versus eager load, device placement, tied embeddings. The decisions here set a hard floor on everything above: a bf16 8B model is roughly 16.1 GB of parameters and *that number becomes the per-token cost of decoding*, as section 4 shows. When this layer is wrong you get a 90-second cold start, or an out-of-memory error during load on a card that has plenty of room for the model at rest, or a silent dtype upcast that doubles your traffic.

**Layer 2 — Kernels.** The actual GPU code: GEMMs, GEMVs, attention, RMSNorm, RoPE, the softmax, the KV write. Fusion, tiling, memory coalescing, tensor-core utilization, launch overhead. When this layer is wrong you leave a factor of two on the table and never know it, because nothing in the Python looks different. This is the layer where [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) stops being a teaching diagram and starts being a budget.

**Layer 3 — Engine.** The KV cache and the scheduler: the two pieces of machinery that turn "run a forward pass" into "serve many people at once". Cache layout, block size, allocation and eviction, prefix sharing, the admission policy, preemption, chunked prefill. This is the layer where most of the surprising behavior lives, because it is the only layer where *requests interact with each other*. When this layer is wrong your p99 detonates while your GPU utilization graph stays flat and reassuring.

**Layer 4 — Decoding.** Everything that happens between the logits and the token id: temperature, top-k, top-p, min-p, penalties, grammar and JSON masks, stop conditions, EOS handling, thinking budgets. It is the smallest layer by FLOPs and the largest by number of production incidents per line of code, because it is where correctness lives. A wrong stop-string implementation truncates a user's answer mid-word. A grammar compiled on the request path adds 300 ms to TTFT. A logit mask applied on the host adds a device synchronization to every single step.

**Layer 5 — API and streaming.** The contract with the client: server-sent events, incremental detokenization, cancellation when the browser tab closes, usage accounting, idempotency, backpressure. This layer decides what "latency" even means to a user, because TTFT is measured from the moment the request arrives here — including any time it spends queued.

**Layer 6 — Economics.** Dollars per million tokens, which is the only number that survives contact with a finance team. It is not an independent layer; it is the arithmetic product of everything below it. Section 6 derives it.

The reason to draw them as a stack rather than a list is that **each layer sets the constraints for the one above and is constrained by the one below.** You cannot design a scheduler without knowing the decode step's cost model, which comes from the kernels, which is bounded by the weight format. And you cannot pick a weight format sensibly without knowing what the scheduler will do with the memory you save. This is why "just turn on quantization" so often disappoints: it is a layer-1 change whose value is realized at layer 3, and only if layer 3 is set up to spend the freed memory on more concurrent sequences.

| Layer | Owns the decision about | Costs you, when wrong | Series track |
| --- | --- | --- | --- |
| 1 Weights | Format, dtype, placement, load path | Cold start, load-time OOM, doubled traffic | A, F |
| 2 Kernels | Fusion, tiling, coalescing, tensor cores | A silent 2x you never see | E |
| 3 Engine | Cache layout, blocks, admission, preemption | p99 collapse at flat GPU utilization | B, C |
| 4 Decoding | Samplers, masks, stops, determinism | Wrong output, broken JSON, host syncs | D |
| 5 API | Streaming, cancellation, accounting | Wasted GPU work on abandoned requests | I |
| 6 Economics | Batch policy, hardware tier, SLO targets | A product that cannot be priced | H, I, J |

## 3. Two programs wearing one trench coat

Here is the fact that makes LLM inference unlike any other model-serving workload, and the reason a ResNet-serving intuition actively misleads you: **a single LLM request runs two completely different programs, one after the other, with opposite hardware bottlenecks.**

![A timeline showing a request arriving, a single dense prefill pass, the first token, then hundreds of small decode steps with the cache growing](/imgs/blogs/what-inference-engineering-is-2.webp)

**Prefill** processes the entire prompt in one shot. All 2,048 prompt tokens go through all 32 layers together, as a matrix. Every matrix multiply is a proper GEMM with a fat inner dimension: a `[2048, 4096] x [4096, 4096]` projection, a `[2048, 4096] x [4096, 14336]` MLP up-projection. Tensor cores love this. Each weight matrix is loaded from HBM once and then reused across 2,048 rows of activations, so the ratio of arithmetic to memory traffic is enormous. Prefill is **compute-bound**: it finishes when the GPU has done the FLOPs, and doing it faster means doing fewer or cheaper FLOPs.

**Decode** produces one token, then uses that token to produce the next one, then that one to produce the next. It cannot be parallelized across time, because token $t+1$ depends on token $t$ having been sampled. So each step processes a batch of exactly one token per sequence. Now that same `[4096, 4096]` projection is a `[1, 4096] x [4096, 4096]` operation — a **GEMV**, a matrix-vector product. The weight matrix is still 16 MB. The activation is still 8 KB. The GPU still has to pull the entire matrix across the memory bus, and then does two FLOPs per element and throws it away. Tensor cores sit idle; the memory controller is the only busy unit on the chip. Decode is **memory-bandwidth-bound**.

The animation below is the shape of a single request. Watch where the time actually goes.

<figure class="blog-anim">
<svg viewBox="0 0 720 250" role="img" aria-label="A wide prefill block fills in a single burst, then decode tokens appear one at a time across the rest of the row" style="width:100%;height:auto;max-width:820px">
<style>
.ie1-frame{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.ie1-bar{fill:var(--accent,#6366f1);opacity:.85}
.ie1-tok{fill:var(--accent,#6366f1);opacity:.9}
.ie1-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ie1-sub{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ie1-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes ie1-burst{0%{transform:scaleX(0)}10%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes ie1-drip{0%,12%{transform:scaleX(0)}100%{transform:scaleX(1)}}
.ie1-grow{animation:ie1-burst 11s linear infinite;transform-box:fill-box;transform-origin:left center}
.ie1-reveal{animation:ie1-drip 11s steps(9,end) infinite;transform-box:fill-box;transform-origin:left center}
@media (prefers-reduced-motion:reduce){.ie1-grow,.ie1-reveal{animation:none}}
</style>
<defs>
<clipPath id="ie1-clip">
<rect class="ie1-reveal" x="320" y="86" width="366" height="56"/>
</clipPath>
</defs>
<text class="ie1-hd" x="24" y="30">One request: 2048 tokens in, 500 tokens out</text>
<text class="ie1-sub" x="24" y="52">the two halves of the same request run on two different hardware limits</text>
<rect class="ie1-frame" x="24" y="80" width="272" height="68" rx="8"/>
<rect class="ie1-bar ie1-grow" x="32" y="88" width="256" height="52" rx="6"/>
<rect class="ie1-frame" x="316" y="80" width="374" height="68" rx="8"/>
<g clip-path="url(#ie1-clip)">
<rect class="ie1-tok" x="326" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="366" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="406" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="446" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="486" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="526" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="566" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="606" y="92" width="30" height="44" rx="5"/>
<rect class="ie1-tok" x="646" y="92" width="30" height="44" rx="5"/>
</g>
<line class="ie1-axis" x1="24" y1="166" x2="690" y2="166"/>
<text class="ie1-hd" x="24" y="192">PREFILL</text>
<text class="ie1-sub" x="24" y="212">2048 tokens, one dense pass</text>
<text class="ie1-sub" x="24" y="230">compute-bound: 32.9 TFLOP of GEMM</text>
<text class="ie1-hd" x="316" y="192">DECODE</text>
<text class="ie1-sub" x="316" y="212">one token per step, 500 steps</text>
<text class="ie1-sub" x="316" y="230">bandwidth-bound: 16.1 GB read per token</text>
</svg>
<figcaption>Prefill lands as a single burst of dense matrix work; decode then drips out one token per step, each step re-reading the entire weight matrix from HBM. Same request, opposite bottlenecks.</figcaption>
</figure>

### The code that makes the split obvious

Most people meet this split for the first time by writing the loop by hand instead of calling `generate()`. Here is the whole thing, using `transformers` purely as a reference implementation — the model we build in Track A replaces it entirely.

```python
# scratch/two_phases.py — the split, made explicit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

msgs = [{"role": "user", "content": "Explain HBM bandwidth to a backend engineer."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

with torch.inference_mode():
    # ---- PHASE 1: PREFILL. All prompt tokens, one pass, GEMM-shaped.
    out = model(ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)
    produced = [next_id]

    # ---- PHASE 2: DECODE. One token in, one token out, GEMV-shaped.
    for _ in range(255):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        produced.append(next_id)

print(tok.decode(torch.cat(produced, dim=-1)[0]))
```

Two things are worth staring at. First, the shape of the input to `model()` changes from `[1, 2048]` to `[1, 1]` between the two phases and never changes again — every subsequent step feeds the model a single token. Second, `past_key_values` is where the previously computed keys and values live, and it *grows by one position per step, per layer, forever*. That growing object is the KV cache, and it is the reason the memory section of this post exists. If you want the full derivation of why it must exist at all, [the KV cache post](/blog/machine-learning/large-language-model/kv-cache) covers the recompute-versus-cache argument; Track B rebuilds it from scratch inside `nanoserve`.

The practical consequence of the two-phase split is that **almost every inference technique you will read about helps exactly one phase.** Chunked prefill is a prefill technique. Speculative decoding is a decode technique. Paged attention is a memory technique that mostly unlocks decode concurrency. Quantization is a bandwidth technique that mostly helps decode and can *hurt* prefill by adding dequantization work to a phase that was already compute-bound. If you cannot say which phase your workload is dominated by, you cannot say which of these will help you.

## 4. The decode floor: one division that explains the whole series

Now the arithmetic. This is the single most useful calculation in LLM inference, it takes ten seconds, and it predicts your batch-1 speed within a factor that is usually better than most people's intuition.

### The mechanism

During one decode step, the GPU must compute a forward pass through every layer. For every weight matrix in the model, the kernel must read that matrix out of HBM into the streaming multiprocessors. There is nowhere else for it to live: an 8B model in bf16 does not fit in the 40–50 MB of L2 cache on any of these cards, so every step is a fresh trip to memory for essentially all of it.

So a lower bound on the time for one decode step is simply:

$$
t_{\text{step}} \;\ge\; \frac{B_{\text{weights}}}{\text{BW}_{\text{HBM}}}
$$

where $B_{\text{weights}}$ is the number of bytes of parameters and $\text{BW}_{\text{HBM}}$ is the GPU's memory bandwidth. Both are numbers you can look up. Invert it and you get a ceiling on tokens per second at batch 1:

$$
\text{tok/s}_{\max} \;=\; \frac{\text{BW}_{\text{HBM}}}{B_{\text{weights}}}
$$

That is it. No profiler, no benchmark, no GPU required. It is a lower bound rather than an estimate because it ignores the KV cache reads, the kernel launches, the sampling, the Python — all of which only add time.

#### Worked example: the batch-1 ceiling for Llama-3.1-8B

Llama-3.1-8B has 8.03 billion parameters. In bf16 that is 2 bytes each:

$$
B_{\text{weights}} = 8.03 \times 10^9 \times 2 = 1.606 \times 10^{10}\ \text{bytes} \approx 16.1\ \text{GB}
$$

Now divide by the published memory bandwidth of each card in the series matrix.

| GPU | HBM bandwidth (spec) | Bytes read per token | Floor, ms/token | Ceiling, tok/s | Source |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 24GB | 300 GB/s | 16.1 GB | 53.5 | 19 | derived; bandwidth cited: NVIDIA L4 datasheet |
| RTX 4090 24GB | 1,008 GB/s | 16.1 GB | 15.9 | 63 | derived; bandwidth cited: NVIDIA RTX 4090 specs |
| A100 80GB SXM | 2,039 GB/s | 16.1 GB | 7.9 | 127 | derived; bandwidth cited: NVIDIA A100 datasheet |
| H100 80GB SXM | 3,350 GB/s | 16.1 GB | 4.8 | 209 | derived; bandwidth cited: NVIDIA H100 datasheet |

Read that table again and notice what it does *not* contain. It does not contain FLOPs. The H100 has roughly three times the tensor-core throughput of the A100 and delivers roughly 1.6x the batch-1 decode speed, because the ratio that matters at batch 1 is the bandwidth ratio, not the compute ratio. Anyone quoting a decode speedup that tracks a card's TFLOPS number is quoting a prefill benchmark.

Here is the calculator, which becomes the first file in `nanoserve`.

```python
# nanoserve/roofline.py — the two numbers that bound everything
from dataclasses import dataclass

# Vendor-published HBM bandwidth, bytes/second. Cite these, do not guess them.
GPUS = {
    "L4 24GB":        300e9,
    "RTX 4090 24GB":  1008e9,
    "A100 80GB SXM":  2039e9,
    "H100 80GB SXM":  3350e9,
}

@dataclass
class ModelSpec:
    name: str
    params: float          # total parameter count
    bytes_per_param: float # 2.0 bf16, 1.0 int8, 0.5 int4 (+ small scale overhead)

    @property
    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_param

def decode_floor(model: ModelSpec, hbm_bw: float) -> tuple[float, float]:
    """Return (seconds per token, tokens per second) at batch 1, weights only."""
    sec = model.weight_bytes / hbm_bw
    return sec, 1.0 / sec

if __name__ == "__main__":
    m = ModelSpec("Llama-3.1-8B", params=8.03e9, bytes_per_param=2.0)
    print(f"{m.name}: {m.weight_bytes/1e9:.2f} GB of weights")
    for gpu, bw in GPUS.items():
        sec, tps = decode_floor(m, bw)
        print(f"  {gpu:16s} {sec*1e3:6.1f} ms/token   {tps:6.0f} tok/s ceiling")
```

```console
Llama-3.1-8B: 16.06 GB of weights
  L4 24GB            53.5 ms/token       19 tok/s ceiling
  RTX 4090 24GB      15.9 ms/token       63 tok/s ceiling
  A100 80GB SXM       7.9 ms/token      127 tok/s ceiling
  H100 80GB SXM       4.8 ms/token      209 tok/s ceiling
```

**What you should actually observe.** Real engines land somewhere between 60% and 85% of this ceiling at batch 1, because the KV cache reads, the attention kernel's own overhead, per-step kernel launches and the sampler all cost time the formula ignores. So on an RTX 4090 running an 8B model in bf16 at batch 1, a well-implemented engine should put you somewhere in the neighbourhood of 40–55 tok/s. Run it yourself and see where you land; if you are far *below* that band, something in your stack is broken, and if you are above the 63 tok/s ceiling, either your model is quantized or your timer is wrong.

### The other half: arithmetic intensity, and why batching is the whole game

The same arithmetic explains why batching works, and it is worth being precise about *why*, because the usual explanation ("the GPU is more efficient with more work") is not wrong but is not actionable.

Define **arithmetic intensity** as FLOPs performed per byte moved from memory. For a decode step at batch size $B$, the weight matrices are read once regardless of $B$, and each of the $B$ sequences does two FLOPs per parameter (a multiply and an add). So:

$$
\text{AI} \;=\; \frac{2 \cdot P \cdot B}{2 \cdot P} \;=\; B \quad \text{FLOPs per byte (bf16 weights)}
$$

At batch 1 the arithmetic intensity is **one FLOP per byte**. Every GPU in the matrix has a ridge point — the intensity at which it stops being memory-bound and starts being compute-bound — that is two orders of magnitude higher than that. For the A100, ${312 \times 10^{12}}/{2.039 \times 10^{12}} \approx 153$ FLOPs per byte. So a batch-1 decode step is running at roughly 0.7% of the card's compute capability, and *the fix is not a faster kernel*. The fix is more sequences.

![Two columns comparing a single-sequence decode step with a batch of thirty-two sharing the same weight read](/imgs/blogs/what-inference-engineering-is-3.webp)

This is the load-bearing insight of the entire discipline. **Batching does not make a decode step faster. It makes the same expensive read produce more tokens.** A batch-32 decode step takes barely longer than a batch-1 step — the weights are read once either way — but emits 32 tokens instead of one. Which is why every serious inference engine is, structurally, a machine for keeping the batch full: continuous batching, chunked prefill, admission control, paged memory, prefix sharing. They are all in service of the same goal, which is not making a step faster but making each step carry more passengers.

#### Worked example: what a batch-32 decode step actually costs

Batch 32 is not free, because the KV cache reads *do* scale with batch size. Take 32 sequences each holding 2,048 tokens of context on an RTX 4090.

KV bytes per token for Llama-3.1-8B (derived in full in section 5): 131,072 bytes. So the KV traffic per decode step is

$$
32 \times 2048 \times 131072 = 8.59 \times 10^{9}\ \text{bytes} = 8.59\ \text{GB}
$$

Total bytes moved in the step: ${16.06 + 8.59 = 24.65}$ GB. At 1,008 GB/s that is 24.5 ms per step, which produces 32 tokens:

$$
\frac{32}{0.0245\ \text{s}} \approx 1{,}308\ \text{tok/s}
$$

Compare to 63 tok/s at batch 1. The step got 54% slower and the output went up 32x, for a net **21x** improvement in aggregate throughput on the same card. Meanwhile each individual user's inter-token latency went from 15.9 ms to 24.5 ms — worse, but still well under the ~50 ms that reads as "faster than I can read". That trade — a modest per-user latency regression for a 20x throughput gain — is the deal that makes LLM serving economically possible at all, and negotiating it precisely is what Track C is about.

Notice also what the KV term does as context grows. At 8,192 tokens of context per sequence, the KV traffic becomes 34.4 GB — now *larger* than the weights, and the step is dominated by cache reads rather than weight reads. This is the crossover that motivates KV cache quantization, multi-query and grouped-query attention, and the entire hybrid-architecture conversation in Track K. Where exactly it lands for your model is a five-line calculation, and Track B's memory-math post does it for every model in the matrix.

## 5. The memory ceiling: the cache you never allocated

Latency has a floor. Concurrency has a ceiling, and the ceiling is set by memory.

### The mechanism

Every token that has ever been in a sequence — prompt or generated — leaves behind a key vector and a value vector *in every layer*, so that future tokens can attend to it. The size of that residue per token is:

$$
B_{\text{kv/token}} \;=\; 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot b
$$

where the leading 2 is for K and V, $L$ is the number of layers, $H_{kv}$ is the number of *key/value* heads (not attention heads — this is the grouped-query attention distinction and it matters enormously), $d_{\text{head}}$ is the head dimension, and $b$ is bytes per element.

#### Worked example: the KV footprint of Llama-3.1-8B

From the published config: 32 layers, 32 attention heads, **8** key/value heads, head dimension 128, bf16.

$$
B_{\text{kv/token}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB}
$$

128 KB per token. Some consequences that fall straight out of it:

- An 8,192-token conversation costs $8192 \times 131072 = 1.07\ \text{GB}$ of VRAM. One conversation.
- The model's advertised 128k context, fully used by a single request, costs $131072 \times 131072 = 1.72 \times 10^{10}$ bytes — **17.2 GB**. On a 24 GB RTX 4090 that already holds 16.1 GB of weights, a single maximum-context request cannot fit its own cache. Not "will be slow". Cannot fit.
- Had Llama-3.1-8B used plain multi-head attention with 32 KV heads instead of 8, the per-token cost would be 512 KB and that same 128k request would need 68.7 GB. Grouped-query attention did not make the model smarter; it made this class of model servable, and it is fair to call it the highest-leverage inference change of its generation.

Here is the calculator, reading real config fields rather than hardcoded constants — the second file in `nanoserve`.

```python
# nanoserve/kvmath.py — cache capacity from a real HF config
from transformers import AutoConfig

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}

def kv_bytes_per_token(model_id: str, kv_dtype: str = "bf16") -> int:
    cfg = AutoConfig.from_pretrained(model_id)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return 2 * cfg.num_hidden_layers * n_kv * head_dim * DTYPE_BYTES[kv_dtype]

def capacity(vram_gb: float, weight_gb: float, overhead_gb: float,
             bytes_per_token: int, ctx_per_seq: int) -> tuple[int, int]:
    free_bytes = (vram_gb - weight_gb - overhead_gb) * 1e9
    total_tokens = int(free_bytes // bytes_per_token)
    return total_tokens, total_tokens // ctx_per_seq

if __name__ == "__main__":
    bpt = kv_bytes_per_token("meta-llama/Llama-3.1-8B")
    print(f"KV bytes/token: {bpt:,} ({bpt/1024:.0f} KB)")
    for gpu, vram in [("RTX 4090", 24.0), ("A100 80GB", 80.0)]:
        toks, seqs = capacity(vram, 16.06, 1.0, bpt, ctx_per_seq=2048)
        print(f"  {gpu:10s} {toks:>8,} cached tokens -> {seqs:>4} concurrent 2k-token chats")
```

```console
KV bytes/token: 131,072 (128 KB)
  RTX 4090     53,406 cached tokens ->   26 concurrent 2k-token chats
  A100 80GB   479,736 cached tokens ->  234 concurrent 2k-token chats
```

![A stacked breakdown of a 24 gigabyte card showing weights, runtime overhead, the remaining cache space and what a paged allocator recovers](/imgs/blogs/what-inference-engineering-is-4.webp)

### Where the concurrency actually goes

Those capacity numbers are the *theoretical* ceiling: what you get if every byte of free VRAM is usable for cache and nothing is wasted. Real allocators do not achieve that, and the gap is the single largest source of "why can I only serve eight people on a card that should hold twenty-six".

A naive implementation reserves a contiguous cache buffer per sequence, sized for the *maximum* the sequence might generate. A request that might produce 2,048 tokens but stops after 90 has reserved 22 times what it used, and that reservation cannot be lent to anyone else. Add internal fragmentation from padding to a fixed shape, and external fragmentation from freed buffers of the wrong size, and the waste compounds. The vLLM paper (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) measured exactly this and reported that existing systems wasted **60% to 80% of KV cache memory** to fragmentation and over-reservation, and that fixing it with a paged allocator yielded 2–4x higher throughput at the same latency. That is not a kernel optimization. That is an allocator, borrowed wholesale from operating-system virtual memory, and it is the single largest published win in the field. Track B builds one from scratch; the existing [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) post covers how the production version behaves if you want the operator's view first.

The general shape of the memory budget, which you should be able to recite:

| Consumer | Size for Llama-3.1-8B bf16 | Grows with | Controlled by |
| --- | --- | --- | --- |
| Weights | 16.1 GB | Model size, dtype | Layer 1: quantization, format |
| CUDA context + workspace | ~0.5–1.0 GB | Library versions, graph capture | Layer 2: fewer, fatter kernels |
| Activations (prefill peak) | Hundreds of MB, spiky | Prompt length x batch | Layer 3: chunked prefill |
| KV cache | 128 KB per token, unbounded | Total live tokens | Layer 3: paging, eviction, quantization |
| Fragmentation | 0% to 80% of the cache | Allocator design | Layer 3: block allocator |

The last row is the one people forget, and it is the only one whose size is entirely a consequence of your own code.

## 6. The scoreboard: seven numbers, and what each one hides

You cannot optimize what you cannot name, and LLM inference has a specific vocabulary that is worth getting exactly right, because vendors and papers use these terms with subtly different definitions and the differences are where the marketing lives.

![A five-row table matching each inference metric to what it measures, what it conceals and which layer determines it](/imgs/blogs/what-inference-engineering-is-5.webp)

**TTFT — time to first token.** From the moment the request arrives at your server to the moment the first token byte is written to the response stream. This is the number a user experiences as "did it hear me". It is the sum of queue time, prefill compute, sampling, detokenization and the first flush of the stream. *What it hides:* the split between queueing and computing. A TTFT of 900 ms could be 900 ms of prefill on a long prompt, or 60 ms of prefill after 840 ms of waiting behind other people. These have completely different fixes — one is a prefill problem, the other is a scheduler problem — and no aggregate TTFT number distinguishes them. Always instrument the split.

**TPOT / ITL — time per output token, or inter-token latency.** The steady-state gap between consecutive streamed tokens after the first. Some sources define TPOT as `(end_to_end - TTFT) / (output_tokens - 1)`, which is the mean; ITL usually refers to the individual gaps, whose distribution matters because a single 400 ms stall is visible to a reader in a way that a shifted mean is not. Report both the mean and the p99 of the gaps. *What it hides:* the batch size you were sharing the GPU with, and therefore whether your number generalizes to any other load level at all.

**End-to-end latency.** TTFT plus the whole decode run. Dominated by output length, which means it is mostly a property of the *prompt and the model's verbosity*, not of your engine. A system that got 30% faster can show worse end-to-end latency after a prompt change that made answers longer. Useful for SLOs, useless for engineering attribution.

**Throughput.** Tokens per second, server-wide. This is the number that gets quoted without context most often, and quoting it without a batch size and an arrival process is close to meaningless: we derived 63 tok/s and 1,308 tok/s for the *same model on the same card* in section 4, and both are honest "tokens per second" figures. Always specify: input tokens or output tokens or both, at what concurrency, under what arrival pattern, at what context length.

**Goodput.** Requests per second that meet their latency SLO. This is the metric that actually corresponds to business value, and the DistServe paper (Zhong et al., OSDI 2024, [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) makes the case for it precisely: a system can raise raw throughput by batching harder while pushing so many requests past their TTFT target that the *useful* rate falls. Throughput counts tokens; goodput counts tokens somebody was still waiting for. If you take one metric from this post into your dashboards, take this one.

**Memory ceiling.** The maximum number of concurrent sequences, or total live tokens, before the allocator refuses. Section 5's arithmetic gives the theoretical value; the achieved value tells you how good your allocator is. The ratio between them is one of the most diagnostic numbers you can track.

**Dollars per million tokens.** Everything above, collapsed into one number:

$$
\text{cost per 1M tokens} \;=\; \frac{\text{GPU price per hour}}{\text{tok/s} \times 3600} \times 10^{6}
$$

#### Worked example: what batching is worth in money

Take an A100 80GB and assume a rented price of \$2.00 per GPU-hour (check your own provider — this is a plausible order-of-magnitude figure, not a quote). Use the derived decode rates from section 4.

| Regime | Derived tok/s | Tokens per GPU-hour | Cost per 1M output tokens | Source |
| --- | --- | --- | --- | --- |
| Batch 1, bf16, 2k ctx | 127 | 457,000 | \$4.37 | derived, at \$2.00 per GPU-hour assumed |
| Batch 32, bf16, 2k ctx | 2,647 | 9,530,000 | \$0.21 | derived, at \$2.00 per GPU-hour assumed |

The engineering work between those two rows is worth a **21x** cost reduction on identical hardware with an identical model. No quantization, no new kernels, no bigger GPU. Just an engine that keeps the batch full. This is why the scheduler track sits so early in the series: it is the highest-leverage code you will write.

Two caveats that keep this honest. The batch-32 row assumes you can *find* 32 concurrent requests — at low traffic your batch is small no matter how good your scheduler is, and your real cost per token is much closer to the top row. And it assumes 2,048 tokens of context; at 8k the KV traffic dominates and the advantage compresses. Cost curves in inference are workload curves, which is why Track H builds a fixed prompt suite and reports everything against it.

### How to measure any of this without lying to yourself

Every number above is derived. The moment you start measuring instead, a specific set of mistakes is waiting, and they all produce *optimistic* results, which is the worst direction to be wrong in.

```python
# nanoserve/bench/timing.py — the minimum honest timing harness
import torch, statistics

def timed_decode_steps(step_fn, n_steps: int = 64, warmup: int = 16) -> dict:
    """Time a decode step with CUDA events. step_fn() runs exactly one step."""
    # 1. WARMUP. The first calls pay for autotuning, allocator growth, JIT and
    #    clock ramp. Timing them measures your cold start, not your steady state.
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    # 2. CUDA EVENTS, not time.time(). Kernel launches are asynchronous: without
    #    a sync, time.time() around a launch measures the launch, not the work.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)]
    for i in range(n_steps):
        starts[i].record()
        step_fn()
        ends[i].record()
    torch.cuda.synchronize()  # 3. SYNC BEFORE READING. Events are not valid until done.

    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    ms.sort()
    return {
        "mean_ms": statistics.fmean(ms),
        "p50_ms": ms[len(ms) // 2],
        "p99_ms": ms[int(len(ms) * 0.99)],
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
```

The rules that harness encodes, plus the ones it cannot:

1. **Warm up, always.** The first iterations pay for cuBLAS autotuning, allocator growth, `torch.compile` tracing and GPU clock ramp. Sixteen throwaway steps is a reasonable minimum.
2. **Use CUDA events, or synchronize.** CUDA kernel launches return immediately. A `time.time()` measurement around an unsynchronized launch measures Python, and will happily report a 40x speedup that does not exist.
3. **Lock or report your clocks.** GPUs downclock under thermal and power limits. A benchmark that starts on a cold card and ends on a hot one shows a regression that is not in your code. `nvidia-smi --lock-gpu-clocks` if you can; report the achieved clock if you cannot.
4. **Distinguish open-loop from closed-loop load.** A closed-loop generator keeps N requests in flight and sends a new one when one finishes — so it *cannot* overload the system, and its latency numbers are self-limiting. An open-loop generator sends requests at a fixed arrival rate regardless of whether the system is keeping up, and it is the only kind that can reveal a latency collapse. Almost every benchmark you will read online is closed-loop, and almost every production traffic pattern is open-loop. This mismatch is the number-one reason systems that benchmarked fine fall over in production.
5. **Report the distribution, not the mean.** Inference latency distributions are heavy-tailed by construction: queueing plus variable output length. A mean TPOT hides the preempted request that stalled for two seconds.
6. **Fix the seed and the prompt suite.** Output length varies with sampling, and throughput measured in tokens per second is trivially gamed by generating longer answers.

The load generator itself is the other half, and it is short enough to show in full:

```python
# nanoserve/bench/openloop.py — arrivals do not wait for the server
import asyncio, random, time

async def poisson_load(send, rate_rps: float, duration_s: float, seed: int = 0):
    """Send requests at a Poisson arrival rate; do NOT wait for responses."""
    rng = random.Random(seed)
    tasks, t_end = [], time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        tasks.append(asyncio.create_task(send()))     # fire and keep going
        await asyncio.sleep(rng.expovariate(rate_rps))  # inter-arrival gap
    return await asyncio.gather(*tasks)

async def one_request(client, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    ttft, prev, gaps = None, None, []
    async for chunk in client.stream(prompt, max_tokens=max_tokens):
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t0            # first token defines TTFT
        else:
            gaps.append(now - prev)    # every later gap is an ITL sample
        prev = now
    return {"ttft_s": ttft, "itl_s": gaps, "e2e_s": time.perf_counter() - t0}
```

Track H formalizes all of this into a reusable protocol and ships the full `bench.py`. Until then, the rule is simple: any number you cannot reproduce with a script and a named GPU is a story, not a measurement.

## 7. `nanoserve`: the engine we are going to build

`nanoserve` is a deliberately small, deliberately readable inference engine. It is not trying to beat vLLM; it is trying to be a thing you can read end to end in an afternoon and modify in an hour, so that when you *do* read vLLM's source you recognize every piece.

![A branching diagram showing a request entering a scheduler that forks into a prefill path and a decode path sharing one cache and rejoining at the sampler](/imgs/blogs/what-inference-engineering-is-6.webp)

The whole engine reduces to one function that the server calls in a loop. Everything else is an implementation detail of that function.

```python
# nanoserve/engine.py — the interface every later post fills in
from dataclasses import dataclass, field

@dataclass
class Request:
    rid: str
    prompt_ids: list[int]
    sampling: "SamplingParams"
    out_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)  # Track B fills this
    finished: bool = False

@dataclass
class StepOutput:
    rid: str
    new_token_id: int | None
    finished: bool
    finish_reason: str | None  # "stop" | "length" | "eos" | "preempted"

class Engine:
    def __init__(self, model, cache, scheduler, sampler):
        self.model, self.cache = model, cache
        self.scheduler, self.sampler = scheduler, sampler

    def add_request(self, req: Request) -> None:
        self.scheduler.enqueue(req)

    def step(self) -> list[StepOutput]:
        """Advance every running sequence by exactly one token. One GPU pass."""
        batch = self.scheduler.schedule(self.cache)      # admit, preempt, chunk
        if not batch:
            return []
        hidden = self.model.forward(batch, self.cache)   # prefill + decode, fused
        logits = self.model.lm_head(hidden)
        token_ids = self.sampler.sample(logits, batch)   # temp, top-p, masks
        return self.scheduler.commit(batch, token_ids)   # append, detect stops
```

That is the entire architecture. Every post in this series either fills in one of those four collaborators or changes what one of them is allowed to cost.

| Component | File it lands in | Track that writes it | The question it answers |
| --- | --- | --- | --- |
| Weight loader | `nanoserve/loader.py` | A | How do 16 GB get onto the device, and how fast? |
| Forward pass | `nanoserve/model.py` | A | RMSNorm, RoPE, GQA attention, SwiGLU, by hand |
| Tokenizer boundary | `nanoserve/tokens.py` | A | How do you stream text that arrives as partial UTF-8? |
| KV cache | `nanoserve/cache.py` | B | What layout, what lifetime, whose memory? |
| Block allocator | `nanoserve/blocks.py` | B | How do you stop wasting 60–80% of the cache? |
| Scheduler | `nanoserve/scheduler.py` | C | Who runs this step, and who waits? |
| Sampler | `nanoserve/sampler.py` | D | How does a logit vector become one token id? |
| Grammar masks | `nanoserve/grammar.py` | D | How do you guarantee valid JSON without a retry loop? |
| CUDA / Triton kernels | `nanoserve/kernels/` | E | Where does the hand-written kernel beat the library? |
| Quantized weights | `nanoserve/quant.py` | F | What does 4-bit buy you, and what does it cost? |
| Tensor parallel | `nanoserve/parallel.py` | G | Where do the all-reduces land, and what do they cost per token? |
| HTTP server | `nanoserve/api.py` | I | SSE, cancellation, usage accounting, backpressure |

There is a deliberate ordering discipline here. `nanoserve` gets *correct* before it gets *fast*: Track A ends with a greedy decode loop whose logits match a reference implementation within numerical tolerance, and that correctness harness is what makes every later optimization checkable. A kernel that is 3x faster and 0.1% different in output is not a 3x win; it is an unquantified regression. The harness is what turns it back into an engineering decision.

The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), runs the finished `nanoserve` against vLLM on the same prompt suite and publishes the gap table. I will tell you the conclusion now so nobody is surprised: we will be competitive on batch-1 latency, respectable on moderate-concurrency throughput, and meaningfully behind on the things that took the vLLM team years — kernel maturity, prefix-cache hit rates, and the long tail of correctness under preemption. The point was never to win. The point is that after 65 posts you will be able to read the gap table and explain every single row.

## 8. The honesty rule, stated plainly

Here is a commitment about the numbers in this series, and it is worth stating up front because it changes how you should read everything that follows.

**I do not have a GPU cluster, and I will never claim to have measured something I did not.** Every quantitative statement in these 65 posts is exactly one of three things, and the prose will always make clear which:

- **Derived** — computed from a formula shown in the text, with the arithmetic visible. The 63 tok/s ceiling in section 4 is derived: 16.06 GB divided by 1,008 GB/s. You can check it in your head.
- **Cited** — taken from a paper, a vendor datasheet, an official benchmark, or a model card, named in the text with a link. The 1,008 GB/s came from NVIDIA's published specification, not from a card I own.
- **Reproducible by you** — accompanied by a script and an expected range on named hardware. "On an RTX 4090 you should land in the 40–55 tok/s band" is a prediction, stated as a prediction, that you can falsify in ten minutes.

Tables of results carry a `Source` column saying which. Where an estimate is order-of-magnitude, it says so.

This is a feature, not a limitation. A derived number carries its own explanation: when your measurement disagrees with the formula, the *gap* is the interesting finding, and you know exactly which term to interrogate. A benchmark table with no derivation behind it tells you what happened once, on hardware you do not have, with a software version you cannot reconstruct, and gives you nothing to reason with when your system behaves differently. The formula generalizes; the benchmark does not. Half the value of this series is teaching you to compute the expected number *first* and treat the measurement as a check on your understanding rather than a substitute for it.

## 9. Who this is for, what you need, and what this is not

**This series is for you if** you have called `model.generate()` and want to know what it did; if you operate an inference endpoint and want to stop guessing which flag to change; if you are being asked how much a feature will cost per user and cannot answer; or if you simply want to write an inference engine because that is a satisfying thing to have written.

**What you need:**

- **Python and PyTorch.** Comfortable with tensors, shapes, `nn.Module`, and enough CUDA-adjacent vocabulary to know that `.cuda()` is asynchronous. Nothing more.
- **A GPU is helpful, and optional.** Everything through Track B runs on a CPU with a small model — slowly, but correctly, and correctness is the point. From Track C onward a single consumer GPU with 16–24 GB is enough to reproduce every result, and the numbers in the text are computed for the RTX 4090 partly for that reason. Rented A100 and H100 time is used only where the point is a comparison, and every such number is derived or cited.
- **A Llama-3.1-8B-class checkpoint**, or any model with the same structural shape. The spine model is Llama-3.1-8B because its config is public, its GQA layout is representative, and 16.1 GB of bf16 weights is a number that fits on hardware you can buy. Qwen3-8B and Gemma-3-12B appear where an architectural difference matters; a small MoE appears in Track G; DeepSeek-V3 is referenced but never run.
- **Patience with arithmetic.** There is more division in this series than there is CUDA.

**What this series is not:**

- **It is not vLLM documentation.** vLLM appears as a benchmark target and as a reference for "here is how the production version differs from ours". If you want an operator's guide to vLLM, the [model serving series](/blog/machine-learning/model-serving/what-is-model-serving) covers it properly.
- **It is not a training course.** No gradients, no optimizers, no data pipelines. The model is a fixed artifact and our job starts after it exists.
- **It is not a survey.** Techniques appear when `nanoserve` needs them, in the order it needs them, and the ones that did not survive contact with production get a paragraph explaining why rather than a section pretending they are alive.
- **It is not about model quality.** Except in one important sense: any optimization that changes outputs must be measured for quality damage, and Track F takes that seriously because "2x faster and quietly dumber" is the most common way inference optimization goes wrong.

One boundary worth naming: architectures that do not have a growing KV cache at all — hybrid Mamba and linear-attention models — change enough of this post's assumptions that they get their own seven-post treatment in Track K rather than a footnote here.

## 10. Public numbers you can check

Four results from the literature that shaped how modern engines are built. Each is cited, each is checkable, and each will be revisited in depth later in the series.

**PagedAttention and vLLM** (Kwon et al., SOSP 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)). The paper's central measurement is a memory one: existing serving systems wasted 60–80% of KV cache memory to internal fragmentation, external fragmentation and over-reservation. Applying operating-system-style paging — fixed-size blocks, a per-sequence block table, non-contiguous physical storage — recovered nearly all of it and produced 2–4x higher throughput at equal latency versus the then-current systems. The lesson generalizes past the specific numbers: **the biggest published win in LLM inference was a memory allocator, not a kernel.** Track B builds one.

**Continuous batching** (Yu et al., Orca, OSDI 2022, [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu)). Orca introduced iteration-level scheduling: instead of batching requests and running them to completion together, the scheduler re-forms the batch every single decode step, so a finished sequence is replaced immediately by a waiting one instead of leaving a hole. The paper reports large throughput and latency improvements over request-level batching at comparable latency. Every production engine does this now, and it is roughly sixty lines of Python. Track C writes those sixty lines.

**Chunked prefill** (Agrawal et al., Sarathi-Serve, OSDI 2024, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)). The observation is that a long prefill monopolizes a step and stalls every decoder in the batch, spiking their inter-token latency. Splitting the prefill into token-budgeted chunks and interleaving them with decode work smooths the tail at a small throughput cost. This is the cleanest example in the field of an explicit TTFT-versus-TPOT trade being made a tunable knob rather than an accident. Track C post 13 reproduces the frontier.

**FlashAttention** (Dao et al., NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)). Attention was memory-bound, not compute-bound: the cost was materializing the $S \times S$ score matrix in HBM. Tiling the computation and keeping the running softmax statistics in on-chip SRAM removes that materialization entirely. It matters here mostly for *prefill*; the decode-time attention problem is a different shape and gets its own kernel in Track E. If you want the full derivation, [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) covers it.

The pattern across all four is worth naming: **none of them made the math cheaper.** They all made the *memory traffic* cheaper — fewer bytes moved, or the same bytes moved once instead of many times, or memory reserved more tightly. That is what inference optimization is, almost every time.

## 11. When to build this yourself, and when to just run vLLM

I am going to spend 65 posts building an engine, so let me be blunt about when you should not.

**Use a production engine — vLLM, SGLang, TensorRT-LLM — when:**

- You are shipping a product and the model is a standard architecture. The engines are mature, the kernels are years ahead of anything you will write, and the throughput gap is real money.
- Your problem is operational: scaling, cold starts, multi-tenancy, observability, quota. Those are serving problems and this series does not solve them.
- You need features that took teams years — prefix caching that actually holds up under load, multi-LoRA batching, speculative decoding with correct rollback, tensor parallelism that does not deadlock on a rank failure.
- Nobody on your team can be on the hook for a decode-loop bug at 2 a.m. An inference engine is infrastructure, and infrastructure you wrote is infrastructure you maintain.

**Build, or at least read and modify, when:**

- Your architecture is not supported yet. A new attention variant, a custom routing scheme, an unusual cache structure. Someone has to add it, and that someone needs everything in this series.
- You are debugging behavior no flag explains. Understanding the scheduler is the only way to explain why p99 doubled at 32 concurrent users while utilization stayed flat.
- You are making an architecture decision with inference consequences — KV head count, context length, MoE expert count, quantization scheme. These are training-time decisions with serving-time bills, and someone has to compute the bill in advance.
- You need to evaluate an engine honestly. Reading a benchmark table and knowing which numbers are load-bearing requires knowing what each one hides.
- You want to be the person in the room who can say what the number *should* be before anyone runs anything. That is the whole skill.

The honest summary: **most people should run vLLM and understand it deeply.** This series is how you get to the second half of that sentence, and building the small version is the fastest route I know.

## The roadmap: 11 tracks, 65 posts

![A branching hierarchy grouping the eleven series tracks into building the engine, making it fast, and proving it works](/imgs/blogs/what-inference-engineering-is-7.webp)

| Track | Posts | Theme | What `nanoserve` gains |
| --- | --- | --- | --- |
| **A** | 1–5 | From weights to a token | Loader, forward pass, tokenizer boundary, a baseline decode loop |
| **B** | 6–10 | The KV cache you write yourself | Contiguous cache, memory math, paged blocks, prefix sharing, eviction |
| **C** | 11–15 | Batching and the scheduler | Continuous batching, chunked prefill, scheduling policy, admission control |
| **D** | 16–21 | The decoding layer | The sampler zoo, determinism, FSM masks, grammars, streaming structured output, stop conditions |
| **E** | 22–28 | CUDA and kernels for inference | RMSNorm and RoPE kernels, the KV append kernel, paged attention, decode GEMV, dequant-fused GEMM, Triton |
| **F** | 29–33 | Precision and compression | GGUF/AWQ/GPTQ loading, FP8 and FP4, KV cache quantization, quality measurement, CUDA graphs |
| **G** | 34–39 | Big models | Tensor and pipeline parallel, MoE routing, MLA, long context, multimodal |
| **K** | 59–65 | Hybrid attention and SSM inference | A second cache for recurrent state, scan-versus-recurrence, state rollback, hybrid scheduling |
| **H** | 40–45 | Experiments on real models | A benchmark protocol and `bench.py`, then sweeps across the model and hardware matrix |
| **I** | 46–51 | The API, the platform edge, operations | OpenAI-compatible API, prompt caching, LoRA swapping, the cost model, reliability, observability |
| **J** | 52–58 | Case studies, hardening, capstone | Five production failures diagnosed end to end, a test suite, and the final vLLM comparison |

Read it in order if you want the engine to make sense as a thing that grows. Jump if you have a specific problem: latency at low load is Tracks A and E; throughput under concurrency is B and C; wrong or malformed output is D; memory limits are B and F; cost is H and I; "it was fine until it wasn't" is C post 15 and J post 52.

The next post picks up exactly where this one stops — with 16.1 GB of weights sitting in a directory, and the question of how they get onto a GPU without a ninety-second cold start.

## Key takeaways

1. **Inference engineering is a design discipline, not an operations or profiling discipline.** It asks what each layer should cost, and its output is a component, not a flag setting.
2. **An LLM request is two programs.** Prefill is compute-bound, parallel and GEMM-rich. Decode is bandwidth-bound, sequential and GEMV-shaped. Nearly every technique helps exactly one of them.
3. **Decode time has a floor you can compute in ten seconds:** weight bytes divided by HBM bandwidth. For an 8B bf16 model that is about 16 ms per token on an RTX 4090 and about 8 ms on an A100. If your number is far worse, something above the hardware is wrong.
4. **Batching does not make a step faster; it makes the read pay for more tokens.** Arithmetic intensity at batch $B$ is $B$ FLOPs per byte against a ridge point above 150. That gap is why the scheduler is the highest-leverage code in the engine.
5. **The KV cache is a per-token tax you never explicitly allocated:** $2 \cdot L \cdot H_{kv} \cdot d \cdot b$ bytes, which is 128 KB per token for Llama-3.1-8B. It sets your concurrency ceiling, and a naive allocator throws most of it away.
6. **Tokens per second is not a measurement without a batch size, a context length and an arrival process.** The same card and model honestly produce 63 tok/s and 1,308 tok/s.
7. **Goodput is the metric that maps to value.** Throughput counts tokens; goodput counts tokens somebody was still waiting for.
8. **Measure with warmup, CUDA events, locked clocks and open-loop arrivals**, or do not report the number. Closed-loop benchmarks cannot show you a latency collapse, and production traffic is open-loop.
9. **Every number should be derived, cited, or reproducible.** When a measurement disagrees with the formula, the gap is the finding.
10. **Most teams should run vLLM and understand it deeply.** Building the small version is the fastest way to earn the second half of that sentence.

## Further reading

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023. The fragmentation measurement and the paged allocator that fixed it.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022. Iteration-level scheduling, the origin of continuous batching.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024. The clearest argument for goodput over throughput.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024. Chunked prefill and the TTFT-versus-TPOT frontier.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022. Why attention was a memory problem.
- [NVIDIA A100 Tensor Core GPU datasheet](https://www.nvidia.com/en-us/data-center/a100/) and the [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) — the bandwidth and throughput figures used throughout this post.
- [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B) — the config fields behind every KV cache calculation here.
- [The roofline model: compute-bound versus memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the general form of section 4's arithmetic.
- [Why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) — the operator's view of the same two-phase workload.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where `nanoserve` meets vLLM and the gap table gets published.
