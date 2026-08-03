---
title: "The inference engineering playbook: nanoserve versus vLLM without benchmark theater"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A reproducible capstone for comparing a small inference engine with vLLM, diagnosing the honest performance gap, and deciding when to build or buy."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "ml-systems",
    "vllm",
    "batching",
    "latency",
    "throughput",
    "pytorch",
    "cuda",
    "gpu",
    "mlops",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 29
---

The dangerous version of an inference benchmark is the one with a decimal point. It says `nanoserve: 87.3 tok/s` and `vLLM: 104.8 tok/s`, then quietly changes the model revision, prompt lengths, CUDA graph policy, sampling settings, or GPU between the two commands. The table looks scientific because the numbers are precise. The comparison is not.

![Two identical prompt streams pass through nanoserve and vLLM before correctness and SLO metrics become a build decision](/imgs/blogs/the-inference-engineering-playbook-1.webp)

The diagram above is the mental model: the prompt suite is the controlled input, both engines must first pass a correctness gate, and only then do TTFT, TPOT, p99, goodput, and memory become comparable. This capstone writes the benchmark boundary and the decision tree around the `nanoserve` pieces built throughout the series. It does not pretend that I ran a GPU experiment here. The gap labels below are targets or reader-reproducible hypotheses, not first-hand results.

This is the final post in the [Inference Engineering series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and it resolves the forward link promised there. The series built a loader, a reference forward pass, KV blocks, a scheduler, samplers, grammar masks, kernels, and an API. Here we put that small engine beside vLLM on the same prompt suite, explain why the simple path can be close, explain why a real service can be five times ahead, and decide when owning the code is worth the bill.

## 1. The only comparison that counts

The comparison boundary is not “two HTTP clients point at two URLs.” It is a contract containing the model bytes, tokenizer, prompt token IDs, sampling semantics, hardware, software, concurrency, stopping rules, and measurement clock. Change any one and you have a new experiment.

![A confounded run changes model, hardware, and metric while a controlled run changes only the engine](/imgs/blogs/the-inference-engineering-playbook-2.webp)

The fixed series matrix gives us a useful spine: Llama-3.1-8B on an RTX 4090 for the consumer baseline, an L4 for the cost tier, an A100 80GB SXM for the established data-center reference, and an H100 80GB SXM for a high-bandwidth reference. Qwen3-8B and Gemma-3-12B are architecture checks. Qwen3-30B-A3B or gpt-oss-20b are MoE checks. DeepSeek-V3-family numbers are cited architecture references, not an invitation to invent a local run.

The prompt suite is also fixed: chat with short input and long output, RAG with long input and short output, code completion, and translation. These are not four interchangeable strings. Chat stresses decode residency, RAG stresses prefill and cache capacity, code completion can expose tokenizer and stop behavior, and translation makes output quality and early stopping visible.

### A manifest is part of the benchmark

Write token IDs, not only source text, to the manifest. A tokenizer update can change a prompt from 1,024 tokens to a different count while leaving the text looking identical. The manifest should include the model repository and revision, tokenizer revision, prompt family, input IDs, target output length, temperature, top-p, seed, and request ID.

```python
# nanoserve/bench/manifest.py
from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    input_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int

def write_manifest(cases: list[Case], path: str) -> None:
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "record-the-exact-revision",
        "tokenizer_revision": "record-the-exact-revision",
        "cases": [asdict(case) for case in cases],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")

def read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())
```

This is intentionally boring. The boring part is the experiment. The engine should receive the same `input_ids` in both branches; neither branch should reapply a chat template during timing.

### Correctness precedes speed

Greedy decoding should agree token-for-token for a short deterministic case. Sampling should agree statistically only if random number generation, logits processors, and floating-point behavior are controlled. A speed result from a model that emits different tokens is not an optimization result. It is a different workload.

```python
# nanoserve/bench/parity.py
import torch

@torch.inference_mode()
def compare_logits(nanoserve_logits: torch.Tensor,
                   reference_logits: torch.Tensor,
                   atol: float = 2e-3) -> dict:
    if nanoserve_logits.shape != reference_logits.shape:
        raise ValueError((nanoserve_logits.shape, reference_logits.shape))
    diff = (nanoserve_logits.float() - reference_logits.float()).abs()
    result = {
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "shape": list(diff.shape),
        "pass": bool(torch.allclose(nanoserve_logits, reference_logits,
                                     atol=atol, rtol=atol)),
    }
    if not result["pass"]:
        raise AssertionError(result)
    return result
```

For a reference, use the same checkpoint through `transformers.AutoModelForCausalLM`, but do not confuse that reference implementation with the service under test. The reference supplies a correctness oracle. It is not automatically an efficient serving engine.

## 2. Define the scoreboard before collecting a result

A single “tokens per second” column hides the queue. We need at least five measurements and the boundaries around each one.

| Metric | Definition | Why it matters | Provenance |
|---|---|---|---|
| TTFT | request submit to first streamed token | user-perceived start latency | reproduce: `bench.py` |
| TPOT | mean inter-token time after first token | streaming smoothness | reproduce: `bench.py` |
| aggregate tok/s | generated tokens divided by wall time | capacity, not experience | reproduce: `bench.py` |
| p99 TTFT | 99th percentile submit-to-first-token | tail SLO and queue collapse | reproduce: open-loop harness |
| goodput | requests meeting TTFT and TPOT SLO per second | useful service capacity | derived from trace |
| peak allocated VRAM | max `torch.cuda.memory_allocated()` | admission and safety margin | reproduce: `memory.py` |

The distinction between throughput and goodput is decisive. A system can keep the GPU busy by accepting more work while violating every latency target. If the target is TTFT below 1 second and TPOT below 50 milliseconds, a request outside either bound is not goodput even if its tokens were eventually generated.

### Open-loop and closed-loop load

Closed-loop load waits for a response before sending the next request. It answers “how fast is one client when it never overlaps itself?” Open-loop load submits according to an arrival process independent of completion. It answers “what happens when demand exceeds service capacity?” The latter is where queues, admission, preemption, and p99 become visible.

```python
# nanoserve/bench/clock.py
import time
import torch

def cuda_ms(fn, warmup: int = 10, iters: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values

def summarize_ms(values: list[float]) -> dict:
    values = sorted(values)
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"n": len(values), "p50_ms": at(.50), "p99_ms": at(.99),
            "mean_ms": sum(values) / len(values)}
```

The warmup is not decorative: lazy CUDA module loading, memory allocation, compilation, and cache population can all occur before steady state. Synchronize before timing. Use CUDA events for device work. Keep tokenization, HTTP serialization, and logging outside the GPU interval, then measure them separately at the client boundary.

### The fixed matrix

The matrix below is a test plan, not a result table. Its range notation tells the reader which cells to run. The hardware names are the fixed series names; record the exact SKU, driver, CUDA runtime, PyTorch, Transformers, vLLM version, and model revision in every output.

| Workload | Input / output shape | Batch or arrival sweep | Primary failure mode | Source |
|---|---:|---:|---|---|
| chat | short / long | batch 1, 4, 16; 0.25–1.0 rps | decode bandwidth and queue | reproduce: `suite.yaml` |
| RAG | long / short | 4k, 32k, 64k input | prefill burst and KV admission | reproduce: `suite.yaml` |
| code | medium / medium | batch 1, 4, 8 | stop and tokenizer parity | reproduce: `suite.yaml` |
| translation | medium / medium | batch 1, 4, 16 | early stop and quality parity | reproduce: `suite.yaml` |

Run the same rows on RTX 4090, L4, A100 80GB SXM, and H100 80GB SXM where the model fits. Do not use a cited H100 result to fill an unrun RTX 4090 cell.

## 3. The gap table: what we expect, what we can claim

Here is the capstone table. It is deliberately not a fabricated benchmark. “Within 20%” and “5× behind” are engineering hypotheses that the labeled script is designed to confirm or reject. A future run may move a row. The table is useful now because it says what mechanism each row is testing.

![The gap table separates correctness, simple single-request paths, batch service, and operational maturity](/imgs/blogs/the-inference-engineering-playbook-3.webp)

| Capability | Comparison statement | Expected range or status | Why the gap should exist | Source |
|---|---|---:|---|---|
| deterministic logits | nanoserve should match the reference within tolerance | max abs diff ≤ 2e-3 target | same weights and math are the contract | reproduce: `parity.py` |
| single request, warm decode | aim for within 20% of vLLM on one fixed cell | 0.8–1.2× target, not observed | both may be dominated by weight traffic; launch overhead decides remainder | reproduce: `single_request.py` |
| short prompt, batch 1 TTFT | compare p50 and p99, not only mean | report measured range | prefill kernels, graph capture, and Python boundaries differ | reproduce: `suite.py` |
| continuous batch throughput | nanoserve is expected to be 5× behind in a stressed cell | 0.2× or better is a hypothesis | mature paged allocation, tuned attention, and batch scheduling compound | reproduce: `open_loop.py` |
| KV utilization | compare live tokens, reserved blocks, and fragmentation | no predeclared percentage | block size and workload shape determine it | derived from block counters |
| operations | vLLM is the production baseline | qualitative | retries, health, metrics, model coverage are not one kernel | cited: [vLLM V1 architecture, 2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release) |

The phrase “within 20%” means an interval chosen before looking at the output. It does not mean we are entitled to fill the interval with a plausible number. The phrase “5× behind” names a failure mode to investigate: when many differently sized requests share a device, the small engine may pay for padding, allocator fragmentation, Python scheduling, separate kernel launches, and conservative admission while vLLM composes those concerns.

### Where a 5× gap can be derived, and where it cannot

Some costs are arithmetic. Llama-3.1-8B’s cited configuration has 32 layers, 8 KV heads, head dimension 128, and bf16 elements. Its KV footprint is:

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes/token} = 128\ \text{KiB/token}.
$$

At 8,192 live tokens, that is:

$$
8{,}192 \times 131{,}072 = 1{,}073{,}741{,}824\ \text{bytes} = 1\ \text{GiB}.
$$

That is a derived memory number. It is not a speed number. A speed ratio requires a kernel, a clock, a batch, a shape, a software stack, and a measurement protocol. We can hypothesize that vLLM’s block allocator and fused kernels reduce overhead; we cannot derive “5×” from KV bytes alone.

#### Worked example: a one-gigabyte cache slice

Suppose four requests hold 2,048 tokens each. The live-token total is $4 \times 2{,}048 = 8{,}192$. With the arithmetic above, their bf16 KV storage is 1 GiB before block padding, metadata, activations, and allocator reserve. If `nanoserve` uses 16-token blocks, the number of logical blocks is $8{,}192 / 16 = 512$ exactly. If the requests end at 2,049 tokens instead, each request needs 129 blocks, so the total is $4 \times 129 = 516$ blocks and the capacity is 8,256 token slots. The 64-slot difference is internal fragmentation, derived from the block size and request lengths. It is not an empirical utilization claim.

## 4. Why simple decode can be close

The fairest place for a small engine to compete is a warmed, single-request decode path with fixed shapes. Both engines repeatedly apply the same model weights. If the GPU is bandwidth-bound, the arithmetic workload may be a smaller differentiator than weight traffic, cache reads, launch overhead, and output handling.

That does not mean a Python loop is free. A naïve loop may launch separate kernels for normalization, rotary position embedding, each projection, attention, MLP, and sampling. It may copy logits to the host to choose a token. It may synchronize between stages. A production engine fuses and captures more of the path. The point is that the comparison needs a profile before an explanation.

```python
# nanoserve/decode_step.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class DecodeState:
    token_ids: torch.Tensor
    position: int

@torch.inference_mode()
def greedy_step(model, state: DecodeState, kv_cache) -> tuple[DecodeState, torch.Tensor]:
    token = state.token_ids[:, -1:]
    logits = model(token, position_ids=torch.tensor([state.position],
                                                     device=token.device),
                   past_key_values=kv_cache).logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    return DecodeState(torch.cat((state.token_ids, next_id), dim=1),
                       state.position + 1), next_id

def decode(model, input_ids, kv_cache, steps: int):
    state = DecodeState(input_ids, input_ids.shape[1] - 1)
    output = []
    for _ in range(steps):
        state, token = greedy_step(model, state, kv_cache)
        output.append(token)
    return torch.cat(output, dim=1)
```

This is a correctness-oriented seam, not the final optimized engine. The important design choice is that `past_key_values` and `position` are explicit. It lets the benchmark compare a stable unit rather than accidentally recomputing the entire prefix.

For the reader-reproducible result, run warmup, then collect at least 50 steady-state steps for each engine. Report the median and tail, the exact output length, and whether CUDA graphs were enabled. On a given named GPU, an expected range may be broad: for example, a script can reasonably state “expect a stable ordering and report your observed p50; do not treat any tok/s interval in this article as measured.” A range becomes honest only when it is attached to a script, hardware, model revision, and shape.

The [vLLM performance update](https://vllm.ai/blog/2024-09-05-perf-update), dated 2024-09-05, is a cited public comparison with its own models, hardware, and load setup. It cannot populate this post’s local gap table. Likewise, vLLM’s [Triton attention backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive), dated 2026-03-04, reports an H100 Triton attention result at 100.7% of FlashAttention 3 for a specified Llama 3.1 8B batch-1 long-decode setup. That is evidence about a vLLM backend under that setup, not evidence that `nanoserve` is 100.7% of anything.

## 5. Why the gap expands under service load

The service path is not a longer single-request path. It is a different system. Requests arrive at different times, have different context lengths, finish at different steps, and compete for KV blocks. The scheduler must admit work without reserving an impossible future, release blocks promptly, and keep the batch shape useful.

<figure class="blog-anim">
<svg viewBox="0 0 760 230" role="img" aria-label="A request queue admits and completes requests while the batch changes over time" style="width:100%;height:auto;max-width:860px">
<style>
.j-line{stroke:var(--border,#d1d5db);stroke-width:2}.j-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.j-live{fill:var(--accent,#6366f1)}.j-wait{fill:var(--text-secondary,#6b7280)}.j-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.j-white{fill:#fff}.j-move{animation:j-admit 8s ease-in-out infinite}.j-done{animation:j-finish 8s ease-in-out infinite}.j-new{animation:j-new 8s ease-in-out infinite}
@keyframes j-admit{0%,20%{transform:translateX(0);opacity:0}35%,70%{transform:translateX(150px);opacity:1}85%,100%{transform:translateX(300px);opacity:0}}
@keyframes j-finish{0%,55%{opacity:1}70%,100%{opacity:.12}}
@keyframes j-new{0%,45%{transform:translateX(0);opacity:.12}60%,100%{transform:translateX(-120px);opacity:1}}
@media (prefers-reduced-motion:reduce){.j-move,.j-done,.j-new{animation:none}.j-move{opacity:1;transform:translateX(150px)}.j-done{opacity:.12}.j-new{opacity:1;transform:translateX(-120px)}}
</style>
<line class="j-line" x1="50" y1="172" x2="710" y2="172"/><rect class="j-box" x="50" y="40" width="190" height="105" rx="12"/><rect class="j-box" x="285" y="40" width="190" height="105" rx="12"/><rect class="j-box" x="520" y="40" width="190" height="105" rx="12"/><text class="j-lbl" x="145" y="70">waiting queue</text><text class="j-lbl" x="380" y="70">running batch</text><text class="j-lbl" x="615" y="70">finished</text><circle class="j-wait" cx="100" cy="110" r="18"/><circle class="j-wait" cx="145" cy="110" r="18"/><circle class="j-wait" cx="190" cy="110" r="18"/><circle class="j-live j-move" cx="100" cy="110" r="18"/><text class="j-lbl j-white" x="100" y="116">R3</text><circle class="j-live j-done" cx="340" cy="110" r="18"/><text class="j-lbl j-white" x="340" y="116">R1</text><circle class="j-live" cx="390" cy="110" r="18"/><text class="j-lbl j-white" x="390" y="116">R2</text><circle class="j-live j-new" cx="560" cy="110" r="18"/><text class="j-lbl j-white" x="560" y="116">R0</text><text class="j-lbl" x="380" y="205">each step admits, decodes, and releases capacity</text>
</svg>
<figcaption>The moving request shows continuous batching: a completed sequence frees a slot while a waiting sequence enters the next decode step.</figcaption>
</figure>

The moving element carries meaning: a still table could show the queue, but not the admission and completion that make continuous batching useful. The waiting request should not be admitted merely because one slot is free now; it must have enough blocks for the next step or the scheduler will create a mid-generation emergency.

```python
# nanoserve/scheduler.py
from dataclasses import dataclass, field

BLOCK_TOKENS = 16

@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    generated_tokens: int = 0
    max_new_tokens: int = 128
    blocks: int = 0

    @property
    def live_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    def required_blocks(self) -> int:
        return (self.live_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS

@dataclass
class Scheduler:
    free_blocks: int
    waiting: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)

    def admit(self, limit: int) -> list[Request]:
        admitted = []
        while self.waiting and len(self.running) < limit:
            request = self.waiting[0]
            need = request.required_blocks()
            if need > self.free_blocks:
                break
            self.waiting.pop(0)
            self.free_blocks -= need
            request.blocks = need
            self.running.append(request)
            admitted.append(request)
        return admitted

    def finish(self, request: Request) -> None:
        self.running.remove(request)
        self.free_blocks += request.blocks
        request.blocks = 0
```

The scheduler is intentionally incomplete: production code must grow blocks as tokens arrive, distinguish prompt blocks from decode blocks, handle prefix sharing, and define a preemption policy. But it exposes the right counters. Log `free_blocks`, `waiting`, `running`, `live_tokens`, and `preemptions` at every step.

### The queue law

Little’s law is an explanatory abstraction, not a formula stated by vLLM: $L = \lambda W$, where $L$ is average work in the system, $\lambda$ is arrival rate, and $W$ is average time in the system. If arrival rate rises while service capacity stays fixed, the average number of queued requests and time in system rise together. It explains why p99 can explode while GPU utilization looks flat: the GPU may be fully occupied by an admitted batch while new requests wait in host memory.

## 6. The five places `nanoserve` falls behind

![The service gap stacks operational, scheduler, cache, kernel, and model-math layers](/imgs/blogs/the-inference-engineering-playbook-4.webp)

### 6.1 Allocator and KV lifetime

Contiguous slabs make a request easy to reason about and hard to pack. If request lengths vary, the unused tail of every reserved slab is stranded until the request finishes. Fixed blocks allow non-contiguous physical storage and a per-request block table. They add indirection and metadata, but they turn free space into a shared pool.

The [PagedAttention paper](https://arxiv.org/abs/2309.06180), published 2023-09-15, is the primary citation for the page-like KV design and its memory-sharing motivation. Use it as a mechanism source, not as a promise that a local Python block table matches vLLM’s kernels.

### 6.2 Scheduler policy

First-come-first-served is explainable but can let a long prefill monopolize the device. A token-budget scheduler can reserve a maximum number of prefill tokens per step while allowing decode tokens to make progress. A shortest-remaining-output policy can improve mean completion but starve long generations. A production choice must name its SLO objective.

The right debugging question is not “which batch size is fastest?” It is “which admission decision produced this queue time?” Record queue wait, prefill service time, decode service time, and network time separately. If TTFT rises while GPU service time stays constant, the fix is admission or capacity, not a new attention kernel.

### 6.3 Kernel coverage and shape selection

`nanoserve` can call `torch.nn.functional.scaled_dot_product_attention` and use a sensible cache layout. vLLM can select specialized attention backends, fuse operations, use CUDA graphs for stable shapes, and maintain more tuned paths. The vLLM [Triton attention post](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) explains that its backend changes the work decomposition for query blocks, KV traversal, and split-KV reductions. That is why a page table alone is not PagedAttention performance.

### 6.4 Host boundaries

An innocent `tensor.cpu()` in a sampling path introduces synchronization and copies the full vocabulary. A Python loop that inspects one request at a time prevents the GPU from seeing the batch as a batch. These costs are often invisible in a model-only profile because they live between kernels.

### 6.5 Operational surface

Retries, cancellation, backpressure, health checks, model loading, structured metrics, timeouts, and graceful degradation are not optional in a service. They are also not part of a single decode kernel. A small engine can be fast on a trace and still be a poor production dependency because an OOM is a process restart, a client disconnect leaks blocks, or a rolling deploy invalidates a CUDA graph.

## 7. A decision tree for incidents

![A branching decision tree maps latency, memory, and quality symptoms to the next diagnostic layer](/imgs/blogs/the-inference-engineering-playbook-5.webp)

Start from the symptom. Do not jump straight to a favored optimization.

```python
# nanoserve/diagnostics.py
def diagnose(sample: dict) -> str:
    if sample["quality_failed"]:
        return "freeze weights and tokenizer; run logit parity before speed work"
    if sample["oom"] or sample["preemptions"] > 0:
        return "inspect live KV tokens, block padding, and request lifetime"
    if sample["queue_ms"] > sample["gpu_ms"]:
        return "reduce admission or arrival rate; kernel work is not first suspect"
    if sample["decode_tpot_ms"] > sample["target_tpot_ms"]:
        return "profile decode bandwidth, launches, and batch shape"
    if sample["prefill_ttft_ms"] > sample["target_ttft_ms"]:
        return "inspect prompt length, chunked prefill, and prefill kernels"
    return "compare full trace and check the measurement boundary"
```

### Symptom: p99 TTFT doubles, GPU utilization is unchanged

Split TTFT into queue and service. If queue dominates, the likely causes are arrival rate above capacity, one long prefill blocking decode, admission that ignores future KV growth, or a retry storm. Try a token budget for prefill, a maximum running sequence count, and explicit rejection rather than letting an unbounded queue form. Only after queue time is controlled should you compare kernels.

### Symptom: OOM at 57% reported utilization

“Utilization” is not “safe free memory.” Check the allocator’s reserved bytes, live KV blocks, temporary workspaces, model weights, CUDA graph pools, and fragmentation. A 24 GB RTX 4090 is not a 24 GB KV cache. Use the derived KV law and leave a headroom policy. If a request needs 200k tokens, its KV requirement can exceed the entire card even when the model weights fit.

### Symptom: outputs differ at batch 8

Freeze sampling and compare logits after each layer on a short prompt. Check padding masks, position IDs, KV block order, RoPE positions, dtype conversions, and batch-dependent kernel paths. The vLLM [bitwise-consistent train/inference post](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference), dated 2025-11-10, describes why batch-dependent kernel selection can change results. It reports a 2.4× cost for its determinism experiment; that number belongs to the cited setup, not to `nanoserve`.

### Symptom: single-request speed is good, batch speed is poor

Inspect padding, launch count, effective tokens per step, block-table reads, and scheduler churn. A batch of 16 requests with lengths 32 through 8,192 is not one shape. Continuous batching helps only if the scheduler and kernels exploit the shape instead of paying for the longest member.

## 8. The reproducible harness

![The benchmark timeline puts environment capture, parity, warmup, steady measurement, tail analysis, and publication in order](/imgs/blogs/the-inference-engineering-playbook-6.webp)

The harness should produce JSON, not a screenshot. Store one record per case and one record for the environment.

```python
# nanoserve/bench/run_case.py
import json
import os
import platform
import time
import torch

def environment() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "driver_visible": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "unknown"),
    }

def run_case(engine, input_ids, max_new_tokens: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    first = None
    output = engine.generate(input_ids, max_new_tokens=max_new_tokens,
                             on_token=lambda _: None)
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return {
        "output_tokens": int(output.shape[-1] - input_ids.shape[-1]),
        "elapsed_s": elapsed,
        "ttft_ms": engine.last_ttft_ms,
        "tpot_ms": engine.last_tpot_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }

def save(record: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
```

The `first` variable above is intentionally a reminder to wire a real stream callback; a production harness should timestamp the first token at the client boundary and independently timestamp the first device event. Keep all measured fields clearly labeled as outputs of the reader’s run.

### Reproduce protocol

Run each engine in a fresh process or document why a shared process is safe. Pin the model revision. Disable unrelated background load. Warm until compilation and allocation settle. Use `torch.cuda.synchronize()` before host timestamps and CUDA events around device regions. Collect enough repetitions for p50 and p99; a small sample cannot support a p99 claim. Record clocks if the platform permits it, but do not claim clocks were locked unless the output proves it.

For open-loop load, generate arrivals from a declared rate, keep the prompt sequence fixed, and report offered load as well as completed load. For closed-loop load, report the number of clients. The expected range in a reproduce row is a reader instruction, not an unverified local claim: the reader should run the script and publish the observed interval, hardware, and software envelope.

### Cost arithmetic

If a GPU rental price is $g$ dollars per GPU-hour and the measured service rate is $r$ generated tokens per second, the derived cost is:

$$
\text{cost per million tokens} = \frac{g}{3600r} \times 1{,}000{,}000.
$$

For an illustrative, explicitly hypothetical rate of $2 per GPU-hour and 100 tok/s, the arithmetic is $2/(3600 \times 100) \times 1{,}000{,}000 = \$5.56$ per million generated tokens. The price and rate are assumptions for the example, not a quote or run. Real economics must include utilization, idle time, replicas, storage, networking, and prompt-token work.

## 9. Case studies and public evidence

### PagedAttention: memory management is a first-class mechanism

The PagedAttention paper describes virtual-memory-like KV blocks and reports its own comparisons under its stated model, hardware, and workload. Its durable lesson is not a universal percentage. It is that KV allocation policy changes the number of requests that can be resident, which changes batching and therefore service capacity. `nanoserve` should reproduce the block accounting first and claim speed only after a kernel-backed experiment.

### vLLM V1: composition beats a local trick

The [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release), dated 2025-01-27, describes architectural changes and reports up to 1.7× higher throughput for its own V1 versus V0 under its ShareGPT setup. “Up to” and “under its setup” are the important words. The result supports the thesis that a runtime’s scheduler, compilation, and execution model compose. It does not supply a multiplier for the capstone’s hardware matrix.

### vLLM speculative decoding: an optimization can reverse under load

The [speculative decoding post](https://vllm.ai/blog/2024-10-17-spec-decode), dated 2024-10-17, reports up to 1.5× for a draft-model setup and up to 2.8× for prompt lookup in specified workloads. The same digest records slowdowns at high QPS in those experiments because draft overhead outweighed acceptance. This is a useful warning for the playbook: an optimization must be plotted against load, not advertised from its best cell.

### vLLM Triton attention: backend specialization is not a wrapper detail

The [Triton attention backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive), dated 2026-03-04, describes query blocks, paged KV traversal, split-KV reductions, and persistent-kernel alternatives. It reports 100.7% of FlashAttention 3 for a specified H100 setup and notes that no single configuration dominates. That caveat is exactly why an honest table needs prompt shape, batch, input length, output length, and backend flags.

#### Worked example: a cost row that stays honest

Suppose a reader runs the same 100 tok/s service on an A100 and pays a hypothetical $1.20 per GPU-hour. The derived cost is $1.20/(3600 \times 100) \times 1{,}000{,}000 = \$3.33$ per million output tokens. If the measured rate falls to 50 tok/s at the SLO boundary, the same arithmetic gives $6.67 per million. Neither number says anything about prompt-token cost or quality. They are two transparent scenarios, not benchmark results.

## 10. Build versus buy

![A three-column grid places learning, custom policy, and production service on a nanoserve-to-vLLM boundary](/imgs/blogs/the-inference-engineering-playbook-7.webp)

Build `nanoserve` when the artifact is understanding, a research mechanism, a narrow hardware target, or a genuinely unusual policy that cannot be expressed at a router boundary. Keep the code small enough that a logit parity test can explain every change. The value is control and learning, not pretending a small engine has the ecosystem of a production runtime.

Use vLLM when the requirement is a maintained production service, broad model support, continuous batching, paged KV management, mature attention backends, metrics, cancellation, model loading, and an OpenAI-compatible API. The right customization order is: tune configuration; measure; place policy in a router or admission layer; contribute upstream if the missing mechanism is broadly useful; only then own the engine core.

The hybrid is often best. Put vLLM behind a thin service boundary and keep the product-specific policy outside it: tenant quotas, prompt routing, model selection, SLO admission, cache isolation, and cost accounting. You keep production machinery while owning the decisions that are actually differentiating.

Do not choose based on the most flattering one-request plot. Choose based on failure recovery, operator time, model velocity, security boundaries, and the cost of being wrong at 2 a.m. A five-times slower research engine can be correct for a one-person learning project. A twenty-percent faster engine with no cancellation path can be a disastrous production purchase.

### A practical ownership boundary

There is a useful boundary between code that benefits from being owned and code that benefits from being shared. Own the policy that expresses your product: which tenant gets admission, which model handles a request, what context budget a plan may consume, when a request degrades from long context to retrieval, and how goodput is charged. Those policies are close to product semantics and change with the business.

Be cautious about owning the mechanism that makes the policy fast: the CUDA attention kernel, the block allocator, the graph-capture catalog, the distributed communicator, and the failure recovery path. Each mechanism has a large interaction surface. A local change to block size can affect cache hit rate, attention tile shape, fragmentation, graph shapes, and the maximum number of active sequences. A local change to retries can duplicate GPU work and make a p99 incident look like a kernel regression. A local change to quantization can improve memory while changing task accuracy.

That boundary also suggests how to structure `nanoserve`. Keep model math and cache interfaces narrow. Put counters at every ownership boundary. Make the scheduler return an admission decision with a reason, not only a list of request IDs. Make the allocator expose requested blocks, granted blocks, released blocks, and shared blocks. Make the runner expose whether work was prefilling, decoding, sampling, or waiting. A future comparison then explains itself from its trace.

The same rule applies to vLLM integration. If a router needs tenant-aware admission, it can often make that decision before calling the engine. If it needs a new model architecture or a new attention kernel, the correct path may be an upstream contribution rather than a private fork. A fork creates a second release stream, and the cost is not only merge conflicts. It is also revalidating numerical parity, driver compatibility, model loading, security fixes, and every benchmark after an upstream scheduler change.

The build-versus-buy decision is therefore a portfolio decision. A team can own a tiny executable for education and a production runtime for users. The tiny executable keeps the mental model honest; the production runtime keeps incident response humane. The two artifacts should share prompt manifests and correctness tests where practical, but their performance claims should remain separately labeled.

## 11. Hardening the handoff

The most useful capstone artifact is not a leaderboard screenshot. It is a handoff package another engineer can rerun six months later and either reproduce or falsify. That package has four layers: an immutable prompt manifest, an environment record, raw per-request events, and a report generator that computes summaries without editing the raw data.

### Keep raw events append-only

Each event should contain `case_id`, engine name, request arrival time, first-token time, completion time, input token count, output token count, queue time, prefill time, decode time, status, peak memory, and an error code. Store nanosecond timestamps only if the clock source is clear. If the client and server clocks are not synchronized, do not subtract their timestamps and call the result GPU latency. Use server-side monotonic timestamps for service decomposition and client-side timestamps for user-visible latency.

Cancellation is a benchmark case, not only an API feature. Submit a request, cancel it after a declared delay, and verify that its KV blocks return to the free pool. Then submit enough requests to reuse those blocks and check logit parity. A leak can remain invisible in a short throughput sweep because the process has plenty of memory at the start.

### Separate correctness, performance, and resilience reports

One green check should not hide three different outcomes. The correctness report answers whether tokens and logit tolerances match. The performance report answers how long admitted work takes. The resilience report answers what happens when a client disconnects, a request exceeds its context budget, a worker restarts, or the arrival rate exceeds capacity.

```python
# nanoserve/bench/report.py
from collections import defaultdict
import json
import statistics

def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty sample")
    values = sorted(values)
    index = min(len(values) - 1, int(q * len(values)))
    return values[index]

def summarize(path: str) -> dict:
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                groups[(row["engine"], row["family"])].append(row)
    report = {}
    for key, rows in groups.items():
        ttft = [float(row["ttft_ms"]) for row in rows]
        tpot = [float(row["tpot_ms"]) for row in rows]
        report["/".join(key)] = {
            "requests": len(rows),
            "ttft_p50_ms": percentile(ttft, .50),
            "ttft_p99_ms": percentile(ttft, .99),
            "tpot_p50_ms": percentile(tpot, .50),
            "ttft_mean_ms": statistics.fmean(ttft),
        }
    return report
```

Do not round raw events. Round the presentation table and retain the unrounded JSON. Do not report a p99 from fewer than 100 observations as if it were a stable tail estimate; label it as a small-sample percentile. Do not average p99s from separate workers without retaining the request-level distribution. If an engine rejects a request, count the rejection and show it beside completed goodput.

### The regression gates

The CI version of the comparison should start small. A short prompt with a fixed greedy output catches shape and position errors. A two-request batch catches padding and batch-index errors. A block-boundary prompt of 15, 16, 17, and 31 tokens catches off-by-one allocation. A cancellation case catches lifetime bugs. A long-prefix case catches context accounting. A grammar case catches a mask that is correct at the first token but stale after the state advances.

The speed gate should be a threshold only after the environment is controlled. For example, a local change may be rejected if its reader-generated median TPOT regresses by more than a declared percentage on the same machine. Do not use an absolute tok/s threshold across an RTX 4090 and an H100. Do not use a cited vLLM number as a CI baseline for a different model or driver.

### What the capstone should publish after a real run

The future completed report should include the command line, commit hashes, model and tokenizer revisions, GPU name and memory, driver and CUDA versions, prompt-manifest checksum, warmup count, repetition count, load mode, SLO definition, and whether CUDA graphs, prefix caching, quantization, or speculative decoding were enabled. It should show failed cells, not only the cells that fit. It should include a link to the raw trace or a reproducible artifact.

The result table then becomes legible. A reader can tell whether “within 20%” was observed at batch 1 with a 128-token output or at open-loop load with 64 active requests. A reader can tell whether “5× behind” means aggregate decode throughput, p99 goodput, or a cold-start path. A reader can disagree with the interpretation without first rebuilding the experiment from a vague paragraph.

### The negative controls matter

Add at least three deliberately bad controls to the harness. First, run a contiguous-cache variant with the same request mix. It should make fragmentation visible in block counters even if a short batch hides the capacity loss. Second, run a host-sampling variant that copies logits to CPU. Its purpose is not to win; it proves that the measurement can detect a synchronization boundary. Third, run closed-loop load beside open-loop load. If their curves look identical at low offered load and diverge near saturation, the harness is observing queueing rather than merely printing a different timer.

These controls protect against a common review failure: an optimization appears to help because the baseline was accidentally handicapped. A fair comparison can include a deliberately slow baseline, but it must label it as such and keep the production comparison separate. The reader should be able to answer “what changed?” from the diff and the trace, not from a story written after seeing the chart.

Do the same for quality. Compare greedy outputs before comparing temperature or top-p. Compare a short ASCII prompt before a Unicode prompt. Compare a prompt that ends exactly on a block boundary before a prompt that crosses it. When the first discrepancy appears, save the input IDs, position IDs, block table, and logits for that step. A final string mismatch is evidence that something differs; the first layer-level mismatch is a diagnosis.

Finally, publish the inconvenient cells. An OOM, rejected request, p99 timeout, and cancellation leak are not embarrassing exceptions to hide below the table. They define the operating envelope. A production buyer needs to know the envelope more than the maximum single-request number.

That is the actual engineering win of the playbook. `nanoserve` may lose the comparison. If it loses for a named reason — block-table indirection, missing fusion, scheduler policy, host synchronization, or missing operational machinery — the loss is useful. If it wins one cell, that win is useful only when the cell is reproducible. The point is not to make a small engine look large. The point is to make every trade-off visible enough that build versus buy becomes a technical decision.

## Key takeaways

1. Freeze model revision, tokenizer, token IDs, hardware, software, sampling, and stopping rules before comparing engines.
2. Make logit parity a gate. A faster engine that produces different tokens is not yet an optimization.
3. Report TTFT, TPOT, p99, goodput, and peak memory. Aggregate tok/s is only one capacity number.
4. Separate queue time from GPU service time before changing a kernel.
5. Derive KV capacity from model geometry; never infer safe capacity from a utilization percentage.
6. Treat “within 20%” and “5× behind” as predeclared hypotheses until a labeled reader run confirms them.
7. Cite public vLLM numbers with their date and setup; never relabel them as `nanoserve` results.
8. Build the small engine to learn or own a narrow differentiator. Buy the production runtime when breadth and reliability are the product.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is)
- [The scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem)
- [Experiment: Llama 3.1 8B on a single RTX 4090](/blog/machine-learning/inference-engineering/experiment-llama-3-8b-on-a-single-4090)
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)
- [A reproducible benchmark setup](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark)
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)
- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
