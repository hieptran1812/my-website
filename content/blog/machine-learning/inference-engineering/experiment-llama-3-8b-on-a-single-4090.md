---
title: "Experiment: Llama 3.1 8B on a single RTX 4090"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Design a reproducible batch, context, and dtype sweep for Llama 3.1 8B, then turn KV arithmetic and roofline reasoning into a safe 4090 serving choice."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "llama",
    "kv-cache",
    "pytorch",
    "gpu",
    "batching",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 30
---

The dangerous question in inference work is not “how many tokens per second does this model produce?” It is “under exactly which memory, batch, context, and measurement conditions does that number exist?” A single RTX 4090 can be a wonderfully honest laboratory for that question: it has enough memory to make an 8B model practical, but not enough to hide sloppy KV-cache accounting or an unbounded batch.

![The experiment connects Llama model geometry to a controlled sweep and then to a safe serving decision](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-1.webp)

The diagram above is the mental model: model geometry determines the cache law; the grid turns that law into experiments; the roofline explains the curve; and the result becomes an admission policy. By the end, we will have a `nanoserve`-oriented benchmark skeleton, a memory admission check, a batch/context/dtype matrix, and a way to decide whether the next optimization should be batching, a kernel, or less context. The post is a design and a reproducible recipe. I do not claim to have run it: there is no first-hand GPU result here.

This is post 41 in the [Inference Engineering series](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It assumes the cache derivation in [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), and it points forward to the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook). The concrete target is Meta’s Llama 3.1 8B Instruct on one GeForce RTX 4090, with a small engine called `nanoserve` rather than a wrapper around a production server.

## 1. Define the experiment before touching CUDA

The first failure mode is a benchmark that changes three things at once. A command is launched with a different context length, a different quantization, and a different concurrency level; the output number is then described as “the model’s speed.” That is not an experiment. It is a number with its causes removed.

The model card from [Meta’s official Llama 3.1 repository](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) identifies the 8B model, its autoregressive transformer architecture, GQA, and a 128k context length. The public configuration exposes the values that matter to cache arithmetic: 32 transformer layers, 32 attention heads, 8 key/value heads, and a head dimension of 128 in [the model configuration](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json). Those are cited facts, not measurements from this post.

The experiment therefore has three independent axes:

| Axis | Values | What it isolates | Source |
|---|---|---|---|
| Decode batch | 1, 2, 4, 8, 16, 32 | Weight reuse and scheduler pressure | reproduce: `bench.py` below |
| Prompt context | 1k, 4k, 8k, 32k, 64k | Prefill work and KV residency | reproduce: `bench.py` below |
| Cache/model dtype | bf16, fp16 | Bytes per weight and cache element | reproduce: `bench.py` below |

The exact grid is intentionally a set of requested cells, not a promise that every cell fits. A cell that cannot be admitted is a useful result. “OOM at batch 16 and 32k context” is a capacity boundary; silently reducing the batch to make the command finish is data corruption.

The primary response metrics are time to first token (TTFT), time per output token (TPOT), total generated tokens per second, peak allocated VRAM, and rejected cells. TTFT is submit-to-first-token latency. TPOT is average inter-token time after the first token. Aggregate decode tok/s is useful for capacity planning, but it must never replace per-request latency: a batch can raise aggregate tok/s while making a single user wait longer.

### The workload contract

Every cell uses the same tokenizer, prompt bytes, generation settings, output length, random seed, and stopping rule. Use four named prompt families from the series protocol: chat with short input and long output, RAG with long input and short output, code completion, and translation. The prompt generator should emit token IDs to a manifest so a later run can replay the exact IDs. Text alone is insufficient because tokenizer revisions can change the token count.

Use a fixed output length for the throughput sweep. If one request stops at 17 tokens and another at 128, “tokens per second” contains a hidden length distribution. A separate quality check can use natural stopping, but the performance matrix should use an explicit maximum and report early-stop counts.

### What is and is not being compared

`nanoserve` is the system under construction: its job is to expose the mechanics. [vLLM’s 2024 performance update](https://vllm.ai/blog/2024-09-05-perf-update) is a public comparison point, but it benchmarks Llama 3 8B on A100 and H100 hardware, not this single-4090 design. The vLLM team reports TTFT, TPOT, and throughput under a defined high-QPS setup; those are cited results with different hardware and should not be pasted into our result table.

The fair question is: can the small engine explain its own curve, and how far is that explanation from the production target? We will not call a vLLM number a `nanoserve` result.

## 2. The memory ceiling is arithmetic, not a feeling

![The 24 GB budget is divided between fixed weights, runtime reserve, the growing KV cache, and admission headroom](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-2.webp)

The 4090’s [official NVIDIA product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) lists 24 GB of GDDR6X memory and 1008 GB/s memory bandwidth. The first number is a hardware limit. It is not the amount a process can safely allocate for model weights and KV blocks, because the CUDA runtime, allocator, activations, temporary workspaces, and fragmentation also need space.

For Llama 3.1 8B in bf16, the simple weight floor is derived as:

$$
8{,}000{,}000{,}000\ \text{parameters} \times 2\ \text{bytes/parameter} = 16{,}000{,}000{,}000\ \text{bytes} \approx 14.9\ \text{GiB}.
$$

That is a rounded 8B calculation, not an assertion about the exact serialized checkpoint size. The checkpoint has metadata and implementation details; the process has runtime allocations. The calculation tells us the scale of the floor.

The KV law is more useful because it is exact once the architecture and dtype are fixed:

$$
B_{\text{KV/token}} = 2 \times L \times H_{kv} \times d_{head} \times b.
$$

The leading 2 counts K and V. For Llama 3.1 8B, the cited configuration gives $L=32$, $H_{kv}=8$, and $d_{head}=128$. For bf16 or fp16, $b=2$ bytes:

$$
2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KiB/token}.
$$

This yields the most important capacity checkpoints:

| Context per sequence | KV arithmetic | Result | Source |
|---:|---|---:|---|
| 1,024 | $1{,}024 \times 131{,}072$ bytes | 128 MiB | derived |
| 8,192 | $8{,}192 \times 131{,}072$ bytes | 1 GiB | derived |
| 32,768 | $32{,}768 \times 131{,}072$ bytes | 4 GiB | derived |
| 65,536 | $65{,}536 \times 131{,}072$ bytes | 8 GiB | derived |
| 131,072 | $131{,}072 \times 131{,}072$ bytes | 16 GiB | derived |

The batch multiplier is real. Four sequences at 8k context need $4 \times 1\ \text{GiB}=4\ \text{GiB}$ of KV, before allocator padding. A 4090 cannot host 14.9 GiB of rounded bf16 weights, 16 GiB of 128k KV, and any runtime reserve at once. The model card’s 128k context is a model capability, not a claim that one consumer GPU can serve a full 128k request in bf16 alongside the weights.

#### Worked example: an 8k admission check

Suppose the measured process baseline after loading the model is 17.2 GiB and the engine reserves 1.0 GiB for activations and temporary workspace. Those are reader measurements, not numbers I observed. The derived remaining budget on a 24 GiB device is $24 - 17.2 - 1.0 = 5.8\ \text{GiB}$. One 8k sequence needs 1 GiB of bf16 KV, so a coarse upper bound is $lfloor 5.8 / 1.0 \rfloor = 5$ sequences. If the block allocator uses 16-token blocks, the logical request still has to be rounded to 512 blocks because $8{,}192 / 16 = 512$; block metadata and fragmentation reduce the safe admission count. The correct policy is “admit at most the measured safe count,” not “the formula says five, therefore five always works.”

### `nanoserve` owns the admission calculation

The engine should calculate bytes from the loaded model configuration, not hard-code “Llama 8B.” This is the first runnable component for this post:

```python
# nanoserve/memory.py
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class CacheSpec:
    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int
    block_tokens: int = 16

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element

    def bytes_for_tokens(self, tokens: int) -> int:
        blocks = ceil(tokens / self.block_tokens)
        return blocks * self.block_tokens * self.bytes_per_token

spec = CacheSpec(32, 8, 128, 2)
print(spec.bytes_per_token, spec.bytes_for_tokens(8192) / 2**30)
# 131072 1.0
```

The rounding is deliberate. A paged cache allocates whole blocks, so a 17-token sequence consumes two 16-token blocks unless the implementation has a special partial-block representation. The arithmetic in the paragraph uses exact tokens; the allocator uses block-rounded tokens. That distinction explains “the formula predicted 1 GiB but `nvidia-smi` moved by slightly more.”

## 3. Build the batch × context × dtype grid

The grid is a contract between a model runner and the measurement harness. `nanoserve` should expose a narrow interface: prepare a batch of token IDs, run prefill, run one or more decode steps, and return token IDs plus device timings. Keep tokenization out of the timed GPU region.

```python
# nanoserve/runner.py
from dataclasses import dataclass
import torch

@dataclass
class StepResult:
    next_ids: torch.Tensor
    elapsed_ms: float

class NanoRunner:
    def __init__(self, model, device="cuda", dtype=torch.bfloat16):
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def decode_step(self, input_ids, past_key_values=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.model(input_ids=input_ids, past_key_values=past_key_values,
                         use_cache=True, return_dict=True)
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        end.record()
        end.synchronize()
        return StepResult(next_ids, start.elapsed_time(end)), out.past_key_values
```

This is intentionally close to the reference `transformers` API while leaving the cache table and scheduler as `nanoserve` responsibilities. A production runner would avoid materializing a full vocabulary argmax in a Python-facing path, but this version is runnable and makes the timing boundary explicit.

The dtype sweep must distinguish model weights from cache dtype. A model may run its matmuls in bf16 while storing KV in fp16, or use a quantized weight format with bf16 activations. Record both fields. Calling a cell “fp16” without defining which tensors changed is an experiment bug.

```python
# bench_grid.py
from itertools import product
import json

GRID = {
    "batch": [1, 2, 4, 8, 16, 32],
    "context": [1024, 4096, 8192, 32768, 65536],
    "dtype": ["bf16", "fp16"],
}

def cells():
    for batch, context, dtype in product(GRID["batch"], GRID["context"], GRID["dtype"]):
        yield {"batch": batch, "context": context, "dtype": dtype}

with open("grid.jsonl", "w", encoding="utf-8") as f:
    for cell in cells():
        f.write(json.dumps(cell) + "\n")
```

The script writes 60 requested cells because $6 \times 5 \times 2 = 60$, a derived count. It does not claim that 60 cells will complete. The runner should return a structured status such as `ok`, `oom`, `unsupported`, or `numerical_error` and continue to the next cell after resetting the process. Continuing in the same Python process after an OOM can leave allocator state that makes later cells misleading.

### Prefill and decode are different experiments

Prefill processes the prompt, often with a matrix-shaped workload and substantial parallelism. Decode appends one token per active sequence, reads the existing KV, and produces the next token. A context sweep mostly changes prefill and cache residency; a batch sweep mostly changes decode reuse and scheduler occupancy. A single “end-to-end tok/s” number mixes both.

For each cell, report at least two phases:

| Phase | Input | Primary metrics | Why it matters |
|---|---|---|---|
| Prefill | all prompt tokens | TTFT, prompt tok/s, peak VRAM | long prompts and activation pressure |
| Decode | one token per live sequence | TPOT, aggregate tok/s, p99 | interactive streaming and weight reuse |
| End-to-end | prompt plus fixed output | total tok/s, completion latency | product-level capacity |

Do not divide prompt tokens by decode time and call it throughput. That gives a flattering but meaningless number for RAG prompts.

## 4. Measure without lying to yourself

![A fair generation measurement moves from load and warmup through synchronized CUDA timing to steady decode and reported percentiles](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-3.webp)

GPU work is asynchronous. A Python timestamp taken immediately after a model call usually measures launch overhead, not completion. The minimum reliable timing sequence is: warm the model, record a CUDA event, enqueue the work, record a second event, synchronize the end event, and use the event interval. The timeline above is not ceremony; it is the boundary between a benchmark and a stopwatch around a launch.

```python
# timing.py
import torch

def timed_cuda(fn, warmup=5, steps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / steps
```

The warmup count here is a protocol parameter, not a performance result. Increase it until lazy module initialization, kernel selection, and allocator growth no longer appear in the per-step trace. Keep it in the manifest. A reader reproducing the experiment should know whether “warmup” meant five calls, 50 calls, or a fixed wall-clock interval.

Record the environment with every result:

```python
# environment.py
import platform, subprocess, torch

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
})
print(subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,clocks.current.memory,temperature.gpu", "--format=csv,noheader"], text=True).strip())
```

Clock behavior belongs in the report. A reader can use a fixed power or clock policy if permitted by the machine, but the exact command is host-specific. Never write “locked clocks” unless the output is recorded. A laptop, a desktop with a background display, and a dedicated lab 4090 can have different thermal behavior even when their model and batch are identical.

### Percentiles and load shape

An offline batch sweep answers “how fast can the device drain this fixed set?” It does not answer “what does a user experience when requests arrive?” Add an open-loop workload with a specified arrival process for service behavior. A Poisson arrival rate is a useful baseline; a closed-loop client that submits the next request only after the previous response is a different workload and usually hides queue buildup.

For each request, keep timestamps for arrival, admission, first token, every token, and completion. Then:

$$
\text{TTFT} = t_{first\ token} - t_{arrival},
\qquad
\text{TPOT} = \frac{t_{last\ token} - t_{first\ token}}{N_{output}-1}.
$$

The denominator is $N_{output}-1$ because there are that many intervals between output tokens. If the request produces one token, TPOT is undefined; report it separately rather than dividing by zero.

Little’s law, $L = \lambda W$, is the sanity check for a stable queue: average in-flight requests equals arrival rate times average time in system. If the reported throughput and latency imply a wildly different in-flight count from the scheduler trace, the harness has dropped or duplicated work.

## 5. Roofline reasoning: why batch helps and then stops helping

![Batch one streams weights for one token while a larger batch reuses the same weights across rows before a bandwidth ceiling flattens the curve](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-4.webp)

Arithmetic intensity is FLOPs per byte transferred. A roofline compares the attainable compute rate with the attainable bandwidth rate:

$$
P \le \min(P_{peak},\; \text{AI} \times BW).
$$

For decode, a simplified dense layer has a weight matrix used by a batch of $B$ token rows. If each weight element is loaded once per step and contributes to $B$ rows, the effective weight traffic per output token decreases approximately with $B$. This is an abstraction for reasoning, not a formula stated by Meta or vLLM. Real kernels also move activations, read KV, write outputs, and choose different tilings.

At batch 1, the matrix-vector shape is skinny. The device cannot amortize a weight load across many independent rows, and launch overhead plus memory traffic matter more than nominal FLOPs. At higher batch, the same weights can feed many rows and GEMM-like tiling becomes more efficient. Eventually the working set, memory bandwidth, tensor-core occupancy, KV reads, or scheduler overhead becomes the new ceiling. The curve rises, bends, and flattens; it does not rise forever.

The 4090’s cited 1008 GB/s bandwidth gives a deliberately optimistic weight-streaming bound. Using the rounded bf16 weight floor of 16 GB:

$$
\frac{16\times 10^9\ \text{bytes}}{1008\times 10^9\ \text{bytes/s}} \approx 0.0159\ \text{s} = 15.9\ \text{ms}.
$$

That is a lower-bound-style intuition for one full weight read at peak bandwidth, not a predicted TPOT. It ignores every other read and write, imperfect utilization, kernel overhead, and the fact that some weights may be cached. The only honest measured bandwidth is:

$$
BW_{achieved} = \frac{\text{bytes attributed to the kernel}}{\text{kernel elapsed seconds}}.
$$

Use Nsight Compute or a profiler to estimate actual traffic, then compare it to the vendor specification. Do not infer bandwidth from tok/s unless the byte model is explicit.

#### Worked example: why “more FLOPs” can be faster

Suppose a batch-1 decode step effectively moves 16 GB of weight data for one token, while a batch-8 step moves the same 16 GB but produces eight tokens, one for each active sequence. The derived weight bytes per output token are 16 GB at batch 1 and $16/8=2$ GB at batch 8. This does not mean the batch-8 step is exactly eight times faster: activations, KV traffic, launch count, and kernel efficiency are not free. It explains why a batch sweep can improve aggregate tok/s even while total work increases.

The experiment should therefore plot both aggregate tok/s and TPOT. Aggregate tok/s can grow because eight users share weight traffic. TPOT can worsen because a single user waits behind the batch. The product decision depends on whether the service is throughput-bound or interactive.

### The animated curve

<figure class="blog-anim">
<svg viewBox="0 0 760 240" role="img" aria-label="Throughput rises with decode batch size and then approaches a flat bandwidth-limited ceiling" style="width:100%;height:auto;max-width:900px">
<style>
.h9-axis{stroke:var(--border,#d1d5db);stroke-width:2}.h9-grid{stroke:var(--border,#d1d5db);stroke-width:1;opacity:.6}.h9-line{fill:none;stroke:var(--accent,#6366f1);stroke-width:5;stroke-linecap:round}.h9-dot{fill:var(--accent,#6366f1)}.h9-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.h9-ceil{stroke:#d97706;stroke-width:2;stroke-dasharray:7 6}.h9-sweep{fill:#6366f1;opacity:.14}
@keyframes h9-grow{0%,12%{stroke-dashoffset:620}75%,100%{stroke-dashoffset:0}}
@keyframes h9-sweep{0%,12%{transform:translateX(0);opacity:.08}75%,100%{transform:translateX(440px);opacity:.22}}
.h9-animated{stroke-dasharray:620;stroke-dashoffset:620;animation:h9-grow 8s ease-in-out infinite}.h9-window{animation:h9-sweep 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.h9-animated{animation:none;stroke-dashoffset:0}.h9-window{animation:none;transform:translateX(440px);opacity:.22}}
</style>
<line class="h9-axis" x1="70" y1="190" x2="710" y2="190"/><line class="h9-axis" x1="70" y1="35" x2="70" y2="190"/>
<line class="h9-grid" x1="70" y1="75" x2="710" y2="75"/><line class="h9-grid" x1="70" y1="125" x2="710" y2="125"/>
<line class="h9-ceil" x1="70" y1="62" x2="710" y2="62"/><text class="h9-label" x="520" y="52">bandwidth ceiling</text>
<path class="h9-line h9-animated" d="M90 178 C150 165 180 125 240 105 S330 78 390 70 S500 63 690 62"/>
<rect class="h9-sweep h9-window" x="88" y="45" width="74" height="150" rx="10"/>
<circle class="h9-dot" cx="100" cy="176" r="6"/><circle class="h9-dot" cx="180" cy="126" r="6"/><circle class="h9-dot" cx="290" cy="86" r="6"/><circle class="h9-dot" cx="430" cy="67" r="6"/><circle class="h9-dot" cx="650" cy="62" r="6"/>
<text class="h9-label" x="82" y="218">batch 1</text><text class="h9-label" x="260" y="218">4–16</text><text class="h9-label" x="600" y="218">32+</text><text class="h9-label" x="12" y="45" transform="rotate(-90 12 45)">tok/s</text><text class="h9-label" x="360" y="238">decode batch</text>
</svg>
<figcaption>As batch grows, each weight load serves more rows; the curve eventually flattens when another resource sets the ceiling.</figcaption>
</figure>

The animation is meaningful because it shows two coupled changes: the active batch sweeps right, and the attainable curve grows until the ceiling. With reduced motion, the completed curve remains visible. The exact height of a reader’s curve is intentionally not drawn as a benchmark result.

## 6. The full reproducible harness

The harness should make a result difficult to fake accidentally. It should save a JSON record for every cell, including failures, and write raw request timings rather than only a final average.

```python
# bench.py
import argparse, json, time
from pathlib import Path
import torch

def cuda_ms(fn):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); value = fn(); end.record(); end.synchronize()
    return float(start.elapsed_time(end)), value

def run_cell(runner, input_ids, output_tokens, warmup=5):
    for _ in range(warmup):
        runner.decode_step(input_ids)
    torch.cuda.synchronize()
    rows = []
    state = None
    for step in range(output_tokens):
        ms, result = cuda_ms(lambda: runner.decode_step(input_ids, state))
        input_ids, state = result.next_ids, result
        rows.append({"step": step, "ms": ms, "batch": input_ids.shape[0]})
    total_ms = sum(row["ms"] for row in rows)
    return {"steps": rows, "total_ms": total_ms,
            "tok_s": (len(rows) * input_ids.shape[0]) / (total_ms / 1000.0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # Model loading and token-ID preparation are intentionally injected here
    # by the nanoserve runner; they are not timed as decode work.
    records = []
    for line in args.manifest.read_text().splitlines():
        cell = json.loads(line)
        try:
            # runner, input_ids = build_runner_and_inputs(cell)
            # result = run_cell(runner, input_ids, output_tokens=128)
            result = {"status": "implement_nanoserve_runner", "cell": cell}
            records.append(result)
        except torch.cuda.OutOfMemoryError as exc:
            records.append({"status": "oom", "cell": cell, "error": str(exc)})
            with torch.no_grad(): torch.cuda.empty_cache()
    args.output.write_text("\n".join(json.dumps(x) for x in records) + "\n")

if __name__ == "__main__":
    main()
```

The deliberately explicit placeholder is a seam, not pseudocode disguised as a result. The `nanoserve` repository owns `build_runner_and_inputs`; the harness owns the protocol and persistence. In a complete local checkout, replace the two commented lines with the model loader and the engine’s batch constructor. The rest of the script is runnable and will emit a manifest-shaped report even before the model runner is wired.

For a real result, use a fresh process per high-risk memory cell. Use `torch.cuda.reset_peak_memory_stats()` immediately before the timed portion and record `torch.cuda.max_memory_allocated()`, plus the process baseline after model load. Report allocated and reserved memory separately; reserved memory can stay high after tensors are freed.

```python
# memory_probe.py
import torch

def memory_snapshot(label):
    torch.cuda.synchronize()
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }

torch.cuda.reset_peak_memory_stats()
print(memory_snapshot("after-load"))
```

Do not use a screenshot of `nvidia-smi` as the only memory record. It includes process and driver effects outside PyTorch’s allocator. Save both views with a timestamp and the cell configuration.

## 7. Read the result table as a curve, not a leaderboard

![Context checkpoints show fixed weight memory while KV memory grows linearly and eventually consumes admission headroom](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-5.webp)

The expected shape is more useful than a fabricated table of tok/s. The following are reader-reproducible expectations, not claimed measurements:

| Observation to test | Expected range or direction | Hardware/setup | Source |
|---|---|---|---|
| bf16 weight floor | about 14.9 GiB for rounded 8B parameters | Llama 3.1 8B, 2 bytes/parameter | derived |
| KV at 8k | exactly 1 GiB before block overhead | 32 layers, 8 KV heads, 128 head dim, bf16 | derived |
| KV at 32k | exactly 4 GiB before block overhead | same | derived |
| decode aggregate tok/s | should rise from batch 1 through a middle batch, then flatten | RTX 4090, fixed context and output | reproduce: `bench.py`; expected qualitative range |
| TPOT | should usually rise or stop improving after saturation | RTX 4090, fixed prompt family | reproduce: `bench.py`; expected qualitative range |
| 128k bf16 admission | expected to fail with a full 8B bf16 model on 24 GB | RTX 4090, no offload | derived capacity argument |

The row “rise then flatten” is a shape expectation from the roofline model. It is not a numeric result. To turn it into a number, the reader must run the script, report the exact driver and PyTorch versions, and publish the raw JSONL.

At fixed context, plot aggregate decode tok/s against batch. Mark the first batch where the incremental gain is small, but define “small” before viewing the plot. For example, a protocol may call the curve saturated when the next batch increases aggregate tok/s by less than a reader-chosen threshold. The threshold is an analysis choice, not a universal constant.

At fixed batch, plot TPOT and peak memory against context. The KV line should be linear in tokens. If peak memory bends sharply upward, investigate activations, block rounding, graph capture, or a leak. If it bends downward, check whether the implementation truncated context or silently skipped requests.

At fixed batch and context, compare bf16 and fp16. The simple KV formula predicts equal bytes because both formats use 2 bytes per element. If the measured cache differs, the implementation is storing a different format or padding differently. This is why dtype labels must be decomposed into weight dtype, activation dtype, and KV dtype.

### A safe cell state machine

![The diagnosis tree routes a slow sweep to memory, bandwidth, or host-side timing tests](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-6.webp)

The state machine for each cell should be boring:

1. Validate the requested geometry against the KV formula.
2. Start a clean worker process.
3. Load weights and record memory.
4. Allocate the requested batch and context.
5. Warm up and check numerical parity.
6. Time prefill and decode separately.
7. Capture peak memory and raw timings.
8. Mark the cell `ok`, `oom`, or `error` and exit cleanly.

If the formula predicts that a cell cannot fit, the harness may skip it with status `predicted_oom`, but at least one boundary cell should be attempted in a controlled process to validate the accounting. Never catch an OOM and continue pretending the allocator state is pristine.

## 8. What `nanoserve` should implement for this experiment

The experiment owns a small but real slice of the engine: configuration introspection, block-rounded admission, a repeatable decode runner, and measurement hooks. It does not need a new attention kernel to answer the first questions.

```python
# nanoserve/admission.py
from dataclasses import dataclass
from math import ceil

@dataclass(frozen=True)
class RequestShape:
    batch: int
    context_tokens: int
    output_tokens: int

def kv_gib(spec, shape):
    tokens = shape.batch * (shape.context_tokens + shape.output_tokens)
    return spec.bytes_for_tokens(tokens) / 2**30

def admit(spec, shape, free_gib, safety_gib=1.0):
    required = kv_gib(spec, shape)
    return {"ok": required + safety_gib <= free_gib,
            "required_gib": required,
            "safety_gib": safety_gib,
            "blocks": ceil(shape.batch * (shape.context_tokens + shape.output_tokens)
                            / spec.block_tokens)}
```

The output length belongs in admission because decode grows the cache. A request with an 8k prompt and a maximum of 8k new tokens can require close to 16k cached tokens, not 8k. If the server admits based only on input length, long completions can evict or kill other requests.

The code above uses a single shared block count for clarity. A real paged allocator tracks blocks per sequence, block tables, and copy-on-write rules. It should reserve by maximum requested length or by a scheduler policy that can stop generation before the reservation is exceeded. “We probably will not generate 8k” is not an admission invariant.

The engine should also expose deterministic sampling for the benchmark. Greedy decoding removes sampling variance from the speed curve. A quality run can use temperature and top-p, but it must keep the random generator on a named device and record the seed. Batch-dependent kernels can still change floating-point reduction order; if exact token parity is a requirement, compare logits with a tolerance and publish the tolerance.

```python
# nanoserve/repro.py
import random, numpy as np, torch

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
```

The seed value here is a protocol example, not a performance result. Determinism has costs and scope. Do not silently enable every deterministic algorithm if the production engine would not use it; instead, run a parity mode and a performance mode, label them, and explain the trade-off.

### A minimal batch scheduler

The scheduler must not confuse a batch with one long sequence. Each row has its own sequence length, block table, position IDs, and finished flag. For the first experiment, use equal context lengths to make the matrix clean, then add ragged prompts as a second stress test.

```python
# nanoserve/batch.py
from dataclasses import dataclass
import torch

@dataclass
class Sequence:
    token_ids: torch.Tensor
    max_new_tokens: int
    finished: bool = False

def pack_decode(sequences, pad_id=0):
    live = [s for s in sequences if not s.finished]
    if not live:
        return None
    ids = torch.cat([s.token_ids[:, -1:] for s in live], dim=0)
    positions = torch.tensor([s.token_ids.shape[1] - 1 for s in live], device=ids.device)
    return {"input_ids": ids, "positions": positions, "sequence_count": len(live)}
```

This intentionally simple representation makes a testable point: `input_ids` is one token per live sequence during decode, while the cache length differs by sequence. A production block table replaces the contiguous `token_ids` storage. If the batch runner pads every sequence to the longest context, the experiment is no longer testing a paged or ragged engine; it is testing padding overhead.

## 9. Case studies and public comparisons

### The vLLM comparison target

The [vLLM 0.6.0 performance post](https://vllm.ai/blog/2024-09-05-perf-update) reports comparisons against other serving systems on A100 and H100, with Llama 3 8B and 70B, and explicitly defines TTFT, TPOT, and high-QPS throughput. The setup is useful because it demonstrates what a public benchmark should disclose. It is not transferable to a 4090: GPU architecture, memory bandwidth, software versions, model revision, and workload differ.

The [vLLM V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) reports up to 1.7× higher throughput for its own V1 versus V0 on Llama 3.1 8B and Llama 3.3 70B under its ShareGPT setup. Again, this is a cited vLLM result, not a `nanoserve` claim. Its lesson for this post is methodological: a number needs a model, workload, hardware, and comparison baseline.

### The model card is not a serving benchmark

Meta’s model card reports model capability evaluations and states the architecture and 128k context. Those scores are useful for identifying the model and its intended behavior; they do not state TTFT, TPOT, or 4090 throughput. A benchmark post that copies a capability score into a performance table has mixed two different kinds of evidence.

### The roofline is a diagnostic, not an oracle

The [roofline explanation in this series](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) provides the general model. For this experiment, the useful consequence is narrow: decode often benefits from batch because weight traffic can be shared across active rows, while prefill has a different shape and should be analyzed separately. If Nsight shows low achieved bandwidth and low occupancy, the next action might be a kernel or batch-shape change. If memory is full, a faster kernel cannot create cache capacity.

### A production target has more moving parts

The production target may have continuous batching, paged attention, prefix caching, CUDA graphs, fused sampling, and more mature admission policies. `nanoserve` should not imitate every feature before the experiment is explainable. First match the reference model’s logits on short sequences, then match memory accounting, then compare speed. Otherwise a faster number can be a model mismatch.

## 10. Failure modes that make the grid untrustworthy

![The graph connects model geometry, cache arithmetic, the controlled grid, roofline interpretation, memory rejection, and the serving recommendation](/imgs/blogs/experiment-llama-3-8b-on-a-single-4090-7.webp)

### Silent truncation

The tokenizer creates a prompt longer than the configured context, and the runner truncates it. The cell is reported as 32k even though the model saw 8k. Save the token count before the model call and reject overlong inputs unless truncation is an explicit separate axis.

### Wrong model revision

“Llama 3.1 8B” is not enough metadata. Save the repository identifier, revision or commit, tokenizer files, Transformers version, dtype, and chat template. The official model card says Llama 3.1 supports 128k context; a local derivative may have a different rope configuration or quantization.

### Cache bytes mistaken for total memory

The KV equation predicts cache payload bytes. It does not include block-table entries, allocator metadata, padding, CUDA graph pools, activations, or temporary attention buffers. Compare the payload prediction to a measured delta after allocation, then explain the gap. Do not “correct” the equation until you know which extra allocation you are comparing against.

### Batch padding

Ragged prompts padded to the maximum sequence length make the long request’s cost appear on every row. Start with equal lengths for the controlled grid; add a ragged workload and report padding tokens. For decode, finished sequences must leave the active batch, or the curve will incorrectly attribute idle rows to useful throughput.

### CPU timing and hidden synchronization

Calling `.item()`, printing a CUDA tensor, or moving logits to CPU inside the loop can force a synchronization. If the benchmark includes that operation, call it out as end-to-end client overhead. If the goal is kernel TPOT, move it outside the timed region. Measure both when the product cares about both.

### Temperature and output variance

Sampling changes the token path and can change early stopping. Use greedy decoding for the performance matrix, then run quality and natural-stop experiments separately. A benchmark that compares one cell’s 128 generated tokens with another cell’s 29 generated tokens is not comparing the same work.

### Thermal drift

The 4090’s cited memory specification is stable; application clocks and temperature are not. Run cells in randomized order or repeat them in a fixed order and record temperature and clocks. If later cells are slower, the explanation may be heat rather than batch size. A reader should see enough repeats to estimate variance, not one “best” run.

## 11. When to reach for this experiment

Run this experiment when you own a single GPU, are choosing a safe default batch, or need to explain why a new kernel changed throughput. It is especially valuable before tuning: the memory curve tells you which cells are possible, the batch curve tells you whether weight reuse exists, and the timing split tells you whether you are optimizing prefill or decode.

Do not run the full 60-cell matrix for every commit. Use a smoke matrix of one short and one long context, batch 1 and a candidate saturated batch, and both dtypes. Run the full grid for a release candidate, a driver change, a model revision, a kernel change, or a scheduler change. Keep raw JSONL and compare distributions, not only a headline average.

Use vLLM instead of `nanoserve` when you need production reliability, broad model support, continuous batching, paged attention, prefix caching, or a maintained OpenAI-compatible service. The point of `nanoserve` is to learn and verify the mechanisms. A production server has already paid the engineering cost of many interactions this post keeps isolated.

On the other hand, do not use a production server to avoid understanding capacity. If a 24 GB GPU rejects a long-context cell, the server’s flag is not magic. It still has to account for weights, runtime reserve, KV blocks, and active sequences. The arithmetic remains the shortest explanation.

### A reproducibility envelope for readers

An expected range is only useful if the reader knows what is allowed to vary. For this post, the named hardware is a stock RTX 4090 with its vendor-listed 24 GB memory and 1008 GB/s bandwidth. The model is the Llama 3.1 8B Instruct revision selected by the reader, with the revision recorded. The software envelope includes the PyTorch, CUDA runtime, driver, Transformers, and tokenizer versions printed by `environment.py`. The prompt manifest is immutable. The engine is `nanoserve`, not a vLLM server called through an OpenAI client.

That still leaves expected variation. The reader may have a different board power limit, a display attached, a different cooling curve, or a different kernel build. An honest report should therefore prefer a range over a single point. One practical protocol is five complete repetitions of each smoke cell after warmup, then report the median and the minimum-to-maximum interval. This is a reader-reproducible expected range protocol; it is not a number this post claims to have observed. For the full grid, three repetitions may be enough to discover regressions, but a p99 latency claim requires more requests than a three-row smoke test.

Do not average across prompt families. A code prompt, a translation prompt, and a RAG prompt can have the same token count while producing different cache access patterns, stopping behavior, and host-side preprocessing costs. Report one row per family, or report a weighted mixture whose weights are declared. If the product traffic is 50% chat, 25% RAG, 15% code, and 10% translation, that mixture is a workload assumption and belongs in the result metadata. It is not a property of Llama.

The benchmark should also separate cold-start from steady state. Loading weights, compiling kernels, constructing CUDA graphs, and populating allocator pools are operational metrics. They matter for autoscaling and failover, but they should not be included in steady-state TPOT. Keep a second record for model load time and the first-request TTFT. A system with excellent steady-state tokens per second can still be unusable if it takes too long to become ready.

Finally, preserve failures. A result directory should contain the cell manifest, environment record, raw request timings, peak-memory snapshots, error trace, and a short interpretation. If a 64k cell is rejected by the admission calculator, the record proves the engine protected existing work. If it is admitted and later OOMs, that is a bug in the accounting or the reservation policy. Both are more valuable than deleting the cell and reporting only the cases that completed.

### What a useful report looks like

The final report does not need a giant leaderboard. It needs a small set of plots and a table that can be audited. Plot one is the derived KV line against context, with measured peak allocated and reserved memory overlaid. Plot two is aggregate decode tok/s against batch, with TPOT on a second axis or in a separate panel. Plot three is TTFT against context for each prompt family. Plot four is a heatmap of cell status: completed, predicted rejection, measured OOM, or numerical failure.

For every plotted point, retain the batch, context, input token count, output token count, weight dtype, activation dtype, KV dtype, sampling mode, prompt family, and repetition number. A point without those fields is not reproducible. For every curve, mark whether its y-axis is a derived expectation, a cited hardware property, or a reader measurement. Mixing a derived 1 GiB KV line with measured peak memory in the same panel is fine, as long as the legend makes the distinction impossible to miss.

This discipline changes the engineering conversation. Instead of saying “the 4090 is slow at long context,” we can say “the payload law predicts 8 GiB of KV at 64k per sequence, the fixed model allocation leaves a measured reserve of a particular size, and the admission policy rejects this batch before the process crosses its safety margin.” Instead of saying “batch 16 is best,” we can say “the tok/s slope flattened after the chosen batch on this driver and prompt family, while TPOT continued to rise; the service should cap admission based on latency, not the throughput maximum.” That is a decision another engineer can challenge and reproduce.

## Key takeaways

- Treat batch, context, and dtype as independent axes; changing all three produces a story, not an experiment.
- For Llama 3.1 8B, the bf16 KV payload is derived as $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes per token, or 128 KiB.
- Eight thousand one hundred ninety-two tokens consume 1 GiB of bf16 KV payload per sequence before block and runtime overhead.
- A 128k context is a model capability cited by Meta, not a promise that a full bf16 request fits beside the weights on a 24 GB card.
- Decode batching improves weight reuse until bandwidth, occupancy, KV traffic, or scheduling becomes the ceiling.
- TTFT and TPOT answer different questions; aggregate tok/s cannot replace either one.
- CUDA events and an end-event synchronization are the minimum credible GPU timing boundary.
- Every result needs a model revision, tokenizer, software stack, hardware, prompt IDs, dtype split, and raw timings.
- An OOM cell is a result. Record it, reset the worker, and do not silently shrink the requested workload.
- Use `nanoserve` to understand the curve; use vLLM when the requirement is a production service.

## Further reading

- [Meta Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), 2024-07-23.
- [Llama 3.1 8B configuration](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json), public model configuration.
- [NVIDIA GeForce RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), accessed 2026-08-03.
- [vLLM 0.6.0 performance update](https://vllm.ai/blog/2024-09-05-perf-update), 2024-09-05.
- [vLLM V1 architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), 2025-01-27.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache).
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).
- [The roofline model for compute-bound versus memory-bound workloads](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).
