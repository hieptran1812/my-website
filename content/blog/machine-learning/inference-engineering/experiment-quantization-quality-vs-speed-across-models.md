---
title: "Quantization quality versus speed across models: INT8, INT4, and FP8 without self-deception"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A reproducible nanoserve-oriented protocol for comparing INT8, INT4, and FP8 on memory, throughput, perplexity, and task quality across the fixed model matrix."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "pytorch",
    "cuda",
    "gpu",
    "latency",
    "throughput",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 38
---

The dangerous sentence in inference engineering is “it is only a little less accurate.” A four-bit checkpoint can fit where a BF16 checkpoint cannot, and it can even look faster in a one-request notebook. Then a reasoning task drops, the server spends its time dequantizing skinny matrices, or a long-context retrieval prompt quietly loses the one fact that mattered.

The diagram below is the mental model: precision is a contract between memory, kernels, and quality gates. INT4, INT8, and FP8 are not three labels on a slider. They move different boundaries in the execution graph, and the useful choice depends on the model family, the GPU, the batch, and the failure you can tolerate.

![The precision choice connects memory residency, kernel execution, and quality gates rather than producing one universal winner](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-1.webp)

This post gives you a protocol and a small implementation path for `nanoserve`. It compares the fixed series matrix: Llama 3.1 8B, Qwen3 8B, Gemma 3 12B, and a small MoE such as Qwen3-30B-A3B, on the fixed hardware set of RTX 4090, L4, A100 80GB, and H100 80GB. It does not pretend that this workspace has a GPU. Every result is either arithmetic, a cited public result, or a reader-reproducible expectation labeled with hardware and script.

The deliverable is not “use FP8.” The deliverable is a decision procedure: prove that the model fits, prove that the kernel path helps, check perplexity, then check two tasks that resemble the product. If the last gate fails, the speedup is not an optimization; it is a regression with a benchmark attached.

## 1. The question we are actually answering

Quantization reduces the number of bits used to represent weights, activations, or the KV cache. Those are different interventions. Weight-only INT4 can leave activations in BF16. W8A8 INT8 changes both sides of a matrix multiply. FP8 can mean FP8 weights with BF16 activations, FP8 weights and activations, or FP8 KV storage. Comparing the names without writing the path is how benchmark charts become folklore.

The fixed matrix deliberately includes several architectural shapes. Meta describes Llama 3.1 8B as an 8B-parameter, 128k-context, GQA model in its official model card ([Meta Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B), published July 23, 2024). Qwen’s model card lists Qwen3-8B at 8.2B parameters, 36 layers, 32 query heads, and 8 KV heads ([Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B), accessed August 3, 2026). Google’s Gemma card identifies the 12B model and its BF16 checkpoint ([Gemma 3 model card](https://huggingface.co/google/gemma-3-12b-it), accessed August 3, 2026). The MoE row is there to stop us from confusing active parameters with bytes that must be resident.

| Question | What the experiment must report | Why a single number lies | Source |
|---|---|---|---|
| Does it fit? | Weight bytes, KV bytes, allocator reserve, peak allocated | A checkpoint size is not the runtime footprint | derived from model config and `torch.cuda.max_memory_allocated()` |
| Is it faster? | TTFT, TPOT, decode tok/s, p50 and p99 | Batch 1 and loaded serving reward different bottlenecks | reproduce: `bench_quant.py` |
| Is it still fluent? | Held-out perplexity and task scores | Perplexity can miss instruction and code regressions | reproduce: `eval_quant.py` |
| Is the comparison fair? | Same prompts, seed, tokenizer, max output, warmup, engine | Changing any of these changes the workload | protocol in this post |

The series already covered the bit-level formats in [quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) and the load-time mechanics in [weight-only quantization in your engine](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time). Here we stay on the experiment boundary: what to measure, how to derive what does not need a GPU, and how to decide when quality has fallen too far.

> A quantized model has not passed because it fits. It has passed when it fits, moves tokens, and answers the questions your product asks.

## 2. Three precisions are three execution paths

The first trap is treating “4 bits” as an instruction to the GPU. A packed INT4 weight is storage. Before a tensor core can use it, a kernel must unpack values, apply a group scale and possibly a zero point, and accumulate into a safer type. A good weight-only kernel fuses much of that work into the matrix multiplication. A bad path materializes BF16 weights and gives back the memory win.

The simplified affine formula for a quantized group is an explanatory abstraction, not a formula claimed by a particular checkpoint:

![Packed weights branch into an INT4 dequantization path or an FP8 scaled matmul before producing BF16 output](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-2.webp)

$$
\hat{w}_i = s \left(q_i - z\right), \qquad q_i \in \{0,\ldots,2^b-1\}.
$$

Here $b$ is the integer bit width, $s$ is a group scale, $z$ is an optional zero point, and $\hat{w}_i$ is the reconstructed value. The choice of group size determines how many weights share $s$ and $z$. Smaller groups track local ranges better but spend more metadata and create more scale loads.

For a weight matrix with $P$ parameters, the first-order storage estimate is:

$$
M_{\text{weights}} \approx \frac{P b}{8} + M_{\text{scales}} + M_{\text{zeros}}.
$$

That is an accounting identity with omitted container headers and alignment, not a claim about a file format. For an 8B dense model, the ideal payload arithmetic is $8{,}000{,}000{,}000 \times 2 / 8 = 2{,}000{,}000{,}000$ bytes at INT2, $8{,}000{,}000{,}000 \times 4 / 8 = 4{,}000{,}000{,}000$ bytes at INT4, $8{,}000{,}000{,}000 \times 8 / 8 = 8{,}000{,}000{,}000$ bytes at INT8, and $8{,}000{,}000{,}000 \times 16 / 8 = 16{,}000{,}000{,}000$ bytes at BF16. Decimal gigabytes are used here so the arithmetic is visible; a loader will add scales, norms, embeddings, temporary buffers, and allocator slack.

The practical formats differ like this:

| Path | Weight storage | Activation storage | Accumulator | Likely benefit | Likely failure |
|---|---:|---:|---:|---|---|
| BF16 | 16 bits/value | BF16 | FP32 or BF16 kernel accumulator | reference quality and broad support | weight traffic and VRAM ceiling |
| INT4 W4A16 | 4 bits/value plus scales | BF16 | BF16/FP32 | capacity and decode bandwidth | dequant overhead; outlier and reasoning cliffs |
| INT8 W8A8 | 8 bits/value plus scales | INT8 | INT32 or FP32 path | conservative quality with native integer kernels | calibration and activation outliers |
| FP8 W8A8 | 8 bits/value plus scales | FP8 | FP16/BF16/FP32 depending on kernel | native tensor-core throughput on supported GPUs | scale error, accumulation, short-context overhead |

PyTorch’s torchao inference documentation describes the distinction directly: weight-only INT4 keeps the input in the original precision, while dynamic FP8 quantization converts both operands before the matmul ([torchao quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html), updated March 25, 2026). Treat that page as an API reference, not as evidence that every model and GPU gets the same speedup.

### Why the advertised two times is not automatic

At decode, a small batch repeatedly streams model weights for one or a few tokens. If the kernel can read half as many bytes and keep the unpacking inside the memory-to-math path, a weight-only format can move the roofline. If the batch is large enough that the GEMM is compute-bound, the same format may add instructions without removing the dominant bottleneck. If the model is MoE, routing and expert imbalance can dominate the weight representation.

FP8 has a different precondition: the GPU must have a useful FP8 kernel path. The vLLM team reports FP8 KV-cache gains on Hopper for long contexts, including a cited break-even around 7k tokens for one Llama 3.1 8B slope experiment, while short contexts can favor BF16; this is a cited vLLM result, not a measurement from this post ([vLLM FP8 KV-cache report](https://vllm.ai/blog/2026-04-22-fp8-kvcache), April 22, 2026). Notice the condition: cache dtype, context length, hardware, and kernel all appear in the sentence.

#### Worked example: ideal weight residency

Take the official nominal 8B size, ignore metadata, and compare BF16 with INT4. The derived payload ratio is $16 / 4 = 4$. The ideal payload falls from 16,000,000,000 bytes to 4,000,000,000 bytes, a 12,000,000,000-byte reduction. That is not a promise that a 24 GB card has 12 GB free: the model has embeddings, layer norms, scales, CUDA workspaces, the KV cache, and the runtime. The correct next step is `torch.cuda.max_memory_allocated()`, not subtraction from the marketing capacity.

## 3. The fixed model matrix and the numbers we can know before running

The model matrix is intentionally small enough to run and broad enough to expose family-specific cliffs.

| Family | Nominal size | Architecture facts that affect this test | What to watch | Source |
|---|---:|---|---|---|
| Llama 3.1 8B | 8B | 32 layers, 8 KV heads, 128k context in the Meta card | strong baseline; GQA reduces KV cost | cited: [Meta card](https://huggingface.co/meta-llama/Llama-3.1-8B), July 23, 2024 |
| Qwen3 8B | 8.2B | 36 layers, 8 KV heads, 32,768 native context | tokenizer and reasoning sensitivity | cited: [Qwen card](https://huggingface.co/Qwen/Qwen3-8B), accessed August 3, 2026 |
| Gemma 3 12B | 12B | multimodal checkpoint family; 12B BF16 model | larger weight floor; keep text prompts identical | cited: [Gemma card](https://huggingface.co/google/gemma-3-12b-it), accessed August 3, 2026 |
| Qwen3-30B-A3B | 30B total, 3B active/token | MoE routing; total resident expert bytes still matter | “3B active” is not “3B loaded” | cited: [Qwen3 collection](https://huggingface.co/Qwen), accessed August 3, 2026; verify exact revision before run |

The last row needs special care. “Active parameters” is a compute description per token, not a memory description. If all experts are resident, the checkpoint and quantization memory tracks total parameters. If experts are offloaded, host traffic and routing become new variables. A benchmark that compares a dense 8B INT4 model with an MoE model while reporting only active parameters is comparing unlike budgets.

The fixed hardware names also matter. The RTX 4090 and L4 are 24 GB-class cards, while the A100 and H100 rows are 80 GB SXM-class references. The post’s expected tok/s ranges must be attached to a named card, CUDA version, engine revision, and workload. “On an NVIDIA GPU” is not provenance.

### Memory arithmetic that includes KV

For a decoder transformer with $L$ layers, $H_{kv}$ KV heads, head dimension $d$, and element size $s$ bytes, the standard per-token BF16 KV estimate is:

$$
B_{\text{KV/token}} = 2 \cdot L \cdot H_{kv} \cdot d \cdot s.
$$

The factor 2 is K and V. This is the series’ cache law; it is independent of weight quantization unless you also quantize the cache. Using the Qwen3-8B card’s 36 layers, 8 KV heads, 128 head dimension, and BF16’s 2 bytes gives $2 \times 36 \times 8 \times 128 \times 2 = 147{,}456$ bytes per token, about 144 KiB. At 8,192 tokens, the derived amount is $147{,}456 \times 8{,}192 = 1{,}207{,}959{,}552$ bytes, about 1.125 GiB. Quantizing weights to INT4 does not change that number. Quantizing KV to FP8 would halve the element-size factor from 2 to 1, subject to a separate accuracy test.

#### Worked example: why INT4 does not double context

Suppose an 8B checkpoint saves 12 GB of ideal weight payload by moving from BF16 to INT4. That free space can hold roughly $12{,}000{,}000{,}000 / 147{,}456 \approx 81{,}383$ Qwen3-style BF16 KV tokens before runtime reserve and fragmentation. The arithmetic is illustrative, not a capacity guarantee. If the same deployment has a 10 GB runtime reserve, a vision encoder, or a batch of unrelated requests, the usable context is lower. Weight quantization and KV quantization solve different memory curves.

## 4. Make `nanoserve` expose the comparison instead of hiding it

The first code change is a representation object. It must say what is quantized and what is not. A string like `int4` is too ambiguous for a serving engine.

```python
# nanoserve/precision.py
from dataclasses import dataclass
from typing import Literal

WeightDType = Literal["bf16", "int4", "int8", "fp8"]
ActivationDType = Literal["bf16", "int8", "fp8"]

@dataclass(frozen=True)
class PrecisionPlan:
    weights: WeightDType
    activations: ActivationDType
    kv_cache: Literal["bf16", "fp8"] = "bf16"
    group_size: int = 128

    def label(self) -> str:
        return f"W{self.weights}-A{self.activations}-KV{self.kv_cache}-G{self.group_size}"

    def weight_bits(self) -> int:
        return {"bf16": 16, "int4": 4, "int8": 8, "fp8": 8}[self.weights]

    def kv_bytes(self, layers: int, kv_heads: int, head_dim: int) -> int:
        element_bytes = {"bf16": 2, "fp8": 1}[self.kv_cache]
        return 2 * layers * kv_heads * head_dim * element_bytes
```

The `kv_bytes` method makes a common test bug visible: changing `weights` does not change KV bytes. A plan named `Wint4-Abf16-KVbf16-G128` is less catchy than `int4`, but it can be reproduced.

Next, add a loader boundary. In a real engine this dispatches to a fused backend such as a Marlin-like INT4 kernel, an INT8 GEMM, or an FP8 GEMM. The reference path below is intentionally slow and correctness-oriented; it demonstrates the contract without claiming that Python dequantization is a serving kernel.

```python
# nanoserve/linear_quant.py
import torch
from torch import Tensor

def dequant_groupwise_int4(packed: Tensor, scale: Tensor, zero: Tensor,
                           out_features: int, group_size: int) -> Tensor:
    """Reference W4A16 unpack; use a fused kernel for serving."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    q = q[:, :out_features]
    groups = (out_features + group_size - 1) // group_size
    q = q.reshape(q.shape[0], groups, group_size)
    w = (q.float() - zero[:, :groups, None]) * scale[:, :groups, None]
    return w.reshape(q.shape[0], out_features).to(torch.bfloat16)

def linear_w4a16(x: Tensor, packed: Tensor, scale: Tensor,
                 zero: Tensor, out_features: int, group_size: int) -> Tensor:
    weight = dequant_groupwise_int4(packed, scale, zero, out_features, group_size)
    return torch.nn.functional.linear(x, weight)
```

Do not put this reference implementation in the decode hot path and then report its tok/s as INT4 performance. It allocates a reconstructed weight. The point is to test numerical parity against a fused implementation, then swap the backend while keeping the same `PrecisionPlan`.

For FP8, the same interface needs scales and a capability check. PyTorch’s current torchao workflow documents CUDA compute capability requirements for its FP8 configurations; the exact backend availability is version-dependent, so fail loudly rather than silently falling back to BF16 ([torchao inference support table](https://docs.pytorch.org/ao/stable/workflows/inference.html), updated March 25, 2026).

```python
# nanoserve/backend.py
import torch

def require_fp8(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("FP8 serving requires a CUDA device in this backend")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"no supported FP8 path for compute capability {major}.{minor}; "
            "choose BF16 or INT8"
        )

def select_linear(plan, device):
    if plan.weights == "fp8" or plan.activations == "fp8":
        require_fp8(device)
        return "fp8_tensor_core"
    if plan.weights == "int4" and plan.activations == "bf16":
        return "fused_w4a16"
    if plan.weights == "int8" and plan.activations == "int8":
        return "int8_tensor_core"
    return "bf16_reference"
```

The production rule is simple: log the selected backend next to every benchmark row. If the log says `bf16_reference` while the row says FP8, the row is invalid.

## 5. The benchmark protocol: cold numbers are not serving numbers

The fourth figure is a timeline because timing errors happen in a sequence. Load, warm, synchronize, measure, score, report. A benchmark that starts its timer before the first CUDA kernel has included lazy initialization. A benchmark that stops it before `torch.cuda.synchronize()` has measured CPU enqueue time.

![A trustworthy quantization benchmark warms the engine, synchronizes CUDA, measures steady state, and scores the same outputs](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-4.webp)

Use the protocol from post 40 in this track as the umbrella: fixed prompt files, deterministic seeds where the backend supports them, explicit warmup, and open-loop load when you care about queueing. This post adds a quality pass because throughput without quality is not a result.

### The minimum performance matrix

Run each model and plan across the four prompt families: short chat input with longer output, long-input RAG with short output, code completion, and translation. For each family, include a batch-oriented sweep and a loaded-serving sweep. The exact batch values are reader parameters; a practical first pass is batch 1, 4, 16, and 32 if the card permits them.

```python
# bench_quant.py
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def timed_decode(model, tokenizer, prompt, new_tokens, repeats):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        stop.record()
        stop.synchronize()
    milliseconds = start.elapsed_time(stop)
    generated = repeats * new_tokens
    return generated / (milliseconds / 1000.0), milliseconds / repeats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain paged attention in one paragraph.")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    tok_s, step_ms = timed_decode(model, tok, args.prompt,
                                  args.new_tokens, args.repeats)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print({"model": args.model, "tok_s": tok_s,
           "decode_ms_per_run": step_ms, "peak_allocated_gib": peak})

if __name__ == "__main__":
    main()
```

This is a baseline harness, not a production server benchmark. To make it `nanoserve`-oriented, replace the model call with the engine’s `generate` endpoint and retain the CUDA-event discipline inside the worker. The same script should accept `--precision-plan` and log the selected backend. For a server, measure TTFT from request arrival to first token, TPOT from the first token onward, and goodput under an open-loop arrival rate. Do not infer p99 from ten sequential requests.

### Expected ranges, explicitly labeled

The following are not measured results. They are reader-reproducible expectations for a correctly fused implementation, with large enough ranges to avoid pretending that clocks, CUDA versions, kernels, prompt mix, and batch are known. Run the supplied harness and replace them with your values.

| Model and plan | Hardware/workload | Expected reader result | Interpretation | Source |
|---|---|---|---|---|
| Llama 3.1 8B BF16 | RTX 4090, batch 1, 128 output tokens | roughly 35–80 decode tok/s | baseline range; prompt and kernel dominate | reproduce: `bench_quant.py`, reader-measured expected range |
| Llama 3.1 8B W4A16 | RTX 4090, batch 1, same prompts | roughly 45–110 decode tok/s if fused | memory win may appear at low batch; reference Python path is excluded | reproduce: fused `nanoserve` backend |
| Llama 3.1 8B FP8 | H100, batch 1–16 | roughly 1.0–1.8× BF16 decode throughput | only if the selected backend is native FP8 | reproduce: `bench_quant.py`; compare same engine revision |
| Qwen3 8B W8A8 | L4, batch 1–8 | roughly 0.8–1.3× BF16 | L4 support and scale overhead can erase the gain | reproduce: `bench_quant.py` |
| Gemma 3 12B W4A16 | RTX 4090, batch 1 | must fit only with explicit reserve check | capacity result, not a speed promise | derived weights plus reader peak memory |

These ranges are intentionally labeled expected. They are not a citation and do not claim a run happened. The only honest way to publish a point value is to attach the raw command, model revision, GPU name, driver, CUDA, engine commit, prompt file hash, warmup count, and measurement window.

### Measuring memory without confusing allocated and reserved

```python
def memory_snapshot(model):
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {"allocated_gib": allocated, "reserved_gib": reserved,
            "peak_allocated_gib": peak}
```

Call the snapshot after model load and again after a representative batch and context. `reserved` is allocator ownership; `allocated` is live tensors; peak allocated is what the kernels actually demanded at the high-water mark. Record all three. A card showing 57% utilization can still OOM because the remaining blocks are not contiguous or because a pending allocation needs a larger workspace.

The before/after figure below is a capacity picture, not a measured result. The BF16 and INT4 weight labels come from the ideal $P \times b/8$ calculation; the runtime and KV boxes are deliberately marked as context-dependent.

![BF16 leaves less weight headroom while INT4 opens capacity that still has to be checked against runtime and quality](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-3.webp)

## 6. Perplexity is necessary and insufficient

Perplexity is the exponentiated average negative log likelihood of held-out tokens. For token sequence $x_1,\ldots,x_T$, the standard evaluation expression is:

$$
\operatorname{PPL}(x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(x_t \mid x_{<t})\right).
$$

This is a useful regression detector. It is not a complete product-quality metric because it weights every token in the corpus and does not directly ask whether a model followed an instruction, wrote compilable code, or preserved a retrieved identifier. Use the same tokenizer and mask padding consistently. Report the number of evaluated tokens and the corpus revision.

```python
# eval_ppl.py
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.inference_mode()
def perplexity(model, tokenizer, texts, max_length=2048):
    total_nll = torch.zeros((), device=model.device)
    total_tokens = 0
    for text in texts:
        batch = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length).to(model.device)
        out = model(**batch, labels=batch["input_ids"])
        tokens = batch["attention_mask"].sum()
        total_nll += out.loss * tokens
        total_tokens += int(tokens)
    return math.exp((total_nll / total_tokens).item()), total_tokens

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [row["text"] for row in dataset if row["text"].strip()]
value, count = perplexity(model, tokenizer, texts)
print({"ppl": value, "tokens": count})
```

The snippet assumes `model` and `tokenizer` have been loaded by the caller. In the actual evaluation script, load BF16 and every quantized variant from pinned revisions, and write JSON Lines with `{model, plan, dataset_revision, tokens, ppl}`. Do not round a value before computing the difference.

Report both absolute and relative drift:

$$
\Delta_{\text{rel}} = 100 \times \frac{\operatorname{PPL}_{q} - \operatorname{PPL}_{\text{BF16}}}{\operatorname{PPL}_{\text{BF16}}}.
$$

If the BF16 reference has perplexity 10 and INT4 has 10.5, the derived relative drift is $100 \times (10.5 - 10) / 10 = 5\%$. That arithmetic does not tell you whether a 5% drift is acceptable; the task gates decide that.

## 7. Two task evals that catch what perplexity misses

For this matrix, use one reasoning/mathematics task and one code task. GSM8K is a familiar arithmetic reasoning benchmark; HumanEval tests code generation. Both are imperfect. Pin the exact harness, prompt template, stop criteria, and pass@k setting. If Gemma is loaded as a multimodal checkpoint, use its text-only path and document that choice; do not mix image inputs into a text-only comparison.

```python
# eval_tasks.py
import json
import subprocess
from pathlib import Path

def run_task(model_id, plan, task_file, output_file):
    command = [
        "python", "-m", "nanoserve.eval",
        "--model", model_id,
        "--precision-plan", plan,
        "--prompts", str(task_file),
        "--temperature", "0",
        "--max-new-tokens", "512",
        "--output", str(output_file),
    ]
    subprocess.run(command, check=True)
    return json.loads(Path(output_file).read_text())

for task in ("gsm8k.jsonl", "humaneval.jsonl"):
    result = run_task(
        "meta-llama/Llama-3.1-8B-Instruct",
        "Wint4-Abf16-KVbf16-G128",
        Path("evals") / task,
        Path("results") / (task + ".json"),
    )
    print(task, result["exact_match"], result["count"])
```

The engine command is the contract to implement in `nanoserve.eval`; it keeps the evaluation prompt path separate from ad hoc notebook generation. For GSM8K, compare exact normalized answers, not free-form vibes. For HumanEval, run the generated function in the sandbox specified by the benchmark harness and report pass rate. Do not run untrusted generated code on a production host.

The second task should be selected from the product if it is not code or arithmetic. A RAG system should include retrieval of exact strings and a citation-preservation check. A translation system should include sacreBLEU or COMET with a pinned version, plus a human spot check. “Two task evals” means two behaviors, not two prompts.

The fifth figure is a tree because regression branches. Activation outliers often point toward calibration or a different scale granularity. Long-context drift can point toward KV or accumulation, not weight-only INT4. Reasoning cliffs can appear even when perplexity barely moves. The remedy is localization, not a global belief about “quantization quality.”

![Quality regressions branch into calibration sensitivity, long-context numerical error, and task-specific reasoning loss](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-5.webp)

### A quality report template

| Model | Plan | PPL | GSM8K | HumanEval | Decision | Source |
|---|---|---:|---:|---:|---|---|
| Llama 3.1 8B | BF16 | reference | reference | reference | control | reproduce: `eval_ppl.py` + `eval_tasks.py` |
| Llama 3.1 8B | INT4 W4A16 | fill after run | fill after run | fill after run | ship only if gates pass | reproduce: pinned scripts |
| Qwen3 8B | INT8 W8A8 | fill after run | fill after run | fill after run | compare calibration | reproduce: pinned scripts |
| Gemma 3 12B | FP8 W8A8 | fill after run | fill after run | fill after run | verify text path | reproduce: pinned scripts |
| Qwen3-30B-A3B | INT4 experts | fill after run | fill after run | fill after run | inspect routing | reproduce: pinned scripts |

An empty cell is better than an invented result. The table is designed to be committed with raw artifacts after a reader runs it on named hardware.

## 8. Family-specific cliffs and what public results actually support

There is no universal cliff location, but there are recurring mechanisms.

### Dense GQA models: Llama and Qwen

Dense GQA models are good first candidates for W4A16 because the workload is familiar and the weight traffic is substantial. The quality question is still layer- and calibration-dependent. A model card can tell you the architecture and a checkpoint can tell you the tensor types; neither guarantees that an arbitrary INT4 recipe preserves instruction behavior.

Qwen3’s card is especially useful for derived cache arithmetic because it exposes the layer and KV-head counts. Its model card also warns about using a sufficiently recent Transformers version for the `qwen3` model type ([Qwen3 usage note](https://huggingface.co/Qwen/Qwen3-8B), accessed August 3, 2026). Pin that dependency in the benchmark environment. If one variant uses a different chat template, the task comparison is invalid before quantization enters the discussion.

### Gemma 3: larger dense model, different surface

Gemma 3 12B is a useful capacity stress case because its nominal weight floor is larger than the 8B rows, and its official model card describes a multimodal family. The text-only benchmark should use the same text prompts, with image processing disabled and that fact recorded. INT4 can be a fit decision on a 24 GB card long before it is a speed decision. If the model only fits after offloading, report host traffic and do not compare its tok/s with a fully resident Llama row.

### MoE: active compute versus resident bytes

Qwen3-30B-A3B is where the simple bits-per-parameter story becomes least reliable. A top-k router selects a subset of experts for each token, so the arithmetic performed per token can resemble a smaller model. The weights may still occupy space for all experts, and a quantized expert path can introduce routing, gather, and scale overhead. Measure expert load balance and all-to-all time if the engine shards experts.

The vLLM team’s public Blackwell benchmark is useful as a contrast because it reports tuned FP4 and FP8 systems with explicit workload setups, not a generic “quantization is faster” claim. Their report gives up to 4.3× higher throughput for gpt-oss 120B at a stated 1k/1k chat setup on B200 relative to Hopper, and separately reports a Llama 3.3 70B result; those are cited vLLM results, not portable expectations for `nanoserve` ([vLLM Blackwell InferenceMAX report](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), October 9, 2025). Use it to understand why hardware and kernel context belong in the row, not to fill an RTX 4090 cell.

## 9. A reproducible command matrix

The commands below are intentionally explicit. The exact quantized checkpoint names vary by release and license; substitute pinned public checkpoints that implement the declared plan and record the URL. Do not silently load a checkpoint whose model card says W4A16 and call it W8A8.

```bash
python -m nanoserve.serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --precision-plan Wbf16-Abf16-KVbf16-G128 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --port 8000

python bench_quant.py \
  --model http://127.0.0.1:8000 \
  --prompts evals/prompt-suite.jsonl \
  --arrival-rate 2 \
  --output results/llama-bf16-4090.json
```

The INT4 and FP8 commands should change only the model revision and `--precision-plan`, not prompt files, output limits, server concurrency, or arrival process. A separate command should run the BF16 control after rebooting or clearing the engine cache if compilation caches affect startup. Warmup is excluded from steady-state throughput but included in a separately reported startup measurement.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python eval_ppl.py --model ./pinned/llama-bf16 --dataset wikitext-2
python eval_ppl.py --model ./pinned/llama-int4 --dataset wikitext-2
python eval_tasks.py --model ./pinned/llama-int4 --tasks gsm8k humaneval
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Do not claim that `CUBLAS_WORKSPACE_CONFIG` makes every fused kernel deterministic. It is a reproducibility aid for supported operations, not a universal guarantee. Record the random seed, sampling mode, and whether the backend uses CUDA graphs.

### Report distributions, not just means

For latency, report p50 and p99 over requests after warmup. For tok/s, report both per-request decode rate and aggregate server output rate. For quality, report confidence intervals or at least the number of examples. For a task with $N$ exact-match examples and $k$ successes, the observed rate is $100k/N$ percent. If $k=73$ and $N=100$, the arithmetic gives 73%; it does not say whether a 2-point difference is statistically meaningful without uncertainty.

At load, use Little’s law as a sanity check: $L = \lambda W$, where $L$ is average in-flight requests, $\lambda$ is arrival rate, and $W$ is average time in system. If a closed-loop client waits for each response before sending the next request, it suppresses queueing and can make a server look healthier than it is. The inference benchmark protocol in this series is the right place to add an open-loop generator.

### Calibration is part of the model artifact

Quantization calibration is not a clerical pre-step. It chooses which activation ranges, channels, and token positions the low-precision representation will see. A calibration set that contains only short English prose can miss code punctuation, multilingual text, long retrieved passages, and the repetitive structures that make activation outliers visible. The calibration set should be small enough to version and large enough to cover the production prompt families.

Keep the calibration set separate from the perplexity and task evaluation sets. Reusing the same examples makes a candidate look more stable than it is. Store a content hash, tokenizer revision, maximum sequence length, and sampling policy in the manifest. If you use an AWQ-, GPTQ-, SmoothQuant-, or torchao-style workflow, record the method, group size, symmetry, zero-point policy, and whether scales are per-tensor, per-channel, or per-group. “INT8” without those qualifiers is not an experiment description.

```python
# calibration_manifest.py
import hashlib
import json
from pathlib import Path

def sha256_lines(path: Path) -> str:
    digest = hashlib.sha256()
    for line in path.read_bytes().splitlines(keepends=True):
        digest.update(line)
    return digest.hexdigest()

manifest = {
    "calibration_file": "calibration/prompts.jsonl",
    "calibration_sha256": sha256_lines(Path("calibration/prompts.jsonl")),
    "tokenizer_revision": "pinned-in-lockfile",
    "sequence_lengths": [512, 2048, 8192],
    "quantizer": "w4a16-groupwise",
    "group_size": 128,
    "symmetric": True,
}
Path("results/calibration-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
```

The numbers in `sequence_lengths` are an experiment design, not a claim about a universal best calibration set. If your service accepts 32k-token RAG requests, add a 32k calibration slice or explicitly label the long-context result as out of calibration distribution. A quality cliff that appears only at 32k is still a production cliff.

### Scale granularity creates a memory-speed-quality triangle

Per-tensor scaling is cheap to load but may be too coarse when one channel has a much larger range than its neighbors. Per-channel or per-group scaling tracks local variation better but introduces more metadata and scale operations. The right comparison therefore holds group size constant while comparing precisions, then runs a second sweep over group size for the winning precision.

For a weight tensor with $P$ values, group size $G$, and one BF16 scale per group, the scale metadata is approximately $2 \times \lceil P/G \rceil$ bytes before alignment. With $P=8{,}000{,}000{,}000$ and $G=128$, the derived scale count is $\lceil 8{,}000{,}000{,}000/128 \rceil = 62{,}500{,}000$ and the BF16 scale payload is about $125{,}000{,}000$ bytes. That is small beside an 8B INT4 payload, but it is not zero, and the access pattern can matter more than the bytes.

| Granularity | Scale count | Quality tendency | Kernel cost | Use in this experiment |
|---|---:|---|---|---|
| Per tensor | 1 per tensor | weakest outlier adaptation | lowest metadata | diagnostic baseline only |
| Per channel | one per output channel | better local range | extra scale loads | compare on INT8/FP8 where supported |
| Group 128 | one per 128 weights | common W4A16 compromise | packed dequant work | primary INT4 row |
| Group 32 | one per 32 weights | can preserve difficult layers | more metadata and work | targeted cliff investigation |

The source column for this table is derived arithmetic plus a qualitative experiment design. It is not a claim that group 128 wins every model. The model-specific row in the final report must carry the actual checkpoint and quantizer source.

### Decode and prefill need separate conclusions

Prefill processes many prompt tokens together and is usually matmul-rich. Decode processes one new token per active sequence and repeatedly reads weights and KV. A quantization method can improve decode and leave prefill unchanged, or improve prefill at a batch shape where decode gets no benefit. Report TTFT and TPOT separately.

For a request with input length $S$ and generated length $T$, a simple accounting split is total time $t_{\text{total}} = t_{\text{prefill}}(S) + T \cdot t_{\text{decode}}$. This is an explicitly explanatory approximation for interpreting traces; scheduler overlap and chunked prefill can violate the simple additive view. If a 2,000-token prompt produces one token, an optimization to decode kernel bandwidth has little opportunity to amortize its cost. If it produces 2,000 output tokens, decode dominates more strongly. Use the prompt suite to expose both regimes.

```python
# summarize_trace.py
import json
import statistics
from pathlib import Path

rows = [json.loads(line) for line in Path("results/requests.jsonl").read_text().splitlines()]
ttft = [row["first_token_ms"] for row in rows]
tpot = [row["inter_token_ms"] for row in rows]
def percentile(values, p):
    values = sorted(values)
    index = min(len(values) - 1, round((p / 100) * (len(values) - 1)))
    return values[index]

print({
    "requests": len(rows),
    "ttft_p50_ms": percentile(ttft, 50),
    "ttft_p99_ms": percentile(ttft, 99),
    "tpot_p50_ms": percentile(tpot, 50),
    "tpot_p99_ms": percentile(tpot, 99),
    "decode_tok_s_median": 1000.0 / statistics.median(tpot),
})
```

The conversion in the last line is derived: if median inter-token time is measured in milliseconds, one token per that interval gives $1000/t_{\text{ms}}$ tokens per second. It is not interchangeable with aggregate server tok/s when multiple requests decode concurrently. Keep both fields in the JSONL result.

### Numerical parity is a release gate

Before a performance run, compare logits or hidden states between the reference dequantization path and the fused backend on small deterministic tensors. Use an absolute tolerance and a relative tolerance appropriate to the accumulation type. Then compare generated text on a fixed greedy prompt set. A fused kernel that is fast but swaps a scale index or a nibble order is not a quantized model; it is a correctness bug.

```python
def assert_close(reference, candidate, atol=2e-2, rtol=2e-2):
    torch.testing.assert_close(
        candidate.float(), reference.float(), atol=atol, rtol=rtol
    )

def greedy_ids(logits):
    return logits.argmax(dim=-1)

reference_ids = greedy_ids(reference_logits)
candidate_ids = greedy_ids(candidate_logits)
agreement = (reference_ids == candidate_ids).float().mean().item()
print({"greedy_token_agreement": agreement})
```

The tolerances above are example starting points, not a universal pass threshold. Calibrate them from BF16-versus-BF16 repeatability and the kernel’s documented accumulation. Token agreement is a debugging signal, not a replacement for PPL or task scores: one changed early token can make the rest of a sampled sequence diverge even when the distributions remain close.

### A result manifest that can survive review

Every published row should be reconstructible from a manifest containing: model repository and revision; quantizer repository and revision; tokenizer revision; `nanoserve` commit; PyTorch, Transformers, CUDA, driver, and kernel backend versions; GPU name and memory; clocks if locked; batch and context; prompt-suite hash; warmup count; measurement count; seed and sampling policy; output length; peak allocated and reserved memory; TTFT, TPOT, p50, p99, aggregate tok/s; PPL token count; task harness revisions; and the source category.

The manifest is not bureaucracy. It answers the most common review questions without rerunning the experiment: was this really FP8, was the card an H100 or an L4, did the candidate use the same tokenizer, and did the quality score use the same prompts? If any answer is unknown, label the row incomplete.

```json
{
  "model": "Qwen/Qwen3-8B@pinned-revision",
  "plan": "Wint8-Aint8-KVbf16-G128",
  "gpu": "A100-SXM-80GB",
  "batch": 8,
  "input_tokens": 2048,
  "output_tokens": 256,
  "warmup_requests": 20,
  "measured_requests": 100,
  "backend": "int8_tensor_core",
  "source": "reproduce: bench_quant.py"
}
```

The JSON values here define the schema, not a completed result. In particular, the manifest does not contain a tok/s claim. That field belongs in the result file after a reader runs the named command on the named hardware.

### How to read a disappointing result

Start with memory. If the candidate does not fit without host offload, stop comparing tok/s. Either reduce context, use a smaller model, quantize the KV cache separately, or choose a larger GPU. An offloaded candidate is a different deployment shape. It may still be the right product choice, but its result belongs in a separate table with host bandwidth, transfer volume, and queue behavior.

If it fits but is slower, inspect the backend string and the trace. Three explanations cover most cases. First, the quantized path is a fallback: a model loader accepted the checkpoint, but the actual matmul is BF16 or a reference dequantization. Second, the workload is compute-bound: the smaller weights reduce traffic that was not the bottleneck at this batch. Third, the quantizer added enough scale, gather, or routing work that the saved bytes did not compensate. These are different fixes. Backend fallback needs a kernel or a rejection. Compute-bound behavior may need batching or a different format. Scale overhead may justify a coarser granularity or BF16.

If it is faster but perplexity drifts, inspect the distribution of token losses rather than only the mean. A small number of extreme examples can be hidden by a large corpus. Bucket by prompt length and task family. If the drift appears only in long sequences, test KV dtype and accumulation separately. If it appears on code, include code calibration. If it appears on multilingual text, do not “fix” it by adding more English calibration examples.

If perplexity passes and a task fails, preserve the failure. Do not average it away with a second task that improves. A model can become slightly more likely to emit common prose while becoming less likely to follow a multi-step instruction. The product owner may still accept it, but the trade must be explicit. The point of the protocol is to make the choice visible.

### Hardware-specific expectations without fake precision

The same plan should be expected to behave differently across the fixed cards. The RTX 4090 is the capacity-first consumer baseline. It is a useful home-lab target for W4A16, but a reader should not assume the H100’s FP8 path exists or has the same efficiency. The L4 is a lower-power deployment target where memory bandwidth and supported kernels can make a nominally smaller type less impressive. The A100 has broad BF16 and INT8 maturity but is not a native Hopper FP8 target. The H100 is the cleanest FP8 experiment in this matrix because the cited vLLM FP8 report explicitly targets Hopper.

Use hardware labels as experimental strata, not as a ranking. A result of 80 tok/s on an RTX 4090 and a result of 70 tok/s on an L4 do not mean the L4 is “slower” in a product sense; power, price, concurrency, and SLO matter. The cost post later in this series can turn throughput into dollars, but this post should report the physical measurements first.

| Observation | Most likely next check | Do not conclude yet |
|---|---|---|
| INT4 saves memory, speed is flat | backend name and dequant kernels | INT4 is useless |
| FP8 wins long context only | context bucket and KV dtype | FP8 always wins |
| INT8 improves PPL but not tasks | task template and calibration coverage | PPL is wrong |
| MoE INT4 fits but p99 rises | expert load balance and gather time | active parameters predict latency |
| A100 and H100 disagree | kernel capability and accumulator | the model is nondeterministic |

### A small ablation beats a giant sweep

When a full matrix is expensive, use an ablation order that isolates variables. Begin with one dense model, one prompt family, and batch 1. Compare BF16 against W4A16 with the same KV cache. Then keep W4A16 fixed and sweep batch. Then keep weight precision fixed and compare BF16 versus FP8 KV at two context lengths. Only after those paths are understood should you add INT8 or the MoE row.

This order prevents a common confound: changing weight type, activation type, KV type, batch, context, and model family at the same time. A large grid can generate many rows while answering none of the causal questions. The smallest useful experiment has a control and changes one declared variable.

For each ablation, write the hypothesis before the run. “W4A16 should reduce decode time at batch 1 because the kernel streams fewer weight bytes” is falsifiable. “INT4 should be faster” is not. If the hypothesis fails, preserve the trace and explain whether the issue was bandwidth, compute, conversion, or scheduling. That explanation is more valuable to `nanoserve` than a leaderboard number.

## 10. The quality gate belongs in the engine

The sixth figure presents the gate as a stack. It is not a suggestion to deploy in that order if the product has a different risk model; it is a useful default because a failing memory gate prevents any later measurement, and a failing task gate invalidates a speed win.

![A quantized candidate must pass memory, speed, perplexity, and task gates before deployment](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-6.webp)

Use explicit thresholds supplied by the product owner. If none exist, define provisional thresholds before looking at outputs. For example, a team might allow a small perplexity drift, require no more than a specified relative regression on GSM8K, and require HumanEval not to cross a minimum. Those threshold numbers are policy, not facts; write them into the experiment manifest.

```python
# nanoserve/quality_gate.py
from dataclasses import dataclass

@dataclass
class Gate:
    max_ppl_relative_drift: float
    max_task_drop_points: float
    min_decode_speed_ratio: float

def check_gate(reference, candidate, gate: Gate):
    ppl_drift = 100.0 * (candidate.ppl - reference.ppl) / reference.ppl
    task_drop = max(reference.gsm8k - candidate.gsm8k,
                    reference.humaneval - candidate.humaneval)
    speed_ratio = candidate.decode_tok_s / reference.decode_tok_s
    checks = {
        "ppl": ppl_drift <= gate.max_ppl_relative_drift,
        "tasks": task_drop <= gate.max_task_drop_points,
        "speed": speed_ratio >= gate.min_decode_speed_ratio,
        "memory": candidate.peak_gib < candidate.device_gib,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "ppl_drift": ppl_drift, "task_drop": task_drop,
            "speed_ratio": speed_ratio}
```

The code’s threshold fields are not universal recommendations. They make the decision reviewable. Store the full reference and candidate rows, not only the boolean. If the candidate fails only the speed threshold, check that it used the intended fused kernel. If it fails only GSM8K, inspect calibration, prompt templates, and reasoning-token behavior. If it fails only long-context retrieval, run a cache-dtype and accumulation experiment before blaming weights.

### Animated figure: tokens crossing the gates

<figure class="blog-anim">
<svg viewBox="0 0 760 220" role="img" aria-label="A quantized model candidate moves through memory, speed, and quality gates, with a rejected candidate returning to calibration" style="width:100%;height:auto;max-width:900px">
<style>
.qg-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#cbd5e1);stroke-width:2}.qg-pass{fill:var(--accent,#6366f1);opacity:.18}.qg-bad{fill:#ef4444;opacity:.12}.qg-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.qg-dot{fill:var(--accent,#6366f1)}
@keyframes qg-travel{0%{transform:translateX(0);opacity:0}12%{opacity:1}78%{opacity:1}100%{transform:translateX(600px);opacity:0}}
@keyframes qg-reject{0%,45%{opacity:0}55%,70%{opacity:1}85%,100%{opacity:0}}
.qg-token{animation:qg-travel 7s linear infinite}.qg-token2{animation:qg-travel 7s linear infinite;animation-delay:3.5s}.qg-reject{animation:qg-reject 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.qg-token,.qg-token2{animation:none;opacity:1}.qg-token{transform:translateX(600px)}.qg-token2{transform:translateX(300px)}.qg-reject{animation:none;opacity:1}}
</style>
<rect class="qg-box" x="20" y="70" width="150" height="72" rx="10"/><rect class="qg-box" x="210" y="70" width="150" height="72" rx="10"/><rect class="qg-box" x="400" y="70" width="150" height="72" rx="10"/><rect class="qg-box" x="590" y="70" width="150" height="72" rx="10"/>
<rect class="qg-pass" x="20" y="70" width="150" height="72" rx="10"/><rect class="qg-pass" x="210" y="70" width="150" height="72" rx="10"/><rect class="qg-pass" x="400" y="70" width="150" height="72" rx="10"/><rect class="qg-pass" x="590" y="70" width="150" height="72" rx="10"/>
<text class="qg-label" x="95" y="112">memory</text><text class="qg-label" x="285" y="112">speed</text><text class="qg-label" x="475" y="112">perplexity</text><text class="qg-label" x="665" y="112">tasks</text>
<circle class="qg-dot qg-token" cx="30" cy="48" r="10"/><circle class="qg-dot qg-token2" cx="30" cy="178" r="10"/><text class="qg-label qg-reject" x="380" y="200">reject → recalibrate</text>
</svg>
<figcaption>A candidate advances through the gates; a quality failure sends the next iteration back to calibration instead of silently shipping.</figcaption>
</figure>

The animation carries the state transition: one candidate advances, another is rejected and returns to calibration. A still pipeline would not show why the same benchmark becomes an iterative loop.

## 11. What vLLM changes in the comparison

Use vLLM as the benchmark target and contrast, not as a first-hand result. Its production stack has specialized kernels, continuous batching, model-specific quantization support, and hardware-aware dispatch that `nanoserve` will not reproduce in a small post. That difference is useful: the experiment tells you which gap comes from the bit width and which gap comes from engine maturity.

The official vLLM precision digest gives several concrete cautions. Its FP8 KV-cache report says that cache storage can halve, but that uncalibrated per-tensor FP8 has hardware and context-length cliffs, including long-contraction accumulation concerns on Hopper; cite the report with its date and setup rather than turning the numbers into a universal rule ([vLLM FP8 KV-cache](https://vllm.ai/blog/2026-04-22-fp8-kvcache), April 22, 2026). Its PTPC-FP8 ROCm post reports a Llama 3.1 8B WikiText perplexity comparison of 9.4281 BF16 versus 9.5093 PTPC and GSM8K of 73.2% versus 70.8% in its cited AMD setup ([PTPC-FP8 on ROCm](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm), February 24, 2025). Those numbers are cited public results, not expected values for an RTX 4090.

The lesson is methodological: report the dataset and task alongside the numeric result. Perplexity 9.5093 is not “good” in a vacuum; it is a comparison to a stated BF16 baseline under a stated evaluation procedure. A 2.4-point task drop may be unacceptable for one product and irrelevant for another.

The seventh figure closes the loop as a workload grid: hardware, capacity, native kernel support, long context, and reasoning risk produce different first experiments.

![The first quantization experiment should be chosen from hardware capacity, kernel support, context length, and task risk](/imgs/blogs/experiment-quantization-quality-vs-speed-across-models-7.webp)

## 12. Case studies: where a plausible benchmark went wrong

### Case 1: The INT4 model that won the memory chart

The team had an 8B dense model and a 24 GB card. The ideal arithmetic promised a fourfold reduction in weight payload from BF16 to INT4. The model loaded, so the dashboard declared victory. Decode tok/s barely moved. The profiler showed a reference dequantization kernel materializing a BF16 weight matrix before the GEMM. The quantized file was smaller, but the hot path was not.

The fix is not “INT4 is bad.” The fix is to make backend selection observable, require a fused W4A16 kernel for the performance row, and retain the reference path only for correctness. The quality gate then runs on the exact fused checkpoint. The lesson generalizes: storage format and execution format are separate artifacts.

### Case 2: FP8 that lost at short context

An H100 deployment enabled FP8 because the hardware supported it. At long RAG prompts, the candidate improved capacity and could keep more KV resident. At short chat prompts, the extra scaling and kernel launch work made latency noisier and did not improve throughput. The vLLM FP8 KV-cache report explicitly frames a context-dependent break-even in one cited setup, so this is not surprising; it is a reminder to include prompt length in the workload key.

The fix is a policy that chooses cache dtype by context and quality profile, not a global environment variable. The weight format and KV format are logged separately. A short-chat row cannot be used to reject an FP8 long-context deployment, and a long-context win cannot justify changing every short-chat route.

### Case 3: Perplexity passed, arithmetic reasoning failed

The INT4 candidate’s held-out perplexity stayed close to BF16. A task regression later appeared on multi-step arithmetic. The benchmark had evaluated next-token likelihood on ordinary prose but had not evaluated the reasoning format used by the product. The model remained fluent and became less reliable on the important path.

The fix is to pin two task suites before quantization, retain the exact chat template, and make the task thresholds part of the deployment manifest. Perplexity remains useful, but it is a gate, not the gate.

### Case 4: The MoE “3B model” that did not fit like a 3B model

An engineer saw Qwen3-30B-A3B and budgeted for 3B active parameters. The router activated only a subset of experts per token, but the deployment loaded the expert weights. The memory plan was wrong before a token was generated. INT4 reduced the resident footprint, but expert routing and gather kernels dominated the measured decode path.

The fix is to report total resident parameter bytes, active compute parameters, expert placement, and routing utilization in separate columns. If experts are offloaded, include PCIe or NVLink traffic. Active parameters answer a compute question; they do not answer a residency question.

### Case 5: The tokenizer changed the quality result

The Qwen3 candidate appeared to have lower perplexity than Llama, and the team compared raw token counts as if the corpora were identical. The tokenizers segmented the same text differently, and one generation path used a different chat template. The arithmetic was correct for each run but the comparison was not.

The fix is to store tokenizer revision, prompt bytes, rendered template, token count, and dataset revision beside every result. A perplexity table without those fields is an attractive serialization of uncertainty.

## When to reach for each format

Reach for INT4 first when the model does not fit, the product can tolerate a calibration and task-evaluation cycle, and the target engine has a fused W4A16 path. It is the capacity escape hatch. It is not automatically the lowest-latency path at high batch.

Reach for INT8 when activation outliers, compatibility, or a conservative quality margin matter more than maximum compression. Calibrate on data that resembles production. Verify that the target GPU and engine actually use an INT8 tensor-core path instead of converting back to BF16.

Reach for FP8 when the target accelerator and backend have a native, tested path, especially for throughput-oriented serving and long-context capacity. Separate W8A8 from FP8 KV cache. Measure short and long prompts because the winning point can move with context.

Stay with BF16 when the model is small enough, the quality risk is high, the product has no representative task suite, or the quantized backend is an unoptimized fallback. A slower answer that is correct is often cheaper than a fast answer that creates a support queue.

## Key takeaways

- Quantization is an execution contract, not a number in a model filename.
- Weight bytes and KV bytes are separate curves; INT4 weights do not reduce BF16 KV storage.
- Derive weight and KV capacity before running a GPU benchmark, then verify peak allocated and reserved memory.
- A fused kernel is part of the quantization result. Log the selected backend.
- Batch, prompt length, and open-loop arrival rate determine whether a memory win becomes a serving win.
- Perplexity is a regression detector; pair it with reasoning and code or product-specific task evaluations.
- MoE active parameters describe per-token compute, not necessarily resident memory.
- Every table row needs a derived, cited, or reader-reproducible source label.
- Use vLLM’s public reports as benchmark targets and contrast, never as first-hand measurements in this post.
- Ship the format that passes the product’s quality gate, not the format with the prettiest bit-width headline.

## Further reading

- [Meta Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B), July 23, 2024.
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B), accessed August 3, 2026.
- [Gemma 3 model card](https://huggingface.co/google/gemma-3-12b-it), accessed August 3, 2026.
- [PyTorch torchao quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html), updated March 25, 2026.
- [vLLM FP8 KV-cache report](https://vllm.ai/blog/2026-04-22-fp8-kvcache), April 22, 2026.
- [PTPC-FP8 on AMD ROCm](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm), February 24, 2025.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks).
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
