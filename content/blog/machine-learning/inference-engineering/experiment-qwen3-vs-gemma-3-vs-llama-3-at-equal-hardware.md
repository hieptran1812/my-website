---
title: "Qwen3 versus Gemma 3 versus Llama 3.1 at equal hardware: The architecture bill arrives at decode time"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Compare three open models on one fixed GPU by tracing KV heads, vocabulary, tokenizer output, context contracts, and tied embeddings all the way to dollars per million tokens."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "benchmarking",
    "tokenizer",
    "kv-cache",
    "latency",
    "throughput",
    "ml-systems",
    "pytorch",
    "gpu",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 32
---

The most expensive sentence in an inference review is usually: “They are all about eight billion parameters, so the speed should be roughly the same.” It sounds sensible until the first decode step. One model carries 32 layers and 8 KV heads of width 128. Another carries 36 layers with the same KV-head count. A third carries 48 layers and doubles the head width. Their weight tables, output vocabularies, tokenizer boundaries, and context policies are different before a kernel launches.

![A fixed RTX 4090 feeding three model architectures whose metadata becomes KV traffic token count weight memory and cost](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-1.webp)

The diagram above is the mental model: the GPU is equal, but the bill is not. A benchmark that reports only “tokens per second” hides the causes. We need to explain how many tokens the prompt became, how many KV bytes every generated token rereads, whether the output projection is duplicated, and whether the advertised context is native or an extension. Then we can attach a price to the result.

This post builds that comparison for Qwen3-8B, Gemma-3-12B, and Llama-3.1-8B. The factual configuration values are dated public values from the model cards and configuration files available on 2026-08-03: Qwen3-8B reports 8.2B parameters, 36 layers, 32 query heads, 8 KV heads, a 32,768-token native context, and a 131,072-token YaRN extension in its [official model card](https://huggingface.co/Qwen/Qwen3-8B) (accessed 2026-08-03); its [official config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) reports a 151,936 vocabulary and untied embeddings (accessed 2026-08-03). Google’s [Gemma-3-12B model card](https://huggingface.co/google/gemma-3-12b-it) (accessed 2026-08-03) reports a 128K context, and the [Gemma config](https://huggingface.co/google/gemma-3-12b-it/blob/main/config.json) reports 48 layers, 16 query heads, 8 KV heads, 256-dimensional heads, a 262,208 vocabulary, sliding-window attention, and tied word embeddings (accessed 2026-08-03). Meta’s [Llama-3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B) and [configuration](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json), both accessed 2026-08-03, report about 8B parameters, 32 layers, 32 query heads, 8 KV heads, 128-dimensional heads, a 128,256 vocabulary, 131,072 positions, and untied embeddings.

I have no GPU and have run none of the measurements below. Derived values are shown as arithmetic. Tokenizer counts and speed ranges are explicitly reader-reproducible. Public vLLM results are cited as vLLM results and are not presented as runs of `nanoserve`.

## 1. Equal hardware is not equal work

The first useful correction is to define “equal.” Equal hardware means the same GPU model, driver, CUDA stack, precision, engine version, clock policy, prompt text, batch policy, and latency target. It does not mean equal input token count, equal output vocabulary, equal cache traffic, or equal model quality.

Here is the compact comparison we will expand throughout the post.

| Dimension | Qwen3-8B | Gemma-3-12B | Llama-3.1-8B | Source |
|---|---:|---:|---:|---|
| Parameters | 8.2B | 12B class | 8.03B | cited: official model cards, accessed 2026-08-03 |
| Layers | 36 | 48 | 32 | cited: official configs, accessed 2026-08-03 |
| Query heads | 32 | 16 | 32 | cited: official configs, accessed 2026-08-03 |
| KV heads | 8 | 8 | 8 | cited: official configs, accessed 2026-08-03 |
| Head dimension | 128 | 256 | 128 | cited: official configs, accessed 2026-08-03 |
| Vocabulary | 151,936 | 262,208 | 128,256 | cited: official configs, accessed 2026-08-03 |
| Word embeddings tied? | No | Yes | No | cited: official configs, accessed 2026-08-03 |
| Context statement | 32K native, 131K with YaRN | 128K | 131K | cited: official cards/configs, accessed 2026-08-03 |
| BF16 KV bytes per token, conservative | 144 KiB | 384 KiB | 128 KiB | derived below |

The last row is the first surprising one. Qwen3 and Llama have the same KV-head count and head width, but Qwen3 has four more layers, so its cache is 12.5% larger. Gemma has the same eight KV heads but a 256-wide head and more layers; the conservative all-layer calculation is three times Llama’s per-token cache. Gemma’s sliding-window pattern changes the long-context behavior of that number, but it does not make the architectural fact disappear: every attention layer still has a layout and a policy the engine must honor.

**Senior rule:** compare the bytes moved by one decode step before comparing the marketing parameter count.

### What the benchmark must hold fixed

The fair test uses one GPU, one dtype, one attention backend, one maximum batch policy, and the same four prompt families used throughout this series: chat, retrieval-augmented generation, code completion, and translation. “Same prompt” means the same Unicode text. It does not mean the same number of input IDs; that number is an output of each tokenizer and must be reported, not silently equalized away.

The engine should be `nanoserve` for the constructive part of the series, with vLLM as a public benchmark target and contrast. The vLLM team’s public material, including its [benchmarking and Blackwell comparison](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) dated 2025-10-09, frames results as a throughput/latency Pareto frontier and reports “up to 4×” higher throughput at similar latency for Blackwell versus Hopper in selected setups. That is useful context, not a result for these three models on our machine. The same discipline applies to every number in this post.

## 2. KV heads turn model config into decode traffic

The key-value cache stores the key and value vectors produced during prefill so decode does not recompute the entire prefix. For one token, one layer, one KV head, and BF16, there are two tensors, each with `head_dim` values, each taking two bytes. Across layers and KV heads:

$$
 B_{\text{KV/token}} = 2 \cdot L \cdot H_{\text{KV}} \cdot D \cdot b,
$$

where $L$ is the layer count, $H_{\text{KV}}$ is the number of key-value heads, $D$ is head dimension, and $b=2$ bytes for BF16. This is an exact layout calculation for a conventional full-attention cache, not a benchmark estimate.

![A side by side cache matrix showing layers KV heads head dimension and the resulting BF16 bytes per token for all three models](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-2.webp)

Substitute the public configuration values:

$$
\begin{aligned}
\text{Llama} &: 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072\text{ bytes} = 128\text{ KiB/token},\\
\text{Qwen3} &: 2 \cdot 36 \cdot 8 \cdot 128 \cdot 2 = 147{,}456\text{ bytes} = 144\text{ KiB/token},\\
\text{Gemma} &: 2 \cdot 48 \cdot 8 \cdot 256 \cdot 2 = 393{,}216\text{ bytes} = 384\text{ KiB/token}.
\end{aligned}
$$

At 8,192 cached tokens, those become 1.125 GiB, 1.266 GiB, and 3.000 GiB respectively. At 32,768 tokens they become 4.500 GiB, 5.063 GiB, and 12.000 GiB. Those are cache bytes only; weights, temporary activations, CUDA graphs, allocator slack, and logits are not included.

#### Worked example: the same 8K request

Suppose three users send an 8,192-token prompt and each begins decoding one token. The cache reservation for one request is just the per-token number times 8,192. Llama needs $128\text{ KiB} \times 8192 = 1.000\text{ GiB}$. Qwen3 needs $144\text{ KiB} \times 8192 = 1.125\text{ GiB}$. Gemma’s conservative full-layer number is $384\text{ KiB} \times 8192 = 3.000\text{ GiB}$.

For three requests, the same arithmetic is 3.000 GiB, 3.375 GiB, and 9.000 GiB. No kernel benchmark is needed to establish this difference. It is the memory reservation implied by the published shapes. A real Gemma engine must additionally apply its local/global attention policy; the model’s configuration names a 1,024-token sliding window and a six-layer pattern, so a production allocator should distinguish local-window storage from global-attention storage instead of blindly using the conservative number. The conservative row remains useful as a safety bound and as a test for a naive engine.

### KV heads are not query heads

All three models use grouped-query attention (GQA): several query heads share one key head and one value head. The query-to-KV grouping ratios are $32/8=4$ for Qwen3 and Llama, and $16/8=2$ for Gemma. A cache calculation using query heads would overstate the Llama and Qwen3 cache by four times and the Gemma cache by two times.

This is exactly the sort of bug that can make a home-grown engine “mysteriously” OOM. The model config says `num_attention_heads`; the cache needs `num_key_value_heads`. In `nanoserve`, make the distinction impossible to miss:

```python
# nanoserve/cache_math.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AttentionShape:
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    bytes_per_value: int = 2

    def kv_bytes_per_token(self) -> int:
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must divide evenly into KV heads")
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_value

    def grouping_ratio(self) -> int:
        return self.query_heads // self.kv_heads

SHAPES = {
    "qwen3-8b": AttentionShape(36, 32, 8, 128),
    "gemma-3-12b": AttentionShape(48, 16, 8, 256),
    "llama-3.1-8b": AttentionShape(32, 32, 8, 128),
}

for name, shape in SHAPES.items():
    print(name, shape.grouping_ratio(), shape.kv_bytes_per_token())
```

Expected output is reproducible from the config values: grouping ratios 4, 2, and 4; cache bytes 147456, 393216, and 131072 in the dictionary order. The script does not claim a GPU result. It is a unit-testable invariant for the allocator.

### The local-attention trap

Gemma 3 is not a plain “store every token forever in every layer” transformer. Its configuration contains sliding-window attention metadata. That changes the growth curve for local layers: once a request exceeds the window, the local layer’s live cache need not grow with the entire prefix. Global layers still grow. This is why the correct Gemma benchmark reports separate local and global cache occupancy, not a single number copied from Llama.

An abstraction is useful here: **the following is a planning model, not Gemma’s exact implementation formula.** If one out of every six layers is global and the other five retain a 1,024-token window, then a rough long-context upper estimate is

$$
B(S) \approx B_{\text{token}}\left(\frac{S}{6}+\frac{5}{6}\cdot 1024\right),
$$

where $S$ is prompt length and $B_{\text{token}}$ is the all-layer per-token figure. Labeling this as an abstraction matters. The actual engine must read the checkpoint’s attention pattern and implement its exact cache semantics.

## 3. Vocabulary changes both memory and token economics

Vocabulary size is normally introduced as a tokenizer property. For inference it is also an output-projection property. If the input embedding and output `lm_head` are untied, the model stores two matrices shaped roughly `[vocab_size, hidden_size]`. If they are tied, the same matrix can serve both roles.

![A before and after comparison showing smaller untied output tables for Llama and Qwen3 versus Gemma’s larger vocabulary with tied embeddings](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-3.webp)

For BF16, one untied output matrix costs:

$$
B_{\text{untied output}} = V \cdot H \cdot 2,
$$

where $V$ is vocabulary size and $H$ is hidden size. Using the public configs:

| Model | Vocabulary $V$ | Hidden size $H$ | One output table | Tied? | Source |
|---|---:|---:|---:|---|---|
| Llama-3.1-8B | 128,256 | 4,096 | $128256 \times 4096 \times 2 = 1.052$ GB = 0.979 GiB | No | derived from cited config |
| Qwen3-8B | 151,936 | 4,096 | $151936 \times 4096 \times 2 = 1.245$ GB = 1.159 GiB | No | derived from cited config |
| Gemma-3-12B | 262,208 | 3,840 | $262208 \times 3840 \times 2 = 2.014$ GB = 1.875 GiB | Yes | derived from cited config |

Gemma’s vocabulary table is physically larger, but tying avoids storing a second copy. Qwen3’s table is about 19% larger than Llama’s because $151936/128256 \approx 1.185$. Gemma’s table is about 1.92 times Llama’s because $262208 \times 3840 /(128256 \times 4096) \approx 1.917$, but the tie means that comparison is not a direct extra-weight comparison. Tied embeddings are a memory win, not a claim that a larger vocabulary is automatically better.

The output logits still have shape `[batch, vocab_size]`. At batch 1, the final projection must produce 151,936, 262,208, or 128,256 scores. At a large batch, the projection becomes a larger matrix operation. At small batch, the extra output width can matter to latency and memory bandwidth even when the transformer blocks dominate total time. Report the output-projection time if your engine can isolate it; otherwise do not attribute every difference to attention.

### Implementing the tied-embedding branch

The forward pass should expose the choice rather than infer it from a parameter name. This snippet is runnable with ordinary PyTorch tensors and is the kind of explicit branch `nanoserve` needs.

```python
# nanoserve/output_head.py
import torch
from torch import nn

class OutputHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, tied: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.tied = tied
        self.lm_head = None if tied else nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.weight if self.tied else self.lm_head.weight
        return hidden @ weight.t()

head = OutputHead(262_208, 3_840, tied=True)
x = torch.randn(2, 4, 3_840)
print(head(x).shape)  # torch.Size([2, 4, 262208])
```

The shape is the same in both cases; the parameter storage is not. A common incorrect optimization aliases the matrices without checking `tie_word_embeddings`. That silently changes a model whose config says `false`. Configuration fidelity beats cleverness.

## 4. Tokenizer efficiency: count words, not vibes

A token is an engine unit, not a linguistic word. The tokenizer may represent one English word as one token, several subword tokens, or a token plus a leading-space convention. Vietnamese is especially important to test because word boundaries include spaces but syllables are separated by spaces; a tokenizer trained on a different distribution can split it differently. The right measurement is not “the vocabulary is larger, therefore Vietnamese is cheaper.” The right measurement is tokens per word on the exact prompt suite.

![A timeline from identical Unicode text through three tokenizers into different ID lengths and then different prefill cache and billing work](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-4.webp)

The tokenizer experiment below deliberately uses escaped Unicode sequences for the Vietnamese strings, so the source file remains plain ASCII while Python reconstructs the actual text. It measures words with a simple whitespace definition, then reports tokens per whitespace-delimited word. This is a reproducible operational metric, not a universal linguistic score.

```python
# nanoserve/bench/tokenizer_efficiency.py
from statistics import mean
from transformers import AutoTokenizer

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

TEXT = {
    "english": [
        "The cache allocator should make memory ownership visible.",
        "A short prompt can still create a long decode stream.",
    ],
    "vietnamese": [
        "Hệ thống phải đếm token trước khi tính chi phí.",
        "Bạn có thể kiểm tra bằng cùng một bộ dữ liệu.",
    ],
}

for label, model_id in MODELS.items():
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    print(label)
    for language, samples in TEXT.items():
        ratios = []
        for text in samples:
            ids = tok(text, add_special_tokens=False).input_ids
            words = text.split()
            ratios.append(len(ids) / len(words))
        print(language, "tokens_per_word", mean(ratios), "ratios", ratios)
```

Run it with the exact tokenizer revisions used by the model weights, record the commit hashes, and include the output in your benchmark artifact. The expected range is reader-reproducible rather than a number I will fabricate: English should usually land near one to two tokens per whitespace word for these modern tokenizers; Vietnamese may be materially higher, and the ordering is an empirical result of this script. Do not turn that qualitative expectation into a precise percentage without a named corpus and date.

The suite should contain at least 1,000 words per language and should preserve punctuation, casing, code identifiers, and diacritics. A pair of hand-written sentences is a smoke test, not a production conclusion. For a real report, publish the corpus checksum, tokenizer revision, special-token policy, and whether chat-template control tokens were included.

### Why token count changes more than billing

If the tokenizer emits $T_{\text{in}}$ input tokens, prefill processes $T_{\text{in}}$ positions and allocates cache for them. If it emits $T_{\text{out}}$ output tokens, the endpoint performs $T_{\text{out}}$ decode iterations and charges that many output tokens under a token-priced provider. Even on self-hosted hardware, output token count changes the time to finish and therefore capacity.

For a simplified decode-dominated cost model, if the GPU rate is $r$ dollars per hour and goodput is $G$ output tokens per second:

$$
\text{cost per million output tokens} = \frac{10^6}{G \cdot 3600} \cdot r.
$$

This is a derived abstraction. It ignores idle time, replicas, storage, electricity, and engineering labor, so it is a comparison tool rather than a finance ledger. At the same $G$, a tokenizer that needs 20% more output tokens makes a fixed semantic workload 20% more expensive. At the same token count, a model with a larger KV footprint may reduce $G$ by lowering concurrency. Both effects compound.

## 5. Context length has three meanings

“Supports 128K” can mean at least three different things:

1. The checkpoint was trained with a native context near 128K.
2. The config advertises a position-extension method such as YaRN or linear RoPE scaling.
3. The engine accepts a `max_model_len` of 128K without immediately rejecting the request.

Those are not equivalent quality or economics claims.

![A layered stack separating native training context position encoding extension engine VRAM budget and practical latency quality SLO](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-5.webp)

Qwen3’s official card says 32,768 native and 131,072 with YaRN, while Llama’s config advertises 131,072 positions with Llama 3 RoPE scaling. Gemma’s official model card describes a 128K context for the 12B size. These are cited metadata values, not proof that all three maintain the same recall quality at the maximum. A benchmark must pair context length with a retrieval or long-context task score.

The memory math exposes why context is an engine constraint. For Llama in BF16, 128 KiB per token means 32,768 cached tokens require $128\text{ KiB} \times 32768 = 4.000$ GiB. Qwen3 requires $144\text{ KiB} \times 32768 = 4.500$ GiB. The conservative Gemma figure requires 12.000 GiB. Add the weight footprint and a 24 GiB card can become cache-bound long before the model’s declared position limit.

#### Worked example: a 24 GiB card at 32K

Use a deliberately transparent planning allowance: reserve 2 GiB for the CUDA context, activations, allocator slack, and workspace. This 2 GiB is an explicit planning assumption, not a hardware measurement. The remaining budget is $24-2=22$ GiB before weights. If BF16 weight bytes are approximated as parameter count times two, the public parameter labels give:

| Model | Approximate BF16 weights | Residual KV budget | Full-cache 32K KV | Source |
|---|---:|---:|---:|---|
| Llama-3.1-8B | $8.03\times10^9\times2=16.06$ GB = 14.96 GiB | $22-14.96=7.04$ GiB | 4.00 GiB | derived + cited parameter count |
| Qwen3-8B | $8.2\times10^9\times2=16.40$ GB = 15.27 GiB | $22-15.27=6.73$ GiB | 4.50 GiB | derived + cited parameter count |
| Gemma-3-12B | $12\times10^9\times2=24.00$ GB = 22.35 GiB | $22-22.35<0$ | 12.00 GiB | derived planning bound |

The Gemma row says the obvious but important thing: a 12B BF16 checkpoint does not fit in this simplified allowance on a 24 GiB card before cache. Quantization, offload, or a larger GPU changes the decision. The arithmetic is not a performance claim and it omits tensor metadata; it is a feasibility screen.

### Native versus extended context in the benchmark

Run separate rows for 4K, 8K, 32K, and the highest context the checkpoint documents. For Qwen3, report 32K as native and the 131K YaRN configuration as an extension row, with the exact RoPE override in the command line. For Gemma, use 128K as the documented context but report quality and cache policy. For Llama, state whether you are using the released configuration and whether the engine honors the scaling fields.

Never pool native and extended rows into a single average. A position extension can preserve usable quality for some tasks and fail for others. The reader needs to see the trade, not a single headline number.

## 6. The benchmark grid and runnable `nanoserve` harness

The experiment is a factorial design: three models crossed with four workload families, four input lengths, two output lengths, and several concurrency levels. The first pass can be smaller; the important thing is that the axes are explicit.

![A grid crossing chat RAG code and translation with fixed input output lengths and the measurement each workload exposes](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-6.webp)

| Workload | Input target | Output target | Primary signal | Source |
|---|---:|---:|---|---|
| Chat | 512 tokens | 256 tokens | decode TPOT, output token count | reproduce: `bench.py` |
| RAG | 4,096 tokens | 128 tokens | TTFT, prefill memory, KV growth | reproduce: `bench.py` |
| Code completion | 1,024 tokens | 256 tokens | tokenizer boundaries and decode | reproduce: `bench.py` |
| Translation | 512 tokens | 256 tokens | English/Vietnamese token ratio | reproduce: `bench.py` |

The input target is per tokenizer. Build text until the target is reached, rather than assuming 4,096 characters equals 4,096 tokens. The output target is a forced length for the performance run; a separate quality run uses natural stopping.

Here is a minimal OpenAI-compatible client harness. It assumes a `nanoserve` server exposes a compatible `/v1/completions` endpoint, but it is equally useful against a vLLM endpoint as a benchmark target. It measures client-visible TTFT and total completion time, which includes queueing.

```python
# nanoserve/bench/equal_hardware.py
import argparse
import asyncio
import json
import time
from statistics import median
import httpx

async def one(client, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    first = None
    async with client.stream("POST", url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data:") and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return {"ttft_ms": (first or end - t0) * 1000,
            "total_ms": (end - t0) * 1000}

async def main(args):
    prompts = ["Explain why KV heads matter for decode."] * args.requests
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        jobs = [one(client, args.url, args.model, p, args.max_tokens)
                for p in prompts]
        results = await asyncio.gather(*jobs)
    print(json.dumps({"model": args.model,
                      "ttft_median_ms": median(r["ttft_ms"] for r in results),
                      "total_median_ms": median(r["total_ms"] for r in results),
                      "n": len(results)}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=120.0)
    asyncio.run(main(p.parse_args()))
```

For a real experiment, add server-side usage accounting, a request ID, p50/p95/p99, inter-token gaps, prompt token count, output token count, peak allocated memory, and an open-loop arrival generator. The closed concurrent batch above is a smoke test, not a capacity benchmark.

### Measurement protocol

Warm each model and each shape before recording. Run at least 30 discarded decode steps per shape, use CUDA events inside the engine for kernel timing, and use client wall time for end-to-end latency. Call `torch.cuda.synchronize()` before and after a diagnostic timing block; do not mistake asynchronous enqueue time for execution time. Fix seeds for prompt generation and sampling, set temperature to zero for deterministic performance prompts, and record the model and tokenizer revisions.

Lock clocks if the hardware and policy permit it. If not, record clock ranges and widen the expected range. On an RTX 4090, the [official NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) list 24 GB memory and 1,008 GB/s memory bandwidth (accessed 2026-08-03). On an A100 80GB SXM, NVIDIA’s [datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) lists 80 GB and 2,039 GB/s HBM2e bandwidth (accessed 2026-08-03). These are specifications, not achieved application bandwidth.

## 7. From bytes to a speed ceiling and a price

Decode is not only attention. The model must read weights, run matrix operations, read the request’s KV cache, and write the next token’s state. A useful lower-bound intuition for a decode-dominated, batch-one step is:

$$
t_{\text{floor}} \approx \frac{B_{\text{weights}} + B_{\text{KV read}}}{\text{HBM bandwidth}}.
$$

This is an abstraction, not an exact latency formula. It ignores kernel efficiency, compute ceilings, launch overhead, quantization, cache reuse, and overlap. It is valuable because it stops impossible claims. If the derived floor is 10 ms, a reported 1 ms decode step needs an explanation such as larger batch amortization, quantization, a different definition of “step,” or a measurement bug.

At batch 1 and a 1,024-token context, the conservative KV read per generated token is approximately 1,024 times the per-token KV footprint: Llama 128 MiB, Qwen3 144 MiB, Gemma 384 MiB. Add approximate BF16 weights of 14.96 GiB, 15.27 GiB, and 22.35 GiB respectively. On a 4090’s specified 1,008 GB/s, the pure-byte floors are not application predictions; they are arithmetic ceilings that show why Gemma has less room for cache and why memory traffic dominates the discussion.

Batching changes the weight term. If one matrix multiplication reuses a weight tile across $B$ requests, the weight bytes per generated token can approach $B_{\text{weights}}/B$ in a simplified model, while each request still reads its own KV. This is why a batch-1 comparison and a batch-32 comparison answer different questions. A model that is slower at batch 1 can have better aggregate economics if it sustains more useful requests before the cache budget collapses; a model with a smaller cache can lose if its tokenizer emits substantially more output tokens for the same task.

### Cost table with provenance

To make cost comparable without pretending to know your cloud bill, choose a stated illustrative rate. The table below uses $1.50 per GPU-hour as a placeholder assumption, not a market quote. Replace it with your reserved or on-demand rate.

| Quantity | Formula | Example result | Source |
|---|---|---:|---|
| Llama cache at 8K | $128\text{ KiB}\times8192$ | 1.000 GiB | derived |
| Qwen3 cache at 8K | $144\text{ KiB}\times8192$ | 1.125 GiB | derived |
| Gemma conservative cache at 8K | $384\text{ KiB}\times8192$ | 3.000 GiB | derived |
| Llama untied table | $128256\times4096\times2$ | 0.979 GiB | derived from cited config |
| Qwen3 untied table | $151936\times4096\times2$ | 1.159 GiB | derived from cited config |
| Gemma one table | $262208\times3840\times2$ | 1.875 GiB | derived from cited config |
| Cost at 100 goodput tok/s | $10^6/(100\times3600)\times1.50$ | $4.17 / 1M | derived illustrative rate |
| Cost at 200 goodput tok/s | $10^6/(200\times3600)\times1.50$ | $2.08 / 1M | derived illustrative rate |

Those last two rows are deliberately not model results. They show the arithmetic a result table should use after the reader measures goodput. If Qwen3’s tokenizer produces 1.2 times as many output tokens for a semantic workload, its effective cost per semantic million tokens is $1.2$ times the token-based figure before any hardware difference. If Gemma’s global attention cache reduces concurrency from 16 to 8 at a fixed SLO, its goodput can fall even if its per-request TPOT looks competitive.

## 8. What the same prompt suite should reveal

The goal is not to crown one model. Each workload stresses a different layer of the engine.

![Three branches from a fixed GPU and SLO to English chat mixed-language translation and long-context retrieval decisions](/imgs/blogs/experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware-7.webp)

### Chat: decode and output vocabulary

Chat with 512 input tokens and 256 forced output tokens is mostly a decode experiment after warmup. Report TPOT, output token count, p99, and aggregate goodput at several concurrency levels. The output projection vocabulary is part of the work, but it may not dominate the transformer blocks. Treat it as a measured component, not a story you assume from vocabulary size.

### RAG: prefill and cache budget

RAG with 4,096 input tokens and 128 output tokens stresses TTFT and cache allocation. A tokenizer that emits 4,600 tokens from the nominal prompt increases prefill and cache by 12.3% before the model has generated anything. Report the actual input IDs, not the intended character length. At long contexts, Gemma’s local/global cache behavior must appear in the memory trace.

### Code: punctuation and identifier boundaries

Code completion is a tokenizer stress test because identifiers, indentation, punctuation, and common code fragments have different merge behavior. Use a fixed repository snapshot and language mix. Do not infer tokenizer quality from natural-language samples.

### Translation: English-to-Vietnamese and Vietnamese-to-English

Translation exposes both input and output token ratios. A direction that emits more tokens costs more decode steps even if the semantic sentence length is the same. Evaluate translation quality separately; a lower token count is not a license to accept a worse translation. The tokenizer script reports token efficiency, while a task evaluator reports quality.

## 9. Failure modes that make the comparison dishonest

**Padding one model to another model’s token count.** This removes a real production cost. Keep both rows: equal Unicode prompt and equal token-count prompt. The first measures user experience; the second isolates architecture at a matched sequence length.

**Using parameter count as weight memory.** Parameter labels are rounded, checkpoints contain metadata and sometimes padding, and tied embeddings change storage. Use the actual safetensors index for a memory report. The arithmetic above is a planning estimate, not a file-size claim.

**Treating maximum context as useful context.** Add a retrieval task at every context length. The [RULER benchmark](https://arxiv.org/abs/2404.06654), published 2024-04-10, reports that many models degrade well before their advertised long-context limit. Cite that public finding; do not turn it into a claim about these exact three checkpoints without running the task.

**Mixing local and global attention.** A generic full-cache allocator overstates some Gemma storage, while a generic sliding-window allocator can under-allocate global layers. Read the model’s attention pattern and test the block table against a reference implementation.

**Comparing different engines.** A vLLM run against a `transformers.generate` run is not a model comparison. Use one engine for all three, then run vLLM as a separate benchmark target if you want to understand the cost of the engine gap. The vLLM team’s [Qwen3-Next serving post](https://vllm.ai/blog/2025-09-11-qwen3-next), dated 2025-09-11, is a good example of why model-specific cache policy matters: it describes a hybrid KV manager that tunes logical blocks for different state sizes. That is a cited contrast, not evidence that our engine matches it.

**Reporting a warmup result.** Compilation, autotuning, allocator growth, and graph capture belong outside the steady-state window. Report cold start separately.

**Measuring only closed-loop throughput.** A closed loop waits for the server to finish before issuing the next request and can hide queue collapse. Add an open-loop Poisson arrival sweep and report p99 plus goodput under the same SLO.

## 10. Case studies and public comparisons

### The vLLM Pareto curve, not a single speed number

The vLLM team’s [NVIDIA Blackwell comparison](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), dated 2025-10-09, reports up to 4× higher throughput at similar latency versus Hopper in selected scenarios and gives workload shapes such as 1K/1K chat, 1K/8K reasoning, and 8K/1K summarization. The lesson for this post is methodological: throughput must be paired with latency and scenario. We should use the same three-model matrix rather than quote a maximum token rate detached from context and output length.

### The tokenizer is part of the endpoint contract

The official [Qwen tokenizer note](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md), accessed 2026-08-03, describes Qwen’s BPE/tiktoken-based tokenizer and distinguishes regular and control vocabulary entries. The Qwen3 model card separately reports multilingual support. Neither source gives a universal English or Vietnamese tokens-per-word result for this exact comparison, so the honest artifact is the script and its corpus-specific output. If someone presents a single ratio without the corpus, tokenizer revision, and special-token policy, ask for those before using the number in a capacity plan.

### The long-context warning

RULER’s [paper](https://arxiv.org/abs/2404.06654), published 2024-04-10, evaluated long-context abilities across multiple tasks and models and found that advertised context length did not imply stable performance at that length. This is why the benchmark here has both memory/latency rows and a recall-sensitive quality row. A model can be the cheapest way to generate a token and still be the wrong model for a long-context retrieval endpoint.

### The architecture difference hidden by “8B”

Llama’s and Qwen3’s same 8-KV-head, 128-wide arrangement makes their cache footprints relatively close, with Qwen3 paying for four additional layers. Gemma’s 256-wide head and 48 layers make the conservative cache footprint three times Llama’s, while tied embeddings recover a separate output table. None of those facts predict answer quality. They do predict which instrumentation to add and which constraint is likely to bind first.

## 11. How to read the result without fooling yourself

The first result table should be boring. One row per model, workload, context, concurrency, precision, and engine revision. The minimum columns are model revision, tokenizer revision, input IDs, output IDs, TTFT p50/p99, TPOT p50/p99, aggregate output goodput, peak allocated memory, peak reserved memory, cache blocks used, and a quality score. If a value is unavailable, write `not reported`; do not fill the cell with a plausible estimate.

The second table should be a derived audit. For each row, recompute the expected cache reservation from the actual input token count and the model shape. If the measured allocator is 5.5 GiB and the shape arithmetic says 4.5 GiB, the difference should be labeled: activations, workspace, block rounding, graph capture, or unexplained. “Framework overhead” is not an explanation; it is a queue for the next instrument.

### A result interpretation table

| Observation | Likely mechanism | Next check | Decision impact |
|---|---|---|---|
| TTFT rises while TPOT is stable | prefill or tokenizer work | input IDs, CUDA event around prefill | shorter prompts or chunked prefill |
| TPOT rises with context | KV read and attention work | cache bytes and achieved bandwidth | smaller KV footprint or shorter context |
| Goodput collapses at high concurrency | admission or cache ceiling | queue time and block occupancy | change batching policy first |
| Tokenizer ratio differs by language | merge coverage and text distribution | corpus histogram by language | price semantic work by language |
| Memory is lower but quality falls | context extension or quantization effect | recall task at each length | reject the apparent capacity win |
| Gemma fits only after offload | weight plus cache budget binds | PCIe transfer timeline | larger GPU or quantization |

This is where `nanoserve` earns its place. A black-box endpoint can tell you that p99 got worse. The engine should tell you whether the request waited for a block, whether that block was local-window or global, how many IDs the tokenizer emitted, and whether the output projection ran with a tied matrix. The numbers become useful when they point to a code path.

### Reader-reproducible expected ranges

The series honesty rule allows expected ranges only when they name the hardware, precision, workload, and script. For this post, do not write a universal “Qwen3 is 15% faster.” A legitimate reader-facing statement looks like: “On an RTX 4090 24GB, BF16, batch 1, 512 input tokens, 128 forced output tokens, after 30 warmup steps, run `python -m nanoserve.bench.equal_hardware`; report the resulting range.” If there is no sourced public range, leave the range open and report the script instead.

This restraint is not evasive. The models have different kernel support, parameter counts, output widths, and graph-capture shapes. A precise expected range without a cited measurement would be invented. The derived cache values are already stronger than a fake speed table because they explain a constraint every correct implementation must face.

### A practical run sequence

Use a fresh process for each model so memory fragmentation and compilation caches do not leak across rows. Load the exact revision, print the config hash, print the tokenizer hash, and record `torch.__version__`, CUDA version, driver version, GPU name, memory size, and clocks. Then perform the following sequence:

```bash
# Reader-run protocol; values are intentionally explicit in the artifact.
export CUDA_VISIBLE_DEVICES=0
python -m nanoserve.serve \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --seed 17 \
  --no-prefix-cache

python -m nanoserve.bench.equal_hardware \
  --url http://127.0.0.1:8000/v1/completions \
  --model Qwen/Qwen3-8B \
  --requests 4 \
  --max-tokens 128
```

Repeat with Gemma and Llama, changing only the model identifier and the context policy required by its official configuration. The `--no-prefix-cache` choice is deliberate for the first comparison: a cache hit would otherwise make prompt order part of the model result. Add a second explicitly labeled experiment with prefix caching if the production workload benefits from it.

For open-loop load, generate arrivals independently of completions. A Poisson process with rate $\lambda$ means an inter-arrival draw $\Delta t=-\ln(U)/\lambda$, where $U$ is uniform on $(0,1)$. This is a reproducible distribution, not a claim about your traffic. Sweep rates until p99 crosses the SLO. Report goodput only for requests inside the SLO, and report offered load separately so a server cannot look healthy by quietly dropping late work.

## 12. The decision is a workload map, not a leaderboard

At the end, draw a frontier for each workload: x-axis is p99 or TPOT, y-axis is goodput, and annotate memory occupancy. A point is useful only if the quality score is attached. The likely shape is more informative than the winner: Llama and Qwen3 should be close in conservative KV bytes, Qwen3 pays extra layer traffic, and Gemma trades a larger model/cache shape for tied embeddings and its local/global attention design. The actual frontier depends on kernels and quality and must come from the reader’s run.

One model can win chat and lose long RAG. Another can win English translation and lose Vietnamese because its tokenizer emits more IDs. A third can fit the target card only after quantization, then lose quality on a retrieval task. That is not a failure of the benchmark; it is the answer the product team needs.

The right artifact is therefore not a leaderboard screenshot. It is a directory containing the prompt corpus checksum, tokenizer counts, model configs, launch commands, raw request traces, memory traces, quality outputs, and a small report that recomputes every derived cell. When a model revision changes its tokenizer or context policy, rerun the artifact and see which part of the bill moved.

This is also how to keep the benchmark useful after the series ends. The later [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) can compare `nanoserve` with vLLM, but only if both engines consume the same tokenized workload and report the same SLO-aware goodput. Otherwise the capstone will accidentally compare a model choice, a tokenizer choice, and an engine choice in one number.

## 13. When to reach for each model, and when not to

Choose Llama-3.1-8B as the baseline when you want a widely integrated dense reference with a smaller vocabulary table and a 128 KiB/token conservative BF16 KV footprint. It is a useful spine for comparing engine changes because the configuration is familiar and the 8-query-head grouping is clear.

Choose Qwen3-8B when multilingual coverage, its documented model behavior, or its instruction/reasoning trade-offs fit the product and the extra four layers plus 19% larger vocabulary table are acceptable. Make the tokenizer audit mandatory for your supported languages; do not assume the vocabulary count tells you the result.

Choose Gemma-3-12B when its quality, multimodal capability, or multilingual behavior earns the additional model and cache complexity. Budget for the 12B-class weights, tied embedding semantics, 256-wide heads, and local/global attention. A 24 GiB BF16 deployment may require quantization or offload under the planning arithmetic above.

Do not choose by tokens per second at batch 1 alone. Do not choose by maximum context alone. Do not choose by a public vLLM number measured on another GPU. And do not write a custom engine for production merely to reproduce a model card: use vLLM when its supported backend, scheduler, and operational surface already solve the problem. `nanoserve` is valuable here because it makes the bills visible and gives us a testable learning artifact, not because a toy engine is automatically better.

## Key takeaways

- KV cache bytes use KV heads, not query heads: $2 \cdot L \cdot H_{\text{KV}} \cdot D \cdot b$.
- Qwen3-8B’s published shape derives to 144 KiB per BF16 KV token; Llama-3.1-8B derives to 128 KiB; Gemma-3-12B’s conservative full-layer value derives to 384 KiB.
- Gemma’s sliding-window configuration means the production cache curve is mixed, not a single full-attention line.
- Vocabulary size changes the output projection; tied embeddings change whether that table is duplicated.
- Tokenizer efficiency must be measured on the exact English and Vietnamese corpus, with revisions and special-token policy recorded.
- Native context, position extension, engine acceptance, and useful quality are separate claims.
- The fair experiment fixes GPU, dtype, engine, prompts, service policy, and SLO, then reports actual token counts, TTFT, TPOT, p99, goodput, memory, and quality.
- Cost per million tokens is derived from goodput and GPU-hour rate; semantic cost also depends on tokens per word.
- vLLM is a benchmark target and contrast. Its public numbers are cited with setup and are not first-hand `nanoserve` results.
- The winner is workload-specific: choose the model whose quality clears the SLO at the lowest measured cost, not the model with the prettiest parameter label.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series layer map and honesty rule.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — the cache derivation used here.
- [The tokenizer boundary and incremental detokenization](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) — why token boundaries become a streaming problem.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) — the shared warmup, timing, and load protocol.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the later capstone comparison against vLLM.
- [Qwen3-8B official model card](https://huggingface.co/Qwen/Qwen3-8B) — configuration and context claims, accessed 2026-08-03.
- [Gemma 3 official model card](https://huggingface.co/google/gemma-3-12b-it) — context and multimodal claims, accessed 2026-08-03.
- [Llama 3.1 official model card](https://huggingface.co/meta-llama/Llama-3.1-8B) — configuration and license information, accessed 2026-08-03.

<figure class="blog-anim">
<svg viewBox="0 0 720 180" role="img" aria-label="The same prompt moves through tokenization prefill cache and decode; the highlighted token travels through the stages" style="width:100%;height:auto;max-width:820px">
<style>
.eq-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.eq-label{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.eq-sub{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.eq-dot{fill:var(--accent,#6366f1)}
@keyframes eq-flow{0%{transform:translateX(0);opacity:0}12%{opacity:1}82%{opacity:1}100%{transform:translateX(570px);opacity:0}}
.eq-move{animation:eq-flow 8s ease-in-out infinite}.eq-move2{animation:eq-flow 8s ease-in-out infinite;animation-delay:2.6s}.eq-move3{animation:eq-flow 8s ease-in-out infinite;animation-delay:5.2s}
@media (prefers-reduced-motion:reduce){.eq-move,.eq-move2,.eq-move3{animation:none;opacity:1}}
</style>
<rect class="eq-stage" x="30" y="55" width="140" height="70" rx="10"/><rect class="eq-stage" x="215" y="55" width="140" height="70" rx="10"/><rect class="eq-stage" x="400" y="55" width="140" height="70" rx="10"/><rect class="eq-stage" x="585" y="55" width="105" height="70" rx="10"/>
<text class="eq-label" x="100" y="85">tokenize</text><text class="eq-sub" x="100" y="106">IDs</text><text class="eq-label" x="285" y="85">prefill</text><text class="eq-sub" x="285" y="106">KV write</text><text class="eq-label" x="470" y="85">decode</text><text class="eq-sub" x="470" y="106">KV read</text><text class="eq-label" x="637" y="85">price</text><text class="eq-sub" x="637" y="106">goodput</text>
<circle class="eq-dot eq-move" cx="30" cy="145" r="9"/><circle class="eq-dot eq-move eq-move2" cx="30" cy="145" r="9"/><circle class="eq-dot eq-move eq-move3" cx="30" cy="145" r="9"/>
</svg>
<figcaption>The animated token makes the causal chain visible: tokenizer output becomes prefill work, cache state, decode work, and finally economics.</figcaption>
</figure>
