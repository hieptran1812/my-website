---
title: "The memory math of the KV cache: how many users actually fit on your GPU"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive the bytes-per-token law that decides your concurrency limit, size the KV budget on a 4090 and an A100 down to the byte, and learn why one integer in a config file was worth an eightfold capacity win."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "kv-cache",
    "memory",
    "gqa",
    "quantization",
    "batching",
    "throughput",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 47
---

The last post gave you a working KV cache and a large speedup. This post gives you the bill.

Here is the shape of the problem. You put Llama-3.1-8B on an RTX 4090. The weights are 14.96 GiB of the card's 24 GiB, which sounds comfortable — nine gigabytes to spare. Then you start serving. At about the twenty-eighth simultaneous conversation the engine stops admitting requests, and `nvidia-smi` cheerfully reports that a large fraction of the card is "just" model weights. Nothing leaked. Nothing is broken. You have simply run out of a resource you never explicitly allocated, whose size nobody printed at startup, and which is governed by a five-factor product that lives in a JSON file you have probably never opened.

That resource is the KV cache, and this post is the arithmetic that governs it. By the end you will be able to look at any `config.json` on Hugging Face and state, in under a minute and without a GPU, exactly how many bytes per token that model's cache costs, how many tokens fit in a given card, how many simultaneous users that translates to, at what context length a single request's cache grows larger than the model itself, and what each of the standard mitigations multiplies that number by. This is `nanoserve/kvmath.py` — twenty lines of pure arithmetic that will save you more GPU-hours than any kernel you ever write.

![A twenty-four gibibyte card broken into fixed regions for context weights and activations with the leftover region labelled as the key-value budget](/imgs/blogs/the-memory-math-of-the-kv-cache-1.webp)

The figure above is the whole post in one picture, and it contains the single most important structural fact about inference memory: **the KV cache is a residual.** Nobody sizes it. It is whatever is left after the weights, the activation working set, and the CUDA context have taken their cut. That is why it behaves so violently — a change that moves the weights by one gigabyte moves your concurrency limit by fourteen percent, and a model that is only fifty percent larger can leave you with a KV budget that is three times smaller.

Two promises up front, both restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is). First: **I have no GPU and have run none of this.** Every number below is derived from arithmetic I show you, cited from a public source with a link, or framed as something you should reproduce yourself with a named script. The results tables carry a `Source` column. Second: this is a derivation post, not a survey. The formula is short enough to memorize and I would rather you leave able to re-derive it than able to recite it.

---

## 1. The master formula, one factor at a time

Start from what a decode step actually needs. When the model generates token at position $t$, every attention layer computes a query vector from the current token and attends over the key and value vectors of **all** positions $0 \ldots t$. Those older keys and values do not depend on the current token, so — as [the previous post](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) established — recomputing them is pure waste. You store them instead. What you store, per token, is the whole formula.

![Five configuration constants merging into a single bytes-per-token figure which is then multiplied by context length and by request count](/imgs/blogs/the-memory-math-of-the-kv-cache-2.webp)

$$B_{\text{tok}} \;=\; 2 \;\cdot\; L \;\cdot\; H_{kv} \;\cdot\; d_{\text{head}} \;\cdot\; b$$

Five factors. Take them one at a time, because each one is a different kind of thing and only two of them are anything you can move.

**The 2** is *K and V*. Attention needs both the key projection and the value projection of every past token. They are the same shape, so the factor is exactly two, not "about two". This is the one factor that is genuinely immovable in a standard transformer. (Multi-head latent attention breaks it, which is the entire point of MLA — more on that in section 7.)

**$L$ is the number of decoder layers.** Every layer has its own attention, its own K and V projections, and therefore its own cache. There is no sharing between layers in a standard architecture. This is why layer count matters far more for cache cost than for anything else you might intuitively associate with model size: a deep-and-narrow model is expensive to serve even when it is small to store. `config.json` calls this `num_hidden_layers`.

**$H_{kv}$ is the number of *key/value* heads**, which since 2023 is usually *not* the number of attention heads. In classic multi-head attention every query head has its own key and value head, so $H_{kv} = H$. Grouped-query attention lets several query heads share one KV head, so $H_{kv} = H / g$ for a group size $g$. Multi-query attention is the extreme $H_{kv} = 1$. This is the field the whole industry moved, and section 7 is about why. `config.json` calls it `num_key_value_heads` — and critically, **if that field is absent, it defaults to `num_attention_heads`**, i.e. plain MHA. Getting that default wrong is the most common way to be off by a factor of eight.

**$d_{\text{head}}$ is the per-head dimension.** Usually `hidden_size / num_attention_heads`, but modern configs increasingly state it explicitly as `head_dim` and it is sometimes *not* the quotient — Gemma-3 is the well-known example. Always read the explicit field when it exists.

**$b$ is bytes per stored element.** Two for bf16 or fp16, one for fp8 or int8, four if you are doing something regrettable. This is the second movable factor and section 9 is about it.

Multiply for Llama-3.1-8B, whose config gives $L = 32$, $H_{kv} = 8$, $d_{\text{head}} = 128$, and bf16 storage:

$$B_{\text{tok}} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KiB per token}$$

Walk it in the order the tensors actually exist, because that is the version you will remember. Per layer, per token, the key is a vector of $8 \times 128 = 1024$ numbers and the value is another 1024 — so 2,048 numbers per layer. Across 32 layers that is 65,536 numbers, which is exactly $2^{16}$, and at two bytes each it is exactly 128 KiB. The number is pleasingly round because transformer dimensions are all powers of two, and that roundness is worth exploiting: **for Llama-3.1-8B in bf16, one token of context costs 128 KiB, one thousand tokens costs 128 MiB, eight thousand tokens costs one gibibyte.** Those three facts will carry you through most capacity conversations you will ever have.

Then the two multipliers that turn a per-token cost into a system limit:

$$M_{\text{total}} = B_{\text{tok}} \cdot \sum_{i=1}^{N} S_i \;\approx\; B_{\text{tok}} \cdot N \cdot \bar{S}$$

where $N$ is the number of live requests and $\bar{S}$ their mean sequence length (prompt plus everything generated so far). Note what this does and does not say. It does **not** care about batch size in the tensor-shape sense, or about how many tokens per second you are producing, or about how long the requests have been alive. The KV cache is charged strictly per token-position held in memory. A request that has been idle for thirty seconds mid-stream costs exactly as much as one being actively decoded. That single observation is the seed of the entire scheduler and eviction discussion later in this track.

### 1.1 The immediate consequence: your cache is not small

The temptation is to file the KV cache under "overhead". Run the numbers before you do. A single 8,000-token RAG request against Llama-3.1-8B holds 1.0 GiB of cache. Thirty of them hold 30 GiB — more than the entire 4090, twice over, before the weights. Meanwhile a single 128,000-token request holds 16.0 GiB, which is *larger than the 14.96 GiB of model weights*.

That last comparison is not a curiosity; it is the defining feature of long-context serving and section 6 derives exactly where the crossover sits.

---

## 2. Checking the formula against a number you did not invent

A derivation you cannot falsify is a derivation you should not trust. So before building anything on $B_{\text{tok}}$, let us test it against numbers published by people with actual GPUs.

The vLLM team's [KV cache offloading post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) mentions, almost in passing, the physical block sizes before and after a KV-layout change (PR #27743). It lists three models:

| Model | Old physical block | New physical block | Source |
| --- | --- | --- | --- |
| Llama-3.1-8B | 32 KB | 2 MB | cited: vLLM KV-offloading post (2026-01-08) |
| Llama-3.2-1B | 16 KB | 0.5 MB | cited: vLLM KV-offloading post (2026-01-08) |
| Llama-3.1-70B | 8 KB | 1.25 MB | cited: vLLM KV-offloading post (2026-01-08) |

vLLM's default block size is 16 tokens, and [the Anatomy of vLLM post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) states the per-block formula as `2 (key/value) * block_size * num_kv_heads * head_size * dtype_bytes`. So the *new* layout — one block covering all layers, both tensors — should be $2 \cdot L \cdot 16 \cdot H_{kv} \cdot d \cdot b$, which is just $B_{\text{tok}} \times 16$. Let us check.

Llama-3.1-8B: $128 \text{ KiB} \times 16 = 2048 \text{ KiB} = 2 \text{ MiB}$. The post says 2 MB. **Match.**

Llama-3.2-1B has 16 layers, 8 KV heads, and $d_{\text{head}} = 64$, so $B_{\text{tok}} = 2 \times 16 \times 8 \times 64 \times 2 = 32{,}768$ bytes = 32 KiB, and $32 \text{ KiB} \times 16 = 512 \text{ KiB} = 0.5 \text{ MiB}$. The post says 0.5 MB. **Match.**

Llama-3.1-70B has 80 layers, 8 KV heads, $d_{\text{head}} = 128$, so $B_{\text{tok}} = 320$ KiB and a 16-token block should be 5 MiB. The post says 1.25 MB — off by exactly a factor of four. That is not a refutation; it is a fingerprint. A 70B model does not fit on one GPU, so those numbers were measured under tensor parallelism, and TP shards the KV heads across ranks. At $\text{TP} = 4$ each rank holds two KV heads, giving $2 \times 80 \times 16 \times 2 \times 128 \times 2 = 1{,}310{,}720$ bytes = 1.25 MiB per rank. **Match, at TP=4.**

And the old layout — one block per layer, per tensor — should be $\text{block\_size} \cdot H_{kv} \cdot d \cdot b$: 8B gives $16 \times 8 \times 128 \times 2 = 32$ KiB ✓, 1B gives $16 \times 8 \times 64 \times 2 = 16$ KiB ✓, 70B at TP=4 gives $16 \times 2 \times 128 \times 2 = 8$ KiB ✓. All six published numbers reproduce, and the one that did not match on the first try told us the parallelism degree of the machine it was measured on.

That is the standard I want for every number in this series. The formula is not a heuristic. It is exact, it is checkable against public data, and when it disagrees with reality the disagreement is informative.

---

## 3. The model zoo: cache cost does not track model size

Now run the formula across the models this series uses. The third column normalizes everything to the same 7.04 GiB budget — derived in section 5 — so the models are directly comparable even though their weights differ.

![A comparison grid of five models showing key-value bytes per token cost of one eight-thousand-token request and token capacity in a fixed budget](/imgs/blogs/the-memory-math-of-the-kv-cache-3.webp)

| Model | $L$ | $H_{kv}$ | $d_{\text{head}}$ | $B_{\text{tok}}$ (bf16) | 8k request | Tokens in 7.04 GiB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 32 | 8 | 128 | 128 KiB | 1.00 GiB | 57,671 | derived from config |
| Llama-3.1-70B | 80 | 8 | 128 | 320 KiB | 2.50 GiB | 23,068 | derived from config |
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 1.13 GiB | 51,264 | derived from config |
| Gemma-3-12B (naive) | 48 | 8 | 256 | 384 KiB | 3.00 GiB | 19,223 | derived from config |
| Llama-3.2-1B | 16 | 8 | 64 | 32 KiB | 0.25 GiB | 230,686 | derived from config |
| 70B-class with MHA | 80 | 64 | 128 | 2.5 MiB | 20.0 GiB | 2,883 | derived, counterfactual |
| DeepSeek-V3 MLA | 61 | latent | — | ≈ 39 KiB | 0.30 GiB | 188,902 | derived from cited latent size |

Four things in that table are worth staring at.

**Qwen3-8B has a bigger cache than Llama-3.1-8B despite being a comparable model.** Same KV head count, same head dimension, but 36 layers instead of 32 — so 12.5% more cache per token, forever. Nothing about parameter count told you that. If you are choosing between two similar-quality 8B models for a high-concurrency deployment, the layer count is a serving-cost decision.

**Gemma-3-12B's naive figure is three times Llama-3.1-8B's**, because `head_dim` is 256 rather than the 240 you would get from dividing hidden size by head count. A 12B model with a 384 KiB-per-token cache is a genuinely different serving proposition from an 8B model with 128 KiB. That naive number is also an *overestimate* for Gemma-3, because most of its layers use sliding-window attention — section 10 handles that correction, and it is a large one. Read the `sliding_window` and layer-type fields in the config you actually downloaded; these fields move between releases.

**A 1B model's cache is a quarter of the 8B's**, not an eighth. Layer count fell by half and head dimension by half, but the KV head count did not move at all. Cache cost scales with $L \cdot H_{kv} \cdot d$, and parameter count scales with roughly $L \cdot h^2$ — those are different functions, and they diverge badly as models get wide.

**The MHA counterfactual is brutal.** A 70B-class model with full multi-head attention costs 2.5 MiB per token: a single 8,000-token conversation reserves twenty gibibytes of cache. That row is the reason section 7 exists.

---

## 4. `nanoserve/kvmath.py`: the twenty lines you should own

Every number above came from a config file, so let us write the thing that reads config files. This is the highest value-per-line code in the entire series.

```python
# nanoserve/kvmath.py
"""KV cache arithmetic. No GPU required, no model download required."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Bytes per stored element, keyed by the string you would pass as a kv dtype.
DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1,
    "int4": 0.5, "fp4": 0.5,
}


@dataclass(frozen=True)
class KVShape:
    """Everything about a model that affects KV cache size, and nothing else."""
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    # MLA models cache a latent vector instead of per-head K and V.
    latent_dim: int | None = None      # kv_lora_rank
    rope_dim: int | None = None        # qk_rope_head_dim, cached uncompressed
    # Interleaved sliding-window models cap most layers at a fixed window.
    window: int | None = None
    window_layers: int = 0             # how many of num_layers are windowed

    def bytes_per_token(self, kv_dtype: str = "bfloat16") -> float:
        """Steady-state bytes added to the cache by ONE new token position."""
        b = DTYPE_BYTES[kv_dtype]
        if self.latent_dim is not None:
            # MLA: one compressed latent + one shared RoPE key per layer.
            # The RoPE part is conventionally kept in bf16 even when the
            # latent is quantized, so price it separately.
            per_layer = self.latent_dim * b + (self.rope_dim or 0) * 2
            return self.num_layers * per_layer
        # Standard MHA / GQA / MQA: 2 tensors, per layer, per KV head.
        full_layers = self.num_layers - self.window_layers
        return 2 * full_layers * self.num_kv_heads * self.head_dim * b

    def fixed_bytes(self, kv_dtype: str = "bfloat16") -> float:
        """Per-request bytes that do NOT grow with sequence length."""
        if not self.window_layers or self.window is None:
            return 0.0
        b = DTYPE_BYTES[kv_dtype]
        # Windowed layers saturate at `window` positions and stop growing.
        return 2 * self.window_layers * self.num_kv_heads * self.head_dim * b * self.window

    def bytes_for(self, seq_len: int, kv_dtype: str = "bfloat16") -> float:
        """Total cache for one request holding `seq_len` positions."""
        grow = self.bytes_per_token(kv_dtype) * seq_len
        fixed = self.fixed_bytes(kv_dtype)
        if self.window is not None and seq_len < self.window:
            # Below the window the windowed layers have not saturated yet.
            b = DTYPE_BYTES[kv_dtype]
            fixed = 2 * self.window_layers * self.num_kv_heads * self.head_dim * b * seq_len
        return grow + fixed
```

The parser is the part that bites people, because the defaults are load-bearing:

```python
# nanoserve/kvmath.py (continued)

def from_hf_config(path: str | Path, name: str | None = None) -> KVShape:
    """Build a KVShape from a Hugging Face config.json.

    The defaults here are the whole point. Get `num_key_value_heads`
    wrong and you are off by the GQA group ratio, which is 4x on
    Llama-3.1 and 8x on Llama-3.1-70B.
    """
    cfg = json.loads(Path(path).read_text())
    # Some repos nest the language model config (multimodal checkpoints).
    cfg = cfg.get("text_config", cfg)

    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    # THE default that matters: absent means MHA, not MQA.
    n_kv = cfg.get("num_key_value_heads", n_heads)
    # Prefer the explicit field; Gemma-3 sets it to something that is NOT
    # hidden_size // num_attention_heads.
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads

    latent = cfg.get("kv_lora_rank")           # DeepSeek-style MLA
    rope_dim = cfg.get("qk_rope_head_dim")

    window = cfg.get("sliding_window")
    window_layers = 0
    if window:
        types = cfg.get("layer_types")
        if types:
            window_layers = sum(1 for t in types if "sliding" in t)
        elif cfg.get("sliding_window_pattern"):
            p = cfg["sliding_window_pattern"]   # 1 global every p layers
            window_layers = n_layers - (n_layers // p)
        else:
            window_layers = n_layers            # every layer windowed

    return KVShape(
        name=name or Path(path).parent.name,
        num_layers=n_layers, num_kv_heads=n_kv, head_dim=head_dim,
        latent_dim=latent, rope_dim=rope_dim,
        window=window, window_layers=window_layers,
    )
```

You do not need to download weights to use this. `huggingface_hub` will fetch the 2 KB config on its own:

```python
# nanoserve/kvmath.py (continued)
from huggingface_hub import hf_hub_download

def from_hub(repo_id: str) -> KVShape:
    p = hf_hub_download(repo_id, filename="config.json")
    return from_hf_config(p, name=repo_id)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:,.2f} {unit}"
        n /= 1024


if __name__ == "__main__":
    import sys
    for repo in sys.argv[1:]:
        s = from_hub(repo)
        bpt = s.bytes_per_token()
        print(f"{s.name}")
        print(f"  layers={s.num_layers} kv_heads={s.num_kv_heads} head_dim={s.head_dim}")
        print(f"  bf16 : {human(bpt)}/token   8k request = {human(s.bytes_for(8192))}")
        print(f"  fp8  : {human(s.bytes_per_token('fp8'))}/token")
```

**Reproduce it yourself.** Run this and compare against the table in section 3:

```bash
python -m nanoserve.kvmath \
  meta-llama/Llama-3.1-8B-Instruct \
  Qwen/Qwen3-8B \
  meta-llama/Llama-3.2-1B-Instruct
```

You should see something very close to this — the exact bytes depend on the config revision you pull, which is precisely why you should run it rather than trust my table:

```console
meta-llama/Llama-3.1-8B-Instruct
  layers=32 kv_heads=8 head_dim=128
  bf16 : 128.00 KiB/token   8k request = 1.00 GiB
  fp8  : 64.00 KiB/token
Qwen/Qwen3-8B
  layers=36 kv_heads=8 head_dim=128
  bf16 : 144.00 KiB/token   8k request = 1.13 GiB
  fp8  : 72.00 KiB/token
meta-llama/Llama-3.2-1B-Instruct
  layers=16 kv_heads=8 head_dim=64
  bf16 : 32.00 KiB/token   8k request = 256.00 MiB
  fp8  : 16.00 KiB/token
```

If your output disagrees with mine, your output is right and my table is stale. That is the correct relationship to have with a capacity table.

---

## 5. The concurrency budget: from bytes to users

Now the question that actually gets asked in planning meetings. *How many people can talk to this thing at once?*

The KV budget is a subtraction, and there are exactly four terms:

$$V_{\text{kv}} = V_{\text{total}} \;-\; V_{\text{weights}} \;-\; V_{\text{act}} \;-\; V_{\text{ctx}}$$

**$V_{\text{total}}$** is what the driver actually exposes, which is a little under the marketing number. Read it, do not assume it — `torch.cuda.mem_get_info()` returns `(free, total)` in bytes and total is the honest ceiling.

**$V_{\text{weights}}$** is parameter count times bytes per parameter, plus whatever your loader failed to free. [The weights post](/blog/machine-learning/inference-engineering/loading-weights-safetensors-dtypes-and-device-placement) covers why this is often larger than the arithmetic suggests.

**$V_{\text{act}}$** is the peak transient working set of one forward pass at your maximum batched-token count: hidden states, the MLP intermediate, attention workspace, and — the one people forget — the logits tensor. For a 2,048-token prefill chunk on Llama-3.1-8B, the MLP intermediate alone is $2048 \times 14336 \times 2 = 58.7$ MB, and if you materialize logits for every position it is $2048 \times 128256 \times 2 = 525$ MB. A budget of 1 GiB on a 4090 and 2 GiB on an A100 with larger chunks is a defensible starting allowance; measure it, do not guess it, and section 11 shows how.

**$V_{\text{ctx}}$** is the CUDA context, cuBLAS/cuDNN workspaces, NCCL buffers if you are sharded, and the allocator's own overhead. Roughly a gigabyte on a modern driver, and it is charged before your first tensor exists.

Then the capacity, and finally the answer:

$$T_{\max} = \left\lfloor \frac{V_{\text{kv}}}{B_{\text{tok}}} \right\rfloor \qquad\qquad N_{\max} = \left\lfloor \frac{T_{\max}}{\bar{S}} \right\rfloor$$

#### Worked example: Llama-3.1-8B on one RTX 4090

The 4090 has 24 GiB of GDDR6X and 1,008 GB/s of bandwidth ([NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)). Llama-3.1-8B is 8.03B parameters, so bf16 weights are $8.03 \times 10^9 \times 2 = 16.06 \times 10^9$ bytes $= 14.96$ GiB.

| Term | Value | Source |
| --- | --- | --- |
| Total VRAM | 24.00 GiB | cited: NVIDIA RTX 4090 spec |
| − CUDA context and workspaces | 1.00 GiB | estimate, order of magnitude |
| − Weights, bf16 | 14.96 GiB | derived: 8.03B × 2 bytes |
| − Activation peak at 2,048-token chunk | 1.00 GiB | estimate, order of magnitude |
| **= KV budget** | **7.04 GiB** | derived |
| ÷ 128 KiB per token | **57,671 tokens** | derived: 7.04 × 8192 |
| ÷ 2,048 tokens per request | **28 concurrent requests** | derived |

Twenty-eight. On a card whose spec sheet says 24 gigabytes, serving a model whose weights are only fifteen. The other two thirds of the card are not available to you, and that is not a bug — it is the shape of the problem.

The sensitivity is the interesting part. Ask for 4,096-token conversations instead of 2,048 and you get 14 users. Ask for 8,192 and you get 7. Ask for 32,768 and you get **one**, with 24,903 tokens of budget left over — not enough for a second. The concurrency curve is a hyperbola in $\bar{S}$, and the knee is much closer to the origin than anyone's intuition places it.

<figure class="blog-anim">
<svg viewBox="0 0 720 268" role="img" aria-label="A 24 GiB VRAM bar in which the weight and context regions stay fixed while the KV region fills one request at a time until a twenty-ninth request has nowhere to go" style="width:100%;height:auto;max-width:840px">
<style>
.mm-fixed{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mm-frame{fill:none;stroke:var(--accent,#6366f1);stroke-width:1.5;stroke-dasharray:4 3}
.mm-kv{fill:var(--accent,#6366f1);opacity:.85}
.mm-rej{fill:#dc2626;opacity:.9}
.mm-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mm-sub{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.mm-tag{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mm-red{font:600 12px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
@keyframes mm-fill{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes mm-flash{0%,86%{opacity:0}93%,100%{opacity:1}}
.mm-grow{animation:mm-fill 11.2s steps(28,end) infinite;transform-box:fill-box;transform-origin:left center}
.mm-late{animation:mm-flash 11.2s linear infinite}
@media (prefers-reduced-motion:reduce){.mm-grow{animation:none}.mm-late{animation:none;opacity:1}}
</style>
<text class="mm-hd" x="24" y="30">One RTX 4090, 24 GiB, serving Llama-3.1-8B in bf16</text>
<text class="mm-sub" x="24" y="52">weights and CUDA context never move; only the KV region grows, one admitted request at a time</text>
<rect class="mm-fixed" x="40" y="86" width="27" height="64" rx="4"/>
<rect class="mm-fixed" x="67" y="86" width="399" height="64" rx="4"/>
<rect class="mm-fixed" x="466" y="86" width="27" height="64" rx="4"/>
<rect class="mm-frame" x="493" y="86" width="187" height="64" rx="4"/>
<rect class="mm-kv mm-grow" x="493" y="86" width="187" height="64" rx="4"/>
<rect class="mm-rej mm-late" x="688" y="86" width="24" height="64" rx="4"/>
<text class="mm-tag" x="266" y="122">weights 14.96 GiB</text>
<text class="mm-tag" x="53" y="172">ctx</text>
<text class="mm-sub" x="53" y="188" style="text-anchor:middle">1.0</text>
<text class="mm-tag" x="479" y="204">act</text>
<text class="mm-sub" x="479" y="220" style="text-anchor:middle">1.0</text>
<text class="mm-tag" x="586" y="172">KV budget 7.04 GiB</text>
<text class="mm-sub" x="586" y="188" style="text-anchor:middle">one step = one request = 0.25 GiB = 2048 tokens</text>
<text class="mm-red mm-late" x="700" y="172">29th</text>
<text class="mm-sub" x="24" y="248">Twenty-eight requests of 2048 tokens exhaust the block pool; the twenty-ninth is queued or refused while the GPU still reports most of its memory as model weights.</text>
</svg>
<figcaption>The KV region is the only part of VRAM that moves at request time. It fills in 0.25 GiB steps — one 2048-token conversation each — and when the last block is handed out the twenty-ninth caller waits, even though nothing about the model changed.</figcaption>
</figure>

#### Worked example: the same model on one A100 80GB

An A100 80GB SXM has 80 GiB of HBM2e at 2,039 GB/s ([NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/)). Same model, same weights, larger activation allowance because you will run bigger prefill chunks:

| Term | Value | Source |
| --- | --- | --- |
| Total VRAM | 80.00 GiB | cited: NVIDIA A100 datasheet |
| − CUDA context and workspaces | 1.00 GiB | estimate, order of magnitude |
| − Weights, bf16 | 14.96 GiB | derived: 8.03B × 2 bytes |
| − Activation peak at 8,192-token chunk | 2.00 GiB | estimate, order of magnitude |
| **= KV budget** | **62.04 GiB** | derived |
| ÷ 128 KiB per token | **508,231 tokens** | derived: 62.04 × 8192 |
| ÷ 2,048 tokens per request | **248 concurrent requests** | derived |

Here is the fact worth internalizing: **3.3× the VRAM bought 8.8× the KV capacity.** The residual structure means capacity is super-linear in card size, because the fixed costs — weights, context — are paid once regardless. The corollary is uncomfortable: the closer your weights sit to filling the card, the more violently capacity swings. Push a 20 GiB model onto a 24 GiB card and your budget is 2 GiB; move to a 32 GiB card and it is 10 GiB, a fivefold jump for a 33% hardware upgrade.

This is also the mechanism behind a number vLLM published that looks impossible at first read. Their [distributed inference post](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17) reports that moving from TP=1 to TP=2 gave 13.9× more KV cache blocks and 3.9× more token throughput. Doubling the GPUs cannot double the memory 13.9 times — but it *can*, because with weights $W$ per copy, TP=1 leaves $V - W - a - c$ while TP=2 leaves ${2(V - W/2 - a - c) = 2V - W - 2a - 2c}$. When $W$ is close to $V$ the ratio explodes. A model whose weights fill 83% of one card yields a 12× KV increase at TP=2 by exactly this arithmetic. Sharding does not just add memory; it removes a duplicated fixed cost.

Here is the budget solver, so you stop doing this on napkins:

```python
# nanoserve/capacity.py
from dataclasses import dataclass
from nanoserve.kvmath import KVShape, human

GiB = 1024 ** 3


@dataclass
class Budget:
    total_gib: float          # from torch.cuda.mem_get_info()[1]
    weight_bytes: float       # params * bytes_per_param / tp_size
    act_gib: float = 1.0      # MEASURE this, do not trust the default
    ctx_gib: float = 1.0
    tp_size: int = 1

    @property
    def kv_bytes(self) -> float:
        used = self.weight_bytes / self.tp_size + (self.act_gib + self.ctx_gib) * GiB
        return max(0.0, self.total_gib * GiB - used)


def capacity(shape: KVShape, budget: Budget, kv_dtype="bfloat16",
             avg_seq=2048) -> dict:
    """Aggregate token capacity and concurrent-request capacity."""
    per_token = shape.bytes_per_token(kv_dtype)
    if budget.tp_size > 1:
        # TP shards KV heads across ranks; each rank stores 1/tp of a token,
        # but there are tp ranks, so aggregate capacity uses the full figure
        # against the SUM of the per-rank budgets.
        per_token = per_token / budget.tp_size
    tokens_per_rank = int(budget.kv_bytes // per_token)
    return {
        "kv_budget": human(budget.kv_bytes),
        "bytes_per_token": human(shape.bytes_per_token(kv_dtype)),
        "tokens": tokens_per_rank,
        "concurrent": tokens_per_rank // avg_seq,
        "gib_per_request": (shape.bytes_per_token(kv_dtype) * avg_seq) / GiB,
    }
```

And a sweep, because a single number is never the answer:

```python
# nanoserve/capacity.py (continued)
if __name__ == "__main__":
    from nanoserve.kvmath import from_hub

    llama8b = from_hub("meta-llama/Llama-3.1-8B-Instruct")
    cards = {
        "RTX 4090 24GB": Budget(24.0, 16.06e9, act_gib=1.0),
        "L4 24GB":       Budget(24.0, 16.06e9, act_gib=1.0),
        "A100 80GB":     Budget(80.0, 16.06e9, act_gib=2.0),
        "H100 80GB":     Budget(80.0, 16.06e9, act_gib=2.0),
    }
    for dt in ("bfloat16", "fp8"):
        print(f"\n== kv dtype = {dt}")
        for name, b in cards.items():
            for seq in (2048, 8192, 32768, 131072):
                c = capacity(llama8b, b, dt, avg_seq=seq)
                print(f"{name:>14}  ctx={seq:>6}  "
                      f"kv={c['kv_budget']:>10}  users={c['concurrent']:>4}")
```

Expected output for the bf16 pass, all of it derived arithmetic you can verify by hand:

```console
== kv dtype = bfloat16
 RTX 4090 24GB  ctx=  2048  kv=  7.04 GiB  users=  28
 RTX 4090 24GB  ctx=  8192  kv=  7.04 GiB  users=   7
 RTX 4090 24GB  ctx= 32768  kv=  7.04 GiB  users=   1
 RTX 4090 24GB  ctx=131072  kv=  7.04 GiB  users=   0
    A100 80GB  ctx=  2048  kv= 62.04 GiB  users= 248
    A100 80GB  ctx=  8192  kv= 62.04 GiB  users=  62
    A100 80GB  ctx= 32768  kv= 62.04 GiB  users=  15
    A100 80GB  ctx=131072  kv= 62.04 GiB  users=   3
```

Note the zero. On a 4090, a single request at Llama-3.1's advertised 128k context does not fit *at all* — it needs 16.0 GiB and there are 7.04. That is not a scheduling problem you can queue your way out of; that request is unservable on that card in bf16, full stop. Which brings us to the term that dominates everything.

---

## 6. Context length is the dominant term

Of the five factors in $B_{\text{tok}}$, four are architectural constants you inherit. The fifth multiplier — sequence length — is set by your users, changes per request, and is the only one that spans four orders of magnitude in normal operation.

![A left-to-right progression of context lengths from one thousand to one hundred twenty-eight thousand tokens showing cache size crossing the weight footprint and then the card limit](/imgs/blogs/the-memory-math-of-the-kv-cache-4.webp)

The cache is exactly linear in context: no compression, no sublinearity, no amortization. Doubling the conversation doubles the memory. So there is a well-defined context at which one request's cache equals the entire model:

$$S^{*} \;=\; \frac{P \cdot b_w}{B_{\text{tok}}} \;=\; \frac{P \cdot b_w}{2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot b}$$

For Llama-3.1-8B: $S^{*} = 16.06 \times 10^9 / 131{,}072 = 122{,}528$ tokens. **The crossover sits at 122.5k — comfortably inside the model's own advertised 128k window.** At full context, one conversation's cache is larger than the entire model that produced it. Serving that model at its documented maximum means the cache, not the weights, is your dominant memory consumer for a single user.

There is a closed form worth carrying in your head. For a Llama-style architecture — GQA with group ratio $g = H / H_{kv}$, SwiGLU MLP with intermediate width around ${3.5 h}$ — the per-layer parameter count is close to $13 h^2$, and the per-layer KV cost is ${2 (h/g) b}$ per token. The layer count cancels:

$$S^{*}_{\text{layers}} \;\approx\; \frac{13 \, h^2 \, b_w}{2 \, (h/g) \, b} \;=\; \frac{13}{2} \cdot g \cdot h \cdot \frac{b_w}{b} \;=\; 6.5 \, g \, h \quad \text{(when } b_w = b\text{)}$$

Check it. Llama-3.1-8B has $g = 4$, $h = 4096$: $6.5 \times 4 \times 4096 = 106{,}496$. The exact answer was 122,528, and the 16,032-token difference is precisely the embedding and output projection — $1.05 \times 10^9$ parameters at two bytes, divided by 128 KiB per token, gives 16,030. The approximation and the exact calculation agree to four significant figures once you account for the term the approximation deliberately drops.

Llama-3.1-70B has $g = 8$, $h = 8192$: $6.5 \times 8 \times 8192 = 425{,}984$, against an exact $141.1 \times 10^9 / 327{,}680 = 430{,}603$. Again a match.

What that closed form tells you is genuinely counterintuitive: **bigger models have *relatively* cheaper caches.** The crossover scales with $g \cdot h$, and both group ratio and hidden size grow with model size, while cache cost grows only with $L \cdot H_{kv} \cdot d$. A 70B model does not reach cache-equals-weights until 431k tokens — well past any window it ships with. An 8B model reaches it at 122k, inside its own window. If your product is long-context, the small model is the one with the pathological memory profile, not the large one.

#### Worked example: the long-context OOM you will actually hit

A customer sends a 90,000-token document to your 4090 deployment for summarization.

| Quantity | Value | Source |
| --- | --- | --- |
| Prompt length | 90,000 tokens | scenario |
| KV needed | $90{,}000 \times 128 \text{ KiB} = 10.99$ GiB | derived |
| KV budget on the card | 7.04 GiB | derived, section 5 |
| Shortfall | 3.95 GiB | derived |
| Longest servable prompt, bf16 | 57,671 tokens | derived |
| Longest servable prompt, fp8 KV | 115,343 tokens | derived |

The request fails, and it fails *during prefill*, typically at around 60% of the way through the prompt — which is why the failure so often gets misfiled as "the prefill kernel has a bug". It does not. The engine ran out of blocks at token 57,672. And note the second-order cruelty: this request does not just fail, it evicts. If your allocator hands out blocks eagerly during a chunked prefill, a request that will never complete has already displaced blocks belonging to twenty conversations that were doing fine. That failure mode — and the admission control that prevents it — is what [the eviction and preemption post](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) sets up and what the scheduler track finishes.

The defensive check is four lines, and every engine has some version of it:

```python
# nanoserve/admit.py
from nanoserve.kvmath import KVShape

def will_fit(shape: KVShape, kv_budget_bytes: float, live_tokens: int,
             prompt_len: int, max_new: int, kv_dtype="bfloat16") -> bool:
    """Reject at admission, not at token 57,672 of a prefill."""
    need = shape.bytes_per_token(kv_dtype) * (prompt_len + max_new)
    used = shape.bytes_per_token(kv_dtype) * live_tokens
    return used + need <= kv_budget_bytes


def max_servable_context(shape: KVShape, kv_budget_bytes: float,
                         kv_dtype="bfloat16") -> int:
    """The honest --max-model-len for a single-request deployment."""
    return int(kv_budget_bytes // shape.bytes_per_token(kv_dtype))
```

This is exactly what vLLM is telling you when it refuses to start with a message about the model's maximum sequence length being larger than the available KV cache, and suggests either lowering `--max-model-len` or raising `gpu_memory_utilization`. Those two flags are the two halves of this arithmetic: one shrinks $\bar{S}$, the other grows $V_{\text{kv}}$ by eating into the safety margin on $V_{\text{act}}$.

---

## 7. Why grouped-query attention was the biggest inference win of its era

Look again at the master formula and ask which factor a model architect can change without changing what the model *is*. Not the 2. Not $L$ — that changes capacity. Not $d_{\text{head}}$ much. That leaves $H_{kv}$, and it turns out you can cut it hard.

![A two-column comparison of full multi-head attention against grouped-query attention on a seventy-billion-parameter model showing cache per token conversation cost and concurrent user count](/imgs/blogs/the-memory-math-of-the-kv-cache-5.webp)

In classic multi-head attention each of $H$ query heads has its own key and value head. Multi-query attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) takes this to the opposite extreme: all query heads share a single key/value head. Grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) sits between them — $H$ query heads share $H_{kv}$ key/value heads in groups of $g = H / H_{kv}$ — and the GQA paper's argument is that this recovers essentially MHA-level quality at close to MQA-level speed. Llama-2's 70B adopted GQA, Llama-3 used it at every size, and by 2024 essentially every new open model shipped with it.

The memory consequence is not subtle, because $H_{kv}$ enters the formula linearly.

#### Worked example: 70B-class, MHA vs GQA vs MQA, on 8× A100 80GB

Weights are 141.1 GB (70.55B parameters, bf16), sharded across eight cards. Budget: $8 \times 80 - 131.4 \text{ (weights)} - 8 \times 1 \text{ (ctx)} - 8 \times 2 \text{ (act)} = 484.6$ GiB of aggregate KV. Test at a 4,096-token conversation.

| Variant | $H_{kv}$ | $B_{\text{tok}}$ | One 4k conversation | Aggregate tokens | Concurrent users | Source |
| --- | --- | --- | --- | --- | --- | --- |
| MHA | 64 | 2.50 MiB | 10.00 GiB | 198,492 | 48 | derived |
| GQA, $g=8$ | 8 | 320 KiB | 1.25 GiB | 1,587,937 | 387 | derived |
| MQA | 1 | 40 KiB | 160 MiB | 12,704,500 | 3,101 | derived |

Forty-eight users to three hundred and eighty-seven, from changing one integer in a config file. No new kernel, no quantization, no distributed system, no accuracy loss worth measuring. That is why GQA propagated through the entire open-model ecosystem in roughly a year: it is the rare architectural change that is nearly free at training time and transformative at serving time.

The MQA row shows why the industry did not go all the way. The capacity is spectacular and the quality is not — MQA was reported to degrade quality noticeably, which is the gap GQA was designed to close. Sixty-four to eight was the sweet spot; eight to one was not.

Note also what GQA does *not* fix. It divides $B_{\text{tok}}$ by $g$ and leaves the linear-in-context behavior completely intact. The MHA row would have hit cache-equals-weights at 53,800 tokens; the GQA row hits it at 430,603. Both are still straight lines through the origin. GQA moved the constant, not the shape.

### 7.1 The modern extreme: multi-head latent attention

Multi-head latent attention takes a different swing: instead of caching per-head keys and values at all, cache a single low-rank latent vector per token per layer and reconstruct K and V from it inside the attention kernel. The repo's [MLA deep dive](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) covers the mechanism; here I only want the arithmetic.

DeepSeek-V3's config gives `kv_lora_rank = 512` and `qk_rope_head_dim = 64` — a 512-dimensional compressed latent plus a 64-dimensional shared RoPE key, per layer, per token. In bf16 that is $(512 + 64) \times 2 = 1{,}152$ bytes per layer. The uncompressed alternative, with its 128 attention heads at 192 key dimensions and 128 value dimensions, would be $128 \times (192 + 128) \times 2 = 81{,}920$ bytes per layer. **A 71× reduction, per layer, per token.**

vLLM's [DeepSeek-V3.2-Exp post](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) (2025-09-29) reports the deployed figure as **656 bytes per token, composed of 512 bytes of quantized NoPE latent, 16 bytes of scales, and 128 bytes of RoPE.** That composition is a perfect fingerprint of the config: 512 latent elements at one byte each (fp8), 64 RoPE dimensions at two bytes each (bf16), plus per-block quantization scales. So the 656 is the *per-layer* latent slot, and across V3's 61 layers a token costs roughly 40,016 bytes — about 39 KiB.

Sit with that. A 671B-parameter model costs 39 KiB per token of cache. Llama-3.1-8B, eighty-four times smaller, costs 128 KiB — **more than three times as much.** MLA does not shift the constant a little; it decouples cache cost from model size almost entirely. It is also the reason DeepSeek-scale models are servable at long context at all: at 39 KiB per token, a 128k conversation holds 4.9 GiB of cache rather than the 583 GiB an MHA version would have needed.

The cost is that MLA is not free to implement. It requires a fused kernel that decompresses during attention, its own paged-cache layout, and — per the same vLLM post — its sparse-attention indexer keeps a separate K cache and complicates continuous batching and paged attention enough that prefill and decode are handled separately. You cannot bolt it onto a model trained with GQA. It is an architecture decision, made once, at pretraining time.

---

## 8. The three movable terms, and why they multiply

Step back and classify. Every KV compression technique that has ever shipped attacks one of exactly three terms in the same product, which is why they stack cleanly.

![A taxonomy tree splitting cache reduction into fewer key-value slots fewer bytes per slot and fewer retained tokens with named techniques under each](/imgs/blogs/the-memory-math-of-the-kv-cache-6.webp)

| Term attacked | Technique | Multiplier on $B_{\text{tok}}$ | Cost | Source |
| --- | --- | --- | --- | --- |
| $H_{kv} \cdot d$ | GQA, $g = 4$ | ÷4 | training-time decision | derived; [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245) |
| $H_{kv} \cdot d$ | GQA, $g = 8$ | ÷8 | training-time decision | derived |
| $H_{kv} \cdot d$ | MQA | ÷$H$ | quality regression | cited: GQA paper |
| $2 \cdot H_{kv} \cdot d$ | MLA latent | ÷71 per layer | pretraining + custom kernel | derived from DeepSeek-V3 config |
| $b$ | fp8 E4M3 KV | ÷2 | small accuracy loss, hardware-gated | cited: vLLM FP8-KV post |
| $b$ | 4-bit KV | ÷4 nominal | throughput regression | cited: vLLM TurboQuant post |
| $S$ | sliding window | caps most layers at $W$ | loses distant context | derived; architecture-dependent |
| $S$ | eviction / offload | frees or relocates blocks | recompute or PCIe latency | cited: vLLM offloading post |

They multiply because they are independent factors in a product. GQA at $g = 4$ plus fp8 KV gives $\div 8$ against an MHA bf16 baseline. That composition is why a modern 8B model at fp8 KV on an A100 supports about a thousand times more aggregate context than a 2022-era 70B in MHA bf16 on the same silicon — and almost none of that came from the GPU getting better.

### 8.1 The precision lever, in one paragraph and one table

Halving $b$ halves $B_{\text{tok}}$ and doubles every capacity number in this post. That is the entire mechanism; there is nothing subtle about the arithmetic. What *is* subtle is whether the resulting model is still the model you shipped, and that belongs to the precision track later in this series rather than here.

The numbers that matter for planning, all cited from vLLM's [State of FP8 KV-Cache and Attention Quantization](https://vllm.ai/blog/2026-04-22-fp8-kvcache) post (2026-04-22):

| Claim | Value | Source |
| --- | --- | --- |
| KV storage under fp8 | halved | cited: vLLM FP8-KV post (2026-04-22) |
| Format used for KV | E4M3 exclusively | cited: vLLM FP8-KV post |
| Default scaling | per-tensor, uncalibrated, scale 1.0 | cited: vLLM FP8-KV post |
| Llama-3.1-8B ITL slope on H100 | 54% of bf16, break-even near 7k tokens | cited: vLLM FP8-KV post |
| Llama-3.1-8B under load, H100 | +14.9% throughput, −14.8% median ITL | cited: vLLM FP8-KV post |
| gpt-oss-20b under the same treatment | only about 4.8% gain | cited: vLLM FP8-KV post |
| Llama-3.3-70B accuracy at 128k | roughly 97–98% AUC recovery | cited: vLLM FP8-KV post |
| Capacity ratios vs bf16 | fp8 2×, k8v4 2.4×, 4-bit-nc up to 3.4× | cited: vLLM TurboQuant post (2026-05-11) |
| 4-bit KV throughput cost | 80% → 66% of bf16, TPOT 1.5–2.5× worse at burst | cited: vLLM TurboQuant post |

Three planning conclusions follow. **One:** fp8 KV is close to a free doubling of concurrency and vLLM's own verdict in the TurboQuant post is that `--kv-cache-dtype fp8` remains the best default. **Two:** the benefit is *not* uniform across models — a 4.8% gain on one model and 14.9% on another, under the same conditions, means you must measure your model rather than adopt a rule. **Three:** the same post documents a genuine cliff — on Hopper, imprecise FP32 accumulation at 100k-plus contraction lengths collapsed a needle-retrieval score from 91% to 13% until a two-level accumulation fix landed, and that fix costs TTFT. Below roughly 7k tokens, bf16 was simply faster. Halving your bytes is arithmetic; keeping your accuracy is engineering.

---

## 9. The same model, three capacity worlds

Put the levers together on one picture, using only numbers already derived in this post.

![A three by three layout of graphics card KV budget token capacity and concurrent user count for two cards and one precision setting](/imgs/blogs/the-memory-math-of-the-kv-cache-7.webp)

| Configuration | KV budget | Token capacity | Users at 2,048 tokens | Cost per seat-hour at \$2 per GPU-hour | Source |
| --- | --- | --- | --- | --- | --- |
| 4090, bf16 KV | 7.04 GiB | 57,671 | 28 | — | derived |
| A100 80GB, bf16 KV | 62.04 GiB | 508,231 | 248 | \$0.0081 | derived, given the stated rental rate |
| A100 80GB, fp8 KV | 62.04 GiB | 1,016,462 | 496 | \$0.0040 | derived, given the stated rental rate |

The dollar column is deliberately conditional: substitute your actual invoice rate, because the point is not the absolute figure but that **the KV dtype is a direct multiplier on your cost per concurrent seat.** Flipping one flag halves it. Very little else in the stack offers that.

A caveat that keeps this honest: concurrency capacity is not the same as throughput. Holding 496 conversations in memory does not mean serving them all at an acceptable per-token latency — that depends on batching, kernels, and bandwidth, and the [roofline post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is where that argument lives. What the KV math gives you is the *ceiling*. You will usually run below it. You can never run above it.

### 9.1 The cache is also a bandwidth cost

One thing the capacity framing hides: every decode step reads the *entire* live cache, because attention attends over all of it. So the cache is charged twice — once in capacity, once in bandwidth.

On the 4090 at full occupancy: 28 requests × 2,048 tokens × 128 KiB = 7.04 GiB = 7.56 GB of KV read per decode step, on top of the 16.06 GB of weights. At the card's 1,008 GB/s that is 7.5 ms for the cache and 15.9 ms for the weights, so **the KV read is roughly a third of the step's bandwidth budget at full occupancy** — and unlike the weight read, it is not amortized across the batch, because each sequence reads only its own cache.

That asymmetry has a consequence people find surprising: as you push concurrency up, the weight read amortizes beautifully (one read serves the whole batch) but the KV read scales linearly with total live tokens. There is therefore a point past which admitting more requests stops helping throughput, because you have traded an amortizable cost for a non-amortizable one. Finding that point is what the scheduler track is for.

---

## 10. Stress tests: where the formula stops being exact

A formula is only useful if you know its edges. Here are the five that matter.

### 10.1 Sliding-window and hybrid layers break the linearity

Gemma-3 interleaves sliding-window layers with global ones. A windowed layer stops growing once the conversation passes its window, so total cache becomes affine rather than linear:

$$M(S) = 2 \cdot L_{\text{full}} \cdot H_{kv} \cdot d \cdot b \cdot S \;+\; 2 \cdot L_{\text{win}} \cdot H_{kv} \cdot d \cdot b \cdot \min(S, W)$$

For Gemma-3-12B — 48 layers, 8 KV heads, $d_{\text{head}} = 256$, a window of 1,024, and five windowed layers for every global one — that is 8 global and 40 windowed layers, with 8 KiB per layer per token. At a 32,768-token context:

| Model | Naive formula | Window-aware | Ratio | Source |
| --- | --- | --- | --- | --- |
| Gemma-3-12B at 32k | 12.58 GiB | 2.31 GiB | 5.4× overestimate | derived from config |

Five and a half times. If you size a Gemma-3 deployment with the naive formula you will provision five times the hardware you need — which is arguably the safer error, but it is still an error. This is why `kvmath.py` parses `layer_types` and `sliding_window_pattern`, and why the honest advice is: run the script against the exact revision you deploy.

### 10.2 Hybrid attention models: a fixed state instead of a growing one

The stress test that genuinely breaks the model is a layer with **no per-token KV dimension at all.** vLLM's [disaggregated serving for hybrid SSM models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) post (2026-04-21) states the engine problem precisely: full-attention layers use a uniform per-token KV layout, while SSM layers hold a fixed-size convolution state plus a temporal state with no per-token dimension, and a single descriptor format cannot address both.

Arithmetically, a hybrid model with $L_f$ full-attention layers and the rest linear-attention costs

$$M_{\text{hyb}}(S) = 2 \cdot L_f \cdot H_{kv} \cdot d \cdot b \cdot S \;+\; C$$

where $C$ is a per-request constant. This is an affine function with a positive intercept and a shallower slope — which means hybrids *lose* at short context (you pay $C$ for a two-token request) and win by an ever-widening margin at long context. The break-even sits at $S_{\text{be}} = C / \big(2 (L - L_f) H_{kv} d\, b\big)$, and for any plausible $C$ that number is small.

MiniMax's numbers give a sense of the magnitude: vLLM's [MiniMax-M1 post](https://vllm.ai/blog/2025-06-30-minimax-m1) (2025-06-30) reports that lightning attention reduces memory by 83% and inference latency by 67% for 100k-token sequences. Cite that as an order-of-magnitude signal, not a formula.

The engine-side consequence is the one to remember. vLLM's [Qwen3-Next post](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11) states it verbatim: *"vLLM automatically tunes the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory."* Read that as an admission that the allocator's one-block-fits-all assumption — the assumption the entire next post is built on — does not survive hybrid architectures. It has to be repaired by making the block sizes *unequal in tokens* so they come out *equal in bytes*. The hybrid track much later in this series takes that apart properly; for now, note that the memory math above assumes every layer contributes identically, and hybrids are exactly where that assumption dies. Do not cite an interleave ratio for these models from a blog post — the vLLM page explicitly does not state one. Read the model card.

### 10.3 Mixture-of-experts changes the weights, not the cache

An MoE model like Qwen3-30B-A3B has 30B parameters but activates about 3B per token. That changes $V_{\text{weights}}$ enormously — you must hold *all* experts resident even though you use a fraction per token — and changes $B_{\text{tok}}$ **not at all**, because attention is dense in every MoE architecture in common use. So MoE models have a distinctive memory profile: enormous fixed weight cost, ordinary per-token cache cost, and therefore a KV budget that is squeezed hard from one side. Size them with the same formula and a much bigger $V_{\text{weights}}$.

### 10.4 Tensor parallelism shards the cache too

Under TP, KV heads are split across ranks along with the attention weights. With $H_{kv} = 8$ and TP=4, each rank holds two KV heads and therefore a quarter of each token's cache. Two consequences: your aggregate capacity is the sum of the per-rank budgets, and **TP degree cannot exceed $H_{kv}$** without replicating KV heads across ranks — which is why an 8-KV-head model at TP=16 wastes memory duplicating what it cannot split. That constraint is invisible until it bites, and it is pure arithmetic from this same formula.

### 10.5 Fragmentation: the formula is an upper bound, not a promise

Everything above computes the cache you would use with perfect packing. A contiguous allocator does not pack perfectly. If you reserve `max_model_len` per request up front, a 200-token conversation with a 32k limit wastes 99% of its reservation. The [PagedAttention paper](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023) is built on exactly this observation — that prior systems lost the majority of their KV memory to fragmentation and over-reservation, and that block-based allocation brings that waste down to a few percent.

So read the capacity numbers in this post as the ceiling that *paging lets you approach*. Without paging you will land far below them, and the next post — [implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — is how you close the gap.

---

## 11. Measuring this honestly

Three habits separate a real capacity number from a plausible one.

**Read the ceiling, do not assume it.** `torch.cuda.mem_get_info()` gives free and total bytes from the driver, which already accounts for the context and anything else on the card. Everything else is guessing.

**Distinguish allocated from reserved.** PyTorch's caching allocator holds onto freed blocks. `torch.cuda.memory_allocated()` is what your tensors occupy; `torch.cuda.memory_reserved()` is what the allocator has claimed from the driver; `nvidia-smi` shows something closer to the latter plus the context. A cache that OOMs while `nvidia-smi` reports 57% utilization is almost always allocator fragmentation between those two numbers, not a leak.

**Profile the activation peak instead of budgeting it.** This is what production engines do: run one dummy forward at your maximum batched-token count, record the peak, and treat the remainder as the KV pool. Here is the pattern, condensed:

```python
# nanoserve/probe.py
import torch


def probe_kv_budget(model, max_batched_tokens: int, utilization: float = 0.90):
    """Measure the real KV budget the way a serving engine does.

    Returns bytes available for the KV pool after a worst-case forward.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()

    # Worst-case shape: one flat sequence of max_batched_tokens.
    dummy = torch.zeros(1, max_batched_tokens, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        model(dummy)                    # no cache: we want the transient peak
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_reserved()   # reserved, not allocated:
    # the allocator's high-water mark is what actually competes with the pool.
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = total * utilization - peak
    return {
        "total": total,
        "weights": weights,
        "peak_reserved": peak,
        "kv_budget": max(0, int(budget)),
    }
```

**Reproduce it yourself.** On a 4090 with Llama-3.1-8B in bf16 and `max_batched_tokens=2048`, you should land in the neighborhood of 6–8 GiB of KV budget at `utilization=0.90`, which at 128 KiB per token is roughly 49,000–65,000 tokens. If you get dramatically less, check for a duplicated fp32 copy of the weights left over from loading; if you get dramatically more, check that you actually ran the forward pass. Report what you see — the range above is the arithmetic of this post applied to a plausible measurement, not a measurement I made.

And the assertion that keeps the formula honest against the allocator:

```python
# tests/test_kvmath.py
import torch
from nanoserve.kvmath import from_hf_config


def test_cache_allocation_matches_formula(tmp_path):
    shape = from_hf_config("configs/llama-3.1-8b.json")
    seq, batch = 1024, 2

    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    cache = [
        (
            torch.empty(batch, shape.num_kv_heads, seq, shape.head_dim,
                        dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, shape.num_kv_heads, seq, shape.head_dim,
                        dtype=torch.bfloat16, device="cuda"),
        )
        for _ in range(shape.num_layers)
    ]
    actual = torch.cuda.memory_allocated() - before
    expected = shape.bytes_per_token() * seq * batch

    # Allocator rounds up to 512-byte segments; allow a small margin.
    assert abs(actual - expected) / expected < 0.01, (actual, expected)
    del cache
```

If that test fails, one of two things is true: your config parse is wrong, or your cache layout is not what you think it is. Both are worth knowing before you write a scheduler on top.

---

## 12. Case studies and public numbers

Four public data points, each cited, that either validate the arithmetic or extend it.

**vLLM's physical block sizes.** Already worked through in section 2 — the [KV offloading post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (2026-01-08) lists 32 KB → 2 MB for Llama-3.1-8B, 16 KB → 0.5 MB for 3.2-1B, and 8 KB → 1.25 MB for 3.1-70B. All six reproduce from $B_{\text{tok}} \times 16$, with the 70B row implying TP=4. This is the strongest available confirmation that the formula is exact rather than approximate.

**The super-linear TP result.** vLLM's [distributed inference post](https://vllm.ai/blog/2025-02-17-distributed-inference) (2025-02-17) reports 13.9× more KV cache blocks and 3.9× more token throughput moving from TP=1 to TP=2. That is not a memory doubling; it is the residual structure amplifying, exactly as section 5 derives. Note the second number too: 13.9× the blocks gave 3.9× the throughput, because capacity and throughput are different resources.

**DeepSeek-V3.2's 656 bytes.** vLLM's [DeepSeek-V3.2-Exp post](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) (2025-09-29) gives the MLA cache as 656 bytes per token, composed of 512 bytes of quantized NoPE latent, 16 bytes of scales, and 128 bytes of RoPE. The composition reconciles exactly with the published `kv_lora_rank=512` and `qk_rope_head_dim=64`, which is how we can be confident it describes a per-layer slot. The same post notes the sparse-attention indexer keeps a separate K cache — a real cost the headline number excludes.

**fp8 KV in production.** vLLM's [FP8 KV-cache post](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (2026-04-22) is the most useful single reference on the precision lever: storage halved, E4M3 only, Hopper via FlashAttention-3 and Blackwell via FlashInfer, Ada and AMD not covered, +14.9% throughput under load on Llama-3.1-8B on an H100 but only ~4.8% on gpt-oss-20b, and a documented accuracy cliff at 100k-plus contraction lengths before the two-level accumulation fix. Read the limitations section before you flip the flag.

---

## 13. When to reach for this (and when not to)

**Always do this arithmetic before buying hardware or setting a context limit.** It costs five minutes, requires no GPU, and it is the difference between a capacity plan and a hope. The most expensive mistakes in LLM serving are provisioning decisions made from parameter count alone.

**Do the arithmetic before choosing between two similar models.** A 12% difference in layer count is a 12% difference in serving capacity forever. If two models are within noise on quality, the one with fewer layers, fewer KV heads, or a smaller head dimension is meaningfully cheaper to run and nobody's benchmark table will tell you that.

**Do not write your own allocator on the strength of this post.** The formula gives you the ceiling; reaching it needs paged allocation, prefix sharing, eviction, and a scheduler. Those are the next several posts, and if you need them in production today the honest answer is to run vLLM or SGLang, which have solved all of it and more. Write your own to *understand* it. Deploy someone else's to *ship* it.

**Do not use the naive formula on sliding-window or hybrid models.** You will overestimate by 5× on Gemma-3 and mis-model hybrids entirely. Parse the layer types.

**Do not treat capacity as throughput.** This post gives you a memory ceiling. Whether you can actually serve 248 concurrent users at an acceptable TPOT is a bandwidth and scheduling question with a different, usually lower, answer.

**Do not skip fp8 KV on Hopper or Blackwell without measuring.** A free 2× on your dominant memory consumer is rare enough that the burden of proof is on *not* using it. But measure your model's accuracy at your context lengths, because the gains and the losses are both model-specific.

---

## 14. Key takeaways

1. **$B_{\text{tok}} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot b$.** Memorize it. For Llama-3.1-8B in bf16 it is exactly 128 KiB per token, 1 GiB per 8k of context.
2. **The KV cache is a residual**, not an allocation. It is whatever survives after weights, activations, and the CUDA context — which is why capacity is super-linear in card size and violently sensitive to weight footprint.
3. **A 24 GiB card serving an 8B model in bf16 holds about 57,000 tokens** — 28 conversations at 2k each, or seven at 8k, or one at 32k. An 80 GiB card holds nine times that, not three.
4. **Cache cost does not track parameter count.** Qwen3-8B costs 12.5% more per token than Llama-3.1-8B; a 671B MLA model costs a third of an 8B GQA model. Read the config, not the model name.
5. **Context is the dominant term.** Cache is exactly linear in sequence length, and for Llama-3.1-8B it exceeds the entire model at 122,528 tokens — inside the model's own advertised window. The closed form is $S^{*} \approx 6.5\,g\,h$.
6. **GQA was worth an 8× capacity win on 70B-class models** — 48 concurrent users to 387 — from changing one integer. It moved the constant and left the linearity untouched.
7. **Only three terms are movable**: KV slots ($H_{kv} \cdot d$, via GQA/MQA/MLA), bytes per slot ($b$, via fp8), and tokens retained ($S$, via windows and eviction). They are independent factors and therefore compose multiplicatively.
8. **Validate your formula against published numbers.** All six of vLLM's stated physical block sizes reproduce from $B_{\text{tok}} \times 16$, and the one that did not match at first told us the tensor-parallel degree of the machine it came from.
9. **The formula is a ceiling, not a promise.** Fragmentation, over-reservation, and hybrid layer types all sit between the arithmetic and the allocator. Paging is what closes that gap.
10. **Measure the activation peak; never budget it.** Profile one worst-case forward pass, take the remainder as the pool. That is what the production engines do and it is four lines of code.

---

## Further reading

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023. The paper that moved $H_{kv}$ for the entire industry.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019. Multi-query attention, the extreme case GQA moderates.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023. Why the ceiling in this post is not reachable without paging.
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — the per-block KV formula and the block pool, from the reference implementation.
- [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm.ai/blog/2026-04-22-fp8-kvcache) — the precision lever, with its hardware gates and its accuracy cliffs.
- [DeepSeek-V3.2-Exp: fine-grained sparse attention in vLLM](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) — the 656-bytes-per-token MLA figure and its composition.
- [Multi-head latent attention, explained](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — the mechanism behind the 71× per-layer reduction.
- [The KV cache, from first principles](/blog/machine-learning/large-language-model/kv-cache) — the conceptual companion to this post's arithmetic.
- [Why recompute is fatal: writing a KV cache](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache) — the previous post in this track, which built the thing we just priced.
- [Paged KV cache: implementing blocks and a block table](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — the next post, which makes these capacity numbers actually reachable.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where every one of these levers gets ranked by what it is worth.
