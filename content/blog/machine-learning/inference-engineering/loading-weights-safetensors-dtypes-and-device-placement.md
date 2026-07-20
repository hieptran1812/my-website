---
title: "Loading weights: safetensors, dtypes, and device placement"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Read a safetensors header byte by byte, derive why a 16 GB model needs a 24 GB card and still barely fits, and write a loader that streams an 8B checkpoint onto the GPU without ever making a second copy on the host."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "safetensors",
    "checkpoints",
    "dtypes",
    "pytorch",
    "gpu",
    "memory",
    "quantization",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 48
---

The first thing your inference engine does is the thing nobody writes about. Before a single token is sampled, before the KV cache exists, before the scheduler has a queue to schedule, some code has to take sixteen gigabytes of floating-point numbers off a disk and put them in the right places in GPU memory, in the right numeric format, with the right shapes, and without accidentally making a second copy of all of it somewhere else.

It sounds like plumbing. It is not. Load-time decisions set a ceiling on everything the rest of this series tries to do. The dtype you pick at load determines your decode-step floor, because a decode step is memory-bound and the number of weight bytes you drag across HBM per token *is* the latency. The bytes you leave unclaimed after loading are the entire budget for the KV cache, which is the entire budget for concurrency. And a loader that materializes the checkpoint twice on the host will happily take a 32-core box with 32 GB of RAM into swap and turn a three-second cold start into a ninety-second one, which your autoscaler will interpret as a dead pod.

Here is the number that starts the argument. Llama-3.1-8B in bf16 is 16.06 GB of weights. An RTX 4090 has 24 GB of VRAM. That looks like eight gigabytes of comfortable headroom, and it is not — by the time the CUDA context, the activation buffers, the allocator's fragmentation slack, and the KV cache have taken their cut, you are fighting for the last gigabyte. Figure 1 is that budget drawn to scale, and it is worth staring at before you write any code, because every later post in this series is a fight over the green band at the right-hand end.

![Stacked budget of a 24 GiB GPU showing CUDA context, bf16 weights, activations, fragmentation headroom and the remaining KV cache space](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-1.webp)

By the end of this post you will be able to read a safetensors file with nothing but `struct.unpack`, explain to a colleague why their 16 GB model OOM'd on a 24 GB card, compute a VRAM budget from a `config.json` before you rent the machine, and write a `load_weights()` that streams tensors from disk into pre-allocated device buffers with a peak host footprint of about one gigabyte. This is the post that writes `nanoserve/weights.py` — the second file in the toy engine this series builds. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post lays out the weights → kernels → engine → decoding → API spine that everything here hangs from; the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) at the end of the series is where all of it gets assembled and benchmarked.

One promise up front, and it holds for every post in this series: I have no GPU and I have run none of this. Every number below is either **derived** from a formula I show you, **cited** to a public source I link, or framed as **reproduce-it-yourself** with a script and an expected range on named hardware. Where I give a range, it is what you should expect to see, not what I claim to have seen.

## 1. What a checkpoint actually is

A "model" on disk is two things: a `config.json` describing the architecture, and a pile of named tensors. That is it. There is no code in a well-formed checkpoint, no graph, no execution plan — just `model.layers.7.self_attn.q_proj.weight` mapped to a 4096×4096 block of bf16 numbers, repeated a few hundred times.

The interesting differences between formats are in how those bytes are framed. Four formats matter in practice, and they differ on exactly three axes that a serving engineer cares about: can opening the file run code, can the bytes be mapped into your address space without being read, and can the format carry quantized weights.

![Comparison of four checkpoint formats across code execution risk, zero-copy mapping, quantization support and their best use case](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-2.webp)

### safetensors: a length, a JSON header, and raw bytes

The [safetensors format](https://huggingface.co/docs/safetensors/index) is almost aggressively boring, which is the point. The layout is:

1. Eight bytes, little-endian unsigned 64-bit integer: the length `N` of the header.
2. `N` bytes of UTF-8 JSON: a dictionary mapping tensor name to `{"dtype": ..., "shape": [...], "data_offsets": [start, end]}`, plus an optional `__metadata__` key holding string-to-string metadata.
3. Everything after byte `8 + N`: the raw tensor bytes, back to back, in C-contiguous order, with `data_offsets` relative to the start of this region.

That is the entire specification. You can parse it with the standard library, and doing so once is the fastest way to stop treating checkpoints as magic:

```python
# nanoserve/formats/safetensors_raw.py
# Read a safetensors header with no dependencies beyond the stdlib.
import json
import struct
from pathlib import Path


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))  # u64 little-endian
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    return header, 8 + header_len


if __name__ == "__main__":
    import sys

    header, data_start = read_header(Path(sys.argv[1]))
    meta = header.pop("__metadata__", {})
    print(f"metadata      : {meta}")
    print(f"tensors       : {len(header)}")
    print(f"data begins at: byte {data_start}")
    total = 0
    for name, spec in list(header.items())[:4]:
        start, end = spec["data_offsets"]
        nbytes = end - start
        total += nbytes
        print(f"  {name:48s} {spec['dtype']:>5s} {str(spec['shape']):>16s} {nbytes / 1e6:8.2f} MB")
    print("  ...")
```

Run it against the first shard of a Llama-3.1-8B checkout and you will see something with this shape:

```console
metadata      : {'format': 'pt'}
tensors       : 98
data begins at: byte 25896
  model.embed_tokens.weight                        BF16   [128256, 4096]  1050.67 MB
  model.layers.0.input_layernorm.weight            BF16           [4096]     0.01 MB
  model.layers.0.mlp.down_proj.weight              BF16   [4096, 14336]   117.44 MB
  model.layers.0.mlp.gate_proj.weight              BF16   [14336, 4096]   117.44 MB
  ...
```

Two things in that output are load-bearing for the rest of this post. First, the header is tiny — roughly 26 KB for a hundred tensors — so reading it costs one disk seek and a couple of milliseconds regardless of how big the file is. You can ask "what is in this checkpoint and how much VRAM will it need" without reading a single weight. Second, `data_offsets` are byte ranges into a flat region, which is exactly what an `mmap` wants. The library never has to `read()` the tensor data at all; it maps the file and hands you tensors that are views into the mapping. The kernel pages bytes in on demand, from the page cache if they are warm.

In practice you use the real library, which does the mapping and the dtype plumbing for you:

```python
# nanoserve/formats/st.py
from safetensors import safe_open

with safe_open("model-00001-of-00004.safetensors", framework="pt", device="cpu") as f:
    print(f.metadata())                    # the __metadata__ dict
    keys = list(f.keys())                  # every tensor name in this shard
    t = f.get_tensor("model.layers.0.mlp.down_proj.weight")
    print(t.shape, t.dtype, t.device)      # torch.Size([4096, 14336]) torch.bfloat16 cpu
    print(t.untyped_storage().nbytes() / 1e6)  # 117.44
```

`safe_open` also accepts `device="cuda:0"`, in which case the library reads the byte range and copies it to the device for you, skipping the intermediate torch CPU tensor. That single argument is most of the difference between a good loader and a bad one, and we will come back to it in section 4.

Note the security property, which is not a footnote: parsing a safetensors file is parsing a length, a JSON blob, and a set of byte ranges. There is no code path that constructs arbitrary objects. A malicious file can at worst claim wrong shapes or overlapping offsets, which a validating parser rejects. This is the entire reason the format exists.

### PyTorch `.bin`: pickle, and why the ecosystem left

A `.bin` or `.pth` checkpoint saved by `torch.save` is a ZIP archive containing a Python pickle. Pickle is not a data format — it is a *bytecode for reconstructing objects*, and its opcodes include `GLOBAL` (import a name from a module) and `REDUCE` (call it). Unpickling a hostile file executes whatever it names. Downloading a `.bin` from a model hub and calling `torch.load` on it was, for years, functionally equivalent to running a stranger's script.

PyTorch closed the door in stages. `torch.load(..., weights_only=True)` restricts unpickling to a small allowlist of tensor-reconstruction functions, and [PyTorch 2.6 flipped that flag to default `True`](https://github.com/pytorch/pytorch/releases/tag/v2.6.0). That is a genuine fix, but it arrives on top of a format that was never designed for untrusted input, and it broke a long tail of checkpoints that legitimately pickled non-tensor objects. Meanwhile safetensors gives you the safety property structurally rather than by allowlist.

There is also a performance argument, and it is subtler than "pickle is slow". A ZIP-wrapped pickle *can* be memory-mapped — [`torch.load(..., mmap=True)`](https://pytorch.org/docs/stable/generated/torch.load.html) does exactly that for archives saved with the newer zipfile serialization — but the tensors inside are not guaranteed to be aligned or contiguous in a way that makes the mapping free, and the default path reads the whole archive into host memory first. HuggingFace publishes their own [speed comparison](https://huggingface.co/docs/safetensors/speed) showing large load-time wins for safetensors; treat those as their measurements on their hardware, and measure your own with the script in section 7.

Practical rule for `nanoserve`: support `.bin` only for checkpoints that have no safetensors sibling, always with `weights_only=True`, and log a warning. Never for anything you did not produce.

### GGUF: metadata and quantized blocks in one file

GGUF is llama.cpp's format, and it solves a different problem: shipping a *quantized* model as a single self-describing file that a C++ program with no Python can open. The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) defines a magic number, a versioned header, an arbitrary key-value metadata section (architecture, rope parameters, tokenizer vocabulary and merges, chat template — all of it, so there is no `config.json` and no `tokenizer.json` beside it), then a tensor-info table, then aligned tensor data.

The part that matters for an engine is the tensor data. GGUF's quantized types are *block* types: `Q4_K_M`, `Q8_0`, `Q6_K` and friends store a run of 32 or 256 weights plus one or two shared scale factors in a packed struct. A `Q4_0` block, for example, is 32 four-bit quants plus one fp16 scale — 18 bytes for 32 weights, or 4.5 bits per weight rather than a clean 4. This is why a "4-bit" GGUF file is never exactly one eighth the size of the fp32 original, and why you cannot compute its footprint by multiplying parameters by 0.5 bytes. You have to know the block layout. We will implement a GGUF loader properly when the series gets to [weight-only quantization](/blog/machine-learning/large-language-model/quantization-in-llm); for now, the thing to internalize is that GGUF bakes the quantization scheme into the file format, whereas safetensors stores quantized tensors as opaque integer arrays plus separate scale tensors, and leaves interpretation to a config field.

### ONNX: a graph, not just weights

ONNX is the outlier because it stores the *computation graph* alongside the weights, as a protobuf. That is useful for runtimes that want to compile an unfamiliar model without a Python implementation, and it is why ONNX Runtime and TensorRT ingest it. It is awkward for LLMs for two reasons: protobuf has a 2 GB message limit, so any real model must spill weights into [external data files](https://onnx.ai/onnx/repo-docs/ExternalData.html) beside the graph; and a frozen graph fights with the dynamic shapes an LLM server needs — varying batch size, varying sequence length, a KV cache that grows. You end up with either a graph specialized per shape bucket or a lot of dynamic-axis machinery. When the series reaches compilers we will pick this up again alongside [TensorRT](/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler); for `nanoserve` we never touch ONNX, because we own the forward pass.

## 2. Sharded checkpoints and the weight map

No hub serves a 16 GB single file. `transformers`' `save_pretrained` splits at a default `max_shard_size` of 5 GB, which for an 8B bf16 model gives four shards, and writes an index beside them:

```json
{
  "metadata": { "total_size": 16060522496 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00004.safetensors",
    "model.layers.17.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    "lm_head.weight": "model-00004-of-00004.safetensors"
  }
}
```

The `weight_map` is a routing table from parameter name to filename. `total_size` is the sum of tensor byte lengths — for Llama-3.1-8B in bf16 that is 16,060,522,496 bytes, a number we are about to derive from first principles rather than trust.

![Routing diagram showing an index file fanning out to four checkpoint shards which merge back into one module and then into GPU buffers](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-3.webp)

A shard-aware reader is about thirty lines, and it is the foundation of the streaming loader:

```python
# nanoserve/weights.py
import json
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


class ShardedCheckpoint:
    """Lazily-opened, possibly sharded safetensors checkpoint."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = index["weight_map"]
            self.total_size: int = index["metadata"]["total_size"]
        else:
            # Single-file checkpoint: build a trivial map.
            single = "model.safetensors"
            with safe_open(self.root / single, framework="pt") as f:
                self.weight_map = {k: single for k in f.keys()}
            self.total_size = (self.root / single).stat().st_size
        self._open: dict[str, object] = {}  # filename -> safe_open handle

    def _handle(self, filename: str):
        if filename not in self._open:
            self._open[filename] = safe_open(
                self.root / filename, framework="pt", device="cpu"
            )
        return self._open[filename]

    def keys(self) -> list[str]:
        return list(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)

    def get_to(self, name: str, device: str) -> torch.Tensor:
        """Read the byte range straight onto `device` with no torch CPU tensor."""
        filename = self.weight_map[name]
        key = (filename, device)
        if key not in self._open:
            self._open[key] = safe_open(
                self.root / filename, framework="pt", device=device
            )
        return self._open[key].get_tensor(name)

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        # Iterate in file order, not dict order: sequential reads beat random ones.
        by_file: dict[str, list[str]] = {}
        for name, filename in self.weight_map.items():
            by_file.setdefault(filename, []).append(name)
        for filename in sorted(by_file):
            for name in by_file[filename]:
                yield name, self.get(name)

    def close(self) -> None:
        self._open.clear()
```

Two design choices there are worth defending. `items()` groups by file and yields in file order, because on a spinning disk or a network filesystem, walking tensor names in `weight_map` insertion order can bounce between shards; grouping keeps reads sequential. And handles are cached per `(file, device)` pair, so opening the same shard for CPU and for CUDA does not thrash.

The names themselves are a contract you do not control. `model.layers.0.self_attn.q_proj.weight` is the `transformers` convention; Meta's original `consolidated.00.pth` releases use `layers.0.attention.wq.weight`; a GGUF file uses `blk.0.attn_q.weight`. A production loader carries a small rename table per checkpoint family. Getting this wrong is not subtle — you get a `KeyError` and go fix it. The failures that hurt are the ones where the name matches and the *tensor* is wrong, which is section 8.

## 3. The memory math: why a 16 GB model does not fit in 24 GB

Now the derivation everything else depends on. We are going to compute the size of Llama-3.1-8B from its architecture, not look it up, because the same arithmetic works for any model you are handed and takes ninety seconds.

Read the relevant fields out of `config.json`: hidden size $d = 4096$, intermediate size $d_{ff} = 14336$, layers $L = 32$, attention heads $H = 32$, key-value heads $H_{kv} = 8$, head dim $d_h = 128$, vocabulary $V = 128256$, and `tie_word_embeddings` false.

Each decoder layer holds four attention projections and three MLP projections. With grouped-query attention the K and V projections are narrower than Q and O by the ratio $H_{kv}/H$:

$$
P_{\text{layer}} = \underbrace{2 d^2 + 2\, d\, H_{kv} d_h}_{\text{attention: Q, O, K, V}} + \underbrace{3\, d\, d_{ff}}_{\text{MLP: gate, up, down}} + \underbrace{2 d}_{\text{two RMSNorms}}
$$

Substituting: the attention block is $2 \cdot 4096^2 + 2 \cdot 4096 \cdot 1024 = 33{,}554{,}432 + 8{,}388{,}608 = 41{,}943{,}040$ parameters. The MLP is $3 \cdot 4096 \cdot 14336 = 176{,}160{,}768$. The two norms add 8,192. So $P_{\text{layer}} = 218{,}112{,}000$ parameters, and thirty-two of them is 6,979,584,000.

The embedding and the output head are each $V d = 128256 \cdot 4096 = 525{,}336{,}576$ parameters, and Llama-3.1 does not tie them, so both are present. Add the final RMSNorm's 4,096:

$$
P = 2 V d + L\, P_{\text{layer}} + d = 1{,}050{,}673{,}152 + 6{,}979{,}584{,}000 + 4{,}096 = 8{,}030{,}261{,}248
$$

Eight-point-oh-three billion parameters — which is why the model is called 8B, and which matches the `total_size` in the index exactly when multiplied by two bytes: 16,060,522,496 bytes. Here is that same arithmetic as a script, so you can run it against any config you are handed:

```python
# nanoserve/budget.py
import json
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass
class ModelShape:
    d: int          # hidden_size
    d_ff: int       # intermediate_size
    layers: int     # num_hidden_layers
    heads: int      # num_attention_heads
    kv_heads: int   # num_key_value_heads
    head_dim: int
    vocab: int
    tied: bool

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelShape":
        c = json.loads(Path(path).read_text())
        heads = c["num_attention_heads"]
        return cls(
            d=c["hidden_size"],
            d_ff=c["intermediate_size"],
            layers=c["num_hidden_layers"],
            heads=heads,
            kv_heads=c.get("num_key_value_heads", heads),
            head_dim=c.get("head_dim", c["hidden_size"] // heads),
            vocab=c["vocab_size"],
            tied=c.get("tie_word_embeddings", False),
        )

    @property
    def params_per_layer(self) -> int:
        attn = 2 * self.d * self.d + 2 * self.d * self.kv_heads * self.head_dim
        mlp = 3 * self.d * self.d_ff
        return attn + mlp + 2 * self.d

    @property
    def params(self) -> int:
        embed = self.vocab * self.d
        head = 0 if self.tied else embed
        return embed + head + self.layers * self.params_per_layer + self.d

    def weight_bytes(self, bytes_per_param: float = 2.0) -> float:
        return self.params * bytes_per_param

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        # K and V, per layer, per kv-head, per head dim.
        return 2 * self.layers * self.kv_heads * self.head_dim * bytes_per_elem


if __name__ == "__main__":
    m = ModelShape.from_config("Llama-3.1-8B/config.json")
    print(f"parameters        : {m.params:,}")
    print(f"bf16 weights      : {m.weight_bytes(2) / 1e9:.2f} GB "
          f"({m.weight_bytes(2) / GIB:.2f} GiB)")
    print(f"KV bytes per token: {m.kv_bytes_per_token() / 1024:.0f} KiB")
```

Expected output for Llama-3.1-8B:

```console
parameters        : 8,030,261,248
bf16 weights      : 16.06 GB (14.96 GiB)
KV bytes per token: 128 KiB
```

### GB is not GiB, and this one bites

16.06 GB and 14.96 GiB are the same quantity. A gigabyte is $10^9$ bytes; a gibibyte is $2^{30} = 1{,}073{,}741{,}824$ bytes, about 7.4% larger. Model cards, `total_size` fields and `ls -l` speak decimal. `nvidia-smi`, `torch.cuda.memory_allocated` and every CUDA error message speak binary MiB.

Worse, GPU marketing "GB" is binary. A card sold as 24 GB reports roughly 24,560 MiB of total memory, which is 23.98 GiB — genuinely 24 binary gigabytes, not the 22.35 GiB you would get if the vendor had meant decimal. Disk vendors do the opposite. So the mental arithmetic that trips people is: "my model is 16 GB and my card is 24 GB, that's 8 GB free." The honest version is 23.98 GiB of card minus 14.96 GiB of weights equals 9.02 GiB free, and that 9 GiB is where the trouble starts.

### The other four tenants

**The CUDA context.** The first time you touch the device, the driver allocates a context: kernel code for every loaded module, constant memory, driver-side bookkeeping. It is not visible to PyTorch's allocator and it does not show up in `torch.cuda.memory_allocated()`, only in `nvidia-smi`. Budget a few hundred MiB and measure yours — the script below prints it. Loading cuBLAS, cuDNN and FlashAttention kernels adds to it, and a `torch.compile` run adds more later.

**Activations.** During decode with batch $B$ these are tiny: a handful of $B \times d$ vectors per layer, freed as you go. During *prefill* with a chunk of $T$ tokens they are not. The largest single activation in a Llama forward is usually the logits: computing them for every position of a 2048-token chunk gives a $2048 \times 128256$ tensor, which is 525 MB in bf16 and 1.05 GB if you upcast to fp32 for a numerically stable softmax. That tensor is why an engine slices to the last position before the LM head during prefill — a change we make in the forward-pass post. Budget 0.5–2 GiB depending on your chunk size and batch.

**Fragmentation slack.** PyTorch's caching allocator carves cached blocks and does not always find a contiguous one for the next request. `torch.cuda.memory_reserved()` exceeding `memory_allocated()` is that gap. Five percent of total is a defensible reserve; long-running servers with varying shapes do worse, which is one of several reasons a paged KV cache exists at all.

**The KV cache.** Everything left over. For Llama-3.1-8B, from the same config fields:

$$
\text{KV bytes/token} = 2\, L\, H_{kv}\, d_h\, b = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 = 128\ \text{KiB}
$$

The factor of 2 in front is K and V; $b = 2$ is bytes per element in bf16. One gibibyte buys 8,192 tokens. This formula is the spine of the whole cache track, and [the KV cache post](/blog/machine-learning/large-language-model/kv-cache) walks the mechanism if it is new to you.

<figure class="blog-anim">
<svg viewBox="0 0 660 190" role="img" aria-label="A GPU memory bar fills with weights, context, activations and six session caches until a seventh request has no space" style="width:100%;height:auto;max-width:820px">
<style>
.v1-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.v1-fixed{fill:var(--text-secondary,#6b7280);opacity:.45}
.v1-kv{fill:var(--accent,#6366f1)}
.v1-frag{fill:var(--border,#d1d5db)}
.v1-deny{fill:#dc2626}
.v1-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.v1-sub{font:500 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.v1-warn{font:700 12px ui-sans-serif,system-ui;fill:#dc2626;text-anchor:middle}
@keyframes v1-pop{0%{opacity:0}5%{opacity:1}86%{opacity:1}92%,100%{opacity:0}}
@keyframes v1-deny{0%,34%{opacity:0}40%{opacity:.95}58%{opacity:.35}76%{opacity:.95}88%,100%{opacity:0}}
.v1-b{animation:v1-pop 12s ease-out infinite backwards}
.v1-b2{animation-delay:.8s}
.v1-b3{animation-delay:1.6s}
.v1-b4{animation-delay:2.4s}
.v1-b5{animation-delay:3.2s}
.v1-b6{animation-delay:4s}
.v1-d{animation:v1-deny 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.v1-b,.v1-d{animation:none;opacity:1}}
</style>
<rect class="v1-track" x="20" y="52" width="600" height="54" rx="6"/>
<rect class="v1-fixed" x="20" y="52" width="374" height="54"/>
<rect class="v1-fixed" x="394" y="52" width="15" height="54"/>
<rect class="v1-fixed" x="409" y="52" width="25" height="54"/>
<rect class="v1-frag" x="590" y="52" width="30" height="54"/>
<rect class="v1-kv v1-b" x="435" y="55" width="24" height="48" rx="3"/>
<rect class="v1-kv v1-b v1-b2" x="461" y="55" width="24" height="48" rx="3"/>
<rect class="v1-kv v1-b v1-b3" x="487" y="55" width="24" height="48" rx="3"/>
<rect class="v1-kv v1-b v1-b4" x="513" y="55" width="24" height="48" rx="3"/>
<rect class="v1-kv v1-b v1-b5" x="539" y="55" width="24" height="48" rx="3"/>
<rect class="v1-kv v1-b v1-b6" x="565" y="55" width="24" height="48" rx="3"/>
<rect class="v1-deny v1-d" x="591" y="55" width="24" height="48" rx="3"/>
<text class="v1-lbl" x="207" y="84">weights bf16 14.96 GiB</text>
<text class="v1-sub" x="421" y="38">ctx + act</text>
<text class="v1-sub" x="512" y="38">KV cache 6.23 GiB</text>
<text class="v1-lbl" x="512" y="132">six sessions at 8k context</text>
<text class="v1-sub" x="512" y="150">1.04 GiB each, 8192 tokens per GiB</text>
<text class="v1-warn" x="530" y="176">session 7 lands in the fragmentation reserve and is rejected</text>
<text class="v1-sub" x="120" y="132">fixed cost, paid once at load</text>
</svg>
<figcaption>The budget fills left to right: weights and context are paid once, then each admitted session claims a slice of what remains until the next arrival has nowhere to go.</figcaption>
</figure>

#### Worked example: how many users fit on a 4090 and an A100

Take the numbers above and subtract. All values in GiB; every line is derived from the formulas in this section except the card totals, which are what `nvidia-smi` reports and you should confirm on your own hardware.

| Line item                      | RTX 4090 | A100 80GB | Source                                             |
| ------------------------------ | -------- | --------- | -------------------------------------------------- |
| Reported total                  | 23.99    | 79.15     | reproduce: `nvidia-smi --query-gpu=memory.total`   |
| CUDA context + kernel modules   | −0.60    | −0.60     | reproduce: `probe_context.py` below                |
| Weights, bf16                   | −14.96   | −14.96    | derived: 8.03e9 params × 2 B                       |
| Activations at your chunk size  | −1.00    | −2.00     | derived: chunk × vocab × dtype, plus residuals     |
| Fragmentation reserve, 5%       | −1.20    | −4.00     | derived: 5% of total                               |
| **Left for KV cache**           | **6.23** | **57.59** | derived: subtraction                               |
| Tokens of cache at 128 KiB each | 51,036   | 471,777   | derived: GiB × 8192                                |
| Concurrent sessions at 8k ctx   | 6        | 57        | derived: tokens ÷ 8192                             |

Six concurrent chat sessions on a 4090, before you have written a single line of scheduler. That is the fact that makes the rest of this series necessary: nearly every technique in it — paged blocks, prefix sharing, KV quantization, GQA-aware layouts — is a way of buying back rows in the last two lines of that table.

And notice what happens if you load in fp32 instead of bf16 out of habit. Weights become 29.92 GiB, which does not fit on a 4090 at all, and on an A100 leaves 42.6 GiB for cache instead of 57.6 — a 26% cut in concurrency for numerics you do not need. Load dtype is a capacity decision.

Here is the context probe, which is the one number in that table you cannot derive:

```python
# nanoserve/probe_context.py
import torch

free_before, total = torch.cuda.mem_get_info()
torch.cuda.init()
# Force the driver to materialize a context and load the default kernel modules.
_ = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
free_after, _ = torch.cuda.mem_get_info()

MIB = 1024**2
print(f"total          : {total / MIB:,.0f} MiB")
print(f"context + libs : {(free_before - free_after) / MIB:,.0f} MiB")
print(f"torch allocated: {torch.cuda.memory_allocated() / MIB:,.1f} MiB")
```

On a recent CUDA release you should see the context land somewhere in the 250–700 MiB range on a consumer card, higher once cuBLAS and a fused attention backend have loaded their kernels. Print it on your own hardware before you plan a budget; the number moves with driver version.

## 4. Four ways to load, and what each one peaks at

Now the part that separates a loader you can ship from one that works on your laptop. The question is not "how fast" — it is "what is the peak host memory, and how many times do the bytes get copied?"

![Side by side comparison of a naive two-copy load against a streamed load, contrasting peak host memory of 32 GB with 1.05 GB](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-4.webp)

### Strategy A: the naive path

```python
model = LlamaForCausalLM(config)          # eager init: 16.1 GB of random floats on CPU
sd = torch.load("pytorch_model.bin")      # + 16.1 GB of checkpoint on CPU
model.load_state_dict(sd)                 # copies sd into model's existing storages
del sd
model.cuda()                              # + 16.1 GB on GPU, while CPU copy still live
```

Count the copies. Constructing the module allocates every parameter and fills it with random values you are about to overwrite — 16.1 GB. `torch.load` allocates another 16.1 GB. Peak host memory is 32 GB before you have touched the GPU. Then `load_state_dict` copies element-wise into the module's existing storages, which is another full pass over 32 GB of memory bandwidth for no benefit. Then `.cuda()` allocates 16.1 GB on the device and copies. The GPU-side result is correct and the host-side journey was three full materializations.

On a 32 GB host this swaps. On a 16 GB CI runner it dies. And the eager random initialization is not free either — filling 8 billion floats with a normal distribution is real work, typically seconds, all of it thrown away.

### Strategy B: mmap plus per-tensor copy

Replace `torch.load` with `safe_open` and the second 16.1 GB disappears — the file is mapped, not read, and pages arrive on demand. But if you still construct the module eagerly you still pay the first 16.1 GB of random init. Better, not fixed.

### Strategy C: `device_map` and `low_cpu_mem_usage`

This is what `transformers` does via [Accelerate's big-model loading](https://huggingface.co/docs/accelerate/usage_guides/big_modeling). `low_cpu_mem_usage=True` builds the module on the meta device — shapes and dtypes, zero bytes — then walks the checkpoint materializing one tensor at a time onto the target. `device_map="auto"` extends that to spreading layers across several GPUs and spilling the remainder to CPU or disk when they do not fit. It is the right default for a research script and it is genuinely well engineered.

It is not what we want for `nanoserve`, for one reason: we are going to own the module, the dtype policy and the placement, and a loader we can read in fifty lines is worth more to this series than one that handles every case. So we write strategy D and understand every line.

### Strategy D: stream into pre-allocated device buffers

Three PyTorch mechanisms make this work, and each is worth knowing on its own.

**The meta device.** Under `with torch.device("meta")`, tensor constructors record shape, dtype and strides but allocate no storage. A whole model built this way costs kilobytes. You can inspect it, count its parameters, and check names against a checkpoint — all before committing a byte.

**`Module.to_empty(device=...)`.** Moves a meta module to a real device by allocating *uninitialized* storage of the right shape. No random fill, no zeroing, no host round-trip. The parameters contain garbage, which is fine because every one of them is about to be overwritten. The catch is right there: any parameter or buffer you forget to load stays garbage, silently. Section 8 has the guard.

**`load_state_dict(..., assign=True)`.** By default `load_state_dict` *copies* source data into the module's existing storages. With `assign=True` it rebinds the parameter to the source tensor's storage instead. When the source tensor already lives on the GPU in the right dtype, assignment means zero additional copies and zero additional peak. This is the difference between "the tensor arrives" and "the tensor arrives twice".

Put together, `nanoserve`'s loader:

```python
# nanoserve/weights.py  (continued)
import torch
from torch import nn


@torch.no_grad()
def load_weights(
    model: nn.Module,
    ckpt: ShardedCheckpoint,
    device: str = "cuda:0",
    dtype: torch.dtype | None = torch.bfloat16,
    rename: dict[str, str] | None = None,
    strict: bool = True,
) -> dict[str, int]:
    """Stream a checkpoint into `model`, one tensor at a time.

    `model` may be on the meta device; it is materialized with to_empty first.
    Peak host memory is one tensor, not one checkpoint.
    """
    if any(p.is_meta for p in model.parameters()):
        model.to_empty(device=device)

    rename = rename or {}
    target = dict(model.named_parameters())
    target.update(dict(model.named_buffers()))
    seen: set[str] = set()
    stats = {"tensors": 0, "bytes": 0, "cast": 0}

    for name in ckpt.keys():
        key = rename.get(name, name)
        param = target.get(key)
        if param is None:
            if strict:
                raise KeyError(f"checkpoint has {name!r} but the model does not")
            continue

        # Read the byte range straight onto the device: no torch CPU tensor.
        src = ckpt.get_to(name, device=device)

        if src.shape != param.shape:
            raise ValueError(
                f"{key}: checkpoint {tuple(src.shape)} vs model {tuple(param.shape)}"
            )
        if dtype is not None and src.dtype != dtype:
            src = src.to(dtype)          # on-device cast, allocates one tensor
            stats["cast"] += 1

        # Rebind rather than copy. param.data = src is the assign=True idea,
        # done explicitly so the ownership is obvious at the call site.
        param.data = src
        seen.add(key)
        stats["tensors"] += 1
        stats["bytes"] += src.untyped_storage().nbytes()

    missing = sorted(set(target) - seen)
    if missing and strict:
        raise RuntimeError(f"{len(missing)} tensors never loaded, first: {missing[:5]}")
    torch.cuda.synchronize()
    return stats
```

And the call site:

```python
# nanoserve/build.py
import torch

from nanoserve.model import Llama          # written in the forward-pass post
from nanoserve.weights import ShardedCheckpoint, load_weights

ckpt = ShardedCheckpoint("Llama-3.1-8B")

with torch.device("meta"):
    model = Llama.from_config("Llama-3.1-8B/config.json")   # 0 bytes allocated

stats = load_weights(model, ckpt, device="cuda:0", dtype=torch.bfloat16)
model.eval()

print(f"loaded {stats['tensors']} tensors, {stats['bytes'] / 1e9:.2f} GB, "
      f"{stats['cast']} casts")
print(f"cuda allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
print(f"peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
```

Expected output for a bf16 Llama-3.1-8B checkpoint loaded as bf16:

```console
loaded 291 tensors, 16.06 GB, 0 casts
cuda allocated: 14.96 GiB
peak allocated: 14.96 GiB
```

Peak equals allocated. That equality is the whole point of the exercise: no transient doubled the footprint. Watch what happens if you load the same checkpoint with `dtype=torch.float32` — `cast` becomes 291, peak allocated exceeds allocated by the size of the largest tensor while both the bf16 source and the fp32 destination are live, and the final figure is 29.92 GiB, which does not fit on the 4090 at all.

<figure class="blog-anim">
<svg viewBox="0 0 660 230" role="img" aria-label="Tensors flow from a checkpoint file through a memory mapping directly into GPU buffers while a host copy slot stays empty" style="width:100%;height:auto;max-width:800px">
<style>
.v2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.v2-ghost{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:6 5}
.v2-dot{fill:var(--accent,#6366f1)}
.v2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.v2-sub{font:500 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.v2-wire{stroke:var(--border,#d1d5db);stroke-width:2}
@keyframes v2-flow{0%{transform:translateX(0);opacity:0}8%{opacity:1}92%{opacity:1}100%{transform:translateX(392px);opacity:0}}
.v2-p{animation:v2-flow 6s linear infinite}
.v2-p2{animation-delay:1.5s}
.v2-p3{animation-delay:3s}
.v2-p4{animation-delay:4.5s}
@media (prefers-reduced-motion:reduce){.v2-p{animation:none;opacity:1}}
</style>
<line class="v2-wire" x1="140" y1="72" x2="250" y2="72"/>
<line class="v2-wire" x1="360" y1="72" x2="470" y2="72"/>
<rect class="v2-box" x="30" y="38" width="110" height="68" rx="8"/>
<rect class="v2-box" x="250" y="38" width="110" height="68" rx="8"/>
<rect class="v2-box" x="470" y="38" width="110" height="68" rx="8"/>
<rect class="v2-ghost" x="250" y="145" width="110" height="52" rx="8"/>
<circle class="v2-dot v2-p" cx="85" cy="72" r="9"/>
<circle class="v2-dot v2-p v2-p2" cx="85" cy="72" r="9"/>
<circle class="v2-dot v2-p v2-p3" cx="85" cy="72" r="9"/>
<circle class="v2-dot v2-p v2-p4" cx="85" cy="72" r="9"/>
<text class="v2-lbl" x="85" y="128">safetensors</text>
<text class="v2-sub" x="85" y="145">16.06 GB on disk</text>
<text class="v2-lbl" x="305" y="128">mapped range</text>
<text class="v2-sub" x="305" y="72" dy="-24">page cache</text>
<text class="v2-lbl" x="525" y="128">GPU buffer</text>
<text class="v2-sub" x="525" y="145">14.96 GiB final</text>
<text class="v2-sub" x="305" y="170">second host copy</text>
<text class="v2-sub" x="305" y="186">never allocated</text>
<text class="v2-sub" x="305" y="215">peak host stays at one tensor: 1.05 GB</text>
</svg>
<figcaption>Each tensor moves from its byte range through the mapping straight to device memory; the host-side staging buffer the naive path would allocate is never created.</figcaption>
</figure>

### Making the copy itself fast

Two knobs matter once the copies are single.

**Pinned host memory.** A host-to-device copy from pageable memory forces the driver to stage through an internal pinned buffer, roughly halving effective bandwidth and blocking. From page-locked memory the DMA engine reads directly. If you are staging through torch on the host, allocate the staging tensor with `pin_memory=True` and copy with `non_blocking=True`. If you are using `safe_open(..., device="cuda:0")` the library handles the transfer for you.

**Overlap.** Disk read and H2D copy are different hardware. With two CUDA streams and a small worker pool you can be reading shard 2 off the disk while shard 1 is crossing PCIe, which turns a serial 2.3 s + 0.64 s into roughly max(2.3, 0.64) plus a tail. That is a 20% cold-start win for maybe forty lines of code, and it is the right kind of complexity to defer until you have measured that cold start actually matters for your deployment.

## 5. Dtypes: exponent bits are the interesting part

Everyone knows fp16 and bf16 are both "two bytes". Almost nobody can say what changes. The answer is entirely in how those sixteen bits are split between exponent and mantissa, and it decides whether your model runs or produces NaNs.

![Decision tree from hardware generation to load dtype, branching to bf16 on Ampere, fp16 or fp32 on older cards, and fp8 on Hopper](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-5.webp)

| Format | Sign | Exponent | Mantissa | Largest finite | Smallest normal | Relative step | Bytes |
| ------ | ---- | -------- | -------- | -------------- | --------------- | ------------- | ----- |
| fp32   | 1    | 8        | 23       | 3.40e38        | 1.18e−38        | 1.2e−7        | 4     |
| bf16   | 1    | 8        | 7        | 3.39e38        | 1.18e−38        | 7.8e−3        | 2     |
| fp16   | 1    | 5        | 10       | 65,504         | 6.10e−5         | 9.8e−4        | 2     |
| fp8 E4M3 | 1  | 4        | 3        | 448            | 1.95e−3         | 1.25e−1       | 1     |
| fp8 E5M2 | 1  | 5        | 2        | 57,344         | 6.10e−5         | 2.5e−1        | 1     |

Every row of that table is `Source: cited` — fp32 and fp16 are IEEE 754 binary32 and binary16, bf16 is [Google's bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus), and the two fp8 encodings are from the [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) specification adopted by NVIDIA, Arm and Intel. The "relative step" column is $2^{-(m+1)}$ where $m$ is the mantissa width, the machine epsilon.

Read the table by column and the design becomes obvious. **bf16 is fp32 with sixteen mantissa bits deleted.** Same exponent width, same dynamic range, same overflow threshold — you lose precision, not range. **fp16 buys three extra mantissa bits by giving up three exponent bits**, and that is the whole story of mixed-precision training pain: a value above 65,504 becomes infinity, a value below 6.1e−5 flushes toward zero, and both happen in real transformers.

### Why bf16 is the inference default on Ampere and later

The A100 was the first NVIDIA data-center part with bf16 tensor cores; Ada (the 4090) and Hopper have them too. Once the hardware runs bf16 at the same rate as fp16, the choice is free, and bf16's range makes it strictly safer. Some models make it more than a preference: a checkpoint trained in bf16 can contain weights or produce intermediate activations whose magnitude exceeds fp16's 65,504 ceiling. Attention logits before the softmax and the residual stream of deep models are the usual offenders. Load such a model in fp16 and you get `inf` in the logits, then `NaN` out of the softmax, then a stream of garbage tokens or an assertion in the sampler.

What silently happens if you load bf16 weights and run in fp16 without overflowing is more insidious: the cast succeeds, the model runs, and quality drops in ways an eyeball test does not catch. Values above the fp16 max saturate. Values in the fp16 subnormal range lose bits. You have introduced a quantization error nobody chose and nobody logged. This is the single most common cause of "the same model is dumber on our stack than on the reference implementation."

Here is a pre-flight check worth having in `nanoserve`, run once at load:

```python
# nanoserve/dtype_check.py
import torch

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05


@torch.no_grad()
def audit_dtype(ckpt, target: torch.dtype, sample_every: int = 1) -> list[str]:
    """Warn about tensors that will lose information when cast to `target`."""
    warnings: list[str] = []
    for i, (name, t) in enumerate(ckpt.items()):
        if i % sample_every:
            continue
        finite = t.float()
        amax = finite.abs().max().item()
        if target is torch.float16:
            if amax > FP16_MAX:
                warnings.append(f"{name}: max |w| = {amax:.1f} overflows fp16")
            elif amax < FP16_MIN_NORMAL:
                warnings.append(f"{name}: max |w| = {amax:.2e} is fp16-subnormal")
        if target is torch.bfloat16 and t.dtype is torch.float32:
            rel = (finite - finite.to(torch.bfloat16).float()).abs().max().item() / (amax + 1e-30)
            if rel > 5e-3:
                warnings.append(f"{name}: bf16 relative error {rel:.2e}")
    return warnings


if __name__ == "__main__":
    from nanoserve.weights import ShardedCheckpoint

    for w in audit_dtype(ShardedCheckpoint("Llama-3.1-8B"), torch.float16):
        print("WARN", w)
```

Weight magnitudes in a healthy transformer sit well inside fp16's range, so on Llama-3.1-8B you should expect this to print nothing — the overflow risk lives in activations, not weights, and this script only sees weights. Its real value is on fine-tuned and merged checkpoints, where a bad merge can produce a projection with a genuinely enormous entry. That is a two-line diagnosis instead of a two-day one.

### Where to pay the casting cost

If the checkpoint dtype and the target dtype differ, someone converts. You have three places to do it:

**At load, on the GPU.** `src.to(dtype)` after the tensor lands. One extra allocation the size of the largest tensor, for a moment, and then it is done forever. This is the default in `load_weights` above and it is almost always right.

**At load, on the CPU.** Convert before transferring. Halves PCIe traffic if you are going fp32 to bf16, but CPU casting of 8 billion elements is slow — single-threaded it is seconds. Worth it only when host memory bandwidth is plentiful and PCIe is the bottleneck.

**Offline, once.** Convert the checkpoint on disk and never think about it again. `safetensors.torch.save_file` with the cast tensors gives you a bf16 checkpoint that is half the size of the fp32 one, loads twice as fast because there are half as many bytes to read, and needs no cast at all. If you serve the same model from many replicas, do this. It is the highest-leverage ten lines in the post.

```python
# tools/convert_dtype.py — convert a checkpoint on disk, once.
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from nanoserve.weights import ShardedCheckpoint

src, dst, dtype = Path("in"), Path("out"), torch.bfloat16
dst.mkdir(exist_ok=True)
ckpt = ShardedCheckpoint(src)

shard: dict[str, torch.Tensor] = {}
weight_map, total, part, budget = {}, 0, 1, 5_000_000_000  # 5 GB shards

def flush():
    global shard, part
    name = f"model-{part:05d}.safetensors"
    save_file(shard, dst / name, metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = name
    shard, part = {}, part + 1

for name, t in ckpt.items():
    t = t.to(dtype) if t.is_floating_point() else t
    shard[name] = t.contiguous()
    total += t.numel() * t.element_size()
    if sum(v.numel() * v.element_size() for v in shard.values()) > budget:
        flush()
if shard:
    flush()

(dst / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
)
print(f"wrote {part - 1} shards, {total / 1e9:.2f} GB")
```

Note `t.contiguous()`. A cast can produce a non-contiguous tensor, and safetensors stores C-contiguous bytes; writing a view would either fail or silently write the wrong layout. Also note the `is_floating_point()` guard — integer tensors in a quantized checkpoint must not be cast, a mistake we will come back to.

## 6. Device placement: what has to be on the GPU

With weights loaded, the next question is where each one lives. The naive answer is "all of it on the GPU," which is right when it fits. When it does not, the interesting question is which tensors are cheapest to exile, and the answer follows from one formula:

$$
t_{\text{offload}} = \frac{\text{bytes the tensor touches per token}}{\text{BW}_{\text{PCIe}}}
$$

PCIe 4.0 x16 has a theoretical 32 GB/s; pinned-memory transfers in practice reach roughly 22–26 GB/s, and you should confirm yours with a bandwidth test rather than trust the marketing figure. Take 25 GB/s as the working number. What matters is that the numerator differs enormously between tensor types.

![Table of model tensors against their bf16 size, bytes touched per token, and the cost of keeping them off the GPU](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-6.webp)

**The embedding table is a gather.** `embed_tokens` is 1.05 GB, but a decode step looks up one row per sequence: $4096 \times 2 = 8192$ bytes. Over PCIe that is 0.3 microseconds. Keeping the embedding pinned on the host and gathering rows across the bus is nearly free, and buys back a full gigabyte of VRAM. Almost nobody does this and it is one of the better tricks in the box for a memory-starved consumer card.

**The LM head is not.** `lm_head` is the same 1.05 GB, but computing logits reads *every* row: the full matrix, every token. $1.05\ \text{GB} / 25\ \text{GB/s} = 42$ ms per token, which by itself would cap you at 24 tok/s. The LM head stays resident.

**Decoder layers are the expensive middle.** One layer is $218{,}112{,}000 \times 2 = 436$ MB in bf16, and a decode step reads all of it. $436\ \text{MB} / 25\ \text{GB/s} = 17.4$ ms per layer per token. Offload eight of the thirty-two layers and you have added 139 ms to every single token.

#### Worked example: the true cost of CPU offload

Compare three placements for Llama-3.1-8B on an RTX 4090. The decode step is memory-bound — its floor is weight bytes divided by the bandwidth of whatever memory holds them — so we can derive each case. The 4090's 1,008 GB/s of GDDR6X bandwidth is from [NVIDIA's product specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).

| Placement                        | Bytes over the slow link per token | Derived TPOT floor | Ceiling at batch 1 | Source            |
| -------------------------------- | ---------------------------------- | ------------------ | ------------------ | ----------------- |
| All 32 layers in VRAM            | 0 (16.06 GB at 1,008 GB/s HBM)     | 15.9 ms            | 63 tok/s           | derived + cited   |
| 8 layers on host, 24 in VRAM     | 3.49 GB at 25 GB/s PCIe            | 139 ms + 12 ms     | 6.6 tok/s          | derived           |
| All weights on host              | 16.06 GB at 25 GB/s PCIe           | 642 ms             | 1.6 tok/s          | derived           |

Ten times slower for offloading a quarter of the model, forty times slower for all of it. And that 15.9 ms figure in the first row is the theoretical floor a perfect engine would hit; a real one lands above it, and getting close to it is what the kernel track of this series is about. On real hardware at batch 1 you should expect somewhere in the 40–60 tok/s band for an 8B bf16 model on a 4090 — run the benchmark from the naive-decode-loop post and report yours.

The honest conclusion: **CPU offload is a correctness feature, not a performance feature.** It lets a model run that otherwise would not run at all, at a speed that is fine for a single developer poking at a prompt and unacceptable for anything serving users. If you find yourself reaching for it in production, the right moves are a smaller model, a quantized model, or a bigger card — and the [quantization for serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) path is usually the cheapest of the three.

There is one exception worth naming: mixture-of-experts models, where a given token activates a small fraction of the expert weights. Offloading cold experts to host memory moves a much smaller numerator across the bus, and the arithmetic can work out. That is a Track G problem and it depends entirely on routing statistics.

If you do offload, pin the host tensors. `tensor.pin_memory()` on a 3.5 GB block takes real time and real host RAM that cannot be swapped, so do it once at load, never per step. Unpinned offload roughly halves the already-bad bandwidth above.

## 7. Load time is a real metric

Cold start does not matter until you autoscale, and then it is the only thing that matters. A pod that takes ninety seconds to serve its first token cannot absorb a traffic spike; you end up over-provisioning to hide the latency, which is the cost you were autoscaling to avoid.

![Timeline of a cold start from process launch through CUDA initialization, disk read, host to device copy and first token](/imgs/blogs/loading-weights-safetensors-dtypes-and-device-placement-7.webp)

Where do the seconds go? Four places, and only one of them is the copy everyone optimizes.

**CUDA initialization.** Creating a context and loading kernel modules is typically one to two seconds, dominated by the CUDA runtime and any libraries you import. It is fixed cost and mostly out of your control, though lazy module loading (`CUDA_MODULE_LOADING=LAZY`, the default since CUDA 11.7) helps and importing fewer libraries helps more.

**Reading the checkpoint off storage.** 16.06 GB at 7 GB/s on a PCIe 4.0 NVMe is 2.3 seconds. At 3.5 GB/s on an older NVMe it is 4.6 s. On network storage at 500 MB/s it is 32 seconds, which is why the first thing to check on a slow cold start is whether the model lives on a volume or a local disk. **If the page cache is warm — the second pod on the same node, or a restart — this term goes to nearly zero**, because the mapped pages are already resident and the "read" is a page-table update. That asymmetry is why a benchmark that reloads in a loop reports a number that has nothing to do with your cold start.

**Host to device.** 16.06 GB at 25 GB/s pinned is 0.64 s; unpinned, closer to 1.3 s. This is the smallest of the three big terms, which is exactly why optimizing it first is a mistake.

**Warmup.** The first forward pass pays for cuBLAS heuristic selection, autotuning, and lazily loaded kernels. Half a second is typical, and `torch.compile` turns it into tens of seconds to minutes — a trade we take up in the compile post, where the fix is a compilation cache.

Here is the staged timer. It reports each phase separately, because a single end-to-end number tells you nothing about which phase to attack:

```python
# tools/time_load.py
import time
from contextlib import contextmanager

import torch

from nanoserve.model import Llama
from nanoserve.weights import ShardedCheckpoint, load_weights

stages: dict[str, float] = {}


@contextmanager
def stage(name: str):
    torch.cuda.synchronize() if torch.cuda.is_initialized() else None
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_initialized() else None
    stages[name] = time.perf_counter() - t0


with stage("cuda_init"):
    torch.zeros(1, device="cuda")

with stage("open_index"):
    ckpt = ShardedCheckpoint("Llama-3.1-8B")

with stage("build_meta"):
    with torch.device("meta"):
        model = Llama.from_config("Llama-3.1-8B/config.json")

with stage("stream_weights"):
    stats = load_weights(model, ckpt, device="cuda:0", dtype=torch.bfloat16)

with stage("first_forward"):
    ids = torch.randint(0, 128000, (1, 8), device="cuda")
    with torch.inference_mode():
        model(ids)

total = sum(stages.values())
for k, v in stages.items():
    print(f"{k:16s} {v * 1000:8.1f} ms  {100 * v / total:5.1f}%")
print(f"{'TOTAL':16s} {total * 1000:8.1f} ms")
print(f"read {stats['bytes'] / 1e9:.2f} GB -> "
      f"{stats['bytes'] / 1e9 / stages['stream_weights']:.2f} GB/s effective")
```

Run it twice. The first run is cold; the second, with the page cache warm, isolates everything that is not disk. On an RTX 4090 with a PCIe 4.0 NVMe you should expect a cold total somewhere in the 4–8 second range and a warm total in the 1.5–3 second range, with `stream_weights` dominating the cold run and `cuda_init` dominating the warm one. If your cold number is above twenty seconds, your checkpoint is on network storage. If your warm number is above ten seconds, you are making a copy you do not know about — check `max_memory_allocated` against `memory_allocated`.

The `torch.cuda.synchronize()` calls in `stage` are not optional. CUDA operations are asynchronous; without a sync you are timing how long it took to *enqueue* the work, which for a load that is mostly DMA gives you an impressively wrong and impressively fast number. This discipline is the subject of [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark), and it applies to every measurement in this series.

Three things that actually move cold start, in order of leverage:

1. **Put the checkpoint on local NVMe.** Network storage is a 10× term. Bake the model into the image, or pull it to a local volume on node start and share it across pods so the page cache stays warm.
2. **Ship the dtype you serve.** A pre-converted bf16 checkpoint halves the bytes read versus fp32. There is no cast and no second allocation.
3. **Overlap read and copy.** Worth roughly 20% of the streaming phase, and worth doing only after the first two.

Things that do not help as much as people expect: faster PCIe (it is the smallest term), more CPU threads for the copy (DMA is doing the work), and compression (decompression is slower than an NVMe read).

## 8. Failure modes, and the guards that catch them

Every failure below is one I have seen reported, debugged, or watched someone else lose an afternoon to. What they share is that most of them do not raise.

### The transpose that only fails on rectangles

`torch.nn.Linear` stores its weight as `[out_features, in_features]` and computes $x W^\top$. Some checkpoint families store `[in_features, out_features]`. For Llama-3.1-8B, `q_proj` and `o_proj` are 4096×4096 — **square** — so a transposed load produces a shape that matches, passes every assertion, and computes a completely different function. Meanwhile `k_proj` is 1024×4096 and will raise a shape error immediately.

The result is a model that loads with one loud error you fix by "helpfully" transposing everything, after which the rectangular tensors are right and the square ones are silently wrong, and the model emits fluent nonsense. The guard is not a shape check — shapes agree. It is a numerical check against a reference implementation, which is why the next post in this series builds a correctness harness that compares logits against `transformers` within a tolerance before anything else is allowed to happen.

### Tied embeddings that were not tied

Some models set `tie_word_embeddings: true`, meaning `lm_head.weight` **is** `model.embed_tokens.weight` — one storage, two names. Such a checkpoint contains no `lm_head.weight` key at all. If your loader is strict about missing tensors it raises, which is annoying but safe. If your model was built with an untied `lm_head` and `to_empty()` gave it uninitialized memory, and your loader is not strict, then `lm_head` holds garbage and the model emits uniform noise. The fix is three lines and belongs in every loader:

```python
# nanoserve/weights.py  (continued)
def apply_tying(model, config: dict, seen: set[str]) -> None:
    """Point lm_head at the embedding when the config says they are tied."""
    if not config.get("tie_word_embeddings", False):
        return
    if "lm_head.weight" in seen:
        raise ValueError("config says tied but the checkpoint ships an lm_head")
    model.lm_head.weight = model.model.embed_tokens.weight  # shared storage
    seen.add("lm_head.weight")
```

The memory this saves is not small. For Llama-3.1-8B an untied head costs $128256 \times 4096 \times 2 = 1.05$ GB, 6.5% of the model. For a model with a large vocabulary and a modest hidden size the ratio is much worse — Gemma 3's vocabulary is roughly 262k tokens, so read your own `config.json`, multiply `vocab_size` by `hidden_size` by two bytes, and see what tying is worth on the model in front of you. On small models it can be a fifth of the parameters.

### A quantized checkpoint loaded as if it were bf16

An AWQ or GPTQ checkpoint stores packed integers plus separate scale and zero-point tensors: `model.layers.0.self_attn.q_proj.qweight` as int32, `.scales` as fp16, `.qzeros` as int32. If your loader casts everything to bf16 with a blanket `.to(dtype)`, the packed integers become floats that mean nothing, and you get a model that loads and produces garbage. Worse, the shapes are plausible — a packed int32 `qweight` for a 4096×4096 layer has a shape that looks like a legitimate tensor.

Two guards. First, never cast non-floating-point tensors — the `is_floating_point()` check in the conversion tool and the dtype filter in `load_weights`. Second, refuse to load a checkpoint whose `config.json` has a `quantization_config` block unless the loader is a quantized loader:

```python
# nanoserve/weights.py  (continued)
SUPPORTED_QUANT = {"fp8", "compressed-tensors"}  # what nanoserve handles today


def assert_loadable(config: dict) -> None:
    qc = config.get("quantization_config")
    if qc is None:
        return
    method = qc.get("quant_method", "unknown")
    if method not in SUPPORTED_QUANT:
        raise NotImplementedError(
            f"checkpoint is {method}-quantized; nanoserve cannot load it as dense. "
            f"Use a bf16 checkpoint or the {method} loader."
        )
```

Failing loudly at load beats debugging a model that answers every question with the same three tokens.

### A `config.json` that does not match the weights

Community re-uploads, merges and fine-tunes routinely ship a config that has drifted from the tensors. A wrong `rope_theta` gives you a model that is coherent for short prompts and degrades past a few thousand tokens — very hard to spot. A wrong `num_key_value_heads` gives a shape error, which is a gift. A wrong `vocab_size` gives an index error at the first out-of-range token, sometimes hours in.

The cheap guard is to derive what you can from the tensors and cross-check the config, at load, in about fifteen lines:

```python
# nanoserve/weights.py  (continued)
def cross_check(ckpt: ShardedCheckpoint, config: dict) -> None:
    """Derive shape facts from the checkpoint and compare them to config.json."""
    with_prefix = [k for k in ckpt.keys() if k.startswith("model.layers.")]
    n_layers = 1 + max(int(k.split(".")[2]) for k in with_prefix)
    embed = ckpt.get("model.embed_tokens.weight")
    vocab, hidden = embed.shape
    kv = ckpt.get("model.layers.0.self_attn.k_proj.weight").shape[0]

    checks = {
        "num_hidden_layers": (n_layers, config["num_hidden_layers"]),
        "vocab_size": (vocab, config["vocab_size"]),
        "hidden_size": (hidden, config["hidden_size"]),
        "kv_proj_out": (kv, config["num_key_value_heads"] * config.get(
            "head_dim", config["hidden_size"] // config["num_attention_heads"])),
    }
    for field, (from_weights, from_config) in checks.items():
        if from_weights != from_config:
            raise ValueError(
                f"{field}: weights say {from_weights}, config says {from_config}"
            )
```

Note what this cannot check: `rope_theta`, `rms_norm_eps`, and the tokenizer. Those have no tensor to compare against, and the only defense is the logits-equivalence test in the next post.

### The tensor you forgot to load

`to_empty()` allocates uninitialized memory. A rotary embedding buffer, a norm weight added by a newer architecture, a bias tensor your module has and the checkpoint does not — any of these left unwritten holds whatever was in that VRAM before. Sometimes zeros, and the model looks broken in an obvious way. Sometimes leftovers from another process, and the model looks *almost* right. The `missing` check at the end of `load_weights` exists for exactly this and should be strict by default. `strict=False` is for people who know which tensor they are skipping and why.

### Stress-testing the loader

Put the loader under pressure the way the rest of the system will:

- **Batch 1 versus batch 64.** Load time does not change; the activation line of the VRAM budget does. Re-derive it at your maximum batch before you size the KV pool, or the pool will be right at batch 1 and OOM at batch 64.
- **128k context.** Weights are unchanged, but the KV budget at 128k tokens is $128{,}000 \times 128\ \text{KiB} = 15.6$ GiB for a *single* sequence — more than the free space on a 4090 after weights. Long context is a cache problem, not a weight problem, and it is why long-context serving usually means quantizing the cache or a bigger card.
- **An L4 instead of an A100.** Same 24 GB class as the 4090 but roughly 300 GB/s of bandwidth instead of 1,008 GB/s, so the derived decode floor becomes $16.06 / 300 = 53.5$ ms per token, about 19 tok/s. Same load code, same budget, a third of the speed. Bandwidth, not capacity, is what you are buying.
- **Two processes on one card.** Each pays its own CUDA context, and neither PyTorch allocator knows about the other. Two 8B models on one 4090 is 29.9 GiB of weights alone: impossible. Even one model plus a small embedding service needs the budget recomputed with both contexts counted.
- **A cold page cache in Kubernetes.** The first pod on a node reads from disk; the second maps pages that are already resident. Your p50 cold start and your p99 cold start are different phenomena, and only the first pod's number is the one that matters during a scale-up.

## 9. Case studies and public numbers

**The safetensors format, and what it was built against.** HuggingFace's [format documentation](https://huggingface.co/docs/safetensors/index) states the design goals plainly: no arbitrary code execution, zero-copy loading via memory mapping, and lazy access to individual tensors. Their [speed comparison page](https://huggingface.co/docs/safetensors/speed) publishes load-time numbers against `torch.load`; those are their measurements on their hardware, and the mechanism behind them — mapping instead of reading, and skipping the pickle interpreter — is what you should expect to reproduce in kind, if not in magnitude.

**PyTorch's `weights_only` default flip.** The [PyTorch 2.6 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.6.0) document `torch.load` defaulting to `weights_only=True`, an explicitly breaking change made on security grounds. It is a useful data point for anyone weighing "is the pickle risk real": a project as conservative as PyTorch about backward compatibility accepted a breaking default change to close it.

**PagedAttention's framing of the memory problem.** The [vLLM paper](https://arxiv.org/abs/2309.06180) measures how much of a serving system's memory goes to waste through fragmentation and over-reservation in a contiguous KV cache, and shows that reclaiming it directly increases the number of requests served. That paper is about the cache, not the weights, but it is the strongest public statement of the point figure 1 makes: after weights, memory *is* the product, and every byte you fail to reclaim is throughput you do not have. The [continuous batching and PagedAttention post](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) covers the mechanism.

**Vendor bandwidth figures, and why to keep them handy.** NVIDIA lists 1,008 GB/s for the [RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), about 2,039 GB/s for the A100 80GB SXM, and 3.35 TB/s for the H100 SXM in their respective datasheets. Those three numbers, divided into your weight bytes, give the theoretical batch-1 decode floor for any model on any of those cards in one division, before you have written any code. The [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) explains why that division is the right first-order model.

**Architectures that change the arithmetic.** Grouped-query attention is the reason `kv_heads` appears in the KV formula at all: Llama-3.1-8B's eight KV heads against thirty-two query heads cut cache bytes per token by 4× versus full multi-head attention, from 512 KiB to 128 KiB. Multi-head latent attention goes further by caching a compressed latent. The [modern LLM architectures post](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) compares these choices across Llama, Qwen, Gemma and DeepSeek; every one of them shows up in your budget table as a different number in the same formula.

## 10. When to reach for this, and when not to

**Write your own loader when:** you are building an engine and need to own placement, dtype policy and shard layout; you need a cold-start budget you can explain line by line; you are loading into a custom module that `transformers` has never seen; or you are debugging a memory problem and need to know exactly how many copies exist and when.

**Use `transformers` with `device_map` and `low_cpu_mem_usage` when:** you want a model in memory to test something and load time is not the product. It is well engineered, it handles the long tail of checkpoint quirks, and reimplementing it to save a second is not a good trade.

**Use vLLM, SGLang or TGI when:** you are serving users. Their loaders already do everything in this post plus tensor-parallel sharding, quantized formats, and per-format weight rewriting, and their engines do the ninety-five percent of the work that comes after loading. This series builds `nanoserve` so you *understand* what they do, not so you deploy it. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is the tour of what a production version of this looks like.

**Do not bother optimizing load time when:** your service runs long-lived pods that reload once a week. Cold start matters when it is on the critical path of a scale-up, and not otherwise. Measure whether it is before you spend a day on overlapping streams.

**Do not reach for CPU offload in production.** The derivation in section 6 is not close. If the model does not fit, quantize it or get a bigger card.

## Key takeaways

1. **A checkpoint is a header and a byte range.** safetensors is eight bytes of length, a JSON header, then raw contiguous tensors — parseable with `struct.unpack` and mappable without reading. Everything else about the format follows from that.
2. **Derive the size, do not look it up.** Parameters equal $2Vd + L(2d^2 + 2dH_{kv}d_h + 3dd_{ff} + 2d) + d$; multiply by bytes per parameter. Ninety seconds with a `config.json` tells you whether the model fits before you rent the machine.
3. **Weights are the first line of the budget, not the budget.** CUDA context, activations, fragmentation reserve and the KV cache all come out of the same card. An 8B bf16 model on a 24 GiB card leaves about 6 GiB of cache, which is about six concurrent 8k-token sessions.
4. **Peak host memory is the metric that kills you, not load speed.** Eager construction plus a full state dict peaks at twice the checkpoint. Meta device plus `to_empty()` plus per-tensor assignment peaks at one tensor.
5. **`max_memory_allocated` should equal `memory_allocated` after a clean load.** If it does not, you made a transient copy. Find it.
6. **bf16 is fp32 with the mantissa cut, fp16 is fp32 with the range cut.** On Ampere and later, load bf16 unless you have a specific reason; fp16 saturates at 65,504 and the damage from a silent bf16-to-fp16 cast does not raise.
7. **Convert the checkpoint on disk once.** Serving the dtype you ship halves the bytes read, removes every cast, and costs you ten lines and one run.
8. **Offload cost is bytes-per-token over PCIe bandwidth.** An embedding gather is 8 KB and nearly free; a decoder layer is 436 MB and costs 17 ms per token. Offload is a way to run a model that would not run, not a way to run one faster.
9. **Cold start is disk, then CUDA init, then the copy — in that order.** Optimize storage locality first; the PCIe transfer is the smallest of the three terms.
10. **Load strictly and cross-check loudly.** Missing tensors, transposed square matrices, untied heads and quantized checkpoints all fail silently. A strict loader plus a config-versus-weights check turns a two-day debugging session into a stack trace.

## Further reading

- [safetensors format and design](https://huggingface.co/docs/safetensors/index) — the specification, the security rationale, and the API.
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — metadata layout and the quantized block types llama.cpp uses.
- [PyTorch `torch.load` documentation](https://pytorch.org/docs/stable/generated/torch.load.html) — `weights_only`, `mmap`, and the map-location semantics.
- [Accelerate: big model inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) — meta-device construction, `device_map`, and disk offload, as the reference implementation of strategy C.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — the E4M3 and E5M2 encodings and the scaling they require.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — where the memory left over after loading actually goes.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the layer map this post's file sits in.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone, where `nanoserve` is assembled and measured against a real engine.

Next in the series: we build the module those weights are loading into. A Llama-3 forward pass in pure PyTorch — RMSNorm, RoPE, grouped-query attention, SwiGLU — with a logits-equivalence harness against `transformers` that catches every silent failure listed in section 8 before it reaches a user.
