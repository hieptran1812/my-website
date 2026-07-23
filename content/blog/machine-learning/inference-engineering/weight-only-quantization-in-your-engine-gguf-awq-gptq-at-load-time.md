---
title: "Weight-only quantization in your engine: GGUF, AWQ, and GPTQ at load time"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Open a quantized checkpoint, understand exactly what it packs, load its int4 weights and group scales into nanoserve, and derive the memory-and-concurrency win — without touching the accuracy question."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "gguf",
    "gptq",
    "awq",
    "pytorch",
    "gpu",
    "ml-systems",
    "kv-cache",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 47
---

Someone hands you a file called `Meta-Llama-3.1-8B-Instruct-GPTQ-Int4` and says "this one fits on your 4090, just load it." You point `nanoserve` at it, the loader you wrote in [the weights post](/blog/machine-learning/inference-engineering/loading-weights-safetensors-dtypes-and-device-placement) opens the safetensors file, and it finds tensors named `model.layers.0.self_attn.q_proj.qweight`, `.qzeros`, `.scales`, and `.g_idx` where it expected a single `weight` of shape 4096 by 4096 in bf16. The `qweight` is `int32` and a quarter the size it should be. There is no `weight` anywhere in the file. Your forward pass has no idea what to do with any of it, and neither, right now, do you.

That file is a quantized checkpoint, and this post is about what is actually inside it and how to get it into your engine correctly. Not the kernel that multiplies with it — [the previous post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) built that, the dequant-fused GEMM that unpacks 4-bit weights in registers and never writes fp16 back to HBM. This post is the layer underneath the kernel: the *format*. What a group of quantized weights encodes on disk, how GGUF and GPTQ and AWQ and compressed-tensors each pack it, how `nanoserve` parses those bytes into the packed weights and group scales the kernel expects, and — the payoff — exactly how much memory it saves and what that does to how many users you can serve.

![Diagram showing a group of full-precision weights collapsing into packed integers plus a shared scale and a zero point that a loader recombines](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-1.webp)

Figure 1 is the whole idea in one picture, and it is the thing to hold onto through every format that follows. A quantized weight is not a mysterious new number type. It is a small integer, plus a scale it shares with its neighbors, plus sometimes a zero point — and every format in this post is a different way of packing those same three things. Master the recombination once and the formats become bookkeeping.

One promise, carried from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and repeated because it is the rule that matters most here: **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as a range you will reproduce yourself with a named script. The memory math is the easy case — it is multiplication and subtraction — so you can check every figure on a napkin. The results table carries a `Source` column. And one scoping promise: this post is format, load, and memory *only*. How much accuracy you lose, which method loses the least, FP8 and FP4, and quantizing the KV cache instead of the weights are all real questions, and all of them belong to later posts I will forward-link at the end. Here we get the bytes in correctly and count the gigabytes. That is a full day's work on its own.

---

## 1. Why loading 4-bit weights correctly is worth a whole post

Start with why anyone quantizes weights at all, because the reason dictates everything about how you load them. In [the skinny-matrix post](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) we established that batch-1 decode is **memory-bandwidth bound**: the step time is set by how many bytes you drag out of HBM, not how many multiplies you do. And [the dequant-fused GEMM post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) showed that the bytes are overwhelmingly *weights* — the whole model, read once per generated token. So the decode floor is:

$$
t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{HBM bandwidth}}
$$

Cut the weight bytes by 4× and you cut the floor by 4×. That is the entire promise of weight-only quantization, and it is a *bandwidth* win, not a compute win — the multiply still happens in fp16, at fp16 rates. Put Llama-3.1-8B on the numbers: 8.03 billion parameters in bf16 is $8.03 \times 10^9 \times 2 = 16.06$ GB of weights. On an RTX 4090 at 1008 GB/s ([NVIDIA's RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)), the decode floor is 15.9 ms per token, about 63 tok/s. In 4-bit the weights are roughly 4.14 GB, so the floor drops to 4.1 ms, about 243 tok/s — the byte ratio and nothing else. *(Source: derived; the full derivation lives in the dequant-fused post.)*

But that speedup only materializes if the 4-bit weights reach the tensor cores as 4-bit weights. Load them wrong — expand them to fp16 in HBM at load time and keep them there — and you have thrown away the entire bandwidth win while keeping all the trouble. That is the trap the kernel post is about. This post is the trap *before* that one: getting the packed integers and their scales into the engine in the exact layout the kernel indexes, with the right group size, the right zero points, and the right per-layer precision, so that when the fused kernel reaches for weight number 5,000,001 it finds the correct nibble and the correct scale. A single off-by-one in how you unpack a group, or a forgotten act-order permutation, and the model does not get slower — it gets *wrong*, fluent-sounding garbage that passes a smoke test and fails a real eval. Loading is not the boring part. It is where the correctness lives.

There is a second, quieter win that is arguably bigger, and it is the reason this post ends on a memory table rather than a speed table. On a fixed card, the KV cache is a *residual* — whatever VRAM is left after the weights. Shrinking the weights does not just make the model fit; it hands the freed gigabytes to the cache, and because concurrency scales with cache, a 4× smaller model on a 24 GB card can serve two to three times as many users. We will derive that exactly in section 6. It is the single most under-appreciated consequence of quantization, and it has nothing to do with tokens per second.

---

## 2. The anatomy of a quantized weight

Here is the mechanism, built from first principles, because if you understand this section every format in the post is a variation on it. Take a contiguous run of $G$ full-precision weights — call $G$ the **group size**, a common value is 128. These weights live in some small numeric range, say $[-0.21, 0.19]$. You are going to represent each of them with a $b$-bit integer, where $b$ is typically 4. A 4-bit integer has 16 possible values, so you are quantizing a smooth range down to 16 levels.

The map from a real weight $w$ to its integer $q$ is an affine rounding:

$$
q = \operatorname{round}\!\left(\frac{w}{s}\right) + z
$$

where $s$ is the **scale** — the width of one quantization step — and $z$ is the **zero point**, the integer that represents $w = 0$. The scale and the zero point are shared by the whole group: one $s$ and one $z$ for all $G$ weights. To reconstruct the weight at inference, you invert the map:

$$
w \approx s \cdot (q - z)
$$

That formula is the entire contract between a quantized checkpoint and every engine that loads it. GGUF obeys it, GPTQ obeys it, AWQ obeys it, compressed-tensors obeys it. The differences are in how $s$, $z$, and $q$ are chosen and packed — never in the shape of the reconstruction. When Figure 1 shows three inputs merging into one value, this is the merge: a nibble, a scale, and a zero point become one fp16 weight.

There is a naming split worth pinning down now because both conventions ship in real files. **Symmetric** quantization fixes $z$ at the midpoint and drops it from the file: for signed int4 that means $z$ is implicitly 8 (or, equivalently, the levels run $-8 \ldots 7$ and you subtract 8), and the reconstruction collapses to $w \approx s \cdot (q - 8)$. It needs no per-group zero point at all, only a scale. **Asymmetric** (affine) quantization lets $z$ float per group so the 16 levels hug an off-center range tightly; it stores a zero point per group and reconstructs with the full $w \approx s \cdot (q - z)$. Symmetric is cheaper and slightly less accurate; asymmetric is the reverse. You will meet both, and a loader has to handle both.

Here is the round trip in code — quantize a group, pack it, and reconstruct it — so the formula stops being abstract. This is pure NumPy and it runs anywhere:

```python
import numpy as np

def quantize_group(w, bits=4, symmetric=True):
    """Quantize one group of fp32 weights to `bits` integers + scale (+ zero)."""
    qmax = (1 << bits) - 1            # 15 for 4-bit
    if symmetric:
        # levels centered on 8: subtract the midpoint on the way back out
        s = np.abs(w).max() / (qmax / 2)     # step covers [-max, +max]
        z = 1 << (bits - 1)                  # 8
        q = np.clip(np.round(w / s) + z, 0, qmax).astype(np.uint8)
    else:
        lo, hi = w.min(), w.max()
        s = (hi - lo) / qmax
        z = np.round(-lo / s).astype(np.uint8)   # integer that maps to 0.0
        q = np.clip(np.round(w / s) + z, 0, qmax).astype(np.uint8)
    return q, np.float16(s), np.uint8(z)

def dequantize_group(q, s, z):
    """The reconstruction every engine performs: w = s * (q - z)."""
    return (q.astype(np.float32) - float(z)) * float(s)

rng = np.random.default_rng(0)
w = (rng.standard_normal(128) * 0.06).astype(np.float32)   # one group of 128
q, s, z = quantize_group(w, bits=4, symmetric=False)
w_hat = dequantize_group(q, s, z)
print("q dtype/range :", q.dtype, q.min(), q.max())        # uint8 0..15
print("max abs error :", np.abs(w - w_hat).max())          # ~ s/2
print("rms error     :", np.sqrt(np.mean((w - w_hat)**2)))
```

The maximum error is about half a step, $s/2$, exactly as [the quantization-mechanism post derives](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) — the error is uniform on $[-s/2, s/2]$, with root-mean-square $s/\sqrt{12}$. And $s$ is proportional to the group's range. That single fact — *error is proportional to scale, scale is proportional to group range* — is the seed of every design decision in every format below. It is why groups exist. It is why AWQ scales salient channels. It is why GGUF quantizes its scales. Hold it.

### Deriving the effective bits per weight

Now the number that decides your memory budget: the *effective* bits per weight, which is more than $b$ because the scale and zero point cost storage too, amortized over the group. If a group of $G$ weights carries a payload of $b$ bits each, a scale of $b_s$ bits, and a zero point of $b_z$ bits, the effective bits per weight is:

$$
b_{\text{eff}} = b + \frac{b_s + b_z}{G}
$$

The overhead term $\tfrac{b_s + b_z}{G}$ is the whole story of the group-size trade-off. Plug in the common case — 4-bit payload, an fp16 scale (16 bits), symmetric so no zero point, group 128:

$$
b_{\text{eff}} = 4 + \frac{16}{128} = 4.125 \text{ bits}
$$

Make it asymmetric with a 4-bit packed zero point and it becomes $4 + \tfrac{16 + 4}{128} = 4.156$ bits. Either way, an "int4" model is really about 4.1 to 4.2 bits per weight once you count the metadata. That 0.125 to 0.156 of a bit is not rounding noise — across 8 billion weights it is the difference between a model that fits your card and one that does not, and it is the reason the memory table in section 6 uses 4.125, not 4.0.

---

## 3. Group size: the memory-versus-accuracy knob

Group size is the one dial in this whole area that you set consciously, so it deserves its own section. Figure 2 stacks the pieces: the 4-bit payload, the per-group scale, the per-group zero point, and how the total slides as the group grows.

![Stacked bars showing the four-bit payload plus per-group scale and zero-point overhead adding up to the effective bits per weight at three group sizes](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-2.webp)

The trade is a tug-of-war between two costs, both driven by the section 2 error law. **Small groups** track local weight statistics tightly: each group of 32 gets a scale tuned to its own narrow range, so the range is small, so $s$ is small, so the error is small. But small groups multiply the metadata — a scale every 32 weights instead of every 128 is four times as many scales. **Large groups** are the opposite: cheap metadata, but one scale now has to cover a wider range of weights, so $s$ is bigger and the error is bigger. The knob trades memory against accuracy, and there is no free setting. Where exactly the accuracy falls off — and it is not linear — is [a later post's job](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) to measure; here we quantify only the memory side, which is exact.

#### Worked example: effective bits at group 32, 128, and 256

Take symmetric int4 with an fp16 scale and no zero point, and walk the formula $b_{\text{eff}} = 4 + 16/G$ across three group sizes. Then convert to a footprint for Llama-3.1-8B's 8.03 billion weights, using $\text{bytes} = \text{params} \times b_{\text{eff}} / 8$:

| Group size $G$ | Scale overhead | Effective bits | Llama-3.1-8B footprint | Source |
| --- | --- | --- | --- | --- |
| 32 | ${16/32 = 0.500}$ | 4.500 bits | ${8.03\text{e}9 \times 4.5/8 = 4.52}$ GB | derived |
| 128 | ${16/128 = 0.125}$ | 4.125 bits | ${8.03\text{e}9 \times 4.125/8 = 4.14}$ GB | derived |
| 256 | ${16/256 = 0.062}$ | 4.062 bits | ${8.03\text{e}9 \times 4.062/8 = 4.08}$ GB | derived |

The span from group 32 to group 256 is 4.52 GB versus 4.08 GB — about 440 MB, roughly a 10% swing in the weight footprint, bought entirely by how often you store a scale. Group 128 is the near-universal default because it sits in the sweet spot: the overhead is already down to an eighth of a bit, and halving it again to group 256 saves only another 60 MB while measurably hurting accuracy on outlier-heavy layers. When a model card says "group size 128," this table is why. Here is the helper that produced it, which you will reuse in section 6:

```python
def effective_bits(payload_bits=4, scale_bits=16, zero_bits=0, group=128):
    return payload_bits + (scale_bits + zero_bits) / group

def model_bytes(num_params, payload_bits=4, scale_bits=16, zero_bits=0, group=128):
    beff = effective_bits(payload_bits, scale_bits, zero_bits, group)
    return num_params * beff / 8, beff

for g in (32, 128, 256):
    b, beff = model_bytes(8.03e9, group=g)
    print(f"group {g:3d}: {beff:.3f} bits/wt -> {b/1e9:.2f} GB")
# group  32: 4.500 bits/wt -> 4.52 GB
# group 128: 4.125 bits/wt -> 4.14 GB
# group 256: 4.062 bits/wt -> 4.08 GB
```

---

## 4. The four format families

Everyone reinvents this trio-plus-one, so it is worth a single map. GGUF, GPTQ, AWQ, and compressed-tensors are the formats you will actually meet on Hugging Face, and Figure 3 lines them up on the axes that matter to a loader: how the integers are chosen, roughly how many bits per weight they land at, whether the file needs any work before it can serve, and which engine reads it natively.

![Matrix comparing four quantization formats across core method, bits per weight, load-time work, and serving engine](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-3.webp)

The row that surprises people is the third column. All four formats pack essentially the same thing — int4 with group scales — and all four land within a whisker of 4.1 to 4.8 bits per weight. The real differentiator at load time is whether the file is *ready to serve as it sits on disk* or needs a transformation first. Three of the four are effectively load-ready; one, act-order GPTQ, needs a repack, and that is the load-time cost this post ends on. Everything else in this section is how each family fills in the "core method" cell — which is to say, how each one chooses the integers to minimize damage.

### 4.1 GGUF and the k-quants

GGUF is llama.cpp's format, and its defining idea is **the scales are quantized too**. [The mechanism post](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) gives the full byte-by-byte tour; here is the loader's-eye view, which is all `nanoserve` needs. A k-quant type like Q4_K groups 256 weights into a **super-block**, splits it into eight **sub-blocks** of 32, and — instead of storing eight fp16 scales — stores eight *6-bit* scales and eight 6-bit minimums, plus two fp16 "master" values shared across the whole super-block. Figure 4 walks the 144 bytes in the order the loader reads them.

![Timeline of the byte regions inside a Q4_K super-block from the two masters through the packed sub-scales to the nibble payload](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-4.webp)

The reconstruction for sub-block $j$ is a two-level affine — a master scale times a sub-scale, minus a master min times a sub-min:

$$
w \approx (d \cdot \mathrm{sc}_j)\, q \;-\; (d_{\min} \cdot \mathrm{m}_j)
$$

where $d$ and $d_{\min}$ are the fp16 masters, $\mathrm{sc}_j$ and $\mathrm{m}_j$ are the 6-bit sub-scale and sub-min, and $q$ is the 4-bit payload. Count the bytes: two masters (4 bytes), packed 6-bit scales and mins (12 bytes), and 256 nibbles ($256 \times 4 / 8 = 128$ bytes), totaling 144 bytes for 256 weights, or $144 \times 8 / 256 = 4.5$ bits per weight. That figure is cited straight from the ggml block layout (`ggml-quants.h`, the `block_q4_K` struct) and confirmed by division. The k-quant siblings scale the same idea: Q5_K is 176 bytes (5.5 bits), Q6_K is 210 bytes (6.5625 bits, with 8-bit sub-scales because at six payload bits the scale precision starts to matter), Q8_0 is a legacy 34-byte block of 32 weights (8.5 bits).

Two GGUF facts change how you load it. First, **`Q4_K_M` is not a format, it is a recipe.** It uses Q4_K for most tensors but promotes the value projection, the feed-forward down projection, and the output head to Q6_K, which is why its *effective* bits per weight is around 4.8, not 4.5 (per the [mechanism post](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded), which dumps the actual per-tensor types). Your loader cannot assume one type for the whole file; it must read the type tag per tensor. Second, that per-layer mixing means a single GGUF file can hold four or five different quant types across its tensors, and the loader dispatches a different unpacker for each. We stress-test exactly that in section 8.

### 4.2 GPTQ and the act-order permutation

GPTQ is the classic GPU post-training method, from Frantar et al.'s ["GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323) (2022). Its core method — the "core method" cell for its row in Figure 3 — is **layer-wise error minimization with second-order information**. Instead of rounding each weight to its nearest level independently, GPTQ quantizes a weight matrix one column at a time and, after fixing each column, nudges all the not-yet-quantized columns to compensate for the error just introduced, using the layer's Hessian ${H = 2XX^\top}$ (built from a small calibration set of activations $X$). The objective it minimizes is the layer's *output* error $\lVert WX - \hat{W}X \rVert^2$, not the per-weight error — which is why GPTQ beats naive round-to-nearest at the same bit width. That is a *calibration-time* story; by the time you load the file, the cleverness is baked into the integers and your job is only to unpack them.

What the loader sees is four tensors per linear layer: `qweight` (packed int4, `int32`), `scales` (fp16, one row per group), `qzeros` (packed int4 zero points), and — the one that bites — `g_idx`. That last tensor is the signature of **act-order** (also called `desc_act`): GPTQ found it quantizes better when it processes columns in order of decreasing importance (decreasing diagonal Hessian), so the columns end up *permuted*, and `g_idx` is the map from each input row back to its group. Figure 5 contrasts the two cases at load.

![Before-and-after comparison of loading an act-order GPTQ checkpoint that needs a row reorder versus a load-ready checkpoint that maps straight into the kernel](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-5.webp)

Without act-order, group boundaries are contiguous: rows 0 to 127 are group 0, 128 to 255 are group 1, and the kernel indexes scales trivially. With act-order, a weight's group is `g_idx[row]`, which is *not* monotonic — the loader (or the kernel) has to honor an indirection on every access, or, more commonly, **repack once at load** so the rows are back in contiguous group order and the fast kernel path applies. That one-time reorder is the load-time cost we quantify in section 7. It is cheap relative to a generation, but it is not free, and it is the reason "just mmap and go" is true for AWQ and false for act-order GPTQ.

### 4.3 AWQ and activation-aware scaling

AWQ, from Lin et al.'s ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) (MLSys 2024), starts from an observation the section 2 error law predicts: not all weights matter equally. A small fraction of weight channels — the ones multiplied by large-magnitude activations — dominate the layer's output, and quantizing *those* coarsely does most of the damage. The obvious fix is to keep the salient channels in fp16 and quantize the rest, but mixed-precision weights are a nightmare for GPU kernels. AWQ's trick is to protect the salient channels *without* mixed precision: **scale the salient weight channels up before quantizing** (and fold the reciprocal into the activations), so those channels occupy more of the integer range and suffer proportionally less rounding error. The per-input-channel scale is chosen by a small grid search that minimizes output error, guided by activation magnitude.

For the loader, the beautiful thing is that all of AWQ's intelligence is, again, baked in before you ever see the file. The scaling is already folded into the stored int4 weights and their group scales; the reciprocal is already folded into the surrounding layers. What lands on disk is a plain group-wise int4 checkpoint — `qweight`, `qzeros`, `scales`, group size usually 128 — with *no* `g_idx`, because AWQ does not permute. So AWQ is load-ready in the Figure 3 sense: mmap it, hand the packed weights and scales to the kernel, serve. The activation-awareness is a quality property of the numbers, invisible to the byte layout. That is a recurring theme worth stating plainly: **the method's cleverness is a calibration-time property; the loader only ever deals with integers, scales, and zero points.**

### 4.4 compressed-tensors and AutoRound

The fourth family is the one vLLM reaches for natively: **compressed-tensors**, the format produced by LLM Compressor. It is a safetensors file plus a quantization config that names the scheme (weight bits, group size, symmetric or not, strategy), and it is deliberately a *container* — it can hold weight-only int4, int8, FP8, and mixed schemes under one loader. One of the more interesting things you can put in it is a checkpoint produced by **AutoRound**. Per the vLLM team's ["AutoRound x LLM Compressor" post](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) (2025-12-09), AutoRound is post-training quantization by **signed gradient descent**: rather than round each weight to nearest, it learns three parameters per tensor — a rounding offset $V$ and a pair of clipping bounds $\alpha, \beta$ — to minimize the block's output reconstruction error, and reports that learned rounding beats round-to-nearest. It targets W4A16 (4-bit weights, 16-bit activations) and saves to compressed-tensors so it serves directly in vLLM. The one accuracy figure the post gives, cited exactly with its setup: Qwen3-8B quantized to W4A16 scores 0.911 on GSM8K.

For a loader, compressed-tensors is the friendliest of the four: the quantization config is explicit and self-describing, so you read the scheme from metadata instead of inferring it from tensor names, and the scale-and-integer layout is standardized across every method that targets it. AutoRound, GPTQ, AWQ-style recipes — they can all emit compressed-tensors, and your loader parses one format. That standardization is precisely why it is the vLLM-native path, and why "which format" in section 10's decision tree often resolves to "compressed-tensors if you are already on vLLM."

---

## 5. Loading a quantized checkpoint into nanoserve

Now the implementation. The goal is `nanoserve/quant/load.py`: parse a packed int4 checkpoint into the tensors the [dequant-fused kernel](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) consumes — packed weights, group scales, zero points — and verify a block round-trips by unpacking it to fp16 and comparing against a reference. We build it bottom-up: pack and unpack a group, then a whole GPTQ-style linear, then a Q4_K block, then wire it to the kernel.

### 5.1 Packing eight nibbles into a word

The GPU formats store eight int4 values in one `int32`, value $i$ in bits $[4i, 4i+3]$. Packing and unpacking are a shift and a mask — the same operation the kernel does in registers, done here in NumPy for verification:

```python
import numpy as np

def pack_int4(q):
    """q: uint8 array of 4-bit values, length multiple of 8 -> int32 words."""
    q = q.astype(np.uint32).reshape(-1, 8)
    word = np.zeros(q.shape[0], dtype=np.uint32)
    for i in range(8):
        word |= (q[:, i] & 0xF) << (4 * i)
    return word.astype(np.int32)

def unpack_int4(word):
    """int32 words -> uint8 4-bit values, 8 per word."""
    w = word.astype(np.uint32)
    out = np.empty((w.shape[0], 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = (w >> (4 * i)) & 0xF
    return out.reshape(-1)

q = np.random.randint(0, 16, size=64, dtype=np.uint8)
assert np.array_equal(unpack_int4(pack_int4(q)), q)   # exact round trip
print("packed 64 nibbles into", pack_int4(q).nbytes, "bytes")  # 32 bytes
```

Sixty-four 4-bit weights that would cost 128 bytes in fp16 now cost 32 bytes — the 4× that makes the HBM lane four times thinner. The round-trip assert is the kind of test that belongs in your loader's unit suite; a packing bug is silent until the model talks nonsense.

### 5.2 A whole GPTQ-style linear, unpacked and verified

Now assemble a realistic layer. A GPTQ checkpoint gives you `qweight`, `qzeros`, `scales`, and optionally `g_idx`. Here we build a small one end-to-end — quantize a random fp16 matrix group-wise, pack it into the GPTQ tensor shapes, then dequantize back and check the error is bounded by the group scale. This is the reference path your fused kernel must match:

```python
import numpy as np

def gptq_quantize_linear(W, group=128, bits=4):
    """W: (in_features, out_features) fp32. Returns GPTQ-style tensors."""
    in_f, out_f = W.shape
    assert in_f % group == 0, f"group {group} must divide in_features {in_f}"
    n_groups = in_f // group
    qmax = (1 << bits) - 1

    scales = np.empty((n_groups, out_f), np.float16)
    zeros  = np.empty((n_groups, out_f), np.uint8)
    qw     = np.empty((in_f, out_f), np.uint8)
    for g in range(n_groups):
        blk = W[g*group:(g+1)*group]              # (group, out_f)
        lo, hi = blk.min(0), blk.max(0)           # per output channel
        s = (hi - lo) / qmax
        z = np.round(-lo / s).astype(np.uint8)
        qw[g*group:(g+1)*group] = np.clip(np.round(blk / s) + z, 0, qmax)
        scales[g], zeros[g] = s.astype(np.float16), z
    # pack qweight along in_features (8 rows per int32)
    qweight = np.stack([pack_int4(qw[:, c]) for c in range(out_f)], axis=1)
    return qweight, scales, zeros

def gptq_dequantize_linear(qweight, scales, zeros, group=128, out_f=None):
    """Reconstruct fp32 W from packed GPTQ tensors: w = s*(q - z)."""
    out_f = out_f or qweight.shape[1]
    qw = np.stack([unpack_int4(qweight[:, c]) for c in range(out_f)], axis=1)
    in_f = qw.shape[0]
    W = np.empty((in_f, out_f), np.float32)
    for r in range(in_f):
        g = r // group
        W[r] = (qw[r].astype(np.float32) - zeros[g].astype(np.float32)) * scales[g].astype(np.float32)
    return W

rng = np.random.default_rng(1)
W = (rng.standard_normal((256, 64)) * 0.05).astype(np.float32)   # in=256, out=64
qweight, scales, zeros = gptq_quantize_linear(W, group=128)
print("qweight shape/dtype:", qweight.shape, qweight.dtype)       # (32, 64) int32
W_hat = gptq_dequantize_linear(qweight, scales, zeros, group=128)
err = np.abs(W - W_hat)
print("max |error|  :", err.max())                # bounded by max scale / 2
print("checkpoint MB:", (qweight.nbytes + scales.nbytes + zeros.nbytes) / 1e6)
```

`qweight` is `(32, 64)` — 256 input rows packed 8-to-a-word — where fp16 would be `(256, 64)`. The reconstruction obeys `w = s * (q - z)`, the exact formula from section 2, and the error is bounded by half a group scale. Real AutoGPTQ has a couple of packing quirks the toy skips — it packs `qzeros` as int4 too, and historically stored the zero point offset by one so you add one on unpack — and a production loader hard-codes those conventions per the format version. The *shape* of the work is what matters here: unpack nibbles, look up the group scale and zero, apply the affine.

### 5.3 Unpacking one Q4_K block

The GGUF path is different enough to show on its own, because of the two-level scales. Here is a loader that takes one 144-byte Q4_K block and reconstructs its 256 fp16 weights. It reads the two masters, expands the eight 6-bit sub-scales and sub-mins, and applies the two-level affine from section 4.1:

```python
import numpy as np

def dequantize_q4k_block(block: bytes):
    """One 144-byte Q4_K super-block -> 256 fp32 weights."""
    assert len(block) == 144
    d    = np.frombuffer(block, np.float16, count=1, offset=0)[0]     # master scale
    dmin = np.frombuffer(block, np.float16, count=1, offset=2)[0]     # master min
    sc_raw = np.frombuffer(block, np.uint8, count=12, offset=4)       # packed 6-bit
    qs   = np.frombuffer(block, np.uint8, count=128, offset=16)       # 256 nibbles

    # unpack eight 6-bit sub-scales and eight 6-bit sub-mins (llama.cpp bit layout)
    sc, mn = _unpack_q4k_scales(sc_raw)      # each length 8, values 0..63
    nib = np.empty(256, np.uint8)
    nib[0::2] = qs & 0xF
    nib[1::2] = qs >> 4

    out = np.empty(256, np.float32)
    for j in range(8):                       # eight sub-blocks of 32
        s_j = np.float32(d)    * np.float32(sc[j])
        m_j = np.float32(dmin) * np.float32(mn[j])
        seg = nib[j*32:(j+1)*32].astype(np.float32)
        out[j*32:(j+1)*32] = s_j * seg - m_j
    return out
```

The bit-fiddling inside `_unpack_q4k_scales` — how sixteen 6-bit values pack into 12 bytes — is the fiddly part, and rather than reproduce it I point you at the [mechanism post](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded), which lays out that exact layout. The point for `nanoserve` is the structure: a GGUF block is a small self-contained record, and the loader's job is to read masters, expand sub-scales, unpack nibbles, and apply `s_j * q - m_j`. Notice this is an *asymmetric, double-quantized* reconstruction — the scales themselves are quantized, which is the double-quant stress test we get to in section 8, already live in the format you use every day.

### 5.4 Wiring it to the kernel

The loader's output is a `QuantLinear` module that holds the packed weights and scales on the device and, at forward time, calls the fused kernel from [the dequant-fused GEMM post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — never materializing fp16 weights in HBM. Here is the nanoserve module, with a reference fallback so it runs (and stays testable) even where the Triton kernel is unavailable:

```python
import torch

class QuantLinear(torch.nn.Module):
    """Holds packed int4 weights + group scales/zeros; forwards via the fused kernel."""
    def __init__(self, qweight, scales, qzeros, group=128, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)   # int32 [in//8, out]
        self.register_buffer("scales", scales)     # fp16  [in//group, out]
        self.register_buffer("qzeros", qzeros)     # int32 [in//group, out//8]
        self.group = group
        self.bias = bias

    def forward(self, x):
        try:
            from nanoserve.kernels.dequant_gemm import dequant_int4_gemm
            # fused: unpacks int4 in registers, feeds tensor cores, no fp16 in HBM
            y = dequant_int4_gemm(x, self.qweight, self.scales, self.qzeros, self.group)
        except ImportError:
            # reference path: correct but expands to fp16 in HBM (the trap of post 27)
            W = _dequant_to_fp16(self.qweight, self.scales, self.qzeros, self.group)
            y = x @ W
        return y if self.bias is None else y + self.bias
```

The two branches are the whole lesson of Track F's kernel post in eight lines. The fused path keeps the int4 weights packed all the way to the tensor cores; the fallback expands them to fp16 in HBM and is *correct but slow* — the exact anti-pattern that turns a "4× faster" checkpoint into no speedup at all. Your loader produces the same tensors either way; which path runs decides whether the bandwidth win survives. That is why the kernel and the loader are two posts: the loader gets the bytes right, the kernel makes them fast, and you need both.

### 5.5 Verifying the load: logit parity, not vibes

A loader that "seems to work" is the most dangerous thing in this post, because the failure mode of a quant bug is not a crash — it is a model that talks fluently and is wrong in ways a smoke test never catches. The section 5.2 round-trip assert checks that *packing* is reversible, but it does not check that you unpacked the *real file's* conventions correctly: the zero-point offset, the act-order permutation, the double-quant master, the per-tensor precision. For that you need the reference the whole series leans on — the [from-scratch forward pass](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) run in fp16 — and a single number that says pass or fail.

The test is logit parity: run the same prompt through the quantized `nanoserve` model and through a trusted fp16 reference (the `transformers` model, or your own fp16 forward), and compare the logits. They will not be bit-identical — quantization *is* a lossy change — but a correctly loaded int4 model tracks the fp16 reference closely, while a mis-loaded one diverges immediately and wildly:

```python
import torch

def logit_parity(quant_model, ref_model, input_ids):
    """Compare quantized logits to an fp16 reference on one prompt."""
    with torch.inference_mode():
        lq = quant_model(input_ids).float()      # quantized path
        lr = ref_model(input_ids).float()        # fp16 reference
    max_abs = (lq - lr).abs().max().item()
    # top-1 agreement over the sequence is the sharpest single signal
    top1 = (lq.argmax(-1) == lr.argmax(-1)).float().mean().item()
    kl = torch.nn.functional.kl_div(
        lq.log_softmax(-1), lr.softmax(-1), reduction="batchmean").item()
    return max_abs, top1, kl
```

What you are checking is *relative* health, not an absolute threshold — so read it as a range you reproduce, never a number I measured. A well-loaded 4-bit checkpoint should show top-1 agreement in the high 0.9s on a plain prompt and a small KL divergence; a broken one shows top-1 near chance and a KL that explodes. The moment top-1 agreement collapses, you have a *loading* bug — a wrong zero point, a skipped `g_idx`, a single-level read of a double-quant scale — not an accuracy-of-quantization question. Keep this in your CI: a golden prompt, an fp16 reference cached once, and a top-1 floor. It is the cheapest insurance in the engine, and it is the test that separates "I support the format" from "I load the format correctly." Measuring *how much* quality the quantization itself costs — perplexity, task evals — is [a later post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff); this test only asks whether your loader is faithful to the file.

---

## 6. The memory math: before and after

This is the payoff section, and it is pure arithmetic you can check. Start from [the VRAM-budget framing](/blog/machine-learning/inference-engineering/loading-weights-safetensors-dtypes-and-device-placement) and [the KV-cache memory math](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache): on a fixed card, the KV cache is whatever VRAM is left after the weights, the activation working set, and the CUDA context take their cut. So shrinking the weights does two things at once — it makes the model fit, and it *enlarges the residual*. Watch it happen:

<figure class="blog-anim">
<svg viewBox="0 0 760 280" role="img" aria-label="A 24 gibibyte card budget bar alternates between an fp16 layout with a large weight region and a small key-value region and an int4 layout with a small weight region and a large key-value region holding many more users" style="width:100%;height:auto;max-width:900px">
<title>Quantizing Llama-3.1-8B from fp16 to int4 shrinks the weight region on a 24 GiB card from 14.96 GiB to 3.86 GiB, and the freed space becomes key-value budget that lifts capacity from 28 to 72 concurrent users.</title>
<style>
.wq-card{fill:none;stroke:var(--border,#d1d5db);stroke-width:2}
.wq-fix{fill:var(--border,#d1d5db);opacity:.55}
.wq-w{fill:var(--text-secondary,#6b7280)}
.wq-kv{fill:var(--accent,#6366f1);opacity:.9}
.wq-tick{stroke:var(--background,#ffffff);stroke-width:1.5;opacity:.6}
.wq-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.wq-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--background,#ffffff);text-anchor:middle}
.wq-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.wq-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes wq-fadeA{0%,42%{opacity:1}54%,96%{opacity:0}100%{opacity:1}}
@keyframes wq-fadeB{0%,42%{opacity:0}54%,96%{opacity:1}100%{opacity:0}}
.wq-A{animation:wq-fadeA 9s ease-in-out infinite}
.wq-B{animation:wq-fadeB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.wq-A{animation:none;opacity:1}.wq-B{animation:none;opacity:0}}
</style>
<text class="wq-hd" x="60" y="44">One RTX 4090 . 24 GiB . serving Llama-3.1-8B</text>
<rect class="wq-card" x="58" y="88" width="644" height="74" rx="8"/>
<rect class="wq-fix" x="60" y="90" width="53" height="70"/>
<text class="wq-sub" x="86" y="182">ctx+act 2 GiB</text>
<g class="wq-A">
<rect class="wq-w" x="113" y="90" width="399" height="70"/>
<rect class="wq-kv" x="512" y="90" width="188" height="70"/>
<text class="wq-lbl" x="312" y="131">fp16 weights 14.96 GiB</text>
<text class="wq-lbl" x="606" y="131">KV 7.04</text>
<text class="wq-cap" x="380" y="222">fp16: weights are the mountain . KV budget fits 28 users</text>
</g>
<g class="wq-B">
<rect class="wq-w" x="113" y="90" width="103" height="70"/>
<rect class="wq-kv" x="216" y="90" width="484" height="70"/>
<line class="wq-tick" x1="270" y1="90" x2="270" y2="160"/>
<line class="wq-tick" x1="324" y1="90" x2="324" y2="160"/>
<line class="wq-tick" x1="378" y1="90" x2="378" y2="160"/>
<line class="wq-tick" x1="432" y1="90" x2="432" y2="160"/>
<line class="wq-tick" x1="486" y1="90" x2="486" y2="160"/>
<line class="wq-tick" x1="540" y1="90" x2="540" y2="160"/>
<line class="wq-tick" x1="594" y1="90" x2="594" y2="160"/>
<line class="wq-tick" x1="648" y1="90" x2="648" y2="160"/>
<text class="wq-lbl" x="164" y="131">int4</text>
<text class="wq-lbl" x="458" y="131">KV budget 18.14 GiB</text>
<text class="wq-cap" x="380" y="222">int4: weights shrink to 3.86 GiB . the freed space fits 72 users</text>
</g>
</svg>
<figcaption>The KV budget is a residual: quantizing the weights to int4 frees 11 GiB on the same 24 GiB card, and that space becomes key-value cache for 2.6x more concurrent users.</figcaption>
</figure>

#### Worked example: VRAM and concurrency, fp16 vs int8 vs int4

![Table comparing fp16, int8, and int4 weights for Llama-3.1-8B on a 4090 across weight bytes, key-value budget, token capacity, and concurrency](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-6.webp)

Figure 6 is that comparison at a glance; the table below is the same numbers with their provenance. Do the subtraction for Llama-3.1-8B on a 24 GiB 4090, holding back 2 GiB for the CUDA context and the activation working set (the allowances derived in [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache)). The KV budget is the residual; capacity is the budget divided by 128 KiB per token — Llama-3.1-8B's cost, [derived there](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — and concurrency is capacity divided by a 2,048-token working length. The one GiB equals 8,192 tokens conversion falls straight out of $1\text{ GiB} / 128\text{ KiB} = 8192$.

| Precision | Weight bytes | KV budget | Token capacity | Concurrency | Source |
| --- | --- | --- | --- | --- | --- |
| fp16 | 14.96 GiB | 7.04 GiB | 57,671 | 28 users | derived |
| int8, group 128 | 7.48 GiB | 14.52 GiB | 118,948 | 58 users | derived |
| int4, group 128 | 3.86 GiB | 18.14 GiB | 148,603 | 72 users | derived |

Read the concurrency column, because it is the surprise. Quantizing the weights 4× — a footprint drop from 14.96 to 3.86 GiB — does not raise concurrency 4×; it raises it about 2.6×, from 28 users to 72. The reason is the residual structure: the KV budget went from 7.04 to 18.14 GiB, a 2.6× jump, not a 4× one, because the 2 GiB of fixed overhead is paid regardless and the card size is fixed. int8 is the quietly excellent middle: half the weight bytes doubles the KV budget and roughly doubles concurrency to 58, at a fraction of int4's accuracy risk. The lesson is that on a memory-constrained card **quantization is a concurrency lever at least as much as a fit lever** — you are not just squeezing the model in, you are buying back the cache that decides how many people it can talk to at once. Here is the calculation, so you never do it on a napkin:

```python
def concurrency(card_gib, weight_gib, overhead_gib=2.0,
                kv_kib_per_token=128, tokens_per_req=2048):
    kv_budget = card_gib - weight_gib - overhead_gib
    tokens = int(kv_budget * (1024*1024) / kv_kib_per_token)   # GiB -> tokens
    return kv_budget, tokens, tokens // tokens_per_req

for name, wgib in [("fp16", 14.96), ("int8", 7.48), ("int4", 3.86)]:
    kv, tok, users = concurrency(24.0, wgib)
    print(f"{name:4s}: KV {kv:5.2f} GiB  {tok:7d} tok  {users} users")
# fp16: KV  7.04 GiB    57671 tok  28 users
# int8: KV 14.52 GiB   118948 tok  58 users
# int4: KV 18.14 GiB   148603 tok  72 users
```

One honest caveat on the int4 weight figure: 3.86 GiB is the pure group-128 payload; a `Q4_K_M` GGUF at ~4.8 effective bits lands closer to 4.5 GiB because of its Q6_K promotions, and a checkpoint that leaves the embedding and head in fp16 (section 8) adds a few hundred MB more. The concurrency numbers shift by a user or two, not by the shape of the story. Use your own file's real byte count — `os.path.getsize` on the checkpoint is the least ambiguous input you have.

---

## 7. The load-time cost: repack-once versus ready-to-serve

Loading a quantized checkpoint is not always free, and the cost splits cleanly along the Figure 3 line. **Load-ready** formats — AWQ, non-act-order GPTQ, compressed-tensors, GGUF — need only what any checkpoint needs: read the packed tensors off disk (ideally mmap'd, so the OS pages them in lazily) and place them on the device. There is no transformation; the bytes on disk are the bytes the kernel indexes. **Repack-at-load** is the act-order GPTQ case: the `g_idx` permutation means the group boundaries are scrambled, and the common fix is to reorder the rows once at load so groups are contiguous again and the fast kernel path applies. The trade is a one-time cost paid at startup against a per-token cost paid forever — and paying once is obviously right, which is why loaders do the repack rather than carry the indirection into the hot loop.

How long does the repack take? I have not run it, so I will not invent a number — instead, here is the honest way to find yours. The reorder is a gather over the weight rows, dominated by moving the packed bytes, so it scales with the checkpoint size and with whether you do it on CPU or on the GPU after upload. Time it directly:

```python
import time, torch

def repack_actorder(qweight, g_idx):
    """One-time row reorder so groups are contiguous. Time it, don't guess it."""
    torch.cuda.synchronize() if qweight.is_cuda else None
    t0 = time.perf_counter()
    order = torch.argsort(g_idx)          # rows sorted by group
    qweight_sorted = qweight[order].contiguous()
    torch.cuda.synchronize() if qweight.is_cuda else None
    return qweight_sorted, time.perf_counter() - t0
```

Run this across your model's layers, sum the times, and report the total. As a framing for what you should expect rather than a claim I am making: this is a memory-bound copy of a few gigabytes, so it belongs in the same order of magnitude as reading the checkpoint off disk — plausibly a fraction of a second to a couple of seconds for an 8B model, dwarfed by the tens of seconds it takes to load and warm up the model in the first place. Measure it with the snippet, compare it to your end-to-end load time, and decide whether it is worth caring about. For most deployments the answer is no; the repack disappears into startup. The reason to know the mechanism is the debugging case: if an act-order model loads *wrong* — coherent but subtly dumber — a botched or skipped `g_idx` reorder is the first thing to check.

---

## 8. Stress tests: the four things that break a loader

A loader that handles the happy path and nothing else will ship, and then a real checkpoint will break it. Here are the four that will, each with the reason and the fix.

**A group size that does not divide the dimension.** Every unpacker above assumed `in_features % group == 0`. Real layers violate this — an intermediate dimension of 14,336 divides 128 cleanly (112 groups), but plenty of models have projection dimensions that do not, and a fused or head layer can be an awkward width. Quantizers handle it one of two ways: **pad** the final partial group up to a full group with zeros (and store a scale for it), or **require** the dimension be a multiple of the group and refuse otherwise — which is why you sometimes cannot quantize a model at group 128 but can at group 64. Your loader must know which convention the file used; if it pads, the packed tensor is slightly larger than `in_features / 8` words and you must not read past the real weights. The assert in section 5.2 is not defensive paranoia — it is the line that turns a silent mis-index into a clear error at load.

**A checkpoint whose scales are themselves quantized.** This is double-quantization, and you have already met it twice: GGUF's k-quants store 6-bit sub-scales under an fp16 master, and QLoRA-style NF4 checkpoints quantize the block scales to 8-bit under a second-level scale to claw back that last fraction of a bit. The loader consequence is that dequantization is *two levels deep* — you cannot read a scale directly; you read a quantized scale and a master and multiply them, as the Q4_K unpacker in section 5.3 does with `d * sc[j]`. A loader written for single-level scales will read the raw 6-bit integer as if it were the scale and produce weights off by a factor of the master. The symptom is a model that is not garbage but is *uniformly wrong* — every weight scaled by a constant — which is a peculiarly hard bug to see because the outputs are plausible.

**An int4 body with an fp16 head.** Open almost any real int4 checkpoint and you will find the token embedding and the `lm_head` output projection left in fp16 or bumped to a higher-precision quant (GGUF's `Q4_K_M` promotes `output.weight` to Q6_K; many GPTQ and AWQ recipes skip the head entirely). There are two good reasons, and your loader must expect the mix. First, the head's errors feed *directly* into the logits and therefore the sampling distribution — a small error there is a distorted probability over the whole vocabulary, whereas an error deep in the residual stream gets averaged out by everything downstream. Second, the embedding is a *lookup*, not a matmul on the bandwidth-critical decode path, so quantizing it buys you almost no decode speedup — it is memory you save at the cost of one specific token's representation being coarse everywhere it appears. The net: the head is high-value to protect and low-value to shrink, so quantizers leave it alone. A loader that assumes uniform precision will try to unpack an fp16 `lm_head` as int4 and fail immediately — read the per-tensor dtype, do not assume the file's headline bit width applies to every tensor.

**A GGUF file where every layer is a different quant type.** Because `Q4_K_M` is a recipe and not a format, a single GGUF file legitimately contains Q4_K, Q6_K, and Q8_0 tensors side by side, chosen per tensor role. The loader cannot read a global "this file is 4-bit" flag and dispatch one unpacker; it must read the type tag on each tensor and dispatch the matching one — Q4_K's 144-byte block unpacker for most, Q6_K's 210-byte unpacker for the promoted value and down projections, fp16 straight-through for anything left dense. This is why a GGUF loader is a *dispatch table keyed on tensor type*, not a single function, and why "I support Q4_K" is not the same as "I can load a Q4_K_M file." You support Q4_K_M when your table covers every type the recipe emits.

---

## 9. Case studies: public numbers with their sources

Four public results, each cited with its setup, to anchor the claims above in something you can go read.

**AutoRound's learned rounding, W4A16.** The vLLM team's ["AutoRound x LLM Compressor" post](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) (2025-12-09) reports that AutoRound quantizes by signed gradient descent — learning a rounding offset $V$ and clipping bounds $\alpha, \beta$ per tensor to minimize block reconstruction error — and lands **Qwen3-8B at W4A16 with a GSM8K score of 0.911**, saved as compressed-tensors and served directly in vLLM. That is a weight-only 4-bit checkpoint holding accuracy on a reasoning benchmark; the post is thin on speed and gives no head-to-head against AWQ or GPTQ, so treat the 0.911 as the one hard number it offers, with its exact setup.

**AutoRound's memory reduction on a larger, multimodal model.** The follow-up ["AutoRound for vLLM-Omni" post](https://vllm.ai/blog/2026-06-02-vllm-omni-autoround) (2026-06-02) uses W4A16 weight-only at group 128, symmetric, and reports **up to a 62% checkpoint reduction** — Qwen3-Omni-30B dropping from 66 GB to 25 GB, and FLUX.1-dev from 23 GB to 7 GB. Those ratios are exactly what section 2's arithmetic predicts: 16-bit down to a bit over 4 effective bits is very close to a 4× shrink, and 62% off is the same 2.6× we saw in the concurrency table, from the other direction. Cited, not measured by me.

**GPTQ's original result.** [Frantar et al.](https://arxiv.org/abs/2210.17323) (2022) introduced the layer-wise Hessian method and showed 3-to-4-bit quantization of models up to 175B with, in their words, negligible accuracy degradation relative to the fp16 baseline on the benchmarks they ran — the paper that made post-training int4 a serious option rather than a curiosity. Read it for the derivation of the column-by-column update; that is the "core method" cell in Figure 3 in full.

**AWQ's activation-aware scaling.** [Lin et al.](https://arxiv.org/abs/2306.00978) (MLSys 2024) is the source for the salient-channel idea and the per-channel scaling search that protects them without mixed precision. Its central empirical claim is that scaling a small fraction of weight channels before quantizing recovers most of the accuracy that naive int4 loses on the models they evaluate. It is worth reading alongside GPTQ because the two make *different* bets — GPTQ optimizes the rounding, AWQ optimizes which weights get the range — and both ship in formats your loader will meet.

A fifth, adjacent result grounds why the *kernel* matters as much as the loader: the vLLM team's [PTPC-FP8 on ROCm post](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) (2025-02-24) reports a fused rowwise-scaled FP8 GEMM running up to 2.5× faster than a naive two-step that dequantizes and then calls a standard GEMM. Different precision, same lesson as [post 27](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly): loading the bytes right is necessary but not sufficient — the kernel has to consume them fused, or the win evaporates.

---

## 10. When to reach for this (and when not)

A weight-only quantized checkpoint is the right default in a specific and common situation, and the wrong tool in a few others. Figure 7 is the decision, keyed on the engine you are targeting more than the model itself.

![Decision tree selecting GGUF for llama.cpp targets and AWQ, GPTQ, or compressed-tensors for GPU-server targets](/imgs/blogs/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time-7.webp)

**Reach for it when the model does not fit, or when you are memory-constrained and decode-bound.** If Llama-3.1-8B in fp16 leaves you 7 GiB of KV budget on a 4090 and you need to serve more than the resulting 28 users, int4 or int8 buys the concurrency directly, as section 6 showed. If the model is a hair too big for the card in fp16, quantization is the difference between running it and not. And because decode is bandwidth-bound, the weight-byte cut is a real latency win at small batch — the premise this whole track is built on.

**Pick the format by the target engine.** Serving through llama.cpp on a mix of CPU and GPU, or on a Mac: GGUF, and `Q4_K_M` is the sane starting recipe. Serving on a GPU through vLLM or a Hugging Face pipeline: AWQ or GPTQ, with a slight edge to AWQ for being load-ready and to GPTQ where a good act-order checkpoint exists — and accept the one-time repack. Already standardized on vLLM and quantizing your own models: compressed-tensors via LLM Compressor, so the config is explicit and one loader covers every scheme. These are the leaves of the tree, and the tree branches on the engine because the format's job is to be read by a specific kernel.

**Skip it — or think hard — in three cases.** First, if you are **compute-bound, not memory-bound**: at large batch, prefill-heavy, high-throughput serving, the GEMM is doing enough arithmetic that weight bytes stop being the bottleneck, and weight-only int4 can even *lose* to fp16 once the dequant overhead lands on a compute-bound path — [post 27 derived exactly that crossover](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly). Second, if you **have the VRAM to spare and the accuracy budget is tight**: fp16 or int8 removes a whole class of "is it quietly dumber?" risk, and int8 already doubles your concurrency. Third, and most important: if your production engine is vLLM or SGLang and they already load your format with a tuned Marlin or Machete kernel, **use theirs, not your own loader** — `nanoserve`'s value is that you now understand exactly what those loaders do, not that you should ship yours against them. The point of building it is the understanding; the point of the understanding is to debug the real one when it loads a checkpoint wrong.

---

## 11. What this post deliberately did not cover

I kept a tight scope, and the forward-links are the map of what got deferred, so you know where each question goes.

- **How much accuracy you lose, and which method loses the least.** This post quantified memory exactly and said nothing rigorous about quality on purpose. Measuring the damage — perplexity versus task evals, where the cliff is per model family, calibration sets — is [the accuracy-cliff post's job](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) and the activation-outlier post after it.
- **FP8 and FP4.** These are *hardware* number formats with native tensor-core support, not integer packing with software dequant, and the trade-offs (per-tensor versus per-channel scales, where the advertised 2× does and does not appear) are a different story told in the FP8/FP4 post next in this track.
- **Quantizing the KV cache instead of the weights.** Everything here shrank the *weights*. Shrinking the *cache* — to gain context length rather than fit the model — is the [KV-cache quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff), with its own per-channel-K-versus-per-token-V geometry and its own cliff.
- **The fused kernel that makes any of this fast.** That is [the dequant-fused GEMM post](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — the piece that consumes what this post loads. Load without it and you keep the memory saving but throw away the speed.

Where it all comes together is [the capstone playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), which benchmarks the whole of `nanoserve` — quantized loader included — against vLLM and reports the honest gap.

---

## Key takeaways

- **A quantized weight is an integer, a shared scale, and an optional zero point.** Reconstruction is always $w \approx s\,(q - z)$; every format is a different way of choosing and packing those three. Learn the recombination once.
- **Effective bits are payload plus overhead:** $b_{\text{eff}} = b + (b_s + b_z)/G$. int4 at group 128 is ~4.125 bits, not 4.0 — and that eighth of a bit sizes real gigabytes.
- **Group size trades memory against accuracy** through the error-proportional-to-scale law. Group 128 is the default because the overhead is already tiny and shrinking it further costs accuracy for almost no bytes.
- **The four families pack the same thing differently.** GGUF quantizes its scales and mixes types per layer; GPTQ minimizes layer output error with a Hessian and may permute (act-order); AWQ scales salient channels; compressed-tensors is the self-describing vLLM-native container (AutoRound targets it).
- **Loading is where correctness lives.** A mis-indexed group or a skipped act-order reorder does not slow the model — it makes it quietly wrong, coherent and dumber.
- **Only act-order GPTQ needs a load-time repack;** the rest are effectively ready to serve. Time the repack, do not guess it — it usually disappears into startup.
- **On a fixed card, quantization is a concurrency lever.** The KV budget is a residual, so cutting Llama-3.1-8B to int4 on a 4090 lifts concurrency about 2.6× (28 to 72 users), not 4× — and int8 already roughly doubles it at lower accuracy risk.
- **Expect a mixed-precision file:** an int4 body with an fp16 head and embedding, double-quantized scales, padded final groups. Read the per-tensor dtype; never assume the headline bit width applies everywhere.

## Further reading

- [Dequant-fused GEMM: int4 weights on the fly](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) — the kernel that consumes what this post loads, and where the 4× win is won or thrown away.
- [The memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) — the residual-budget derivation the concurrency table stands on.
- [KV-cache quantization: FP8, INT8, and the accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) — quantizing the cache instead of the weights, and where quality falls off.
- [How quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) — the byte-by-byte tour of GGUF blocks, scales, and the k-quant packing this post summarized.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022, the layer-wise Hessian method.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024, the salient-channel scaling idea.
- [AutoRound x LLM Compressor](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) — the vLLM team on signed-gradient-descent rounding into compressed-tensors (Qwen3-8B W4A16, GSM8K 0.911).
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the series frame and the capstone that benchmarks the whole engine.
