---
title: "Dequant-fused GEMM: int4 weights on the fly"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build the kernel that makes 4-bit weights actually fast: unpack packed int4 in registers, feed the tensor cores, and never write the dequantized weights back to HBM — then derive exactly where the win evaporates."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "cuda",
    "triton",
    "gpu",
    "pytorch",
    "ml-systems",
    "throughput",
    "latency",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 46
---

Someone hands you a 4-bit checkpoint of Llama-3.1-8B and tells you it will make decode "4× faster." You load it, you run it, and it is exactly as fast as the bf16 model. Not slower — the memory footprint really did drop from 15 GiB to 4 GiB, and now the KV cache has room to breathe — but the tokens-per-second needle did not move. You profile a decode step and find the culprit immediately: before every matmul, a kernel reads the 4-bit weights, expands them to fp16, and writes 15 GiB of fp16 weights back to HBM. Then the matmul reads those 15 GiB right back. You quantized the storage and threw away the entire speedup at the door.

That is the trap this post is about, and the kernel that avoids it. The premise, which we established in [the skinny-matrix post](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem), is that batch-1 decode is **memory-bandwidth bound**, and the bytes it moves are overwhelmingly **weights** — the whole model, read once per token. So 4-bit quantization is a *bandwidth* win: you move roughly 4× fewer weight bytes per token, and because decode time is set by bytes-over-bandwidth, you go up to roughly 4× faster. It is emphatically **not** a FLOP win — the actual multiply still happens in fp16, at fp16 rates. Weight-only quantization speeds up decode because you *read less*, not because you *compute less*. Hold onto that sentence; half the confusion in this area comes from forgetting it.

![Side-by-side flow contrasting a naive two-step that writes and re-reads fp16 weights in HBM against a fused kernel that unpacks int4 in registers](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-1.webp)

The kernel that captures the win is the one that **never writes the dequantized weights to HBM**. It loads the packed 4-bit weights (4× less traffic), unpacks them to fp16 *inside registers and shared memory* with a shift, a mask, and a scale, and feeds them straight into the tensor cores — the fp16 values live for microseconds and die in the register file. This is the idea behind Marlin and its Hopper successor Machete, and this post builds a working version of it in Triton, on top of the GEMV we wrote last time. By the end you will have `nanoserve/kernels/dequant_gemm.py`: a dequant-fused int4 GEMV, tested against a dequantize-then-matmul reference, autotuned, and benchmarked with a script you can run. You will also be able to derive, from a roofline, the exact batch size where this kernel stops helping and starts *hurting* — the part most treatments skip.

One standing promise from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as something you will reproduce yourself with a named script and an expected range. The results tables carry a `Source` column. The bandwidth arithmetic is the easy case — it is division — so a script that measures achieved bandwidth gives you the same ratios I derive. End-to-end tokens-per-second is the hard case, and those stay cited or reproduce-it-yourself.

---

## 1. Weight-only quantization is a bandwidth trick

Start from the decode roofline, because it is the whole argument. A batch-1 decode step does one thing that costs real time: it streams the entire weight matrix of the model through the arithmetic units, once, to produce one token's worth of activations. Everything else — the KV read, the activations, the sampler — is small by comparison. So to first order, the step time is:

$$
t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{HBM bandwidth}}
$$

This is the memory-bound regime, and [the roofline model post](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is where it comes from: when a kernel's arithmetic intensity (FLOPs per byte) sits below the machine's ridge point, the memory system is the bottleneck and the compute units idle waiting for bytes. A GEMV — matrix times a single vector — has an arithmetic intensity near 1, which is about as memory-bound as a kernel gets.

Put Llama-3.1-8B on the numbers. It is 8.03B parameters. In bf16 that is $8.03 \times 10^9 \times 2 = 16.06$ GB of weights, which is 14.96 GiB. On an RTX 4090, whose GDDR6X delivers 1008 GB/s (NVIDIA's [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), a memory-bandwidth figure), the floor on a decode step is:

$$
t_{\text{decode}} \approx \frac{16.06 \text{ GB}}{1008 \text{ GB/s}} = 15.9 \text{ ms} \;\Rightarrow\; 62.8 \text{ tok/s}
$$

That is a ceiling, ignoring the KV read and every overhead, and it is why an 8B model in bf16 lands somewhere in the 40–60 tok/s band on a 4090 at batch 1 — real overheads eat the gap between 62.8 and what you see. Now quantize the weights to 4 bits. A weight is half a byte, so the ideal footprint is $8.03 \times 10^9 \times 0.5 = 4.02$ GB. Group-wise quantization adds a small overhead — one fp16 scale shared by every group of 128 weights is $2/128 = 0.0156$ bytes per weight, about 0.125 extra bits — bringing the real footprint to roughly 4.14 GB, or about 4.13 bits per weight. Re-run the floor:

$$
t_{\text{decode}} \approx \frac{4.14 \text{ GB}}{1008 \text{ GB/s}} = 4.11 \text{ ms} \;\Rightarrow\; 243 \text{ tok/s}
$$

The speedup is the byte ratio and nothing else: $16.06 / 4.14 = 3.88\times$. Notice what did *not* appear anywhere in that derivation: the number of FLOPs. The matmul does the identical amount of arithmetic in both cases — the dequantized weights are fp16, the activations are fp16, the multiply-accumulate is fp16. We did not make the GPU compute less. We made it *read less*, and reading is what the clock was spent on.

![Stacked view of what a single decode step reads, with fp16 weights dominating the byte budget over KV and activations](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-2.webp)

The ratio is also bandwidth-independent, which is a useful sanity check. On an A100 80GB SXM at 2039 GB/s ([NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)), the fp16 floor is $16.06/2039 = 7.9$ ms (127 tok/s) and the int4 floor is $4.14/2039 = 2.0$ ms (492 tok/s) — same 3.88× because both numerator and denominator moved together. The card sets the absolute speed; the quantization sets the ratio.

#### Worked example: how much of a decode step is weights?

Take one decode step of Llama-3.1-8B at batch 1 and 8k of context, and count the bytes actually moved:

- **Weights (bf16):** the whole model, 16.06 GB. Read once.
- **KV cache read:** attention reads all past keys and values. At 128 KiB per token ([derived in the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache)) and 8,000 tokens, that is $8000 \times 128 \text{ KiB} = 0.98$ GB.
- **Activations:** the residual stream is one vector of 4096 fp16 values per layer, a few hundred KB total across the step. Round it to a few MB.

Weights are 16.06 of roughly 17.04 GB moved — about 94%, and that fraction only climbs as context shrinks. *(Source: derived.)* This is why weight-only quantization is the highest-leverage byte you can cut for decode: it attacks the term that dominates. Cutting the KV bytes helps too, and [that is its own post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff), but at batch 1 the weights are the mountain.

---

## 2. The naive two-step, and why it is not free

Here is the implementation almost everyone writes first, because it reuses code they already trust. Keep the weights packed in HBM as int4 to save memory. Before each matmul, run a **dequantize kernel** that reads the int4 weights, expands them to fp16, and writes the fp16 weights to a scratch buffer in HBM. Then call your normal, battle-tested fp16 GEMM on that scratch buffer. Two kernels, two well-understood pieces, correct output. And no speedup at all — often a slowdown. Count the HBM traffic per matmul:

$$
\underbrace{4.14 \text{ GB}}_{\text{read int4}} + \underbrace{16.06 \text{ GB}}_{\text{write fp16}} + \underbrace{16.06 \text{ GB}}_{\text{read fp16 for GEMM}} = 36.26 \text{ GB}
$$

You moved **36.26 GB** to do a matmul that, in plain fp16, moves 16.06 GB. The naive two-step is $36.26 / 16.06 = 2.26\times$ *more* HBM traffic than not quantizing at all, so on the memory-bound decode path it is about 2.26× slower than the bf16 model you were trying to beat. You took a checkpoint that promised 4× and shipped something 2× *worse*, while congratulating yourself on the memory saving. *(Source: derived.)*

"But dequantize once and reuse it across all decode steps," someone always says. You can — and then you are storing 16.06 GB of fp16 weights resident in HBM, which means you gave back the entire memory saving that was the other reason to quantize, *and* every decode step still reads those 16.06 GB, so you are exactly as fast as bf16 and no faster. There is no arrangement of the two-step that wins. Either you re-expand every step (slower) or you cache the expansion (no faster, no smaller). The dequantized weights are the problem no matter when you produce them, because they are 4× larger than the thing you were trying to move less of.

The only way out is to never let the fp16 weights exist in HBM. Produce them, consume them, and discard them without the round trip — which means the dequantization has to happen *inside the matmul kernel*, in registers, between the load and the multiply. That fusion is not a nice-to-have optimization on top of the two-step; it is the entire mechanism. Everything else is bookkeeping.

The evidence that fusion is the whole game comes straight from the field. The vLLM team's [PTPC-FP8 on ROCm post](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) (2025-02-24) reports that their fused rowwise-scaled FP8 GEMM runs "up to 2.5× faster than a naive two-step unfused implementation" that dequantizes and then calls a standard GEMM — the same two-step we just costed, in a slightly different precision. When a production team measures a 2.5× gap between fused and unfused for the identical math, the lesson is unambiguous: the fusion, not the low precision, is where the speed lives.

---

## 3. The fused kernel: unpack in registers, feed the tensor cores

The fused kernel does the two-step's work in a different order and a different place. It loads the packed int4 weights from HBM — 4× fewer bytes, and this is the load that costs real time on the memory-bound path. It unpacks each weight from a nibble to an fp16 value using a shift, a mask, a subtract, and a multiply — a handful of arithmetic instructions that run on the integer/vector units while the tensor cores would otherwise be idle. And it feeds the reconstructed fp16 value straight into the multiply-accumulate. The dequantized weight never has an address in HBM; it lives in a register for the few cycles between reconstruction and consumption.

![Graph showing a masked nibble, a group scale, and a zero point merging into one fp16 value that flows into the tensor core and an fp32 accumulator](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-3.webp)

Three ingredients merge to reconstruct each weight. The **nibble** is the raw 4-bit integer, extracted with `(word >> (4*i)) & 0xF`, giving an unsigned value in 0–15. The **zero point** re-centers it: for symmetric int4 you subtract 8 to land in the signed range −8…7; for asymmetric schemes you subtract a per-group learned zero point instead. The **group scale** — one fp16 number shared by a run of 128 consecutive weights — turns the small integer back into a real magnitude. The reconstructed weight is:

$$
w = (\text{nibble} - z) \cdot s
$$

where $z$ is the zero point and $s$ the group scale. That is the arithmetic. The engineering is making it cost nothing, and the trick is *when* the ingredients arrive. The scale is read once per group of 128 weights, so its cost amortizes to almost nothing. The nibble arrives already in the register that held the packed word. And the whole reconstruction happens while the load of the *next* tile of packed weights is in flight, so the arithmetic hides behind memory latency instead of adding to it.

Watch the bytes move. The packed weights ride a lane out of HBM that is a quarter the width of the fp16 lane they replace; they expand to fp16 only after they arrive, inside the register file, and only the small fp16 tile — never the packed weights and never a write-back — flows onward to the tensor cores:

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Packed 4-bit weights stream out of HBM along a lane four times thinner than the fp16 lane, expand to fp16 inside the register file, and feed the tensor cores without ever being written back" style="width:100%;height:auto;max-width:900px">
<title>Packed int4 weights stream from HBM into registers, unpack to fp16, and feed the tensor cores; the int4 lane is four times thinner than the fp16 lane it replaces.</title>
<style>
.dqf-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.dqf-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dqf-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dqf-ghost{fill:var(--border,#d1d5db);opacity:.35}
.dqf-lane{fill:var(--accent,#6366f1);opacity:.18}
.dqf-dot{fill:var(--accent,#6366f1)}
.dqf-fp16{fill:var(--accent,#6366f1);opacity:.9}
@keyframes dqf-flow{0%{transform:translateX(0);opacity:0}8%{opacity:1}70%{opacity:1}100%{transform:translateX(250px);opacity:0}}
@keyframes dqf-out{0%,55%{transform:translateX(0);opacity:0}62%{opacity:1}94%{opacity:1}100%{transform:translateX(200px);opacity:0}}
@keyframes dqf-pop{0%,45%{opacity:0;transform:scale(.4)}60%{opacity:1;transform:scale(1)}90%{opacity:1}100%{opacity:0}}
.dqf-p{animation:dqf-flow 7s linear infinite}
.dqf-p2{animation-delay:2.3s}
.dqf-p3{animation-delay:4.6s}
.dqf-o{animation:dqf-out 7s linear infinite}
.dqf-o2{animation-delay:2.3s}
.dqf-o3{animation-delay:4.6s}
.dqf-e{animation:dqf-pop 7s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
.dqf-e2{animation-delay:2.3s}
.dqf-e3{animation-delay:4.6s}
@media (prefers-reduced-motion:reduce){.dqf-p,.dqf-o,.dqf-e{animation:none;opacity:1}}
</style>
<rect class="dqf-box" x="20" y="60" width="120" height="180" rx="10"/>
<text class="dqf-lbl" x="80" y="150">HBM</text>
<text class="dqf-sub" x="80" y="172">3.35 TB/s</text>
<rect class="dqf-ghost" x="140" y="90" width="200" height="48" rx="6"/>
<text class="dqf-sub" x="240" y="82">fp16 lane · 16 GB</text>
<rect class="dqf-lane" x="140" y="180" width="200" height="12" rx="6"/>
<text class="dqf-sub" x="240" y="212">int4 lane · 4.1 GB · 4x thinner</text>
<rect class="dqf-box" x="340" y="60" width="170" height="180" rx="10"/>
<text class="dqf-lbl" x="425" y="120">registers</text>
<text class="dqf-sub" x="425" y="142">unpack to fp16</text>
<rect class="dqf-box" x="600" y="80" width="140" height="140" rx="10"/>
<text class="dqf-lbl" x="670" y="145">tensor cores</text>
<text class="dqf-sub" x="670" y="167">fp16 x fp16</text>
<circle class="dqf-dot dqf-p" cx="150" cy="186" r="7"/>
<circle class="dqf-dot dqf-p dqf-p2" cx="150" cy="186" r="7"/>
<circle class="dqf-dot dqf-p dqf-p3" cx="150" cy="186" r="7"/>
<rect class="dqf-fp16 dqf-e" x="405" y="168" width="40" height="30" rx="5"/>
<rect class="dqf-fp16 dqf-e dqf-e2" x="405" y="168" width="40" height="30" rx="5"/>
<rect class="dqf-fp16 dqf-e dqf-e3" x="405" y="168" width="40" height="30" rx="5"/>
<circle class="dqf-dot dqf-o" cx="520" cy="150" r="7"/>
<circle class="dqf-dot dqf-o dqf-o2" cx="520" cy="150" r="7"/>
<circle class="dqf-dot dqf-o dqf-o3" cx="520" cy="150" r="7"/>
</svg>
<figcaption>Packed int4 weights ride a lane four times thinner than the fp16 lane, expand to fp16 inside the register file, and feed the tensor cores; the dequantized weights are never written back to HBM.</figcaption>
</figure>

There is one more layer of engineering that separates a toy fused kernel from Marlin, and it is worth naming because it is the reason hand-writing this well is hard. Tensor-core instructions (the `mma` and `wgmma` families) demand their operands in a very specific arrangement across the 32 threads of a warp — each thread must hold particular elements in particular registers. If you unpack the nibbles in their natural order, the values land in the wrong lanes and you pay for a flurry of register shuffles to move them into place, which can cost as much as you saved. Marlin's real contribution, per its [repository and technical write-up](https://github.com/IST-DASLab/marlin), is a **pre-permuted weight layout**: the int4 values are stored in HBM in exactly the scrambled order that makes them unpack directly into the register lanes the `mma` wants, no shuffles. Machete, vLLM's Hopper-generation W4A16 kernel, generalizes this on top of CUTLASS for the newer `wgmma` instructions. The lesson for our own kernel: we will get the fusion right and the bandwidth win real, and we will *not* match Marlin's layout artistry — that gap is exactly why you reach for the library in production, a theme we will close on.

---

## 4. The packing format: eight weights in a word, one scale per group

Before we can unpack, we have to know how the weights are packed, and the format is simpler than it sounds. A 4-bit value is a nibble. A 32-bit word holds eight nibbles. So you take eight consecutive weights along the contraction dimension and pack them into one `int32`, value $v_i$ occupying bits $[4i, 4i+3]$:

$$
\text{word} = \sum_{i=0}^{7} v_i \cdot 16^{i}
$$

![Grid of eight nibbles laid out inside a single 32-bit word, each labeled with its bit range](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-4.webp)

Concretely, `word = v0 | (v1 << 4) | (v2 << 8) | ... | (v7 << 28)`, and you recover value $i$ with `(word >> (4*i)) & 0xF`. Four bytes now carry what sixteen fp16 bytes would have — that 4× is the packing ratio, and it is why the load off HBM is 4× thinner. A weight matrix $W$ of shape $[N, K]$ (output features $N$, input features $K$) becomes a packed tensor of shape $[N, K/8]$ of `int32`, plus a scales tensor.

The scales are where the group size lives, and it is the one knob with real consequences. Group-wise quantization splits each row's $K$ elements into contiguous groups of $G$ (a common default is 128), and every group gets its own fp16 scale, so the scales tensor has shape $[N, K/G]$. Smaller groups track local weight statistics more tightly — better accuracy — but cost more scale storage and more scale lookups. Larger groups are cheaper but blunter. The overhead in bits per weight is $16/G$: at $G = 128$ that is 0.125 bits, negligible; at $G = 32$ it is 0.5 bits, which is a real 12% tax on top of the 4-bit payload. Section 9 stress-tests both ends. The *accuracy* consequences of group size — which values to pick, how much you lose, how to calibrate — are the subject of [the weight-only quantization post](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time), and I am deliberately not relitigating them here. This post is the kernel: given a packed checkpoint, make the matmul fast and correct. The numerics are Track F's job.

That division of labor is the honest framing. AWQ, GPTQ, and GGUF are *methods for choosing the quantized values* — which weights to round which way, how to protect the salient channels — and they all emit the same shape of artifact: packed nibbles plus group scales (and sometimes zero points). Intel's AutoRound, covered in the vLLM team's [AutoRound integration post](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) (2025-12-09), is a newer such method that learns the rounding by signed gradient descent and ships W4A16 with group-128 symmetric quantization, reporting Qwen3-8B at 0.911 on GSM8K. Symmetric means no zero point — the subtract-8 is implicit — which shaves the reconstruction to a shift, a mask, and a multiply. Our kernel below targets exactly that format, so it reads any group-128 symmetric W4A16 checkpoint those tools produce.

---

## 5. Building it in Triton

Now the code. We build in Triton because it lets us write the unpack-in-registers loop at a readable level while still compiling to a real GPU kernel; [the Triton post](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) makes the case for when this is the right tool and when to drop to CUDA. Everything here is a diff on `nanoserve`; the file is `nanoserve/kernels/dequant_gemm.py`.

Start with the reference path — quantize a weight matrix and pack it — so we have something correct to test against. This runs on CPU or GPU and is pure PyTorch:

```python
import torch

def quantize_int4_groupwise(w: torch.Tensor, group: int = 128):
    """Symmetric group-wise int4. w: [N, K] fp16/bf16.
    Returns q in [-8, 7] as int8, and per-group fp16 scales [N, K//group]."""
    N, K = w.shape
    assert K % group == 0, "K must be a multiple of the group size"
    wg = w.reshape(N, K // group, group).float()
    absmax = wg.abs().amax(dim=-1, keepdim=True)          # [N, K//group, 1]
    scale = (absmax / 7.0).clamp(min=1e-8)                 # int4 signed range is -8..7
    q = torch.clamp(torch.round(wg / scale), -8, 7).to(torch.int8)
    return q.reshape(N, K), scale.reshape(N, K // group).to(torch.float16)

def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: [N, K] int8 in [-8, 7]  ->  [N, K//8] int32, eight nibbles per word."""
    N, K = q.shape
    assert K % 8 == 0, "K must be a multiple of the pack width (8)"
    u = ((q.to(torch.int32) + 8) & 0xF).reshape(N, K // 8, 8)   # 0..15
    shifts = (torch.arange(8, device=q.device) * 4).to(torch.int32)
    return (u << shifts).sum(dim=-1).to(torch.int32)             # [N, K//8]
```

Two invariants are baked in and will matter later: `K` must be a multiple of the group size (so groups are whole) and a multiple of 8 (so words are whole). The `+8` is the symmetric zero point folded into the packing, so the stored nibbles are unsigned 0–15 and we subtract 8 back on the way out. Here is a dequantize-then-matmul reference — the naive two-step, in PyTorch, which we will use *only* as the correctness oracle, never as the fast path:

```python
def dequant_reference(packed: torch.Tensor, scale: torch.Tensor,
                      group: int, K: int) -> torch.Tensor:
    """The naive two-step, materialized. Returns fp32 weights [N, K]."""
    N = packed.shape[0]
    shifts = (torch.arange(8, device=packed.device) * 4)
    nib = (packed[:, :, None] >> shifts) & 0xF                  # [N, K//8, 8]
    w = nib.reshape(N, K).float() - 8.0                         # de-offset
    s = scale.float().repeat_interleave(group, dim=1)           # [N, K]
    return w * s
```

Now the fused kernel. This is the centrepiece, so read the inner loop closely — the unpack happens between the load and the accumulate, and no dequantized weight ever leaves the kernel. Because batch-1 decode is a GEMV (matrix times a single vector), there are no tensor cores to feed yet; the reduction *is* the GEMV, and the point to watch is that the packed weights are the only thing read from HBM:

```python
import triton
import triton.language as tl

@triton.jit
def dequant_gemv_kernel(
    x_ptr,            # [K]       fp16 activation vector
    qw_ptr,           # [N, K//8] int32 packed int4 weights
    scale_ptr,        # [N, K//G] fp16 group scales
    y_ptr,            # [N]       fp16 output
    N, K,
    GROUP: tl.constexpr,      # quant group size along K, e.g. 128
    BLOCK_N: tl.constexpr,    # output rows handled per program
    BLOCK_K: tl.constexpr,    # K elements per inner step (multiple of 8)
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)     # [BLOCK_N] output-row ids
    row_mask = rows < N

    K_packed = K // 8
    G_count = K // GROUP
    ppb = BLOCK_K // 8                               # packed words per K-tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)           # [BLOCK_K] absolute K indices
        k_ok = k_idx < K

        # 1) load the activation slice (fp16 -> fp32 for accumulation)
        x = tl.load(x_ptr + k_idx, mask=k_ok, other=0.0).to(tl.float32)   # [BLOCK_K]

        # 2) load ONLY the packed int4 weights: this is the 4x-thinner HBM read
        pk = (k0 // 8) + tl.arange(0, ppb)                                # [ppb]
        pk_ok = pk < K_packed
        qw = tl.load(
            qw_ptr + rows[:, None] * K_packed + pk[None, :],
            mask=row_mask[:, None] & pk_ok[None, :], other=0,
        )                                                                 # [BLOCK_N, ppb] int32

        # 3) UNPACK IN REGISTERS: eight nibbles per word, de-offset by 8
        shifts = tl.arange(0, 8) * 4                                      # [8]
        nib = (qw[:, :, None] >> shifts[None, None, :]) & 0xF             # [BLOCK_N, ppb, 8]
        w = nib.to(tl.float32) - 8.0
        w = tl.reshape(w, (BLOCK_N, BLOCK_K))                            # column order == K order

        # 4) group-scale lookup (once per group, broadcast across the tile)
        g_idx = k_idx // GROUP                                            # [BLOCK_K]
        s = tl.load(
            scale_ptr + rows[:, None] * G_count + g_idx[None, :],
            mask=row_mask[:, None] & k_ok[None, :], other=0.0,
        ).to(tl.float32)                                                  # [BLOCK_N, BLOCK_K]
        w = w * s                                                         # dequant, still in regs

        # 5) accumulate the GEMV; the fp16 weights die here, never written out
        acc += tl.sum(w * x[None, :], axis=1)                            # [BLOCK_N]

    tl.store(y_ptr + rows, acc.to(tl.float16), mask=row_mask)
```

The five steps map one-to-one onto the mechanism from section 3. Step 2 is the only HBM read that scales with model size, and it reads packed bytes — a quarter of the fp16 traffic. Steps 3 and 4 are the register-resident reconstruction. Step 5 consumes the fp16 weights immediately. There is no scratch buffer, no write-back, no second kernel. A host wrapper picks the grid and, importantly, does *not* transpose or copy the packed weights — they stay exactly as they were loaded:

```python
def dequant_gemv(x, qw, scale, N, K, group=128, block_n=64, block_k=128):
    y = torch.empty(N, dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(N, block_n),)
    dequant_gemv_kernel[grid](
        x, qw, scale, y, N, K,
        GROUP=group, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return y
```

Correctness first, speed second — always in that order for a kernel, because a fast wrong answer wastes more of your life than a slow one. The test compares the fused kernel against the materialized reference on the *identical* packed weights, so the only permitted difference is floating-point accumulation order:

```python
torch.manual_seed(0)
N, K, group = 4096, 4096, 128
w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.float16)
q, scale = quantize_int4_groupwise(w, group)
packed = pack_int4(q)
x = torch.randn(K, device="cuda", dtype=torch.float16)

y_ref = dequant_reference(packed, scale, group, K) @ x.float()   # oracle
y = dequant_gemv(x, packed, scale, N, K, group).float()          # fused

rel = (y - y_ref).abs().max() / y_ref.abs().max().clamp(min=1e-6)
print(f"max relative error: {rel.item():.2e}")
# expect ~1e-3 to 1e-2 — fp16/fp32 accumulation noise, not a logic bug
```

If that prints something near $10^{-2}$ or smaller, the fusion is correct: the kernel computes the same dot products as dequantize-then-matmul, just without ever storing the fp16 weights. If it prints something near 1, the usual culprit is a nibble-order mismatch between `pack_int4` and the kernel's `reshape` — the packed axis and the unpacked axis must agree on which nibble is which K index, and getting that backwards silently scrambles the weights. That is the single most common bug in a dequant kernel, and the reference test is how you catch it in seconds instead of discovering it as mysteriously-worse eval scores three days later.

Finally, autotuning. The best `BLOCK_N` and `BLOCK_K` depend on the matrix shape and the card, so let Triton search:

```python
configs = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w)
    for bn in (32, 64, 128) for bk in (64, 128, 256) for w in (2, 4, 8)
]

@triton.autotune(configs=configs, key=["N", "K"])
@triton.jit
def dequant_gemv_kernel(...):   # body unchanged from above
    ...
```

`triton.autotune` benchmarks each config once per new `(N, K)` and caches the winner, so the search cost is paid once at warmup, not per call. Which brings us to the honest way to measure this at all.

#### Worked example: measuring the kernel without lying to yourself

You want the *achieved bandwidth* of the fused kernel, because bandwidth — not tok/s — is the metric that tells you whether the fusion worked. The recipe, which mirrors [the reproducible-benchmark discipline](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark):

```python
def bench_gemv(fn, *args, warmup=25, iters=100):
    for _ in range(warmup):                 # let autotune settle + caches warm
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    bytes_moved = packed.numel() * 4 + scale.numel() * 2 + x.numel() * 2
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    print(f"{ms:.3f} ms/call   {gbps:.0f} GB/s achieved")
```

Warm up so autotuning and the caches settle; `torch.cuda.synchronize()` before and after because kernel launches are asynchronous and timing without it measures nothing; use CUDA events, not wall-clock; average over many iterations for steady state. On a 4090 whose ceiling is 1008 GB/s, a well-tuned dequant GEMV should reach a large fraction of that — call it 700–900 GB/s in the achievable band once launch overhead and imperfect coalescing are paid — and the fp16 GEMV baseline should reach a similar *fraction*, so the int4 kernel's advantage shows up as roughly 4× fewer bytes moved at comparable efficiency. Run it and report your GB/s; the ratio to the fp16 baseline is the honest speedup, and it is a division you can trust because you are not comparing against my hardware. *(Source: reproduce with `bench_gemv`; ceilings cited from the RTX 4090 and A100 specs above.)*

---

## 6. Where it stops paying: the crossover batch size

Everything so far assumed batch 1, where the GEMV is deeply memory-bound and the byte ratio is destiny. That assumption has a shelf life, and the most important skill in this whole area is knowing when it expires. As you batch more sequences together, the matmul stops being a GEMV (one vector) and becomes a GEMM (many vectors), and a GEMM does more arithmetic per byte of weight read — because the same weight, read once, now multiplies against $B$ different activation vectors. At some batch size the arithmetic, not the memory, becomes the bottleneck, and once you are compute-bound the whole premise of weight-only quantization collapses: reducing weight bytes cannot speed up a kernel that is waiting on the math.

Let me derive the crossover exactly, because it is a clean roofline argument. Take one weight matrix $W$ of shape $[N, K]$ and a batch of $B$ token vectors. The matmul does $2 B N K$ FLOPs. In fp16 it reads $2 N K$ bytes of weights (read once, reused across the batch). The kernel is memory-bound while the time to move the weights exceeds the time to do the math:

$$
\underbrace{\frac{2NK}{\text{BW}}}_{\text{weight read}} \;\gt\; \underbrace{\frac{2BNK}{F}}_{\text{compute}}
\quad\Longleftrightarrow\quad
B \;\lt\; \frac{F}{\text{BW}}
$$

The $N$ and $K$ cancel — the crossover does not depend on the matrix shape, only on the machine. The quantity $F/\text{BW}$ is the ridge point of the roofline, the machine's FLOPs-per-byte, and it is the **critical batch size** $B^{*}$. Below it, memory-bound: weight-only int4 helps. Above it, compute-bound: it cannot. On an A100, $F = 312$ TFLOP/s of bf16 tensor throughput and $\text{BW} = 2039$ GB/s, so:

$$
B^{*} = \frac{312 \times 10^{12}}{2039 \times 10^{9}} \approx 153
$$

On a 4090 ($165.2$ TFLOP/s dense fp16 tensor, 1008 GB/s) it is about 164; on an H100 ($989.5$ TFLOP/s bf16, 3.35 TB/s) about 295. These are the fp16 ridge points. Decode at batch 1 sits three orders of magnitude below them, which is why it is so memory-bound and why int4 helps so much there.

![Timeline of the int4 speedup across batch sizes, holding near four times then decaying to break-even and turning into a loss](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-5.webp)

But the int4 kernel has its own, *lower*, crossover, and this is the subtle part almost everyone misses. Because the int4 kernel reads only $0.5 N K$ bytes of weights, its memory time is 4× smaller, so it becomes compute-bound at a batch 4× smaller: around $B^{*}/4 \approx 38$ on an A100. Now trace the three regimes carefully, comparing the int4 fused kernel against a plain fp16 GEMM at each batch:

- **$B \lt 38$ (both memory-bound).** fp16 time is $2NK/\text{BW}$, int4 time is $0.5NK/\text{BW}$. The int4 kernel is a flat 3.8–4× faster. Full win, exactly as the decode derivation promised.
- **$38 \lt B \lt 153$ (int4 compute-bound, fp16 still memory-bound).** The int4 kernel's time is now $2BNK/F$ (the math), while fp16 is still $2NK/\text{BW}$ (the weight read). int4 is faster while $2BNK/F \lt 2NK/\text{BW}$, i.e. while $B \lt B^{*}$ — but the margin shrinks as $B^{*}/B$: about 2.4× at batch 64, 1.5× at batch 100, and exactly break-even at 153. The win is evaporating, and worse, the int4 kernel is now paying its dequant instructions on the *compute-bound* path where they can no longer hide behind memory latency.
- **$B \gt 153$ (both compute-bound).** Both kernels do the same $2BNK/F$ of tensor-core math, but the int4 kernel additionally runs the unpack instructions, which now add real time because the tensor cores are the bottleneck and the integer units are on the critical path. The int4 kernel is *slower than fp16 by the dequant overhead*. Weight-only quantization has become a net loss.

$$
t_{\text{int4}}(B \gt B^{*}) \approx \frac{2BNK}{F} + \varepsilon_{\text{dequant}}
\;\gt\;
t_{\text{fp16}}(B \gt B^{*}) \approx \frac{2BNK}{F}
$$

That is the honest boundary, and it is why "quantize the weights to 4-bit" is decode advice, not prefill advice. Prefill processes the whole prompt at once — hundreds or thousands of tokens — which is a large-batch GEMM sitting well above $B^{*}$, squarely in the regime where int4 does nothing good and something slightly bad.

#### Worked example: the batch-256 serving GEMM

You are serving Llama-3.1-8B on an A100 and continuous batching has assembled 256 sequences into one step — a healthy, well-utilized server. Each layer's projection is now a GEMM with $B = 256$, which is above the A100's $B^{*} = 153$. The GEMM is compute-bound. Your int4 kernel reads 4× fewer weight bytes, but the weight read is no longer the bottleneck — the tensor cores are saturated either way — so the byte saving buys nothing. Meanwhile the unpack instructions add, say, a few percent of overhead on the now-critical compute path. The int4 kernel runs at roughly 0.9× the fp16 GEMM: a small but real regression. If you deployed weight-only int4 to cut latency and then scaled the server up to healthy batch sizes, you would watch your per-step time get *worse* as load increased, which is a genuinely confusing thing to debug if you do not have this derivation in hand. *(Source: derived; the $B^{*}$ ridge points from the A100 and 4090 specs cited above.)*

The practical resolution most engines land on: use int4 weight-only for latency-first, low-batch deployments, and switch to a format with a genuine compute story — FP8 — when you are throughput-first and running large batches. Which is the next section.

---

## 7. int4 is bandwidth, fp8 is compute

Here is the distinction that ties the whole post together, and getting it wrong is how people ship the wrong quantization for their workload. **Weight-only int4 is a bandwidth technique.** It moves fewer weight bytes; it does the math in fp16 at fp16 rates. It helps exactly when and where bandwidth is the bottleneck — decode, small batch — and it hurts, mildly, where compute is the bottleneck. **Native FP8 is a compute technique** (that happens to also save bandwidth). On Hopper and Blackwell the tensor cores execute FP8 matmuls *natively*, at roughly double the fp16 rate — an H100 does 989.5 TFLOP/s of bf16 but 1978.9 TFLOP/s of FP8 (NVIDIA's [H100 datasheet](https://resources.nvidia.com/en-us-tensor-core), dense, no sparsity). So FP8 halves the weight bytes *and* halves the compute time, which moves both the memory ceiling and the compute ceiling. It keeps winning above $B^{*}$, precisely where int4 gives up.

![Matrix comparing int4 weight-only, native fp8, and fp16 across weight bytes, decode speed, large-batch speed, and the kind of win each delivers](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-6.webp)

The Blackwell generation pushes this further down to 4 bits with *native* FP4 tensor cores, which is a categorically different thing from our int4-unpacked-to-fp16 kernel: FP4 on Blackwell is a compute win at 4-bit, not a bandwidth-only win. The vLLM team's [Blackwell InferenceMAX post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) (2025-10-09) reports "up to 4× higher throughput at similar latency" versus Hopper on a B200 (192 GB HBM3e at 8 TB/s, native FP4), using FlashInfer's FP8 and FP4 GEMMs and fused AllReduce+RMSNorm+quant kernels. Read that number with its setup — it is a Pareto-frontier throughput claim across a full latency curve, not a single peak — but the mechanism is the point: when the tensor cores speak the low-precision format directly, low precision becomes a compute win that survives large batch. Weight-only int4 never becomes that, because the compute always happens in fp16.

| Approach | Weight bytes (8B) | Decode (B=1) | Large batch (B=256) | Kind of win | Source |
| --- | --- | --- | --- | --- | --- |
| fp16 baseline | 16.06 GB | 1.0× | 1.0× | reference | derived |
| int4 weight-only | 4.14 GB | up to ~3.8× | ~0.9× (loss) | bandwidth only | derived |
| FP8 (Hopper native) | 8.03 GB | ~2× | up to ~2× | compute + bandwidth | derived; H100 datasheet |
| FP4 (Blackwell native) | ~4 GB | large | large | compute + bandwidth | cited: InferenceMAX |

The decode column and the large-batch column tell the whole story in two numbers each. int4 is the biggest decode win and the only large-batch *loss* in the table. FP8 is a smaller decode win but holds under load. That is not a contradiction to resolve — it is a choice to make based on your batch regime, which is why the last section is a decision tree, not a winner.

The mechanics of choosing FP8 scales (per-tensor versus per-channel), the Hopper-versus-Blackwell support matrix, and where the advertised 2× does and does not materialize are [the FP8/FP4 post's](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you) subject. Here I only need the roofline contrast: int4 moves the memory ceiling, FP8 moves both.

---

## 8. Case studies and real numbers

Four public results, each cited with its setup, to anchor the derivations above in things other people measured.

**PTPC-FP8, fused vs unfused (AMD MI300X).** The clearest evidence that fusion is the mechanism. The vLLM team's [PTPC-FP8 post](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm) reports their fused rowwise-scaled FP8 GEMM running up to 2.5× faster than a naive two-step that dequantizes and then calls a standard GEMM. The accuracy cost was small — Llama-3.1-8B Wikitext perplexity moved from 9.4281 (bf16) to 9.5093 (PTPC-FP8), a 0.86% change — and 70B throughput came in at 1.01× versus per-tensor FP8. The number to take is the 2.5×: it is the fused-versus-unfused gap for the same math, on real hardware, and it is why we refused to ship the two-step.

**AutoRound W4A16 packing and memory (Intel × vLLM).** For the format side, the [AutoRound integration post](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc) documents the exact artifact our kernel consumes — W4A16, group-128 symmetric — produced by learned rounding rather than round-to-nearest, with Qwen3-8B reaching 0.911 on GSM8K. The companion [vLLM-Omni AutoRound post](https://vllm.ai/blog/2026-06-02-vllm-omni-autoround) (2026-06-02) quantifies the memory side of the same format: up to 62% checkpoint reduction, Qwen3-Omni-30B dropping from 66 GB to 25 GB. That 62% is the storage win; the *speed* win is a separate thing that only the fused kernel delivers, and only at low batch.

**Marlin and Machete, the reference kernels.** Marlin ([IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)) is the mixed-precision W4A16 kernel that established the pre-permuted-layout technique and near-ideal small-batch speedups, and it is integrated into vLLM. Machete is vLLM's Hopper-generation successor, built on CUTLASS for the `wgmma` instructions. These are what our teaching kernel is a simplified stand-in for: they get the register-lane layout right so the unpacked nibbles feed the tensor cores without shuffles, and they pipeline the packed load against the compute. When you need this in production, you use them — the gap between our correct-but-plain kernel and Marlin's tuned layout is real, and section 10 is honest about it.

**Blackwell native FP4 (InferenceMAX).** The [InferenceMAX post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) is the contrast case: up to 4× higher throughput at similar latency versus Hopper on B200, driven by native FP4 tensor cores and FlashInfer's FP8/FP4 GEMMs. This is the compute-win-at-4-bit that weight-only int4 can never be, and it is the reason the frontier is moving toward hardware-native low precision rather than software dequantization.

| Result | Setup | Number | Source |
| --- | --- | --- | --- |
| Fused vs naive two-step | MI300X, FP8 rowwise GEMM | up to 2.5× faster | cited: PTPC-FP8 (2025-02-24) |
| W4A16 checkpoint shrink | Qwen3-Omni-30B, group 128 | 66 GB → 25 GB (~62%) | cited: vLLM-Omni AutoRound |
| AutoRound accuracy | Qwen3-8B W4A16, GSM8K | 0.911 | cited: AutoRound × LLMC |
| Blackwell FP4 throughput | B200 vs Hopper, Pareto | up to 4× at similar latency | cited: InferenceMAX (2025-10-09) |
| Decode speedup, 8B | 4090, weight-traffic ratio | up to 3.88× (bandwidth floor) | derived |
| Critical batch, A100 | ridge point F/BW | ~153 | derived |

---

## 9. Stress tests and failure modes

A kernel is only as good as its behavior at the edges. Push the fused GEMV on the cases that break naive implementations.

**Group size too small.** Drop the group from 128 to 32 and two things happen. The scale storage quadruples — $16/32 = 0.5$ bits per weight instead of 0.125 — so the effective footprint climbs from ~4.13 to ~4.5 bits and the bandwidth win shrinks proportionally. Worse, the inner loop now looks up a new scale every 32 elements instead of every 128, and if those scale loads are not perfectly cached they add HBM traffic on the very path you were trying to thin. There is a floor below which the scale bookkeeping dominates and the whole exercise stops saving bytes. Group 128 is the common default because it sits comfortably above that floor while keeping accuracy acceptable; the accuracy trade at the *large*-group end (per-channel, one scale per row) is where you start losing eval points, and that is [Track F's](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time) territory, not the kernel's.

**K not a multiple of the pack width.** Our `pack_int4` asserts `K % 8 == 0`, and `quantize_int4_groupwise` asserts `K % group == 0`. Real weight matrices usually satisfy both — hidden dimensions are chosen to be friendly — but not always, and a fused kernel that assumes alignment and gets a ragged `K` will read past the end of the packed buffer and either crash or, worse, read garbage into the high nibbles of the last word and quietly corrupt the output. The fix is the same one every tiled kernel uses: pad `K` up to the next multiple of `lcm(8, group)` with zeros before packing, and rely on the kernel's `k_idx < K` mask to zero the padding's contribution. The mask is already in the code above (`k_ok`); the discipline is to *never* ship a dequant kernel without it, because the failure is silent.

**Mixing int4 weights with an fp8 KV cache.** These live in different kernels and compose cleanly. The int4 dequant happens inside the projection GEMMs; the fp8 KV cache lives in the attention kernel, which reads and writes the cache in fp8 ([the KV-quantization post](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) covers that path). One attacks the weight traffic, the other attacks the KV traffic, and on the memory-bound decode path both help because both cut bytes moved. The only thing to watch is that they are *independent* accuracy risks that stack: 4-bit weights lose a little, fp8 KV loses a little, and you should evaluate the combination, not each in isolation, because a model that survives either alone can still drift when you apply both. That evaluation is a numerics question; the kernels themselves do not interfere.

**The batch sweep from decode to prefill.** This is the stress test that matters most operationally, and it is the crossover from section 6 made concrete. Run the same int4 kernel at batch 1, 8, 32, 64, 128, and 256 and plot achieved speedup versus the fp16 baseline. You should see it hold near 3.8× through the low batches, start bending down somewhere past batch 30–40 as the kernel goes compute-bound, cross 1.0× near the ridge point in the low 100s, and dip below 1.0× beyond it. If your plot does *not* bend — if int4 stays 3.8× ahead at batch 256 — you have a bug or a mismeasurement, because the roofline forbids it: a compute-bound kernel cannot be sped up by reading fewer weight bytes. That plot is the single best diagnostic that you understand your own kernel, and it is a `reproduce`-class experiment: run it, and the *shape* of the curve (not my exact numbers) should match the derivation. *(Source: reproduce with the batch sweep; the ridge point is derived above.)*

---

## 10. When to reach for this (and when not to)

Weight-only int4 with a dequant-fused kernel is not a general "make the model faster" button. It is a precise tool for a precise regime, and the decision comes down to where your workload sits on the roofline.

![Decision tree mapping decode, prefill, and memory-fit regimes to int4 fused, fp8 native, and the large-batch loss to avoid](/imgs/blogs/dequant-fused-gemm-int4-weights-on-the-fly-7.webp)

**Reach for it when** you are latency-bound at low batch — interactive chat, a single user, agentic loops that decode long outputs one token at a time — and the weight traffic is your decode bottleneck. Reach for it when the model *does not otherwise fit*: dropping Llama-3.1-8B from 15 GiB to under 4 GiB is what lets it share a 24 GiB card with a serious KV budget, or lets a 70B model fit where it otherwise would not, and that memory win is real regardless of batch. In both of those cases int4 earns its place.

**Do not reach for it when** you are throughput-bound at large batch. Above the ridge point the fused kernel is a net loss, and you are better served by FP8 on Hopper or Blackwell, where the tensor cores give you a genuine compute win that survives load. Do not reach for a *hand-written* dequant kernel, in almost any case, when Marlin or Machete already exist and are wired into vLLM — they get the register-lane layout and the pipelining right in ways this post's teaching kernel deliberately does not, and matching them is weeks of work for a result someone already shipped. Our kernel exists so you *understand* the mechanism and can debug it, reason about its crossover, and know what the library is doing for you. That is worth a great deal; it is not worth re-deriving in production.

The build-versus-buy line for this piece of `nanoserve` is unusually clean. Build the understanding — the roofline, the fusion, the crossover — because without it you will deploy int4 to the wrong regime and be baffled when throughput drops under load. Buy the kernel — use Marlin through vLLM — because the last 2× over a plain fused kernel is layout artistry that is genuinely hard and genuinely done. The [capstone](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) benchmarks `nanoserve` against vLLM and puts a number on exactly this gap; this post is where you learn what the gap is made of.

---

## Key takeaways

- **Weight-only int4 is a bandwidth win, not a FLOP win.** Decode is memory-bound and dominated by weight traffic; 4-bit weights move ~4× fewer bytes, so decode goes up to ~3.8× faster while the math stays in fp16 at fp16 rates.
- **The naive two-step defeats the purpose.** Dequantizing to fp16 in HBM and calling a normal GEMM moves 2.26× *more* weight traffic than not quantizing at all. The fusion — unpack in registers, never write fp16 weights to HBM — is the entire source of the speedup, and PTPC-FP8's fused-vs-unfused 2.5× is the public proof.
- **The mechanism is a shift, a mask, a subtract, and a multiply, done in registers.** Eight int4 weights pack into one 32-bit word; one fp16 scale is shared per group of 128; the reconstructed weight lives for microseconds and feeds the tensor core directly.
- **The layout is the hard part.** Marlin's real contribution is a pre-permuted weight order that lands unpacked nibbles in the exact register lanes the `mma` wants, avoiding shuffles. That is why you use the library in production.
- **There is a crossover batch size, and it is the machine's ridge point $B^{*} = F/\text{BW}$** — about 153 on an A100. Below it int4 helps; above it int4 is a net loss because dequant instructions run on the compute-bound critical path. Weight-only int4 is decode advice, not prefill advice.
- **int4 moves the memory ceiling; FP8/FP4 move both ceilings.** Native low precision on Hopper/Blackwell is a compute win that survives large batch; weight-only int4 never becomes one.
- **Test against dequantize-then-matmul, measure achieved bandwidth, and always mask ragged `K`.** The nibble-order bug is silent and the alignment bug is silent; the reference test and the `k_idx < K` mask are how you keep them from reaching your eval scores.

---

## Further reading

- [GEMM for decode: the skinny-matrix problem](/blog/machine-learning/inference-engineering/gemm-for-decode-the-skinny-matrix-problem) — the memory-bound GEMV this kernel builds on, and where the critical batch size first appears.
- [Triton for inference kernels, and when to stop writing CUDA](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) — the maintenance argument for the tool we used here.
- [Weight-only quantization in your engine: GGUF, AWQ, GPTQ at load time](/blog/machine-learning/inference-engineering/weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time) — the numerics side: which method, how much accuracy you lose, how group size trades off.
- [FP8 and FP4 inference: what the hardware actually gives you](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you) — the compute-win formats, and where the advertised 2× does and does not materialize.
- [The roofline model: compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) and [the memory hierarchy: registers, shared memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — the two ideas the entire crossover argument rests on.
- [Marlin](https://github.com/IST-DASLab/marlin) and the vLLM team's [PTPC-FP8](https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm), [AutoRound](https://vllm.ai/blog/2025-12-09-intel-autoround-llmc), and [Blackwell InferenceMAX](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax) posts — the primary sources for every cited number above.
- [How quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) and [quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — background on the formats if you are new to them.
- The full arc: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
