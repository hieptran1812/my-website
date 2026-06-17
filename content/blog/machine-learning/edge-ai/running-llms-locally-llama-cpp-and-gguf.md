---
title: "Running LLMs locally: llama.cpp, GGUF, and the local-inference stack"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take a 7B model from a cloud API key to your own laptop — understand why int4 plus a memory-bound decoder makes it feasible, walk the convert → quantize → serve flow in llama.cpp, and read the size/RAM/tokens-per-second tables that tell you which quant and backend to pick."
tags:
  [
    "edge-ai",
    "model-optimization",
    "llama-cpp",
    "gguf",
    "local-llm",
    "quantization",
    "inference",
    "efficient-ml",
    "on-device",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-1.png"
---

The first time I watched a 7-billion-parameter language model answer a question on my laptop — no GPU dedicated to it, no cloud, no API key, the Wi-Fi literally switched off — it felt like a small magic trick. I typed a prompt into a terminal, hit enter, and tokens started streaming back at a readable pace from a 4 GB file sitting on my SSD. The same model that, a year earlier, I had only ever reached through an HTTPS request to someone else's datacenter was now running on the machine in front of me, and it was *fast enough to use*. That moment is what `llama.cpp` made real for a lot of people, and the goal of this post is to take you all the way from "that's impossible on consumer hardware" to "here is exactly why it works and here are the commands."

This is a post about a whole stack, not a single trick. Three things have to line up for a 7B model to run usefully on a laptop. First, the weights have to shrink: full fp16 weights for a 7B model are about 13.5 GB, which won't fit comfortably alongside your browser and IDE, so we lean on **weight-only int4 quantization** (covered in depth in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)). Second, the workload has to be the *right kind* of workload for the hardware: token-by-token decoding is **memory-bandwidth-bound**, not compute-bound, which — counterintuitively — is what lets a laptop with no datacenter GPU keep up (this is the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) in action). Third, the software has to be lean enough to exploit both facts, which is what `llama.cpp` and its tensor library `ggml` are. Figure 1 contrasts the cloud-API path you probably started with against the local path we are about to build.

![A two-column comparison showing a cloud API call sending the prompt off device with a network round trip and metered cost versus a local llama.cpp run keeping the GGUF on disk with no network and tens of tokens per second](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-1.png)

By the end of this post you will be able to: take any Hugging Face checkpoint and convert it to the single-file **GGUF** format; quantize it to `Q4_K_M` (or pick a different k-quant on purpose); run it with `llama-cli` and `llama-server`; offload layers to a GPU with `-ngl`; measure real tokens per second and peak memory with `llama-bench`; and — most importantly — predict from first principles how fast a given model will run on a given machine *before* you download 13 GB. We will keep one running example, **Llama-2-7B**, as the spine, and we will tie everything back to the series' recurring frame: the four optimization levers (quantization, pruning, distillation, efficient architecture) sitting on a runtime, validated by profiling, read off the accuracy–efficiency Pareto frontier. Here the lever is quantization and the runtime is `llama.cpp`; see [the model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the map of all of them.

## Why a 7B model on a laptop is even possible

Let me start with the objection, because it is a good one. A 7B model in fp16 is about $7\times10^9 \times 2 = 14$ GB of weights. A forward pass through a transformer is dominated by matrix multiplications against those weights. Naively, that sounds like a job for a server-class accelerator with tens of teraFLOPs and hundreds of gigabytes per second of memory bandwidth. So why does it run at all on a laptop whose integrated GPU is a fraction of that?

The answer has two parts, and getting them straight is the whole foundation of local inference. The first part is that we do not run the model in fp16 — we run it in roughly 4 bits per weight, which cuts the 14 GB down to about 4 GB. That alone is what makes it *fit*. The second part is subtler and more important: token generation is **memory-bound**, so the thing that limits your speed is not how many FLOPs your chip can do but how fast it can *read the weights out of memory*. And consumer laptops, while they have modest compute, have surprisingly decent memory bandwidth — 50 to 100 GB/s on a modern machine, and Apple Silicon with unified memory pushes 100 to 400 GB/s. That bandwidth is what buys you usable tokens per second.

### The arithmetic-intensity argument, made concrete

Here is the science, derived rather than asserted. When you generate one token in the decode phase, the model reads every weight exactly once (you are doing a matrix-vector product, weight matrix times the single new token's hidden vector) and performs roughly two floating-point operations per weight (one multiply, one add). So for a model with $P$ parameters, generating one token costs:

$$\text{bytes read} \approx P \times b \quad\quad \text{FLOPs} \approx 2P$$

where $b$ is bytes per weight. The **arithmetic intensity** — FLOPs per byte, the quantity that decides whether you are compute-bound or memory-bound — is therefore:

$$I = \frac{2P}{P \times b} = \frac{2}{b}$$

For int4, $b = 0.5$ bytes per weight, so $I = 4$ FLOPs/byte. For fp16, $b = 2$, so $I = 1$ FLOP/byte. Both of those are *tiny*. A modern CPU or GPU has a ridge point (the arithmetic intensity above which you become compute-bound) of tens to hundreds of FLOPs per byte. We are nowhere near it. **Single-token decode is firmly in the memory-bound regime**, which means the roofline model predicts your throughput from bandwidth alone:

$$\text{tokens/s} \approx \frac{\text{memory bandwidth (bytes/s)}}{\text{bytes read per token}} = \frac{\text{BW}}{P \times b}$$

This single equation is the most useful thing in this entire post. Memorize it. It lets you predict speed before you download anything. See [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the full derivation of the ridge point and why low-FLOP models can still be slow.

#### Worked example: predict tokens/s before downloading

Suppose you have a 7B model quantized to `Q4_K_M`, which averages about 4.8 bits per weight (k-quants are not exactly 4 bits — more on that later). That is $b \approx 0.6$ bytes/weight, so the working set read per token is roughly $7\times10^9 \times 0.6 \approx 4.2$ GB. Now plug in three machines:

- **A laptop CPU at 60 GB/s effective bandwidth:** $60 / 4.2 \approx 14$ tokens/s. Slow but readable — about as fast as a person reads aloud.
- **An M3 MacBook with ~100 GB/s unified memory:** $100 / 4.2 \approx 24$ tokens/s in theory; in practice Metal gets you into the 40–55 tok/s range because the GPU also overlaps work and the effective bandwidth to the GPU cores is higher than the raw figure.
- **An RTX 4090 with ~1000 GB/s VRAM:** $1000 / 4.2 \approx 238$ tokens/s upper bound; real decode lands around 130–150 tok/s once you account for the KV-cache reads, attention, and kernel-launch overhead per step.

Notice the pattern: the ranking of the machines is set by their memory bandwidth, in the same order, every time. That is the memory-bound regime telling you what to optimize. If you want a 7B model to go faster on a fixed machine, the highest-leverage moves are (1) use fewer bits per weight (smaller $b$) and (2) use a backend that reaches faster memory. Throwing more *compute* at single-token decode does almost nothing.

There is one important caveat I will return to: the **prefill** phase (processing the prompt) is a different animal entirely. There you process many tokens at once, the matmuls become matrix-matrix products, arithmetic intensity shoots up, and you become compute-bound. So a long prompt can be slow to "digest" even on a machine that then decodes quickly. Hold that thought.

Before moving on, it's worth nailing down *why* the decode pass reads every weight exactly once, because that single fact is doing all the heavy lifting. A transformer layer's expensive operations are the four attention projections ($W_q$, $W_k$, $W_v$, $W_o$) and the two or three feed-forward matrices ($W_\text{gate}$, $W_\text{up}$, $W_\text{down}$ in a SwiGLU block). During decode you feed in a *single* new token, so its hidden state is a vector $x \in \mathbb{R}^{d}$, and each projection is a matrix-vector product $y = W x$ where $W \in \mathbb{R}^{m \times d}$. To compute $Wx$ the hardware must touch every one of the $m \times d$ entries of $W$ exactly once and multiply-add it against the corresponding entry of $x$. There is no reuse: each weight participates in exactly one multiply for this token. That is the structural reason arithmetic intensity is pinned at $2/b$ and cannot be improved by a cleverer kernel — the data-movement floor is physics, not implementation. The only knobs are $b$ (bits per weight) and the bandwidth of the memory the weights live in.

Contrast that with a *general* matrix-matrix multiply $C = AB$ where $A$ is $m \times k$ and $B$ is $k \times n$. That costs $2mnk$ FLOPs but only reads $mk + kn$ elements, so its arithmetic intensity is $\frac{2mnk}{mk + kn} \approx \frac{2 \cdot mn}{m + n}$, which grows with the matrix dimensions. Each weight in $A$ gets reused $n$ times — once per column of $B$ — so the data you fetched does $n$ units of work instead of one. This is exactly what prefill exploits: stack $T$ prompt tokens into a $T \times d$ activation matrix and the projection becomes a GEMM with reuse factor $T$. Push $T$ high enough and you climb above the ridge point into compute-bound territory. The whole game of efficient *serving* (batching many users together) is to manufacture this reuse artificially; the whole reality of *single-user chat* is that you cannot, because there is only one token in flight.

#### Worked example: deriving the M3's theoretical decode ceiling from first principles

Let me work the bandwidth-to-tokens derivation end to end on a concrete machine so the chain of reasoning is unbroken. Take an M3 Pro with a stated 150 GB/s of unified-memory bandwidth and a 7B model at `Q4_K_M` ($b \approx 0.6$ bytes/weight). Per decode step the engine must read the weights ($7\times10^9 \times 0.6 = 4.2$ GB) plus the KV-cache for attention at the current context. At a modest 2k tokens of context with multi-head attention the KV reads add roughly $2 \times 32 \times 32 \times 128 \times 2048 \times 2 \approx 1.07$ GB, so the total bytes moved per token is about $4.2 + 1.07 = 5.27$ GB. The bandwidth-bound ceiling is then $150 \times 10^9 / (5.27 \times 10^9) \approx 28$ tokens/s. Notice that the KV-cache term, often forgotten, just knocked roughly 20% off the naive weight-only estimate of $150 / 4.2 \approx 36$ tok/s. As context grows the KV term grows linearly and the gap widens — a 7B model that decodes at 36 tok/s on an empty context can sag toward 20 tok/s at 16k tokens *purely* because there are more KV bytes to stream every step. That sag is not a bug; it is the same bandwidth law applied to a working set that grew. When you see decode "slow down as the conversation gets long," this is the mechanism, and quantizing the KV-cache (`-ctk q8_0`) is the direct lever against it because it shrinks that second term.

### Why 4 bits is the magic number, not 8 or 2

The other half of feasibility is that we can drop to ~4 bits per weight *without the model getting noticeably dumber*. That is not obvious — naively, throwing away 12 of every 16 bits should be catastrophic. The reason it isn't is worth understanding, because it tells you exactly where the wall is.

Quantizing a weight tensor means mapping a continuous range of fp16 values onto a small grid of integer levels. With $b$ bits you have $2^b$ levels. The rounding error per weight is roughly uniform over half a quantization step, $\Delta/2$, where $\Delta$ is the spacing between levels. The classic result for the signal-to-quantization-noise ratio of a uniform quantizer is:

$$\text{SQNR} \approx 6.02\,b + 1.76 \ \text{dB}$$

Each extra bit buys you about 6 dB — a 4× reduction in error power. Read that the other way: going from 16 bits to 4 bits *increases* the quantization noise by about $12 \times 6 = 72$ dB. That sounds fatal, and for activations it often is. But weights are special. A transformer's matmul is a sum over thousands of weight-times-activation products. The per-weight rounding errors are roughly independent and zero-mean, so when you sum $N$ of them the error *averages out*: the error in the output of a dot product grows like $\sqrt{N}$ while the signal grows like $N$, so the *relative* error of the matmul output shrinks as $1/\sqrt{N}$. With $N$ in the thousands, a per-weight error that looks alarming becomes a tiny output perturbation. That averaging is the deep reason weight-only 4-bit quantization barely moves perplexity while 4-bit *activations* (which don't get the same averaging in the same place) are much harder. The full SQNR-and-error-variance derivation lives in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq); the takeaway here is the boundary.

So where is the wall? Around 3–4 bits. Above 4 bits the averaging keeps the model essentially intact (the +0.06 PPL you saw for `Q4_K_M`). Below 4 bits two things break the averaging assumption: the grid gets so coarse that a few large-magnitude **outlier weights** can't be represented at all without clamping (and outliers carry disproportionate signal), and the per-block scale itself starts to dominate the error budget. That is why `Q3_K` needs an importance matrix to stay usable and `Q2_K` visibly degrades — the clean averaging story falls apart. Four bits is the sweet spot because it sits just on the safe side of that cliff. This is the same "int4 is the practical floor for naive PTQ" boundary derived from a different angle in [the sub-8-bit post](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks).

## GGUF: the one file that holds everything

Before we run anything we need a file format, and `llama.cpp`'s format is **GGUF** (GPT-Generated Unified Format), the successor to the older GGML and GGJT formats. GGUF is deceptively simple and that simplicity is the point. It is a single, self-contained binary file that holds the model weights, all the metadata an engine needs to reconstruct the architecture, and the tokenizer — everything in one place, with no external config JSON, no separate tokenizer files, no Python environment required to load it. Figure 2 shows the layout.

![A vertical stack diagram of a GGUF file showing the magic header then a metadata key-value section then the tokenizer then a tensor index then alignment padding and finally the quantized tensor data blocks](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-2.png)

The file starts with a 4-byte magic number (`GGUF`) and a version. Then comes a **key-value metadata** section: arbitrary typed key-value pairs that describe the model. This is where the architecture lives — `llama.block_count`, `llama.attention.head_count`, `llama.context_length`, the RoPE settings, the feed-forward dimension, and so on. Because the metadata is general key-value, the same format describes Llama, Mistral, Qwen, Phi, Gemma, and dozens of other architectures; the loader reads the `general.architecture` key and dispatches to the right graph builder. The tokenizer is stored as metadata too — the vocabulary, the merge rules for BPE, the special-token IDs. After the metadata comes a **tensor index**: for each tensor, its name (like `blk.0.attn_q.weight`), its shape, its quantization type, and its byte offset into the data section. Then padding to align the data to a 32-byte boundary, and finally the raw **tensor data** — the quantized weight blocks themselves.

### Why memory-mapping matters so much

The layout exists to be **memory-mapped**, and that is one of the quiet reasons local inference feels so light. When `llama.cpp` opens a GGUF file it does not `read()` 4 GB into a buffer. It calls `mmap()`, which tells the OS "make this file appear in my address space." The OS then pages the weights in lazily, on demand, as the compute graph touches them, and — critically — it can share those pages across processes and reclaim them under memory pressure without thrashing. If you launch two `llama.cpp` processes pointed at the same GGUF, the OS backs them with the *same physical pages*. The model's resident memory is the page cache, which the kernel already knows how to manage. This is why a 4 GB model can start answering in well under a second: the first forward pass faults in only the pages it needs, and subsequent tokens reuse the page cache.

Memory-mapping is also why the "RAM needed" figures you see are slightly fuzzy. The *file* is 4.1 GB for `Q4_K_M`, but resident memory is file size plus the KV-cache plus the compute buffers plus a little overhead. On a 16 GB laptop you can comfortably run a 7B `Q4_K_M` model with a couple thousand tokens of context. We will size this precisely later.

### The k-quant types — a recap

GGUF stores weights in `llama.cpp`'s family of quantization types. The legacy types are simple: `Q4_0`, `Q8_0`, and so on — fixed-bit blocks with a per-block scale (and sometimes a min). The modern, default types are the **k-quants**: `Q4_K`, `Q5_K`, `Q6_K`. The "K" types use a smarter, hierarchical block structure — a super-block of 256 weights split into sub-blocks of 16 or 32, with the sub-block scales themselves quantized — so they pack more quality into the same bit budget than the legacy types. You will also see suffixes `_S`, `_M`, `_L` (small/medium/large), which control a *mixed* assignment: the attention and feed-forward weights that hurt accuracy most when squeezed get more bits, the rest get fewer.

`Q4_K_M` — the one everyone defaults to — is not uniformly 4 bits. It puts most weights in `Q4_K` but bumps the `attn_v` and `ffn_down` tensors (empirically the most sensitive) up to `Q6_K`, which is why it averages about 4.8 bits per weight rather than 4.0. The full reasoning behind why some tensors are more sensitive than others — and the SQNR math behind bit allocation — is in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq); for the GGUF-specific block layout and how each type is decoded, [how quantization works: GGUF quant types decoded](/blog/machine-learning/large-language-model/how-quantization-works-gguf-quant-types-decoded) goes block-by-block.

It is worth doing the bits-per-weight accounting explicitly, because once you see where the "extra" bits come from, the whole `_S`/`_M`/`_L` family stops being magic. Take `Q4_K`. A super-block holds 256 weights, split into 8 sub-blocks of 32. Each of those 256 weights gets a 4-bit quantized value, which is $256 \times 4 = 1024$ bits of payload. Then comes the overhead: each of the 8 sub-blocks needs a scale and a minimum (`Q4_K` is an *affine* quantizer storing both a scale $s$ and a min $m$ so each weight reconstructs as $w \approx s \cdot q + m$), and these per-sub-block scales and mins are themselves quantized to 6 bits and packed, costing roughly $8 \times (6 + 6) = 96$ bits, plus two fp16 super-block-level values ($d$ and $d_\text{min}$) at $2 \times 16 = 32$ bits. So one super-block is about $1024 + 96 + 32 = 1152$ bits for 256 weights, which is $1152 / 256 = 4.5$ bits per weight. That is the source of the "it's not really 4 bits" surprise: the metadata to *describe* the blocks is real and it never goes to zero. Smaller sub-blocks would lower quantization error (a tighter scale per group) but raise the overhead share; the super-block-of-256 with sub-blocks-of-32 is the empirically tuned balance.

Now layer the mixed assignment on top. `Q4_K_M` runs most tensors at this 4.5 bpw `Q4_K`, but promotes `attn_v.weight` and `ffn_down.weight` to `Q6_K` (about 6.6 bpw). Those two tensor types are a minority of the parameter count but carry outsized sensitivity, so spending ~2 extra bits on them and leaving everything else at 4.5 lifts the *whole-model* average to about 4.8 bpw while recovering most of the quality you'd lose from a flat 4.5-bit quant. `Q4_K_S` ("small") skips those promotions and stays near 4.5 bpw; `Q4_K_L` (where it exists) promotes even more tensors toward `Q6_K`/`Q8_0`. So the suffix is literally a bit-allocation policy, and the per-weight average you read in the table is the weighted mean of the per-tensor policies. This is the GGUF-native version of the same principle GPTQ and AWQ formalize: spend bits where the error costs you most. Here is the practical summary you can act on:

| Type     | Bits/weight (approx) | 7B file size | Quality vs fp16        | When to use                          |
| -------- | -------------------- | ------------ | ---------------------- | ------------------------------------ |
| `Q4_K_M` | ~4.8                 | 4.1 GB       | +0.06 PPL (near-loss)  | The default. Best size/quality.      |
| `Q5_K_M` | ~5.7                 | 4.8 GB       | +0.02 PPL              | When 700 MB more RAM is free.        |
| `Q6_K`   | ~6.6                 | 5.5 GB       | ~0.00 PPL (negligible) | Quality-sensitive, RAM available.    |
| `Q8_0`   | 8.5                  | 7.2 GB       | indistinguishable      | Reference / when you have the RAM.   |
| `Q3_K_M` | ~3.9                 | 3.3 GB       | +0.3–0.6 PPL (visible) | Tight RAM; accept some degradation.  |
| `Q2_K`   | ~3.0                 | 2.8 GB       | +1.0 PPL (rough)       | Last resort; quality clearly drops.  |

The "+PPL" column is the increase in perplexity over the fp16 baseline on a held-out corpus (lower is better, so a small positive number means a small quality loss). These figures are representative of the values the `llama.cpp` community has measured on 7B Llama-class models; treat them as approximate order-of-magnitude guidance, not exact constants — the precise numbers shift with model and corpus. The headline is that `Q4_K_M` loses almost nothing while halving the memory of `Q8_0`, which is why it is the universal default. Below 4 bits the loss becomes visible; for the science of *why* int4 is roughly the wall for naive PTQ and what it takes to go lower, see [the sub-8-bit post](/blog/machine-learning/edge-ai/sub-8-bit-int4-ternary-and-binary-networks).

## llama.cpp and ggml: the architecture under the hood

`llama.cpp` is, at its core, a thin model-definition layer on top of a small tensor library called **ggml**. Understanding the split clarifies everything else. `ggml` is the engine; `llama.cpp` is one application of it (`whisper.cpp`, `stable-diffusion.cpp`, and others are siblings). Let me describe how a forward pass actually happens.

When you load a GGUF, `llama.cpp` reads the metadata, figures out the architecture, and builds a **compute graph** in `ggml`. A `ggml` compute graph is a directed acyclic graph of tensor operations — matmul, add, RMS-norm, RoPE, softmax, SiLU — with the GGUF weights as the leaf tensors. The graph is *symbolic first*: you describe the operations and their dependencies, then a scheduler walks the graph and executes each node on a **backend**. This separation of graph from execution is exactly what lets the same model run on a CPU, an Apple GPU, an NVIDIA GPU, or a Vulkan device without changing the model code. Figure 5 shows the dispatch.

![A graph diagram showing the per-layer compute graph flowing into a backend scheduler controlled by the dash n g l flag which then dispatches layers to CPU with AVX2 or NEON, Metal on Apple GPUs, CUDA on NVIDIA GPUs, or Vulkan across vendors](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-5.png)

### The backends

A **backend** in `ggml` is an implementation of the tensor operations for a particular piece of hardware. The major ones:

- **CPU.** Hand-written SIMD kernels: AVX2 and AVX-512 on x86, NEON on ARM (including Apple Silicon and the Raspberry Pi). The int4/k-quant matmul kernels are the secret sauce — they dequantize a block of weights into registers and do the dot product in one fused pass so the weights are touched once. This is the universal fallback; it works everywhere.
- **Metal.** Apple's GPU API. On a Mac this is the big win because Apple Silicon has **unified memory** — the CPU and GPU share the same physical RAM, so there is no copy across a PCIe bus. The GPU reads the same mmap'd weights the CPU would, at the GPU's higher effective bandwidth.
- **CUDA.** NVIDIA GPUs. Fastest decode of all because of high-bandwidth VRAM, but the model (or the offloaded layers) must fit in VRAM, which is the binding constraint.
- **Vulkan.** A cross-vendor GPU backend that runs on AMD, Intel Arc, and NVIDIA. Slower than the native CUDA/Metal paths but a lifesaver for non-NVIDIA GPUs.
- **HIP/ROCm, SYCL, CANN** and others exist for AMD, Intel, and Huawei accelerators respectively.

The scheduler can split a single model *across* backends. That is what `-ngl` (number of GPU layers) controls.

### `-ngl`: offloading layers to the GPU

`-ngl N` tells `llama.cpp` to place the first `N` transformer layers (plus, if there's room, the output layer) on the GPU backend and leave the rest on the CPU. Setting `-ngl 999` (or `-ngl 99`, any number ≥ the layer count) offloads *everything*. This single flag is the most consequential performance knob on a machine with a discrete GPU and limited VRAM, because it lets you trade a fully-on-GPU run (fast, needs lots of VRAM) against a hybrid run (slower, fits in less VRAM). When a model is too big for VRAM, you offload as many layers as fit and the rest stay on CPU. The catch: a hybrid run is gated by the *slow* part. If 24 of 32 layers are on a fast GPU and 8 are on a slow CPU, your tokens/s is dragged toward the CPU's rate, because every token must pass through all 32 layers in sequence. Offloading is close to all-or-nothing for speed; partial offload mostly buys you the ability to run a model that otherwise wouldn't fit at all.

### Prefill vs decode: the two phases

I flagged this earlier and now it earns its own figure. Generation has two phases with opposite hardware profiles. Figure 4 contrasts them.

![A two-column comparison of prefill processing all prompt tokens at once with big GEMM matmuls and being compute-bound versus decode emitting one token per step by reading all weights every step and being memory-bound](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-4.png)

**Prefill** (also called the prompt phase) processes all the prompt tokens in one shot. Because there are many tokens, the per-layer operation is a matrix-*matrix* multiply (a GEMM), which has high arithmetic intensity and saturates the chip's FLOPs. Prefill is **compute-bound**, and its speed is reported as "prompt tokens per second" or "tokens to first token." A 512-token prompt on a fast GPU is digested in tens of milliseconds; on a CPU it can take a noticeable second or two.

**Decode** (the generation phase) emits one token at a time. Each step is a matrix-*vector* product, low arithmetic intensity, reading every weight once — exactly the memory-bound case we derived. Decode speed is the "decode tokens per second" figure and it is what determines how fast the answer streams.

Why does the distinction matter operationally? Because the two phases respond to different optimizations. To speed up prefill you want more compute (a GPU, more cores, bigger batch). To speed up decode you want more bandwidth and fewer bits. And it explains a confusing observation: a model with a huge system prompt can feel slow to *start* (long prefill) but then stream quickly (fast decode), or vice versa. If you only measure end-to-end latency you can't tell which phase to fix; measure them separately, which `llama-bench` does for you.

Let me make the prefill arithmetic intensity concrete so the "compute-bound" claim isn't hand-waved. During prefill of a $T$-token prompt, each weight matrix $W \in \mathbb{R}^{m \times d}$ is multiplied against the full $T \times d$ activation block, so the FLOPs are $2 \times T \times m \times d$ while the bytes read for that weight are still just $m \times d \times b$ (you read the weight once and reuse it across all $T$ tokens). The arithmetic intensity is therefore $\frac{2Tmd}{mdb} = \frac{2T}{b}$ — it scales *linearly with the prompt length*. For `Q4_K_M` ($b \approx 0.6$) a single-token decode sits at $I = 2/0.6 \approx 3.3$ FLOPs/byte, but a 512-token prefill sits at $I = 2 \times 512 / 0.6 \approx 1700$ FLOPs/byte. The ridge point of a typical GPU is in the tens-to-low-hundreds of FLOPs/byte, so decode ($3.3$) is far below it (memory-bound) while a 512-token prefill ($1700$) is far above it (compute-bound). The *same model, same weights, same hardware* lives on opposite sides of the roofline depending only on how many tokens are in flight. That is not a metaphor; it is the literal reason the two phases have opposite optimization recipes, and it is why batching (which raises the effective token count) is the master lever for server throughput while doing nothing for single-user decode.

### RAM, VRAM, and the KV-cache budget

Your memory budget has to hold three things: the **weights**, the **KV-cache**, and the **compute/activation buffers**. The weights we've covered. The KV-cache is the per-token attention state that grows with context length, and it can be surprisingly large. For a model with $L$ layers, $H$ KV-heads, head dimension $d$, context length $T$, and $c$ bytes per cache element (2 for fp16), the KV-cache size is:

$$\text{KV bytes} = 2 \times L \times H \times d \times T \times c$$

The factor of 2 is for keys and values. For Llama-2-7B ($L=32$, and with multi-head attention $H=32$, $d=128$) at $T=4096$ tokens in fp16:

$$2 \times 32 \times 32 \times 128 \times 4096 \times 2 \approx 2.1 \text{ GB}$$

That is on top of the 4.1 GB of `Q4_K_M` weights — so a 4096-token context roughly *doubles* nothing but adds a real 2 GB. This is why context length is a memory knob, not a free parameter, and why `llama.cpp` lets you quantize the KV-cache too (`-ctk q8_0 -ctv q8_0` halves it). Models that use **grouped-query attention** (GQA), like Llama-3 and Mistral, have far fewer KV-heads (e.g. 8 instead of 32), which shrinks the KV-cache by 4× and is a big reason newer models handle long context on laptops. The full treatment of KV-cache growth and the tricks to tame it is in [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).

#### Worked example: does this fit in 16 GB?

You have a 16 GB MacBook and want Llama-2-7B at 4096 context. Budget: weights 4.1 GB (`Q4_K_M`) + KV-cache 2.1 GB (fp16) + compute buffers ~0.5 GB + the OS and your apps taking ~6–8 GB. That sums to about 6.7 GB for the model against maybe 8 GB free — it fits, with headroom. Push context to 16k tokens and the KV-cache alone becomes ~8.4 GB, which blows the budget; the fix is GQA models, KV-cache quantization, or a shorter context. The sizing rule of thumb: **leave 1.5–2× the weight size free** for KV-cache and buffers, and you'll rarely be surprised.

## The practical flow: convert, quantize, run

Now the hands-on part. The lifecycle is three commands plus a build, summarized in Figure 3. We will use Llama-2-7B as the running example, but the flow is identical for any supported architecture.

![A left-to-right timeline of the local inference flow showing download Hugging Face weights then convert to fp16 GGUF then quantize to Q4 K M then benchmark with llama-bench then serve with llama-server exposing an OpenAI compatible API](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-3.png)

### Step 0: build llama.cpp

`llama.cpp` builds with CMake and auto-detects your hardware, but you can be explicit. On a Mac, Metal is enabled by default:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# macOS (Metal is on by default):
cmake -B build
cmake --build build --config Release -j

# Linux with an NVIDIA GPU:
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Linux/Windows with any Vulkan GPU (AMD/Intel/NVIDIA):
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j

# CPU-only, force the best SIMD for this machine:
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j
```

The build produces the binaries we'll use: `llama-cli` (interactive/one-shot generation), `llama-server` (the HTTP server), `llama-quantize` (the quantizer), `llama-bench` (the benchmark), and `convert_hf_to_gguf.py` (the converter, a Python script in the repo root).

### Step 1: convert Hugging Face weights to GGUF

The converter reads a Hugging Face model directory (the `safetensors` shards plus `config.json` and the tokenizer files) and emits a single GGUF in fp16 (or bf16/fp32). It does *no* quantization — it just repackages.

```bash
# Assume you've already downloaded the HF model to ./Llama-2-7b-hf
# (e.g. via `huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ...`)

python convert_hf_to_gguf.py ./Llama-2-7b-hf \
    --outfile llama-2-7b-f16.gguf \
    --outtype f16
```

This writes `llama-2-7b-f16.gguf`, about 13.5 GB. At this point the model is runnable but huge — this fp16 GGUF is your *reference* artifact, the one you measure quality against.

What is the converter actually doing? It is not training, calibrating, or rounding anything — it is a *repackager* with an architecture-aware tensor-renaming pass. The Hugging Face checkpoint names tensors in PyTorch's convention (`model.layers.0.self_attn.q_proj.weight`); `llama.cpp` expects its own canonical names (`blk.0.attn_q.weight`). Each supported architecture has a small Python class in the converter that declares this mapping plus any layout fix-ups (Llama, for instance, stores the RoPE-permuted Q and K projections, and the converter un-permutes them so `ggml`'s RoPE implementation sees the weights in the order it expects). It also reads `config.json` and the tokenizer files and *writes them into the GGUF metadata* — `general.architecture = "llama"`, `llama.block_count = 32`, the RoPE base frequency, the vocabulary and BPE merges, the special-token IDs. After this single pass everything the runtime needs lives in one file with no external dependency. A useful sanity check before you sink time into quantizing is to confirm the converter recognized the architecture and the metadata looks right:

```bash
# Inspect a GGUF's metadata and tensor list without loading weights
python gguf-py/scripts/gguf_dump.py llama-2-7b-f16.gguf | head -40
```

If `general.architecture` is wrong or a tensor is missing, the model will load but produce garbage — catching it here saves an hour. One real failure mode: a brand-new architecture that the converter doesn't know yet will either error out or, worse, map tensors incorrectly; in that case you either update to a newer `llama.cpp` (the community adds architectures fast) or hand-write the conversion class. This is the price of GGUF's self-containment — the converter has to *understand* each architecture, it can't blindly copy bytes.

### Step 2: quantize to Q4_K_M

`llama-quantize` takes the fp16 GGUF and the target type and writes a new, smaller GGUF. The type names map to the table above.

```bash
# The default sweet spot:
./build/bin/llama-quantize llama-2-7b-f16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# A few others to compare:
./build/bin/llama-quantize llama-2-7b-f16.gguf llama-2-7b-q5_k_m.gguf Q5_K_M
./build/bin/llama-quantize llama-2-7b-f16.gguf llama-2-7b-q6_k.gguf   Q6_K
./build/bin/llama-quantize llama-2-7b-f16.gguf llama-2-7b-q8_0.gguf   Q8_0
```

The `Q4_K_M` output is about 4.1 GB. Quantization here is **post-training and data-free by default** — it just rounds the weights into the k-quant blocks, no calibration set required, which is part of why it's so convenient. Mechanically, `llama-quantize` streams the fp16 tensors one at a time, and for each one it picks the target type (honoring the `_M` promotions for sensitive tensors), then for each super-block of 256 weights it searches for the scale and min that minimize reconstruction error over that block. For the legacy `Q4_0` family that search is a closed-form min/max-to-range mapping; for the k-quants it's a small numerical optimization per sub-block (and this is why quantizing a 7B takes a couple of minutes of CPU work rather than being instantaneous — it's solving thousands of tiny least-squares-ish problems). The output GGUF carries the same metadata as the fp16 input with the tensor types and offsets updated. Because it operates tensor-by-tensor and streams, `llama-quantize` needs only a little more than one tensor's worth of RAM, not the whole model — you can quantize a model far larger than your RAM.

You *can* improve low-bit quants with an **importance matrix** (an "imatrix"), which weights the quantization error by how much each weight actually matters on a calibration corpus; it's most worthwhile for the aggressive sub-4-bit types:

```bash
# Build an importance matrix from a small calibration text, then quantize with it
./build/bin/llama-imatrix -m llama-2-7b-f16.gguf -f calibration.txt -o imatrix.dat
./build/bin/llama-quantize --imatrix imatrix.dat \
    llama-2-7b-f16.gguf llama-2-7b-iq3_m.gguf IQ3_M
```

For `Q4_K_M` and above the imatrix gives a small benefit; for `IQ2`/`IQ3` it's close to mandatory. The conceptual link to GPTQ/AWQ-style error-aware quantization — which solve a similar "round the weights that matter least" problem — is drawn out in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).

### Step 3: run it

The simplest run is a one-shot completion with `llama-cli`:

```bash
./build/bin/llama-cli \
    -m llama-2-7b-q4_k_m.gguf \
    -p "Explain why decode is memory-bound in one paragraph." \
    -n 256 \
    -ngl 999 \
    -c 4096 \
    -t 8
```

The flags that matter:

- `-m` — the GGUF model file.
- `-p` — the prompt (or `-i` for interactive chat, or `-cnv` for conversation mode).
- `-n` — number of tokens to generate.
- `-ngl 999` — offload all layers to the GPU (Metal/CUDA/Vulkan). On a CPU-only build this is ignored.
- `-c 4096` — context size (the KV-cache budget). Bigger context = more RAM.
- `-t 8` — CPU threads. Set this to your number of *performance* cores, not logical cores; hyperthreads rarely help a memory-bound workload and oversubscribing can hurt.
- `-b` / `-ub` — logical and physical batch size for prefill (defaults are usually fine; raise `-ub` to speed up prefill on a GPU if you have memory).
- `--temp`, `--top-p`, `--top-k`, `--repeat-penalty` — sampling controls.

Threading deserves a word. Because decode is memory-bound, adding threads past the point where you saturate memory bandwidth does nothing — and on a hybrid CPU (performance + efficiency cores) putting work on the slow E-cores can *reduce* throughput. The reliable rule: set `-t` to the performance-core count and measure. More is not better here.

### The OpenAI-compatible server

The real reason `llama.cpp` slots into your existing tooling is `llama-server`. It exposes an HTTP API that is **drop-in compatible with the OpenAI Chat Completions API**, so any client or library that talks to OpenAI can point at your laptop instead.

```bash
./build/bin/llama-server \
    -m llama-2-7b-q4_k_m.gguf \
    -ngl 999 \
    -c 8192 \
    --host 127.0.0.1 \
    --port 8080 \
    --parallel 2
```

Now you can hit it exactly like the OpenAI endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "local",
      "messages": [{"role": "user", "content": "What is GGUF?"}],
      "temperature": 0.7
    }'
```

Or from Python, with the official OpenAI client pointed at localhost — you don't even need a real key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",  # llama-server ignores it unless you set --api-key
)

resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Summarize GGUF in two sentences."}],
)
print(resp.choices[0].message.content)
```

This is the moment local inference stops being a toy: any app you wrote against the OpenAI SDK runs unchanged against a 4 GB file on your laptop, offline, for free. The `--parallel` flag lets the server handle multiple concurrent requests by splitting the KV-cache into slots — useful if a few apps share one server, though throughput per request drops as you add slots since they all contend for the same memory bandwidth.

### Measuring honestly with llama-bench

Never trust a single hand-timed run. `llama-bench` runs prefill and decode separately, warms up, repeats, and reports a mean with standard deviation:

```bash
./build/bin/llama-bench \
    -m llama-2-7b-q4_k_m.gguf \
    -ngl 999 \
    -p 512 \
    -n 128 \
    -r 5
```

It prints a table: `pp512` (prefill throughput on a 512-token prompt, tokens/s) and `tg128` (decode/text-generation throughput on 128 tokens, tokens/s), each with a ± standard deviation across the 5 repetitions. A few honesty notes that apply to *any* on-device benchmark:

- **Warm up.** The first run pays page-fault and kernel-compile costs. `llama-bench` discards a warmup run by default; if you roll your own, do the same.
- **Watch thermals.** Laptops throttle. A number from the first 10 seconds can be 20–30% higher than the sustained number after the chassis heats up. Report sustained throughput for anything that runs longer than a few seconds.
- **Batch=1 is the reality for chat.** Server benchmarks love big batches because they amortize the weight reads across many sequences. But interactive single-user chat is batch=1, which is the pure memory-bound case. Quote batch=1 numbers for the laptop use case.
- **Pin the context.** Decode speed drifts as the KV-cache grows (more bytes to read for attention each step). Report it at a stated context length.

If you want to roll your own measurement — say, to log tokens/s and peak resident memory together while a real workload runs through `llama-server` — a small Python harness against the OpenAI-compatible endpoint does the job. It streams the completion, times the first token (prefill latency) separately from the rest (decode), and reads the process's resident set size so you can put a real RAM number next to the speed number:

```python
import time, requests, subprocess, json

def rss_mb(pid):
    # macOS/Linux: resident set size in MB via ps
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out) / 1024  # ps reports KB

def bench(server_pid, prompt, n_predict=128):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": False,
    }
    t0 = time.perf_counter()
    first_tok_t = None
    n_tokens = 0
    with requests.post("http://127.0.0.1:8080/completion",
                       json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("content"):
                if first_tok_t is None:
                    first_tok_t = time.perf_counter()
                n_tokens += 1
    t_end = time.perf_counter()

    prefill_s = first_tok_t - t0
    decode_s = t_end - first_tok_t
    return {
        "ttft_ms": prefill_s * 1000,
        "decode_tok_s": (n_tokens - 1) / decode_s if decode_s > 0 else 0,
        "peak_rss_mb": rss_mb(server_pid),
    }

# Run a 512-ish-token prompt and report
report = bench(server_pid=12345, prompt="Summarize the GGUF format. " * 40)
print(json.dumps(report, indent=2))
```

The two numbers that matter fall out directly: `ttft_ms` (time to first token, dominated by prefill) and `decode_tok_s` (the streaming rate). Run it at a couple of context lengths and you'll watch the decode rate slide downward exactly as the KV-cache term in the bandwidth equation predicts — the harness turns the theory into a curve you can see. Pair `peak_rss_mb` with the model size and you can confirm your RAM budget (weights + KV-cache + buffers) empirically rather than trusting the back-of-envelope.

## Worked examples on real machines

Let me put concrete numbers on two scenarios so you can calibrate your expectations. These are representative figures for Llama-2-7B-class models from `llama.cpp` community benchmarks; your exact mileage depends on the chip revision, thermal headroom, and driver versions, so treat them as well-grounded approximations rather than guarantees.

#### Worked example: 7B fp16 → Q4_K_M on an M2/M3 Mac with Metal

Start with the fp16 GGUF on an M3 MacBook Air (a fanless machine, ~100 GB/s unified memory). The fp16 model is 13.5 GB, which means on a 16 GB machine it barely fits and the page cache fights your other apps; decode crawls because you're reading 27 GB worth of bytes per token's worth of work and the system is swapping. This is the "doesn't fit" failure everyone hits first.

Now quantize to `Q4_K_M` (4.1 GB) and run with `-ngl 999 -c 4096`. Predicted decode from the roofline: $100 / 4.2 \approx 24$ tok/s; measured on Metal is higher, around **40–48 tok/s**, because the GPU's effective bandwidth to its compute units exceeds the nominal figure and Metal overlaps the dequantize with the matmul. Prefill on a 512-token prompt runs around **400–550 tok/s**, so a half-page prompt is digested in roughly a second. Peak memory sits near 6.5 GB (weights + KV-cache + buffers), comfortable on 16 GB. Quality versus the fp16 reference: a perplexity increase of around +0.06, which is imperceptible in practice. This is the canonical local setup — and it is genuinely pleasant to use.

| Config       | Size    | Peak RAM | Prefill (pp512) | Decode (tg128) | Quality (ΔPPL) |
| ------------ | ------- | -------- | --------------- | -------------- | -------------- |
| fp16         | 13.5 GB | ~15 GB   | ~450 tok/s      | ~10 tok/s\*    | baseline       |
| `Q8_0`       | 7.2 GB  | ~9 GB    | ~520 tok/s      | ~30 tok/s      | ~0.00          |
| `Q5_K_M`     | 4.8 GB  | ~7 GB    | ~540 tok/s      | ~42 tok/s      | +0.02          |
| **`Q4_K_M`** | 4.1 GB  | ~6.5 GB  | ~550 tok/s      | ~46 tok/s      | +0.06          |

\*The fp16 decode is depressed because the model is near the RAM ceiling and the page cache thrashes; on a 32 GB Mac fp16 decode would land near ~14 tok/s, exactly the roofline prediction $100/27$. That asterisk *is* the lesson: quantization isn't only about quality, it's about staying off the swap cliff.

#### Worked example: the same model on a CPU-only laptop and on a small GPU (the -ngl effect)

Take the same `Q4_K_M` file to a CPU-only x86 laptop (say ~60 GB/s effective DRAM bandwidth, 8 performance cores). With a CPU-only build, `-ngl` is a no-op and everything runs on AVX2 kernels. Decode lands around **10–14 tok/s** — right on the roofline prediction $60/4.2 \approx 14$. Prefill is the painful part: with only CPU compute, a 512-token prompt runs at maybe **70–110 tok/s**, so digesting a long system prompt takes several seconds before the first token appears. Usable for short prompts, sluggish for long ones. Setting `-t` above the performance-core count doesn't help and sometimes hurts.

Now add a modest discrete GPU — say a laptop RTX 4060 with 8 GB VRAM — and rebuild with `-DGGML_CUDA=ON`. The whole `Q4_K_M` model (4.1 GB + ~2 GB KV-cache at 4k) fits in 8 GB, so `-ngl 999` offloads everything. Decode jumps to roughly **70–90 tok/s** (the GPU's VRAM bandwidth is ~270 GB/s, far above the CPU's 60), and prefill leaps to **2000+ tok/s** so prompts are digested almost instantly. That is the `-ngl` effect: moving the layers to faster memory is the single biggest lever once the bits are already small.

The interesting middle case is a *bigger* model that doesn't fully fit. Suppose you try a 13B `Q4_K_M` (~7.9 GB) on that 8 GB card. The full model plus KV-cache exceeds VRAM, so you offload as many layers as fit — say `-ngl 24` of 40 layers — and leave 16 on the CPU. Now every token must traverse 24 fast GPU layers *and* 16 slow CPU layers in sequence, and the CPU portion dominates: decode might be ~12 tok/s, barely better than CPU-only, because the slow tail gates the whole pipeline. The lesson: **partial offload buys you the ability to run, not the speed of a full offload.** If speed matters, pick a model and quant that fully fit your VRAM. Figure 7 makes the cross-backend pattern explicit.

![A matrix comparing decode tokens per second, prefill throughput, memory bandwidth, and where the model fits across a CPU, an Apple Metal GPU, and an NVIDIA RTX 4090, showing tokens per second tracking memory bandwidth](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-7.png)

## Results: quant types and backends, side by side

Two tables consolidate the trade-offs you'll actually reason about. First, the quant-type Pareto for a 7B model — this is the figure you consult when deciding *which file to download*. Figure 6 visualizes the same data as a matrix.

![A matrix of k-quant types for a 7 billion parameter model showing bits per weight, file size, RAM needed, and quality perplexity delta for Q4 K M, Q5 K M, Q6 K, Q8 0, and fp16](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-6.png)

| Quant        | Bits/wt | 7B size | RAM (4k ctx) | Decode (M3 Metal) | Quality (ΔPPL) | Verdict                    |
| ------------ | ------- | ------- | ------------ | ----------------- | -------------- | -------------------------- |
| **`Q4_K_M`** | ~4.8    | 4.1 GB  | ~6.5 GB      | ~46 tok/s         | +0.06          | The default. Pick this.    |
| `Q5_K_M`     | ~5.7    | 4.8 GB  | ~7.0 GB      | ~42 tok/s         | +0.02          | Slightly better quality.   |
| `Q6_K`       | ~6.6    | 5.5 GB  | ~7.8 GB      | ~38 tok/s         | ~0.00          | When quality is critical.  |
| `Q8_0`       | 8.5     | 7.2 GB  | ~9.5 GB      | ~30 tok/s         | ~0.00          | Reference; rarely needed.  |
| fp16         | 16      | 13.5 GB | ~15.5 GB     | ~14 tok/s\*       | baseline       | Only for re-quantizing.    |

\*On a 32 GB machine where fp16 isn't thrashing. The clean read: smaller bits → faster decode (fewer bytes per token) *and* smaller RAM, with quality flat until you drop below 4 bits. `Q4_K_M` dominates the frontier.

Second, the backend table — the figure you consult when deciding *which machine or build to run on*. Same `Q4_K_M` 7B model, three backends:

| Backend         | Mem bandwidth | Decode (tg) | Prefill (pp512) | Constraint               |
| --------------- | ------------- | ----------- | --------------- | ------------------------ |
| CPU (x86 AVX2)  | ~60 GB/s      | ~12 tok/s   | ~90 tok/s       | Slow prefill; universal. |
| Metal (M3)      | ~100 GB/s     | ~46 tok/s   | ~550 tok/s      | Unified mem; Mac only.   |
| CUDA (RTX 4090) | ~1000 GB/s    | ~140 tok/s  | ~5000 tok/s     | Must fit in VRAM.        |

The decode column climbs in lockstep with the bandwidth column — the memory-bound law, visible in data. The prefill column climbs even faster because prefill is compute-bound and the GPUs have far more FLOPs, not just more bandwidth. If your use case is long prompts (RAG, document QA), the prefill column is what you optimize; if it's short-prompt chat, the decode column is.

### A note on what "quality" means here

The perplexity deltas above are a proxy, and a coarse one. Perplexity measures how surprised the model is by held-out text; a +0.06 PPL increase is well below the noise floor of most downstream tasks. But perplexity can hide failures that matter — `Q2_K` might show a tolerable-looking PPL while clearly degrading on reasoning or code. The honest practice is to **evaluate on a task you care about**: run your actual prompts through the quantized model and the fp16 reference side by side and read the outputs. For anything `Q4_K_M` and up on a 7B+ model you will struggle to tell them apart; below that, trust your own task eval over the PPL number. This connects to the broader [model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) point that every compression lever must be validated on the metric you actually ship against, not a proxy.

## Case studies and real numbers from the wild

A few grounding data points from shipped work and well-known community results, so the figures above don't feel invented:

**Whisper.cpp on a laptop.** The same `ggml` engine powers `whisper.cpp`, which runs OpenAI's Whisper speech-recognition models on CPU in real time. Georgi Gerganov's original `whisper.cpp` demo — transcribing audio faster than real-time on a MacBook with no GPU — is what kicked off the whole `ggml` ecosystem in 2022, and it's a clean demonstration that the memory-bound + quantized + tight-kernels recipe generalizes beyond text LLMs.

**Llama on a Raspberry Pi.** People have run quantized 7B models on a Raspberry Pi 5 (8 GB, ~17 GB/s LPDDR4X bandwidth). The roofline predicts $17 / 4.2 \approx 4$ tok/s, and measured results land around 2–4 tok/s — slow, but it *runs*, and the prediction is dead on. It's the most extreme confirmation of the bandwidth law: cut the bandwidth to a fifth of a laptop's and you get a fifth of the tokens.

**Phi and Gemma on-device.** Microsoft's Phi-3-mini (3.8B) and Google's Gemma-2-2B were explicitly designed to be strong-per-parameter so they'd run well locally. Quantized to `Q4_K_M` they're ~2.2 GB and ~1.6 GB respectively and decode at 50–80+ tok/s on a Mac — comfortably interactive — which is why "small, well-trained model + `Q4_K_M`" has become the default local-assistant recipe. For why these small models punch above their parameter count, see [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design).

**GGUF as the de-facto distribution format.** Hugging Face now hosts tens of thousands of pre-quantized GGUF files (the community quantizers like "TheBloke" and "bartowski" popularized one-click `Q4_K_M` downloads). In practice you rarely run `convert_hf_to_gguf.py` yourself — you download a ready GGUF — but knowing the flow means you can quantize a model nobody else has, or build a custom imatrix for your domain.

**GQA's effect on the KV-cache, measured.** A clean illustration of the KV-cache math from earlier: Llama-2-7B uses full multi-head attention (32 KV-heads) and its fp16 KV-cache at 4k context is ~2.1 GB; Llama-3-8B uses grouped-query attention with 8 KV-heads, so the *same* 4k-context KV-cache is ~0.5 GB — a 4× reduction that follows directly from $H$ dropping from 32 to 8 in the $2LHdTc$ formula. This is why the newer 8B model, despite having *more* parameters than the older 7B, can actually run *longer* contexts on the same laptop: the weight term grew slightly but the KV-cache term — the part that scales with context — shrank fourfold. When you read "this model is better for long-context local use," GQA in the architecture is usually the reason, and you can verify it by inspecting `llama.attention.head_count_kv` in the GGUF metadata before you download. It is a concrete payoff of reading the bandwidth-and-memory equations rather than guessing.

## Stress test: what actually breaks at the edges

Everything above is the happy path. The instructive part of local inference is what happens when you push past the comfortable region, because the failure modes are not random — each one is a specific term in the equations above going out of budget. Let me walk the three failures you are most likely to hit and trace each back to its cause, because diagnosing them is the difference between "it's broken" and "it's bandwidth, here's the fix."

**Failure 1: the model plus KV-cache exceeds physical RAM.** This is the most common and the most punishing. Say you load a 13B `Q4_K_M` (~7.9 GB of weights) at 8k context on a 16 GB laptop. Weights 7.9 GB + KV-cache (non-GQA, 8k) ~4.2 GB + buffers ~0.7 GB ≈ 12.8 GB for the model, against maybe 9–10 GB actually free after the OS and your apps. You are over budget. Two distinct things can happen depending on the OS. If `mmap` is in use (the default), the kernel will *evict* weight pages it has already faulted in to make room for the ones the next layer needs, then fault them back in next token — so every decode step re-reads weights from the SSD instead of RAM. Your effective "bandwidth" collapses from ~60 GB/s (DRAM) to ~3–5 GB/s (NVMe), and the bandwidth law does the rest: decode falls off a cliff to well under 1 tok/s, with the disk light pegged. This is the "swap cliff," and the fp16-on-16GB row in the M3 table (\*-marked at ~10 tok/s) is a mild version of it. The fix is not "add threads" or "wait it out" — it is to *make the working set fit*: drop to a smaller quant (`Q3_K_M`), shorten context, quantize the KV-cache, or pick a smaller model. On Linux you can also disable `mmap` with `--no-mmap` to force a single up-front load, which fails fast with an out-of-memory error instead of silently thrashing — sometimes a clean crash is the better outcome because it tells you the truth immediately.

**Failure 2: partial `-ngl` offload, where the CPU tail gates everything.** Suppose that same 13B sits on an 8 GB GPU. The full model won't fit in VRAM, so you offload what fits — `-ngl 24` of 40 layers — and the other 16 stay on the CPU. Now reason about what one token costs. The 24 GPU layers each read their weights from ~270 GB/s VRAM; the 16 CPU layers each read theirs from ~60 GB/s DRAM. Every token must pass through *all 40 layers in sequence* — there is no skipping. So the per-token time is the sum of the GPU-layer time and the CPU-layer time, and the CPU portion, being on memory four-plus times slower, dominates. Concretely: if the 16 CPU layers alone would decode at the CPU rate for 16/40 of a model (≈ 30 tok/s for that fraction) and the 24 GPU layers are near-instant by comparison, your end-to-end rate is dragged toward the slow tail — you might measure ~12 tok/s, barely above pure CPU and far below the ~70 tok/s a fully-offloaded model would hit. This is the trap that makes people think "I gave it a GPU and it barely helped." The diagnosis is to watch the GPU utilization: if it's low and oscillating, the GPU is idle waiting on the CPU tail. The fix is to make the model fit *fully* in VRAM (smaller quant, smaller model) so `-ngl 999` offloads everything, or accept that partial offload bought you the ability to run, not the speed. There is one nuance worth knowing: which layers you keep on CPU matters slightly — keeping the token-embedding and output layers on the device that can dispatch them cheaply, and offloading the contiguous transformer blocks, is what the default placement tries to do, but you cannot beat the fundamental "every token traverses every layer" constraint.

#### Worked example: the swap cliff, measured

Here is the cliff in numbers so you can recognize it. On a 16 GB Mac, `llama-2-7b-q4_k_m.gguf` (4.1 GB) at 4k context fits with headroom and decodes around 46 tok/s on Metal. Now force it over the edge: load the same model at a 32k context, which pushes the KV-cache (non-GQA fp16) to roughly $2 \times 32 \times 32 \times 128 \times 32768 \times 2 \approx 17$ GB. Weights 4.1 GB + KV 17 GB = 21 GB, well past 16 GB. The instant the KV-cache can't be held resident, decode does not degrade gracefully by 20% — it falls to single-digit or sub-1 tok/s as the system pages, and the machine becomes unresponsive because the page cache is fighting the rest of the OS for RAM. The lesson is that the failure is a *cliff, not a slope*: you are fine, fine, fine, then catastrophically not fine the moment the working set crosses physical RAM. The defense is to compute the KV-cache term *before* you set `-c`, not after the machine locks up. At 32k you'd want a GQA model (which cuts that 17 GB to ~4 GB) plus `-ctk q8_0 -ctv q8_0` (another halving), bringing the cache to ~2 GB and back inside budget.

**Failure 3: the CPU-only box where prefill, not decode, is the wall.** People obsess over decode tok/s and forget that on a CPU-only machine the *prompt* is often the bottleneck. Take a CPU-only x86 server with no GPU, running 7B `Q4_K_M`. Decode is a tolerable ~12 tok/s (the bandwidth law, $60/5$). But prefill is compute-bound, and a CPU has maybe 1–2% of a datacenter GPU's FLOPs, so a 4k-token RAG context prefills at perhaps 80 tok/s — meaning $4096 / 80 \approx 51$ seconds before the *first* token appears. The user stares at a blank screen for nearly a minute, then gets a perfectly snappy 12 tok/s stream. End-to-end latency is dominated entirely by a phase most people don't measure. The fixes here are different from the decode fixes: a smaller prompt (trim the RAG context, summarize before injecting), prompt caching (`--cache-prompt` so a repeated system prompt is prefilled once), a larger prefill batch (`-ub`) to better use the cores, or — the real answer — any GPU at all, because prefill is where the GPU's FLOPs help most. This is the operational reason the decision tree later sends long-prompt workloads toward a GPU backend even when decode would be fine on CPU.

The through-line across all three failures: **identify which term in the budget went out of range** — weights+KV vs physical RAM, slow-layer fraction vs fast-layer fraction, prefill FLOPs vs prompt length — and the fix follows mechanically. Local inference rewards engineers who size the budget before they run, which is exactly why we spent so long on the equations.

## When llama.cpp is the right tool — and when it isn't

`llama.cpp` is excellent, but it is not the only local stack, and choosing well saves you a lot of grief. The decisive question is **where the model has to run and who it serves**. Figure 8 routes the decision.

![A decision tree routing by deployment target where a personal device splits into laptop or desktop going to llama.cpp with GGUF and Metal versus a phone going to MLC-LLM on TVM, while a shared server goes to vLLM for batched GPU serving](/imgs/blogs/running-llms-locally-llama-cpp-and-gguf-8.png)

**Reach for `llama.cpp` when:**

- You're targeting a **desktop or laptop** (Mac, Windows, Linux) — its CPU/Metal/CUDA/Vulkan coverage is unmatched and it's the path of least resistance.
- **Privacy and offline** matter: the data never leaves the device, no network, no key, no logs in someone's datacenter.
- You value **hackability**: it's a small C/C++ codebase you can read, patch, and embed. GGUF is trivial to inspect. The OpenAI-compatible server drops into existing tooling.
- You want **single-user, batch=1 interactive** generation — the case it's most tuned for.

**Reach for something else when:**

- **Mobile (iOS/Android).** `llama.cpp` runs on phones but isn't the most efficient path to a phone's NPU. **MLC-LLM** compiles models through TVM down to mobile GPUs and NPUs and ships a clean mobile runtime; that's the subject of the companion post [running LLMs locally with MLC and mobile stacks](/blog/machine-learning/edge-ai/running-llms-locally-mlc-and-mobile-stacks). Apple's **Core ML** and **MLX** are also strong on Apple devices when you want ANE/NPU acceleration.
- **High-throughput multi-user serving.** If you're serving many concurrent users on a server GPU, **vLLM** (with PagedAttention and continuous batching) or **TensorRT-LLM** will crush `llama.cpp` on aggregate throughput, because they're built to amortize weight reads across large dynamic batches. `llama.cpp`'s `--parallel` works but isn't in the same league for fleet serving. (For the GPU-serving side of edge, see [TensorRT and GPU edge inference on Jetson](/blog/machine-learning/edge-ai/tensorrt-and-gpu-edge-inference-on-jetson).)
- **Microcontrollers and tiny MCUs.** Way out of scope — that's TFLite-Micro / CMSIS-NN territory, kilobytes of SRAM, not gigabytes.

### The model-size-vs-RAM sizing rule

Before you pick a model at all, size it against your hardware with one rule:

$$\text{free RAM needed} \approx \text{(weight size)} + \text{KV-cache} + \text{buffers} \approx 1.5 \times \text{(weight size)} \;\text{for short context}$$

So a machine with 8 GB *free* comfortably runs a model whose `Q4_K_M` weights are ≤ ~5 GB — that's any 7B–8B model, and even some 13B at a tighter quant. A 16 GB free budget reaches 13B–14B `Q4_K_M` happily and 30B-class at a pinch. For long context, add the KV-cache term explicitly (it scales linearly with context length and, for non-GQA models, can rival the weight size at 16k+ tokens). And remember the speed law on top of the fit law: even if a 70B model *fits* by spilling to CPU, decode will crawl — fit and speed are two separate gates, and on a laptop you usually want the largest model that fits *fully in your fastest memory*, not the largest that fits at all.

### Stress-testing the decision

Let me poke at the edges, because that's where the real engineering lives:

- *What if the prompt is enormous (32k-token RAG context)?* Then prefill — compute-bound — dominates, and a CPU-only setup will make you wait many seconds before the first token. You want a GPU backend (more FLOPs) and possibly KV-cache quantization to fit the context. The decode law no longer tells the whole story.
- *What if the model barely doesn't fit in VRAM?* Partial `-ngl` offload runs it but the CPU tail gates speed; better to step down one quant level (`Q4_K_M` → `Q3_K_M`) to make it fit fully, accepting a small quality hit, than to live with the hybrid penalty. Measure both.
- *What if decode is fast but the answer is wrong?* That's a quality problem, not a speed problem — re-test at a higher quant (`Q5_K_M`/`Q6_K`) or with an imatrix; if the fp16 reference is also wrong, it's the model, not the quantization.
- *What if you're memory-bound and threads don't help?* Correct and expected — that's the regime telling you to reduce bytes-per-token (lower quant, KV-cache quant) or move to faster memory (GPU), not to add cores. For the full theory of diagnosing this, see [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast).

## Key takeaways

1. **Local 7B inference works because of two facts at once:** int4-class quantization makes the weights *fit* (14 GB → ~4 GB), and single-token decode is *memory-bound*, so a laptop's modest 50–100 GB/s bandwidth still yields usable tokens per second.
2. **The one equation to remember:** $\text{tokens/s} \approx \text{bandwidth} / (\text{params} \times \text{bytes-per-weight})$. It predicts decode speed on any machine before you download a thing.
3. **GGUF is one self-contained, memory-mapped file** — weights, metadata, and tokenizer together — which is why models start instantly, share pages across processes, and need no Python to load.
4. **`Q4_K_M` is the default for a reason:** ~4.8 bits/weight, ~4.1 GB for a 7B, near-zero quality loss. Go up to `Q5_K_M`/`Q6_K` only when RAM is free and quality is critical; go below 4 bits only under RAM pressure and verify on your task.
5. **Prefill and decode are opposite workloads** — prefill is compute-bound (optimize with FLOPs/GPU), decode is memory-bound (optimize with bandwidth and fewer bits). Measure them separately with `llama-bench`.
6. **`-ngl` is the biggest speed lever on a GPU machine, but it's close to all-or-nothing:** partial offload lets a too-big model run, it does not make it fast. Pick a model+quant that fits fully in your fastest memory.
7. **Size before you download:** budget ~1.5× the weight size in free RAM for short context, and add the KV-cache term ($2LHdTc$ bytes) explicitly for long context. GQA models shrink the KV-cache 4×.
8. **`llama-server` is OpenAI-compatible**, so any app written against the OpenAI SDK runs unchanged against a local 4 GB file — offline, free, private.
9. **Choose the stack by target:** `llama.cpp` for desktop/laptop, MLC for mobile NPUs, vLLM/TensorRT-LLM for multi-user server throughput.
10. **Benchmark honestly:** warm up, watch thermals, quote batch=1 sustained numbers at a stated context length — the first 10 seconds always lie.

## Further reading

- **`llama.cpp`** — the project, build docs, and the full quant-type list: the [ggml-org/llama.cpp repository](https://github.com/ggml-org/llama.cpp) and its `examples/` directory.
- **`ggml`** — the tensor library underneath, and the **GGUF format spec** (the file-layout and metadata-key documentation in the `ggml` docs) for anyone implementing a loader.
- **The k-quant discussion** — the original `llama.cpp` GitHub discussions/PRs introducing the K-quants (`Q4_K`, `Q5_K`, `Q6_K`) and the super-block structure, plus the later IQ (importance-matrix) quants; the best primary source for *why* the bit allocation is what it is.
- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)** — for the **NF4** 4-bit format as a contrast to the k-quants: an information-theoretically-motivated normal-float quantization, used by `bitsandbytes`, that solves the same 4-bit problem from a different angle.
- **Frantar et al., "GPTQ" (2022)** and **Lin et al., "AWQ" (2023)** — error-aware weight-only quantization that informs the imatrix idea; walked through in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).
- **Within this series:** [the model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for the memory-bound derivation, [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast) for end-to-end latency tuning, the companion [MLC and mobile stacks](/blog/machine-learning/edge-ai/running-llms-locally-mlc-and-mobile-stacks) post, and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that ties every lever together.
