---
title: "Channels Last and Memory Formats: When Tensor Layout Decides Your Kernel Speed"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "How the physical byte layout of a 4D tensor silently picks which convolution kernel runs, why channels_last unlocks the tensor core fast path, and how to prove the win landed in the trace."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "channels-last",
    "memory-format",
    "profiling",
    "pytorch",
    "cuda",
    "tensor-cores",
    "convolution",
    "latency",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 39
---

A vision service on an A100 was serving a ResNet-50 classifier at batch 32. The team had already done the obvious things: half precision through autocast, a warmed-up model, pinned-memory host copies, a tidy request handler. Steady-state p50 sat at 6.8 ms per batch, about 4,700 images per second, and the GPU looked busy. Then someone added a single line before the serving loop — converting the model and the input tensor to `channels_last` — and p50 dropped to 5.3 ms, throughput climbed to roughly 6,000 images per second, and the accuracy was bit-for-bit unchanged. A 22% speedup, no retraining, no new kernel, no code change inside the model. Just a different byte layout for the exact same numbers.

That result is confusing the first time you see it, and it should be. The convolution computes the identical math either way — same weights, same inputs, same outputs to the last ULP. Moving the bytes around cannot change the answer, so how can it change the speed? The answer is that a GPU library does not run "a convolution." It runs one of dozens of *specific kernels*, and which one it picks depends on how your tensor is laid out in memory. Change the layout and you change the kernel — and on a modern GPU, one of those kernels uses the Tensor Cores at full tilt while the other either falls back to a slower path or pays to transpose your data first. The figure below shows the two outcomes side by side: same `conv2d` call, two different kernels, a 4.2 ms convolution becoming a 3.1 ms one.

![a two column comparison showing a default NCHW convolution paying a hidden transpose and running a 4.2 millisecond kernel versus a channels last convolution running a 3.1 millisecond tensor core kernel directly](/imgs/blogs/channels-last-and-memory-formats-1.webp)

This post is about that lever. By the end you will be able to explain exactly what a memory format *is* (it is a stride pattern, not a shape), why cuDNN and the Tensor Cores want channels innermost, how to turn `channels_last` on for a model and its inputs with one line each, how to *verify* in the profiler that it actually took, and — the part most people miss — how to catch the single format-breaking operator that silently reverts your network to NCHW mid-forward and quietly gives back the entire speedup. It fits the series' recurring frame: this is a **bandwidth wall plus kernel-selection** win. The default layout wastes memory bandwidth on transposes and hands the work to a weaker kernel; the fix is a layout, not a rewrite. If you have not yet read [why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu), that post frames the four wastes this one attacks two of at once.

## What a memory format actually is

Start with the thing everyone gets subtly wrong: a tensor's **shape** and its **memory format** are two different objects. The shape is the logical size along each axis — a 4D activation is `(N, C, H, W)`: batch, channels, height, width. That never changes when you switch formats. What changes is the **stride**: the number of elements you step forward in the flat backing buffer to move one index along a given axis. A GPU tensor is always, physically, a one-dimensional array of numbers. The stride vector is the recipe that maps a 4D index `(n, c, h, w)` to a single offset into that flat array:

$$\text{offset}(n,c,h,w) = n \cdot s_N + c \cdot s_C + h \cdot s_H + w \cdot s_W$$

where $s_N, s_C, s_H, s_W$ is the stride vector. The default PyTorch layout — the one you get from `torch.empty(N, C, H, W)` — is called **NCHW** or "contiguous," and its strides are $(C \cdot H \cdot W,\ H \cdot W,\ W,\ 1)$. Reading right to left: stepping one pixel in `W` moves 1 element; one row in `H` moves `W` elements; one channel in `C` jumps a whole `H×W` plane; one batch item jumps `C` full planes. The channel stride is large. All of channel 0's spatial plane is stored first, then all of channel 1's, and so on.

`channels_last` is the same logical `(N, C, H, W)` shape stored with a different stride vector: $(H \cdot W \cdot C,\ 1,\ W \cdot C,\ C)$. Now the *channel* stride is 1 — channels are innermost, adjacent in memory — and the width stride is `C`. Physically this is the NHWC layout: for each pixel, its channels sit next to each other, then you move to the next pixel. The figure below shows both stride vectors hanging off one unchanged shape and one unchanged buffer of 48 values, which is the whole point: the format is the stride, not the shape.

![a layered stack showing one logical shape mapping to two different stride vectors over a single flat buffer of forty eight values](/imgs/blogs/channels-last-and-memory-formats-2.webp)

**Contiguity** is the concept that ties this together. A tensor is "contiguous" in a given format when its strides match that format's canonical pattern exactly, with no gaps. The default `x.is_contiguous()` asks "are you contiguous in NCHW?" There is a second question, `x.is_contiguous(memory_format=torch.channels_last)`, that asks "are you contiguous in NHWC?" A freshly allocated NCHW tensor answers True to the first and False to the second; a `channels_last` tensor flips both answers. The bytes are laid out one way or the other, and these two predicates tell you which. This matters enormously later, because the *only* reliable way to know your fast layout survived a block of the network is to ask this second question after it.

#### Worked example: computing both stride vectors by hand

Take the smallest tensor the PyTorch tutorial uses, shape `(1, 3, 4, 4)` — one image, 3 channels, 4×4 spatial. It holds `1 × 3 × 4 × 4 = 48` values. In NCHW the strides are $(3 \cdot 16,\ 16,\ 4,\ 1) = (48, 16, 4, 1)$. In `channels_last` they are $(16 \cdot 3,\ 1,\ 4 \cdot 3,\ 3) = (48, 1, 12, 3)$.

Now locate the three channels of the top-left pixel, indices `(0,0,0,0)`, `(0,1,0,0)`, `(0,2,0,0)`. In NCHW their offsets are `0`, `16`, `32` — the three channels of one pixel are 16 elements apart, scattered across three separate planes. In `channels_last` their offsets are `0`, `1`, `2` — the three channels of one pixel are adjacent, right next to each other. That adjacency is not cosmetic. When a convolution kernel needs all `C` input channels at a spatial location to compute one output — which is exactly what convolution does — NHWC hands it a contiguous run it can load in one coalesced burst, while NCHW forces it to gather `C` values that are `H×W` apart. Hold onto that; it is the entire mechanism.

```python
import torch

N, C, H, W = 1, 3, 4, 4
x = torch.arange(N * C * H * W).reshape(N, C, H, W)

print("shape        :", tuple(x.shape))
print("NCHW stride  :", x.stride())
print("NCHW contig  :", x.is_contiguous())
print("chlast contig:", x.is_contiguous(memory_format=torch.channels_last))

xcl = x.to(memory_format=torch.channels_last)
print("---- after .to(channels_last) ----")
print("shape        :", tuple(xcl.shape))       # unchanged
print("chlast stride:", xcl.stride())
print("NCHW contig  :", xcl.is_contiguous())
print("chlast contig:", xcl.is_contiguous(memory_format=torch.channels_last))
```

```console
shape        : (1, 3, 4, 4)
NCHW stride  : (48, 16, 4, 1)
NCHW contig  : True
chlast contig: False
---- after .to(channels_last) ----
shape        : (1, 3, 4, 4)
chlast stride: (48, 1, 12, 3)
NCHW contig  : False
chlast contig: True
```

The shape printed identically before and after. Only the stride vector and the two contiguity predicates flipped. `x.to(memory_format=torch.channels_last)` did not reshape anything and did not change a single value — it produced a new view-plus-copy whose strides describe the NHWC ordering. That is the API in miniature, and everything else in this post is about making that one call propagate through a whole network and stay there.

## What those strides mean in memory

The stride vector is precise but abstract, so pin down what it does to the actual bytes. The figure below reads memory as a flat tape and marks what sits in each slot under each format for our `(1, 3, 4, 4)` example.

![a two row grid contrasting NCHW storing each channel plane as a contiguous block of sixteen values against channels last interleaving the three channels of every pixel](/imgs/blogs/channels-last-and-memory-formats-3.webp)

Under NCHW, memory slots 0 through 15 hold the entire red channel — all 16 spatial positions — then slots 16 through 31 hold the entire green channel, then 32 through 47 the blue. The data is *plane-major*: a whole channel at a time. Under `channels_last`, slots 0, 1, 2 hold the red, green, blue of pixel `(0,0)`; slots 3, 4, 5 hold RGB of pixel `(0,1)`; and so on. The data is *pixel-major*: all channels of one pixel, then the next pixel.

Why does a convolution care which one it gets? A convolution at each output position reads a small spatial window across *every* input channel and dots it with the filter. The inner sum runs over channels. If the channels of a given spatial location are contiguous — the NHWC case — the kernel walks a straight, contiguous run of memory for its inner reduction, and consecutive GPU threads read consecutive addresses. That is **coalesced** access: the memory system fuses many thread requests into a few wide transactions and delivers close to peak bandwidth. If the channels are `H×W` apart — the NCHW case — the same inner reduction strides through memory in big jumps, threads touch scattered cache lines, and effective bandwidth collapses. For the deeper mechanics of why coalesced access maps onto the hardware's warps and memory transactions, the HPC piece on [the GPU's SMs, warps, and SIMT execution](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) is the right companion; here we only need the consequence.

Put a number on the coalescing gap. A GPU services memory in fixed transactions — typically 32-byte or 128-byte sectors. A warp of 32 threads that reads 32 consecutive fp16 values touches 64 contiguous bytes, which the hardware satisfies in a couple of sector transactions; that is coalesced and runs near the bandwidth ceiling. The same warp reading 32 values that are each `H×W` elements apart — the NCHW channel stride for a 56×56 map is 3,136 elements, or 6,272 bytes in fp16 — touches 32 *different* sectors, one useful value per sector, and wastes roughly 31 out of every 32 bytes it drags across the bus. The reduction over channels is exactly this access pattern, so the NCHW convolution can burn an order of magnitude more effective bandwidth for the same delivered data. NHWC collapses those 32 scattered reads back into a contiguous burst. This is the bandwidth half of the win, and it is why the profile shows lower HBM read traffic after the switch even before you count the transposes.

There is a second consequence beyond raw bandwidth, and it is the bigger one on modern hardware: the *choice of kernel*. Which brings us to the Tensor Cores.

## Why the same convolution runs faster

A modern NVIDIA GPU has two ways to multiply matrices. The general CUDA cores do it one fused-multiply-add at a time. The **Tensor Cores** do it a whole small tile at a time — a matrix-multiply-accumulate, or **MMA**, that consumes, say, a 16×16 by 16×8 tile of low-precision inputs and produces a 16×8 accumulator in a single hardware instruction. On an A100 that is the difference between roughly 19.5 fp32 TFLOP/s on the CUDA cores and about 312 bf16 TFLOP/s on the Tensor Cores — more than an order of magnitude. Any kernel that wants to be fast on convolutions in fp16 or bf16 *must* feed the Tensor Cores. That is not optional; it is where the FLOP/s live.

Now, cuDNN does not implement convolution as a literal sliding window. It implements it as an **implicit GEMM**: it treats the convolution as one big matrix multiply where the contraction (the "K" dimension that gets summed over) is the input channels times the filter's spatial extent, `C × R × S`. The Tensor Core MMA loads its input fragments *along that K dimension*. For the load to be efficient — contiguous, coalesced, and mapped cleanly onto the fragment layout the MMA expects — the channel dimension needs to be innermost in memory. That is exactly what `channels_last` provides and exactly what NCHW does not.

So when you hand cuDNN an NCHW fp16 tensor and ask for a convolution, it faces a decision. It can pick a native NCHW kernel, which either avoids the Tensor Cores or uses them inefficiently, and eat the slowdown. Or it can **transpose your tensor to NHWC first**, run the fast Tensor Core kernel, and transpose the output back. The heuristic makes this choice per layer based on shapes and its internal cost model. Either branch costs you. The figure below draws the fork explicitly.

![a branching dataflow where an NCHW input either takes a slow kernel or pays a transpose before the fast kernel while a channels last input reaches the tensor core kernel directly](/imgs/blogs/channels-last-and-memory-formats-4.webp)

The transpose branch is the one worth quantifying, because it is pure waste — it moves bytes and computes nothing. A transpose reads the whole tensor and writes the whole tensor, so it moves $2B$ bytes for a tensor of $B$ bytes, and it is entirely memory-bound. Its time is governed by HBM bandwidth:

$$t_\text{transpose} = \frac{2B}{\text{BW}_\text{HBM}}$$

This is the same roofline reasoning the series leans on throughout — an operation with essentially zero arithmetic intensity $\text{AI} = \text{FLOPs}/\text{bytes} \approx 0$ can only ever run at the bandwidth ceiling, never the compute ceiling. If you want that model in full, [the roofline for your service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) is the sibling post. Here, plug in numbers.

#### Worked example: the transpose tax on one conv layer

Take a mid-network activation in a ResNet-style trunk: `N=32, C=256, H=56, W=56`, fp16 (2 bytes). Its size is

$$B = 32 \times 256 \times 56 \times 56 \times 2 \approx 51.4\ \text{MB}.$$

One transpose moves $2B \approx 103\ \text{MB}$. On an A100 with HBM2e at 2.0 TB/s:

$$t_\text{transpose} = \frac{103 \times 10^6}{2.0 \times 10^{12}} \approx 51\ \mu s.$$

If cuDNN transposes the input in *and* the output back out, that is roughly 100 µs of pure bandwidth waste on this single layer, doing zero math. A ResNet-50 has 53 convolutions. You will not pay a transpose on every one — cuDNN often keeps a run of NCHW layers on a native NCHW kernel to avoid churning formats — but on the layers where it does convert, the tax stacks into the hundreds of microseconds to low milliseconds per forward pass. That is the 4.2 ms versus 3.1 ms convolution total from the opening figure: the 1.1 ms delta is transpose traffic plus the gap between a native NCHW kernel and the direct NHWC Tensor Core kernel. Convert the whole network to `channels_last` up front and cuDNN's cheapest path is the direct fast kernel with no transposes at all — the bytes are already in the order the Tensor Core wants.

The intuition to keep: `channels_last` does not make the convolution "do less math." It makes the *fast kernel eligible without a copy*. You are removing a bandwidth-bound tax and upgrading the kernel selection in one move, which is why it shows up as both lower HBM traffic and higher SM efficiency in a profile.

#### The convolution as one big matrix multiply

The implicit-GEMM formulation is worth making concrete, because it is the reason the channel axis is special rather than just another dimension. A convolution with `C_in` input channels, `C_out` output filters, and an `R×S` kernel, applied to an `H×W` feature map, is algebraically a single matrix multiply. Flatten every output spatial position into a row — there are `N · H_out · W_out` of them — and flatten every `(input-channel, kernel-row, kernel-col)` triple into a column, of which there are `C_in · R · S`. The convolution is then that `[N·H_out·W_out] × [C_in·R·S]` activation matrix times the `[C_in·R·S] × [C_out]` weight matrix. The contraction dimension, the one that gets summed over and is conventionally called K in GEMM notation, is `C_in · R · S`.

The Tensor Core MMA streams its input fragments *along K*. For that stream to be contiguous and to pack cleanly into the fragment layout the hardware expects, the `C_in` factor of K must be adjacent in memory — and that is precisely NHWC. In NCHW, two K indices that differ only in their channel component are `H · W` elements apart, so the MMA's operand load becomes a strided gather and the entire efficiency case for using a Tensor Core evaporates. The one-sentence version: the Tensor Core sums over channels, so channels should be contiguous. cuDNN would rather spend a transpose to make that true than run the MMA against strided memory, which is exactly the fork the earlier figure drew.

There is a sharp corollary that bites in practice — **channel alignment**. The Tensor Core fp16 path wants the channel count to be a multiple of 8, and a multiple of 16 packs even better, so the MMA tiles divide the channel dimension without a remainder. A convolution whose `C_in` or `C_out` is 3, 20, or 100 either falls off the fast path or pads internally, and no memory format can rescue a badly-aligned channel count. The very first convolution of most vision models consumes a 3-channel RGB image, and that specific layer will not use a Tensor Core regardless of layout, because 3 is not a multiple of 8 — which is fine, it is one small, cheap layer. But if a custom architecture threads odd channel widths through its whole trunk, check the alignment before you blame the layout: `channels_last` gets the bytes in the right order, but the MMA still needs the shapes to cooperate.

## Where the layout win lives (and where it does not)

Before wiring this into a service, be honest about when it pays. `channels_last` is not free lunch for every model — it is a specific win for a specific shape of workload, and knowing the boundary keeps you from cargo-culting it onto a Transformer and reporting a rounding-error "speedup." The figure below is the decision at a glance.

![a matrix rating the layout win as large for convolutional networks in fp16 on Ampere small for fp32 little for Transformers and none for tensors already in NHWC](/imgs/blogs/channels-last-and-memory-formats-5.webp)

The **large** win is a convolution-heavy network in fp16 or bf16 on Ampere-or-later Tensor Cores: ResNets, EfficientNets, ConvNeXts, U-Nets, detection and segmentation backbones. These are dominated by `conv2d`, `conv2d` is exactly the op with an NHWC Tensor Core fast path, and low precision makes that path eligible. Reported and reproduced speedups land in the 1.2 to 1.6x range on end-to-end forward time depending on the network and batch, with the convolution-only fraction improving more.

The **small** win is the same convolutional network in **fp32**. Plain fp32 convolution does not use the Tensor Cores at all, so the biggest lever — the NHWC MMA fast path — is not even on the table. You still get some benefit from better coalescing and fewer transposes, but it is often within noise. A caveat worth knowing: on Ampere, fp32 convolutions can opt into **TF32** Tensor Cores (a 19-bit reduced-mantissa format), and `channels_last` does help that path — so "fp32 with TF32 enabled" sits between small and large. If you are running true fp32 with TF32 disabled, expect near nothing.

The **little** win is a pure Transformer. Its heavy ops are already `matmul` / `linear` — batched GEMMs on 2D and 3D tensors, which do not have a channel axis and are largely layout-agnostic because the GEMM kernels handle their own tiling. Attention runs through fused kernels like scaled dot-product attention regardless of `channels_last`. Some elementwise and normalization ops on 4D tensors inside a hybrid model can still benefit, but a text Transformer sees essentially nothing. And the **none** case: a tensor that is *already* NHWC (some data pipelines and camera/codec sources deliver NHWC natively) has nothing to convert — `channels_last` is a no-op and any further conversion would be a wasted copy.

| Workload | Precision | Layout win | Root cause |
|---|---|---|---|
| ResNet / U-Net / detector | fp16 / bf16 | Large (1.2–1.6x) | NHWC Tensor Core conv, no transpose |
| Same CNN | fp32 (TF32 on) | Medium | TF32 Tensor Cores engage on NHWC |
| Same CNN | fp32 (TF32 off) | Near zero | No Tensor Core path for fp32 conv |
| Text Transformer | fp16 / bf16 | Little | GEMM/attention are layout-agnostic |
| Already-NHWC input | any | None | Nothing to convert |

The rule that falls out: reach for `channels_last` when your service is convolution-bound and running low precision on Tensor Cores. If it is a Transformer or you are stuck in fp32, spend your effort elsewhere — the [bandwidth-bound-and-fusion](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion) sibling covers the memory-wall fixes that *do* move the needle on those.

#### Beyond 2D: channels_last_3d for video and volumes

The same idea extends to 5D tensors. A 3D convolution operates on `(N, C, D, H, W)` — batch, channels, depth, height, width — for video models, volumetric medical imaging, and 3D scene work. Its default layout is NCDHW; the channels-innermost variant is NDHWC, and PyTorch exposes it as `torch.channels_last_3d`. Everything in this post transfers: the strides put channels at stride 1, the cuDNN 3D convolution Tensor Core kernels prefer NDHWC, and the same transpose tax appears when you feed NCDHW. If anything the win is *larger* for 3D, because the tensors are bigger (a depth dimension multiplies the byte count) so the transpose traffic you delete is proportionally heavier. The conversion call is identical in spirit: `x = x.to(memory_format=torch.channels_last_3d)` and `model = model.to(memory_format=torch.channels_last_3d)`. Watch the same failure mode — a 5D-incompatible reshape reverts you to NCDHW just as readily.

#### How this composes with torch.compile

If you are already compiling the model, the layout story does not disappear — it moves. Inductor, the default `torch.compile` backend, runs a layout-optimization pass and will frequently choose `channels_last` for convolutions on its own when it decides that path is cheaper, so a compiled model sometimes captures part of this win without any explicit call. Two caveats keep the explicit conversion worthwhile. First, Inductor decides *inside* the compiled region, but it does not control the layout of the *input you hand it* — if you feed an NCHW tensor, the first thing the graph may do is convert it, and doing that conversion yourself device-side (folded into the H2D copy) is cheaper and clearer. Second, making the intent explicit means the profiler and the stride checks below tell a consistent story; when the layout is half-chosen by a compiler pass and half by your code, debugging a revert is harder. The pragmatic recipe: convert the model and inputs to `channels_last` explicitly, *then* compile — the two compose, and the explicit conversion documents what you expect Inductor to preserve.

## The one-line API and how it propagates

The mechanics are genuinely two lines: convert the model's weights, and convert each input. Everything format-aware in between preserves the layout on its own.

```python
import torch, torchvision

device = "cuda"
model = torchvision.models.resnet50(weights="DEFAULT").eval().to(device)

# 1) convert the model's parameters and buffers to channels_last
model = model.to(memory_format=torch.channels_last)

# 2) convert each input batch to channels_last (do it on the GPU)
x = torch.rand(32, 3, 224, 224, device=device)
x = x.to(memory_format=torch.channels_last)

with torch.no_grad():
    y = model(x)

print("output stride NHWC-contig:",
      y.is_contiguous(memory_format=torch.channels_last))
```

Converting the model reorders the convolution weights so cuDNN reads them in the layout its fast kernel wants. Converting the input puts the activation in NHWC so the first convolution starts on the fast path. From there, the format is *sticky*: `conv2d`, `batch_norm`, `relu`, `max_pool2d`, `adaptive_avg_pool2d`, and elementwise `add` (as in a residual, when both operands share the format) are all format-aware — given a `channels_last` input they produce a `channels_last` output. So a residual block stays NHWC end to end without you touching it. That propagation is the whole reason two lines suffice for a 53-layer network.

But some operators are *not* format-aware, and they are where the win leaks out. Any op that internally calls `.contiguous()` (defaulting to NCHW), certain `view`/`reshape` calls that cannot express the NHWC strides, older or custom CUDA ops that only handle contiguous NCHW, a `.cpu()` round-trip, some interpolation and grid-sample paths depending on the version — these hand back an NCHW tensor. When that happens mid-network, every downstream convolution sees NCHW again and cuDNN is back to the slow-kernel-or-transpose fork. The network still produces correct numbers; it just quietly stops being fast. This is the single most common way a `channels_last` rollout reports "no speedup" — the layout took at the input and died in block 3.

The combined recipe that actually engages Tensor Cores is `channels_last` *plus* autocast to a low precision. `channels_last` chooses the NHWC kernel; autocast makes the math fp16/bf16 so that kernel is a Tensor Core kernel:

```python
model = model.to(device, memory_format=torch.channels_last)

def infer(batch):
    batch = batch.to(device, memory_format=torch.channels_last,
                     non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
        return model(batch)
```

Two independent switches, and you need both. `channels_last` in fp32 (TF32 off) gives you the NHWC kernel but not the Tensor Cores — small win. Autocast in NCHW gives you Tensor Cores but forces a transpose to reach them — you pay back part of the win in copies. Together they give you the direct NHWC Tensor Core kernel, which is the whole prize. This is also why the two often ship together in serving-optimization guides; if you are also stacking `torch.compile` on top, note that Inductor is layout-aware and will frequently choose `channels_last` for convs on its own, but doing the conversion explicitly makes the intent legible and guarantees the input side is right.

| Op | Preserves channels_last? |
|---|---|
| `conv2d`, `conv_transpose2d` | Yes (this is the point) |
| `batch_norm2d`, `group_norm` | Yes |
| `relu`, `gelu`, elementwise unary | Yes |
| `add` (residual, matching formats) | Yes |
| `max_pool2d`, `adaptive_avg_pool2d` | Yes |
| `view` / `reshape` (NHWC-incompatible) | No — reverts to NCHW |
| `flatten` to 2D before the classifier | Collapses spatial; no longer 4D |
| ops that call `.contiguous()` internally | No — forces NCHW |
| custom / legacy CUDA ops (NCHW-only) | No — reverts to NCHW |

## channels_last in training, not just inference

Everything so far framed inference, but the layout win applies to training too, and often matters *more* there because a training step runs the convolutions three times — forward, plus the two backward passes (gradient with respect to input, and gradient with respect to weights). Both backward convolutions have the same NHWC Tensor Core fast path and the same NCHW transpose tax, so a fully-converted training step compounds the saving across all three. The rule is the same and just as short: convert the model once, and convert each input batch. The gradients then inherit the format from the activations they are computed against, so you do not convert them by hand.

```python
model = torchvision.models.resnet50(num_classes=1000).to(
    "cuda", memory_format=torch.channels_last)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scaler = torch.amp.GradScaler("cuda")

for images, targets in loader:
    images = images.to("cuda", memory_format=torch.channels_last,
                       non_blocking=True)
    targets = targets.to("cuda", non_blocking=True)

    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = criterion(model(images), targets)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

Three things about this loop are worth calling out. The `targets` tensor is **not** converted — it is a 1D label vector with no channel axis, so `channels_last` is meaningless for it and the call would be a no-op or an error; only the 4D image tensor takes the format. The `GradScaler` and `autocast` are the standard AMP machinery, and they are what make the fast kernel a Tensor Core kernel, exactly as in inference. And the same silent-revert risk exists in the backward graph: a custom autograd `Function` that returns a `.contiguous()` gradient reverts the format for every upstream convolution's weight-gradient computation, so the stride-check discipline below is not inference-only. If a training run adopts `channels_last` and the step time barely moves, run the forward-hook check on a single batch first — the leak is usually one custom op.

One footnote on memory: `channels_last` does not increase peak memory. The tensor is the same number of bytes in a different order, so the activation and gradient footprints are unchanged. The only transient cost is the one-time conversion copy when you call `.to(memory_format=...)`, which is a single memory-bound pass over the tensor and is dwarfed by the per-step savings once training is running.

## Did the layout hold end to end?

You cannot trust that `channels_last` took just because you called `.to()` twice. You have to check, and there are two levels of checking: cheap stride assertions in code, and the ground truth in the profiler. Do the cheap one first. The figure below is the decision tree — the sequence of yes/no questions that separates a network that stayed NHWC all the way through from one that reverted.

![a decision tree asking whether the input was converted the model was converted and whether any breaking operator reverted the tensor with a single no dropping back to NCHW](/imgs/blogs/channels-last-and-memory-formats-6.webp)

The most surgical way to find a revert is a forward hook on every module that inspects the output stride. This walks the whole network in one pass and prints exactly which module dropped the format:

```python
def make_hook(name):
    def hook(module, inputs, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        if t.dim() == 4 and not t.is_contiguous(
                memory_format=torch.channels_last):
            print(f"[REVERT] {name:<28} stride={tuple(t.stride())} "
                  f"({module.__class__.__name__})")
    return hook

handles = [m.register_forward_hook(make_hook(n))
           for n, m in model.named_modules() if len(list(m.children())) == 0]

with torch.no_grad():
    _ = model(x)          # x is channels_last

for h in handles:
    h.remove()
```

On a clean ResNet-50 that has been fully converted, this prints nothing — every 4D activation stays NHWC-contiguous. If instead you had, say, a custom attention or reshape block spliced into the trunk, you would see something like this, and the culprit is named:

```console
[REVERT] layer3.0.custom_reshape     stride=(200704, 3136, 56, 1) (Reshape)
[REVERT] layer3.1.conv1              stride=(200704, 3136, 56, 1) (Conv2d)
[REVERT] layer3.1.conv2              stride=(200704, 3136, 56, 1) (Conv2d)
```

The stride `(200704, 3136, 56, 1)` is the NCHW signature — channel stride 3136 = 56×56, the large one. The first line is the reshape that reverted the format; every convolution *after* it inherited NCHW and is now back on the slow fork. The fix is targeted: re-convert right after the offending op with `t = t.contiguous(memory_format=torch.channels_last)`, or replace the reshape with a format-preserving equivalent. One line at the leak restores the rest of the network.

That is the code-level check. It tells you the *strides* are right, but it does not prove cuDNN actually picked the Tensor Core kernel — for that you read the trace.

## Reading the trace to confirm it took

Strides are necessary but not sufficient. The definitive proof that `channels_last` bought you anything is in the kernel names and timings the profiler reports. Capture a steady-state window with the standard `torch.profiler` block — the same tool the series builds on in [profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler):

```python
from torch.profiler import profile, ProfilerActivity

# warm up first — cuDNN benchmarks and picks kernels on early iters
for _ in range(10):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        _ = model(x)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(20):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _ = model(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

Two things distinguish a run where the layout took from one where it reverted, and the figure below contrasts them: the **kernel names** and the **presence of transpose kernels**.

![a two column comparison of a reverted trace showing nchw kernels and explicit transpose copies against a channels last trace showing nhwc tensor core kernels and zero transpose kernels](/imgs/blogs/channels-last-and-memory-formats-7.webp)

Here is the top of the table for the run that took. Notice the convolution kernel names contain `nhwc` and an MMA tile tag like `s16816` (the sm80 16×8×16 Tensor Core shape), and there is *no* transpose kernel anywhere in the list:

```console
-------------------------------------------------  ------------  ----------
Name                                                 CUDA total    # of Calls
-------------------------------------------------  ------------  ----------
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_          3.11 ms          1060
    ..._nhwc_tilesize128x128...
void cudnn::bn_fw_inf_1C11_kernel_NHWC<...>            0.42 ms           720
void at::native::vectorized_elementwise_...           0.28 ms           980
void cudnn::maxpool_nhwc<...>                          0.09 ms            20
-------------------------------------------------  ------------  ----------
Self CUDA time total: 5.31 ms
```

And here is the same model when a reverting op crept in. The convolution kernels lost the `nhwc` tag, an explicit `nchwToNhwc` / `nhwcToNchw` transpose kernel appears (that is the copy tax made visible), batch-norm fell back to its NCHW kernel, and the total is up:

```console
-------------------------------------------------  ------------  ----------
Name                                                 CUDA total    # of Calls
-------------------------------------------------  ------------  ----------
sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_          3.98 ms          1060
    ..._nchwkrsc_nchw...
void cudnn::ops::nchwToNhwcKernel<__half>             0.61 ms           410
void cudnn::ops::nhwcToNchwKernel<__half>             0.34 ms           410
void cudnn::bn_fw_inf_1C11_kernel_new<...>            0.55 ms           720
-------------------------------------------------  ------------  ----------
Self CUDA time total: 6.79 ms
```

Read those two tables side by side and the story is unambiguous. The `nhwc` kernels versus `nchw` kernels tell you which fast path ran. The `nchwToNhwcKernel` / `nhwcToNhwcKernel` rows — 0.95 ms of them here — are the transpose tax, and their *absence* is your confirmation the layout is clean. And the convolution self-time (3.11 ms versus 3.98 ms) plus the total (5.31 ms versus 6.79 ms) quantify the win. If your strides say NHWC but you still see transpose kernels in the trace, a downstream op is reverting and cuDNN is bouncing formats back and forth — go back to the forward-hook check and find it. This is the discipline: the code check finds *where* the format broke; the trace confirms *whether cuDNN cared*.

To confirm the *bandwidth* half of the win rather than just the kernel names, drop down to Nsight Compute on a single convolution. `ncu` reports the memory-throughput and Tensor Core utilization for the exact kernel, so you can see the effective DRAM traffic fall and the Tensor Core pipe activate:

```bash
# profile just the forward convolution kernels, full metric set
ncu --set full \
    --kernel-name-base demangled \
    -k "regex:fprop|nchwToNhwc|nhwcToNchw" \
    --launch-count 20 \
    python infer_one_batch.py
```

For the `channels_last` run, the Speed-of-Light section shows the convolution kernel at high Tensor Core utilization with no companion transpose kernels; for the NCHW run, the same section shows a lower compute pipe, a memory workload dominated by the transpose kernels, and DRAM read bytes that are visibly higher for the same math:

```console
channels_last  fprop_implicit_gemm  DRAM read 2.01 GB  Tensor pipe 71%  transpose kernels 0
NCHW           fprop_implicit_gemm  DRAM read 2.58 GB  Tensor pipe 44%  transpose kernels 820
```

Those two lines are the mechanism made measurable: the same convolution, 22% fewer DRAM bytes read and a Tensor Core pipe that went from lightly used to busy, with the 820 transpose-kernel launches on the NCHW side accounting for the difference. Nsight Compute is heavier than the profiler table — it serializes and replays kernels to gather metrics, so use it on one kernel, not a whole serving loop — but it is the tool that turns "the trace looks faster" into "the DRAM traffic is down and the Tensor Cores are engaged." The [profiling PyTorch](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) sibling covers the lightweight table; `ncu` is the microscope for the one kernel you care about.

#### Worked example: A100 ResNet-50 service, NCHW → channels_last

Put the whole thing on named hardware. ResNet-50, batch 32, 224×224 input, fp16 autocast, A100 80GB SXM, measured with CUDA events around a synchronized steady-state loop of 200 iterations after 30 warmups (locked clocks, no dataloader in the timed path). The table is the before→after the opening figure promised:

| Metric | NCHW (contiguous) | channels_last | Change |
|---|---|---|---|
| p50 latency / batch | 6.79 ms | 5.31 ms | −22% |
| Throughput | 4,712 img/s | 6,026 img/s | +28% |
| Conv kernel self-time | 3.98 ms | 3.11 ms | −22% |
| Transpose kernel time | 0.95 ms | 0.00 ms | eliminated |
| Batch-norm kernel time | 0.55 ms | 0.42 ms | −24% |
| Tensor Cores on conv | partial | yes | upgraded |
| HBM read per batch | ~2.6 GB | ~2.0 GB | −23% |
| Accuracy (top-1) | unchanged | unchanged | 0 |

Every row traces to the trace you just read. The 22% p50 improvement is dominated by two effects: deleting the 0.95 ms of transpose kernels outright, and running the convolutions on the direct NHWC Tensor Core kernel instead of a native NCHW one. The HBM read dropping ~23% is those transposes no longer streaming the activations through memory for nothing. And the accuracy line is the reason this optimization is a no-brainer where it applies: it is a pure layout change, mathematically identical, so there is no correctness risk to weigh against the speed. Note also that throughput improved *more* than latency (28% vs 22%) — that is the usual pattern when you remove a fixed per-batch tax, because the saved time is a larger fraction of the throughput accounting.

How to measure this honestly, per the series' rules: warm up until cuDNN has finished its kernel autotuning (the first several iterations are not representative — cuDNN benchmarks kernel candidates), call `torch.cuda.synchronize()` before reading the clock so you are not timing an async launch queue, use CUDA events rather than wall time to avoid host jitter, lock the GPU clocks with `nvidia-smi -lgc` to remove boost-throttle noise, keep the dataloader out of the timed region so you are measuring the model and not the input pipeline, and report a percentile over a steady-state window rather than a single call. If you skip the warmup you will measure cuDNN's autotuning, not your kernel; if you skip the sync you will measure Python, not the GPU. The [setting-up-a-reproducible-benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) sibling is the full checklist.

#### Worked example: the same change on an L4 inference GPU

Repeat the measurement on an L4 — a common cost-efficient inference card with roughly 242 fp16 TFLOP/s of Tensor Core throughput but only about 300 GB/s of memory bandwidth, versus the A100's 2.0 TB/s. The relative win holds and the *mechanism* sharpens, because the L4's much lower bandwidth makes the transpose tax hurt proportionally more per byte moved. Same ResNet-50, batch 16 (a more typical L4 batch), fp16 autocast:

| Metric | NCHW | channels_last | Change |
|---|---|---|---|
| p50 latency / batch | 11.4 ms | 8.9 ms | −22% |
| Throughput | 1,404 img/s | 1,798 img/s | +28% |
| Transpose kernel time | 1.7 ms | 0.00 ms | eliminated |
| Tensor Cores on conv | partial | yes | upgraded |

The transpose kernels cost 1.7 ms on the L4 versus 0.95 ms on the A100 for a smaller batch — that is the 300 GB/s bandwidth showing up directly, since $t_\text{transpose} = 2B / \text{BW}_\text{HBM}$ scales inversely with bandwidth. The takeaway generalizes: `channels_last` is not an A100-only trick. On the cheaper, bandwidth-starved inference GPUs where a lot of serving actually runs, deleting the transpose traffic matters *more*, not less, because those cards are closer to their bandwidth ceiling to begin with. Run the numbers on your actual serving hardware; the relative win tends to be similar, and the absolute millisecond savings often larger on the smaller card.

## Stress-testing the win

A speedup you cannot break is a speedup you do not understand. Push on it from several directions and watch it hold or fold — this is how you learn its real shape.

**Batch 1 versus batch 64.** At batch 1, the convolutions are smaller and closer to memory-bound; the transpose tax is proportionally smaller (less data) but so is the conv time, and the fixed per-kernel launch overhead starts to dominate — the layout win shrinks to maybe 8–12% because you are no longer bandwidth-limited on the transposes. At batch 64, the convolutions are firmly compute-bound and the Tensor Core fast path matters most; the win is at its largest, often the full 1.2–1.6x. The lesson: `channels_last` pays best exactly when you are pushing throughput, which is when serving usually runs it anyway. If your service is latency-bound at batch 1, measure — the win is real but smaller, and other levers (CUDA graphs to kill launch overhead) may matter more.

**fp32 with no TF32.** Flip autocast off and run the same ResNet-50 in true fp32. Convert to `channels_last` and re-measure: the improvement collapses to a few percent, within noise on some layers. Read the trace and you will see the convolution kernels no longer carry the `nhwc`+`s16816` Tensor Core tag — they are CUDA-core fp32 kernels, because fp32 convolution has no Tensor Core path (short of TF32). The strides are correct, the format took, and it barely mattered. This is the diagnostic that teaches the mechanism: the win was never "NHWC is faster," it was "NHWC lets the Tensor Core kernel run without a copy." Remove the Tensor Cores and you remove most of the prize.

**A Transformer.** Take a ViT or a text Transformer, wrap the same two `.to(memory_format=...)` calls around it, and measure. On a pure text Transformer you will see essentially nothing — the heavy ops are `linear` and attention, which do not have a channel axis and route through layout-agnostic GEMM and fused-attention kernels. On a ViT you might see a small win on the patch-embedding convolution and the occasional 4D norm, but the transformer blocks that dominate its runtime are unaffected. If you *reported* a big speedup here, suspect you changed something else (autocast, a warm cache) — `channels_last` alone on a Transformer is a rounding error, and knowing that saves you from chasing it.

**Varying input shapes per request.** Unlike `torch.compile` or CUDA graphs, `channels_last` does not care whether the input shape changes between requests — it is a property of the stride pattern, not of a captured graph, so a service that sees 224×224, 256×256, and 320×320 images in the same batch stream keeps the fast layout on all of them with no recompilation and no per-shape setup. This is a genuine advantage in dynamic-shape serving: the layout win composes with variable resolution for free, where graph-based optimizations force you into bucketing. The only per-shape cost is cuDNN's own kernel autotuning the first time it sees a new shape, which is orthogonal to the format.

**The silent-revert stress test.** Deliberately splice a `.contiguous()` (NCHW default) into the middle of the trunk and re-run the forward hook and the trace together. The hook fires `[REVERT]` at that op; the trace grows transpose kernels from that point downstream; the p50 climbs back toward the NCHW baseline. This is the failure mode in a controlled setting, and rehearsing it once means you recognize it instantly in production: "we rolled out `channels_last` and saw 3%" almost always means one op is reverting, not that the optimization does not work. Fix the leak, and the 22% comes back.

**Under concurrency.** Run the converted model under 50 concurrent request streams sharing one GPU and the per-request win holds, but its character shifts: the GPU is now closer to saturation, so the deleted transpose traffic frees bandwidth that other in-flight requests immediately consume, and aggregate throughput rises rather than each request getting individually faster. In other words, at low load `channels_last` shows up as lower latency; at high load it shows up as higher throughput, because the bandwidth you stopped wasting on transposes is exactly the resource the concurrent requests were contending for. Either way the mechanism is the same — you removed a memory-bound tax — but the metric it improves depends on where the service sits on its load curve.

## Case studies and real numbers

**The PyTorch Channels Last tutorial (Volta and Ampere).** The canonical reference — PyTorch's "Channels Last Memory Format in PyTorch" deep-dive tutorial by Vitaly Fedyunin — is where this API and its stride semantics are documented, including the exact `(1,3,4,4)` stride example this post uses and the `model.to(memory_format=torch.channels_last)` recipe. It demonstrates the conversion propagating through a torchvision classifier and shows meaningful end-to-end speedups on Tensor Core GPUs when combined with AMP. If you take one further-reading link from this post, take that one; every claim here about the API surface is grounded in it.

**cuDNN's NHWC preference.** NVIDIA's cuDNN documentation and its convolution performance guidance state plainly that the Tensor Core convolution kernels are optimized for the NHWC layout and that feeding NCHW data can trigger internal format transformations. This is not a PyTorch quirk — it is a property of how cuDNN implements implicit-GEMM convolution on Tensor Cores, and it is why the `nchwToNhwcKernel` transpose shows up in traces of NCHW fp16 convolutions. The 51 µs-per-transpose figure in the worked example is a direct application of the tensor size and the A100's 2.0 TB/s HBM bandwidth from NVIDIA's A100 spec.

**Segmentation and detection backbones.** Convolution-dominated dense-prediction models — U-Nets for segmentation, feature-pyramid detectors — are where `channels_last` reliably lands in the upper half of the 1.2–1.6x range, because a larger fraction of their runtime is convolution and their activation tensors are large enough that the transpose tax and coalescing effects are both significant. Teams shipping these in fp16 on Ampere routinely report double-digit-percent latency reductions from the two-line change, matching the mechanism: more conv fraction plus low precision equals more win. For serving-side context on stacking this with other kernel-selection wins, the model-serving series' treatment of [kernel fusion, CUDA graphs, and torch.compile](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) composes cleanly on top.

**The channel-alignment false negative.** A recurring support pattern: a team converts a custom CNN to `channels_last`, measures no improvement, and concludes the optimization is a myth. The forward-hook check shows the strides are correct end to end — the format took. The trace shows the convolutions are *not* on the Tensor Core path anyway, because the architecture uses channel widths like 3, 12, or 100 that are not multiples of 8. The layout was never the bottleneck; the channel alignment was. The lesson bundles the two facts: `channels_last` is necessary for the fast path but not sufficient — the channel count has to be Tensor-Core-friendly too. When the layout is verified and the win is still absent, check `C_in` and `C_out` against the multiple-of-8 rule before spending another day on it.

**Inductor's automatic layout choice.** In compiled inference, part of this win can arrive without the explicit conversion, because Inductor's layout pass elects `channels_last` for convolutions when its cost model favors it. Reports of `torch.compile` speeding up a CNN service often bundle a layout change the team never wrote by hand — which is a good default, but it also means a compiled model whose *input* is still NCHW pays a conversion on entry. Converting the input yourself, device-side, removes that entry cost and makes the trace legible. The composition — explicit `channels_last`, then `torch.compile` — is the belt-and-suspenders recipe several production serving stacks converge on.

## When to reach for this (and when not to)

Reach for `channels_last` when your service is a convolution-heavy vision model running in fp16 or bf16 on an Ampere-or-newer GPU. That is the sweet spot, the change is two lines, the risk is zero because the math is identical, and the payoff is commonly 15–30% end-to-end. It is one of the highest-return-per-line optimizations in the entire performance toolbox, and it should be near the top of your checklist for any CNN inference service — try it before you reach for anything exotic.

Do **not** reach for it, or do not *expect* much, in these cases. Do not bother on a pure text Transformer — there is nothing to convert that matters. Do not expect a win in true fp32 with TF32 disabled — you get the NHWC kernel but not the Tensor Cores, so the biggest lever is absent; enable TF32 or autocast first, then `channels_last` earns its keep. Do not assume it took — a single format-breaking op mid-network silently reverts you to NCHW, and a rollout that "did nothing" is almost always a leak, not a dud; always run the forward-hook stride check and confirm the kernel names in the trace. Do not convert inputs that are already NHWC — that is a wasted copy. And do not convert on the CPU host and then ship over PCIe if you can convert on the GPU instead — the conversion is itself a memory-bound copy, so do it device-side where the bandwidth is 10x higher, ideally folded into the same `.to(device, memory_format=..., non_blocking=True)` call.

The deeper principle the series keeps returning to: layout is a first-class performance knob, not an implementation detail. The same numbers, ordered differently in memory, select different kernels and burn different bandwidth. `channels_last` is the vision-model instance of that principle; the memory-bandwidth and fusion post is the elementwise-and-attention instance; and the whole optimization loop — profile, find the waste, change one thing, re-measure — is captured in [the performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) capstone.

## Key takeaways

- **A memory format is a stride pattern, not a shape.** `channels_last` keeps the logical `(N,C,H,W)` shape and the byte count identical; it only changes the stride vector so channels become innermost (NHWC in memory). Same values, different order.
- **Layout selects the kernel.** cuDNN's fast Tensor Core convolution kernels want channels innermost. Hand them NCHW and they either run a slower kernel or transpose your data first — a bandwidth-bound copy that computes nothing.
- **The win is bandwidth plus kernel selection, tied to the roofline.** You delete the ~2B-byte transpose traffic and upgrade to the direct NHWC Tensor Core kernel. That is why it shows up as both lower HBM reads and higher SM efficiency.
- **Two lines, both required:** `model.to(memory_format=torch.channels_last)` and `input.to(memory_format=torch.channels_last)`, plus autocast to fp16/bf16 so the NHWC kernel is actually a Tensor Core kernel.
- **It propagates, until it doesn't.** conv, batch-norm, relu, pooling, and residual-add preserve the format; a stray `.contiguous()`, an NHWC-incompatible reshape, or a legacy op reverts it and every downstream conv pays again.
- **Verify in two places.** Stride-check every module output with `is_contiguous(memory_format=torch.channels_last)` to find leaks; confirm in the profiler that conv kernels carry the `nhwc` tag and that no transpose kernels appear.
- **Know the boundary.** Large win for fp16 CNNs on Ampere; small for fp32-without-TF32; little for Transformers; none for already-NHWC data. Do not cargo-cult it where the mechanism cannot fire.
- **A "no speedup" almost always means a silent revert.** Before concluding the optimization does not work, find the one op that dropped the format — the win is usually still there once the leak is closed.

## Further reading

- [Channels Last Memory Format in PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) — the primary source: the stride semantics, the `(1,3,4,4)` example, and the model-conversion recipe this post is grounded on.
- [NVIDIA cuDNN Developer Guide — Tensor Core convolutions and NHWC](https://docs.nvidia.com/deeplearning/cudnn/) — why the fast convolution kernels prefer NHWC and when internal format transforms happen.
- [NVIDIA Deep Learning Performance Guide — convolutional layers](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) — the tiling, channel-alignment, and Tensor Core eligibility rules behind the kernel selection.
- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the four wastes this post attacks: the bandwidth wall and bad kernel selection.
- [The roofline for your service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) — arithmetic intensity and why a zero-FLOP transpose can only run at the bandwidth ceiling.
- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — the schedule, `key_averages()`, and reading the kernel table you used to confirm the layout took.
- [Bandwidth-bound kernels and fusion](/blog/machine-learning/performance-engineering/bandwidth-bound-and-fusion) — the same memory-wall reasoning applied to elementwise and attention, for the non-conv parts of your model.
- [Inside the GPU: SMs, warps, and the SIMT execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) — why coalesced, channel-innermost access maps onto the hardware and NCHW's strided access does not.
