---
title: "The selective-scan and delta-rule decode kernel: one step, one launch"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Write the one-step recurrent update that hybrid models run instead of attention, and learn why this kernel is starved by launch overhead and low occupancy rather than by memory bandwidth."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "triton",
    "cuda",
    "mamba",
    "linear-attention",
    "kernel-fusion",
    "gpu",
    "pytorch",
    "ml-systems",
    "latency",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
---

You swapped three quarters of your attention layers for Mamba-2 blocks. The memory math came out beautifully — the per-request cache stopped growing with context, 128k prompts stopped being a capacity planning exercise, and the block allocator got a whole class of problem removed from it. Then you measured single-stream latency and decode got *slower*. Not by a rounding error. Slower, on a model that does strictly less work per token than the dense one it replaced.

This is the most confusing performance result in hybrid serving, and it has an almost embarrassing explanation. The recurrent layer's per-step arithmetic is so small, and the memory it touches so tiny, that neither one is what you are paying for. You are paying for the *act of asking the GPU to do the work*: fifteen kernel launches per layer, each one a few microseconds of driver and scheduler time, wrapped around a couple of microseconds of actual computation. Twenty-four layers of that and you have burned a millisecond per token on ceremony. The attention layers you kept are bandwidth-bound and behave the way you expect. The layers you added are latency-bound and behave like nothing else in the engine.

![Dataflow graph showing a hidden vector fanning out into a convolution update, a decay gate and two projections that all merge into a single state read-modify-write and one store](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-1.webp)

This post writes the kernel that fixes it: `nanoserve/kernels/ssd_decode.py`, a `@triton.jit` single-step selective scan that keeps the state tile in registers, folds the convolution update, the gates, the output projection and the norm into one pass, and writes the state back exactly once. Then the same shape for the gated delta rule, which is the other half of the hybrid world. You will get a correctness harness that diffs both against a pure-PyTorch oracle, the occupancy arithmetic that tells you how many states a streaming multiprocessor can hold, and a measurement script that separates launch cost from memory cost so you can tell which one you are actually fighting.

The standing promise from [the introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is) holds here as strictly as anywhere in this series: **I have no GPU and I have run none of this.** Every number below is derived from arithmetic I show you, cited from a paper or an official post with a link, or framed as something you will reproduce with a named script and an expected range. The results tables carry a `Source` column. The kernel-launch costs in particular are the ones people fabricate most often, and I will not — you will measure yours.

This is the kernel companion to [the hybrid-model memory post](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption), which derived the state sizes we are about to move around, and the sibling of [the paged attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand), which is the kernel this one is the opposite of in almost every way that matters.

---

## 1. What one decode step of a recurrent layer actually computes

Both families we care about — the Mamba-2 state-space dual (SSD) recurrence and the gated delta rule — reduce at decode time to the same three-line shape. There is a fixed-size state tensor, an update that mixes the new token into it, and a read that projects it back down to a vector. Nothing about that shape depends on how long the sequence is, which is exactly the property that made hybrids attractive and exactly the property that makes this kernel weird.

**The Mamba-2 SSD single step.** For one head with head dimension $P$ and state dimension $N$, the state is a matrix $\mathbf{H} \in \mathbb{R}^{P \times N}$. At step $t$ the layer produces from the input a value vector $\mathbf{x}_t \in \mathbb{R}^{P}$, a step size $\Delta_t$ (a scalar per head), and two group-shared vectors $\mathbf{B}_t, \mathbf{C}_t \in \mathbb{R}^{N}$. Then:

$$
a_t = \exp(\Delta_t A_h), \qquad
\mathbf{H}_t = a_t \mathbf{H}_{t-1} + (\Delta_t \mathbf{x}_t)\,\mathbf{B}_t^{\top}, \qquad
\mathbf{y}_t = \mathbf{H}_t \mathbf{C}_t + D_h \mathbf{x}_t
$$

Read it left to right. The old state decays by a scalar factor $a_t$ that the model computes per token — that is the "selective" part, the thing that distinguishes Mamba from a fixed linear recurrence. The new token is written in as a rank-one outer product. The output is a matrix-vector product against $\mathbf{C}_t$, plus a skip term. That is the entire sequence-mixing computation for one token, one head.

Two details that matter for the kernel and get glossed over in the papers. First, $\Delta_t$ usually arrives as a raw projection and passes through a softplus with a learned bias, so the kernel has to compute $\Delta_t = \text{softplus}(\Delta_t^{\text{raw}} + \text{bias}_h)$ itself rather than receiving it clean. Second, $\mathbf{B}_t$ and $\mathbf{C}_t$ are shared across a *group* of heads, not per head — a model with 128 heads and 8 groups gives every group of 16 heads the same $\mathbf{B}$ and $\mathbf{C}$. Your kernel's indexing has to know that, and it is a common source of silently wrong output that still looks like plausible text.

**The gated delta rule single step.** Here the state is a key-to-value associative memory $\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$. The layer produces a query $\mathbf{q}_t$, key $\mathbf{k}_t$, value $\mathbf{v}_t$, a scalar decay $\alpha_t$ and a scalar write strength $\beta_t$. The recurrence from [Gated Delta Networks](https://arxiv.org/abs/2412.06464) (Yang et al., arXiv:2412.06464) is:

$$
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1}\bigl(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top}\bigr) + \beta_t \mathbf{v}_t \mathbf{k}_t^{\top}
$$

which, written in the order a kernel would actually evaluate it, is much friendlier:

$$
\mathbf{S}'= \alpha_t \mathbf{S}_{t-1}, \qquad
\hat{\mathbf{v}} = \mathbf{S}'^{\top}\mathbf{k}_t, \qquad
\mathbf{S}_t = \mathbf{S}' + \beta_t\, \mathbf{k}_t (\mathbf{v}_t - \hat{\mathbf{v}})^{\top}, \qquad
\mathbf{o}_t = \mathbf{S}_t^{\top} \mathbf{q}_t
$$

Decay the memory, look up what the memory currently predicts for this key, compute the error against the true value, and write back a rank-one correction proportional to that error. It is online least-squares with a forgetting factor. The Kimi Linear paper ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) generalises $\alpha_t$ from a scalar to a per-channel diagonal gate, which changes the math slightly and the kernel structure not at all — you replace one broadcast scalar with one broadcast vector.

Both recurrences have the same computational fingerprint: a fixed-size tensor, three or four passes over it, one read and one write to memory. Everything in this post follows from that fingerprint.

The figure above is the SSD step drawn against the layer that contains it. The hidden vector fans out into the convolution update, the decay gate and the projections; all three converge on a single state tile that is read, modified and written once; the output and the state leave together in one store. Every arrow into the state node is a value that a fused kernel can compute in registers and never write to memory. That convergence is the whole design.

---

## 2. The mechanism: arithmetic intensity of a fixed-state update

Here is the derivation the rest of the post rests on. It is short, and its conclusion is stronger than I expected the first time I worked it through.

Take any decode step whose only persistent tensor is a state $\mathbf{S}$ holding $M$ elements at $b$ bytes each. Suppose the step makes $c$ passes over that state — a decay multiply is one pass, a rank-one accumulate is two (a multiply and an add), a matrix-vector read is two. Then:

$$
\text{FLOPs} = c\,M, \qquad
\text{HBM bytes} = 2\,M\,b \;+\; O(\sqrt{M})
$$

The bytes term is a read of the state and a write of the state. The $O(\sqrt{M})$ correction is the input and output vectors, which are of dimension $P$ or $N$ while the state is $P \times N$ — genuinely negligible. So the [arithmetic intensity](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) is:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{bytes}} = \frac{c\,M}{2\,M\,b} = \frac{c}{2b}
$$

**$M$ cancels.** The arithmetic intensity of a fixed-state recurrent decode step does not depend on the head dimension, the state dimension, the number of heads, or the sequence length. It depends on exactly two things: how many passes the recurrence makes over its state, and what dtype you store the state in. Nothing else.

Count $c$ for our two recurrences. For SSD: the decay multiply $a_t \mathbf{H}$ is $PN$ operations; the outer-product accumulate $(\Delta_t \mathbf{x})\mathbf{B}^{\top}$ is $PN$ multiplies plus $PN$ adds; the output $\mathbf{H}\mathbf{C}$ is $PN$ multiplies plus roughly $PN$ adds. That is $c = 5$. For the gated delta rule: decay is one pass, the lookup $\mathbf{S}'^{\top}\mathbf{k}$ is two, the rank-one write is two, the output read $\mathbf{S}^{\top}\mathbf{q}$ is two — $c = 7$.

| Recurrence | Passes over state | AI, bf16 state | AI, fp32 state | Source |
| --- | --- | --- | --- | --- |
| Mamba-2 SSD step | 5 | 1.25 FLOP/byte | 0.63 FLOP/byte | derived |
| Gated delta rule step | 7 | 1.75 FLOP/byte | 0.88 FLOP/byte | derived |
| Kimi-style channel gate | 7 | 1.75 FLOP/byte | 0.88 FLOP/byte | derived |
| Depthwise conv-state update | ~3 | 0.75 FLOP/byte | 0.38 FLOP/byte | derived |

Now place those on a roofline. NVIDIA's H100 datasheet lists roughly 989 teraFLOP/s of bf16 tensor-core throughput against 3.35 TB/s of HBM3 bandwidth, so the ridge point — the intensity at which compute and bandwidth balance — sits near 295 FLOPs per byte. Our kernels live at 1.25 and 1.75. They are more than two hundred times below the ridge.

By the letter of the roofline, that makes them memory-bound and the story is over: go read memory as fast as the card allows and you are done. Except the story is not over, because the roofline tells you the *ratio* and says nothing about the *magnitude*. A kernel with an intensity of 1.25 that has to move 1 GiB is a bandwidth problem. A kernel with an intensity of 1.25 that has to move 4 MiB is a scheduling problem. Recurrent decode is the second kind, and the next two sections are about what that changes.

#### Worked example: the byte budget of one Nemotron-H-8B Mamba-2 layer

Take the published config for [Nemotron-H-8B](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json): `hidden_size` 4096, `expand` 2 giving $d_{\text{inner}} = 8192$, `mamba_num_heads` 128, `mamba_head_dim` 64, `ssm_state_size` 128, `n_groups` 8, `conv_kernel` 4, bf16.

Per head, the state is $64 \times 128 \times 2 = 16{,}384$ bytes, which is 16 KiB. Across 128 heads, one layer holds 2.00 MiB of SSM state. The convolution retains three prior positions across the inner projection plus the group-shared projections, $3 \times (8192 + 2 \times 8 \times 128) \times 2 = 61{,}440$ bytes, or 60 KiB.

One decode step reads and writes both, and reads a handful of vectors:

$$
\text{bytes} = \underbrace{2 \times 2{,}097{,}152}_{\text{SSM state RW}} + \underbrace{2 \times 61{,}440}_{\text{conv state RW}} + \underbrace{16{,}384}_{\mathbf{x}} + \underbrace{4{,}096}_{\mathbf{B},\mathbf{C}} + \underbrace{256}_{\Delta} \approx 4{,}337{,}920
$$

That is 4.14 MiB per layer per request. Across the model's 24 Mamba-2 layers, **104 MB per decode step, regardless of context length**. Source: derived from the model card config.

FLOPs for the same step: $5 \times d_{\text{inner}} \times N = 5 \times 8192 \times 128 = 5{,}242{,}880$, about 5.24 MFLOP per layer. Confirming the intensity: $5{,}242{,}880 / 4{,}337{,}920 = 1.21$, which is the predicted 1.25 diluted slightly by the conv state and the input vectors. The derivation holds.

Now the number that should make you sit up. At the H100's peak bandwidth, moving 104 MB takes $104 \times 10^{6} / 3.35 \times 10^{12} \approx 31$ microseconds. Thirty-one microseconds is the *entire* memory cost of every recurrent layer in an 8B model for one token at batch 1. That is not a budget you optimise. That is a budget you accidentally spend ten times over on overhead.

---

## 3. Bytes are bounded, so the launches set the clock

![Layered stack showing registers, shared memory, L2 and HBM3 with the recurrent state tile resident on chip and the key-value tiles streamed through memory](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-2.webp)

The figure is the memory hierarchy annotated with what each of our two kernels puts where. Registers on an H100 SM total 65,536 32-bit slots, which is 256 KB; shared memory tops out at 228 KB per block; L2 is 50 MB; HBM3 is 80 GB at 3.35 TB/s. A single head's 16 KiB SSM state fits comfortably in registers. The attention layer's 32 MiB of KV per layer at 8k context fits nowhere on chip and has to be streamed. That difference is not a detail — it is the whole reason the two kernels have different bottlenecks.

Every kernel launch costs something. The CPU has to build a launch packet, the driver has to submit it, the GPU's work distributor has to find SMs, and at the end the kernel has to drain before a dependent kernel starts. On current hardware this is commonly in the low single-digit microseconds per launch, and under a captured CUDA graph the host-side portion shrinks but does not vanish. NVIDIA's own [kernel-fusion post](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead) is blunt about the limit of graphs: "Wrapping a naive baseline in a graph shaves microseconds off the host side, but the one GiB round-trip through the intermediate buffer remains unchanged." Graphs remove the ceremony. Only fusion removes the memory traffic between the steps.

I am deliberately not going to hand you a launch-overhead number as fact. Measure it, because it varies with driver version, CUDA version, whether you are inside a graph, and how many kernels are already in flight:

```python
# nanoserve/bench/launch_probe.py
# Measures the marginal cost of issuing one trivial kernel, eager and under a graph.
import torch, triton, triton.language as tl

@triton.jit
def _noop(p):
    tl.store(p + tl.program_id(0), 0.0)

def eager_launch_cost(n=2000, device="cuda"):
    buf = torch.zeros(1, device=device)
    for _ in range(50):                      # warmup: JIT compile + caches
        _noop[(1,)](buf)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(n):
        _noop[(1,)](buf)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / n  # microseconds per launch

def graph_launch_cost(n=2000, device="cuda"):
    buf = torch.zeros(1, device=device)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(50):
            _noop[(1,)](buf)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(n):
            _noop[(1,)](buf)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); g.replay(); end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / n

if __name__ == "__main__":
    print(f"eager : {eager_launch_cost():.2f} us / launch")
    print(f"graph : {graph_launch_cost():.2f} us / launch")
```

On a modern data-centre card you should expect the eager number to land somewhere in the two-to-ten microsecond band and the graph number to be meaningfully lower, often under two. Run it and write your own number down; the rest of this section is arithmetic on whatever you get, and I will use three microseconds as a placeholder that you should replace.

Now count the kernels a naive PyTorch Mamba-2 decode step issues. Do not trust my count — take yours:

```python
# nanoserve/bench/count_kernels.py
# Counts device kernels in ONE decode step of one layer. The number, not my number,
# is what your launch bill is built from.
import torch
from torch.profiler import profile, ProfilerActivity

def count_step_kernels(layer, hidden, state, conv_state):
    for _ in range(10):                                   # warmup
        layer.step(hidden, state, conv_state)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        layer.step(hidden, state, conv_state)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    for e in sorted(kernels, key=lambda e: -e.device_time)[:12]:
        print(f"{e.device_time:8.1f} us  {e.key[:64]}")
    print(f"\ntotal device kernels in one step: {len(kernels)}")
    return len(kernels)
```

A straightforward implementation — the one you get from transcribing the equations into PyTorch — issues a kernel for the input projection, two or three for the convolution-state roll and depthwise multiply-accumulate, one for the softplus, one for the exponential decay, three or four for the broadcast multiply, outer product and add that form the state update, one for the output contraction, one for the skip term, two for the SiLU gate, two or three for the gated RMSNorm, and one for the output projection. In the profile that is typically somewhere around fifteen device kernels for the sequence-mixing part of a single layer.

#### Worked example: the launch bill at batch 1

Nemotron-H-8B on an H100, 24 Mamba-2 layers, batch 1, launch cost taken as 3 microseconds.

Unfused: $24 \times 15 \times 3\,\mu s = 1{,}080\,\mu s$ of launch overhead, wrapped around 31 microseconds of unavoidable memory traffic. The ratio is roughly 35 to 1. Fused down to three kernels per layer — input projection GEMM, one fused conv-plus-scan-plus-norm kernel, output projection GEMM — you get $24 \times 3 \times 3\,\mu s = 216\,\mu s$, a ratio of about 7 to 1. Source: derived, using a placeholder launch cost you must replace with your own measurement from `launch_probe.py`.

Two honest caveats on that arithmetic. First, it treats launches as serialised, which they are not entirely — the CPU can run ahead and enqueue while the GPU works, so the *observed* penalty is smaller than the sum of launch costs unless the CPU is the bottleneck, which at batch 1 with fifteen tiny kernels per layer it very often is. Second, CUDA graphs collapse most of the host-side portion, which is why every serious engine captures the decode step. What graphs do not collapse is the GPU-side per-kernel ramp and, more importantly, the fact that each kernel in the chain writes its intermediate to HBM and the next one reads it back. Fifteen kernels means fourteen round-trips through memory for tensors that could have stayed in registers.

![Side-by-side comparison of an unfused decode step with fifteen kernel launches per layer against a fused step with three, showing the launch cost collapsing](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-3.webp)

That is the trade the figure draws. On the left, fifteen launches per layer and every intermediate spilling to HBM and back; on the right, three launches with the gates and the norm living in registers. The right-hand column is not faster because it computes less. It computes exactly the same thing. It is faster because it asks the GPU fewer times and keeps more of the answer on chip.

There is a well-documented precedent for the size of this win, though it comes from the prefill side rather than decode. The PyTorch team's [Accelerating Mamba2 with Kernel Fusion](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) post reports fusing the five SSD prefill kernels — chunk cumsum, chunk state, state passing, batched matmul, and chunk scan — into a single Triton kernel, yielding a 1.50× to 2.51× speedup on the SSD portion on NVIDIA A100 and H100, benchmarked at batch sizes 1 to 32 and sequence lengths from 1K to 256K with fp16 states. Their end-to-end figure is 8–13% for Mamba-2 2.7B at batch 1 and 128K context, rising to roughly 20% at 1K context. Their stated reasons are precisely ours: "Eliminating Kernel Launch Overheads: One launch instead of five reduces CPU-GPU synchronization and scheduling delays" and "Improving Cache Locality: Data produced in one stage is immediately consumed by the next within the same threadblock."

Note carefully what that citation does and does not establish. It is prefill, explicitly — the post says so — and the decode step is a different kernel with a different shape. It establishes that fusing the SSD pipeline is worth a large constant factor on the SSD portion. It does not license me to claim a specific decode speedup, and I am not going to.

---

## 4. Where the state lives, and why that is the whole trick

The reason this kernel can be fused at all is that its working set is small enough to stay on chip for the entire step. Sixteen kilobytes of state per head. Not per layer — per head, per request. A block that owns one head's state can load it into registers, do every operation the layer needs, and write it back, without a single intermediate touching memory.

Contrast that with what [the paged attention kernel](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) has to do. That kernel's working set is the whole KV cache for the sequence, which at 8k context on Llama-3.1-8B is 1 GiB. It cannot hold that. It holds a tile, streams the next tile, reduces, and repeats, carrying a running maximum and a running sum so the softmax stays exact across the stream. The loop structure exists because the data does not fit. Here there is no loop, because it does.

<figure class="blog-anim">
<svg viewBox="0 0 680 350" role="img" aria-label="A recurrent decode step loads one sixteen kilobyte state tile once, folds four gate values into it in place, and stores it once, while an attention step below streams thirty-two megabytes of key and value tiles past the same multiprocessor" style="width:100%;height:auto;max-width:820px">
<style>
.ss1-h{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ss1-s{font:400 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ss1-c{font:600 12.5px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ss1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ss1-tile{fill:none;stroke:var(--accent,#6366f1);stroke-width:2}
.ss1-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.ss1-chip{fill:var(--accent,#6366f1)}
.ss1-cl{font:600 11px ui-sans-serif,system-ui;fill:var(--background,#ffffff);text-anchor:middle}
.ss1-arr{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;fill:none}
.ss1-hd{fill:var(--text-secondary,#6b7280)}
.ss1-rule{stroke:var(--border,#d1d5db);stroke-width:1}
.ss1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.2}
.ss1-win{fill:var(--accent,#6366f1);opacity:.22}
@keyframes ss1-fold{0%{transform:translateY(0);opacity:0}5%{opacity:1}28%{transform:translateY(40px);opacity:1}36%{transform:translateY(40px);opacity:0}100%{transform:translateY(40px);opacity:0}}
@keyframes ss1-b1{0%,34%{transform:scaleY(.42)}44%,100%{transform:scaleY(.88)}}
@keyframes ss1-b2{0%,34%{transform:scaleY(.86)}44%,100%{transform:scaleY(.36)}}
@keyframes ss1-b3{0%,34%{transform:scaleY(.55)}44%,100%{transform:scaleY(.94)}}
@keyframes ss1-b4{0%,34%{transform:scaleY(.72)}44%,100%{transform:scaleY(.48)}}
@keyframes ss1-b5{0%,34%{transform:scaleY(.38)}44%,100%{transform:scaleY(.68)}}
@keyframes ss1-b6{0%,34%{transform:scaleY(.90)}44%,100%{transform:scaleY(.58)}}
@keyframes ss1-store{0%,52%{opacity:.15}62%,92%{opacity:1}100%{opacity:.15}}
@keyframes ss1-sw{0%{transform:translateX(0)}100%{transform:translateX(560px)}}
.ss1-f{animation:ss1-fold 10s ease-in-out infinite}
.ss1-f2{animation-delay:.9s}
.ss1-f3{animation-delay:1.8s}
.ss1-f4{animation-delay:2.7s}
.ss1-v1{animation:ss1-b1 10s ease-in-out infinite}
.ss1-v2{animation:ss1-b2 10s ease-in-out infinite}
.ss1-v3{animation:ss1-b3 10s ease-in-out infinite}
.ss1-v4{animation:ss1-b4 10s ease-in-out infinite}
.ss1-v5{animation:ss1-b5 10s ease-in-out infinite}
.ss1-v6{animation:ss1-b6 10s ease-in-out infinite}
.ss1-st{animation:ss1-store 10s ease-in-out infinite}
.ss1-mv{animation:ss1-sw 10s steps(8,end) infinite}
@media (prefers-reduced-motion:reduce){.ss1-f,.ss1-v1,.ss1-v2,.ss1-v3,.ss1-v4,.ss1-v5,.ss1-v6,.ss1-mv{animation:none}.ss1-st{animation:none;opacity:1}}
</style>
<text class="ss1-h" x="24" y="20">one recurrent step: the state never leaves the SM</text>
<text class="ss1-s" x="24" y="38">load 16 KiB once, fold four gates in, store 16 KiB</text>
<rect class="ss1-chip ss1-f" x="200" y="52" width="46" height="20" rx="5"/>
<rect class="ss1-chip ss1-f ss1-f2" x="254" y="52" width="46" height="20" rx="5"/>
<rect class="ss1-chip ss1-f ss1-f3" x="308" y="52" width="46" height="20" rx="5"/>
<rect class="ss1-chip ss1-f ss1-f4" x="362" y="52" width="46" height="20" rx="5"/>
<text class="ss1-cl ss1-f" x="223" y="66">dt</text>
<text class="ss1-cl ss1-f ss1-f2" x="277" y="66">decay</text>
<text class="ss1-cl ss1-f ss1-f3" x="331" y="66">B</text>
<text class="ss1-cl ss1-f ss1-f4" x="385" y="66">C</text>
<rect class="ss1-box" x="24" y="96" width="70" height="52" rx="8"/>
<text class="ss1-c" x="59" y="127">HBM</text>
<path class="ss1-arr" d="M98 122 L178 122"/>
<path class="ss1-hd" d="M178 117 L189 122 L178 127 Z"/>
<rect class="ss1-tile" x="192" y="84" width="224" height="76" rx="8"/>
<rect class="ss1-bar ss1-v1" x="208" y="96" width="24" height="52" rx="3"/>
<rect class="ss1-bar ss1-v2" x="242" y="96" width="24" height="52" rx="3"/>
<rect class="ss1-bar ss1-v3" x="276" y="96" width="24" height="52" rx="3"/>
<rect class="ss1-bar ss1-v4" x="310" y="96" width="24" height="52" rx="3"/>
<rect class="ss1-bar ss1-v5" x="344" y="96" width="24" height="52" rx="3"/>
<rect class="ss1-bar ss1-v6" x="378" y="96" width="24" height="52" rx="3"/>
<path class="ss1-arr ss1-st" d="M420 122 L556 122"/>
<path class="ss1-hd ss1-st" d="M556 117 L567 122 L556 127 Z"/>
<rect class="ss1-box" x="570" y="96" width="70" height="52" rx="8"/>
<text class="ss1-c" x="605" y="127">HBM</text>
<text class="ss1-s" x="192" y="180">state tile 64 x 128, 16 KiB, rewritten in place</text>
<text class="ss1-s" x="192" y="198">HBM traffic this step: 4.14 MiB per layer</text>
<line class="ss1-rule" x1="24" y1="216" x2="656" y2="216"/>
<text class="ss1-h" x="24" y="242">one attention step on the same SM, 8k context</text>
<text class="ss1-s" x="24" y="260">nothing is resident, every tile is fetched and dropped</text>
<rect class="ss1-cell" x="24" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="94" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="164" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="234" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="304" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="374" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="444" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-cell" x="514" y="276" width="56" height="40" rx="5"/>
<rect class="ss1-win ss1-mv" x="24" y="276" width="56" height="40" rx="5"/>
<text class="ss1-s" x="24" y="338">HBM traffic this step: 32 MiB per layer, and it grows with context</text>
</svg>
<figcaption>Top: the recurrent step loads its 16 KiB state tile once, folds the step size, decay and both projections into it in registers, and stores it once — 4.14 MiB of HBM traffic per layer regardless of how long the conversation is. Bottom: the attention step on the same multiprocessor holds nothing, sweeping 32 MiB of key and value tiles past itself for one token, and that number grows every step.</figcaption>
</figure>

Watch the top half once through and the design constraint becomes concrete. The tile is loaded, four gate values drop into it, the bars change value in place, and one store closes the step. Nothing in the middle goes to memory. That is what "fused" means here, and it is only possible because 16 KiB fits.

Now the register arithmetic, because "it fits" needs a number. An H100 SM has 65,536 32-bit registers, a hard cap of 255 registers per thread, and a maximum of 64 resident warps. Suppose a program (a Triton block) owns one head's state as a $64 \times 128$ tile in fp32 — that is 8,192 values. Spread across a 256-thread block, each thread holds 32 of them. Add working registers for the gates, the pointers, the accumulator and the compiler's temporaries and you land somewhere near 64 registers per thread total. Then:

$$
\text{blocks per SM} = \left\lfloor \frac{65{,}536}{256 \times 64} \right\rfloor = 4
$$

Four blocks of eight warps each is 32 resident warps out of 64, or 50% occupancy. Hold the state at bf16 in registers instead and you halve the state's register footprint, which pushes you toward six or seven blocks — but you also lose numerical headroom in the accumulation, which for a recurrence that runs for thousands of steps is a real risk. Most production implementations keep the state in fp32 for exactly this reason, and pay for it in both bytes and occupancy.

Put the tile in shared memory instead and the arithmetic changes: 228 KB of shared memory per block divided by 32 KiB of fp32 state gives seven blocks per SM by that constraint alone, but shared memory is slower than registers and the recurrence's dependency chain is serial, so you would be trading latency you cannot hide for occupancy you may not need. The right answer depends on your state dimensions, which is why the kernel gets autotuned rather than hand-tuned. The [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) has the general treatment of this trade.

---

## 5. Occupancy: the grid is batch times heads, and that is a problem

Here is the second half of the diagnosis, and it is the one that catches people who have already done the fusion work and are confused that it did not help enough.

A single-step recurrent kernel has exactly one natural parallel decomposition: one program per (request, head) pair. Within a head, the update is a serial dependency — you cannot compute $\mathbf{H}_t$ without $\mathbf{H}_{t-1}$ — and there is only one time step, so there is nothing to split along the sequence. Across heads and requests, everything is independent. So the grid is:

$$
\text{grid} = B \times H
$$

For Nemotron-H-8B with 128 Mamba heads, batch 1 gives you a grid of 128 blocks. An H100 SXM has 132 SMs. Every SM gets at most one block. Each of those blocks runs perhaps four to eight warps. The machine's warp slots are 132 × 64 = 8,448, and you have filled somewhere between 512 and 1,024 of them — six to twelve percent.

That matters because GPUs hide latency with parallelism, not with speed. Every memory access has a latency of several hundred cycles; the SM covers it by switching to another warp that has work ready. With four warps resident on an SM that has room for 64, there is nothing to switch to. The kernel's runtime becomes the length of one block's dependency chain — load, decay, outer product, contract, store — with the memory latency fully exposed at each step. You are not moving many bytes. You are just waiting.

![Timeline showing the block count and wave count of the recurrent kernel rising from batch one through batch two hundred fifty six](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-4.webp)

Define a **wave** as the number of blocks the machine can hold resident at once: SMs times blocks-per-SM. With the four-blocks-per-SM figure derived above, one wave is 132 × 4 = 528 blocks on an H100. Extending the timeline in the figure with a slightly more generous eight blocks per SM (a smaller state tile, or bf16 state) gives a wave of 1,056, and that is the number the figure uses:

| Batch | Grid blocks | Waves | What limits you | Source |
| --- | --- | --- | --- | --- |
| 1 | 128 | 0.12 | dependency-chain latency, exposed | derived |
| 4 | 512 | 0.48 | still latency; SMs half idle | derived |
| 8 | 1,024 | 0.97 | first full wave; latency starts hiding | derived |
| 32 | 4,096 | 3.9 | transitioning to bandwidth | derived |
| 64 | 8,192 | 7.8 | bandwidth, mostly | derived |
| 256 | 32,768 | 31.0 | bandwidth, entirely | derived |

This is the single most important operational fact about this kernel and it is the opposite of the intuition you built on attention. **The same kernel is in a different performance regime at batch 1 than at batch 64.** At batch 1 you are launch- and latency-bound and fusion is the fix. At batch 64 you are bandwidth-bound and fusion buys you almost nothing, because the memory system is saturated and the launches are lost in the noise.

#### Worked example: the same kernel in both regimes

Batch 1, Nemotron-H-8B, H100. Grid 128 blocks on 132 SMs. State traffic 104 MB across 24 layers, which at peak bandwidth would be 31 microseconds — but a grid covering an eighth of a wave cannot come near peak, because achieved bandwidth scales with the number of concurrent memory requests in flight. Fusing fifteen kernels per layer into three removes roughly 864 microseconds of launch ceremony (derived, at the 3-microsecond placeholder). Fusion is the dominant win.

Batch 64, same model, same H100. Grid 8,192 blocks, 7.8 waves — the machine is full and staying full. State traffic is now $64 \times 104\,\text{MB} = 6.7\,\text{GB}$ read and written per decode step, which at 3.35 TB/s is about 2.0 milliseconds. The launch overhead is still 216 microseconds fused, or about 11% of the step, and it would have been 1.08 ms unfused — still worth removing, but no longer the story. The story is that you are now streaming 6.7 GB of state for one token across the batch, and no amount of kernel cleverness changes that number. Source: derived from the config arithmetic in section 2.

That second regime deserves a warning, because it is a genuine architectural cost of hybrids that the memory-savings pitch tends to skip. The state is *flat in sequence length* but *linear in batch size*, exactly like the KV cache. What it is not is linear in both. A dense model at batch 64 and 128k context reads an enormous KV cache; a hybrid at batch 64 and 128k context reads 6.7 GB of state, the same as it would at 512 tokens. The break-even against this model's own four attention layers, which the companion post derives at 16 KiB of KV per token, falls at:

$$
S^{*} = \frac{104 \times 10^{6}}{16{,}384} \approx 6{,}350 \text{ tokens}
$$

Below roughly 6,350 tokens of context, the recurrent layers of Nemotron-H-8B move more bytes per decode step than its attention layers do. Above it, the attention layers dominate and keep growing. Source: derived. Both facts are true simultaneously, and which one you feel depends entirely on your workload's context distribution.

---

## 6. Writing the fused kernel in Triton

Time to write code. The plan: a PyTorch oracle first so we have something to diff against, then the Triton kernel, then a launcher that handles paged state slots, then the tests.

### 6.1 The oracle

Write the reference in the most boring PyTorch you can, straight from the equations. Its job is to be obviously correct, not fast.

```python
# nanoserve/models/ssd_ref.py
import torch
import torch.nn.functional as F

@torch.inference_mode()
def ssd_step_ref(state, x, dt_raw, A, B, C, D=None, z=None,
                 dt_bias=None, dt_softplus=True):
    """One Mamba-2 SSD decode step, written for clarity.

    state   (batch, nheads, head_dim, dstate)  fp32, updated in place
    x       (batch, nheads, head_dim)
    dt_raw  (batch, nheads)
    A       (nheads,)          negative; the continuous-time decay rate
    B, C    (batch, ngroups, dstate)
    D       (nheads,) or None  skip connection
    z       (batch, nheads, head_dim) or None   SiLU gate
    returns y (batch, nheads, head_dim)
    """
    nheads = x.shape[1]
    ngroups = B.shape[1]
    heads_per_group = nheads // ngroups

    dt = dt_raw.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)                                  # (b, h)

    # scalar decay per (batch, head); A is negative so decay lands in (0, 1)
    decay = torch.exp(dt * A.float())                        # (b, h)

    # expand the group-shared projections to per-head views
    Bh = B.float().repeat_interleave(heads_per_group, dim=1)  # (b, h, n)
    Ch = C.float().repeat_interleave(heads_per_group, dim=1)  # (b, h, n)

    # state <- decay * state + (dt * x) outer B
    dtx = dt.unsqueeze(-1) * x.float()                        # (b, h, p)
    state.mul_(decay[..., None, None])
    state.add_(dtx.unsqueeze(-1) * Bh.unsqueeze(-2))          # (b,h,p,1)*(b,h,1,n)

    y = torch.einsum("bhpn,bhn->bhp", state, Ch)              # (b, h, p)
    if D is not None:
        y = y + D.float()[None, :, None] * x.float()
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(x.dtype)
```

Notice how many separate CUDA kernels those eleven lines of math turn into. `softplus`, `mul`, `exp`, two `repeat_interleave`s, `unsqueeze`+`mul`, `mul_`, `add_`, `einsum`, two more for the skip and the gate. This function *is* the problem from section 3, written honestly.

### 6.2 The fused Triton kernel

One program per (request, head). The state tile is loaded once into registers, everything happens there, one store closes it.

```python
# nanoserve/kernels/ssd_decode.py
import torch
import triton
import triton.language as tl

@triton.jit
def _ssd_decode_kernel(
    state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
    dt_bias_ptr, out_ptr, slot_ptr,
    # strides
    s_slot, s_head, s_p, s_n,
    x_b, x_h, x_p,
    dt_b, dt_h,
    B_b, B_g, B_n,
    C_b, C_g, C_n,
    z_b, z_h, z_p,
    o_b, o_h, o_p,
    # sizes
    head_dim, dstate, heads_per_group,
    # compile-time
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr, HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)          # request index within the batch
    pid_h = tl.program_id(1)          # head index

    # Paged states: the request's physical slot is an indirection, exactly like
    # a block table entry. A negative slot marks a padded batch row: skip it.
    slot = tl.load(slot_ptr + pid_b)
    if slot < 0:
        return

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < head_dim
    mask_n = offs_n < dstate
    mask_pn = mask_p[:, None] & mask_n[None, :]

    state_ptrs = (state_ptr + slot * s_slot + pid_h * s_head
                  + offs_p[:, None] * s_p + offs_n[None, :] * s_n)

    # ---- the single load: 16 KiB of state into registers -------------------
    state = tl.load(state_ptrs, mask=mask_pn, other=0.0).to(tl.float32)

    # ---- gates, computed in registers, never written to memory -------------
    dt = tl.load(dt_ptr + pid_b * dt_b + pid_h * dt_h).to(tl.float32)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    if DT_SOFTPLUS:
        # log1p(exp(dt)) with the standard large-input guard
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)

    a = tl.load(A_ptr + pid_h).to(tl.float32)
    decay = tl.exp(a * dt)

    x = tl.load(x_ptr + pid_b * x_b + pid_h * x_h + offs_p * x_p,
                mask=mask_p, other=0.0).to(tl.float32)

    g = pid_h // heads_per_group          # group-shared B and C
    Bv = tl.load(B_ptr + pid_b * B_b + g * B_g + offs_n * B_n,
                 mask=mask_n, other=0.0).to(tl.float32)
    Cv = tl.load(C_ptr + pid_b * C_b + g * C_g + offs_n * C_n,
                 mask=mask_n, other=0.0).to(tl.float32)

    # ---- the recurrence: decay, rank-one write, contract -------------------
    state = state * decay + (dt * x)[:, None] * Bv[None, :]
    y = tl.sum(state * Cv[None, :], axis=1)

    if HAS_D:
        y += tl.load(D_ptr + pid_h).to(tl.float32) * x
    if HAS_Z:
        zv = tl.load(z_ptr + pid_b * z_b + pid_h * z_h + offs_p * z_p,
                     mask=mask_p, other=0.0).to(tl.float32)
        y = y * (zv * tl.sigmoid(zv))     # SiLU, fused

    # ---- the single store --------------------------------------------------
    tl.store(state_ptrs, state, mask=mask_pn)
    tl.store(out_ptr + pid_b * o_b + pid_h * o_h + offs_p * o_p,
             y.to(out_ptr.dtype.element_ty), mask=mask_p)
```

Read the body and count the memory operations: one `tl.load` of the state, four small loads of vectors and scalars, one `tl.store` of the state, one `tl.store` of the output. The softplus, the exponential, the decay multiply, the outer product, the contraction, the skip and the SiLU gate all happen between the load and the store, on values that never leave the register file. That is the entire point of the exercise.

A few implementation notes that are easy to get wrong:

- **`BLOCK_P` and `BLOCK_N` must be powers of two** and at least as large as `head_dim` and `dstate`. Triton requires power-of-two block shapes for `tl.arange`; the masks handle the remainder. For our config, `BLOCK_P = 64` and `BLOCK_N = 128` are exact.
- **The state accumulates in fp32 even when the model is bf16.** The `.to(tl.float32)` on load and the fp32 store are not decoration. A recurrence that decays and accumulates thousands of times will drift visibly in bf16, and the failure mode is a model that degrades over long generations rather than one that crashes.
- **The negative-slot early return** is how you handle a padded batch dimension without a separate gather. vLLM's own single-step kernel exposes the same idea through `state_batch_indices` and a `pad_slot_id` argument; see the [`selective_state_update` API surface](https://docs.vllm.ai/en/v0.11.2/api/vllm/model_executor/layers/mamba/ops/mamba_ssm/) in the vLLM docs.
- **`tl.sum(state * Cv[None, :], axis=1)`** is the contraction. It is a reduction across the state dimension within each program, which Triton lowers to a warp-level reduction — no shared memory round-trip, no second kernel.

### 6.3 The launcher

```python
# nanoserve/kernels/ssd_decode.py (continued)

def ssd_decode(state, x, dt_raw, A, B, C, slot_idx,
               D=None, z=None, dt_bias=None, dt_softplus=True,
               num_warps=4, num_stages=2):
    """Fused single-step SSD update over a paged state pool.

    state     (num_slots, nheads, head_dim, dstate) fp32, updated in place
    slot_idx  (batch,) int32 -- physical slot per batch row, -1 for padding
    """
    batch, nheads, head_dim = x.shape
    dstate = state.shape[-1]
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert state.dtype == torch.float32, "keep the recurrent state in fp32"

    out = torch.empty_like(x)
    grid = (batch, nheads)            # one program per (request, head)

    _ssd_decode_kernel[grid](
        state, x, dt_raw, A, B, C,
        D if D is not None else x,     # unused pointers still need to be valid
        z if z is not None else x,
        dt_bias if dt_bias is not None else A,
        out, slot_idx,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt_raw.stride(0), dt_raw.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0),
        out.stride(0), out.stride(1), out.stride(2),
        head_dim, dstate, nheads // ngroups,
        BLOCK_P=triton.next_power_of_2(head_dim),
        BLOCK_N=triton.next_power_of_2(dstate),
        DT_SOFTPLUS=dt_softplus,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        HAS_DT_BIAS=dt_bias is not None,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
```

The `slot_idx` indirection is the same design as the block table from [the paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table), and for the same reason: the engine's batch composition changes every step as requests finish and new ones are admitted, and you do not want to move state tensors around when it does. The request keeps its slot; the batch row points at it. One integer load per program buys you a completely stable state pool.

### 6.4 The gated delta-rule variant

Same shape, different algebra. The state is now a key-value outer-product memory and the update has an extra read.

```python
# nanoserve/kernels/delta_decode.py
import torch, triton, triton.language as tl

@triton.jit
def _gated_delta_kernel(
    S_ptr, q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, out_ptr, slot_ptr,
    s_slot, s_head, s_k, s_v,
    q_b, q_h, q_d, v_b, v_h, v_d,
    a_b, a_h, o_b, o_h, o_d,
    dk, dv,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    CHANNEL_GATE: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_ptr + pid_b)
    if slot < 0:
        return

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mk, mv = offs_k < dk, offs_v < dv
    mkv = mk[:, None] & mv[None, :]

    S_ptrs = (S_ptr + slot * s_slot + pid_h * s_head
              + offs_k[:, None] * s_k + offs_v[None, :] * s_v)
    S = tl.load(S_ptrs, mask=mkv, other=0.0).to(tl.float32)

    q = tl.load(q_ptr + pid_b * q_b + pid_h * q_h + offs_k * q_d,
                mask=mk, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + pid_b * q_b + pid_h * q_h + offs_k * q_d,
                mask=mk, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + pid_b * v_b + pid_h * v_h + offs_v * v_d,
                mask=mv, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + pid_b * a_b + pid_h * a_h).to(tl.float32)

    if CHANNEL_GATE:
        # Kimi-style diagonal gate: one decay per key channel, not one per head
        alpha = tl.load(alpha_ptr + pid_b * a_b + pid_h * a_h + offs_k,
                        mask=mk, other=1.0).to(tl.float32)
        S = S * alpha[:, None]
    else:
        alpha = tl.load(alpha_ptr + pid_b * a_b + pid_h * a_h).to(tl.float32)
        S = S * alpha

    # delta rule: predict, take the error, write back a rank-one correction
    v_hat = tl.sum(S * k[:, None], axis=0)           # (BLOCK_V,)
    err = v - v_hat
    S = S + beta * k[:, None] * err[None, :]

    o = tl.sum(S * q[:, None], axis=0)               # (BLOCK_V,)

    tl.store(S_ptrs, S, mask=mkv)
    tl.store(out_ptr + pid_b * o_b + pid_h * o_h + offs_v * o_d,
             o.to(out_ptr.dtype.element_ty), mask=mv)
```

The `CHANNEL_GATE` branch is the only structural difference between a scalar-decay gated delta rule and the finer-grained diagonal gate that Kimi Delta Attention introduced. One is a broadcast scalar multiply, the other a broadcast vector multiply, and Triton compiles both to the same shape of code. What changes is memory: a per-channel gate means loading $d_k$ floats per head per step instead of one, which for a 128-dimensional key head is 512 extra bytes against 64 KiB of state — nothing. The expressiveness is close to free at the kernel level, which is exactly the argument the Kimi Linear paper makes for it.

### 6.5 Autotuning the occupancy knobs

The two things worth searching over are `num_warps` (how many warps share the state tile, which sets registers per thread) and `num_stages` (how aggressively the compiler software-pipelines the loads). Both move occupancy directly.

```python
# nanoserve/kernels/ssd_decode.py (autotuned variant)
import triton

_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in (1, 2, 4, 8)
    for s in (1, 2, 3)
]

@triton.autotune(
    configs=_CONFIGS,
    # Retune whenever the shape of the problem changes. batch matters here in a
    # way it does not for most kernels: it changes how full the grid is.
    key=["head_dim", "dstate", "batch"],
)
@triton.jit
def _ssd_decode_autotuned(...):   # same body as _ssd_decode_kernel
    ...
```

Two warnings from experience with this pattern in a serving loop. First, `triton.autotune` runs its search on the *first* call for each key, which means a several-hundred-millisecond stall in the middle of serving unless you warm every shape at startup. Add batch to the key and you have a lot of shapes. The usual fix is to bucket the batch dimension — pad to the next power of two — so the key space stays small. Second, autotuned kernels and CUDA graphs interact badly if the autotuner has not converged before capture; warm every configuration you intend to capture, then capture.

The vLLM team's [Triton attention backend deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) hits the same wall from the attention side and is worth reading for the shape of the problem: they report that "no single configuration dominates" across their workload matrix, and that variable launch grids "replay badly under CUDA graphs," which is what pushed them toward a persistent-kernel variant with a fixed number of instances reading work metadata from GPU memory. If your recurrent kernel ends up inside a captured graph with a batch dimension that changes, that persistent-kernel design is the pattern to copy.

---

## 7. Correctness: diffing against the oracle, and the two bugs you will hit

A recurrent decode kernel has a nasty property: it is stateful, so a bug does not produce garbage on the first step. It produces something slightly wrong that feeds into the next step, and the error compounds. A single-step test passes. A hundred-step test fails. Write the hundred-step test.

```python
# tests/test_ssd_decode.py
import torch, pytest
from nanoserve.models.ssd_ref import ssd_step_ref
from nanoserve.kernels.ssd_decode import ssd_decode

@pytest.mark.parametrize("batch,nheads,ngroups,head_dim,dstate",
                         [(1, 128, 8, 64, 128), (5, 32, 4, 64, 128),
                          (17, 8, 1, 128, 64)])
@pytest.mark.parametrize("steps", [1, 128])
def test_matches_reference(batch, nheads, ngroups, head_dim, dstate, steps):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16

    ref_state = torch.randn(batch, nheads, head_dim, dstate,
                            device=dev, dtype=torch.float32) * 0.1
    # scatter the same states into a paged pool at non-contiguous slots
    num_slots = batch * 3
    pool = torch.zeros(num_slots, nheads, head_dim, dstate,
                       device=dev, dtype=torch.float32)
    slots = torch.randperm(num_slots, device=dev)[:batch].to(torch.int32)
    pool[slots.long()] = ref_state.clone()

    A = -torch.rand(nheads, device=dev) - 0.5      # negative decay rates
    D = torch.randn(nheads, device=dev)
    dt_bias = torch.randn(nheads, device=dev) * 0.1

    for _ in range(steps):
        x = torch.randn(batch, nheads, head_dim, device=dev, dtype=dt)
        z = torch.randn(batch, nheads, head_dim, device=dev, dtype=dt)
        dt_raw = torch.randn(batch, nheads, device=dev, dtype=dt)
        B = torch.randn(batch, ngroups, dstate, device=dev, dtype=dt)
        C = torch.randn(batch, ngroups, dstate, device=dev, dtype=dt)

        y_ref = ssd_step_ref(ref_state, x, dt_raw, A, B, C, D, z, dt_bias)
        y_tri = ssd_decode(pool, x, dt_raw, A, B, C, slots, D, z, dt_bias)

        torch.testing.assert_close(y_tri.float(), y_ref.float(),
                                   rtol=2e-2, atol=2e-2)

    # the states must still agree after all those compounding updates
    torch.testing.assert_close(pool[slots.long()], ref_state,
                               rtol=1e-3, atol=1e-4)
```

Three things this test does deliberately. It uses non-contiguous, shuffled slots, because a stride bug in the slot indirection is invisible when slot $i$ happens to equal batch row $i$. It uses `ngroups` values that are not equal to `nheads`, because the group-sharing index `pid_h // heads_per_group` is the single most common source of wrong-but-plausible output. And it checks the final state, not just the outputs, because a state that drifts while the outputs stay within tolerance is a bug that will eat you at generation length 2,000.

**The two bugs you will actually hit.** The first is the group index. If your model has 128 heads and 8 groups and you index `B` with `pid_h` instead of `pid_h // 16`, you read past the end of the group dimension. On a well-behaved allocator that produces garbage; on an unlucky one it silently reads the next tensor. Symptom: coherent-looking output that is subtly off-topic, and a perplexity regression you cannot localise. The fix is the parametrised test above with `ngroups != nheads`.

The second is the softplus guard. Writing `tl.log(1.0 + tl.exp(dt))` without the `dt <= 20.0` branch overflows `exp` for large `dt`, producing `inf`, then `inf * A` gives `-inf`, then `exp(-inf)` gives a decay of exactly zero — and the state is silently wiped. Symptom: the model stops using long-range context, on some prompts only. This one is genuinely hard to find from the outside, and it is why the reference implementations all carry the guard.

For a third opinion on correctness, diff against the reference kernel that the ecosystem actually ships. Both `mamba_ssm` and vLLM expose a single-step selective state update:

```python
# tests/test_against_mamba_ssm.py
# A-B check against the reference Triton kernel from the mamba_ssm package.
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

def test_against_upstream(state, x, dt_raw, A, B, C, D, z, dt_bias, slots):
    ours   = ssd_decode(state.clone(), x, dt_raw, A, B, C, slots,
                        D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    theirs = selective_state_update(state.clone(), x, dt_raw, A, B, C,
                                    D=D, z=z, dt_bias=dt_bias,
                                    dt_softplus=True,
                                    state_batch_indices=slots)
    torch.testing.assert_close(ours.float(), theirs.float(),
                               rtol=1e-2, atol=1e-2)
```

If your kernel matches both a naive PyTorch oracle and an independent Triton implementation over 128 compounding steps with shuffled slots and non-trivial group sharing, it is correct. That is the bar.

---

## 8. Measuring it honestly

Everything in this post says the bottleneck is launch cost and occupancy, not bandwidth. That claim is falsifiable on your hardware in about ten minutes, and you should falsify it rather than believe me.

```python
# nanoserve/bench/bench_ssd_decode.py
# Separates the three candidate bottlenecks: launch cost, memory traffic,
# and dependency-chain latency. Run it before you optimise anything.
import torch, argparse
from nanoserve.kernels.ssd_decode import ssd_decode

def timed(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()                 # never time without this
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters      # microseconds

def main(batch, nheads=128, ngroups=8, head_dim=64, dstate=128, layers=24):
    dev = "cuda"
    pool = torch.randn(batch, nheads, head_dim, dstate,
                       device=dev, dtype=torch.float32) * 0.1
    slots = torch.arange(batch, device=dev, dtype=torch.int32)
    x  = torch.randn(batch, nheads, head_dim, device=dev, dtype=torch.bfloat16)
    z  = torch.randn_like(x)
    dtr = torch.randn(batch, nheads, device=dev, dtype=torch.bfloat16)
    B  = torch.randn(batch, ngroups, dstate, device=dev, dtype=torch.bfloat16)
    C  = torch.randn_like(B)
    A  = -torch.rand(nheads, device=dev) - 0.5
    D  = torch.randn(nheads, device=dev)

    step = lambda: ssd_decode(pool, x, dtr, A, B, C, slots, D=D, z=z)
    us = timed(step)

    # what the roofline says this should cost if bandwidth were the limit
    ssm_bytes = 2 * batch * nheads * head_dim * dstate * 4      # fp32 state RW
    peak_bw = 3.35e12                                            # H100 SXM3, cited
    floor_us = ssm_bytes / peak_bw * 1e6

    print(f"batch {batch:4d} | grid {batch * nheads:7d} blocks")
    print(f"  measured   : {us:8.2f} us / layer")
    print(f"  bw floor   : {floor_us:8.2f} us / layer  (peak HBM, one layer)")
    print(f"  achieved bw: {ssm_bytes / (us * 1e-6) / 1e9:8.1f} GB/s")
    print(f"  whole model: {us * layers:8.2f} us / token ({layers} layers)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    main(p.parse_args().batch)
```

Run it as a sweep — `for b in 1 2 4 8 16 32 64 128; do python bench_ssd_decode.py --batch $b; done` — and read the achieved-bandwidth column. What you should see, and what the derivation in section 5 predicts: at batch 1 the achieved bandwidth is a small fraction of peak and the measured time is far above the floor, because the grid is a fraction of a wave. As batch grows the achieved bandwidth climbs and the measured time converges toward the floor. The batch at which the curve flattens is your machine's crossover from latency-bound to bandwidth-bound, and it is the single most useful number this benchmark produces.

Expected ranges, framed honestly as reproduce-it-yourself: on an H100 or A100 with a 128-head model you should expect batch 1 to sit well under a quarter of peak bandwidth, and the curve to be close to flat somewhere between batch 16 and batch 64. On an RTX 4090 — 128 SMs and 1,008 GB/s of peak bandwidth, of which NVIDIA's own kernel-fusion benchmark achieves roughly 850 GiB/s — the crossover lands in a similar place because the SM count is similar, even though the bandwidth is a third. Report yours.

The measurement discipline is the same as everywhere in this series and worth restating because it is violated constantly. Warm up before timing, so the Triton JIT and the autotuner have finished. Call `torch.cuda.synchronize()` before starting the clock and before reading it, because CUDA is asynchronous and a naive `time.perf_counter()` around a kernel launch measures how long it took to *enqueue* the work. Use CUDA events rather than wall-clock where you can. Lock the clocks with `nvidia-smi -lgc` if you want run-to-run stability, or accept a few percent of noise from boost behaviour. And treat batch-1 latency as one point on a curve, never as the headline — [the reproducible benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) makes that case at length.

One profiling tip specific to this kernel. If you profile with `nsys` and see the SSM layers as a dense picket fence of two-microsecond kernels separated by gaps, you are launch-bound and fusion is your fix. If you see a single kernel per layer that is much longer than the bandwidth floor, you are occupancy-bound and the fix is a smaller state tile or fewer warps per block. If you see a single kernel per layer sitting close to the floor, you are done — the remaining cost is physics.

---

## 9. Against the attention kernel in the same model

![Comparison matrix of five decode kernels against their byte counts, arithmetic intensity, limiting factor and where their working set lives](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-5.webp)

The matrix above is the honest summary and it is worth dwelling on, because the two kernels sitting in the same model behave so differently that intuition transferred from one to the other will mislead you.

| Property | Paged attention decode | SSD / delta-rule decode | Source |
| --- | --- | --- | --- |
| Bytes per step, batch 1 | 1.0 GiB at 8k, grows with S | 104 MB, flat in S | derived |
| Bytes per step, batch 64 | 64 GiB at 8k | 6.7 GB | derived |
| Arithmetic intensity | about 1.0 with GQA | exactly 1.25 (SSD, bf16) | derived |
| Natural grid | batch x KV heads, split-K to fill | batch x heads, no split available | derived |
| Working set | streamed tiles through SMEM | one 16 KiB tile in registers | derived |
| Limit at batch 1 | HBM bandwidth | launch cost, then latency | derived |
| Limit at batch 64 | HBM bandwidth | HBM bandwidth | derived |
| Helped by fusion? | modestly; it is one kernel already | dramatically at low batch | derived |
| Helped by split-K? | yes, materially at low batch | no; the step is serial | derived |

That "no split available" row is the structural difference and it deserves a paragraph. Paged attention at batch 1 has a fill problem too — one sequence, a handful of heads, not enough blocks to occupy the card — and it has a clean solution: split the KV traversal across multiple blocks, let each compute a partial softmax, and combine the partials in a second kernel. The vLLM Triton backend does exactly this and calls it "parallel tiled softmax." The reason it works is that attention's reduction is over the *sequence*, and the sequence is long.

The recurrent step has no such dimension. There is one time step. The state update is a single serial dependency. You cannot split the work of $\mathbf{H}_t = a_t\mathbf{H}_{t-1} + \mathbf{x}_t\mathbf{B}_t^{\top}$ across blocks in any way that reduces its latency, because there is no reduction to parallelise — only $P \times N$ independent elements, which one block already handles. Your only sources of parallelism are heads and requests. Heads are fixed by the architecture. Requests are the batch. Which means, bluntly: **the only lever you have for occupancy on this kernel is batch size, and if your product is single-stream chat you do not get to pull it.**

That is a genuinely uncomfortable conclusion and it is the correct one. It is also why the fusion work matters so much more here than it does for attention. If you cannot fill the machine, the least you can do is stop asking it for permission fifteen times per layer.

There is one more asymmetry, and it points at prefill. The same recurrence that is a serial one-step update at decode time is a *scan* at prefill time — and a scan is parallelisable, which is the entire content of the chunked SSD algorithm and the chunkwise-parallel delta-rule formulations. The prefill kernel and the decode kernel for the same layer share almost no code. [The scan-versus-recurrence post](/blog/machine-learning/inference-engineering/prefill-is-a-scan-decode-is-a-recurrence) is the theory; this post is the decode half of the practice.

---

## 10. Where it breaks

Five stress tests, in roughly the order you will trip over them.

**Batch 1 with a small head count.** Everything in section 5 gets worse if the model has fewer heads. A hybrid with 32 linear-attention heads gives you a grid of 32 blocks at batch 1 — a quarter of the SMs on an H100, three warps of work on each. At that size the kernel is essentially a latency measurement of one dependency chain, and no kernel engineering fixes it. The engine-level fix is to batch, and if you cannot batch, the architecture-level fix is to stop pretending single-stream latency is the metric you are optimising.

**fp32 state and a big state dimension.** The register arithmetic in section 4 assumed a $64 \times 128$ tile. Double the state dimension to 256 and the tile is 64 KiB in fp32 — 16,384 values across a 256-thread block is 64 registers per thread of state alone, before working values. You will blow past the 255-register cap or force the compiler to spill to local memory, which is HBM wearing a hat. The symptom is a kernel that gets dramatically slower for a modest increase in state size, and `ncu` will show you register spills directly. The fix is to tile the state dimension across multiple programs and accept a second reduction, or to keep the state in bf16 and accept the numerical risk.

**Speculative decoding.** Every kernel in this post assumes exactly one token per request per step. Speculative decoding proposes $k$ tokens and then verifies them, which for attention is a small generalisation — you attend with $k$ query rows instead of one — and for a recurrent layer is a genuine problem. To verify $k$ draft tokens you must advance the state $k$ times, and if the verification rejects at position $j$ you must *roll the state back* to position $j$. There is no rollback for a tensor you overwrote in place. Your options are to snapshot the state before the speculative window (costing a full state copy per step, which for a 49.4 MiB state is 100 MB of traffic — worse than the work you were trying to save), or to recompute the accepted prefix from the last checkpoint, or to keep the *inputs* rather than the state and replay. vLLM tracks exactly this as an open design question; their [hybrid SSM disaggregation post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) lists the interaction between speculative decoding and hybrid models as "not extensively validated," which is a polite way of saying be careful.

**Tensor parallelism.** Split a Mamba-2 layer across two GPUs and each rank owns half the heads, which for the kernel changes nothing — the grid shrinks, occupancy gets worse, that is all. What it changes is the *state transfer* problem, and that is where the production engines have done real work. Per the same vLLM post, the conv state is decomposed into three sub-projections corresponding to the $x$, $\mathbf{B}$ and $\mathbf{C}$ paths, and the SSM state is stored in a `(dim, state_len)` "DS" layout, so that a decode rank fetching state over the network reads only its own one-over-TP slice rather than the whole tensor and then discarding most of it. The post reports that padding is consequently never transferred, saving roughly 50 MB per request on bf16 — a real number with a real setup, cited, not mine.

![Dataflow graph showing a convolution state splitting into three sub-projections that merge with the state-space state into per-rank slices](/imgs/blogs/the-selective-scan-and-delta-rule-decode-kernel-7.webp)

The layout in that figure is what a production engine does that a from-scratch kernel usually does not: it treats the state as a structured object with named sub-projections rather than one opaque blob, precisely so that a consumer needing one slice can address one slice. If you are building toward multi-GPU or disaggregated serving, design the state layout for that from the start — retrofitting a slice-addressable layout after you have a working kernel is a rewrite.

**Prefix caching and preemption.** The block-based KV cache lets you evict half a sequence and recompute it. A recurrent state is meaningless in pieces — you cannot evict the second half of a state matrix, and you cannot reconstruct the state at position $t$ from the state at position $t + 100$. Any engine feature that assumes partial, position-indexed, recomputable cache contents needs a separate story for these layers. The companion post covers the allocator consequences; the kernel consequence is simply that your state pool has no eviction policy more sophisticated than "drop the whole request."

---

## 11. Case studies and public numbers

Four results with provenance, none of them mine.

**PyTorch, fusing the SSD prefill pipeline.** The [Accelerating Mamba2 with Kernel Fusion](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) post reports 1.50×–2.51× on the SSD portion on NVIDIA A100 and H100 by fusing five kernels — chunk cumsum, chunk state, state passing, batched matmul, chunk scan — into one Triton kernel, benchmarked at batch 1–32 and sequence lengths 1K–256K with fp16 states, translating to roughly 8–13% end-to-end for Mamba-2 2.7B at batch 1 and 128K context and about 20% at 1K context. This is prefill, explicitly. Cite it for the *mechanism* — launch elimination plus locality — not as a decode number.

**vLLM, disaggregated hybrid SSM serving.** The [hybrid SSM disaggregation post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) describes the engine-side consequences of the state layout: dual descriptor views over the same physical memory so one index format addresses attention blocks and another addresses SSM blocks; attention blocks subdivided to satisfy a kernel's token granularity (FlashInfer's 16-token requirement) while SSM layers use logical blocks directly; the three-way conv-state decomposition and DS layout described above. Their benchmark is `Nemotron-3-Super-120B-A12B-FP8` on 8×H200 with prefill TP4 and decode TP4, which they report "Pareto-dominates the co-located baseline at higher batch sizes." Note the qualifier — at higher batch sizes — which is the same regime boundary this post derived from first principles.

**vLLM, the Triton attention backend.** The [backend deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) reports their Triton attention kernel reaching 100.7% of FlashAttention 3 on an H100 for Llama 3.1 8B at batch 1 with a 500-token input and a long decode, in roughly 800 lines against FA3's roughly 70,000. The relevance here is the counterfactual: that is what a well-tuned Triton kernel achieves on the *bandwidth-bound* problem. It is the ceiling you should expect for the attention layers of your hybrid, and the reason you should not spend your optimisation budget there.

**Tencent's HPC-Ops backends.** The [HPC-Ops post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops) (2026-07-06) describes a fused prologue they call `HpcRopeNorm`, combining QK-Norm, RoPE and the KV-cache write into a single kernel, alongside a three-stage persistent attention kernel. Their attention numbers are up to 2.95× over static split-KV on an H20 with Hy3 FP8, and 2.25× on average against FlashInfer and FlashAttention. The reason it belongs in this post is the *pattern*, not the number: the industry's answer to "many small kernels around one big one" is to fold the small ones into the prologue of the big one. Our fused SSD kernel is the same move applied to a layer where the small kernels are all there is.

**Kimi Linear.** The [Kimi Linear paper](https://arxiv.org/abs/2510.26692) (arXiv:2510.26692) introduces Kimi Delta Attention, extending the gated delta rule with a per-channel diagonal gate, and reports a bespoke chunkwise algorithm built on a specialised Diagonal-Plus-Low-Rank formulation reaching close to twice the speed of the general DPLR form. That is a prefill-side algorithmic result; the decode-side consequence, as section 6.4 showed, is that the finer gate costs essentially nothing per step.

---

## When to reach for this (and when not to)

**Write this kernel** if you are building an engine that must serve a hybrid model and you have measured a picket fence of tiny kernels in the profile. The fusion win at low batch is large, derivable, and yours to take. Write it also if you are learning — this is the cleanest small kernel in modern inference, with no online-softmax subtlety and no tiling loop, and it teaches occupancy better than any tutorial because the occupancy problem is unavoidable rather than incidental.

**Do not write this kernel** if you are serving at batch 32 or above and the profile shows a single kernel per layer near the bandwidth floor. You are done. Any further effort belongs in the attention layers, the MoE routing, or the scheduler.

**Do not write this kernel from scratch for production** unless you have a specific reason. `mamba_ssm` ships `selective_state_update`, and vLLM ships its own Triton port with paged state indices, padding slots and the layout work described above. Both are correct, tested against real models, and maintained by people who can measure. The version in this post exists so that you understand what those kernels do and can debug them when they are the thing that is slow. That has been the deal for the whole series: build it once to see it, then use the production one and know exactly what it is doing.

**Reach for CUDA C++ instead of Triton** when you need something Triton will not express: a persistent kernel with a work queue read from device memory, cluster-level cooperation across SMs on Hopper, or explicit control over the register allocation because the compiler is spilling and you can see a way it should not. The [Triton post](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) has the full decision rule; for this kernel specifically, the trigger is almost always register pressure at large state dimensions.

**Capture a CUDA graph regardless.** Whatever you do about fusion, a decode step with dozens of small kernels benefits enormously from graph capture, and the [CUDA graphs post](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) covers the mechanics and the shape-stability requirements. Graphs and fusion attack different halves of the same overhead, and you want both.

---

## Key takeaways

1. **The arithmetic intensity of a fixed-state recurrent decode step is $c/(2b)$ — a constant.** No dimension appears in it. For Mamba-2 SSD in bf16 that is exactly 1.25 FLOPs per byte; for the gated delta rule, 1.75.
2. **Low intensity does not mean bandwidth-bound.** The roofline gives you a ratio, not a magnitude. This kernel moves about 104 MB per decode step for an 8B hybrid at batch 1, which is roughly 31 microseconds of HBM time on an H100 — far less than the launch overhead of a naive implementation.
3. **The state fits on chip, so the whole layer can be one kernel.** Sixteen kilobytes per head lives in registers; the gates, the norm and the skip never touch memory. That is not an optimisation, it is the design.
4. **The grid is batch times heads and there is nothing else to parallelise.** No split-K, no sequence dimension, no reduction to tile. Batch size is your only occupancy lever.
5. **The same kernel changes regime with batch.** Latency- and launch-bound below roughly one full wave of blocks; bandwidth-bound above it. Sweep the batch and find your crossover before you optimise anything.
6. **State traffic is flat in context and linear in batch.** For Nemotron-H-8B the recurrent layers move more bytes than the attention layers below about 6,350 tokens of context, and fewer above it. Both facts are true; your workload picks which one you feel.
7. **Test over a hundred compounding steps, with shuffled slots and non-trivial group sharing.** A stateful kernel's bugs do not show up on step one. The group index and the softplus overflow guard are the two that will get you.
8. **Speculative decoding and recurrent state do not compose for free.** Overwriting a state in place destroys the rollback path, and snapshotting costs more traffic than the step itself.
9. **Design the state layout for slicing from day one.** Production engines decompose it into named sub-projections precisely so a tensor-parallel rank can fetch its own share; retrofitting that is a rewrite.
10. **Fusion and CUDA graphs solve different halves.** Graphs remove host-side launch ceremony; only fusion removes the HBM round-trip between the steps.

---

## Further reading

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — the gated delta-rule recurrence in section 1, with the parallel training algorithm that becomes the prefill kernel.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Delta Attention's per-channel gate and the DPLR chunkwise algorithm.
- [Accelerating Mamba2 with Kernel Fusion](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) — PyTorch's five-into-one SSD prefill fusion, with the launch-overhead and locality argument stated plainly.
- [Disaggregated Serving for Hybrid SSM Models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) — vLLM's dual descriptor views, the three-way conv-state decomposition, and the DS layout for tensor-parallel state transfer.
- [vLLM Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) — persistent kernels, autotuning under CUDA graphs, and what a tuned Triton kernel achieves on the bandwidth-bound problem.
- [Kernel Fusion in NVIDIA CUDA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead) — the distinction between what graphs remove and what only fusion removes.
- [Hybrid models and the end of the KV cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) — the memory derivation this kernel operates on.
- [Paged attention kernel by hand](/blog/machine-learning/inference-engineering/paged-attention-kernel-by-hand) — the kernel this one is the mirror image of.
- [Kernel fusion and FlashAttention: beating the memory wall](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) — the general theory of why fusion wins.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — where this kernel sits in the whole engine.
