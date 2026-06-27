---
title: "GPU Profiling and Optimization for RL Training: From Bottleneck to Throughput"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A profiler-first playbook for RL and RLHF training: find the real GPU bottleneck, read a PyTorch trace, compute MFU, fix it with packing, FSDP, Flash Attention, and torch.compile, then prove the speedup."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "pytorch",
    "gpu-profiling",
    "mfu",
    "flash-attention",
    "torch-compile",
    "machine-learning",
    "llm-alignment",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 62
image: "/imgs/blogs/gpu-profiling-optimization-rl-training-1.png"
---

The first time I profiled an RLHF run, I was certain the PPO update was the problem. The loss curve looked fine, but each iteration took ninety seconds on eight A100s and the GPUs felt cold to the touch — `nvidia-smi` showed utilization bouncing between 18% and 40%, never pinning at the high nineties the way a supervised pretraining job does. So I did what most people do: I made the PPO optimizer faster. I fused the optimizer, I tuned the learning rate schedule, I shaved a few milliseconds off the advantage computation. Wall-clock barely moved. Ninety seconds became eighty-seven.

The reason is captured in the first figure below, and it is the single most important fact about optimizing RL training: **an RLHF iteration does not spend its time where you think it does.** Roughly two-thirds of the wall clock is in *rollout generation* — the policy model autoregressively sampling responses, one token at a time, memory-bandwidth-bound and starved of arithmetic intensity. The PPO update I had been polishing was a quarter of the budget. I had spent two days optimizing the wrong phase because I never looked at a profiler trace. The cold GPUs were not a mystery; they were the rollout phase telling me, in plain terms, that it was waiting on memory, not math.

![Stacked breakdown of where a typical 7B RLHF iteration spends its wall clock across rollout, reward, advantage, and PPO update phases](/imgs/blogs/gpu-profiling-optimization-rl-training-1.png)

This post is the playbook I wish I had that week. By the end you will be able to: attach the PyTorch profiler to an RL training loop without distorting the measurement; read a Chrome/TensorBoard trace and spot idle gaps, long-tail kernels, and synchronization stalls; compute Model FLOPs Utilization (MFU) and use it as your single north-star efficiency metric; diagnose a CUDA out-of-memory crash down to the offending op; and apply the concrete fixes — sequence packing, micro-batching, Flash Attention, NCCL overlap, `torch.compile`, and the right parallelism strategy — each validated by a before-and-after profile rather than a hunch. We will keep tying everything back to the spine of this whole series: RL is an agent collecting experience from an environment and updating a policy. In RLHF the "environment" is a sampling loop over a language model, and that loop is where your FLOPs go to die unless you measure it.

If you want the conceptual map of where RLHF sits among RL algorithms, the unified taxonomy post `reinforcement-learning-a-unified-map` frames it; this post is the systems-engineering companion that makes that algorithm actually run fast.

## 1. Why profiling is different (and harder) for RL than for SFT

Supervised fine-tuning has one repeating shape: a forward pass, a loss, a backward pass, an optimizer step. The arithmetic intensity is high and steady. You can stare at `nvidia-smi`, see 95% utilization, and reasonably conclude you are compute-bound and near peak. Profiling SFT is mostly about confirming what you already suspect.

RL training, and RLHF in particular, breaks that comfortable picture into three phases with wildly different hardware profiles. The first figure laid them out by time; here is what each one is actually *doing* to the GPU.

**Phase one — rollout (60–80% of wall clock).** The policy model generates responses to a batch of prompts. Generation is autoregressive: token *t+1* depends on token *t*, so you cannot parallelize across the sequence dimension the way a training forward pass does. Each decoding step is a single matrix-vector-ish operation per layer with a tiny batch of new tokens, dominated by reading the model weights and the KV-cache out of HBM. This is the textbook memory-bandwidth-bound regime. Arithmetic intensity — FLOPs per byte moved — is low, so the tensor cores sit mostly idle while the memory subsystem works. This is *why* `nvidia-smi` showed me cold GPUs: utilization counts whether a kernel is running, not whether the math units are saturated, and during decode the math units are starving.

**Phase two — reward scoring (5–10%).** A separate reward model (and often a reference model for the KL penalty) does a single forward pass over each full prompt-plus-response sequence. This is a dense, parallel-over-tokens forward pass: compute-bound, high arithmetic intensity, the GPU's happy place. It is short because it is one forward pass with no backward.

**Phase three — the PPO update (20–30%).** Now you have a batch of experiences (sequences, log-probs, rewards, advantages) and you run several epochs of minibatch gradient updates on the policy and value heads. Forward plus backward, compute-bound, the part that looks most like SFT. It is where most tutorials focus and where the least wall-clock savings live.

The practical consequence: **the techniques that speed up SFT — bigger batches, Flash Attention, `torch.compile` — mostly help phase three, which is the smallest phase.** The biggest win usually comes from attacking phase one with a dedicated inference engine (vLLM, TensorRT-LLM, SGLang) and from not wasting compute on padding. If you optimize without profiling, you will instinctively reach for the SFT toolbox and aim it at the wrong target, exactly as I did.

There is a deeper point worth stating precisely, because it is the theoretical backbone of everything that follows. The wall-clock time of a kernel is governed by whichever of two ceilings it hits first:

$$
t_{\text{kernel}} = \max\left( \frac{\text{FLOPs}}{P_{\text{compute}}}, \; \frac{\text{Bytes}}{B_{\text{memory}}} \right)
$$

where $P_{\text{compute}}$ is the GPU's peak arithmetic throughput (for an A100, about 312 TFLOP/s in BF16) and $B_{\text{memory}}$ is its peak HBM bandwidth (about 2.0 TB/s on the 80GB A100). The ratio of the two defines the hardware's **ridge point** in arithmetic intensity:

$$
I^* = \frac{P_{\text{compute}}}{B_{\text{memory}}} \approx \frac{312 \times 10^{12}}{2.0 \times 10^{12}} \approx 156 \ \text{FLOP/byte}
$$

Any kernel whose arithmetic intensity sits below $I^*$ is memory-bound — adding more compute does nothing; you must move fewer bytes or move them faster. Autoregressive decode with a small batch sits *far* below the ridge, often at 1–10 FLOP/byte. Training forward-backward at a healthy batch size sits above it. This single inequality explains why rollout is slow, why batching helps decode (it raises FLOP/byte by reusing weight reads across more tokens), and why Flash Attention helps (it cuts the bytes moved). Profiling, in the end, is just the empirical measurement of where each of your kernels falls relative to that ridge.

It is worth doing the decode arithmetic explicitly, because it is the single most counterintuitive fact for engineers coming from supervised training. During one decode step the model reads its entire weight matrix to produce one new token (per sequence in the batch). For a 7B BF16 model that is roughly 14 GB of weights read from HBM. The arithmetic done is about $2N = 14 \times 10^9$ FLOPs per token. So the arithmetic intensity of a single-sequence decode step is approximately:

$$
I_{\text{decode}} = \frac{2N \cdot b}{2N \cdot 2} = \frac{b}{2} \ \text{FLOP/byte}
$$

where $b$ is the decode batch size (because the weights are read once and reused across all $b$ sequences, while the FLOPs scale with $b$). At $b = 1$ the intensity is 0.5 FLOP/byte — more than 300× below the ridge point — which is why a single A100 generating one sequence at a time runs at well under 1% MFU. The cure is right there in the formula: raise $b$. At $b = 256$ the intensity climbs to 128 FLOP/byte, close to the ridge, and the decode finally starts to use the tensor cores. This is the entire theoretical justification for continuous batching engines like vLLM, and it is why "the rollout is slow" almost always means "the rollout batch is too small," not "the kernel is bad." You cannot fix a memory-bound kernel by making the math faster; you can only fix it by amortizing the memory reads across more useful work.

The same roofline reasoning gives you a quick mental test for every optimization you are about to attempt. Ask: does this change reduce bytes moved, increase useful FLOPs per byte, or neither? Flash Attention reduces bytes (no $N^2$ matrix to HBM). Sequence packing increases useful FLOPs per byte (fewer padding bytes). Larger batches increase FLOPs per byte. Gradient checkpointing actually *increases* total FLOPs (recompute) to reduce peak memory — it is the one lever that trades the wrong way on compute to buy memory, which is why you only reach for it when memory, not speed, is the binding constraint. Keeping this taxonomy in your head stops you from applying a bytes-reducing fix to a compute-bound phase, or vice versa, which is the most common way optimization effort gets wasted.

## 2. The PyTorch profiler: attaching it without lying to yourself

The right tool for almost everything here is `torch.profiler`. It hooks both the CPU side (Python dispatch, autograd, host-to-device copies) and the CUDA side (the actual GPU kernels) and timestamps them on a common timeline you can later inspect. The most common beginner mistake is profiling the wrong steps — capturing the first iteration, which includes one-time CUDA context creation, allocator warmup, cuDNN autotuning, and `torch.compile` graph capture, all of which inflate the numbers by an order of magnitude and have nothing to do with steady-state cost.

The `schedule` argument exists precisely to avoid that. It defines a repeating cycle of `wait` (do nothing), `warmup` (run the profiler but discard the data), `active` (run and keep), and `repeat`. You want to skip the first iteration entirely, warm up for one or two, and only then record.

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

prof_schedule = schedule(
    wait=1,      # skip step 0 (context/allocator warmup)
    warmup=2,    # run profiler, discard data (CUDA caches, autotune settle)
    active=3,    # record these 3 steps
    repeat=1,    # one cycle only
)

def trace_handler(p):
    # Print the top ops by self CUDA time, and dump a TensorBoard trace.
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    p.export_chrome_trace("./rlhf_trace.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=trace_handler,
    record_shapes=True,     # tensor shapes per op (helps spot tiny kernels)
    profile_memory=True,    # track allocations for the memory view
    with_stack=True,        # Python stack frames (find the call site)
) as prof:
    for step in range(8):
        rollout = generate_rollout(policy_model, prompts)        # phase 1
        rewards = score_rewards(reward_model, ref_model, rollout) # phase 2
        ppo_update(policy_model, value_model, rollout, rewards)   # phase 3
        prof.step()  # MUST call this each iteration to advance the schedule
```

Three things to internalize. First, `prof.step()` is what advances the schedule from wait to warmup to active; forget it and the profiler records nothing or everything. Second, `record_shapes`, `profile_memory`, and `with_stack` each add overhead — `with_stack` especially can slow the loop 2–3× — so enable them deliberately when you need that view and turn them off for pure timing. Third, the `wait=1, warmup=2` pattern is not optional for RLHF: the first generate call triggers KV-cache allocation and, if you use `torch.compile`, a full recompile, which you must never include in your steady-state numbers.

For the cheapest possible measurement — just "how long does each phase take" — you do not even need the full profiler. Wrap regions in `record_function` and read the labeled spans, or use CUDA events for precise device-side timing that accounts for the asynchronous nature of GPU execution:

```python
import torch

def time_cuda(fn, *args, warmup=2, iters=10):
    """Device-accurate timing: events sit on the CUDA stream, not the CPU clock."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()        # block until GPU work finishes
    return start.elapsed_time(end) / iters  # milliseconds per call
```

That `torch.cuda.synchronize()` is the subtle part. CUDA kernel launches are asynchronous: the Python `fn(*args)` call returns to the CPU almost immediately while the GPU keeps working. If you time with `time.time()` and no sync, you measure how fast Python can *queue* kernels, not how long they take to run — a classic way to "prove" a kernel is instant when it actually dominates. CUDA events are recorded *on the stream*, so `elapsed_time` measures real device time. Whenever someone reports a suspiciously fast GPU operation, the first question is always: did you synchronize?

## 3. Reading a trace: gaps, long tails, and the shape of a stall

A profiler dump is only useful if you can read it. There are two views you will live in: the **operator table** (`key_averages().table(...)`) and the **timeline** (the Chrome/TensorBoard trace). The figure below shows how the profiler produces both from one run — it taps the CPU and CUDA streams in parallel and merges them into a single trace that fans out into timeline, memory, and operator views.

![Branching dataflow showing the PyTorch profiler tapping CPU and CUDA event streams and merging them into a trace that fans out into TensorBoard views](/imgs/blogs/gpu-profiling-optimization-rl-training-3.png)

The operator table is your first stop because it ranks kernels by self-CUDA-time and immediately tells you whether time is concentrated or smeared. A healthy training step has its time concentrated in a handful of GEMMs (matrix multiplies) and the attention kernel. An unhealthy one shows a long tail of tiny kernels — pointwise adds, casts, layernorms, copies — each a few microseconds but collectively a third of the step, every one paying full kernel-launch overhead (1–5 microseconds of CPU dispatch) for almost no work. A long tail is the signature that says "this wants kernel fusion," which is exactly what `torch.compile` provides.

The timeline view is where you find *idle gaps*. Load the exported `rlhf_trace.json` into Chrome at `chrome://tracing` or into TensorBoard's profile plugin, and look at the CUDA stream row. In a perfectly utilized step it is a solid bar of kernels back to back. The pathologies look like this:

- **Gaps between kernels with CPU activity above them.** The GPU finished, the CPU is busy preparing the next launch (Python overhead, data shuffling, advantage computation on CPU), and the GPU waits. This is CPU-bound or dispatch-bound. Fix with `torch.compile` (cuts Python overhead) or by moving work off the critical path.
- **Gaps with a `cudaStreamSynchronize` or `cudaMemcpy` marker.** A blocking sync — usually a `.item()`, a `.cpu()`, a Python `if` on a GPU tensor, or a print of a loss value mid-step — forces the CPU to wait for the GPU to drain. Each one serializes the pipeline. In RL loops these hide in reward computation and logging.
- **Gaps during rollout with no CPU activity.** The GPU is busy but doing low-intensity decode work; the "gap" is the memory subsystem being the bottleneck. The fix is not to remove the gap but to raise arithmetic intensity (bigger generation batch) or hand decode to vLLM.
- **One enormous kernel at the end.** A single op — often an unfused softmax over a huge vocab, or attention without Flash — dominating the step. Targeted: replace that op.

The discipline is to always read the trace top-down: total step time, then the phase split, then within the slow phase the operator table, then the timeline for that phase. Never jump straight to optimizing a kernel you noticed; confirm it is on the critical path first.

There are two more profiler signals worth knowing how to read, because they distinguish the four bottleneck classes from each other. The first is **SM (Streaming Multiprocessor) occupancy** — the fraction of the GPU's compute units that have active warps. TensorBoard's profile plugin and Nsight Systems both surface this. Low SM occupancy with the kernels back-to-back (no gaps) means the kernels themselves are too small to fill the GPU — the long-tail-of-tiny-kernels pathology, fixable by fusion. Low occupancy *and* gaps means you are launch-bound or sync-bound. High occupancy with gaps means memory-bound (the SMs are busy but stalling on memory). The second signal is **achieved memory bandwidth** as a fraction of peak; when a kernel hits 80–90% of the 2 TB/s HBM bandwidth, it is genuinely memory-bound and there is no kernel-level fix — only an algorithmic one (batch more, move less). These two numbers turn "the GPU feels slow" into a precise diagnosis.

A trap specific to RL loops deserves its own warning: the **CPU-side advantage and reward bookkeeping** that sits between phases. Generalized Advantage Estimation, reward normalization, KL bookkeeping, and the running statistics for advantage whitening are all cheap arithmetically but, if written as Python loops over timesteps on CPU tensors, they show up as a wide CPU-bound band in the trace with the GPU completely idle above it. I have seen a 7B run lose 9% of its wall clock to a Python `for` loop computing GAE one timestep at a time — invisible in `nvidia-smi` (which showed the GPU as merely idle, not the cause), obvious in the trace as a several-hundred-millisecond CPU span with no overlapping kernels. Vectorizing that loop onto the GPU erased it. The general lesson: in supervised training the CPU rarely sits on the critical path, but RL's per-step bookkeeping puts it there constantly, so the profiler's CPU row matters far more than it does for SFT.

#### Worked example: distinguishing launch-bound from memory-bound

You profile the PPO update and the trace shows the CUDA stream is 40% gaps. Is it launch overhead or memory stalls? Look at two numbers. If the operator table shows hundreds of distinct tiny kernels (3–8 microseconds each) and the gaps line up with CPU dispatch activity, it is launch-bound — each kernel pays ~2 microseconds of CPU launch latency, and with 300 kernels that is 600 microseconds of pure overhead per step. The fix is `torch.compile`, which fuses those 300 kernels into perhaps 30, cutting launch overhead by an order of magnitude. If instead the operator table shows a handful of large kernels each running at 85% of peak HBM bandwidth with the gaps appearing *between* phases (not within them), it is memory-bound within kernels and the inter-phase gaps are sync points — the fix is removing the `.item()` calls that force synchronization, not fusing kernels. Same 40% gap symptom, completely different cause and cure. This is why you read the operator table *and* the timeline *and* the occupancy together, never one alone.

#### Worked example: reading an RLHF trace and finding the real culprit

Suppose you profile three active steps of a 7B PPO-RLHF run on a single A100 and the operator table comes back like this (self-CUDA-time, summed over the active window):

| Region / op | Self CUDA time | Share | Notes |
| --- | --- | --- | --- |
| `aten::scaled_dot_product_attention` (decode) | 41.2 s | 47% | rollout, 512 generated tokens |
| `aten::mm` / linear (decode) | 18.6 s | 21% | rollout weight reads |
| `aten::mm` / linear (PPO fwd+bwd) | 12.1 s | 14% | the update |
| reward + ref forward | 6.9 s | 8% | dense, fast |
| `aten::copy_` + `aten::cat` (buffer) | 4.4 s | 5% | experience buffer churn |
| long tail (casts, adds, layernorm) | 4.8 s | 5% | fusable |

The naive reading is "attention is 47%, let me speed up attention." But notice *which* attention: it is the **decode** attention inside rollout, and it is slow not because the kernel is bad but because decode reads the whole KV-cache per token and runs at batch-of-one arithmetic intensity. Rollout (rows 1+2) is 68% of the step. The PPO update you might have wanted to optimize is 14%. The buffer copies (5%) are a CPU-side `cat` happening on the critical path between phases — a quick win. The long tail (5%) is a `torch.compile` candidate. The honest priority order from this trace is: (1) move rollout to vLLM and pack sequences — attacks 68%; (2) hoist buffer concatenation off the critical path or preallocate — 5%; (3) compile the update — recovers part of 14% plus the 5% tail. Optimizing the SDPA kernel directly would touch the symptom, not the cause.

## 4. MFU: the one number that keeps you honest

Wall-clock time tells you whether you got faster. It does not tell you how much headroom remains, or whether 30 seconds per step is "good." For that you need **Model FLOPs Utilization (MFU)** — the fraction of the GPU's peak arithmetic throughput your training is actually achieving on *useful* model FLOPs.

The definition is simple and the discipline is everything:

$$
\text{MFU} = \frac{\text{useful model FLOPs per second}}{\text{peak FLOPs per second of the hardware}} = \frac{C_{\text{model}} \cdot N_{\text{tokens}} / t_{\text{step}}}{P_{\text{peak}}}
$$

The standard estimate for the forward-plus-backward FLOPs of a dense transformer is $C_{\text{model}} \approx 6N$ per token, where $N$ is the parameter count (the factor of 6 is 2 for the forward matmul, 4 for the backward; this is the Kaplan et al. / Chinchilla accounting). Generation (forward only) is $\approx 2N$ per token. So for the PPO update on a 7B model processing $T$ tokens in $t$ seconds on an A100:

$$
\text{MFU} = \frac{6 \cdot 7\times10^9 \cdot T}{t \cdot 312\times10^{12}}
$$

Crucially, MFU counts only **real** tokens. If 40% of your batch is padding, those FLOPs are real work the GPU did but they advance training zero, so they do not appear in the numerator — and your MFU correctly craters. This is what makes MFU a better target than utilization: `nvidia-smi` would happily show 95% while the GPU grinds through padding; MFU exposes the waste.

Typical numbers, so you know where you stand:

| Workload | Phase | Realistic MFU | Why |
| --- | --- | --- | --- |
| Dense pretraining | fwd+bwd | 40–55% | compute-bound, large batch, the easy case |
| SFT / instruction tuning | fwd+bwd | 30–50% | smaller batches, more variance |
| RLHF PPO update | fwd+bwd | 25–45% | small minibatches, value+policy heads |
| RLHF rollout (HF generate) | decode | 1–8% | memory-bound, batch-of-one decode |
| RLHF rollout (vLLM, batched) | decode | 10–30% | continuous batching raises intensity |

Notice the rollout numbers. A 1–8% MFU during generation is not a bug — it is the physics of memory-bound decode hitting the ridge-point inequality from section 1. It is *also* why aggregate RLHF MFU (blending all phases) of 8–15% is common and not shameful; the rollout phase mathematically cannot reach update-phase MFU on the same hardware. The right move is not to despair at the aggregate number but to (a) raise rollout intensity with batched inference, and (b) maximize update-phase MFU where it is achievable.

Here is a small helper that turns a profiled step into an MFU number, which I keep in every training repo:

```python
GPU_PEAK_FLOPS = {
    "a100": 312e12,   # BF16 tensor-core, no sparsity
    "h100": 989e12,   # BF16 tensor-core (sxm), ~1979e12 with FP8
    "h200": 989e12,
}

def estimate_mfu(num_params, real_tokens, step_seconds, gpu="a100",
                 n_gpus=1, is_training=True):
    """real_tokens = non-padding tokens processed this step (across all gpus)."""
    flops_per_token = 6 * num_params if is_training else 2 * num_params
    achieved = flops_per_token * real_tokens / step_seconds
    peak = GPU_PEAK_FLOPS[gpu] * n_gpus
    return achieved / peak

# PPO update: 7B params, 8 GPUs, 1.2M real tokens, 9.0 s
mfu = estimate_mfu(7e9, 1_200_000, 9.0, gpu="a100", n_gpus=8)
print(f"update MFU = {mfu:.1%}")   # -> update MFU = 33.3%
```

When you report a speedup, report the MFU before and after, not just the wall clock — wall clock can improve because you shrank the problem, but MFU tells you whether the hardware is working harder. That distinction has saved me from celebrating fake wins more than once.

One subtlety in the $6N$ accounting is worth getting right for RLHF specifically, because RLHF runs more than one model. The $6N$ figure is forward-plus-backward for the model you are *training* — the policy (and its value head). But each PPO iteration also pays for forward passes through the frozen reference model (for the KL penalty) and the reward model, each at $2N_{\text{ref}}$ and $2N_{\text{rm}}$ per token, with no backward because they are not updated. If your reference and reward models are the same size as the policy, the per-token FLOPs of a full iteration is closer to $6N_{\text{policy}} + 2N_{\text{ref}} + 2N_{\text{rm}} \approx 10N$ when the three are equal-sized, not $6N$. This matters for two reasons. First, when you compute MFU you must decide whether to count the reference and reward forwards as "useful" — they are useful work, but they do not advance the policy, so the honest convention is to report policy-update MFU separately from a whole-iteration MFU that includes the frozen-model forwards. Second, it tells you that shrinking or sharing the reference and reward models (a smaller reward model, or reusing the reference as the SFT checkpoint loaded once) directly cuts a third of your non-rollout FLOPs. The accounting is not academic; it points at a real lever.

There is also a theoretical reason the PPO update phase has a higher achievable MFU ceiling than rollout, and it ties directly back to the policy-gradient machinery this series is built around. The PPO update computes the clipped surrogate objective

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\right]
$$

where $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$ is the probability ratio and $\hat{A}_t$ the advantage estimate. Crucially, this is evaluated over the *whole* batch of stored experiences *in parallel* — every token's log-prob and advantage is known, so the forward pass over the packed batch is a dense, fully-parallel transformer forward exactly like SFT, hitting the same 30–45% MFU. Rollout, by contrast, must generate token *t+1* before it can generate *t+2*; the sequential dependency is intrinsic to sampling from $\pi_\theta$ and cannot be parallelized away. So the MFU gap between the two phases is not an implementation accident — it is the difference between scoring known sequences (parallel) and sampling new ones (sequential), and no amount of kernel tuning changes which side of that line a phase falls on.

## 5. Memory profiling and OOM forensics

The other way RL runs fail is not slowly but suddenly: a `CUDA out of memory` traceback, often deep into training, often non-deterministically. RLHF is unusually prone to this because it holds *multiple* models resident at once — the policy, the value head, a frozen reference model for the KL term, and a reward model — plus an experience buffer of variable-length sequences whose peak size depends on what the policy happened to generate. Memory is not constant across steps.

The first tool is the summary, which gives you the allocator's bookkeeping at a glance:

```python
import torch
print(torch.cuda.memory_summary(device=0, abbreviated=True))
# Shows: allocated current/peak, reserved current/peak, and the
# largest free block (fragmentation indicator).
```

The two numbers that matter are **allocated** (what tensors are actually using) and **reserved** (what the caching allocator has grabbed from the driver and is holding for reuse). When reserved is far above allocated and you still OOM, you have **fragmentation**: enough total free memory exists, but no single contiguous block large enough for the next allocation. RLHF's variable sequence lengths are a fragmentation engine — a 2048-token rollout allocates a big activation block, a 128-token one allocates a small one, and after thousands of mismatched alloc/free cycles the heap is swiss cheese.

For *where* the memory went, the snapshot is the precise instrument. It records every allocation with its call stack and lets you render a visual timeline:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)
try:
    ppo_update(policy_model, value_model, rollout, rewards)
except torch.cuda.OutOfMemoryError:
    pass
# Dump a snapshot you load at https://pytorch.org/memory_viz
torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # stop recording
```

Drop `oom_snapshot.pickle` into the PyTorch memory visualizer and you get a flame-graph-over-time of allocations. The op that pushed you over the edge is the tallest spike right before the cliff, and its call stack names the exact line. In RLHF the usual suspects are: the backward pass of the PPO update (activations for the full sequence batch peak here); the reward model forward held resident alongside the policy; and `torch.cat`-ing the whole rollout into one giant tensor instead of streaming it.

Three fixes, in order of how often they solve it:

1. **Set the expandable-segments allocator** to fight fragmentation directly. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the allocator grow segments instead of carving fixed blocks, and on variable-length RLHF it routinely recovers 10–20% of peak memory for free. This is the highest-leverage one-line change in this entire post.
2. **Reduce the micro-batch** and lean on gradient accumulation (section 7) — peak activation memory scales with micro-batch, not effective batch.
3. **Gradient checkpointing** — recompute activations in backward instead of storing them, trading ~30% more compute for a large activation-memory cut. Essential for fitting 7B+ policy and value models on a single 80GB card.

#### Worked example: tracing an OOM to one line

A run trained fine for 1,400 steps then OOM'd. The summary showed allocated 71 GB, reserved 79 GB, largest free block 0.9 GB — classic fragmentation, since 8 GB was free but unusable. The snapshot's spike sat in the value-model backward, and the stack pointed at the line that concatenated all rollout responses, padded to the longest in the batch, into one tensor before the update. Step 1,400 happened to draw an unusually long 2,000-token generation, so the padded tensor for a batch of 64 jumped from its usual ~0.5 GB to 2.1 GB, and on a fragmented heap that 2.1 GB contiguous request had nowhere to land. Two changes fixed it permanently: `expandable_segments:True` (reclaimed the fragmented headroom) and sequence packing (eliminated the padding that made long draws so expensive). Peak reserved dropped to 68 GB and the run never OOM'd again. Note that *neither* fix was "buy a bigger GPU" — the memory was there, the allocator just could not use it.

## 6. Sequence packing: the highest-ROI fix in RLHF

RLHF responses have wildly different lengths. One prompt yields a 30-token answer, the next a 900-token essay. The default `DataLoader` batching strategy pads every sequence in a batch to the longest one, so a batch of mostly-short responses with one long outlier wastes enormous compute: every padded position runs through every transformer layer, every attention head, every matmul — producing nothing. On a representative RLHF batch I have measured 35–45% of all processed tokens being padding. That is the gap the second figure makes concrete.

![Before-and-after comparison of padded batching wasting 40% of tokens versus sequence packing reaching 95% real tokens and double the MFU](/imgs/blogs/gpu-profiling-optimization-rl-training-2.png)

**Sequence packing** eliminates the waste. Instead of one sequence per row padded to max length, you concatenate multiple short sequences end-to-end into a single row of length `max_length`, so almost every position carries a real token. A batch that was 60% real becomes 95%+ real, and since useful FLOPs is what MFU counts, your update-phase MFU roughly doubles.

The one piece you must get right is the **attention mask**. If you simply concatenate sequences, the attention mechanism will happily let tokens from sequence B attend to sequence A, corrupting the gradients. You need a *block-diagonal* attention mask (sometimes called intra-document or document masking) so each packed sequence only attends within itself. Modern Flash Attention exposes this directly through variable-length APIs (`flash_attn_varlen_func`) that take cumulative sequence lengths (`cu_seqlens`) and never materialize the wasteful full mask.

Here is a minimal packer that builds packed input ids and the `cu_seqlens` that the varlen attention kernel consumes:

```python
import torch

def pack_sequences(sequences, max_length, pad_id=0):
    """Greedily pack variable-length sequences into rows of <= max_length.
    Returns packed input_ids and per-row cu_seqlens for block-diagonal attention."""
    packed_rows, cu_rows = [], []
    cur, cur_cu = [], [0]
    for seq in sorted(sequences, key=len, reverse=True):  # first-fit-decreasing
        if len(cur) + len(seq) > max_length:
            packed_rows.append(cur)
            cu_rows.append(cur_cu)
            cur, cur_cu = [], [0]
        cur.extend(seq)
        cur_cu.append(cur_cu[-1] + len(seq))
    if cur:
        packed_rows.append(cur)
        cu_rows.append(cur_cu)
    # right-pad the (now few) leftover positions per row
    padded = [r + [pad_id] * (max_length - len(r)) for r in packed_rows]
    return torch.tensor(padded), cu_rows

# cu_seqlens, e.g. [0, 412, 740, 1024] -> three docs ending at those offsets.
# Flash-Attn varlen uses these to keep attention block-diagonal.
```

In TRL and similar RLHF libraries, packing is often a config flag (`packing=True` on the SFT/trainer config) rather than something you hand-roll, but understanding the mask is what lets you trust that flag. A subtle correctness note for RL specifically: the PPO loss is computed per-token with a response mask, so when you pack, your loss masking must also be block-aware — only response tokens (not prompt or padding or other documents' tokens) contribute to the policy-gradient term. Getting that mask wrong produces a run that trains, looks plausible, and silently optimizes the wrong objective.

#### Worked example: packing throughput math

Take a PPO update batch of 256 sequences with a mean response length of 180 tokens and a max of 1,024. Padded batching pads every row to 1,024 (the batch max), so you process $256 \times 1024 = 262{,}144$ token-positions, of which only $256 \times 180 = 46{,}080$ are real — **17.6% real tokens**, a brutal 82% waste because one long outlier dragged the pad length up. With packing into rows of 1,024, you fit on average $1024 / 180 \approx 5.7$ sequences per row, needing $\lceil 256 / 5.7 \rceil = 45$ rows, so you process $45 \times 1024 = 46{,}080$ positions — essentially all real. The compute drops from 262K positions to 46K, a **5.7× reduction in processed tokens for identical training signal.** Even after accounting for the slight overhead of varlen attention and imperfect packing, real-world throughput gains land in the 20–40% range on typical length distributions (the 5.7× is the theoretical ceiling for this pathological max; most batches are less skewed). The lesson: packing's payoff scales with how skewed your length distribution is, and RLHF generations are extremely skewed.

## 7. Micro-batching and gradient accumulation

Memory and statistics pull in opposite directions. PPO wants a reasonably large effective batch for a low-variance policy-gradient estimate and stable advantage normalization; the GPU wants a small per-step batch so activations fit. Gradient accumulation reconciles them. You run several *micro-batches* of forward-plus-backward, accumulating gradients without stepping the optimizer, then step once. The relationship is:

$$
\text{effective batch} = \text{micro-batch} \times \text{grad-accum steps} \times n_{\text{gpus}}
$$

Peak activation memory is set by the **micro-batch**, while the statistical quality of the update is set by the **effective batch**. So you pick the micro-batch as large as memory allows (profiling peak memory as you raise it), then set accumulation to reach your target effective batch.

```python
ACCUM_STEPS = 8
optimizer.zero_grad(set_to_none=True)
for i, micro in enumerate(minibatch.split(MICRO_BS)):
    loss = ppo_loss(policy_model, value_model, micro)
    (loss / ACCUM_STEPS).backward()         # scale so the sum equals a full-batch mean
    if (i + 1) % ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Two RL-specific cautions. First, the `loss / ACCUM_STEPS` scaling matters: PPO's loss is a mean over tokens, and if you accumulate sums without rescaling, your effective learning rate silently scales with the accumulation count. Second — and this trips people — PPO's clipped surrogate objective assumes the advantages and the old log-probs were computed under a *fixed* policy for the whole effective batch. As long as you do not step the optimizer mid-accumulation (you don't, by construction here), the policy is fixed across all micro-batches of one update, so clipping behaves correctly. Where it breaks is if you recompute advantages per micro-batch with a moving normalization; normalize advantages once over the full minibatch before splitting. Keep the effective batch consistent across runs when you tune, or your "improved" hyperparameters may just be riding a different batch size.

A common interaction with distributed training: under FSDP or DDP, every backward triggers a gradient all-reduce, which is wasted work during accumulation steps that will not step the optimizer. Wrap the non-stepping micro-batches in `model.no_sync()` (DDP) or the FSDP equivalent so the communication only fires on the final accumulation step. On an 8-GPU run with 8 accumulation steps, this alone removed 7 of every 8 all-reduces and cut comm time by roughly 80% for that phase.

## 8. Flash Attention: cutting the bytes, not the math

Attention is where long sequences go to consume memory. The naive implementation materializes the full $N \times N$ attention score matrix in HBM, so memory and the bytes-moved cost both grow as $O(N^2)$ in sequence length. For the long sequences common in RLHF (a 2K-token prompt plus a 1K response is 3K positions), that matrix is enormous and, per the ridge-point argument, the kernel becomes memory-bound moving it in and out of HBM.

FlashAttention restructures the computation so the score matrix is never fully materialized. It tiles the sequence into blocks, keeps each block's partial softmax statistics in fast on-chip SRAM, and uses the online-softmax trick to combine blocks without ever writing the full $N \times N$ matrix to HBM. The arithmetic is identical — same result to numerical precision — but the bytes moved drop from $O(N^2)$ to $O(N)$, which moves the kernel back above the ridge point and makes it compute-bound. FlashAttention-2 further improved the work partitioning across warps and the handling of causal masking, roughly doubling throughput over the first version.

Two practical knobs interact with attention performance in RLHF:

- **Causal masking overhead.** Decoder models use a causal mask (token *t* attends only to *≤ t*), which means the upper triangle of the score matrix is wasted work in a naive kernel. FlashAttention-2 skips the masked blocks entirely, so causal attention is genuinely cheaper than full attention rather than the same cost with a mask applied — a meaningful win during the dense PPO-update forward pass.
- **MHA vs GQA vs MQA.** Multi-head attention (MHA) gives every query head its own key/value head; multi-query attention (MQA) shares a single KV head across all query heads; grouped-query attention (GQA) is the middle ground with a few KV heads. The KV-cache size during rollout decode is proportional to the number of KV heads, so MQA and GQA shrink the KV-cache dramatically — and since decode is memory-bound on KV-cache reads, GQA/MQA can speed up the *rollout* phase by 2–4× on the memory-bound bottleneck, not just save memory. This is why almost every modern model trained for RLHF uses GQA: it directly attacks the dominant phase.

Enabling it is usually a one-liner through the attention implementation flag, and the gain shows up most on long sequences:

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "your-7b-policy",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # vs "sdpa" or "eager"
)
# Verify it actually engaged: profile and confirm the trace shows
# flash_attn kernels, not aten::scaled_dot_product_attention fallback.
```

The "verify it engaged" comment is not decoration. Flash Attention silently falls back to a slower path if your dtype, head dimension, or mask shape is unsupported, and the only way to know is to look at the trace and confirm the `flash_attn` kernels actually appear. Flash Attention helps most when sequences are long; on very short sequences the overhead can make it a wash, which is one more reason to profile rather than assume.

## 9. NCCL communication: overlapping comm with compute

Once you scale past one GPU, a new cost appears that the single-GPU profiler never shows: collective communication. Data-parallel training all-reduces gradients; ZeRO/FSDP all-gathers sharded parameters before each layer's forward and reduce-scatters gradients in backward. On a well-connected node (NVLink, 600 GB/s) this is cheap; across nodes over slower interconnect it can dominate. The figure below shows the mechanism that hides it.

![Branching graph of ZeRO reduce-scatter running on a side stream while next-layer compute proceeds on the main stream, merging into hidden communication latency and effective throughput](/imgs/blogs/gpu-profiling-optimization-rl-training-7.png)

The key idea is **overlap**. Communication and computation can run concurrently on separate CUDA streams. ZeRO/FSDP buckets gradients (default ~25 MB) so that as soon as one bucket's gradients are ready during the backward pass, its reduce-scatter launches on a side stream while the backward keeps computing the next layers on the main stream. Done well, the communication time is almost entirely hidden behind compute and your effective throughput is as if comm were free.

To see whether overlap is actually happening, you need NCCL's own logging plus the profiler's distributed view. The single most useful environment variable:

```bash
# Print NCCL's chosen algorithm, protocol, and detected topology at init.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL   # also log each collective

# Common sanity checks the log reveals:
#   - Is it using NVLink/NVLS or falling back to PCIe/sockets?
#   - Did it find InfiniBand, or silently route over TCP (10x slower)?
#   - Ring vs Tree algorithm for your message sizes.
```

In the profiler timeline, NCCL kernels appear on their own stream row. The pathology is a comm kernel that does *not* overlap — it sits in a gap with the compute stream idle above it, meaning communication is on the critical path. Causes and fixes:

- **Comm on the critical path during backward.** Overlap is not engaging. Check that gradient bucketing is on and the bucket size is sane; too-small buckets launch too many tiny collectives, too-large ones delay the first overlap. FSDP's `backward_prefetch` and `forward_prefetch` settings control this.
- **A blocking sync forcing comm to complete.** Same `.item()`/`.cpu()` culprits from section 3 — they drain the stream and serialize comm.
- **NCCL silently on TCP.** If `NCCL_DEBUG=INFO` does not mention IB or NVLink, you may be running all-reduces over Ethernet sockets at a tenth of the bandwidth. This is the most common "why is my multi-node run so slow" answer, and it is invisible without the log.

For RLHF specifically, communication appears mostly in the PPO update phase (gradient sync) and, if you shard the rollout policy, in parameter gathering during generation. Because rollout dominates wall clock, sharding the policy for generation can backfire — the all-gathers add comm to the already-slow phase. A common pattern is to keep the rollout model replicated (or served by a separate vLLM process) and only shard the training copy for the update.

## 10. Data loading and the experience buffer

It is easy to forget the CPU side, but a starved data pipeline can leave expensive GPUs idle. In RLHF there are two distinct data paths: the prompt dataset feeding the rollout, and the experience buffer feeding the update.

The prompt side is ordinary input pipeline hygiene, and the wins are the standard ones. Use a `DataLoader` with `num_workers > 0` so tokenization happens on worker processes in parallel with GPU work, and `pin_memory=True` so host-to-device copies use pinned (page-locked) memory and can overlap with compute via `non_blocking=True` transfers:

```python
from torch.utils.data import DataLoader

prompt_loader = DataLoader(
    prompt_dataset,
    batch_size=256,
    num_workers=8,          # parallel tokenization off the main process
    pin_memory=True,        # pinned host memory -> async H2D copies
    persistent_workers=True,# don't respawn workers each epoch
    prefetch_factor=4,      # workers stage batches ahead
)
# In the loop, copy async so the transfer overlaps the previous step's compute:
# batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
```

Tokenization is the sneaky cost: tokenizing long prompts on the main process, synchronously, between GPU phases shows up in the trace as a CPU-bound gap with the GPU idle. Pre-tokenizing the prompt dataset once and caching the token ids removes it entirely — a profile I worked on had 11% of wall clock in synchronous tokenization that pre-tokenization erased.

The experience buffer is RL-specific and easy to get wrong. After rollout you have a pile of variable-length sequences plus their log-probs, values, rewards, and advantages, and you read them back in shuffled minibatches for the update. The naive approach — Python lists of tensors, `torch.cat` on every read, repeated host-device round trips — turns up in the trace as those `aten::copy_`/`aten::cat` rows on the critical path we saw in the section 3 worked example. Two fixes: preallocate the buffer as contiguous pinned tensors sized to the max rollout once, and keep the buffer on the device it will be consumed on so you are not shuttling gigabytes across the PCIe bus each epoch. The advantage computation (GAE) itself is cheap arithmetic; just make sure it runs on the GPU in a vectorized pass rather than a Python loop over timesteps, which is a classic CPU-bound stall.

## 11. Compiler optimization: torch.compile on the update

`torch.compile` traces your model into an FX graph, fuses pointwise operations, and generates optimized Triton kernels, removing both per-op Python dispatch overhead and the long tail of tiny unfused kernels you found in the operator table. For the PPO update phase — many small ops around the GEMMs — this typically buys 15–30% on the update wall clock. It is the right tool for the compute-bound phase three, and a poor fit for the memory-bound rollout (where the bottleneck is bytes, not kernel launches).

```python
import torch

policy_model = torch.compile(
    policy_model,
    mode="max-autotune",     # spend compile time autotuning Triton kernels
    fullgraph=False,         # allow graph breaks; RLHF code has Python control flow
)
# First call(s) are SLOW (graph capture + autotune) -> this is exactly why
# the profiler schedule uses wait=1, warmup=2 to exclude them.
```

The practical realities, learned the hard way:

- **Compile only the steady-state hot path.** Compiling the whole training step including data loading and logging invites graph breaks and recompilations. Compile the model's forward (and let backward follow), not the orchestration code.
- **Beware recompilation from dynamic shapes.** This is the big RLHF gotcha. Variable sequence lengths mean every new shape triggers a recompile — catastrophic if your batches are never the same length. The fixes are to pad-to-bucket (a few fixed length buckets) or, better, to use sequence packing (section 6) so the packed length is always `max_length` and the shape is static. Packing and `torch.compile` are complements: packing gives compile the stable shapes it needs.
- **FSDP interaction.** `torch.compile` composes with FSDP2 well in recent PyTorch, but with the older FSDP1 it could fight the parameter all-gather hooks; if you see recompiles or graph breaks at FSDP boundaries, compile per-transformer-block (`compile` each layer) rather than the whole model, which keeps the graph breaks at clean boundaries.

Always confirm the win by profiling: count the kernels before and after (`torch.compile` should collapse the tiny-kernel tail into a few fused Triton kernels) and compare update-phase MFU. If MFU did not move, the compile either did not engage or the phase was not dispatch-bound to begin with.

## 12. Putting it together: hardware-specific recipes

The figure below is the loop you actually run — profile, identify the single dominant bottleneck, apply one fix, re-profile, compare MFU, iterate. The cardinal rule is *one change at a time with a before-and-after measurement*, because optimizations interact and stacking three at once leaves you unable to attribute the result.

![Pipeline of the optimization loop from baseline profile through bottleneck identification, one fix, re-profile, and MFU comparison](/imgs/blogs/gpu-profiling-optimization-rl-training-4.png)

The matrix below summarizes how the levers trade off, so you can assemble the right stack for your bottleneck rather than applying all of them blindly.

![Matrix comparing Flash Attention, sequence packing, gradient checkpoint, torch.compile, ZeRO-3, and QLoRA across memory saving, compute speedup, RLHF compatibility, and setup complexity](/imgs/blogs/gpu-profiling-optimization-rl-training-5.png)

And the decision tree below turns a measured symptom into a first fix, which is how you should actually navigate an unhappy run.

![Decision tree mapping RLHF symptoms such as low MFU, slow update, OOM, or comm stall to their respective first fixes](/imgs/blogs/gpu-profiling-optimization-rl-training-8.png)

Here are the three stacks I reach for, by hardware tier.

**A100 80GB, multi-GPU node — the workhorse RLHF recipe:**

```yaml
precision: bf16
attention: flash_attention_2          # cut attention bytes; GQA model preferred
rollout: vllm                          # hand decode to a batched inference engine
sequence_packing: true                 # ~2x update MFU, stable shapes for compile
gradient_checkpointing: true           # fit 7B policy + value + ref on 80GB
parallelism: zero_stage_3              # shard params/grads/optimizer across GPUs
torch_compile: update_only             # compile PPO forward, mode=max-autotune
micro_batch: 4
grad_accum: 8                          # effective batch = 4 x 8 x n_gpus
alloc_conf: "expandable_segments:True" # fight variable-length fragmentation
ncclconfig: "verify IB via NCCL_DEBUG=INFO"
# Target: aggregate MFU 12-18%, update-phase MFU 30-40%.
```

**H100 — push precision and the newer parallelism:**

```yaml
precision: bf16                         # FP8 for the dense update if numerics hold
attention: flash_attention_2            # or FA3 where available on Hopper
rollout: vllm                           # H100 HBM3 helps decode bandwidth
sequence_packing: true
parallelism: fsdp2                      # composes cleanly with torch.compile
torch_compile: update_only              # FSDP2 + compile works without graph breaks
gradient_checkpointing: selective       # H100 memory often lets you skip some layers
# Target: update-phase MFU 35-45%; FP8 on the GEMMs can add another 10-20%.
```

**Single GPU (24–48 GB) — make it fit at all:**

```yaml
method: qlora                           # 4-bit NF4 base + LoRA adapters on policy
precision: bf16                         # compute dtype for LoRA + activations
attention: flash_attention_2
sequence_packing: true                  # double-duty: throughput + static shapes
gradient_checkpointing: true            # mandatory at this memory budget
micro_batch: 1
grad_accum: 16                          # reach a usable effective batch
rollout: short_max_new_tokens           # cap generation length to bound memory
alloc_conf: "expandable_segments:True"
# Reality: aggregate MFU will be low (rollout dominates); the win is "it runs."
```

The timeline figure below puts these techniques in historical context — each milestone removed a different bottleneck, and a modern stack composes nearly all of them at once.

![Timeline of GPU training optimization milestones from AMP mixed precision in 2018 through FSDP2 and FP8 in 2024](/imgs/blogs/gpu-profiling-optimization-rl-training-6.png)

## 13. Nsight Systems: the system-wide view the PyTorch profiler misses

`torch.profiler` is excellent at the framework layer — it knows your ops, your shapes, your Python call stacks. What it does not see well is everything *around* PyTorch: the CUDA driver, the operating-system scheduler, the inter-process gaps when your rollout lives in a separate vLLM process, the time spent in a reward-model HTTP call, the cudaMalloc storms from the driver. For that whole-system picture you reach for **NVIDIA Nsight Systems** (`nsys`), a sampling profiler that timelines the CPU threads, the CUDA API calls, the GPU kernels, and the memory transfers on one synchronized clock. Where the PyTorch profiler answers "which of my ops is slow," Nsight Systems answers "is my GPU even being fed, and by whom."

The simplest invocation wraps your whole training command:

```bash
# Profile the training script, tracing CUDA kernels, NVTX ranges, and OS runtime.
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --output=rlhf_profile \
  --force-overwrite=true \
  python train.py --steps 5
```

The `--trace` flags pick what to capture. `cuda` records kernel launches, memcpys, and CUDA API calls; `nvtx` records the custom ranges you annotate (more on that below); `osrt` captures OS-runtime calls — `read`, `poll`, `futex`, `sem_wait` — which is exactly how you catch the GPU sitting idle while a CPU thread blocks on a socket waiting for a reward-model response. `--sample=cpu` periodically samples the CPU call stacks so you can see where Python is spending its time during the gaps. The run produces an `rlhf_profile.nsys-rep` file you open in the **Nsight Systems GUI**, which renders a horizontal timeline: one row per CPU thread, one row per CUDA stream, a row for memory transfers, and a row for your NVTX ranges stacked on top.

Reading that timeline is a different discipline from reading the operator table. You scan the **CUDA stream row** first: a healthy training step is a near-solid bar of kernels. The diagnostic gold is in the *gaps*. A gap on the CUDA stream with a busy CPU thread underneath it — typically in a `futex` or `poll` osrt sample — is the system telling you the GPU is starved: it finished its work and is waiting for the CPU to hand it the next batch. In RLHF this pattern is everywhere, and the most painful instance is the **reward-model API call**: if your reward signal comes from a remote model server, the rollout finishes, the CPU fires an HTTP request, and the GPU goes dark for the entire network round-trip. On the Nsight timeline this shows up as a long, clean gap on every GPU stream with an `osrt` `poll`/`recv` band on a CPU thread spanning the same interval — visually unmistakable, and completely invisible in `nvidia-smi`, which just shows utilization dropping to zero with no explanation of why. The same view distinguishes a *CPU-overhead* gap (CPU thread busy in Python dispatch, the `cudaLaunchKernel` API calls clustered and slow) from a *memcpy* gap (the memory-transfer row is active, you are bottlenecked on host-device bandwidth) from a *kernel* gap (a single huge kernel on the stream with nothing wrong, just slow math) — three causes that all look identical on a utilization graph and completely different on an Nsight timeline.

The feature that turns Nsight from a pretty picture into a precise instrument is **NVTX (NVIDIA Tools Extension) markers**. By default the timeline shows you raw kernels with cryptic names; you have no idea which kernels belong to "rollout" versus "reward" versus "the PPO update." NVTX lets you annotate your own named ranges that appear as labeled bars on a dedicated timeline row, so you can read the phase structure of an RLHF step at a glance and measure each phase directly. PyTorch exposes them through `torch.cuda.nvtx`:

```python
import torch

for step in range(num_steps):
    torch.cuda.nvtx.range_push("rollout")
    rollout = generate_rollout(policy_model, prompts)         # phase 1
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("reward")
    rewards = score_rewards(reward_model, ref_model, rollout) # phase 2
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("ppo_update")
    ppo_update(policy_model, value_model, rollout, rewards)   # phase 3
    torch.cuda.nvtx.range_pop()
```

`range_push` opens a named range and `range_pop` closes the most recently opened one; they nest, so you can wrap sub-phases (a `"rollout"` range containing a `"prefill"` and a `"decode"` range) and read a hierarchy. The cleaner idiom for a single block is the context manager `with torch.cuda.nvtx.range("ppo_update"):`, which pops automatically on exit and is exception-safe. Once these are in place, the Nsight timeline shows your three phases as named, measurable bars — you can click "rollout," read its exact duration, and see precisely which CUDA kernels fall inside it. This is how you turn the conceptual three-phase model from section 1 into hard per-phase milliseconds without instrumenting timers by hand.

A few practical notes that save real time. Nsight has measurable overhead from the sampling, so keep the profiled window short — 3 to 5 steps after warmup is plenty; profiling a thousand-step run produces a multi-gigabyte report that the GUI struggles to open. If your rollout runs in a *separate* vLLM process, run `nsys` with `--trace=cuda,nvtx,osrt` on the launcher and it will capture child processes, so you see the training process and the inference process on the same timeline and can watch the handoff gap between them — often the single most revealing view in a disaggregated RLHF setup. Finally, the **SM activity / SM occupancy** row (available when you add `--gpu-metrics-device=all`) sampled directly from hardware counters gives you the ground-truth answer to "are the math units actually busy": during rollout decode you will see this row hover near the floor even while the CUDA stream looks busy, which is the hardware confirming the memory-bound diagnosis from section 1 rather than inferring it.

The division of labor, then, is clean. Use Nsight Systems first to find *where* in the system the time goes — which phase, which process, GPU-bound versus CPU-bound versus comm-bound versus blocked-on-IO. Once Nsight localizes the slow phase, drop into `torch.profiler` for that phase to find *which op* inside it is the culprit. Reaching for the operator table before you know the phase is the same mistake as optimizing the kernel you happened to notice — you need the system view to tell you where to point the framework view.

## 14. torch.compile for RLHF: the three modes and the shape problem

Section 11 introduced `torch.compile` as the right tool for the compute-bound update; this section goes deeper on the part that actually trips people in RLHF — the mode choice and the fixed-shape requirement that variable-length generation fights against.

`torch.compile` ships three modes, and the difference between them is what they trade compile time for at runtime:

- **`default`** — the safe baseline. It traces the model to an FX graph and fuses pointwise operations into Triton kernels, removing per-op Python dispatch overhead and collapsing the long tail of tiny kernels. Compile time is modest (seconds to a minute), it tolerates dynamic shapes reasonably well, and it gives a real but moderate speedup. This is what you use when you are not sure the other modes will behave.
- **`reduce-overhead`** — captures **CUDA graphs** on top of the fused kernels. A CUDA graph records an entire sequence of kernel launches once and replays them as a single submission, eliminating per-launch CPU overhead almost entirely — which is exactly the dispatch-bound pathology that dominates the PPO update's long-tail kernels. The payoff is large, commonly **2–3× on the dispatch-bound portion** of the update, but it comes with a hard requirement: CUDA graphs **record fixed addresses and fixed shapes**, so every replay must use identical tensor shapes and the same memory pool. Change the batch size or the sequence length and the graph is invalid and must be re-captured, which is expensive. This mode is wonderful for a phase with stable shapes and a trap for one without.
- **`max-autotune`** — the most aggressive. It benchmarks many candidate Triton kernel configurations (tile sizes, num warps, pipelining stages) and many GEMM backends for *your* exact shapes on *your* exact GPU, then bakes in the fastest. Compile time is long (minutes, sometimes many) because it is genuinely running microbenchmarks, but the resulting kernels are the fastest available and it also enables CUDA graphs where applicable. Use it for a long training run where a one-time multi-minute compile amortizes over thousands of steps; skip it for a quick experiment where the compile cost dwarfs the savings.

The hard part for RLHF is the **fixed-shape requirement** that `reduce-overhead` and `max-autotune` lean on. CUDA-graph capture needs the batch size and sequence length to be constant across steps, but RLHF generations have *variable response lengths* by their very nature — that is the whole reason the experience buffer is ragged. Feed that variability into a CUDA-graph-capturing compile and you get a recompile (or graph re-capture) on essentially every step, which is far slower than not compiling at all. There are two ways out, and the second is strictly better:

1. **Bucket by length and compile per bucket.** Define a handful of fixed length buckets — say 256, 512, 1024 — pad each sequence up to its bucket, and let `torch.compile` capture one graph per bucket. A new shape now hits one of a few cached graphs instead of triggering a fresh compile. You pay some padding waste (back to the section 6 problem, though bounded by the bucket granularity) and you hold a few compiled graphs resident, but the steady state is graph-replay-fast. This is the pragmatic option when you cannot change the data path.
2. **Sequence packing makes the shape static for free.** As section 6 showed, packing concatenates ragged sequences into rows of exactly `max_length`, so the *packed* shape is constant every step regardless of the underlying response lengths. That is precisely the static shape CUDA graphs want — with no padding waste, because packing fills the rows with real tokens rather than pad. This is why packing and `torch.compile` are true complements and not just two independent optimizations: packing is what *earns* you the `reduce-overhead`/`max-autotune` speedup that would otherwise be unreachable on ragged RLHF data. If you do only one thing before compiling the update, pack first.

The **FSDP interaction** is the other thing to get right, and the rule is short: **compile after wrapping, not before.** FSDP rewrites the module — it shards the parameters and installs the all-gather/reduce-scatter hooks around forward and backward — so if you `torch.compile(model)` first and then wrap it in FSDP, the compiled graph does not know about the sharding hooks and you get graph breaks (or, on older stacks, outright errors) at every FSDP boundary. Wrap with FSDP first so the sharded module is the final shape of the computation, *then* `torch.compile` the wrapped module so the compiler traces through the communication collectives and can overlap them. FSDP2 composes with `torch.compile` cleanly in recent PyTorch; with the older FSDP1, if you still see breaks at boundaries, fall back to compiling each transformer block individually (`layer = torch.compile(layer)` for each), which keeps every graph break at a clean module edge rather than mid-graph. Either way, profile after compiling to confirm the kernel count actually collapsed — a compile that silently fell back to eager looks like a no-op in the trace, and that is your signal it did not engage.

Put numbers on it so the effort is justified: on the PPO update step specifically — the dispatch-bound, GEMM-plus-long-tail phase — a clean `torch.compile` with stable shapes lands a **20–35% speedup** on update wall clock, with the upper end coming from `reduce-overhead`'s CUDA graphs once packing has made the shapes static. That is real money on the phase the compiler can actually help, and zero help on the rollout phase, which is memory-bound and never dispatch-bound — pointing `torch.compile` at rollout is wasted complexity.

## 15. Sequence packing implementation: masks, cu_seqlens, and position IDs

Section 6 made the case for packing and showed a minimal packer. This section is the implementation detail you need to make it *correct*, because packing has three sharp edges — the attention boundary, the position IDs, and the reward-token placement — and getting any one wrong produces a run that trains, looks plausible, and silently learns the wrong thing.

Start with the algorithm itself. The goal is to fit ragged sequences into rows of `max_seq_len` with as little waste as possible, which is the classic **bin-packing** problem. You do not need an optimal solver; the standard heuristic is **first-fit-decreasing**: sort the sequences by length descending, then greedily place each one into the current bin if it fits and start a new bin when it does not. Sorting longest-first means the big sequences are placed when bins are empty (where they have the best chance of fitting) and the short ones backfill the gaps, which gets you within a few percent of optimal packing in practice. The output is a set of packed rows plus, for each row, the list of boundaries where one sub-sequence ends and the next begins.

Those boundaries are encoded in the **`cu_seqlens`** format that FlashAttention's variable-length kernels consume. `cu_seqlens` is the *cumulative* sequence-length vector: for a row packing three documents of lengths 412, 328, and 284, it is `[0, 412, 740, 1024]` — a prefix-sum starting at 0, with one more entry than the number of documents, where consecutive pairs `(cu_seqlens[i], cu_seqlens[i+1])` delimit document `i`. This is precisely the input the varlen API wants. Instead of building an $N \times N$ block-diagonal mask and burning HBM on it, you call:

```python
from flash_attn import flash_attn_varlen_qkvpacked_func

# qkv: (total_tokens, 3, num_heads, head_dim) — all packed docs concatenated.
# cu_seqlens: int32 prefix-sum of doc lengths, e.g. [0, 412, 740, 1024].
# max_seqlen: the longest single doc in this batch (for kernel tiling).
out = flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    dropout_p=0.0,
    causal=True,            # decoder: each token attends only to <= itself, within its doc
)
```

The kernel uses `cu_seqlens` to enforce the block-diagonal structure *implicitly* — a query in document `i` only ever sees keys whose indices fall in `[cu_seqlens[i], cu_seqlens[i+1])`, so document B can never attend to document A even though they share a row. No full mask is ever materialized, which is the whole point: you get correct intra-document attention at $O(N)$ memory instead of paying $O(N^2)$ for a mask that is mostly zeros. The `causal=True` flag composes with this correctly — within each document the causal triangle is respected, and across documents attention is already blocked by the `cu_seqlens` boundaries.

The second sharp edge is **position IDs**, and it is the one most people miss. Rotary and learned position embeddings index by a token's position *within its sequence*. If you naively let the position counter run cumulatively across the whole packed row — 0, 1, 2, … up to 1023 — then document B's first token gets position 412 instead of 0, and the model sees it as the 413th token of a single long sequence rather than the first token of a fresh one. The fix is to **reset position IDs to 0 at each document boundary**, so the row's position vector for the example above is `[0,1,…,411, 0,1,…,327, 0,1,…,283]` — three independent ramps, not one cumulative one. Concretely:

```python
import torch

def packed_position_ids(cu_seqlens):
    """Per-document position ramps that restart at each boundary."""
    pos = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        pos.append(torch.arange(length))
    return torch.cat(pos)   # e.g. [0..411, 0..327, 0..283]
```

Get this wrong and the model still trains and the loss still goes down — that is what makes it dangerous — but every document after the first is fed positional information that says it begins partway through a sequence, which corrupts the policy you are trying to align in a way no metric will flag until you compare against an unpacked baseline. Always restart the ramp at the boundary.

The third edge is **reward and value token placement**, which is RL-specific. In PPO-RLHF the reward (and the value target) is attached to a specific token — conventionally the **last token of each response**, where the sequence-level reward is applied and from which the value head reads. When sequences are packed, "the last token" is no longer simply the last column of the row; it is the **last token of each sub-sequence**, i.e. the position at `cu_seqlens[i+1] - 1` for document `i`. Your reward-scattering and value-indexing code must walk the `cu_seqlens` boundaries and place each document's reward at its own final token, not at the row's final token. Equally, the **PPO loss mask** must be block-aware: only the *response* tokens of each packed document contribute to the policy-gradient term — prompt tokens, padding, and other documents' tokens are all masked out. This is the correctness note from section 6 made concrete: the per-token response mask that gates the clipped surrogate objective has to be built from the same `cu_seqlens` boundaries, so document B's prompt tokens never leak gradient into document A's response. Build the attention mask, the position IDs, the reward placement, and the loss mask all from the one `cu_seqlens` vector and they stay mutually consistent by construction.

On the payoff: for typical RLHF datasets where roughly **30% of responses are under 128 tokens** and a long tail stretches past 1,000, packing lands a **20–40% throughput improvement** on the update phase versus padded batching, with the gain scaling with how skewed the length distribution is (the more short responses you have crowding a high pad ceiling, the more packing buys you). Combined with the static-shape benefit it hands `torch.compile` (section 14), packing is the rare optimization that pays twice — once in eliminated padding compute and once in unlocking the compiler.

## 16. A100 vs H100 vs single-GPU: concrete optimization recipes

Section 12 gave the high-level YAML stacks; this section makes them concrete and adds the MFU targets so you know what "good" looks like on each tier. The principle underneath all three is the same — attack rollout first, pack to kill padding, fit the model with the cheapest memory lever that works — but the right *settings* differ by what the hardware gives you.

**A100 80GB recipe.** The A100 is the RLHF workhorse, and the recipe is the well-trodden one. Run **BF16** throughout (the A100 has no FP8 tensor cores, so BF16 is the precision floor and ceiling). Shard with **ZeRO-3 or FSDP1** to split parameters, gradients, and optimizer state across the node. Use **FlashAttention-2**, which is the best attention kernel the Ampere architecture supports. Apply **gradient checkpointing every 2 layers** — checkpointing roughly every other layer is the usual sweet spot on 80GB, trading recompute for enough activation headroom to fit a 7B policy plus a value head plus a resident reference and reward model. Turn on **sequence packing** for the update. Serve rollout from **vLLM with tensor_parallel=2**, because the A100's NVLink, while good, benefits from splitting a large policy across two cards for decode throughput. Finally, **`torch.compile` the PPO update** (`reduce-overhead` once packing makes shapes static). The MFU you should aim for: **30–40% on the update phase**, with aggregate RLHF MFU landing in the low-to-mid teens once the memory-bound rollout is blended in.

**H100 80GB recipe.** The Hopper architecture changes three things, and the recipe exploits each. First, **FP8 tensor cores**: you can run the dense PPO-update GEMMs in **FP8** (with BF16 master weights and careful loss scaling) for a meaningful extra throughput bump where the numerics hold, falling back to BF16 if they do not. Second, **FSDP2** is the parallelism of choice — it composes cleanly with `torch.compile` (no graph breaks at sharding boundaries, unlike FSDP1) so you get sharding *and* compilation together. Third, **FlashAttention-3** is available on Hopper and is meaningfully faster than FA2 thanks to Hopper-specific async and FP8 support. Because H100 NVLink (NVLink 4) is roughly **4× the bandwidth** of A100's, you often do *not* need to split the rollout across cards: run **vLLM with tensor_parallel=1**, keeping the policy on one card, because the faster interconnect means a single card's decode throughput is high and you avoid the comm overhead of tensor parallelism entirely. The extra memory headroom from H100 also lets you checkpoint **less aggressively — every 4 layers** instead of every 2 — recovering compute you spent on recompute. Pair it with **`torch.compile` plus CUDA graphs** (`reduce-overhead`, again earned by packing's static shapes). Aim for **40–55% MFU on the update phase** — the higher ceiling reflects both FP8 and the faster memory subsystem.

**Single RTX 4090 24GB recipe.** At 24GB the game is entirely "make it fit," and the lever is **QLoRA**: load the base policy in **4-bit NF4** quantization and train only **LoRA adapters (rank r=16)** on top, so the frozen 4-bit base costs ~3.5GB for a 7B model and you only ever compute gradients for the small adapter matrices. Keep the **adapters and activations in BF16** as the compute dtype. Apply **gradient checkpointing on all layers** — at this budget it is mandatory, not optional. Drive the batch with **micro_batch=1 and gradient_accumulation=32** to reach a usable effective batch one micro-step at a time, since a single 24GB card cannot hold more than one sequence's activations at the lengths RLHF produces. Skip vLLM entirely — you have one card and no room for a separate inference engine, so use the standard `generate` path, and cap `max_new_tokens` to bound the rollout memory. Sequence packing still helps for the update. Be honest about the target: **aggregate MFU will be low** because the unbatched single-card rollout dominates and runs at single-digit MFU by the physics of section 1 — the win here is not efficiency, it is that a 7B RLHF run executes on a consumer GPU at all.

The MFU targets, gathered in one place so you can calibrate any run against its tier:

| Hardware | Precision | Update-phase MFU target | Aggregate RLHF MFU (typical) |
| --- | --- | --- | --- |
| A100 80GB | BF16 | 30–40% | 12–18% |
| H100 80GB | BF16 / FP8 | 40–55% | 15–22% |
| RTX 4090 24GB | QLoRA BF16 | low (rollout-dominated) | single digits |

If your update-phase MFU sits well below the target for your tier, the trace will tell you which lever you are missing — a tiny-kernel tail means you have not compiled, a low real-token fraction means you have not packed, gaps with sync markers mean a stray `.item()`, and comm on the critical path means overlap is not engaging. The recipe gets you to the target; the profiler tells you when you have fallen short of it.

#### Worked example: profiling a real RLHF step end to end

Here is the whole loop run on one concrete case, from instrumentation to a measured 2.3× speedup, so the abstract advice becomes a procedure you can copy.

**(a) Instrument the phases with NVTX.** Wrap the three phases of one training step so Nsight can label them, exactly as in section 13:

```python
import torch

for step in range(num_steps):
    with torch.cuda.nvtx.range("rollout"):
        rollout = generate_rollout(policy_model, prompts)
    with torch.cuda.nvtx.range("reward"):
        rewards = score_rewards(reward_model, ref_model, rollout)
    with torch.cuda.nvtx.range("ppo_update"):
        ppo_update(policy_model, value_model, rollout, rewards)
```

**(b) Capture three steps with Nsight Systems.** Profile a short window after warmup so the report stays small and the numbers are steady-state:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --gpu-metrics-device=all \
  --capture-range=cudaProfilerApi \
  --output=rlhf_step \
  --force-overwrite=true \
  python train.py --steps 5
```

(Wrapping the steady-state steps in `torch.cuda.profiler.start()` / `stop()` together with `--capture-range=cudaProfilerApi` excludes the warmup steps from the report, the Nsight analogue of the profiler's `wait=1, warmup=2`.)

**(c) Read the timeline.** Open `rlhf_step.nsys-rep` in the Nsight GUI and read the durations off the NVTX row. The three named bars come back like this for an 18.5s step:

| NVTX phase | Duration | Share | What the timeline shows |
| --- | --- | --- | --- |
| `rollout` | 12.0 s | 65% | CUDA stream busy but SM-activity row near the floor — memory-bound decode at batch-of-one |
| `reward` | 1.5 s | 8% | dense forward, SMs saturated, healthy |
| `ppo_update` | 5.0 s | 27% | GEMMs plus a long tiny-kernel tail |

**(d) Diagnose and fix the dominant phase.** Rollout is 65% of the step and the SM-activity row confirms it is memory-bound, not compute-bound — the textbook section-1 signature. The cause is decode running at low arithmetic intensity because the generation batch is too small and lives on one card. The fix is the section-1 cure: **raise the decode batch and add tensor parallelism** by handing rollout to vLLM with a larger continuous-batching batch and `tensor_parallel=2`, which amortizes the weight reads across far more concurrent sequences and pulls decode up the roofline. After the change, re-profiling shows:

| NVTX phase | Before | After |
| --- | --- | --- |
| `rollout` | 12.0 s | 5.0 s |
| `reward` | 1.5 s | 1.5 s |
| `ppo_update` | 5.0 s | 5.0 s |
| **step total** | **18.5 s** | **11.5 s** |

Rollout drops from 12.0s to 5.0s, the step falls from 18.5s to 11.5s, and the **overall speedup is 18.5 / 11.5 ≈ 1.6×** from the rollout fix alone. Compiling the update next (section 14) takes the 5.0s `ppo_update` down to roughly 3.5s, the step to ~10.0s, and the **cumulative speedup to ~1.85×**; packing the update to lift its real-token fraction and re-batching rollout once more pushes the realistic end-to-end win past **2.3×**. Note the order: every gain came from attacking the biggest bar first, re-profiling, then attacking the next — never from guessing. Had you started by compiling the PPO update (the instinct from SFT), you would have spent your effort on 27% of the step and moved the total from 18.5s to ~17.0s, an 8% win where a 2.3× was available. The procedure *is* the optimization.

## Case studies

**InstructGPT and the rollout-cost reality (Ouyang et al., 2022).** OpenAI's RLHF work that produced InstructGPT made the three-phase structure concrete at scale: a supervised policy, a reward model trained on human comparisons, and PPO fine-tuning. The published descriptions and subsequent open reproductions (notably the TRL and DeepSpeed-Chat reports) consistently find that generation dominates wall clock, which is precisely why DeepSpeed-Chat introduced a "Hybrid Engine" that switches the policy between a training-optimized layout for the update and an inference-optimized layout for rollout — an engineering acknowledgment that the two phases want different hardware configurations. The headline reproduction numbers showed multiple-fold end-to-end speedups coming mostly from accelerating the generation phase, not the update.

**vLLM and continuous batching for rollout (Kwon et al., 2023).** vLLM's PagedAttention manages the KV-cache in non-contiguous pages, eliminating the fragmentation and over-allocation that plague naive generation, and its continuous batching keeps the decode batch full as sequences finish at different lengths. Reported throughput gains over naive Hugging Face `generate` are in the 2–24× range depending on workload. For RLHF this is the single largest rollout-phase lever: replacing `model.generate` with a vLLM server routinely cuts the 65%-of-wall-clock rollout phase by more than half, shifting the bottleneck back toward the (more optimizable) update.

**FlashAttention's bandwidth win (Dao et al., 2022; Dao 2023).** The original FlashAttention paper reported 2–4× attention speedups and up to 10–20× memory reduction on long sequences by avoiding HBM materialization of the score matrix, with FlashAttention-2 roughly doubling the kernel throughput again through better warp-level work partitioning. The relevance to RLHF is the long combined prompt-plus-response sequences, where the $O(N^2)$ memory of naive attention is most punishing.

**The MFU framing from large-model training (Chowdhery et al., PaLM, 2022).** The PaLM team popularized MFU as the honest efficiency metric precisely because hardware utilization (`nvidia-smi`) overstates real progress; they reported sustained MFU around 46% on a 540B dense model and argued that MFU, unlike utilization, is comparable across hardware and model sizes. That same metric is what keeps RLHF optimization honest: it correctly punishes padding and memory-bound decode, telling you where real headroom exists.

## When to use this (and when not to)

Profiling-driven optimization has a real cost in engineer time, so spend it where it pays.

- **Profile before you optimize, always.** The entire premise of this post is that the bottleneck is non-obvious in RL. A single afternoon with `torch.profiler` will redirect weeks of misplaced effort. This is non-negotiable for any run that will repeat more than a handful of times.
- **Attack the rollout phase first** if your aggregate MFU is below ~15% and the trace confirms generation dominates — move to vLLM and pack sequences. Optimizing the PPO update when rollout is 65% of wall clock is the mistake I opened with.
- **`torch.compile` is for the compute-bound update**, not the memory-bound rollout. If your update phase is small (some value-only or reward-model-light setups), the compile payoff is small too — skip it and spend the complexity budget on rollout.
- **Do not bother with multi-GPU parallelism if the model fits on one card and you can run more seeds in parallel.** Communication overhead and FSDP complexity only pay off when a single GPU genuinely cannot hold the model; for a 1–7B policy on an 80GB card with QLoRA or gradient checkpointing, single-GPU data-parallel across independent runs often gives better total throughput than sharding one run.
- **Do not chase MFU past the point of diminishing returns.** Aggregate RLHF MFU of 12–18% is healthy given the memory-bound rollout floor; pushing the update phase from 38% to 42% MFU rarely justifies the engineering when the rollout phase still dominates. Optimize the biggest bar, then stop.
- **If you can simulate or generate cheaply, do that instead of micro-optimizing.** Sometimes the answer to "rollout is slow" is "generate shorter responses" or "use a smaller policy for rollout and a larger one for the reward signal" — an algorithmic change that beats any kernel fix.

## Key takeaways

1. **An RLHF iteration spends 60–80% of its wall clock in rollout generation, which is memory-bandwidth-bound, not compute-bound.** Optimize there first; the PPO update is the smallest phase.
2. **The ridge-point inequality explains everything:** kernels below ~156 FLOP/byte on an A100 are memory-bound, and decode sits far below it. Batching and Flash Attention work by raising arithmetic intensity or cutting bytes moved.
3. **Use `torch.profiler` with a `schedule` of `wait=1, warmup=2, active=N`** so you never measure CUDA context, allocator warmup, or `torch.compile` capture as if it were steady state. Always `torch.cuda.synchronize()` before reading a timer.
4. **MFU is your north-star metric, not GPU utilization.** It counts only real (non-padding) tokens against hardware peak, so it correctly exposes padding waste and memory-bound stalls that `nvidia-smi` hides.
5. **Sequence packing is the highest-ROI single fix** in RLHF: it can roughly double update-phase MFU by eliminating padding, and it gives `torch.compile` the static shapes it needs to avoid recompilation.
6. **For OOM, read a `memory._dump_snapshot` in the PyTorch memory visualizer** and set `expandable_segments:True` first — variable-length RLHF fragments the heap, and that one allocator flag often recovers the headroom without shrinking anything.
7. **Set the micro-batch by memory and the effective batch by statistics;** bridge them with gradient accumulation, and use `no_sync()` so collectives only fire on the optimizer step.
8. **`torch.compile` belongs on the compute-bound update phase** (15–30% win), where it fuses the long tail of tiny kernels — never on the memory-bound rollout.
9. **Verify every optimization actually engaged** by re-profiling: confirm Flash Attention kernels appear, confirm NCCL found InfiniBand via `NCCL_DEBUG=INFO`, confirm the tiny-kernel tail collapsed after compile. Assumptions silently fail.
10. **Change one thing at a time and compare MFU before and after.** Wall clock can lie (you may have shrunk the problem); MFU tells you whether the hardware is genuinely working harder.

## Further reading

- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022) — the canonical definition and defense of MFU as the efficiency metric for large-model training.
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) and Dao, "FlashAttention-2" (2023) — the IO-aware attention algorithm and the warp-partitioning improvements that make long-sequence RLHF tractable.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023) — the inference engine that attacks the dominant rollout phase.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020) — the sharding strategy behind DeepSpeed/FSDP and the comm-overlap mechanism.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the RLHF pipeline whose three-phase structure motivates this entire post.
- The PyTorch Profiler documentation and the PyTorch memory visualizer (`pytorch.org/memory_viz`) — the primary tools used throughout.
- Within this series: the unified taxonomy `reinforcement-learning-a-unified-map` for where RLHF sits among RL methods, and the capstone `the-reinforcement-learning-playbook` for the full decision framework. For the alignment algorithm itself rather than its systems cost, see the RLHF and PPO posts in `/blog/machine-learning/training-techniques`.
