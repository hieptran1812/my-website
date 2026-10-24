---
title: "Setting Up the Lab: the okc Loop, the SGLang Image, Nsight, and the No-H200 Path"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "How to build a development loop for the Open Kernel Challenge that spends the judge's daily runs on conclusions, not experiments, including what changes when your local GPUs are two L40S cards instead of an H200."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "sglang", "nsight", "cuda-graphs", "h200", "l40s", "profiling", "developer-workflow"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: true
readTime: 13
---

The Open Kernel Challenge gives you ten scored runs and twenty check runs per problem per day, and every one of them executes on a GPU you probably do not own: a single H200 SXM inside the `lmsysorg/sglang:dev-muse-spark-open` image. That arithmetic shapes everything about how you work. If you treat the judge as your compiler, you burn a day's budget learning that a tile size was wrong. If you treat it as a stopwatch you consult only after the code is already right, the same budget carries a week of real progress.

This post builds that lab: the `okc` workflow, the pinned image, a local copy of the judge's clock, Nsight inside the container, and the question most participants face: what if your local hardware is not an H200? The running example is two L40S cards (Ada, SM89), a common rented box whose differences from Hopper are instructive.

![Two machines, one loop: a local workstation with two L40S cards handles correctness, CUDA graph capture and rough timing, while the remote H200 judge receives okc check and okc submit runs and returns a profile](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-1.webp)

The diagram above is the mental model: correctness is decided locally, and the clock that counts lives on the judge's H200.

## 1. The okc workflow, command by command

Install the CLI (the installer checks the download against the release's SHA256SUMS), then sign in. Signing in is a person's job: `okc login` shows the rules and the terms and runs GitHub's device flow. An agent cannot accept the rules on your behalf, and submissions return 403 until a person has.

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
okc login                          # a person: rules prompt, then GitHub device flow
okc problems                       # the five problems and today's limits
okc init muse_bf16_proj lab/proj   # starter, spec and AGENTS.md into lab/proj
okc check lab/proj/submission.py   # full test + timing run, off the board
okc submit lab/proj/submission.py  # the same run, scored
okc status <job> --wait            # a run's result again
okc jobs                           # your recent runs
okc leaderboard muse_bf16_proj --json
```

`okc init` writes more than a starter. You get `submission.py` (plain PyTorch, correct but slow), `baseline.py` (SGLang's own kernel, the one you are timed against), `reference.py` and `checks.py` (the computation and the correctness spec as the judge checks them), `nvfp4.py` for the three NVFP4 problems, plus `TASK.md` and `AGENTS.md`. Keep the first line of `submission.py`, `# okc: problem=muse_bf16_proj`; `okc submit` reads the problem from it.

Two planning details. The judge runs its own copies of `reference.py` and `checks.py` with its own seeds and fresh inputs, so a local pass is necessary, not sufficient. And the same code submitted twice returns the earlier result, so resubmitting an unchanged file does nothing.

If you work from a fresh container or a rented box, do not sign in again there. Run `okc api token` on a machine that is already signed in and export the result as `OKC_TOKEN` in the sandbox.

### What a passing run gives back

A passing run reports each shape's time against SGLang's and your speedup with a 95% interval of at least plus or minus 1.5%, and okc saves `profiles/submission.<job>.txt` (where the time went) and `profiles/submission.<job>.log` (the last 16,000 characters you printed). A failing run names its first failing case. Those two files are your only window into the H200.

## 2. The daily budget is the real constraint

![Five problems each with 20 check tokens and 10 submit coins, a narrower serving column with 3 checks and 1 submit, under a clock that resets at 00:00 UTC](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-4.webp)

The grid above is the whole allowance. Spend checks like experiments and submits like conclusions.

| Resource | Limit (from llms.txt) | What it means in practice |
| --- | --- | --- |
| Scored runs | 10 per problem per day | Your best run is your row; extras only help if they beat it |
| Check runs | 20 per problem per day, separate cap | The H200 stopwatch for experiments |
| Run wall clock | 10 minutes, compile included | A slow `load_inline` build eats your timing window |
| `prepare()` | up to 300 s, untimed | Repack weights here, never in `kernel_fn` |
| Rate limit | 6 submissions/min per token, 10/min per account | Do not script a tight resubmit loop |
| Serving (from Nov 5) | 1 scored + 3 checks per day | Each check is one rung at concurrency 16 |
| Reset | 00:00 UTC | Plan your heavy experiments for just before reset |

Across five kernel problems that is ${5 \times (10 + 20) = 150}$ judge runs a day, and each check answers exactly one question. Write it down before spending the check ("does `BLOCK_N=128` beat 64 on the m=256 verify shapes?"); if you cannot phrase it, the check is premature.

## 3. The image: pin it, then live inside it

Every submission runs inside `lmsysorg/sglang:dev-muse-spark-open`, with no network and nothing else installed. You may import anything already in the image: PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS, plus SGLang itself and its prebuilt `sgl-kernel`. Your local lab should be the same image, not a conda environment that happens to have similar versions, because a Triton or PyTorch version skew changes autotuning results and compile behavior.

Tags move; digests do not. The serving spec (`GET /api/v1/serving`) pins its image as `lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a` alongside SGLang commit `a55ab7d5`. Pull the tag, compare its digest, and record which one you developed against.

```bash
docker pull lmsysorg/sglang:dev-muse-spark-open
docker inspect --format '{{index .RepoDigests 0}}' lmsysorg/sglang:dev-muse-spark-open

# Run the lab: all GPUs, host IPC for PyTorch shared memory, the repo mounted,
# and SYS_ADMIN so Nsight Compute can read the performance counters.
docker run --rm -it --gpus all --ipc=host --cap-add=SYS_ADMIN \
  -v "$PWD/lab:/lab" -w /lab \
  -e OKC_TOKEN \
  lmsysorg/sglang:dev-muse-spark-open bash

# Inside: confirm what you are actually running against.
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
which nsys ncu || echo "Nsight not in the image: bind-mount the host install read-only"
```

![A layered stack of the image's libraries, each marked for whether it runs on Ada SM89: PyTorch, Triton, FlashInfer, Marlin, TileLang yes; CUTLASS and CuTe DSL partly; DeepGEMM and FA3 no](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-2.webp)

The figure above shows the stack and, for later, which layers survive a move to Ada. The FA3 layer is the one to notice: `baseline.py` for attention decode, the decode layer and the prefill layer all import `flash_attn_with_kvcache` from `sgl_kernel.flash_attn`, which is FlashAttention-3, a Hopper kernel.

## 4. Reproducing the judge's clock locally

The judge does not time your function the way `time.perf_counter()` around a call would. Per `TASK.md`, each shape is captured as a CUDA graph and replayed; your replays and SGLang's take turns on the same fresh inputs; every timed replay starts with a cold L2 because the judge reads a buffer larger than L2 just before it; the shape's speedup is the median of paired ratios with the order swapped every other pair; and roughly 4 µs of fixed graph-replay cost is subtracted from both sides. [Inside the judge](/blog/machine-learning/gpu-kernel-engineering/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking) covers why each of those choices exists. Here we only copy them.

`AGENTS.md` says it plainly: at these sizes, configurations that win eagerly can lose under graphs. Eager timing includes Python dispatch and launch overhead that a graph removes, and a 2 µs launch gap is a large fraction of a 1.8 µs router floor. The harness below is illustrative, my sketch of the method rather than the judge's code, and it has not been run here:

```python
# Illustrative local timer in the judge's style. Not the judge's code; not run here.
import statistics
import torch

def capture(fn, data, state, warmup=3):
    """Warm up on a side stream (compile, autotune, allocate), then capture one call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn(data, state)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data, state)
    return graph, out

L2 = torch.cuda.get_device_properties(0).L2_cache_size
SCRUB = torch.empty(4 * L2 // 4, dtype=torch.float32, device="cuda")  # 4x L2 in bytes

def replay_us(graph):
    SCRUB.zero_()                          # evict: every timed replay starts cold
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record(); graph.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1e3

def paired_speedup(g_base, g_mine, reps=60):
    ratios = []
    for i in range(reps):                  # swap the order every other pair
        if i % 2 == 0:
            tb, tm = replay_us(g_base), replay_us(g_mine)
        else:
            tm, tb = replay_us(g_mine), replay_us(g_base)
        ratios.append(tb / tm)
    return statistics.median(ratios)
```

Three things to add before trusting it. Measure an empty graph's replay time and subtract it from both sides, as the judge does. Feed fresh inputs by copying new tensors into the captured input buffers between replays, because the rules forbid caching anything derived from `data`. And run your correctness check on the output of the graph replay, not the eager call: a kernel that reads a stale pointer captured at graph time passes eagerly and fails replayed. [CUDA graph gotchas](/blog/machine-learning/performance-engineering/cuda-graphs-gotchas-and-debugging) catalogs the rest of that failure family.

Run the nine pre-timing test shapes from `TASK.md` locally as well. For `muse_bf16_proj` they include m 7, 33, 130, 512 and 1000, and three use the "outliers" distribution with 1% of channels scaled 30 times. A kernel tuned only on the scored m values often breaks on m=130.

## 5. Nsight inside the container

The judge's profile tells you where the time went on the H200. Nsight on your own GPU tells you why. Two tools, two questions. [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services) answers "what ran, in what order, with what gaps"; [Nsight Compute](/blog/machine-learning/performance-engineering/nsight-compute-kernel-deep-dive) answers "why is this one kernel slow".

```bash
# Timeline: trace inside the CUDA graph so each node shows, not one opaque launch.
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o proj_m256 python bench.py --m 256

# One kernel, full metric set. Note Nsight Compute's defaults: it flushes caches
# before each pass (--cache-control all) and locks clocks to base (--clock-control base).
ncu --set full -k regex:gemm -c 1 -o proj_m256 python bench.py --m 256
```

The cache flush matches the judge's cold-L2 policy; the clock lock matches nothing the judge documents, so treat `ncu` durations as relative and its throughput percentages (DRAM bandwidth vs peak, occupancy, stall reasons) as the real output. `ERR_NVGPUCTRPERM` means the container lacks counter access, which `--cap-add=SYS_ADMIN` fixes.

The loop that ties these pieces together looks like this.

![The iteration loop: edit submission.py, check correctness against reference.py and checks.py, capture as a CUDA graph, time locally with a cold L2, profile with nsys and ncu, spend an okc check on the H200, read the profile, then submit only if the gain clears the interval](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-5.webp)

Most laps close on the local GPU; the judge only sees candidates that already passed locally. The decision diamond is the discipline: a check that shows a 1% gain sits inside the judge's own plus or minus 1.5% interval, so it is noise, not a reason to submit.

## 6. The no-H200 path: developing on two L40S cards

Two L40S cards make a perfectly good lab, as long as you are precise about which questions they can answer. The published numbers:

| | H200 SXM | L40S |
| --- | --- | --- |
| Architecture | Hopper, SM90 | Ada Lovelace, SM89 |
| Memory | 141 GB HBM3e | 48 GB GDDR6 |
| Bandwidth | 4.8 TB/s | 864 GB/s |
| Dense BF16 tensor | 989 TFLOPS (1,979 with sparsity) | 362.05 TFLOPS (733 with sparsity) |
| Shared memory per SM | 228 KB | 100 KB |
| Hopper-only features | WGMMA, TMA, thread block clusters | none of these |

Sources: [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) and [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/) product pages (accessed 2026-10-24), and the compute-capability table in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) for shared memory. NVIDIA lists H200 tensor throughput with sparsity, so the dense figure is half of it.

### What the L40S can answer

**Correctness.** `reference.py` and `checks.py` are plain PyTorch and run anywhere. Every tolerance in `TASK.md`, from 1% relative L2 on the projections to 0.01% on the router and the 0.1% probe rows, can be checked locally against your kernel's output.

**CUDA graph capture.** Capturability is a property of your code, not of the GPU. A host sync, a value-dependent allocation or a `.item()` breaks capture on Ada exactly as it does on Hopper.

**Relative ordering, within limits.** Ada has FP8 tensor-core instructions, so FP8 KV dequantization paths compile and run. Triton, FlashInfer, TileLang and Marlin all target Ada.

### What it cannot answer

**Absolute speed, or even a constant ratio.** The tempting shortcut is "the L40S is about 5.5 times slower, so divide." The judge's own floor formula shows why that fails. The script below re-implements `speed_of_light_ns` from `muse_bf16_proj`'s `reference.py` (floor = max of bytes over bandwidth, FLOPs over dense BF16) and evaluates it for both GPUs:

```python
# The judge's speed_of_light_ns formula for muse_bf16_proj, evaluated on two GPUs.
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}
GPUS = {"H200 SXM": dict(bw=4.8e12, flops=989e12),       # NVIDIA spec, dense BF16
        "L40S":     dict(bw=864e9,  flops=362.05e12)}    # NVIDIA spec, dense BF16

def floor_us(m, layer, gpu):
    n, k = LAYERS[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024                      # this GPU's slice at up to 8 tokens
    wb = 4 if layer == "router" else 2
    bytes_moved = n * k * wb + m * k * 2 + m * n * wb
    flops = 2.0 * m * n * k
    t_mem, t_math = bytes_moved / gpu["bw"], flops / gpu["flops"]
    return 1e6 * max(t_mem, t_math), "math" if t_math > t_mem else "bytes"

for gpu in GPUS:
    print(f"{gpu}: ridge = {GPUS[gpu]['flops'] / GPUS[gpu]['bw']:.0f} FLOP/byte")
print(f"{'shape':<18}{'H200 us':>9}{'bound':>7}{'L40S us':>9}{'bound':>7}{'ratio':>7}")
for m, layer in [(1, "pooler"), (32, "pooler"), (128, "pooler"), (256, "pooler"),
                 (256, "qkv_gate"), (1, "router")]:
    h, hb = floor_us(m, layer, GPUS["H200 SXM"])
    l, lb = floor_us(m, layer, GPUS["L40S"])
    print(f"m={m:<4}{layer:<12}{h:>9.1f}{hb:>7}{l:>9.1f}{lb:>7}{l / h:>6.2f}x")
```

Run with `python3 -I`, it prints:

```console
H200 SXM: ridge = 206 FLOP/byte
L40S: ridge = 419 FLOP/byte
shape               H200 us  bound  L40S us  bound  ratio
m=1   pooler           14.0  bytes     77.7  bytes  5.56x
m=32  pooler           14.1  bytes     78.6  bytes  5.56x
m=128 pooler           14.6  bytes     81.3  bytes  5.56x
m=256 pooler           17.4   math     85.0  bytes  4.89x
m=256 qkv_gate          9.8   math     49.9  bytes  5.11x
m=1   router            1.8  bytes      9.7  bytes  5.56x
```

The H200 column reproduces the floors printed in `TASK.md` (14.0, 14.1, 14.6, 17.4, 9.8 and 1.8 µs), which is a useful sanity check on the formula. Worked through for the m=256 pooler: the weight is 4096 by 8192 BF16, so 67.1 MB, plus 4.2 MB of input and 2.1 MB of output, about 73.4 MB in all. At 4.8 TB/s that is 15.3 µs of traffic, but the 17.2 GFLOP of math takes 17.4 µs at 989 TFLOPS, so the H200 is compute-bound. On the L40S the same 73.4 MB takes 85.0 µs at 864 GB/s while the math needs only 47.5 µs, so it is still waiting on memory.

![Grouped bars of the judge's floor on H200 and L40S for six bf16_proj shapes, with the ratio above each pair: 5.56x for bandwidth-bound shapes, 4.89x and 5.11x where the H200 becomes math-bound](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-3.webp)

The gap is not a constant: at m=256 the H200 crosses into compute while the L40S is still memory-bound. The L40S ridge point is about twice the H200's (419 vs 206 FLOP per byte), so every verify shape looks more bandwidth-bound locally than it is on the judge. A tile configuration that wins on the L40S because it maximizes memory-level parallelism can lose on the H200, where the same shape wants tensor-core throughput. The [roofline post](/blog/machine-learning/gpu-kernel-engineering/roofline-for-decode-on-h200-bytes-not-flops) plots every scored shape on the H200 alone.

**Tuned parameters.** Beyond the ridge, the hardware the tuning targets is different. Hopper's fast GEMM path is WGMMA fed by TMA, with up to 227 KB of shared memory per block; Ada has neither instruction and 99 KB per block. A Triton autotune winner on SM89 picks `num_stages` and block sizes against the smaller shared memory and the older MMA path, and its choice says nothing about SM90. The SM counts differ too (142 on the L40S, 132 on the H200), so split-K and persistent-grid decisions that tile neatly over one card leave a ragged last wave on the other. The rule I use: tune on the judge, with checks, and keep the local box for "is it right" and "did it get faster relative to my previous version on the same card".

**The Hopper-only baselines.** Because attention decode, the decode layer and the prefill layer call FA3, their `baseline.py` will not run on Ada at all. You cannot time against SGLang locally on those three problems. You can still time your version against your previous version, and you can still check correctness against `reference.py`, which does not need FA3.

### The second card

Every kernel problem is single-GPU, so use the second card for parallel experiments: a correctness sweep on GPU 0 while GPU 1 runs an autotune or an `ncu` session, pinned with `CUDA_VISIBLE_DEVICES`. Profiling on the card that is timing corrupts both.

## 7. The serving track cannot run locally

The serving track (opening Nov 5) scores a `git diff` to SGLang `a55ab7d5` on a whole 8×H200 node, launched with `--tp-size 8 --ep-size 8 --quantization modelopt_fp4 --moe-runner-backend marlin` and DFlash speculative decoding, per `GET /serving`.

![Memory blocks drawn to scale: two L40S cards hold 96 GB, one H200 holds 141 GB, an 8xH200 node holds 1,128 GB; the roughly 459 GB of NVFP4 weights overflow the first two and fit in the node](/imgs/blogs/setting-up-the-lab-okc-sglang-image-and-nsight-6.webp)

The arithmetic is short. The model has 816B parameters, and NVFP4 stores 4 bits per weight plus one FP8 scale per 16 weights, so 4.5 bits per parameter: ${816 \times 10^9 \times 4.5 / 8 \approx 459}$ GB, treating every parameter as NVFP4 (the BF16 parts make the true figure larger). Two L40S cards hold 96 GB. The serving model is about 4.8 times that before a single KV page is allocated.

So the serving lab splits the same way as the kernel lab. Locally, test patch logic on a shrunken model: a Muse Spark config with fewer layers, fewer experts and a narrower hidden size, loaded with random weights through SGLang's dummy load format, so the same model code, scheduler and CUDA-graph capture paths execute on hardware you have. The challenge's rules do not let a submitted patch shrink the model (8 routed experts per token, the 2,048-token window and every norm stay), but nothing stops you exercising your control flow on a toy locally. Then spend the three daily `okc serve check` runs, each one short rung at concurrency 16 with a 35-minute deadline, as the real-hardware timer, the same way `okc check` serves the kernel track. [SGLang's internals post](/blog/machine-learning/inference-frameworks/sglang-inference) is the right primer for where a patch can live.

## 8. Two ways the local lab lies

**The warm cache.** A local benchmark that reruns the same 67 MB pooler weight back to back will find part of it resident in L2 and report a time the judge never sees. The scrub buffer in section 4, sized from `L2_cache_size`, is not optional, and it must scale with whatever card you happen to be on.

**The architecture win.** A split-K setting that is best on 142 SMs gets spent as one of your twenty checks and comes back slower on 132 SMs. Nothing was wrong with the kernel; the parameter simply did not transfer. Budget for this: the first two or three checks on any new kernel are tuning checks, not validation checks.

## When to rent the H200, when not to

The decision is about iteration rate. When your ideas are structural (fusing two kernels, removing a host-side gap, restructuring the MoE dispatch), the L40S plus twenty checks a day is enough, because correctness and capture are the slow part. When you are in the last 5%, tuning tiles and stages for WGMMA and TMA, every useful signal comes from SM90, and an hourly H200 rental buys you an unlimited local judge. Pay for it then, not before.

> **On the board**
>
> **Moves:** every kernel problem (`muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer`) and the serving track indirectly, by raising the number of real ideas each day's 10 submits and 20 checks can test.
>
> **Lever:** a local loop that mirrors the judge (pinned image, graph replay, cold L2, paired ratios, the nine test shapes) so checks answer tuning questions on the H200 rather than catching bugs a local run would have found.
>
> **Rule fence:** the judge's timing and harness are off-limits: no patching timing, reading harness frames, touching harness I/O, importing the reference, starting processes or using the network from a submission. A person accepts the rules with `okc login`; an agent never does. Same code resubmitted returns the old result, and a run that games the judge can be withdrawn with `okc withdraw <job>`.

Next, the series moves from the lab to the machine: Hopper's SMs and the features the L40S was missing. Start from [the series introduction](/blog/machine-learning/gpu-kernel-engineering/the-open-kernel-challenge-decoded), [the model's shapes](/blog/machine-learning/gpu-kernel-engineering/muse-spark-open-for-kernel-engineers) and [where a layer's microseconds go](/blog/machine-learning/gpu-kernel-engineering/one-layer-on-one-h200-where-the-microseconds-go).
