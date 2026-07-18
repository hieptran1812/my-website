---
title: "Profiling Compiled Code: Did the Fusion Actually Happen?"
date: "2026-07-14"
publishDate: "2026-07-14"
description: "You called torch.compile and the log said it worked. But is the model faster because Inductor fused your kernels, or did it just recompile thirty times and land back at eager speed? The only honest answer is in the profile. This post teaches you to read a compiled trace, prove the fusion happened, and account for compile time versus runtime."
tags:
  [
    "performance-engineering",
    "gpu-optimization",
    "torch-compile",
    "profiling",
    "pytorch",
    "cuda",
    "cuda-graphs",
    "latency",
    "throughput",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Performance Engineering"
author: "Hiep Tran"
featured: true
readTime: 36
---

Here is a claim you have almost certainly made, or heard someone make, and been unable to defend: "we torch.compiled the model and it's faster now." The compile call did not error. The log printed something about Inductor. The demo notebook felt snappier. Ship it. Except a week later the p99 latency is exactly where it was before the change, and nobody can say why, and the honest answer — the one nobody wants to give in the standup — is that *we never actually checked*. `torch.compile` returns a callable whether or not it made anything faster. It returns a callable when it fuses forty kernels into three. It returns a callable when it hits a graph break on line one and silently runs the rest in eager. It returns a callable when it recompiles on every new batch shape and burns more time compiling than it ever saves at runtime. The function signature cannot tell you which of these happened. Only the profile can.

This post is the close of the `torch.compile` track in the [Profiling & Optimizing AI Services](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) series, and it exists to answer one question that the earlier posts in the track carefully set up but did not close: *how do you SEE, in a real profile, whether the compile helped — and by how much?* The earlier posts told you [what torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) (Dynamo captures a graph, Inductor generates fused kernels), how to [debug graph breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks) that quietly kneecap it, and how [reduce-overhead composes compile with CUDA graphs](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead). This post hands you the verification discipline that turns "it compiled" into "here is the fused kernel, here is the closed idle gap, here is the p99 that dropped 31%, and here is the number of steps after which the compile time paid for itself."

![two stacked trace panels contrasting a long list of small aten operations against a short list of fused triton kernels under a compiled region span](/imgs/blogs/profiling-compiled-code-1.webp)

The figure above is the whole thesis in one image, and it is the first thing you should learn to recognize. On the left is a slice of an eager trace: a long picket-fence of tiny `aten::` operations — `aten::add`, `aten::mul`, `aten::native_layer_norm`, `aten::gelu`, one after another, 1,920 of them per step, each one a separate kernel launch. On the right is the same forward pass after `torch.compile`: the fence collapses into a handful of fatter kernels with names you have never seen in eager mode — `triton_poi_fused_add_gelu_native_layer_norm_5`, `triton_per_fused_native_layer_norm_2` — wrapped inside a span labeled `Torch-Compiled Region`. Those fused names are not cosmetic. Each one is a receipt: the suffix `add_gelu_native_layer_norm` literally lists the operations that Inductor merged into a single kernel, doing them all in one pass over the data instead of five round-trips to HBM. If you can read that name, you can prove the fusion happened without trusting a single log line. That is the skill this post is about.

By the end you will be able to: open a `torch.profiler` trace of a compiled model and read the fusion off the kernel names; use `TORCH_LOGS="output_code"` to see the exact generated Triton and confirm a pointwise chain collapsed into one kernel; measure compile time with `torch._dynamo.utils.compile_times()` and compute the break-even step count where the compile pays for itself; profile steady-state to compare eager and compiled p50/p99 honestly; spot the `reduce-overhead` CUDA-graph replay node in the trace; and — most valuable of all — run the autopsy on a service that "compiled" and got no faster, and name exactly which of four failure modes killed the win. This is the last of the four wastes from the series intro made visible: **redundant work** — the same bytes read and written to HBM again and again because nothing fused — and the tool that proves you killed it.

## The trace changes shape when you compile

Start with what your eyes should register in the first two seconds of looking at a compiled trace, because the shape of the trace changes before any number does. In eager mode, every operation your model calls is dispatched and launched individually, so the GPU kernel row of a [Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) is a dense stripe of hundreds of narrow bars, each a few microseconds wide, most of them elementwise ops — adds, multiplies, activations, the pointwise glue between the big matmuls. A compiled trace of the *same model* looks different in four specific, learnable ways, and each one is a distinct signal you can verify independently.

First, **there are dramatically fewer kernels**, and the ones that remain are fatter. Inductor's whole job is to take chains of pointwise operations — a residual `add`, a `layer_norm`, a `gelu`, a bias-`add` — that eager mode would run as separate kernels, and generate a single Triton kernel that does the entire chain in one pass. Where eager launched 1,920 kernels for a 12-layer encoder step, the compiled version launches roughly 456. That collapse is the single most important thing you are looking for, and it is trivially measurable: the `# of Calls` column in `torch.profiler`'s table, summed, drops by three to five times.

Second, **the kernel names change**. Eager kernels carry `aten::` names — `aten::add`, `aten::addmm`, `aten::native_layer_norm` — because they are ATen library kernels. Inductor-generated kernels carry the prefix `triton_` followed by a category, the word `fused`, the list of operations they merged, and a disambiguating index: `triton_poi_fused_add_mul_0`, `triton_red_fused_native_layer_norm_1`, `triton_per_fused_softmax_3`. Seeing `triton_..._fused_...` in your GPU kernel row is *direct evidence Inductor generated and ran a fused kernel*. Seeing only `aten::` names means either you are looking at an eager region, or the compile produced no fusion (which, as we will see, is a real failure mode).

Third, **a `Torch-Compiled Region` span appears** on the CPU/annotation row. `torch.compile` wraps each compiled subgraph in a `record_function` range so that in the profiler timeline you see a labeled bracket — often rendered as `Torch-Compiled Region: 0/0`, where the numbers identify the frame and the graph index — enclosing all the fused kernels it dispatched. If your model has graph breaks, you will see *multiple* such spans with eager operations sandwiched between them, and that pattern is itself a diagnosis: every gap between compiled regions is a place Dynamo gave up and fell back to eager. One clean span from start to finish is what `fullgraph=True` gets you.

Fourth, **if you compiled with `mode="reduce-overhead"`**, which layers CUDA graphs on top of Inductor, the trace changes again: on the CPU side, the hundreds of individual `cudaLaunchKernel` calls collapse into a single `cudaGraphLaunch`, because the entire recorded sequence replays as one driver operation. The fused kernels still appear on the GPU row — the GPU still does the same work — but the CPU row that used to be a dense stripe of launch bars becomes almost empty, one launch feeding the whole step. That is the launch-overhead fix from the [CUDA-graphs track](/blog/machine-learning/performance-engineering/the-kernel-launch-overhead-problem) showing up in a compiled trace.

Those four signals — fewer kernels, fused names, the compiled-region span, and the optional graph-replay launch — are the vocabulary. The rest of this post is about reading each one precisely and, critically, refusing to believe the win until you have. Let us start with the richest signal, the fused kernel name, because it carries more information than any log line the compiler will ever print.

## Reading the fusion off a kernel name

A fused kernel name is the most honest artifact `torch.compile` produces, because Inductor generates it *from the operations it actually fused*, not from an intention. If you can decode the name, you can read the fusion straight off a profiler row.

![a vertical stack showing three separate aten pointwise operations narrowing through an inductor lowering step into a single fused triton kernel](/imgs/blogs/profiling-compiled-code-2.webp)

The figure walks the collapse. At the top are three separate eager operations on the same tensor: a residual `add` (read two tensors from HBM, write one back), a `gelu` activation (read one, write one), and a `layer_norm` (read one, compute a reduction, write one). In eager mode that is three kernel launches and, worse, three full round-trips of the activation tensor to and from HBM — the same bytes streamed off the chip and back on twice for no reason. The middle band is Inductor's lowering pass: it recognizes that these three ops form a chain where each consumes the previous one's output, so it can compute all three *while the data is still in registers*, touching HBM only to read the original inputs once and write the final output once. The bottom is the result: a single Triton kernel named `triton_poi_fused_add_gelu_native_layer_norm_5`.

Every piece of that name means something:

- **`triton_`** — this is an Inductor-generated Triton kernel, not an ATen library call or a cuBLAS/cuDNN kernel. Its mere presence proves codegen ran.
- **`poi`** — the *category* of the kernel. `poi` = **pointwise** (elementwise, one output element per input element). The other categories you will see: **`red`** = a reduction with a non-trivial reduction dimension (a sum, a mean, a max over an axis); **`per`** = a *persistent* reduction (the whole reduction fits in one block, common for layer-norm over the hidden dim); **`tem`** = a *template* kernel, Inductor's generated matmul or convolution (you only see these with autotuning enabled, since by default matmuls stay on cuBLAS); **`for`** = a `foreach` kernel that batches many small elementwise ops (the fused optimizer step is the classic case).
- **`fused_`** — followed by the list of ATen operations that were merged, in lowering order. `add_gelu_native_layer_norm` is the receipt: those three ops now share one kernel. A name like `triton_poi_fused_add_mul_add_0` tells you two adds and a multiply fused; `triton_poi_fused_0` with no op list is a trivial pointwise copy or cast.
- **`5`** — a disambiguating index. Inductor numbers its kernels in generation order; the number carries no semantics beyond uniqueness.

So a profiler row reading `triton_poi_fused_add_gelu_native_layer_norm_5 · 1.402 ms · 96 calls` tells you, without any further tooling: Inductor generated a pointwise kernel that fuses a residual add, a GELU, and a layer-norm; it ran 96 times (four fused blocks per layer times 12 layers plus the embedding); and it cost 1.4 ms of GPU time total. In eager mode those same three operations across the same 96 sites would have been at least 288 separate kernel launches. You just read a 3× launch reduction and a 2× HBM-traffic reduction off a single kernel name. This is why the name is the proof: it is generated from the fusion decision itself.

There is a crucial subtlety here that separates people who trust the win from people who verify it. **Inductor does not fuse matmuls into the pointwise kernels by default.** A Transformer's GEMMs — the QKV projection, the attention output projection, the two MLP layers — remain calls to cuBLAS (`aten::addmm`, which shows up in the trace as a kernel name like `ampere_fp16_s16816gemm_...`) even after compile, because cuBLAS is usually faster than a generated matmul. What Inductor fuses is the *epilogue and the glue*: the bias-add after the matmul, the activation, the residual, the norm. So a correctly compiled Transformer trace is a mix — big cuBLAS GEMM kernels (unchanged from eager) interleaved with `triton_..._fused_...` kernels (the collapsed pointwise chains). If you see *zero* `triton_` kernels, the pointwise fusion did not happen and something is wrong. If you see the GEMMs became `triton_tem_fused_addmm_...`, you compiled with `max-autotune` and Inductor decided its generated matmul template beat cuBLAS — a separate, later story.

## The four signals of a compiled trace

Before we go to the tools, here is the checklist in one place — the four independent signals, what each one is, what it proves, and where you read it. Treat it as the rubric you run every time someone claims a compile helped.

![a four by three table matching each compiled trace signal to what it is what it proves and the tool that reads it](/imgs/blogs/profiling-compiled-code-3.webp)

| Signal | What it is | What it proves | Where you read it |
|---|---|---|---|
| Fewer kernels | `# of Calls` drops 3–5× | Pointwise chains collapsed | `torch.profiler` table, summed calls |
| Fused names | `triton_..._fused_op_op_...` | Inductor generated & ran fused kernels | GPU kernel row of the Chrome trace |
| Compiled-region span | `Torch-Compiled Region: n/m` | Code ran through Dynamo, not eager | CPU/annotation row; count = 1 + graph breaks |
| Graph-replay launch | one `cudaGraphLaunch` per step | `reduce-overhead` CUDA graph is replaying | CPU row; launch bars collapsed to one |

The power of having four independent signals is that they cross-check each other and localize failures. Fewer kernels but no `triton_` names would be contradictory — that combination cannot happen, so if you think you see it, you are misreading the trace (probably summing across an eager warm-up region). One `Torch-Compiled Region` span means a clean full-graph capture; five spans with eager ops between them means four graph breaks, and *that* explains why the kernel count barely dropped. A `cudaGraphLaunch` node confirms the CUDA-graph layer engaged; its *absence* under `mode="reduce-overhead"` means the graph capture silently bailed (usually a dynamic shape or a disallowed op), and you are getting Inductor fusion but paying full launch cost. Each signal is a different subsystem reporting in, and the profile is where they all report.

Notice what is *not* on this list: `nvidia-smi` GPU-Util, wall-clock feel, and the absence of a compile error. None of those are signals. `nvidia-smi` reads 100% before and after (it always does — see [metrics that actually matter](/blog/machine-learning/performance-engineering/metrics-that-actually-matter)); the demo felt faster because you ran it once warm; and the lack of an error means nothing because `torch.compile` is designed to *never* error on a graph break — it falls back to eager and keeps going. The whole reason this post exists is that the easy signals are the lying ones.

## Verifying fusion happened: the profiler and TORCH_LOGS

Now the practical flow. There are two tools that prove fusion, and you should use both because they answer different questions. `torch.profiler` tells you *which* fused kernels ran and how much time they took; `TORCH_LOGS="output_code"` shows you the *generated source* so you can confirm the fusion is what you think it is.

Start with the profiler. The one non-obvious rule when profiling compiled code is that **you must skip the compiling steps** — the first call (or first few, if shapes vary) pays the entire compile cost, and if you profile it you are measuring the compiler, not the model. Use the profiler's `schedule` to warm up past the compile, then record steady state:

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

model = build_encoder().cuda().eval()                       # 12-layer, hidden 768
compiled = torch.compile(model, fullgraph=True)             # default mode
x = torch.randn(16, 256, 768, device="cuda", dtype=torch.float16)

# wait=2 skips setup, warmup=3 pays the compile + captures nothing,
# active=5 records steady-state compiled steps. repeat=1.
sched = schedule(wait=2, warmup=3, active=5, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    on_trace_ready=tensorboard_trace_handler("./log/compiled"),
    record_shapes=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):          # 2 wait + 3 warmup + 5 active
            compiled(x)
            torch.cuda.synchronize()
            prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
```

The `warmup=3` is doing real work here: the first `active`-phase step is guaranteed to be a fully-compiled, cache-hot replay, not a compile. Here is the kind of table this prints for the compiled encoder, GPU rows sorted by total CUDA time — read it top to bottom as a fusion receipt:

```console
-------------------------------------------------------  ------------  ------------  ------------
Name                                                      Self CUDA    CUDA total   # of Calls
-------------------------------------------------------  ------------  ------------  ------------
ampere_fp16_s16816gemm_fp16_256x128_...  (cuBLAS)            3.998ms       3.998ms           72
aten::_scaled_dot_product_flash_attention                    2.101ms       2.101ms           24
triton_poi_fused_add_gelu_view_5                             1.402ms       1.402ms           96
triton_per_fused_native_layer_norm_2                         0.907ms       0.907ms           48
triton_poi_fused_add_native_layer_norm_1                     0.486ms       0.486ms           24
triton_poi_fused_clone_3                                     0.221ms       0.221ms           24
-------------------------------------------------------  ------------  ------------  ------------
Self CUDA time total: 9.115ms
```

Read it: the GEMMs are still cuBLAS (`ampere_fp16_s16816gemm`), attention is the fused flash kernel, and *everything else collapsed into four `triton_..._fused_...` kernels*. Sum the `# of Calls`: 72 + 24 + 96 + 48 + 24 + 24 = 288 kernels in the hot path, versus 1,920 in the eager trace of the same model. The fusion happened, it is on the row, and you can hand this table to a skeptic.

But a profiler table tells you *that* ops fused, not *how*. For that, read the generated code. Set the environment variable and Inductor prints the Triton source it wrote:

```bash
TORCH_LOGS="output_code" python serve.py 2> inductor_output.py
```

Somewhere in that dump you will find the kernel the profiler named `triton_poi_fused_add_gelu_view_5`, and it is real, readable Triton:

```python
@triton.jit
def triton_poi_fused_add_gelu_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)     # activations, read ONCE from HBM
    tmp1 = tl.load(in_ptr1 + xindex, xmask)     # residual, read ONCE from HBM
    tmp2 = tmp0 + tmp1                            # <- the residual add, fused
    tmp3 = 0.5 * tmp2
    tmp4 = tmp2 * 0.7071067811865476
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tmp5 + 1.0
    tmp7 = tmp3 * tmp6                            # <- the GELU, fused, no HBM trip
    tl.store(out_ptr0 + xindex, tmp7, xmask)     # write ONCE to HBM
```

This is the proof at its most concrete. Two `tl.load`s at the top, one `tl.store` at the bottom, and everything in between — the add and the full GELU expansion — happening in registers. In eager mode the add would have read two tensors and written one, then GELU would have read that one back and written it again: four HBM transactions of the activation tensor. Here there are three total, and the intermediate never leaves the chip. *That* is what "fusion saved HBM bandwidth" means, and you are looking at the exact instructions that make it true.

If you want to steer or inspect the fusion decisions themselves, `torch._inductor.config` is the knob box. A few that matter for verification:

```python
import torch._inductor.config as ind

ind.trace.enabled = True          # dump the full compile trace (kernels, scheduling) to a dir
ind.debug = True                  # verbose lowering decisions
ind.epilogue_fusion = True        # fuse pointwise epilogues into matmul templates (default on)
ind.max_autotune = False          # keep GEMMs on cuBLAS; True benchmarks Triton templates
```

Setting `torch._inductor.config.trace.enabled = True` (or `TORCH_COMPILE_DEBUG=1`) writes a directory per compiled graph containing the generated code, the scheduling decisions, and an `ir_post_fusion.txt` that lists every fusion group Inductor formed — the ground truth if a profiler row ever surprises you.

#### Worked example: proving a norm-heavy block fused

A vision-adjacent service runs a stack of `LayerNorm → Linear → GELU → residual add` blocks and the team compiled it, expecting the three pointwise ops around each Linear to fuse. Did they? The eager profile shows, per block: `aten::native_layer_norm` (2 kernels — a reduction and an affine), `aten::addmm` (1), `aten::gelu` (1), `aten::add` (1) = 5 kernels, and the layer-norm plus gelu plus add together read and wrote the hidden tensor five times. After `torch.compile`, the profile shows per block: one `ampere_..._gemm` (the addmm, still cuBLAS), one `triton_per_fused_native_layer_norm_0` (the norm reduction), and one `triton_poi_fused_add_gelu_1` (the gelu and the residual, fused). Three kernels, and `TORCH_LOGS="output_code"` confirms `triton_poi_fused_add_gelu_1` does both the activation and the add in registers. The receipt: 5 kernels → 3, and the activation tensor's HBM traffic dropped from 5 round-trips to 3. Measured on an A100, the block's GPU time fell from 0.61 ms to 0.44 ms — a 28% local win — entirely from the two fusions, with the GEMM untouched. The verification took ninety seconds and turned "we think it fused" into a number.

## The compile-time vs runtime tradeoff

Here is where "we compiled it" gets a bill attached. Every second the model runs faster at inference was bought with seconds of compile time paid up front, and whether that trade is worth it depends entirely on how long the job lives. This is the accounting nobody does, and it is why some teams torch.compile a thirty-second batch job and make it slower.

![a left to right timeline of a first cold call spending time in dynamo tracing inductor codegen and triton autotune then many fast steady calls](/imgs/blogs/profiling-compiled-code-4.webp)

The figure shows where the first call's wall time actually goes. The very first invocation of a compiled model — the *cold* call — does not run the model at the speed you profiled. It runs the compiler. In order: **Dynamo tracing** (~1–3 s) walks your Python bytecode and builds an FX graph, inserting guards; **AOTAutograd** (~1–2 s) traces through to get the functionalized graph; **Inductor lowering and scheduling** (~3–6 s) turns that graph into a schedule of fused kernels and decides what fuses with what; and **Triton compilation** (~8–15 s) is the big one — every generated kernel is compiled to PTX and then to a cubin by the Triton/NVCC toolchain, and this dominates. If you asked for `max-autotune`, add **autotuning** on top, which *benchmarks* multiple kernel implementations on real data before picking one, and can push the cold call to minutes. Only after all of that does the *first fast step* run. Every step after is the warm steady-state number you want.

You do not have to guess these numbers — PyTorch instruments them. `torch._dynamo.utils.compile_times()` returns a table of cumulative time per compile phase:

```python
import torch._dynamo.utils as dynamo_utils

compiled = torch.compile(model)
x = torch.randn(16, 256, 768, device="cuda", dtype=torch.float16)
with torch.no_grad():
    compiled(x)                              # the cold call — pays the whole compile
    torch.cuda.synchronize()

print(dynamo_utils.compile_times(repr="str", aggregate=True))
```

```console
TorchDynamo compilation metrics:
Function                                     Runtimes (s)
-------------------------------------------  -------------
_compile.compile_inner                       21.8402
   OutputGraph.call_user_compiler            18.3115
      create_aot_dispatcher_function          3.0871
      compile_fx_inner                       14.9980
         GraphLowering.compile_to_module     13.2114
            async_compile.wait                8.7663      <- Triton kernels compiling in parallel
      Scheduler.__init__                      0.9420
   convert_frame                              2.1774       <- Dynamo bytecode tracing
```

Read it top-down: 21.8 s total, of which `async_compile.wait` (8.8 s) is the Triton toolchain compiling kernels, `GraphLowering.compile_to_module` (13.2 s) is all of Inductor, and `convert_frame` (2.2 s) is Dynamo's trace. That 21.8 s is the price. Now the question is whether the runtime savings ever pay it back, and that is a clean piece of arithmetic.

### The break-even law

Let a job run `$N$` inference steps. In eager, each step costs `$t_e$` and the job costs `$T_\text{eager} = N \cdot t_e$`. Compiled, you pay a one-time compile cost `$C$` up front and then each step costs `$t_c$`, where `$t_c \lt t_e$` because the fusion helped. So the compiled job costs:

$$T_\text{comp} = C + N \cdot t_c$$

Compiled is the better choice only when `$T_\text{comp} \lt T_\text{eager}$`, that is when `$C + N t_c \lt N t_e$`. Solving for `$N$` gives the **break-even step count**:

$$N^{*} = \frac{C}{t_e - t_c}$$

Below `$N^{*}$` steps, compiling *loses* — you spent more time compiling than you saved running. Above it, you win, and the win grows without bound as `$N \to \infty$` because the compile cost amortizes to nothing. Plug in the encoder: cold compile `$C = 21.8$` s, eager step `$t_e = 11.4$` ms, compiled step `$t_c = 7.9$` ms. Then:

$$N^{*} = \frac{21.8}{0.0114 - 0.0079} = \frac{21.8}{0.0035} \approx 6{,}230 \text{ steps}$$

So this compile pays for itself after about 6,200 inference steps. A long-running serving process that handles millions of requests? The compile cost is a rounding error and you should absolutely compile. A batch job that scores 2,000 items and exits? You just made it *slower* — the compile cost `$C$` exceeds the `$2000 \times 3.5\text{ ms} = 7$` s you saved. A CI test that imports the model, runs it twice, and asserts on the output? Compiling is pure loss.

This is exactly the regime where `torch.compile` gets a bad reputation: someone benchmarks it on a short job, sees it run slower wall-to-wall, and concludes it does not work. It works; they measured a job shorter than `$N^{*}$`. The fix is not to abandon compile — it is to amortize `$C$`.

### Amortizing compile with the persistent cache

The break-even law assumes you pay `$C$` in full on every run. You do not have to. Inductor writes its compiled Triton kernels to a persistent on-disk cache, and PyTorch caches the FX graph too, so a *second* process compiling the *same* graph reuses the artifacts and pays a fraction of `$C$`.

```bash
# Where Inductor caches compiled kernels (default: /tmp/torchinductor_<user>).
# Point it at persistent, warm storage so it survives restarts and is shared.
export TORCHINDUCTOR_CACHE_DIR=/mnt/fast/torchinductor_cache

# The FX graph cache (on by default in recent PyTorch) skips re-lowering
# an already-seen graph. Explicit env in case a base image disabled it:
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
```

With a warm cache, the encoder's cold compile drops from 21.8 s to about 1.8 s — Dynamo still has to trace the bytecode and check guards, but Inductor finds the kernels already compiled on disk and skips the 13 s of codegen and the 8.8 s of Triton compilation. That collapses `$N^{*}$` from ~6,200 to about `$1.8 / 0.0035 \approx 515$` steps. The lesson for production: bake the cache into your container image, or mount it on shared warm storage, and your break-even point becomes almost irrelevant. The lesson for benchmarking: a cold-cache first run and a warm-cache first run are *different measurements*, and you must say which one you report.

#### Worked example: the batch job that compile made slower

A team wraps a nightly embedding job — encode 5,000 documents, write vectors, exit — in `torch.compile` because "compile makes it faster," and the job's wall time goes *up* from 71 s to 88 s. The autopsy is pure arithmetic. Eager: `$5000 \times 11.4\text{ ms} = 57$` s of compute plus 14 s of I/O = 71 s. Compiled cold: 21.8 s compile + `$5000 \times 7.9\text{ ms} = 39.5$` s + 14 s I/O = 75.3 s — already worse than eager once you count compile, and the extra was recompiles on ragged final-batch shapes pushing it to 88 s. The break-even was `$N^{*} \approx 6{,}200$` steps and the job runs 5,000; it was always going to lose cold. Two fixes recover it: warm the Inductor cache (compile drops to 1.8 s, and `$5000 \gt 515$`, so compiled now wins at 1.8 + 39.5 + 14 = 55.3 s), and pad the last batch to a fixed shape to kill the recompiles. Same code, same compile call — the difference is entirely in *whether the compile cost was amortized*, which is a question only the compile-time accounting answers.

## Confirming the runtime win honestly

You have proven the fusion happened and you have accounted for the compile cost. Now prove the *runtime* is actually better, and do it honestly — which mostly means measuring steady state and refusing to be fooled by the confounds.

The cardinal rule: **never time the compiling steps.** A naive `time.time()` around the first call includes 20+ seconds of compile and makes compiled look catastrophically slow; a naive loop that times all steps averages the compile over them and makes the number meaningless. Warm up past the compile, synchronize, then time the steady state with CUDA events:

```python
import torch, numpy as np

compiled = torch.compile(model)
x = torch.randn(16, 256, 768, device="cuda", dtype=torch.float16)

# 1) Warm up PAST the compile. First calls trace, lower, codegen, autotune.
for _ in range(10):
    with torch.no_grad():
        compiled(x)
torch.cuda.synchronize()

# 2) Time steady state with CUDA events, one event pair per step, then
#    take percentiles — a mean hides the p99 tail you actually care about.
starter = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
ender   = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
with torch.no_grad():
    for i in range(200):
        starter[i].record()
        compiled(x)
        ender[i].record()
torch.cuda.synchronize()
lat = np.array([s.elapsed_time(e) for s, e in zip(starter, ender)])
print(f"p50 ={np.percentile(lat,50):.2f} ms   p99 ={np.percentile(lat,99):.2f} ms")
```

Run the same harness on the eager model and you have a fair, apples-to-apples comparison. Here is the full before→after on named hardware, measured exactly this way — steady state, CUDA events, locked clocks, warm cache — for the 12-layer encoder:

| Metric (A100 80GB, 12-layer encoder, batch 16, seq 256, fp16) | Eager | torch.compile (default) | reduce-overhead |
|---|---|---|---|
| Kernels / step | 1,920 | 456 | 1 replay |
| Fused `triton_` kernels | 0 | 138 | 138 (in graph) |
| Compile time (cold) | 0 s | 21.8 s | 30.6 s |
| Compile time (warm cache) | 0 s | 1.8 s | 3.1 s |
| Host launch / step | 13.1 ms | 3.4 ms | 0.2 ms |
| p50 latency | 11.4 ms | 7.9 ms | 7.1 ms |
| p99 latency | 12.8 ms | 8.6 ms | 7.4 ms |
| Throughput | 1,404 samples/s | 2,025 samples/s | 2,253 samples/s |
| SM occupancy | 41% | 47% | 47% |
| Peak memory | 6.2 GB | 6.4 GB | 7.1 GB |

Every number is one the profiler you just ran produces. Note the honest details. The `default` compile drops p50 from 11.4 to 7.9 ms — a real 31% win, mostly from the fused pointwise kernels cutting HBM traffic and the reduced launch overhead (host launch fell from 13.1 to 3.4 ms). `reduce-overhead` shaves another 0.8 ms by collapsing the remaining launches into one graph replay, at the cost of 0.7 GB more memory (the CUDA-graph static pools) and a longer, more fragile compile. Occupancy rose modestly (41% → 47%) because the fused kernels are fatter and keep the SMs busier. And p99 tracked p50 down — which is the *other* thing you must verify, because a compile that lowers p50 while *raising* p99 has hidden a stall.

That last point deserves its own check, because it is where compiled services quietly rot. A compiled model whose input shapes vary — different sequence lengths per request — can pass a p50 benchmark on fixed shapes and then, in production, **recompile on every new shape**, and each recompile is a multi-second stall that shows up only in the tail. The steady-state benchmark on one fixed shape will never see it. You have to check for recompiles explicitly:

```python
import torch._dynamo as dynamo
# after running your real, shape-varied traffic through the compiled model:
print(dynamo.utils.counters["stats"])
# {'unique_graphs': 1, 'recompiles': 0, ...}   <- what you WANT
# {'unique_graphs': 47, 'recompiles': 46, ...}  <- a recompilation storm
```

or run with `TORCH_LOGS="recompiles"` and watch for a line every time a guard fails. A clean steady state has `recompiles: 0`; a tail problem shows up as dozens of unique graphs. This is the bridge to [debugging graph breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks) — a recompilation storm is the same disease seen from the profiler's side. The rule: **profile with the shape distribution you serve, not one fixed shape**, or you are measuring a service you do not run.

## When "compiled" isn't faster: an autopsy

The most valuable skill in this post is the one you reach for when the numbers refuse to move. Someone compiled the model, the p50 did not budge, and everyone is confused because "torch.compile is supposed to be faster." It is faster when it fuses and stays fused. When it does not, one of four things went wrong, and the profile tells you which in about five minutes. Run them in order.

![a decision tree from a compiled but not faster symptom branching through fusion breaks recompiles and compile cost to four distinct causes](/imgs/blogs/profiling-compiled-code-5.webp)

The tree walks the four failure modes as a sequence of yes/no questions against the profile:

- **Did anything fuse?** Open the profiler table and look for `triton_..._fused_...` names. If there are *none* — all `aten::` — then Inductor generated no fused kernels, and the usual cause is that your model is already all big GEMMs with nothing pointwise to fuse (a pure matmul stack has nothing for Inductor to collapse; the GEMMs were already cuBLAS in eager). Verdict: **compile had nothing to do**; the model was never launch- or bandwidth-bound. Accept it and move on — this is a *correct* null result, not a bug.
- **Are there graph breaks?** Count the `Torch-Compiled Region` spans. More than one means Dynamo fell back to eager between them, so the fusion only applies to fragments and the eager glue between fragments still runs op-by-op. Run `torch._dynamo.explain(model)(x)` to get the break reasons and `TORCH_LOGS="graph_breaks"` to locate them. Verdict: **fusion is real but partial**; fix the breaks (a data-dependent `if`, a `.item()`, an unsupported op) and the win returns. This is the single most common cause of "compiled but no faster."
- **Is it recompiling?** Check `dynamo.utils.counters["stats"]` for `recompiles`. A high count means guards are failing — usually varying shapes — so the model spends its time re-invoking the compiler instead of running compiled kernels, and steady state is never reached. Verdict: **the win is being eaten by recompiles**; bucket the shapes or pass `dynamic=True`. In the trace this looks like periodic multi-second CPU-bound stalls with no GPU work.
- **Is compile time dominating?** If fusion happened, there are no breaks, and no recompiles, but the *job* is still slower wall-to-wall, then you are below the break-even `$N^{*}$` — the job is too short to amortize the compile. Verdict: **amortize or don't compile**; warm the cache, or accept that this workload should run eager.

That ordering is deliberate: it goes from "the compiler did nothing" (cheapest to rule out, and a legitimate outcome) through the two runtime failures (breaks, recompiles) to the accounting failure (compile-time dominance). Each question is answered by one profiler artifact, and each verdict is a different fix. Nobody should ever again say "torch.compile didn't help" without naming which of these four it was.

#### Worked example: the service that compiled and stayed at 30% util

A batch-1 chat-serving handler on an L4 reads 30% GPU util and 118 ms p50. The team compiles it; p50 moves to 116 ms — nothing. Run the tree. **Did anything fuse?** Yes, `triton_poi_fused_*` names are present. **Graph breaks?** The trace shows *seven* `Torch-Compiled Region` spans per step. `torch._dynamo.explain` reports the breaks: a `if input_ids.shape[1] > self.max_len:` guard and a `.tolist()` call in the sampling loop, each forcing a fallback. So fusion happened but only inside seven fragments, and the eager tokenizer glue and sampling between them — hundreds of tiny launches — still dominate the batch-1 step, which was launch-bound to begin with. The fix is not "compile harder"; it is to hoist the data-dependent Python out of the compiled region (do the length check before the call, keep sampling on tensors) so Dynamo captures one graph, *then* add `mode="reduce-overhead"` so the now-unbroken graph replays as one launch. After: one `Torch-Compiled Region` span, one `cudaGraphLaunch`, util 30% → 71%, p50 118 → 74 ms. The compile was never the problem; the graph breaks were, and only the span count in the profile revealed it. This is the same [30%-util war story](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) the series opens with, seen through the compiled-trace lens.

## max-autotune: paying more compile for more runtime

There is a compile mode that deliberately buys more runtime with a *lot* more compile time, and profiling is the only way to know whether the trade paid off. `mode="max-autotune"` tells Inductor to stop trusting cuBLAS for the GEMMs and instead *benchmark* several implementations — multiple Triton matmul templates at different tile sizes, plus cuBLAS itself — on your actual shapes, and pick the fastest. It also enables coordinate-descent tuning of kernel parameters and turns on CUDA graphs. The cost is compile time measured in minutes, not seconds; the payoff, when it comes, is GEMMs that beat the library default on your specific shapes.

You watch the autotuning happen with `TORCH_LOGS="+inductor"`, which prints an `AUTOTUNE` block per benchmarked operation:

```console
AUTOTUNE addmm(4096x3072, 3072x768)
  triton_mm_12 0.0389 ms 100.0%
  triton_mm_9  0.0421 ms  92.4%
  bias_addmm   0.0498 ms  78.1%   <- the cuBLAS baseline
  triton_mm_16 0.0511 ms  76.1%
SingleProcess AUTOTUNE benchmarking takes 6.84 seconds for addmm
```

Read it: Inductor benchmarked four candidates for this one `addmm` and the winning Triton template (`triton_mm_12`) beat cuBLAS (`bias_addmm`) by 22% *on this shape*. Multiply that 6.8-second benchmarking cost across every unique GEMM shape in the model and you see why max-autotune compiles for minutes. In the resulting trace, the winning GEMMs show up as `triton_tem_fused_addmm_...` (template kernels) instead of the `ampere_..._gemm` cuBLAS names — a fifth signal, specific to autotuning, that the matmuls themselves got replaced.

The honest question is whether that 22%-per-GEMM benchmark-time investment shows up at runtime, and it often does *not* the way you hope. On an A100 with standard Transformer shapes, cuBLAS is extremely well tuned and max-autotune frequently picks cuBLAS anyway (the `AUTOTUNE` block shows the baseline winning), so you pay minutes of compile for a sub-1% runtime change. It pays off on *unusual* shapes cuBLAS was not tuned for, on cards where the library is weaker, and on models dominated by many GEMMs of the same odd shape. The decision procedure is the same discipline as the whole post: measure the compiled p50 with and without `max-autotune`, measure the extra compile time, and compute whether the runtime delta clears the break-even `$N^{*}$` given the much larger `$C$`. If `max-autotune` adds 180 s of compile to save 0.2 ms per step, `$N^{*} = 180 / 0.0002 = 900{,}000$` steps — worth it for a model serving billions of requests, absurd for anything else.

## The verification loop, end to end

Pull the pieces into one repeatable procedure. Verifying a compile is not a single measurement; it is a small loop that runs the eager and compiled models through the *same* profiler and diffs the artifacts. Automate it and you never again have to argue about whether a compile helped.

![a dataflow where eager profiling and compiled profiling both feed a single kernel and bandwidth diff that emits a verified verdict](/imgs/blogs/profiling-compiled-code-6.webp)

The two profiling runs — eager and compiled, same input, same schedule — each emit a `key_averages()` table. Both feed into a single diff step that computes three quantities: the kernel-count ratio (should drop 3–5×), the presence of `triton_..._fused_...` names (should be nonzero), and the total-CUDA-time and HBM-traffic delta (should drop). The diff emits a verdict — *verified win*, *no fusion*, *partial (breaks)*, or *compile-dominated* — and that verdict is what you record in the PR, not "feels faster." Here is the diff step reduced to the essential asserts you would put in a regression test:

```python
def verify_compile(model, x, min_kernel_reduction=2.0):
    def kernel_stats(m):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            with torch.no_grad():
                m(x); torch.cuda.synchronize()
        rows = p.key_averages()
        n_kernels = sum(r.count for r in rows if r.device_type.name == "CUDA")
        n_fused   = sum(1 for r in rows if "triton_" in r.key and "fused" in r.key)
        cuda_us   = sum(r.self_device_time_total for r in rows)
        return n_kernels, n_fused, cuda_us

    # warm both past compile / cache before measuring
    for _ in range(10):
        with torch.no_grad(): model(x)
    torch.cuda.synchronize()

    ke, fe, te = kernel_stats(model)                 # eager reference
    compiled = torch.compile(model, fullgraph=True)
    for _ in range(10):
        with torch.no_grad(): compiled(x)            # pay the compile once
    torch.cuda.synchronize()
    kc, fc, tc = kernel_stats(compiled)

    assert fc > 0,                       f"NO FUSION: 0 triton_fused kernels"
    assert ke / kc >= min_kernel_reduction, f"WEAK FUSION: {ke}->{kc} kernels"
    assert tc < te,                      f"SLOWER: cuda {te:.0f}->{tc:.0f} us"
    return {"kernels": (ke, kc), "fused": fc, "cuda_us": (te, tc)}
```

Wire that into CI against a representative input and a compile that silently stops fusing — because someone added a `.item()` in a hot path — fails the build instead of shipping a dead optimization. The verification loop is the whole series' *profile → hypothesize → fix → measure* discipline applied to one specific tool, and it turns the compile from an act of faith into a checked invariant.

## When to reach for compiled-code profiling (and when not)

Compiling and then profiling the result is a cost, and like every fix in this series it is not always worth paying. The decision is a function of two things: how long the job lives (does the compile amortize?) and how stable its shapes are (does it stay compiled?).

![a three by three grid rating long lived short and shape varying jobs on compile cost runtime win and verdict](/imgs/blogs/profiling-compiled-code-7.webp)

- **Long-lived, fixed-shape service** (a batched inference server, a training loop of millions of steps): compile, verify the fusion, and use `reduce-overhead` if you are launch-bound. The compile cost amortizes to nothing and the fused kernels are pure win. This is the case `torch.compile` was built for, and skipping it leaves 20–40% on the table.
- **Short-lived job** (a nightly batch of a few thousand items, a CLI tool, a CI test): compile *only* with a warm persistent cache, or don't compile at all. Below the break-even `$N^{*}$`, the compile cost dominates and eager wins wall-to-wall. Profile the *whole job*, not the steady state, because the compile is part of this job's cost.
- **Shape-varying service** (variable sequence lengths, dynamic batch sizes): do not compile naively — you will get a recompilation storm that the fixed-shape benchmark hides. Bucket the shapes, pass `dynamic=True`, and verify `recompiles: 0` under the real shape distribution before you trust any p50. If the shapes are truly unbounded, compile may not be the right tool at all.

And the meta-rule that this whole post serves: **do not claim a compile helped until you have read the fused kernel in the profile and the p99 in the steady-state benchmark.** "It compiled without error" is not evidence. "The demo felt faster" is not evidence. A `triton_..._fused_...` row, a kernel-count drop, and a p99 that moved the right way — measured on the shapes you serve — are evidence. Everything upstream of that is hope.

## Case studies: real fusion and compile numbers

A few results from the primary sources, framed honestly and with the caveat that exact numbers depend on hardware, shapes, and PyTorch version — treat them as order-of-magnitude, verifiable-on-your-own-hardware claims.

**Inductor pointwise fusion on elementwise-heavy models.** The PyTorch team's `torch.compile` tutorials and the Inductor design documents report that the largest speedups come precisely where this post focuses: models with long chains of pointwise operations between GEMMs (normalization, activations, residual adds, dropout in training), where fusion collapses many small memory-bound kernels into few. The reported inference speedups on such models cluster in the 1.3–2.1× range on A100-class hardware in the default mode, and the mechanism is exactly the HBM-traffic reduction you read off the `triton_..._fused_...` kernel earlier. Models that are *already* a stack of large GEMMs (nothing to fuse) see little — which the tree above classifies correctly as "compile had nothing to do."

**reduce-overhead / CUDA graphs on launch-bound inference.** PyTorch's documentation on `mode="reduce-overhead"` and the CUDA Graphs integration reports that small-batch, launch-bound inference — the batch-1 and batch-2 regime where per-kernel launch overhead dominates compute — is where CUDA graphs on top of Inductor pay off most, with the host-side launch cost collapsing to a single replay. The published guidance is consistent with the encoder table above: the biggest wins are at small batch where the CPU cannot keep the GPU fed, and the win shrinks toward zero at large batch where the GPU is compute-bound anyway. The tradeoff the docs are careful about — extra memory for the static graph pools and fragility under dynamic shapes — matches the +0.7 GB and the capture-bail failure mode described here.

**max-autotune on non-standard GEMM shapes.** Inductor's autotuning documentation and community benchmarks report that `max-autotune` most reliably beats cuBLAS on shapes the library was not specifically tuned for and on GPUs where the vendor library is less mature, with per-GEMM wins in the 10–30% range when a Triton template does win — but frequently selecting cuBLAS anyway on standard Transformer shapes on well-supported cards, in which case you paid the (large) autotuning compile time for a negligible runtime change. This is the "measure whether it paid off" case, and it is why the `AUTOTUNE` log showing the *baseline winning* is a legitimate and common outcome, not a failure.

**The graph-break tax, quantified.** The PyTorch profiling and Dynamo documentation, and the [reading-a-chrome-trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) methodology, make the same point this post's autopsy does: a single unnecessary graph break in a hot path can erase most of a compile's benefit by forcing the surrounding glue back to eager, and the fix is diagnostic (`torch._dynamo.explain`, `TORCH_LOGS="graph_breaks"`) rather than heuristic. The consistent finding is that the *count* of `Torch-Compiled Region` spans in the trace is the fastest proxy for how much of the model actually compiled.

## Key takeaways

- **A compiled trace has four learnable signals: fewer kernels, `triton_..._fused_...` names, a `Torch-Compiled Region` span, and (with `reduce-overhead`) a single `cudaGraphLaunch`.** Learn to read all four; each is a different subsystem reporting in.
- **The fused kernel name is the proof.** `triton_poi_fused_add_gelu_native_layer_norm_5` literally lists the ops Inductor merged. `poi`/`red`/`per`/`tem` tell you the kernel category; the suffix is the fusion receipt. Read it off the profiler row, no log line required.
- **Inductor fuses the pointwise glue, not the GEMMs** (unless `max-autotune` wins with a `triton_tem_` template). A correct Transformer trace is cuBLAS GEMMs interleaved with `triton_..._fused_...` kernels. Zero `triton_` kernels means no fusion happened.
- **Confirm the source, not just the table.** `TORCH_LOGS="output_code"` prints the generated Triton; two `tl.load`s and one `tl.store` around a chain of math is fusion you can see instruction by instruction.
- **Compile time is a real cost with a break-even.** `$N^{*} = C / (t_e - t_c)$` steps. Below it, compiling makes the job slower. Measure `$C$` with `torch._dynamo.utils.compile_times()`; amortize it with `TORCHINDUCTOR_CACHE_DIR` and the FX graph cache.
- **Profile steady state, never the compiling call.** Warm up past the compile, time with CUDA events, report p50 *and* p99, and check `dynamo.utils.counters["recompiles"] == 0` under the shapes you actually serve.
- **Run the four-question autopsy when the numbers don't move:** did anything fuse? graph breaks? recompiles? compile-time dominating? Each has a distinct profiler artifact and a distinct fix — never say "compile didn't help" without naming which one.
- **max-autotune buys runtime with minutes of compile; verify it paid off.** Watch the `AUTOTUNE` log; the cuBLAS baseline winning is a legitimate outcome, and the huge `$C$` only clears break-even on very long-lived jobs.

## Further reading

- [Why your AI service wastes CPU and GPU](/blog/machine-learning/performance-engineering/why-your-ai-service-wastes-cpu-and-gpu) — the series intro and the four wastes; compiled-code profiling is how you prove you killed the *redundant work* waste.
- [What torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does) — Dynamo capture, guards, and Inductor codegen; the mechanism this post verifies from the outside.
- [Debugging graph breaks](/blog/machine-learning/performance-engineering/debugging-graph-breaks) — `torch._dynamo.explain`, `TORCH_LOGS="graph_breaks"`, and recompilation storms; the disease behind "compiled but partial."
- [Compile plus CUDA graphs: reduce-overhead](/blog/machine-learning/performance-engineering/compile-plus-cuda-graphs-reduce-overhead) — how `mode="reduce-overhead"` composes Inductor with CUDA graphs, and the graph-replay node you see in the trace.
- [Profiling PyTorch with torch.profiler](/blog/machine-learning/performance-engineering/profiling-pytorch-with-torch-profiler) — the schedule, `key_averages()`, and Chrome-trace export used throughout this post.
- [Reading a Chrome trace](/blog/machine-learning/performance-engineering/reading-a-chrome-trace) — the timeline lanes and how to spot the compiled-region spans and launch collapse visually.
- [The performance engineering playbook](/blog/machine-learning/performance-engineering/the-performance-engineering-playbook) — the capstone decision tree tying every tool and fix together.
- [Kernel fusion, CUDA graphs, and torch.compile in serving](/blog/machine-learning/model-serving/kernel-fusion-cuda-graphs-torch-compile) — how these same techniques land inside a production serving stack.
- PyTorch documentation: the `torch.compile` tutorial, the Inductor / TorchDynamo deep-dive docs, the `torch.profiler` recipe, and the CUDA Graphs integration notes — the primary sources for the APIs, flags, and reported speedups cited above.
