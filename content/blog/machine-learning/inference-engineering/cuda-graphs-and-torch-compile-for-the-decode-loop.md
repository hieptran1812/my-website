---
title: "CUDA graphs and torch.compile for the decode loop: fewer launches, not fewer bytes"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Track F closes with the other decode-loop win: the loop is host-bound, so you fix it by launching fewer kernels, not by moving fewer bytes. Capture the step into a CUDA graph, bucket it by batch size, compile the fusible parts, and plumb it all into nanoserve's continuous-batching loop."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "cuda-graphs",
    "torch-compile",
    "pytorch",
    "cuda",
    "gpu",
    "latency",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 45
---

Track F has spent four posts making your decode step move fewer bytes. Weight-only quantization shrank the 16 GB of Llama-3.1-8B weights that a decode step drags across HBM every token; FP8 halved the KV cache; the [dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) unpacked 4-bit weights in registers so the smaller footprint actually turned into a smaller memory bill. Every one of those wins attacked the same wall: decode is memory-bound, so read less memory. This post closes the track by attacking a *different* wall, and it is a wall that quantization cannot touch no matter how many bits you shave.

Here is the symptom. You quantized Llama-3.2-1B to INT4, you serve it on an H100, and your `nsys` trace is a picket fence — kernels an eighth of a millimeter wide with grey gaps between every one of them. The GPU is idle more than it is busy. You made the weights four times smaller and the tokens-per-second barely moved, because the bottleneck was never the bytes. It was the **launches**. The decode loop issues on the order of 330 tiny kernels per token, and on a small model or a fast GPU the CPU cannot push those launches out fast enough to keep the GPU fed. The tensor cores you paid for spend the step waiting for the host to say "go" 330 times.

![A comparison matrix showing quantization cutting bytes and CUDA graphs cutting launches as two different wins that compose](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-1.webp)

The figure above is the frame for the whole post, and it is the reason this belongs at the end of Track F rather than in some other track. Quantization and graphs are *orthogonal*: one removes bytes, the other removes launches, they fix different bottlenecks, and — this is the part that makes them worth stacking — their speedups multiply instead of overlapping. By the end you will have `nanoserve/graphed_decode.py`: the decode step wrapped in a captured [CUDA graph](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) with static input and output buffers, a pool of graphs bucketed by batch size, and the whole thing plumbed into the `step()` loop from [the continuous-batching post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) so the running-set size picks the bucket. Then a `torch.compile` variant that does the adjacent job — fusing the small kernels so there are fewer to launch in the first place — and an honest account of which one to reach for.

Standard promise for this series, restated from [the introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is arithmetic I show you in full, a citation with a link, or a range framed as what you will see when you run the script. The results tables carry a `Source` column. The one headline benchmark I lean on hardest — a 56% throughput gain on a tiny model — is the vLLM team's, cited and dated, never mine.

---

## 1. The launch-bound regime, derived

[The kernel landscape post](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) counted the launches, so I will restate the count and then push it to the place where it bites. A Llama-style decoder layer, at decode time, launches roughly ten small GPU programs in order: an input RMSNorm, the QKV projection, the RoPE rotation, the KV-cache write, the attention kernel, the output projection, a residual add, the post-attention RMSNorm, the two or three MLP projections with a SwiGLU activation, and a second residual add. Call it ten kernels. Multiply by 32 layers, add the final norm and the vocabulary projection and the sampler, and you get **about 330 kernel launches to produce one token**.

Each launch is a small amount of *host* work: the CPU builds a launch descriptor, pushes it onto the CUDA stream, and lets the driver hand it to the GPU. That costs a few microseconds of CPU time per launch. NVIDIA's [CUDA Graphs documentation](https://developer.nvidia.com/blog/cuda-graphs/) motivates the entire feature by exactly this per-launch cost; the number to reproduce on your own box with `nsys` is around 5 microseconds. Do the arithmetic. If each launch costs $L$ microseconds and a step issues $K$ launches, the host needs

$$t_{\text{launch}} = K \cdot L = 330 \times 5\,\mu s = 1.65\ \text{ms}$$

just to *submit* the step's work — before the GPU has done a single useful multiply.

Now put that next to the GPU-side floor. A decode step is memory-bound, so its floor is the time to stream the weights across HBM once: weight bytes divided by bandwidth (the derivation lives in [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) and [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)). For Llama-3.1-8B at 16 GB in bf16 on an A100 at about 2.0 TB/s, the floor is $16 / 2039 \approx 7.9$ ms. The host submits all 330 launches in 1.65 ms — comfortably ahead of the 7.9 ms of GPU work, so the launches hide completely behind the compute and the GPU stays fed. Launch overhead is invisible. This is why nobody notices it on a big model.

![A timeline of one decode step showing many tiny kernel launches separated by idle gaps where the GPU waits for the host](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-2.webp)

The regime flips when the GPU work shrinks below the host work. Formally, you are launch-bound when

$$t_{\text{launch}} \gt t_{\text{GPU}} \quad\Longleftrightarrow\quad K \cdot L \gt \frac{\text{weight bytes}}{\text{bandwidth}}.$$

Read the two sides. The left grows with the number of kernels, which is roughly proportional to layer count and barely moves when you quantize — quantization changes the bytes each kernel reads, not how many kernels there are. The right *shrinks* when the model gets smaller (fewer weight bytes) or the GPU gets faster (more bandwidth). So the launch-bound regime is reached from two directions at once: **small models and fast GPUs**. A 405B model on an A100 is never launch-bound; a 0.6B model on an H100 almost always is. That is the single most important intuition in this post — the fix you are about to build helps *least* exactly where quantization helps most (big models, where you are drowning in bytes) and helps *most* where quantization does nothing (tiny models on fast silicon, where you are drowning in launches).

#### Worked example: when does a small model on a fast GPU cross over?

Take Llama-3.2-1B: 16 layers, about 2.5 GB in bf16, roughly 163 launches per token (16 layers times ~10 kernels, plus the head and sampler). On an A100 at 2.0 TB/s the GPU floor is $2.5 / 2039 \approx 1.2$ ms; the launch time is $163 \times 5\,\mu s = 0.82$ ms, which is **68% of the floor** — borderline, launches are a real tax but compute still just barely dominates. Move the same model to an H100 at 3.35 TB/s and the floor drops to $2.5 / 3350 \approx 0.75$ ms, while the launch time is unchanged at 0.82 ms. Now $t_{\text{launch}} \gt t_{\text{GPU}}$: **the host cannot submit kernels as fast as the GPU could run them.** The 1B model on an H100 is launch-bound, and no kernel optimization rescues it, because the kernels are not the problem — the gaps between them are. `Source: derived` (launch count from architecture; per-launch 5 µs cited from NVIDIA's CUDA-Graphs blog; verify both with `nsys`).

Push it one model smaller and the effect is not borderline, it is dominant. A 0.6B model is roughly 1.2 GB in bf16 and issues on the order of 280 launches. On an H100 the GPU floor is $1.2 / 3350 \approx 0.36$ ms, while the launch time is about $280 \times 5\,\mu s = 1.4$ ms — the host work is nearly **four times** the GPU work. This is not a hypothetical corner. The vLLM team's [Model Runner V2 write-up](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) reports a **56% throughput gain** (25K vs 16K output tokens/s) on Qwen3-0.6B on a single GB200 — a deliberately host-overhead-dominated stress case — from a redesign whose whole purpose is to stop the launch loop from being the wall (they build the step's input tensors on the GPU and schedule step N+1 while the GPU runs step N). That number is measured by people with the hardware, and it is the launch-bound regime made concrete: shave the host overhead on a tiny model and throughput jumps by more than half. `Source: cited (vLLM Model Runner V2, 2026-03-24)`.

So the target is set. On the models where quantization wins — 8B, 70B, 405B, where you are bandwidth-bound — CUDA graphs are a modest 10-to-20% cleanup. On the models where quantization does nothing for latency — the 0.6B and 1B draft models, the small routers, anything on a Blackwell-class GPU — CUDA graphs are the *primary* win, worth 1.5× and up. Keep that asymmetry in mind; it is the recommendation section in miniature.

---

## 2. CUDA graphs from first principles

The launch overhead is host work: 330 times per step, the CPU builds a descriptor and pushes it onto the stream. If the *sequence* of launches were identical every step — same kernels, same order, same arguments — then rebuilding that sequence 330 times per token is pure waste. That is the premise CUDA graphs exploit.

A CUDA graph is a **recording of a launch sequence**. You put the stream into capture mode, run your step once, and instead of executing the kernels the driver records them into a graph object: the ordered DAG of kernels, their launch configurations, and — critically — the exact memory addresses each kernel reads and writes. You then *instantiate* the graph once, and from then on a single `graph.replay()` call re-issues the whole recorded DAG with one launch from the host. The CPU says "go" once; the GPU runs all 330 kernels back to back with no host round-trip between them.

The saving is exactly the arithmetic from the last section, run backwards. Without a graph the host pays $K \cdot L$ per step. With a graph it pays one launch — call it $L$ — plus a small fixed cost to copy the fresh inputs in. So the host submission cost falls from $330 \times 5\,\mu s = 1.65$ ms to a few microseconds, and every idle gap that the per-launch cost created disappears.

![A before and after comparison of a decode step with graph capture off versus graph replay on](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-3.webp)

There is no free lunch, and the price is a pair of constraints that are the whole story of why graphs are easy in the decode loop and impossible in prefill. Because the graph records specific memory addresses, replay is only correct if:

1. **Static shapes.** Every tensor the recorded kernels touch must have the same shape on replay as at capture. The launch grid of a kernel is a function of its input shape; a graph baked at batch 4 launches the grid for batch 4, and replaying it with batch 5 reads and writes the wrong number of elements.
2. **Static addresses.** The graph does not record "read the input tensor," it records "read the 96 bytes starting at address `0x7f...`." So the inputs must live in **fixed buffers you overwrite in place** between replays. Allocate a new input tensor each step and its address changes and the graph reads stale or freed memory.

Now hold those two constraints against the two phases of LLM inference and the fit is uncanny.

**Decode is a perfect match.** A decode step processes exactly one new token per running request. The batch shape is `[B, 1]` — B rows, one token each — and it does not change from step to step as long as B is fixed. The KV cache is a preallocated tensor whose address never moves; the step only *appends* to it (see [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table)). If you keep the input token ids, the positions, and the block table in fixed buffers and overwrite them in place, both constraints hold and the whole step captures cleanly. Decode was practically designed for graph capture.

**Prefill is a perfect mismatch.** A prefill step processes the entire prompt at once, and prompts have variable length: 40 tokens here, 4,000 there. The shape `[1, seq_len]` changes on every request, so a graph captured at `seq_len = 40` is useless at `seq_len = 4000`. You could capture one graph per possible length, but the length space is effectively unbounded. This is why production engines capture graphs for decode and run prefill in eager mode (or [chunked prefill](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff), which caps the chunk size but still varies within the cap). The vLLM team's [V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) describes exactly this split: `torch.compile` plus piecewise CUDA graphs for the steady-state path, with FlashAttention 3 handling the mixed prefill-plus-decode dynamism that graphs cannot. `Source: cited (vLLM V1, 2025-01-27)`.

The reason the animation below matters more than a still frame: the win is temporal. It is not that the graph does less work — it does *exactly the same 330 kernels* — it is that it stops leaving the GPU idle *between* them. Watch the top lane light kernels one at a time with gaps, and the bottom lane fire them all from a single replay call with the gaps closed.

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="Two decode lanes under one clock: the top lane lights kernels one launch at a time with idle gaps and finishes late, the bottom lane fires one graph replay that lights all kernels at once and finishes early" style="width:100%;height:auto;max-width:860px">
<title>A launch-at-a-time decode lane versus a single graph-replay lane</title>
<style>
.cg-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.cg-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.cg-k{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cg-hi{fill:var(--accent,#6366f1);opacity:.22}
.cg-fill{fill:var(--accent,#6366f1);opacity:0}
.cg-gap{fill:var(--text-secondary,#6b7280);opacity:.18}
.cg-fin{font:600 12px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);opacity:0}
@keyframes cg-sweep{0%{transform:translateX(0)}70%{transform:translateX(560px)}100%{transform:translateX(560px)}}
@keyframes cg-late{0%,66%{opacity:0}72%,100%{opacity:1}}
@keyframes cg-replay{0%,4%{opacity:0}10%,100%{opacity:.55}}
@keyframes cg-early{0%,8%{opacity:0}14%,100%{opacity:1}}
.cg-hi{animation:cg-sweep 10s steps(8,end) infinite}
.cg-finlate{animation:cg-late 10s ease-in-out infinite}
.cg-fill{animation:cg-replay 10s ease-in-out infinite}
.cg-finearly{animation:cg-early 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.cg-hi{animation:none;transform:translateX(560px)}.cg-fill{animation:none;opacity:.55}.cg-finlate,.cg-finearly{animation:none;opacity:1}}
</style>
<text class="cg-lbl" x="24" y="34">One launch at a time (host-bound)</text>
<text class="cg-sub" x="24" y="54">CPU submits one kernel, GPU runs it, idle gap, repeat &#215; 330</text>
<rect class="cg-gap" x="24" y="72" width="632" height="56" rx="8"/>
<rect class="cg-k" x="32"  y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="108" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="184" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="260" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="336" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="412" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="488" y="76" width="60" height="48" rx="6"/>
<rect class="cg-k" x="564" y="76" width="60" height="48" rx="6"/>
<rect class="cg-hi" x="32" y="72" width="60" height="56" rx="6"/>
<text class="cg-finlate" x="636" y="104" text-anchor="end">finishes late</text>
<text class="cg-lbl" x="24" y="196">One graph replay (captured once)</text>
<text class="cg-sub" x="24" y="216">CPU submits one call, the GPU runs the whole recorded DAG</text>
<rect class="cg-k" x="32"  y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="108" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="184" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="260" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="336" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="412" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="488" y="238" width="60" height="48" rx="6"/>
<rect class="cg-k" x="564" y="238" width="60" height="48" rx="6"/>
<rect class="cg-fill" x="32" y="238" width="592" height="48" rx="6"/>
<text class="cg-finearly" x="230" y="266" text-anchor="start">one launch &#183; gaps closed &#183; finishes early</text>
</svg>
<figcaption>Under one clock the top lane lights kernels one launch at a time with idle gaps and finishes late; the bottom lane replays the same kernels from a single captured graph, lighting them together with the gaps closed, and finishes early.</figcaption>
</figure>

---

## 3. Capturing nanoserve's decode step

Enough principle. Here is the decode step `nanoserve` has been running eager since the [baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) — one token per running request, paged KV cache, block-table indirection:

```python
# nanoserve/graphed_decode.py  -- the eager step we are about to capture
import torch
import torch.nn.functional as F

@torch.inference_mode()
def decode_step(model, input_ids, positions, kv_cache, block_table, seq_lens):
    # input_ids : [B]     one new token id per running request
    # positions : [B]     absolute position of that token in each sequence
    # kv_cache  : preallocated paged tensor -- address is FIXED for the run
    # block_table: [B, max_blocks] int32 -- maps each row to its physical KV blocks
    # seq_lens  : [B] int32 -- current length of each sequence (for the attention mask)
    hidden = model.embed(input_ids)                    # [B, H]
    for layer in model.layers:
        hidden = layer(hidden, positions, kv_cache, block_table, seq_lens)
    hidden = model.norm(hidden)
    return model.lm_head(hidden)                        # [B, vocab]  (logits)
```

To capture it we need every input to live in a fixed buffer, and we need to warm the step up before capture so that all the lazy, one-time work — memory-pool growth, cuBLAS handle creation, Triton autotuning — happens *outside* the recording. If any of that fires during capture, it either fails the capture outright or bakes a one-time allocation into the graph. The canonical warmup-then-capture dance runs on a side stream:

```python
class GraphedStep:
    """Captures decode_step for ONE fixed batch size B into a CUDA graph."""
    def __init__(self, model, kv_cache, batch_size, max_blocks, device="cuda"):
        self.model, self.kv_cache, self.B = model, kv_cache, batch_size
        V = model.config.vocab_size
        # STATIC input/output buffers: their addresses get baked into the graph,
        # so from now on we OVERWRITE them in place and never reallocate.
        self.in_ids   = torch.zeros(batch_size, dtype=torch.long,  device=device)
        self.pos      = torch.zeros(batch_size, dtype=torch.long,  device=device)
        self.blk_tbl  = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
        self.seq_lens = torch.ones (batch_size, dtype=torch.int32, device=device)
        self.logits   = torch.zeros(batch_size, V, dtype=torch.float32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._capture()

    def _run(self):
        # references ONLY the static tensors -- this is what gets recorded
        out = decode_step(self.model, self.in_ids, self.pos,
                          self.kv_cache, self.blk_tbl, self.seq_lens)
        self.logits.copy_(out)                          # write into the static output

    def _capture(self):
        # 1) warm up on a side stream so allocations / cuBLAS / autotune finish FIRST
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        # 2) record the launch sequence exactly once
        with torch.cuda.graph(self.graph):
            self._run()

    def replay(self, in_ids, pos, blk_tbl, seq_lens):
        # overwrite the static inputs IN PLACE, then fire ONE launch
        self.in_ids.copy_(in_ids)
        self.pos.copy_(pos)
        self.blk_tbl[:, :blk_tbl.shape[1]].copy_(blk_tbl)
        self.seq_lens.copy_(seq_lens)
        self.graph.replay()
        return self.logits            # a VIEW of the static output buffer
```

Three details in that code are the difference between a graph that works and one that silently corrupts output.

**The `copy_` into static buffers is the entire contract.** `self.in_ids.copy_(in_ids)` overwrites the recorded buffer in place; it does *not* rebind `self.in_ids` to a new tensor. If you had written `self.in_ids = in_ids`, the graph would keep reading the old address and you would decode the same token forever. Every input crosses the boundary by value-copy into a fixed address.

**The output is a view, not a fresh tensor.** `replay()` returns `self.logits`, which the graph writes into. If two replays happen before you read the first result, the second clobbers the first. In the single-threaded decode loop that is fine — you sample from the logits immediately — but it is a real hazard the moment you try to overlap steps, and it is why the sampler must consume the logits before the next `replay()`.

**The warmup count is not decoration.** The first replay after capture is *not* representative — the first few real iterations still pay for anything the warmup did not trigger. Three warmup iterations is the usual floor; some backends need more if they autotune per-shape. This is the same warmup discipline the [benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) insists on, now load-bearing for correctness and not just for honest timing.

There is a higher-level API, `torch.cuda.make_graphed_callables`, that does the warmup-and-capture for you and hands back a drop-in callable. It is cleaner when your step is a single `nn.Module` with fixed-shape inputs:

```python
# Convenience wrapper: warm up + capture in one call, returns a graphed callable.
sample_in = (torch.zeros(B, dtype=torch.long, device="cuda"),
             torch.zeros(B, dtype=torch.long, device="cuda"))
graphed_embed_and_layers = torch.cuda.make_graphed_callables(
    model.decode_module,      # an nn.Module whose forward is the fixed-shape step
    sample_in,                # example inputs; their SHAPES are frozen into the graph
    num_warmup_iters=3,
)
```

`make_graphed_callables` is the right tool when the captured region is a clean module boundary. The hand-rolled `CUDAGraph` above is the right tool when you need to control exactly which tensors are static and which stay outside the graph — which, once you add a paged KV cache and a block table, you do. We keep the hand-rolled version for `nanoserve` because section 6's gotcha about block-table pointers needs that control.

#### Worked example: what does one captured step actually save?

Take the launch-bound 1B-on-H100 case: GPU floor 0.75 ms, launch cost 0.82 ms, so eager decode runs at roughly $\max(0.82, 0.75) \approx 0.82$ ms per step in the best case and worse in practice, because every host stall between kernels serializes rather than overlaps. Call the eager step 0.9 ms. After capture the host cost collapses to a single launch of a few microseconds, so the step runs at the GPU floor of 0.75 ms — a step-time drop from about 0.9 ms to 0.75 ms, roughly **17% faster**, which translates to ~17% more tokens/s at batch 1. `Source: derived` (floors from bandwidth, eager step from the launch arithmetic; reproduce with CUDA events). That is the *conservative* estimate, because it credits the eager path with perfect launch overlap. The vLLM-measured 56% on a 0.6B model is what happens when the eager path was *not* overlapping and the host was truly the wall — which is the regime the smaller you go. Frame your own expectation between those two: single-digit-to-mid-teens percent on an 8B model, tens of percent on a sub-1B model, and reproduce it before you quote it.

---

## 4. One graph per batch size: bucketing

A captured graph is frozen at one batch size. But `nanoserve`'s [continuous-batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) changes its running-set size *every step* — request C finishes, requests G and H are admitted, and the batch goes from 4 rows to 5. A graph baked at B=4 cannot replay at B=5. So how does any of this work in a real engine where the batch breathes?

The answer is **bucketing**: capture a graph for each of a fixed set of batch sizes, and at each step pad the actual running set up to the nearest captured size and replay that graph. Production engines do exactly this. The vLLM team's [torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20) exposes it as `compile_sizes: [1, 2, 4]` — a list of static batch sizes to specialize and capture — and captures piecewise graphs for those sizes while everything else runs a more general path. `Source: cited (vLLM torch.compile, 2025-08-20)`.

![A matrix mapping running-set sizes to the nearest captured bucket, the padding waste incurred, and whether the graph or eager path is taken](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-6.webp)

The trade-off the matrix above makes visible is a padding tax, and it is the same tax [static batching](/blog/machine-learning/inference-engineering/static-batching-and-the-padding-tax) pays, reappearing in a new place. If your buckets are `{1, 2, 4, 8, 16, 32}` and the running set holds 5 requests, you pad up to bucket 8 and replay the batch-8 graph — running 8 rows of work to serve 5, a 37% waste on that step. Denser buckets (say every power of 2, or even every integer up to some cap) cut the padding but multiply the number of graphs you capture and the VRAM they pin. Sparser buckets save memory and capture time but pad harder. The sweet spot depends on your batch-size distribution: if you almost always run 28-to-32 requests, capture 32 and eat the small pad; if your load is spiky and often tiny, capture the small sizes densely because that is where the launch-bound regime lives.

Here is the pool and the padding, as a class that wraps a dict of `GraphedStep`s:

```python
class BucketedGraphs:
    """A pool of decode graphs, one per captured batch size, with an eager fallback."""
    def __init__(self, model, kv_cache, buckets=(1, 2, 4, 8, 16, 32), max_blocks=512):
        self.model, self.kv_cache = model, kv_cache
        self.buckets = sorted(buckets)
        self.graphs = {b: GraphedStep(model, kv_cache, b, max_blocks) for b in self.buckets}

    def pick(self, n):
        for b in self.buckets:      # smallest captured size that fits n
            if b >= n:
                return b
        return None                 # n exceeds the largest bucket -> escapes to eager

    @torch.inference_mode()
    def run(self, in_ids, pos, blk_tbl, seq_lens):
        n = in_ids.shape[0]
        b = self.pick(n)
        if b is None:               # too big for any graph: run it eager, no capture
            return decode_step(self.model, in_ids, pos, self.kv_cache, blk_tbl, seq_lens)
        pad = b - n
        if pad:                     # pad the ragged running set UP to the bucket size
            in_ids   = F.pad(in_ids,   (0, pad))
            pos      = F.pad(pos,      (0, pad))
            seq_lens = F.pad(seq_lens, (0, pad), value=1)   # len 1, not 0: avoid an empty attend
            blk_tbl  = F.pad(blk_tbl,  (0, 0, 0, pad))
        logits = self.graphs[b].replay(in_ids, pos, blk_tbl, seq_lens)
        return logits[:n]           # drop the padded rows before sampling
```

Two subtleties that bite if you skip them. First, **pad `seq_lens` to 1, not 0.** A padded row with sequence length 0 makes the attention kernel attend over nothing, which some backends handle and some turn into a NaN or an illegal read; giving it a length-1 dummy sequence pointing at a scratch block keeps the kernel on a well-defined path, and you throw the row's output away anyway. Second, **slice `logits[:n]` before sampling.** The graph computed 8 rows of logits; only the first `n` are real. Sample those and discard the pad. Forget the slice and you will admit garbage tokens for requests that do not exist.

Now plumb it into the engine. The `step()` from the continuous-batching post built a ragged batch from `self.sched.running` and called the model eagerly; the only change is to route the forward through the graph pool, keyed on the running-set size:

```python
class Engine:
    def __init__(self, model, scheduler, kv_store):
        self.model, self.sched, self.kv = model, scheduler, kv_store
        # capture the decode graphs ONCE at startup, after weights are loaded
        self.graphs = BucketedGraphs(model, kv_store.cache, buckets=(1, 2, 4, 8, 16, 32))

    @torch.inference_mode()
    def step(self):
        out = self.sched.schedule()                     # admit + build the running set
        batch = self._build_decode_batch(out.running)   # ragged -> [n] tensors + block table
        # the running-set size n picks the bucket; padding + replay happen inside run()
        logits = self.graphs.run(batch.in_ids, batch.positions,
                                 batch.block_table, batch.seq_lens)
        next_ids = self.sampler(logits)                 # [n] sampled token ids
        return self._commit(out.running, next_ids)      # append KV, retire finished reqs
```

That is the entire integration. The running-set size — the thing that changes every step — is precisely the bucketing key, and everything downstream (sampling, KV append, retirement) is unchanged because `run()` hands back exactly `n` rows of logits. The graph machinery is invisible to the rest of the engine, which is how it should be.

### The memory cost, derived

Every captured graph pins memory, and that memory comes out of the same VRAM budget the KV cache wants — so bucketing is not free, it is a trade of KV-cache capacity for launch overhead.

![A stacked VRAM budget showing weights, a KV cache shrunk by the graph buffers, per-bucket static IO, replay workspace, and a shared capture pool](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-5.webp)

What does each graph actually hold? Two things. The **static I/O buffers** are small: for bucket B, the input ids and positions are $B$ longs, the block table is $B \times \text{max\_blocks}$ int32, and the output logits are $B \times V$ floats. The logits dominate — at $V = 128{,}000$ and fp32, one row is $128{,}000 \times 4 = 512$ KB, so a batch-32 graph's logit buffer is 16 MB. Across six buckets summing to ${1+2+4+8+16+32 = 63}$ rows, that is about 32 MB of logit buffers — real, but small next to a 16 GB model. The **replay workspace** — the intermediate activations the captured kernels scribble through — is the bigger and subtler cost, because by default each graph's captured allocations are private and pinned for the life of the graph.

The fix production engines use is a **shared memory pool** across graphs. `torch.cuda.graph(g, pool=shared_pool)` lets several graphs draw their captured allocations from one pool, so you reserve the workspace of the *largest* bucket once instead of summing every bucket's workspace. Even so, that reserved workspace is VRAM the KV cache cannot use. This is why vLLM exposes a knob to cap it — the [GPT-OSS optimization post](https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations) (2026-02-01) mentions `--cuda-graph-capture-size` to bound the largest captured batch and therefore the reserved workspace. `Source: cited (vLLM GPT-OSS, 2026-02-01)`. The honest framing for `nanoserve`: capturing graphs steals a few hundred megabytes to a couple of gigabytes from your KV-cache budget, which at 128 KB/token for Llama-3.1-8B (from [the memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache)) is thousands of tokens of context you gave up to close the launch gaps. On a launch-bound small model that trade is obviously worth it; on a bandwidth-bound large model where graphs buy little, it may not be.

---

## 5. torch.compile: fewer kernels to launch in the first place

CUDA graphs make the 330 launches cost one launch. `torch.compile` does the adjacent and complementary job: it makes there be *fewer than 330 kernels* by fusing them. The two are not rivals — the production path uses both, and `torch.compile`'s `reduce-overhead` mode literally wraps its fused output in CUDA graphs. But the mechanism is worth understanding on its own, because it is what turns the tiny elementwise kernels (the norms, the activation, the residual adds — the ones that are pure launch overhead) into a handful of fused ones.

`torch.compile` is a two-stage JIT. The vLLM team's [torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20) describes both stages, and the mechanism matches PyTorch's own [documentation of what torch.compile does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does):

- **TorchDynamo** traces your Python by intercepting bytecode and building an FX graph of the tensor operations. When it hits something it cannot trace — a data-dependent Python branch, a `.item()`, an unsupported op — it inserts a **graph break**: it compiles the traced portion, runs the untraceable bit in eager Python, and resumes tracing after. Graph breaks are not fatal; they just carve the model into compiled islands separated by eager water.
- **TorchInductor** takes each FX graph and lowers it: it fuses pointwise and reduction ops so a norm-then-scale-then-add becomes one kernel instead of three, generates [Triton](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) for the fused kernels, picks a matmul backend (cuBLAS, Triton, or CUTLASS) by autotuning, and — in `reduce-overhead` mode — caches CUDA graphs of the result.

The fusion wins are real and cited. The vLLM torch.compile post reports **SiLU+Quant FP8 fusion up to 8%** (on Llama 3.1 405B FP8, 8×MI300) and **AllReduce+RMSNorm fusion up to 15%**, with TorchBench showing a 1.8–2× geomean speedup across 80+ models. `Source: cited (vLLM torch.compile, 2025-08-20)`. Notice *which* kernels those fusions target: the SiLU activation and the RMSNorm — exactly the tiny elementwise kernels that the [kernel-landscape post](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs) flagged as pure launch overhead. Fusing the big GEMMs buys little because they are already at their bandwidth floor; fusing the small kernels removes launches, which is the whole game in the launch-bound regime.

### Piecewise CUDA graphs

Here is where the two techniques merge, and it is the design vLLM ships. Not every part of the model is graph-safe. The attention kernel, in particular, can have a **variable launch grid** — its grid depends on sequence lengths and, in cascade or dynamic forms, on runtime metadata — which is exactly the static-shape constraint graphs cannot honor. So you cannot capture the whole model into one graph if attention is dynamic.

**Piecewise CUDA graphs** are the answer: `torch.compile` splits the model into graph-safe regions and graph-unsafe regions, captures the safe regions into CUDA graphs, and runs the unsafe regions (the dynamic attention) in eager mode between them. The vLLM torch.compile post names cascade attention as the unsafe region that forces the split. `Source: cited (vLLM torch.compile, 2025-08-20)`. You get graphs where you can and eager where you must, and the fused, captured MLP-and-norm regions carry most of the launch-count reduction anyway.

![A branching dataflow showing Dynamo tracing into a graph-safe fused region captured as a CUDA graph and a graph-unsafe attention region run eager, both merging into one decode step](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-4.webp)

The `nanoserve` version of the compiled path is almost anticlimactic, which is the point — the compiler does the work:

```python
# nanoserve/compiled_decode.py  -- the torch.compile alternative to hand-rolled graphs
# reduce-overhead: Inductor fuses the pointwise/reduction ops AND wraps the
# result in CUDA graphs automatically. dynamic=False specializes on shape.
compiled_step = torch.compile(model.decode_module, mode="reduce-overhead", dynamic=False)

# Warm each bucket ONCE at startup so its graph is captured up front, not mid-serving.
for b in (1, 2, 4, 8, 16, 32):
    warm_ids = torch.zeros(b, dtype=torch.long, device="cuda")
    warm_pos = torch.zeros(b, dtype=torch.long, device="cuda")
    for _ in range(3):
        compiled_step(warm_ids, warm_pos, kv_cache, block_table[:b], seq_lens[:b])
torch.cuda.synchronize()
# from here, compiled_step(...) at a warmed batch size replays a captured graph
```

The flags you need to know, all from the cited vLLM post:

- **Disable it** with `-O0` or `--enforce-eager` — the escape hatch when the compiler misbehaves or startup time is unacceptable.
- **`compile_sizes: [1, 2, 4]`** — the static batch sizes to specialize and capture, the `CompilationConfig` analog of the buckets you hand-built in section 4.
- **The cache lives in `~/.cache/vllm/torch_compile_cache`** — compiled artifacts persist across runs so you pay the compile cost once. Disable the cache with `VLLM_DISABLE_COMPILE_CACHE=1` when you suspect a stale artifact.

`Source: cited (vLLM torch.compile, 2025-08-20)` for all four.

### Hand-rolled graphs vs torch.compile: which for nanoserve?

| Property | Hand-rolled `CUDAGraph` | `torch.compile(reduce-overhead)` | Source |
|---|---|---|---|
| Reduces launch count | No — same kernels, one launch | Yes — fuses small kernels, then graphs | derived |
| Removes host submit cost | Yes — 330 launches to 1 | Yes — via wrapped CUDA graphs | derived |
| Fusion wins (SiLU+Quant, AllReduce+RMSNorm) | No | up to 8% / up to 15% | cited: vLLM torch.compile |
| Control over static buffers / paged KV | Full — you own every tensor | Partial — compiler decides | derived |
| Startup cost | Warmup + capture, seconds | Compile + capture, tens of seconds+ | cited: vLLM torch.compile |
| Handles dynamic attention | You must split by hand | Piecewise split is automatic | cited: vLLM torch.compile |
| Lines of code | ~80 (the pool + step) | ~5 | derived |

The honest recommendation: use `torch.compile(mode="reduce-overhead")` first — it does both jobs (fuse and graph) in five lines and gets the piecewise split right automatically. Reach for the hand-rolled `CUDAGraph` when you need explicit control over which tensors are static — specifically when your paged KV cache and block table need to stay outside the graph while the block-table *contents* change every step, which is the next section's gotcha and the reason `nanoserve` keeps both implementations.

---

## 6. The gotchas that bite

Every one of these I have watched turn a "2× faster" graph into a silent corruption or a recapture storm. They are the reason CUDA graphs have a reputation for being finicky, and every one of them traces back to the two constraints from section 2: static shapes, static addresses.

**Dynamic shapes break capture — bucket them.** This is the whole reason section 4 exists. If any input shape varies and you have not bucketed it, the graph reads or writes the wrong extent. The failure mode is not always a crash; a graph replayed at the wrong batch size can read adjacent live memory and produce plausible-but-wrong logits. Bucket every dynamic dimension, pad up, and slice down.

**The attention kernel's variable launch grid replays badly.** The vLLM team's [Triton attention backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) is explicit about this: their persistent-kernel attention variant uses a **fixed number of kernel instances that read work metadata from GPU memory**, precisely because a kernel whose launch grid varies with the batch "replay[s] badly under CUDA graphs." A fixed launch grid can be captured and replayed; a grid that changes with sequence lengths cannot. `Source: cited (vLLM Triton backend, 2026-03-04)`. This is the mechanism behind the piecewise split — the attention region is the unsafe one — and the persistent-kernel design is how you make even attention capturable, by moving the variability from the launch grid (which the graph freezes) into GPU-side metadata (which the graph does not).

**Recapture storms if shapes vary every step.** If you feed `torch.compile` a shape it has not seen, Dynamo recompiles — and if your shapes vary every single step (because you did not bucket, or because your `seq_lens` tensor changes rank, or a stray `.item()` leaks a dynamic value into a shape), you recompile every step. Recompilation is orders of magnitude slower than a step, so your throughput craters and your `nsys` trace fills with compile activity instead of kernels. This is the same recompile hazard the [Triton post](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda) warned about for autotuned kernels, and the guard is the same:

```python
# Turn silent recompiles into a loud error while you debug the decode loop.
import torch._dynamo
torch._dynamo.config.error_on_recompile = True   # raises on any recompilation
# and pin the dynamic dims so shapes are stable across steps:
# - keep block_table at a fixed [B, max_blocks] and slice, never reshape
# - bucket the batch dim; never let n leak into a shape unbucketed
# - keep seq_lens as a fixed-length int32 tensor, updated in place
```

**The memory cost of many graphs.** Covered in section 4 — every bucket pins buffers and reserves workspace out of the KV budget. The failure mode is an OOM at capture time, or a KV cache so shrunk you preempt requests you would not otherwise have. Cap the number and size of buckets; do not capture 64 of them because you can.

**The paged KV cache interaction — the subtle one.** This is where the hand-rolled graph earns its keep, and it is the most-asked question about graphs plus paged attention: if the graph bakes in memory addresses, and paged attention reads KV from *different physical blocks* for different requests, how can the graph be valid across replays with different block tables?

The resolution is a clean separation between the tensor and its contents. The **KV cache is one big preallocated tensor** whose base address never moves — the graph bakes in that address and it stays correct. The **block table is also a fixed tensor** — shape `[B, max_blocks]`, address baked into the graph — but its *contents* are integers that the attention kernel reads at replay time to know which blocks to gather. So across replays, the graph reads the same block-table tensor at the same address; you have overwritten its *contents* in place with this step's block indices; and the attention kernel dereferences those fresh indices into the fixed KV tensor. Nothing the graph recorded moved; only the integers inside the fixed buffer changed. That is exactly why `replay()` does `self.blk_tbl[:, :].copy_(blk_tbl)` — an in-place content update of a static tensor — and never rebinds the tensor. Get this wrong (allocate a fresh block table each step) and the graph reads a stale or freed table and gathers garbage KV, which shows up as a model that was fine at capture and degrades over the run.

**torch.compile startup time is a real autoscaling pain.** The vLLM torch.compile post names it directly: startup time is "a pain for autoscaling," and many of the torch.compile APIs it relies on are private. `Source: cited (vLLM torch.compile, 2025-08-20)`. The mechanism: when a new replica boots to absorb a traffic spike, it must compile and capture before it can serve at full speed, and that compile can take tens of seconds. Your autoscaler adds a replica to shed load, but the replica is slow for its first minute — exactly when you needed it fast. The persistent cache in `~/.cache/vllm/torch_compile_cache` is the mitigation (bake the cache into the image so a fresh replica loads instead of compiles), and `--enforce-eager` is the escape hatch when you would rather serve immediately at eager speed than wait for the compile.

#### Worked example: a graph that was fine at capture and wrong by token 200

You capture the batch-8 decode graph at startup with a fresh block table full of zeros — every row points at block 0. Warmup runs clean, capture succeeds, the first tokens look great. By token 200, requests have grown past their first KV block and the block table now points at blocks 12, 47, 103. If your `replay()` rebinds `self.blk_tbl = blk_tbl` instead of `self.blk_tbl.copy_(blk_tbl)`, the graph still reads the *original* zeroed table at the *original* address — every request attends over block 0 — and the model produces fluent nonsense that gets worse as sequences diverge from their capture-time state. The tell is that quality degrades with sequence length and is fine for short prompts, which points straight at the block table, not the weights. `Source: derived` (a failure mode of address-baking; reproduce by rebinding vs copying and diffing logits at token 200).

---

## 7. Numbers, with provenance

Every number below is derived from the formulas in section 1 or cited from a named source. No row is a first-hand benchmark of mine.

**The launch-bound crossover.** When is a model launch-bound, i.e. when does closing the gaps actually help? Compare $t_{\text{launch}} = K \cdot 5\,\mu s$ against the GPU floor $= \text{weight bytes} / \text{bandwidth}$.

| Model | Layers | Weights (bf16) | Launches K | GPU | HBM BW | GPU floor | $t_{\text{launch}}$ | Launch-bound? | Source |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 16 GB | 330 | A100 | 2.0 TB/s | 7.9 ms | 1.65 ms | No — 21% of floor | derived |
| Llama-3.1-8B | 32 | 16 GB | 330 | H100 | 3.35 TB/s | 4.8 ms | 1.65 ms | No — 34% of floor | derived |
| Llama-3.2-1B | 16 | 2.5 GB | 163 | A100 | 2.0 TB/s | 1.2 ms | 0.82 ms | Borderline — 68% | derived |
| Llama-3.2-1B | 16 | 2.5 GB | 163 | H100 | 3.35 TB/s | 0.75 ms | 0.82 ms | Yes — exceeds floor | derived |
| ~0.6B | ~28 | ~1.2 GB | ~280 | H100 | 3.35 TB/s | 0.36 ms | 1.4 ms | Yes — ~4× floor | derived |
| RTX 4090 baseline | — | — | — | RTX 4090 | ~1.0 TB/s | see note | — | cited: NVIDIA spec |

Note on the 4090: at ~1.0 TB/s (GDDR6X, NVIDIA spec) the 4090's memory floor is higher than the H100's, so *the same small model is less launch-bound on a 4090 than on an H100* — the slower memory gives the launches more compute to hide behind. This is the crossover's counterintuitive edge: a "worse" GPU can be *less* in need of CUDA graphs, because launch-bound is about the ratio of host work to GPU work, and slow memory raises the GPU-work side.

**The speedups.** Two kinds, kept separate:

| Win | Magnitude | Setup | Source |
|---|---|---|---|
| CUDA-graph replay (launch collapse) | host submit 1.65 ms → few µs | Llama-3.1-8B, per-step, derived | derived |
| CUDA-graph replay, 1B on H100 | step 0.9 ms → 0.75 ms (~17%) | launch-bound case, derived | derived |
| Model Runner V2 (host-overhead redesign) | +56% throughput (25K vs 16K tok/s) | Qwen3-0.6B, 1×GB200 | cited: vLLM MRv2 2026-03-24 |
| SiLU+Quant FP8 fusion | up to 8% | Llama-3.1-405B FP8, 8×MI300 | cited: vLLM torch.compile 2025-08-20 |
| AllReduce+RMSNorm fusion | up to 15% | reported range | cited: vLLM torch.compile 2025-08-20 |
| TorchBench geomean | 1.8–2× | 80+ models | cited: vLLM torch.compile 2025-08-20 |

### How to measure this honestly

The failure mode when you time CUDA graphs is that you accidentally measure the wrong thing, and the direction of the error is always "graphs look better than they are" because the whole point of graphs is to overlap host and device — so a naive timer that includes the host-side `copy_` calls but not the device work will flatter them. The discipline:

```python
# nanoserve/bench_graphed.py
def time_step(fn, iters=200, warmup=30):
    for _ in range(warmup):     # capture-warmup PLUS timer-warmup; do not skip
        fn()
    torch.cuda.synchronize()    # drain the queue before we start the clock
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()                    # for the graphed path this is one replay()
    end.record()
    torch.cuda.synchronize()    # WAIT for the GPU before reading the timer
    return start.elapsed_time(end) / iters   # ms per step, device time
```

The rules baked in, each one a mistake people make the first time: **warm up past the capture** (the first replays after capture still pay one-time costs and would inflate the average); **use CUDA events, not `time.time()`** (launches are async, so wall-clock timing on the host measures submission, not execution — exactly the thing graphs change, which is why host timers lie about graphs specifically); **synchronize before you read the timer** (or you time the launch of the last replay, not its completion); and **measure at steady state and at a fixed batch size** so you are timing one bucket's replay, not a mix. And the load-shaped honesty from the [reproducible-benchmark post](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark): tok/s at batch 1 tells you the launch-bound story but nothing about a loaded server, where the batch grows, the buckets shift, and the padding tax appears. Measure the batch-1 win *and* the loaded-server win; they are different numbers with different provenance.

---

## 8. Composition: this stacks with quantization

Return to the frame from figure 1, now that you have the mechanism. Quantization (posts 29–31 of this track, and the [model-serving quantization deep dive](/blog/machine-learning/model-serving/quantization-for-llm-serving)) cuts the *bytes* a decode step reads: INT4 weights quarter the 16 GB weight stream, FP8 KV halves the cache traffic. CUDA graphs cut the *launches*: 330 host submissions become one. These attack two different terms in the decode-step cost, and that is exactly why they compose rather than overlap.

Write the decode step time, crudely, as the max of two things it must wait on:

$$t_{\text{step}} \approx \max\big(\,t_{\text{GPU}},\ t_{\text{launch}}\,\big) = \max\Big(\frac{\text{weight bytes}}{\text{bandwidth}},\ K \cdot L\Big).$$

Quantization shrinks the first argument (fewer weight bytes). CUDA graphs shrink the second (fewer launches worth of host cost). If you only quantize, you drive $t_{\text{GPU}}$ down until it drops *below* $t_{\text{launch}}$ — and then you are launch-bound, and further quantization does nothing, because the max is now pinned by the host term. This is the trap the intro described: you quantized the 1B model to INT4, drove its GPU floor down, and hit the launch wall, so the tokens/s stopped moving. The only thing that helps past that point is cutting launches.

Run it the other way and the same logic holds. If you only apply CUDA graphs to a big bandwidth-bound model, you drive $t_{\text{launch}}$ to nearly zero — but $t_{\text{GPU}}$ was the max all along, so the step time barely moves. The [dequant-fused GEMM](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) is what moves that model. The point is symmetric: **whichever term is your max, that is the technique you need, and applying the other one alone does little.** The win only compounds when you apply both, because after quantization drops $t_{\text{GPU}}$ under $t_{\text{launch}}$, graphs then drop $t_{\text{launch}}$ under the new $t_{\text{GPU}}$, and you have moved *both* walls. That is the optimization-composition idea the [capstone](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) makes into a discipline: profile to find which term is the max, apply the technique that moves *that* term, then re-profile, because the max has probably switched to the other term.

#### Worked example: quantize then graph a 1B model, in order

Start with Llama-3.2-1B in bf16 on an H100, eager. GPU floor ${2.5/3350 = 0.75}$ ms; launch cost ${0.82}$ ms; step $\approx \max(0.75, 0.82) = 0.82$ ms, launch-bound. **Quantize to INT4:** weights drop to ~0.7 GB, GPU floor drops to ${0.7/3350 = 0.21}$ ms — but the step is still $\max(0.21, 0.82) = 0.82$ ms, because you are now *deeply* launch-bound and the quantization bought nothing for latency. Tokens/s unchanged. **Now add CUDA graphs:** launch cost collapses to a few µs, so the step becomes $\max(0.21, \sim 0) = 0.21$ ms — a step-time drop from 0.82 ms to 0.21 ms, nearly **4× faster**, and *none* of it would have shown up from quantization alone. `Source: derived` (floors from bandwidth; launch cost from section 1; reproduce with CUDA events). The order does not matter for the final number, but it does for the lesson: neither technique alone gets you there, and if you had measured only the quantization step you would have concluded, wrongly, that quantizing a 1B model is useless.

---

## 9. Stress tests

The kit's rule for this series is to build the thing and then try to break it. Four ways this breaks.

**A step that is already compute-bound (large batch).** At batch 128 on an 8B model, the GPU is busy the whole step — the weight read is amortized across 128 rows, the GEMMs are fat and efficient, and the launches hide completely behind the compute. Capture a graph here and you save almost nothing, because there were no idle gaps to close. Be honest about this: **CUDA graphs are a batch-1-and-small-model win.** The bigger your batch and the bigger your model, the less they buy, until at large batch on a large model they are within noise. This is the exact inverse of quantization, which helps *more* at large batch (more bytes to save). It is also why the composition in section 8 matters: the two techniques cover each other's weak regimes.

**A shape that escapes the buckets every step.** Suppose your load oscillates and the running set is 33, then 31, then 40 — and your top bucket is 32. Every step above 32 falls to the eager path (the `pick()` returning `None` branch), so you pay full launch overhead on exactly the steps where the batch is largest. Worse, if you naively "just capture a bigger bucket" up to 64, you have doubled the reserved workspace and shrunk the KV cache. The fix is to size your top bucket to your `max_running` and never let `n` exceed it — which the [scheduler](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) already caps — so there is no escape in steady state. The escape path is a correctness fallback, not a hot path; if your profile shows it firing often, your buckets are mis-sized.

**A model whose attention cannot be captured.** Cascade attention, or any attention with a launch grid that depends on runtime metadata, cannot go into a whole-model graph — this is the piecewise case. You do not give up graphs; you split. The MLP-and-norm regions (most of the launch count) capture into graphs, and the attention region runs eager between them. You lose the launches you cannot capture and keep the ones you can, and the vLLM Triton persistent-kernel design (section 6) shows the further step: re-engineer attention to a fixed launch grid so even it becomes capturable. The stress test's lesson is that "can't capture attention" is a reason to go piecewise, not a reason to abandon graphs.

**The first token after capture.** Capture happens once at startup, but the *first replay* after capture is not free — the buffers are cold, and the very first real step through a freshly captured graph pays for anything the warmup did not trigger (page faults on first touch, a cuBLAS handle the warmup shape did not exercise). In a server this is invisible because it happens before the first request. But if you capture lazily — the first time a bucket is hit — you will see a latency spike on the first request that lands in a new bucket, and it will look like a mysterious p99 outlier. Capture *all* buckets eagerly at startup (as the `BucketedGraphs.__init__` does) so the spike is paid once, at boot, and never on the serving path. The tell of a lazy-capture bug is a p99 that has occasional multi-hundred-millisecond spikes correlated with the batch size hitting a value it has not hit before.

---

## 10. Case studies and real numbers

Four public results, each cited, that ground everything above.

**vLLM Model Runner V2 — the launch-bound regime, measured.** The [MRv2 write-up](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) reports **+56% throughput (25K vs 16K output tok/s) on Qwen3-0.6B on one GB200**, a deliberately host-overhead-dominated case, from a redesign that builds the step's input tensors on the GPU (via Triton kernels) and schedules step N+1 while the GPU runs step N. On the flip side, on a large FP8 MoE (GLM-4.7-FP8, 4×GB200, MTP=1) the same redesign moved TPOT only −6.3%, because that workload was not host-bound. The two numbers together *are* the crossover: enormous on the tiny model, marginal on the big one. `Source: cited (vLLM MRv2, 2026-03-24)`.

**vLLM torch.compile — the fusion wins.** The [torch.compile integration post](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20) reports SiLU+Quant FP8 fusion **up to 8%** (Llama-3.1-405B FP8, 8×MI300), AllReduce+RMSNorm fusion **up to 15%**, and a TorchBench geomean of **1.8–2×** across 80+ models — and names the costs plainly: startup time is a pain for autoscaling, and many of the torch.compile APIs are private. It is the clearest public account of both the win and the price. `Source: cited (vLLM torch.compile, 2025-08-20)`.

**vLLM Triton attention backend — why attention resists capture.** The [Triton backend deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) states that variable launch grids "replay badly under CUDA graphs" and describes a persistent-kernel variant with a *fixed* number of instances reading work metadata from GPU memory as the fix — the concrete engineering that makes even attention capturable. It also reports the Triton attention backend hitting 100.7% of FlashAttention-3 on an H100 (Llama-3.1-8B, batch 1, 500-token input, long decode), which matters here because batch-1 long-decode is the launch-bound regime where graphs and a capturable attention kernel pay off most. `Source: cited (vLLM Triton backend, 2026-03-04)`.

**vLLM V1 — the production split.** The [V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) confirms the whole design as shipped: torch.compile plus piecewise CUDA graphs for the steady-state path, FlashAttention 3 for the mixed prefill-plus-decode dynamism graphs cannot capture, reported at up to 1.7× throughput over V0 on Llama-3.1-8B / 3.3-70B (ShareGPT). This is the reference architecture `nanoserve`'s hand-rolled version approximates. `Source: cited (vLLM V1, 2025-01-27)`.

---

## 11. When to reach for this (and when not)

![A decision tree routing a step by whether it is prefill or decode, whether the batch size is bucketed, and whether the attention region is safe to capture](/imgs/blogs/cuda-graphs-and-torch-compile-for-the-decode-loop-7.webp)

The tree above is the routing every real engine does per step, and it is the decision to internalize: prefill runs eager, a bucketed decode with capturable attention replays a full graph, and an uncapturable attention region drops just that one region to piecewise. Read as a recommendation:

**Reach for CUDA graphs when** your `nsys` trace shows white space between kernels on the GPU row and the model is small or the GPU is fast — the launch-bound regime, where $K \cdot L \gt \text{weight bytes} / \text{bandwidth}$. Draft models for [speculative decoding](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), sub-1B routers and classifiers, anything you serve at batch 1 on Hopper or Blackwell — these are where graphs are the primary win, worth tens of percent to nearly 2×.

**Reach for torch.compile(reduce-overhead) first**, in general, because it fuses *and* graphs in five lines and gets the piecewise split right automatically. Fall back to hand-rolled `CUDAGraph` only when you need explicit control over which tensors stay outside the graph — the paged-KV-and-block-table case — or when the compile startup time is unacceptable for your autoscaling and you would rather warm a hand-rolled graph in seconds than compile in tens of seconds.

**Do not bother when** the model is large and the batch is big — you are bandwidth-bound or compute-bound, the launches already hide behind the compute, and graphs buy you noise. Spend that effort on quantization instead. And do not bother when your shapes genuinely cannot be bucketed (a workload with wildly variable structure per step); the recapture storms will cost you more than the launches.

**And the honest ceiling: use vLLM or SGLang instead of your own graphs in production.** They ship the piecewise-CUDA-graph-plus-torch.compile path, tuned across dozens of models and both vendors' GPUs, with the persistent-kernel attention that makes even the dynamic regions capturable and the shared memory pools that keep the KV-cache cost bounded. `nanoserve`'s version exists so you understand what they are doing and can debug it when it misbehaves — so you can read an `nsys` trace and say "that white space is launch overhead, this is a mis-sized bucket, that p99 spike is a lazy capture" — not so you should hand-roll graphs for a production endpoint. Build it to see it; run theirs to serve it.

---

## 12. Key takeaways

- **Decode has two walls, not one.** Bytes (memory-bound) and launches (host-bound). Quantization moves the first; CUDA graphs and torch.compile move the second. Profile to learn which is your max before you optimize.
- **You are launch-bound when $K \cdot L \gt \text{weight bytes} / \text{bandwidth}$** — small models and fast GPUs, reached from both directions. This is exactly where quantization stops helping and graphs start.
- **A CUDA graph records a launch sequence and replays it with one host launch.** N launches become 1; the idle gaps between kernels close. The price is static shapes and static addresses.
- **Decode fits graph capture perfectly** (fixed batch slot, one token per step, fixed KV tensor); **prefill does not** (variable length). Capture decode, run prefill eager or chunked.
- **Bucket by batch size.** One graph per captured size, pad the running set up to the nearest, slice the logits back down. The running-set size is the bucketing key; the cost is a padding tax and pinned VRAM out of the KV budget.
- **torch.compile fuses the small kernels so there are fewer to launch**, then reduce-overhead mode wraps them in CUDA graphs. Piecewise graphs split graph-safe (MLP, norms) from graph-unsafe (dynamic attention) and run each accordingly.
- **The block table is a static tensor whose contents change.** The KV cache and block table keep fixed addresses; you overwrite the block-table integers in place each step. Rebind either and the graph gathers garbage — a bug that looks like quality decay with sequence length.
- **Graphs and quantization compose because they move different terms in the step-time max.** Whichever is your max is the technique you need; the win compounds only when you apply both and re-profile after each.
- **Measure with CUDA events and a sync, at steady state, per bucket.** Host wall-clock timers flatter graphs specifically, because graphs change exactly the async-submission behavior a host timer mismeasures.
- **In production, run vLLM or SGLang.** Build `nanoserve`'s version to understand and debug the launch-bound regime; do not hand-roll graphs for a real endpoint.

---

## Further reading

- vLLM — [torch.compile Integration with vLLM](https://vllm.ai/blog/2025-08-20-torch-compile) (2025-08-20): the two-stage JIT, piecewise CUDA graphs, the fusion numbers, and the flags (`-O0`, `compile_sizes`, the compile cache, `VLLM_DISABLE_COMPILE_CACHE`).
- vLLM — [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24): the host-overhead redesign and the +56% launch-bound result on Qwen3-0.6B.
- vLLM — [Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04): why variable launch grids replay badly under CUDA graphs, and the persistent-kernel fix.
- vLLM — [V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27): torch.compile plus piecewise CUDA graphs as the shipped steady-state path.
- NVIDIA — [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/): the per-launch host cost that motivates the whole feature.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), [the inference kernel landscape](/blog/machine-learning/inference-engineering/the-inference-kernel-landscape-what-actually-runs), [writing a continuous-batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop), [Triton for inference kernels](/blog/machine-learning/inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda), and the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) capstone.
- Sibling depth on the same tools: [CUDA graphs in a serving loop](/blog/machine-learning/performance-engineering/cuda-graphs-in-a-serving-loop) and [what torch.compile actually does](/blog/machine-learning/performance-engineering/what-torch-compile-actually-does).
