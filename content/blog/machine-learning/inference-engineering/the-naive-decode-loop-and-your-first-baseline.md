---
title: "The naive decode loop and your first baseline: measuring before you optimize"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Write the simplest possible generation loop on top of your hand-built forward pass, prove why it is catastrophically slow, derive the decode floor from HBM bandwidth, and build the benchmark harness that keeps every later number in this series honest."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "benchmarking",
    "latency",
    "throughput",
    "kv-cache",
    "pytorch",
    "gpu",
    "ml-systems",
    "roofline",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
---

You now have a forward pass you wrote yourself and a tokenizer boundary you control. Put them together and you get a language model that can produce exactly one token. To get the second token you have to call the model again — and that innocent little `for` loop is where every interesting problem in LLM serving is hiding.

Here is the shape of the trouble. The loop you are about to write processes a 1,000-token prompt, generates 200 tokens, and in doing so pushes **219,900 token positions** through a 32-layer transformer. The same job done properly needs **1,199**. That is a factor of 183 in wasted arithmetic, and it comes from a single missing data structure. Meanwhile, the part of the loop that *is* necessary — the 199 single-token steps — runs at maybe 0.6% of the GPU's arithmetic capability, because each of those steps has to drag all 16.1 GB of Llama-3.1-8B's weights across HBM to multiply them against one vector. Two completely different pathologies, stacked on top of each other, in twelve lines of Python.

![A single decode step where the forward pass produces logits for every position but only the final row is used by the sampler](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-1.webp)

This post writes that loop anyway — deliberately, in its most naive form — and then does the thing that actually matters: it measures it properly. By the end you will have `nanoserve/generate.py` (the loop) and `nanoserve/bench.py` (the harness), a derived performance floor for four named GPUs, precise definitions of TTFT, TPOT, end-to-end latency and goodput, and a set of habits that stop you from fooling yourself with numbers. Every optimization in the remaining 60 posts of this series is measured against this baseline. If the baseline is a lie, so is everything built on it.

One promise up front, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is either derived from arithmetic I show you, cited from a vendor spec or paper with a link, or framed as a range you should expect when you run the script yourself. The results table carries a `Source` column for exactly this reason. That discipline is not decoration — it is the whole point of a baseline post.

---

## 1. The loop, in its most honest form

The generation loop is the simplest algorithm in machine learning and it fits in a paragraph of English. Feed the prompt through the model. Take the logits at the last position. Pick a token. Append it to the sequence. Feed the whole thing back through. Stop when the model emits an end-of-sequence token or you hit a budget.

Here it is as real code, dropped into `nanoserve/generate.py` on top of the model from [the hand-written forward pass](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) and the tokenizer from [the tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization):

```python
# nanoserve/generate.py
import torch


@torch.inference_mode()
def generate_naive(model, input_ids, max_new_tokens=200, eos_id=None):
    """The smallest correct decode loop: no cache, no sampling, no batching.

    input_ids: LongTensor of shape [1, T] already on the model's device.
    Returns (full_ids, new_ids) where new_ids excludes the prompt.
    """
    ids = input_ids
    new_ids = []

    for _ in range(max_new_tokens):
        logits = model(ids)                       # [1, T, vocab] — full recompute
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)

        token = int(next_id)                      # host sync — see below
        if eos_id is not None and token == eos_id:
            break
        new_ids.append(token)

    return ids, new_ids
```

Twelve lines of body. It is correct, it is complete, and it will produce exactly the same tokens as `transformers`' `model.generate(..., do_sample=False)` given the same weights — which is precisely the property that makes it a usable reference. Four things in it deserve more attention than they usually get.

### `torch.inference_mode()` and why it is not `no_grad()`

`torch.no_grad()` stops autograd from recording operations, but tensors created inside it still carry a version counter and still participate in view tracking, because PyTorch has to stay ready in case you hand one of them back into a graph later. `torch.inference_mode()` goes further: it turns off the version counter and view-tracking machinery entirely, which makes tensor allocation and every in-place operation slightly cheaper. The trade is that a tensor produced inside inference mode cannot later be recorded by autograd — try it and you get a runtime error. For a serving path that is exactly the deal you want. The [PyTorch autograd mechanics documentation](https://pytorch.org/docs/stable/notes/autograd.html) spells out the difference; use `inference_mode` everywhere in `nanoserve` and never think about it again.

The saving is small in absolute terms — we are talking microseconds of CPU work per operation — but decode steps are short. On an A100 a batch-1 decode step for an 8B model has a floor around 7.9 ms (we derive that in section 4), and a 32-layer model issues on the order of 200–300 kernel launches per step. Anything that shaves per-op CPU overhead matters when the CPU is trying to stay ahead of the GPU, which during decode it barely is. This is the thread that eventually leads to CUDA graphs.

### `logits[:, -1, :]` is where most of the waste lives

The model returns logits for **every** position in the sequence: shape `[1, T, 128256]` for Llama-3.1-8B's vocabulary. We use one row. At T = 1,000 that tensor is 1,000 × 128,256 × 2 bytes = **256 MB** of pure allocation, produced so we can index into 256 KB of it.

Worse, producing it costs real FLOPs. The language-model head is a matmul from hidden size 4,096 to vocabulary 128,256, so per position it costs $2 \times 4096 \times 128256 \approx 1.05$ GFLOP. Doing that at all 1,000 positions costs about **1.05 TFLOP**; doing it at the one position we need costs 1.05 GFLOP. For context, a full forward pass over one token for this model is roughly $2N = 2 \times 8.03 \times 10^{9} \approx 16.1$ GFLOP, so the head alone accounts for about 6.5% of a single-token pass — and the naive loop inflates that one operation by a factor of a thousand.

The fix is one line and it does not need a cache:

```python
# nanoserve/model.py — inside Llama.forward, just before the LM head
if self.slice_last_only and not self.training:
    h = h[:, -1:, :]          # [B, 1, d_model] — keep the graph shape, drop the rest
logits = self.lm_head(h)      # [B, 1, vocab] instead of [B, T, vocab]
```

Do this and `generate_naive` changes its indexing from `logits[:, -1, :]` to `logits[:, 0, :]`, or you keep the `-1` and it still works because the sequence dimension is now length one. Real engines do exactly this, and they generalize it: during a batched prefill they gather only the *last* position of each sequence in the batch before the head, using an index tensor of sampling positions. It is the first optimization in this series and it costs nothing.

I am pointing at it early because it makes a broader point. There are two categories of inefficiency in the naive loop: **redundant work you can delete for free** (the head over all positions) and **redundant work that requires a new data structure** (recomputing attention over the prefix). The first kind is embarrassing but easy. The second kind is the KV cache, and it is the subject of the entire next track.

### `int(next_id)` is a synchronization point

That innocuous cast copies one integer from device memory to host memory, and to do so it must wait for every kernel queued ahead of it to finish. CUDA operations are asynchronous: when Python returns from `model(ids)`, the GPU has probably not started, let alone finished. The call to `int()` (equivalently `.item()`, `.tolist()`, `.cpu()`, or a Python `if` on a tensor) forces a full pipeline drain.

In the naive loop this is nearly free, because we were going to wait for the result anyway to build the next input. But it becomes a real problem the moment you want to overlap CPU work with GPU work, and it is the single biggest reason a naive decode loop can be *host-bound* rather than GPU-bound: the CPU spends its time blocked in a sync instead of racing ahead to enqueue the next step's 300 kernel launches. The eventual fixes — keeping stop conditions on-device, sampling on the GPU, capturing the step in a CUDA graph — are Track E and Track F material. For now, know that it is there, and know that it is one reason your measured tok/s will sit below the derived floor rather than at it.

### Greedy `argmax` is the only sampler here, on purpose

`argmax` picks the highest-scoring token. It is deterministic given deterministic kernels, which makes it the right choice for a baseline: two runs should produce the same text, so if the text changes you have a bug, not a temperature. Temperature, top-k, top-p, min-p, repetition penalties, and everything else that lives between logits and tokens is Track D's territory. Introducing sampling now would mean introducing an RNG, a seed, and a whole class of "the same prompt gave different output" questions into a post whose job is to establish a stable measuring stick.

There is one honest caveat even for greedy decoding: **greedy is not bit-reproducible across batch sizes on real hardware.** Floating-point reductions are not associative, and a matmul with a different batch dimension may tile and reduce in a different order, changing the last bits of a logit and occasionally flipping which of two near-tied tokens wins. That is a real phenomenon and it gets its own post in Track D. Just do not be surprised when your batch-1 and batch-8 outputs diverge after 40 tokens.

### Driving it

```python
# scripts/run_naive.py
import torch
from nanoserve.model import Llama
from nanoserve.tokenizer import Tokenizer
from nanoserve.generate import generate_naive

model = Llama.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                              dtype=torch.bfloat16, device="cuda").eval()
tok = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain the KV cache in two paragraphs."}],
    add_generation_prompt=True,
)
ids = torch.tensor([prompt], device="cuda")

full, new = generate_naive(model, ids, max_new_tokens=200, eos_id=tok.eos_id)
print(tok.decode(new))
```

That is a working, if pitiful, inference engine. It has no cache, no batching, no scheduler, no streaming, no cancellation, and no way to serve two users. It is also — and this is the point — **correct**, which none of the fast versions will be until you have something to check them against.

---

## 2. Counting the work you throw away

The most useful thing you can do before touching a profiler is arithmetic. The naive loop's central sin is visible without a GPU, without a timer, and without any measurement at all: it is a property of the algorithm.

<figure class="blog-anim">
<svg viewBox="0 0 680 220" role="img" aria-label="A cacheless decode loop rereads an ever longer prefix; the highlighted region grows by one token on every step" style="width:100%;height:auto;max-width:820px">
<style>
.n1-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.n1-new{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2}
.n1-band{fill:var(--accent,#6366f1);opacity:.20;transform-box:fill-box;transform-origin:left center}
.n1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.n1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.n1-tick{font:500 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes n1-grow{0%,20%{transform:scaleX(.667)}25%,45%{transform:scaleX(.750)}50%,70%{transform:scaleX(.833)}75%,97%{transform:scaleX(.917)}100%{transform:scaleX(.667)}}
@keyframes n1-c1{0%,22%{opacity:.10}27%,100%{opacity:1}}
@keyframes n1-c2{0%,47%{opacity:.10}52%,100%{opacity:1}}
@keyframes n1-c3{0%,72%{opacity:.10}77%,100%{opacity:1}}
@keyframes n1-c4{0%,96%{opacity:.10}99%,100%{opacity:1}}
.n1-band{animation:n1-grow 10s steps(1,end) infinite}
.n1-p1{animation:n1-c1 10s steps(1,end) infinite}
.n1-p2{animation:n1-c2 10s steps(1,end) infinite}
.n1-p3{animation:n1-c3 10s steps(1,end) infinite}
.n1-p4{animation:n1-c4 10s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.n1-band{animation:none;transform:scaleX(.917)}.n1-p1,.n1-p2,.n1-p3,.n1-p4{animation:none;opacity:1}}
</style>
<text class="n1-sub" x="222" y="34">prompt · 8 tokens, already known</text>
<text class="n1-sub" x="530" y="34">generated one at a time</text>
<rect class="n1-band" x="30" y="58" width="594" height="54" rx="8"/>
<rect class="n1-cell" x="30"  y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="80"  y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="130" y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="180" y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="230" y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="280" y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="330" y="62" width="44" height="46" rx="6"/>
<rect class="n1-cell" x="380" y="62" width="44" height="46" rx="6"/>
<rect class="n1-new n1-p1" x="430" y="62" width="44" height="46" rx="6"/>
<rect class="n1-new n1-p2" x="480" y="62" width="44" height="46" rx="6"/>
<rect class="n1-new n1-p3" x="530" y="62" width="44" height="46" rx="6"/>
<rect class="n1-new n1-p4" x="580" y="62" width="44" height="46" rx="6"/>
<text class="n1-tick" x="52"  y="132">1</text>
<text class="n1-tick" x="202" y="132">4</text>
<text class="n1-tick" x="402" y="132">8</text>
<text class="n1-tick" x="602" y="132">12</text>
<text class="n1-lbl" x="340" y="172">the shaded region is re-read on every single step</text>
<text class="n1-sub" x="340" y="196">8 positions, then 9, then 10, then 11 — nothing is remembered</text>
</svg>
<figcaption>Without a cache each decode step reprocesses the entire prefix, so the shaded work region grows by one token every step instead of staying at one.</figcaption>
</figure>

### The derivation

Let $p$ be the prompt length in tokens and $n$ the number of tokens to generate. The naive loop makes $n$ forward passes, and the $i$-th of them (for $i = 0, 1, \ldots, n-1$) processes a sequence of length $p + i$. Count *token positions pushed through the network* — a clean, hardware-independent unit of work:

$$
W_{\text{naive}}(p, n) \;=\; \sum_{i=0}^{n-1} (p + i) \;=\; n\,p \;+\; \frac{n(n-1)}{2}.
$$

The second term is the quadratic one. It is quadratic in the number of *generated* tokens, and it is there because the loop has no memory: every step reconstructs from scratch the internal state it already built one step earlier.

Now count the same job for an engine that keeps a cache. The prompt is processed once, in a single parallel pass over $p$ positions. Each subsequent step processes exactly one new position, and there are $n - 1$ of them (the prompt pass already produced the first generated token):

$$
W_{\text{cached}}(p, n) \;=\; p \;+\; (n - 1).
$$

The ratio is the whole story:

$$
R(p, n) \;=\; \frac{W_{\text{naive}}}{W_{\text{cached}}} \;=\; \frac{n\,p + \tfrac{n(n-1)}{2}}{p + n - 1}.
$$

#### Worked example: a 1,000-token prompt and 200 generated tokens

Take the RAG-shaped workload from the series prompt suite — a long retrieved context and a short answer. Set $p = 1000$, $n = 200$.

$$
W_{\text{naive}} = 200 \times 1000 + \frac{200 \times 199}{2} = 200{,}000 + 19{,}900 = 219{,}900 \text{ positions.}
$$

$$
W_{\text{cached}} = 1000 + 199 = 1{,}199 \text{ positions.}
$$

$$
R = \frac{219{,}900}{1{,}199} \approx 183.
$$

**The naive loop does 183 times the token-position work of a correct one.** *(Source: derived — pure arithmetic, no hardware involved.)*

If you isolate just the decoding phase and ask "how much extra work did the 199 decode steps cost?", the number is uglier still: 219,900 positions to produce what 199 positions should have produced, a factor of about **1,105**.

And that is the *charitable* accounting, because it counts positions as if all work were linear in sequence length. Attention is not. In a naive pass over a length-$t$ sequence, the score matrix is $t \times t$, so attention FLOPs scale as $t^2$ per layer. Summing over the loop:

$$
\sum_{t=p}^{p+n-1} t^2 \;\approx\; \frac{(p+n)^3 - p^3}{3} \;=\; \frac{1199^3 - 999^3}{3} \approx 2.42 \times 10^{8},
$$

against roughly $p^2/2 + (n-1)\cdot p \approx 7.0 \times 10^{5}$ position-pairs for the cached version. That is a factor of about **345** on the attention term specifically. *(Source: derived.)* For an 8B model at 1k context the attention term is not what dominates wall clock — the dense projections and MLP are — but at 32k or 128k context it very much is, which is why long-context prefill gets its own post later in the series.

![Work accounting comparing a cacheless decode loop against a cached one for the same 200-token generation](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-2.webp)

### Prove it to yourself without a GPU

You do not need hardware to confirm the accounting. Instrument the loop with a counter and run it on CPU with a tiny random model, or run the counter alone:

```python
# nanoserve/workcount.py
def naive_positions(prompt_len: int, n_new: int) -> int:
    return sum(prompt_len + i for i in range(n_new))


def cached_positions(prompt_len: int, n_new: int) -> int:
    return prompt_len + (n_new - 1)


if __name__ == "__main__":
    rows = [(1000, 200), (1000, 1000), (128, 512), (32000, 200), (8000, 2000)]
    print(f"{'prompt':>7} {'new':>6} {'naive':>12} {'cached':>8} {'ratio':>8}")
    for p, n in rows:
        a, b = naive_positions(p, n), cached_positions(p, n)
        print(f"{p:>7} {n:>6} {a:>12,} {b:>8,} {a / b:>7.1f}x")
```

Running that prints a table you can verify by hand — it is arithmetic, not measurement:

```console
 prompt    new        naive   cached    ratio
   1000    200      219,900    1,199    183.4x
   1000   1000    1,499,500    1,999    750.1x
    128    512      258,816      639    405.0x
  32000    200    6,419,900   32,199    199.4x
   8000   2000    9,999,000    9,999   1000.0x
```

Two patterns jump out of that table, and both matter for the rest of the series.

**The waste scales with output length, not just input length.** A chat workload with a short prompt and a long answer (128 in, 512 out) is punished harder in relative terms than a RAG workload with a long prompt and a short answer (32,000 in, 200 out). This is counterintuitive if you think of long contexts as "the expensive case," and it is the reason chat and RAG traffic need different engine tuning.

**The ratio grows without bound.** There is no prompt length at which the naive loop becomes acceptable; it just becomes differently bad. Any system that generates thousands of tokens — an agent loop, a reasoning model with a long thinking budget, a code generator — is in the worst regime.

### The one honest defense of the naive loop

It is not entirely worthless. Two things it gives you that a cached implementation does not:

1. **A correctness oracle.** A naive loop has no cache invalidation, no block table, no position-offset bookkeeping. When your paged-attention kernel starts producing subtly wrong tokens after position 4,096, the naive loop is what you diff against. Keep it in the repo forever, behind a `--reference` flag, and make it part of the test suite.
2. **A ceiling on how wrong your cache can be.** If cached and naive generation diverge on the same greedy prompt, one of them is broken. Divergence beyond floating-point noise is a bug, and this is the cheapest bug detector you will ever write.

That is exactly how `nanoserve` will use it. The naive loop is not deleted in post 6 — it is demoted to a test fixture.

---

## 3. Two phases, two entirely different machines

The single most important structural fact about LLM inference is that a request is not one workload. It is two, and they stress different parts of the hardware so completely that treating them as one thing is the root cause of most bad serving decisions.

![A request timeline showing one wide compute-bound prefill followed by a long train of narrow bandwidth-bound decode steps](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-3.webp)

### Prefill: one wide, parallel, compute-bound pass

The prompt's tokens are all known in advance. Nothing about token 500 depends on token 501, so all $p$ positions go through the network **together**, in one pass. Every weight matrix multiplication is then a matrix-matrix product — a GEMM — with an $M$ dimension of $p$. For $p = 1000$ that is a big, well-shaped GEMM: exactly what tensor cores were designed for.

The FLOP count is easy. Using the standard ${2N}$ estimate for a dense forward pass (two FLOPs — a multiply and an add — per parameter per token):

$$
\text{FLOPs}_{\text{prefill}} \approx 2 N p = 2 \times 8.03\times10^{9} \times 1000 \approx 1.61 \times 10^{13} = 16.1\ \text{TFLOP}.
$$

*(Source: derived, using the 8.03 B parameter count from the [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B).)* The estimate ignores attention's own FLOPs, which at 1k context add a few percent; at 32k they add a lot, and you should stop ignoring them.

The corresponding memory traffic is roughly one read of the weights — 16.1 GB — plus activations. So the **arithmetic intensity** of prefill, meaning FLOPs performed per byte moved, is on the order of

$$
\text{AI}_{\text{prefill}} \approx \frac{2 N p}{2 N} = p \approx 1000\ \frac{\text{FLOP}}{\text{byte}}.
$$

A thousand FLOPs per byte is a compute-bound regime on every accelerator you can buy. That is the definition of compute-bound and it is exactly what the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) predicts. Prefill is the part of inference where a GPU's advertised TFLOPS number means something.

### Decode: many narrow, serial, bandwidth-bound steps

Now the second phase. Token $p+1$ cannot be computed before token $p$ exists. The sequential dependency is absolute — no amount of hardware removes it. So each step processes exactly one new position, and each weight matrix multiplication becomes a matrix-**vector** product, a GEMV, with $M = 1$.

The FLOP count per step is $2N \approx 16.1$ GFLOP. The memory traffic per step is, again, a full read of the weights: 16.1 GB in bf16. So:

$$
\text{AI}_{\text{decode}} \approx \frac{2N}{2N} = 1\ \frac{\text{FLOP}}{\text{byte}}.
$$

**One FLOP per byte.** Every byte you pull out of HBM gets used for a single multiply-accumulate and is then thrown away. There is no reuse, because there is nothing to reuse it against: the "matrix" on the other side of the multiply is a single vector.

That single number — arithmetic intensity of one — explains essentially every design decision in every LLM serving system that exists. Hold onto it.

### The physical picture

During prefill, the GPU's tensor cores are busy and its memory system is comfortably feeding them. During decode, the memory system is saturated and the tensor cores are almost entirely idle — they finish their tiny GEMV and then wait for the next slab of weights to arrive. The GPU is not "slow" during decode; it is *starving*. Watching `nvidia-smi` during decode shows high "GPU utilization," which is one of the most misleading numbers in the entire stack: that counter reports whether *any* kernel was resident, not whether the machine was doing useful arithmetic. A GEMV that keeps one SM busy waiting on HBM reports as 100% utilized.

| Property | Prefill | Decode |
| --- | --- | --- |
| Shape of the matmuls | GEMM, $M = p$ | GEMV, $M = 1$ |
| Parallelism available | all $p$ positions at once | one position, strictly serial |
| Arithmetic intensity | $\approx p$ FLOP/byte | $\approx 1$ FLOP/byte (per sequence) |
| Bottleneck | tensor-core throughput | HBM bandwidth |
| What makes it faster | better kernels, higher clocks | fewer bytes, or more sequences |
| Latency metric it owns | TTFT | TPOT |
| Effect of batching | already saturated, batching adds little | close to linear throughput gain |

That last row is the punchline of this post, and section 5 derives it. But notice what the table implies: **the two phases want opposite things from a scheduler.** Prefill wants the GPU to itself so it finishes fast and TTFT stays low. Decode wants as many concurrent sequences as possible so each byte of weight traffic gets amortized. Reconciling those two desires is what chunked prefill, continuous batching, and prefill/decode disaggregation are all about; the repo's existing post on [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) covers the extreme version where you put them on different machines.

### Where the KV cache sits in this accounting

I keep saying "decode reads the weights." It also reads the KV cache, and it is worth knowing when that term stops being negligible. For Llama-3.1-8B with grouped-query attention — 32 layers, 8 key/value heads, head dimension 128, bf16 — the cache costs

$$
2 \times L \times H_{kv} \times d_{\text{head}} \times b = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB per token.}
$$

*(Source: derived from the model config; the general formula and its consequences get a whole post in Track B, and the repo's [KV cache primer](/blog/machine-learning/large-language-model/kv-cache) covers the concept.)*

At our 1,200-token working example, one sequence's cache is $1200 \times 128\ \text{KB} \approx 157$ MB — about 1% of the 16.1 GB of weights. Utterly negligible; the decode floor is a weights story. But at batch 64 with 8k of context each, it is $64 \times 8192 \times 128\ \text{KB} \approx 68.7$ GB, which does not fit alongside the weights on an 80 GB card at all. The cache goes from a rounding error to the dominant term as you scale exactly the knob (batch size) that section 5 says you must scale. That tension is the entire reason PagedAttention exists.

---

## 4. The decode floor: weight bytes over bandwidth

Here is the derivation that should be the first thing you reach for when someone asks "how fast can this model possibly go?"

![A derivation ladder from weight bytes and HBM bandwidth down to the per-step time floor and the idle tensor-core fraction](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-4.webp)

### The argument

To compute one decode step, every weight in the model must be read from HBM into the SMs at least once. There is no way around this: each parameter participates in the computation, and there is nowhere on the chip to keep 16.1 GB between steps (an H100 has 50 MB of L2). Therefore the time for one decode step is bounded below by the time to move the weights:

$$
t_{\text{step}} \;\gtrsim\; \frac{B_{\text{weights}}}{\text{BW}_{\text{HBM}}},
$$

and the batch-1 token rate is bounded above by its reciprocal:

$$
\text{tok/s}_{\max} \;\lesssim\; \frac{\text{BW}_{\text{HBM}}}{B_{\text{weights}}}.
$$

The bound holds regardless of how good your kernels are, what precision your tensor cores support, or how clever your attention implementation is. It is a statement about physics and about the model's size, and it is the most useful sanity check in inference engineering. The [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) has more on why HBM is the wall it is.

#### Worked example: Llama-3.1-8B in bf16 on a 4090 and an A100

**Weight bytes.** The [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B) gives 8.03 B parameters. In bf16 that is

$$
B_{\text{weights}} = 8.03\times10^{9} \times 2\ \text{bytes} = 1.606\times10^{10}\ \text{bytes} = 16.06\ \text{GB}.
$$

**RTX 4090.** NVIDIA's [Ada Lovelace GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) lists 1,008 GB/s of memory bandwidth for the RTX 4090.

$$
t_{\text{step}} \gtrsim \frac{16.06\ \text{GB}}{1.008\ \text{TB/s}} = 15.9\ \text{ms} \quad\Longrightarrow\quad \text{tok/s}_{\max} \approx 63.
$$

**A100 80GB SXM.** NVIDIA's [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) lists 2,039 GB/s.

$$
t_{\text{step}} \gtrsim \frac{16.06\ \text{GB}}{2.039\ \text{TB/s}} = 7.9\ \text{ms} \quad\Longrightarrow\quad \text{tok/s}_{\max} \approx 127.
$$

*(Source: derived, from the cited parameter count and the cited bandwidth specs.)*

Notice how little the two GPUs differ in *kind*. The A100 is roughly twice as fast at batch-1 decode as a consumer card that costs a fraction as much — because the only thing that matters here is bandwidth, and the A100 has roughly twice as much. Its 312 TFLOPS of BF16 tensor-core throughput, which is where most of its price lives, contributes nothing at batch 1.

### The compute side, for contrast

Compute time for one decode step is the FLOPs divided by the achievable rate. Using the peak dense BF16 tensor-core numbers from the vendor specs:

- **RTX 4090:** the Ada whitepaper lists 165.2 TFLOPS for FP16/BF16 tensor operations with FP32 accumulate, dense (the whitepaper's larger figures are with structured sparsity, and GeForce Ada halves the FP32-accumulate rate relative to FP16-accumulate — 165.2 is the number that matches what a PyTorch `bf16` matmul actually gets).
  $$16.06\times10^{9} / 165.2\times10^{12} = 0.097\ \text{ms}.$$
- **A100 80GB:** 312 TFLOPS dense BF16 per the datasheet.
  $$16.06\times10^{9} / 312\times10^{12} = 0.051\ \text{ms}.$$
- **H100 80GB SXM:** NVIDIA's [H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) lists 1,979 TFLOPS BF16 *with sparsity*; dense is 989 TFLOPS.
  $$16.06\times10^{9} / 989\times10^{12} = 0.016\ \text{ms}.$$

Line those up against the memory floors and the picture is stark. On an A100, a decode step needs **7.9 ms of memory time and 0.051 ms of compute time**. The tensor cores are doing useful work for

$$
\frac{0.051}{7.876} = 0.65\% \text{ of the step; they are idle } 99.35\% \text{ of the time.}
$$

*(Source: derived from the cited specs.)*

This is not a bug in your code. It is not a bad kernel. It is what batch-1 autoregressive decoding *is*. If you have ever profiled a decode loop, seen near-zero tensor-core activity, and gone looking for the bug, this derivation is the answer: there was no bug.

### Why the real number will be lower than the floor

Your measured tok/s will not hit the derived ceiling, and you should expect a gap of 20–50%. The reasons, roughly in order of size:

- **Kernels do not achieve peak bandwidth.** A well-tuned GEMV reaches maybe 70–90% of peak HBM bandwidth; a mediocre one much less. Peak is a marketing number measured by a streaming benchmark, not by a real kernel with indexed loads.
- **You read more than the weights.** The KV cache, activations, the logits tensor, RMSNorm statistics, RoPE tables — all real traffic.
- **Kernel launch overhead.** A 32-layer model issues hundreds of kernels per step. At a few microseconds of launch latency each, on a step whose floor is 7.9 ms, that can be a meaningful fraction — and on a *faster* GPU where the floor is 4.8 ms, a larger one. This is why CUDA graphs matter more on an H100 than on an L4.
- **The host sync per step.** The `int(next_id)` discussed earlier drains the pipeline every iteration.
- **Python.** The naive loop runs its control flow in the interpreter between GPU steps.

So the honest framing for a reader with hardware: **on an RTX 4090 running Llama-3.1-8B in bf16 at batch 1, expect somewhere in the region of 40–60 tok/s once the cache is in place; the derived ceiling is 63.** Run `bench.py` from section 9 and see where you land. If you are *above* the derived ceiling, you have a bug in your measurement — most likely a missing synchronization — and section 7 will tell you which one.

---

## 5. The one fact that explains every serving system

Take the two rates from section 4 and form their ratio. That ratio has a name: **machine balance**, the arithmetic intensity at which a device transitions from memory-bound to compute-bound.

$$
\text{balance} = \frac{\text{peak FLOP/s}}{\text{peak bytes/s}}.
$$

For the four GPUs in this series' hardware matrix:

$$
\text{A100: } \frac{312\times10^{12}}{2.039\times10^{12}} \approx 153\ \frac{\text{FLOP}}{\text{byte}}, \qquad
\text{4090: } \frac{165.2\times10^{12}}{1.008\times10^{12}} \approx 164,
$$

$$
\text{H100: } \frac{989\times10^{12}}{3.35\times10^{12}} \approx 295, \qquad
\text{L4: } \frac{60.5\times10^{12}}{0.300\times10^{12}} \approx 202.
$$

*(Source: derived from the cited datasheets; the [L4 datasheet](https://resources.nvidia.com/en-us-l4/l4-datasheet) lists 300 GB/s and 121 TFLOPS BF16 with sparsity, so 60.5 dense.)*

Now the key step. Batch $B$ sequences together and run one decode step for all of them. The weights are read **once** and used $B$ times. So:

$$
\text{AI}_{\text{decode}}(B) = \frac{2NB}{2N} = B\ \frac{\text{FLOP}}{\text{byte}}.
$$

Arithmetic intensity is *equal to the batch size*. Which means the transition point — the batch at which decode stops being bandwidth-bound and starts being compute-bound — is simply the machine balance. On an A100, that is a batch of about **153**. Below it, the step time barely changes as you add sequences, because the dominant cost (reading the weights) is fixed.

![A comparison of four GPUs showing bandwidth, per-step decode floor, token-rate ceiling, and the batch size needed to become compute bound](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-5.webp)

### What "batching is nearly free" actually means

Say it precisely, because the loose version is wrong. **Per-step latency is nearly flat in batch size up to the machine balance; therefore throughput is nearly linear in batch size over that range.** Concretely, on an A100:

| Batch | Weight bytes read | Per-step time (floor) | Aggregate tok/s (ceiling) | Per-request tok/s |
| --- | --- | --- | --- | --- |
| 1 | 16.06 GB | 7.9 ms | 127 | 127 |
| 8 | 16.06 GB | 7.9 ms | 1,016 | 127 |
| 32 | 16.06 GB | 7.9 ms | 4,063 | 127 |
| 128 | 16.06 GB | 7.9 ms | 16,254 | 127 |
| 153 | 16.06 GB | 7.9 ms | 19,424 | 127 |
| 512 | 16.06 GB | ~26.4 ms (compute-bound) | ~19,424 | ~38 |

*(Source: derived. The floor ignores KV-cache traffic and activation traffic, both of which grow with batch and pull the real curve below these ceilings well before 153.)*

Read the third column again. **Sixteen thousand tokens per second versus one hundred and twenty-seven, on the same GPU, at the same per-request speed.** No new hardware, no quantization, no kernel work. Just batching.

This is why every production inference system — vLLM, TGI, TensorRT-LLM, SGLang, all of them — is fundamentally a *batching* system with a scheduler wrapped around it. It is why [continuous batching](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) was a bigger practical win than any kernel optimization of the same era. It is why the KV cache's memory footprint is the central resource-management problem in the field: memory is what limits $B$, and $B$ is what buys throughput.

The chain is short enough to memorize: *decode is bandwidth-bound → batching amortizes the bandwidth → memory limits the batch → therefore memory management is the whole game.*

### The two ceilings you actually hit first

You will never reach a batch of 153 on an 80 GB A100 with long sequences, and the reason is the cache math from section 3. Free memory after weights is $80 - 16.06 \approx 64$ GB. At 128 KB per token:

$$
\text{tokens that fit} = \frac{64\times10^{9}}{131072} \approx 488{,}000.
$$

At 4k context per sequence that is about **119 concurrent sequences**; at 8k about **59**; at 32k about **14**. *(Source: derived.)* So on a realistic chat workload you get most of the way to machine balance, and on a long-context RAG workload you fall far short of it and stay firmly memory-bound. That gap between "the batch you want" and "the batch that fits" is the motivation for KV-cache quantization, for GQA and MLA, and for paged allocation that stops you from wasting the memory you do have.

The second ceiling is latency. Batching raises aggregate throughput but not per-request speed, and past machine balance it actively *hurts* per-request speed (see the last row of the table). A user watching tokens appear does not care about your aggregate. Which brings us to the metrics.

---

## 6. The metrics, defined precisely

Most arguments about inference performance are actually arguments about definitions. Fix them now, use them for 60 more posts.

![How time to first token and time per output token compose into end-to-end latency and then into goodput](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-6.webp)

**TTFT — time to first token.** Wall-clock time from the moment the request is accepted to the moment the first token of the response is emitted to the client. It includes queueing, tokenization, the prefill forward pass, sampling the first token, and detokenizing enough of it to send something. It is dominated by prefill for a long prompt, and by *queue time* under load. This is the "is it alive?" metric: the user is staring at an empty box.

**TPOT — time per output token**, also called inter-token latency (ITL). The average wall-clock gap between consecutive tokens after the first. For a well-behaved engine at steady state it is close to the decode step time; under contention it also contains the time your sequence spent waiting while other sequences' prefills ran. This is the "is it faster than I can read?" metric.

A number worth carrying around: adults read continuous non-fiction prose at roughly 238 words per minute on average, per Brysbaert's [meta-analysis of reading rate](https://doi.org/10.1016/j.jml.2019.104047). At roughly 1.3 tokens per English word, that is about **5 tokens per second**. A TPOT of 20 ms is 50 tok/s — ten times reading speed, which is why chat UIs feel instant at that rate and why pushing TPOT below ~15 ms buys you very little perceptually for a human reader. It buys a great deal for an *agent* consuming the output programmatically, which is a different SLO.

**End-to-end latency.** The composition is exact and worth writing down:

$$
L_{\text{e2e}} = \text{TTFT} + (n - 1)\cdot \text{TPOT},
$$

for a response of $n$ tokens. Everything else follows from this one line.

**Tokens per second, per request** is just $1/\text{TPOT}$. **Aggregate tokens per second** is the sum across all in-flight requests — for a batch of $B$ at steady state, $B/\text{TPOT}$. These two numbers can differ by two orders of magnitude on the same machine at the same instant (see the table in section 5), and quoting one when someone asked about the other is the most common form of benchmark dishonesty in this field. Always say which.

**Goodput.** Requests per second that *meet a stated SLO*, as opposed to requests per second, full stop. If your SLO is "TTFT under 500 ms and TPOT under 30 ms at p95," then a server that completes 40 rps of which 12 violate the SLO has a goodput of 28 rps, not 40. Throughput is the wrong objective function for a serving system because it is trivially maximized by making everyone wait: fill an enormous batch, run it, and report a big number. Goodput is the metric that cannot be gamed that way, and the scheduler post in Track C builds a policy around it.

#### Worked example: two regressions that feel nothing alike

Take a chat response of $n = 200$ tokens with a healthy baseline of TTFT = 250 ms and TPOT = 20 ms.

$$
L_{\text{e2e}} = 250 + 199 \times 20 = 250 + 3980 = 4230\ \text{ms} = 4.23\ \text{s}.
$$

Now regress each metric by 2× and recompute.

| Scenario | TTFT | TPOT | End-to-end | Change | How it feels |
| --- | --- | --- | --- | --- | --- |
| Baseline | 250 ms | 20 ms | 4.23 s | — | fine |
| p99 TTFT doubles | 500 ms | 20 ms | 4.48 s | +5.9% | a beat of hesitation, then normal |
| p99 TPOT doubles | 250 ms | 40 ms | 8.21 s | +94% | visibly, painfully slow throughout |
| Both double | 500 ms | 40 ms | 8.46 s | +100% | broken |

*(Source: derived from the composition formula.)*

The asymmetry is the point. **A TTFT regression costs you a fixed amount once; a TPOT regression costs you that amount 199 times.** They also fail differently: a doubled TTFT reads as "the server hiccuped," while a doubled TPOT reads as "the product got worse." And they have different causes — TTFT regressions usually come from queueing or from a prefill that got longer, TPOT regressions usually come from batch pressure, cache pressure, or a per-token overhead you just added (a logits processor, a grammar mask, a guardrail model).

If you can only alert on one, alert on p99 TPOT. If you can only optimize one for a chat product, optimize TTFT below 500 ms and then never think about it again. For a RAG product where prompts are 8k tokens and answers are 100, invert that advice entirely: TTFT is nearly all of the latency, and TPOT barely registers.

**Report percentiles, not means.** A mean latency over a mixed workload is a number about no user. Report p50 and p99 separately for TTFT and for TPOT, and report them per workload class, because a suite that mixes 128-token and 8,000-token prompts produces a bimodal distribution whose p99 is just "the long prompts" — which tells you nothing you did not already know.

---

## 7. How to benchmark without fooling yourself

This is the most reusable section in the post. The techniques below are not specific to LLM inference; they are what separates a number you can regress against from a number that describes your warmup path.

![A decision tree asking whether a latency number survives warmup, clock locking, and open-loop load before it can be believed](/imgs/blogs/the-naive-decode-loop-and-your-first-baseline-7.webp)

### The first iteration is always a lie

Never, ever include the first iteration. Possibly not the first fifty. Everything that can be lazy in a GPU stack *is* lazy:

- **CUDA context creation** happens on the first device operation and costs hundreds of milliseconds.
- **The caching allocator grows on demand.** PyTorch calls `cudaMalloc` when its pools cannot satisfy a request, and `cudaMalloc` is synchronous and slow. Once your steady-state working set is allocated, the allocator serves from its pools and the cost vanishes. Until then, every new shape you touch can trigger one.
- **cuBLAS and cuDNN initialize handles and workspaces on first use**, and cuBLASLt runs heuristics to pick an algorithm for each new problem shape.
- **`torch.compile` and Triton JIT-compile kernels on first encounter**, which can take *seconds* per graph, and autotuning multiplies that.
- **Clocks are cold.** A GPU idling at low clocks takes a moment to boost.

The consequence: a script that times one generation and prints a number is measuring compilation and allocation, not inference. This is the single most common benchmarking error in ML, and it is usually off by more than an order of magnitude.

```python
# WRONG — measures lazy init, allocator growth, autotuning, and cold clocks
import time
t0 = time.perf_counter()
out = generate(model, ids, max_new_tokens=64)
print(f"{(time.perf_counter() - t0) * 1e3:.1f} ms")   # meaningless
```

```python
# RIGHT — warm up, synchronize, then measure a population
import time, torch

for _ in range(5):                                  # warmup: same shapes as the real run
    generate(model, ids, max_new_tokens=64)
torch.cuda.synchronize()                            # drain everything before the timer

samples = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate(model, ids, max_new_tokens=64)
    torch.cuda.synchronize()                        # the launch returned; the work may not be done
    samples.append((time.perf_counter() - t0) * 1e3)
```

Warm up with the **same shapes** you will measure. Warming up at sequence length 128 and measuring at 8,192 re-triggers allocator growth and algorithm selection, and you are back to measuring initialization.

### Synchronize, or measure nothing

CUDA kernel launches are asynchronous. `time.perf_counter()` around a launch measures how long it took the CPU to *enqueue* the work, which on a warm path is a few microseconds regardless of how much work it was. That is why unsynchronized benchmarks report impossible speeds — and why "I got 900 tok/s at batch 1 on a 4090" is always a missing `synchronize()`.

The rule: **`torch.cuda.synchronize()` immediately before you start the timer and immediately before you stop it.** The first sync ensures prior work is not attributed to your measurement; the second ensures your work is actually finished.

### CUDA events versus the wall clock

Two tools, two purposes, and you want both.

```python
import time, torch

# GPU-side: measures device execution time, minimal CPU perturbation
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out = model(ids)
end.record()
torch.cuda.synchronize()          # required before reading elapsed_time
gpu_ms = start.elapsed_time(end)  # milliseconds, ~0.5 us resolution

# Host-side: measures what the user experiences, including Python and syncs
torch.cuda.synchronize()
t0 = time.perf_counter()
out = model(ids)
torch.cuda.synchronize()
wall_ms = (time.perf_counter() - t0) * 1e3

print(f"gpu {gpu_ms:.2f} ms   wall {wall_ms:.2f} ms   host overhead {wall_ms - gpu_ms:.2f} ms")
```

CUDA events are recorded into the stream and timestamped by the GPU, so they measure device time without the CPU's scheduling noise. Wall clock measures the thing a user actually waits for, including Python, tokenization, the host syncs, and any time the GPU sat idle waiting for the CPU to catch up.

**The difference between them is the most informative number in the whole harness.** If wall time is 12 ms and GPU time is 8 ms on a decode step, you have 4 ms of host overhead per step — a third of your TPOT is Python and launch latency, and no kernel optimization will touch it. That diagnosis points straight at CUDA graphs, not at your attention implementation. The repo's post on [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) goes deeper on harness construction generally.

One caveat: events measure *stream* time. If you use multiple streams, or if the GPU is shared with another process, the numbers need more care.

### Lock the clocks, or your p99 is thermal noise

GPUs do not run at a fixed frequency. They boost when cool and throttle when hot or when they hit a power limit. A benchmark that runs for 30 seconds may see clocks decline monotonically throughout, so "latency increased over the run" means "the card got warm," not "there is a leak."

```bash
# Persistence mode on, so the driver stays loaded and clocks do not reset between runs
sudo nvidia-smi -pm 1

# Lock SM clocks to a fixed value (query supported values with -q -d SUPPORTED_CLOCKS)
sudo nvidia-smi --lock-gpu-clocks=1200,1200

# Watch for throttling during the run — this is the column that explains weird p99s
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,\
clocks_throttle_reasons.active --format=csv -l 1

# Release the lock afterwards
sudo nvidia-smi --reset-gpu-clocks
```

Pick a locked clock somewhat below the boost ceiling so the card can sustain it indefinitely. Your absolute numbers will be a bit lower than an unlocked run's best case, and that is fine — a reproducible number that is 5% pessimistic beats an irreproducible number that is optimistic. Log the throttle reasons alongside the results; `SW_POWER_CAP` and `HW_THERMAL_SLOWDOWN` in that column explain most mysterious tails.

On a shared or virtualized machine you often cannot lock clocks. Say so in the writeup, and increase your sample count.

### Steady state, and enough samples for a p99

Two separate requirements that people conflate.

**Steady state** means the system has reached the operating point you want to measure: the allocator is not growing, the batch is full, the cache is at its working size, clocks have settled. For a serving benchmark, that means running load for a warmup period and *discarding* it, then measuring a window in the middle. Ramp up, hold, measure the hold, ramp down.

**Sample count** is about statistics. To estimate a p99 you need enough observations above it to be meaningful. The rule of thumb is $n(1 - p) \geq 10$: for p99 that is $n \geq 1000$ requests. With 100 samples your "p99" is the maximum of the sample and it will swing wildly between runs. If you are reporting p99 from 50 iterations, you are reporting noise.

Related: report the **variance across whole runs**, not just within one run. Run the benchmark three times end to end and see whether p50 moves. If run-to-run p50 varies by 10%, then a 5% "improvement" from your change is not a result.

### Isolate what you are measuring

Decide explicitly what is inside the timer.

- **Tokenization.** Pre-tokenize the prompt suite once, before the measurement loop, and keep the id lists in memory. A Python-side BPE encode of an 8k prompt is not nothing, and if it is inside your decode-step timer it will confuse you. But do **not** conclude that tokenization is free — in production it is on the request path, so measure it *separately* and report it separately. The same goes for `apply_chat_template`, which does string formatting and a full encode.
- **Data loading.** Reading prompts from disk, JSON parsing, dataset shuffling: all outside.
- **Detokenization.** Incremental detokenization runs per token in a streaming server, and it can be surprisingly expensive when it has to handle partial UTF-8 sequences. Measure it, but keep it out of the GPU-step timer.
- **The first token's special status.** TTFT includes prefill; TPOT must not. Compute them from separate timestamps, never by dividing total latency by token count. That average is a number about nothing.

### Closed-loop and open-loop load, and why closed-loop flatters you

This is the subtlest error in the list and the one most likely to make a system look fine right up until it collapses in production.

<figure class="blog-anim">
<svg viewBox="0 0 680 300" role="img" aria-label="Closed-loop load arrives as an evenly spaced train while open-loop Poisson load arrives in bursts that build a queue" style="width:100%;height:auto;max-width:820px">
<style>
.n2-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.n2-dot{fill:var(--accent,#6366f1)}
.n2-q{fill:var(--accent,#6366f1);opacity:.35;transform-box:fill-box;transform-origin:bottom center}
.n2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.n2-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes n2-flow{0%{transform:translateX(0);opacity:0}6%{opacity:1}94%{opacity:1}100%{transform:translateX(520px);opacity:0}}
@keyframes n2-queue{0%,10%{transform:scaleY(.12)}30%,44%{transform:scaleY(1)}64%,74%{transform:scaleY(.25)}88%,100%{transform:scaleY(.9)}}
.n2-a{animation:n2-flow 8s linear infinite}
.n2-a2{animation-delay:2s}
.n2-a3{animation-delay:4s}
.n2-a4{animation-delay:6s}
.n2-b{animation:n2-flow 8s linear infinite}
.n2-b2{animation-delay:.5s}
.n2-b3{animation-delay:.9s}
.n2-b4{animation-delay:4.4s}
.n2-b5{animation-delay:4.8s}
.n2-qa{animation:n2-queue 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.n2-a,.n2-b{animation:none;opacity:1}.n2-qa{animation:none;transform:scaleY(.9)}}
</style>
<text class="n2-lbl" x="30" y="34">Closed loop · 4 clients, each waits for its reply</text>
<rect class="n2-lane" x="30" y="52" width="600" height="48" rx="10"/>
<circle class="n2-dot n2-a"  cx="54" cy="76" r="11"/>
<circle class="n2-dot n2-a n2-a2" cx="54" cy="76" r="11"/>
<circle class="n2-dot n2-a n2-a3" cx="54" cy="76" r="11"/>
<circle class="n2-dot n2-a n2-a4" cx="54" cy="76" r="11"/>
<text class="n2-sub" x="30" y="126">arrival rate falls when the server slows — the queue can never build</text>
<text class="n2-lbl" x="30" y="184">Open loop · Poisson arrivals at a fixed rate</text>
<rect class="n2-lane" x="30" y="202" width="600" height="48" rx="10"/>
<rect class="n2-q n2-qa" x="34" y="206" width="86" height="40" rx="8"/>
<circle class="n2-dot n2-b"  cx="54" cy="226" r="11"/>
<circle class="n2-dot n2-b n2-b2" cx="54" cy="226" r="11"/>
<circle class="n2-dot n2-b n2-b3" cx="54" cy="226" r="11"/>
<circle class="n2-dot n2-b n2-b4" cx="54" cy="226" r="11"/>
<circle class="n2-dot n2-b n2-b5" cx="54" cy="226" r="11"/>
<text class="n2-sub" x="30" y="276">bursts arrive regardless of load — queue time becomes most of p99</text>
</svg>
<figcaption>Closed-loop load self-throttles into an even train, while open-loop Poisson arrivals keep coming during a slowdown and grow the queue that dominates tail latency.</figcaption>
</figure>

A **closed-loop** load generator keeps a fixed number of clients in flight. Each client sends a request, waits for the full response, then sends the next one. This is what almost every quick benchmark script does, and it has a built-in negative feedback loop. By Little's law, with $N$ clients in flight, throughput $X$ and response time $R$ satisfy $N = X \cdot R$. Hold $N$ fixed and double $R$ — throughput automatically halves. **The load backs off exactly when the server gets slow.** You cannot build a queue. You cannot observe latency collapse. Every closed-loop curve looks like a graceful, saturating hyperbola.

An **open-loop** generator sends requests at a rate $\lambda$ drawn from an arrival process — Poisson, typically, so inter-arrival times are exponentially distributed — with no regard for whether previous requests have finished. When the server slows, requests keep arriving, the queue grows, queue time dominates latency, and past a critical $\lambda$ the queue grows without bound and latency goes to infinity. That is what production traffic does, and it is why "it was fine until it wasn't" is the universal story of overloaded services.

The classic treatment is Schroeder, Wierman and Harchol-Balter's ["Open Versus Closed: A Cautionary Tale" (NSDI 2006)](https://www.usenix.org/legacy/events/nsdi06/tech/schroeder.html), which shows that the two models can produce qualitatively different conclusions about which scheduling policy is better — not just different numbers, different *rankings*. If you take one methodological idea from this post into unrelated work, take that one.

Practical guidance:

- Use **closed-loop** to find peak throughput and to sweep batch size. It is the right tool for "how much can this thing do."
- Use **open-loop** for anything involving latency SLOs, tail latency, admission control, or capacity planning. It is the only tool that will show you the knee.
- Report the arrival process and rate, not just "we used 64 concurrent users." Those two sentences describe different experiments.
- Watch for **coordinated omission**: if your generator measures latency from *send* rather than from *intended send time*, a stalled server hides its own stall, because the generator did not send during the stall. Record the scheduled arrival time and measure from that.

### Report the configuration, always

A latency number without its configuration is not a result. The minimum you must record alongside every measurement: model and revision, dtype, GPU model and count, driver and CUDA version, PyTorch version, engine version, batch size or concurrency, arrival process, prompt length distribution, output length distribution, sampling parameters, whether clocks were locked and to what, warmup count, sample count, and the percentiles reported. Have the harness dump this as JSON next to the results. Future-you comparing two runs six weeks apart will otherwise have no idea which of nineteen variables changed.

---

## 8. `bench.py` — the harness you can actually run

Here is the script, in the state it reaches at the end of this post. It measures TTFT and TPOT correctly, uses CUDA events and wall clock together, warms up, synchronizes, reports percentiles, and captures its environment. Track H's experiment-protocol post grows it into a full load generator; this version is the batch-1 latency harness.

```python
# nanoserve/bench.py
"""Batch-1 latency harness for nanoserve. Measures TTFT and TPOT honestly.

Usage:
  python -m nanoserve.bench --model meta-llama/Llama-3.1-8B-Instruct \
      --prompt-len 1000 --new-tokens 200 --warmup 5 --iters 30
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict

import torch


@dataclass
class Sample:
    ttft_ms: float          # wall clock, prompt in -> first token out
    tpot_ms: float          # mean gap between subsequent tokens
    decode_ms: list[float]  # per-step wall clock, first step excluded
    gpu_prefill_ms: float   # CUDA-event time for the prefill pass alone
    n_tokens: int

    @property
    def e2e_ms(self) -> float:
        return self.ttft_ms + (self.n_tokens - 1) * self.tpot_ms
```

The measurement loop itself. Note that it re-implements the decode loop rather than calling `generate_naive`, because the timing instrumentation has to sit *between* the steps:

```python
@torch.inference_mode()
def timed_generate(model, ids: torch.Tensor, n_new: int, eos_id: int | None) -> Sample:
    ev_a = torch.cuda.Event(enable_timing=True)
    ev_b = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ---- prefill: the first forward pass over the whole prompt ----
    ev_a.record()
    logits = model(ids)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ev_b.record()
    torch.cuda.synchronize()                  # the token is real only after this
    t_first = time.perf_counter()

    ttft_ms = (t_first - t_start) * 1e3
    gpu_prefill_ms = ev_a.elapsed_time(ev_b)

    ids = torch.cat([ids, next_id], dim=1)
    produced = 1
    step_ms: list[float] = []
    t_prev = t_first

    # ---- decode: one token per iteration ----
    for _ in range(n_new - 1):
        logits = model(ids)                   # naive: full recompute, no cache
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        torch.cuda.synchronize()
        t_now = time.perf_counter()
        step_ms.append((t_now - t_prev) * 1e3)
        t_prev = t_now
        produced += 1
        if eos_id is not None and int(next_id) == eos_id:
            break

    tpot_ms = statistics.mean(step_ms) if step_ms else float("nan")
    return Sample(ttft_ms, tpot_ms, step_ms, gpu_prefill_ms, produced)
```

Two details that are easy to get wrong and that change the answer:

1. **The synchronize after prefill is what makes TTFT real.** Without it you are timing an enqueue. With it, `t_first` is the moment the token genuinely exists on the device — which is still slightly optimistic versus a real server, where the token must also be detokenized and written to a socket.
2. **TPOT is the mean of the per-step gaps, not total time divided by token count.** Dividing conflates prefill into TPOT and makes long prompts look like slow decoding. Every serving benchmark that reports a single "tok/s" for the whole request is making this mistake.

Percentile reporting and the driver:

```python
def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def environment() -> dict:
    def nvsmi(query: str) -> str:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                text=True, timeout=10)
            return out.strip().splitlines()[0]
        except Exception:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": nvsmi("name"),
        "driver": nvsmi("driver_version"),
        "sm_clock_mhz": nvsmi("clocks.sm"),
        "throttle": nvsmi("clocks_throttle_reasons.active"),
        "power_w": nvsmi("power.draw"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=1000)
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    from nanoserve.model import Llama
    from nanoserve.tokenizer import Tokenizer

    model = Llama.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda").eval()
    tok = Tokenizer.from_pretrained(args.model)

    # Pre-tokenize OUTSIDE the timer. Synthetic ids keep prompt length exact.
    ids = torch.randint(10, 30000, (1, args.prompt_len), device="cuda")

    for _ in range(args.warmup):                       # same shapes as the real run
        timed_generate(model, ids, min(16, args.new_tokens), None)
    torch.cuda.synchronize()

    samples = [timed_generate(model, ids, args.new_tokens, None)
               for _ in range(args.iters)]

    ttft = [s.ttft_ms for s in samples]
    tpot = [s.tpot_ms for s in samples]
    e2e = [s.e2e_ms for s in samples]
    all_steps = [ms for s in samples for ms in s.decode_ms]

    report = {
        "config": vars(args),
        "env": environment(),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "ttft_ms": {"p50": pct(ttft, .50), "p99": pct(ttft, .99)},
        "tpot_ms": {"p50": pct(tpot, .50), "p99": pct(tpot, .99)},
        "step_ms": {"p50": pct(all_steps, .50), "p99": pct(all_steps, .99)},
        "e2e_ms": {"p50": pct(e2e, .50), "p99": pct(e2e, .99)},
        "tok_s_per_request_p50": 1000.0 / pct(tpot, .50),
        "n_samples": len(samples),
        "n_step_samples": len(all_steps),
    }
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
```

The output has this shape. **The values below are illustrative placeholders showing the report's structure — they are not a measurement I took, and yours will differ:**

```json
{
  "config": {"model": "...", "prompt_len": 1000, "new_tokens": 200, "iters": 30},
  "env": {"gpu": "NVIDIA A100-SXM4-80GB", "throttle": "Not Active", "torch": "..."},
  "peak_mem_gb": 0.0,
  "ttft_ms": {"p50": 0.0, "p99": 0.0},
  "tpot_ms": {"p50": 0.0, "p99": 0.0},
  "step_ms": {"p50": 0.0, "p99": 0.0},
  "e2e_ms": {"p50": 0.0, "p99": 0.0},
  "tok_s_per_request_p50": 0.0,
  "n_samples": 30,
  "n_step_samples": 5970
}
```

Note `n_step_samples`: 30 iterations × 199 steps gives nearly 6,000 decode-step observations, which is enough for a meaningful step-level p99 even though 30 request-level samples are not enough for a request-level p99. When you need a request p99, raise `--iters` past 1,000 or reduce `--new-tokens`.

### Run it in the order that teaches you the most

```bash
# 1. The naive baseline: quadratic recompute, batch 1.
python -m nanoserve.bench --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-len 1000 --new-tokens 200 --out naive.json

# 2. Vary the prompt length. TTFT should grow roughly linearly with prompt length;
#    with the naive loop TPOT will ALSO grow with prompt length, which is the tell.
for p in 128 512 1000 2000 4000; do
  python -m nanoserve.bench --model meta-llama/Llama-3.1-8B-Instruct \
      --prompt-len $p --new-tokens 64 --out naive_p$p.json
done

# 3. Vary the output length. With the naive loop, per-step time should CLIMB across
#    the run — plot s.decode_ms and you will see the quadratic with your own eyes.
python -m nanoserve.bench --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-len 128 --new-tokens 512 --out naive_long.json
```

Experiment 3 is the one to actually look at. Plot the per-step times from `decode_ms` against step index. **A cached engine gives you a flat line; the naive loop gives you a line with positive slope**, because step $i$ processes $p + i$ positions. That upward slope is the quadratic, visible without any theory. It is also the cleanest regression test imaginable for post 6: after you add the cache, the slope must go to approximately zero.

### The expected-range table

This is what to expect on named hardware. To be explicit about the honesty rule: the **ceiling** rows are derived from vendor specs, and the **expected** rows are ranges you should verify — not measurements.

| Quantity | Value | Source |
| --- | --- | --- |
| Llama-3.1-8B parameter count | 8.03 B | cited: [model card](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| Weight bytes, bf16 | 16.06 GB | derived: $8.03\times10^{9} \times 2$ |
| KV bytes per token (GQA, bf16) | 128 KB | derived: $2\cdot32\cdot8\cdot128\cdot2$ |
| RTX 4090 HBM bandwidth | 1,008 GB/s | cited: [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) |
| A100 80GB HBM bandwidth | 2,039 GB/s | cited: [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) |
| H100 80GB HBM bandwidth | 3,350 GB/s | cited: [H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) |
| L4 HBM bandwidth | 300 GB/s | cited: [L4 datasheet](https://resources.nvidia.com/en-us-l4/l4-datasheet) |
| Decode floor, 4090 | 15.9 ms/step | derived: 16.06 GB / 1.008 TB/s |
| Decode floor, A100 80GB | 7.9 ms/step | derived |
| Decode floor, H100 80GB | 4.8 ms/step | derived |
| Decode floor, L4 | 53.5 ms/step | derived |
| Batch-1 ceiling, 4090 | 63 tok/s | derived: reciprocal of the floor |
| Batch-1 ceiling, A100 80GB | 127 tok/s | derived |
| Compute time per decode step, A100 | 0.051 ms | derived: 16.1 GFLOP / 312 TFLOPS |
| Tensor-core idle fraction, A100 | 99.35% | derived |
| Machine balance, A100 | ~153 FLOP/byte | derived: 312 TFLOPS / 2.039 TB/s |
| Prefill FLOPs, 1,000-token prompt | 16.1 TFLOP | derived: ${2Np}$ |
| **Expected batch-1 tok/s, 4090, cached** | 40–60 | reproduce: `bench.py` |
| **Expected batch-1 tok/s, A100 80GB, cached** | 80–115 | reproduce: `bench.py` |
| **Expected TTFT, 1k prompt, A100** | ~100–200 ms | reproduce: `bench.py` |
| **Expected naive-vs-cached speedup, 1k/200** | large, workload-dependent | reproduce: `bench.py` both ways |
| Peak VRAM, 8B bf16 batch 1, 1.2k ctx | ~16.3 GB | derived: weights + 157 MB cache |

The expected ranges deserve a word on where they come from. The 4090 range of 40–60 tok/s is the derived ceiling of 63 multiplied by an achieved-bandwidth fraction of roughly 65–95%, which is the band a reasonable PyTorch implementation using `scaled_dot_product_attention` lands in. The A100 range of 80–115 is the same reasoning against its ceiling of 127. The TTFT range is $16.1\ \text{TFLOP}$ divided by 40–50% of the A100's 312 TFLOPS peak — an MFU band that dense prefill on a well-tuned stack tends to reach. **If your measured value falls outside these bands, that is information, and the most likely explanations are: above the band, a missing synchronize; below the band, thermal throttling, a mis-set dtype, or activations spilling to host memory.**

---

## 9. What we will beat, and roughly by how much

You now have a baseline and a scoreboard. Here is the arc of the series against it, with each technique's mechanism stated in terms of the quantities we just derived. Every figure below is cited from published work, not measured by me.

| Technique | What it changes in the model above | Public claim (cited) | Post |
| --- | --- | --- | --- |
| KV cache | $W$ from $np + n^2/2$ to $p + n$ | structural, not a benchmark | Track B, post 6 |
| Paged KV blocks | raises achievable batch by cutting fragmentation waste | vLLM paper: 2–4× throughput at equal latency vs FasterTransformer and Orca ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)) | Track B, post 8 |
| Continuous batching | raises average in-flight $B$ toward machine balance | Orca: up to 36.9× throughput at the same latency vs FasterTransformer for GPT-3 175B ([Yu et al., OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)) | Track C, post 12 |
| Weight quantization | cuts $B_{\text{weights}}$, so cuts the decode floor proportionally | 4-bit weights are 4× fewer bytes than fp16 — derived; see [AWQ](https://arxiv.org/abs/2306.00978) for the quality argument | Track F, post 29 |
| KV-cache quantization | cuts cache bytes/token, so raises the batch that fits | derived from the bytes/token formula | Track F, post 31 |
| CUDA graphs | removes per-step host overhead (the gap between wall and GPU time) | helps in proportion to that gap | Track F, post 33 |
| Speculative decoding | more than one token per weight read | 2–3× faster inference with identical output distribution ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)) | Track E cross-links, post 45 |

### The compounding is not a product

If you multiply those headline numbers together you get an absurd figure, and someone will quote it at you in a meeting. It is wrong, for three reasons worth understanding.

**They fix different bottlenecks, and only one bottleneck binds at a time.** This is Amdahl's law in its original form. Once the KV cache removes the quadratic recompute, decode is bandwidth-bound. Quantization then attacks bandwidth and helps a lot. But once quantization has cut the weight bytes by 4×, the decode step is short enough that per-step host overhead becomes a large fraction of it — so CUDA graphs, which did almost nothing before, suddenly matter. Each fix promotes a different thing to bottleneck status.

**Some techniques trade one metric for another.** Continuous batching multiplies aggregate throughput and does approximately nothing for a single user's TPOT — it may even worsen it slightly through contention. Chunked prefill improves TPOT under load and *costs* TTFT. Speculative decoding improves TPOT at low batch and can be a net loss at high batch, because the verification pass consumes the compute headroom that batching was using. A "3× speedup" is meaningless without saying 3× on which metric, at which load.

**The baselines differ.** Orca's 36.9× is against a static-batching system serving GPT-3 175B under a specific arrival pattern. vLLM's 2–4× is against systems that already do continuous batching. Stacking numbers measured against different baselines is not arithmetic; it is wishful thinking.

The honest way to state the arc: **each technique in this series removes a specific, identifiable inefficiency, and you should expect the total to be much less than the product and much more than the maximum.** The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), measures `nanoserve` end to end against vLLM and says plainly what it still loses.

---

## Case studies and real numbers

Four public results worth knowing, each cited, each illustrating something this post derived.

**PagedAttention and vLLM (Kwon et al., SOSP 2023).** The [paper](https://arxiv.org/abs/2309.06180) reports 2–4× throughput improvement at the same level of latency versus FasterTransformer and Orca, and attributes it specifically to reducing KV-cache memory waste — the authors measure that existing systems waste 60–80% of KV memory to internal and external fragmentation and to over-provisioned reservations. Read that through section 5's lens: cutting memory waste raises the batch that fits, and batch is what buys throughput on a bandwidth-bound decode. The mechanism is memory management; the *result* is throughput. That indirection is the most important idea in modern serving.

**Orca (Yu et al., OSDI 2022).** The [paper](https://www.usenix.org/conference/osdi22/presentation/yu) introduces iteration-level scheduling — what everyone now calls continuous batching — and reports up to 36.9× throughput improvement at the same latency level versus FasterTransformer on GPT-3 175B. The enormous number is a consequence of how bad the baseline was: static batching forces every request in a batch to wait for the longest one, so a batch of mixed-length requests wastes most of its slots. The gain is not from doing anything faster; it is from stopping the GPU from idling. Again, exactly what section 5 predicts.

**Speculative decoding (Leviathan et al., 2022).** The [paper](https://arxiv.org/abs/2211.17192) reports 2–3× faster inference while provably preserving the target model's output distribution. The mechanism is the cleanest possible illustration of "decode is bandwidth-bound": you spend the idle 99% of compute verifying several draft tokens in one pass, so one read of the weights yields more than one accepted token. It only works *because* the tensor cores were idle. On a compute-bound workload the same trick would be a straight loss.

**MLPerf Inference.** [MLCommons' inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) are worth studying less for their numbers than for their methodology. They define separate scenarios — Server (open-loop Poisson arrivals with a latency constraint), Offline (maximum throughput, no latency constraint), Single-Stream, Multi-Stream — precisely because a single "performance" number is meaningless. They mandate accuracy targets alongside performance, so a submission cannot buy speed by silently degrading quality. And they require a full system description with every result. If you ever need to justify why your benchmark harness is so pedantic, point at MLPerf: an industry consortium arrived at the same pedantry independently.

---

## When to reach for this (and when not to)

**Write the naive loop when:**

- You are learning, and you want a correctness oracle you fully understand.
- You need a reference implementation to diff a fast path against. Keep it in the test suite forever.
- You are debugging a cache or a kernel and need to know whether the *model* is right before you ask whether the *engine* is right.
- The generation is genuinely tiny — a classifier emitting one token, a router emitting a label. At $n = 1$ there is no cache to build, and the naive path is simply the correct path.

**Do not ship the naive loop when:** anything else is true. There is no production workload where recomputing the prefix on every step is defensible.

**Write your own benchmark harness when:** you are optimizing something, comparing two configurations of the same engine, or need numbers you can defend. The harness in section 8 is 150 lines and it will save you from a dozen wrong conclusions.

**Use vLLM's or SGLang's built-in benchmark scripts instead when:** you want to compare *engines* rather than *changes to your own engine*. They already implement open-loop arrival processes, prompt-suite handling, and percentile reporting, and using someone else's harness removes a whole class of "your benchmark was unfair" arguments. Reserve your own harness for the things it uniquely enables — instrumenting inside your engine, isolating a component, driving a regression test in CI.

**Stop writing your own engine and just use vLLM when:** you need to serve real traffic this quarter, your model is a standard architecture, and your constraints are ordinary. `nanoserve` exists to teach you what vLLM is doing and why; it is not going to beat a system with hundreds of contributors and hand-tuned kernels. Where writing your own genuinely pays: a nonstandard architecture nobody supports yet, a decoding scheme that needs engine-level surgery, a hard latency constraint that requires cutting features, or a deployment target where the big engines do not run. The repo's [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) covers what you would be giving up.

**Skip the derivation and just measure when:** you are debugging a specific regression on a specific machine. The roofline tells you where the ceiling is; it does not tell you why you are 40% below it. That is what profilers are for.

**Do the derivation before you measure when:** you are planning capacity, choosing hardware, sizing a model, or evaluating a vendor's claim. Five minutes of arithmetic with the bandwidth number and the parameter count will tell you whether a claim is plausible, and it costs nothing.

---

## Key takeaways

1. **The naive decode loop is quadratic in generated tokens.** Generating 200 tokens after a 1,000-token prompt pushes 219,900 token positions through the model instead of 1,199 — a factor of 183, all of it from one missing data structure.
2. **A request is two workloads, not one.** Prefill is a wide compute-bound GEMM with arithmetic intensity around the prompt length. Decode is a narrow bandwidth-bound GEMV with arithmetic intensity of one per sequence. Tune, schedule and measure them separately.
3. **The decode floor is weight bytes over HBM bandwidth.** For Llama-3.1-8B in bf16: 15.9 ms/step on a 4090, 7.9 ms on an A100 80GB, 4.8 ms on an H100. No kernel beats it, and knowing it takes thirty seconds of arithmetic.
4. **At batch 1 the tensor cores are idle over 99% of every decode step.** That is not a bug to find; it is the shape of the problem, and it is the headroom that batching and speculative decoding spend.
5. **Arithmetic intensity during decode equals the batch size**, so batching is nearly free until you reach machine balance (~153 on an A100) or run out of memory — and you will run out of memory first. That is why KV-cache management is the central problem of LLM serving.
6. **Latency composes as TTFT plus $(n-1)$ times TPOT.** A doubled TTFT costs 6% of a 200-token response; a doubled TPOT costs 94%. They are not interchangeable and they have different causes.
7. **The first iteration is always a lie.** Warm up with the same shapes, `torch.cuda.synchronize()` on both sides of the timer, lock the clocks, reach steady state, and take at least 1,000 samples before you say the word "p99."
8. **Closed-loop load generators cannot show you a collapse**, because fixed concurrency makes arrival rate fall as latency rises. Use open-loop Poisson arrivals for anything involving tails, SLOs or capacity.
9. **Every number needs a provenance.** Derived, cited, or reproducible. A results table without a `Source` column is a table of assertions.
10. **Keep the naive loop forever, as a test.** It is the cheapest correctness oracle you will ever have, and you will need it the first time your paged attention kernel goes subtly wrong past position 4,096.

---

## Further reading

- [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B) — parameter count, config, and the architecture numbers used throughout this post.
- [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) and [Ada Lovelace architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) — the bandwidth and TFLOPS figures behind every derived floor here.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180) — the vLLM paper; read section 3 for the memory-waste measurements.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling, the origin of continuous batching.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) — how to spend the idle compute this post derived.
- [Open Versus Closed: A Cautionary Tale (Schroeder, Wierman, Harchol-Balter, NSDI 2006)](https://www.usenix.org/legacy/events/nsdi06/tech/schroeder.html) — why your load generator's arrival model changes your conclusions.
- [MLPerf Inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) — an industry-scale example of benchmark methodology done properly.
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) and [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — `inference_mode`, asynchronous execution, streams, and events.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) · [a forward pass by hand](/blog/machine-learning/inference-engineering/a-forward-pass-by-hand-llama-from-scratch) · [the tokenizer boundary](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) · [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)
- Related on this blog: [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) · [the memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) · [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) · [the KV cache](/blog/machine-learning/large-language-model/kv-cache) · [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) · [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) · [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive)

Next in the series: **why recompute is fatal — writing a KV cache.** We take the 183× from section 2, build the data structure that removes it, and watch the per-step time curve go flat.
