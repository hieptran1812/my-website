---
title: "Static batching and the padding tax: why the obvious batch loop wastes half your GPU"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Derive why batching is the single highest-leverage thing in LLM inference, then derive the two taxes that the obvious pad-to-longest loop charges you, implement it in nanoserve with a slot ledger that makes the waste countable, and prove that a skewed batch of eight can be slower than serving those eight requests one at a time."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "batching",
    "throughput",
    "latency",
    "scheduler",
    "kv-cache",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 48
---

Batching is the highest-leverage idea in LLM inference. It is also the place where the obvious implementation does the most damage, and it does that damage silently — with the GPU pinned at 100% utilization and every dashboard green.

Here is the shape of the trouble, and it is worth stating as arithmetic before we state it as a story. On an A100 a single decode step for Llama-3.1-8B has to drag 16.06 GB of weights across HBM to produce **one** token. Serve thirty-two requests in the same step and the same 16.06 GB produces **thirty-two** tokens. The weight read is a fixed toll and batching splits it thirty-two ways. That is not a 10% optimization, it is a 25× one, and it is why every serving system on earth batches.

Then you write the batch loop the obvious way — collect some requests, pad them to the longest, run them in lockstep until they are all done — and you discover that a batch of eight chat requests where one of them happens to carry an 8,192-token document spends **85% of its prefill arithmetic on padding tokens** and **74% of its decode steps on rows that already emitted end-of-sequence and are just riding along**. Fewer than one token slot in six does any work. Under a realistic mix that loop is not merely inefficient; we will derive, from nothing but bandwidth arithmetic, that it can finish those eight requests **slower than serving them strictly one at a time**.

![A three by three batch rectangle showing each request's real prompt tokens beside the padding slots and the idle decode steps it burns after finishing](/imgs/blogs/static-batching-and-the-padding-tax-1.webp)

This post opens Track C of the series. It writes `nanoserve/static_batch.py` — a genuine, runnable, lockstep batching loop on top of the paged KV cache from [the paged cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) — and, more importantly, it writes `nanoserve/ledger.py`, the instrumentation that turns "the GPU feels busy" into a number you can put in a table. By the end you will be able to derive the two taxes from the length distribution of your own traffic, predict what batch size buys you before you build it, and say precisely why the fix in [the continuous batching post](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) is not a tuning knob but a change of data structure.

One promise carried forward from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is derived from arithmetic shown in the text, cited from a vendor spec or a paper with a link, or framed as something you should reproduce yourself with a script and an expected range. The results tables carry a `Source` column for exactly that reason. In this post that discipline is unusually easy to honor, because the central claims are counting arguments — the padding tax is combinatorics, not a benchmark — and you can reproduce every one of them on a laptop with no GPU at all.

---

## 1. The one law that makes batching worth doing

Start from the fact established in [the naive decode loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline): decode is memory-bandwidth-bound. A decode step multiplies a handful of vectors against every weight matrix in the model. The matrices are enormous and the vectors are tiny, so the step is not limited by how fast the GPU can multiply — it is limited by how fast the GPU can *fetch the weights*.

Make that quantitative. Let $P$ be the parameter count and let the weights be stored in bfloat16, so they occupy ${2P}$ bytes. A forward pass over $N$ token positions costs roughly ${2PN}$ floating-point operations, because each parameter participates in one multiply and one add per token. Now run a decode step for a batch of $B$ requests: each request contributes exactly one new token position, so $N = B$.

$$
\text{FLOPs per step} \approx 2 P B
\qquad
\text{weight bytes per step} = 2 P
$$

Arithmetic intensity — the ratio of work done to bytes moved, the x-axis of [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — is therefore

$$
\text{AI}_{\text{weights}} = \frac{2PB}{2P} = B
$$

The parameter count cancels. **The arithmetic intensity of the weight-matrix part of a decode step is exactly the batch size.** Not approximately, not asymptotically — exactly, to the precision of the "two FLOPs per parameter per token" model. At batch 1 you get one FLOP per byte, which is a catastrophic place to be on any modern accelerator. At batch 32 you get thirty-two. The entire economics of LLM serving follows from that one line.

### 1.1 The part batching does not amortize

There is a second term, and it behaves completely differently. Each request also reads its own KV cache. From [the memory math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), Llama-3.1-8B costs

$$
2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dims} \times 2 \text{ bytes} = 131{,}072 \text{ bytes per token}
$$

which is 128 KiB per token of context. A batch of $B$ requests each holding $S$ tokens of context reads $B \cdot S \cdot 131{,}072$ bytes per step. That term scales with $B$, so batching does *not* amortize it — every extra request brings its own cache and pays its own bandwidth.

What is the arithmetic intensity of the attention part alone? Per layer, per request, per step, the query attends over $S$ cached positions. The score computation and the value-weighted sum each cost $2 \cdot n_h \cdot d \cdot S$ FLOPs where $n_h$ is the number of query heads and $d$ the head dimension. The bytes read are $2 \cdot n_{kv} \cdot d \cdot S \cdot 2$, for K and V in bfloat16. So

$$
\text{AI}_{\text{attn}} = \frac{2 \cdot 2 \cdot n_h \cdot d \cdot S}{2 \cdot 2 \cdot n_{kv} \cdot d \cdot S} = \frac{n_h}{n_{kv}}
$$

Everything cancels except the grouped-query ratio. For Llama-3.1-8B that is 32 query heads over 8 KV heads, so the attention part of a decode step runs at an arithmetic intensity of **exactly 4 FLOPs per byte**, forever, no matter what batch size you choose. GQA is not only a memory-footprint trick; the group size *is* the attention kernel's arithmetic intensity. That is worth sitting with, because it sets a hard ceiling on how far batching can carry you.

### 1.2 The ceiling, derived

Put the two terms together. Per decode step:

$$
\text{AI}(B, S) = \frac{2PB + 4 n_h d L B S}{2P + 2 n_{kv} d L B S \cdot 2}
$$

where $L$ is the layer count. Substituting Llama-3.1-8B's numbers ($P = 8.03$ B, $L = 32$, $n_h = 32$, $n_{kv} = 8$, $d = 128$) and taking $B \to \infty$, the constant weight term drops out and the ceiling is

$$
\text{AI}_{\max}(S) = \frac{2P}{131{,}072\, S} + 4
$$

Compare that against the *ridge point* of the hardware — the arithmetic intensity at which peak FLOPs and peak bandwidth are balanced, computed as peak dense BF16 throughput divided by peak HBM bandwidth. Using NVIDIA's published specifications:

| GPU | HBM bandwidth | Dense BF16 peak | Ridge, FLOP/byte | Context beyond which decode is memory-bound at any batch | Source |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 24GB | 1,008 GB/s | 165.2 TFLOP/s | 164 | about 766 tokens | cited: [NVIDIA Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) + derived |
| L4 24GB | 300 GB/s | 121 TFLOP/s | 403 | about 307 tokens | cited: [NVIDIA L4 datasheet](https://resources.nvidia.com/en-us-l4/l4-datasheet) + derived |
| A100 80GB SXM | 2,039 GB/s | 312 TFLOP/s | 153 | about 822 tokens | cited: [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf) + derived |
| H100 80GB SXM | 3,350 GB/s | 989 TFLOP/s | 295 | about 421 tokens | cited: [NVIDIA H100 datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) + derived |

Read the last column carefully, because it is the most useful line in this section. On an A100, if your requests carry more than roughly 800 tokens of context on average, **no batch size will ever make decode compute-bound**. The KV term grows exactly as fast as the compute term, and the ceiling sits below the ridge. Batching still helps enormously — it just helps by moving you along a bandwidth-bound line, not by crossing over to a compute-bound one. Anyone who tells you to "batch until the tensor cores are busy" has not done this subtraction.

### 1.3 What batching actually buys, in milliseconds

Now the payoff, computed rather than asserted. Take an A100 80GB SXM at its datasheet 2,039 GB/s, Llama-3.1-8B in bf16, and fix the context at 1,024 tokens per request. Per decode step the GPU must move

$$
16.06 \text{ GB (weights)} + B \times 1024 \times 131{,}072 \text{ bytes (KV)}
$$

Divide by bandwidth to get a step-time floor, and divide $B$ by that to get aggregate token throughput:

| Batch | KV bytes/step | Total bytes/step | Step-time floor | Aggregate tok/s | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.13 GB | 16.19 GB | 7.94 ms | 126 | derived |
| 4 | 0.54 GB | 16.60 GB | 8.14 ms | 491 | derived |
| 8 | 1.07 GB | 17.13 GB | 8.40 ms | 952 | derived |
| 32 | 4.29 GB | 20.36 GB | 9.98 ms | 3,205 | derived |
| 64 | 8.59 GB | 24.65 GB | 12.09 ms | 5,294 | derived |
| 128 | 17.18 GB | 33.24 GB | 16.30 ms | 7,853 | derived |

![A timeline of batch sizes from one to sixty-four showing step time barely rising while aggregate throughput climbs twenty-five fold before the ridge point](/imgs/blogs/static-batching-and-the-padding-tax-2.webp)

Going from batch 1 to batch 32 costs **26% more time per step** and returns **25× the tokens**. Going from 32 to 128 costs 63% more time per step and returns 2.4× the tokens — still a win, but the curve is bending, because by then the KV traffic is comparable to the weight traffic and you are no longer amortizing anything. This is the whole reason the rest of Track C exists: batch size is the single biggest lever you own, and every mechanism in this track is about turning the *nominal* batch size into an *effective* one.

These are floors, not predictions. They assume perfect bandwidth utilization and ignore kernel launch overhead, the host-side Python that will dominate at batch 1 until you get to CUDA graphs, and the sampler. Reproduce them yourself on real hardware with the harness from the baseline post and expect to land somewhere in the 55–75% of floor range at batch 1 on an eager-mode PyTorch loop, tightening as batch grows and the host stops being the bottleneck.

---

## 2. The obvious loop

You now want to batch, so you write the loop that any reasonable engineer writes first. It has four steps and every one of them is defensible in isolation.

1. Collect $B$ requests off a queue.
2. Pad their token sequences to the longest one so they form a rectangular tensor.
3. Build an attention mask so the padding is invisible to attention.
4. Run prefill once, then decode in lockstep until every row has finished.

Call it **static batching**, because the membership of the batch is fixed at launch and does not change until the batch is done. It is also called request-level batching, in contrast to the iteration-level scheduling introduced by Orca that we get to in section 10.

Here is step 2 and 3 as real code. It is short, and it contains one decision that engineers get wrong roughly half the time.

```python
# nanoserve/static_batch.py
import torch


def pad_batch(prompt_ids: list[list[int]], pad_id: int, device="cuda"):
    """Pack ragged prompts into a rectangle, LEFT padded.

    Returns:
      input_ids [B, Lmax]  int64
      attn_mask [B, Lmax]  bool, True where a real token lives
      positions [B, Lmax]  int64, RoPE position of each real token
      lengths   [B]        int64, the true prompt lengths
    """
    B = len(prompt_ids)
    lengths = torch.tensor([len(p) for p in prompt_ids], dtype=torch.int64)
    Lmax = int(lengths.max())

    input_ids = torch.full((B, Lmax), pad_id, dtype=torch.int64)
    attn_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for i, p in enumerate(prompt_ids):
        input_ids[i, Lmax - len(p):] = torch.tensor(p, dtype=torch.int64)
        attn_mask[i, Lmax - len(p):] = True

    # RoPE positions must count REAL tokens only. A padded row must start
    # its first real token at position 0, not at position (Lmax - len).
    positions = attn_mask.cumsum(dim=1) - 1
    positions = positions.clamp(min=0)

    return (input_ids.to(device), attn_mask.to(device),
            positions.to(device), lengths.to(device))
```

**Left padding, not right padding.** This is the decision. With right padding, the last column of the rectangle is padding for every row except the longest one, so `logits[:, -1]` — the row every decode loop reads to pick the next token — is the model's prediction after a run of `<pad>` tokens for most of the batch. With left padding, the last column is the true final prompt token for every row, and `logits[:, -1]` is correct without any per-row gather. Right padding is fine for training, where you compute a loss over a mask; it is wrong for generation, where you need one specific position.

**Positions come from the mask, not from `arange`.** The second landmine. Rotary position embeddings encode absolute position, so if a left-padded row naively uses `arange(Lmax)`, its first real token is told it sits at position `Lmax - len(p)` instead of 0. The model still produces fluent text — that is what makes this bug so expensive — but the effective context window is shifted per row, long-range behavior degrades, and outputs differ between batch 1 and batch 8 for the same prompt. The `cumsum(mask) - 1` above is the fix and it is one line. If you take nothing else from this section, take the habit of deriving positions from the mask.

The attention mask itself is the third piece. Padding must be invisible in two directions: a real token must not attend to a pad token, and pad rows must not blow up the softmax with all-masked rows.

```python
# nanoserve/static_batch.py (continued)

def build_prefill_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """Causal mask AND key-padding mask, as an additive float mask.

    attn_mask : [B, L] bool, True where a real token lives
    returns   : [B, 1, L, L] float, 0.0 where allowed, -inf where not
    """
    B, L = attn_mask.shape
    device = attn_mask.device
    causal = torch.ones(L, L, dtype=torch.bool, device=device).tril()
    allowed = causal.view(1, 1, L, L) & attn_mask.view(B, 1, 1, L)

    # A fully-masked query row (a pad row) would make softmax produce NaN.
    # Let pad queries attend to themselves; their output is discarded anyway.
    self_ok = torch.eye(L, dtype=torch.bool, device=device).view(1, 1, L, L)
    allowed = allowed | (~attn_mask.view(B, 1, L, 1) & self_ok)

    mask = torch.zeros(B, 1, L, L, device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask
```

That self-attention escape hatch for pad rows deserves a note. Without it, a query row that is itself padding has every key masked, softmax over an all-`-inf` row produces `NaN`, and the `NaN` propagates through the residual stream into rows that were perfectly healthy. It is the single most common way a first batching implementation produces garbage for the *whole batch* rather than for the padded rows. Feed the mask to `torch.nn.functional.scaled_dot_product_attention` as `attn_mask` and it will do the right thing.

Notice what this mask costs. It is a `[B, 1, Lmax, Lmax]` float32 tensor. For a batch of 8 with a longest prompt of 8,192 tokens that is $8 \times 8192^2 \times 4 = 2.15$ GB of mask — larger than the KV cache the batch will ever use, allocated purely to describe padding. In practice you use a fused attention kernel that takes a causal flag plus sequence lengths and never materializes the matrix, which is exactly what FlashAttention and its descendants exist to do. But the fact that the *naive* expression of the mask is measured in gigabytes is the first hint that the rectangle is the wrong shape.

---

## 3. Tax one: the padding rectangle

Now derive the cost. Let the batch contain prompt lengths $\ell_1, \dots, \ell_B$. Prefill computes over $B \cdot \ell_{\max}$ token positions but only $\sum_i \ell_i$ of them are real. Define **prefill slot efficiency**:

$$
\eta_{\text{pad}} = \frac{\sum_{i=1}^{B} \ell_i}{B \cdot \ell_{\max}} = \frac{\overline{\ell}}{\ell_{\max}}
$$

The wasted fraction is $1 - \overline{\ell}/\ell_{\max}$. That is the whole law, and it is worth noticing that it involves the batch's *mean over max*, which is a statistic that behaves badly in exactly the situations you care about.

### 3.1 The distribution decides everything

Take $B$ lengths drawn independently from a distribution and ask what $\eta_{\text{pad}}$ looks like in expectation. For the exponential distribution with mean $\lambda$, the expected maximum of $B$ draws has a beautiful closed form:

$$
\mathbb{E}[\ell_{\max}] = \lambda \sum_{k=1}^{B} \frac{1}{k} = \lambda H_B
$$

where $H_B$ is the $B$-th harmonic number. Since $\mathbb{E}[\overline{\ell}] = \lambda$, the efficiency is

$$
\eta_{\text{pad}}(B) \approx \frac{1}{H_B} \approx \frac{1}{\ln B + 0.577}
$$

**Padding efficiency decays as one over the logarithm of batch size.** This is the sentence that should change how you think about the tuning knob `max_num_seqs`. Under static batching, every doubling of the batch buys you throughput on the weight-amortization axis and simultaneously *loses* you efficiency on the padding axis, and the loss is unbounded — it never stops decaying.

| Batch $B$ | $H_B$ | Exponential lengths, $\eta_{\text{pad}}$ | Uniform lengths, $\eta_{\text{pad}}$ | Source |
| --- | --- | --- | --- | --- |
| 1 | 1.000 | 100.0% | 100.0% | derived |
| 2 | 1.500 | 66.7% | 75.0% | derived |
| 4 | 2.083 | 48.0% | 62.5% | derived |
| 8 | 2.718 | 36.8% | 56.3% | derived |
| 16 | 3.381 | 29.6% | 53.1% | derived |
| 32 | 4.059 | 24.6% | 51.6% | derived |
| 64 | 4.744 | 21.1% | 50.8% | derived |
| 128 | 5.434 | 18.4% | 50.4% | derived |
| 256 | 6.124 | 16.3% | 50.2% | derived |

The uniform column comes from the same order-statistics argument: for $B$ draws uniform on $(0, M]$, $\mathbb{E}[\ell_{\max}] = M \cdot B/(B+1)$ and $\mathbb{E}[\overline{\ell}] = M/2$, so $\eta_{\text{pad}} = (B+1)/(2B)$, which converges to a floor of 50%. Uniform lengths cost you half your prefill and stop there. Exponential lengths keep taking more.

Real chat traffic is heavier-tailed than exponential. Prompt-length distributions from conversational traces are closer to lognormal, with a long right tail from pasted documents, code files and retrieved passages, and lognormal has a heavier tail than exponential, so its maximum grows faster and $\eta_{\text{pad}}$ falls faster. **Treat $1/H_B$ as an optimistic bound on what your chat traffic will do.** The recent agentic workloads are worse still: the vLLM team's [Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) reports that across 610 Codex and SWE-bench agent traces the input-to-output token ratio is 131 to 1, the median session runs 33 turns, and context grows by roughly 2,242 tokens per turn to a median of about 80K tokens by turn 30. A batch that mixes a first-turn agent request with a thirtieth-turn one is mixing 2K against 80K.

### 3.2 Worked example: the one-outlier batch

Take a concrete batch of eight, the kind an interactive endpoint assembles in a fraction of a second. Seven are ordinary chat turns; one user pasted a document.

```python
prompt_lengths = [96, 128, 152, 180, 204, 256, 320, 8192]
output_lengths = [20,  35,  40,  60,  80, 120, 200,  512]
```

Prefill accounting:

- Real prompt tokens: $96 + 128 + 152 + 180 + 204 + 256 + 320 + 8192 = 9{,}528$
- Rectangle: $8 \times 8192 = 65{,}536$ token slots
- $\eta_{\text{pad}} = 9{,}528 / 65{,}536 = 14.5\%$, so **85.5% of prefill arithmetic is padding**

Now delete the outlier and rerun the same arithmetic on the remaining seven:

- Real tokens: 1,336. Rectangle: $7 \times 320 = 2{,}240$. $\eta_{\text{pad}} = 59.6\%$

**Removing one request from the batch quadruples the prefill efficiency of the other seven.** That is the padding tax in one line: it is not a property of the batch, it is a property of the batch's worst member, and every other member pays it. A scheduler that could simply have run that one request by itself would have made the other seven four times cheaper.

![A layered breakdown of sixty nine thousand token slots showing prompt padding and post-EOS idle slots consuming all but fifteen percent](/imgs/blogs/static-batching-and-the-padding-tax-3.webp)

Notice also what the padding costs in *memory*, not just FLOPs. A dense KV cache allocated for the rectangle needs $8 \times 8192 \times 131{,}072 = 8.59$ GB of KV for a batch whose real content is $9{,}528 \times 131{,}072 = 1.25$ GB. Seven-eighths of the cache holds nothing. This is the failure mode that the paged cache from Track B already fixed — with block-level allocation each request holds $\lceil \ell_i / 16 \rceil$ blocks and no more — and it is important to be precise here: **paging fixes the memory padding, and does nothing at all about the compute padding.** The rectangle is still a rectangle in the tensor you feed the model. We will use that distinction hard in section 7.

---

## 4. Tax two: the stragglers

The second tax lives on the output side and it has a different flavor, because output lengths are not merely unequal — they are *unknown in advance*. You cannot look at a prompt and know whether the model will answer in twelve tokens or twelve hundred.

A static batch runs until every row is finished. A row that emits an end-of-sequence token at step 20 does not leave; there is no mechanism for it to leave, because the batch is a fixed-shape tensor and removing a row would require rebuilding it. So it sits there, its slot occupied, its logits computed and discarded, for every one of the remaining steps.

<figure class="blog-anim">
<svg viewBox="0 0 620 300" role="img" aria-label="Five batch rows decode in lockstep; each row turns grey once it emits end of sequence but keeps its slot until the longest row finishes at step ten" style="width:100%;height:auto;max-width:840px">
<style>
.an1-bg{fill:var(--background,#ffffff)}
.an1-live{fill:var(--accent,#6366f1)}
.an1-dead{fill:var(--border,#d1d5db);opacity:.6}
.an1-grid{stroke:var(--border,#d1d5db);stroke-width:1;opacity:.45}
.an1-frame{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.an1-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.an1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.an1-cur{stroke:var(--accent,#6366f1);stroke-width:3}
@keyframes an1-sweep{0%,4%{transform:translateX(0)}90%,100%{transform:translateX(466px)}}
.an1-mv{animation:an1-sweep 13s linear infinite}
@media (prefers-reduced-motion:reduce){.an1-mv{animation:none;transform:translateX(466px)}.an1-cur{opacity:0}}
</style>
<rect class="an1-bg" x="0" y="0" width="620" height="300"/>
<rect class="an1-live" x="106" y="46" width="88" height="34" rx="4"/>
<rect class="an1-dead" x="198" y="46" width="364" height="34" rx="4"/>
<rect class="an1-live" x="106" y="90" width="134" height="34" rx="4"/>
<rect class="an1-dead" x="244" y="90" width="318" height="34" rx="4"/>
<rect class="an1-live" x="106" y="134" width="180" height="34" rx="4"/>
<rect class="an1-dead" x="290" y="134" width="272" height="34" rx="4"/>
<rect class="an1-live" x="106" y="178" width="272" height="34" rx="4"/>
<rect class="an1-dead" x="382" y="178" width="180" height="34" rx="4"/>
<rect class="an1-live" x="106" y="222" width="456" height="34" rx="4"/>
<rect class="an1-bg an1-mv" x="102" y="38" width="470" height="230"/>
<rect class="an1-frame" x="102" y="42" width="464" height="222" rx="5"/>
<line class="an1-grid" x1="150" y1="42" x2="150" y2="264"/>
<line class="an1-grid" x1="196" y1="42" x2="196" y2="264"/>
<line class="an1-grid" x1="242" y1="42" x2="242" y2="264"/>
<line class="an1-grid" x1="288" y1="42" x2="288" y2="264"/>
<line class="an1-grid" x1="334" y1="42" x2="334" y2="264"/>
<line class="an1-grid" x1="380" y1="42" x2="380" y2="264"/>
<line class="an1-grid" x1="426" y1="42" x2="426" y2="264"/>
<line class="an1-grid" x1="472" y1="42" x2="472" y2="264"/>
<line class="an1-grid" x1="518" y1="42" x2="518" y2="264"/>
<line class="an1-cur an1-mv" x1="104" y1="38" x2="104" y2="268"/>
<text class="an1-sub" x="106" y="32">decode steps, all rows in lockstep</text>
<text class="an1-lbl" x="8" y="68">req A - 2 tok</text>
<text class="an1-lbl" x="8" y="112">req B - 3 tok</text>
<text class="an1-lbl" x="8" y="156">req C - 4 tok</text>
<text class="an1-lbl" x="8" y="200">req D - 6 tok</text>
<text class="an1-lbl" x="8" y="244">req E - 10 tok</text>
<text class="an1-sub" x="8" y="288">25 of 50 slot-steps carry a token; the grey half is the straggler tax</text>
</svg>
<figcaption>Each row goes grey the step it emits end-of-sequence, but it keeps its slot and the batch grinds on until the longest generation finishes at step ten.</figcaption>
</figure>

The accounting is the same shape as the padding tax, with output lengths $o_i$ in place of prompt lengths:

$$
\eta_{\text{dec}} = \frac{\sum_{i=1}^{B} o_i}{B \cdot o_{\max}} = \frac{\overline{o}}{o_{\max}}
$$

and the same order statistics apply. If output lengths are exponentially distributed then $\eta_{\text{dec}} \approx 1/H_B$ as well. Two independent taxes, one derivation, both decaying logarithmically in batch size.

For our worked example, $\sum o_i = 1{,}067$ and $o_{\max} = 512$, so:

$$
\eta_{\text{dec}} = \frac{1067}{8 \times 512} = \frac{1067}{4096} = 26.0\%
$$

Three quarters of the decode slot-steps compute logits for rows that finished. Combined over prefill and decode, the batch consumes $65{,}536 + 4{,}096 = 69{,}632$ token slots and does $9{,}528 + 1{,}067 = 10{,}595$ tokens of useful work: **15.2% overall**.

### 4.1 Why the straggler tax is worse than it looks

Three aggravating factors, in increasing order of nastiness.

**Output length is unpredictable and heavy-tailed.** Prompt lengths you can at least *see* before you batch, which means you can sort them. Output lengths you cannot see at all. Reasoning models have made this dramatically worse: a request that triggers an extended thinking budget can generate ten thousand tokens while its batch-mates generate thirty. In a static batch that single request holds the other rows hostage for the entire run.

**The idle rows still cost bandwidth.** A finished row does not merely fail to produce useful output; in a dense implementation its KV is still read every step, because the attention kernel processes the full `[B, ...]` tensor. So the straggler tax shows up in the step-time term too, not just in the efficiency ratio. A paged implementation can skip dead rows' KV in the gather — we do exactly that in section 6 — but the weight read, which is the dominant term, is paid regardless.

**It compounds with the padding tax through the mask.** Once decoding begins, each row's total sequence length is $\ell_{\max} + t$ in a dense implementation, not $\ell_i + t$, because the KV tensor was allocated to the rectangle. So a row whose real prompt was 96 tokens is attending over 8,192 positions of mostly-masked padding at every decode step. The two taxes multiply rather than add.

---

## 5. Tax three: the door that does not open

The third cost is not about wasted arithmetic at all. It is about what a static batch does to the requests that are not in it.

A static batch is atomic. Once the rectangle is built and prefill has run, there is no mechanism to add a row. A request that arrives one millisecond after launch waits for the *entire batch* to drain before its prefill can start.

Let $T_{\text{batch}}$ be the wall-clock duration of a batch. For a saturated server with requests arriving roughly uniformly across the batch window, the expected queueing delay before a request's prefill even begins is

$$
\mathbb{E}[W_{\text{queue}}] = \frac{T_{\text{batch}}}{2}
\qquad
W_{\text{queue}}^{p99} \approx T_{\text{batch}}
$$

and time-to-first-token is that plus the prefill time of the batch it eventually joins:

$$
\text{TTFT} = W_{\text{queue}} + T_{\text{prefill}}
$$

![A branching diagram showing an arriving request forced either to wait for a running batch to drain or to wait for a forming batch to fill, both paths ending in a twenty three second p99](/imgs/blogs/static-batching-and-the-padding-tax-4.webp)

### 5.1 Worked example: the p99 nobody can explain

Continue with the same batch of eight, on an A100. We will need two derived quantities.

**Padded prefill time.** The rectangle is 65,536 token positions. At ${2PN}$ FLOPs with $P = 8.03$ B, that is $1.05 \times 10^{15}$ FLOPs. Prefill is a genuinely compute-bound GEMM workload, but nobody hits datasheet peak; assume 40% of the A100's 312 TFLOP/s dense BF16, so 125 TFLOP/s. That gives

$$
T_{\text{prefill}} = \frac{1.05 \times 10^{15}}{1.25 \times 10^{14}} = 8.42 \text{ s}
$$

**Padded decode time.** The batch runs 512 steps. Each step moves 16.06 GB of weights plus $8 \times 8192 \times 131{,}072 = 8.59$ GB of dense padded KV, so 24.65 GB at 2,039 GB/s gives 12.09 ms per step, and $512 \times 12.09 = 6.19$ s.

$$
T_{\text{batch}} = 8.42 + 6.19 = 14.61 \text{ s}
$$

Now the queueing consequences, all derived from that one number:

| Quantity | Value | Source |
| --- | --- | --- |
| Batch wall clock | 14.61 s | derived |
| Median queueing delay | 7.31 s | derived |
| p99 queueing delay | about 14.6 s | derived |
| Median TTFT (queue + padded prefill) | about 15.7 s | derived |
| p99 TTFT | about 23.0 s | derived |
| GPU utilization as reported by `nvidia-smi` | near 100% | derived from the loop having no idle gaps |
| Useful token fraction | 15.2% | derived |

**A p99 TTFT of 23 seconds on a GPU that is 100% utilized.** That is the signature failure of static batching, and it is the reason "the GPU is busy so we must be efficient" is one of the most expensive beliefs in this field. The dashboard measures whether the SMs are issuing instructions. It does not measure whether those instructions are computing anything anyone asked for.

### 5.2 The batch-formation bind

There is a second queueing cost, at the other end. To run a batch of $B$ you must first *collect* $B$ requests. At Poisson arrival rate $\lambda$, the expected time to accumulate $B$ arrivals is $B/\lambda$, and the first arrival waits $(B-1)/\lambda$ of that. At $\lambda = 1$ request per second and $B = 8$, the unlucky first arrival waits seven seconds before the batch even launches.

So you add a formation timeout: launch after $B$ requests *or* after $\tau$ milliseconds, whichever comes first. Now you have traded one problem for another. A short $\tau$ keeps TTFT low but produces small batches, which throws away the weight amortization from section 1. A long $\tau$ produces full batches but adds $\tau$ to every request's TTFT at low load, when the server has nothing better to do. There is no setting of $\tau$ that is right at both low load and high load, and this is where most people first meet the throughput-latency frontier that [the batching fundamentals post](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) maps out.

Little's law gives the honest framing. If $L$ is the mean number of requests in the system, $\lambda$ the arrival rate and $W$ the mean time in system, then $L = \lambda W$. Under static batching, $L$ is quantized to the batch size and $W$ is floored by $T_{\text{batch}}/2$. You cannot get $W$ below that floor by tuning; the floor is structural, and it is set by the *slowest member of a group the request never asked to join*.

---

## 6. Implementing it in nanoserve, with a ledger

Enough derivation. Write the thing, and — this is the part that matters — instrument it so the waste is a number rather than an argument.

The loop sits on the paged KV store from Track B, which means we get to skip the memory padding entirely and isolate the compute padding. That is deliberate: it gives us the *best possible* static batching implementation, so when it still loses we know the loss is structural.

### 6.1 The ledger

```python
# nanoserve/ledger.py
from dataclasses import dataclass, field


@dataclass
class SlotLedger:
    """Counts token slots, split into useful and wasted, for one batch."""

    prompt_useful: int = 0     # real prompt tokens
    prompt_pad: int = 0        # padding token positions in the prefill rectangle
    decode_useful: int = 0     # decode steps that produced a token someone wants
    decode_dead: int = 0       # decode steps for rows that already emitted EOS
    steps: int = 0             # number of lockstep decode iterations
    per_step_live: list[int] = field(default_factory=list)

    def record_prefill(self, lengths: list[int]) -> None:
        b, lmax = len(lengths), max(lengths)
        self.prompt_useful += sum(lengths)
        self.prompt_pad += b * lmax - sum(lengths)

    def record_step(self, live: int, batch_size: int) -> None:
        self.steps += 1
        self.decode_useful += live
        self.decode_dead += batch_size - live
        self.per_step_live.append(live)

    @property
    def total_slots(self) -> int:
        return (self.prompt_useful + self.prompt_pad
                + self.decode_useful + self.decode_dead)

    @property
    def useful_slots(self) -> int:
        return self.prompt_useful + self.decode_useful

    def report(self) -> str:
        def pct(a, b):
            return f"{100.0 * a / b:5.1f}%" if b else "   n/a"
        pu, pp = self.prompt_useful, self.prompt_pad
        du, dd = self.decode_useful, self.decode_dead
        return "\n".join([
            f"prefill  useful {pu:>8,}  pad  {pp:>8,}   eff {pct(pu, pu + pp)}",
            f"decode   useful {du:>8,}  dead {dd:>8,}   eff {pct(du, du + dd)}",
            f"overall  useful {self.useful_slots:>8,}  "
            f"of {self.total_slots:>8,}   eff {pct(self.useful_slots, self.total_slots)}",
            f"steps    {self.steps:>4}   mean live rows "
            f"{sum(self.per_step_live) / max(1, self.steps):.2f}",
        ])
```

Four counters and a print method. This is the most valuable 40 lines in the post, because it converts a design argument into a runtime observation. Every serving system should be able to emit these four numbers on demand, and almost none do.

### 6.2 The lockstep loop

```python
# nanoserve/static_batch.py (continued)
import torch

from nanoserve.blocks import PagedSequence, build_batch_tables
from nanoserve.ledger import SlotLedger


@torch.inference_mode()
def run_static_batch(model, store, allocator, prompt_ids,
                     max_new_tokens=512, eos_id=None, pad_id=0):
    """Static (request-level) batching: fixed membership, lockstep decode.

    Returns (outputs, ledger). outputs[i] is the list of new token ids.
    """
    B = len(prompt_ids)
    device = store.data.device
    ledger = SlotLedger()
    ledger.record_prefill([len(p) for p in prompt_ids])

    # --- prefill -------------------------------------------------------
    input_ids, attn_mask, positions, lengths = pad_batch(
        prompt_ids, pad_id=pad_id, device=device)
    seqs = [PagedSequence(allocator, num_tokens=0) for _ in range(B)]
    for i, p in enumerate(prompt_ids):
        seqs[i].append(len(p))            # reserve blocks for REAL tokens only

    mask = build_prefill_mask(attn_mask)
    logits = model(input_ids, positions=positions, attn_mask=mask,
                   store=store, seqs=seqs)         # [B, Lmax, vocab]
    next_tok = logits[:, -1, :].argmax(dim=-1)     # left padding makes -1 correct

    outputs = [[int(t)] for t in next_tok.tolist()]
    finished = torch.tensor(
        [eos_id is not None and int(t) == eos_id for t in next_tok],
        device=device)

    # --- decode, in lockstep -------------------------------------------
    for step in range(1, max_new_tokens):
        if bool(finished.all()):
            break

        live = int((~finished).sum())
        ledger.record_step(live=live, batch_size=B)

        # THE POINT: every row is fed to the model, finished or not.
        # A finished row's slot cannot be released; the tensor is a rectangle.
        for i, s in enumerate(seqs):
            if not bool(finished[i]):
                s.append(1)

        table, seq_lens = build_batch_tables(seqs, device=device)
        pos = seq_lens - 1                          # [B] next RoPE position
        logits = model.decode_step(next_tok.view(B, 1), positions=pos,
                                   store=store, block_table=table,
                                   seq_lens=seq_lens)
        next_tok = logits[:, -1, :].argmax(dim=-1)

        for i in range(B):
            if not bool(finished[i]):
                outputs[i].append(int(next_tok[i]))
        if eos_id is not None:
            finished |= (next_tok == eos_id)

    for s in seqs:
        s.release()
    return outputs, ledger
```

Read the comment marked "THE POINT". Everything about static batching lives in those three lines: `finished[i]` is `True`, we skip appending KV for that row, we skip recording its output — and we still hand its row to the model, because `next_tok` is a `[B]` tensor and the model consumes `[B, 1]`. There is no expression in this function that could remove row $i$. Removing it would mean rebuilding `next_tok`, rebuilding `seqs`, rebuilding the block table, and — crucially — remapping every downstream index. The rectangle is load-bearing.

One optimization is already in there and worth calling out, because it separates this from the truly naive version: `s.append(1)` is only called for live rows, so a finished row's block table stops growing and `build_batch_tables` reports its true length. A paged attention kernel reading `seq_lens` will therefore not re-read a dead row's KV beyond its real content. That saves the KV bandwidth term. It does not save the weight read, which is 79% of the traffic at batch 8, and it does not save a single FLOP of the projection GEMMs.

### 6.3 A simulator you can run without a GPU

The ledger tells you what happened. To predict what *will* happen — and to reproduce every number in this post on a laptop — write the accounting as pure arithmetic.

```python
# nanoserve/batchsim.py
"""Wall-clock and slot accounting for batching strategies. No GPU required."""
from dataclasses import dataclass

KV_BYTES_PER_TOKEN = 131_072          # Llama-3.1-8B, bf16, GQA 8 KV heads
WEIGHT_BYTES = 16.06e9                # 8.03B params in bf16
FLOPS_PER_TOKEN = 2 * 8.03e9          # 2 FLOPs per parameter per token


@dataclass
class Machine:
    name: str
    hbm_bytes_per_s: float
    prefill_flops_per_s: float        # achieved, not datasheet peak


A100 = Machine("A100 80GB SXM", 2.039e12, 0.40 * 312e12)


def decode_step_seconds(m: Machine, live_tokens: int) -> float:
    """One lockstep decode iteration: all weights + the live KV."""
    return (WEIGHT_BYTES + live_tokens * KV_BYTES_PER_TOKEN) / m.hbm_bytes_per_s


def static_batch(m: Machine, prompts: list[int], outputs: list[int],
                 pad_prefill: bool, pad_kv: bool) -> dict:
    B, lmax, omax = len(prompts), max(prompts), max(outputs)
    prefill_tokens = B * lmax if pad_prefill else sum(prompts)
    t_prefill = prefill_tokens * FLOPS_PER_TOKEN / m.prefill_flops_per_s

    t_decode = 0.0
    for step in range(omax):
        if pad_kv:
            live = B * (lmax + step)                      # dense rectangle
        else:
            live = sum(p + min(step, o) for p, o in zip(prompts, outputs)
                       if o > step)                       # paged, dead rows skipped
        t_decode += decode_step_seconds(m, live)

    useful = sum(outputs)
    slots = B * lmax + B * omax
    return {
        "prefill_s": t_prefill,
        "decode_s": t_decode,
        "total_s": t_prefill + t_decode,
        "goodput_tok_s": useful / (t_prefill + t_decode),
        "slot_efficiency": (sum(prompts) + useful) / slots,
    }


def sequential(m: Machine, prompts: list[int], outputs: list[int]) -> dict:
    t = 0.0
    for p, o in zip(prompts, outputs):
        t += p * FLOPS_PER_TOKEN / m.prefill_flops_per_s
        for step in range(o):
            t += decode_step_seconds(m, p + step)
    return {"total_s": t, "goodput_tok_s": sum(outputs) / t}


def continuous(m: Machine, prompts: list[int], outputs: list[int]) -> dict:
    """Steady state: a freed slot is refilled immediately from a full queue."""
    B = len(prompts)
    t_prefill = sum(prompts) * FLOPS_PER_TOKEN / m.prefill_flops_per_s
    steps = -(-sum(outputs) // B)                        # ceil division
    avg_live = B * (sum(prompts) / B + sum(outputs) / (2 * B))
    t_decode = steps * decode_step_seconds(m, int(avg_live))
    return {"total_s": t_prefill + t_decode,
            "goodput_tok_s": sum(outputs) / (t_prefill + t_decode)}
```

Running it on the worked-example batch prints:

```console
$ python -m nanoserve.batchsim
workload: 8 requests, prompts sum 9,528 (max 8,192), outputs sum 1,067 (max 512)

sequential, batch 1                total  9.71 s   goodput   110 tok/s
static, dense padded KV            total 14.61 s   goodput    73 tok/s   slots 15.2%
static, paged KV + varlen prefill  total  5.54 s   goodput   193 tok/s   slots 15.2%
continuous, slots refilled         total  2.34 s   goodput   456 tok/s   slots  100%
```

This is arithmetic, not a benchmark — the script contains no GPU call and its output is fully determined by the four constants at the top. That is exactly why it is worth having: it lets you argue about batching strategy in a design review with numbers instead of adjectives, and it lets you sanity-check a real measurement against a floor.

---

## 7. The numbers, with provenance

The single most important row in that console output is the second one.

**A padded static batch of eight finishes in 14.61 seconds. Serving those same eight requests strictly one at a time finishes in 9.71 seconds.** Batching — the thing we spent all of section 1 proving is a 25× lever — made this workload 1.5× *slower*.

![A matrix of five length distributions against prompt slot use, decode slot use and overall useful fraction showing a hundred percent for fixed shapes and fifteen percent for a skewed chat batch](/imgs/blogs/static-batching-and-the-padding-tax-5.webp)

Here is the full comparison with every term's provenance:

| Strategy | Prefill | Decode | Total | Output goodput | Slot efficiency | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Sequential, batch 1 | 1.22 s | 8.49 s | 9.71 s | 110 tok/s | 100% | derived |
| Static, dense padded KV | 8.42 s | 6.19 s | 14.61 s | 73 tok/s | 15.2% | derived |
| Static, paged KV + varlen prefill | 1.22 s | 4.32 s | 5.54 s | 193 tok/s | 15.2% | derived |
| Continuous, slots refilled | 1.22 s | 1.12 s | 2.34 s | 456 tok/s | 100% | derived |
| Theoretical decode-only ceiling at $B = 8$ | — | — | — | 952 tok/s | — | derived, section 1.3 |

Every cell traces back to arithmetic already shown: prefill from ${2PN}$ over 125 TFLOP/s achieved, decode from weight bytes plus live KV bytes over 2,039 GB/s, goodput from useful output tokens over total wall clock. Reproduce all of it with `batchsim.py`; verify the achieved-FLOPs assumption on your own hardware with the harness from the baseline post and expect prefill MFU somewhere in the 30–50% band for an eager PyTorch implementation with `scaled_dot_product_attention`, higher with a fused attention backend.

Four things fall out of that table and each is worth stating plainly.

**Paging rescues static batching from being actively harmful, and no further.** Going from row two to row three is a 2.6× improvement and it comes entirely from *not padding the KV cache and not padding the prefill*. Slot efficiency is unchanged at 15.2%, because the lockstep structure is unchanged. Paging fixed the memory, not the schedule.

**The remaining gap is the schedule, and it is 2.4×.** Row three to row four changes nothing about kernels, memory layout, or precision. It changes only *who is in the batch at step $t$*. That is a scheduler change, and it is worth more than most kernel work you will ever do.

**Goodput and utilization point in opposite directions.** Row two has the highest GPU utilization of any row in the table and the second-worst goodput. Any optimization target that is not goodput will happily lead you to row two.

**The ceiling is set by the workload's shape, not the hardware.** Row five is 952 tok/s of pure decode, but this workload has 9,528 prompt tokens for 1,067 output tokens — an input-to-output ratio of 8.9 to 1 — so prefill is half the wall clock even in the ideal case. That ratio is a property of your product, not of your engine, and it determines which optimizations are worth doing. A retrieval-augmented workload lives or dies on prefill and prefix caching. A chat workload lives or dies on decode batch size.

### 7.1 Worked example: reading the tax off your own traffic

You do not need any of my numbers. You need three statistics from your access log: the distribution of prompt lengths, the distribution of output lengths, and the arrival rate. From those, the tax follows.

Take a service with prompt lengths that are roughly lognormal with a median of 400 tokens and a 99th percentile of 12,000 — a very ordinary shape for a chat product with document upload. Assemble batches of 32.

- The batch's maximum is dominated by the tail. With 32 draws, the chance that at least one exceeds the 96.9th percentile is $1 - 0.969^{32} \approx 63\%$, so more than half your batches contain a prompt in the top 3% of the distribution.
- Sampling the mean-over-max for that shape lands $\eta_{\text{pad}}$ in the 10–20% band, worse than the exponential bound of $1/H_{32} = 24.6\%$ because lognormal has the heavier tail.
- If output lengths are exponential with mean 150, $\eta_{\text{dec}} \approx 1/H_{32} = 24.6\%$.

So a nominal batch of 32 is doing the useful work of a batch of roughly 4 to 8, at the memory cost of 32 and the tail-latency cost of 32. That is the sentence to bring to the design review. The script that computes it from a log is fifteen lines and you should write it before you tune a single flag.

---

## 8. How to measure this honestly

Everything above is derivation. When you go to confirm it on real hardware, the measurement itself is a place where people fool themselves, so here is the protocol that keeps the numbers meaningful.

**Warm up, then synchronize, then time.** The first few iterations pay for CUDA context creation, cuBLAS handle setup, autotuning and allocator growth. Discard at least 10 iterations. CUDA kernel launches are asynchronous, so a Python `time.perf_counter()` around a launch measures the *launch*, not the work.

```python
# nanoserve/bench_batch.py
import torch


def timed_decode_steps(fn, warmup=10, iters=50):
    """Median per-step time in ms, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]
```

**Report the median and the p99, never the mean.** A batching loop's distribution is bimodal by construction: steps where a row finishes look different from steps where none does. A mean hides that.

**Lock the clocks or report that you did not.** Consumer cards in particular will boost for a few seconds and then thermally throttle, so a 30-second run and a 5-minute run give different answers. `nvidia-smi --lock-gpu-clocks` on hardware that supports it; otherwise say so in the results table.

**Measure open-loop, not closed-loop.** This is the one that matters most for batching. A closed-loop load generator sends request $n+1$ only after receiving response $n$, which means it *cannot* produce a queue, which means it cannot reveal any of the queueing pathologies from section 5. Static batching looks perfectly reasonable under closed-loop load and falls apart under Poisson arrivals at the same mean rate. Generate arrivals from an exponential inter-arrival distribution at a fixed target rate, and let the queue grow if the server cannot keep up. If your benchmark never produces a queue, it never tested your scheduler.

**Report goodput, not throughput.** Total tokens per second counts tokens produced for rows that already finished if your implementation is sloppy, and counts tokens for requests whose clients timed out and left. Goodput counts only tokens delivered to a client within its latency budget. The vLLM team's [Anatomy of a High-Throughput Inference System post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) gives the standard metric definitions worth adopting verbatim: TTFT is submit to first token, ITL is the gap between consecutive tokens, and TPOT is the mean ITL over the output.

**State your workload shape in every number you publish.** A tok/s figure without input length, output length, batch size, model and dtype is not a measurement, it is a mood. The same Anatomy post documents vLLM's own defaults for exactly this reason — its latency benchmark uses 32 input and 128 output tokens at batch 8, and its throughput benchmark runs 1,000 ShareGPT samples at unbounded request rate. Two completely different shapes producing two completely different numbers, both correctly labeled.

---

## 9. The stress test: when static batching is exactly right

The honest version of this post has to include the case where the obvious loop is not just acceptable but optimal, because that case is common and the reflex to reach for continuous batching everywhere is its own failure mode.

![A decision tree splitting offline jobs with sortable inputs from interactive serving with random arrivals and giving the slot efficiency of each leaf](/imgs/blogs/static-batching-and-the-padding-tax-6.webp)

### 9.1 Fixed shapes: static batching is optimal

If every request has the same prompt length and the same output length, then $\ell_{\max} = \overline{\ell}$ and $o_{\max} = \overline{o}$, so $\eta_{\text{pad}} = \eta_{\text{dec}} = 100\%$. There is no tax, because there is nothing to pad and nothing to straggle. And static batching is *better* than continuous in this case, because it has no per-step scheduling overhead, no block-table rebuild, no admission logic, and — most importantly — a completely static tensor shape, which means you can capture the whole decode step in a CUDA graph and eliminate the host-side launch overhead entirely.

Workloads that genuinely have this shape: embedding generation, classification with a fixed template, reward-model scoring during RLHF, log-probability evaluation over a benchmark, and any pipeline that has already bucketed its inputs.

### 9.2 Offline batch jobs: sort, and the tax mostly evaporates

If you have all the inputs on disk before you start, you can sort by length. Derive what that buys.

Lengths uniform on $(0, M]$, sorted into $K$ equal-width buckets, batches formed within a bucket. Bucket $j$ covers lengths in $((j-1)M/K,\ jM/K]$, so within it the mean is about $(j - 0.5)M/K$ and the max is $jM/K$. Token-weighted across all buckets:

$$
\eta_{\text{sorted}} = \frac{\sum_{j=1}^{K} (j - 0.5)}{\sum_{j=1}^{K} j}
= \frac{K^2/2}{K(K+1)/2} = \frac{K}{K+1}
$$

**Bucketing into $K$ length buckets gives efficiency $K/(K+1)$ regardless of batch size.** Eight buckets gives 88.9%, sixteen gives 94.1%, four gives 80%. Compare that against the unsorted $(B+1)/(2B) \to 50\%$ and the exponential $1/H_B \to 0$. Sorting does not merely improve the constant; it removes the dependence on $B$ entirely, which means you can now use the large batch that section 1 says you want.

```python
# nanoserve/offline.py
def length_bucketed_batches(items, key, batch_size, num_buckets=8):
    """Group items so each batch contains similar-length inputs.

    items : list of anything; key(item) -> token length
    Yields lists of at most batch_size items, longest buckets first so the
    biggest memory demand is faced when the pool is empty rather than full.
    """
    ordered = sorted(items, key=key)
    n = len(ordered)
    edges = [round(n * j / num_buckets) for j in range(num_buckets + 1)]
    for j in range(num_buckets - 1, -1, -1):
        bucket = ordered[edges[j]:edges[j + 1]]
        for start in range(0, len(bucket), batch_size):
            yield bucket[start:start + batch_size]
```

Note the equal-*population* buckets rather than equal-width ones: with a heavy-tailed distribution, equal-width buckets put 99% of the items in bucket one and defeat the purpose.

There is an important limit here, and it is the reason sorting is only half a fix. **You can sort by prompt length because you can see it. You cannot sort by output length, because it does not exist yet.** So bucketing takes $\eta_{\text{pad}}$ from 50% to 89% and leaves $\eta_{\text{dec}}$ exactly where it was. For an offline job with long inputs and short outputs — scoring, classification, extraction, summarization — that is nearly the whole win, because prefill dominates. For an offline job with short inputs and long outputs — synthetic data generation, reasoning traces — the straggler tax is untouched and continuous batching still wins.

You can go one step further and sort by *predicted* output length, using a small classifier over the prompt or a simple heuristic on task type. It is a real technique. It also introduces a prediction that can be wrong, and a mispredicted straggler in a bucket of supposedly-short generations costs exactly as much as an unsorted one.

### 9.3 The other stress cases

**Batch 1.** Static and continuous batching are identical at batch 1; there is nothing to pad against and nothing to straggle behind. All of section 1's amortization is unavailable, so you sit at 126 tok/s on an A100 and the weights are 99.2% of the traffic. If your product is one user on one GPU, none of this track applies to you and you should go straight to Track E and F — kernels and quantization — which are the only things that help at batch 1.

**Batch 128 on a 24 GB card.** You will not get there. At 128 KiB per token, 128 concurrent requests averaging 1,024 tokens need 17.2 GB of KV, and Llama-3.1-8B's weights already took 16.1 GB of the RTX 4090's 24 GB. The padding tax on a dense implementation would demand $128 \times \ell_{\max}$ tokens of cache, which for any realistic $\ell_{\max}$ is an immediate out-of-memory. This is the concrete sense in which the padding tax and the memory wall are the same wall.

**A 128K-token request in the batch.** One request at 128K tokens costs 16.8 GB of KV by itself, more than the weights. Under dense static batching the rectangle would demand that much *per row*. Under paged static batching it fits, but its prefill is $2 \times 8.03\text{e}9 \times 131{,}072 = 2.1 \times 10^{15}$ FLOPs — about 17 seconds at 125 TFLOP/s — during which every other row in its batch produces nothing at all. This is precisely the problem chunked prefill exists to solve, and it is post 13 of this series.

**An L4 instead of an A100.** The L4's 300 GB/s makes the weight read ${16.06/0.300 = 53.5}$ ms per decode step, nearly seven times an A100's. Batching matters *more* on slow-bandwidth hardware, not less, because the fixed toll you are amortizing is larger. But the L4's ridge point of 403 FLOP/byte is also much higher, so you would need a batch in the hundreds to approach it — and the memory to hold those requests' KV does not exist on a 24 GB card. Slow-memory, small-VRAM accelerators are where static batching's inefficiency hurts most and where you have least room to fix it by turning knobs.

---

## 10. Case studies and public results

**Orca (OSDI 2022).** The paper that named the problem. Yu et al., ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu), introduced *iteration-level scheduling*: rather than scheduling at the granularity of a request, schedule at the granularity of a single decode iteration, so a finished request leaves and a new one enters between steps. It also introduced *selective batching*, the observation that the attention operation cannot be batched across sequences of different lengths the way the projection GEMMs can, so you batch what is batchable and handle attention per-sequence. The paper reports throughput improvements over request-level batching in prior systems on the order of tens of times at a matched latency target — the headline figure is 36.9× against FasterTransformer. Treat that as a comparison against a specific 2022 baseline on a specific workload, not as a number you should expect to reproduce, but the mechanism it isolates is exactly the two taxes derived above.

**vLLM and PagedAttention (SOSP 2023).** Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180), measured the memory half of the problem and reported that existing serving systems wasted 60–80% of KV cache memory to fragmentation and over-reservation, and that paging plus continuous batching delivered 2–4× the throughput of prior state-of-the-art systems at the same latency level. Section 7's table reproduces the structure of that result from arithmetic: our paged row is 2.6× the dense padded row.

**vLLM's super-sequence framing.** The vLLM team's [Anatomy of a High-Throughput Inference System post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) describes what replaces the rectangle. Continuous batching in vLLM flattens all sequences in the running set into **one concatenated "super sequence"**, with position indices and attention masks doing the work of isolating each request from its neighbors. There is no `B` dimension to pad. The scheduler holds a waiting queue and a running queue alongside the KV manager, and its decision each step is simply a mapping from request id to a token count. The same post documents the default block size of 16 tokens and the free-block-queue pool that makes such a mapping cheap to satisfy.

That framing is the punchline of this whole post. The reason static batching pads is that it insists the batch be a *rectangle*, and a rectangle has one width. Once the batch is a flat concatenation with per-request offsets, "width" stops existing as a concept and there is nothing to pad to.

**vLLM V1's scheduler.** The [vLLM V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) reports that representing every scheduling decision as `{request_id: num_tokens}` erased the prefill/decode distinction entirely, letting chunked prefill, prefix caching and speculative decoding share one policy, and cites up to 1.7× throughput over V0 on Llama 3.1 8B and 3.3 70B with ShareGPT. That is what the *second-generation* fix to this problem looks like: not "add continuous batching" but "stop having two kinds of work".

**Model Runner V2.** The vLLM team's [Model Runner V2 post](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) goes further in the same direction with a decoupled persistent batch, where each request holds a fixed row in a state table independent of its ordering in the step, and per-step gathers build the ordered block tables. They report a 56% throughput gain on Qwen3-0.6B on a single GB200 in a host-overhead-dominated setting, and a 6.3% TPOT reduction on GLM-4.7-FP8 across four GB200s. Note the shape of that result: the gains are in *host overhead and ordering*, which is what remains to be optimized once the padding tax is gone.

---

## 11. The fix, named

The two taxes have the same cause and therefore the same cure. Padding waste comes from forcing every row to the width of the widest. Straggler waste comes from forcing every row to the length of the longest. Both are consequences of the batch being a **fixed rectangle whose membership cannot change**.

So change the data structure. Make the batch a *set* that is re-evaluated at every step: a request that emits end-of-sequence is removed from the set the same iteration, its KV blocks return to the free pool, and a waiting request takes the freed capacity on the very next step. Nothing is ever padded to anything, because there is no shared width — each request contributes exactly the tokens it has.

![A two column comparison of a static rectangle keeping three thousand idle slots against a continuous set that refills a slot the step a request finishes](/imgs/blogs/static-batching-and-the-padding-tax-7.webp)

That is continuous batching, and [the next post writes it](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) — the `step()` function, the waiting queue, the running set, the admit-and-finish logic. It is about sixty lines on top of what you now have, and section 7's table says those sixty lines are worth 2.4× on this workload with no kernel changes at all.

Three things to carry into it.

**The win comes from admission, not from removal.** Removing a finished row from a *closed* set of eight requests saves you nothing on the critical path, because the longest generation still takes 512 steps either way. The win appears only when a freed slot is *refilled* from a queue of waiting work. Continuous batching is an admission-control mechanism that happens to look like a batching mechanism. That framing matters when you get to [the scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) and start asking which waiting request should get the slot.

**Per-step scheduling is not free.** Every step now rebuilds the block table, recomputes sequence lengths, and runs the admission policy on the host. At a 8 ms step time on an A100 you have a few hundred microseconds of budget before the scheduler itself becomes the bottleneck, which is exactly why vLLM builds its input tensors with GPU kernels and overlaps CPU scheduling of step $N+1$ with GPU execution of step $N$.

**Dynamic shapes fight CUDA graphs.** A static batch has one shape and captures cleanly into a CUDA graph. A continuous batch changes shape every step. The production answer is to bucket batch sizes and capture a graph per bucket, padding *up* to the nearest bucket — which is to say, production engines reintroduce a small, bounded, deliberate amount of padding to buy back the launch overhead. The tax is not evil; the tax being unbounded and invisible is what was evil.

---

## When to reach for this (and when not to)

**Use static batching when:**

- Every request in the job has the same shape — fixed-length classification, embedding, scoring, log-probability evaluation. Efficiency is 100% and you get free CUDA graph capture.
- You have an offline job whose inputs are all available up front. Sort into length buckets, get $K/(K+1)$ efficiency on prefill, and enjoy the simplicity.
- The workload is prefill-dominated with short, uniform outputs. The straggler tax is small when $o_{\max}$ is close to $\overline{o}$.
- You are writing your first batching loop and want to understand what continuous batching is buying you. Write the ledger, run the simulator, then read the next post.

**Do not use static batching when:**

- Requests arrive independently over time. The closed door in section 5 is structural and no batch-size or timeout tuning removes it.
- Prompt lengths vary by more than about 2× within a batch. Below that, the padding tax is under 50% and the simplicity may be worth it; above it, you are burning more than you save.
- Output lengths are unbounded, which for reasoning models is always. One extended-thinking request holds a whole batch hostage.
- You care about p99 TTFT at all. Static batching's p99 is bounded below by the batch duration, which is bounded below by the longest generation in it.

**And use vLLM instead of your own code when** you are shipping a product rather than learning how the machine works. Everything in this post exists inside vLLM, SGLang and TensorRT-LLM in a far better form: fused kernels, GPU-side tensor construction, piecewise CUDA graphs, prefix caching, chunked prefill, and a scheduler that has survived years of production traffic. Write `nanoserve` to understand what those systems are doing and why their flags exist; run the real one when it counts.

---

## Key takeaways

1. **The arithmetic intensity of the weight-matrix part of a decode step is exactly the batch size.** That single line is why every serving system batches, and it is why batch 1 wastes 99% of the GPU's arithmetic capability.
2. **Batching does not amortize KV traffic.** The attention part of decode has arithmetic intensity equal to the GQA ratio — 4 for Llama-3.1-8B — forever, at any batch size. Beyond roughly 800 tokens of context on an A100, decode is memory-bound at every batch size and no amount of batching crosses the ridge.
3. **Prefill slot efficiency is $\overline{\ell}/\ell_{\max}$**, and for exponential lengths it decays as $1/H_B \approx 1/\ln B$. Bigger static batches are monotonically worse on this axis, without limit.
4. **Decode slot efficiency is $\overline{o}/o_{\max}$** — the same law with output lengths — and output lengths cannot be sorted because they do not exist until you generate them.
5. **One outlier taxes the whole batch.** In the worked example, removing a single 8,192-token prompt from a batch of eight quadrupled prefill efficiency for the other seven.
6. **A static batch can be slower than no batching at all.** 14.61 s padded versus 9.71 s sequential, both derived from the same bandwidth arithmetic, on the same GPU.
7. **Paging fixes the memory padding and none of the compute padding.** It takes this workload from 73 to 193 tok/s and leaves slot efficiency at 15.2%. The remaining 2.4× is purely a scheduling change.
8. **High GPU utilization is not evidence of efficiency.** The worst row in the results table is the one where the SMs never go idle. Measure goodput.
9. **Static batching is optimal for fixed-shape and sortable offline work.** Bucketing into $K$ length buckets gives $K/(K+1)$ efficiency independent of batch size, and a fixed shape captures cleanly into a CUDA graph.
10. **Ship the ledger before you ship the optimization.** Four counters — useful and wasted prompt slots, useful and dead decode slots — turn every argument in this post into a number your service can print.

---

## Further reading

- Yu et al., ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022 — iteration-level scheduling and selective batching, the origin of the fix.
- Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180), SOSP 2023 — the memory half, with the 60–80% waste measurement and the 2–4× throughput result.
- vLLM, ["Inside vLLM: Anatomy of a High-Throughput Inference System"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) — the super-sequence formulation, the scheduler's queues, and the metric definitions used throughout this series.
- vLLM, ["vLLM V1: A Major Upgrade to vLLM's Core Architecture"](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) — the `{request_id: num_tokens}` scheduling representation that erases the prefill/decode split.
- [PyTorch `scaled_dot_product_attention` documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — mask semantics, and which backends accept which mask forms.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) · [the naive decode loop and your first baseline](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) · [the memory math of the KV cache](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) · [the paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) · [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) · [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook)
- Elsewhere on this blog: [batching fundamentals and the latency-throughput tradeoff](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) · [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) · [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound)
