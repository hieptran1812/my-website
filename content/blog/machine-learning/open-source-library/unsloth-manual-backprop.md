---
title: "Hand-Derived Backpropagation: How Unsloth Beats Autograd"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "PyTorch autograd is general and correct, but it pays for that generality in saved tensors and generic kernels. Unsloth hand-derives the gradients for its fused blocks, saving only what the math needs and writing gradients in place — for big VRAM and speed wins with identical numbers."
tags: ["unsloth", "backpropagation", "autograd", "lora", "qlora", "gradient", "memory-optimization", "triton", "swiglu", "rmsnorm"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 28
---

The first time I profiled a QLoRA fine-tune and watched the backward pass eat more VRAM than the forward, I assumed I had a leak. I did not. That was autograd doing exactly what it is designed to do: recording every operation and pinning every intermediate tensor it might need to differentiate later. The backward pass was not leaking — it was *remembering*, and remembering is expensive.

Unsloth's headline numbers — "up to 2x faster with up to 70% less VRAM," "80% less VRAM for GRPO" — come from refusing to remember more than the math requires. The trick is not a clever approximation; the gradients are bit-for-bit the same as what autograd would compute. The trick is that Unsloth *throws away the autograd graph for its hot blocks* and writes the backward pass by hand. When you derive the gradient on paper, you discover that most of what autograd saves is recomputable from a tiny residue, that several backward operations collapse into a single fused kernel, and that the gradient can be written straight back over a buffer you already own.

![Autograd's generic tape versus Unsloth's hand-derived backward: autograd records a node and saves a tensor for every op, while Unsloth fuses the block and saves only what the math needs.](/imgs/blogs/unsloth-manual-backprop-1.webp)

The diagram above is the mental model for this entire post. On the left, generic autograd: one graph node per primitive op, a saved-tensor pool that grows with model depth, a reverse pass that walks the graph op by op through generic kernels, and a fresh gradient tensor allocated at nearly every node. On the right, Unsloth's hand-derived backward: one fused Triton kernel per block, a `save_for_backward` list that keeps only the irreducible minimum, a closed-form gradient computed in a single reverse pass, and a gradient written *in place* into a buffer that is already on the GPU. Same answer, a fraction of the memory.

This is post 3 in the [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series. If you want the system-level view of *where* these wins land in an end-to-end fine-tune, read the [speedup anatomy post](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) first. This one goes down to the gradient math.

## 1. The cost of generality

PyTorch's autograd is one of the great pieces of systems engineering in modern ML. It lets you write a forward pass as ordinary Python — `x.pow(2).mean().rsqrt() * w` — and get correct gradients for *any* differentiable composition, with no derivation on your part. That generality is the entire value proposition, and it is also the entire cost.

To differentiate an arbitrary program, autograd has to do three things at runtime. First, it **records the computation as a graph** — a "tape" of nodes, one per operation, each remembering which operation it was and which tensors flowed into it. Second, for each node it **saves the inputs needed to compute that node's local gradient** via `save_for_backward`; those tensors are pinned in memory until the backward pass consumes them. Third, when you call `.backward()`, it **walks the graph in reverse**, invoking each node's backward function as a separate generic kernel and threading the upstream gradient through.

None of those three steps is wasteful *for a general program*. But a transformer block is not a general program. It is the same handful of operations — RMSNorm, a couple of projections, a SwiGLU, attention — repeated tens of times. We know the closed-form gradient of every one of them. Paying the autograd tax on a block whose derivative we could write on an index card is the mismatch this post is about.

Three concrete costs follow from the three runtime steps:

| Autograd does this (generically) | The cost on a transformer block |
| --- | --- |
| Records a `Node` per primitive op | Graph bookkeeping over thousands of ops per step |
| `save_for_backward` pins every needed input | Activation memory dominates VRAM; grows with depth × seq-len |
| Reverse pass = one generic kernel per node | Many small launches, each round-tripping HBM, no fusion |

The middle row is the killer. In a training step the *parameters* are a fixed cost — for a QLoRA run the base weights are even frozen in 4 bits — but the *activations* saved for backward scale with batch size, sequence length, and model depth, all at once. That is why the backward pass surprised me with its memory: it was holding a slice of the forward activations at every layer simultaneously, waiting for the reverse walk to reach them.

Unsloth's wager is that for its specific, known blocks it can do better on all three axes by deriving the backward pass by hand. Let me show what each axis costs first, then take it apart block by block.

## 2. How autograd works, and what it costs

To make the savings concrete we need a crisp picture of what autograd actually does at runtime. Take RMSNorm, because it is small enough to hold in your head and it is the first block Unsloth rewrites. The forward, in plain PyTorch, is roughly:

```python
# RMSNorm forward as a generic PyTorch composition.
# Each line is one (or more) autograd Node(s); each saves what its
# backward needs.
var      = x.pow(2).mean(dim=-1, keepdim=True)   # Node: pow, Node: mean
inv_rms  = torch.rsqrt(var + eps)                # Node: add, Node: rsqrt
normed   = x * inv_rms                           # Node: mul  (saves x, inv_rms)
y        = normed * w                            # Node: mul  (saves normed, w)
```

![What autograd does at runtime: each forward op appends a Node to the tape and pins its inputs, and backward replays the tape in reverse op by op.](/imgs/blogs/unsloth-manual-backprop-2.webp)

The figure traces it. The input `X` flows left to right through one node per operation — `pow`, `mean`, `rsqrt`, `mul` — and each node that needs an input for its backward pins it into a saved-tensor pool: `pow` keeps `X`, the final `mul` keeps `normed` and `W`, and so on. That pool is the dashed feed into the bottom of the diagram, and it *grows with the depth of the model*: every block at every layer contributes its own saved tensors, and they all stay resident until the reverse pass reaches them. When you call `.backward()`, autograd reads the pool and the output gradient `dY`, then replays each node in reverse, one generic kernel at a time.

Three observations fall out of this picture, and they are the three levers Unsloth pulls.

**Activations dominate, not parameters.** RMSNorm's only parameter is the weight vector `w` of length `hidden_dim`. But the saved activations — `X`, `normed`, the variance — are each shaped `(batch × seq_len, hidden_dim)`. For a batch of 8 at sequence length 4096 with hidden dimension 4096, `X` alone is `8 × 4096 × 4096 × 2 bytes ≈ 256 MB` in bf16. Multiply by the number of saved activations, by the number of layers, and you see why backward memory balloons. The parameter is a rounding error; the activations are the bill.

**Each backward op is a separate generic kernel.** The reverse pass for that four-line forward is *also* four-plus operations: differentiate the `mul`, the `rsqrt`, the `mean`, the `pow`, each launched as its own CUDA kernel, each reading its operands from HBM and writing its result back to HBM. The arithmetic is trivial; the memory traffic is not. Modern GPUs are bandwidth-bound on operations like this, so the cost is dominated by how many times the data crosses the HBM boundary, and a generic op-by-op reverse pass crosses it many times.

**Most saved tensors are recomputable.** Here is the quiet point that the whole post hinges on. The variance, `normed`, and `inv_rms` are all deterministic functions of `X` and `w`. Autograd saves them because it does not know they are cheap to recompute — it treats every saved tensor as precious because, in general, it might be the output of an expensive sub-network. But for RMSNorm, recomputing `inv_rms` from `X` is a single fused reduction. If you are willing to recompute, you can drop nearly everything from the saved pool.

Generic autograd cannot make that trade for you, because it does not know your block. You can. That is the whole game.

## 3. Manual backward, case 1: RMSNorm

RMSNorm is the cleanest demonstration because the closed-form gradient is short and the memory trick is stark. Unsloth implements it as a custom `torch.autograd.Function` whose `forward` runs a single fused Triton kernel and whose `backward` runs another. Here is the structure of the autograd `Function`, from `unsloth/kernels/rms_layernorm.py`:

```python
class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma = False):
        shape = X.shape; dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = X.device)
        r = torch.empty(n_rows, dtype = torch.float32, device = X.device)  # 1 float/row
        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_gpu_device(X.device):
            fx[(n_rows,)](Y, Y.stride(0), X, X.stride(0), W, W.stride(0),
                          r, r.stride(0), n_cols, eps,
                          BLOCK_SIZE = BLOCK_SIZE, num_warps = num_warps)
        ctx.eps = eps; ctx.BLOCK_SIZE = BLOCK_SIZE; ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)   # save only X, W, and the tiny r
        return Y.view(*shape)
```

The single most important line is `ctx.save_for_backward(X, W, r)`. Compare it to what generic autograd would save for the same computation: `X`, `normed`, the variance, `inv_rms`, and `W` — several full activation tensors. Unsloth saves `X`, `W`, and `r`, where `r` is the per-row inverse RMS — **one float32 per row**, not per element. For our `8 × 4096`-row example, `r` is `32768 × 4 bytes = 128 KB`, against the 256 MB of a single full activation. The forward kernel stores it explicitly with `tl.store(r, inv_var)` precisely so the backward pass does not have to recompute the reduction.

### The closed-form gradient

The forward computes, per row, $\text{normed} = X \cdot r$ where $r = 1/\sqrt{\text{mean}(X^2) + \epsilon}$, then $Y = \text{normed} \cdot W$. Differentiating this by hand (the row-wise RMSNorm Jacobian) gives, for the gradient with respect to the input,

$$
dX = \frac{r}{n}\left(n \cdot (dY \odot W) - \text{normed} \cdot \sum_i (dY \odot W)_i \, \text{normed}_i\right)
$$

where $n$ is `n_cols`, $\odot$ is elementwise product, and the sum is a per-row reduction. This is the entire backward kernel. Here it is verbatim from the source — read it next to the formula:

```python
# _rms_layernorm_backward (Triton kernel, one program per row)
def _rms_layernorm_backward(
    dY, dY_row_stride, dX, dX_row_stride,
    X, X_row_stride, W, W_row_stride, r, r_row_stride,
    n_cols: tl.constexpr, eps: tl.constexpr,
    GEMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY += row_idx * dY_row_stride
    X  += row_idx * X_row_stride
    r  += row_idx * r_row_stride
    if GEMMA: dX += row_idx * dY_row_stride
    else:     dX = dY                        # <-- writes gradient IN PLACE into dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)      # reuse the saved 1/rms — not recomputed
    normed = X_row * inv_var
    dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)
```

Two lines carry the whole memory argument, and the figure below makes them visual.

![RMSNorm backward writes dX in place into dY: the kernel reloads the saved 1/rms, applies the closed-form gradient, and stores the result over the dY buffer.](/imgs/blogs/unsloth-manual-backprop-4.webp)

**`inv_var = tl.load(r)` — reuse, don't recompute.** The variance reduction that autograd's reverse pass would re-derive (or that it pinned a tensor to avoid) is loaded from the 128 KB `r` buffer the forward already wrote. One scalar load per row replaces a full reduction or a full saved activation.

**`dX = dY` — write the gradient in place.** This is the line that surprises people. On the non-Gemma path, `dX` is not a new tensor; the kernel literally aliases `dX` to the incoming `dY` pointer and stores the computed gradient *over* the upstream gradient buffer. The upstream gradient is consumed exactly once (we read `dY_row` before overwriting it), so overwriting it is safe, and the payoff is that the backward pass allocates **zero** new activation-sized tensors. The host-side `backward` confirms it:

```python
    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape; dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dX = torch.empty_like(dY) if ctx.GEMMA else dY   # only Gemma allocates
        with torch_gpu_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](dY, dY.stride(0), dX, dX.stride(0),
                X, X.stride(0), W, W.stride(0), r, r.stride(0),
                n_cols, ctx.eps, GEMMA = ctx.GEMMA,
                BLOCK_SIZE = ctx.BLOCK_SIZE, num_warps = ctx.num_warps)
        dX = dX.view(*shape)
        return dX, None, None, None
```

`dX = torch.empty_like(dY) if ctx.GEMMA else dY`. Outside the Gemma path — which needs a separate buffer because its `(1 + weight)` scaling makes the in-place aliasing unsafe — `dX` *is* `dY`. The fused Triton kernel also collapses the four-plus generic reverse ops into one launch: one program per row does the load, the reduction, the closed-form arithmetic, and the store, with the row's data crossing HBM once. Note `fast_rms_layernorm` is wrapped with `@torch.compiler.disable` — Unsloth deliberately keeps these hand-written kernels out of `torch.compile`, because the manual derivation already does the fusion that the compiler would attempt, and the compiler's graph capture would only get in the way. The [Triton kernel-fusion post](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) covers the forward side of this same kernel in depth.

So for RMSNorm the scorecard is: saved tensors drop from several full activations to `X`, `W`, and a 128 KB scalar buffer; backward kernel launches drop from four-plus to one; new activation allocations in backward drop to zero.

## 4. The chain rule by hand, case 2: the LoRA MLP

RMSNorm is the warm-up. The real workhorse — and the place the manual derivation earns its keep on a QLoRA fine-tune — is the SwiGLU MLP with LoRA adapters on the gate, up, and down projections. This is `LoRA_MLP` in `unsloth/kernels/fast_lora.py`, and Unsloth ships the hand-derivation directly in the docstring. Here it is verbatim (the `###` lines are the comment block from the real source):

```python
### LoRA weights:  W = W + A @ B   (per gate/up/down)
### SwiGLU(X):  e = X@G ; f = e*sigmoid(e) ; g = X@U ; h = f*g ; i = h@W
### Backprop (hand-derived, see Unsloth blog):
df = sigmoid(e)*(1 - f) + f
dC/dW = h.T @ dY
dC/dU = X.T @ (D @ W.T * f)
dC/dG = X.T @ (D @ W.T * df * g)
# down-proj LoRA:  dC/dAw = h.T @ dY @ B.T ;   dC/dBw = A.T @ h.T @ dY
# up-proj LoRA:    dC/dAu = X.T @ (D@W.T * f) @ B.T ;  dC/dBu = A.T @ X.T @ (D@W.T * f)
# gate-proj LoRA:  dC/dAg = X.T @ (D@W.T * df * g) @ B.T ;  dC/dBg = A.T @ X.T @ (D@W.T * df * g)
```

Read the forward line first: `e = X@G`, `f = e·sigmoid(e)` (that is SiLU), `g = X@U`, `h = f·g` (that is SwiGLU), `i = h@W`. The base weights $G$, $U$, $W$ each carry a LoRA adapter $A @ B$, so the effective weight is $W + A@B$. The backward then says: the derivative of the SiLU gate is `df = sigmoid(e)*(1 - f) + f`, the down-projection weight gradient is `h.T @ dY`, and the input contributions thread back through `D @ W.T` (where `D` is the gradient arriving at the MLP output) modulated by `f` for the up path and `df * g` for the gate path. The per-adapter gradients `dC/dA` and `dC/dB` for each of gate/up/down are the same expressions left- and right-multiplied by the *other* adapter factor — which is the whole reason LoRA backward is cheap: the heavy `D@W.T` term is computed once and reused across the adapter grads.

### The backward code, and three memory tricks

The docstring is the math. The code is where the memory wins live. Here is the backward, with the long middle elided (`# ...`) to keep the load-bearing lines in view:

```python
    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (gateW, gateW_quant, gateS, upW, upW_quant, upS,
         downW, downW_quant, downS, _backward_function) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors
        # ...
        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)      # fused SwiGLU backward
        h, df, de = DW, e, g                         # rebind the overwritten buffers

        # LoRA grads via addmm_ (alpha = scaling, beta = 0 -> OVERWRITE, no alloc)
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)
        # ...

        # dX: dequantize the frozen 4-bit base weight on the fly, matmul, free it
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace else None)  # reuse X
        del upW
        dX.addmm_(df @ upB.t(), upA.t(), alpha = upS)
        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW.t()); del gateW
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha = gateS)
        return (dX.view(batch, seq_len, hd), None, None, d_gateA.t(), d_gateB.t(), None, ...)
```

![LoRA SwiGLU MLP hand-derived backward dataflow: dY flows back through the down projection, the fused SwiGLU backward splits it into df and de, and the frozen base weights are dequantized on the fly to accumulate dX.](/imgs/blogs/unsloth-manual-backprop-5.webp)

The figure shows the dataflow: `dY` arrives, goes back through the down projection as `DW = matmul_lora(dY, downW.t(), ...)`, gets split by the fused SwiGLU backward into `df` and `de`, and those accumulate into `dX` from the on-the-fly-dequantized base weights, while the adapter gradients branch off. Three tricks make this lean.

**The `addmm_(..., alpha=scale, beta=0)` trick.** `Tensor.addmm_(M, N, alpha=a, beta=b)` computes `out = b·out + a·(M @ N)` in place. With `beta=0` the existing contents of `out` are *ignored and overwritten* — there is no read of the old value, so it is a fused multiply-then-store, not a multiply-accumulate. Unsloth uses `beta=0` for the *first* write to each adapter-gradient buffer (`d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)`), which means the buffer does not need to be zero-initialized and no temporary is allocated for the matmul result — the product lands directly in `d_downA`. The `alpha=downS` folds the LoRA scaling factor into the same fused op rather than as a separate elementwise multiply.

**`fast_dequantize` then immediate `del`.** The base weight `upW` lives on the GPU as 4-bit NF4 — it is *frozen*, so it carries no gradient and never needs an fp16 copy except transiently for this one matmul. The code dequantizes it (`upW = fast_dequantize(upW.t(), upW_quant)`), uses it once (`torch.matmul(df, upW.t(), ...)`), and immediately `del upW` frees the fp16 reconstruction. The 16-bit weight never persists; at any instant at most one of the three base weights is materialized in fp16. This is the same `fast_dequantize` covered in the [4-bit NF4 quantization post](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) — here it is being used as a *just-in-time* weight source inside backward.

**`out = X if ctx.inplace else None` — buffer reuse for dX.** The first `dX` write is `torch.matmul(df, upW.t(), out = X if ctx.inplace else None)`. When `inplace=True` (the default), the output of the matmul is written *over the saved input activation `X`*. By the time we compute `dX`, the forward's `X` has done its job — it was needed to recompute the projections, but after `DW` and the SwiGLU backward we no longer need it — so its buffer is recycled to hold `dX`. The subsequent `dX.addmm_(...)` calls then accumulate the remaining three chain-rule terms into that same buffer.

That last point deserves its own picture, because the four chain-rule terms for `dX` do not appear all at once — they *accumulate*, one `addmm_` at a time, into a buffer that was an input activation moments earlier.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Four gradient terms accumulate into the dX buffer one after another via successive in-place addmm_ calls" style="width:100%;height:auto;max-width:820px">
<style>
.mb7-buf{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:8}
.mb7-term{fill:var(--accent,#6366f1)}
.mb7-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mb7-code{font:600 13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:start}
.mb7-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes mb7-t1{0%,8%{opacity:0}16%,100%{opacity:1}}
@keyframes mb7-t2{0%,28%{opacity:0}36%,100%{opacity:1}}
@keyframes mb7-t3{0%,50%{opacity:0}58%,100%{opacity:1}}
@keyframes mb7-t4{0%,72%{opacity:0}80%,98%{opacity:1}100%{opacity:0}}
.mb7-a1{animation:mb7-t1 10s ease-in-out infinite}
.mb7-a2{animation:mb7-t2 10s ease-in-out infinite}
.mb7-a3{animation:mb7-t3 10s ease-in-out infinite}
.mb7-a4{animation:mb7-t4 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.mb7-a1,.mb7-a2,.mb7-a3,.mb7-a4{animation:none;opacity:1}}
</style>
<rect class="mb7-buf" x="40" y="60" width="200" height="200" rx="8"/>
<text class="mb7-lbl" x="140" y="48">dX buffer (= reused X)</text>
<rect class="mb7-term mb7-a1" x="40" y="210" width="200" height="50" rx="8"/>
<rect class="mb7-term mb7-a2" x="40" y="160" width="200" height="50" rx="8"/>
<rect class="mb7-term mb7-a3" x="40" y="110" width="200" height="50" rx="8"/>
<rect class="mb7-term mb7-a4" x="40" y="60" width="200" height="50" rx="8"/>
<text class="mb7-code mb7-a1" x="300" y="240">1.  dX = matmul(df, upW.t, out=X)</text>
<text class="mb7-code mb7-a2" x="300" y="190">2.  dX.addmm_(df@upB.t, upA.t)</text>
<text class="mb7-code mb7-a3" x="300" y="140">3.  dX.addmm_(de, gateW.t)</text>
<text class="mb7-code mb7-a4" x="300" y="90">4.  dX.addmm_(de@gateB.t, gateA.t)</text>
<text class="mb7-cap" x="360" y="290">Each addmm_ multiply-accumulates its term into the same dX buffer.</text>
</svg>
<figcaption>The four chain-rule terms for dX land one after another into a single reused buffer: the first matmul overwrites it (out=X), then three addmm_ calls accumulate the LoRA-adapter and base-weight contributions in place.</figcaption>
</figure>

Step 1 overwrites the buffer with the up-projection base term (`out=X`, `beta`-equivalent overwrite). Steps 2 through 4 are accumulating `addmm_` calls: the up-adapter term, the gate base term, and the gate-adapter term, each adding into the same `dX`. Four chain-rule contributions, one buffer, zero extra activation-sized allocations. The `del upW` and `del gateW` between the steps keep the transient fp16 base weights from coexisting.

## 5. What gets saved versus what autograd would save

Now we can be precise about the saved-tensor list, which is the heart of the memory win. The `LoRA_MLP.forward` ends with:

```python
        ctx.custom_saved_tensors = (gateW, gateW_quant, gateS, upW, upW_quant, upS,
                                    downW, downW_quant, downS, _backward_function)
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i
```

The `save_for_backward` list is `(gateA, gateB, upA, upB, downA, downB, X, e, g)`. Look at what is *and is not* there:

- **The six LoRA adapters** `gateA, gateB, upA, upB, downA, downB`. These are tiny — rank `r` (default 64) factors, so each is `hidden × 64` or `64 × hidden`, a sliver of a full weight. They carry the gradients we actually want.
- **`X`**, the block input. Needed to recompute projections and as the recyclable `dX` buffer.
- **`e` and `g`**, the two SwiGLU pre-products (`e = X@gate`, `g = X@up`). These are the *minimum* needed to reconstruct everything else in the SwiGLU backward.
- **Not saved:** `f` (the SiLU output), `h` (the SwiGLU output), and every matmul intermediate. Those are *recomputed* from `e` and `g` inside the fused SwiGLU backward kernel.

The base weights `gateW, upW, downW` ride in `custom_saved_tensors` alongside their quant states, but they are the frozen 4-bit weights — stored once, not per-activation, and never differentiated. The grid below lines this up against what generic autograd would have pinned.

![Saved tensors: what autograd keeps versus what Unsloth keeps. For both RMSNorm and the LoRA MLP, Unsloth's ctx.save_for_backward holds a tiny subset; the rest is recomputed in the backward kernel.](/imgs/blogs/unsloth-manual-backprop-3.webp)

For RMSNorm, autograd would keep `X, normed, var, 1/rms, W` — several full activations — against Unsloth's `X, W, r`. For the LoRA MLP, autograd would keep `e, f, g, h` *and every matmul intermediate* inside `matmul_lora` — against Unsloth's six adapters plus `X, e, g`. The memory delta is not subtle. Each unsaved full activation is, in our running example, on the order of a hundred-plus megabytes; dropping `f`, `h`, and the projection intermediates across every layer is where the "70% less VRAM" comes from. The trade is recomputation in backward — but that recomputation is fused into the kernel that was going to run anyway, so it is nearly free in wall-clock terms, paid for many times over in memory headroom.

This is the same idea as gradient checkpointing, taken to a block-specific extreme. Checkpointing recomputes a block's *internal* activations in backward instead of saving them; Unsloth's hand-derived backward recomputes them too, but it also knows the *exact* minimal residue (`e`, `g`) needed to do so, instead of re-running the whole block. The series' checkpointing-and-offload post covers the complementary trick — moving the block's *input* off the GPU entirely.

## 6. Fused SwiGLU backward in one pass

The `_backward_function` in the LoRA backward is `swiglu_DWf_DW_dfg_kernel`, the fused SwiGLU backward from `unsloth/kernels/swiglu.py`. It is the engine that turns the saved `e` and `g` and the incoming `DW` into the three derivatives `h`, `df`, `de` — in a single pass, overwriting its inputs. Here is the kernel:

```python
@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE, LONG_INDEXING):
    """ writes back, in place: h=f*g into DW; df=DW*f into e; de into g """
    # ... offset/mask setup ...
    se_row = tl.sigmoid(e_row)
    f_row  = (se_row * e_row).to(DW_row.dtype)         # SiLU(e), recomputed from e
    h_row  = f_row * g_row                             # SwiGLU output
    df_row = DW_row * f_row                            # grad through the SiLU branch
    dg_row = DW_row * g_row
    de_row = (dg_row.to(tl.float32) * se_row * (1.0 + e_row*(1.0 - se_row))).to(DW_row.dtype)
    tl.store(DW + offsets, h_row,  mask=mask)          # DW <- h
    tl.store(e  + offsets, df_row, mask=mask)          # e  <- df
    tl.store(g  + offsets, de_row, mask=mask)          # g  <- de
```

Notice that `f`, which autograd would have saved, is *recomputed* on line 2 as `se_row * e_row` straight from the saved `e`. That single recomputation is what licenses dropping `f` from the saved-tensor list. And the three `tl.store` calls write `h`, `df`, and `de` back over the `DW`, `e`, and `g` buffers respectively — the same three tensors the kernel read from. There is no fourth allocation for the gradients.

<figure class="blog-anim">
<svg viewBox="0 0 640 260" role="img" aria-label="Three buffers DW, e, g are overwritten in place by the fused SwiGLU backward kernel with h, df, de" style="width:100%;height:auto;max-width:760px">
<style>
.mb8-in{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mb8-out{fill:var(--accent,#6366f1)}
.mb8-name{font:600 16px ui-monospace,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mb8-val{font:600 15px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.mb8-valin{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mb8-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes mb8-fadeIn{0%,38%{opacity:1}52%,92%{opacity:0}100%{opacity:1}}
@keyframes mb8-fadeOut{0%,38%{opacity:0}52%,92%{opacity:1}100%{opacity:0}}
.mb8-A{animation:mb8-fadeIn 9s ease-in-out infinite}
.mb8-B{animation:mb8-fadeOut 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.mb8-A{animation:none;opacity:0}.mb8-B{animation:none;opacity:1}}
</style>
<text class="mb8-name" x="100" y="40">DW</text>
<text class="mb8-name" x="320" y="40">e</text>
<text class="mb8-name" x="540" y="40">g</text>
<g class="mb8-A">
<rect class="mb8-in" x="40" y="60" width="120" height="90" rx="8"/>
<rect class="mb8-in" x="260" y="60" width="120" height="90" rx="8"/>
<rect class="mb8-in" x="480" y="60" width="120" height="90" rx="8"/>
<text class="mb8-valin" x="100" y="110">dY @ downW.t</text>
<text class="mb8-valin" x="320" y="110">X @ gate</text>
<text class="mb8-valin" x="540" y="110">X @ up</text>
<text class="mb8-cap" x="320" y="200">before: input buffers (read by the kernel)</text>
</g>
<g class="mb8-B">
<rect class="mb8-out" x="40" y="60" width="120" height="90" rx="8"/>
<rect class="mb8-out" x="260" y="60" width="120" height="90" rx="8"/>
<rect class="mb8-out" x="480" y="60" width="120" height="90" rx="8"/>
<text class="mb8-val" x="100" y="110">h = f*g</text>
<text class="mb8-val" x="320" y="110">df = DW*f</text>
<text class="mb8-val" x="540" y="110">de</text>
<text class="mb8-cap" x="320" y="200">after: same buffers, overwritten with the derivatives</text>
</g>
<text class="mb8-cap" x="320" y="235">_DWf_DW_dfg_kernel: one pass, three derivatives, zero new tensors</text>
</svg>
<figcaption>The fused SwiGLU backward reads DW, e, and g, then stores h, df, and de back over those exact three buffers; no fourth allocation appears for the gradients.</figcaption>
</figure>

The animation shows the buffer reuse as a before/after: the three slots hold `dY @ downW.t`, `X @ gate`, and `X @ up` going in, and `h`, `df`, `de` coming out — same memory, new contents. The static map of the same kernel makes the three closed-form expressions explicit:

![Fused SwiGLU backward: three derivatives, zero new buffers. The kernel reads DW, e, g in one pass and stores h, df, de back over those exact buffers.](/imgs/blogs/unsloth-manual-backprop-6.webp)

This is why the host-side `LoRA_MLP.backward` rebinds the result as `h, df, de = DW, e, g` immediately after calling `_backward_function` — the variable names change but the storage does not move. Compare with what autograd would do: separate kernels for the SiLU derivative, the product-rule terms of the SwiGLU, and the elementwise multiplies, each with its own intermediate tensor and HBM round-trip. The fused kernel does the arithmetic of all of them in one pass over the data, recomputing `f` and `se` on the fly rather than reading them back from saved tensors.

## 7. Is it still correct?

Here is the question every reader should be asking by now: if you throw away autograd and write the backward by hand, how do you know you got the chain rule right? A hand-derived gradient with a sign error or a dropped term will train — it will just train *wrong*, silently, and you will spend a week blaming your data.

The answer is the trust contract, and it is the same one PyTorch itself uses to validate custom `Function`s: **gradient-check against autograd.** Unsloth's tests apply the fused kernel and a reference implementation to the same input, then assert that both the output *and* the input gradient match within a numerical tolerance. The RMSNorm test has the shape:

```python
# test_rms_layernorm (shape of Unsloth's gradient-check)
def test_rms_layernorm():
    X = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    ref = LlamaRMSNorm(4096).cuda().bfloat16()      # stock PyTorch RMSNorm

    # forward: fused vs reference
    Y_fast = fast_rms_layernorm(ref, X)
    Y_ref  = ref(X)
    torch.testing.assert_close(Y_fast, Y_ref, atol=0.05, rtol=0.05)

    # backward: gradient of the fused path vs autograd through the reference
    Y_fast.sum().backward(); g_fast = X.grad.clone(); X.grad = None
    Y_ref.sum().backward();  g_ref  = X.grad.clone()
    torch.testing.assert_close(g_fast, g_ref, atol=0.05, rtol=0.05)
```

The reference path runs *through autograd* — `Y_ref.sum().backward()` invokes PyTorch's generic reverse pass on a stock `LlamaRMSNorm`. The fast path runs the hand-derived kernel. The assertion is that the two gradients agree to roughly `0.05` absolute/relative tolerance, which is the bf16 rounding floor for an operation of this size, not an approximation budget. The point is decisive: **the manual backward is not an approximation of the gradient — it is the gradient, computed by a different schedule.** Every memory and speed trick in this post — recomputing `f`, reusing `r`, aliasing `dX` to `dY`, dequantizing-then-deleting the base weight — changes *when and where* tensors live, never *what value* the gradient takes.

This is the core of Unsloth's "zero accuracy loss" claim, and it is worth stating plainly because the QLoRA literature is full of methods that *do* trade accuracy for memory. Unsloth does not. The NF4 base weights are dequantized to exactly the same fp16 values bitsandbytes would produce; the fused kernels are algebraic rewrites of the same closed-form gradient; the in-place writes touch buffers whose old contents are provably dead. If a hand-derived kernel had a bug, the gradient-check would catch it against autograd's ground truth before the kernel ever shipped. That is the contract that lets Unsloth replace autograd without asking you to trust it on faith.

It also explains why the tolerance is set where it is. The `0.05` figure is not a knob you tune until the test passes — it is the expected disagreement between two bf16 reduction orders summing the same values. Run the same check in fp32 and the tolerance tightens to near machine epsilon, because the only remaining difference is floating-point reassociation, not the math. This is the diagnostic to keep in your back pocket: if you ever suspect a fused kernel of an actual algebraic error rather than a rounding difference, rerun the gradient-check in fp32. A genuine bug stays large as the dtype widens; a rounding artifact collapses. Autograd, by being the slow-but-obviously-correct reference, is what makes that distinction observable at all.

## 8. Case studies: when hand-derived backward pays off

The pattern generalizes beyond these two blocks. Here are the situations where I have seen manual backprop earn its complexity, and where it does not.

**1. RMSNorm and LayerNorm in every transformer layer.** This is the canonical win. The norm appears twice per layer, its activations are full-width, and its closed-form gradient is short. Saving one float32 per row instead of multiple full activations, fused into one kernel with an in-place write, is pure upside — no recomputation cost worth mentioning, and the in-place `dX = dY` is safe on every non-Gemma model. If you implement one manual backward, implement this one.

**2. The SwiGLU/GeGLU MLP under LoRA.** The MLP is the largest single consumer of activation memory in a transformer (the intermediate is typically 2.7–4× the hidden dimension). Dropping `f` and `h` and every matmul intermediate, recomputing them from the minimal `e`/`g` residue, is where the bulk of the QLoRA VRAM saving lands. The `addmm_`/dequantize/`del`/in-place sequence is intricate, but it is intricate *once*, in a kernel that runs at every layer.

**3. Cross-entropy over a large vocabulary.** Covered in its own series post, but it belongs on this list: a naive `F.cross_entropy` materializes a full `(batch·seq, vocab)` softmax tensor — for Gemma's 256K vocab at long sequence length that is billions of floats. The manual backward saves only the per-row logsumexp (one float per token) and recomputes the softmax on the fly in the backward kernel, writing the gradient back over the logits buffer. Same in-place, recompute-don't-save philosophy; even more dramatic memory delta because the vocab dimension is so large.

**4. RoPE, the orthogonal-rotation case.** RoPE is the prettiest manual backward in the codebase: because the rotation matrix is orthogonal, its transpose is the negative-angle rotation, so the *backward kernel is the forward kernel with `sin → -sin`*. No separate derivation, no saved activations beyond the cached cos/sin tables. When your operation has a clean algebraic inverse, the manual backward can be nearly free to write and to run.

**5. When NOT to hand-derive: a one-off custom layer you are still iterating on.** Manual backward is a maintenance commitment. Every time you change the forward you must re-derive the backward and re-run the gradient-check, or you ship a silent training bug. For a block you are still designing — where the forward changes weekly — let autograd do its job. The whole value of autograd is that it tracks your changes for free. Reach for manual backward only once the forward is *frozen* and the block is *hot*.

**6. When NOT to hand-derive: a block that is not memory- or launch-bound.** If a block's saved activations are small and it already fuses into a couple of kernels under `torch.compile`, the manual derivation buys little and costs the maintenance. Profile first. The reason RMSNorm, the SwiGLU MLP, and cross-entropy are worth it is that they are simultaneously *frequent*, *activation-heavy*, and *launch-heavy* — all three. A block missing any of those three rarely justifies the hand-derivation.

**7. The Gemma exception, as a cautionary tale.** Notice that the in-place `dX = dY` trick is gated on `if GEMMA` precisely *not* taking the in-place path. Gemma's RMSNorm uses a `(1 + weight)` scaling that makes the aliasing arithmetic unsafe, so Unsloth allocates a separate `dX` for it. The lesson: in-place gradient writes are correct only when the old buffer contents are genuinely dead at the moment you overwrite. Get that wrong and you corrupt the gradient. The Gemma branch is the codebase admitting, in one `if`, exactly where the in-place trick stops being safe — which is itself a small endorsement of the gradient-check discipline that found the boundary.

### When to reach for hand-derived backward, and when not

Reach for it when a block is **frozen in design, hot in the training loop, and heavy in either saved-activation memory or kernel-launch count** — RMSNorm, the LoRA SwiGLU MLP, cross-entropy, RoPE. In that intersection, the three autograd costs from Section 1 are all live, and the three manual-backward wins — save only the residue, fuse the reverse into one kernel, write the gradient in place — all land. The payoff is real VRAM headroom (the difference between a model fitting on your GPU and not) and real wall-clock speedup, with *identical* gradients verified against autograd.

Do not reach for it on a block you are still changing, or on one that is neither memory- nor launch-bound. Autograd's generality is a feature there, not a tax. The art is knowing which of your blocks are in the frozen-hot-heavy intersection — and Unsloth's source is, in effect, a curated list of exactly those blocks for a transformer, with the hand-derivation done and gradient-checked for you. The next time your backward pass eats more memory than your forward, you will know it is not a leak. It is autograd remembering. And you will know exactly which blocks are worth teaching to forget.

For the rest of the series: the [overview post](/blog/machine-learning/open-source-library/unsloth-lib) frames where these kernels sit, the [speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) shows the end-to-end numbers, the [Triton kernel-fusion post](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) covers the forward side of these same kernels, and the [4-bit NF4 quantization post](/blog/machine-learning/open-source-library/unsloth-4bit-quantization-nf4) explains the `fast_dequantize` that feeds the LoRA backward.
