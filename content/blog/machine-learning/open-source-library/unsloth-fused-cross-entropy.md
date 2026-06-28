---
title: "The Cross-Entropy Memory Wall: How Unsloth Never Materializes the Softmax"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "For a large-vocabulary model the logits tensor and the softmax computed over it are the single biggest hidden memory cost of a training step. Unsloth's fused Triton cross-entropy stores one logsumexp float per token, recomputes the softmax on the fly in backward, and writes the gradient in place over the logits buffer — never allocating a softmax tensor."
tags: ["unsloth", "cross-entropy", "logits", "softmax", "triton", "memory-optimization", "logsumexp", "vocabulary", "qlora", "kernels"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 29
---

The first time I OOM'd a fine-tune at the very last layer, I was certain I had misconfigured the optimizer. I had a 9B model loaded in 4-bit, LoRA adapters that fit in a couple hundred megabytes, gradient checkpointing on, a batch that was almost embarrassingly small — and the run still died, reliably, the instant the forward pass reached the language-model head. The traceback pointed at `cross_entropy`. The optimizer was innocent. The killer was a single tensor I had never thought about: the logits.

This is the cross-entropy memory wall, and for large-vocabulary models it is the most underestimated line item in a training step. The weights are quantized, the activations are checkpointed, the optimizer state for LoRA is tiny — and then the final projection produces a `(batch × seq × vocab)` block of floats that can, on its own, dwarf everything else on the device. Worse, the naive way to turn those logits into a loss allocates a *second* tensor of the same shape. You pay for the wall twice.

![Where the VRAM actually goes at the loss step: the logits tensor and a naive softmax copy are each ~8 GB for a 256K-vocab model, dwarfing the 4-bit weights, the LoRA optimizer state, and the checkpointed activations.](/imgs/blogs/unsloth-fused-cross-entropy-1.webp)

The diagram above is the mental model for this entire post. On the left, the things we usually worry about: 4-bit weights, LoRA optimizer state, checkpointed activations — all under a couple of gigabytes. In the middle, the thing we don't: the logits tensor, and right below it the log-softmax copy that a textbook cross-entropy materializes. On the right, what Unsloth keeps instead — one logsumexp float per token, around 32 KB total. The two enormous blocks in the middle simply never get allocated. This post is about exactly how that collapse works, why it is mathematically *exact* and not an approximation, and why the trick is one of the cleanest examples of "derive the math, then refuse to store what the math doesn't need" in the whole Unsloth codebase.

This is part of the [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series. If you want the system-level view of where these wins land across a whole step, read the [speedup anatomy post](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) first; the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop) covers the in-place-gradient theme in general, and the [Triton kernel-fusion post](/blog/machine-learning/open-source-library/unsloth-triton-kernel-fusion) covers the kernel mechanics. This one drills into one kernel: `unsloth/kernels/cross_entropy_loss.py`.

## 1. The memory math of logits

Let us put numbers on the wall, because the wall is entirely arithmetic and the arithmetic is unforgiving.

A transformer's language-model head produces a logit for every vocabulary token at every sequence position. If the (flattened) batch-times-sequence length is $N$ and the vocabulary size is $V$, the logits tensor has $N \times V$ elements. In mixed-precision training the loss is almost always computed in fp32 for numerical stability, so each element is 4 bytes. The tensor is therefore

$$
\text{size}_{\text{logits}} = N \times V \times 4 \ \text{bytes}.
$$

Plug in a long-context Gemma-style run: sequence length 8192, batch 1, so $N = 8192$, and $V = 256000$. That is

$$
8192 \times 256000 \times 4 \ \text{B} \approx 8.4 \times 10^9 \ \text{B} \approx 8 \ \text{GB}.
$$

Eight gigabytes for a single intermediate tensor, on a model whose 4-bit weights are about 1.5 GB. The logits are more than five times the size of the weights. This is not a pathological corner case; it is what 256K-vocabulary models do at long context, and the trend in frontier models is toward *larger* vocabularies, not smaller, because a bigger vocabulary buys better tokenization efficiency.

Now the part that makes it a wall rather than a bump. The textbook way to compute cross-entropy from logits is `F.log_softmax(logits, dim=-1)` followed by a gather of the label entries. `log_softmax` is an elementwise transform: it reads the `(N × V)` logits and writes an `(N × V)` log-probability tensor. PyTorch does not write that result back over the input — the input logits are still needed for the backward pass, so autograd keeps both alive. You now hold *two* vocab-wide fp32 tensors at peak: the logits and the log-softmax. For our Gemma example that is roughly 16 GB just to turn a forward pass into a scalar loss.

| Quantity | Symbol | Gemma (V=256K, N=8192) | Llama-3 (V=128K) | Mistral (V=32K) |
|---|---|---|---|---|
| Logits tensor | $N V \cdot 4$ | ~8 GB | ~4 GB | ~1 GB |
| Naive log-softmax copy | $N V \cdot 4$ | ~8 GB | ~4 GB | ~1 GB |
| Naive peak (both live) | $2 N V \cdot 4$ | ~16 GB | ~8 GB | ~2 GB |
| Logsumexp vector | $N \cdot 4$ | ~32 KB | ~32 KB | ~32 KB |

The last row is the punchline waiting to happen. The per-token logsumexp — one fp32 scalar for each of the $N$ tokens — is $8192 \times 4 = 32{,}768$ bytes. Thirty-two kilobytes. Against a tensor it can stand in for, that is a rounding error. The entire game is to notice that backward needs only this scalar per row, never the full log-softmax tensor.

A note on why fp32. You could compute the loss in bf16 to halve the tensor, but the sum-of-exponentials inside softmax is exactly the kind of reduction that loses catastrophic precision in 16-bit — small logits underflow, the normalizer drifts, and your loss curve develops a mysterious noise floor. Every serious implementation, Unsloth included, accumulates the cross-entropy reduction in fp32. So the wall is an fp32 wall, and there is no cheap escape by lowering the dtype.

## 2. Cross-entropy from first principles

To see why one scalar suffices, derive the loss from scratch. For a single token, the model emits a logit vector $x \in \mathbb{R}^V$. The softmax turns it into a probability distribution:

$$
P_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}.
$$

The cross-entropy loss against a one-hot target at index $t$ (the true next token) is $-\log P_t$. Expand it:

$$
\text{CE} = -\log P_t = -\log \frac{e^{x_t}}{\sum_j e^{x_j}} = -\left( x_t - \log \sum_j e^{x_j} \right) = \log\!\sum_j e^{x_j} \; - \; x_t.
$$

Define the **logsumexp** of the row:

$$
\text{lse}(x) = \log \sum_j e^{x_j}.
$$

Then the loss collapses to a strikingly small expression:

$$
\boxed{\ \text{CE} = \text{lse}(x) - x_t\ }
$$

This is the whole forward pass, conceptually. You do **not** need the softmax distribution to compute the loss. You need exactly two numbers per token: the scalar $\text{lse}(x)$, which is a reduction over the whole row, and $x_t$, the single logit at the label index. The probability vector $P$ is a means to an end that we can skip entirely.

There is one subtlety that every numerical-stability lecture hammers: $\sum_j e^{x_j}$ overflows the moment any $x_j$ is even modestly large, because $e^{x_j}$ blows up. The fix is the **max-shift**. Let $c = \max_j x_j$. Then

$$
\text{lse}(x) = c + \log \sum_j e^{x_j - c}.
$$

Subtracting the row max before exponentiating guarantees the largest term is $e^0 = 1$ and every other term is in $(0, 1]$, so the sum is bounded and well-conditioned. This identity is algebraically exact — pulling $c$ out of the log is just $\log(e^c \cdot S) = c + \log S$. It costs one extra pass over the row to find the max, which a Triton reduction does for free alongside the sum.

![What the loss needs from a logit row: a whole vocab-wide row reduces through the stable logsumexp formula to one float per token.](/imgs/blogs/unsloth-fused-cross-entropy-2.webp)

The figure above is the reduction the kernel performs for every token. The full 256000-value row goes in; the max-shifted logsumexp comes out as a single float; the loss is that float minus the logit at the label column. The row itself is never copied into a softmax tensor. What gets *stored* for later is just `logsumexp_ptr[row]` — one scalar — and that is the seed from which the backward pass will reconstruct everything it needs.

Here is the gradient, which is where the saved scalar earns its keep. Differentiating $\text{CE} = \text{lse}(x) - x_t$ with respect to an arbitrary logit $x_i$:

$$
\frac{\partial \,\text{CE}}{\partial x_i} = \frac{\partial \,\text{lse}(x)}{\partial x_i} - \frac{\partial x_t}{\partial x_i} = P_i - \mathbb{1}[i = t] = \frac{e^{x_i}}{\sum_j e^{x_j}} - \mathbb{1}[i = t].
$$

The derivative of logsumexp is the softmax — that is a clean, famous identity — and the derivative of the label term is one only at the label column. So the gradient is "softmax minus the one-hot," the well-known result. The thing to notice for memory purposes: $P_i = e^{x_i} / \sum_j e^{x_j} = e^{x_i - \text{lse}(x)}$. Given the saved logsumexp scalar and the original logit $x_i$, you can *recompute* any softmax entry on demand with a single `exp`. You never had to store the softmax during forward; backward rebuilds it from the scalar.

## 3. The fused forward kernel

Now the real code. Unsloth's forward kernel, `_cross_entropy_forward`, is a Triton kernel that runs one program instance per token (per logit row) and does the entire reduction in registers and shared memory, writing back only the loss and the logsumexp scalar.

```python
MAX_FUSED_SIZE = 65536  # 2**16  -> the chunk width

@triton.jit
def _cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr, logsumexp_ptr, labels_ptr,
    VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, DO_SOFTCAPPING, SOFTCAP,
    DO_LOGIT_SCALING, LOGIT_SCALE):
    """
    CE_i = -y log(Pi),  Pi = exp(xi)/sum(exp(xi))
         = y * (logsumexp(x) - x)
    logsumexp stable form:  c = max(x);  lse = c + log(sum(exp(x - c)))
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr + row_idx, logsumexp)   # store ONE float per row, not the softmax
    tl.store(loss_ptr + row_idx, loss)
```

Read it line by line, because every line maps directly to the math above.

`row_idx = tl.program_id(0)` — the kernel is launched on a 1-D grid of size `n_rows`, so each program owns exactly one token's logit row. `logits_ptr += row_idx * logits_row_stride` advances the pointer to the start of that row; the cast to `tl.int64` matters because $N \times V$ for a big model overflows 32-bit indexing, and a silent integer wraparound here would corrupt the loss for tokens past the 2-billion-element mark.

`col_offsets = tl.arange(0, BLOCK_SIZE)` with `mask = col_offsets < VOCAB_SIZE` loads the whole row in one vectorized `tl.load`, padding past the true vocab with `-inf`. The `-inf` is deliberate: `exp(-inf) = 0`, so the padding contributes nothing to the sum, and it can never be the max. The row is immediately upcast `.to(tl.float32)`.

Then the three lines that *are* the math:

- `c = tl.max(logits, 0)` — the row max, the stability shift.
- `logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))` — the stable logsumexp, exactly the boxed identity from the previous section, computed as a register-level reduction over the block.
- `loss = logsumexp - x` where `x` is the single logit at `label_idx` — the boxed $\text{CE} = \text{lse}(x) - x_t$.

The `-100` branch is PyTorch's `ignore_index` convention. Padding tokens and prompt tokens you do not want to train on are labeled `-100`; for those the loss is forced to `0.0` and they contribute nothing to the gradient. Handling it inside the kernel means no separate masking pass over the loss vector afterward.

Finally, the two stores. `tl.store(logsumexp_ptr + row_idx, logsumexp)` writes one fp32 scalar — the seed for backward. `tl.store(loss_ptr + row_idx, loss)` writes the per-token loss. **Nowhere does this kernel write an `(N × V)` tensor.** It reads the logits, reduces them to two scalars, and the softmax distribution exists only transiently inside `tl.exp(logits - c)`, in registers, for the duration of the `tl.sum`. There is no `log_softmax` allocation because there is no `log_softmax` call.

Compare the data movement, too. A naive PyTorch path reads the logits for the max, reads them again for the exp-sum, allocates and writes the full log-softmax (another $NV$ writes plus reads), then gathers. Unsloth reads each logit row once into the kernel and emits two scalars. The memory traffic drops from "several passes over a vocab-wide tensor plus a vocab-wide allocation" to "one pass, two scalar writes." That is also why it is *faster*, not only lighter — cross-entropy on a big vocab is bandwidth-bound, and the fused kernel moves far less data across HBM.

![Naive F.cross_entropy versus Unsloth fused cross-entropy on the same logits: the naive path allocates a second vocab-wide log_softmax tensor; the fused path keeps only a logsumexp vector and a loss vector and overwrites the logits in place.](/imgs/blogs/unsloth-fused-cross-entropy-3.webp)

The figure above sets the two side by side on the same input. Both start from the identical logits tensor. The naive path (left) adds the red full-vocab `log_softmax` copy — the wall, doubled. The fused path (right) adds only the two small green vectors, `logsumexp` and `loss`, each `seq` floats, and later overwrites the logits with the gradient in place. Same loss out of both columns; one of them allocates a tensor the other never does.

## 4. The chunked path for huge vocabularies

There is a hardware limit lurking in that elegant single-pass kernel. A Triton program loads `BLOCK_SIZE` elements, and `BLOCK_SIZE` is bounded by what fits in a block's registers and shared memory. Unsloth caps it at `MAX_FUSED_SIZE = 65536` — that is $2^{16}$, the chunk width. For a vocabulary of 65536 or fewer tokens (Mistral's 32K, for instance) the whole row fits in one block and the single-pass kernel above runs. But anything larger — Llama-3's 128K, and certainly Gemma's 256000 — does not fit in a 65536-element block. You cannot load the whole row at once.

The naive reaction is "then go back to PyTorch for big vocabs." Unsloth instead splits the row into chunks of `MAX_FUSED_SIZE` and exploits an exact identity that lets the chunks be reduced independently and then combined. This is the chunked-logsumexp identity, and it is the mathematical heart of the big-vocab path.

**The identity.** Logsumexp is associative under concatenation. If you split a vector $x$ into chunks $x^{(1)}, x^{(2)}, \dots, x^{(k)}$, then

$$
\text{lse}(x) = \log \sum_{\text{all } j} e^{x_j} = \log \sum_{m=1}^{k} \sum_{j \in \text{chunk } m} e^{x_j} = \log \sum_{m=1}^{k} e^{\,\text{lse}(x^{(m)})}.
$$

The middle step just regroups the sum by chunk; the last step uses $\sum_{j \in m} e^{x_j} = e^{\text{lse}(x^{(m)})}$, which is the definition of logsumexp run backward. So:

$$
\boxed{\ \text{lse}(x) = \text{lse}\big(\big[\,\text{lse}(x^{(1)}), \dots, \text{lse}(x^{(k)})\,\big]\big)\ }
$$

**The logsumexp of the per-chunk logsumexps is the global logsumexp.** This is exact — no approximation, no error term. The Unsloth source states it plainly in the docstring: `logsumexp(concat chunks) == logsumexp([logsumexp(chunk_1), ..., logsumexp(chunk_k)])`.

![Chunked logsumexp for vocab > 65536: each chunk of 65536 logits produces one logsumexp scalar, and a second torch.logsumexp over those scalars recovers the exact global value.](/imgs/blogs/unsloth-fused-cross-entropy-4.webp)

The figure shows the reduction tree for Gemma's 256000-token vocab split into four chunks. Each blue chunk is processed by one program instance of the chunked kernel and emits one amber per-chunk logsumexp; the four scalars stack into a small `(n_rows × n_chunks)` tensor; a single `torch.logsumexp` over the chunk dimension produces the exact global logsumexp, one float per token. The intermediate `logsumexp` tensor here is `n_rows × n_chunks` — for Gemma that is `8192 × 4` floats, about 128 KB, still negligible.

The same combination, watched as a flow, makes the "scalars in, one scalar out" structure obvious:

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Four per-chunk logsumexp values flow into a single reducer that emits the exact global logsumexp" style="width:100%;height:auto;max-width:820px">
<title>Per-chunk logsumexps reduce to the exact global logsumexp</title>
<style>
.ce2-chunk{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:6}
.ce2-red{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2.5;rx:8}
.ce2-out{fill:var(--accent,#6366f1);stroke:none;rx:8}
.ce2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ce2-out-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--background,#fff);text-anchor:middle}
.ce2-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ce2-dot{fill:var(--accent,#6366f1)}
@keyframes ce2-flow0{0%{transform:translate(0,0);opacity:0}12%{opacity:1}48%{transform:translate(220px,90px);opacity:1}60%,100%{transform:translate(220px,90px);opacity:0}}
@keyframes ce2-flow1{0%{transform:translate(0,0);opacity:0}12%{opacity:1}48%{transform:translate(80px,90px);opacity:1}60%,100%{transform:translate(80px,90px);opacity:0}}
@keyframes ce2-flow2{0%{transform:translate(0,0);opacity:0}12%{opacity:1}48%{transform:translate(-80px,90px);opacity:1}60%,100%{transform:translate(-80px,90px);opacity:0}}
@keyframes ce2-flow3{0%{transform:translate(0,0);opacity:0}12%{opacity:1}48%{transform:translate(-220px,90px);opacity:1}60%,100%{transform:translate(-220px,90px);opacity:0}}
@keyframes ce2-emit{0%,55%{opacity:0;transform:scale(.6)}70%,100%{opacity:1;transform:scale(1)}}
.ce2-d0{animation:ce2-flow0 6s ease-in infinite}
.ce2-d1{animation:ce2-flow1 6s ease-in infinite}
.ce2-d2{animation:ce2-flow2 6s ease-in infinite}
.ce2-d3{animation:ce2-flow3 6s ease-in infinite}
.ce2-emit{transform-box:fill-box;transform-origin:center;animation:ce2-emit 6s ease-out infinite}
@media (prefers-reduced-motion:reduce){.ce2-d0,.ce2-d1,.ce2-d2,.ce2-d3{animation:none;opacity:1}.ce2-emit{animation:none;opacity:1;transform:none}}
</style>
<rect class="ce2-chunk" x="40"  y="40" width="120" height="60"/>
<rect class="ce2-chunk" x="200" y="40" width="120" height="60"/>
<rect class="ce2-chunk" x="360" y="40" width="120" height="60"/>
<rect class="ce2-chunk" x="520" y="40" width="120" height="60"/>
<text class="ce2-lbl" x="100" y="76">lse_0</text>
<text class="ce2-lbl" x="260" y="76">lse_1</text>
<text class="ce2-lbl" x="420" y="76">lse_2</text>
<text class="ce2-lbl" x="580" y="76">lse_3</text>
<rect class="ce2-red" x="260" y="160" width="200" height="60"/>
<text class="ce2-lbl" x="360" y="196">torch.logsumexp</text>
<circle class="ce2-dot ce2-d0" cx="100" cy="100" r="8"/>
<circle class="ce2-dot ce2-d1" cx="260" cy="100" r="8"/>
<circle class="ce2-dot ce2-d2" cx="420" cy="100" r="8"/>
<circle class="ce2-dot ce2-d3" cx="580" cy="100" r="8"/>
<rect class="ce2-out ce2-emit" x="280" y="244" width="160" height="40"/>
<text class="ce2-out-lbl ce2-emit" x="360" y="269">global lse</text>
<text class="ce2-cap" x="360" y="140">exact: lse([lse_0..lse_3]) == lse over the full vocab row</text>
</svg>
<figcaption>Each chunk emits one logsumexp scalar; a second torch.logsumexp over the four scalars yields the exact global logsumexp for the row. The chunked path changes the memory footprint, not the answer.</figcaption>
</figure>

The forward dispatch picks the path by counting chunks. Here is the branch from `Fast_CrossEntropyLoss.forward`:

```python
class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape
        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        if n_chunks == 1:
            # vocab <= 65536 (Llama, Mistral): single pass
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
            _cross_entropy_forward[(n_rows,)](logits, logits.stride(0), losses, logsumexp, labels,
                VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE, ...)
        else:
            # vocab > 65536 (Gemma 256K): chunked logsumexp then reduce
            logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=logits.device)
            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](logits, logits.stride(0), losses,
                logsumexp, labels, VOCAB_SIZE=vocab_size, N_CHUNKS=n_chunks, BLOCK_SIZE=MAX_FUSED_SIZE, ...)
            logsumexp = torch.logsumexp(logsumexp, dim=1)   # logsumexp of per-chunk logsumexps == global lse
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses
```

`divmod(vocab_size, MAX_FUSED_SIZE)` computes how many 65536-wide chunks the vocab spans; `n_chunks = div + (mod != 0)` rounds up. For Mistral's 32K and any vocab up to 65536, `n_chunks == 1` and you take the single-pass kernel from section 3. For Gemma's 256000, `n_chunks == 4` and you take the chunked path.

In the chunked branch the launch grid is 2-D, `(n_rows, n_chunks)`: one program per `(token, chunk)` pair, each computing the logsumexp of its 65536-element slice into `logsumexp[row, chunk]`. The chunked kernel also accumulates the partial loss contribution (the `-x_t` term lands in whichever chunk contains the label). Then the host does the combination with stock PyTorch: `logsumexp = torch.logsumexp(logsumexp, dim=1)` reduces the `(n_rows, n_chunks)` partials down to `(n_rows,)` — that is the identity from above, implemented in one line. `losses += logsumexp` finishes $\text{CE} = \text{lse}(x) - x_t$, and `masked_fill_` applies the `-100` ignore convention. Note the difference from the single-pass branch, where the kernel already produced the final per-row logsumexp; the chunked branch produces *partial* logsumexps and the host reduces them.

The critical memory observation survives the chunking: the `logsumexp` tensor is `(n_rows, n_chunks)` before the reduce and `(n_rows,)` after. Even at its widest it is $8192 \times 4 = 32768$ floats, about 128 KB. The chunked path still never materializes a vocab-wide softmax. The 65536 cap is a hardware constraint, not a memory-budget one; the memory win is identical whether you take one chunk or four.

## 5. The backward trick: recompute, then overwrite

The forward pass kept only the logsumexp scalar. The backward pass has to produce a full $(N \times V)$ gradient — softmax minus the one-hot — and that gradient is exactly the same shape as the wall we were trying to avoid. If backward allocated a fresh gradient tensor, we would be right back at two vocab-wide tensors live at once. Unsloth refuses to allocate it. Instead it does two things at once: it *recomputes* the softmax on the fly from the saved scalar, and it writes the result **in place over the logits buffer**, reusing the memory the logits already occupy.

```python
@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride, dloss_ptr, dloss_row_stride,
    logsumexp_ptr, labels_ptr, VOCAB_SIZE, BLOCK_SIZE, ...):
    """
    dC/dx = softmax(x) - 1{x == label}  =  exp(x - logsumexp) - 1{label}
    """
    row_idx = tl.program_id(0); block_idx = tl.program_id(1)
    ...
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)                       # recompute softmax from the saved scalar
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    tl.store(logits_ptr + col_offsets, dloss * y, mask = mask)   # gradient written IN PLACE over logits
```

The mechanism, line by line:

- `logsumexp = tl.load(logsumexp_ptr + row_idx)` — reload the one scalar saved during forward.
- `y = tl.exp(x - logsumexp)` — this is $e^{x_i - \text{lse}} = P_i$, the softmax probability, **recomputed** from the original logit and the saved scalar. No softmax tensor was stored; it is being rebuilt right here, in registers, exactly when and only where it is needed. This is the recompute-vs-store tradeoff that runs through all of Unsloth: an `exp` is cheap, HBM is expensive, so recompute.
- `y = tl.where(col_offsets == label_idx, y - 1.0, y)` — subtract one at the label column, giving $P_i - \mathbb{1}[i = t]$, the gradient of the loss.
- `tl.store(logits_ptr + col_offsets, dloss * y, ...)` — scale by the upstream gradient `dloss` and write the result back **to `logits_ptr`** — the same buffer the logits came in on. The logits are overwritten by their own gradient.

That last store is the whole trick. The buffer that held $x$ now holds $\partial \text{CE} / \partial x$. There is no second allocation. And it is legal precisely because by the time backward runs, the forward loss has already been consumed and the original logit values are needed only to recompute $y$ — which the kernel does *before* it overwrites that same element. Each program reads `x`, computes `dloss * y`, and stores; the read happens before the write to the same address, so there is no hazard.

<figure class="blog-anim">
<svg viewBox="0 0 720 280" role="img" aria-label="Backward kernel recomputes the softmax from the saved logsumexp and overwrites the same logits buffer with the gradient" style="width:100%;height:auto;max-width:820px">
<title>The backward kernel overwrites the logits buffer in place with the gradient</title>
<style>
.ce1-cell{stroke:var(--border,#d1d5db);stroke-width:1.5;rx:6}
.ce1-fill-x{fill:var(--surface,#f3f4f6)}
.ce1-fill-g{fill:var(--accent,#6366f1)}
.ce1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ce1-val{font:600 14px ui-monospace,monospace;text-anchor:middle}
.ce1-x{fill:var(--text-primary,#1f2937)}
.ce1-g{fill:var(--background,#fff)}
.ce1-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ce1-buf{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:start}
@keyframes ce1-toGrad{0%,22%{opacity:0}40%,100%{opacity:1}}
@keyframes ce1-fadeX{0%,22%{opacity:1}40%,100%{opacity:0}}
@keyframes ce1-sweep{0%,18%{opacity:0}24%,34%{opacity:.9}46%,100%{opacity:0}}
.ce1-g-anim{animation:ce1-toGrad 7s ease-in-out infinite}
.ce1-x-anim{animation:ce1-fadeX 7s ease-in-out infinite}
.ce1-sweep{fill:none;stroke:var(--accent,#6366f1);stroke-width:3;rx:8;animation:ce1-sweep 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ce1-g-anim{animation:none;opacity:1}.ce1-x-anim{animation:none;opacity:0}.ce1-sweep{animation:none;opacity:0}}
</style>
<text class="ce1-buf" x="40" y="40">one logits row in HBM  (label = index 2)</text>
<rect class="ce1-cell ce1-fill-x" x="40"  y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-x" x="180" y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-x" x="320" y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-x" x="460" y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-g ce1-g-anim" x="40"  y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-g ce1-g-anim" x="180" y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-g ce1-g-anim" x="320" y="70" width="120" height="80"/>
<rect class="ce1-cell ce1-fill-g ce1-g-anim" x="460" y="70" width="120" height="80"/>
<text class="ce1-val ce1-x ce1-x-anim" x="100" y="116">x=2.1</text>
<text class="ce1-val ce1-x ce1-x-anim" x="240" y="116">x=-0.4</text>
<text class="ce1-val ce1-x ce1-x-anim" x="380" y="116">x=5.7</text>
<text class="ce1-val ce1-x ce1-x-anim" x="520" y="116">x=1.0</text>
<text class="ce1-val ce1-g ce1-g-anim" x="100" y="116">y</text>
<text class="ce1-val ce1-g ce1-g-anim" x="240" y="116">y</text>
<text class="ce1-val ce1-g ce1-g-anim" x="380" y="116">y-1</text>
<text class="ce1-val ce1-g ce1-g-anim" x="520" y="116">y</text>
<rect class="ce1-sweep" x="36" y="66" width="128" height="88"/>
<text class="ce1-lbl" x="360" y="200">y = exp(x - logsumexp)   then   dloss * y written in place</text>
<text class="ce1-cap" x="360" y="240">no gradient tensor is allocated; the logits buffer becomes the gradient</text>
</svg>
<figcaption>Backward recomputes the softmax y from the one saved logsumexp scalar, subtracts the one-hot at the label column, and overwrites the same logits buffer with dloss * y. The buffer that held logits now holds the gradient.</figcaption>
</figure>

The backward dispatch tiles the row differently from forward. It uses a fixed `BLOCK_SIZE = 4096` and a 2-D grid `(n_rows, n_blocks)` so that even a 256K row is covered by several blocks per token, and the `block_idx = tl.program_id(1)` selects which 4096-element slice this program handles:

```python
    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE); n_blocks = div + (mod != 0)
        _cross_entropy_backward[(n_rows, n_blocks,)](logits, logits.stride(0),
            dlosses, dlosses.stride(0), logsumexp, labels, VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE, ...)
        return logits, None, None, None    # logits buffer now holds the gradient
```

`return logits, None, None, None` is the signature of the whole technique. The four return values correspond to the four `forward` arguments (`logits`, `labels`, `logit_softcapping`, `logit_scaling`); only the first has a gradient, and that gradient *is* the `logits` tensor — now holding `dloss * (softmax - one_hot)` after the kernel overwrote it. Autograd hands `logits` upstream as $\partial \text{CE} / \partial \text{logits}$ and never knew a separate gradient buffer was supposed to exist. The `save_for_backward(logits, logsumexp, labels)` in forward saved the *logits buffer itself* (which will be repurposed), the tiny logsumexp vector, and the labels — and that is the entire backward state. No softmax, no log-softmax, no fresh gradient tensor.

![Forward saves a scalar; backward rebuilds the softmax. save_for_backward keeps logits, the logsumexp vector, and labels; backward reloads the scalar, recomputes the softmax, and writes the gradient in place over the logits buffer.](/imgs/blogs/unsloth-fused-cross-entropy-5.webp)

The figure traces the full lifecycle. Forward runs the kernel, emits the logsumexp scalar and the loss, and stashes `(logits, logsumexp, labels)` into `ctx`. Backward reloads, runs its kernel — `y = exp(x - lse)` — and writes `dloss * y` straight over the logits buffer (the red node). No tensor crosses the forward/backward boundary except the ones that were already going to be alive: the logits buffer, which gets reused, and a 32 KB scalar vector.

## 6. Why it is exact

It is worth being emphatic about this, because "we saved 8 GB" usually comes with an asterisk, and here it does not. Unsloth's fused cross-entropy is **numerically identical** to a correct `F.cross_entropy`, up to floating-point reordering of a reduction — which is the same caveat that applies to *any* two implementations of a sum on a GPU.

Walk the claims:

1. **Same loss.** The forward kernel computes $\text{lse}(x) - x_t$ with the max-shift for stability. That is the textbook stable cross-entropy. The chunked path computes the *same* logsumexp via the exact concatenation identity — `torch.logsumexp` of per-chunk logsumexps equals the global logsumexp by algebra, not by approximation.
2. **Same gradient.** The backward kernel computes $e^{x_i - \text{lse}} - \mathbb{1}[i=t]$, which is the analytic derivative $P_i - \mathbb{1}[i=t]$. Recomputing $P_i$ from the saved $\text{lse}$ gives bit-identical results to computing it during forward, because $e^{x_i - \text{lse}}$ is the same arithmetic either way.
3. **No lossy step anywhere.** There is no quantization of the logits, no low-rank approximation of the softmax, no top-k truncation of the vocabulary. The fp32 accumulation matches PyTorch's.

The memory and speed win comes entirely from *fusion* (one kernel instead of several), *recomputation* (rebuild the cheap softmax instead of storing it), and *in-place writes* (reuse the logits buffer for its own gradient). None of those change the answer. This matters in practice because a fine-tune is a long feedback loop; if the loss were subtly wrong you would not find out until your eval numbers came back soft, days later. "Same math, never materialized" is the property that lets you turn this on and forget about it. It is the same philosophy as the rest of the library — see the [manual-backprop post](/blog/machine-learning/open-source-library/unsloth-manual-backprop) for the LoRA-MLP version of "derive it, save only the residue, write in place."

## 7. Integration: patching the model's loss

A kernel nobody calls is a museum piece. Unsloth wires the fused cross-entropy into the model through a wrapper and a patch. The user-facing entry point is `fast_cross_entropy_loss`, which calls the autograd `Function` and handles the normalization that a training loss needs:

```python
def fast_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    batch, seq_len, d = logits.shape
    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),     # flatten (B, S, V) -> (B*S, V): one row per token
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)   # number of trained-on tokens
    return loss.sum() / n_items
```

The flatten `logits.view(batch * seq_len, d)` is what makes "one program per token" possible — the kernel sees a 2-D `(n_rows, vocab)` matrix, one row per token, batch and sequence collapsed. The `n_items` normalization is the detail people get wrong when they roll their own loss: cross-entropy over a batch should be averaged over the number of *real* (non-`-100`) tokens, not over the padded length. Dividing the summed per-token loss by `count_nonzero(labels != -100)` gives the correct mean even with ragged sequences and packed batches.

To make the model actually use this instead of HuggingFace's loss, Unsloth monkeypatches the loss function during model patching:

```python
def patch_loss_functions(...):
    # replace transformers' ForCausalLMLoss / fixed_cross_entropy with the fused path
    import transformers.loss.loss_utils
    transformers.loss.loss_utils.fixed_cross_entropy = unsloth_fixed_cross_entropy
    # the patched fn flattens logits, calls fast_cross_entropy_loss, normalizes by n_items
```

After patching, any `model(...)` call whose head computes a causal-LM loss routes through the fused kernel transparently. The training loop you wrote against the HuggingFace API does not change; the 8 GB tensor just stops being allocated. This is the same patch-don't-fork strategy the [speedup anatomy post](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) describes for the rest of the model: keep the upstream `transformers` control flow, swap the hot leaves for fused kernels.

Two model-family extras ride along in the kernel signature, which is why you see `DO_SOFTCAPPING / SOFTCAP` and `DO_LOGIT_SCALING / LOGIT_SCALE` as kernel arguments. **Logit softcapping** (Gemma 2) squashes logits through $\text{softcap} \cdot \tanh(x / \text{softcap})$ before the softmax, to keep them bounded; the kernel applies it inside the same pass so it costs nothing extra in memory. **Logit scaling** (Cohere's Command-R) multiplies logits by a constant before the softmax. Both are pre-softmax transforms, so they slot naturally into the fused forward — and crucially their derivatives fold into the same in-place backward, preserving the no-extra-allocation property. They are a side note for our memory story, but they are the reason the real kernel signature is longer than the toy one.

## 8. The numbers

Let us close the loop on the arithmetic with the peak-memory comparison, since that is what you will actually feel on the device.

![Logit-step peak memory across three vocabulary sizes: naive keeps logits plus a full softmax copy, fused keeps logits plus a tiny logsumexp vector, and the saving scales with vocabulary size.](/imgs/blogs/unsloth-fused-cross-entropy-6.webp)

| Model | Vocab | Naive peak ($2NV \cdot 4$) | Fused peak ($NV \cdot 4 + N \cdot 4$) | Softmax tensor never built |
|---|---|---|---|---|
| Mistral | 32K | ~2 GB | ~1 GB | ~1 GB |
| Llama-3 | 128K | ~8 GB | ~4 GB | ~4 GB |
| Gemma | 256K | ~16 GB | ~8 GB | ~8 GB |

(All at $N = 8192$, fp32. The fused "peak" still holds the logits, which the backward then reuses as the gradient buffer — so even the logits are not double-counted across forward and backward.)

The fused path roughly *halves* the logit-step peak, and the absolute saving grows linearly with vocabulary. For the 256K-vocab model that is 8 GB freed at the single hottest moment of the step — the difference between fitting on a 24 GB card and not. And because the saving is the second vocab-wide tensor, the bigger the vocabulary, the more the wall hurts the naive path and the more the fused path saves. The `logsumexp` vector that replaces the softmax tensor is ~32 KB at this sequence length — five orders of magnitude smaller than the thing it stands in for.

This is also where the speed shows up. Cross-entropy on a large vocab is memory-bandwidth-bound: the dominant cost is moving the logits across HBM, not the arithmetic. The naive path streams the logits several times and writes a second vocab-wide tensor; the fused path streams them once in forward and once in backward, writing only scalars in forward. Less traffic, fewer kernel launches, no giant allocation to zero and free. The headline "3x faster training and 30% less VRAM via new Triton kernels" that Unsloth cites for recent releases has this kernel as one of its load-bearing pieces, especially on the long-context, big-vocab runs where the wall is tallest.

## 9. War stories: when the wall falls on you

These are the shapes of the problem as it shows up in practice, and how the fused kernel changes the outcome.

**1. The last-layer OOM on a 24 GB card.** A 9B Gemma-2 in 4-bit, LoRA rank 16, sequence length 8192, batch 1. The weights are 1.5 GB, the adapters and their optimizer state are a few hundred megabytes, activations are checkpointed down to about a gigabyte. The run dies the instant the head computes the loss, with a 16 GB allocation request the card cannot satisfy. The naive `F.cross_entropy` wanted logits (8 GB) plus log-softmax (8 GB) live at once. Switching to the fused kernel drops the loss-step peak to ~8 GB and the run fits with room to spare. Nothing else in the config changes; the single tensor stops being doubled.

**2. The bf16 loss that wouldn't converge.** A team trying to fit a big-vocab model halved the logits to bf16 to dodge the wall, computing the loss in 16-bit. The loss curve was noisy and plateaued early. The culprit was the softmax normalizer: $\sum e^{x_j}$ over a 256K vocab in bf16 loses precision badly, and the per-token loss developed a bias. The fused kernel sidesteps the temptation entirely — it accumulates the reduction in fp32 inside the kernel regardless of the logits' storage dtype, so you get fp32-accurate loss without ever holding an fp32 vocab-wide tensor. The fix was not "use a smaller dtype"; it was "stop storing the tensor at all."

**3. The chunked path silently saving a different model.** An engineer benchmarked the single-pass kernel on Llama (`n_chunks == 1`) and assumed Gemma would behave the same. Gemma's 256K vocab takes the chunked branch, which writes a `(n_rows, n_chunks)` partials tensor and does a host-side `torch.logsumexp`. The two paths produce identical losses — that is the whole point of the chunked identity — but the chunked path has a slightly different kernel-launch profile (a 2-D grid, an extra small reduce). Knowing *which* branch your model takes, gated purely by whether `vocab_size > 65536`, explains a benchmark discrepancy that otherwise looks like noise.

**4. The "my gradient is wrong" false alarm.** A contributor instrumented the backward pass, read `logits` after `loss.backward()`, and reported that the logits had been "corrupted" — they no longer matched the forward values. They had not been corrupted; they had been *overwritten by their own gradient*, which is the design. `return logits, None, None, None` hands the repurposed buffer upstream as the gradient. The lesson: with in-place backward, the input buffer is not a stable place to read the original activation after backward runs. If you need the pre-gradient logits for logging, snapshot them before `.backward()`.

**5. Packed sequences and the `n_items` average.** A run that packed many short examples into each 8192-token sequence reported a loss that looked too low compared to an unpacked baseline. The packed batch had many `-100` padding/boundary tokens; averaging over the full padded length instead of the trained-on token count deflates the mean. `fast_cross_entropy_loss` normalizes by `count_nonzero(labels != -100)`, which is the correct denominator. The "bug" was a denominator mismatch between two loss implementations, and the fused wrapper had it right.

**6. The vision model with a huge text head.** A multimodal model fed thousands of image-patch tokens into a text decoder with a large vocab. The image tokens are not text targets, so they are labeled `-100`, but they still occupy rows in the flattened `(n_rows, vocab)` logits. The fused kernel handles them for free — the `-100` branch forces their loss to 0 and the backward subtracts a one-hot only at real label columns — and the per-token-row structure means thousands of ignored rows cost only their share of bandwidth, not a separate masking allocation.

**7. Logit softcapping mismatch on Gemma 2.** Someone ported a custom loss for Gemma 2 and forgot the $\tanh$ softcap, getting a loss that diverged from the reference by a small but persistent amount. The fused kernel carries `DO_SOFTCAPPING / SOFTCAP` precisely so the softcap is applied *inside* the same pass that computes the logsumexp, keeping the loss exact for that family without a second tensor. The fix was to route through the fused path rather than reimplement the family-specific pre-softmax transform.

## When to reach for the fused cross-entropy — and when not to

Reach for it whenever the vocabulary is large and the sequence is long — which today means essentially every modern LLM fine-tune. The bigger the vocabulary, the more the second vocab-wide tensor dominates the step, and the more the fused kernel saves; at 256K vocab it is the difference between fitting and not. It is exact, so there is no accuracy cost to weigh against the memory win, and inside Unsloth it is on by default through `patch_loss_functions` — you get it for free when you load a model with `FastLanguageModel.from_pretrained`.

It is *not* a magic bullet for small-vocab, short-sequence regimes — a 32K-vocab model at sequence 512 has a logit tensor of a few hundred megabytes, and the naive path is annoying but survivable; the win is real but you may not notice it. It does not help if your bottleneck is the weights or the attention activations rather than the logits — profile first; if the loss step is not your peak, this kernel is not your fix. And the in-place backward changes one habit: do not read the logits buffer after `.backward()` expecting the original values, because it now holds the gradient. Snapshot what you need before the backward pass runs.

The deeper lesson generalizes past this one kernel. The cross-entropy wall stood for years not because it was hard to climb but because nobody questioned whether the softmax tensor needed to exist at all. Derive the loss, notice it depends on the row only through a scalar logsumexp, notice the gradient can be rebuilt from that scalar, and the wall turns out to be optional. That move — *write down the math and store only its irreducible residue* — is the throughline of the entire [Inside Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) series, and cross-entropy is its cleanest single instance.
