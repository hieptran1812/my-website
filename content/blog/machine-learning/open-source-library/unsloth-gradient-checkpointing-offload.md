---
title: "Smart Gradient Checkpointing: How Unsloth Offloads Activations to System RAM"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Activations are the long-context memory wall. Unsloth's offloaded gradient checkpointer recomputes each block in backward and parks its input in pinned CPU RAM with non_blocking copies, collapsing on-GPU activation memory to roughly one block's worth."
tags: ["unsloth", "gradient-checkpointing", "activation-memory", "offload", "long-context", "cuda-streams", "gpu-memory", "qlora", "pytorch", "training-efficiency"]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 29
---

Ask anyone who has tried to fine-tune a long-context model on a single consumer GPU what killed them, and they will not say the weights. A 7B model in 4-bit NF4 is about 3.5 GB. An 8-bit optimizer state for a LoRA adapter is a rounding error. What blows up — what turns a 24 GB card into an out-of-memory traceback at sequence length 8192 — is the activations. Every layer, on the forward pass, stores the tensors its backward pass will need, and that pile grows with batch size, with sequence length, and with depth, all at once. Push the context window and you do not pay linearly; you pay linearly *per layer*, and there are a lot of layers.

[Unsloth](/blog/machine-learning/open-source-library/unsloth-lib) has a sharp answer to this that goes one step past what PyTorch ships. Standard gradient checkpointing already trades compute for memory: it refuses to store a transformer block's inner activations and instead recomputes the block's forward during backward. Unsloth keeps that trade and adds a second one. It also takes each checkpointed block's *input* — the `hidden_states` tensor handed to that block — and copies it off the GPU into system RAM during the forward pass, using an asynchronous `non_blocking` transfer that overlaps with compute. The activation that would have sat in VRAM for the entire depth of the model now lives on the host, and the GPU's activation footprint across the whole network collapses to roughly one block's worth. The throughput cost is small because the PCIe copy hides behind the matmuls you were going to run anyway.

![Stored activation VRAM grows linearly without checkpointing, sublinearly with standard checkpointing, and stays near one block under Unsloth offload](/imgs/blogs/unsloth-gradient-checkpointing-offload-1.webp)

The diagram above is the mental model. Same model, same batch, same sequence length — three columns, three retention policies. With `use_gradient_checkpointing=False` you store everything and pay $O(L)$ in depth. With `=True` you save block boundaries and recompute, paying roughly $O(\sqrt{L})$. With `="unsloth"` you recompute *and* offload, and the on-GPU activation memory flattens to about one block. This post walks the whole mechanism: why activations are the wall, how standard checkpointing works, the exact code of `Unsloth_Offloaded_Gradient_Checkpointer`, why `non_blocking` makes the copy nearly free, how to turn it on, and where it helps versus where it hurts.

## 1. Why activations are the long-context killer

Start from a single linear layer, because the whole story is already there. A transformer is mostly matmuls, and the canonical one is $Y = XW$, where $X$ is the layer input of shape (batch, seq, hidden) and $W$ is the weight. On the forward pass this is one GEMM. The trouble is the backward pass.

To train, you need the gradient of the loss $C$ with respect to the weight. By the chain rule, for $Y = XW$:

$$\frac{\partial C}{\partial W} = X^\top \frac{\partial C}{\partial Y} = X^\top dY$$

Read that carefully. The gradient of the weight is the *forward input* $X$, transposed, times the upstream gradient $dY$. You cannot compute $dW$ without $X$. So PyTorch, when it builds the autograd graph for this matmul, calls `save_for_backward(X)` — it pins $X$ in memory until the backward pass consumes it. That saved $X$ is an *activation*. Every linear layer in the network does this. Every attention projection, every MLP matmul, every layernorm input — each one keeps a tensor alive from its forward call until its backward call fires, which for the first layer of the model is essentially the entire step.

![dC/dW = X transpose times dY: the saved input X is what makes activations the long-context memory wall](/imgs/blogs/unsloth-gradient-checkpointing-offload-2.webp)

Now count the memory. The activation footprint of a transformer scales as:

$$\text{activation memory} \;\propto\; \text{batch} \times \text{seq} \times \text{layers} \times \text{hidden}$$

Three of those four factors are fixed by the model and your batch choice, but `seq` is the one you reach for when you want long context, and it multiplies *every layer*. Double the sequence length and you double the activation memory in every one of the $L$ blocks simultaneously. That is why the bars in the figure above grow the way they do: at seq 2k you are comfortable, at seq 8k you are sweating, and at seq 32k the activations alone have eaten the card while the 4-bit weights sit there using a constant 3.5 GB. The weights do not grow with context. The activations do, and they do it $L$ times over.

This is the asymmetry that makes activation memory the thing to attack. A concrete number: for a Llama-style 7B at hidden size 4096, 32 layers, batch 1, bf16, the activations you must retain for backward run into the tens of gigabytes by the time the sequence is long enough to be interesting. The model is small. The training state is the problem. And unlike the weights — which you cannot shrink below their information content without [4-bit quantization](/blog/machine-learning/open-source-library/unsloth-manual-backprop) — the activations are *recomputable*. You stored them only because recomputing was expensive. That recomputability is the lever the next three sections pull.

It is worth doing the arithmetic once, because the orders of magnitude are what make the design decisions obvious. Take the residual stream — the (batch, seq, hidden) tensor that flows between blocks — at batch 1, seq 8192, hidden 4096, in bf16 (2 bytes). That is $1 \times 8192 \times 4096 \times 2 = 64$ MB for *one* tensor at *one* boundary. With 32 blocks, the boundary tensors alone are about 2 GB. But the residual stream is the small part: inside each block, attention produces query/key/value projections, attention scores, the attention output, and the MLP produces a gate and up projection at the (typically 4×) intermediate width plus the SwiGLU product — easily 5–10× the residual-stream size per block in retained intermediates if you store everything. Multiply by 32 blocks and you are well past the activation budget of a 24 GB card before the optimizer or the gradients get a look in. Now double the sequence to 16K and every one of those numbers doubles. The weights did not move. This is why "fit a longer context" is almost never a weight problem and almost always an activation problem, and why the techniques in this post are aimed squarely at the activation term.

## 2. Standard gradient checkpointing: trade compute for memory

The classic move, available in `torch.utils.checkpoint`, is to stop storing a block's inner activations and instead recompute them when backward needs them. You keep only the *input* to each checkpointed segment — the block boundary — and throw away everything the block produced internally. When the backward pass reaches that segment, you re-run the segment's forward from the saved input, regenerate the inner activations, and then do the backward over the freshly-recomputed graph.

Here is the standard idiom, the baseline Unsloth is improving on:

```python
import torch
import torch.utils.checkpoint as checkpoint

class TransformerBlock(torch.nn.Module):
    def forward(self, hidden_states, attention_mask):
        # attention + MLP; lots of inner activations created here
        ...
        return hidden_states

# Naive: every block stores all of its inner activations for backward.
def forward_no_checkpoint(blocks, hidden_states, mask):
    for block in blocks:
        hidden_states = block(hidden_states, mask)
    return hidden_states

# Checkpointed: store only the boundary; recompute the inner forward in backward.
def forward_checkpointed(blocks, hidden_states, mask):
    for block in blocks:
        # `checkpoint` saves `hidden_states` (the input), runs `block` under
        # no_grad in forward, and re-runs `block` with grad in backward.
        hidden_states = checkpoint.checkpoint(
            block, hidden_states, mask,
            use_reentrant=False,
        )
    return hidden_states
```

The contract is simple. `checkpoint.checkpoint(fn, *args)` runs `fn(*args)` without building the autograd graph (so no inner activations are retained), but it remembers the inputs. When the loss backpropagates to that point, it runs `fn(*args)` *again*, this time with the graph enabled, materializes the inner activations transiently, computes the local gradients, and frees them. You pay one extra forward pass per checkpointed segment, and in exchange you do not hold that segment's interior in memory between the original forward and the backward.

![Standard checkpointing stores only block-boundary inputs and recomputes the inner forward during backward, trading compute for memory](/imgs/blogs/unsloth-gradient-checkpointing-offload-3.webp)

How much memory does this save? The textbook analysis assumes a chain of $N$ uniform stages. If you checkpoint every stage, you store $N$ boundary tensors and recompute each stage once, so memory is $O(N)$ in boundaries but you have dropped all the interiors. The famous result — from Chen et al.'s "sublinear memory" paper — is that if you instead checkpoint only every $\sqrt{N}$-th stage, you can drive the peak activation memory down to $O(\sqrt{N})$ while still doing only one extra forward overall. The intuition: you keep $\sqrt{N}$ "segment" boundaries, and within a segment you recompute from the nearest saved boundary, holding at most $\sqrt{N}$ interior activations live at once. That is the sawtooth you saw in figure 1 — memory rises within a segment as you recompute, then drops at the next boundary.

In practice, transformer libraries checkpoint at the block granularity rather than tuning the $\sqrt{N}$ schedule, because a block is a natural, self-contained unit and the boundary tensor (the residual-stream `hidden_states`) is exactly the small thing worth keeping. The cost is real but bounded: one extra forward through each block during backward, which empirically lands around a 20–30% throughput hit on a compute-bound model, in exchange for a large drop in peak activation VRAM. For most long-context fine-tuning that trade is obviously worth it, because without it you simply OOM.

But notice what standard checkpointing does *not* do. The boundary tensors it keeps — one residual-stream activation per block — still live on the **GPU**, all $L$ of them, for the duration of the step. That residual stream is shape (batch, seq, hidden), and at long context it is not small. $L$ copies of it is the memory standard checkpointing still pays. This is the seam Unsloth pries open.

## 3. Unsloth's offloaded checkpointer

Unsloth's contribution, living in `unsloth_zoo/gradient_checkpointing.py`, is a custom `torch.autograd.Function` called `Unsloth_Offloaded_Gradient_Checkpointer`. It does everything standard checkpointing does — recompute the block in backward — and adds one thing: it moves the saved boundary tensor off the GPU and into system RAM. Here is the class, essentially verbatim:

```python
class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        ctx.device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)   # async GPU->CPU
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.device, non_blocking = True).detach()  # async CPU->GPU
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)   # recompute THIS block
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
```

Walk the forward pass line by line. `ctx.device = hidden_states.device` stashes which GPU the tensor came from, so backward knows where to send it back. Then the load-bearing line: `saved_hidden_states = hidden_states.to("cpu", non_blocking=True)`. This kicks off an asynchronous device-to-host copy of the block input into CPU RAM and returns immediately — the copy is queued, not waited on. Next, `with torch.no_grad(): output = forward_function(hidden_states, *args)` runs the block's forward *without building an autograd graph*, exactly like standard checkpointing: no inner activations are retained. Finally `ctx.save_for_backward(saved_hidden_states)` saves the **CPU** copy as the tensor to bring back later, and `ctx.forward_function` / `ctx.args` remember how to recompute.

The original `hidden_states` GPU tensor is still passed into `forward_function` for the real forward computation, but because it is not saved for backward (the *CPU* copy is), once this block's forward finishes and the next block does not need the previous block's GPU input, that GPU memory is reclaimable. What persists between forward and backward is the host-side copy.

![Unsloth's checkpointer stashes hidden_states on the host in forward and reloads them in backward](/imgs/blogs/unsloth-gradient-checkpointing-offload-4.webp)

Now the backward pass. `(hidden_states,) = ctx.saved_tensors` pulls the CPU copy back out. The mirror line is `hidden_states = hidden_states.to(ctx.device, non_blocking=True).detach()`: an asynchronous host-to-device copy back to the original GPU, detached so it starts a fresh autograd subgraph. Then `hidden_states.requires_grad_(True)` marks it as a leaf we want a gradient for — this is what lets us recover the gradient that flows into this block's input and pass it upstream. `with torch.enable_grad(): (output,) = ctx.forward_function(hidden_states, *ctx.args)` re-runs the block's forward, this time *with* the graph, regenerating exactly the inner activations we threw away in the original forward. `torch.autograd.backward(output, dY)` then backpropagates the upstream gradient `dY` through that freshly-built local graph, populating `hidden_states.grad`. The return tuple `(None, hidden_states.grad,) + (None,)*len(ctx.args)` hands `None` for the non-tensor `forward_function` argument, the input gradient for `hidden_states`, and `None` for each extra arg.

That is the entire mechanism. It is strikingly small — a few dozen lines — because it leans entirely on PyTorch's autograd and the CUDA memory model rather than reimplementing anything. The two ideas it composes are not new individually: recompute-in-backward is standard checkpointing, and host offload is a known technique. What makes it work in practice is the fourth word in each copy: `non_blocking`.

Two details in this code are easy to skim past but are exactly why it is correct, not just memory-cheap. The first is the `.detach()` on the reloaded tensor in backward. Without it, the reloaded `hidden_states` would still carry whatever autograd history clung to the CPU copy, and re-running the forward on top of stale history would either error or silently double-count gradients. Detaching gives you a clean leaf, and `requires_grad_(True)` then makes it a leaf that *accumulates* a gradient — which is the gradient this block must hand to the block before it. The second is that `torch.autograd.backward(output, dY)` is a *local* backward: it backpropagates `dY` through only the freshly-recomputed subgraph of this one block, lands the input gradient in `hidden_states.grad`, and stops. It does not recurse into earlier blocks. Earlier blocks are separate `Unsloth_Offloaded_Gradient_Checkpointer` instances; PyTorch's outer autograd engine calls each one's `backward` in turn, and each reloads, recomputes, backprops locally, and returns its input gradient up the chain. The recompute is strictly per-block, which is what keeps the live activation footprint at one block rather than the whole graph.

This per-block isolation is also what makes the offload safe across the depth of the model. Because no block's recompute depends on another block's *retained* GPU activations — each one regenerates its own interior from its own reloaded input — there is no tensor that must stay resident on the GPU across block boundaries except the residual-stream input currently being processed. That independence is the precondition for moving everything else to the host.

## 4. Why `non_blocking` hides the cost

Here is the obvious objection. You just added two PCIe transfers per block per step — one device-to-host in forward, one host-to-device in backward. PCIe is slow relative to HBM. A residual-stream tensor at long context is not tiny. Surely shuttling it across the bus dominates?

It would, if the copies were synchronous. The trick is that they are not. `non_blocking=True` on a `.to()` between a CUDA device and *pinned* host memory issues the copy as an asynchronous DMA transfer on a CUDA stream and returns control immediately. The CPU thread does not block waiting for the bytes to land; it goes on to enqueue the next GPU operation. As long as the data being copied is not needed by the very next kernel, the transfer runs *concurrently* with GPU compute. The DMA engine moves the activation across PCIe while the streaming multiprocessors chew through the next block's matmuls.

<figure class="blog-anim">
<svg viewBox="0 0 760 280" role="img" aria-label="A GPU compute lane and a PCIe copy lane run on the same timeline; the non-blocking copy overlaps the next block's compute, so the transfer adds almost no wall-clock time" style="width:100%;height:auto;max-width:820px">
<title>The non_blocking copy overlaps GPU compute, hiding PCIe transfer time</title>
<style>
.g2-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.g2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.g2-tick{font:500 12px ui-monospace,monospace;fill:var(--text-secondary,#6b7280)}
.g2-compute{fill:var(--accent,#6366f1);opacity:.9}
.g2-copy{fill:var(--text-secondary,#6b7280);opacity:.55}
.g2-cursor{stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:4 4}
@keyframes g2-sweep{0%{transform:translateX(0)}100%{transform:translateX(620px)}}
.g2-mv{animation:g2-sweep 7s linear infinite}
@media (prefers-reduced-motion:reduce){.g2-mv{animation:none;transform:translateX(310px)}}
</style>
<text class="g2-lbl" x="20" y="70">GPU compute</text>
<text class="g2-lbl" x="20" y="170">PCIe copy</text>
<rect class="g2-compute" x="120" y="46" width="150" height="48" rx="6"/>
<rect class="g2-compute" x="280" y="46" width="150" height="48" rx="6"/>
<rect class="g2-compute" x="440" y="46" width="150" height="48" rx="6"/>
<rect class="g2-compute" x="600" y="46" width="120" height="48" rx="6"/>
<rect class="g2-copy" x="282" y="146" width="120" height="48" rx="6"/>
<rect class="g2-copy" x="442" y="146" width="120" height="48" rx="6"/>
<rect class="g2-copy" x="602" y="146" width="110" height="48" rx="6"/>
<text class="g2-tick" x="125" y="120">block n fwd</text>
<text class="g2-tick" x="287" y="222">stash n (hidden)</text>
<text class="g2-tick" x="447" y="222">stash n+1 (hidden)</text>
<line class="g2-axis" x1="110" y1="240" x2="730" y2="240"/>
<text class="g2-tick" x="540" y="262">wall-clock time &rarr;</text>
<line class="g2-cursor g2-mv" x1="110" y1="36" x2="110" y2="240"/>
</svg>
<figcaption>The async <tt>.to(non_blocking=True)</tt> copy (lower lane) runs while the GPU computes the next block (upper lane). Because the transfer hides behind compute, offload costs only the small slice that does not overlap.</figcaption>
</figure>

The animation shows the two lanes on a shared timeline. The GPU-compute lane runs block forwards back-to-back. The PCIe-copy lane stashes block $n$'s hidden state while the GPU is already computing block $n+1$. The copy lives entirely inside the shadow of the compute. Unsloth's own docstring says it plainly: *"Tiny hit to performance, since we mask the movement via non blocking calls."* The cost is not the full transfer time; it is only the sliver that fails to overlap — startup latency, and any stretch where the copy is longer than the compute it hides behind.

To see why the overlap is even possible, you have to think in terms of CUDA's execution model rather than the synchronous-looking Python. When you call a kernel or an async copy, you are not running it — you are *enqueuing* it on a CUDA stream, a FIFO queue of GPU work. The CPU returns immediately and races ahead to enqueue the next thing. The GPU drains the stream in order, but a copy issued on the same stream as compute will still serialize behind it; the real concurrency comes from the fact that the GPU has dedicated *copy engines* (DMA units) separate from its compute SMs. A host-device transfer rides a copy engine while the SMs run matmuls, and the two proceed in parallel as long as nothing forces a synchronization between them. `non_blocking=True` is what keeps the CPU from inserting that synchronization — it says "do not block my thread waiting for this copy to finish," which in turn lets the CPU keep feeding the compute stream so the copy and the compute have something to overlap. Drop the flag and the copy becomes a synchronization point: the CPU stalls, the compute queue drains, and the transfer time lands directly on the wall clock.

The size of the un-masked sliver, then, is a race between two quantities per block: the time to compute the block's forward, and the time to push the residual stream across PCIe. When compute dominates — long sequences, large hidden, the regime that caused the memory problem in the first place — the copy disappears entirely behind it. The cost only becomes visible when the copy is the longer of the two, which is the short-context, slow-bus corner that did not need offloading anyway. This is the happy coincidence at the heart of the design: the technique is cheapest exactly when you need it most.

Two requirements make this overlap real, and both matter:

- **Pinned (page-locked) host memory.** A normal `malloc`'d host buffer is pageable; the OS can move or swap it, so the CUDA driver cannot DMA directly into it — it has to stage through an internal pinned buffer, which serializes the copy and kills the async benefit. PyTorch's CPU tensors that the framework hands to pinned-aware paths, and tensors explicitly pinned, are page-locked so the DMA engine can read/write them without CPU involvement. This is why the figures call the host side *pinned* host RAM. (For the host-to-device reload to truly overlap, the destination must be on a non-default stream or the source pinned; in practice the win comes from the copies not stalling the compute stream.)
- **The copied data is not needed immediately.** In forward, the stashed copy is only consumed in backward, far in the future — maximum slack. In backward, the reload is needed before the recompute, so there is less slack, but it is still issued ahead of the kernels that use it.

Now stack this across the whole model and the memory picture inverts.

<figure class="blog-anim">
<svg viewBox="0 0 760 320" role="img" aria-label="During forward, each block's input is copied from GPU VRAM down to CPU RAM so the GPU holds only one live block; during backward the inputs stream back one at a time" style="width:100%;height:auto;max-width:820px">
<title>Block inputs offload to CPU RAM in forward and stream back in backward</title>
<style>
.g1-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.g1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.g1-sub{font:500 12px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.g1-live{fill:var(--accent,#16a34a);opacity:.9}
.g1-host{fill:var(--text-secondary,#6b7280);opacity:.35}
.g1-drop{fill:var(--text-secondary,#6b7280)}
@keyframes g1-fall{0%{transform:translateY(0);opacity:0}8%{opacity:1}40%{transform:translateY(150px);opacity:1}100%{transform:translateY(150px);opacity:.35}}
@keyframes g1-rise{0%{transform:translateY(0);opacity:.35}55%{opacity:.35}70%{transform:translateY(-150px);opacity:1}100%{transform:translateY(-150px);opacity:1}}
@keyframes g1-pulse{0%,40%{opacity:1}50%,100%{opacity:.25}}
.g1-d1{animation:g1-fall 9s ease-in-out infinite}
.g1-d2{animation:g1-fall 9s ease-in-out infinite;animation-delay:.7s}
.g1-d3{animation:g1-fall 9s ease-in-out infinite;animation-delay:1.4s}
.g1-r1{animation:g1-rise 9s ease-in-out infinite}
.g1-blink{animation:g1-pulse 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.g1-d1,.g1-d2,.g1-d3,.g1-r1,.g1-blink{animation:none}.g1-d1,.g1-d2,.g1-d3{transform:translateY(150px);opacity:.35}.g1-r1{transform:translateY(0);opacity:1}}
</style>
<rect class="g1-lane" x="30" y="40" width="700" height="100" rx="10"/>
<rect class="g1-lane" x="30" y="200" width="700" height="100" rx="10"/>
<text class="g1-lbl" x="380" y="28">GPU VRAM &mdash; only the live block stays resident</text>
<text class="g1-lbl" x="380" y="320">pinned host RAM &mdash; every other block input parked here</text>
<rect class="g1-host" x="70"  y="220" width="64" height="60" rx="8"/>
<rect class="g1-host" x="150" y="220" width="64" height="60" rx="8"/>
<rect class="g1-host" x="230" y="220" width="64" height="60" rx="8"/>
<rect class="g1-drop g1-d1" x="310" y="60" width="64" height="60" rx="8"/>
<rect class="g1-drop g1-d2" x="390" y="60" width="64" height="60" rx="8"/>
<rect class="g1-drop g1-d3" x="470" y="60" width="64" height="60" rx="8"/>
<rect class="g1-live g1-blink" x="610" y="60" width="80" height="60" rx="8"/>
<text class="g1-sub" x="650" y="96">recompute</text>
<rect class="g1-live g1-r1" x="70" y="220" width="64" height="60" rx="8"/>
</svg>
<figcaption>Forward stashes each block input to host RAM (squares fall to the lower lane); only the one block being computed stays bright on the GPU. Backward streams an input back up to recompute that block, then frees it again.</figcaption>
</figure>

In the forward pass, each block's input falls from the GPU lane into the host lane as it is offloaded; only the block actively computing stays bright on the GPU. By the time forward finishes, the residual-stream inputs for all $L$ blocks are sitting in host RAM, and the GPU is holding essentially one block's activations plus the resident weights and optimizer state. In backward, the inputs stream back up one at a time — the block being recomputed reloads its input, regenerates its interior, backprops, frees, and the next one comes back. The GPU never holds more than about one block's worth of activations at any instant. That is the whole point: the activation memory that standard checkpointing kept as $L$ residual-stream copies on the GPU now lives on the host, and the GPU footprint is flat in depth.

![VRAM holds weights, optimizer state, and one live block; the rest of the depth's inputs sit in system RAM](/imgs/blogs/unsloth-gradient-checkpointing-offload-5.webp)

The static figure above pins down where each thing physically lives at a backward instant. VRAM holds the 4-bit frozen weights (resident, constant), the 8-bit optimizer state, and the *one* active block's inner activations, with the rest of the card now free — the headroom long context used to consume. Pinned host RAM holds the $L-1$ other block inputs, one tensor per block, page-locked so each H2D/D2H copy is a DMA transfer the bus engine handles without stalling the GPU.

## 5. Turning it on

You do not interact with `Unsloth_Offloaded_Gradient_Checkpointer` directly. Unsloth wires it in through one argument that appears in both `from_pretrained` and `get_peft_model`. From `unsloth/models/loader.py`, the relevant default:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    dtype = None,                              # auto: bf16 on Ampere+, else fp16
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",    # <-- the offloaded checkpointer
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",    # <-- same flag, set it here too
    random_state = 3407,
)
```

The argument takes three values, and they correspond exactly to the three columns of the first figure:

| Setting | What it does | Peak activation VRAM | Extra recompute | Host RAM |
| --- | --- | --- | --- | --- |
| `False` | store every block's activations | $O(L)$, highest | none | none |
| `True` | standard `torch.utils.checkpoint` | $O(\sqrt{L})$ | +1 forward | none |
| `"unsloth"` | recompute **and** offload inputs to host RAM | ~1 block on GPU | +1 forward | $L-1$ inputs |

![Each setting trades peak activation VRAM against recompute cost and throughput in a different way](/imgs/blogs/unsloth-gradient-checkpointing-offload-6.webp)

Note the default. `FastLanguageModel.from_pretrained` ships with `use_gradient_checkpointing = "unsloth"` already set — alongside `load_in_4bit = True`, `max_seq_length = 2048`, and `random_state = 3407`. So if you do nothing, you get offloaded checkpointing for free. You set `False` only if you are short on system RAM and long on VRAM (rare for the single-GPU crowd), or you are benchmarking. You set `True` if for some reason you want recompute without the host transfer — for instance, if your PCIe link is genuinely the bottleneck and you would rather eat the $\sqrt{L}$ VRAM. For nearly everyone fine-tuning long context on one card, `"unsloth"` is the right answer and the default reflects that.

One subtlety worth stating: this is exact. Recomputing the block's forward under `torch.enable_grad()` produces numerically the same activations the original forward would have stored (modulo the usual nondeterminism of some CUDA kernels, which checkpointing inherits regardless). Offloading and reloading a tensor over PCIe is a bit-exact copy. So `="unsloth"` does **not** change your loss curve relative to `=True` — it is a pure memory-placement optimization, consistent with Unsloth's broader stance that its kernels make no approximations. You are trading a little wall-clock for a lot of VRAM, not trading accuracy for anything.

## 6. The compounding effect

Offloaded checkpointing is one of three memory levers Unsloth pulls, and they compound. Look again at the VRAM-vs-host-RAM figure: the GPU is holding 4-bit weights, 8-bit optimizer state, and one block of activations. Each of those three lines is a separate optimization:

- **4-bit NF4 weights.** The base model is quantized to 4 bits with double-quantized absmax scales, dequantized transiently inside the [manual-backprop kernels](/blog/machine-learning/open-source-library/unsloth-manual-backprop) and freed immediately, so the fp16 weight never persists. A 7B base drops from ~14 GB (fp16) to ~3.5 GB.
- **8-bit paged optimizer state.** Only the small LoRA adapters carry gradients and optimizer moments, and those moments are held in 8 bits with paging, so the optimizer state is a sliver instead of the 2× of fp32 Adam over full weights.
- **Offloaded activations.** The subject of this post: the residual-stream inputs across depth move to host RAM, leaving ~one block on the GPU.

Put together, these are why Unsloth can advertise fitting a long-context reasoning fine-tune — a 32K-context model with LoRA — into something like 5 GB of VRAM, on hardware that could never hold the fp16 weights, let alone the full activation stack. The weights are 4-bit, the optimizer state is 8-bit and tiny, and the activations are mostly on the host. None of the three alone gets you there; the activation offload is the one that makes long *context* specifically affordable, because activations are the term that scales with `seq`. For the full story of stretching the window, see the [long-context training](/blog/machine-learning/open-source-library/unsloth-long-context-training) post — offloaded checkpointing is the foundation that work builds on, often combined with [8-bit paged optimizers](/blog/machine-learning/open-source-library/unsloth-8bit-paged-optimizers) to free even more headroom.

It also stacks with the kernel-level fusion described in the [speedup anatomy](/blog/machine-learning/open-source-library/unsloth-speedup-anatomy) post. The fused Triton kernels reduce the *number* and *size* of intermediate activations created per block (a fused SwiGLU does not materialize separate gate/up/SiLU buffers; a fused cross-entropy never builds the full softmax). Smaller per-block activations mean less to offload and less to recompute. The two techniques are complementary: fusion shrinks the per-block footprint, offloading flattens the across-depth footprint.

## 7. Caveats and when it bites

Offloaded checkpointing is not a free lunch in every regime. Knowing where it stops helping is part of using it well.

**PCIe bandwidth is the ceiling.** The whole scheme rests on the copy hiding behind compute. If your per-block compute is small relative to the per-block activation size — short sequences, tiny hidden, a memory-bound rather than compute-bound block — then the copy may *not* fully overlap, and you pay a real fraction of the transfer time. On a slow PCIe 3.0 x8 link with a large residual stream, the masking is less complete than on PCIe 4.0/5.0 or NVLink-attached host memory. The throughput "tiny hit" assumes the common case (compute-bound, long context); it grows as you move toward the memory-bound corner.

**Pinned memory has a cost.** Page-locked host memory is a finite OS resource; pinning too much can starve the rest of the system and, in extreme cases, degrade overall throughput. The offloaded inputs are bounded ($L$ residual-stream tensors), so this is rarely a problem for one model on a workstation, but if you are running other memory-heavy host processes alongside training, keep an eye on it.

**It helps most exactly where you need it.** The benefit is proportional to depth times residual-stream size — that is, $L \times (\text{batch} \times \text{seq} \times \text{hidden})$ moved off-GPU. So it pays off hardest for deep models at long context with limited VRAM, which is precisely the single-GPU long-context fine-tuning scenario. For a shallow model at short context where you already fit comfortably, offloading is pure overhead — turn it down to `True` or `False` and skip the transfers. The default is tuned for the hard case, not the easy one.

**Single vs multi-GPU.** Older Unsloth lore claimed no multi-GPU support; that is outdated — multi-GPU is available now, with a major upgrade noted as on the way. On a single GPU the offload story is exactly as described: one device, one host, copies over its PCIe link. In a multi-GPU setting the same per-device offloading applies, but you now also have inter-GPU bandwidth and the placement of pinned buffers to think about; the principle (mask host transfer behind compute) holds, the bookkeeping grows.

**Recompute is still recompute.** You are doing roughly one extra forward pass through every block. On a compute-bound model that is the ~20–30% throughput cost standard checkpointing already imposed; offloading adds only the un-masked sliver of copy time on top. If your training is already throughput-critical and you have VRAM to spare, the honest move is to checkpoint less, not more — memory you are not using is throughput you are leaving on the table.

**It interacts with `torch.compile` and graph capture.** A custom `autograd.Function` with a host round-trip inside it is a deliberate graph break. The offload, the `no_grad`/`enable_grad` regions, and the CPU copy are not the kind of thing you want a compiler tracing through and trying to fuse or capture into a CUDA graph — the host transfer has no GPU representation to capture. Unsloth's broader design already keeps several kernels out of `torch.compile` on purpose (the RMSNorm and RoPE paths are wrapped to disable the dynamo tracer), and the checkpointer fits that same philosophy: it is a control-flow boundary, not a fusable op. If you are layering your own compilation on top, expect this boundary and do not fight it; the memory win is worth the break.

**RAM is not infinite either.** The offloaded inputs are bounded at $L$ residual-stream tensors, but at very long context that is still real host memory — recall the ~64 MB per boundary at seq 8K, so ~2 GB across 32 blocks, and four times that at seq 32K. On a workstation with 64–128 GB of system RAM this is comfortable; on a memory-starved box, or one already running a large dataloader and tokenization pipeline in RAM, the host side can become the new pressure point. The failure mode here is not an OOM you will misread as a GPU problem — it is host swapping, which silently destroys the async-copy overlap because swapped pages cannot be DMA'd. Watch host memory the same way you watch VRAM.

## 8. Numbers and case studies

Concrete scenarios, to ground the trade. The percentages are representative of the compute-bound long-context regime Unsloth targets; your exact numbers depend on model shape, sequence length, and PCIe generation.

**Case 1 — 7B QLoRA, seq 2048, 24 GB card, `False`.** No checkpointing. Activations across 32 layers at a modest 2K context already crowd the card alongside the 4-bit weights and optimizer state. You fit, barely, at small batch. Throughput is maximal — zero recompute, zero copies. This is the "I have VRAM headroom and want speed" corner, and at short context it is a legitimate choice.

**Case 2 — 7B QLoRA, seq 8192, 24 GB card, `False` → OOM.** Push the context to 8K and the activation term quadruples relative to 2K. The forward pass allocates the residual stream and inner activations for all 32 blocks and the allocator runs out of room before the first backward. This is the wall. No amount of smaller batch helps once batch is already 1; the per-token activation cost times sequence length times depth is the problem.

**Case 3 — same, `True`.** Standard checkpointing. Inner activations are dropped; only $L$ residual-stream boundaries are kept on the GPU. Peak activation VRAM falls to roughly $O(\sqrt{L})$ of the segmented schedule, or in the per-block case, the $L$ boundary tensors plus one block's recomputed interior. You now fit at 8K, paying ~20–30% throughput for the extra forward. For many users this is enough and they never need more.

**Case 4 — same, `"unsloth"`.** Offloaded checkpointing. The $L$ boundary tensors move to host RAM; the GPU holds ~one block's worth. Peak activation VRAM is now nearly flat in depth, freeing several more gigabytes versus `True`. Throughput is `True`'s cost plus the small un-masked copy sliver — in the compute-bound case, close to `True`. The freed VRAM is what lets you raise batch size, lengthen the sequence further, or fit a larger base model on the same card.

**Case 5 — 32K-context reasoning fine-tune in ~5 GB.** The headline. Combine 4-bit weights (~3.5 GB for 7B, less for smaller bases), 8-bit paged optimizer over LoRA adapters (sliver), and offloaded checkpointing (activations on host, ~one block on GPU). The activation term — which at 32K would otherwise dwarf everything — is mostly on the host, so the GPU residency is dominated by the constant weight cost. This is the configuration Unsloth points at when it talks about long-context training on small cards, and offloaded checkpointing is the leg of the tripod that specifically tames the `seq` dimension.

**Case 6 — slow PCIe, short context, `"unsloth"` is the wrong default.** Flip the assumptions: a PCIe 3.0 x8 link, seq 1024, a model that already fits. Here the per-block compute is small, the copy does not fully hide, and you are paying transfer time for memory you did not need. The fix is to step down to `True` (recompute without host transfer) or `False` (neither). The lesson: the offload default is optimal for the hard case it was designed for, and you should override it when your case is easy.

**Case 7 — debugging an unexpected throughput cliff.** I have personally chased a case where offloaded checkpointing looked far more expensive than the "tiny hit" promise. The culprit was that the host buffers were not pinned in the path being exercised, so every `non_blocking=True` copy silently degraded to a synchronous staged transfer — the async flag is a no-op when the host memory is pageable. The transfer stopped overlapping, and the copy time landed fully on the critical path. The tell is a profiler trace showing the copy stream serialized against the compute stream with no overlap. The fix is ensuring the offload path uses page-locked memory; once pinned, the overlap returned and the cost dropped back to the masked sliver. The mechanism is only as good as the pinning behind it.

## When to reach for offloaded checkpointing, and when not to

Reach for `use_gradient_checkpointing="unsloth"` — and it is the default, so mostly this means *leave it on* — when you are fine-tuning a deep model at long context on a single GPU with constrained VRAM. That is the scenario the technique was built for, and it is the scenario where the activation term dominates and moving it to the host is the difference between training and OOM. It compounds cleanly with 4-bit weights and 8-bit optimizer state, and because it is exact, it costs you nothing in accuracy. The throughput hit, in the compute-bound long-context regime it targets, is the same recompute cost standard checkpointing already pays plus a small un-masked copy sliver.

Step down to `True` when you have system-RAM pressure, a slow PCIe link where the copy does not hide, or a profiler trace showing the transfer is not overlapping despite pinning — standard checkpointing gives you the recompute memory win without the host round-trip. Step down to `False` when you have VRAM to spare and want maximum throughput at short context, where recompute is pure overhead. And whenever a "tiny hit" turns into a real one, check the pinning first: an async copy into pageable memory is a synchronous copy wearing a costume, and it will quietly serialize the very transfer the whole design depends on hiding.

The deeper lesson generalizes past Unsloth. Activations are recomputable and weights are not, so activations are where the cleverness goes — first by not storing them (checkpointing), then by storing them somewhere cheaper (the host), then by hiding the cost of getting them there (async DMA). Each step is small. Stacked, they turn the activation wall from the thing that ends your training run into a constant you barely notice.
