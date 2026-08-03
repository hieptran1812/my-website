---
title: "Speculative decoding and rollback with recurrent state: the undo button a state machine does not have"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Rejecting a drafted token means rewinding the model to a state that a recurrent layer has already destroyed, and this post derives the memory and bandwidth price of every known way to get it back."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "speculative-decoding",
    "state-space-models",
    "mamba",
    "hybrid-models",
    "kv-cache",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
    "decoding",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
---

Speculative decoding is the one optimization in this series that is pure profit on a transformer. A small draft model proposes four tokens, the big target model checks all four in a single forward pass, and on a good day you emit three tokens for the price of one. The accounting is airtight because the *undo* is free: if the target rejects the third drafted token, you move the KV cache write pointer back three slots and the bytes past it become garbage that nobody will ever read again. No copy, no recompute, no bookkeeping. Rollback is a subtraction on an integer.

Now run the same loop on a hybrid model — Nemotron-H, Qwen3-Next, Granite 4.0 H, anything with Mamba-2 or gated DeltaNet layers interleaved among the attention layers — and that airtight accounting springs a leak that you cannot patch with a pointer. A recurrent layer does not append to a cache. It **overwrites a fixed-size state in place**, once per token. Advancing through four drafted tokens runs that overwrite four times. When the target rejects at position two, the state you need to return to was destroyed on the first of those four steps, and there is no suffix to truncate because there is no per-token dimension in the object at all. You cannot truncate a state. You can only have saved it, or be prepared to build it again.

![Two columns comparing rollback in an attention layer against rollback in a recurrent layer after four drafted tokens are rejected at position two](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-1.webp)

The figure above is the whole post compressed into two columns, and the asymmetry it shows is the entire subject. On the left, an attention layer during a four-token speculation window: four KV slots get appended, two get rejected, and the rollback costs exactly zero bytes of data movement — you decrement a length counter and the block allocator gets two slots back. On the right, the same window in a recurrent layer of Nemotron-H-8B: the state advanced four times through the same 49.4 MiB tensor, there is no slot to drop, and getting back to where you were costs a full 49.4 MiB copy per request. Both numbers are derived below from a published model config. The left column is why speculative decoding became standard. The right column is why the vLLM team, writing about hybrid SSM serving in April 2026, listed the interaction with speculative decoding under *acknowledged limitations* rather than under *features*.

By the end of this post you will be able to state exactly which of your engine's layers can be rolled back and which have to be reconstructed, derive the memory and bandwidth cost of the four known reconstruction strategies as a function of draft length, batch size and state bytes, compute the batch at which one strategy overtakes another, and write the rollback path yourself. You will have built `nanoserve/rollback.py`: a speculation window that snapshots recurrent state, a rollback that truncates KV and restores state in the same call, the keep-k and replay-inputs variants, a correctness harness that catches the exact class of bug that has been filed against production engines, and a cost model that tells you which variant your batch size wants. And you will understand why the same problem shows up, unprompted, the moment anyone asks your engine for `n=2`.

Two promises carried over from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and they bind hard in a post whose subject is an open problem. **I have no GPU and I have run nothing.** Every number below is either derived from arithmetic I show you, cited to a paper, model card, issue tracker or vendor post with a link, or framed as something you should reproduce yourself with an expected range. The results tables carry a `Source` column. **And I do not describe a mechanism I could not trace to a primary source.** Where the public record says "not extensively validated", I say that, rather than inventing the missing detail.

---

## 1. What rollback actually means

Start with the shape of a speculation window, because half the confusion in this area comes from being vague about what "position" means when three different models are stepping at once. If you want the full derivation of why drafting works at all — the acceptance-rate math, the rejection-sampling proof that output distribution is preserved — that is [the core speculative decoding post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), and I will not repeat it. Here is only the part that touches state.

A sequence has been decoded up to position $t$. The engine holds, for that sequence:

- **KV cache**: $t$ token-slots of keys and values in every attention layer, laid out in paged blocks.
- **Recurrent state**: in every recurrent layer, one fixed-size tensor that is a lossy summary of all $t$ tokens. Call the total across layers $\sigma$ bytes. It is the same $\sigma$ at $t = 1$ and at $t = 128{,}000$.

A draft proposer — a small model, an n-gram lookup, an EAGLE head, a multi-token-prediction head — emits $k$ candidate tokens $d_1 \ldots d_k$. The target model runs **one** forward pass over $k+1$ positions (the $k$ drafts plus the position that produces the bonus token) and returns $k+1$ logit vectors. A rejection sampler walks them left to right and returns $j$, the number of drafted tokens accepted, with $0 \le j \le k$. The engine emits $j+1$ tokens and the sequence is now at position $t + j + 1$.

Every part of that is architecture-neutral except the last sentence. *The sequence is now at position $t+j+1$* is a claim about state, and the engine has to make it true. Concretely it has to arrange that every layer holds exactly what it would hold if the model had decoded $t+j+1$ tokens one at a time and never speculated. Anything less and you have silently corrupted the sequence.

For an attention layer that arrangement is trivial, and it is worth being precise about why. The target's forward pass wrote $k+1$ token-slots of K and V into the cache at positions $t, t+1, \ldots, t+k$. Those writes are **positionally addressed and mutually independent**: slot $t+3$ does not depend on slot $t+2$ having been written, because K and V at a position are a pure function of that position's hidden vector. So the cache after $j$ accepted tokens is *literally a prefix* of the cache after $k$ accepted tokens. Rolling back is deleting a suffix — set `seq_len = t + j + 1`, return any block that is now entirely past the end to the free list, and stop. The stale bytes inside the last partial block are never read because attention masks them out by length.

Try the same sentence on a recurrent layer and it falls apart at the second clause. The Mamba-2 style step is a recurrence:

$$h_{t} \;=\; A_t \odot h_{t-1} \;+\; B_t\, x_t^{\top}, \qquad y_t \;=\; C_t^{\top} h_{t}$$

where $h$ is the state matrix, and $A_t$, $B_t$, $C_t$ are input-dependent (that is the "selective" part). Read the first equation as an assignment statement, because that is what the kernel does: `h = A * h + B @ x`. The right-hand side consumes $h_{t-1}$ and the left-hand side destroys it. Position $t+3$'s state depends on position $t+2$'s state *having been computed and stored*, and the storing is the destroying.

So the three properties that make KV rollback free are all absent:

1. **There is no per-token dimension.** The state is one tensor for the whole prefix. There is nothing to index by position, nothing to slice, nothing to hand back to a block allocator.
2. **The update is not invertible in practice.** $h_{t-1} = (h_t - B_t x_t^\top) \oslash A_t$ is algebraically true and numerically useless: $A_t$ contains decay factors that are frequently very close to zero, so dividing by them amplifies rounding error without bound, and in bf16 you have ten mantissa bits to lose. Undoing four steps this way produces a tensor that is not the state you started from.
3. **The state is not recoverable from a suffix.** You can rebuild an attention layer's dropped blocks by re-prefilling those tokens. To rebuild $h_t$ you must replay the recurrence from $h_0$, which means the entire prefix, which means a full re-prefill of $t$ tokens.

That third point deserves emphasis because it is the one that surprises people. In a dense engine, *recompute* is always the fallback — it is how preemption works, it is how eviction works, it is why you can be sloppy with the cache. In a recurrent layer, recompute-from-nothing costs the whole prompt. The escape hatch you have leaned on for the entire series is priced out of the market.

![Timeline of a single speculation window showing snapshot, drafting, verification, rejection and restore, with the byte cost attached to each event](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-2.webp)

Lay the window out on a timeline and the shape of the bill becomes obvious. Seven events, and the two expensive ones are both copies of the same 49.4 MiB object: the snapshot you take before you dare advance, and the restore you perform when the draft turns out to be wrong. Everything else in the window — the four small draft passes, the one big verification pass, the rejection sampler's comparison — was already in your budget when you decided to speculate on a transformer. The two copies are new, and they are the whole problem.

---

## 2. The state you are trying to get back

Before pricing strategies, price the object. Everything downstream is a multiple of $\sigma$, the per-request state bytes, so it is worth deriving once, exactly, from a config file you can open yourself.

Take `nvidia/Nemotron-H-8B-Base-8K`. Its [published config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) gives `num_hidden_layers` 52 and a `hybrid_override_pattern` string containing 24 `M` (Mamba-2), 24 `-` (feed-forward) and 4 `*` (self-attention). That 4-to-24 split is a 1:6 interleave, which is the worked example the rest of this post uses. The relevant fields:

| Field | Value | What it drives |
| --- | --- | --- |
| `hidden_size` | 4096 | model width |
| `expand` | 2 | $d_{\text{inner}} = 8192$ |
| `mamba_num_heads` | 128 | state heads |
| `mamba_head_dim` | 64 | per-head channel width |
| `ssm_state_size` | 128 | state dimension $N$ |
| `n_groups` | 8 | shared B and C groups |
| `conv_kernel` | 4 | depthwise conv width |
| `num_key_value_heads` | 8 | KV heads (attention layers) |
| `attention_head_dim` | 128 | head dim (attention layers) |
| `torch_dtype` | bfloat16 | 2 bytes per element |

Per Mamba-2 layer there are two pieces of carried state. The **SSM state** is one matrix per head of shape (head dim × state dim):

$$S_{\text{ssm}} \;=\; 128 \times 64 \times 128 \times 2\ \text{B} \;=\; 2{,}097{,}152\ \text{B} \;=\; 2.00\ \text{MiB}$$

The **convolution state** is the last $k_{\text{conv}} - 1$ positions of the depthwise convolution's input, across all convolved channels. Those channels are $d_{\text{inner}}$ plus the B and C projections:

$$C_{\text{ch}} = 8192 + 2 \times 8 \times 128 = 10{,}240, \qquad S_{\text{conv}} = 10{,}240 \times 3 \times 2\ \text{B} = 61{,}440\ \text{B} = 60\ \text{KiB}$$

So one recurrent layer carries $S_b = 2{,}158{,}592$ bytes, or **2.06 MiB**, and across 24 of them:

$$\sigma \;=\; L_r \cdot S_b \;=\; 24 \times 2{,}158{,}592 \;=\; 51{,}806{,}208\ \text{B} \;=\; 49.41\ \text{MiB per request}$$

For contrast, the four attention layers contribute $2 \times 4 \times 8 \times 128 \times 2 = 16{,}384$ bytes per token, or 16 KiB per token — so at 8192 tokens of context, 128.0 MiB. Hold both numbers: **49.41 MiB of state that does not grow, and 128.0 MiB of KV that does.** Source for all of the above: derived from the linked config.

Now the number that decides everything downstream, and the one I have not seen stated plainly anywhere. What are the *inputs* to one recurrent step, in bytes? The scan at position $t$ consumes, per layer: the projected $x_t$ of width $d_{\text{inner}}$, the $B_t$ and $C_t$ vectors of width $n_{\text{groups}} \times N$ each, and the per-head timestep $\Delta_t$ of width $n_{\text{heads}}$.

$$I_b \;=\; \big(8192 + 1024 + 1024 + 128\big) \times 2\ \text{B} \;=\; 20{,}736\ \text{B} \;=\; 20.25\ \text{KiB}$$

Divide:

$$\frac{S_b}{I_b} \;=\; \frac{2{,}158{,}592}{20{,}736} \;\approx\; 104$$

**A Mamba-2 layer's state is about 104 times larger than the input that advances it by one token.** Every strategy in this post is a different answer to the question "given that ratio, what should you keep?" Keep states and you pay 104× more than you have to. Keep inputs and you pay 1×, but you have to be able to replay them, which means you have to be allowed to modify the kernel. That tension is the whole design space, and we can now walk it with real numbers instead of adjectives.

#### Worked example: the anatomy of one rollback

One request on Nemotron-H-8B, 8192 tokens of context, $k = 4$ drafted tokens, target rejects at $j = 2$. What moves?

- **Attention layers.** Four KV slots were written at positions 8192–8195. Accepting 2 plus the bonus token means the new length is 8195. Slots at 8195 are stale but inside a live block; nothing is copied, nothing is freed (a 16-token block boundary was not crossed). **Bytes moved: 0.**
- **Recurrent layers.** All 24 advanced their 2.06 MiB state through four positions. The state now summarizes tokens 1–8196 with three drafted tokens in it that the model just disowned. Getting to the correct state for 8195 tokens requires either a saved copy (49.41 MiB read plus 49.41 MiB write) or a replay of two steps from a saved copy. **Bytes moved: 98.8 MiB, minimum.**
- **Ratio:** the recurrent layers are 46% of this model's sequence-mixing layers and 100% of its rollback cost.

Source: derived from the config above. The one thing this example hides is that at batch 1 nobody cares — 98.8 MiB at 3.35 TB/s is 30 microseconds. Section 6 is about what happens when you multiply it by a batch size that makes your GPU worth owning.

---

## 3. Strategy one: checkpoint and restore

The first strategy is the one every engineer invents in the first thirty seconds, and it is correct, and it is what you should write first. Before advancing the state through drafted tokens, copy it. On rejection, copy it back. On full acceptance, throw the copy away.

Watch it happen once before we write it, because the thing that makes this hard is a *sequence* of overwrites and a still frame cannot show a sequence.

<figure class="blog-anim">
<svg viewBox="0 0 660 300" role="img" aria-label="A four token speculation window: KV slots append and two are dropped at no cost, while the recurrent state advances in place and must be copied back from a saved snapshot" style="width:100%;height:auto;max-width:820px">
<style>
.sr-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.sr-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.sr-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.sr-draft{fill:var(--accent,#6366f1);opacity:.85;stroke:none}
.sr-box{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2}
.sr-snap{fill:none;stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:6 5}
.sr-txt{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.sr-tag{font:600 12px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.sr-dot{fill:var(--accent,#6366f1)}
.sr-cut{stroke:#dc2626;stroke-width:2.5;stroke-dasharray:5 4}
.sr-cutlbl{font:600 12px ui-sans-serif,system-ui;fill:#dc2626}
@keyframes sr-d1{0%,12%{opacity:0}18%,100%{opacity:1}}
@keyframes sr-d2{0%,18%{opacity:0}24%,100%{opacity:1}}
@keyframes sr-d3{0%,24%{opacity:0}30%,62%{opacity:1}70%,100%{opacity:.14}}
@keyframes sr-d4{0%,30%{opacity:0}36%,62%{opacity:1}70%,100%{opacity:.14}}
@keyframes sr-cutin{0%,58%{opacity:0}64%,100%{opacity:1}}
@keyframes sr-snapin{0%{opacity:0}9%,100%{opacity:1}}
@keyframes sr-save{0%{opacity:0;transform:translateX(0)}6%{opacity:1}9%{opacity:1;transform:translateX(130px)}12%,100%{opacity:0;transform:translateX(130px)}}
@keyframes sr-back{0%,70%{opacity:0;transform:translateX(0)}74%{opacity:1;transform:translateX(0)}88%{opacity:1;transform:translateX(-130px)}92%,100%{opacity:0;transform:translateX(-130px)}}
@keyframes sr-s0{0%,36%{opacity:1}40%,74%{opacity:0}80%,100%{opacity:0}}
@keyframes sr-s4{0%,36%{opacity:0}40%,70%{opacity:1}76%,100%{opacity:0}}
@keyframes sr-s2{0%,76%{opacity:0}82%,100%{opacity:1}}
.sr-a1{animation:sr-d1 14s ease-in-out infinite}
.sr-a2{animation:sr-d2 14s ease-in-out infinite}
.sr-a3{animation:sr-d3 14s ease-in-out infinite}
.sr-a4{animation:sr-d4 14s ease-in-out infinite}
.sr-ac{animation:sr-cutin 14s ease-in-out infinite}
.sr-as{animation:sr-snapin 14s ease-in-out infinite}
.sr-av{animation:sr-save 14s ease-in-out infinite}
.sr-ab{animation:sr-back 14s ease-in-out infinite}
.sr-t0{animation:sr-s0 14s ease-in-out infinite}
.sr-t4{animation:sr-s4 14s ease-in-out infinite}
.sr-t2{animation:sr-s2 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sr-a1,.sr-a2,.sr-ac,.sr-as,.sr-t2{animation:none;opacity:1}.sr-a3,.sr-a4{animation:none;opacity:.14}.sr-av,.sr-ab,.sr-t0,.sr-t4{animation:none;opacity:0}}
</style>
<text class="sr-lbl" x="22" y="26">attention KV</text>
<text class="sr-sub" x="128" y="26">16 KiB per token, appended</text>
<rect class="sr-slot" x="30" y="44" width="64" height="46" rx="7"/>
<rect class="sr-slot" x="102" y="44" width="64" height="46" rx="7"/>
<text class="sr-txt" x="62" y="73">t-1</text>
<text class="sr-txt" x="134" y="73">t</text>
<rect class="sr-draft sr-a1" x="174" y="44" width="64" height="46" rx="7"/>
<rect class="sr-draft sr-a2" x="246" y="44" width="64" height="46" rx="7"/>
<rect class="sr-draft sr-a3" x="318" y="44" width="64" height="46" rx="7"/>
<rect class="sr-draft sr-a4" x="390" y="44" width="64" height="46" rx="7"/>
<text class="sr-tag sr-a1" x="206" y="73">d1</text>
<text class="sr-tag sr-a2" x="278" y="73">d2</text>
<text class="sr-tag sr-a3" x="350" y="73">d3</text>
<text class="sr-tag sr-a4" x="422" y="73">d4</text>
<line class="sr-cut sr-ac" x1="314" y1="36" x2="314" y2="100"/>
<text class="sr-cutlbl sr-ac" x="322" y="114">reject at j = 2</text>
<text class="sr-sub" x="470" y="73">rollback: 0 bytes</text>
<line x1="22" y1="140" x2="638" y2="140" stroke="var(--border,#d1d5db)" stroke-width="1"/>
<text class="sr-lbl" x="22" y="172">recurrent state</text>
<text class="sr-sub" x="140" y="172">49.4 MiB, one tensor, no positions</text>
<rect class="sr-box" x="174" y="190" width="136" height="56" rx="10"/>
<text class="sr-txt sr-t0" x="242" y="224">live = S_t</text>
<text class="sr-txt sr-t4" x="242" y="224">live = S_t+4</text>
<text class="sr-txt sr-t2" x="242" y="224">live = S_t+2</text>
<rect class="sr-snap sr-as" x="440" y="190" width="150" height="56" rx="10"/>
<text class="sr-txt sr-as" x="515" y="215">snapshot S_t</text>
<text class="sr-sub sr-as" x="474" y="234">49.4 MiB copy</text>
<circle class="sr-dot sr-av" cx="318" cy="218" r="7"/>
<circle class="sr-dot sr-ab" cx="432" cy="218" r="7"/>
<text class="sr-sub" x="22" y="278">save 49.4 MiB before drafting, restore 49.4 MiB on rejection, then replay 2 steps to reach S_t+2</text>
</svg>
<figcaption>The same four drafted tokens in both lanes. The KV lane drops two slots and copies nothing; the state lane advances in place, loses S_t, and has to pull it back from a snapshot before replaying the two accepted steps.</figcaption>
</figure>

Two details in that loop are worth naming. First, the snapshot is taken *unconditionally*, before you know whether you will need it — that is the tax you pay on every window, including the windows where all four drafts are accepted and the copy is thrown away. Second, restoring $S_t$ is not the end of the job: $S_t$ is the state for the verified prefix, and you accepted two tokens, so you still owe two forward steps of the recurrence before the state is correct. Restore is a copy *plus* a replay. Engines that get this wrong restore the snapshot and forget the replay, which produces a model that has silently un-read the tokens it just emitted.

### The state handle

Here is the first piece of `nanoserve/rollback.py`. It gives the engine a single object per request that owns both the paged KV metadata and the recurrent tensors, because the whole point of this post is that rollback has to touch both and they behave differently.

```python
# nanoserve/rollback.py
from __future__ import annotations
from dataclasses import dataclass, field
import torch

@dataclass
class HybridStateSpec:
    """Per-layer shapes for one hybrid model, read from config.json."""
    n_layers: int
    recurrent_layers: list[int]      # indices of Mamba-2 / GDN layers
    attention_layers: list[int]      # indices of full-attention layers
    n_heads: int                     # mamba_num_heads
    head_dim: int                    # mamba_head_dim
    d_state: int                     # ssm_state_size
    conv_channels: int               # d_inner + 2 * n_groups * d_state
    conv_kernel: int                 # conv_kernel
    dtype: torch.dtype = torch.bfloat16

    @property
    def ssm_bytes(self) -> int:
        el = torch.empty((), dtype=self.dtype).element_size()
        return self.n_heads * self.head_dim * self.d_state * el

    @property
    def conv_bytes(self) -> int:
        el = torch.empty((), dtype=self.dtype).element_size()
        return self.conv_channels * (self.conv_kernel - 1) * el

    @property
    def bytes_per_layer(self) -> int:
        return self.ssm_bytes + self.conv_bytes

    @property
    def sigma(self) -> int:
        """Total recurrent state bytes for one request."""
        return len(self.recurrent_layers) * self.bytes_per_layer


NEMOTRON_H_8B = HybridStateSpec(
    n_layers=52,
    recurrent_layers=list(range(24)),   # stand-in; parse hybrid_override_pattern for real
    attention_layers=[3, 9, 15, 21],
    n_heads=128, head_dim=64, d_state=128,
    conv_channels=8192 + 2 * 8 * 128,
    conv_kernel=4,
)

print(f"per recurrent layer : {NEMOTRON_H_8B.bytes_per_layer/2**20:.2f} MiB")
print(f"sigma per request   : {NEMOTRON_H_8B.sigma/2**20:.2f} MiB")
```

```console
per recurrent layer : 2.06 MiB
sigma per request   : 49.41 MiB
```

Those two lines are the arithmetic from section 2, reproduced by code rather than asserted, and they are the only two numbers the rest of the file needs.

### Snapshot and restore

Now the actual mechanism. The critical design decision is that snapshot buffers are **preallocated once per sequence slot**, not allocated per window. A speculation window happens every decode step; allocating 49 MiB per step would hand the allocator a job it cannot do at that rate, and would fragment the pool that your KV blocks live in. Preallocate at admission, reuse forever, free at completion — the same lifetime discipline the recurrent state itself already has.

```python
# nanoserve/rollback.py  (continued)

class RecurrentStatePool:
    """Live state plus one snapshot buffer per sequence slot.

    Layout mirrors what a fused scan kernel wants: state for all slots of one
    layer is contiguous, so a batched step reads a single tensor per layer.
    """

    def __init__(self, spec: HybridStateSpec, max_seqs: int, device="cuda"):
        self.spec, self.max_seqs = spec, max_seqs
        L = len(spec.recurrent_layers)
        # (layers, slots, heads, head_dim, d_state)
        self.ssm = torch.zeros(
            L, max_seqs, spec.n_heads, spec.head_dim, spec.d_state,
            dtype=spec.dtype, device=device)
        # (layers, slots, channels, kernel-1)
        self.conv = torch.zeros(
            L, max_seqs, spec.conv_channels, spec.conv_kernel - 1,
            dtype=spec.dtype, device=device)
        # snapshot mirrors, same shapes
        self.ssm_snap = torch.empty_like(self.ssm)
        self.conv_snap = torch.empty_like(self.conv)
        self.snap_valid = torch.zeros(max_seqs, dtype=torch.bool, device=device)

    def snapshot(self, slots: torch.Tensor) -> None:
        """Save live -> snap for the given sequence slots. One copy per window."""
        self.ssm_snap[:, slots] = self.ssm[:, slots]
        self.conv_snap[:, slots] = self.conv[:, slots]
        self.snap_valid[slots] = True

    def restore(self, slots: torch.Tensor) -> None:
        """Copy snap -> live. Only correct if snapshot() ran this window."""
        assert bool(self.snap_valid[slots].all()), "restore without snapshot"
        self.ssm[:, slots] = self.ssm_snap[:, slots]
        self.conv[:, slots] = self.conv_snap[:, slots]

    def bytes_moved_per_snapshot(self, n_slots: int) -> int:
        """Read + write, which is what the bandwidth bill actually sees."""
        return 2 * n_slots * self.spec.sigma
```

Three things to notice, because each is a place production engines have gone wrong.

`self.ssm_snap[:, slots] = self.ssm[:, slots]` is an **advanced-indexing gather followed by a scatter**, which materializes an intermediate. For a hot path that runs every decode step you want `index_select` into a preallocated staging tensor, or better, a single fused copy kernel over the slot list. I have written it the readable way; if you profile a real implementation and find the snapshot costing more than the $2 B \sigma$ the arithmetic predicts, this line is where the extra went.

`snap_valid` exists because "restore without a matching snapshot" is a silent corruption, not a crash. The state will contain *something*, the model will keep generating, and the output will drift. An assert here costs nothing and converts a class of heisenbug into a stack trace.

And the snapshot covers **both** tensors. The convolution state is only 60 KiB of the 2.06 MiB, which makes it exactly the kind of thing people forget. Forgetting it produces a model whose recurrence is correct and whose short-range convolution is three tokens ahead of itself — subtly wrong output that passes a smoke test and fails on long generations.

### The asymmetric rollback

Now the function this whole post exists for. One call, two completely different mechanisms, chosen by layer type.

```python
# nanoserve/rollback.py  (continued)

@dataclass
class SpecWindow:
    """Bookkeeping for one draft-and-verify round on one sequence."""
    slot: int
    base_pos: int          # verified length before drafting
    k: int                 # number of drafted tokens
    accepted: int = 0      # j, filled in by the rejection sampler

def rollback(engine, win: SpecWindow, accepted: int) -> int:
    """Return the engine to the state of (base_pos + accepted + 1) tokens.

    Attention layers: truncate. Recurrent layers: restore, then replay.
    """
    win.accepted = accepted
    new_len = win.base_pos + accepted + 1        # + 1 for the bonus token

    # --- attention lane: a pointer move and maybe some freed blocks ---
    engine.block_table.set_length(win.slot, new_len)
    freed = engine.block_table.trim_blocks_past(win.slot, new_len)

    # --- recurrent lane: the state for new_len does not exist any more ---
    slots = torch.tensor([win.slot], device=engine.device)
    if accepted == win.k:
        # every draft accepted: the live state is already correct
        engine.state_pool.snap_valid[slots] = False
        return freed

    engine.state_pool.restore(slots)             # back to base_pos
    if accepted + 1 > 0:
        # replay the tokens we actually kept, through the recurrent layers only
        engine.replay_recurrent(
            slot=win.slot,
            token_ids=engine.output_ids[win.slot][win.base_pos : new_len],
        )
    engine.state_pool.snap_valid[slots] = False
    return freed
```

Read the two lanes side by side and the asymmetry is impossible to miss. The attention lane is two integer operations and a free-list splice; its cost does not depend on $k$, on $\sigma$, or on the batch. The recurrent lane is a full-state copy plus a replay whose length is the number of tokens you *kept* — you pay most when you succeed, which is a genuinely perverse incentive that section 6 quantifies.

Note also the `accepted == win.k` fast path. When every drafted token is accepted, the live state is already exactly right and the snapshot is dead weight. At an acceptance rate of $\alpha = 0.7$ and $k = 4$, that happens $0.7^4 = 24\%$ of the time — so about a quarter of your snapshots are pure waste, and you cannot know which quarter in advance.

![A branching diagram in which a snapshot leads into four drafted tokens that split into a fully accepted path and a rejected path, both rejoining at the next speculation window](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-3.webp)

The branch structure above is what the code encodes. One window, one snapshot, then a fork the sampler decides: the accept path walks straight to the next window with the live state already correct, and the reject path detours through a restore and a replay before rejoining at exactly the same place. The two paths are indistinguishable from outside the engine — same tokens, same distribution — and differ by 49.4 MiB of traffic per request. An engine that reports a flat "tokens per second" number is averaging over that fork, which is why your speculative throughput on a hybrid is much more sensitive to acceptance rate than the standard speedup formula suggests.

### Driving it: the speculation loop

```python
# nanoserve/rollback.py  (continued)

@torch.inference_mode()
def speculate_step(engine, slot: int, k: int = 4) -> list[int]:
    """One draft-and-verify window with checkpoint rollback. Returns new tokens."""
    win = SpecWindow(slot=slot, base_pos=engine.seq_len(slot), k=k)
    slots = torch.tensor([slot], device=engine.device)

    # 1. save the state we may have to come back to (unconditional)
    engine.state_pool.snapshot(slots)

    # 2. draft k tokens with the cheap proposer (n-gram, EAGLE head, small model)
    drafts = engine.proposer.propose(slot, k)                 # list[int], len k

    # 3. one target forward over k+1 positions; this ADVANCES the recurrent
    #    state through all k drafted tokens, destroying S_t in the process
    logits = engine.target_forward(slot, drafts)              # (k+1, vocab)

    # 4. rejection sampling decides how many drafts survive
    accepted, bonus = engine.sampler.verify(logits, drafts)   # j, next token

    # 5. put every layer back into a consistent state for base_pos + j + 1
    rollback(engine, win, accepted)

    return drafts[:accepted] + [bonus]
```

That is a complete, honest speculative decoder for a hybrid model in twenty lines, and it is what I would ship first. It is correct, it is obvious, it has one failure mode (forgetting the replay in step 5) and one assert that catches it. Its problem is not correctness. Its problem is that step 1 costs $2\sigma$ bytes of bandwidth per request per window whether or not it turns out to be needed, and at the batch sizes where a serving engine earns its keep, that is not a rounding error. Section 6 puts a number on it. First, the two strategies that attack it from opposite directions.

---

## 4. Strategy two: keep every intermediate state

If the problem is that you destroyed $S_t$, the brute-force answer is to stop destroying anything. Advance the state through the drafted tokens, but write each intermediate result to a *different* buffer: $S_t$ stays where it is, $S_{t+1}$ goes to buffer 1, $S_{t+2}$ to buffer 2, and so on. When the sampler returns $j$, you do not copy anything at all — you swap a pointer so the live state *is* buffer $j$, and the other buffers become scratch for the next window.

```python
# nanoserve/rollback.py  (continued)

class KeepKStatePool:
    """k+1 state versions per slot; rollback is an index, not a copy."""

    def __init__(self, spec: HybridStateSpec, max_seqs: int, k: int, device="cuda"):
        self.spec, self.k = spec, k
        L = len(spec.recurrent_layers)
        # one extra version axis: [0] is the verified state, [1..k] the drafts
        self.ssm = torch.zeros(
            k + 1, L, max_seqs, spec.n_heads, spec.head_dim, spec.d_state,
            dtype=spec.dtype, device=device)
        self.conv = torch.zeros(
            k + 1, L, max_seqs, spec.conv_channels, spec.conv_kernel - 1,
            dtype=spec.dtype, device=device)
        # which version index is "live" for each slot
        self.live = torch.zeros(max_seqs, dtype=torch.long, device=device)

    def version(self, v: int):
        return self.ssm[v], self.conv[v]

    def commit(self, slot: int, accepted: int) -> None:
        """Rollback: version (accepted+1) becomes the new verified state."""
        v = accepted + 1                        # +1 for the bonus token
        if v != 0:
            self.ssm[0, :, slot] = self.ssm[v, :, slot]
            self.conv[0, :, slot] = self.conv[v, :, slot]
        self.live[slot] = 0

    def extra_bytes_per_request(self) -> int:
        return self.k * self.spec.sigma


pool = KeepKStatePool(NEMOTRON_H_8B, max_seqs=256, k=4, device="meta")
print(f"extra per request : {pool.extra_bytes_per_request()/2**20:.1f} MiB")
print(f"extra at batch 64 : {64*pool.extra_bytes_per_request()/2**30:.2f} GiB")
```

```console
extra per request : 197.6 MiB
extra at batch 64 : 12.35 GiB
```

Twelve gigabytes. On an 80 GB H100 that is more than the model weights, spent entirely on the ability to undo. And note that `commit` still copies once — moving version $j+1$ down to version 0 — because the next window's scan wants to start from a known index. You can avoid even that with a rotating base index, at the cost of a level of indirection in every kernel that touches the state, which is the sort of thing that turns a clean Triton kernel into a maze.

So keep-k is the expensive option, and I would not reach for it to solve the problem in section 3. But it is not a strawman, because there is one thing it does that neither other strategy can: **it gives you random access to every position in the window.** That matters the moment your drafts stop being a chain.

Chain drafting proposes one sequence of $k$ tokens. Tree drafting — Medusa heads, EAGLE-2, any of the tree-speculation schemes covered in [the tree speculation post](/blog/machine-learning/speculative-decoding/tree-speculation-drafting-multiple-futures) — proposes a *tree* of candidate continuations and verifies many of them in one pass. In an attention layer that works beautifully: you build a tree attention mask so each candidate node attends only to its ancestors, and one forward pass evaluates every branch. The KV cache holds all the nodes at once and the mask sorts out who sees whom.

There is no mask trick for a recurrence. A state is not a set of positions you can mask; it is a single accumulated value, and two sibling branches of the draft tree need two *different* states because they consumed different tokens. Evaluating a tree of $m$ nodes on a recurrent layer requires materializing up to $m$ distinct states, or serializing the branches. This is not my inference — it is one of the three challenges named explicitly in [SpecMamba (arXiv 2509.19873, Zhong et al., ICCAD 2025)](https://arxiv.org/abs/2509.19873), whose abstract lists "tree-based parallel verification incompatibility" alongside "hidden state backtracking difficulties" as the obstacles to speculative decoding on SSMs. Their answer is a first-in-first-out tree verification scheme with tiling to bound the memory access, paired with a "memory-aware hybrid backtracking strategy" — which is to say, they too concluded that no single rollback mechanism wins and blended them.

**So the rule is:** chain drafting on a recurrent layer has three viable rollback strategies; tree drafting has one, and it is the expensive one. If your engine's speculative speedup comes from tree drafting, budget $m \cdot \sigma$ and plan your batch size around it before you write a line of code.

---

## 5. Strategy three: do not advance the state at all

The third strategy inverts the question. Checkpointing and keep-k both accept that the verification pass will advance the state and then arrange to undo it. What if it simply did not advance it?

This is more achievable than it sounds, because of a property of the scan kernel that section 2 already exposed. Verifying $k+1$ positions on a recurrent layer is a **chunked scan**: the kernel loads $S_t$, walks the chunk producing an output for every position, and stores the final state. The intermediate states live in registers and shared memory and are never written to HBM. So the kernel already computes all $k+1$ outputs without materializing $k+1$ states — the only thing that makes the operation destructive is the *final store*.

Suppress that store and you get verification for free, in rollback terms. The kernel returns $k+1$ output vectors, the sampler picks $j$, and the canonical state in HBM is still $S_t$, untouched. Then you run a second, tiny scan over exactly the $j+1$ accepted tokens and store *that* result. The state was never wrong, so nothing has to be undone.

The price is the second scan, and this is where a naive implementation quietly loses. To re-run the recurrence over the accepted tokens you need their per-token SSM inputs — the $x$, $B$, $C$ and $\Delta$ from section 2 — and those are the output of the layer's input projection. If you did not keep them, you must recompute them, which means reading the projection weights of every recurrent layer out of HBM a second time. For Nemotron-H-8B those projections are the bulk of the Mamba layers' parameters: per layer, an input projection of $4096 \times 18{,}560$ and an output projection of $8192 \times 4096$, about 110 M parameters, times 24 layers is roughly 2.6 B of the model's 8 B. Call it a third of the weights, or about 5.3 GB in bf16, re-read on every window that rejects.

Which is exactly why you keep the inputs instead. They are $I_b$ = 20.25 KiB per token per layer, against $S_b$ = 2.06 MiB of state — the 104× ratio from section 2, and it is the whole reason this strategy is worth building.

```python
# nanoserve/rollback.py  (continued)

class ReplayBuffer:
    """Cache the scan's INPUTS for the window, not its states.

    20.25 KiB per token per layer instead of 2.06 MiB of state per layer,
    for the Nemotron-H-8B shapes. Rollback becomes a length assignment.
    """

    def __init__(self, spec: HybridStateSpec, max_seqs: int, k: int, device="cuda"):
        self.spec, self.k = spec, k
        L, W = len(spec.recurrent_layers), k + 1
        d_inner = spec.n_heads * spec.head_dim
        n_groups = (spec.conv_channels - d_inner) // (2 * spec.d_state)
        self.x = torch.zeros(L, max_seqs, W, d_inner, dtype=spec.dtype, device=device)
        self.B = torch.zeros(L, max_seqs, W, n_groups * spec.d_state,
                             dtype=spec.dtype, device=device)
        self.C = torch.zeros(L, max_seqs, W, n_groups * spec.d_state,
                             dtype=spec.dtype, device=device)
        self.dt = torch.zeros(L, max_seqs, W, spec.n_heads,
                              dtype=spec.dtype, device=device)
        self.valid_len = torch.zeros(max_seqs, dtype=torch.long, device=device)

    def bytes_per_token_per_layer(self) -> int:
        el = torch.empty((), dtype=self.spec.dtype).element_size()
        d_inner = self.spec.n_heads * self.spec.head_dim
        n_groups = (self.spec.conv_channels - d_inner) // (2 * self.spec.d_state)
        return (d_inner + 2 * n_groups * self.spec.d_state + self.spec.n_heads) * el

    def extra_bytes_per_request(self) -> int:
        return (self.k + 1) * len(self.spec.recurrent_layers) \
            * self.bytes_per_token_per_layer()

    def truncate(self, slot: int, accepted: int) -> None:
        """The entire rollback. No copy, no kernel, one integer."""
        self.valid_len[slot] = accepted + 1


rb = ReplayBuffer(NEMOTRON_H_8B, max_seqs=256, k=4, device="meta")
print(f"inputs / token / layer : {rb.bytes_per_token_per_layer()/2**10:.2f} KiB")
print(f"state  / layer         : {NEMOTRON_H_8B.bytes_per_layer/2**20:.2f} MiB")
print(f"ratio                  : "
      f"{NEMOTRON_H_8B.bytes_per_layer / rb.bytes_per_token_per_layer():.0f}x")
print(f"replay buffer / request: {rb.extra_bytes_per_request()/2**20:.2f} MiB")
```

```console
inputs / token / layer : 20.25 KiB
state  / layer         : 2.06 MiB
ratio                  : 104x
replay buffer / request: 2.37 MiB
```

**2.37 MiB against 49.41 MiB.** That is the same undo, bought for one twentieth of the memory, and the rollback itself — `self.valid_len[slot] = accepted + 1` — moves no data whatsoever. The scan for the *next* window starts from the checkpoint state and folds in whatever inputs are still marked valid.

I want to be careful about credit here, because this is not my idea and the version of it that works well is more refined than the sketch above. Tri Dao published **ReplaySSM** on [his blog on 15 June 2026](https://tridao.me/blog/2026/replayssm/), with a corresponding [vLLM RFC (issue #47572)](https://github.com/vllm-project/vllm/issues/47572), and the framing is exactly "cache SSM inputs instead of state". His version keeps a checkpoint state plus a **ring buffer** of recent inputs, computes each step's output from the checkpoint plus the buffered inputs without materializing the full state, and only folds the buffer into the checkpoint on a periodic *flush* step — which means the dominant state traffic is paid once every $L$ steps rather than every step. For Mamba-2 the buffer holds $(v, k)$ pairs and decay factors; for gated DeltaNet it caches the pre-computed correction term. On speculative decoding specifically, the post states that the baseline stores "a full state snapshot for every draft position" while ReplaySSM gets "O(1) rollback via pointer move", because the rejected draft inputs are simply still sitting in the buffer and you move the pointer past them.

The numbers he reports, with the caveat that I am citing them and have measured nothing: up to 1.48× end-to-end on standard decoding (1.43× on large mixture-of-experts models), kernel speedups of 1.43× to 1.84×, 1.87–1.96× for speculative decoding over standard decoding and up to 2.14× over the vLLM baseline, and support for 3.0–3.3× more concurrent requests under a fixed memory budget. The mechanism claim underneath them is the one I would trust most because it is arithmetic: writing the state back once every $L$ steps instead of every step roughly halves the dominant state traffic, from $8dn$ to $4dn$ bytes per head per step.

That concurrency figure is worth pausing on. It is not primarily a speculative-decoding result — it is the flat consequence of the fact that the per-request resident state shrinks when you stop keeping a state per draft position, which is precisely the axis section 6 is about to derive.

![A four by four table comparing checkpoint, keep-k states, re-scan and replay inputs across extra memory, extra traffic, undo cost and the situation each suits](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-4.webp)

Four strategies, four currencies. Read the table above by column rather than by row: the *extra memory* column separates the two viable strategies from the two expensive ones, the *undo cost* column shows that only checkpoint actually moves bytes at rollback time, and the *reach for it when* column is the honest answer — the choice is dictated less by elegance than by whether you are allowed to modify the scan kernel and how wide your draft structure is. Every figure in that table is derived in this post for Nemotron-H-8B at $k = 4$; on a model with a different state size the numbers move but the ordering does not.

---

## 6. The mechanism: pricing the four strategies

Time to do this properly, with symbols, so you can plug in your own model instead of borrowing mine.

Let

- $k$ — drafted tokens per window,
- $j$ — accepted tokens, $0 \le j \le k$, with mean $\bar{j}$,
- $B$ — batch size, the number of sequences speculating concurrently,
- $L_r$ — number of recurrent layers,
- $S_b$ — recurrent state bytes per layer per request, so $\sigma = L_r S_b$,
- $I_b$ — SSM input bytes per token per layer,
- $W$ — total model weight bytes, $f_r$ the fraction of them in recurrent layers,
- $K_b$ — KV bytes per token, $S$ the context length,
- $\text{BW}$ — achieved HBM bandwidth,
- $p_{\text{rej}} = 1 - \alpha^{k}$ — probability a window rejects at least one draft, for per-token acceptance rate $\alpha$.

**The baseline window.** One target forward pass over $k+1$ positions moves the weights once, the KV cache once, and the state once (a single chunked scan reads $\sigma$ and writes $\sigma$):

$$T_{\text{window}} \;=\; \frac{W \;+\; B\,K_b\,S \;+\; 2B\sigma}{\text{BW}}$$

That third term is already interesting on its own, and it is a fact about hybrid decoding that has nothing to do with speculation. State traffic overtakes weight traffic when $2B\sigma \gt W$, that is at

$$B^{\dagger} \;=\; \frac{W}{2\sigma}$$

For Nemotron-H-8B — 8.0 B parameters in bf16, so $W \approx 16.0$ GB, and $\sigma = 51.8$ MB — that is $B^{\dagger} \approx 154$. **Past batch 154, this model's decode step is bound by moving recurrent state, not by moving weights.** The flat term that made hybrids attractive at long context becomes the dominant traffic at high concurrency. Keep that in your pocket; it is the reason the rollback strategies diverge so violently as batch grows.

**Checkpoint and restore.** Adds one unconditional save and one conditional restore, each a read plus a write of $\sigma$ per request:

$$\Delta T_{\text{ckpt}} \;=\; \frac{2B\sigma \,\bigl(1 + p_{\text{rej}}\bigr)}{\text{BW}}, \qquad \Delta M_{\text{ckpt}} \;=\; B\sigma$$

Note what is *absent* from that expression: $k$. The checkpoint cost is completely independent of draft length, which makes it the strategy of choice if you want to push $k$ up. Note also that it scales with $B$ in lockstep with the baseline's state term, so as a *fraction* of the window it is roughly constant at $2(1+p_{\text{rej}})\sigma B / (W + BK_bS + 2B\sigma)$ — which starts small and asymptotes to $(1+p_{\text{rej}})$ over $(1 + K_bS/2\sigma)$ as $B$ grows.

**Keep-k states.** No copy at rollback, but the scan must spill a state per position instead of one at the end, and the buffers are resident for the request's whole life:

$$\Delta T_{\text{keep}} \;=\; \frac{k B \sigma}{\text{BW}}, \qquad \Delta M_{\text{keep}} \;=\; k B \sigma$$

Both terms scale with $k$. This is the strategy that punishes you for drafting deeper, and drafting deeper is the main lever you have on speculative speedup.

**Re-scan without cached inputs.** No extra resident memory at all, and no state copy — but the accepted-token replay re-reads the recurrent layers' projection weights:

$$\Delta T_{\text{rescan}} \;=\; \frac{f_r W \, p_{\text{rej}}}{\text{BW}}, \qquad \Delta M_{\text{rescan}} \;=\; 0$$

The striking property here is that the cost is **independent of $B$**. A weight re-read is a weight re-read whether one sequence needs it or two hundred do. That immediately gives the break-even the whole section has been building toward. Re-scan beats checkpoint when

$$f_r W \, p_{\text{rej}} \;\lt\; 2B\sigma\,(1 + p_{\text{rej}}) \quad\Longrightarrow\quad B^{*} \;=\; \frac{f_r W \, p_{\text{rej}}}{2\sigma\,(1 + p_{\text{rej}})}$$

With $f_r \approx 1/3$ (derived above from the projection matrices, and a lower bound since it ignores the small per-head parameters), $W = 16.0$ GB, $\sigma = 51.8$ MB, and $p_{\text{rej}} = 1 - 0.7^4 = 0.76$:

$$B^{*} \;=\; \frac{0.33 \times 16.0\times 10^{9} \times 0.76}{2 \times 51.8\times 10^{6} \times 1.76} \;\approx\; 22$$

**Below batch 22, snapshot the state. Above it, re-scan.** The crossover is that low because $\sigma$ is large and the recurrent projections are a modest slice of the weights; on a model with a smaller state the crossover moves right, and on a model that is mostly recurrent it moves left. The formula is the deliverable, not the number.

**Replay cached inputs.** The synthesis: no state copy, no weight re-read, and a buffer that is $I_b/S_b$ of the state:

$$\Delta T_{\text{replay}} \;=\; \frac{B L_r I_b \bigl((k+1) + \bar{j}\,\bigr)}{\text{BW}}, \qquad \Delta M_{\text{replay}} \;=\; (k+1)\, B L_r I_b$$

Compare the traffic term against checkpoint's. Dividing both by $B L_r$, replay wins whenever

$$I_b\bigl((k+1) + \bar{j}\bigr) \;\lt\; 2 S_b \bigl(1 + p_{\text{rej}}\bigr)$$

With $S_b / I_b \approx 104$, $k = 4$ and $\bar{j} \approx 2.8$, the left side is about 7.8 input-units and the right side about 366. Replay wins by roughly a factor of 47, at every batch size, for every draft length up to $k \approx 360$. There is no crossover to find. The only reason not to build it is that it requires the scan kernel to accept a variable-length input buffer and start from a checkpoint — which, if you are calling a vendor kernel, you are not allowed to do.

### The numbers, with provenance

Nemotron-H-8B, one H100 SXM 80 GB, 8192 tokens of context, $k = 4$, $\alpha = 0.7$. Free memory after weights: 80 GB is 74.5 GiB, weights are 16.0 GB or 14.9 GiB, and leaving about 4 GiB for activations and workspace gives roughly 55.6 GiB for per-request state.

| Strategy | Extra bytes / request | Per-request total at 8k | Concurrency at 55.6 GiB | Source |
| --- | --- | --- | --- | --- |
| no speculation | 0 | 177.4 MiB | 320 | derived |
| checkpoint | 49.4 MiB | 226.8 MiB | 251 | derived |
| keep-4 states | 197.6 MiB | 375.0 MiB | 151 | derived |
| re-scan (no cache) | 0 | 177.4 MiB | 320 | derived |
| replay inputs | 2.4 MiB | 179.8 MiB | 316 | derived |

Per-request total is $\sigma$ plus $8192 \times 16$ KiB of KV plus the strategy's extra. The arithmetic for the middle row: $49.41 + 197.62 + 128.0 = 375.03$ MiB, and $56{,}934 / 375.03 = 151.8$.

The concurrency column is the one that should change your mind. **Turning on keep-4 speculative decoding cuts your maximum batch from 320 to 151** — you are trading away 53% of your concurrency for a per-request latency win. If your service is throughput-limited, that trade is a straight loss no matter how good the acceptance rate is. Checkpoint costs 22% of concurrency. Replay costs 1.3%.

![A layered breakdown of one request's memory showing KV blocks and live state as the baseline and the three rollback strategies as surcharges on top](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-5.webp)

Stacking the same numbers makes the outlier obvious: the keep-4 surcharge is larger than the entire 8k KV cache of the request it belongs to. That is a genuinely strange sentence to write about a model whose selling point is that it barely has a KV cache, and it is the clearest illustration I know of the trap in hybrid serving — you retire one memory term and a different one walks in through a door you did not know existed.

#### Worked example: the rollback tax at batch 32 and at batch 251

Take the checkpoint strategy and price one window at two batch sizes, using the H100 SXM datasheet figure of 3.35 TB/s of HBM3 bandwidth ([NVIDIA H100 datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet)). These are theoretical-bandwidth numbers and therefore optimistic; a real scan kernel lands somewhere around 60–80% of peak, so scale the times up accordingly.

**At batch 32**, 8192 tokens of context:

- weights: 16.0 GB
- KV read: $32 \times 134.2$ MB = 4.29 GB
- state read and write: $2 \times 32 \times 51.8$ MB = 3.32 GB
- window baseline: 23.6 GB, or **7.04 ms**
- checkpoint save: 3.32 GB, or 0.99 ms
- restore, expected: $0.76 \times 3.32$ GB = 2.52 GB, or 0.75 ms
- **rollback tax: 1.74 ms on a 7.04 ms window, 25%**

**At batch 251** (the concurrency the checkpoint strategy allows):

- weights: 16.0 GB
- KV read: $251 \times 134.2$ MB = 33.7 GB
- state read and write: $2 \times 251 \times 51.8$ MB = 26.0 GB
- window baseline: 75.7 GB, or **22.6 ms**
- checkpoint save plus expected restore: $1.76 \times 26.0$ GB = 45.8 GB, or **13.7 ms**
- **rollback tax: 61% of the window**

Source for every line: derived from the config-based byte counts in section 2 and the cited bandwidth figure. Same code, same model, same draft length — and the overhead goes from an annoyance to the largest single term in the step. With replay-inputs instead, the batch-251 tax is $251 \times 24 \times 20{,}736 \times 7.8$ bytes = 0.97 GB, or 0.29 ms: **1.3% instead of 61%.**

#### Worked example: what the tax does to the actual speedup

A window emits $1 + \bar{j}$ tokens where $\bar{j} = \sum_{i=1}^{k} \alpha^{i}$. At $\alpha = 0.7$, $k = 4$: $\bar{j} = 0.7 + 0.49 + 0.343 + 0.2401 = 1.773$, so 2.773 tokens per window. Ignore the draft proposer's cost for a moment (it is a separate, well-understood term — see [the core spec-dec post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify)) and ask only what the rollback tax does.

At batch 32, per-token time goes from $7.04/2.773 = 2.54$ ms to $8.78/2.773 = 3.17$ ms — the speculative decoder is still far ahead of the 7.04 ms it would spend per token without speculation, and the tax has eaten about 20% of the win. At batch 251, per-token time goes from $22.6/2.773 = 8.15$ ms to $36.3/2.773 = 13.1$ ms, against 22.6 ms unspeculated. Speculation still wins, but a 2.77× theoretical speedup has been clipped to 1.73×, and **all of the loss is rollback, none of it is drafting**. Source: derived from the two lines above.

That is the shape of the problem in one paragraph. Speculative decoding on a hybrid is not broken; it is *taxed*, the tax scales with batch, and every engineering hour you spend on the rollback path buys back speedup that the draft model already earned.

---

## 7. The same problem, wearing four other hats

Rollback is the sharpest version of this problem, but it is not the only one. The underlying fact is more general and worth stating on its own:

> A KV cache can be **forked cheaply** — two sequences share physical blocks and copy only on write. A recurrent state cannot be forked at all; it must be **copied in full, immediately**.

Everywhere your engine branches or rewinds a sequence, that sentence bites.

![A two lane grid comparing the cost of forking a sequence one, two and four ways in the KV lane against the recurrent state lane](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-6.webp)

The grid above prices the fork. In the KV lane, forking is a reference-count increment on the shared blocks — the mechanism from [the prefix sharing and copy-on-write post](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) — so going from one sample to four costs zero bytes until the branches start writing, and then costs one 16-token block each. In the state lane there is no shared representation and no write barrier to hang copy-on-write off: the branches diverge at the first token, so you copy $\sigma$ per branch, immediately. Four samples cost $3\sigma = 148.2$ MiB. Here are the four places that shows up.

**`n>1` sampling.** A user asks for four completions of the same prompt. On a transformer this is nearly free: prefill once, fork the block table four ways with a shared prefix, and pay only for the divergent suffixes. On a hybrid you prefill once and then immediately clone the state four ways — 148.2 MiB of copies for Nemotron-H-8B, before a single token has been generated. The economics of `n>1` invert: on a transformer it is the cheapest way to get diversity, and on a hybrid it is a per-sample tax with no prefix discount.

**Beam search.** Worse, because beams do not just fork once — they get reshuffled every step. A width-8 beam holds eight live states ($8 \times 49.41 = 395$ MiB per request) and every step's re-ranking permutes which beam descends from which, so each step is a gather-scatter across the whole 395 MiB. The KV lane handles the same permutation by rewriting eight block-table rows: eight pointer arrays, no data. Beam search on a recurrent model is a bandwidth problem that beam search on a transformer simply does not have.

**Copy-on-write branching in an agent loop.** Speculative *execution* at the request level — fork a conversation, try two tool calls, keep the better one — is the same operation with a longer time horizon, and it is exactly the workload the vLLM team characterized in their Mooncake Store post: agentic traces with a median of 33 turns and around 80,000 tokens of context by turn 30 ([vLLM, 2026-05-06](https://vllm.ai/blog/2026-05-06-mooncake-store)). Every fork of such a session costs $\sigma$ on a hybrid and approximately nothing on a transformer with prefix caching.

**Prefix caching itself.** This is the deepest version, and [the hybrid scheduling post](/blog/machine-learning/inference-engineering/batching-and-scheduling-hybrid-models) goes into it properly. Two requests sharing a 2000-token prefix produce byte-identical KV for those tokens, which is why block-hash sharing works at all. They also produce an identical *state* at token 2000 — but that state is a single value at one point, not a sequence of shareable pieces, so the only available mechanism is to snapshot the state at chosen boundaries and copy it wholesale into the new request. The hit-rate profile is completely different: exact-boundary matches only, and every snapshot costs a full $\sigma$ of resident memory to hold.

The through-line is that all four are the *same* mechanism as rollback, and all four are solved by the same insight. If what you cache is the layer's inputs rather than its state, forking is cheap again — two branches share the checkpoint state and the common prefix of the input buffer, and diverge only in the buffered suffix, which is 20.25 KiB per token per layer. Copy-on-write comes back. That is a bigger claim than anything in the public record supports today, so I flag it as an implication of the arithmetic rather than a reported result, but the arithmetic is the arithmetic.

---

## 8. Proving it works, and measuring it honestly

The failure mode of every strategy in this post is the same, and it is the worst kind: **the engine does not crash, it just produces slightly wrong text.** A state that is two tokens ahead of where it should be still has the right shape, still produces plausible logits, still passes every assertion you thought to write. What it produces is output that degrades — repetition, truncation, a drift into nonsense that gets worse the longer you generate.

This is not a hypothetical failure mode. It has been filed. [vLLM issue #39273](https://github.com/vllm-project/vllm/issues/39273) reports that n-gram speculative decoding produces corrupted, repetitive output on a hybrid gated-DeltaNet model when some speculative tokens are rejected, on an NVIDIA GH200. The issue's diagnosis is the exact sequence this post has been describing: the forward pass evolves the state for all proposed tokens, the rejection sampler accepts a subset, and the state-copy mechanism fails because the state was already advanced. Its key sentence is the thesis of this post stated as a bug report: *"There is no mechanism to rollback/revert the GDN state to the position after only the accepted tokens."* The reported symptom at `temperature=0` is output degenerating into repeated and truncated fragments. A companion issue, [#39809](https://github.com/vllm-project/vllm/issues/39809), reports a startup crash when Mamba prefix caching is combined with multi-token-prediction speculative decoding on a Nemotron-3 hybrid, and characterizes it as "a kernel-level incompatibility between the mamba prefix caching block indices and the speculative decode state management in the Triton SSM kernel". Both are cited, not reproduced by me.

So write the test first. The test is cheap and it is decisive, because there is a reference implementation available for free: the same model, decoding the same tokens, one at a time, with no speculation at all.

```python
# nanoserve/rollback_test.py
import torch

@torch.inference_mode()
def assert_rollback_is_exact(engine, prompt_ids, n_new=32, k=4, atol=0.0):
    """Speculative decode must be bit-identical to greedy decode.

    Runs the same prompt twice with a fixed seed: once through the plain decode
    loop, once through speculate_step(). Any divergence in the emitted tokens
    OR in the final recurrent state is a rollback bug.
    """
    engine.set_seed(1234)
    slot_a = engine.admit(prompt_ids)
    ref_ids = []
    for _ in range(n_new):
        ref_ids.append(engine.decode_one(slot_a))
    ref_ssm = engine.state_pool.ssm[:, slot_a].clone()
    ref_conv = engine.state_pool.conv[:, slot_a].clone()
    engine.release(slot_a)

    engine.set_seed(1234)
    slot_b = engine.admit(prompt_ids)
    spec_ids = []
    while len(spec_ids) < n_new:
        spec_ids.extend(engine.speculate_step(slot_b, k=k))
    spec_ids = spec_ids[:n_new]

    assert spec_ids == ref_ids, (
        f"token divergence at index "
        f"{next(i for i, (a, b) in enumerate(zip(spec_ids, ref_ids)) if a != b)}")

    # the harder assertion: the STATE must match, not just the tokens
    ssm_err = (engine.state_pool.ssm[:, slot_b] - ref_ssm).abs().max().item()
    conv_err = (engine.state_pool.conv[:, slot_b] - ref_conv).abs().max().item()
    print(f"tokens match: {len(spec_ids)}   "
          f"ssm max|err|={ssm_err:.3e}   conv max|err|={conv_err:.3e}")
    assert ssm_err <= atol and conv_err <= atol, "state diverged after rollback"
```

```console
tokens match: 32   ssm max|err|=0.000e+00   conv max|err|=0.000e+00
```

Three notes on why this test is shaped the way it is.

**Assert on the state, not only the tokens.** A rollback bug can leave the state wrong while the next few greedy tokens happen to come out the same, because argmax is robust to small perturbations. The token assertion catches the bug eventually; the state assertion catches it immediately, at the exact window where it happened. If you can only afford one, take the state one.

**`atol=0.0` is the right default for checkpoint and keep-k, and the wrong one for replay.** Restoring a saved copy is a bitwise operation, so the states must match exactly. Replaying inputs re-runs the recurrence, and a chunked scan over $j$ tokens does not necessarily accumulate in the same order as $j$ single steps, so bf16 rounding can differ in the last bits. That is not a bug, but you have to decide it is not a bug *before* you see it, or you will spend a day chasing it. Set a small `atol` for the replay path and document why. This is the same class of concern as [batch-invariance in sampling](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) — different reduction orders, different last bits.

**Seed both runs identically and use greedy decode.** With sampling on, the rejection sampler's random draws differ between the two paths by construction, and you will be comparing noise. Speculative decoding preserves the output *distribution*, not the output *sequence*, so a distributional test needs thousands of samples and a statistical criterion. Start with `temperature=0`, where the guarantee is exact equality and the test is a one-liner.

### How to measure the rollback tax without lying to yourself

The tax is a fraction of a decode step, and decode steps are short, so measuring it badly is easy. The protocol that works, and the reasons for each part:

1. **Warm up until the numbers stop moving.** At least 20 windows before you record anything: CUDA context, autotuner, allocator caching, and any `torch.compile` graph all settle in the first handful of steps.
2. **Time with CUDA events, not the wall clock.** `torch.cuda.Event(enable_timing=True)` around the region, with a `torch.cuda.synchronize()` before you read the elapsed time. A Python-side `time.perf_counter()` around an async launch measures the launch, not the work.
3. **Time the snapshot separately from the window.** Put events immediately around `state_pool.snapshot()` and around `state_pool.restore()`. You want three numbers per window — snapshot, verify, restore — because the whole argument of section 6 is about their ratio, and an aggregate tok/s number hides it completely.
4. **Sweep batch, not just $k$.** The entire thesis is that the strategies reorder as $B$ grows. Measuring at batch 1 and concluding that checkpointing is free is the single most likely way to get this wrong. Run $B \in \{1, 8, 32, 64, 128, 256\}$ and plot the tax as a fraction of the window.
5. **Report accepted tokens per window alongside everything.** The tax is conditional on rejection. A run whose acceptance rate happens to be high looks great and tells you nothing about the run where it is not.
6. **Lock clocks.** `nvidia-smi -lgc <freq>` before a comparison sweep, or the GPU's own boost behavior will manufacture a 10% effect out of thin air.

Here is the measurement harness, which is also the cost model — it prints predicted and measured side by side, so a mismatch is visible instead of implicit.

```python
# nanoserve/rollback_bench.py
import torch

HBM_BW = 3.35e12          # H100 SXM datasheet peak, bytes/s (cited)

def predict(spec, batch, ctx, k, alpha, weight_bytes, kv_bytes_per_tok):
    sigma = spec.sigma
    p_rej = 1.0 - alpha ** k
    base = weight_bytes + batch * kv_bytes_per_tok * ctx + 2 * batch * sigma
    ckpt = 2 * batch * sigma * (1.0 + p_rej)
    keep = k * batch * sigma
    Ib = 20736                                   # bytes/token/layer, derived
    Lr = len(spec.recurrent_layers)
    jbar = sum(alpha ** i for i in range(1, k + 1))
    replay = batch * Lr * Ib * ((k + 1) + jbar)
    return {
        "window_ms":  1e3 * base / HBM_BW,
        "ckpt_ms":    1e3 * ckpt / HBM_BW,
        "keep_ms":    1e3 * keep / HBM_BW,
        "replay_ms":  1e3 * replay / HBM_BW,
        "tok_per_win": 1.0 + jbar,
    }

def measure_snapshot(pool, slots, iters=50):
    """Isolate the snapshot copy. Events, sync, warmup, steady state."""
    for _ in range(20):
        pool.snapshot(slots)
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        pool.snapshot(slots)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters
    moved = pool.bytes_moved_per_snapshot(len(slots))
    print(f"snapshot {ms:7.3f} ms   {moved/2**20:8.1f} MiB   "
          f"{moved/(ms*1e-3)/1e9:6.1f} GB/s achieved")
    return ms

for B in (1, 8, 32, 64, 128, 256):
    p = predict(NEMOTRON_H_8B, B, 8192, 4, 0.70, 16.0e9, 16384)
    tax = 100 * p["ckpt_ms"] / (p["window_ms"] + p["ckpt_ms"])
    print(f"B={B:4d}  window {p['window_ms']:6.2f} ms   "
          f"ckpt {p['ckpt_ms']:6.2f} ms ({tax:4.1f}%)   "
          f"replay {p['replay_ms']:5.2f} ms")
```

```console
B=   1  window   4.82 ms   ckpt   0.05 ms ( 1.1%)   replay  0.00 ms
B=   8  window   5.55 ms   ckpt   0.44 ms ( 7.3%)   replay  0.01 ms
B=  32  window   7.04 ms   ckpt   1.74 ms (19.8%)   replay  0.04 ms
B=  64  window   9.15 ms   ckpt   3.49 ms (27.6%)   replay  0.08 ms
B= 128  window  13.38 ms   ckpt   6.97 ms (34.3%)   replay  0.15 ms
B= 256  window  21.83 ms   ckpt  13.95 ms (39.0%)   replay  0.30 ms
```

Those are **predictions from the derivation, not measurements** — the script prints what the formula says, and `measure_snapshot()` is what you run to check it. The expected result on real hardware: the achieved bandwidth on the snapshot copy should land somewhere in the 1.8–2.6 TB/s range on an H100 for a copy of this size and shape, so the measured milliseconds should come out 30–80% higher than the predicted ones, uniformly across batch. If they do not scale linearly with $B$, your snapshot is doing a gather-scatter through an intermediate instead of a straight copy — go back to section 3 and fix the indexing. Run it and report yours.

The shape of that table is the whole argument, and it survives whatever bandwidth you actually achieve, because both the window and the tax scale with the same $\text{BW}$ in the denominator. At batch 1 the checkpoint is 1% and you should not think about it. At batch 256 it is 39% and it is the second-largest line item in your decode step.

---

## 9. Case studies and the public record

Four public data points, each cited with its setup and its caveat. Note how thin this section is compared to the equivalent section in a post about, say, paged attention — that thinness is itself the finding.

**vLLM's hybrid SSM disaggregation work names this as an open edge.** The vLLM team's [Disaggregated Serving for Hybrid SSM Models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) is the most complete public account of running hybrids in a production engine. It describes dual descriptor views over the same physical memory so that attention blocks and SSM blocks can be addressed by different index formats without reshuffling, a conv state decomposed into three sub-projections with the SSM state in a `(dim, state_len)` layout so each decode rank reads only its own tensor-parallel slice, and a `Nemotron-3-Super-120B-A12B-FP8` deployment on 8×H200 with prefill TP4 and decode TP4 that they report "Pareto-dominates the co-located baseline at higher batch sizes". And then it lists what does not work yet: Mamba1 unsupported, gated DeltaNet pending, mixed-block-size hybrid allocation unsupported, and **the interaction with speculative decoding "not extensively validated"**. The vLLM team explicitly flags spec-dec-on-hybrid as not extensively validated — in a post whose entire purpose is to demonstrate that hybrid serving works. That is the most honest sentence in the Track K literature and it is the reason this post exists.

**The bug reports are the ground truth.** [Issue #39273](https://github.com/vllm-project/vllm/issues/39273) (corrupted output from n-gram speculative decoding on a hybrid GDN model, "no mechanism to rollback/revert the GDN state") and [issue #39809](https://github.com/vllm-project/vllm/issues/39809) (startup crash combining Mamba prefix caching with MTP speculative decoding on Nemotron-3, described as a kernel-level incompatibility in the Triton SSM kernel) are what "not extensively validated" looks like from the user side. Read them as a specification: the first tells you the semantic requirement, the second tells you that the requirement interacts with prefix caching in a way that the kernel signature does not currently express. Both are cited from the trackers; I have reproduced neither.

**SpecMamba is the academic statement of the same problem.** [arXiv 2509.19873](https://arxiv.org/abs/2509.19873) (Zhong, Xu, Wen, Xie, Guo, Wang, Li — ICCAD 2025) opens by naming three obstacles to speculative decoding on SSMs — hidden state backtracking difficulties, tree-based parallel verification incompatibility, and hardware workload mismatch — and answers with a "memory-aware hybrid backtracking strategy", a FIFO-based tree verification with tiling, and a co-designed dataflow. Reported results: 2.27× over GPU baselines and 2.85× over prior FPGA solutions on AMD VHK158 and VCK190 platforms, with 5.41× and 1.26× better energy efficiency respectively. The caveat matters and it is large: this is an **FPGA accelerator**, so the speedups are not transferable to your H100 and the memory hierarchy it optimizes against is not yours. What transfers is the taxonomy — that backtracking and tree verification are two separate problems, and that the authors also landed on a *hybrid* of strategies rather than one.

**ReplaySSM is the strongest public claim that the problem has a good answer.** Tri Dao's [ReplaySSM post](https://tridao.me/blog/2026/replayssm/) (2026-06-15) and the associated [vLLM RFC #47572](https://github.com/vllm-project/vllm/issues/47572) propose caching SSM inputs rather than states, with a ring buffer and periodic flush, giving "O(1) rollback via pointer move" against a baseline that "stores a full state snapshot for every draft position". Reported: 1.87–1.96× for speculative decoding over standard decoding and up to 2.14× over the vLLM baseline, 1.43–1.84× kernel speedups, and 3.0–3.3× more concurrent requests at a fixed memory budget. The caveat: these are the author's reported figures for his own method, published on a personal blog and an open RFC rather than in a peer-reviewed venue, and at the time of writing the RFC is a proposal rather than merged behavior. Cite it as a strong signal about direction, not as a property of your engine today.

Put the four together and the picture is coherent and unusual for this series: the mechanism is understood, the arithmetic is settled, the production engines have shipped the *hard* parts (two-cache allocation, disaggregated state transfer) and left this specific interaction as a known gap, and the best-looking fix is recent enough that it has not landed. That is what "open problem" means in inference engineering. Not that nobody knows what to do — that nobody has finished doing it.

---

## 10. When to reach for this, and when not to

![A decision tree that routes a rollback strategy choice based on whether you control the scan kernel and whether your drafts form a chain or a tree](/imgs/blogs/speculative-decoding-and-rollback-with-recurrent-state-7.webp)

The decision tree above is short because the decision is genuinely determined by two questions, and neither of them is about your workload.

**If you are serving a hybrid model on a production engine today, the first honest answer is: check whether speculative decoding is supported at all, and believe the release notes.** The public record above says the combination is a known gap in vLLM as of the April 2026 write-up, with open corruption bugs against it. This is not a case where you should reach for the flag and see what happens, because the failure mode is silent output degradation rather than an exception. Turn it on, run the temperature-zero equality test from section 8 against the same model without speculation, and only then look at throughput. If the tokens do not match exactly, you do not have a performance question, you have a correctness bug.

**If you are building the engine and you can modify the scan kernel, build replay-inputs and skip the other three.** The arithmetic in section 6 is not close: 2.4 MiB against 49.4 MiB per request, a rollback that moves no data at all, and a factor of roughly 47 on the traffic term at every batch size and every draft length you would plausibly use. The cost is that your scan kernel needs to accept a checkpoint state plus a variable-length input buffer, which is a real kernel change, not a wrapper. Budget for it as kernel work.

**If you cannot modify the kernel, checkpoint — and cap your batch.** The checkpoint path is twenty lines, has one failure mode, and costs 1% of a window at batch 1 and 39% at batch 256. That is a perfectly reasonable trade at the low end. Derive your own $B^{*}$ from section 6, and treat it as an admission-control limit rather than a performance curiosity: past that batch, turn speculation off, because you are paying more in rollback than the draft model is earning.

**Do not build keep-k unless you are doing tree drafting.** It is worse than checkpoint on both axes for chain drafting — more memory *and* more traffic — and its one advantage, random access to any position in the window, is only worth something when your drafts branch.

**And the case for not speculating at all on a hybrid is stronger than on a transformer.** The reason is worth stating plainly, because it runs against the intuition the rest of the series builds. Speculative decoding earns its keep by amortizing the weight read across several tokens. On a hybrid at high batch, section 6 showed that the weight read is *not* the dominant term — past batch 154 for Nemotron-H-8B, the state traffic is. Speculation amortizes the state traffic too (one chunked scan per window, not per token), so the win survives, but the rollback machinery attacks the same term it just saved. If your service runs at high concurrency and your acceptance rate is mediocre, the honest expectation is a modest win that a batch-size increase would have given you for free, with a new class of silent-corruption bug attached. Measure before you commit.

Where speculation on a hybrid clearly *is* worth it: low-concurrency, latency-sensitive deployments — a single-user coding assistant, an interactive agent, anything where you are at batch 8 and the GPU is mostly idle. There the checkpoint tax is under 8%, the acceptance rate on code and structured output is high, and the per-token latency win is exactly what the user feels. That is the same conclusion the vLLM team reached for transformers back in [their 2024 speculative decoding post](https://vllm.ai/blog/2024-10-17-spec-decode), which reports up to 1.5× with a draft model and up to 2.8× with prompt-lookup n-grams at low QPS, and a *slowdown* of 1.4× and 1.8× respectively at high QPS. Hybrids do not change that shape. They steepen it.

---

## 11. Key takeaways

1. **Rollback is free on attention and expensive on recurrence, and the reason is structural.** KV writes are positionally addressed and independent, so the cache after $j$ tokens is a prefix of the cache after $k$. A recurrent state has no per-token dimension, its update destroys its input, and inverting it in bf16 amplifies rounding without bound.
2. **Price $\sigma$ before you design anything.** For Nemotron-H-8B it is 49.41 MiB per request — 24 layers of 2.06 MiB — derived from the published config, and every strategy's cost is a multiple of it.
3. **The ratio that decides the design is state bytes over input bytes.** For Mamba-2 at these shapes it is about 104. Caching what advances the state is two orders of magnitude cheaper than caching the state.
4. **Checkpoint cost is independent of $k$; keep-k cost is linear in $k$; re-scan cost is independent of $B$.** Those three facts are why the strategies reorder as you change batch and draft length, and why there is no universal winner.
5. **Re-scan beats checkpoint above $B^{*} = f_r W p_{\text{rej}} / (2\sigma(1+p_{\text{rej}}))$**, which is about batch 22 for Nemotron-H-8B at $\alpha = 0.7$ and $k = 4$. Derive yours; do not borrow mine.
6. **Keep-4 states cuts your concurrency from 320 to 151 at 8k context on an H100.** The surcharge for the ability to undo is larger than the entire KV cache of the request it protects.
7. **Tree drafting has exactly one rollback strategy and it is the expensive one.** Attention handles a draft tree with a mask; a recurrence needs one materialized state per branch.
8. **The same asymmetry is why `n>1`, beam search and copy-on-write forking all invert their economics on a hybrid.** A KV cache forks by reference count; a state forks by full copy.
9. **The failure mode is silent.** Assert bitwise equality against unspeculated greedy decode on both the tokens *and* the state, and do it before you look at a single throughput number.
10. **Believe the release notes.** The vLLM team flags spec-dec on hybrids as not extensively validated, and there are open corruption issues against it. This is the one place in the series where "just use vLLM" is not yet the safe answer, and where reading the arithmetic yourself has direct operational value.

---

## Further reading

- [Disaggregated Serving for Hybrid SSM Models](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) — vLLM, 2026-04-21. The dual-descriptor design, the DS conv-state layout, and the limitations list that this post is built around.
- [ReplaySSM: Cache SSM Inputs, Not State](https://tridao.me/blog/2026/replayssm/) — Tri Dao, 2026-06-15, with [vLLM RFC #47572](https://github.com/vllm-project/vllm/issues/47572). The ring-buffer formulation of strategy four and its reported speculative-decoding numbers.
- [SpecMamba: Accelerating Mamba Inference on FPGA with Speculative Decoding](https://arxiv.org/abs/2509.19873) — Zhong et al., ICCAD 2025. The academic taxonomy of the three obstacles, including tree-verification incompatibility.
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) — Dao and Gu, 2024. The Mamba-2 recurrence and the chunked-scan formulation the whole rollback question hangs off.
- [Nemotron-H-8B-Base-8K config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) and the [Nemotron-H technical report](https://arxiv.org/abs/2504.03624) — every byte count in this post traces back here.
- [vLLM issue #39273](https://github.com/vllm-project/vllm/issues/39273) and [#39809](https://github.com/vllm-project/vllm/issues/39809) — what the rollback gap looks like as a bug report.
- [Hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) — the Track K opener: where $\sigma$ comes from, and the five other things a hybrid breaks.
- [Batching and scheduling hybrid models](/blog/machine-learning/inference-engineering/batching-and-scheduling-hybrid-models) — admission control, preemption and prefix caching for two-cache engines.
- [Prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) — the forking mechanism that section 7 shows does not transfer.
- [Speculative decoding: draft fast, verify in parallel](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) — the acceptance-rate math and the rejection-sampling proof this post assumes.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — where this piece sits in the series, and the capstone that ties the scoreboard together.


