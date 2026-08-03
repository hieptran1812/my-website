---
title: "Batching and scheduling hybrid models: what a fixed-size state does to your scheduler"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Rebuild the admission, preemption and prefix-cache logic of your engine for a model whose memory is a line with an intercept, and find the one characteristic length that decides every scheduling question a hybrid asks."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "batching",
    "scheduling",
    "hybrid-models",
    "state-space-models",
    "kv-cache",
    "mamba",
    "vllm",
    "ml-systems",
    "gpu",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
---

Here is the bug report you are going to get. A hybrid model is deployed, traffic is short prompts, the dashboard says the KV cache is 4.6% utilized, and the engine is returning "capacity exceeded" to one caller in seven. Ninety-five percent of the block pool is free and the scheduler will not take the request. Nobody's runbook covers this, because in every engine you have ever operated, "free blocks" was the answer to "can I take another request", and here it is not even the right question.

The cause is a single structural fact from [the previous post in this track](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption): a hybrid model's per-request memory is a line **with an intercept**. Some layers keep a KV cache that grows one entry per token; the rest keep a recurrent state whose size is fixed at model-definition time and does not move between token 1 and token 128,000. Your allocator now has two populations to feed, and they exhaust independently. The scheduler you wrote in Track C — [the continuous-batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) and [the policy layer on top of it](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) — has exactly one budget in it, and one budget cannot express two exhaustion conditions.

![Two columns comparing a dense engine with a single block budget against a hybrid engine that must satisfy a block budget and a slot budget at the same time](/imgs/blogs/batching-and-scheduling-hybrid-models-1.webp)

The figure above is the shape of the change. On the left, the dense engine: one budget, and concurrency starts falling as soon as contexts pass about 1,800 tokens, ending at three concurrent requests at 128k. On the right, the same GPU running a hybrid: two budgets, concurrency pinned flat at your chosen slot count all the way out to about 11,000 tokens, then falling far more slowly to 27 at 128k. Both sets of numbers are derived below from published configs and a datasheet, and the interesting part is not the 9× at the right-hand end. It is the *flat region* — the range of context lengths over which a hybrid engine's capacity is a number you chose rather than a number your traffic chose for you.

By the end of this post you will have rewritten four parts of the scheduler. **Admission** becomes a two-term budget, and you will be able to derive, for any hybrid config and any GPU, the exact context length at which the binding constraint flips from slots to blocks. **Prefix caching** mostly stops paying, and you will be able to say precisely how much it loses and at what prefix length it starts winning again. **Preemption** becomes a fixed-size copy instead of a recompute-or-swap decision, which makes it roughly 300× cheaper and changes what policy you should run. And **prefill/decode disaggregation** starts handing over a state tensor rather than a block list, which is where the current engineering frontier is. The code lands in a new file, `nanoserve/hybrid_sched.py`, plus a small diff on `nanoserve/policy.py`.

Two promises inherited from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and they bind hard here because this post is mostly arithmetic. **I have no GPU and have run nothing.** Every number below is derived from arithmetic I show you, cited from a vendor post or datasheet with a link, or framed as something you should reproduce with an expected range; results tables carry a `Source` column. **And every fact about a production engine is attributed and linked**, because several of the mechanisms in section 7 are newer than my training data and I will not assert them from memory.

---

## 1. The line in your admission check that stops being true

Pull up the FCFS reference policy from the Track C post. The whole admission decision is five lines:

```python
# nanoserve/policy.py — the dense version, for reference
def select_admits(self, st: SchedulerState) -> list[Admission]:
    admits: list[Admission] = []
    free, budget = st.free_blocks, st.token_budget
    slots = self.max_running - len(st.running)
    for r in self._order(st):
        if slots <= 0 or budget <= 0:
            break
        take = min(budget, r.prompt_len - r.prefilled)
        need = r.blocks_for(take)
        if take <= 0 or free - need < self.watermark:
            break
        admits.append(Admission(r, take, need))
        free -= need
        budget -= take
        slots -= 1
    return admits
```

Three resources are being rationed: `free` (KV blocks), `budget` (tokens of compute this step), and `slots` (a cap on the running-set size, there to bound Python overhead and per-step launch count, not because the hardware runs out of anything at 256). Only the first is a real memory constraint, and it is one scalar because the cache is one thing.

The `max_running` cap deserves a moment, because it is about to change meaning entirely. In a dense engine it is a *soft* knob. You set it to 256 because the scheduler's Python loop starts costing you at a few hundred sequences, or because the attention kernel's batch dimension gets awkward, or because you want a ceiling on the tail latency of a step. Raise it and nothing breaks; you just spend more CPU per step. It is not tied to a physical allocation.

In a hybrid engine, `max_running` becomes a **hard memory constraint**. Every running request holds a recurrent state, that state is a real allocation of real bytes, and the number of those allocations you can hold is fixed by how much memory you fenced off for them. You cannot raise the cap without taking bytes from the KV pool. You cannot lower it without stranding bytes you already reserved. The soft knob became a hard partition, and the failure mode in the opening paragraph is what it looks like when a soft knob quietly turns hard: a scheduler that refuses work while its main memory pool is nearly empty.

Everything else in this post follows from working out the arithmetic of that partition honestly.

Let me fix notation, all of it from the previous post's derivations so you can check it against a config file rather than against me.

- $B_{\text{tok}}$ — bytes of KV per token, summed over the full-attention layers only. For **Nemotron-H-8B**, four attention layers with eight key/value heads of dimension 128 in bf16 give $2 \times 4 \times 8 \times 128 \times 2 = 16{,}384$ bytes, so 16 KiB per token.
- $\Sigma$ — total fixed state bytes per request, summed over the recurrent layers. For Nemotron-H-8B, 24 Mamba-2 layers at 2,158,592 bytes each give $\Sigma = 51{,}806{,}208$ bytes, so 49.4 MiB.
- $F$ — free device bytes after weights, activations and workspace. On an H100 80GB SXM (NVIDIA's [H100 datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-datasheet) lists 80 GB of HBM3 at 3.35 TB/s, which is 74.5 GiB) minus about 14.9 GiB of bf16 weights for an 8B model and 4 GiB of headroom: $F \approx 55.6$ GiB $= 59{,}700{,}045{,}414$ bytes.
- $N$ — the number of state slots you reserved at startup. This is the old `max_running`, now with teeth.

The comparison model throughout is **Llama-3.1-8B**, whose 32 attention layers with the same head configuration give $B_{\text{tok}} = 131{,}072$ bytes, so 128 KiB per token and $\Sigma = 0$.

---

## 2. The state-equivalent length

Write the per-request footprint of a hybrid at context $S$:

$$M(S) \;=\; \Sigma \;+\; B_{\text{tok}} \cdot S$$

Now factor out the slope. That single algebraic move is the most useful thing in this post:

$$M(S) \;=\; B_{\text{tok}}\,\bigl(S \;+\; S^{\dagger}\bigr), \qquad S^{\dagger} \;\equiv\; \frac{\Sigma}{B_{\text{tok}}}$$

$S^{\dagger}$ has units of tokens. It is **the number of tokens of KV cache whose bytes equal the fixed state** — call it the *state-equivalent length*. And once you have it, the hybrid stops being a new kind of object to your scheduler. A hybrid request at context $S$ occupies exactly as much memory as a dense request at context $S + S^{\dagger}$ would, at the hybrid's own per-token rate. The state is a **phantom prefix**: a fixed number of tokens that every request carries from the moment it is admitted, that never grows and never shrinks, and that costs the same as real context.

#### Worked example: the state-equivalent length of Nemotron-H-8B

$$S^{\dagger} \;=\; \frac{51{,}806{,}208}{16{,}384} \;=\; 3{,}162 \text{ tokens}$$

Exactly 3,162, with no remainder, because both quantities are powers of two times small integers. Source: derived from the [Nemotron-H-8B config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json).

Read that as an operational statement, because it is one: **every Nemotron-H-8B request behaves, for scheduling purposes, as if its prompt were 3,162 tokens longer than it is.** A 200-token classification request costs what a 3,362-token request would. A 100,000-token document request costs what a 103,162-token one would, which is to say the phantom prefix has become a rounding error. The whole engineering story of hybrid scheduling is the relationship between your actual context lengths and this one number.

Three consequences fall out immediately, and each one is a section of this post.

**Consequence one: admission is a two-term budget.** With $N$ state slots reserved, the state pool consumes $N\Sigma$ bytes and the KV pool gets what is left. Concurrency at uniform context $S$ is the smaller of the two limits:

$$n(S) \;=\; \min\!\Bigl(N,\; \Bigl\lfloor \frac{F - N\Sigma}{B_{\text{tok}}\,S} \Bigr\rfloor \Bigr)$$

**Consequence two: below a knee, concurrency is constant in $S$.** Set the two terms equal and solve for the context length at which the binding constraint flips:

$$S_{\text{knee}} \;=\; \frac{F}{N\,B_{\text{tok}}} \;-\; S^{\dagger}$$

**Consequence three: the footprint's variability is damped.** More on that in a moment, once the numbers are on the table.

![A layered budget showing free VRAM split into a fixed state pool and the remaining KV pool, with the phantom prefix and the resulting concurrency knee](/imgs/blogs/batching-and-scheduling-hybrid-models-2.webp)

The stack above is the partition for a concrete choice. Reserve $N = 256$ slots. The state pool takes $256 \times 51{,}806{,}208 = 13{,}262{,}389{,}248$ bytes, which is 12.35 GiB — nearly a quarter of everything you had. The KV pool gets the remaining 46,437,656,166 bytes, or 43.2 GiB, which at 16 tokens and 262,144 bytes per block is 177,145 blocks. Expressed in phantom-prefix terms, each request charges $\lceil 3162/16 \rceil = 198$ phantom blocks plus $\lceil S/16 \rceil$ real ones.

#### Worked example: where the knee falls, hybrid versus dense

For the hybrid with $N = 256$:

$$S_{\text{knee}} \;=\; \frac{59{,}700{,}045{,}414}{256 \times 16{,}384} \;-\; 3{,}162 \;=\; 14{,}233.7 - 3{,}162 \;=\; 11{,}071 \text{ tokens}$$

For Llama-3.1-8B with the same $F$ and the same running cap of 256 (and no state pool, so no phantom prefix):

$$S_{\text{knee}} \;=\; \frac{59{,}700{,}045{,}414}{256 \times 131{,}072} \;=\; 1{,}779 \text{ tokens}$$

Source: derived, using the datasheet's 80 GB for the memory figure. The flat region is **6.2× wider** for the hybrid. Below 1,779 tokens both engines run at their configured 256; between 1,779 and 11,071 the dense engine's concurrency is decaying while the hybrid's is not; above 11,071 both decay, but the hybrid decays from a slope eight times gentler.

Here is the full capacity comparison, which is the table you actually want pinned to the wall:

| Context $S$ | Llama-3.1-8B | Nemotron-H-8B (unified pool) | Nemotron-H-8B ($N=256$) | Source |
| --- | --- | --- | --- | --- |
| 512 | 444 (capped 256) | 991 (capped 256) | 256 | derived |
| 1,024 | 444 (capped 256) | 870 (capped 256) | 256 | derived |
| 4,096 | 111 | 502 (capped 256) | 256 | derived |
| 8,192 | 55 | 320 (capped 256) | 256 | derived |
| 32,768 | 13 | 101 | 101 | derived |
| 131,072 | 3 | 27 | **21** | derived |

Two things in that table are worth stopping on.

**The last row is the cost of a bad partition.** At 128k context the unified-pool arithmetic gives 27 concurrent requests, matching the previous post's figure exactly. The $N=256$ partition gives 21 — because 12.35 GiB is fenced off for 256 state slots when only 21 requests can possibly fit, so 235 slots and 11.3 GiB sit unusable. That is a 22% concurrency loss caused by nothing but a startup flag. Set $N$ too high and you starve the KV pool; set it too low and you cap yourself below what memory would allow. There is no value of $N$ that is right at both 512 tokens and 128k tokens, which is the central annoyance of hybrid capacity planning and the reason section 6 builds a growable pool instead of a fixed one.

**The middle rows are where hybrids pay off, and it is not where the marketing points.** Between 4k and 32k the hybrid holds 4.5× to 7.8× the concurrency of the dense model. That is the RAG and agentic band — not the 1M-token demo.

#### The variance argument, made precise

The flat region buys something a capacity table does not show: **predictability**. Take a traffic mix with mean context $\bar S$ and standard deviation $\sigma_S$. The per-request footprint is a linear function of $S$, so its coefficient of variation is:

$$\mathrm{CV}\bigl(M\bigr) \;=\; \frac{B_{\text{tok}}\,\sigma_S}{B_{\text{tok}}\,(\bar S + S^{\dagger})} \;=\; \frac{\sigma_S}{\bar S + S^{\dagger}}$$

against a dense engine's $\sigma_S / \bar S$. The damping factor is $\bar S / (\bar S + S^{\dagger})$, which is a clean, checkable statement about how much less your memory demand swings:

| Mean context $\bar S$ | Damping factor | Interpretation | Source |
| --- | --- | --- | --- |
| 512 | 0.14 | footprint varies 7.2× less than dense | derived |
| 1,024 | 0.24 | 4.1× less | derived |
| 4,096 | 0.56 | 1.8× less | derived |
| 32,768 | 0.91 | 1.10× less, essentially undamped | derived |

Be honest about what this says. At short and medium contexts the constant term genuinely stabilizes the scheduler: a burst of long requests moves your memory demand far less than it would on a dense model, so you can run closer to the memory ceiling without a watermark scare. At long contexts the damping vanishes, because the phantom prefix has become negligible next to the real one. The predictability benefit and the memory benefit therefore live in *different regimes* — the stability is a short-context property and the capacity win is a long-context property. Any sentence claiming a hybrid is uniformly better at scheduling is eliding that.

---

## 3. Writing `can_admit()`

Now build it. The design goal is that when admission refuses a request, the scheduler knows **which** resource refused it, because the two failure modes have opposite remedies: block exhaustion means preempt somebody, slot exhaustion means preempt somebody *or* repartition, and slot exhaustion with a nearly empty block pool means you configured the server wrong.

![A request splitting into a block claim and a slot claim which merge into a single admission verdict that rejects on the exhausted pool](/imgs/blogs/batching-and-scheduling-hybrid-models-3.webp)

The figure traces the case from the opening paragraph. One request fans out into two independent claims; each claim queries a different pool; the verdict is the weaker of the two answers, not a blend of them; and the rejection reason names the pool that said no. Encode that shape directly.

```python
# nanoserve/hybrid_sched.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

BLOCK_SIZE = 16  # tokens per KV block, matching nanoserve/blocks.py


class Refusal(str, Enum):
    OK = "ok"
    NO_BLOCKS = "no_blocks"          # KV pool exhausted or below watermark
    NO_STATE_SLOT = "no_state_slot"  # recurrent-state pool exhausted
    NO_TOKEN_BUDGET = "no_token_budget"


@dataclass(frozen=True)
class HybridShape:
    """The two memory terms, straight from the model config."""
    kv_bytes_per_token: int      # summed over full-attention layers only
    state_bytes: int             # summed over recurrent layers

    @property
    def state_equivalent_tokens(self) -> int:
        """S-dagger: tokens of KV whose bytes equal the fixed state."""
        return self.state_bytes // self.kv_bytes_per_token

    @property
    def phantom_blocks(self) -> int:
        """The fixed state expressed in your existing block currency."""
        return math.ceil(self.state_equivalent_tokens / BLOCK_SIZE)

    def bytes_at(self, seq_len: int) -> int:
        return self.state_bytes + self.kv_bytes_per_token * seq_len


NEMOTRON_H_8B = HybridShape(kv_bytes_per_token=16_384, state_bytes=51_806_208)
LLAMA_31_8B = HybridShape(kv_bytes_per_token=131_072, state_bytes=0)
```

`state_equivalent_tokens` is the whole post in one property. Print it once at startup and put it in your logs, because every capacity question you will ever ask about this deployment is answered relative to it.

Next, the pool that did not exist before. A state slot is not a block: it is not subdividable, not shareable, not partially allocatable. It is a fixed-size box with a request in it or nothing.

```python
class StateSlotPool:
    """Fixed-size recurrent-state slots. One per in-flight request.

    Slots are opaque: no offsets, no partial allocation, no sharing.
    The only operations are take one, give one back.
    """

    def __init__(self, num_slots: int, bytes_per_slot: int):
        self.bytes_per_slot = bytes_per_slot
        self._free: list[int] = list(range(num_slots))
        self._owner: dict[int, str] = {}   # slot index -> request id
        self.num_slots = num_slots

    @property
    def free_slots(self) -> int:
        return len(self._free)

    @property
    def reserved_bytes(self) -> int:
        return self.num_slots * self.bytes_per_slot

    def take(self, rid: str) -> int | None:
        if not self._free:
            return None
        slot = self._free.pop()
        self._owner[slot] = rid
        return slot

    def give_back(self, slot: int) -> None:
        self._owner.pop(slot, None)
        self._free.append(slot)
```

Then the two-term check itself. Note that it returns a `Refusal`, not a bool, and that the block claim charges the phantom prefix only in unified-pool mode — in split-pool mode the state bytes were already fenced off, so charging them again would double-count.

```python
@dataclass
class HybridPools:
    shape: HybridShape
    free_blocks: int
    state: StateSlotPool | None      # None => unified byte-budget mode
    watermark: int = 256             # blocks held back for decode growth

    def blocks_needed(self, req, new_tokens: int) -> int:
        """Blocks to reserve for `new_tokens` more tokens of this request."""
        held = math.ceil(req.ctx / BLOCK_SIZE)
        after = math.ceil((req.ctx + new_tokens) / BLOCK_SIZE)
        extra = after - held
        if self.state is None and req.ctx == 0:
            # Unified mode: a brand-new request also pays the phantom prefix.
            extra += self.shape.phantom_blocks
        return extra

    def can_admit(self, req, new_tokens: int, token_budget: int) -> tuple[Refusal, int]:
        """Return (verdict, blocks_to_reserve). Never mutates anything."""
        if new_tokens <= 0 or token_budget < new_tokens:
            return Refusal.NO_TOKEN_BUDGET, 0

        need = self.blocks_needed(req, new_tokens)
        if self.free_blocks - need < self.watermark:
            return Refusal.NO_BLOCKS, need

        # Only a request that is not yet running consumes a state slot.
        if self.state is not None and req.ctx == 0 and self.state.free_slots == 0:
            return Refusal.NO_STATE_SLOT, need

        return Refusal.OK, need
```

Two details in there are the difference between code that works and code that looks like it works.

**The slot is claimed at first admission, not at every step.** A request that is mid-prefill across several chunked steps already holds its slot; charging it again on step two would deadlock the request against itself. The `req.ctx == 0` guard is what distinguishes "this request is entering the running set" from "this request is continuing".

**The check is pure.** It reserves nothing and returns what it *would* reserve. That matters because the policy layer loops over candidates and needs to speculatively decrement a local copy of the budget, exactly as the dense `select_admits` does with its local `free` variable. A `can_admit` that mutates cannot be used inside such a loop without a rollback path, and rollback paths in admission code are where the double-free bugs live.

Now the policy override. Because the Track C policy layer put the admission decision behind a small interface, the hybrid version is a subclass that changes one method:

```python
from nanoserve.policy import FCFS, Admission, SchedulerState


class HybridFCFS(FCFS):
    """FCFS with a two-term budget and per-reason refusal accounting."""
    name = "hybrid-fcfs"

    def __init__(self, pools: HybridPools, **kw):
        super().__init__(**kw)
        self.pools = pools
        self.refusals: dict[Refusal, int] = {r: 0 for r in Refusal}

    def select_admits(self, st: SchedulerState) -> list[Admission]:
        admits: list[Admission] = []
        # Local, speculative copies. Nothing is committed until the engine acts.
        free_blocks = st.free_blocks
        free_slots = self.pools.state.free_slots if self.pools.state else 1 << 30
        budget = st.token_budget

        for r in self._order(st):
            take = min(budget, r.prompt_len - r.prefilled)
            probe = HybridPools(
                shape=self.pools.shape,
                free_blocks=free_blocks,
                state=_FakeSlots(free_slots) if self.pools.state else None,
                watermark=self.watermark,
            )
            verdict, need = probe.can_admit(r, take, budget)
            if verdict is not Refusal.OK:
                self.refusals[verdict] += 1
                if verdict is Refusal.NO_TOKEN_BUDGET:
                    break          # nothing later will fit either
                break              # head-of-line: see the Track C post on skip-ahead
            admits.append(Admission(r, take, need))
            free_blocks -= need
            budget -= take
            if r.ctx == 0:
                free_slots -= 1
        return admits


@dataclass
class _FakeSlots:
    """A speculative view of the slot pool for the admission loop."""
    free_slots: int
```

The `refusals` counter is not decoration. It is the single most valuable metric a hybrid engine can emit, and section 8 explains why: the *ratio* between `NO_BLOCKS` and `NO_STATE_SLOT` tells you, with no profiling, which side of the knee your traffic is actually on and whether your partition is wrong.

Run the reference case — the burst from the opening paragraph — and watch the two pools disagree:

```python
def burst_demo(num_requests=300, prompt_len=512, num_slots=256):
    shape = NEMOTRON_H_8B
    free_bytes = int(55.6 * 1024 ** 3)
    block_bytes = BLOCK_SIZE * shape.kv_bytes_per_token

    state = StateSlotPool(num_slots, shape.state_bytes)
    kv_bytes = free_bytes - state.reserved_bytes
    total_blocks = kv_bytes // block_bytes

    per_req_blocks = math.ceil(prompt_len / BLOCK_SIZE)
    admitted = min(num_requests, num_slots, total_blocks // per_req_blocks)
    used = admitted * per_req_blocks

    print(f"state slots : {num_slots:>7} total  {admitted:>7} used  "
          f"{num_slots - admitted:>7} free")
    print(f"kv blocks   : {total_blocks:>7} total  {used:>7} used  "
          f"{100 * used / total_blocks:.1f}% used")
    print(f"admitted    : {admitted} / {num_requests}")
    print(f"rejected    : {num_requests - admitted}  (reason: no_state_slot)")


burst_demo()
```

```console
state slots :     256 total      256 used        0 free
kv blocks   :  177145 total     8192 used      4.6% used
admitted    : 256 / 300
rejected    : 44  (reason: no_state_slot)
```

There is the bug report, reproduced in pure arithmetic with no GPU. Forty-four requests rejected with 95.4% of the block pool free. The pool that ran out is invisible to every dashboard built for a dense engine, and the request that was refused would have fit ten times over in the memory you can see.

---

## 4. Prefix caching mostly stops paying

Now the part that costs you something real.

[The prefix-sharing post](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) built a hash-indexed cache over the block allocator: hash each complete block of tokens, look up the longest chain of matching hashes, and point the new request's block table at the physical blocks that already exist. It works because a KV cache has a property that is easy to take for granted — **it is a map, not a fold.**

Precisely: for a full-attention layer, the cache entry at position $t$ is a function of the hidden state at position $t$ alone,

$$k_t = W_K h_t, \qquad v_t = W_V h_t$$

and $h_t$ in turn depends on the prefix up to $t$. So the *array* of entries for a prefix is determined by the prefix, and — this is the part that matters — **any contiguous run of that array is independently addressable, hashable and shareable.** Two requests with a common prefix produce byte-identical entries for exactly those positions, and you can share the first 1,024 of them without caring about the rest.

A recurrent layer's state is a fold over the whole prefix:

$$h^{\text{state}}_T \;=\; \Bigl(\textstyle\prod_{t=1}^{T} A_t\Bigr) h^{\text{state}}_0 \;+\; \sum_{t=1}^{T}\Bigl(\textstyle\prod_{j=t+1}^{T} A_j\Bigr) B_t x_t$$

One tensor. Not an array of $T$ things — one thing, produced by consuming all $T$ inputs in order. There is no "the first 1,024 tokens' worth" of it to share, because the object has no positional axis to slice along. You can share $h^{\text{state}}_T$ with another request whose prefix is *exactly* $T$ tokens long and identical; you can do nothing at all with a request whose prefix matches for 900 tokens and then diverges.

![A four by two comparison of prefix reuse, preemption, admission and handoff under a dense engine against a hybrid engine](/imgs/blogs/batching-and-scheduling-hybrid-models-4.webp)

The matrix above is the scorecard for the whole post, and prefix reuse is the row where the hybrid loses. The other three rows are wins, which is worth saying plainly before the bad news: two of the four mechanisms get *cheaper and more predictable* under a fixed state. But prefix reuse is the one most production deployments lean on hardest, so let us quantify the loss rather than gesture at it.

### 4.1 Why caching the attention KV alone buys you nothing

The first instinct is to keep the existing prefix cache for the attention layers and accept the loss on the recurrent ones. For a model like Nemotron-H that would be four layers out of twenty-eight sequence mixers, so you would expect to keep about one seventh of the benefit.

You keep essentially none of it, and the reason is layer coupling.

Suppose you have a prefix-cache hit for a 2,048-token prefix: the K and V entries for all four attention layers are sitting in blocks you can point at. To decode the 2,049th token you need the recurrent state at position 2,048 in all twenty-four Mamba-2 layers. To produce the state at layer $\ell$ you must feed it $h^{(\ell)}_t$ for every $t \le 2048$. To produce $h^{(\ell)}_t$ you must run layer $\ell-1$ at position $t$. To run an attention layer at position $t$ you need its query, which comes from that layer's input at position $t$, which comes from the layer below. So you must execute the full stack at every prefix position anyway — which regenerates the K and V you had cached, as a byproduct, for free.

What did the cache save? The K and V projection GEMMs at four layers, and the write into the cache. Against a full forward pass over 2,048 positions through fifty-two layers, that is a rounding error.

There is one structural escape, and it is worth naming because it is an architecture-design consequence that most people would not connect to scheduling: **if every attention layer sat above every recurrent layer, the top block would retain ordinary prefix caching.** Cache hits would let you skip prefix positions for the attention-only suffix of the stack. Real hybrids disperse their attention layers through the depth — Nemotron-H's `hybrid_override_pattern` places its four attention layers evenly, with the deepest recurrent layer near the very top — precisely because dispersion is better for quality. Dispersion is what costs you prefix caching. That trade is not usually stated in an architecture paper, and it is entirely real for whoever has to serve the result.

The vLLM team's [Qwen3-Next support post](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11) lists prefix caching as a roadmap gap for hybrid models rather than a solved feature, which is consistent with the argument above: there is no cheap version of it.

### 4.2 What snapshot caching costs, and the length at which it wins

The mechanism that does work is a **state snapshot store**: at chosen prefix boundaries, copy the whole recurrent state aside and index it by the prefix hash. A later request whose prompt begins with exactly that prefix restores the snapshot and skips prefill entirely — attention KV and all.

The right way to compare it against a block-granular KV cache is *value density*: how much prefill compute does one cached byte buy you? Using the standard approximation that prefill costs about $2 N_p$ FLOPs per token for a model with $N_p$ parameters (it ignores the attention term, which is small for an 8B model at these lengths):

- **A dense KV cache** over a prefix of length $P$ stores $B_{\text{tok}} P$ bytes and saves $2 N_p P$ FLOPs. Value density $= 2 N_p / B_{\text{tok}}$, **independent of $P$**.
- **A hybrid state snapshot** at prefix length $P$ stores $\Sigma$ bytes — the same $\Sigma$ whether $P$ is 100 or 100,000 — and saves $2 N_p P$ FLOPs. Value density $= 2 N_p P / \Sigma$, **linear in $P$**.

Set them equal, using the hybrid's own $B_{\text{tok}}$ as the reference, and the crossing point is:

$$P^{*} \;=\; \frac{\Sigma}{B_{\text{tok}}} \;=\; S^{\dagger}$$

The same characteristic length again. **Snapshot caching a hybrid prefix is worth fewer bytes-per-FLOP than an ordinary KV cache below $S^{\dagger}$ tokens and more above it.** For Nemotron-H-8B that boundary is 3,162 tokens.

| Prefix length $P$ | Snapshot bytes per cached token | Value density vs KV cache | Source |
| --- | --- | --- | --- |
| 512 | 98.8 KiB | 0.16× (6.2× worse) | derived |
| 2,048 | 24.7 KiB | 0.65× | derived |
| 3,162 | 16.0 KiB | 1.00× (break-even) | derived |
| 8,192 | 6.2 KiB | 2.6× | derived |
| 32,768 | 1.6 KiB | 10.4× | derived |

So the advice inverts the dense intuition. In a dense engine you cache every prefix you can, and short shared system prompts are the easiest win there is. In a hybrid engine, **short shared prefixes are the worst thing you can snapshot** and long ones are the best. A 300-token system prompt costs 49.4 MiB to remember, which is 169 KiB per cached token — more than a dense Llama-3.1-8B spends per token of real KV. A 32k retrieved document costs the same 49.4 MiB and is a spectacular deal.

#### Worked example: a 2,048-token shared system prompt, 100 requests

An agent deployment where every request begins with the same 2,048-token instruction block.

**Dense Llama-3.1-8B with a radix prefix cache.** Ninety-nine of the hundred requests hit. Prefill FLOPs avoided: $99 \times 2 \times 8\times10^{9} \times 2048 = 3.24\times10^{15}$. Assume 40% model FLOPs utilization against the H100 datasheet's 989 TFLOP/s of dense bf16, giving about 396 TFLOP/s: **8.2 seconds of GPU time saved across the batch, about 83 ms off each request's TTFT.** Cache cost: $2048 \times 131{,}072 = 256$ MiB, shared by all of them. Source: derived, with the 40% MFU stated as an assumption and the peak rate cited from the datasheet.

**Nemotron-H-8B with a KV-only prefix cache.** Zero saved, for the layer-coupling reason in 4.1.

**Nemotron-H-8B with one state snapshot at exactly 2,048 tokens.** The same 99 hits, the same prefill avoided. Cache cost: 49.4 MiB of state plus 32 MiB of KV for the prefix. That is 81.4 MiB, which is *less* than the dense model's 256 MiB in absolute terms — but it consumes one whole state slot out of 256, permanently, because a snapshot is a state-shaped object and the state pool is where state-shaped objects live.

That last sentence is the real constraint, and it is a coverage constraint rather than a bytes constraint. **Each distinct cached prefix costs you one concurrent request.** A radix tree over 43 GiB of KV pool can hold thousands of distinct prefixes at 16-token granularity. A snapshot store that reserves ten percent of a 256-slot pool holds exactly 25 prefixes. If your traffic has three system prompts, that is fine and you should do it. If your traffic has ten thousand distinct conversation prefixes, the snapshot store does not have a shape that fits.

And partial matching is gone. A radix tree matching an 1,800-token common prefix with 2,048 cached recovers 1,792 of those tokens, losing at most fifteen to block rounding. A snapshot store with snapshots every $G$ tokens recovers $\lfloor \text{LCP}/G \rfloor \cdot G$ and loses $G/2$ on average — 256 tokens at $G = 512$, and you paid 49.4 MiB for each of those snapshot points.

Here is the lookup, written so the two paths are visibly different objects:

```python
# nanoserve/hybrid_sched.py (continued)
@dataclass
class StateSnapshot:
    prefix_hash: int
    prefix_len: int          # exact, not rounded to a block
    state_slot: int          # a slot in the state pool, pinned
    kv_blocks: list[int]     # the attention KV for the same prefix
    refs: int = 0


class HybridPrefixCache:
    """Block-granular reuse for attention, whole-object reuse for state.

    The attention half is the radix cache from the prefix-sharing post and is
    only ever consulted alongside a state hit: a KV hit without a matching
    state hit saves nothing, because the recurrent layers force a full
    re-prefill that regenerates the KV anyway.
    """

    def __init__(self, radix, state_pool: StateSlotPool, max_snapshots: int):
        self.radix = radix                      # from nanoserve/kv/radix.py
        self.state_pool = state_pool
        self.max_snapshots = max_snapshots
        self.snapshots: dict[int, StateSnapshot] = {}
        self.hits = self.misses = self.wasted_kv_hits = 0

    def lookup(self, token_ids: list[int]) -> StateSnapshot | None:
        """Longest snapshot whose prefix is an exact prefix of token_ids."""
        best: StateSnapshot | None = None
        for snap in self.snapshots.values():
            if snap.prefix_len > len(token_ids):
                continue
            if hash(tuple(token_ids[:snap.prefix_len])) != snap.prefix_hash:
                continue
            if best is None or snap.prefix_len > best.prefix_len:
                best = snap
        if best is None:
            self.misses += 1
            # A KV-only match here is real but worthless; count it so the
            # dashboard does not report a hit rate you cannot spend.
            if self.radix.match(token_ids)[1] > 0:
                self.wasted_kv_hits += 1
            return None
        self.hits += 1
        best.refs += 1
        return best

    def should_snapshot(self, prefix_len: int, shape: HybridShape) -> bool:
        """Only snapshot prefixes long enough to beat a plain KV cache."""
        if len(self.snapshots) >= self.max_snapshots:
            return False
        return prefix_len >= shape.state_equivalent_tokens
```

The `wasted_kv_hits` counter is the honest version of a metric that will otherwise lie to you. A hybrid engine that reuses the dense prefix-cache instrumentation will report a healthy hit rate on the attention blocks and show no throughput improvement at all, and the counter above is how you find out why in one glance instead of one week.

`should_snapshot` is the derivation from earlier turned into a policy: refuse to remember prefixes shorter than $S^{\dagger}$. It is three lines and it stops the snapshot store from filling with 200-token system prompts that each cost a concurrent request.

---

## 5. Preemption becomes a copy, not a recompute

![A timeline showing a hybrid request checkpointing its fixed recurrent state while attention blocks are released, then restoring the state without a full prefill](/imgs/blogs/batching-and-scheduling-hybrid-models-5.webp)

Now the good news, and it is genuinely good.

[The eviction and preemption post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) framed preemption as a choice between two bad options. **Recompute**: drop the victim's blocks, and when it resumes, re-run prefill over its whole context. **Swap**: copy the victim's KV to host memory over PCIe and copy it back later. Which is cheaper depends on context length, on host link bandwidth, and on how much compute the GPU has spare, and the crossover moves with all three.

For a hybrid, that decision collapses, and it collapses in the direction you want.

**Recompute is not available for the state.** The state at position $S$ is a fold over positions 1 through $S$. There is no suffix you can drop and rebuild; to get the state back you replay the entire prefill from token zero. Not the tail — all of it. And because the attention KV cannot be usefully reconstructed without also running the recurrent layers (section 4.1, same argument), dropping the KV and keeping the state does not help either. Preemption for a hybrid is all-or-nothing: either you keep the request's memory somewhere, or you pay a **complete** re-prefill.

**But the thing you must keep is small, fixed-size and contiguous** — the friendliest possible object to copy.

#### Worked example: preemption cost at 8,192 tokens of context

Assume a pinned-memory host transfer achieving 50 GB/s. PCIe Gen5 x16 tops out near 63 GB/s in one direction from the link spec, and pinned copies typically land somewhere in the 45–55 GB/s band; measure yours rather than trusting 50. Assume prefill at 396 TFLOP/s, that is 40% MFU against the H100 datasheet's 989 TFLOP/s dense bf16.

| Strategy | Bytes moved | Time | Source |
| --- | --- | --- | --- |
| Llama-3.1-8B, recompute | 0 | 331 ms | derived (assumes 40% MFU) |
| Llama-3.1-8B, swap KV | 1.00 GiB | 21.5 ms | derived (assumes 50 GB/s) |
| Nemotron-H-8B, recompute | 0 | 331 ms | derived (assumes 40% MFU) |
| Nemotron-H-8B, swap state only | 49.4 MiB | 1.04 ms | derived (assumes 50 GB/s) |
| Nemotron-H-8B, swap state + KV | 177 MiB | 3.72 ms | derived (assumes 50 GB/s) |

The re-prefill figure: $2 \times 8\times10^{9} \times 8192 = 1.31\times10^{14}$ FLOPs at $3.96\times10^{14}$ FLOP/s gives 331 ms. The state copy: $51{,}806{,}208 / 50\times10^{9}$ gives 1.04 ms. **A factor of 318 between the only two options a hybrid gives you**, which is not a trade-off, it is an answer.

Two further properties make this better than the raw ratio suggests.

**The state copy is flat in $S$.** One millisecond whether the request is at 1,000 tokens or 128,000. A dense engine's swap cost grows linearly — 344 ms to move 16 GiB at 128k — so preempting a long dense request is nearly as expensive as recomputing it, which is why dense preemption policies get so fussy about picking a young victim. A hybrid's state cost has no such gradient. Adding the KV back in, the total is $B_{\text{tok}}(S + S^{\dagger})$ bytes, so even at 128k the hybrid moves 2.05 GiB against the dense model's 16 GiB.

**Preemption becomes cheap enough to use as a routine scheduling tool.** At about 1 ms of copy per victim against a decode step of a few tens of milliseconds, you can preempt aggressively to protect a latency SLO without the thrash risk that a dense engine carries. The `ThrashDetector` from the preemption post is still worth keeping, but its alarm threshold should be a great deal higher here — a hybrid engine that preempts on 10% of steps is spending a percent or two of its step time, not melting down.

Watch the two memory kinds behave differently across a preemption:

<figure class="blog-anim">
<svg viewBox="0 0 700 360" role="img" aria-label="Three request slots where the key-value bars grow each step while the fixed state boxes stay the same size, then one slot is preempted and its state box moves to a host checkpoint shelf before returning" style="width:100%;height:auto;max-width:860px">
<style>
.h63-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.h63-sub{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.h63-slot{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.h63-track{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:4 4}
.h63-kv{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.h63-state{fill:var(--surface,#f3f4f6);stroke:var(--text-primary,#1f2937);stroke-width:2}
.h63-stxt{font:600 11px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.h63-shelf{fill:none;stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:6 5}
.h63-rule{stroke:var(--border,#d1d5db);stroke-width:1}
@keyframes h63-grow{0%{transform:scaleX(.10)}44%{transform:scaleX(1)}52%{transform:scaleX(.10)}100%{transform:scaleX(.72)}}
@keyframes h63-grow2{0%{transform:scaleX(.16)}40%{transform:scaleX(.92)}100%{transform:scaleX(.55)}}
@keyframes h63-kvgone{0%,42%{transform:scaleX(.88);opacity:1}54%,80%{transform:scaleX(0);opacity:.12}90%{transform:scaleX(.10);opacity:1}100%{transform:scaleX(.34);opacity:1}}
@keyframes h63-move{0%,42%{transform:translate(0px,0px)}56%,78%{transform:translate(232px,96px)}92%,100%{transform:translate(0px,0px)}}
@keyframes h63-pulse{0%{opacity:.45}20%{opacity:1}45%{opacity:.55}70%{opacity:1}100%{opacity:.45}}
@keyframes h63-shelfon{0%,44%{opacity:.22}58%,76%{opacity:1}90%,100%{opacity:.22}}
.h63-a1{animation:h63-grow 12s ease-in-out infinite}
.h63-a3{animation:h63-grow2 12s ease-in-out infinite}
.h63-a2{animation:h63-kvgone 12s ease-in-out infinite}
.h63-mv{animation:h63-move 12s ease-in-out infinite}
.h63-pl{animation:h63-pulse 12s ease-in-out infinite}
.h63-sh{animation:h63-shelfon 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.h63-a1,.h63-a2,.h63-a3,.h63-mv,.h63-pl,.h63-sh{animation:none}}
</style>
<text class="h63-lbl" x="24" y="22">fixed state, one slot each</text>
<text class="h63-lbl" x="300" y="22">attention KV, grows every step</text>
<text class="h63-slot" x="24" y="60">slot 1</text>
<rect class="h63-state" x="72" y="44" width="86" height="34" rx="6"/>
<text class="h63-stxt" x="115" y="66">49.4 MiB</text>
<rect class="h63-track" x="300" y="44" width="374" height="34" rx="6"/>
<rect class="h63-kv h63-a1" x="300" y="44" width="374" height="34" rx="6"/>
<text class="h63-slot" x="24" y="124">slot 2</text>
<g class="h63-mv">
<rect class="h63-state" x="72" y="108" width="86" height="34" rx="6"/>
<text class="h63-stxt" x="115" y="130">49.4 MiB</text>
</g>
<rect class="h63-track" x="300" y="108" width="374" height="34" rx="6"/>
<rect class="h63-kv h63-a2" x="300" y="108" width="374" height="34" rx="6"/>
<text class="h63-slot" x="24" y="188">slot 3</text>
<rect class="h63-state" x="72" y="172" width="86" height="34" rx="6"/>
<text class="h63-stxt" x="115" y="194">49.4 MiB</text>
<rect class="h63-track" x="300" y="172" width="374" height="34" rx="6"/>
<rect class="h63-kv h63-a3" x="300" y="172" width="374" height="34" rx="6"/>
<line class="h63-rule" x1="24" y1="226" x2="676" y2="226"/>
<rect class="h63-shelf h63-sh" x="296" y="196" width="100" height="42" rx="8"/>
<text class="h63-lbl" x="24" y="262">host checkpoint shelf</text>
<text class="h63-sub" x="24" y="284">the state box leaves the GPU whole, 49.4 MiB in about 1.0 ms</text>
<text class="h63-sub" x="24" y="306">its KV blocks are freed and never copied at all</text>
<text class="h63-sub" x="24" y="328">on resume the box comes back and decoding continues, no re-prefill</text>
<rect class="h63-state h63-pl" x="536" y="252" width="140" height="30" rx="6"/>
<text class="h63-stxt" x="606" y="272">rewritten in place</text>
<text class="h63-sub" x="536" y="304">size never changes</text>
</svg>
<figcaption>Three slots of a hybrid batch. The KV bars extend on every decode step; the state boxes change contents but never size. When slot 2 is preempted, its KV is simply freed while its state box travels intact to the host and back — the only object in the request that cannot be recomputed is also the only one small enough to copy in a millisecond.</figcaption>
</figure>

The checkpoint arena is a fixed allocation, sized by how many preempted requests you are willing to hold:

```python
# nanoserve/hybrid_sched.py (continued)
import torch


class StateCheckpointArena:
    """Pinned host storage for preempted recurrent states.

    Every checkpoint is the same size, so this is a slab of identical rows
    with a free list -- no fragmentation, no size classes, no compaction.
    """

    def __init__(self, state_numel: int, rows: int, dtype=torch.bfloat16):
        self.buf = torch.empty((rows, state_numel), dtype=dtype,
                               device="cpu", pin_memory=True)
        self._free: list[int] = list(range(rows))
        self.stream = torch.cuda.Stream()
        self.bytes_per_row = self.buf.element_size() * state_numel

    def capacity_gib(self) -> float:
        return self.buf.numel() * self.buf.element_size() / 1024 ** 3

    def save(self, device_state: torch.Tensor) -> tuple[int, torch.cuda.Event]:
        """Copy a request's whole state to the host. Returns (row, event)."""
        if not self._free:
            raise RuntimeError("checkpoint arena full; lower max_preempted")
        row = self._free.pop()
        with torch.cuda.stream(self.stream):
            self.buf[row].copy_(device_state.reshape(-1), non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        return row, done

    def restore(self, row: int, device_state: torch.Tensor) -> torch.cuda.Event:
        with torch.cuda.stream(self.stream):
            device_state.reshape(-1).copy_(self.buf[row], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self._free.append(row)
        return done
```

Three things there are load-bearing.

**A separate CUDA stream.** The copy has no data dependency on the next decode step for any *other* request, so it should overlap with compute rather than serializing behind it. Record an event and make the resume path wait on that event instead of calling `synchronize()`, exactly as the KV swap path does.

**Pinned host memory.** Pageable memory forces the driver through a staging buffer and roughly halves your achievable bandwidth. The whole argument of this section rests on the copy being about a millisecond; on pageable memory it is two or three, and the preempt-aggressively conclusion gets shakier.

**The row is freed at the start of `restore`, not the end.** The copy is asynchronous, so the row is logically reclaimable as soon as the transfer is enqueued on the stream — but only because this arena has exactly one stream and therefore a total order on its transfers. If you shard the arena across streams, move the `append` after the event wait, or you will hand the same row to two requests.

Now the preemption path itself:

```python
class HybridPreemptor:
    def __init__(self, pools: HybridPools, arena: StateCheckpointArena,
                 states: dict[str, torch.Tensor], block_tables: dict[str, list[int]]):
        self.pools = pools
        self.arena = arena
        self.states = states                # rid -> device state tensor
        self.block_tables = block_tables    # rid -> physical block ids
        self.checkpointed: dict[str, tuple[int, torch.cuda.Event, int]] = {}

    def preempt(self, req) -> None:
        """Free everything this request holds, keeping only its state."""
        row, event = self.arena.save(self.states[req.rid])
        # The KV is NOT copied: it will be regenerated by the resumed forward
        # pass only if we re-prefill, and we are not going to re-prefill.
        blocks = self.block_tables.pop(req.rid, [])
        self.pools.free_blocks += len(blocks)
        self.pools.state.give_back(req.state_slot)
        self.checkpointed[req.rid] = (row, event, req.ctx)
        req.preemptions += 1

    def resume(self, req) -> bool:
        """Bring a checkpointed request back. False if there is no room yet."""
        row, save_done, ctx = self.checkpointed[req.rid]
        need_blocks = math.ceil(ctx / BLOCK_SIZE)
        if self.pools.free_blocks - need_blocks < self.pools.watermark:
            return False
        slot = self.pools.state.take(req.rid)
        if slot is None:
            return False

        save_done.synchronize()             # the save must have landed
        restore_done = self.arena.restore(row, self.states[req.rid])
        req.state_slot = slot
        self.pools.free_blocks -= need_blocks
        req.pending_event = restore_done    # engine waits before the next step
        del self.checkpointed[req.rid]
        return True
```

And here is the wrinkle that the tidy version above hides, which you will hit the first time you run this against a real model.

**The KV is gone but the context is not.** After `preempt`, the request's attention KV for its 8,192 tokens has been freed. After `resume`, it holds a fresh set of blocks with garbage in them, and a state that is correct at position 8,192. The recurrent layers can continue. The attention layers **cannot** — they need K and V for positions 1 through 8,192 and those bytes are gone.

So there are exactly two honest designs, and you must pick one:

1. **Checkpoint the KV too.** Copy $B_{\text{tok}} \cdot S$ bytes alongside the state, restore both, and resume with everything intact. At 8,192 tokens that is 177 MiB and 3.7 ms total instead of 49.4 MiB and 1.0 ms. Still 89× cheaper than a re-prefill. This is the correct default and it is what the "swap state + KV" row of the table measures.
2. **Checkpoint the state and re-prefill only for the attention layers.** Tempting, and wrong: to compute the attention layers' K and V at prefix positions you need those positions' hidden states, which requires running the layers below, which are recurrent, which would advance the state you just restored. There is no partial forward pass that fills the KV without corrupting the state.

Design 1 it is. The code above is therefore incomplete on purpose — it is the version that looks right until you trace the attention layers, which is a failure mode worth seeing in print rather than in production. The fix is one extra `save` call for the block contents, using the existing `HostSwapSpace` from the preemption post, gated on the same event.

The lesson generalizes past this function: **in a hybrid engine the state and the KV are not independent objects, even though they live in independent pools.** They are two projections of one request's history, and any operation that touches one without the other has to justify itself.

---

## 6. The engine diff, end to end

Assemble the pieces, and fix the partition problem from section 2 while doing it.

The core issue was that $N$ is chosen at startup and no single value is right across the context range. The clean fix — one byte-denominated budget, with state and KV drawn from the same physical pool — requires the two kinds to share a physical granularity, which they do not naturally: a Mamba-2 layer's state is 2,158,592 bytes, and 16 tokens of this model's KV is 262,144. Neither divides the other.

The vLLM team solved exactly this for Qwen3-Next by moving the granularity rather than the objects. From [their post](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11), verbatim: "vLLM automatically tunes the 'logical' block size of the full attention layers to ensure that the state for the full attention layers and linear attention layers occupy the same amount of 'physical' GPU memory." Force both into a common physical unit and one free list serves both. The cost is a block size you no longer choose freely, which then constrains your attention kernel — their [hybrid SSM disaggregation post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) describes the downstream bookkeeping, including subdividing attention blocks to satisfy FlashInfer's 16-token requirement and hybrid-memory-allocator padding that inflates an attention block to 400 tokens. All cited; I have built none of it.

`nanoserve` takes the cruder route that fits in a hundred lines: **grow the state pool in chunks carved from the block pool.** Coarse, but dynamic, and it removes the startup bet.

```python
# nanoserve/hybrid_sched.py (continued)
class GrowableStatePool(StateSlotPool):
    """State slots carved from the block pool in fixed-size chunks.

    A chunk is `slots_per_chunk` slots' worth of bytes, rounded up to whole
    blocks so the block accountant stays integral. Chunks are never split and
    are returned only when every slot in them is free -- coarse, but it means
    the partition follows the traffic instead of a startup flag.
    """

    def __init__(self, bytes_per_slot: int, block_bytes: int,
                 slots_per_chunk: int = 16, max_slots: int = 4096):
        super().__init__(0, bytes_per_slot)
        self.block_bytes = block_bytes
        self.slots_per_chunk = slots_per_chunk
        self.max_slots = max_slots
        self.blocks_per_chunk = math.ceil(
            slots_per_chunk * bytes_per_slot / block_bytes)
        self.chunks = 0

    def grow(self, pools: "HybridPools") -> bool:
        """Take one chunk from the KV pool. False if it would not fit."""
        if self.num_slots >= self.max_slots:
            return False
        if pools.free_blocks - self.blocks_per_chunk < pools.watermark:
            return False
        pools.free_blocks -= self.blocks_per_chunk
        base = self.num_slots
        self._free.extend(range(base, base + self.slots_per_chunk))
        self.num_slots += self.slots_per_chunk
        self.chunks += 1
        return True

    def shrink(self, pools: "HybridPools") -> bool:
        """Return the top chunk if all of its slots are free."""
        if self.chunks == 0:
            return False
        top = set(range(self.num_slots - self.slots_per_chunk, self.num_slots))
        if not top.issubset(self._free):
            return False
        self._free = [s for s in self._free if s not in top]
        self.num_slots -= self.slots_per_chunk
        self.chunks -= 1
        pools.free_blocks += self.blocks_per_chunk
        return True
```

Wire it into admission so that a slot shortage tries to grow before it refuses:

```python
class AdaptiveHybridFCFS(HybridFCFS):
    name = "hybrid-fcfs-adaptive"

    def select_admits(self, st: SchedulerState) -> list[Admission]:
        pool = self.pools.state
        if isinstance(pool, GrowableStatePool) and pool.free_slots == 0:
            # Slots are the binding constraint right now. Convert some blocks.
            pool.grow(self.pools)
        return super().select_admits(st)

    def on_step(self, st: SchedulerState, served: dict[str, int]) -> None:
        pool = self.pools.state
        if not isinstance(pool, GrowableStatePool):
            return
        # Give a chunk back only when slots are clearly over-provisioned and
        # blocks are tight -- the opposite signal from the growth path.
        idle = pool.free_slots
        if idle >= 2 * pool.slots_per_chunk and self.pools.free_blocks < 4 * pool.blocks_per_chunk:
            pool.shrink(self.pools)
```

The asymmetry between `grow` and `shrink` is deliberate, and it is the standard shape for any adaptive partition: grow on a hard signal (a request was about to be refused), shrink on a conservative one (slots are visibly idle *and* blocks are visibly tight). Symmetric hysteresis oscillates. The double condition on the shrink path is what stops the pool flapping a chunk back and forth every few steps at the knee.

Finally the planner, which is the tool you actually run before provisioning anything:

```python
def plan(shape: HybridShape, free_bytes: int, lengths: list[int],
         slot_counts: list[int]) -> None:
    print(f"S-dagger (state-equivalent length): "
          f"{shape.state_equivalent_tokens:,} tokens "
          f"= {shape.phantom_blocks} phantom blocks")
    for n in slot_counts:
        kv_bytes = free_bytes - n * shape.state_bytes
        if kv_bytes <= 0:
            print(f"  N={n:<5} state pool exceeds free memory")
            continue
        knee = kv_bytes / (n * shape.kv_bytes_per_token)
        row = []
        for s in lengths:
            cap = kv_bytes // (shape.kv_bytes_per_token * s)
            row.append(f"{min(n, cap):>6}")
        print(f"  N={n:<5} knee={knee:>8.0f} tok  " + " ".join(row))


plan(NEMOTRON_H_8B, int(55.6 * 1024 ** 3),
     lengths=[1024, 4096, 8192, 32768, 131072],
     slot_counts=[64, 128, 256, 512])
```

```console
S-dagger (state-equivalent length): 3,162 tokens = 198 phantom blocks
  N=64    knee=   54280 tok      64     64     64     64     26
  N=128   knee=   26538 tok     128    128    128    101     26
  N=256   knee=   14234 tok     256    256    256    101     21
  N=512   knee=    7082 tok     444    438    320    101     12
```

Read the columns and the trade is undeniable. At 128k context, `N=64` yields 26 concurrent requests and `N=512` yields 12 — a small state pool wins, because it leaves the KV pool intact. At 1,024 tokens, `N=512` yields 444 and `N=64` yields 64 — a large state pool wins by a factor of seven. There is no row that is good everywhere, which is precisely why the growable pool exists. Source: derived, same arithmetic as section 2.

One caveat on that table that the arithmetic cannot express: the `N=512` row at short context reports 444 rather than 512 because the block pool binds first, and 444 concurrent sequences is far past the point where a Python scheduler loop and per-step kernel launches become the real limit. The memory model tells you what is *possible*; it does not tell you what your step time will tolerate. Keep the old soft `max_running` as a second, independent cap and set it from step-time measurements, not from this table.

---

## 7. Handing a state vector across a prefill/decode boundary

The last mechanism, and the one where public engineering is most active.

[Prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) splits the two phases onto separate GPU pools so a long prefill cannot stall a room full of decoders. The handoff is the whole design: prefill finishes, and the request's cache has to reach the decode instance before its first decode step.

For a dense model the handoff object is well understood — a list of KV blocks, per-token, uniform layout, shardable along the head dimension in exactly the way tensor parallelism already shards it. For a hybrid there are two objects with incompatible shapes, and the second one has no per-token dimension to index along at all.

![A physical pool indexed through a full-attention descriptor and a state descriptor whose reads merge at the decode instance](/imgs/blogs/batching-and-scheduling-hybrid-models-6.webp)

The vLLM team's [disaggregated serving for hybrid SSM models post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) is the primary public account of solving this, and the figure above traces the mechanism they describe. Everything in this section is cited from that post; I have run none of it and I am not extrapolating past what it says.

Their statement of the problem matches the derivation in section 4 exactly: attention layers use a uniform per-token KV layout while SSM layers hold a fixed convolution state plus temporal state with no per-token dimension, so one descriptor or index format cannot address both. Their answer is **dual descriptor views over the same physical memory** — one view indexes attention blocks, another indexes SSM state — which lets a transfer happen between instances at different tensor-parallel degrees without reshuffling the underlying bytes. They also describe bridging physical and logical blocks, including subdividing attention blocks to satisfy FlashInfer's 16-token requirement, and hybrid-memory-allocator padding that inflates an attention block to 400 tokens.

The second mechanism is a layout choice, and it is the one worth internalizing because it generalizes. They decompose the convolution state into three sub-projections and lay the SSM state out as `(dim, state_len)` so that **each decode rank RDMA-reads only its own tensor-parallel slice**, and padding is never transferred at all — reported as roughly 50 MB per request saved on bf16. That is a layout designed so that the natural shard boundary is also the natural read boundary, which is the same trick that makes a per-token KV layout shardable in the first place. The hybrid case just had to invent it twice.

For results, the post reports a `Nemotron-3-Super-120B-A12B-FP8` deployment on 8×H200 with prefill and decode disaggregated (prefill at TP4 plus decode at TP4) that "Pareto-dominates the co-located baseline at higher batch sizes", and gives a worked configuration for `Nemotron-3-Nano-30B-A3B-FP8` at TP=2, described as 52 layers alternating Mamba and full attention with five hybrid-memory-allocator groups and six shared KV tensors. Availability is stated as vLLM v0.20.0 and later, with `VLLM_SSM_CONV_STATE_LAYOUT=DS` and a `NixlConnector` in the `kv_both` role. Note what that headline claim does and does not say: it is a Pareto statement about a curve at higher batch sizes, not a single speedup number, and the post names its own limits — Mamba1 unsupported, gated DeltaNet pending, mixed block sizes unsupported, and the interaction with speculative decoding "not extensively validated".

### 7.1 What the handoff costs, derived

You can compute the transfer bill from the same two terms. Bytes handed over per request:

$$T(S) \;=\; \Sigma \;+\; B_{\text{tok}}\,S \;=\; B_{\text{tok}}\bigl(S + S^{\dagger}\bigr)$$

At 50 GB/s of achieved network bandwidth — a 400 Gb/s InfiniBand link is 50 GB/s at the signaling rate, and RDMA typically lands somewhat under it, so measure yours:

| Context $S$ | Llama-3.1-8B transfer | Nemotron-H-8B transfer | Hybrid time at 50 GB/s | Source |
| --- | --- | --- | --- | --- |
| 1,024 | 128 MiB | 65.4 MiB | 1.37 ms | derived |
| 8,192 | 1.00 GiB | 177 MiB | 3.72 ms | derived |
| 32,768 | 4.00 GiB | 561 MiB | 11.8 ms | derived |
| 131,072 | 16.0 GiB | 2.05 GiB | 44.0 ms | derived |

The interesting entry is the top row, and it is the one that will bite an unprepared deployment. **The hybrid handoff has a floor of $\Sigma$, and the floor does not shrink with the prompt.** Below $S^{\dagger}$ tokens the transfer is mostly state, so the disaggregation tax stops improving as prompts get shorter. Dense disaggregation gets cheaper without limit as prompts shrink; hybrid disaggregation does not.

#### Worked example: the state floor at high request rate

A short-prompt endpoint at 500 requests per second, disaggregated. Every request ships its state across the fabric exactly once:

$$500 \times 51{,}806{,}208 \text{ bytes/s} \;=\; 25.9 \text{ GB/s}$$

That is roughly half of a 400 Gb/s link consumed by state alone, before a single byte of KV, and it is invariant to prompt length. Source: derived. The same endpoint on Llama-3.1-8B with 512-token prompts ships $500 \times 512 \times 131{,}072 = 33.6$ GB/s — more, but it *falls* if prompts get shorter, whereas the hybrid's does not.

The practical reading: **short-prompt hybrid traffic is a poor candidate for disaggregation.** Co-locate it. The architecture that makes hybrids attractive at long context is the same one that puts a fixed tax on every handoff, and at short context the tax is the whole bill. This is the same regime split as everywhere else in this post, keyed to the same number.

---

## 8. Measuring it honestly

Everything above is arithmetic, which means it is checkable, which means you should check it rather than trust it.

**Four metrics your dense engine does not emit.** Add these before anything else; they answer the questions this post raises in one glance.

| Metric | What it tells you | Alarm condition |
| --- | --- | --- |
| `state_slots_used / state_slots_total` | whether the slot pool binds | near 1.0 while block use is low |
| `refusals{reason}` split by reason | which side of the knee traffic is on | `no_state_slot` dominating |
| `prefix_cache_wasted_kv_hits` | attention hits with no state hit | any non-trivial rate |
| `checkpoint_bytes_per_second` | preemption traffic on the host link | above about 20% of link rate |

The refusal split is the one to build first. If `no_state_slot` dominates, your median context is below the knee and you should grow the state pool. If `no_blocks` dominates, you are above it and the state pool is stealing memory the KV pool needs. If both are non-trivial you are near the knee, which is the good place to be and also the place where a fixed partition is the most expensive.

**Verifying the two-term footprint.** The formula has two parameters, so two measurements determine it. Run one request at two widely separated context lengths, read steady-state allocation after prefill, and fit:

```python
def fit_two_terms(run_prefill, s1: int, s2: int) -> tuple[float, float]:
    """Return (kv_bytes_per_token, fixed_state_bytes) from two prefills."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()      # weights and workspace only

    run_prefill(s1)
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated() - base

    run_prefill(s2)
    torch.cuda.synchronize()
    m2 = torch.cuda.memory_allocated() - base

    slope = (m2 - m1) / (s2 - s1)
    intercept = m1 - slope * s1
    return slope, intercept
```

```bash
python -m nanoserve.tools.fit_two_terms \
  --model nvidia/Nemotron-H-8B-Base-8K --s1 2048 --s2 32768 --batch 1
```

**What you should see.** On any 24 GB or larger card, an 8B-class hybrid in bf16 should give a slope within a few percent of 16 KiB per token and an intercept between 49 and 100 MiB — the low end if the runtime keeps SSM states in bf16, the high end if it keeps them in fp32 for numerical stability, which several implementations do. If the intercept comes back at roughly double the derived value, that is the fp32 state and not a bug; recompute $S^{\dagger}$ with the measured intercept, because every threshold in this post moves with it. Fit at batch 1: at batch $n$ the constant term is multiplied by $n$ while the weights are not, and fitting at batch 8 will attribute state bytes to weights.

**Timing the checkpoint path.** The 1 ms figure in section 5 assumed 50 GB/s. Measure it:

```python
def time_checkpoint(arena, device_state, iters=50):
    for _ in range(5):                        # warm up the pinned path
        row, ev = arena.save(device_state); ev.synchronize(); arena._free.append(row)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        row, ev = arena.save(device_state); ev.synchronize(); arena._free.append(row)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    gbs = arena.bytes_per_row / (ms * 1e-3) / 1e9
    print(f"{ms:.3f} ms per checkpoint, {gbs:.1f} GB/s effective")
```

On a PCIe Gen5 host with pinned memory you should land somewhere in the 45–55 GB/s band, so roughly 0.9–1.2 ms for a 49.4 MiB state; on Gen4, halve the bandwidth and double the time. If you get under 15 GB/s, your buffer is not actually pinned — that is the usual cause and it is worth checking before you conclude anything about the policy.

**The load-generator discipline does not change.** Warm up until numbers stop moving, `torch.cuda.synchronize()` around every timed region or use CUDA events, lock clocks if you can, and measure in steady state. Batch-1 tokens per second still tells you nothing about a server; an open-loop generator with Poisson arrivals reporting TTFT, TPOT and p99 still tells you something. That machinery is [the reproducible-benchmark post's](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) subject.

What *does* change is that **every hybrid scheduling number must be reported with its context length**, because the binding constraint flips at the knee and a benchmark that averages over a mixed length distribution is reporting a weighted average of two different regimes. Report at least a short-context point below $S^{\dagger}$, one between $S^{\dagger}$ and the knee, and one above the knee. A single aggregate number for a hybrid engine is close to meaningless.

| Claim | Value | Source |
| --- | --- | --- |
| Nemotron-H-8B state-equivalent length $S^{\dagger}$ | 3,162 tokens | derived from config.json |
| Concurrency knee at $N=256$ on one H100 | 11,071 tokens | derived (80 GB cited: H100 datasheet) |
| Dense Llama-3.1-8B knee, same cap | 1,779 tokens | derived |
| Concurrency at 128k, unified vs $N=256$ | 27 vs 21 requests | derived |
| Snapshot break-even prefix length | 3,162 tokens | derived |
| State checkpoint at 50 GB/s | 1.04 ms, flat in $S$ | derived (assumes 50 GB/s) |
| Re-prefill at 8k, 40% MFU | 331 ms | derived (assumes 40% MFU; 989 TFLOP/s cited: H100 datasheet) |
| Handoff floor per request | 49.4 MiB, invariant in $S$ | derived |
| vLLM hybrid P/D padding saved | about 50 MB per request, bf16 | cited: vLLM blog 2026-04-21 |
| Your measured slope and intercept | within a few percent of derived | reproduce: `fit_two_terms` |

---

## 9. Case studies and public numbers

Three public results, each with its setup and its limits.

**Disaggregated serving for hybrid SSM models (vLLM).** The [2026-04-21 post](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) is the most complete public account of hybrid scheduling machinery. The value for this post is not the throughput claim but the *shape* of the solution: dual descriptor views over one physical pool, a conv-state layout chosen so each decode rank reads only its own slice, and roughly 50 MB per request of padding never sent. Reported deployment: `Nemotron-3-Super-120B-A12B-FP8` on 8×H200 with prefill TP4 and decode TP4, Pareto-dominating the co-located baseline at higher batch sizes; worked config `Nemotron-3-Nano-30B-A3B-FP8` at TP=2, 52 layers alternating Mamba and full attention, five hybrid-memory-allocator groups, six shared KV tensors; vLLM v0.20.0 and later. Its own stated limits are the honest part: Mamba1 unsupported, gated DeltaNet pending, mixed block sizes unsupported, speculative-decoding interaction not extensively validated.

**Qwen3-Next hybrid cache management (vLLM).** The [2025-09-11 post](https://vllm.ai/blog/2025-09-11-qwen3-next) gives the equal-physical-memory trick verbatim, which is the cleanest published answer to the two-allocator problem and the one section 6 contrasts `nanoserve` against. It also lists prefix caching as a roadmap gap, corroborating section 4 from the engine side rather than from arithmetic. What it does not give: any throughput number, or the model's interleave ratio — the page does not state one, so neither do I.

**Model Runner V2 (vLLM).** The [2026-03-24 post](https://vllm.ai/blog/2026-03-24-mrv2) is about a decoupled persistent batch in which each request holds a fixed row in a state table independent of ordering, with per-step gathers building the ordered block tables. That design is a natural fit for a state pool, since a state slot is exactly a fixed row that must not move. The post's limits section is the citable part for this track: at v0.18.0 it lists linear-attention models — naming Qwen3.5 and Nemotron 3 Super — among the things the new runner does not yet cover. Two of the three vLLM posts most relevant to hybrid scheduling say some part of it is not done yet, which is a reasonable measure of how much machinery the dense assumption was holding up.

**On the throughput side, and where the caution belongs.** The Nemotron-H technical report ([arXiv 2504.03624](https://arxiv.org/abs/2504.03624)) reports roughly 3× the inference throughput of Llama-3.1-8B for the 8B model at long context. The mechanism, per section 2's derivation, is concurrency: 27 concurrent requests versus 3 at 128k on one H100. That is a *scheduling* result wearing an architecture costume, and it evaporates below the knee where both engines run at the same configured cap. Any hybrid throughput number without a stated context length should be treated as unlabeled.

---

## 10. When to reach for this (and when not to)

![A decision tree branching on where the median context sits relative to the state-equivalent length and the concurrency knee](/imgs/blogs/batching-and-scheduling-hybrid-models-7.webp)

The tree above is the sizing decision, and the three branches are three genuinely different operating regimes rather than three flavors of the same one.

**Below $S^{\dagger}$ (median context under about 3,162 tokens for this model): slots decide everything.** Your concurrency is exactly the slot count you configured; the block pool is scenery. Size $N$ from step-time measurements, not memory, because memory is not what is stopping you. Do not disaggregate — the state floor is most of the transfer. Do not snapshot prefixes — every snapshot costs more than the KV it replaces. And ask hard whether you want a hybrid at all: below a few hundred tokens the previous post showed the hybrid holds *more* memory per request than the dense model it replaces, and here it also gives up prefix caching, which short-prompt traffic with shared system prompts leans on more than any other workload. Short-prompt serving is the regime where hybrids lose twice.

**Between $S^{\dagger}$ and the knee (roughly 3k to 11k here): both budgets matter and the partition is worth tuning.** This is where the growable pool from section 6 earns its complexity, and where the refusal-reason split from section 8 is the metric you watch. It is also where hybrids deliver their best *ratio* against a dense model on a memory-bound endpoint — 4.5× to 7.8× the concurrency, per the section 2 table.

**Above the knee (over about 11k): blocks decide everything and the state pool should be small.** Set $N$ to roughly $F / (\Sigma + B_{\text{tok}} S)$ for your target length and no larger, because every extra slot is stolen KV. This is the long-context, agentic, document-processing regime, and it is where hybrid architectures actually pay off. Preempt freely — a millisecond per victim is nothing. Disaggregate freely — the state is 2% of the transfer at 128k. Snapshot your long shared prefixes aggressively, since at 32k a snapshot is worth ten times its bytes.

**When to just use vLLM instead of your own code.** Sooner here than anywhere else in this series. The two-term admission check is genuinely worth writing yourself: it is fifty lines, it makes the failure mode visible, and the planner in section 6 answers capacity questions before you provision anything. But the dual descriptor views, the equal-physical-granularity allocator, the state snapshot path that survives speculative decoding, and the chunk-aligned prefill scheduler are a serious body of work that the vLLM team has been building for over a year and is still labeling as partially complete in its own posts. Build the planner. Build `can_admit`. Read the engine for the rest.

**And the one thing to carry even if you never serve a hybrid.** Compute $S^{\dagger}$ for whatever you are serving, including models that do not have a state — for a dense model it is zero, and that zero is exactly why one budget was ever enough. A scheduler is a function of the memory law of the thing it schedules. Change the law and the scheduler is not tunable into correctness; it has to be rewritten. That is the frame [the capstone playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) uses to tie the series together, and hybrids are the cleanest example of it in the whole book.

---

## Key takeaways

1. **Compute $S^{\dagger} = \Sigma / B_{\text{tok}}$ first and print it in your startup logs.** For Nemotron-H-8B it is 3,162 tokens. Every scheduling threshold in this post is that one number in a different disguise.
2. **A hybrid request costs what a dense request with $S^{\dagger}$ extra tokens would cost.** The fixed state is a phantom prefix, which means your existing block-denominated admission code works if you charge each request 198 extra blocks up front.
3. **`max_running` stopped being a soft knob.** It is now a hard memory partition, and a request can be refused while 95% of the block pool is free. Emit refusals split by reason or you will never see it.
4. **Concurrency is flat below the knee $F/(N B_{\text{tok}}) - S^{\dagger}$ and falls above it.** Derived here: 11,071 tokens for a hybrid at 256 slots versus 1,779 for the dense model — a flat region 6.2× wider, and inside it capacity is a number you chose.
5. **No single slot count is right across the context range.** At 128k, 64 slots beat 512 by 2.2×; at 1k, 512 beat 64 by 7×. Grow the pool in chunks on a hard signal and shrink it on a conservative one.
6. **KV-only prefix caching for a hybrid saves essentially nothing**, because the recurrent layers force a full forward pass over the prefix that regenerates the cached bytes anyway. Count attention hits that lack a state hit as *wasted*, not as hits.
7. **Snapshot a hybrid prefix only when it is longer than $S^{\dagger}$.** Below that a snapshot buys less prefill per byte than an ordinary KV cache; above it the advantage grows linearly, reaching about 10× at 32k.
8. **Preemption becomes a copy, and the copy is flat in $S$.** Derived: 1.04 ms for 49.4 MiB of state against 331 ms to re-prefill 8k tokens. Preempt aggressively; raise your thrash alarm threshold.
9. **State and KV are not independent even though their pools are.** You cannot restore a state and re-prefill only the attention layers — the forward pass that fills the KV would advance the state. Checkpoint both or neither.
10. **Disaggregation gains a fixed per-request tax of $\Sigma$ bytes.** At 500 requests per second that is 25.9 GB/s of pure state traffic that does not shrink when prompts do. Co-locate short-prompt hybrid traffic.

---

## Further reading

- vLLM, [*Disaggregated Serving for Hybrid SSM Models*](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) (2026-04-21) — dual descriptor views over one physical pool, the DS conv-state layout, the Nemotron-3 configurations, and a candid list of what is still unsupported. The primary public source for section 7.
- vLLM, [*Qwen3-Next hybrid attention-MoE support*](https://vllm.ai/blog/2025-09-11-qwen3-next) (2025-09-11) — the equal-physical-memory block-size trick, and prefix caching named as a roadmap gap.
- vLLM, [*Model Runner V2*](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) and [*Inside vLLM: Anatomy of a High-Throughput Inference System*](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05) — the persistent-batch state table that a slot pool fits into, and the dense scheduler this post diffs against.
- [*Nemotron-H* technical report (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624) and the [Nemotron-H-8B config](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) — the layer pattern and dimensions every number in this post is derived from.
- [*Transformers are SSMs* (Dao and Gu, arXiv 2405.21060)](https://arxiv.org/abs/2405.21060) — the recurrence whose fold structure is why the state is not sliceable.
- Within this series: [hybrid models and the end of the KV-cache assumption](/blog/machine-learning/inference-engineering/hybrid-models-and-the-end-of-the-kv-cache-assumption) for the memory law this post schedules against, [the scheduler as a policy problem](/blog/machine-learning/inference-engineering/the-scheduler-as-a-policy-problem) and [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) for the code being diffed, [prefix sharing, radix trees and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) for the cache that mostly stops working here, and [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) for the dense preemption machinery.
