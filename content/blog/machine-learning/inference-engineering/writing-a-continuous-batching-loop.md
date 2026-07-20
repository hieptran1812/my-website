---
title: "Writing a continuous batching loop: the sixty lines that change everything"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Stop running a batch to completion and start mutating a running set every iteration: build nanoserve's step() function, the flattened ragged batch it feeds, and the arithmetic that turns 31 percent slot utilization into nearly all of it."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "batching",
    "scheduler",
    "continuous-batching",
    "kv-cache",
    "throughput",
    "latency",
    "pytorch",
    "vllm",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 45
---

Here is the number that should bother you. Take eight chat requests whose completions run 24, 41, 63, 88, 112, 190, 260 and 512 tokens — a perfectly ordinary spread for an assistant endpoint. Put them in one batch, run the batch to completion the way every tutorial does, and the GPU executes 512 decode steps across 8 sequence slots. That is 4,096 slot-steps of work. Only 1,290 of them produce a token anybody wanted. The other 2,806 are the GPU faithfully computing attention and MLP outputs for sequences that emitted their stop token minutes of wall-clock ago and are now being carried along as expensive ballast because the tensor shape says they must be.

That is 31.5% useful. You paid for a card that does a trillion useful operations a second and you are using roughly a third of it, and no amount of kernel optimization will get the other two thirds back, because the waste is not in the kernels. It is in the *shape of the loop*.

![Two-column comparison of a fixed eight-slot batch running to completion against a running set whose membership changes every iteration](/imgs/blogs/writing-a-continuous-batching-loop-1.webp)

This post writes the fix, and the fix is small. It is one function, `Engine.step()`, plus the scheduler function that feeds it. Together they are about sixty lines of Python, and they are the reason vLLM and TGI and SGLang exist as separate artifacts rather than as wrappers around `model.generate()`. Everything else those engines do — paged memory, prefix caching, CUDA graphs, speculative decoding — is an optimization *on top of* this loop. This loop is the thing that changes the economics.

By the end you will have `nanoserve/scheduler.py` and `nanoserve/engine.py` written and running on top of the block allocator from [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table); you will understand why a batch of requests with different lengths gets flattened into one long token array with a cumulative offset table instead of padded into a rectangle; you will be able to derive, from a length distribution alone, exactly how much throughput continuous batching buys you and exactly how much it cuts the queueing part of time-to-first-token; and you will know the specific point at which the Python scheduler itself becomes the bottleneck and the GPU starts waiting on you. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post frames the scoreboard — TTFT, TPOT, tokens per second, memory, goodput — that this one moves.

---

## 1. The reframe: a batch is a set you mutate, not a rectangle you run

Training taught all of us the wrong instinct. In training, a batch is a rectangle. You assemble `[B, S]`, you push it through, you get a loss, you throw it away and assemble the next one. Membership is fixed for the lifetime of the batch because the batch's lifetime is a single forward-backward pass. That instinct is correct there and catastrophic here.

Inference is different in one specific way: **the batch has a lifetime measured in hundreds of forward passes, and its members do not finish together.** A request that generates 24 tokens is done after 24 steps. A request that generates 512 tokens is done after 512. If membership is fixed at admission, the first request occupies a slot for 488 steps after it stopped being useful. The waste is not a rounding error; it is a function of how skewed your length distribution is, and LLM output lengths are extremely skewed.

So: stop treating the batch as a rectangle. Treat it as a **running set** — a mutable collection of live requests that the engine edits at the top of every iteration. Three operations, executed every single step:

1. **Admit.** Move requests from the waiting queue into the running set, as many as free KV blocks and the step's token budget allow.
2. **Step.** Run exactly one forward pass over whatever is in the running set right now, and sample exactly one new token for each member.
3. **Retire.** Remove requests that emitted a stop token or hit their length cap, and return their KV blocks to the free pool *immediately*, in the same iteration, so the next iteration can hand those blocks to somebody else.

The batch composition therefore changes every step. At step 417 the running set holds requests {A, C, D, F}; at step 418 it holds {A, D, F, G, H} because C finished and two more were admitted. There is no point at which the engine "starts a batch" or "finishes a batch". There are only iterations.

<figure class="blog-anim">
<svg viewBox="0 0 700 250" role="img" aria-label="Four generation lanes advance one token per step; the second lane finishes and empties, and a queued request immediately moves into the freed slot and starts generating" style="width:100%;height:auto;max-width:820px">
<style>
.cb-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cb-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.cb-chip{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cb-chip-a{fill:var(--accent,#6366f1);opacity:.85}
.cb-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.cb-lbl-r{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
.cb-ghost{fill:none;stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:6 5}
@keyframes cb-ga{0%{transform:scaleX(.12)}100%{transform:scaleX(.85)}}
@keyframes cb-gb{0%{transform:scaleX(.34)}100%{transform:scaleX(.99)}}
@keyframes cb-gc{0%{transform:scaleX(.05)}100%{transform:scaleX(.58)}}
@keyframes cb-old{0%{transform:scaleX(.55);opacity:1}34%{transform:scaleX(.95);opacity:1}44%,100%{transform:scaleX(.95);opacity:0}}
@keyframes cb-gho{0%,38%{opacity:0}46%,56%{opacity:.9}64%,100%{opacity:0}}
@keyframes cb-fly{0%,46%{transform:translate(0,0);opacity:0}50%{opacity:1}62%{transform:translate(80px,-30px);opacity:1}66%,100%{transform:translate(80px,-30px);opacity:0}}
@keyframes cb-new{0%,62%{transform:scaleX(.04);opacity:0}66%{opacity:1}100%{transform:scaleX(.42);opacity:1}}
@keyframes cb-adv{0%,60%{transform:translateX(0)}72%,100%{transform:translateX(52px)}}
.cb-a{animation:cb-ga 12s linear infinite}
.cb-b{animation:cb-gb 12s linear infinite}
.cb-c{animation:cb-gc 12s linear infinite}
.cb-o{animation:cb-old 12s ease-in-out infinite}
.cb-gh{animation:cb-gho 12s ease-in-out infinite}
.cb-fl{animation:cb-fly 12s ease-in-out infinite}
.cb-nw{animation:cb-new 12s ease-in-out infinite}
.cb-av{animation:cb-adv 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.cb-a{animation:none;transform:scaleX(.6)}.cb-b{animation:none;transform:scaleX(.8)}.cb-c{animation:none;transform:scaleX(.35)}.cb-av{animation:none}.cb-o{animation:none;opacity:1;transform:scaleX(.95)}.cb-gh,.cb-fl,.cb-nw{animation:none;opacity:0}}
</style>
<text class="cb-lbl" x="26" y="72">waiting queue</text>
<g class="cb-av">
<rect class="cb-chip" x="26" y="96" width="44" height="30" rx="6"/>
<rect class="cb-chip" x="78" y="96" width="44" height="30" rx="6"/>
</g>
<rect class="cb-chip cb-chip-a cb-fl" x="130" y="96" width="44" height="30" rx="6"/>
<text class="cb-lbl-r" x="200" y="50">req 1</text>
<text class="cb-lbl-r" x="200" y="95">req 2</text>
<text class="cb-lbl-r" x="200" y="140">req 3</text>
<text class="cb-lbl-r" x="200" y="185">req 4</text>
<rect class="cb-lane" x="210" y="30" width="450" height="30" rx="6"/>
<rect class="cb-lane" x="210" y="75" width="450" height="30" rx="6"/>
<rect class="cb-lane" x="210" y="120" width="450" height="30" rx="6"/>
<rect class="cb-lane" x="210" y="165" width="450" height="30" rx="6"/>
<rect class="cb-bar cb-a" x="210" y="30" width="450" height="30" rx="6"/>
<rect class="cb-bar cb-o" x="210" y="75" width="450" height="30" rx="6"/>
<rect class="cb-bar cb-b" x="210" y="120" width="450" height="30" rx="6"/>
<rect class="cb-bar cb-c" x="210" y="165" width="450" height="30" rx="6"/>
<rect class="cb-ghost cb-gh" x="212" y="77" width="446" height="26" rx="6"/>
<rect class="cb-bar cb-nw" x="210" y="75" width="450" height="30" rx="6"/>
<text class="cb-lbl" x="26" y="222">one iteration = one token for each live request</text>
<text class="cb-lbl" x="26" y="240">a finished request frees its blocks and its slot refills at once</text>
</svg>
<figcaption>Four requests generate side by side. Request 2 hits its stop token, empties its lane and returns its blocks; the freed slot is taken by the head of the waiting queue on the very next iteration rather than after the whole batch drains.</figcaption>
</figure>

The idea is not new and it is worth knowing whose it is. Yu et al.'s **Orca** (OSDI 2022, *Orca: A Distributed Serving System for Transformer-Based Generative Models*) introduced **iteration-level scheduling**: return to the scheduler after every iteration rather than after every request, and let the scheduler change the batch. The paper reports throughput improvements over a request-level-batching baseline that are more than an order of magnitude at matched latency — a claim I quote as an order of magnitude rather than a precise figure because the setup matters enormously and you should read the paper for it. Every production engine since has adopted the idea, sometimes under the name *continuous batching*, sometimes *in-flight batching*, sometimes *dynamic batching* (which unhelpfully also names a different, request-level technique in classical serving).

What changed between Orca and today is *where the idea gets pushed*. vLLM's own architecture write-up, [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (2025-09-05), describes the mature form: a scheduler holding a waiting queue and a running queue alongside a KV-cache manager, and a batch that "flattens" all sequences into a single concatenated *super sequence* whose members stay isolated only through position indices and attention masking. Hold onto that word — flattened. It is the mechanical core of section 4 and the single most common thing people get wrong when they write this loop themselves.

---

## 2. Three objects and nothing else

Before the loop, the nouns. Continuous batching needs exactly three data structures, and if you find yourself adding a fourth you have probably invented a bug.

A **`Request`**: everything about one client's generation, including its block table (which lives in the `PagedSequence` from the paged-cache post) and, crucially, a counter for how many of its tokens already have KV in the cache.

```python
# nanoserve/request.py
"""One in-flight request and the states it can occupy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nanoserve.blocks import PagedSequence


class State(str, Enum):
    WAITING = "waiting"     # queued, holds no KV blocks
    RUNNING = "running"     # in the running set, holds blocks
    FINISHED = "finished"   # done, blocks returned to the pool


@dataclass
class Request:
    req_id: str
    prompt_ids: list[int]
    max_tokens: int = 256
    eos_id: int | None = None
    stop_ids: frozenset[int] = frozenset()

    # engine-owned mutable state
    state: State = State.WAITING
    seq: PagedSequence | None = None          # block table, created on admission
    output_ids: list[int] = field(default_factory=list)
    num_computed: int = 0                     # tokens whose KV is already written
    arrival_step: int = -1
    admit_step: int = -1
    first_token_step: int = -1
    finish_reason: str | None = None

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def tokens_pending(self) -> int:
        """Tokens that exist but whose KV has NOT been computed yet.

        A freshly admitted request: len(prompt). A running request that just
        received a sampled token: exactly 1. That single number is the entire
        prefill/decode distinction, and the scheduler never looks at anything
        else.
        """
        return self.num_tokens - self.num_computed

    def stop_reason(self) -> str | None:
        """Called after the newest token has been appended."""
        last = self.output_ids[-1]
        if last == self.eos_id or last in self.stop_ids:
            return "stop"
        if len(self.output_ids) >= self.max_tokens:
            return "length"
        return None
```

Read `tokens_pending()` twice. It is the most important four lines in the file. A brand-new request with a 512-token prompt has 512 pending tokens. A request that just got sampled has exactly 1 pending token. **The scheduler never asks "is this prefill or decode?"** — it asks "how many tokens do you owe me?", and the answer happens to be large for newcomers and 1 for everybody else. That collapse is deliberate, and it is precisely the framing vLLM's V1 rewrite adopted: per the [vLLM V1 announcement](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27), the V1 scheduler represents each scheduling decision as a simple dictionary mapping request id to a token count, "erasing the traditional distinction between prefill and decode phases" and letting chunked prefill, prefix caching and speculative decoding all fall out of the same policy. We will build the same shape.

The **waiting queue** and the **running set** are the other two objects, and they are a `deque` and a `list`. The queue is FIFO by default. The running set is a list rather than a set because *order matters* — it defines the row order of every tensor the step builds, and keeping it stable across steps is what lets a real engine keep a persistent per-request row in its state tables. We will come back to that in section 8, because it is exactly what vLLM's Model Runner V2 rebuilt.

That is it. No batch objects, no futures, no pools of workers. The engine's whole world is `(waiting, running, allocator)`.

---

## 3. `step()`: the loop itself

![Branching diagram of one step forking into admitted and carried requests, merging into a single flat batch, then splitting into retired and continuing requests](/imgs/blogs/writing-a-continuous-batching-loop-2.webp)

The figure above is the shape to hold in your head while reading the code: a fork at the top (newcomers versus survivors), a merge in the middle (one flat batch, one forward pass), a fork at the bottom (retire versus continue), and a merge at the end (tokens out to streams). The step never branches on "is this prefill or decode". It branches on "did this finish".

Start with the scheduler, because it makes all the decisions and touches no tensors at all. Keeping the policy layer free of PyTorch means you can unit-test admission, fairness and starvation on a laptop with no GPU, which you will want to do a great deal once the policy stops being first-come-first-served.

```python
# nanoserve/scheduler.py
"""Decide what runs this iteration. Pure policy: no tensors, no CUDA."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nanoserve.blocks import BlockAllocator, PagedSequence
from nanoserve.request import Request, State


@dataclass
class SchedulerOutput:
    """The complete decision for one iteration."""

    num_tokens: dict[str, int] = field(default_factory=dict)  # req_id -> tokens
    admitted: list[str] = field(default_factory=list)
    stalled: list[str] = field(default_factory=list)


class Scheduler:
    def __init__(self, allocator: BlockAllocator, *,
                 max_running: int = 64, max_batched_tokens: int = 2048):
        self.alloc = allocator
        self.block_size = allocator.block_size
        self.max_running = max_running
        self.max_batched_tokens = max_batched_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.step_idx = 0

    def add(self, req: Request) -> None:
        req.arrival_step = self.step_idx
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_batched_tokens
        free = self.alloc.num_free          # a tally, so two requests cannot
                                            # both be promised the last block

        # (1) Running requests get served first. Decode never yields to a
        #     newcomer: a token already promised to a client outranks a token
        #     nobody has waited for yet.
        for req in self.running:
            need = req.seq.blocks_needed_for(1)
            if need > free or budget < 1:
                out.stalled.append(req.req_id)   # post 10 turns this into preemption
                continue
            free -= need
            budget -= 1
            out.num_tokens[req.req_id] = 1

        # (2) Leftover budget buys prefills from the head of the queue.
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            pending = req.tokens_pending()
            if pending > budget:
                break                            # post 13 chunks it instead
            need = -(-pending // self.block_size)
            if need > free:
                break
            self.waiting.popleft()
            req.seq = PagedSequence(req.req_id, self.alloc)
            req.state = State.RUNNING
            req.admit_step = self.step_idx
            self.running.append(req)
            out.admitted.append(req.req_id)
            free -= need
            budget -= pending
            out.num_tokens[req.req_id] = pending

        return out
```

Four decisions in there are load-bearing and none of them are obvious.

**Running requests are scheduled before waiting ones.** This is not a fairness statement, it is a latency statement. A client watching tokens stream in will notice a stall of one step; a client still waiting for its first token cannot tell the difference between 300 ms and 320 ms. Serving decode first also keeps the KV cache's occupancy monotone within a step, which makes the block accounting a single pass instead of a fixed-point iteration.

**`free` is a local tally, not a live query.** If you call `allocator.num_free` inside the loop and only actually allocate later, two requests can each be told there is one block available. That bug shows up as an `OutOfBlocks` raised from deep inside tensor construction, three frames from the decision that caused it, at 3 a.m. Decrement a local counter as you commit.

**A request that cannot get its block is *stalled*, not dropped.** It stays in `running`, holds its blocks, and simply produces no token this iteration. This is the correct minimal behaviour and also a genuinely bad policy: under memory pressure it silently converts a throughput problem into an unbounded-latency problem, because a stalled request keeps its memory while making no progress. The honest fix is preemption — evict somebody, free real blocks, make progress — and that is what [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) builds. Ship the stall first, watch it misbehave, then replace it. Naming the failure in the code (`out.stalled`) is what makes the metric exist.

**Whole prefills only, for now.** `if pending > budget: break` refuses to admit a request whose prompt does not fit in this step's token budget. That is the conservative choice and it has a real cost: a 4,000-token prompt with a 2,048-token budget will *never* be admitted, forever, no matter how empty the machine gets. The fix is to admit it in pieces, which is chunked prefill, which is [the next post](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff), and which is a two-line change to the block above. Leaving the bug visible here is deliberate; section 9 stresses it.

Now the engine, which does no policy at all.

```python
# nanoserve/engine.py
"""The loop. Everything else in nanoserve exists to keep these lines short."""

from __future__ import annotations

import torch

from nanoserve.batch import build_batch
from nanoserve.request import State


class Engine:
    def __init__(self, model, scheduler, kv_store):
        self.model = model
        self.sched = scheduler
        self.kv = kv_store
        self.step_idx = 0
        self.finished: list = []
        self.metrics = {"steps": 0, "tokens": 0, "stalls": 0, "admits": 0}

    @torch.inference_mode()
    def step(self) -> list[tuple[str, int, str | None]]:
        """Advance every live request by exactly one token.

        Returns (req_id, token_id, finish_reason) per request that ran.
        """
        plan = self.sched.schedule()                       # (a) admit + plan
        self.metrics["stalls"] += len(plan.stalled)
        self.metrics["admits"] += len(plan.admitted)
        if not plan.num_tokens:
            self.step_idx += 1
            return []                                      # idle: nothing to run

        scheduled = [r for r in self.sched.running
                     if r.req_id in plan.num_tokens]
        batch = build_batch(scheduled, plan.num_tokens,
                            self.kv, device=self.model.device)

        hidden = self.model.forward(batch)                 # (b) ONE forward pass
        logits = self.model.lm_head(hidden[batch.logits_idx])
        next_ids = self.model.sample(logits, scheduled)    # (c) one token each

        events, retired = [], []
        for req, tok in zip(scheduled, next_ids):          # (d) retire or continue
            req.output_ids.append(tok)
            if req.first_token_step < 0:
                req.first_token_step = self.step_idx
            reason = req.stop_reason()
            events.append((req.req_id, tok, reason))
            if reason:
                req.finish_reason = reason
                req.state = State.FINISHED
                retired.append(req)

        for req in retired:                                # (e) free blocks NOW
            req.seq.release()
            self.sched.running.remove(req)
            self.finished.append(req)

        self.step_idx += 1
        self.sched.step_idx = self.step_idx
        self.metrics["steps"] += 1
        self.metrics["tokens"] += len(events)
        return events                                      # (f) yield to streams
```

Counting the scheduler's `schedule` at thirty-one lines of body and the engine's `step` at thirty-two, that is sixty-three lines, and it is a complete continuous-batching engine. Not a toy version of one: the same control flow production engines run, minus the policies that make it survive adversarial load.

Two ordering rules inside `step()` are worth stating as invariants, because violating either produces a bug that is nearly impossible to find by reading logs.

**Retire before the next schedule, never after.** Blocks must go back to the pool in the same iteration the request finished. If you defer the release to the top of the next step, the pool is short by exactly the finished requests' footprint for one whole iteration, which is enough to reject an admission that should have succeeded. Under steady load with many short requests, this shaves several percent off your resident set permanently and looks like nothing at all in a profile.

**`num_computed` is advanced by whoever writes the KV, and by nobody else.** In this design `build_batch` advances it, because `build_batch` is what calls `seq.append(n)` and produces the physical slots the model will scatter K and V into. If two places update that counter, they will disagree by one somewhere, and a request will either skip a token or recompute one, and the output will look *almost* right — which is far worse than looking wrong.

---

## 4. The ragged batch: one flat array plus an offset table

Here is the part everyone underestimates. The running set contains a request contributing 512 query tokens (it was just admitted, so its whole prompt has to be prefilled) alongside two requests contributing one token each. Those do not stack into a rectangle. `[3, 512]` with padding would mean running 1,536 token-positions of attention and MLP work to get 514 tokens' worth — a 3× waste, which is exactly the padding tax [the static batching post](/blog/machine-learning/inference-engineering/static-batching-and-the-padding-tax) measures, reappearing in a new place.

The answer is to stop having a batch dimension. Concatenate every request's query tokens into one flat array of length $T = \sum_i n_i$, and carry a small integer array that says where each request's slice starts.

![Grid showing three requests contributing 512, 1 and 1 query tokens to one flat array with cumulative start offsets and per-request KV lengths](/imgs/blogs/writing-a-continuous-batching-loop-3.webp)

For the step drawn above: request A contributes 512 new tokens, B contributes 1 (it has 631 cached, so its new token is position 631 and its KV length becomes 632), C contributes 1 (88 cached, KV length 89). Then

$$
\texttt{query\_start\_loc} = [0,\ 512,\ 513,\ 514], \qquad
\texttt{seq\_lens} = [512,\ 632,\ 89]
$$

and $T = 514$. `query_start_loc` has $B+1$ entries and is the cumulative sum of query lengths with a leading zero, so request $i$ owns flat rows `[qsl[i], qsl[i+1])`. `seq_lens` is a different quantity: how many *cached* tokens each request attends over, which for a decode step is almost always much larger than its query length. Conflating the two is the classic first bug — it produces attention that silently truncates history and output that degrades gradually as contexts get long.

This is not nanoserve being clever. It is the layout the attention kernels want. FlashAttention's variable-length entry point takes `cu_seqlens_q` and `cu_seqlens_k` in exactly this cumulative form, and vLLM's Model Runner V2 write-up ([Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2), 2026-03-24) names `input_ids`, `positions`, `query_start_loc` and `seq_lens` as the per-step input tensors it builds — the same four arrays, with the interesting difference that MRv2 constructs them *on the GPU* with Triton kernels instead of in Python. Hold that thought for section 8.

Here is the builder.

```python
# nanoserve/batch.py
"""Flatten a ragged running set into the tensors one forward pass needs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor         # [T]   every request's new tokens, concatenated
    positions: torch.Tensor         # [T]   each token's index in ITS OWN sequence
    query_start_loc: torch.Tensor   # [B+1] cumulative query lengths, leading 0
    seq_lens: torch.Tensor          # [B]   cached tokens each request attends over
    slot_mapping: torch.Tensor      # [T]   flat physical KV slot per new token
    block_tables: torch.Tensor      # [B, max_blocks] physical block ids
    logits_idx: torch.Tensor        # [B]   row in T holding each request's last token


def build_batch(reqs, num_tokens: dict[str, int], kv, device="cuda") -> Batch:
    ids: list[int] = []
    pos: list[int] = []
    slots: list[int] = []
    starts: list[int] = [0]
    lens: list[int] = []

    for req in reqs:
        n = num_tokens[req.req_id]
        first = req.num_computed                  # first position lacking KV
        ids.extend(req.all_ids[first:first + n])
        pos.extend(range(first, first + n))
        slots.extend(req.seq.append(n))           # allocates blocks if needed
        starts.append(starts[-1] + n)
        lens.append(first + n)                    # KV the LAST new token sees
        req.num_computed = first + n              # exactly one writer

    max_bt = max(len(r.seq.block_table) for r in reqs)
    bt = torch.zeros(len(reqs), max_bt, dtype=torch.int32)
    for i, r in enumerate(reqs):
        table = r.seq.block_table
        bt[i, : len(table)] = torch.tensor(table, dtype=torch.int32)

    def dev(x, dt=torch.int32):
        return torch.tensor(x, dtype=dt, device=device)

    qsl = dev(starts)
    return Batch(
        input_ids=dev(ids, torch.int64),
        positions=dev(pos, torch.int64),
        query_start_loc=qsl,
        seq_lens=dev(lens),
        slot_mapping=dev(slots, torch.int64),
        block_tables=bt.to(device, non_blocking=True),
        logits_idx=qsl[1:] - 1,
    )
```

Three details deserve their own paragraph.

**`positions` is per-request, not per-batch.** Request B's single new token has position 631 even though it sits at flat row 512. RoPE reads `positions`; get this wrong and every request except the first gets rotary embeddings for the wrong place in its own sequence, which produces text that is fluent and subtly wrong — the worst failure mode there is.

**`slot_mapping` is the whole write path.** It is the flat physical slot for each new token, produced by `PagedSequence.append(n)`, and it is what the KV write kernel scatters into. Build it once per step and reuse it across all thirty-two layers; building it per layer means thirty-two host-to-device copies per iteration on a loop that is already host-bound.

**`logits_idx` saves more than you would guess.** Only the *last* token of each request needs logits — the other 511 prefill positions are computed so their K and V land in the cache, and their output distributions are thrown away. Slicing the hidden states before the LM head rather than after turns a $[514, 4096] \times [4096, 128256]$ GEMM into a $[3, 4096] \times [4096, 128256]$ one. For Llama-3.1-8B that is

$$
2 \times 514 \times 4096 \times 128{,}256 \approx 5.4 \times 10^{11} \text{ FLOPs}
$$

versus $3.2 \times 10^{9}$ for three rows — a factor of 171 on the single largest matmul in the model. On an RTX 4090, whose bf16 tensor-core throughput NVIDIA lists at roughly 165 TFLOP/s with FP32 accumulate, that discarded work alone is on the order of 3 ms per step. Slice early.

### The attention contract

With the flat layout, attention needs a different signature. Here is the reference implementation — correct, slow, and exactly the contract a real paged-attention kernel implements.

```python
# nanoserve/attention_varlen.py
"""Reference variable-length paged attention. Correct, not fast."""

import torch
import torch.nn.functional as F

from nanoserve.blocks import gather_kv


@torch.inference_mode()
def varlen_attention(q, batch, kv, layer: int, n_heads: int):
    """q : [T, n_heads, head_dim] queries for ALL requests, concatenated."""
    qsl = batch.query_start_loc.tolist()
    seq_lens = batch.seq_lens.tolist()
    out = torch.empty_like(q)

    for i, s in enumerate(seq_lens):
        lo, hi = qsl[i], qsl[i + 1]
        nq = hi - lo
        k, v = gather_kv(kv, layer, batch.block_tables[i], s)   # [s, n_kv, D]
        rep = n_heads // k.shape[1]                             # GQA group size
        k = k.repeat_interleave(rep, 1).permute(1, 0, 2).unsqueeze(0)
        v = v.repeat_interleave(rep, 1).permute(1, 0, 2).unsqueeze(0)
        qi = q[lo:hi].permute(1, 0, 2).unsqueeze(0)             # [1, H, nq, D]
        # The nq new tokens are the LAST nq of the s cached positions, so row r
        # may attend to columns 0 .. (s - nq + r): causal within the new tail,
        # fully open over the prefix. For a decode step nq == 1 and the mask is
        # all ones, which is why decode needs no mask at all.
        mask = torch.ones(nq, s, dtype=torch.bool, device=q.device).tril(s - nq)
        o = F.scaled_dot_product_attention(qi, k, v, attn_mask=mask)
        out[lo:hi] = o.squeeze(0).permute(1, 0, 2)
    return out
```

The `tril(s - nq)` line is the whole ragged-batch mask in one expression, and it works uniformly for a pure prefill ($nq = s$, giving a standard lower-triangular mask), a pure decode ($nq = 1$, giving all ones), and every mixed case in between. That uniformity is what makes mixing prefill and decode in one step tractable at all.

What this reference gets *wrong*, deliberately, is that it loops over requests in Python and materializes each request's K and V. A production kernel launches once for the whole flat batch, reads `query_start_loc` and `seq_lens` from device memory, and walks the block table inside the kernel. The vLLM V1 post notes that FlashAttention 3 was chosen partly because it handles this mixed prefill-and-decode dynamism in a single kernel. Writing that kernel is post 25; getting the *interface* right is this post, and the interface is now right: attention consumes offsets and a block table, never a base pointer and a stride.

---

## 5. One iteration, phase by phase

![Timeline of seven phases inside one iteration where only the forward pass runs on the GPU](/imgs/blogs/writing-a-continuous-batching-loop-4.webp)

Seven phases, and six of them are Python. That ratio is the single most important operational fact about this loop, and it is invisible in every tutorial because tutorials run one request at a time where the host cost is amortized over a huge prefill.

Let us put a floor under the one GPU phase, because it is derivable. Decode is memory-bound: at modest batch sizes the forward pass reads essentially every weight once and does very little arithmetic per byte, so the step time cannot go below the time to stream the weights out of HBM. Llama-3.1-8B in bf16 is

$$
8.03 \times 10^9 \text{ params} \times 2 \text{ bytes} = 16.06 \text{ GB}
$$

and an RTX 4090's memory bandwidth is 1,008 GB/s per NVIDIA's specification for the card. So

$$
t_{\text{step}} \ge \frac{16.06\ \text{GB}}{1{,}008\ \text{GB/s}} = 15.9\ \text{ms}
$$

which is a TPOT floor of 15.9 ms, or about 63 tokens per second at batch 1 — squarely inside the 40–60 tok/s range you should expect to actually see on that card once real kernels and launch overhead are included. That is the budget. Every microsecond your Python scheduler spends is a microsecond the GPU is not streaming weights, and 15.9 ms is not a lot of room when `build_batch` is constructing five host lists, one of which has 514 entries.

Do not guess where it goes. Measure it, and measure it correctly — host time and device time are different clocks and mixing them is how people convince themselves the GPU is the problem.

```python
# nanoserve/instrument.py
"""Per-phase timing for the step loop. Host time and device time, separately."""

import time
from collections import defaultdict

import torch


class StepTimer:
    def __init__(self):
        self.host = defaultdict(float)
        self.n = 0
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.device_ms = 0.0

    def phase(self, name):
        return _Phase(self, name)

    def gpu_begin(self):
        self._start.record()

    def gpu_end(self):
        self._end.record()
        # Do NOT synchronize here every step in production: it destroys the
        # very overlap you are trying to measure. Sample every k-th step.
        torch.cuda.synchronize()
        self.device_ms += self._start.elapsed_time(self._end)

    def report(self):
        total_host = sum(self.host.values())
        print(f"steps                 : {self.n}")
        print(f"device (CUDA events)  : {self.device_ms / self.n:8.3f} ms/step")
        print(f"host   (perf_counter) : {total_host * 1e3 / self.n:8.3f} ms/step")
        for k, v in sorted(self.host.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<20}: {v * 1e3 / self.n:8.3f} ms/step "
                  f"({100 * v / total_host:5.1f}% of host)")


class _Phase:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.timer.host[self.name] += time.perf_counter() - self.t0
```

Used like this inside `step()`:

```python
with self.timer.phase("schedule"):
    plan = self.sched.schedule()
with self.timer.phase("build_batch"):
    batch = build_batch(scheduled, plan.num_tokens, self.kv, self.model.device)
self.timer.gpu_begin()
hidden = self.model.forward(batch)
logits = self.model.lm_head(hidden[batch.logits_idx])
next_ids = self.model.sample(logits, scheduled)
self.timer.gpu_end()
with self.timer.phase("retire"):
    ...
```

Three measurement rules that this harness encodes and that most home-grown benchmarks violate:

- **Warm up before you record.** The first few steps pay for kernel autotuning, lazy module initialization and allocator growth. Discard at least twenty iterations.
- **Use CUDA events for device time and `perf_counter` for host time, and never subtract one from the other.** They measure overlapping intervals; the difference is not "overhead", it is nonsense.
- **Sample the synchronization.** `torch.cuda.synchronize()` on every step turns an asynchronous pipeline into a lockstep one and inflates your measured host cost. Synchronize every 50th step, or accept that the numbers are only valid in aggregate.

For the same reason, do not report tokens per second at batch 1 and call it a server number. Batch 1 measures how fast you can stream 16 GB of weights. A server's number is aggregate output tokens per second at a stated concurrency, with a TTFT and a TPOT distribution attached, which is what [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) is for.

---

## 6. Prefill and decode in the same iteration

![Layered breakdown of a step token budget where decode claims one token per running request and the remainder funds prefills](/imgs/blogs/writing-a-continuous-batching-loop-5.webp)

Now the consequence of `tokens_pending()` that makes the whole design pay off: because the scheduler asks only "how many tokens", a single forward pass can serve a 512-token prefill and thirty-one single-token decodes simultaneously. There is no separate prefill pass and no separate decode pass. There is a pass.

Why this matters is a hardware argument. Prefill is compute-bound: 512 tokens through a $[512, 4096] \times [4096, 4096]$ projection is a fat GEMM that saturates tensor cores. Decode is memory-bound: 31 tokens through the same projection is a skinny matrix that leaves the tensor cores mostly idle while the weights stream past. Run them in separate steps and each step is bad at the thing the other is good at. Run them together and the prefill's arithmetic rides along on memory traffic the decodes were paying for anyway. You get the prefill nearly for free in bandwidth terms, and the decodes get a step that is only slightly longer.

That "slightly" is doing real work, and it is the trade this whole area lives on. The step now takes as long as the prefill needs, and every decoding request's inter-token latency goes up by that amount. One 512-token prefill mixed in is a mild bump. One 8,000-token prefill mixed in is a visible stutter in thirty-one people's streams simultaneously. The budget in the figure — `max_batched_tokens` — is the knob that bounds the damage, and the technique that lets you admit an arbitrarily long prompt without ever exceeding the budget is chunked prefill, which splits the prompt across several steps. That is [the next post's](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) entire subject, and the change to the code above is small:

```python
# nanoserve/scheduler.py — the chunked-prefill variant of step (2)
            pending = req.tokens_pending()
            take = min(pending, budget)        # was: if pending > budget: break
            if take <= 0:
                break
            need = req.blocks_needed_for(take)
            if need > free:
                break
            ...
            out.num_tokens[req.req_id] = take  # a PARTIAL prefill is legal
```

The reason that is a two-line change rather than a redesign is that `num_tokens` was a dictionary of counts from the beginning. A request contributing 2,048 of its 8,000 prompt tokens this step is not a special case; it is a request that owes 2,048 tokens. Its `num_computed` advances by 2,048, `tokens_pending()` returns 5,952 next step, and everything downstream — `build_batch`, the mask, `slot_mapping` — already handles it. The vLLM V1 post makes exactly this argument for why the unified representation was worth the rewrite: chunked prefill, prefix caching and speculative decoding stopped being separate scheduler paths and became different token counts.

One subtlety the figure makes concrete. Decode claims its 32 tokens *first*, and only the remainder funds prefills. Reverse that order and a burst of arrivals can consume the whole budget on prefills, leaving zero for decode, and every streaming client stalls for a step. Then it happens again the next step. Serving decode first is what prevents a queue of newcomers from starving the people already being served — a scheduling policy hiding in an ordering choice, which is how most scheduling policies hide.

---

## 7. What it actually buys, derived rather than asserted

![Comparison table of static batching, continuous batching and continuous batching with chunked prefill across utilization, queue wait and long-request behavior](/imgs/blogs/writing-a-continuous-batching-loop-6.webp)

Time to earn the claim in the first paragraph. Two quantities move: **slot utilization** (what fraction of the batch's capacity does useful work) and **queue wait** (how long a request sits before it gets a slot). They move for different reasons and it is worth deriving them separately.

### Slot utilization

Let a batch of $B$ requests have output lengths $\ell_1 \dots \ell_B$. Static batching runs the rectangle for $\max_i \ell_i$ steps, so it spends $B \cdot \max_i \ell_i$ slot-steps and produces $\sum_i \ell_i$ tokens. Utilization is therefore

$$
U_{\text{static}} = \frac{\sum_i \ell_i}{B \cdot \max_i \ell_i} = \frac{\bar{\ell}}{\ell_{\max}}
$$

The mean over the max. That is the entire formula, and it is brutal, because it depends only on the *shape* of your length distribution and not at all on how good your kernels are. Under continuous batching with a non-empty queue, every slot that frees is refilled on the next iteration, so every slot-step produces a token and

$$
U_{\text{cont}} \to 1 \quad \text{(bounded only by queue depth and admission granularity)}
$$

The ratio $\ell_{\max} / \bar{\ell}$ is the multiplier, and for the eight lengths from the opening it is:

$$
\bar{\ell} = \frac{1{,}290}{8} = 161.25, \qquad
\frac{\ell_{\max}}{\bar{\ell}} = \frac{512}{161.25} = 3.18
$$

#### Worked example: 200 requests through eight slots

Take the same eight-length pattern repeated 25 times — 200 requests, 32,250 total output tokens, mean 161.25, max 512.

Static batching in groups of eight runs 25 rectangles of 512 steps each: $25 \times 8 \times 512 = 102{,}400$ slot-steps for 32,250 tokens, giving $32{,}250 / 102{,}400 = 31.5\%$ utilization. Continuous batching with the same eight slots never lets a slot idle while the queue is non-empty, so the run takes at least $\lceil 32{,}250 / 8 \rceil = 4{,}032$ steps and cannot exceed 32,250 useful slot-steps plus a short drain at the tail. Utilization lands above 99%. **Source: derived; and reproducible with the simulator below.**

Two honest caveats before anyone takes 3.18× to a capacity meeting.

**A deep queue is a precondition, not a bonus.** With only eight requests in the whole world and eight slots, continuous batching gains you nothing at all — there is nobody to refill with, and the utilization is the same 31.5%. Continuous batching converts *offered load* into throughput. At low load it converts nothing, which is correct behaviour and also why it never shows up in a single-user demo.

**Step time is not constant in batch size.** The derivation above counts slot-steps as if a step costs the same regardless of how many requests are in it. That is close to true for a memory-bound decode at modest batch, but not exactly true, because the KV cache traffic scales with the number of resident tokens. On the 4090, at batch 32 with an average of 640 cached tokens each:

$$
\text{KV bytes read} = 32 \times 640 \times 128\ \text{KiB} = 2.68\ \text{GB}
$$

against 16.06 GB of weights, so the step goes from 15.9 ms to $(16.06 + 2.68)/1008 = 18.6$ ms — an 17% tax for a 32× larger batch. That is the *good* regime, and it is exactly why batching works at all: the weight traffic is amortized and only the KV traffic is marginal. Push to batch 128 with long contexts and the KV term dominates, the step stretches, and per-request TPOT degrades even though aggregate throughput is still climbing. Deciding where on that curve to sit is what goodput is for.

### Queue wait

The second win is the one users feel. Under static batching, a request that arrives while a batch is running cannot start until that batch drains, because every slot is committed. With 512-step batches at 15.9 ms per step, the batch occupies $512 \times 15.9\ \text{ms} = 8.14$ s, and a uniformly arriving request waits a mean of half that:

$$
W_{\text{static}} = \tfrac{1}{2} \times 8.14\ \text{s} = 4.07\ \text{s}
$$

before its prefill even begins. Under continuous batching, the wait is until the *next slot frees*, not until all of them do. With $B$ slots and mean output length $\bar{\ell}$, the completion rate in steady state is $B / \bar{\ell}$ per step, so the expected wait for a free slot is $\bar{\ell} / B$ steps:

$$
W_{\text{cont}} = \frac{161.25}{8} \times 15.9\ \text{ms} = 20.2 \times 15.9\ \text{ms} = 321\ \text{ms}
$$

A factor of 12.7 on the queueing component of TTFT, from the same hardware, the same model and the same kernels. Note what this is *not*: it is not a reduction in prefill time. Your prompt still takes as long to prefill as it ever did. What vanished is the dead time before the prefill was allowed to start, and on a loaded server that dead time is usually the majority of TTFT.

#### Worked example: useful tokens per second on one RTX 4090

Combine both effects, because in practice you change two things at once — you stop wasting slots *and* you can afford more of them, since the padding tax no longer scales with the widest slot.

*Static, 8 slots* (vLLM's own latency benchmark uses batch 8 as its default, per the Anatomy post, so it is a fair baseline). Step time with 8 requests averaging 640 cached tokens: KV traffic $8 \times 640 \times 128\ \text{KiB} = 0.67$ GB, so $t = (16.06 + 0.67)/1008 = 16.6$ ms. The eight requests take 512 steps = 8.5 s and produce 1,290 real tokens: **152 useful tok/s**.

*Continuous, 64 slots.* KV traffic $64 \times 640 \times 128\ \text{KiB} = 5.37$ GB, so $t = (16.06 + 5.37)/1008 = 21.3$ ms, and every one of the 64 slot-steps is useful: ${64 / 0.0213}$ = **3,005 tok/s**.

That is 19.8× — decomposing into roughly 3.18× from utilization and roughly 6.2× from being able to run a wider batch without the padding tax. **Source: derived** from the bandwidth model and the stated length distribution; it is an upper bound that ignores kernel inefficiency, host overhead and the fact that 64 concurrent requests of 640 tokens need $64 \times 640 / 16 = 2{,}560$ blocks out of the 3,430 the 4090's KV budget affords. Treat it as the shape of the win, not a promise.

### Reproduce it yourself

You do not need a GPU to check the utilization half. This simulator counts slot-steps under both policies on any length distribution you hand it, in pure Python.

```python
# nanoserve/sim_batching.py — pure Python, no torch, no GPU
"""Count useful and wasted slot-steps under static vs continuous batching."""

from __future__ import annotations


def static_batching(lengths: list[int], batch_size: int):
    used = total = 0
    for i in range(0, len(lengths), batch_size):
        chunk = lengths[i:i + batch_size]
        steps = max(chunk)
        used += sum(chunk)
        total += steps * len(chunk)
    return used, total


def continuous_batching(lengths: list[int], num_slots: int):
    """Every freed slot is refilled on the next iteration."""
    pending = list(lengths)
    running: list[int] = []
    used = total = steps = 0
    while pending or running:
        while pending and len(running) < num_slots:
            running.append(pending.pop(0))
        steps += 1
        used += len(running)
        total += num_slots            # capacity offered this step
        running = [r - 1 for r in running]
        running = [r for r in running if r > 0]
    return used, total, steps


def main():
    lengths = [24, 41, 63, 88, 112, 190, 260, 512] * 25   # 200 requests
    n, slots = len(lengths), 8
    su, st = static_batching(lengths, slots)
    cu, ct, csteps = continuous_batching(lengths, slots)
    print(f"requests            : {n}")
    print(f"output tokens       : {sum(lengths):,}")
    print(f"mean / max length   : {sum(lengths)/n:.2f} / {max(lengths)}")
    print(f"static  slot-steps  : {st:,} used {su:,}  -> {100*su/st:.1f}%")
    print(f"cont.   slot-steps  : {ct:,} used {cu:,}  -> {100*cu/ct:.1f}%")
    print(f"cont.   steps        : {csteps:,} (lower bound {-(-su//slots):,})")


if __name__ == "__main__":
    main()
```

The static line is fully determined by arithmetic and will print `102,400 used 32,250 -> 31.5%` every time. The continuous line will print a utilization above 99% and a step count a little above the 4,032-step lower bound, with the excess coming entirely from the tail drain where fewer than eight requests remain. Change `lengths` to your own production distribution and the number it prints is the *maximum* throughput multiple continuous batching can give you on that workload. If it prints 92%, continuous batching is not your bottleneck and you should go read a profile instead.

| Quantity | Value | Source |
| --- | --- | --- |
| Llama-3.1-8B weights, bf16 | 16.06 GB | derived: 8.03B params × 2 B |
| RTX 4090 memory bandwidth | 1,008 GB/s | cited: NVIDIA GeForce RTX 4090 specification |
| Decode step floor, small batch | 15.9 ms (63 tok/s) | derived: 16.06 GB ÷ 1,008 GB/s |
| KV bytes per token, Llama-3.1-8B | 128 KiB | derived: 2·32·8·128·2 B |
| KV blocks on a 4090, 6.7 GiB budget | 3,430 (54,880 tokens) | derived (see the paged-cache post) |
| Static slot utilization, this workload | 31.5% | derived: mean ÷ max |
| Continuous slot utilization, deep queue | above 99% | reproduce: `sim_batching.py` |
| Mean queue wait, static batch of 8 | 4.07 s | derived: half of 512 × 15.9 ms |
| Mean queue wait, continuous, 8 slots | 321 ms | derived: (161.25 ÷ 8) × 15.9 ms |
| Discarded LM-head FLOPs without `logits_idx` | 5.4 × 10¹¹ per step | derived: 2·514·4096·128,256 |
| PagedAttention vs prior serving systems | 2–4× throughput | cited: vLLM paper, arXiv:2309.06180 |
| vLLM V0 → V1 throughput | up to 1.7× | cited: vLLM V1 blog, Llama 3.1 8B / 3.3 70B, ShareGPT |
| Model Runner V2, host-overhead-bound case | +56% (25K vs 16K out tok/s) | cited: vLLM MRv2 blog, Qwen3-0.6B, 1×GB200 |

---

## 8. When the scheduler becomes the bottleneck

Section 5 put a 15.9 ms floor under the forward pass. Now flip the question: what happens when the host work per iteration approaches that number?

The loop as written is strictly serial. Python schedules, Python builds five lists, PyTorch copies them to the device, the GPU runs, the sampler reads a token id back to the host, and only then does the next iteration's Python start. Every host microsecond is GPU idle time. The costs that grow are the ones proportional to the running set: `schedule()` iterates over every running request, `build_batch` builds a `slot_mapping` entry per new token and a block-table row per request, and the block-table tensor is rebuilt from Python lists every step. At batch 8 none of that matters. At batch 256 with long contexts, the block table alone is 256 rows of up to a few hundred `int32`s, constructed by a Python loop, every 16 ms.

Two things make it much worse. **Small models**, where the GPU's share of the iteration shrinks but the host's does not: a 0.6B model's forward pass might be a millisecond, and suddenly a two-millisecond scheduler is doubling your step time. **Fast GPUs**, for the same reason from the other direction.

The production answers are three, and they are worth knowing even if nanoserve implements none of them yet.

**Stop rebuilding what did not change.** vLLM's Model Runner V2 introduces a *decoupled persistent batch*: each live request holds a fixed row in a state table for its whole lifetime, independent of its position in this step's ordering, and the per-step work becomes a gather that assembles ordered block tables rather than a from-scratch reconstruction. Our `running` list already preserves order across steps, which is the precondition; turning it into stable row indices with a free-slot list is the change. The same write-up describes building `input_ids`, `positions`, `query_start_loc` and `seq_lens` directly on the GPU with Triton kernels, which removes the host-to-device copy from the critical path entirely.

**Overlap the scheduler with the forward pass.** MRv2 calls this zero-sync asynchronous scheduling: the CPU schedules step $N+1$ while the GPU is still executing step $N$, and sampled token ids are copied back on a separate CUDA stream so the host never has to block on the device to decide what runs next. The catch is that step $N+1$ must be scheduled *before* step $N$'s tokens are known, so the scheduler cannot see which requests just hit EOS — the engine has to speculate that everyone continues and correct afterwards. The vLLM team reports this design delivering **+56% throughput (25K versus 16K output tokens per second) on Qwen3-0.6B on a single GB200 in a host-overhead stress test**, and a smaller −6.3% TPOT on GLM-4.7-FP8 across 4×GB200 — the gap between those two numbers is precisely the gap between host-bound and GPU-bound, and it tells you when this optimization is worth its complexity.

**Stop launching kernels one at a time.** Even with a free scheduler, a decode step launches hundreds of tiny kernels, and at a few microseconds of launch overhead each that is milliseconds of pure dispatch. CUDA graphs capture the whole launch sequence once and replay it as a single submission. That interacts awkwardly with everything this post built — the shapes change every step and the block-table pointers move — which is why real engines capture per batch-size bucket and pad up to it. That is a Track E and Track F subject; [the model-serving deep dive on continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) sketches how the pieces fit in vLLM specifically.

It is worth being precise about the 1.7× figure from the V1 rewrite, because it is easy to misattribute. vLLM V0 already did continuous batching. V1's reported *up to 1.7× throughput improvement over V0 on Llama 3.1 8B and 3.3 70B with ShareGPT* came from isolating the engine core in its own process, unifying the scheduler around the token-count representation, and cutting CPU overhead — not from introducing the loop this post writes. The loop is table stakes. The 1.7× is what you get for making the loop cheap.

---

## 9. Stress tests, and the failure modes they expose

![Decision tree showing five outcomes for one admission attempt including whole admission chunking deferral preemption and rejection](/imgs/blogs/writing-a-continuous-batching-loop-7.webp)

A loop that works on a benign workload is not evidence of anything. Here are the four stresses that break naive continuous batching, in the order you will hit them.

### Stress 1: one very long generation among many short ones

Admit 63 requests that generate about 100 tokens each and one that generates 4,000. The long one holds its slot for 4,000 steps, which is exactly correct and exactly what you want — under static batching it would have held *all 64 slots* for 4,000 steps. This is the stress continuous batching passes trivially, and it is worth running precisely so you can see it pass.

What it does expose is memory. The long request's block table grows to $\lceil 4000/16 \rceil = 250$ blocks, and grows by one block every sixteenth step forever. If the pool is tight, that slow drip is what eventually triggers the first `OutOfBlocks`, and it will trigger it during a step where nothing unusual is happening, which makes it maddening to reproduce. Log free-block count as a time series, not just at the failure.

### Stress 2: a burst that exceeds free blocks

Fire 200 requests with 2,000-token prompts at a pool of 3,430 blocks. Each needs 125 blocks, so 27 fit and the rest queue. Our scheduler handles this correctly-ish: the `need > free` check breaks the admission loop and the queue grows. Two things then go wrong.

First, running requests that need to grow a block are *stalled* rather than preempted — they keep their memory, make no progress, and their clients see a stream that has simply stopped. The metric `plan.stalled` counts it; the fix is to evict somebody, which is [eviction, preemption and KV swapping](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping).

Second, nothing anywhere pushes back on the client. The queue grows without bound, every request's TTFT grows with it, and the system passes smoothly from "slow" to "useless" without a single error being returned. Bounding the queue and rejecting past the bound is admission control, and the queueing-theory picture of why the transition is so abrupt is post 15's subject.

### Stress 3: the prompt that never gets admitted

Set `max_batched_tokens = 2048` and submit a 4,000-token prompt. Trace the scheduler: `pending = 4000 > budget`, break, forever. The request sits at the head of the queue immortally, and because it is at the *head*, a strict FIFO policy will not admit anything behind it either. One oversized prompt has deadlocked the server.

This is the single best argument for chunked prefill, and it is why production engines do not treat chunking as an optimization. It is a correctness property. Until you implement it, at minimum reject at submission time anything larger than the budget, with a clear error, rather than queueing something you can never serve.

### Stress 4: fairness and starvation

Mix a stream of short chat requests with occasional long RAG requests, run FIFO, and watch. FIFO is not obviously unfair, but it has a specific pathology: a request already running is never displaced, so under saturation the running set becomes whoever happened to be admitted before the load arrived, and everybody else waits behind them in arrival order. Long generations therefore get disproportionate service simply by occupying a slot for longer, and short requests — the ones with the tightest latency expectations — pay for it.

Fixing this means the scheduler needs an actual policy: priorities, fair-share, deadline-awareness, or preemption of long runners in favour of short ones. It also means the objective changes from throughput to **goodput** — tokens delivered inside their latency SLO rather than tokens delivered. That reframing is post 14's, and it is worth internalizing early: an engine tuned for raw throughput will happily produce more tokens that nobody is still waiting for. The model-serving repo's [request scheduling and preemption post](/blog/machine-learning/model-serving/request-scheduling-and-preemption) covers what the mature policies look like.

| Symptom | Immediate cause | Where it is fixed |
| --- | --- | --- |
| Stream stops mid-generation, no error | Running request stalled on a missing block | preemption, post 10 |
| A large prompt never starts | `pending > budget` with whole-prefill admission | chunked prefill, post 13 |
| TTFT climbs without bound under load | Unbounded waiting queue, no backpressure | admission control, post 15 |
| Short requests starve behind long ones | FIFO plus non-displaceable running set | scheduler policy, post 14 |
| Output subtly degrades on long contexts | `seq_lens` confused with query length | this post, section 4 |
| Fluent but wrong text after batching lands | `positions` computed per-batch not per-request | this post, section 4 |
| GPU utilization falls as batch grows | Host-side scheduler and tensor build dominate | async scheduling, section 8 |

---

## 10. Case studies and public numbers

Four data points worth knowing, each cited, each with its setup, because a throughput multiple without a workload attached is marketing.

**Orca (Yu et al., OSDI 2022)** introduced iteration-level scheduling and selective batching, and is the origin of everything in this post. Its reported gains over request-level batching at matched latency are more than an order of magnitude on the workloads it studies. Read it for the selective-batching detail: Orca observes that not every operator in a transformer needs the same batching treatment, which is a subtlety our flat-array design sidesteps by not having a batch dimension at all.

**The vLLM paper (Kwon et al., SOSP 2023, arXiv:2309.06180)** reports 2–4× throughput improvements over prior serving systems at the same latency, attributing them to PagedAttention's near-zero KV waste enabling larger effective batch sizes. Note that the baselines there already did continuous batching — the 2–4× is what *paged memory* adds on top of the loop this post wrote, which is why [the paged KV cache post](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) had to come first.

**vLLM V1** ([announcement](https://vllm.ai/blog/2025-01-27-v1-alpha-release), 2025-01-27) reports **up to 1.7× throughput versus V0 on Llama 3.1 8B and Llama 3.3 70B with ShareGPT traces**. The mechanisms named are the unified `{request_id: num_tokens}` scheduler representation, an isolated engine-core process communicating over ZeroMQ, and overlapped tokenization and detokenization. The same post reports "zero-overhead" prefix caching with under 1% throughput cost even at a 0% hit rate. Its stated limitations at release are equally instructive: Ampere-or-newer NVIDIA only, and no Mamba, encoder-decoder, LoRA, pipeline parallelism or structured decoding at that time.

**Model Runner V2** ([write-up](https://vllm.ai/blog/2026-03-24-mrv2), 2026-03-24) reports **+56% throughput, 25K versus 16K output tokens per second, on Qwen3-0.6B on a single GB200** in a deliberately host-overhead-bound configuration, and a **−6.3% TPOT on GLM-4.7-FP8 across 4×GB200 with MTP=1**. The 40× difference between those two improvements is the lesson: the size of the async-scheduling win is a direct readout of how host-bound your configuration already was. It is enabled with `VLLM_USE_V2_MODEL_RUNNER=1` and as of v0.18.0 excludes linear-attention models, LoRA and logits processors.

**The Anatomy post** ([Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05) is the best free description of the mature structure: waiting and running queues, a KV-cache manager holding a `free_block_queue` of hundreds of thousands of 16-token blocks depending on VRAM, a `req_to_blocks` map from request id to block list, and the flattening of all scheduled sequences into a single concatenated super sequence kept separate by position indices and masking. It also pins down the metric definitions this series uses — TTFT as submit to first token, ITL as the gap between consecutive tokens, TPOT as the mean ITL over an output — which is worth adopting verbatim so your numbers are comparable to everyone else's.

---

## 11. When to reach for this, and when not to

Write this loop yourself when you are learning how engines work, when you are building an engine, when you need scheduling behaviour nothing off the shelf offers, or when your model is unusual enough that vLLM does not support it. Those are real cases and they are why this series exists.

Do **not** write it yourself for production if all four of these hold: your model is a mainstream architecture, your hardware is mainstream, your workload is ordinary chat or RAG, and you have fewer than a few engineers to spend on inference. Install vLLM or SGLang or TGI. The loop in section 3 is sixty lines; the *policies* that make it survive a real load — preemption without thrashing, chunked prefill tuned against TTFT, prefix caching with correct eviction, CUDA graph capture across batch buckets, guarded fallbacks when a kernel does not support your head dimension — are tens of thousands, and every one of them is a bug you will find in production rather than in a test.

There are also workloads where continuous batching is the wrong shape entirely:

- **Offline batch scoring with uniform lengths.** If every request has the same output length, $\bar{\ell} = \ell_{\max}$ and static batching is already at 100% utilization. Continuous batching adds host overhead for nothing.
- **Single-user local inference.** With one request there is no set to churn. `llama.cpp` on a laptop is not leaving throughput on the table by not doing this.
- **Extremely strict per-request latency with reserved capacity.** If a request must have deterministic TPOT, sharing a step with a variable set of neighbours is precisely what you cannot allow. Isolation costs utilization; sometimes you buy it deliberately.
- **Very small models on very fast GPUs**, where the Python loop costs more than the forward pass. There, section 8's techniques are not an optimization but a prerequisite, and if you are not prepared to implement them you should use an engine that has.

The capstone, [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), puts this decision alongside the others in the series so you can see which knobs interact.

---

## Key takeaways

1. **A serving batch is a mutable set, not a rectangle.** Membership changes every iteration; there is no "start of a batch" and no "end of a batch", only steps.
2. **Utilization under static batching is exactly $\bar{\ell} / \ell_{\max}$.** It depends on your length distribution and nothing else, which means you can compute your ceiling before writing a line of code.
3. **Retire and free blocks in the same iteration the request finishes.** Deferring the release by one step costs you resident capacity permanently and shows up in no profile.
4. **Represent the plan as request id to token count.** That single choice erases the prefill/decode distinction and makes chunked prefill a two-line change instead of a second code path.
5. **Flatten, do not pad.** One concatenated query array plus `query_start_loc` and `seq_lens`; and never confuse the two, because the failure mode is silent quality loss rather than a crash.
6. **Compute logits only for the last token of each request.** Slicing hidden states before the LM head instead of after removes the largest matmul in a prefill-heavy step.
7. **Serve decode before admitting prefills.** Reversing the order lets a burst of newcomers stall everyone already streaming.
8. **Continuous batching converts offered load into throughput.** With no queue it buys nothing, which is why it never shows up in a single-user benchmark.
9. **Measure host and device time on separate clocks**, warm up first, and never quote batch-1 tokens per second as a server number.
10. **The loop is the easy part.** Preemption, chunking, admission control and fairness are what turn it into an engine, and each of them exists because this loop, unmodified, fails in a specific way you can reproduce.

---

## Further reading

- Yu et al., [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) — the origin of iteration-level scheduling and selective batching.
- Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (SOSP 2023) — the vLLM paper; read sections 4 and 5 alongside this post's section 4.
- vLLM, [*Inside vLLM: Anatomy of a High-Throughput Inference System*](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — waiting and running queues, the KV manager, the super sequence, and the metric definitions.
- vLLM, [*vLLM V1: A Major Upgrade to vLLM's Core Architecture*](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the unified token-count scheduler and where the 1.7× came from.
- vLLM, [*Model Runner V2*](https://vllm.ai/blog/2026-03-24-mrv2) — persistent batch state, GPU-side input construction, and zero-sync async scheduling.
- Within this series: [static batching and the padding tax](/blog/machine-learning/inference-engineering/static-batching-and-the-padding-tax) for the baseline this post beats, [paged KV cache](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table) for the allocator underneath it, and [chunked prefill and the TTFT/TPOT trade-off](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) for the very next thing this loop needs.
