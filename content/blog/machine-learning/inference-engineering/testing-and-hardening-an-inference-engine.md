---
title: "Testing and hardening an inference engine: Make every token explainable"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "Build a layered correctness and chaos-testing harness for nanoserve so logits, cache state, shapes, traces, and distributed failures become reproducible evidence."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "testing",
    "reproducibility",
    "pytorch",
    "cuda",
    "distributed-inference",
    "ml-systems",
    "latency",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 29
---

The dangerous inference bug is rarely a crash. A crash gives you a stack trace, a request id, and permission to stop the world. The expensive bug is the one that returns a plausible answer, leaves the GPU busy, and changes one token only when a request lands on a block boundary at batch size three.

That is why this post is about more than unit tests. We are going to make `nanoserve` produce evidence for five different questions: did the numerical implementation preserve the reference logits, did the decode state transition write exactly the slots it claims, do awkward shapes survive, can a failing rank stop a collective cleanly, and can a future engineer replay the failure without the original machine? The first figure is the mental model: a correctness envelope, from cheap local checks to expensive deliberate faults.

![A layered correctness envelope moves from unit checks through parity, state, fuzz, chaos, and canary evidence](/imgs/blogs/testing-and-hardening-an-inference-engine-1.webp)

The diagram above is the mental model: confidence is a ladder, not a switch. A green end-to-end response does not prove that the cache is correct, and a passing cache test does not prove that a tensor-parallel communicator can recover. Every layer owns a failure class and leaves an artifact that the next layer can inspect.

This post writes `nanoserve/tests/` rather than another optimization. It adds a reference harness, golden trace format, cache invariants, shape generators, a failure-injection seam, and a CI policy. The code is intentionally small enough to paste into the toy engine and strict enough to expose the places where a production engine eventually needs a much larger test matrix.

## 1. Correctness is a vector, not a boolean

The first design mistake is to ask whether an inference engine is “correct” as if it had one answer. Correctness has coordinates.

| Coordinate | Question | Typical evidence | Source |
| --- | --- | --- | --- |
| Numerical | Are tensors close to a trusted implementation? | Logit and intermediate parity | `reproduce: tests/test_parity.py` |
| Semantic | Does the selected token obey the sampler contract? | Golden token ids and stop behavior | `reproduce: tests/test_trace.py` |
| State | Does every logical token own exactly one physical slot? | Cache ledger assertions | `derived: one append increments one owner` |
| Shape | Do ragged batches and boundary lengths preserve invariants? | Parametrized fuzz cases | `reproduce: tests/test_shapes.py` |
| Distributed | Does a failed rank become a bounded error? | Fault-injection log and restart outcome | `reproduce: tests/test_chaos.py` |
| Operational | Can another machine repeat the verdict? | Locked manifest, seed, image digest | `derived: same inputs + same seed define the replay key` |

These coordinates are related but not interchangeable. A greedy token can match while one logit row is wrong because the winning logit had a large margin. A cache can contain the right values while its free-list metadata says a slot is available twice. A test can pass on a square tensor while an empty sequence or a one-token sequence triggers an out-of-bounds access.

The rule I use is simple: every optimization gets a reference boundary before it gets a speed claim. A fused RMSNorm compares to unfused PyTorch. A paged attention path compares to a dense attention path on the same small tensor. A scheduler compares the emitted request trace, not merely the final text. A rank-recovery path compares the request lifecycle: complete, retry, or explicit failure.

### The smallest useful failure report

“Expected tensor mismatch” is not enough for an inference test. A useful report carries the model revision, tokenizer revision, input ids, dtype, device, batch shape, position ids, random seed, sampling parameters, cache layout, and the first coordinate that diverged. If a test fails only on an A100 80GB SXM but not an RTX 4090 24GB, the device name and kernel path belong in the artifact.

The report should also distinguish a numerical tolerance failure from a semantic failure. PyTorch’s [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html) defines closeness as an absolute tolerance plus a relative tolerance and documents dtype-specific defaults; that is a useful primitive, not a complete inference contract. For a logit tensor, we should record maximum absolute error, maximum relative error, mismatch count, and argmax disagreement count. For a token stream, we should record the first differing position and the surrounding top-k candidates.

## 2. Start with logit parity, not text parity

Text is a lossy assertion. If two distributions select the same argmax, their tails may still be badly wrong. If sampling is enabled, two correct implementations can select different tokens because their random draws occur in a different order. The comparison boundary should therefore be logits before sampling, then tokens under a separate deterministic sampler test.

![A prompt branches into independent Transformers and nanoserve paths before their logits converge at a parity verdict](/imgs/blogs/testing-and-hardening-an-inference-engine-2.webp)

The parity harness feeds the same integer ids, attention mask, position ids, dtype, and seed to both paths. It does not compare a string produced by two tokenizers. Tokenization belongs in its own test because a chat template difference can masquerade as a model error.

### A runnable parity helper

The following diff is deliberately explicit. The reference callable and candidate callable have the same signature, but they do not share the candidate’s internal modules. That independence matters: a test that calls the same helper twice can preserve the same bug twice.

```python
# nanoserve/tests/parity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ParityReport:
    max_abs: float
    max_rel: float
    argmax_mismatches: int
    first_mismatch: tuple[int, ...] | None


def compare_logits(
    reference: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    *,
    rtol: float = 2e-2,
    atol: float = 2e-3,
) -> ParityReport:
    with torch.inference_mode():
        expected = reference(input_ids).float()
        actual = candidate(input_ids).float()

    if expected.shape != actual.shape:
        raise AssertionError(f"shape mismatch: {expected.shape} != {actual.shape}")

    delta = (actual - expected).abs()
    scale = expected.abs().clamp_min(atol)
    relative = delta / scale
    close = delta <= (atol + rtol * expected.abs())
    mismatches = (~close).nonzero(as_tuple=False)
    argmax_mismatches = (actual.argmax(-1) != expected.argmax(-1)).sum().item()
    first = tuple(mismatches[0].tolist()) if mismatches.numel() else None

    report = ParityReport(
        max_abs=float(delta.max().item()),
        max_rel=float(relative.max().item()),
        argmax_mismatches=int(argmax_mismatches),
        first_mismatch=first,
    )
    if first is not None:
        raise AssertionError(report)
    return report
```

The tolerances above are an example policy for comparing a bf16 candidate after promoting both outputs to fp32. They are not a universal promise about Llama-3.1-8B. The test should calibrate them against a small, fixed reference corpus and review any relaxation as a code change. A fused kernel that needs a tolerance ten times larger deserves an explanation: accumulation order, approximate math, quantization, or a bug.

For exact greedy decoding, add a second assertion at the boundary where sampling begins:

```python
# nanoserve/tests/test_parity.py
import torch
from nanoserve.tests.parity import compare_logits


def test_forward_matches_reference(reference_model, nanoserve_model):
    ids = torch.tensor([[1, 314, 159, 265, 358]], dtype=torch.long)
    report = compare_logits(
        reference_model,
        nanoserve_model,
        ids,
        rtol=2e-2,
        atol=2e-3,
    )
    assert report.argmax_mismatches == 0
```

The literal ids are a fixture, not a claim about a tokenizer. A real repository should serialize the ids generated from the pinned tokenizer revision. Keep at least one hand-written tensor fixture too: it makes the test independent of a tokenizer package outage.

### What parity does and does not prove

Parity proves a relationship at a chosen boundary. It does not prove that a model is good, that a quantized model has acceptable task quality, or that the engine’s scheduler is fair. It also does not require bitwise equality across devices. The vLLM issue tracker has documented cases where output can differ across GPUs even at temperature zero, which is why comparing logits or log probabilities is often more informative than demanding identical sampled text; see the [vLLM GPU-difference discussion](https://github.com/vllm-project/vllm/issues/11526), published 2025-03-06 and crawled 2026-08-04.

For a quantized path, the reference is not expected to be numerically identical. The contract changes to “within the quality budget” and should be checked with perplexity plus task-specific evaluations. That is the same distinction this series made when loading weight-only quantization: a loader can be structurally correct while the representation is intentionally lossy.

## 3. Golden traces turn a moving decode loop into a replay

A final string hides the journey. A token can be wrong because the position id was incremented before the cache write, because the block table pointed at a recycled block, because EOS was masked too early, or because the sampler consumed a random number on a rejected candidate. The trace must preserve enough intermediate state to locate the first wrong transition.

![A golden trace records seed and ids, a decode step, cache write, scores, and the selected token as one replayable timeline](/imgs/blogs/testing-and-hardening-an-inference-engine-3.webp)

The trace should not dump every activation from every layer. That would be expensive, brittle, and hard to review. Record boundaries: request metadata, token ids, position, logical length, physical block ids, cache write coordinates, a digest of the newly written K/V slice, top-k logits, selected token, and allocator counters. A digest detects mutation without making the fixture depend on a particular tensor serialization format.

### The trace schema

```python
# nanoserve/tests/trace.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:24]


@dataclass
class DecodeEvent:
    step: int
    token_in: int
    position: int
    logical_length: int
    physical_blocks: list[int]
    write_offset: int
    kv_digest: str
    top_ids: list[int]
    top_values: list[float]
    token_out: int


def save_trace(events: list[DecodeEvent], path: Path, *, seed: int, model_id: str) -> None:
    payload = {
        "schema": 1,
        "seed": seed,
        "model_id": model_id,
        "events": [asdict(event) for event in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_trace(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError(f"unsupported trace schema: {payload.get('schema')}")
    return payload
```

Replay is a function, not a screenshot. It loads the fixture, sets the seed, runs exactly the recorded input sequence, and fails at the first event whose structural fields or numeric summaries differ. Store the trace beside the test with a model revision and tokenizer revision; do not silently regenerate it after a failure.

```python
# nanoserve/tests/test_trace.py
from pathlib import Path

from nanoserve.tests.trace import load_trace


def test_golden_decode_trace(engine_factory, trace_runner):
    expected = load_trace(Path("tests/golden/llama31-8b-greedy.json"))
    actual = trace_runner(engine_factory(), seed=expected["seed"])

    for index, (want, got) in enumerate(zip(expected["events"], actual)):
        assert got["step"] == want["step"], index
        assert got["position"] == want["position"], index
        assert got["physical_blocks"] == want["physical_blocks"], index
        assert got["write_offset"] == want["write_offset"], index
        assert got["kv_digest"] == want["kv_digest"], index
        assert got["top_ids"] == want["top_ids"], index
        assert got["token_out"] == want["token_out"], index

    assert len(actual) == len(expected["events"])
```

Do not compare floating-point top values with string equality in a cross-device fixture. Compare the ids exactly and compare values with an explicit tolerance. The ids are the semantic boundary; the values explain a near miss.

### Golden is not immutable forever

A golden trace is a contract. If a deliberate kernel change alters accumulation order but preserves the documented numerical and semantic budget, update the fixture in a review that explains why. If the fixture changes because someone ran a formatter, that is a process failure. Put the old and new trace summaries in the pull request and require a human to approve the contract change.

## 4. Cache consistency: assert conservation after every transition

The KV cache is where an inference engine stops being a pure function. The output depends on a mutable allocator, a block table, logical lengths, copy-on-write rules, and reclamation. The strongest tests are not “the answer looks right after ten tokens.” They are conservation laws checked after each append, fork, finish, preempt, and rollback.

![The cache comparison shows an unchecked length and slot mismatch becoming an asserted one-to-one ownership transition](/imgs/blogs/testing-and-hardening-an-inference-engine-4.webp)

For each request, define a logical length $S$, a block size $B$, and a block table with $\lceil S/B \rceil$ entries. The minimum derived invariant is:

$$
\text{allocated\_slots}(r) = S_r,
\qquad
\text{table\_blocks}(r) = \left\lceil \frac{S_r}{B} \right\rceil.
$$

That equality is only valid for a request that owns every token privately. Prefix sharing changes ownership: shared slots have a reference count, and the private suffix must still satisfy the equality. Copy-on-write must increase the number of physical blocks before a write, not after the write.

### A reference ledger

```python
# nanoserve/tests/cache_invariants.py
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RequestState:
    request_id: str
    logical_length: int
    block_table: list[int]


def assert_cache_consistent(
    state: RequestState,
    *,
    block_size: int,
    owner_of: dict[int, str | None],
    free_blocks: set[int],
) -> None:
    expected_blocks = math.ceil(state.logical_length / block_size)
    assert len(state.block_table) == expected_blocks, (
        state.request_id,
        state.logical_length,
        state.block_table,
    )
    assert len(set(state.block_table)) == len(state.block_table)

    for physical in state.block_table:
        assert physical not in free_blocks
        assert owner_of.get(physical) == state.request_id

    owned = {
        block for block, owner in owner_of.items() if owner == state.request_id
    }
    assert owned == set(state.block_table)

    assert free_blocks.isdisjoint(owned)
```

In a shared-prefix allocator, replace the single owner string with an owner set or reference count and assert that the count agrees with the number of request tables containing the block. Also assert the inverse: every free block is absent from every request table. This catches the particularly nasty bug where the active request is correct but a stale table still points at a block that has already been handed to another request.

### The animated invariant

<figure class="blog-anim">
<svg viewBox="0 0 760 190" role="img" aria-label="Decode cache fills one slot at a time, then rolls back exactly when the draft is rejected" style="width:100%;height:auto;max-width:820px">
<style>
.hc-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.hc-live{fill:var(--accent,#6366f1);opacity:.9}.hc-rollback{fill:#ef4444;opacity:.85}.hc-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.hc-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}@keyframes hc-fill{0%,12%{transform:translateX(0);opacity:0}18%,68%{transform:translateX(0);opacity:1}75%,100%{transform:translateX(0);opacity:0}}@keyframes hc-fill2{0%,24%{transform:translateX(0);opacity:0}30%,68%{transform:translateX(0);opacity:1}75%,100%{transform:translateX(0);opacity:0}}@keyframes hc-fill3{0%,36%{transform:translateX(0);opacity:0}42%,68%{transform:translateX(0);opacity:1}75%,100%{transform:translateX(0);opacity:0}}@keyframes hc-fill4{0%,48%{transform:translateX(0);opacity:0}54%,68%{transform:translateX(0);opacity:1}75%,100%{transform:translateX(0);opacity:0}}@keyframes hc-reject{0%,68%{opacity:0}75%,88%{opacity:1}100%{opacity:0}}@keyframes hc-clear{0%,68%{opacity:0}75%,88%{opacity:1}100%{opacity:0}}.hc-a{animation:hc-fill 8s linear infinite}.hc-b{animation:hc-fill2 8s linear infinite}.hc-c{animation:hc-fill3 8s linear infinite}.hc-d{animation:hc-fill4 8s linear infinite}.hc-r{animation:hc-reject 8s linear infinite}.hc-cursor{animation:hc-clear 8s linear infinite}@media (prefers-reduced-motion:reduce){.hc-a,.hc-b,.hc-c,.hc-d,.hc-r,.hc-cursor{animation:none}.hc-a,.hc-b,.hc-c,.hc-d{opacity:1}.hc-r,.hc-cursor{opacity:.85}}
</style>
<text class="hc-lbl" x="380" y="25">golden decode trace: append, assert, reject, rollback</text><rect class="hc-slot" x="100" y="70" width="92" height="62" rx="8"/><rect class="hc-slot" x="210" y="70" width="92" height="62" rx="8"/><rect class="hc-slot" x="320" y="70" width="92" height="62" rx="8"/><rect class="hc-slot" x="430" y="70" width="92" height="62" rx="8"/><rect class="hc-slot" x="540" y="70" width="92" height="62" rx="8"/><rect class="hc-live hc-a" x="100" y="70" width="92" height="62" rx="8"/><rect class="hc-live hc-b" x="210" y="70" width="92" height="62" rx="8"/><rect class="hc-live hc-c" x="320" y="70" width="92" height="62" rx="8"/><rect class="hc-live hc-d" x="430" y="70" width="92" height="62" rx="8"/><rect class="hc-rollback hc-r" x="430" y="70" width="92" height="62" rx="8"/><text class="hc-lbl" x="146" y="105">t</text><text class="hc-lbl" x="256" y="105">t+1</text><text class="hc-lbl" x="366" y="105">t+2</text><text class="hc-lbl" x="476" y="105">draft</text><text class="hc-lbl" x="586" y="105">free</text><text class="hc-note" x="380" y="165">the red slot is not merely hidden: the allocator returns its ownership</text>
</svg>
<figcaption>Motion carries the invariant: a rejected speculative step must remove precisely the state it appended.</figcaption>
</figure>

The motion matters because rollback is not “subtract one from the length.” It is a transaction over logical length, block-table entries, physical ownership, and any recurrent or auxiliary state. The same principle applies when `nanoserve` later adds speculative decoding: an accepted prefix commits; a rejected suffix must disappear from every state representation.

### Assert at the mutation boundary

Call the invariant immediately after mutation in debug and test builds. Do not wait until request completion, when the corrupting operation may be hundreds of steps behind the visible failure.

```python
# nanoserve/cache.py — testable mutation boundary
class PagedCache:
    def append(self, request_id: str, key, value) -> None:
        state = self.requests[request_id]
        physical, offset = self._reserve_slot(state)
        self.keys[physical, offset].copy_(key)
        self.values[physical, offset].copy_(value)
        state.logical_length += 1
        self._assert_if_enabled(state)

    def rollback(self, request_id: str, count: int) -> None:
        state = self.requests[request_id]
        if count < 0 or count > state.logical_length:
            raise ValueError(f"invalid rollback {count} for {state.logical_length}")
        for _ in range(count):
            physical = self._last_physical_slot(state)
            self._release_slot(state, physical)
            state.logical_length -= 1
        self._assert_if_enabled(state)

    def _assert_if_enabled(self, state) -> None:
        if self.debug_invariants:
            from nanoserve.tests.cache_invariants import assert_cache_consistent
            assert_cache_consistent(
                state,
                block_size=self.block_size,
                owner_of=self.owner_of,
                free_blocks=self.free_blocks,
            )
```

A production build may turn the Python assertion off, but it should retain cheap device-side checks where practical: bounds checks, a generation number on each block, and a request id in debug metadata. The point is not to make every request pay the full test cost. The point is to make the cost a deliberate configuration choice rather than an accidental absence of evidence.

#### Worked example: one block boundary

Take block size $B=16$ and a request with $S=17$ tokens. The derived block count is $\lceil 17/16 \rceil=2$. If a buggy append increments $S$ from 16 to 17 but forgets to allocate the second block, the model may still read the first 16 valid positions and produce plausible text. The invariant fails immediately because `len(block_table)=1` while the derived requirement is 2. No GPU benchmark is needed; the arithmetic is the source.

## 5. Shape fuzzing finds the assumptions in the corners

Most inference tests are shaped like the developer’s favorite request: batch one, a comfortable sequence length, a full block, and one decode step. Production traffic is not shaped like that. It has empty queues, one-token prompts, ragged sequences, EOS at the first generated token, a vocabulary-sized logits row, and a final block with one live slot.

![A shape matrix crosses batch, prefill, decode, cache, and sampling cases to expose boundary assumptions](/imgs/blogs/testing-and-hardening-an-inference-engine-5.webp)

Fuzzing here does not mean generating meaningless random tensors until CUDA crashes. It means generating small, interpretable cases from the engine’s legal domain, then checking invariants that must hold for all of them. Small shapes are valuable because a failing case can be printed, replayed, and reasoned about.

### Generate legal request shapes

```python
# nanoserve/tests/shapes.py
from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ShapeCase:
    batch: int
    lengths: tuple[int, ...]
    vocab: int
    block_size: int
    mode: str


def cases(seed: int = 17, count: int = 128):
    rng = random.Random(seed)
    for _ in range(count):
        batch = rng.choice([1, 2, 3, 4])
        lengths = tuple(rng.randint(1, 33) for _ in range(batch))
        yield ShapeCase(
            batch=batch,
            lengths=lengths,
            vocab=rng.choice([7, 31, 257, 1024]),
            block_size=rng.choice([1, 4, 16]),
            mode=rng.choice(["prefill", "decode", "rollback"]),
        )
```

The seed and count are part of the test artifact. If a failure occurs, print the `ShapeCase` and rerun exactly that case before changing the generator. A fixed seed is not a substitute for coverage, but it gives every failure a stable address while the generator evolves.

```python
# nanoserve/tests/test_shapes.py
import pytest
from nanoserve.tests.shapes import cases


@pytest.mark.parametrize("case", list(cases(seed=17, count=128)))
def test_shape_case_preserves_cache_and_logits(case, reference_runner, candidate_runner):
    reference = reference_runner(case)
    candidate = candidate_runner(case)
    assert candidate.cache_lengths == case.lengths
    assert candidate.block_tables == reference.block_tables
    assert candidate.logits.shape == (case.batch, 1, case.vocab)
    assert candidate.selected_tokens == reference.selected_tokens
```

The generator is intentionally not pretending to be a statistical proof. It is a seam finder. Pair it with explicit boundary cases: length 1, 15, 16, 17, and 32 for block size 16; batch 1 and the maximum configured batch; vocabulary 1 and the real vocabulary; all requests finishing simultaneously; and one request finishing while another is appended.

For the full model, run a smaller matrix on every pull request and a larger matrix nightly. On an RTX 4090 24GB, the nightly suite should be bounded by a time budget rather than a fabricated duration. The reproducible claim is the command and the case count; the reader’s machine supplies the wall-clock range.

```bash
# Reader-reproducible shape lane; no local result is implied here.
pytest -q nanoserve/tests/test_shapes.py \
  --maxfail=1 \
  --junitxml=artifacts/shapes.xml
```

## 6. Determinism needs a scope statement

“Deterministic inference” can mean at least four things: identical logits on one device, identical greedy tokens on one device, identical sampled tokens for a fixed RNG stream, or identical output across GPU architectures. Those are different promises.

For a test harness, declare the scope. A useful first promise is: same software image, same device class, same dtype, same seed, same input ids, same scheduler order, and deterministic algorithms enabled where supported. Do not claim cross-device bitwise equality unless the kernels and reduction order actually guarantee it.

The seed must be owned by the request or decode stream, not by a global process. If batch membership changes the order in which a global generator is consumed, request A can change its sampled token merely because request B joined the batch. Give each request a generator or a counter-based random stream and record its state in the trace.

```python
# nanoserve/sampling.py
import torch


def sample_top_k(logits: torch.Tensor, k: int, *, seed: int) -> torch.Tensor:
    if k < 1 or k > logits.shape[-1]:
        raise ValueError(f"k={k} outside vocabulary {logits.shape[-1]}")
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)
    values, ids = torch.topk(logits, k, dim=-1)
    probabilities = torch.softmax(values.float(), dim=-1)
    chosen = torch.multinomial(probabilities, 1, generator=generator)
    return ids.gather(-1, chosen).squeeze(-1)
```

This function makes one narrow promise: the same logits, device, `k`, and seed produce the same sampled id under the same runtime. It does not make a promise about a fused GPU sampler or another device. Test the sampler separately from the model so a sampler RNG regression does not look like a forward-pass regression.

## 7. Chaos: kill a rank where the engine can still explain itself

Distributed inference makes failure a protocol problem. A rank that dies in the middle of an all-reduce does not merely lose one worker; the surviving ranks can wait forever unless the communicator is polled, aborted, and rebuilt or the request is rejected.

![A decode step sends an injected rank failure through asynchronous error polling, communicator abort, state restore, and a bounded resume-or-reject decision](/imgs/blogs/testing-and-hardening-an-inference-engine-6.webp)

NCCL’s current documentation says that asynchronous errors should be detected with `ncclCommGetAsyncError`, the failed operation should be aborted, and the communicator destroyed with `ncclCommAbort`; it also documents communicator shrink for more advanced recovery. See NVIDIA’s [NCCL communicator error-handling guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html), current documentation crawled 2026-08-04. That is the source for the recovery shape here, not a claim that this toy implementation has run on an A100 or H100.

### Put failure injection behind an interface

Never make chaos tests depend on sending SIGKILL from arbitrary test code without a recovery contract. Put the fault at a named boundary and make the test record the expected outcome.

```python
# nanoserve/distributed/faults.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultPlan:
    kill_rank: int | None = None
    kill_after_step: int | None = None

    def should_kill(self, rank: int, step: int) -> bool:
        return (
            self.kill_rank == rank
            and self.kill_after_step is not None
            and step >= self.kill_after_step
        )


def check_fault(plan: FaultPlan, *, rank: int, step: int) -> None:
    if plan.should_kill(rank, step):
        raise RuntimeError(f"injected rank failure rank={rank} step={step}")
```

The launcher translates that exception into a process failure in a real multi-process test. The engine’s recovery code must still perform the same state transition: stop issuing new collectives, mark the request uncertain, and choose a bounded result. “Retry” is only valid if the request state is checkpointed at a safe boundary and the retry cannot duplicate externally visible tokens.

```python
# nanoserve/distributed/recovery.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SafePoint:
    request_id: str
    emitted_tokens: tuple[int, ...]
    logical_length: int
    block_table: tuple[int, ...]


def recover_after_rank_failure(point: SafePoint, *, can_rebuild: bool) -> str:
    if not can_rebuild:
        return f"reject request={point.request_id} after {len(point.emitted_tokens)} tokens"
    return f"resume request={point.request_id} at length={point.logical_length}"
```

The string outcome is intentionally boring. A real engine would return a structured status to the API layer, but the invariant is the same: a client must not receive a second copy of a token merely because the collective failed after the token was emitted. The [reliability post](/blog/machine-learning/inference-engineering/reliability-timeouts-retries-hedging-and-degraded-modes) covers the API-level retry tradeoff; this post focuses on the engine boundary that makes a retry safe or unsafe.

### What the chaos test must assert

Inject the fault at several points: before the collective is enqueued, after it is enqueued but before synchronization, after cache append but before token emission, and after token emission. The expected result differs. A failure before a cache mutation can retry from the previous safe point. A failure after a cache mutation must either roll it back or resume from the committed point. A failure after an SSE token was sent cannot be hidden by a transparent retry.

```bash
# Reader-reproducible distributed chaos command; expected result is a bounded
# resume or reject record, not a claimed latency number.
torchrun --standalone --nproc-per-node=2 \
  -m nanoserve.tests.chaos \
  --kill-rank 1 \
  --kill-after-step 3 \
  --artifact-dir artifacts/chaos-rank-1-step-3
```

The test passes when all surviving ranks exit or recover within the configured timeout, every communicator is closed, no request emits duplicate tokens, and the artifact identifies the last safe point. It fails on a hang even if the process is eventually killed by CI. A hang is evidence that the failure protocol is not bounded.

## 8. CI should be layered by cost and failure class

CI becomes unreliable when every test is placed in one undifferentiated job. Developers skip it because a one-line sampler change waits for a multi-GPU chaos lane. Conversely, a fast CPU lane can turn green while the GPU cache path is completely broken.

![The CI stack climbs from commit linting through CPU parity, GPU cache checks, nightly fuzzing, scheduled rank faults, and a release canary](/imgs/blogs/testing-and-hardening-an-inference-engine-7.webp)

The right layering is an evidence ladder. The bottom is deterministic and cheap. The top is expensive, failure-prone, and scheduled. Every layer publishes enough metadata for a failure to move downward into a smaller reproducer.

| Lane | Runs | Artifact | Policy | Source |
| --- | --- | --- | --- | --- |
| Commit | format, import, type checks | diff and tool versions | required | `reproduce: pre-commit` |
| Pull request CPU | cache ledger, sampler, tiny parity | JUnit + JSON traces | required | `reproduce: pytest -q` |
| Pull request GPU | CUDA parity, shapes, memory bounds | device + driver manifest | required for kernel changes | `reproduce: pytest --gpu` |
| Nightly | larger shape corpus, golden traces | failing seed and trace | block promotion | `reproduce: nightly workflow` |
| Scheduled chaos | rank and process faults | fault plan + safe point | investigate every failure | `reproduce: torchrun` |
| Release | pinned prompt suite and canary | complete manifest | release gate | `cited: vLLM benchmark scripts as comparison tooling` |

The [vLLM benchmark README](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md), accessed 2026-08-04, is a useful reminder that serving benchmarks need their own scripts and online-load concerns. It is not evidence for a `nanoserve` result. Our CI can borrow the separation between correctness tests and serving benchmarks while keeping the provenance boundary explicit.

### A minimal GitHub Actions shape

```yaml
# .github/workflows/inference.yml
name: inference-engine

on:
  pull_request:
  push:
    branches: [main]

jobs:
  cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install -e ".[test]"
      - run: pytest -q nanoserve/tests/test_parity.py nanoserve/tests/test_trace.py
        env:
          CUBLAS_WORKSPACE_CONFIG: ":4096:8"

  shape:
    needs: cpu
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest -q nanoserve/tests/test_shapes.py --maxfail=1

  gpu:
    if: contains(github.event.pull_request.labels.*.name, 'cuda')
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: python -m pip install -e ".[test]"
      - run: pytest -q nanoserve/tests --gpu --junitxml=artifacts/gpu.xml
      - uses: actions/upload-artifact@v4
        with:
          name: gpu-test-artifacts
          path: artifacts/
```

The YAML is a runnable workflow skeleton, but the self-hosted label and package extras must match the repository. Do not copy a hosted runner label and assume it has an NVIDIA driver. The manifest should include Python, PyTorch, CUDA runtime, driver, GPU model, NCCL version, git commit, model revision, tokenizer revision, and the command line.

### Reproducibility is part of the test API

Pinning dependencies is necessary but not sufficient. Save the generated shape seed, the trace schema version, the model and tokenizer revisions, and the test command. For GPU tests, record the device name and driver. For distributed tests, record world size and transport-related environment variables. If the test uses a clock or performance claim, record clock settings; correctness tests should avoid depending on performance.

The repository should expose one command that prints the manifest before running:

```python
# nanoserve/tests/manifest.py
import json
import os
import platform
import subprocess
import sys


def build_manifest() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "commit": commit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "torch_deterministic": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    }


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))
```

This does not magically make a future run identical. It makes an unexplained difference less likely and an explained difference cheaper.

### A failure workflow that does not destroy evidence

The moment a test fails, the harness should stop trying to be helpful by cleaning up. Keep the smallest failing input, the generated shape case, the golden-trace prefix, and the manifest. A CI job that uploads only a red line has outsourced the expensive part of debugging to whoever receives the notification.

Use a stable artifact layout. Put `manifest.json` at the root, `command.txt` beside it, `case.json` for generated inputs, `trace.json` for decode events, and `stdout.log` and `stderr.log` for process output. If CUDA is involved, include the selected device and the test’s requested deterministic settings. If a test uses a real model, write the model id and revision rather than copying an enormous checkpoint into the artifact. The revision is the pointer that makes the input meaningful.

The first retry should use the exact artifact, not regenerate from the original seed. Regeneration is a useful second check, but it answers a different question: whether the generator is stable. The exact artifact answers whether the engine can reproduce the observed failure. Keep those questions separate in the test report.

When the failure is numerical, shrink it in this order: remove requests from the batch, shorten the sequence while preserving the first mismatch position, reduce the vocabulary fixture if the mismatch is not in the sampler, and replace the full model with a tiny configuration that preserves the relevant dimensions. A useful reduced test is not merely smaller; it preserves the same transition. If the bug occurs when a request crosses block 16, a reduction to length 8 has deleted the bug rather than explained it.

When the failure is distributed, shrink the world size only after preserving the communicator behavior. A two-rank test can catch a missing abort; it cannot prove that a four-rank topology has no straggler path. Keep one scheduled test at the production world size and use the smaller test for every pull request. The fixed hardware matrix gives the comparison vocabulary—RTX 4090 and L4 for single-device paths, A100 and H100 for multi-device paths—but the test report must say which one actually ran.

Coverage should be described in terms of contracts, not only line percentages. A cache test can execute every line in `rollback()` while never checking shared-prefix ownership. A sampler test can execute top-k while never exercising `k=1`, `k=vocab`, or an all-masked row. Add a contract checklist to the test module: “append at an empty cache,” “append at a full block,” “fork before write,” “rollback across a block boundary,” “finish one member of a ragged batch,” and “abort after an externally visible token.” Reviewers can then see which behavior a new test protects.

The practical loop is:

1. reproduce the failure from the saved artifact;
2. identify the earliest violated contract, not the latest symptom;
3. add the smallest permanent fixture that fails before the fix;
4. implement the fix;
5. run the cheap lane, then the relevant GPU or distributed lane;
6. retain the fixture and its provenance in the repository.

That loop is slower than deleting a flaky test and faster than rediscovering the same cache corruption during a release. The evidence ladder in figure 7 is useful only if failures travel downward to a smaller, durable test.

## 9. Measuring performance without weakening correctness

Hardening and benchmarking interact. A debug invariant can add overhead; a golden trace can copy tensors; deterministic kernels can be slower. Keep correctness modes and performance modes explicit rather than removing checks when a benchmark looks bad.

For timing, use CUDA events around the region of interest and synchronize before reading them. Warmup belongs outside the measured sample. Report TTFT, TPOT, p50, p99, throughput, and goodput separately. The [benchmark protocol post](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) defines the series-wide prompt suite and open- versus closed-loop load; this post adds one rule: a performance result without a green correctness artifact is not a release result.

| Claim | How to earn it | Expected output |
| --- | --- | --- |
| “The kernel is faster” | `bench.py`, named GPU, warmup and CUDA events | Reader-reproducible range, not a local claim |
| “The optimization is equivalent” | parity and golden trace on the same fixture | Exact first mismatch or pass |
| “The engine is robust” | shape matrix plus scheduled chaos | Seeded artifact and bounded outcome |
| “The release is reproducible” | immutable container and manifest | Commit, versions, device, command |

Do not attach a percentage speedup to a code diff merely because a unit test passes. A correctness test tells you whether the result is acceptable. It does not tell you whether the kernel is saturated, whether the scheduler improved goodput, or whether the added trace copy changed p99.

#### Worked example: a tolerance is not a benchmark

Suppose a parity fixture contains 5 batches, 17 positions, and a vocabulary of 32,000. The logit tensor has $5 \times 17 \times 32{,}000 = 2{,}720{,}000$ scalar entries, derived from the shape. A report that says “the test passed” is incomplete: it should also say how many entries exceeded the tolerance, how many argmax positions differed, and the maximum absolute error. A timing number would require a separate reader-runnable script and named hardware; it cannot be inferred from the count.

## 10. Case studies from real engineering patterns

### The one-token drift

A fused decode path emits the same text for short prompts but diverges at the first generated token for a prompt whose length is exactly a block size. The wrong first hypothesis is sampling. The parity trace shows that the reference and candidate logits match through the last prefill position, then diverge after the candidate writes K/V at offset zero of a new block. The cache ledger catches that the logical position was incremented before the block-table lookup. The fix is to make “position used for rotary embedding,” “position used for cache offset,” and “logical length after append” three named values, then assert their relationship.

### The passing argmax with broken tails

A quantized model keeps choosing the expected greedy token, so a text-only test stays green. A task evaluation later regresses because constrained decoding needs probability mass in a tail that greedy decoding never visits. Logit parity is not the right contract for a lossy quantized model, but top-k overlap, perplexity, and task-specific evaluations are. The lesson is to test the behavior the product uses, not the behavior that is easiest to snapshot.

### The stale block that waits to hurt you

A request finishes and its blocks return to the free list. A second request reuses one of them. The first request’s stale block table is never read again in the happy path, so ordinary tests pass. A preemption or cancellation path later resumes the first request and attends to the second request’s data. The inverse ownership assertion catches this at release time: no block can be both free and present in any request table. A generation number makes the eventual error report even clearer.

### The shape that only exists in production

A batch of three has lengths 1, 16, and 17. The scheduler constructs a compact batch, the cache allocates two blocks for the final request, and the sampler sees one finished row. A test that pads every sequence to 32 never exercises the transition. The shape matrix makes the case ordinary. The expected value is not a latency number; it is a preserved length vector, a valid table, and one output decision per live row.

### The rank that dies after the token is visible

Rank one fails after the API has streamed a token but before all ranks finish their next collective. A naive retry restarts the request from the previous checkpoint and sends that token again. The client sees a duplicate; the engine sees a successful retry. The safe-point record makes the ambiguity explicit. If the external token is committed, recovery must resume after it or return an error that tells the API not to replay it. Distributed correctness includes the wire protocol.

### The flaky nightly test

A shape fuzz test fails once every few days with no seed in the report. Engineers rerun the entire nightly suite and get a green result. The failure was real, but the evidence was destroyed. Storing the seed, generated case, manifest, and trace turns the “flaky” test into a deterministic regression. If the case cannot be reproduced, the harness itself is the bug under investigation.

## When to reach for this hardening plan, and when not to

Use this plan when:

- you are changing cache layout, batching, sampling, quantization, or a CUDA kernel;
- a regression can be silent, workload-specific, or dependent on batch membership;
- you need to support more than one GPU class in the fixed matrix: RTX 4090, L4, A100 80GB SXM, or H100 80GB SXM;
- you are adding tensor parallelism or any asynchronous communicator;
- a future engineer must reproduce a failure without access to the original process.

Do not build the full chaos harness for a CPU-only educational model with no mutable cache or distributed path. Start with parity and state invariants, then add fuzzing when the first boundary bug appears. Do not claim cross-GPU bitwise determinism if your product only needs same-device greedy stability. Do not use a golden text file as the only test for a stochastic or quantized engine.

And when you are shipping a product rather than learning how an engine works, use vLLM, SGLang, or TensorRT-LLM unless you have a specific reason to own this surface. Production engines already carry years of fixes around cache reuse, scheduler races, kernels, cancellation, and distributed failure. `nanoserve` is valuable because it makes those hidden contracts visible. It is not valuable as a reason to recreate them without a test budget.

## Key takeaways

- Compare logits before comparing text; text hides numerical and sampling failures.
- Keep the reference path independent from the optimized path.
- Record the first divergent decode event, not only the final token sequence.
- Treat cache metadata as a conservation system: owners, free blocks, lengths, and tables must agree after every mutation.
- Generate small ragged shapes and block-boundary positions deliberately.
- Scope determinism by device, dtype, seed, scheduler order, and kernel contract.
- A distributed fault test passes only when failure is bounded and externally visible tokens are not duplicated.
- Store seeds, model revisions, environment manifests, and trace schema versions with every failure.
- Separate correctness gates from performance measurements, then require both for release.
- Build `nanoserve` to understand the contract; choose a mature engine when the contract is your product.

## Further reading

- [PyTorch `torch.testing` documentation](https://docs.pytorch.org/docs/stable/testing.html), last updated 2025-06-10: tensor comparison semantics and dtype tolerances.
- [NVIDIA NCCL communicator error handling](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html), current documentation accessed 2026-08-04: asynchronous errors, abort, and recovery boundaries.
- [vLLM benchmark scripts](https://github.com/vllm-project/vllm/tree/main/benchmarks), accessed 2026-08-04: serving benchmark tooling as a comparison target, not a `nanoserve` result.
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks), for provenance and load-generation discipline.
- [Sampling numerics and determinism](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance), for RNG ownership and batch invariance.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook), the later capstone that will compare the complete engine honestly.
