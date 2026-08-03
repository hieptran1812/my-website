---
title: "The p99 that doubled after turning on JSON mode: Constrained decoding on the request path"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Trace a JSON-mode latency regression to grammar compilation and token masks, then make structured output predictable with schema caches and device-side filtering."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "structured-output",
    "constrained-decoding",
    "json-schema",
    "latency",
    "decoding",
    "cuda",
    "pytorch",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 29
---

The endpoint was healthy until a product flag changed from `response_format=text` to `response_format=json_schema`. Mean latency moved by a few milliseconds. The p99 moved by roughly a factor of two. GPU utilization looked normal, so the first hypothesis was a scheduler regression. It was wrong. The request path had acquired a compiler, and every decode step had acquired a vocabulary-sized question: “which tokens are legal after this prefix?”

![A JSON request splits into a cold grammar path and a hot token-mask path before joining the decoder](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-1.webp)

The diagram above is the mental model: JSON mode is not one feature. It is a cold path that turns a schema into a recognizer, followed by a hot path that advances that recognizer once per generated token. In the illustrative incident used here, a cold request adds 38 ms of compilation and a hot request adds 12 ms of mask work. Those values are derived examples, not a benchmark I ran. By the end, you will have the accounting, cache key, `nanoserve` interfaces, GPU-mask design, quality checks, and rollout gates needed to make that cost explicit.

This is post #53 in [Inference Engineering](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It continues the same weights → kernels → engine → decoding → API spine and writes a constrained-decoding component rather than a full production grammar library. The next operational question belongs in [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook): when should your team build this path, and when should it use vLLM as the benchmark target and buy the feature?

## 1. JSON mode changes the latency equation

![The uncached request pays grammar compilation and per-token masking while the cached request pays only a small lookup and mask step](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-2.webp)

Start with a latency identity, not a library name. A normal request can be described as queue time plus model work plus transport. A structured request adds schema work and constrained-decoder work:

$$
T_{\text{TTFT,JSON}} = T_{\text{queue}} + T_{\text{lookup}} + T_{\text{compile}} + T_{\text{prefill}} + T_{\text{first-decode}} + T_{\text{transport}}
$$

The compile term is usually paid before the first token. The mask term is paid during decode, so it contributes to TPOT, inter-token latency, and completion time rather than only TTFT. This is an explanatory accounting identity, not a formula quoted from a particular engine.

### A deliberately reproducible incident

Take a request whose baseline p99 decomposition is illustrative but easy to reproduce in a spreadsheet or a local simulator:

| Component | JSON cold path | JSON warm path | Provenance |
|---|---:|---:|---|
| Queue and admission | 42.0 ms | 42.0 ms | derived: fixed scenario input |
| Schema lookup / compile | 38.0 ms | 0.4 ms | derived: 0.4 ms cache lookup plus 38 ms cold compile |
| Prefill | 12.0 ms | 12.0 ms | derived: fixed scenario input |
| First decode | 110.0 ms | 110.0 ms | derived: fixed scenario input |
| Transport and client | 18.0 ms | 18.0 ms | derived: fixed scenario input |
| Total | **220.0 ms** | **182.4 ms** | derived: row sum |

The arithmetic is $42 + 38 + 12 + 110 + 18 = 220$ ms for a cold request and $42 + 0.4 + 12 + 110 + 18 = 182.4$ ms for a warm request. The scenario is not evidence that your service has these timings. It is a unit-test fixture for verifying that a trace exposes the compile term. If your p99 doubles, replace these rows with your trace's rows before changing kernels.

Why can a compile cost hurt p99 more than mean? A cache miss is not evenly distributed. One schema may be common and warm while a stream of tenant-specific schemas misses repeatedly. Let $p$ be the cold-request fraction and $C$ the compile cost. The average addition is $pC$, but the tail can contain almost all of $C$ when misses cluster behind a lock or a CPU pool. A single histogram of total latency cannot tell you whether the tail is a grammar problem or a queueing problem. Emit `grammar_cache_hit`, `compile_ms`, `mask_ms`, and `schema_fingerprint` as separate dimensions.

### The first instrumentation diff

The component owns no model weights. It records the costs around a recognizer and makes the decoder accept a precomputed mask provider.

```python
# nanoserve/structured_metrics.py
from dataclasses import dataclass
from time import perf_counter_ns


def elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000


@dataclass
class GrammarTrace:
    schema_fingerprint: str
    cache_hit: bool = False
    compile_ms: float = 0.0
    lookup_ms: float = 0.0
    mask_ms: float = 0.0
    rejected_tokens: int = 0
    generated_tokens: int = 0

    @property
    def rejected_fraction(self) -> float:
        if self.generated_tokens == 0:
            return 0.0
        return self.rejected_tokens / self.generated_tokens


def timed_lookup(cache, key: str, trace: GrammarTrace):
    start = perf_counter_ns()
    grammar = cache.get(key)
    trace.lookup_ms = elapsed_ms(start)
    trace.cache_hit = grammar is not None
    return grammar
```

Do not infer cache behavior from a single end-to-end number. The trace must distinguish “compile is slow” from “the compiled mask is slow” and “the request waited for another compile.” The metric labels should be bounded: a fingerprint, model revision, grammar backend, and hit/miss are safer than putting the raw schema in a time-series label.

## 2. What is actually constrained: characters, bytes, and tokens

![Character-level JSON legality flows through UTF-8 bytes and tokenizer tokens into a parser state and an allowed token set](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-3.webp)

JSON is written as characters, but the model emits tokenizer tokens. That mismatch is where many “the mask is just a regex” explanations become misleading. A tokenizer token can contain punctuation, a complete word, multiple UTF-8 bytes, or only a continuation fragment. The recognizer must answer whether appending the token's byte sequence keeps the output inside the language defined by JSON plus the schema.

The useful state is therefore not only “inside a string” or “after a colon.” It includes at least:

- the parser state: object, array, string, number, literal, or closed value;
- the schema state: which property is expected, which properties remain required, and whether additional properties are allowed;
- the UTF-8 state: whether a partial code point is waiting for continuation bytes;
- the tokenizer boundary: whether a candidate token is safe to append as a whole;
- the model vocabulary: the integer token IDs whose byte strings can be tested against that state.

This is why a character-only validator can accept a prefix that no tokenizer token can complete cleanly, and why a token-only heuristic can accidentally permit an invalid UTF-8 or JSON boundary. The safe predicate is:

$$
\operatorname{allowed}(q, t) = \mathbf{1}\left[\operatorname{parse}(q \mathbin{\Vert} \operatorname{bytes}(t)) \text{ has a valid continuation}\right]
$$

Here $q$ is the current recognizer state, $t$ is a candidate token, and $\mathbin{\Vert}$ means concatenation. This explanatory model is an interface boundary; a concrete library may represent `q` as FSM states, PDA stacks, or compiled bitsets.

The tokenizer is part of the correctness boundary. Record the tokenizer revision in the grammar cache key. A schema compiled against one vocabulary cannot safely be reused with a model whose token-to-byte mapping changed, even if both models are called “Llama-3.1-8B.” The same warning applies to Qwen3-8B and Gemma-3-12B in the fixed series matrix: the model name is not a sufficient cache identity.

### A small token legality kernel on CPU

The following pedagogical implementation is slow by design. It makes the semantics testable before you optimize the representation. It assumes a recognizer object returns the next state or `None` when the candidate bytes are illegal.

```python
# nanoserve/grammar_mask.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Recognizer(Protocol):
    def step_bytes(self, state: int, data: bytes) -> int | None: ...


@dataclass(frozen=True)
class TokenTable:
    token_bytes: tuple[bytes, ...]


def allowed_token_ids(
    recognizer: Recognizer, state: int, table: TokenTable
) -> list[int]:
    allowed: list[int] = []
    for token_id, token_bytes in enumerate(table.token_bytes):
        if recognizer.step_bytes(state, token_bytes) is not None:
            allowed.append(token_id)
    return allowed
```

For a vocabulary of $V$ tokens, the straightforward work per decode step is $O(V \cdot c)$, where $c$ is the candidate-token byte-processing cost. The complexity is not a prediction of wall-clock time; memory layout, branch behavior, and batching dominate the constant. It is a useful warning: scanning 128k candidates for every output token is a poor place to spend Python dispatches.

### Prefix correctness beats permissive speed

There are two failure directions. A false positive lets the model emit an invalid response, which often becomes a downstream parse failure. A false negative rejects a valid token and can make the distribution too narrow or empty. Empty allowed sets are not an ordinary sampling event. They mean the recognizer, tokenizer assumptions, schema, or stop handling disagree.

Treat the empty set as a typed error with the prefix and state metadata, not as “unmask everything”:

```python
class GrammarDeadEnd(RuntimeError):
    def __init__(self, state: int, token_count: int, schema_id: str):
        super().__init__(
            f"no legal token at state={state}, tokens={token_count}, schema={schema_id}"
        )
        self.state = state
        self.token_count = token_count
        self.schema_id = schema_id


def require_nonempty(ids: list[int], state: int, n: int, schema_id: str) -> list[int]:
    if not ids:
        raise GrammarDeadEnd(state, n, schema_id)
    return ids
```

The recovery policy belongs to the product contract. For an API that promises schema-valid JSON, returning unconstrained text is a correctness violation. A service may instead return a structured error before streaming or retry with a known-safe grammar version. That choice should be visible in the API response and in the goodput metric.

## 3. The cold path: compile once, reuse safely

![Schema identity flows through canonicalization and versioned cache lookup before a compiled grammar is admitted to workers](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-4.webp)

Compilation is work that should be paid at schema deployment or first use, not by every request. The key word is safely. A cache hit is only valid when every input that affects legality is represented in the key.

An explanatory cache key can be modeled as:

$$
K = H(\operatorname{canonical}(S), M, R, B, O)
$$

where $S$ is the schema, $M$ the model and tokenizer identity, $R$ the grammar-runtime version, $B$ the backend, and $O$ the relevant options such as additional-property policy or whitespace policy. The hash $H$ is not security by itself; it is an index. If schemas cross trust boundaries, keep tenant and authorization scope outside the deduplicated value or include a policy namespace in $K$.

### Canonicalization is not cosmetic

Two JSON documents can differ in whitespace and object-member order while describing the same schema. If the compiler treats them as equivalent, canonicalization improves reuse. If it treats order or formatting as semantically meaningful, canonicalization can change the language. Use the compiler's documented normalization rules, not a home-grown `sort_keys=True` assumption.

```python
# nanoserve/grammar_cache.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


def schema_key(schema: dict, *, model_revision: str, tokenizer_revision: str,
               grammar_version: str, backend: str) -> str:
    payload = {
        "schema": schema,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "grammar_version": grammar_version,
        "backend": backend,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class CompiledGrammar(Generic[T]):
    key: str
    value: T
    estimated_bytes: int


class GrammarCache(Generic[T]):
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.items: dict[str, CompiledGrammar[T]] = {}
        self.lock = RLock()

    def get_or_compile(self, key: str, compile_fn: Callable[[], T], size_fn: Callable[[T], int]) -> tuple[T, bool]:
        with self.lock:
            hit = self.items.get(key)
            if hit is not None:
                return hit.value, True
            value = compile_fn()
            size = size_fn(value)
            if size > self.max_bytes:
                raise MemoryError("compiled grammar exceeds cache budget")
            while self.used_bytes + size > self.max_bytes and self.items:
                evicted_key, evicted = self.items.popitem()
                self.used_bytes -= evicted.estimated_bytes
            self.items[key] = CompiledGrammar(key, value, size)
            self.used_bytes += size
            return value, False
```

The lock in this teaching version protects correctness but serializes cold compiles. A production path should use per-key single-flight: the first request owns compilation, followers await the same future, and cancellation does not tear down a grammar that other requests need. Bound the number of simultaneous compiles. Otherwise “turning on JSON mode” can turn a harmless cache miss storm into a CPU denial of service.

### Precompile schemas before traffic

If schemas are known at deploy time, compile them in a readiness phase. A readiness check should verify the schema, tokenizer compatibility, supported keywords, maximum compiled size, and at least one legal continuation from the initial state. Do not mark the worker ready merely because the JSON document parsed.

```python
def precompile_schema(registry, cache, schema, identity):
    key = schema_key(schema, **identity)
    grammar, hit = cache.get_or_compile(
        key,
        lambda: registry.compile(schema, tokenizer_revision=identity["tokenizer_revision"]),
        lambda compiled: compiled.memory_bytes,
    )
    if hit:
        raise AssertionError("readiness should compile a cold schema exactly once")
    return key, grammar
```

This function deliberately treats an unexpected hit as a failed readiness assertion. In a real deployment, make it an informational log after the first restart. The point is to test the intended lifecycle: compile before accepting production traffic, then serve only cache hits for the allow-listed schemas.

## 4. The hot path: a mask is a representation, not a loop

![A latency and quality matrix compares CPU token loops, cached bitmasks, GPU masking, and jump-forward decoding](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-5.webp)

At decode time the model produces logits with shape $[B, V]$: one vocabulary score vector per active request. A grammar state gives each row a legal set. The simplest implementation writes negative infinity into illegal positions:

$$
\ell'_{b,v} =
\begin{cases}
\ell_{b,v}, & v \in A(q_b) \\
-\infty, & v \notin A(q_b)
\end{cases}
$$

where $A(q_b)$ is the allowed token set for request $b$. Softmax then assigns zero probability to illegal tokens. This operation is mathematically clean, but its cost depends on how $A(q_b)$ is represented and where the write happens.

### Bitsets make the state transferable

A dense bitset stores one bit per vocabulary entry. For vocabulary size $V$, its storage is $\lceil V/8 \rceil$ bytes per grammar state. For $V=128{,}000$, the arithmetic is $128{,}000 / 8 = 16{,}000$ bytes, or about 15.6 KiB using $1{,}024$ bytes per KiB. This is a derived storage estimate, not a claim about a specific model's compiled grammar.

The bitset is convenient for GPU broadcast and intersection, but it does not mean every state needs a unique dense allocation. States can share immutable rows, use sparse transition tables, or cache masks only for states actually reached by the prompt and output. The choice changes memory, compile time, and lookup cost.

```python
import torch


def apply_dense_mask(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """allowed: bool tensor with shape [batch, vocab]."""
    if logits.shape != allowed.shape:
        raise ValueError(f"shape mismatch: {logits.shape} vs {allowed.shape}")
    return logits.masked_fill(~allowed, float("-inf"))


def sample_argmax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    masked = apply_dense_mask(logits, allowed)
    if not torch.isfinite(masked).any(dim=-1).all():
        raise ValueError("grammar mask removed every candidate in a row")
    return masked.argmax(dim=-1)
```

The code is runnable on CPU and is a reference oracle for a fused implementation. It is not an honest performance claim. The dense boolean matrix has $B \times V$ entries, and creating it on the host for every token is exactly the pattern that creates a hidden p99 tax.

### The cost model

Let $C_g$ be grammar-state transition work, $C_m$ mask application work, and $C_s$ sampling work. A CPU implementation has an approximate per-step cost:

$$
T_{\text{step}} \approx T_{\text{model}} + C_g(B,V) + C_m(B,V) + C_s(B,V)
$$

If the mask is prepared on the CPU and copied to the GPU, add host-device transfer and synchronization. If it is compiled as a device-resident bitset and applied in the logits kernel, $C_m$ may overlap with other memory traffic, but the legality transition still needs a state update. This is why “cache the grammar” and “move masking to the GPU” are different optimizations.

| Strategy | Cold work | Hot work | Quality / correctness risk | When it fits |
|---|---|---|---|---|
| CPU scan | compile + token scan | Python/C++ scan of vocabulary | reference semantics easiest | oracle and low traffic |
| Cached CPU bitset | compile bitsets | lookup + transfer + mask | stale key or sync bug | small batch, correctness-first |
| GPU bitset | compile and upload | device intersection / fused write | kernel parity and empty masks | high batch or long output |
| Jump-forward | compile transitions | skip known literal spans | tokenizer and stop semantics | schemas with predictable literals |
| Hybrid | compile selected states | GPU hot path, CPU fallback | two paths to test | heterogeneous schemas |

All qualitative rows are engineering judgments. The exact crossover is reader-reproducible with the timing harness below on the fixed matrix of RTX 4090, L4, A100 80GB SXM, and H100 80GB SXM. Do not publish a crossover number without the harness, model revision, vocabulary, batch, output length, and clock policy.

## 5. Implement a reference decoder before optimizing it

![The decode step branches to grammar state update and model logits, then merges at a single masked sampler](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-6.webp)

The safest construction sequence is a slow reference path, a parity test, then a faster representation. The reference path should consume exactly the token IDs that the model emitted. Do not detokenize to text and then retokenize between steps: whitespace, byte fallback, and special-token handling can produce a different sequence.

```python
from dataclasses import dataclass
import torch


@dataclass
class DecodeState:
    grammar_state: int
    token_ids: list[int]
    finished: bool = False


def constrained_step(model, state, recognizer, token_bytes, temperature=1.0):
    with torch.inference_mode():
        logits = model(state.token_ids)[0, -1].float()
    allowed = []
    next_states = {}
    for token_id, raw in enumerate(token_bytes):
        next_state = recognizer.step_bytes(state.grammar_state, raw)
        if next_state is not None:
            allowed.append(token_id)
            next_states[token_id] = next_state
    if not allowed:
        raise GrammarDeadEnd(
            state.grammar_state, len(state.token_ids), "active-schema"
        )
    candidate = torch.full_like(logits, float("-inf"))
    ids = torch.tensor(allowed, device=logits.device)
    candidate[ids] = logits[ids] / temperature
    token_id = int(torch.argmax(candidate).item())
    state.token_ids.append(token_id)
    state.grammar_state = next_states[token_id]
    return token_id
```

This uses argmax to make tests deterministic. A production sampler may apply temperature, top-k, top-p, or repetition penalties before or after the grammar mask according to its contract. The order matters. If a penalty is applied to illegal tokens before masking, correctness is unchanged but wasted work remains. If a sampler renormalizes or truncates before the mask, the result can differ from “sample from the legal distribution.” Specify the order in a test.

### Property tests for legality

The key invariant is not “the output looks JSON-like.” It is that every emitted prefix is accepted by the same recognizer used to create the mask. Test random schemas, Unicode strings, escaped quotes, exponent forms, empty arrays, duplicate keys if your policy rejects them, and special tokens.

```python
def assert_prefixes_are_legal(output_ids, token_bytes, recognizer, start_state):
    state = start_state
    consumed = 0
    for token_id in output_ids:
        next_state = recognizer.step_bytes(state, token_bytes[token_id])
        assert next_state is not None, (consumed, token_id, state)
        state = next_state
        consumed += 1


def assert_mask_matches_oracle(logits, mask, oracle_ids):
    observed = set(torch.nonzero(mask, as_tuple=False).flatten().tolist())
    assert observed == set(oracle_ids)
```

The second assertion should run against the CPU oracle for thousands of states before enabling a device kernel. A mismatch is a release blocker even if the GPU path is faster. A grammar bug is a product correctness incident; a slow grammar is a performance incident. Do not trade one for the other silently.

### Where tokenization changes the quality trade-off

A grammar mask can make the output valid while changing the model's preferred continuation. If a natural token is illegal because it crosses a parser boundary, the sampler may choose several smaller tokens instead. That can increase output length and alter style. Measure both validity and task quality: exact schema validity, field-level accuracy, refusal behavior, and a semantic score appropriate to the endpoint.

The fixed model matrix makes this visible. Llama-3.1-8B, Qwen3-8B, and Gemma-3-12B do not share a tokenizer vocabulary, so the same schema can produce different legal-token densities. A lower mask density does not automatically mean better latency; it may mean more fragmented output. Report `output_tokens`, `valid`, and `field_accuracy` together.

## 6. Move the mask to the device without moving the bug

The per-token mask overhead usually has three parts: determine the next grammar state, obtain the legal-token representation, and apply it to logits. The model's decode kernel already owns a large device tensor. A host implementation can add a synchronization boundary exactly where the engine wants asynchronous work.

The device-side design should keep the state transition small and the mask immutable. One practical shape is:

1. each request stores a compact grammar-state ID on the device;
2. a device table maps state IDs to bitset words or transition metadata;
3. a fused logits kernel reads the state, checks vocabulary lanes, and writes negative infinity to illegal lanes;
4. the selected token ID updates the next state through a small transition table.

The word “fused” is not magic. It means fewer launches and fewer round trips, not zero grammar work. If the transition requires variable-length parsing or a stack operation, keep that operation in a bounded device representation or use a carefully synchronized fallback.

```cuda
// illustrative CUDA kernel: one thread owns one vocabulary lane
// The production kernel must define bitset layout and state transitions explicitly.
__global__ void apply_mask_bitset(
    float* logits,
    const unsigned int* allowed_words,
    const int* grammar_state_ids,
    int vocab_size,
    int words_per_state) {
  int lane = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y;
  if (lane >= vocab_size) return;

  int state = grammar_state_ids[batch];
  int word = lane >> 5;
  int bit = lane & 31;
  unsigned int word_value = allowed_words[state * words_per_state + word];
  if (((word_value >> bit) & 1u) == 0u) {
    logits[batch * vocab_size + lane] = -CUDART_INF_F;
  }
}
```

This kernel is an implementation sketch, not a claimed benchmark result. It assumes a dense bitset, one row per grammar state, and a batch-major logits layout. A real implementation must consider vocabulary padding, tensor-parallel vocabulary shards, dtype conversion, sampling temperature, and the state update after sampling. In tensor parallel, an “allowed token” mask must be addressed in the same global token-ID coordinate system as the shard's logits.

### GPU-side masking and tensor parallelism

For one GPU with Llama-3.1-8B on an RTX 4090 or A100, the simplest bitset is globally indexed. For tensor-parallel workers, each rank can receive the slice of the global bitset matching its vocabulary range. The all-rank sampler must still agree on the global token ID and next grammar state. A rank-local mask that silently uses local IDs is a classic correctness bug because the output can remain syntactically plausible while selecting the wrong byte string.

On H100 and A100 systems, the speed question is not just raw HBM bandwidth. A mask kernel may be launch-bound at batch 1, memory-bound at high vocabulary, or synchronization-bound if the state update lives on the host. On an L4 or RTX 4090, a CPU path can be competitive for low concurrency if it avoids a device launch. The expected result is hardware- and workload-dependent; the honest deliverable is a benchmark range from the reader, not a universal “GPU is faster” sentence.

## 7. The animated part is the state machine, not decoration

The important motion is that the legal set changes after every accepted token. A still image can show one state, but it cannot show why the mask must be recomputed and why a cached mask for the previous state is wrong. The inline animation supplied with this post shows a parser state advancing through a small object while the highlighted legal-token region changes. It is a teaching animation, not a performance trace.

<figure class="blog-anim">
<svg viewBox="0 0 760 250" role="img" aria-label="A grammar state advances through object, key, and value states while the highlighted legal token set changes after each accepted token" style="width:100%;height:auto;max-width:820px">
<style>
.gm-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.gm-hot{fill:var(--accent,#6366f1);opacity:.9}.gm-label{font:600 15px ui-sans-serif,system-ui;fill:var(--text,#111827)}.gm-small{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}@keyframes gm-state{0%,18%{transform:translateX(0)}33%,51%{transform:translateX(190px)}66%,84%{transform:translateX(380px)}100%{transform:translateX(0)}}@keyframes gm-mask{0%,18%{transform:scaleX(.9)}33%,51%{transform:scaleX(.42)}66%,84%{transform:scaleX(.16)}100%{transform:scaleX(.9)}}.gm-state{animation:gm-state 9s ease-in-out infinite}.gm-mask{transform-box:fill-box;transform-origin:left center;animation:gm-mask 9s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.gm-state{animation:none;transform:translateX(190px)}.gm-mask{animation:none;transform:scaleX(.42)}}
</style>
<text class="gm-small" x="28" y="28">grammar state</text><rect class="gm-box" x="28" y="48" width="180" height="62" rx="10"/><rect class="gm-box" x="218" y="48" width="180" height="62" rx="10"/><rect class="gm-box" x="408" y="48" width="180" height="62" rx="10"/><text class="gm-label" x="62" y="84">expect key</text><text class="gm-label" x="260" y="84">inside string</text><text class="gm-label" x="456" y="84">expect value</text><g class="gm-state"><circle class="gm-hot" cx="118" cy="145" r="16"/><path d="M136 145H300" stroke="var(--accent,#6366f1)" stroke-width="4"/><path d="M300 145H490" stroke="var(--accent,#6366f1)" stroke-width="4"/><text class="gm-small" x="48" y="184">accepted token</text><rect class="gm-box" x="28" y="198" width="520" height="22" rx="6"/><rect class="gm-mask" x="28" y="198" width="520" height="22" rx="6"/></g><text class="gm-small" x="570" y="213">legal set changes</text>
</svg>
<figcaption>After each accepted token, the recognizer moves to a new parser state and the legal-token mask changes; reusing the previous mask would be incorrect.</figcaption>
</figure>

The animation also gives a useful debugging question: at which token did the state diverge from the CPU oracle? Log the token ID, byte string, old state, new state, and the number of legal candidates. A mismatch at the first quote points to tokenizer or string-entry handling. A mismatch after a Unicode continuation points to byte-state logic. A mismatch only after a closing brace points to schema completion or stop-token policy.

## 8. Cache policy is a latency policy

![A decision tree chooses schema compilation reuse by identity cache state tenant scope and invalidation version](/imgs/blogs/case-study-the-p99-that-doubled-after-turning-on-json-mode-7.webp)

Caching compiled masks changes more than CPU utilization. It changes who pays compile latency, how much memory is reserved, and how a tenant can influence another tenant's tail. The cache therefore needs an explicit policy.

### Key dimensions

The minimum identity is schema content, model revision, tokenizer revision, grammar runtime, backend, and options. Add tenant or policy namespace when the same schema text is not allowed to share compiled artifacts. Do not put request IDs in the key; that defeats reuse. Do not omit the tokenizer revision; that makes the cache appear healthy while invalidating legality.

### Eviction and admission

Suppose a cache budget is 64 MiB and a compiled grammar uses 256 KiB. The upper bound from arithmetic is $64 \times 1{,}024 / 0.25 = 262{,}144$ entries if every byte were available and every entry were exactly 256 KiB. Real capacity is lower because metadata, allocator alignment, locks, and fragmentation consume space. State the calculation as a ceiling, not a capacity measurement.

Use weighted LRU or TinyLFU only after you know the schema distribution. An allow-listed schema registry can be safer than admitting arbitrary user schemas. If user schemas are required, cap compilation time, compiled bytes, and per-tenant concurrency. A cache miss should fail fast with a typed error or queue behind a bounded single-flight, not start an unbounded compiler pool.

### Invalidation

Invalidate on any change that can alter the accepted language or token mapping. That includes grammar-runtime upgrades, tokenizer changes, model revisions if the vocabulary changes, and feature flags for whitespace or additional properties. Retain the old version during a rolling deploy only if workers never mix incompatible grammar objects.

```python
def cache_namespace(model, tokenizer, runtime, options):
    return ":".join([
        model.revision,
        tokenizer.revision,
        runtime.version,
        options.whitespace_mode,
        str(options.allow_additional_properties),
    ])


def invalidate_on_namespace_change(cache, previous, current):
    if previous != current:
        cache.clear()
        return True
    return False
```

The `clear()` operation should be generation-aware in a multi-worker service. Mark old entries retired, stop assigning them to new requests, and free them after in-flight requests release references. Abrupt destruction can turn a cache optimization into a use-after-free in a native backend.

## 9. Measure the regression with a reader-runnable harness

The harness must compare four modes, not just “JSON on” and “JSON off”:

1. unconstrained sampling;
2. constrained cold schema;
3. constrained warm schema with CPU mask;
4. constrained warm schema with device mask.

Hold model, tokenizer, prompt suite, output cap, sampling seed, batch, and arrival process fixed. The series prompt suite is chat, RAG, code completion, and translation; JSON extraction is most representative of RAG and API tool calls, but include at least one short and one long output because per-token overhead scales with output length.

```python
# bench_structured.py
import argparse
import json
import statistics
import time


def percentile(values, p):
    values = sorted(values)
    index = min(len(values) - 1, int((p / 100) * len(values)))
    return values[index]


def run_case(server, requests, mode):
    observations = []
    for request in requests:
        start = time.perf_counter_ns()
        result = server.generate(request, structured_mode=mode)
        finish = time.perf_counter_ns()
        observations.append({
            "mode": mode,
            "e2e_ms": (finish - start) / 1_000_000,
            "output_tokens": result.output_tokens,
            "valid": result.valid_json,
            "grammar_cache_hit": result.grammar_cache_hit,
            "compile_ms": result.compile_ms,
            "mask_ms": result.mask_ms,
        })
    return observations


def summarize(rows):
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    for mode, items in by_mode.items():
        latencies = [item["e2e_ms"] for item in items]
        valid = sum(item["valid"] for item in items) / len(items)
        print(json.dumps({
            "mode": mode,
            "count": len(items),
            "p50_ms": percentile(latencies, 50),
            "p99_ms": percentile(latencies, 99),
            "valid_fraction": valid,
            "mean_compile_ms": statistics.fmean(item["compile_ms"] for item in items),
            "mean_mask_ms": statistics.fmean(item["mask_ms"] for item in items),
        }, sort_keys=True))
```

This script is a measurement harness, not a source of numbers in this article. Run it after warmup and with an open-loop load generator when you care about service p99. For GPU timing, put `torch.cuda.synchronize()` before and after a reference region or use CUDA events; otherwise the CPU timer can stop while kernels remain queued. For a server measurement, record queue time separately and use the same request arrival process for every mode.

Expected ranges must be hardware-specific and reported only after a reader runs the script. A reasonable report format is “on an A100 80GB SXM, batch 8, Llama-3.1-8B, 512 input tokens, 128 output tokens, backend commit X, warmup Y, the device-mask case landed in the range Z–W ms p99.” The range is an expected reader result, not a fact this post can assert. If no run exists, say `unavailable`.

### Derived break-even arithmetic

Let $C_c$ be cold compilation, $C_l$ lookup, $M$ per-token mask cost, and $N$ generated tokens. The warm-cache path is preferable to compiling on every request when:

$$
C_c + N M_{\text{cold}} \gt C_l + N M_{\text{warm}}
$$

Rearranging gives the token count at which the per-token optimization repays a larger lookup or upload cost:

$$
N \gt \frac{C_l - C_c}{M_{\text{cold}} - M_{\text{warm}}}
$$

The inequality only has an intuitive positive threshold when the denominator is positive. Use actual measurements from the harness, with units aligned. If compilation is a one-time deployment cost, remove $C_c$ from the per-request comparison and include it in readiness time instead.

#### Worked example: a cold request

Assume a schema compiler costs 38 ms, a cache lookup costs 0.4 ms, and a response produces 32 tokens. The warm path saves $38 - 0.4 = 37.6$ ms per request. If 100 requests use that schema, the arithmetic is $100 \times 37.6 = 3{,}760$ ms of request-path work avoided, before considering contention. This is a derived scenario, not a production measurement. It is enough to justify precompilation when the schema is known before traffic.

#### Worked example: per-token overhead

Assume a baseline decode step is 2.0 ms and a CPU mask adds 0.15 ms. For 64 output tokens, the added decode time is $64 \times 0.15 = 9.6$ ms, and the constrained decode step is 2.15 ms. The relative step increase is $0.15 / 2.0 = 7.5\%$. If a device mask reduces the added cost to 0.03 ms, the analogous increase is $0.03 / 2.0 = 1.5\%$. These are explanatory inputs chosen to show the arithmetic; use the harness to obtain an expected range on RTX 4090, L4, A100, and H100.

## 10. Quality, latency, and graceful failure

Structured decoding is a contract with two dimensions: syntactic validity and model utility. A mask can guarantee that a response parses while making the output less natural, longer, or less accurate. A jump-forward optimization can reduce token steps for fixed literals while making debugging and token accounting harder. A permissive fallback can protect availability while violating the endpoint's schema promise.

Track at least these outcomes:

| Outcome | Meaning | Release interpretation | Source |
|---|---|---|---|
| Valid JSON fraction | parser accepted complete output | hard floor for strict endpoint | derived: valid / total |
| Schema validity | required fields and types hold | hard floor for typed API | derived: validator result |
| Field accuracy | values answer the task | quality guardrail | reproduce: evaluation harness |
| Output tokens | tokenization and verbosity cost | latency and cost input | derived: token counter |
| p50 / p99 TTFT | cold and queue behavior | user-visible latency | reproduce: load harness |
| TPOT / ITL | hot per-token cost | streaming smoothness | reproduce: CUDA-event trace |
| Empty-mask count | recognizer/tokenizer disagreement | zero tolerance | derived: error counter |

The official vLLM structured-decoding write-up dated 2025-01-14 describes FSM and PDA approaches, identifies compilation as a TTFT contributor, and reports up to 5× TPOT improvement for XGrammar under its stated setup. That is a cited result from [vLLM's structured decoding article](https://vllm.ai/blog/2025-01-14-struct-decode-intro), not a result from this post. The same article discusses fallback behavior and limitations; treat backend coverage as a compatibility question, not a promise that every schema gets the same speedup.

### A safe failure ladder

For a strict JSON-schema endpoint:

1. reject unsupported schema keywords at validation time;
2. reject a schema that cannot be compiled within its resource budget;
3. fail readiness if an allow-listed schema has no legal initial transition;
4. fail the request with a structured error on an empty mask;
5. never silently disable masking and return text that violates the contract.

For an assistant-style endpoint where JSON is a preference rather than a guarantee, the API may expose an explicit degraded mode. It must still label the response as unconstrained, so downstream consumers do not mistake a best effort for a validated object.

## Case studies / real numbers

### The compile cost hiding inside TTFT

The vLLM structured-decoding article is the closest public reference for this incident pattern. Its description separates grammar compilation from the per-token FSM/PDA path and explicitly treats compilation as a TTFT contributor. The relevant lesson is architectural: if compilation happens during request handling, its latency is part of the endpoint even when the model forward pass is unchanged. The article's reported “up to 5× TPOT” is setup-specific and must not be transplanted into a Llama-3.1-8B on RTX 4090 claim.

### The tokenizer boundary that breaks a plausible parser

The [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer), accessed for this post on 2026-08-04, exposes conversion between text and token IDs and the model-specific vocabulary boundary. That interface is why the grammar component must retain token IDs and tokenizer identity rather than treating decoded text as the canonical intermediate. A parser that accepts a character prefix is not enough to prove that a model can emit the prefix as one or more legal token bytes.

### The device mask that is easy to get subtly wrong

PyTorch's [`masked_fill`](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html) is a useful reference operation: it fills positions selected by a boolean mask and gives the CPU implementation a simple oracle. A fused CUDA kernel should match its semantics on finite logits, `-inf`, dtype conversion, vocabulary padding, and empty rows. The API documentation is the source for the operation's behavior, not evidence for a speedup.

### The model matrix is part of the experiment

The [Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), the [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-8B), and the [Gemma model documentation](https://ai.google.dev/gemma/docs) are the places to verify model and tokenizer revisions before a run. The fixed series matrix names these models so comparisons remain interpretable. Do not present a mask density or compile-size result without naming the vocabulary and tokenizer revision that produced it.

## When to reach for this (and when not to)

Build the constrained-decoding path when the endpoint has a real schema contract, schemas repeat enough to amortize compilation, and you can test token-level legality against a reference recognizer. Precompile a registry when product schemas are known. Use a bounded cache for user-supplied schemas. Move mask application to the GPU when the measured host synchronization and per-token work are a material part of p99 at the target batch and output length.

Use a mature server as the benchmark target when you need broad grammar coverage, tensor parallelism, streaming, and operational support before your team has a parity suite. vLLM's documented structured-decoding backends are a useful comparison point, but their flags and backend behavior are versioned. Cite the version and rerun the compatibility matrix rather than copying a command from an old incident report.

Do not add a grammar compiler to the hot path for a one-off schema that can be validated after generation. Do not use a dense host mask when output is one or two tokens and the endpoint's p99 budget is tight. Do not optimize the CUDA write before you can prove the CPU oracle and GPU result select exactly the same legal token IDs. And do not trade away schema validity to make a dashboard's p99 look better.

## Key takeaways

- JSON mode adds both a cold compilation path and a hot per-token mask path.
- Measure compile, lookup, mask, queue, and model time separately; one p99 cannot diagnose all five.
- Cache keys include canonical schema, model revision, tokenizer revision, grammar runtime, backend, and relevant options.
- Precompile allow-listed schemas before readiness and use bounded single-flight for user schemas.
- Token legality is a byte-and-token boundary problem, not only a character parser problem.
- Keep a CPU recognizer and mask oracle until the device kernel passes parity and empty-mask tests.
- A device-side bitset can remove host synchronization, but it does not remove grammar-state transitions or correctness work.
- Measure validity, field accuracy, output tokens, TTFT, TPOT, and p99 together.
- Every latency number must be derived, cited with setup, or reproduced by the reader; this post claims no first-hand run.
- If the schema is optional and rare, post-validation may be the better product trade-off.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series frame and `nanoserve` artifact.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone decision tree.
- [vLLM structured decoding](https://vllm.ai/blog/2025-01-14-struct-decode-intro), 2025-01-14 — FSM/PDA design and cited performance context.
- [JSON Schema specification](https://json-schema.org/specification), accessed 2026-08-04 — schema vocabulary and validation contract.
- [Hugging Face tokenizer API](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer), accessed 2026-08-04 — token ID and tokenizer identity boundaries.
- [PyTorch `masked_fill`](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html), accessed 2026-08-04 — reference mask semantics.
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) — warmup, synchronization, and provenance discipline.
