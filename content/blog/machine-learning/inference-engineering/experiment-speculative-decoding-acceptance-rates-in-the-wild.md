---
title: "Speculative decoding acceptance rates in the wild: When the shortcut pays"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Learn how to measure draft acceptance, derive the break-even point, and keep speculative decoding from becoming a high-batch tax."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "speculative-decoding",
    "decoding",
    "batching",
    "latency",
    "throughput",
    "pytorch",
    "gpu",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 29
---

The tempting version of speculative decoding is a magic trick: let a small model guess five tokens, ask the large model to check them in one pass, and collect a speedup. The operational version is a bet. You pay for a second model, extra KV-cache slots, a more complicated scheduler, and a verification pass whose shape changes with batch size. You win only when enough guesses survive and the target was idle enough for the saved serial steps to matter.

![A speculative decoding round branches from draft proposals into accepted and rejected prefixes before the next round](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-1.webp)

The diagram above is the mental model: draft, verify, commit the accepted prefix, and make the rejected suffix disappear. This post turns that picture into an experiment you can run in `nanoserve`. The goal is not to promise a universal multiplier. It is to learn which draft/target pair behaves well on code, chat, and translation, then let measured acceptance and measured net latency decide whether the feature stays enabled.

This is post 45 in the [Inference Engineering introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It adds a small speculative runner and an acceptance-rate harness to `nanoserve`, and it points forward to [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook). The central claim is intentionally narrow: code is usually the most promising workload, chat is mixed, translation is the workload most likely to expose a tokenizer or language-domain mismatch, and high batch can turn a good acceptance rate into a net loss.

## 1. What acceptance rate actually measures

Speculative decoding starts with two autoregressive distributions over the same vocabulary. The draft model $q$ cheaply proposes a sequence. The target model $p$ checks those proposals. In greedy decoding, “accepted” means the draft token equals the target's selected token at that position. In sampling, exact speculative sampling accepts with a probability derived from both distributions and resamples at the first disagreement; the [original speculative-decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html), published in 2023, proves that the output distribution can remain the target distribution.

For a draft length $k$, define $a_i$ as the probability that position $i$ survives verification, conditional on positions before it surviving. The expected number of accepted draft tokens in one round is not simply $k$ times a single global rate unless the positions behave similarly. The useful expression is

$$
E[A_k] = \sum_{i=1}^{k} \Pr(\text{positions }1\ldots i\text{ are accepted}) = \sum_{i=1}^{k} \prod_{j=1}^{i} a_j.
$$

This is the first place where an average can lie. A pair can report a healthy first-token acceptance rate while collapsing at position four. The target then does not get four useful serial steps back; it gets a prefix whose length is governed by the product of the conditional probabilities. `nanoserve` should therefore log both `accepted_tokens` and a histogram indexed by draft position.

### Greedy acceptance versus sampled acceptance

The two modes answer different questions.

| Measurement | What is compared | What it tells you | Source |
|---|---|---|---|
| Greedy agreement | `argmax(q)` against `argmax(p)` | A deterministic compatibility signal | derived from the decoding rule |
| Sampled acceptance | The exact accept/resample rule | Whether target-distributed sampling remains exact | cited: [Leviathan, Kalman, and Matias, 2023](https://proceedings.mlr.press/v202/leviathan23a.html) |
| Accepted length | The committed prefix per round | How many serial target steps a round replaces | derived from the prefix rule |
| Net TPOT | End-to-end time per emitted token | Whether the system is actually faster | reproduce: `nanoserve/bench_spec.py` |

The first experiment should use greedy decoding because it makes disagreements easy to inspect. Once the pair looks promising, repeat with the production sampling parameters. Do not compare greedy acceptance from one setup with sampled throughput from another and call the result a speedup.

### Why position matters

Suppose a five-token draft has conditional acceptance probabilities $a_1=0.80$, $a_2=0.75$, $a_3=0.65$, $a_4=0.50$, and $a_5=0.40$. The expected accepted length is

$$
0.80 + (0.80)(0.75) + (0.80)(0.75)(0.65) + (0.80)(0.75)(0.65)(0.50) + (0.80)(0.75)(0.65)(0.50)(0.40) = 2.856.
$$

That is a derived example, not a measurement. The draft proposed five positions, but the target receives fewer than three committed draft tokens on average. If your dashboard reports only “80% acceptance” from the first position, it will overstate the useful work saved.

## 2. The workload is the drafter's test set

A draft model is not “good” in isolation. It is good when its conditional distribution is close to the target distribution on the exact continuation the service produces. A code completion often has a narrow path: indentation, punctuation, a familiar API call, and a small set of legal next tokens. Chat has a wider set of valid continuations, even when the style is predictable. Translation has a particularly sharp failure mode: a draft that is competent in English can still choose a different segmentation or a different target-language construction from the larger model.

![A workload matrix compares draft signals, reader-reproducible acceptance ranges, risks, and token budgets for code, chat, and translation](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-2.webp)

The ranges in this figure are labeled reader expectations, not results from this workspace. They are deliberately broad. On a fixed pair and a fixed prompt suite, a reader may see code around 0.55–0.85, chat around 0.25–0.65, and translation around 0.15–0.50 as an initial hypothesis range on an RTX 4090 or A100. Those ranges are not a claim about all models; they are a decision boundary for choosing the first sweep. The benchmark must replace them with its own histogram.

### Code: predictable syntax, expensive mistakes

Code has two properties that help. The prompt frequently contains the local vocabulary and indentation context, and the output often repeats patterns already present in the repository. A draft can be small without being brilliant if it gets the local continuation right. The cost of a bad guess is also visible: the first mismatch often occurs at a branch, a string literal, or a function name, and the rest of the speculative tail is invalid.

Do not let “code is predictable” become “all code is predictable.” Generated tests, SQL, configuration files, and code with long natural-language comments have different entropy. Measure per repository or task family, not only with a single synthetic function.

### Chat: agreement is style plus facts

Chat is less forgiving because the target may choose a different conversational move even when several responses would be acceptable to a person. A small model can match punctuation and politeness while missing a factual pivot. That creates a short accepted prefix followed by a correction. Temperature matters: higher temperature broadens both distributions and can lower exact agreement, while lower temperature can improve latency but change the product's quality and diversity.

The experiment should pin `temperature`, `top_p`, stop conditions, chat template, and random seed. Otherwise the acceptance rate is partly measuring configuration drift. The [Hugging Face generation guide](https://huggingface.co/docs/transformers/v4.49.0/en/generation_strategies) documents that assisted decoding supports greedy search and sampling, and that the assistant and target normally need a shared tokenizer.

### Translation: language and tokenizer fit dominate

Translation is not one workload. English-to-German, English-to-Japanese, and English-to-a-language with a different script exercise different tokenization and morphology. A draft may be semantically close but disagree token by token. If the two models use different tokenizers, Universal Assisted Decoding must re-encode text and find a common suffix; the [Transformers documentation](https://huggingface.co/docs/transformers/v4.49.0/en/generation_strategies) describes that extra work and its tokenization-discrepancy guard.

For a first production experiment, choose a draft sharing the target tokenizer. If no compatible draft exists, test prompt lookup on prompts with repeated source text, or treat speculative decoding as a research project rather than a serving flag. This is why a small model name alone is not a pair specification: checkpoint, tokenizer, chat template, language direction, decoding settings, and context distribution all belong in the experiment record.

## 3. The break-even formula

The useful question is not “what is the acceptance rate?” It is “how many target-equivalent serial steps does one round buy for its total cost?”

Let:

- $T_d(k)$ be draft time for $k$ candidate tokens;
- $T_v(k)$ be target verification time for those candidates;
- $T_o$ be orchestration, sampling, cache, and scheduler overhead;
- $E[A_k]$ be expected committed draft tokens;
- $T_0$ be the baseline time for one ordinary target decode step.

The speculative round time is

$$
T_{spec}(k) = T_d(k) + T_v(k) + T_o.
$$

The baseline time for the same expected amount of progress is approximately

$$
T_{base}(k) = (E[A_k] + 1)T_0,
$$

where the extra one represents the target's correction or next-token result at the end of a round. This is an explanatory abstraction, not a formula stated verbatim by the original paper. It is a service-cost model for deciding whether to enable a feature; the exact implementation has batching, kernel launch, and cache effects that the terms summarize.

The simple break-even condition is

$$
E[A_k] + 1 > \frac{T_d(k) + T_v(k) + T_o}{T_0}.
$$

If the right side is 2.4, a round must yield more than 1.4 accepted draft tokens to beat baseline. This number is derived from the assumed cost ratio, not a hardware measurement. The formula explains why acceptance alone is insufficient: a five-token draft with a 70% first-position rate can lose if the draft model is slow or if verification is nearly as expensive as five ordinary target steps.

![A before-and-after ledger contrasts one-token baseline decode with draft plus verification and a measured net outcome](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-3.webp)

### Worked example: a low-load win

Assume a reader measures on a named RTX 4090, after warmup, that a baseline step costs 10 ms, a five-token draft costs 3 ms, verification costs 8 ms, and orchestration costs 1 ms. These are reader-reproducible input values for the arithmetic, not results from this post. If the acceptance histogram gives $E[A_5]=3.0$, then

$$
T_{spec}=3+8+1=12\text{ ms}, \qquad T_{base}=(3+1)10=40\text{ ms}.
$$

The derived round-level ratio is $40/12=3.33$. This does not mean the service is 3.33× faster: the setup may be batch one, and it ignores prompt time, queueing, output stopping, and GPU overlap. It says the local cost ledger has room to win.

### Worked example: the same acceptance at high batch

Now suppose batch pressure changes the target step to 4 ms, because the target is using the GPU more efficiently, while the draft plus verification costs 3 ms + 5 ms + 1 ms = 9 ms. Keep the same $E[A_5]=3.0$.

$$
T_{spec}=9\text{ ms}, \qquad T_{base}=4\times4=16\text{ ms}.
$$

The derived ratio is still 1.78, so this particular hypothetical remains positive. Increase verification to 10 ms because candidate slots now collide with the target batch, and $T_{spec}=14$ ms; the ratio falls to $16/14=1.14$. Add a 3 ms queue or synchronization penalty and the ratio becomes $16/17=0.94$: the same acceptance is now a net loss. This is the high-batch regime in one line. The target got faster, but speculation did not become proportionally cheaper.

## 4. Implementing the acceptance ledger in nanoserve

The implementation should start as a pure Python control path around model-call interfaces. The objective is to make correctness and accounting visible before adding CUDA graphs, fused verification, or continuous-batch integration.

The following interface intentionally separates proposal from verification. A real `nanoserve` model runner can replace the two callables with its cached forward pass. The return value is a token id, so the cache manager can commit or roll back by length rather than by decoding text.

```python
from dataclasses import dataclass
from typing import Callable, Sequence

Token = int
LogitsFn = Callable[[Sequence[Token]], Sequence[float]]


@dataclass
class SpecRound:
    proposed: list[Token]
    accepted: list[Token]
    replacement: Token | None


def greedy_token(logits: Sequence[float]) -> Token:
    return max(range(len(logits)), key=logits.__getitem__)


def propose(prefix: Sequence[Token], draft: LogitsFn, k: int) -> list[Token]:
    work = list(prefix)
    out: list[Token] = []
    for _ in range(k):
        token = greedy_token(draft(work))
        out.append(token)
        work.append(token)
    return out
```

The proposal path is serial inside the draft model. That is fine: the draft is cheaper, and its job is to create a candidate block that the target can score in parallel. The target runner should receive the prefix plus the candidates in one call and return one selected token per candidate position. A minimal verifier is explicit about the first mismatch.

```python
def verify_greedy(
    prefix: Sequence[Token],
    candidates: Sequence[Token],
    target_next: Callable[[Sequence[Token]], Token],
) -> SpecRound:
    accepted: list[Token] = []
    work = list(prefix)
    replacement: Token | None = None

    for candidate in candidates:
        expected = target_next(work)
        if expected != candidate:
            replacement = expected
            break
        accepted.append(candidate)
        work.append(candidate)

    return SpecRound(list(candidates), accepted, replacement)


def commit(round_: SpecRound) -> list[Token]:
    if round_.replacement is None:
        return list(round_.accepted)
    return [*round_.accepted, round_.replacement]
```

This toy verifier calls the target once per candidate and is therefore a correctness skeleton, not a performance implementation. In `nanoserve`, `target_next` becomes a batched target forward pass whose logits are indexed at the candidate positions. Keep this slow reference path: it is the oracle for the optimized path.

### Rollback belongs to the cache owner

The target may have written KV entries for all proposed tokens. Only the accepted prefix plus the replacement token is part of the sequence. If the allocator reserves `k + 1` slots and the round accepts `a` candidates, the committed length is `a + 1` when a mismatch occurs, or `k` when every proposal is accepted and the target supplies the next position separately. The uncommitted suffix must be returned to the request's free list before the next scheduling step.

![The nanoserve timeline reserves candidate slots, drafts, verifies, commits the prefix, rolls back the suffix, and records metrics](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-4.webp)

```python
@dataclass
class CacheCursor:
    length: int
    capacity: int

    def reserve(self, count: int) -> int:
        end = self.length + count
        if end > self.capacity:
            raise MemoryError(f"need {count} slots, have {self.capacity - self.length}")
        start = self.length
        self.length = end
        return start

    def rollback(self, committed: int) -> None:
        if not 0 <= committed <= self.length:
            raise ValueError("committed length must be within the cursor")
        self.length = committed


def apply_round(cursor: CacheCursor, round_: SpecRound) -> list[Token]:
    committed_tokens = commit(round_)
    cursor.rollback(cursor.length - len(round_.proposed) + len(committed_tokens))
    return committed_tokens
```

The cursor example assumes the caller reserved exactly the candidate count before verification and that the cursor started at the prefix end. A production block allocator should store the original cursor and a list of physical blocks, then release whole unused blocks and retain a partially filled final block. This is the same ownership boundary established by [paged KV-cache work](/blog/machine-learning/inference-engineering/paged-kv-cache-implementing-blocks-and-a-block-table): speculative decoding does not make the allocator optional; it gives the allocator a rollback transaction.

### Instrument positions, not just totals

```python
from collections import Counter


@dataclass
class AcceptanceStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    by_position: Counter[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.by_position = Counter()

    def add(self, round_: SpecRound) -> None:
        self.rounds += 1
        self.proposed += len(round_.proposed)
        self.accepted += len(round_.accepted)
        for index in range(len(round_.accepted)):
            self.by_position[index] += 1

    def report(self) -> dict[str, object]:
        return {
            "rounds": self.rounds,
            "mean_acceptance": self.accepted / max(1, self.proposed),
            "mean_accepted_length": self.accepted / max(1, self.rounds),
            "accepted_rounds": dict(sorted(self.by_position.items())),
        }
```

The `mean_acceptance` field is a descriptive ratio, not a performance result. Log the denominator, because a run with very short generations can make a noisy ratio look authoritative. Also log request id, workload label, draft checkpoint, target checkpoint, tokenizer hash, sampling settings, draft length, batch size, input length, output length, and whether the round ended by EOS.

## 5. Add the real benchmark harness

Acceptance-rate experiments fail most often because the measurement harness silently changes the workload. Define four prompt families: code completion, chat, RAG, and translation. This post concentrates on the three in the brief, but RAG is a useful control because prompt lookup can be unusually strong when the answer copies source text.

The fixture format should be boring and inspectable.

```json
[
  {"id":"code-001","task":"code","prompt":"def bounded_sum(xs, limit):\n    "},
  {"id":"chat-001","task":"chat","prompt":"Explain why a cache miss increases token latency."},
  {"id":"translation-001","task":"translation","prompt":"Translate to German: The cache is full."}
]
```

Do not include private production prompts in a public repository. Use stable public prompts or synthetic fixtures, record the license, and keep the same tokenized inputs for baseline and speculative modes. For translation, record the source and target language as metadata rather than burying it in a prompt string.

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


@dataclass
class Row:
    task: str
    batch: int
    draft_k: int
    accepted: int
    proposed: int
    elapsed_s: float


def run_case(case, runner, batch: int, draft_k: int) -> Row:
    start = perf_counter()
    result = runner(case["prompt"], batch=batch, draft_k=draft_k)
    elapsed = perf_counter() - start
    return Row(case["task"], batch, draft_k, result.accepted,
               result.proposed, elapsed)


def main(path: str, runner) -> None:
    cases = json.loads(Path(path).read_text())
    rows = []
    for batch in (1, 2, 4, 8, 16, 32):
        for draft_k in (0, 2, 4, 6):
            for case in cases:
                rows.append(run_case(case, runner, batch, draft_k))
    print(json.dumps([asdict(row) for row in rows], indent=2))
```

The batch values and draft lengths above are a reader-reproducible sweep plan, not a claim that every model fits every point on every GPU. For Llama-3.1-8B on an RTX 4090, reduce the largest batch or context if VRAM is insufficient; for an A100 80GB, retain the same logical grid and report the hardware separately. The point of a fixed grid is comparability, not bravado.

### Timing rules

Warm up the target and draft independently. On CUDA, synchronize before starting and after stopping the timed region, or use CUDA events. Exclude model load and tokenizer initialization from steady-state TPOT. Include draft, verification, cache writes, rollback, and scheduler overhead in speculative TPOT. If the user sees streamed output, measure time to first token and inter-token latency, not only total batch duration.

```python
import torch


def cuda_seconds(fn, warmup: int = 10, steps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / steps / 1000.0
```

The value returned is seconds per call, not tokens per second. Convert only after counting committed output tokens. For a stream with $N$ committed tokens and elapsed steady-state time $t$, `tok/s = N / t`; if the output length is short, report the sample count and confidence interval rather than a false precision.

### Open-loop versus closed-loop load

Closed-loop load sends the next request only after the previous request finishes. It answers “how fast can one client go?” Open-loop load injects requests at a specified arrival rate even while earlier work is queued. It answers “what happens to p99 and goodput as the queue fills?” Speculative decoding can look excellent under closed-loop batch one and fail under open-loop load because the target runner is already full.

Use a fixed seed for fixture ordering, but do not use a seed as a substitute for determinism validation. Compare baseline and speculative output token ids under greedy decoding. For sampling, compare distributions or a large enough sampled set under the exact correction rule; byte-for-byte output equality is not the contract.

## 6. Read acceptance by pair and workload

The table below is a protocol, not a results table. Each row says what a reader should report and how its number gets provenance.

| Pair or condition | Workload | What to record | Expected interpretation | Source |
|---|---|---|---|---|
| Llama-3.1-1B draft → Llama-3.1-8B target, if compatible checkpoints are available | code | position histogram, TPOT, VRAM | reader-reproducible high opportunity; validate tokenizer | reproduce: `bench_spec.py` |
| Same pair | chat | position histogram, p50/p99 TPOT | reader-reproducible mixed opportunity | reproduce: `bench_spec.py` |
| Same pair and language direction | translation | per-language acceptance | reader-reproducible and language-dependent | reproduce: `bench_spec.py` |
| Prompt lookup | repeated RAG or summarization | copy rate and accepted length | strong only when continuation overlaps prompt | cited: [vLLM, 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode) |
| Four H100s, Llama 3 70B, ShareGPT, QPS 1 | chat-like | speedup up to 1.5× | public vLLM result, not this engine | cited: [vLLM, 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode) |
| Four H100s, Llama 3 70B, CNN/DailyMail, QPS 1 | summarization | speedup up to 2.8× | public vLLM result, not this engine | cited: [vLLM, 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode) |
| Same vLLM examples at high QPS | service load | 1.4× or 1.8× slowdown | public warning that overhead can dominate | cited: [vLLM, 2024-10-17](https://vllm.ai/blog/2024-10-17-spec-decode) |

The vLLM figures are valuable because they include setup: model family, hardware count, dataset, and QPS. They are not transferable as a promise for `nanoserve`, Llama-3.1-8B, an RTX 4090, or a different draft. This is the standard we want for our own results.

### The acceptance histogram to keep

For every request, record the number of candidates accepted before the first mismatch. At draft length five, the buckets are 0 through 5. A bucket of five means the entire proposal survived; it does not mean the target emitted five tokens without doing work. Plot the histogram separately for code, chat, and each translation direction. Report mean and median accepted length; the median exposes a long tail of lucky completions.

Also report acceptance conditional on position. If position one is 0.75 and position five is 0.18, lower `k` may improve net latency even though the total number of proposed tokens is smaller. A dynamic policy can start at `k = 2`, grow when recent accepted lengths are near the cap, and shrink when the queue or verification cost rises. Treat that policy as a controller with hysteresis, not a per-request coin flip; otherwise the metric becomes noisy and the scheduler churns.

## 7. Why high batch creates a net-loss regime

The key hardware distinction is whether the target decode step is memory-bound or compute-bound. At low batch, a large model often streams weights and cache state for a small amount of useful arithmetic. A small draft may fit in spare capacity or at least hide part of the target's memory latency. At high batch, the target can use more of the GPU's matrix throughput. Candidate verification adds work, extra KV writes, and more scheduler bookkeeping exactly when the target is already productive.

![A batch-pressure grid shows a low-load memory-bound regime, a high-load compute-bound regime, and the policy transition between them](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-5.webp)

The phrase “high batch” is not a universal integer. It depends on model, context length, precision, kernel, and hardware. On an RTX 4090, an 8B model with short contexts can reach a different regime from a 70B model split across H100s. The reader should sweep batch and QPS until baseline achieved bandwidth or GPU utilization flattens, then repeat with speculation. That crossing, not a folklore batch threshold, is the regime boundary.

### A tiny queueing model

Let arrival rate be $\lambda$ requests per second and service rate be $\mu$ requests per second. Utilization is $\rho=\lambda/\mu$. As $\rho$ approaches one, queueing delay becomes sensitive to small service-time changes. This is a queueing abstraction, not an exact model of a continuous-batching engine. If speculation increases average service time by 5% while acceptance reduces compute by 4%, the utilization can still rise because the net service time grew. At a lower load the same change may be invisible to p50 but visible in TPOT.

This is why open-loop tests must report both throughput and goodput. Define goodput for the experiment, for example as output tokens that meet a TPOT SLO divided by wall-clock time. A system that emits more tokens while violating the latency target is not necessarily better for users.

### The vLLM contrast

The [vLLM speculative decoding article dated October 17, 2024](https://vllm.ai/blog/2024-10-17-spec-decode) is a useful public contrast. Its ShareGPT setup on four H100s reports up to 1.5× speedup at QPS 1 with a draft model, while its CNN/DailyMail prompt-lookup setup reports up to 2.8× at QPS 1. The same article reports high-QPS slowdowns of 1.4× and 1.8× in those examples. vLLM integrates draft and target runners with continuous batching and modifies its scheduler and memory manager to handle multiple token slots and both KV caches. That is the benchmark target, not an implementation claim about this post.

The practical lesson is not “never use speculation at high QPS.” It is “make load a first-class input to the policy.” vLLM's own article describes dynamic adjustment as the direction for shortening proposal length under load. In `nanoserve`, a conservative first version can disable speculation above a configured queue-depth or target-utilization threshold, then replace the threshold with a measured controller later.

## 8. Make the experiment reproducible

The benchmark report should be sufficient for someone else to reconstruct the decision. Include:

- target and draft model identifiers, revisions, quantization, dtype, tokenizer hashes, and chat templates;
- GPU model, count, interconnect, driver, CUDA, PyTorch, and engine commit;
- input-token and output-token distributions for each workload;
- decoding parameters, seed, stop rules, and draft length schedule;
- warmup count, timed token count, number of requests, and whether the load was open-loop;
- TTFT, TPOT p50/p95/p99, output tok/s, goodput, acceptance by position, VRAM peak, and target/draft utilization;
- baseline and speculative correctness checks.

![The measurement stack places correctness and acceptance beside latency, throughput, resources, and load](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-6.webp)

The numbers need a source label in the report. “Derived” is appropriate for bytes, ratios, and the break-even arithmetic. “Cited: vLLM blog, 2024-10-17” is appropriate for the public vLLM examples. “Reproduce: `bench_spec.py`, expected range on RTX 4090” is appropriate for measurements the reader must run. Never write “we measured” unless the report is actually describing a run by a named reader with the command, hardware, and artifact.

### A command-line contract

```bash
python -m nanoserve.bench_spec \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --draft meta-llama/Llama-3.2-1B-Instruct \
  --workloads fixtures/speculative.json \
  --batch-sizes 1,2,4,8,16,32 \
  --draft-lengths 0,2,4,6 \
  --warmup 10 \
  --steps 50 \
  --output results/speculative-4090.json
```

The model names in this command are an example configuration to validate, not a guarantee that the checkpoints have identical tokenizers or that they fit together on the selected GPU. The harness should fail loudly when tokenizer vocabularies, special-token ids, or model revisions are incompatible. “It ran” is not evidence that token alignment was correct.

For a reader using an RTX 4090, an acceptable expected-range statement is: after matching software and reducing context or batch when memory requires it, the code path should produce a measurable acceptance histogram and a baseline-vs-speculative TPOT comparison; the exact tok/s is hardware-, version-, and prompt-dependent. If you publish an expected tok/s range, name the exact model revision, dtype, context, batch, and kernel. This post does not have a GPU run and therefore does not invent one.

### Avoiding timing traps

Do not include the first CUDA graph capture in steady state. Do not synchronize after the draft but before the target unless that synchronization exists in production. Do not let a CPU tokenizer serialize the GPU while claiming to measure decode. Do not allocate a fresh KV tensor for every speculative round; that measures the Python allocator and can hide the actual rollback cost. Do not compare a speculative batch of five candidate positions with a baseline that uses a different maximum sequence length without reporting the shapes.

For p99, use enough independent requests to make the percentile meaningful and report the sample count. A p99 from a handful of completions is a label, not evidence. For open-loop load, log the injected arrival timestamps, queue wait, GPU time, and completion time. If queue wait dominates, a lower TPOT from speculation may not improve end-to-end latency.

## 9. Choose the drafter by workload, not fashion

![A decision tree selects prompt lookup, a small model, or a shared-target drafter before measuring net loss under load](/imgs/blogs/experiment-speculative-decoding-acceptance-rates-in-the-wild-7.webp)

The decision tree starts with the product requirement: lower TPOT. If the prompt contains repeated source text, prompt lookup is a cheap first test. The vLLM article describes n-gram lookup as useful for summarization and question answering where the answer copies the prompt. If the target and draft share a tokenizer, a small model is the straightforward control. If the target was trained with multi-token prediction or an assisted head, a self-speculative or MTP-style path may avoid a second full model, but it introduces its own checkpoint and kernel compatibility work.

### Draft model pairs

Pair size is an overhead decision, not a prestige decision. A draft that is too large may have excellent acceptance and still lose because its forward pass consumes the time it was meant to save. A draft that is too small may be cheap but produce a first mismatch so early that verification is wasted. Test at least two draft sizes or one model and one prompt-lookup baseline. Keep target weights fixed while sweeping the draft so the comparison has one moving part.

The target's tokenizer is a hard boundary for the simplest algorithm. The vLLM blog notes that Llama 3 vocabulary differences can make draft selection difficult. Hugging Face documents Universal Assisted Decoding for incompatible tokenizers, but the re-encoding and longest-common-subsequence work belongs in the time ledger. Do not call UAD “free compatibility.”

### Prompt lookup

Prompt lookup deserves its own control because it changes the cost curve. It needs no second neural model, but it can be useless on a prompt whose answer is novel. It is attractive for RAG summarization, code repair with repeated context, and templated outputs. A lookup hit should record the matched n-gram length, the copied candidate length, and the accepted length; otherwise a high acceptance rate may simply reflect a tiny candidate.

### Multi-token heads and self-speculation

Self-speculation can share some weights or hidden states, lowering memory relative to a separate draft, but it requires a model trained for the behavior or a compatible auxiliary head. Treat it as a separate pair in the experiment matrix. “Same target” does not mean “same cost”: skipped layers, extra heads, hidden-state transfers, and verification kernels alter $T_d$, $T_v$, and $T_o$.

## 10. Case studies and public results

### The original lossless algorithm

The 2023 PMLR paper by Leviathan, Kalman, and Matias frames the fundamental opportunity: autoregressive generation of $K$ tokens requires $K$ serial model runs, while a smaller approximation can propose a block and the target can verify it in parallel. Its abstract reports 2×–3× acceleration on T5-XXL against the standard T5X implementation with identical outputs. That is a cited result on the paper's setup, not a prediction for `nanoserve`. The portable lesson is the exactness condition: acceleration is useful only if the acceptance mechanism preserves the target distribution.

### vLLM's draft-model result

The vLLM team reports up to 1.5× token-generation speedup for Llama 3 70B on four H100s with a draft model, ShareGPT data, and QPS 1. The setup matters more than the headline. It is low-QPS, multi-GPU, and uses a particular draft checkpoint. The article also explains that vLLM changed its scheduler and memory manager to handle multiple token slots and both caches. A small `nanoserve` runner can reproduce the control flow, but not the production integration maturity.

### vLLM's prompt-lookup result

The same dated vLLM article reports up to 2.8× on Llama 3 70B with four H100s, CNN/DailyMail summarization, QPS 1, and n-gram prompt lookup. This is a valuable reminder that “draft model” is not synonymous with “speculative decoding.” When the answer repeats the prompt, the prompt is the drafter. The result should not be generalized to free-form chat or translation without measuring overlap.

### The high-QPS reversal

The strongest operational result in the vLLM post is the negative one: at high QPS, it reports a 1.4× slowdown for the draft-model ShareGPT setup and a 1.8× slowdown for the n-gram CNN/DailyMail setup. These are cited public values with the same broad model and hardware context as the low-QPS examples, but the article's high-QPS plots are the authority for the exact test conditions. The lesson is broader than vLLM: a technique that saves target serial depth can still increase total work per scheduling round.

### The API constraint

The current Hugging Face generation documentation says assisted decoding assumes a shared tokenizer and that assisted decoding does not support batched inputs in the documented interface. That is not a statement that every production engine lacks batching; vLLM's own implementation integrates speculation with continuous batching. It is a warning about API boundaries. A convenient library API can support a narrower feature set than a purpose-built serving engine. Benchmark the layer you will ship.

### Failure modes worth forcing

Force a mismatch at every candidate position in a unit test. The allocator should end with exactly the original prefix plus one replacement token, and no physical block should remain marked as live for the discarded suffix. Then force every candidate to match. The allocator should keep the full candidate prefix and not release a block that the next attention step still references. These two tests catch the most dangerous rollback bug: the token stream looks plausible while a stale KV pointer quietly belongs to another request.

Force an early EOS next. If the target chooses EOS at the first checked position, the engine must stop the request and release all reserved suffix capacity. Do not wait for the next scheduler tick to discover completion; otherwise a completed request can occupy speculative slots while the queue grows. Force the request to end exactly at a block boundary and one token before a block boundary. Those cases exercise both whole-block release and partial final-block retention.

Force an exhausted allocator. If the request has room for two more physical tokens but the policy asks for six candidates, the scheduler should either reduce the draft length or run ordinary decode. It should not reserve speculative slots optimistically and discover the shortage after the draft has written state. That fallback is a policy decision, not an exceptional crash. Log it as `speculation_skipped_no_capacity` so a capacity problem cannot masquerade as a low-acceptance problem.

Force cancellation during verification. The target kernel may still be in flight when an HTTP client disconnects. The request state needs a cancellation bit that the scheduler checks before committing the candidate block. The exact mechanism depends on the endpoint layer, but the invariant is stable: a cancelled request owns no blocks after its final cleanup path. This is where a toy engine earns its keep; making ownership explicit before adding streams is cheaper than debugging a production-only use-after-free.

Finally, force a tokenizer mismatch and reject the configuration before loading both models. Compare vocabulary size, special-token mapping, and the tokenizer implementation hash. A shared vocabulary size is not a proof of shared tokenization, and equal decoded text is not a proof that token ids align. If the pair needs universal assisted decoding, put re-encoding time and text normalization in the same timed region as drafting. A compatibility warning printed at startup is not enough; write the decision into the JSON result beside the acceptance histogram.

These adversarial cases are also the reason not to optimize the verifier first. A fused target pass that is 10% faster but commits an incorrect suffix is a regression. Keep the reference implementation, compare token ids and cache cursors after every round in a short deterministic fixture, and only then move the verification logits into a CUDA kernel or a captured graph. Correctness is the first performance feature because every later measurement depends on it.

## When to reach for speculative decoding, and when not to

Reach for it when:

- the product's pain is decode latency rather than prefill time;
- the target is large enough that ordinary decode is the bottleneck;
- you have a compatible draft or prompt-lookup signal;
- code or repeated-context workloads dominate;
- batch and QPS are low enough that the target has spare memory or scheduling capacity;
- you can log accepted length, rollback, and net TPOT per workload.

Skip it, or keep it disabled by policy, when:

- the workload is mostly translation with a poorly matched draft;
- prompts are open-ended and acceptance collapses at position one;
- the target is already compute-bound at the latency SLO;
- the extra draft model cannot fit without evicting useful KV blocks;
- the product needs stable p99 and you cannot measure rollback and queue time;
- your team is actually looking for faster prefill, batching, quantization, or a better attention kernel.

Use vLLM when the goal is production serving rather than learning engine construction. It already has a scheduler, memory manager, continuous batching, multiple speculative methods, and public documentation of the trade-offs. Use `nanoserve` here to understand the contract, create a test oracle, and make a narrowly scoped experiment reproducible. Replacing a production engine with a toy runner is not an optimization.

## Key takeaways

- Acceptance rate is a model-agreement signal, not a speedup.
- Accepted length is prefix-shaped; log every draft position.
- Code is the first workload to test because syntax and local repetition can make continuations predictable.
- Chat needs fixed sampling settings and often has a shorter accepted prefix than its first-token rate suggests.
- Translation must be measured per language direction and tokenizer pair.
- The break-even condition compares expected committed tokens with draft, verification, and orchestration time.
- High batch can expose speculative overhead even when acceptance is unchanged.
- Correct rollback is an allocator transaction: commit the accepted prefix and release the suffix.
- Open-loop QPS, p99 TPOT, goodput, and VRAM belong beside acceptance histograms.
- If the evidence is not from a formula, a named public source, or a runnable script with named hardware, label it as an expectation rather than a result.

## Further reading

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html), Leviathan, Kalman, and Matias, 2023.
- [How Speculative Decoding Boosts vLLM Performance by up to 2.8x](https://vllm.ai/blog/2024-10-17-spec-decode), vLLM Team, October 17, 2024.
- [Speculative Decoding in vLLM](https://docs.vllm.ai/en/stable/features/spec_decode/), official documentation.
- [Text generation strategies](https://huggingface.co/docs/transformers/v4.49.0/en/generation_strategies), Hugging Face documentation.
- [Speculative decoding core idea: draft and verify](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify).
- [An experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks).

<figure class="blog-anim">
<svg viewBox="0 0 760 220" role="img" aria-label="Five drafted tokens move through verification; accepted tokens turn green and the first rejected suffix is rolled back" style="width:100%;height:auto;max-width:820px">
<style>
.sd45-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.sd45-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.sd45-token{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.sd45-ok{fill:#b2f2bb}.sd45-bad{fill:#ffc9c9}.sd45-arrow{stroke:var(--text-secondary,#6b7280);stroke-width:2;fill:none}.sd45-ghost{fill:none;stroke:var(--text-secondary,#6b7280);stroke-dasharray:5 5;stroke-width:2}@keyframes sd45-commit{0%,20%{transform:translateX(0);opacity:1}48%,78%{transform:translateX(0);opacity:1}90%,100%{transform:translateX(-72px);opacity:1}}@keyframes sd45-reject{0%,48%{opacity:1}72%,100%{opacity:.18}}.sd45-move{animation:sd45-commit 8s ease-in-out infinite}.sd45-rejected{animation:sd45-reject 8s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.sd45-move,.sd45-rejected{animation:none}}
</style>
<text class="sd45-label" x="120" y="28">draft</text><text class="sd45-label" x="380" y="28">target verifies</text><text class="sd45-label" x="650" y="28">commit</text><line class="sd45-arrow" x1="190" y1="100" x2="280" y2="100"/><line class="sd45-arrow" x1="480" y1="100" x2="555" y2="100"/><rect class="sd45-box" x="25" y="58" width="150" height="84" rx="10"/><rect class="sd45-box" x="280" y="58" width="200" height="84" rx="10"/><rect class="sd45-box" x="555" y="58" width="180" height="84" rx="10"/><text class="sd45-label" x="100" y="94">k = 5</text><text class="sd45-label" x="100" y="119">candidates</text><text class="sd45-label" x="380" y="94">compare prefix</text><text class="sd45-label" x="380" y="119">then choose</text><text class="sd45-label" x="645" y="94">accepted</text><text class="sd45-label" x="645" y="119">prefix</text><g class="sd45-move"><rect class="sd45-token sd45-ok" x="130" y="170" width="48" height="34" rx="6"/><rect class="sd45-token sd45-ok" x="190" y="170" width="48" height="34" rx="6"/><rect class="sd45-token sd45-ok" x="250" y="170" width="48" height="34" rx="6"/><text class="sd45-label" x="154" y="193">t1</text><text class="sd45-label" x="214" y="193">t2</text><text class="sd45-label" x="274" y="193">t3</text></g><g class="sd45-rejected"><rect class="sd45-token sd45-bad" x="310" y="170" width="48" height="34" rx="6"/><rect class="sd45-token sd45-bad" x="370" y="170" width="48" height="34" rx="6"/><text class="sd45-label" x="334" y="193">t4</text><text class="sd45-label" x="394" y="193">t5</text></g><path class="sd45-ghost" d="M130 214H418"/><text class="sd45-label" x="375" y="218">rollback rejected suffix</text>
</svg>
<figcaption>Motion shows the accepted prefix committing while the rejected suffix is rolled back before the next round.</figcaption>
</figure>
