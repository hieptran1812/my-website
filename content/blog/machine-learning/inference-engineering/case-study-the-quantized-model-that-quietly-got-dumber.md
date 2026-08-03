---
title: "The quantized model that quietly got dumber: Why perplexity is not a release gate"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build a reproducible INT4 evaluation gate that catches task regressions, preserves provenance, and rolls back safely before users discover the quality cliff."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "quantization",
    "transformers",
    "pytorch",
    "cuda",
    "latency",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 28
---

The release note said “INT4 enabled on the 24 GB target.” The dashboard said perplexity improved by 0.1. The first customer report said the model had stopped producing compilable patches.

That combination is not paradoxical. Perplexity is an average log-likelihood over a token stream. A coding assistant is a collection of contracts: preserve the API, close the bracket, use the right import, obey the requested format, and do not confidently invent a dependency. A quantizer can preserve the average probability of ordinary prose while damaging one of those narrow, high-value behaviors.

![A quantization release branches into perplexity and task evaluation before a promotion or rollback decision](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-1.webp)

The diagram is the mental model for this post: one artifact must travel down two paths. The first asks whether next-token fit looks acceptable. The second asks whether the model still does the work that makes it valuable. Only their combined result should reach the deployment switch. By the end, we will add a small, runnable evaluation and provenance component to `nanoserve`, define an FP16 reference contract, evaluate the fixed model and hardware matrix, and make rollback an ordinary output of the gate rather than an emergency ritual.

This is a case study, not a report of a benchmark I ran. I have no GPU in this environment and make no first-hand performance claim. Every number below is either derived from an explicit formula, cited to a primary source with its date, or presented as a reader-reproducible expectation with a named script and an expected range. The point is to make the decision process trustworthy even when the result is inconvenient.

## Why the green perplexity check was a red release signal

Perplexity is useful. It is also easy to overinterpret. For a tokenized corpus with tokens $x_1, x_2, \ldots, x_N$, a causal model assigns a conditional probability to each next token. The average negative log-likelihood is

$$
\operatorname{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log p(x_i \mid x_{<i}),
$$

and perplexity is $\operatorname{PPL}=\exp(\operatorname{NLL})$. The expression rewards the model for distributing probability well over the corpus. It does not know that one missing closing brace can invalidate an entire patch, or that one wrong citation in a RAG answer can make the answer unusable.

The same issue appears in ordinary classification. A mean score can remain stable while a small subgroup gets worse. In language modeling, the subgroup might be code delimiters, rare identifiers, JSON punctuation, multilingual scripts, or the final answer position after a long retrieved context. If those tokens are rare in the calibration corpus, their damage is diluted by the common tokens that dominate the mean.

The failure was therefore a measurement-design failure, not evidence that INT4 is inherently unsafe. Weight-only quantization can be an excellent inference technique. The [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) literature explains why reducing weight traffic can help decode, and the [INT4 fused GEMM implementation](/blog/machine-learning/inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly) in this series explains the kernel path. Neither technique decides whether a product contract survives. That is an evaluation responsibility.

| Question | Perplexity answers | Product evaluation must answer | Failure if omitted |
|---|---|---|---|
| Distribution | Is average next-token likelihood similar? | Does the model choose acceptable answers? | Averages hide tails |
| Format | How probable are corpus tokens? | Is JSON valid and complete? | Parsers reject output |
| Code | How well does text resemble the corpus? | Does a patch compile or pass tests? | Plausible code is still broken |
| Retrieval | How well does it model prose? | Does it use the supplied evidence? | Fluent hallucination |
| Operations | What is average loss? | Does the candidate fit latency and memory budgets? | Quality win can be too expensive |

The operational lesson is simple: perplexity is a health check, not a release contract. Keep it because it is cheap, stable, and useful for detecting a broad numerical catastrophe. Add task tests because the user is not buying a lower NLL; the user is buying behavior.

## 1. Reconstruct the incident as an evidence problem

The original release process had four steps: take the FP16 checkpoint, quantize it, run a small text perplexity job, and publish the faster artifact. Each step was locally reasonable. The missing step was a named comparison against the behaviors that mattered.

The incident can be expressed as a causal chain. A group-wise INT4 representation stores four bits per weight value plus scales and metadata. The dequantization kernel reconstructs values during the matrix multiplication. A scale chosen from a calibration group is a compromise: it represents the large, common values well enough, but it may distort a small outlier pattern that a particular attention head or MLP feature uses for code syntax or retrieval attribution. The model remains broadly fluent. A narrow behavior regresses.

That explanation is an engineering hypothesis until an evaluation isolates it. Do not convert it into a claim about a particular model layer without inspecting the artifact and comparing traces. The reliable facts are the observed contract, the candidate configuration, and the reproducible delta.

### A release needs a reference, not a vibe

The reference should be the exact FP16 or BF16 checkpoint served through the same tokenizer, chat template, prompt manifest, sampling policy, and evaluator. “The original model” is not specific enough. A changed tokenizer revision, a different system prompt, or a different stop condition can create a false regression.

For deterministic comparisons, use greedy decoding or fixed sampling seeds where the runtime makes seeded sampling reproducible. For stochastic production behavior, run multiple seeds and report a confidence interval or pass-rate range. The test should record the prompt ID, model hash, quantizer configuration, runtime version, GPU name, CUDA version, and evaluator commit. These are not bureaucracy. They are the coordinates needed to reproduce the decision.

```python
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import json

@dataclass(frozen=True)
class Provenance:
    candidate: str
    reference: str
    tokenizer: str
    evaluator_commit: str
    quantizer: str
    group_size: int
    decode_seed: int

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()

record = Provenance(
    candidate="llama-3.1-8b-int4",
    reference="llama-3.1-8b-bf16",
    tokenizer="meta-llama/Llama-3.1-8B@locked-revision",
    evaluator_commit="git:replace-with-ci-commit",
    quantizer="gptq-weight-only",
    group_size=128,
    decode_seed=17,
)
print(json.dumps(asdict(record), sort_keys=True, indent=2))
```

The `group_size=128` value above is an input to a reproducible example, not a universal recommendation. GPTQ’s paper describes a second-order, weight-only post-training quantization method and reports its own setup; use the [GPTQ paper](https://arxiv.org/abs/2210.17323), published October 2022, for the algorithm rather than treating this example as a result from that paper. AWQ, GPTQ, SmoothQuant, and vendor formats make different compromises; the [quantization in LLMs overview](/blog/machine-learning/large-language-model/quantization-in-llm) is the right place to compare mechanisms.

### The symptom and the wrong first hypothesis

The symptom was a code score falling while perplexity stayed within its release budget. The tempting diagnosis was “the evaluator is noisy.” That diagnosis is possible, so the first response should be to replay the same prompt IDs and seeds against the reference and candidate, then inspect failures by category. If all categories move together, suspect a harness issue. If only code formatting and compilation move, suspect a task-sensitive change or an evaluator mismatch.

Do not start by tuning the threshold until the input is stable. A threshold can only express a business decision; it cannot repair a moving target.

## 2. Why INT4 can preserve language fit and damage behavior

Weight-only quantization maps a floating-point group to a lower-precision integer representation. A symmetric toy form is

$$
q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{w_i}{s}\right), -Q, Q\right),\qquad \hat{w}_i=sq_i,
$$

where $w_i$ is a weight, $s$ is a scale, and $Q$ is the largest representable magnitude. For signed INT4, the commonly used integer range is $[-8,7]$ or a related asymmetric convention; the exact range belongs to the file format and kernel, so the evaluator must record it rather than infer it from the word “INT4.”

For a group of $g$ weights represented with $b=4$ bits each and one scale stored in $s_b$ bytes, the approximate storage is

$$
B_{\text{group}}=\frac{gb}{8}+s_b+ B_{\text{metadata}}.
$$

If $g=128$, $b=4$, and the scale is 2 bytes with no extra metadata, that is $128\cdot4/8+2=66$ bytes, or $66/128=0.515625$ bytes per weight. Compared with BF16’s 2 bytes per weight, the idealized reduction is $2/0.515625\approx3.88\times$. This is derived storage arithmetic, not a measured end-to-end speedup. Scales, packing, alignment, activations, and kernel choice determine the real footprint.

The error is not evenly distributed. Rounding error for a single value is bounded by roughly half a quantization step when clipping does not occur, but clipping can dominate when an outlier exceeds the group’s range. More importantly, a small weight perturbation can be amplified by a sequence of matrix multiplications and nonlinearities. A language-model loss averaged over many ordinary tokens may barely change while a particular logit margin flips on a decision-critical token.

The right mental model is a ranking problem. Suppose two tokens, `}` and `]`, have logits 4.10 and 4.05 before quantization. A perturbation of −0.08 to the first and +0.04 to the second changes the ranking. The absolute perturbations are small, but greedy decoding changes. If that token is a closing delimiter, the product behavior changes discontinuously. This is why “the loss delta is small” cannot imply “the program remains valid.”

![The old gate compares one average perplexity score while the stronger gate checks task behavior and provenance before shipping](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-2.webp)

The before-and-after figure is intentionally asymmetric. The old path has one green number and a red blind spot. The new path has more work, but each result has a named contract: task score, allowable delta, and evidence needed to replay the decision.

#### Worked example: storage is not quality

Take a simplified 8-billion-weight model. The BF16 weight payload is $8{,}000{,}000{,}000\times2=16{,}000{,}000{,}000$ bytes, which is about 14.9 GiB after dividing by $2^{30}$. With the illustrative 0.515625 bytes per INT4 weight above, the packed payload is $8{,}000{,}000{,}000\times0.515625=4{,}125{,}000{,}000$ bytes, about 3.84 GiB. The difference is about 11.1 GiB before runtime buffers and alignment.

That arithmetic explains why INT4 may make a 24 GB RTX 4090 deployment possible. It says nothing about code pass rate, RAG faithfulness, or translation adequacy. Those must be measured separately. A capacity win is an opportunity to evaluate more, not permission to evaluate less.

## 3. Build the fixed model, hardware, and workload matrix

One task suite can overfit the gate just as one corpus can overfit perplexity. The series uses a fixed matrix: Llama-3.1-8B as the spine model, with Qwen3-8B and Gemma-3-12B as architecture comparisons; RTX 4090 24GB, L4 24GB, A100 80GB SXM, and H100 80GB SXM as hardware targets; and chat, RAG, code completion, and translation as workload shapes.

The hardware names are a coverage contract, not a claim that each run happened. When a spec is needed, cite the primary vendor source: NVIDIA’s [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), [L4 datasheet](https://www.nvidia.com/en-us/data-center/l4/), [A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/), and [H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/). The dates of those product pages can change; the CI record should store the retrieval date alongside any copied bandwidth or memory number.

![The evaluation matrix crosses three fixed models with four task contracts instead of collapsing quality into one corpus score](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-3.webp)

The matrix has two jobs. First, it prevents the gate from silently becoming “Llama on WikiText.” Second, it tells us what to do when a format is not supported equally by every runtime. A candidate may be eligible for an RTX 4090 canary but not for an A100 path if the kernel implementation or calibration artifact differs. The release record should say “not evaluated” rather than converting absence into a pass.

### Workload contracts

For chat, score instruction adherence, refusal behavior where relevant to the product, and structured-output validity. For RAG, include a supplied context and score answer support against that context; do not score only fluency. For code completion, compile or run tests in a sandbox. For translation, use a fixed source set and a human- or reference-based metric, but retain sampled outputs because a scalar metric can hide a terminology failure.

The matrix can start small. Four workloads times three models gives 12 model-task cells. If each cell has 100 prompt IDs, that is 1,200 examples per candidate before multiple seeds. This is derived counting, not a performance claim. The suite can be tiered: a fast presubmit subset, a nightly full suite, and a pre-production canary. The important property is that the same IDs and evaluator version are used for the reference and candidate.

```python
from dataclasses import dataclass
from itertools import product

MODELS = ("Llama-3.1-8B", "Qwen3-8B", "Gemma-3-12B")
HARDWARE = ("RTX 4090 24GB", "L4 24GB", "A100 80GB SXM", "H100 80GB SXM")
TASKS = ("chat", "RAG", "code", "translation")

@dataclass(frozen=True)
class Cell:
    model: str
    hardware: str
    task: str

cells = [Cell(*item) for item in product(MODELS, HARDWARE, TASKS)]
print(f"matrix cells: {len(cells)}")
print(cells[0])
```

This prints 48 cells because $3\times4\times4=48$. A team can reduce the presubmit matrix to the spine model on two representative GPUs while retaining the full matrix for nightly evaluation, but that reduction must be explicit in the policy. “We tested the model” is not enough; “we tested Llama-3.1-8B on the RTX 4090 and A100 in presubmit, and all 48 cells nightly” is reproducible.

### What a cell records

Every cell should record at least task score, reference score, delta, sample count, seed set, latency summary, peak allocated memory, and status. For tasks with pass/fail outcomes, report both the count and the denominator. “92%” without “92 of 100 prompt IDs” makes later auditing harder.

| Field | Example | Provenance |
|---|---|---|
| Model | Llama-3.1-8B | cited: model card, retrieved 2026-07-20 |
| GPU | A100 80GB SXM | cited: NVIDIA datasheet, retrieved 2026-07-20 |
| Task | code completion | fixed series workload |
| Prompt IDs | code-0001…code-0100 | evaluator manifest |
| Candidate | INT4 group 128 | quantizer config |
| Reference | BF16 checkpoint | immutable model hash |
| Score delta | candidate minus reference | derived arithmetic |
| Decision | pass / rollback / unevaluated | policy evaluation |

The date in the table is the evaluation record date, not a claim about a vendor page’s publication date. Pin the URLs and hashes in the artifact. This is especially important for model cards, which may be updated after a release.

## 4. Implement a task-aware gate in `nanoserve`

The post adds a narrow component: `nanoserve/eval_gate.py`. It does not replace a mature evaluator. It makes the release decision explicit, typed, and serializable so deployment code can consume it.

First define metrics as values with direction and budgets. A higher-is-better task score uses `candidate - reference >= -budget`. A lower-is-better perplexity score uses `candidate - reference <= budget`. Do not mix directions in a generic subtraction without naming the rule.

```python
from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower"]

@dataclass(frozen=True)
class MetricBudget:
    name: str
    direction: Direction
    max_drop: float

    def passes(self, reference: float, candidate: float) -> bool:
        delta = candidate - reference
        if self.direction == "higher":
            return delta >= -self.max_drop
        return delta <= self.max_drop

budgets = (
    MetricBudget("perplexity", "lower", 0.20),
    MetricBudget("code_pass_rate", "higher", 0.02),
    MetricBudget("rag_support", "higher", 0.03),
    MetricBudget("json_valid", "higher", 0.01),
)
print(budgets[1].passes(0.80, 0.79))
```

The example prints `True`: a one-percentage-point drop is within a two-point budget. This threshold is illustrative policy, not a cited benchmark. A real team should set it from product risk, sample size, and historical variability. If the code metric is measured on 100 examples, one example is one percentage point; if it is measured on 20 examples, one example is five points. That denominator must be visible.

Next, make a cell result. A missing result is not a pass. Treating `None` as zero is one of the easiest ways to ship an unevaluated path.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MetricResult:
    budget: MetricBudget
    reference: Optional[float]
    candidate: Optional[float]

    def status(self) -> str:
        if self.reference is None or self.candidate is None:
            return "unevaluated"
        return "pass" if self.budget.passes(self.reference, self.candidate) else "fail"

results = [
    MetricResult(budgets[0], 8.40, 8.31),
    MetricResult(budgets[1], 0.80, 0.71),
    MetricResult(budgets[2], 0.76, 0.75),
]
for result in results:
    print(result.budget.name, result.status())
```

The numbers in this snippet are deliberately synthetic and labeled as such by the code’s role: they demonstrate the branch behavior. They are not claims about Llama-3.1-8B or any quantizer. In a production repository, test fixtures should name themselves `synthetic_*` so nobody mistakes them for a benchmark.

Finally, aggregate results with a fail-closed policy. The promotion decision should distinguish `fail` from `unevaluated`, even if both block shipping.

```python
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class GateDecision:
    status: str
    blockers: tuple[str, ...]

def decide(results: Iterable[MetricResult]) -> GateDecision:
    blockers = []
    for result in results:
        state = result.status()
        if state != "pass":
            blockers.append(f"{result.budget.name}:{state}")
    return GateDecision("promote" if not blockers else "block", tuple(blockers))

decision = decide(results)
print(decision.status, decision.blockers)
```

This synthetic fixture blocks the candidate because the code pass-rate delta is nine points, larger than the two-point budget. That is the incident in miniature: perplexity is green, behavior is red, and the correct output is block.

### Avoiding false confidence from flaky tasks

Task evaluators need their own reliability checks. A code test can fail because a package mirror is unavailable. A RAG judge can fail because the evaluator model timed out. A translation reference can contain an alternate but valid phrase. Record infrastructure failures separately from model failures, retry only the infrastructure category, and never silently drop a failed prompt.

For a binary task with $n$ examples and observed pass rate $\hat{p}=k/n$, the rough standard error under an independent Bernoulli model is $\sqrt{\hat{p}(1-\hat{p})/n}$. This is an explanatory statistical abstraction, not a claim that task examples are independent. At $n=100$ and $\hat{p}=0.80$, the expression is $\sqrt{0.8\cdot0.2/100}\approx0.04$, or four percentage points. A two-point budget is therefore smaller than one rough standard error in this illustrative sample. That tells us to either increase the sample, use paired bootstrap confidence intervals, or set the policy around an agreed uncertainty rule.

Paired evaluation is especially valuable: run the same prompt ID against reference and candidate, then analyze per-prompt disagreement. If both models fail the same prompt, it does not explain the regression. If only the candidate fails, the example is diagnostic. Keep the raw outputs for failures with redaction and retention controls.

## 5. Make provenance a first-class release artifact

The quality result is only as useful as its ability to answer “which bits produced it?” A quantized model directory can contain a tokenizer, configuration, packed weights, scales, calibration metadata, and runtime-specific kernels. A symlink or mutable object-store tag can make a later replay load different bytes.

![The evidence stack places the decision above results, inputs, runtime, quantizer settings, and the immutable model identity](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-5.webp)

The evidence stack is deliberately bottom-heavy. A decision cannot be stronger than the identity of the model and evaluator beneath it. Capture hashes for every large file or, better, a manifest hash over sorted relative paths and content hashes. Store the exact command line, environment, GPU name, driver, CUDA runtime, quantizer version, and evaluator commit.

```python
import json
import platform
import subprocess
from datetime import date

def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()

provenance = {
    "recorded_on": str(date.today()),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "git": command("git", "rev-parse", "HEAD"),
    "torch": __import__("torch").__version__,
    "cuda": __import__("torch").version.cuda,
}
print(json.dumps(provenance, sort_keys=True, indent=2))
```

This script is reader-reproducible metadata collection. It should run in the evaluation container, not on a developer laptop whose environment is not the production image. The date is intentionally generated at runtime. The post’s fixed frontmatter date is a series convention; an evaluation record needs its own date.

A good record also stores the prompt manifest hash. The manifest should contain stable IDs, not only prompt text, because a prompt can contain secrets and a text diff can be hard to review. Keep a private mapping from ID to content under the retention policy, and publish aggregate results with anonymized IDs where appropriate.

### Runtime parity matters

A reference in Transformers and a candidate in a custom INT4 kernel can differ in tokenizer normalization, BOS handling, rotary embedding implementation, attention backend, stopping criteria, or sampling. First compare them in a controlled configuration. Then separately measure the production runtime.

The [Transformers text-generation documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation), retrieved 2026-07-20, is the reference for generation controls; it is not evidence that a custom server has identical defaults. The [vLLM documentation](https://docs.vllm.ai/en/latest/), retrieved 2026-07-20, is the benchmark target for a production engine, not an excuse to claim parity without running the same prompt suite.

## 6. Add rollback as an automated, boring state transition

The safe response to a failing candidate is not to debate the metric in a release channel. It is to keep the known-good artifact addressable, mark the candidate blocked, and route new traffic to the last passing version. Existing streams need a policy: finish on the old version, cancel with a clear error, or drain before the switch. That is a product choice, but it must be explicit.

![The rollback tree checks perplexity first, then protected task budgets, and sends any failure to a deterministic block or rollback path](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-6.webp)

The tree’s first question is not speed. If a candidate is faster but violates a protected task contract, it cannot be promoted. The candidate can remain available for debugging under an isolated name. Do not delete it before the incident is understood; deletion destroys evidence.

```python
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    PROMOTE = "promote"
    BLOCK = "block"
    ROLLBACK = "rollback"

@dataclass(frozen=True)
class DeploymentState:
    active: str
    candidate: str

def release(state: DeploymentState, decision: GateDecision) -> tuple[DeploymentState, Action]:
    if decision.status == "promote":
        return DeploymentState(active=state.candidate, candidate=state.candidate), Action.PROMOTE
    return state, Action.ROLLBACK

state = DeploymentState(active="llama-3.1-8b-bf16", candidate="llama-3.1-8b-int4")
next_state, action = release(state, GateDecision("block", ("code_pass_rate:fail",)))
print(action.value, next_state.active)
```

The function is intentionally small. The real serving system needs an atomic alias update, health checks, drain handling, and an audit event. But the decision should remain pure: given a state and a gate result, the outcome is deterministic.

### Rollback budgets and canaries

A canary is useful for discovering runtime failures, not for excusing an incomplete offline gate. Start with a small traffic fraction only after the fixed matrix passes. During the canary, monitor task-specific online proxies where legally and operationally appropriate: structured-output parse failures, tool-call retries, user corrections, empty retrieval citations, and support tickets. These proxies are signals, not ground truth; route them back into offline prompt IDs.

Rollback should have a time budget. If the candidate causes a spike in parser failures, the system should be able to return to the previous alias without waiting for a human to reconstruct a command. The cost of keeping one prior artifact is storage. The cost of not keeping it is a longer quality incident.

## 7. Stress the gate against the failure modes that fooled it

The gate itself can fail. Here are the tests I would add before trusting it.

### Calibration leakage

If the task prompts are in the quantizer’s calibration set, the candidate may look better than it generalizes. Keep calibration, development, and holdout IDs disjoint. The quantizer may use representative text, but the final behavior suite should include hidden or access-controlled examples. Report the counts and the split rule.

### Chat-template drift

A model can appear to regress because the candidate received a different role separator or because one runtime applied a system prompt twice. Serialize the rendered token IDs for a small debug subset and compare them before comparing outputs. A tokenizer is part of the model contract. The [BPE tokenizer deep dive](/blog/machine-learning/large-language-model/bpe-tokenizer) and [choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) explain why byte-level details can matter to code and multilingual tasks.

### Sampling drift

Greedy output is stable but can miss distribution changes that appear under temperature and top-p. Run a deterministic pass for diagnosis, then a seeded stochastic pass for production-shaped behavior. Compare distributions or multiple samples where the product uses sampling. Do not claim a candidate is “the same” because one greedy completion matches.

### Long context and the needle

RAG and long-context quality should place a decisive fact at multiple positions. A quantized model may preserve short-context text while losing a low-frequency fact near the middle or end. The number of context lengths must be derived or cited for the chosen model; do not invent a context limit. The evaluator should record prompt token count, retrieved chunk IDs, and answer support evidence.

### Kernel-specific differences

The same packed checkpoint can run through different dequantization and accumulation paths. A fused INT4 kernel may accumulate in FP16, BF16, or FP32 depending on implementation. Treat the kernel and runtime as part of the candidate identity. If the same weights pass in one engine and fail in another, that is not a contradiction; it is two candidates.

### Hardware coverage

The fixed matrix includes the RTX 4090, L4, A100, and H100 because memory hierarchy and kernel support differ. A reader can reproduce the runtime measurements with `bench_quantized.py`; an honest expected range must be hardware-specific and treated as a hypothesis until collected. The gate should block an unevaluated hardware path rather than extrapolate from the RTX 4090 to an H100.

| Failure mode | Detection | Safe response |
|---|---|---|
| Calibration leakage | disjoint manifest hashes | replace holdout |
| Template drift | token-ID diff | fix harness, rerun |
| Sampling drift | fixed seeds and multiple samples | separate deterministic and stochastic budgets |
| Needle loss | position-stratified RAG set | block candidate or recalibrate |
| Kernel divergence | runtime in provenance key | evaluate every engine path |
| Missing GPU result | explicit `unevaluated` | do not promote |

### Honest latency measurement

Quality is the primary gate in this incident, but quantization is normally justified by memory or latency. Measure those claims with a separate script. Warm up the model, synchronize the device before timing, use CUDA events for GPU intervals, discard startup, and report steady-state p50 and p99. Record prompt length and generated length because prefill and decode have different costs. The [reproducible benchmark setup](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) and [roofline for a service](/blog/machine-learning/performance-engineering/the-roofline-for-your-service) cover the measurement discipline.

```python
import time
import torch

def timed_decode(fn, warmup: int = 5, trials: int = 20) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples

if __name__ == "__main__":
    print("Run with a loaded candidate and reference; report p50/p99, not one sample.")
```

This is a measurement harness skeleton, not a benchmark result. A reader running it should publish the script, commit, model hash, GPU, driver, batch, input/output lengths, and an expected range agreed before the run. Do not backfill an expected range after seeing a favorable result.

## 8. The release gate and its numbers with provenance

The gate needs a result table whose source is visible. Here is the shape I would put in CI. The example values are derived or synthetic policy examples, not measurements from this post.

| Quantity | Value or rule | Source |
|---|---:|---|
| INT4 payload per 128-weight group | $128\times4/8+2=66$ bytes | derived from stated packing assumption |
| Idealized BF16-to-INT4 payload ratio | $2/0.515625\approx3.88\times$ | derived from stated packing assumption |
| Matrix size | $3\times4\times4=48$ cells | derived from fixed matrix |
| Example code budget | candidate drop at most 0.02 | reproducible policy fixture |
| Example perplexity budget | candidate increase at most 0.20 | reproducible policy fixture |
| NVIDIA GPU memory and bandwidth | use the named product page | cited: NVIDIA pages, retrieved 2026-07-20 |
| GPTQ method | weight-only post-training quantization | cited: [GPTQ paper, October 2022](https://arxiv.org/abs/2210.17323) |
| Runtime behavior | p50/p99 from the supplied script | reproduce: `bench_quantized.py` |

Notice what is not in the table: “INT4 is 2x faster.” That claim would require a setup, a device, a batch, a prompt suite, a kernel, and a reader-reproducible script. It is not implied by four bits per weight. A simple bandwidth-bound estimate can be written as $T\approx B/\beta$, where $B$ is bytes moved and $\beta$ is sustained memory bandwidth. But sustained bandwidth is not the vendor peak, and decode also pays for activations, scales, synchronization, and attention. Use the formula to reason about a ceiling, not to manufacture a result.

#### Worked example: a task gate catches the quiet failure

Suppose the FP16 reference scores 80 of 100 code prompts, and the INT4 candidate scores 71 of 100. The reference pass rate is $80/100=0.80$ and the candidate is $71/100=0.71$. The delta is $0.71-0.80=-0.09$, or negative nine percentage points. Under the illustrative budget of 0.02, the candidate fails because $-0.09<-0.02$.

Now suppose perplexity is 8.40 for the reference and 8.31 for the candidate. The delta is $8.31-8.40=-0.09$, which passes a lower-is-better budget of 0.20 because $-0.09\le0.20$. Both statements can be true simultaneously: average loss improves while the code contract fails.

The arithmetic is enough to make the rollback explainable. No appeal to intuition is required, and no fabricated throughput number is needed to justify blocking the artifact.

### A compact CI command

The evaluator should emit a machine-readable result and a human-readable summary. A release job can then make promotion depend on the exit code.

```bash
python -m nanoserve.eval_gate \
  --reference artifacts/llama-3.1-8b-bf16 \
  --candidate artifacts/llama-3.1-8b-int4 \
  --manifest eval/manifests/wave-11-v1.json \
  --hardware "A100 80GB SXM" \
  --output reports/int4-a100.json

python -m nanoserve.verify_report reports/int4-a100.json \
  --require tasks,perplexity,provenance
```

The command names are the proposed `nanoserve` interface; the reader can implement them around the Python functions above. It should exit nonzero for `fail` and `unevaluated`, and zero only for a complete passing report. Deployment automation should consume the report’s signed digest, not parse terminal prose.

## The incident’s corrected timeline

The original story ends at “code quality fell.” The corrected process inserts evidence at every handoff.

![The incident timeline shows how a green perplexity result reached traffic before a task regression triggered rollback](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-4.webp)

At day zero, freeze the FP16 reference and task manifest. At quantization time, write the quantizer config and packed-weight hash. At evaluation time, run the matrix and store paired outputs for failures. At promotion time, require all protected cells to pass and attach the report digest to the deployment record. At canary time, watch online proxies and keep the old alias available. At rollback time, preserve the failed artifact for analysis.

This sequence also clarifies ownership. The quantization engineer owns the artifact and its metadata. The evaluation owner owns the prompt manifest and scoring code. The inference owner owns runtime parity and performance measurement. The release owner owns the promotion policy. One person can hold several roles in a small team, but the artifacts should remain separate enough to review.

## Case studies from public methods and this incident

### 1. GPTQ: a method is not a product contract

The GPTQ paper, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,” describes a one-shot weight quantization method and reports model-quality results under its experimental setup ([arXiv, submitted October 2022](https://arxiv.org/abs/2210.17323)). That is evidence for the method’s design and reported experiments. It is not evidence that a particular INT4 checkpoint, tokenizer, kernel, and task suite will satisfy a new product’s contract.

The engineering translation is to preserve the paper citation in the model card while adding a candidate-specific report. The report needs the quantizer implementation version, calibration data identity, group size, zero-point convention, packing format, and accumulation type. Without those, “GPTQ INT4” is a family name, not an executable identity.

### 2. PagedAttention: capacity wins create room for more evaluation

The PagedAttention paper explains how virtual-memory-like paging improves KV-cache utilization for serving ([vLLM paper, arXiv 2023](https://arxiv.org/abs/2309.06180)). The lesson for this case study is not to re-document paged attention. It is that a memory optimization changes what the server can admit. If INT4 frees weight memory, use some of that capacity for a broader evaluation or a safer canary. Do not spend every saved byte on more concurrency while leaving quality coverage unchanged.

### 3. SmoothQuant and activation sensitivity

SmoothQuant moves quantization difficulty between weights and activations using an equivalent transformation ([SmoothQuant paper, arXiv 2022](https://arxiv.org/abs/2211.10438)). Its existence is a reminder that “quantization” is not one error profile. Weight-only INT4, weight-and-activation INT8, FP8, and KV-cache quantization affect different tensors and kernels. The evaluation matrix should identify which path changed. A task regression can come from the candidate weights, activation scales, attention cache, or runtime accumulation.

### 4. FlashAttention: numerical changes deserve functional checks

FlashAttention’s paper presents an IO-aware exact attention algorithm that reduces memory traffic without changing the mathematical attention definition in its intended precision ([FlashAttention paper, NeurIPS 2022](https://arxiv.org/abs/2205.14135)). Even when a method is designed to preserve a computation, an implementation still needs parity and task tests. A kernel optimization and a quantization optimization may be individually reasonable yet interact through precision, ordering, or fallback paths.

## The release artifact: one immutable record

The final design is a small report that joins metric rows to budgets, evidence, and a decision. The grid below is the shape, not a screenshot of a run.

![A release grid joins metric rows to budgets and evidence before emitting a ship or rollback decision](/imgs/blogs/case-study-the-quantized-model-that-quietly-got-dumber-7.webp)

```json
{
  "schema": "nanoserve.eval.v1",
  "candidate": {"name": "llama-3.1-8b-int4", "sha256": "..."},
  "reference": {"name": "llama-3.1-8b-bf16", "sha256": "..."},
  "cell": {"model": "Llama-3.1-8B", "hardware": "A100 80GB SXM", "task": "code"},
  "metrics": [
    {"name": "code_pass_rate", "reference": 0.80, "candidate": 0.71, "budget": 0.02, "status": "fail"},
    {"name": "perplexity", "reference": 8.40, "candidate": 8.31, "budget": 0.20, "status": "pass"}
  ],
  "decision": "rollback",
  "provenance": {"manifest_sha256": "...", "evaluator_commit": "...", "seed": 17}
}
```

The JSON numbers are the worked example above, not a claim about a real checkpoint. In production, use decimal conventions consistently and include sample counts. If the task score is a percentage, either store `80` with a unit or `0.80` as a fraction; never mix them.

## When to reach for INT4, and when not to

Reach for INT4 when the memory budget is the real constraint, the target runtime has a tested kernel, and the fixed matrix shows that the task contracts survive. It is especially attractive when weight traffic dominates a decode step and the saved capacity enables useful batching or a larger context. The speed benefit must be measured on the actual hardware and workload; the bit count alone is not a benchmark.

Do not reach for it as a blind emergency fix for an OOM. First identify whether weights, KV cache, activations, fragmentation, or concurrency causes the failure. Quantizing weights will not solve a KV-cache capacity problem. The [KV-cache memory math](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache) is the relevant diagnostic for that path.

Do not ship it when the evaluator has no reference, the prompt manifest is mutable, the candidate is unevaluated on a supported GPU, or a protected task falls outside its budget. “We can watch it in production” is not a substitute for an offline gate when rollback may expose users to broken code or unsupported claims.

And do not build this release system instead of using vLLM unless the engineering goal is the engine itself or the product needs behavior vLLM cannot provide. The right benchmark target is the production engine you would otherwise run. A small `nanoserve` gate is valuable because it makes the contract visible; it is not a reason to replace mature scheduling, kernels, and observability without evidence.

## Key takeaways

- Perplexity is an average next-token fit metric, not a complete behavior contract.
- A quantized artifact is defined by weights, scales, packing, kernels, runtime, tokenizer, and evaluator—not by “INT4” alone.
- Compare the candidate with an immutable FP16 or BF16 reference on paired prompt IDs.
- Keep chat, RAG, code, and translation contracts in the matrix, even when one corpus score is green.
- Treat missing cells as `unevaluated`, and make `unevaluated` block promotion.
- Choose thresholds from product risk and sample uncertainty; do not copy a convenient number without naming it as policy.
- Store hashes, seeds, templates, runtime versions, and evaluator commits with every result.
- Separate quality gates from latency measurements, and publish the setup for every performance number.
- Keep the last passing artifact addressable and make rollback an atomic, tested state transition.
- Use the memory win from quantization to buy capacity or evaluation coverage, not permission to remove the gate.

## Further reading

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), arXiv, October 2022.
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), arXiv, 2022.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), arXiv, 2023.
- [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving).
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

<figure class="blog-anim">
<svg viewBox="0 0 760 190" role="img" aria-label="A candidate moves from perplexity approval to task regression and rollback" style="width:100%;height:auto;max-width:860px">
<style>
.j1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.j1-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.j1-sub{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.j1-dot{fill:var(--accent,#6366f1)}
.j1-alert{fill:#ef4444;opacity:.15}
@keyframes j1-travel{0%,18%{transform:translateX(0);opacity:1}38%,55%{transform:translateX(210px);opacity:1}72%,100%{transform:translateX(450px);opacity:0}}
@keyframes j1-alert{0%,48%{opacity:0}55%,100%{opacity:.9}}
.j1-move{animation:j1-travel 8s ease-in-out infinite}
.j1-alert{animation:j1-alert 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.j1-move,.j1-alert{animation:none}.j1-move{transform:translateX(210px);opacity:1}.j1-alert{opacity:.9}}
</style>
<rect class="j1-box" x="25" y="60" width="145" height="70" rx="10"/>
<rect class="j1-box" x="210" y="60" width="145" height="70" rx="10"/>
<rect class="j1-box" x="395" y="60" width="145" height="70" rx="10"/>
<rect class="j1-box" x="580" y="60" width="145" height="70" rx="10"/>
<text class="j1-label" x="97" y="88">INT4 candidate</text><text class="j1-sub" x="97" y="110">artifact hashed</text>
<text class="j1-label" x="282" y="88">PPL green</text><text class="j1-sub" x="282" y="110">approval signal</text>
<text class="j1-label" x="467" y="88">Task −9 pt</text><text class="j1-sub" x="467" y="110">regression found</text>
<text class="j1-label" x="652" y="88">Rollback</text><text class="j1-sub" x="652" y="110">FP16 restored</text>
<rect class="j1-alert" x="395" y="60" width="145" height="70" rx="10"/>
<circle class="j1-dot j1-move" cx="97" cy="150" r="9"/>
</svg>
<figcaption>The candidate advances through the release states; motion makes the delayed task regression and rollback explicit.</figcaption>
</figure>
