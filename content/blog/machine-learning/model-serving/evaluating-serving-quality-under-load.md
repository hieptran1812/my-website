---
title: "Evaluating serving quality under load: the third SLO nobody gates on"
date: "2026-07-04"
publishDate: "2026-07-04"
description: "Your serving optimizations can silently break the model's output. Learn to treat output correctness as a first-class SLO with a pre-ship regression gate, logit-drift checks, determinism tests, and online quality monitoring."
tags:
  [
    "model-serving",
    "inference",
    "llm",
    "evaluation",
    "quantization",
    "regression-testing",
    "kl-divergence",
    "lm-eval-harness",
    "vllm",
    "observability",
    "quality-assurance",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/evaluating-serving-quality-under-load-1.webp"
---

A team I worked with shipped an FP8 build of their 8B chat model on a Friday. FP8 — eight-bit floating point, the native tensor-core format on H100 — cut their per-token decode latency by 35% and roughly doubled the tokens per second they could push through each GPU. The latency dashboard turned green. The throughput dashboard turned green. The cost-per-million-tokens number dropped by nearly half. Everyone went home happy.

By Monday, the support queue had a cluster of tickets that all rhymed: "the assistant used to solve my math homework and now it gets it wrong," "the code it writes doesn't run anymore," "it feels dumber." Nobody had touched the prompt. Nobody had touched the weights — not really. They had only changed the *numerics* of how those weights were multiplied. And that was enough to knock 4.2 points off the model's grade-school math accuracy, a regression that no latency percentile, no GPU-utilization graph, and no error-rate alert would ever show you. The outputs were still well-formed, still fast, still 200-OK. They were just, more often, wrong.

This is the failure mode that the entire model-serving discipline is quietly bad at. We have decades of tooling for latency and throughput. We have p99 histograms, queue-depth alerts, Little's Law capacity models, and autoscalers that react in seconds. What we usually do not have is an equivalent discipline for the *correctness of the output* — the one property the user actually came for. Every optimization in the serving stack, from quantization to KV-cache compression to speculative decoding to a new attention kernel, is a place where accuracy can silently regress while every operational metric stays perfectly healthy.

![Matrix comparing the three serving SLOs — latency, throughput, and quality — across what each measures, its primary metric, how you detect a regression, and its gate or tolerance.](/imgs/blogs/evaluating-serving-quality-under-load-1.webp)

The argument of this post is simple: **output quality is the third SLO**, and it deserves exactly the same treatment as latency and throughput — a metric, a detector, and a gate with a tolerance. The recurring spine of this series is `Model → Packaging → Runtime → Server → Infrastructure → Observability → Scale`, and the recurring trade is the SLO triangle of latency, throughput, and cost. Quality is the axis that turns that triangle into a tetrahedron. You can buy latency and throughput and cost with an optimization — but the currency you spend is often correctness, and if you are not measuring it, you do not know the price you paid.

By the end of this post you will be able to: build a pre-ship regression gate that runs a fixed benchmark against your optimized server and blocks the deploy on a tolerance; compute a logit-drift signal (KL divergence between the reference and optimized model) that catches subtle regressions accuracy alone will miss; reason quantitatively about how much accuracy you can trade for throughput; wire online quality proxies that work on production traffic where you have no ground truth; and test whether heavy load itself — batching, preemption, fallback models — changes your outputs. Let us start with where the changes come from.

## 1. The seven places serving silently changes your outputs

If training produces a set of weights, serving is the machinery that turns weights plus a prompt into tokens. Almost every performance optimization in that machinery perturbs the arithmetic, and a perturbed arithmetic can produce a different token. Here are the seven most common sources, roughly in order of how often they bite.

**Weight quantization (FP8, INT8, INT4, AWQ, GPTQ).** Quantization stores the weight matrices in fewer bits — 4-bit integers instead of 16-bit floats, say — so each decode step reads less data from high-bandwidth memory (HBM) and runs faster on the memory-bound decode path. The rounding is lossy by construction. Good schemes like AWQ (Activation-aware Weight Quantization) and a well-calibrated FP8 are *near*-lossless; a badly calibrated scale factor or a naive round-to-nearest INT4 can cost several points of accuracy. Quantization is the single largest source of silent quality regressions in serving.

**KV-cache quantization.** The KV cache stores the keys and values of every past token so you do not recompute attention from scratch each step; it is the dominant memory consumer in LLM serving. Quantizing it to FP8 or INT8 frees memory for larger batches, but it compresses exactly the state that long-context attention depends on. The impact is usually small on short prompts and grows with context length — a regression that a 512-token benchmark will completely miss.

**Speculative decoding.** A small draft model proposes several tokens, the big target model verifies them in one forward pass, and accepted tokens are emitted for free. The rejection-sampling scheme is *provably* distribution-preserving: the accepted sequence is drawn from exactly the target model's distribution. In theory it is lossless. In practice, greedy variants, numerical mismatches between draft and verify passes, and implementation bugs in the acceptance test can all leak a difference. "Should be lossless" is a hypothesis to verify, not a fact to assume.

**Chunked prefill.** Long prompts are split into fixed-size chunks and interleaved with ongoing decode steps so a single 8k-token prompt does not stall everyone else's tokens. Splitting the prefill changes the order of some reductions and can change which numerically-adjacent token wins at a decision boundary. The effect is tiny but nonzero.

**A different attention or GEMM kernel backend.** FlashAttention-2, FlashInfer, xFormers, a Triton kernel, a TensorRT engine — each computes the same mathematical function with a different tiling, accumulation order, and sometimes a different accumulation dtype. Two backends that are each "correct" can produce logits that differ in the last few bits, and that is occasionally enough to flip a token.

**Batching numerics and nondeterminism.** Floating-point addition is not associative: `(a + b) + c` does not always equal `a + (b + c)`. GPU reductions — the softmax denominator in attention, the accumulation in a matmul, the all-reduce across tensor-parallel ranks — sum their terms in an order that depends on the batch size and how many sequences are running concurrently. The same prompt at batch 1 and at batch 32 can therefore produce different logits, and under greedy decoding a hair's-breadth logit difference near a boundary flips a token that then compounds. This is the reason "quality under load" is its own problem, and we will return to it.

**Sampling parameters.** Temperature, top-p, top-k, repetition penalty, and the RNG seed are part of the serving configuration, not the model. A default that drifts between your eval harness and your production server — eval at temperature 0, prod at temperature 0.7 — will make your gate measure a different model than the one users hit.

It is worth understanding *why* quantization, the biggest source, is so easy to get subtly wrong, because it sharpens your intuition for what a gate must catch. Transformer activations are famously heavy-tailed: a small number of feature channels carry values one or two orders of magnitude larger than the rest, and those outlier channels turn out to matter enormously for the model's behavior. A quantization scheme has to pick a scale factor that maps the float range onto the available integer or low-precision-float levels. Choose a scale that covers the outliers and the vast majority of normal-magnitude values get crushed into a few levels, losing precision where most of the computation happens. Choose a scale tuned to the normal values and the outliers clip, discarding exactly the signals the model relies on most. AWQ resolves this by protecting the roughly 1% salient channels; SmoothQuant migrates the difficulty from activations into weights by a per-channel rescale; FP8's floating-point format handles a wider dynamic range than INT8 at the same bit width. Every one of these is a bet about where the model's information lives, and a bet that is slightly wrong for *your* model produces a build that looks fine on a smoke test and fails on the reasoning-heavy tail — which is precisely the failure the gate exists to catch.

The uncomfortable property shared by all seven is that they are *invisible to operational monitoring*. The following table lays out each source, its expected quality impact, how you detect it, and a sane starting tolerance.

| Optimization | Expected quality impact | How to detect it | Starting tolerance |
|---|---|---|---|
| FP8 weights (calibrated) | Near-lossless (< 0.5pp) | Benchmark + mean per-token KL | Δacc ≤ 1pp, KL ≤ 0.01 |
| INT4 / AWQ weights | Small (0.5–2pp) | Benchmark + KL, long-context set | Δacc ≤ 1.5pp, KL ≤ 0.02 |
| Naive / miscalibrated quant | Large (2–6pp), silent | Benchmark, full test set | Block on Δacc > 1pp |
| KV-cache FP8/INT8 | Grows with context length | Long-context benchmark + KL | Δacc ≤ 1pp at 8k ctx |
| Speculative decoding | ~0 in theory, verify | Logit/output equivalence vs target | Exact match at temp 0 |
| New attention/GEMM kernel | Last-bit logit differences | KL drift, determinism test | KL ≤ 0.005 |
| Batching / concurrency | Nondeterministic, load-dependent | Determinism test across batch sizes | Bitwise or KL ≤ 0.005 |

![Before-and-after contrast showing a ship-blind FP8 deploy causing a silent four-point accuracy drop versus a regression-gated deploy that catches the same drop and blocks the promotion.](/imgs/blogs/evaluating-serving-quality-under-load-2.webp)

The figure above shows the two worlds side by side. On the left, the same FP8 build that shaved 35% off latency also dropped grade-school math (GSM8K) accuracy from 78.4% to 74.2% — a 4.2-point regression, silent, that reached users as roughly one wrong answer in twelve. On the right, a pre-ship eval gate ran the full benchmark, saw the 4.2-point delta blow past a 1-point tolerance, and blocked the promotion so an engineer could go recalibrate the FP8 activation scales before anyone was affected. The build was not bad because it was FP8; FP8 is usually near-lossless. It was bad because of a miscalibrated scale, and the only thing standing between that mistake and production is a gate. Let us build one.

## 2. The pre-ship regression gate

The core discipline is unglamorous and effective: before you promote an optimized server, run a fixed benchmark against it, compare the score to a stored reference, and refuse to promote if the drop exceeds a tolerance. This is the same idea as a unit test, applied to a probabilistic system, with a tolerance band instead of an exact assertion.

The industry-standard tool for the benchmark half is EleutherAI's `lm-evaluation-harness` (usually called `lm-eval`), the same harness behind the Hugging Face Open LLM Leaderboard. It knows how to format hundreds of tasks — MMLU (multiple-choice knowledge across 57 subjects), GSM8K (grade-school math word problems), HumanEval (Python code generation), TruthfulQA, and many more — and it can score them against a model exposed as an OpenAI-compatible HTTP endpoint. That last part matters: you want to evaluate the *actual serving path*, the vLLM or TGI server with all its kernels and batching, not a pristine `transformers` model in a notebook. The bug you are hunting lives in the serving path.

Here is the gate's first half — start the optimized server, then point the harness at it.

```bash
# 1. Start the OPTIMIZED server (FP8) exposing an OpenAI-compatible API.
#    Pin the seed so the run is reproducible; pin max-model-len so the
#    eval context matches production.
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --port 8000 \
  --max-model-len 4096 \
  --seed 0 \
  --disable-log-requests &

# Wait for the server to report ready.
until curl -sf http://localhost:8000/health >/dev/null; do sleep 2; done

# 2. Run lm-eval against the LIVE endpoint. local-chat-completions speaks the
#    OpenAI /v1/chat/completions protocol, so this scores the real serving path.
lm_eval \
  --model local-chat-completions \
  --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
  --tasks gsm8k,mmlu \
  --num_fewshot 5 \
  --apply_chat_template \
  --batch_size 16 \
  --output_path results/fp8/
```

Run the identical command against your reference server (the BF16 build you trust) into `results/fp16/`, and you have two JSON files with comparable scores. The comparison is where the tolerance lives, and getting the tolerance right requires understanding the statistics — which is the next section, because a tolerance chosen without the noise math is either so tight it blocks healthy builds or so loose it waves regressions through.

A few operational notes that save you a bad afternoon. Set `num_concurrent` high enough to actually load the server — a gate that runs at batch 1 will not catch batching-dependent regressions (see §6). Pin `temperature=0` (greedy) for the accuracy tasks so run-to-run variance from sampling does not contaminate the measurement; you want the only variance to be the harness's own item sampling. And log the exact server flags, model revision hash, and harness version into the results directory, because a quality number without the config that produced it is not reproducible and therefore not a gate.

The single most common way this gate lies to you is a **chat-template mismatch**. The harness has to wrap each prompt in the model's chat template — the special tokens and role markers the model was trained on — and if the template the harness applies differs from the one your production server applies, you are evaluating a subtly different input than users send. The symptom is a gate that shows a "regression" on a build that is actually fine, or worse, passes a build that is actually broken, because the template difference swamps the quantization difference. Pin the template explicitly, apply it identically in the eval and in production, and sanity-check by diffing the exact tokenized request the harness sends against a real production request for the same prompt. A second, subtler pitfall: **answer extraction.** GSM8K is scored by pulling the final number out of the model's response with a regex; if quantization changes the model's formatting (it stops writing "The answer is 42" and starts writing "42."), the extractor can fail and report a regression that is a parsing artifact, not a reasoning loss. Always eyeball a handful of "wrong" items before you trust a red gate — sometimes the model is right and your extractor is wrong.

## 3. Choosing the benchmark and the metric

A gate is only as good as the benchmark behind it, and the right benchmark is a function of what your model actually does. Running MMLU on a model that only ever writes SQL is theater. The task shape picks both the dataset and the scoring metric.

![Decision tree that routes a serving quality eval by task shape — reasoning tasks to GSM8K with exact-match and drift, code to HumanEval pass@1, and fixed-label tasks to MMLU accuracy plus logprob.](/imgs/blogs/evaluating-serving-quality-under-load-8.webp)

The tree above walks the three big branches. **Free-form reasoning** (math, multi-step logic) wants an exact-match metric on the final answer plus a drift check on the logits, because the reasoning chain is where quantization damage shows up first. GSM8K's 1,319-item test set is the canonical choice. **Code generation** wants `pass@1` — does the generated program pass the hidden unit tests — measured on HumanEval's 164 problems or the larger MBPP. Code is a brutal, honest metric: a single wrong token often means a non-compiling program, so code benchmarks catch quantization damage that prose benchmarks smooth over. **Fixed-label tasks** (classification, multiple choice) want accuracy plus a logprob check: MMLU scores by comparing the model's logprob of each answer letter, so it is already a logprob-space metric and is sensitive to distributional shift.

Two principles cut across all three branches. First, **use a task-specific golden set, not only academic benchmarks.** If you serve a customer-support model, your most valuable eval is 300–1,000 real (anonymized) production prompts with known-good responses, scored the way your product actually judges quality. Academic benchmarks catch gross regressions; a golden set catches the ones that matter to your users. Second, **freeze the set.** The gate compares optimized-versus-reference on the *same* items with the *same* few-shot examples in the *same* order. The moment the eval set drifts, the comparison is meaningless — you are measuring two different things and calling the difference a regression.

**Building the golden set is worth doing carefully**, because it is the asset the whole gate rests on. Sample real production prompts across the traffic distribution — not just the happy path, but the long prompts, the multi-turn conversations, the edge-case formats, and the categories where your model is known to be weak, since those are where a regression will land first. Aim for a few hundred to a couple thousand items, stratified so each important category has enough items to resolve a regression within it (the noise math applies per-stratum, so a 20-item category cannot gate anything). For each item, store a reference output — either a human-written known-good answer, or the output of the model you currently trust in production, captured once and frozen. Anonymize aggressively: strip PII before anything enters the eval set, because a golden set is a durable artifact that will be copied into CI logs and artifact stores. And version it: when you add or retire items, bump the set's version, because a quality number is only comparable against numbers from the *same* set. A living golden set that grows as you discover new failure modes is one of the highest-leverage pieces of ML infrastructure a serving team owns.

Modern services also serve **multi-turn and agentic** traffic, and evaluating those is harder than single-turn benchmarks admit. A regression can leave single-turn quality untouched while breaking tool-call formatting, or degrade the model's ability to stay coherent across a ten-turn conversation. If you serve agents, your golden set needs multi-turn trajectories and tool-call schemas scored for structural validity, not just final-answer accuracy — a model that quantization pushed from 99% to 96% valid tool calls will look fine on GSM8K and fall over in production, because a 4% tool-call failure rate compounds across the many calls a single agent task makes.

There is a subtlety worth internalizing before we do the math: accuracy is a coarse metric. It collapses the model's entire probability distribution over the vocabulary into a single bit — right or wrong — and then averages those bits. That collapse throws away almost all the information in the model's output and, as we will see, buries small regressions inside sampling noise. The distribution itself carries a far sharper signal.

## 4. The mechanics: sampling noise, the tolerance band, and why logit drift is sharper

This is the section that makes the difference between a gate that works and a gate that lies to you. Two numbers govern everything: how noisy an accuracy measurement is, and how much sharper a distributional measurement is.

### Accuracy is a coin-flip estimate, and it has a standard error

An accuracy score is an average of Bernoulli trials. You run `n` benchmark items; each is right (1) or wrong (0) with some true probability `p`; the observed accuracy is the sample mean. The standard error of that mean — the one-sigma noise band you get from re-running on a different sample of items — is the standard error of a proportion:

$$\mathrm{SE} = \sqrt{\frac{p\,(1-p)}{n}}$$

Plug in real numbers. At a true accuracy of `p = 0.77` on `n = 100` items:

$$\mathrm{SE} = \sqrt{\frac{0.77 \times 0.23}{100}} = \sqrt{0.00177} \approx 0.042$$

That is **±4.2 percentage points at one sigma**, and roughly **±8.3 points at the 95% level** (two sigma). Read that again: on a 100-item eval, an *observed* drop from 78% to 76% is well inside the one-sigma noise. It is statistically indistinguishable from having drawn a slightly harder sample of 100 questions. If your gate fires on a 2-point drop at n = 100, it will false-positive constantly on healthy builds and miss real regressions in the noise. A 100-item gate cannot resolve a 2-point regression, period.

Now increase `n`. The standard error shrinks as `1/√n`, so it costs you a hundredfold more items to tighten the band tenfold. On the full GSM8K test set, `n = 1319` at `p ≈ 0.78`:

$$\mathrm{SE} = \sqrt{\frac{0.78 \times 0.22}{1319}} = \sqrt{0.00013} \approx 0.0114$$

That is **±1.14 points at one sigma**. Now the picture is completely different. The 4.2-point FP8 regression from the intro is `4.2 / 1.14 ≈ 3.7` sigma — an event with probability well under one in a thousand under the null hypothesis of "no change." That is a real regression, and a 1,319-item gate resolves it easily. The lesson is mechanical: **your tolerance band must be derived from your item count, not guessed.** A defensible gate sets the tolerance at `k · SE` for `k = 2` or `3`, or equivalently picks `n` large enough that the smallest regression you care about is at least two or three sigma above the noise. Set `k` too small and you block good builds; too large and you ship bad ones.

#### Worked example: sizing the gate

You want to catch any regression of 1 point or more on GSM8K, with a false-positive rate under about 2%. A 2-point false-positive threshold means your tolerance is roughly two sigma, so you need `2 · SE ≤ 1pp`, i.e. `SE ≤ 0.5pp = 0.005`. Solve for `n`:

$$n \geq \frac{p(1-p)}{\mathrm{SE}^2} = \frac{0.78 \times 0.22}{0.005^2} = \frac{0.1716}{0.000025} \approx 6864$$

So resolving a 1-point regression cleanly needs roughly 7,000 items, not 1,319 — you would augment GSM8K with additional math problems, or accept a coarser 2–3 point resolution on the 1,319-item set. This is the kind of calculation that turns "we run a benchmark" into "we run a benchmark that can actually detect the thing we care about." The alternative to buying more items is to stop measuring a coarse 0/1 outcome and start measuring the distribution.

### Logit drift: KL divergence is a lower-variance signal

Here is the key insight that upgrades a gate from adequate to sharp. The model does not emit a bit; it emits a probability distribution over the entire vocabulary at every position. Comparing those distributions between the reference and the optimized model uses *all* of that information, so it is a far lower-variance estimator of "did the model change" than accuracy is.

The standard measure is the Kullback–Leibler (KL) divergence. For a reference distribution `P` and an optimized distribution `Q` over the vocabulary `V` at a token position:

$$D_{\mathrm{KL}}(P \,\|\, Q) = \sum_{v \in V} P(v)\, \log \frac{P(v)}{Q(v)}$$

KL is zero when the distributions are identical and grows as they diverge. You compute it teacher-forced — feed the same fixed text through both models, read off the per-position distributions, and average the per-token KL across a calibration set of a few hundred sequences. Because each token contributes a full-distribution comparison rather than a single right/wrong bit, the mean-KL estimate has dramatically lower variance than accuracy for the same number of tokens. A few hundred sequences of a few hundred tokens each gives you tens of thousands of per-token comparisons.

The magnitudes you learn to recognize: a well-calibrated FP8 build against its BF16 reference sits around a mean per-token KL of `0.002–0.005`. A healthy AWQ INT4 build sits around `0.01–0.02`. A miscalibrated quantization or a genuinely lossy KV-cache scheme pushes mean KL to `0.04` and up — and it shows this at a few hundred sequences, long before the accuracy metric moves outside its own noise band. KL sees the regression building before it flips enough argmax decisions to register as wrong answers.

![Before-and-after contrast showing an accuracy-only check missing a subtle regression inside its noise band while a KL-divergence check on the logits catches the same drift immediately.](/imgs/blogs/evaluating-serving-quality-under-load-7.webp)

#### Worked example: the drift catch

An AWQ build is evaluated on a 100-item slice of your golden set to keep the gate fast. It scores 76% against the reference's 78%. As we computed, the one-sigma band at n = 100 is ±4.2 points, so a 2-point drop is buried — the accuracy gate passes. But the same 100 items, run teacher-forced, give a mean per-token KL of 0.043 against the reference, versus a baseline of 0.008 for a healthy build. That is a five-fold jump, far outside anything a healthy optimization produces, and the drift gate blocks. When the team then re-ran accuracy on the full 1,319-item golden set, the true drop was 2.8 points — a real regression the 100-item accuracy check could never have resolved, but the KL check flagged from the small sample. The distribution knew before the score did.

Here is a compact script that computes the drift signal offline, using vLLM's `prompt_logprobs` to read the per-position distribution of both models on the same fixed texts.

```python
# drift_check.py — mean per-token KL between a reference and an optimized build.
# Teacher-forced: feed the SAME texts through both models and compare the
# per-position top-k distributions. Emits mean_kl for the CI gate to assert on.
import json
import math
from vllm import LLM, SamplingParams

# A frozen calibration set: a few hundred representative sequences.
with open("golden/calibration_texts.json") as f:
    TEXTS = json.load(f)            # list[str], e.g. 300 sequences

TOP_K = 20                          # vLLM returns top-k logprobs per position


def per_position_dists(model_id, quantization=None):
    """Return, for each text, a list of {token_id: logprob} dicts (one per token)."""
    llm = LLM(model=model_id, quantization=quantization, seed=0,
              max_logprobs=TOP_K, enforce_eager=True)
    # max_tokens=0 + prompt_logprobs asks for the logprobs of the PROMPT tokens
    # themselves under the model — a pure teacher-forced read, no generation.
    sp = SamplingParams(temperature=0.0, max_tokens=0, prompt_logprobs=TOP_K)
    outs = llm.generate(TEXTS, sp)
    dists = []
    for out in outs:
        seq = []
        for pos in (out.prompt_logprobs or []):
            if pos is None:         # first token has no conditional distribution
                continue
            seq.append({tid: lp.logprob for tid, lp in pos.items()})
        dists.append(seq)
    return dists


def kl_topk(p_logprobs, q_logprobs):
    """Approximate KL(P||Q) over the union of top-k supports, renormalized."""
    support = set(p_logprobs) | set(q_logprobs)
    floor = math.log(1e-8)          # mass for tokens outside a model's top-k
    p = {t: p_logprobs.get(t, floor) for t in support}
    q = {t: q_logprobs.get(t, floor) for t in support}
    # renormalize each to a proper distribution over the shared support
    zp = math.log(sum(math.exp(v) for v in p.values()))
    zq = math.log(sum(math.exp(v) for v in q.values()))
    kl = 0.0
    for t in support:
        lp, lq = p[t] - zp, q[t] - zq
        kl += math.exp(lp) * (lp - lq)
    return kl


ref = per_position_dists("meta-llama/Llama-3.1-8B-Instruct")
opt = per_position_dists("meta-llama/Llama-3.1-8B-Instruct", quantization="fp8")

kls = [kl_topk(pr, po)
       for seq_r, seq_o in zip(ref, opt)
       for pr, po in zip(seq_r, seq_o)]
mean_kl = sum(kls) / len(kls)
print(f"mean per-token KL: {mean_kl:.5f} over {len(kls)} tokens")
json.dump({"mean_kl": mean_kl, "n_tokens": len(kls)},
          open("results/fp8/drift.json", "w"))
```

The top-k KL is an approximation — you only see each model's top 20 tokens, not the full vocabulary — but for detecting drift it is more than sharp enough, and it is what you can get from a running server. If you control both models offline and want the exact KL, read the full logits with a forward hook and skip the top-k renormalization. For most gates, top-k is the right trade.

### Two refinements: paired tests and the right divergence

Two statistical refinements make the accuracy half of the gate sharper without buying more items. First, **use a paired test.** Your optimized and reference models are evaluated on the *same* items, so the two accuracy numbers are not independent — they agree on most questions and disagree on a few. The right test is McNemar's test on the discordant pairs: count the items the reference gets right and the optimized gets wrong (`b`), and the items the optimized gets right and the reference gets wrong (`c`), and test whether `b` and `c` differ by more than chance. Because it looks only at the questions where the two models actually disagree, McNemar's test has far more power than comparing two independent proportions — it can call a 1-point regression significant on a set where the unpaired two-sample test would shrug. Practically: log per-item correctness for both builds, not just the aggregate scores, so you can run the paired test instead of throwing away the pairing.

Second, **KL is not the only divergence, and it is not always the best one.** KL is asymmetric and unbounded — a single position where the optimized model puts near-zero mass on a token the reference loved contributes an enormous term, which makes mean-KL sensitive to outliers. Three alternatives are worth knowing. **Jensen–Shannon divergence** is the symmetric, bounded cousin of KL (it maxes out at `log 2`), which makes its mean stable and easy to threshold. **Total variation distance** — half the L1 distance between the distributions — is the most interpretable: it is the largest difference in probability the two models assign to any event, and a total-variation of 0.1 means "the models disagree on at most 10% of the probability mass." And **top-1 agreement rate** — the fraction of positions where both models' argmax token matches — is the crudest but most operational: under greedy decoding it directly predicts how often the served output will differ. A healthy quantization keeps top-1 agreement above 99%; a build that drops below 97% will visibly change outputs. Pick one primary divergence for the gate (mean symmetric-KL or JS is a good default) and log top-1 agreement alongside it as the human-legible companion number.

### The named before-and-after: a representative gate report

Putting the two signals together, here is what a gate report looks like across a family of optimizations for a representative 8B instruct model on an H100 80GB. These are illustrative figures — a worked example of the shape of the data, not a benchmark citation — but they are consistent with published quantization results and with the mechanics above.

| Build | Precision | GSM8K (n=1319) | ΔGSM8K | Mean per-token KL | MMLU | Decode tok/s (H100) | Verdict |
|---|---|---|---|---|---|---|---|
| Reference | BF16 | 78.4% | — | 0 (self) | 68.1% | 1,050 | baseline |
| FP8 E4M3 (calibrated) | FP8 | 77.9% | −0.5pp | 0.004 | 67.9% | 1,780 | ship |
| AWQ W4A16 | INT4 | 77.1% | −1.3pp | 0.013 | 67.2% | 1,520 | review |
| GPTQ W4A16 (bad calib) | INT4 | 75.3% | −3.1pp | 0.041 | 65.8% | 1,510 | block |
| KV-cache FP8 | BF16 w / FP8 KV | 78.2% | −0.2pp | 0.006 | 68.0% | 1,180 | ship |
| FP8 (miscalibrated) | FP8 | 74.2% | −4.2pp | 0.052 | 65.1% | 1,780 | block |

Read the verdicts against the tolerances. The calibrated FP8 build loses 0.5 points — inside the 1.14-point one-sigma band, so statistically indistinguishable from zero — with a KL of 0.004, and it buys a 70% throughput increase. That ships. The miscalibrated FP8 at the bottom is the intro's Friday-afternoon disaster: same throughput win, but −4.2 points at 3.7 sigma and KL 0.052. That blocks, loudly. The interesting case is AWQ: only −1.3 points, borderline on accuracy, but its KL of 0.013 is just over the 0.01 tolerance, which is why it lands in "review" — you would run the full golden set and a long-context slice before deciding. This is the whole discipline on one page: a metric, a detector, a tolerance, a verdict.

### The accuracy-for-throughput trade is a decision, not a default

The reason quality is genuinely the *third* SLO, and not just a red-light on a dashboard, is that it trades against the other two. An optimization that costs a little accuracy and buys a lot of throughput is often a good deal — the whole reason you quantize is that the accuracy price is small and the throughput and cost wins are large. The gate does not exist to reject every regression; it exists to make the trade *visible and deliberate* instead of silent and accidental. So the tolerance is not "zero accuracy loss." It is a budget: how much correctness are you willing to spend, and what do you demand in return.

Frame it as a Δ-gate, in the style of a competition leaderboard where a submission is accepted only if it beats the baseline by more than the noise. Here the submission is an optimized build, the "score" is throughput or cost-per-token, and the constraint is that quality must not fall by more than a budget `Δ_max`. Set `Δ_max = k · SE` from the noise math — pick `k = 2` so the budget is two sigma, meaning you will only ever "spend" accuracy you can actually measure. Then the accept rule is: **ship the optimization if and only if `Δquality ≤ Δ_max` AND it delivers a throughput or cost win above your threshold.** An optimization that costs quality inside the budget *and* delivers the win is accepted; one that costs quality it cannot pay for, or that delivers no win, is rejected.

#### Worked example: spending the accuracy budget

Your product owner sets the quality budget at 1.5 points of GSM8K — the most correctness the business will trade for infrastructure savings — and your gate runs on the full 1,319-item set where one sigma is 1.14 points, so `Δ_max = 1.5pp` is a hair over one sigma. Three candidates land on your desk. The **calibrated FP8** build costs 0.5 points (well inside budget, and inside the noise) and buys a 70% decode-throughput increase, which at your volume cuts serving cost roughly 40%. Accept — this is the deal you quantized for. The **GPTQ** build costs 3.1 points, more than double the budget and 2.7 sigma of real regression, for a 44% throughput win. Reject — the win is real but you cannot pay for it; a 3.1-point drop is thousands of newly-wrong answers a day. The **AWQ** build costs 1.3 points (inside budget) for a 45% win, but its KL of 0.013 exceeds the drift tolerance, which means the distribution moved more than the accuracy number admits — a warning that the drop may be larger than 1.3 points on traffic your 1,319-item set underrepresents. The decision: accept AWQ *conditionally*, run the full golden set and an 8k-context slice to confirm the accuracy cost stays inside budget on the long-context traffic where INT4 damage concentrates, and only then promote. The budget turned three fuzzy "it depends" builds into three clean decisions, and it did so with numbers a business owner can sign off on.

The general principle: the quality budget is a business input, the noise floor is a statistical input, and the gate is where they meet. Neither an engineer eyeballing outputs nor a product owner reading a latency graph can make this trade alone — it needs the accuracy number, the drift number, the throughput number, and the budget, on one page.

## 5. Wiring the gate into CI

A gate that a human remembers to run is a gate that will be skipped on the Friday it matters most. The gate has to be code that blocks a merge. The pattern is a CI job that starts the optimized server, runs the benchmark and the drift check, and then asserts the results against tolerances in a test that fails the pipeline.

![Graph of a CI quality gate — an optimized build fans out to a parallel accuracy eval and a logit-drift check, then a single tolerance decision routes it to promote or to block and page.](/imgs/blogs/evaluating-serving-quality-under-load-4.webp)

The graph shows the control flow. The optimized build fans out to two checks that run in parallel — the accuracy eval and the KL-drift computation against the frozen fp16 reference — and both feed a single tolerance decision. If accuracy and KL are both within tolerance, the build promotes to staging. If either fails, the build is blocked and the pipeline pages whoever pushed it. There is exactly one decision node, and it has exactly two exits, because a gate that has a "well, maybe" exit is not a gate.

The assertion half is an ordinary pytest module. It reads the JSON that the eval and drift steps wrote and turns tolerances into failing tests with actionable messages.

```python
# test_quality_gate.py — fails CI if the optimized build regresses beyond tolerance.
import json
import pytest

TOL_ACC_PP = 1.0      # max allowed accuracy drop, in percentage points
TOL_KL     = 0.01     # max allowed mean per-token KL vs reference


def _acc(path, task):
    with open(path) as f:
        return json.load(f)["results"][task]["exact_match,strict-match"]


def test_gsm8k_within_tolerance():
    ref = _acc("results/fp16/results.json", "gsm8k")
    opt = _acc("results/fp8/results.json", "gsm8k")
    drop_pp = (ref - opt) * 100
    assert drop_pp <= TOL_ACC_PP, (
        f"GSM8K regressed {drop_pp:.2f}pp (ref={ref:.3f} opt={opt:.3f}), "
        f"tolerance is {TOL_ACC_PP}pp. Block the promote and recalibrate."
    )


def test_logit_drift_within_tolerance():
    kl = json.load(open("results/fp8/drift.json"))["mean_kl"]
    assert kl <= TOL_KL, (
        f"Mean per-token KL {kl:.4f} exceeds {TOL_KL}. The distribution shifted "
        f"even if accuracy did not — investigate before shipping."
    )


@pytest.mark.parametrize("task,tol", [("gsm8k", 1.0), ("mmlu", 0.8)])
def test_no_task_regresses(task, tol):
    ref = _acc("results/fp16/results.json", task)
    opt = _acc("results/fp8/results.json", task)
    assert (ref - opt) * 100 <= tol, f"{task} regressed beyond {tol}pp"
```

And the GitHub Actions workflow that runs it on every change to the serving config or the model revision. Note the GPU runner label and the ordering: start the server, wait for health, eval, drift, then the pytest step that actually gates the merge.

```yaml
# .github/workflows/quality-gate.yml
name: serving-quality-gate
on:
  pull_request:
    paths:
      - "serving/**"          # server flags, quantization config
      - "models/revision.txt" # pinned model revision
jobs:
  quality-gate:
    runs-on: [self-hosted, gpu, h100]   # a runner with a real GPU
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4

      - name: Start optimized vLLM server
        run: |
          vllm serve "$(cat models/revision.txt)" \
            --quantization fp8 --port 8000 --seed 0 \
            --max-model-len 4096 --disable-log-requests &
          until curl -sf http://localhost:8000/health; do sleep 2; done

      - name: Run accuracy benchmark against the live endpoint
        run: |
          lm_eval --model local-chat-completions \
            --model_args model="$(cat models/revision.txt)",base_url=http://localhost:8000/v1/chat/completions,num_concurrent=16,tokenized_requests=False \
            --tasks gsm8k,mmlu --num_fewshot 5 --apply_chat_template \
            --output_path results/fp8/

      - name: Compute logit drift vs frozen reference
        run: python drift_check.py

      - name: Gate — block the merge on any regression
        run: pytest test_quality_gate.py -v
```

Two design choices make this gate survivable in practice. First, **cache the reference results.** Re-running the BF16 baseline on every PR doubles your GPU cost for no benefit — the reference only changes when the base model does, so compute it once, store the JSON as an artifact keyed by model revision, and reuse it. Second, **make the gate fast enough that people do not route around it.** A 45-minute gate on a subset (a 500-item golden slice plus the KL check) that runs on every PR is worth more than a 6-hour full-MMLU gate that engineers disable "just this once." Speed is a feature of a gate, because a gate that is skipped protects nothing.

There is a third choice that determines whether the gate survives its first month: **it must not be flaky.** A quality gate that fails one PR in five for no reason — because the tolerance is set at one sigma and normal item-sampling noise trips it — will be disabled by the team within a fortnight, and a disabled gate is worse than no gate because it creates false confidence. This is the same noise math from §4, applied to your own credibility. Set the tolerance at two or three sigma above the noise floor so that a *passing* build passes reliably and only a *real* regression fails. If you find the gate flapping, the fix is almost never "loosen the tolerance and hope" — it is either to increase the item count so the noise band shrinks, or to switch the primary signal to KL drift, which has far lower variance than accuracy and therefore flaps far less. A gate earns the right to block a merge only by being trustworthy when it blocks; the fastest way to lose that right is a false alarm on someone's clean Friday PR.

## 6. Quality under load: does heavy batching change your outputs?

Everything so far measures quality on a quiet server. Production is not quiet. The question this section answers is whether the load itself — the batching, the preemption, the fallback logic that only activates when you are overwhelmed — changes the outputs. The answer, uncomfortably, is that it can.

![Grid showing greedy output determinism across batch sizes — identical at batch one, diverging at later token positions as batch size grows to eight and thirty-two.](/imgs/blogs/evaluating-serving-quality-under-load-5.webp)

The grid traces the same greedy prompt across three batch sizes. At batch 1 it is the reference. At batch 8 the output is identical for the first few hundred tokens and then diverges near token 500. At batch 32 it diverges earlier, near token 200, and by token 500 it is producing materially different text. Nothing changed except how many *other* sequences were being decoded alongside this one. The cause is the non-associativity of floating-point addition from §1: the attention softmax denominators and the matmul accumulations sum their terms in an order that depends on the batch geometry, so the logits differ in their last bits, and greedy decoding occasionally resolves a near-tie differently. Once one token differs, the autoregressive process feeds that difference forward and the sequences drift apart.

This matters for evaluation in a specific, sharp way: **a gate that runs at batch 1 measures a model your users never hit.** Your users hit the model at whatever batch size the server happens to be running under load, and that batch size changes second to second. If you want your gate to be honest, it has to load the server — which is why the `num_concurrent=16` in the lm-eval command was not incidental. Here is a direct determinism probe that makes the effect concrete.

```python
# determinism_check.py — does the same prompt produce the same greedy output
# at different batch sizes? Runs the target prompt alone, then padded into a
# larger batch of decoys, and reports the first divergent token position.
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", seed=0, enforce_eager=True)
sp = SamplingParams(temperature=0.0, max_tokens=512)   # greedy, deterministic seed

PROMPT = "Solve step by step: A train travels 60 km in 45 minutes. What is its speed in km/h?"

def greedy_at_batch(batch_size):
    decoys = [f"Write one sentence about topic number {i}." for i in range(batch_size - 1)]
    outputs = llm.generate([PROMPT] + decoys, sp)
    return outputs[0].outputs[0].token_ids   # the TARGET prompt is index 0

base = greedy_at_batch(1)
for b in (8, 32, 64):
    other = greedy_at_batch(b)
    first_div = next((i for i, (x, y) in enumerate(zip(base, other)) if x != y),
                     None)
    if first_div is None and len(base) == len(other):
        print(f"batch={b:>3}: identical to batch-1 output")
    else:
        print(f"batch={b:>3}: diverges at token {first_div}")
```

If this prints "diverges at token 200" for your stack, you have learned something important: your server is not bitwise-deterministic across load, and any per-request golden-output assertion will flap. There are three responses, in increasing order of cost. **Accept it and gate on distributions, not exact strings** — measure mean KL between batch-1 and batch-N outputs and require it below a small tolerance (say 0.005), which tolerates last-bit jitter while catching real divergence. **Turn on batch-invariant kernels** if your runtime supports them — these fix the reduction order regardless of batch size, restoring bitwise determinism at a throughput cost, and are the subject of active work in vLLM and the broader community. **Or reduce the blast radius** by capping batch size for requests where reproducibility is contractually required, routing them to a determinism-pinned pool.

#### Worked example: the price of determinism

A financial-services customer has a compliance requirement that identical inputs must produce byte-identical outputs — an auditor needs to reproduce any decision the model made. On the default vLLM configuration the determinism probe above diverges at token 200 under load, so the requirement is violated. Turning on batch-invariant kernels fixes the reduction order and restores bitwise reproducibility, but the fixed-order reductions cannot exploit the fastest tiling, so decode throughput drops from roughly 1,050 to about 750 tokens per second on the H100 — a 30% loss. The decision is not "determinism good, throughput bad"; it is a routing decision. You stand up two pools: a determinism-pinned pool with batch-invariant kernels and a capped batch size for the compliance-flagged tenant, absorbing the 30% throughput cost only for the ~5% of traffic that needs it, and the default high-throughput pool for everyone else. The quality SLO — reproducibility, here — becomes a property you provision for a slice of traffic rather than a global setting, and the throughput cost is paid only where the requirement actually applies. This is the SLO triangle at work: reproducibility bought with throughput, spent only where it earns its keep.

Two more load-specific quality hazards deserve a mention. **Preemption:** under memory pressure, continuous-batching schedulers preempt a running sequence — evicting its KV cache and recomputing or swapping it later. A correct implementation resumes to the identical state, but a bug in the swap/recompute path can corrupt the cache and produce garbage for that one unlucky request, which no aggregate benchmark will catch because it is one request in a million under a specific memory condition. **Fallback models:** many production stacks fail over to a smaller or older model when the primary is overloaded or timing out. That fallback is a different model with different quality, activated exactly when you are least able to notice. If you have a fallback, its quality is part of your quality SLO, and it needs its own gate and its own online monitoring — because a fallback that silently serves a worse model during your busiest hour is a quality regression that correlates perfectly with load.

## 7. Online quality monitoring: no ground truth, real traffic

You cannot run GSM8K on production traffic. Real prompts have no reference answer, arrive one at a time, and are often private. So the pre-ship gate, however good, is only half the system: it certifies the build you are about to ship, but it cannot certify that the build stays healthy once weights get hot-swapped, a fallback engages, or the traffic distribution shifts. Online, you monitor *proxies* — cheap, computable-without-ground-truth signals that correlate with quality — and you compare a challenger against a champion on live traffic.

![Layered defense stack for serving quality — pre-ship eval, then numerical drift check, then a one-percent canary, then online proxies, then user feedback as the slow last resort.](/imgs/blogs/evaluating-serving-quality-under-load-3.webp)

The stack shows the layers of defense, each catching a class of regression the one above it missed, at increasing detection latency. Pre-ship eval and the numerical drift check are offline and certain but only see the build, not production. The canary is the first layer that touches real traffic. Below it, online proxies watch every request, and user feedback is the slow, expensive last resort. A regression that beats all five layers is rare; one that beats the top two but is caught by the canary is common, which is exactly why the lower layers exist.

The workhorse online proxies, cheapest first:

- **Format and schema validity.** If your service returns structured output, validate every response against its JSON schema. A quantization regression frequently shows up first as a spike in malformed JSON — the model loses the precision to reliably close braces and quote keys. This is nearly free to compute and one of the earliest signals.
- **Refusal rate.** The fraction of responses that are refusals or safety deflections. A sudden jump often means the model degraded into hedging, or a system-prompt change interacted badly with the new build.
- **Output-length distribution.** Track the p50 and p99 of response length. A regressed model often gets terser (gives up early) or more repetitive (loops). A shift in the length histogram is a strong, cheap early warning.
- **LLM-as-judge on a sample.** Take a small random sample — 1% of traffic — and have a stronger model score each response for helpfulness and correctness on a rubric. This is the most informative proxy and the most expensive, which is why you sample rather than judge everything.
- **User feedback.** Thumbs up/down, regenerate-clicks, escalations to a human. The highest-signal, highest-latency measure — it is what you are ultimately optimizing, but it arrives hours to days late.

Here is a monitor that computes the cheap proxies on every request and fires the expensive judge on a 1% sample, emitting everything to your metrics backend.

```python
# online_quality.py — per-request quality proxies + a sampled LLM judge.
import json
import random
import jsonschema
from openai import OpenAI

judge = OpenAI(base_url="http://judge-model:8001/v1")   # a stronger judge model
RESPONSE_SCHEMA = json.load(open("schemas/response.json"))
JUDGE_SAMPLE_RATE = 0.01
REFUSAL_MARKERS = ("i can't", "i cannot", "i'm unable", "as an ai")


def observe(request_text: str, response_text: str, emit) -> None:
    # 1. JSON-schema validity — cheap, run on every request.
    try:
        jsonschema.validate(json.loads(response_text), RESPONSE_SCHEMA)
        emit("quality_json_valid", 1)
    except (json.JSONDecodeError, jsonschema.ValidationError):
        emit("quality_json_valid", 0)      # a spike here is an early quant warning

    # 2. Refusal rate and 3. output length — also cheap, every request.
    emit("quality_refusal", int(any(m in response_text.lower() for m in REFUSAL_MARKERS)))
    emit("quality_output_tokens", len(response_text.split()))

    # 4. LLM-as-judge on a 1% sample — expensive, so sample it.
    if random.random() < JUDGE_SAMPLE_RATE:
        verdict = judge.chat.completions.create(
            model="judge",
            temperature=0.0,
            messages=[
                {"role": "system", "content":
                 "Score the assistant response 1-5 for correctness and helpfulness. "
                 "Reply with only the integer."},
                {"role": "user", "content":
                 f"PROMPT:\n{request_text}\n\nRESPONSE:\n{response_text}"},
            ],
        )
        try:
            emit("quality_judge_score", int(verdict.choices[0].message.content.strip()))
        except ValueError:
            pass    # judge returned something unparseable; skip this sample
```

The final piece is the **champion–challenger canary**. Route 1% of traffic to the challenger build (the new FP8 server) and 99% to the champion (the current production build), and compare the *same* proxies across the two arms on the *same* traffic distribution. Because both arms see statistically identical prompts, a difference in refusal rate, JSON-validity, length distribution, or sampled judge score between champion and challenger is attributable to the build, not to a traffic shift. This is how you catch a regression that passed the pre-ship gate — because the golden set did not represent some slice of real traffic — before it reaches more than 1% of users. Alert on any proxy diverging beyond its historical noise band, and wire the alert to an automatic rollback of the canary.

| Online proxy | Cost per request | Catches | Detection latency |
|---|---|---|---|
| JSON-schema validity | Negligible | Malformed structure, quant damage | Seconds |
| Refusal rate | Negligible | Over-hedging, prompt interaction | Minutes |
| Output-length shift | Negligible | Terseness, repetition loops | Minutes |
| LLM-as-judge (1% sample) | High (extra inference) | Helpfulness/correctness drift | Minutes–hours |
| Champion–challenger diff | Medium | Any build-attributable regression | Minutes |
| User feedback | Free but sparse | The thing you actually optimize | Hours–days |

A note on trusting the judge: an LLM-as-judge is itself a model with biases — it favors longer answers, it can be gamed, and it is noisy on any single example. It is a fine *aggregate trend* detector across thousands of samples and a poor *single-verdict* oracle. Use it to watch whether the challenger's mean judge score is drifting below the champion's, not to adjudicate individual responses. And guard the judge's own version: if you silently upgrade the judge model, your quality trend will step-change for reasons that have nothing to do with the served model.

### Sizing the canary and validating the judge

Two practical questions decide whether the canary catches anything. First, **how much traffic and for how long?** The same noise math from §4 governs the canary: to detect a difference of size `d` in a proportion (say refusal rate) between champion and challenger, you need roughly `n ≈ 16 · p(1−p) / d²` samples *per arm* for a comfortable-power test. For a refusal rate around 5% and a difference you care about of 1 point (`d = 0.01`), that is about `16 · 0.0475 / 0.0001 ≈ 7600` requests per arm. If the challenger sees 1% of a 1,000-QPS service, it collects 10 requests per second, so it reaches 7,600 samples in about 13 minutes — meaning a genuine 1-point refusal-rate regression is detectable within a quarter hour, but a 0.2-point one needs 25× the traffic and is better left to the pre-ship gate. Right-size the canary duration to the smallest regression you actually need it to catch, and do not read a divergence out of the first two minutes of a canary — that is noise, not signal.

Second, **validate the judge before you trust its trend.** Before an LLM-as-judge goes into your monitoring, measure its agreement with human labels on a few hundred examples spanning good and bad responses. If the judge agrees with human raters only 70% of the time, its "score dropped 0.2" is mostly its own noise; if it agrees 90%+, the trend is meaningful. Tighten the rubric until agreement is high: replace "rate helpfulness 1–5" with a specific, checkable rubric ("5 = fully correct and directly answers; 3 = partially correct or partially off-topic; 1 = wrong or refuses a valid request"), give the judge one or two anchor examples per score, and force it to emit the score in a parseable format. A calibrated judge with a tight rubric is one of the highest-value online signals you have; an uncalibrated one is an expensive random-number generator wired to a pager.

## 8. Guarding against silent weight and version mismatch

The most embarrassing quality regression is not a subtle numerics drift — it is serving the wrong weights entirely. A canary that promotes but points at a stale checkpoint; a config that loads the base model instead of the fine-tune; a multi-replica fleet where three of ten pods came up with last week's revision after a partial rollout. Each of these serves a *different model* than you evaluated, and the gate you ran means nothing because it ran against a different artifact.

![Timeline of a silent regression — a weight swap passes a five-prompt smoke test, reaches full traffic, degrades reasoning for six hours, and is rolled back only after users complain.](/imgs/blogs/evaluating-serving-quality-under-load-6.webp)

The timeline is the anatomy of exactly this failure. A weight swap from revision v2.3 to v2.4 passes a five-prompt smoke test — five prompts cannot resolve a 3-point regression, per the noise math — and deploys to 100% of traffic. GSM8K silently drops 3.1 points, but nobody is measuring it online. The sampled judge score drifts from 4.2 to 3.9. Thumbs-down rate climbs 18%. Six hours later, after enough user complaints accumulate to get someone's attention, the team rolls back to v2.3. The entire six-hour incident was preventable by two cheap guards.

**Guard one: assert the served artifact's identity.** Every response path should be able to report the exact model revision hash, quantization config, and tokenizer version it is running, and your monitoring should assert that every replica reports the expected value. vLLM and TGI expose the loaded model on a metadata endpoint; scrape it and alert on any replica that disagrees with the intended revision. A hash mismatch is a page, not a dashboard curiosity.

```bash
# artifact_guard.sh — assert every replica serves the intended revision.
EXPECTED="$(cat models/revision_hash.txt)"
for pod in $(kubectl get pods -l app=llm-serving -o name); do
  served=$(kubectl exec "$pod" -- \
    curl -sf http://localhost:8000/v1/models | jq -r '.data[0].id')
  if [ "$served" != "$EXPECTED" ]; then
    echo "MISMATCH: $pod serves '$served', expected '$EXPECTED'" >&2
    exit 1     # fail the rollout verification; trigger rollback
  fi
done
echo "all replicas serve $EXPECTED"
```

The identity you assert has to cover more than the weight file. A model artifact is the weights *plus* the tokenizer, the chat template, the quantization config, and the generation defaults, and a mismatch in any of them changes the served behavior as surely as swapping the weights. A tokenizer that adds a leading space differently, a chat template that moves the system prompt, a default temperature that shifted from 0 to 0.7 in a config refactor — each is a "same weights, different model" bug that your weight-hash check would wave through. Fold the tokenizer hash and the effective generation config into the identity you assert, and diff the *entire* serving config against the one your gate ran on, not just the checkpoint. The question the guard answers is not "are these the right weights" but "is this the exact artifact I evaluated," and the artifact is the whole stack.

**Guard two: a continuous canary probe.** Independent of user traffic, fire a tiny fixed set of golden prompts — say 20 items with known-good answers — at the production endpoint every few minutes, and alert if the pass rate drops. Twenty items cannot resolve a 1-point regression, but they can absolutely resolve a 3-point one (three sigma at n = 20 is roughly a 30-point band, so a 3-point drop is not catchable there either — use this probe for *gross* failures like a wrong-model load or a crashed decoder, and rely on the champion–challenger diff for subtle drift). The continuous probe is your smoke detector for catastrophic mismatch; the online proxies are your instrument panel for slow drift. You need both. The rule that ties this section to the whole post: **you evaluated an artifact; make sure production serves that exact artifact, and keep proving it serves it.**

## Case studies

**FP8 is near-lossless when calibrated — and only then.** NVIDIA's H100-native FP8 (E4M3) format and the FP8 support in vLLM and TensorRT-LLM consistently report accuracy recovery above 99% of the BF16 baseline across standard benchmarks when the activation and weight scales are calibrated properly. The published accuracy-recovery numbers from the vLLM and Neural Magic / Red Hat quantization work put most tasks within a fraction of a point of BF16. The failure mode is not the format — it is a static activation scale that clips the outlier channels that transformer activations are famous for. The lesson for a gate: FP8 usually passes, so a gate that only ever sees passing FP8 builds is not wasted effort — it is the thing that catches the one miscalibrated build in twenty, which is exactly the one that would have shipped otherwise.

**AWQ's accuracy claims, measured.** The AWQ paper (Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," 2023) protects the roughly 1% of weight channels that are salient to the activations and reports 4-bit perplexity within a few tenths of a point of FP16 on WikiText-2, substantially better than a naive round-to-nearest INT4. On downstream tasks the drop is typically small but nonzero and model-dependent — which is precisely why the measurement table above lands AWQ in "review" rather than "auto-ship." Trust the paper's *method*; verify the *number* on your model and your tasks, because the salient-channel assumption holds to different degrees across architectures.

**Speculative decoding is provably distribution-preserving — verify the implementation anyway.** Both Leviathan et al. ("Fast Inference from Transformers via Speculative Decoding," 2023) and Chen et al. ("Accelerating Large Language Model Decoding with Speculative Sampling," 2023) prove that the modified rejection-sampling scheme yields tokens drawn from exactly the target model's distribution — the speedup is free of any quality cost, in expectation. That is a strong theoretical guarantee, and it is the reason speculative decoding is safe to deploy. But the guarantee is about the *algorithm*, not your *code*. A determinism test — same prompt at temperature 0 with and without the draft model, asserting identical output — is a five-minute check that confirms your acceptance test is implemented correctly, and it has caught real bugs where a draft/target tokenizer mismatch or a wrong acceptance threshold leaked a quality difference the theory says cannot exist.

**Batching nondeterminism is real and now well-characterized.** The observation that LLM inference is not reproducible across batch sizes was folklore for years; the 2025 work from Thinking Machines Lab ("Defeating Nondeterminism in LLM Inference," Horace He and collaborators) traced it precisely to the batch-size-dependent reduction order in attention and matmul kernels and demonstrated batch-invariant kernels that restore bitwise determinism. The practical takeaway for evaluation is the one from §6: if you have not turned on batch-invariant kernels, do not write per-request golden-output assertions, because they will flap under load for reasons unrelated to any regression — gate on distributional drift instead, and reserve exact-match assertions for the determinism-pinned pool.

**KV-cache quantization hides its damage at short context.** FP8 and INT8 KV-cache compression is one of the most attractive memory optimizations in LLM serving — the KV cache dominates memory at long context, and halving it lets you roughly double batch size or context length. The published results and the vLLM implementation notes show it is close to lossless on typical short-to-medium prompts. The trap is that its error concentrates exactly where you are least likely to test: at long context, where a request attends over thousands of quantized past tokens and the accumulated rounding error in the compressed keys and values degrades the model's ability to retrieve a fact buried early in the prompt. A gate that evaluates KV-FP8 only on 512-token benchmarks will report "lossless" and ship a build that quietly fails needle-in-a-haystack retrieval at 32k context. The fix is methodological, not exotic: any KV-cache-quantization gate must include a long-context slice — a retrieval task at the context lengths you actually serve — because that is the only place the regression is visible. This is a specific instance of the general rule that your eval set must match your traffic distribution, and it is the one teams most often get wrong.

## When to use this (and when not to)

Run the **full gate** — benchmark plus KL drift, blocking CI — whenever a change touches the *numerics of the compute graph*: any quantization change (weights or KV cache), a new attention or GEMM kernel, enabling speculative decoding or chunked prefill, a runtime upgrade that changes kernels, or a new model revision. These are the changes that can silently move outputs, and they are exactly the changes whose whole point is a latency or throughput win — so the temptation to skip the quality check is highest precisely when the risk is highest.

Run only the **cheap layers** — online proxies and the continuous artifact-identity probe, no blocking benchmark — for changes that do not touch the compute graph: autoscaling policy, load-balancer weights, replica counts, timeout tuning, or a Kubernetes rollout of the *same* artifact. These cannot change the math, so a full benchmark is wasted GPU time. Keep the online proxies on always, though, because a rollout can still serve the wrong artifact.

Do **not** over-invest. A full MMLU-plus-everything gate on every pull request is a common way to make the gate so slow that engineers route around it — and a routed-around gate protects nothing. A 500-item golden slice plus a KL check that runs in under an hour beats a six-hour full suite that runs "when we remember." Similarly, do not build an LLM-as-judge pipeline for a model that returns a fixed enum — schema validity and accuracy are cheaper and sharper. And if you serve at genuinely tiny scale — a handful of internal users, no latency pressure, no quantization — the honest answer is that a frozen golden set of 100 prompts you eyeball before each deploy may be all the gate you need. The machinery here scales with your blast radius; match the investment to it.

## Key takeaways

- **Quality is the third SLO.** Latency and throughput have metrics, detectors, and gates; give output correctness the same three things or you are flying blind on the property users actually care about.
- **Every serving optimization is a place accuracy can silently regress.** Quantization, KV-cache compression, speculative decoding, chunked prefill, kernel swaps, batching numerics, and sampling params all perturb the arithmetic. None of them trip an operational alert.
- **Derive your tolerance from the noise, not from a guess.** Accuracy has a standard error of `√(p(1−p)/n)`. At n = 100 that is roughly ±4 points at one sigma; a 100-item gate cannot resolve a 2-point regression. Size `n` to make the smallest regression you care about at least two sigma.
- **KL divergence on the logits is a sharper signal than accuracy.** It uses the full distribution instead of a 0/1 collapse, so it catches subtle drift at a few hundred sequences, long before accuracy moves outside its noise band.
- **Gate before you ship, in CI, blocking the merge.** A human-remembered gate is skipped on the day it matters. Start the optimized server, run the benchmark and the drift check against a cached reference, and fail the pipeline on a tolerance.
- **Load changes outputs.** Floating-point non-associativity makes greedy decoding batch-size-dependent. Evaluate under concurrency, gate on distributions rather than exact strings, and turn on batch-invariant kernels where reproducibility is required.
- **Online, monitor proxies and a champion–challenger canary.** JSON-schema validity, refusal rate, output-length shift, and a sampled LLM-judge score catch what the pre-ship gate could not, on traffic that has no ground truth.
- **Prove production serves the artifact you evaluated.** Assert the model revision hash on every replica and run a continuous smoke probe. The worst regression is serving the wrong weights, and it is the cheapest to prevent.

## Further reading

- EleutherAI, **`lm-evaluation-harness`** — the standard benchmark harness with OpenAI-compatible endpoint support ([github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).
- Lin et al., **"AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"** (2023) — the accuracy-vs-compression method behind most production INT4 serving.
- Leviathan, Kalman, Matias, **"Fast Inference from Transformers via Speculative Decoding"** (2023) and Chen et al., **"Accelerating Large Language Model Decoding with Speculative Sampling"** (2023) — the proofs that speculative decoding is distribution-preserving.
- Thinking Machines Lab, **"Defeating Nondeterminism in LLM Inference"** (2025) — the definitive account of batch-size-dependent reduction order and batch-invariant kernels.
- **vLLM documentation** on FP8 quantization and reproducibility, and **NVIDIA TensorRT-LLM** quantization accuracy notes — the practical accuracy-recovery numbers for production quantization.
- Within this series: [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle; [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for the schemes whose accuracy you are gating; [Speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production) for the losslessness you should verify; [Model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) for the latency and throughput SLOs quality joins; and [Observability for LLM serving](/blog/machine-learning/model-serving/observability-for-llm-serving) for wiring the online proxies into your dashboards.
