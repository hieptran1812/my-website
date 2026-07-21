---
title: "Sampling numerics: why temperature zero is not reproducible"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "The same prompt, the same seed and the same GPU give you two different answers because a kernel changed its reduction order when other people joined your batch — here is the mechanism, the fix, what the fix costs, and a harness that finds the exact token where two runs part."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "determinism",
    "batch-invariance",
    "sampling",
    "decoding",
    "numerics",
    "pytorch",
    "gpu",
    "vllm",
    "ml-systems",
    "reproducibility",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 49
---

Here is a support ticket you will receive, verbatim, if you run an inference endpoint long enough:

> Same prompt. `temperature=0`, `seed=42`. Same model version, same server, same GPU. I sent it twice and got two different answers. The first 40 words are identical and then they diverge completely. Is this a bug?

It is not a bug in the sense the reporter means — nothing is broken, no memory is corrupted, no race is being lost. It is also not "just floating point noise", which is the dismissive answer engineers usually give and which is wrong in an interesting way. The honest answer is that the reporter's output depended on **who else was talking to your server at that moment**, and nobody told them that was a property of the system they were buying.

The instinct is to blame the sampler. It is the obvious suspect: it is the only component with the word "random" attached to it, it is where the seed goes, and it is the last thing to touch the tokens. The instinct is wrong, and disproving it takes four lines of code. The sampler is the one part of the decode loop that is exactly, bitwise, boringly reproducible. Everything upstream of it is not.

![Five stacked layers where reproducibility leaks in an inference engine, from float arithmetic at the bottom to retokenization above the model](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-1.webp)

This post takes those layers apart from the bottom. We derive why `(a+b)+c` and `a+(b+c)` are different numbers on real hardware, with a bf16 example you can check with a pencil. We follow that into GPU reduction order, and then into the mechanism that actually explains the ticket above: **libraries pick different kernels for different batch sizes, and different kernels sum in different orders**, so in a continuously batched server your logits are a function of other users' traffic. Then we build the fix into `nanoserve` — a per-request generator, an fp32 sampler, tie-break rules that do not depend on memory layout — and, most usefully, `nanoserve/tools/divergence.py`, a harness that runs one prompt at batch 1, 4 and 16 and prints the exact step index where the runs part. That harness works against any engine with a Python API, including vLLM, and it is the artifact from this post I would most want you to steal.

Finally we price it. Determinism is not free and the bill has been published by three separate teams; we quote all three with their setups. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post lays out the scoreboard this one trades against, and [the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) covers the decoding knobs whose *numerics* we are auditing here.

---

## 1. First, exonerate the sampler

Before theorizing, eliminate the accused. Greedy decoding is `argmax` over a vector of logits. `argmax` is an exact operation: it does one pass, compares, and returns an index. Given the identical input vector it returns the identical index, every time, on every device, forever. There is no accumulation, no reassociation, no scheduling freedom that could change the answer.

```python
# nanoserve/tests/test_sampler_is_innocent.py
import torch

logits = torch.randn(128_256, dtype=torch.float32, device="cuda")

# Greedy: the same vector in, the same token id out, a thousand times.
first = torch.argmax(logits).item()
assert all(torch.argmax(logits).item() == first for _ in range(1000))

# Seeded multinomial: the same generator state in, the same draw out.
def draw(seed: int) -> int:
    g = torch.Generator(device="cuda").manual_seed(seed)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=g).item()

assert draw(42) == draw(42)
print("sampler is a pure function of (logits, generator state)")
```

Run this and it passes. It will pass on your 4090, on an A100, on CPU. The sampler is a pure function of two inputs: the logit vector and the generator state. If your output changed between two runs, then one of those two inputs changed. Since you fixed the seed, and assuming the seed actually reaches the same generator both times — a real assumption we will interrogate in section 5 — **the logit vector changed**.

That reframes the whole problem. Nondeterminism in LLM serving is not a sampling problem. It is a *forward pass* problem that the sampler faithfully reports. The sampler is the messenger.

So the question becomes: how does the same model, with the same weights, given the same token ids, produce a different logit vector? Not a *wildly* different one — the vectors will agree to five or six decimal digits. Different in the last few bits. And that turns out to be enough, for a reason we will make precise: a language model spends a surprising fraction of its steps genuinely undecided between two tokens, and when the gap between the top two logits is smaller than the numerical noise, the noise decides.

Four candidate mechanisms, in the order a good debugger should consider them:

| Mechanism | Changes logits? | Under your control? | Section |
| --- | --- | --- | --- |
| Float non-associativity | It is the substrate | No — it is the arithmetic | 2 |
| Reduction order in a kernel | Yes, last bits | Partly — kernel choice | 3 |
| Batch-size-dependent kernel selection | Yes, last bits | Yes — with a cost | 4 |
| RNG placement and batch order | Not the logits, the draw | Yes — cheaply | 5 |
| Retokenization, templates, cache state | Yes, grossly | Yes — cheaply | 6 |

Only one of those five is genuinely expensive to fix. The rest are engineering hygiene, and most teams that complain about nondeterminism have not done the hygiene yet.

---

## 2. Floating point is not associative, and here is the proof by hand

Every reduction in a transformer — a dot product, a softmax denominator, an RMSNorm sum of squares, an all-reduce across tensor-parallel ranks — is a sum of many terms. In real arithmetic, the order of a sum does not matter. In floating point it does, because each partial result is *rounded to the nearest representable value* before the next addition happens, and rounding is not distributive over reassociation.

Take bfloat16, the format your engine almost certainly runs in. It has 1 sign bit, 8 exponent bits and 7 stored mantissa bits, so 8 significant bits counting the implicit leading one. For a value $x$ in the binade $[2^{k}, 2^{k+1})$, the spacing between representable neighbours — the unit in the last place — is

$$\text{ulp}(x) = 2^{k-7}$$

For $x \in [256, 512)$ that is $k = 8$, so $\text{ulp} = 2^{1} = 2$. The representable bf16 values in that range are 256, 258, 260, 262, and so on. **There is no bf16 number equal to 257.**

Now take three bf16 values: $a = 256$, $b = 1$, $c = 1$. All three are exactly representable.

**Left-to-right.** First $a + b = 257$ exactly, in the accumulator. Rounding 257 to bf16 means choosing between 256 and 258; 257 is exactly halfway, so round-half-to-even picks the one whose mantissa is even, which is 256. Now add $c$: ${256 + 1 = 257}$, rounds to 256 again. The result is **256**.

**Right-to-left.** First $b + c = 2$, exactly representable, no rounding. Then $a + 2 = 258$, exactly representable. The result is **258**.

$$(a + b) + c = 256 \qquad a + (b + c) = 258$$

Two valid orders, both implemented correctly, differing by 2 — which here is a relative error of about 0.8%. Neither is wrong. IEEE-754 guarantees each individual operation is correctly rounded; it says nothing about which order you perform them in, because it cannot.

```python
# nanoserve/tools/numerics_demo.py — run this anywhere, CPU is fine
import torch

a = torch.tensor(256.0, dtype=torch.bfloat16)
b = torch.tensor(1.0, dtype=torch.bfloat16)
c = torch.tensor(1.0, dtype=torch.bfloat16)

print(float((a + b) + c))   # expect 256.0
print(float(a + (b + c)))   # expect 258.0

# fp32 version of the same trap, at a different scale
x = torch.tensor(1.0)
tiny = torch.tensor(1e-8)
print(float((x + tiny) - x))   # expect 0.0  — tiny vanished
print(float(x + (tiny - x) + x))  # a different, also-valid answer
```

The fp32 case matters because your sampler runs in fp32. There, the significand is 24 bits, so $\varepsilon = 2^{-24} \approx 5.96 \times 10^{-8}$ and $\text{ulp}(1.0) = 2^{-23} \approx 1.19 \times 10^{-7}$. Adding $10^{-8}$ to 1.0 changes nothing at all: the increment is smaller than half an ulp, so it rounds away. Add it a hundred million times sequentially and you still have 1.0. Add the hundred million copies to each other first and you get 1.0, then ${1.0 + 1.0 = 2.0}$. The same arithmetic, two answers, off by 100%.

#### Worked example: how much error does a 4096-term dot product carry?

Llama-3.1-8B has hidden size 4096. The final logit for token $v$ is a dot product of the last hidden state with row $v$ of the output embedding: a sum of $N = 4096$ products. Classical error analysis gives two useful bounds.

For **sequential** summation with unit roundoff $\varepsilon$, the worst-case relative error grows like $N\varepsilon$, and the *typical* error — treating rounding errors as independent zero-mean perturbations, which is a well-behaved model in practice — grows like $\sqrt{N}\,\varepsilon$ because the errors random-walk:

$$\text{err}_{\text{seq}} \approx \sqrt{N}\,\varepsilon \cdot \lVert x \rVert$$

For **pairwise or tree** summation with $\log_2 N$ levels, only $\log_2 N$ roundings compose along any path:

$$\text{err}_{\text{tree}} \approx \sqrt{\log_2 N}\,\varepsilon \cdot \lVert x \rVert$$

With $N = 4096$: $\sqrt{4096} = 64$ and $\sqrt{\log_2 4096} = \sqrt{12} \approx 3.46$. The two strategies differ in typical error magnitude by roughly $64 / 3.46 \approx 18\times$.

Put a number on the absolute size. Accumulating in fp32 with $\varepsilon \approx 6 \times 10^{-8}$, and taking a logit of magnitude around 15 as representative of a mid-sized model's output scale, a tree reduction lands near $3.46 \times 6\times 10^{-8} \times 15 \approx 3 \times 10^{-6}$ while a sequential one lands near $64 \times 6 \times 10^{-8} \times 15 \approx 6 \times 10^{-5}$. *Source: derived from the error model above; the logit scale is an assumption, not a measurement — swap in your model's actual logit magnitudes and the conclusion moves proportionally.*

So the noise floor on a logit is somewhere in the neighbourhood of $10^{-6}$ to $10^{-4}$ depending on how the reduction is arranged. Hold that number. It is the entire story.

---

## 3. Who chooses the reduction order, and why they keep changing their mind

Nothing above is GPU-specific. What GPUs add is that the reduction order is *not fixed by your code* — it is an emergent property of how a kernel decided to parallelize.

A GPU computes a 4096-term dot product by splitting it across threads. A typical decode-shaped kernel might give each of 8 warps a 512-element chunk, have each warp reduce its chunk internally (itself a tree over 32 lanes, then a shuffle-based reduction), write 8 partial sums to shared memory, and have one warp add those 8 partials. Or it might use 4 chunks. Or 16. Or it might use `atomicAdd` into a single accumulator, in which case the order is decided by which blocks happen to arrive first — and that ordering is not stable across launches, which is a genuinely different and worse kind of nondeterminism.

![One hidden vector summed by two different reduction strategies, producing two logits that agree to six digits and differ in the last bits](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-2.webp)

The number of chunks is not an arbitrary aesthetic choice. It is a performance decision driven by occupancy: a GPU has a fixed number of streaming multiprocessors, and a kernel that produces too few independent work items leaves most of them idle. For decode attention specifically, one query attending over a long KV sequence is a *single* logical reduction over thousands of elements, and if you assign it to one block you use one SM out of 108 on an A100. So kernels split the KV dimension into pieces, compute partial softmax numerators and denominators over each piece, and combine them in a second pass — the "split-K" or "split-KV" strategy, sometimes called a parallel tiled softmax.

vLLM's Triton attention backend deep dive (2026-03-04) describes exactly this design: a decode path where a 3D kernel splits the KV traversal and a second reduction kernel combines the partials, with the split **heuristic-gated**, and the team notes plainly that "no single configuration dominates" across shapes. That is a completely reasonable engineering position for a performance library. It is also, precisely, a statement that the reduction order is a function of the shape.

Three practical consequences follow, and they are the ones that bite in production:

1. **Different GPUs, different order.** An L4 has fewer SMs than an A100, which has fewer than an H100. A heuristic tuned to fill the device produces a different split count per device. Bitwise reproducibility across heterogeneous fleets is therefore not achievable by configuration alone; you have to fix the strategy, not the tuning.
2. **Different library versions, different order.** An upgrade that adds an autotuned config changes your outputs. This is why "pin the engine version" appears in every reproducibility checklist that survives contact with reality.
3. **Different sequence lengths, different order.** A 500-token context and a 50,000-token context will not use the same split count. So a prefix cache hit — which changes how many tokens the prefill kernel actually processes — can change the numerics of the tokens it *does* process. We return to this in section 11.

What none of that explains yet is the ticket at the top: same GPU, same version, same prompt, same context length, different answer. For that we need the fourth mechanism.

---

## 4. The real culprit: your batch has strangers in it

Here is the mechanism, stated as directly as I can.

A performance library does not have one matmul kernel. It has dozens, and it selects among them based on the shape of the problem. The vLLM team put the reason in one sentence in their bitwise-consistent training and inference post (2025-11-10): "Kernels for high batch sizes parallelize heavily on the batch dimension, while kernels for low batch sizes parallelize more within a single instance."

Read that twice, because it is the whole post. At batch 1 there is no batch dimension to spread across SMs, so the kernel must find parallelism *inside* the single row — by splitting the reduction. At batch 32 there are 32 independent rows, which is plenty of parallelism, so the kernel keeps each row's reduction intact and assigns whole rows to blocks. Both kernels are correct. Both compute the same mathematical function. **They sum in different orders, so they return different bits.**

Now put that inside a continuously batched server. In [the continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) the running set is mutated every single iteration: requests are admitted when blocks free up, requests retire when they hit a stop token. Batch size at step 417 is 9; at step 418 it is 12 because two arrived and one left. Every one of those transitions can cross a kernel-selection threshold.

Therefore: **your output is a function of other users' traffic.** Not metaphorically. Mechanically. The same request, submitted at 3 a.m. to an idle replica and at 1 p.m. to a busy one, goes through different kernels and gets different logits in the last bits. Nobody wrote that behaviour down. It fell out of a performance heuristic.

<figure class="blog-anim">
<svg viewBox="0 0 720 330" role="img" aria-label="The same request scored alone and inside a batch of eight; the two top logits agree to six digits, then the low-order digits shift, the second candidate overtakes the first, and the two continuations separate" style="width:100%;height:auto;max-width:820px">
<style>
.a17-panel{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a17-hd{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a17-tok{font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.a17-num{font:400 15px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.a17-low{fill:var(--accent,#6366f1);font:700 15px ui-monospace,SFMono-Regular,Menlo,monospace}
.a17-win{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5;rx:6}
.a17-cont{font:600 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.a17-foot{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes a17-fadeA{0%,32%{opacity:1}44%,92%{opacity:0}100%{opacity:1}}
@keyframes a17-fadeB{0%,32%{opacity:0}44%,92%{opacity:1}100%{opacity:0}}
.a17-A{animation:a17-fadeA 12s ease-in-out infinite}
.a17-B{animation:a17-fadeB 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a17-A{animation:none;opacity:0}.a17-B{animation:none;opacity:1}}
</style>
<rect class="a17-panel" x="16" y="34" width="688" height="112" rx="10"/>
<text class="a17-hd" x="34" y="58">batch 1 &#183; this request is alone in the step</text>
<rect class="a17-win" x="150" y="68" width="330" height="28"/>
<text class="a17-tok" x="46" y="88">the</text>
<text class="a17-num" x="164" y="88">12.406<tspan class="a17-low">250</tspan></text>
<text class="a17-tok" x="46" y="122">a</text>
<text class="a17-num" x="164" y="122">12.406<tspan class="a17-low">189</tspan></text>
<text class="a17-cont" x="500" y="88">picks &#8220;the&#8221;</text>
<text class="a17-foot" x="500" y="122">&#8230;the quiet hum</text>
<rect class="a17-panel" x="16" y="166" width="688" height="112" rx="10"/>
<text class="a17-hd" x="34" y="190">batch 8 &#183; seven strangers share the step</text>
<g class="a17-A">
<rect class="a17-win" x="150" y="200" width="330" height="28"/>
<text class="a17-tok" x="46" y="220">the</text>
<text class="a17-num" x="164" y="220">12.406<tspan class="a17-low">250</tspan></text>
<text class="a17-tok" x="46" y="254">a</text>
<text class="a17-num" x="164" y="254">12.406<tspan class="a17-low">189</tspan></text>
<text class="a17-cont" x="500" y="220">picks &#8220;the&#8221;</text>
<text class="a17-foot" x="500" y="254">&#8230;the quiet hum</text>
</g>
<g class="a17-B">
<rect class="a17-win" x="150" y="234" width="330" height="28"/>
<text class="a17-tok" x="46" y="220">the</text>
<text class="a17-num" x="164" y="220">12.406<tspan class="a17-low">182</tspan></text>
<text class="a17-tok" x="46" y="254">a</text>
<text class="a17-num" x="164" y="254">12.406<tspan class="a17-low">241</tspan></text>
<text class="a17-cont" x="500" y="254">picks &#8220;a&#8221;</text>
<text class="a17-foot" x="500" y="220">&#8230;a soft low drone</text>
</g>
<text class="a17-foot" x="16" y="306">Same weights, same prompt, same seed. Only the company it keeps in the batch changed.</text>
<text class="a17-foot" x="16" y="324">The gap between the candidates is smaller than the noise the reduction order introduces.</text>
</svg>
<figcaption>Illustrative logit values for one request scored alone and inside a batch of eight: the leading digits never move, the trailing digits do, and once they cross, the argmax flips and the two continuations never rejoin.</figcaption>
</figure>

The values in that figure are illustrative, chosen to sit at the noise scale derived in section 2, not measured. What is not illustrative is the shape of the failure: a difference of $6 \times 10^{-5}$ in a logit is invisible until it lands on a near-tie, and then it is total.

### 4.1 The fix, named: batch invariance

The property you want has a name. Horace He, working with Thinking Machines Lab, published *Defeating Nondeterminism in LLM Inference* on 2025-09-10 and defined it crisply: a kernel is batch-invariant when "the reduction order for each element must be fixed regardless of the batch-size of the kernel." Not "fast for every shape". *Fixed* for every shape. The post also released a library of drop-in batch-invariant PyTorch ops, `batch-invariant-ops`, covering RMSNorm, matmul and attention, along with a vLLM integration example.

Their strategies per op, as described in that post:

- **RMSNorm** — a data-parallel strategy, one reduction per row, never splitting a row across blocks even when that leaves the device underfilled at small batch.
- **Matmul** — fixed tile sizes rather than shape-selected ones, at a cost the post gives as roughly 20% slower than cuBLAS.
- **Attention** — a fixed split-size strategy along the KV dimension, so the number of partials does not depend on sequence length or batch.

The vLLM team's own bitwise-consistency work names the ops they had to make batch-invariant to get a trainer and a sampler to agree: "heavily optimized fused operations, such as the SiLU MLPs and RMSNorms (with added residuals)". The pattern is consistent — the fused, autotuned, performance-critical operations are exactly the ones whose reduction order moves.

![Two-column comparison of shape-selected kernels against fixed-strategy batch-invariant kernels for the same request row](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-3.webp)

### 4.2 One flipped bit becomes a different essay

The reason a $10^{-5}$ perturbation is not a rounding detail is autoregression. Sampling is a hard decision — an `argmax` or a categorical draw — applied to a continuous quantity, and the token you pick is then fed back as input. The system has no restoring force. Once two runs emit different tokens at step $t$, they are conditioning on different prefixes at step $t+1$, and the divergence is not small any more; it is a different sentence, then a different paragraph, then a different answer.

He's post makes this concrete with a measurement worth quoting exactly. Sampling 1,000 completions from Qwen3-235B at temperature 0 with the prompt "Tell me about Richard Feynman", the run produced **80 unique completions**, with the first divergence appearing at **token 103**. With batch-invariant kernels enabled, all 1,000 completions were identical.

![Ordered stages showing two runs identical for a hundred steps then permanently separating after a single flipped argmax](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-4.webp)

#### Worked example: how often is the model genuinely undecided?

We can invert that cited observation into a number that is useful for your own planning.

Let $p$ be the probability that, at a given step, the gap between the top two logits is smaller than the numerical noise, so that the noise decides the token. Divergence at step $t$ requires no flip in the first $t-1$ steps and a flip at $t$; the expected index of the first divergence is then approximately ${1/p}$. Taking the cited first divergence around token 103:

$$p \approx \frac{1}{103} \approx 0.0097$$

So roughly **one step in a hundred is a coin flip decided by the last bits of a float**. That is much higher than the naive estimate you would get by assuming logit gaps are spread uniformly over an $O(1)$ range — under that assumption a noise floor of $6 \times 10^{-5}$ would give $p \sim 10^{-4}$ and a first divergence around token 10,000. The discrepancy is the interesting part: language models are not uniformly confident. They are frequently and genuinely torn between two near-equivalent continuations — "the" versus "a", a comma versus a semicolon, two spellings of the same whitespace — and near-ties cluster far more densely than a uniform model predicts.

*Source: $p$ is derived from a single cited data point (Qwen3-235B, one prompt, per He and Thinking Machines Lab, 2025-09-10), so treat it as an order of magnitude for one model on one prompt, not a constant of nature. The uniform-gap comparison is derived from section 2's noise estimate.*

Now propagate it. For an output of $T$ tokens, the probability that two runs produce identical text is $(1-p)^T$:

| Output length | P(identical text) at p = 0.0097 | Source |
| --- | --- | --- |
| 20 tokens | 82% | derived |
| 100 tokens | 38% | derived |
| 300 tokens | 5.4% | derived |
| 1000 tokens | 0.006% | derived |

Which tells you something operationally important: **short completions look reproducible and long ones never are.** Teams routinely convince themselves their pipeline is deterministic because they tested it on short answers. Test it on a 1,000-token answer and the illusion evaporates.

---

## 5. Where the seed lives: RNG placement

Everything so far concerns the logits. There is a second, entirely independent, and much cheaper-to-fix source of nondeterminism: *how your engine consumes randomness*.

The naive sampler calls `torch.multinomial(probs)` with the global CUDA generator. That generator is a single stream of numbers with a position counter that advances every time anyone draws from it. In a batched engine this creates a dependency nobody intends: the random numbers a request receives depend on **how many requests were sampled before it in the same step, and in what order the engine iterated the batch**. Reorder the running set — which continuous batching does constantly — and request C gets the numbers request A would have gotten.

Worse, `seed=42` on the request does not fix this if the engine's implementation of "seed" is to call `torch.manual_seed(42)` at the top of the step. Then every request in that step shares one stream, and a request's draw depends on its index in the batch.

The fix is to give every request its own generator, seeded from something stable about the request, and to consume from it only for that request.

```python
# nanoserve/sampling/rng.py
import hashlib
import torch


class RequestRNG:
    """One torch.Generator per request, seeded deterministically.

    The seed is a stable hash of (server_seed, request seed, request id) so
    that: a client-supplied seed always reproduces; two concurrent requests
    with the same client seed do NOT collide into the same token stream; and
    nothing about batch position or arrival order enters the state.
    """

    def __init__(self, server_seed: int, device: str = "cuda"):
        self.server_seed = server_seed
        self.device = device

    def _seed_for(self, request_id: str, client_seed: int | None) -> int:
        payload = f"{self.server_seed}:{client_seed}:{request_id}".encode()
        digest = hashlib.sha256(payload).digest()
        # torch generators take a 64-bit seed; take the low 63 bits.
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def make(self, request_id: str, client_seed: int | None) -> torch.Generator:
        g = torch.Generator(device=self.device)
        g.manual_seed(self._seed_for(request_id, client_seed))
        return g
```

Two properties are worth calling out because they are easy to get wrong.

First, **the generator is created once per request, not once per step**. It lives on the request object and its state advances across the request's own decode steps. Re-seeding every step would make every token draw from the same position in the stream, which produces subtly correlated sampling — a real bug that shows up as repetitive outputs at high temperature and is maddening to diagnose.

Second, **the request id goes into the hash**. If two clients both send `seed=42` in the same second and you seed both generators with 42, they draw the identical noise, and if their prompts are similar you will serve them near-identical completions. That is a surprising-behaviour bug and, in a multi-tenant setting, arguably an information leak.

### 5.1 Gumbel-max: make sampling batch-invariant by construction

There is a stronger trick. Categorical sampling can be rewritten as an argmax over perturbed logits — the Gumbel-max trick. For probabilities $p_i \propto \exp(z_i / T)$ and independent uniforms $u_i \sim U(0,1)$:

$$\arg\max_i \left( \frac{z_i}{T} + g_i \right), \qquad g_i = -\log(-\log u_i)$$

draws exactly from the categorical distribution defined by the softmax. This matters here for a structural reason: it removes the softmax normalization from the sampling path entirely — no denominator to reduce, so no reduction whose order can vary — and it turns the draw into a pure elementwise function of the logits and a noise vector you generated yourself, from your own generator, indexed by `(request_id, step)`. The noise does not depend on batch composition because you did not ask the batch for it.

vLLM's Model Runner V2 post (2026-03-24) describes their Triton sampler as using Gumbel-Max with no explicitly materialized softmax and a stateless in-kernel RNG — the same structural argument, implemented at production scale.

```python
# nanoserve/sampling/sampler.py
import torch


def sample_gumbel(
    logits: torch.Tensor,          # [vocab], any dtype
    generator: torch.Generator,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Deterministic single-sequence sampling.

    Everything happens in fp32. The noise is drawn from this request's own
    generator, so the draw does not depend on how many other sequences share
    the step or in what order the engine iterated them.
    """
    z = logits.float()                       # upcast BEFORE any arithmetic

    if temperature <= 0.0:
        return int(_stable_argmax(z))

    z = z / temperature

    if top_k is not None and top_k < z.numel():
        # topk with sorted=True has a defined tie-break, but we do not rely on
        # it: we threshold by value and keep the lowest indices among ties.
        kth = torch.topk(z, top_k, sorted=True).values[-1]
        z = torch.where(z >= kth, z, torch.full_like(z, float("-inf")))

    if top_p is not None and top_p < 1.0:
        z = _top_p_mask(z, top_p)

    # Gumbel noise from THIS request's generator.
    u = torch.rand(z.shape, generator=generator, device=z.device, dtype=torch.float32)
    g = -torch.log(-torch.log(u.clamp_min(torch.finfo(torch.float32).tiny)))
    return int(_stable_argmax(z + g))


def _stable_argmax(x: torch.Tensor) -> int:
    """argmax that always returns the LOWEST index among exact ties.

    torch.argmax does not document its tie-break, and on CUDA the winner among
    equal values can depend on the reduction tree. Ties are common after a
    top-k mask sets a block of entries to -inf, and after quantized or
    heavily-rounded logits collapse neighbours onto the same value.
    """
    best = torch.max(x)
    idx = torch.nonzero(x == best, as_tuple=False)
    return int(idx[0].item())


def _top_p_mask(z: torch.Tensor, top_p: float) -> torch.Tensor:
    # Sort descending. torch.sort is stable=False by default; make it explicit,
    # otherwise equal logits can swap positions between runs and shift the
    # nucleus boundary by one token.
    sorted_z, sorted_idx = torch.sort(z, descending=True, stable=True)
    probs = torch.softmax(sorted_z, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Keep everything up to and including the first index crossing top_p.
    keep = cumulative - probs < top_p
    masked = torch.full_like(z, float("-inf"))
    masked[sorted_idx[keep]] = z[sorted_idx[keep]]
    return masked
```

Three deliberate choices in that code, each one a bug I would expect to see in a first implementation:

- **`z = logits.float()` before anything else.** Computing softmax in bf16 is not just imprecise, it is *unstable*: with only 8 significant bits, `exp` of two logits that differ by $10^{-3}$ can round to the same value, manufacturing ties that do not exist. The cost of the upcast is quantified in section 9 and it is negligible.
- **`_stable_argmax` returns the lowest index among exact ties.** `torch.argmax` does not document a tie-break rule and the CUDA implementation's winner can depend on the reduction tree, which is the exact thing we are trying to eliminate. Ties are not rare: after a top-k mask, everything below the cut is exactly `-inf`, and if the whole vector were masked you would be selecting among identical values.
- **`stable=True` on the sort.** An unstable sort can swap two equal logits, and if the nucleus boundary falls between them, top-p keeps a different token set. The set is the same size and the same probability mass, but not the same *tokens*.

---

## 6. Nondeterminism above the model

Two runs can also diverge without any float ever misbehaving, because the *input* was not the same. These bugs are cheaper to fix and, in my experience, more common than kernel numerics — and they masquerade as numerics because the symptom is identical.

**Retokenization drift.** You detokenize a generation into a string, then something downstream tokenizes that string again — a trainer, an eval harness, a logging pipeline, the next turn of a conversation. The re-encoded ids need not match the ones you generated, even when the strings match exactly. The vLLM blog's Agent Lightning post (2025-10-22) lists the causes: tokenization is not unique (their example: "HAVING" can encode as H + AVING or HAV + ING), tool-call serialization reformats whitespace, and chat templates differ across frameworks. The consequence they state for RL is severe — the resulting off-policy effect is "not even at the token level, cannot be corrected through token-level importance sampling". Their fix is an API change rather than a numerical one: pass `"return_token_ids": true` (vLLM v0.10.2 and later) and the response carries `prompt_token_ids` and `token_ids`, so the consumer never has to guess. If you are building a training loop on top of an inference server, **transport token ids, not strings.** [The tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) works through why the encode/decode round trip is not an identity, and [the BPE tokenizer post](/blog/machine-learning/large-language-model/bpe-tokenizer) covers the merge-order mechanics underneath it.

**Chat template drift.** `apply_chat_template` output depends on the tokenizer config shipped with the checkpoint, the `add_generation_prompt` flag, and whether your framework normalizes message content. vLLM's Kimi K2 debugging post (2025-10-28) is a case study in exactly this: vLLM inspected the signature of `apply_chat_template` and dropped arguments hidden in `**kwargs`, so `add_generation_prompt=True` was never passed and the assistant-turn tokens went missing (fixed in PR #27622); separately, vLLM rewrote `content: ''` into a list form that the model's Jinja template rendered literally. Neither is a numerics problem. Both change the prompt, which changes everything after it. Their diagnostic method is the one to copy: render the template *outside* the server and post the resulting token ids to a completions endpoint, which isolates templating from inference.

**Cache state.** A prefix cache hit and a prefix cache miss are not numerically identical operations. On a miss, the tokens are computed by a prefill kernel over the full prompt. On a hit, the KV entries are read back from blocks that were written by an *earlier* prefill, possibly under a different batch shape, possibly with a different chunking. vLLM's anatomy post (2025-09-05) notes that only complete blocks are cacheable, so a partial prefix recomputes `long_prefix_len % block_size` tokens — meaning the *boundary* between cached and recomputed tokens moves depending on what was cached, and the recomputed segment has a different shape than it did last time. If you want bitwise reproducibility, you must either fix the cache state or disable the cache. [Prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) covers the mechanism in depth; the point here is only that the cache is part of your reproducibility surface.

---

## 7. Building it: nanoserve's determinism knobs and the divergence harness

Now the practical part. There are two things to build: the settings that make a single process as deterministic as it can be, and the harness that tells you whether you succeeded.

### 7.1 The knobs, and what they do not buy you

```python
# nanoserve/determinism.py
import os
import torch


def enable_determinism(strict: bool = True) -> None:
    """Make a single nanoserve process as reproducible as PyTorch allows.

    IMPORTANT: this buys you rung 0 and part of rung 2. It does NOT buy you
    batch invariance. Every knob here fixes the algorithm chosen for a GIVEN
    shape; none of them stop the shape from changing when other requests join
    the running set.
    """
    # cuBLAS workspace must be fixed for its reductions to be reproducible
    # across streams. Must be set BEFORE the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          # autotune picks per shape

    # TF32 silently drops the mantissa to 10 bits inside matmuls. It is a fine
    # default for throughput and a terrible one for logit-parity testing.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # fp32 accumulation for reduced-precision matmuls, where the backend
    # supports choosing.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
```

Every line there earns its place, and the docstring is the important part. `torch.use_deterministic_algorithms(True)` makes PyTorch refuse to dispatch to kernels that use atomics or otherwise produce run-to-run variation *for a fixed input shape*. `cudnn.benchmark = False` stops cuDNN from timing several algorithms on first sight of a shape and caching the winner, which is a classic source of "the first run differs from the rest". Disabling TF32 stops the tensor cores from truncating fp32 inputs to a 10-bit mantissa, which is invisible in most training and extremely visible when you are diffing logits.

And none of it makes the engine batch-invariant. This is the single most common misunderstanding in this area, so it is worth being blunt: **`torch.use_deterministic_algorithms(True)` does not fix the ticket at the top of this post.** It guarantees that shape $S$ always uses algorithm $A$. It does not guarantee that your request always sees shape $S$, and in a continuously batched server it never does.

### 7.2 The harness

This is the artifact. The idea is simple and I have not seen it packaged anywhere, which is why I think it is worth writing: run the *same* request at several batch sizes, pad the batch with filler requests to force the batch dimension without changing the request under test, and report the first step index at which the token ids diverge from the batch-1 reference.

```python
# nanoserve/tools/divergence.py
"""Find the first token index at which a prompt's output depends on batch size.

Usage:
    python -m nanoserve.tools.divergence \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --prompt "Explain why a heap is not a stack." \
        --max-tokens 256 --batch-sizes 1 4 16

Prints one line per batch size. A line reading `identical` means the run was
bit-for-bit equal to the batch-1 reference for the whole generation.
"""
import argparse
from dataclasses import dataclass


@dataclass
class Run:
    batch_size: int
    token_ids: list[int]
    top_gap: list[float]      # logit gap between top-1 and top-2 at each step


def run_at_batch(engine, prompt: str, filler: list[str], n: int, max_tokens: int) -> Run:
    """Generate `prompt` with n-1 filler requests sharing every forward pass.

    The filler prompts must be LONGER than the prompt under test so they do not
    retire early and shrink the batch mid-generation — a shrinking batch would
    change the kernel selection partway through and confuse the measurement.
    """
    reqs = [prompt] + filler[: n - 1]
    outs = engine.generate(
        reqs,
        max_tokens=max_tokens,
        temperature=0.0,           # greedy: isolate the forward pass
        seed=42,
        logprobs=2,                # we want top-2 to measure the gap
    )
    target = outs[0]
    gaps = [lp[0].logprob - lp[1].logprob for lp in target.step_logprobs]
    return Run(batch_size=n, token_ids=target.token_ids, top_gap=gaps)


def first_divergence(ref: Run, other: Run) -> int | None:
    for i, (a, b) in enumerate(zip(ref.token_ids, other.token_ids)):
        if a != b:
            return i
    if len(ref.token_ids) != len(other.token_ids):
        return min(len(ref.token_ids), len(other.token_ids))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from nanoserve.engine import Engine          # or: from vllm import LLM
    engine = Engine(args.model)
    filler = [
        "Write a detailed history of the printing press, at length. " * 8
    ] * max(args.batch_sizes)

    ref = run_at_batch(engine, args.prompt, filler, 1, args.max_tokens)
    print(f"reference: batch=1  tokens={len(ref.token_ids)}")

    # Same-batch repeatability first: if THIS fails, the problem is not
    # batch invariance, it is atomics or autotuning inside one shape.
    for r in range(args.repeats - 1):
        again = run_at_batch(engine, args.prompt, filler, 1, args.max_tokens)
        d = first_divergence(ref, again)
        print(f"  batch=1 repeat {r + 1}: " + ("identical" if d is None else f"diverges at step {d}"))

    for n in args.batch_sizes:
        if n == 1:
            continue
        run = run_at_batch(engine, args.prompt, filler, n, args.max_tokens)
        d = first_divergence(ref, run)
        if d is None:
            print(f"  batch={n:<3}: identical")
        else:
            print(f"  batch={n:<3}: diverges at step {d}  (top-2 gap there: {ref.top_gap[d]:.2e})")


if __name__ == "__main__":
    main()
```

What you should expect to see when you run it against a stock engine on an A100 or a 4090, with a 256-token generation: the batch-1 repeats come back `identical` — that path is usually stable within one process and one shape — and the batch-4 and batch-16 runs diverge somewhere in the first few hundred steps, at a step whose printed top-2 gap is very small, typically several orders of magnitude below 1. If your batch-1 repeats *also* diverge, stop reading about batch invariance and go look for atomics, `cudnn.benchmark`, or an autotuning cache: you have a more basic problem.

![Three rows of harness output showing a batch-1 reference and two padded runs that match until one step and differ afterwards](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-5.webp)

The printed top-2 gap is the part I would not skip. A divergence at a step where the gap is $10^{-6}$ is the expected physics of the machine, and the answer is batch-invariant kernels or nothing. A divergence at a step where the gap is 0.5 is not numerics at all — that is a real bug, and you should go hunting for a mask built from a padded length, a position id computed from a batch-relative offset, or a cache read that picked up the wrong block.

That distinction has saved me more time than anything else in this post. **Numerical divergence lands on near-ties. Logic bugs do not care about the gap.** Print the gap and the two failure modes separate themselves.

### 7.3 The regression test you actually keep

```python
# nanoserve/tests/test_batch_invariance.py
import pytest
from nanoserve.tools.divergence import run_at_batch, first_divergence

PROMPTS = [
    "Summarize the causes of the 1929 crash in three sentences.",
    "def quicksort(a):",                       # code: low-entropy, few near-ties
    "Translate to French: the weather is fine today.",
]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("batch_size", [4, 16])
def test_output_does_not_depend_on_batch(engine, filler, prompt, batch_size):
    ref = run_at_batch(engine, prompt, filler, 1, max_tokens=128)
    run = run_at_batch(engine, prompt, filler, batch_size, max_tokens=128)
    idx = first_divergence(ref, run)
    assert idx is None, (
        f"batch={batch_size} diverges from batch=1 at step {idx}; "
        f"top-2 logit gap there was {ref.top_gap[idx]:.3e}"
    )
```

Mark it `xfail` on a stock engine — it will fail, and that is the correct result to record. Flip it to a hard assertion the day you turn batch-invariant kernels on, and it becomes the test that tells you when an upgrade silently turned them off again.

---

## 8. The four rungs of determinism

"Deterministic" is not one property, and most arguments about it are two people defending different rungs of the same ladder. Name the rung you need.

![Decision tree splitting on whether outputs are compared across runs and whether a gradient is taken through them](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-6.webp)

**Rung 0 — repeatable.** Same input, same process, same hardware, same engine version, *same batch composition* produces the same output. This is what `torch.use_deterministic_algorithms(True)` plus a per-request generator gives you. Cost: essentially zero. Most engines are already close to this and the gap is usually one autotuning cache.

**Rung 1 — batch-invariant.** Same input, same hardware, same version, **any** batch composition produces the same output. This is the rung the support ticket wants. It requires kernels whose reduction order is fixed by construction rather than selected by shape. Cost: real and measured, see section 9.

**Rung 2 — run-to-run reproducible.** Same input reproduces across process restarts, across replicas, across days. Adds requirements beyond rung 1: identical engine and kernel-library versions, identical tensor-parallel degree, identical GPU model, and control over cache state (either a warmed cache with a known content, or no cache). Cost: mostly operational discipline — pinning, and giving up heterogeneous fleets for the workloads that need this.

**Rung 3 — framework-equal.** The inference engine's logits are bitwise equal to a *different* framework's forward pass on the same weights — typically a trainer. This is the hardest rung because two independent codebases must agree on every fused-op boundary, every accumulation dtype, every epsilon placement in every norm. It is also the only rung where the payoff is mathematical rather than aesthetic, which is why it exists at all.

| Rung | Property | What you must control | Typical cost |
| --- | --- | --- | --- |
| 0 | Repeatable in-process | Autotuning, atomics, TF32, per-request RNG | ~free |
| 1 | Batch-invariant | Kernel reduction strategy | 20–40% throughput, cited section 9 |
| 2 | Run-to-run reproducible | Versions, TP degree, GPU model, cache state | Fleet homogeneity |
| 3 | Equal to the trainer | Every fused-op boundary in two codebases | 2.4× on an RL run, cited section 9 |

The ladder is monotone: rung 3 implies rung 2 implies rung 1 implies rung 0. So the practical question is never "should we be deterministic" but "what is the lowest rung that makes our downstream consumer correct", which is section 10.

---

## 9. What determinism costs

Three teams have published the bill with their setups. None of these numbers are mine; I have no GPU and have run nothing. Quote them with their configurations or do not quote them.

| Measure | Reported cost | Setup | Source |
| --- | --- | --- | --- |
| Batch-invariant matmul kernel | ~20% slower than cuBLAS | fixed tile sizes vs shape-selected | cited: He / Thinking Machines Lab, 2025-09-10 |
| Deterministic vLLM, first cut | 26 s to 55 s (2.1×) | Qwen3-8B, 1000 sequences | cited: He / Thinking Machines Lab |
| Deterministic vLLM, improved attention kernel | 26 s to 42 s (1.6×) | same 1000-sequence run | cited: He / Thinking Machines Lab |
| SGLang deterministic mode, average | +34.35% latency | Qwen3-8B, H200 140GB, TP1, FlashInfer and FA3 | cited: LMSYS / SGLang, 2025-09-22 |
| SGLang, FlashInfer backend | +42.6% latency | 1024 in / 1024 out, same setup | cited: LMSYS / SGLang |
| SGLang, FA3 backend | +27.2% latency | same setup | cited: LMSYS / SGLang |
| Bitwise RL training run, end to end | 2.4× slower | TorchTitan + vLLM, Qwen3 1.7B, GSM8K | cited: vLLM, 2025-11-10 |
| fp32 sampler upcast in nanoserve | <1% of a decode step | Llama-3.1-8B, batch 64, A100 80GB | derived below |
| Per-request `torch.Generator` | ~0 | one small object per request | derived |

Two observations about that table.

First, the spread — 1.6× to 2.4× depending on how much kernel work has been done — is not noise, it is a maturity curve. He's own numbers show the cost dropping from 2.1× to 1.6× with one improved attention kernel. SGLang's report of avg +34.35% is lower still, and their post attributes part of that to integrating CUDA graphs with the deterministic path. Determinism is expensive because the fast path has been optimized for years and the invariant path has not. It is not expensive because invariance is intrinsically slow.

Second, the RL number is a *different kind* of number. The 2.4× from vLLM's bitwise-consistency post is end-to-end wall clock on a training run, not a per-request latency, and it is paid against a payoff we will get to in section 10.

#### Worked example: what the fp32 sampler actually costs

The one determinism measure this post asks you to take unconditionally is running the sampler in fp32. Let us price it so you can stop worrying about it.

Llama-3.1-8B has a vocabulary of 128,256 tokens. Per request per decode step:

- bf16 logits: $128{,}256 \times 2 = 256.5$ KB
- fp32 logits: $128{,}256 \times 4 = 513$ KB

At batch 64, the fp32 buffer is $64 \times 513\ \text{KB} = 32.8$ MB against 16.4 MB in bf16 — an extra 16.4 MB of traffic per pass over the logits. An A100 80GB SXM lists 2,039 GB/s of HBM2e bandwidth (NVIDIA A100 datasheet), so one extra pass over 16.4 MB costs

$$\frac{16.4 \times 10^{6}\ \text{B}}{2.039 \times 10^{12}\ \text{B/s}} \approx 8\ \mu s$$

Even at five passes (upcast, temperature divide, mask, softmax, argmax) that is about 40 µs. Compare it with the decode step itself: at batch 64 the step is still dominated by streaming the weights, $8.03 \times 10^{9}$ parameters at 2 bytes each is 16.1 GB, and

$$\frac{16.1\ \text{GB}}{2.039\ \text{TB/s}} \approx 7.9\ \text{ms}$$

So the fp32 sampler is roughly $40\ \mu s / 7.9\ \text{ms} \approx 0.5\%$ of the step. *Source: derived from the vocabulary and parameter counts in the Llama-3.1-8B model card and the A100 datasheet bandwidth; the five-pass count is an upper-bound assumption on a naive implementation, and a fused sampler does better.*

**Run your sampler in fp32.** It costs half a percent and it removes an entire class of manufactured ties. If somebody argues about it, show them this arithmetic. The expensive part of determinism is not here — it is in the attention and matmul kernels, and that is where the argument belongs.

#### Worked example: what non-determinism costs an eval

The reverse accounting is more persuasive to a product owner. Suppose you run a 500-prompt eval, 300 output tokens each, and you use the per-step flip probability derived in section 4, $p \approx 0.0097$.

From the table in section 4, the probability a single prompt's text is identical across two runs at 300 tokens is 5.4%. So of your 500 prompts, roughly $500 \times 0.054 \approx 27$ produce identical text between runs and **473 do not**. A text diff between two runs of your eval is therefore useless as a signal — it is red on essentially every row, whether or not you changed anything.

The score is more stable than the text, because most divergences are cosmetic. But not perfectly: suppose 2% of divergences change whether an item is graded correct — a labeled assumption, not a measurement; substitute your own rate by rerunning the eval twice and counting grade flips. With 500 items and a per-item flip probability of 0.02, the number of flipped grades is binomial with standard deviation

$$\sigma = \sqrt{500 \times 0.02 \times 0.98} \approx 3.13\ \text{items} \approx 0.63\ \text{percentage points}$$

and the difference between two runs has standard deviation $\sqrt{2} \times 0.63 \approx 0.89$ points. So a 1-point regression in your eval is inside the noise of your own harness. If your model-quality decisions are made on 1-point movements — and most are — then you have been reading noise, and rung 1 is not a luxury, it is the precondition for the number to mean anything. *Source: derived; the 2% grade-flip rate is a stated assumption you should measure for your own eval.*

---

## 10. Who needs which rung

![Five workloads mapped against the rung they need, what breaks without it, and the price of getting it](/imgs/blogs/sampling-numerics-determinism-and-batch-invariance-7.webp)

**A chat product: rung 0.** Users do not resubmit the same prompt and diff the bytes. They will not notice, and paying 30% throughput for a property nobody observes is a bad trade — you are converting money into an aesthetic. Ship the cheap hygiene (per-request generator, fp32 sampler, no shared RNG) and stop.

**Regression tests and golden traces: rung 1.** If your CI asserts that a prompt produces a known output, and the CI runner's batch composition varies with what else is running, the test is flaky by construction and the team will disable it within a month. Either make the engine batch-invariant, or pin the test path to batch 1 with the rest of the queue drained — a legitimate cheaper option if your CI can afford a dedicated single-tenant run.

**Evals and leaderboards: rung 1.** The arithmetic above is the argument. Without batch invariance your eval score carries a run-to-run standard deviation that you cannot separate from real regressions, and every model comparison inherits it.

**Debugging an incident: rung 2.** A user reports a bad answer. You need to reproduce it to fix it. Rung 2 means recording enough state — engine version, TP degree, GPU model, seed, cache configuration — that you can replay it tomorrow on a different box. Notice that most of this rung is *logging*, not kernels. Log the engine build and the sampling parameters with every request and you get most of the value for nothing.

**RL training: rung 3, and the argument deserves spelling out.** Policy-gradient methods assume the samples you train on were drawn from the policy you are updating. If the sampler (your inference engine) and the trainer (your training framework) compute even slightly different probabilities for the same tokens, the data is off-policy with respect to the thing being differentiated. The correction machinery — importance sampling ratios — exists for exactly this, but it only works if you can *write down* the sampling distribution. Numerical drift between two frameworks is not a distribution you can write down. Agent Lightning's post makes the sharpest version of the point for the retokenization case: the resulting mismatch is "not even at the token level, cannot be corrected through token-level importance sampling."

The payoff, when you pay for rung 3, is reported by vLLM's bitwise-consistency post: with batch-invariant, bitwise-matched kernels the measured `kl_div` between sampler and trainer stays exactly 0.0, and their run reached **higher total reward in fewer steps** than the non-bitwise baseline (TorchTitan + vLLM, Qwen3 1.7B, GSM8K). That is the shape of the trade: 2.4× slower per unit of wall clock, but fewer steps needed and a better endpoint. Whether that nets out positive is a question about your run, not a general law — but it is the only case in this post where determinism buys *quality* rather than just *comfort*.

| Consumer | Rung | Cheapest sufficient action |
| --- | --- | --- |
| Chat / assistant product | 0 | Per-request generator, fp32 sampler |
| Streaming API with retries | 0 | Idempotency key so a retry replays, not regenerates |
| CI golden traces | 1 | Batch-invariant kernels, or pin CI to batch 1 |
| Evals, model comparison | 1 | Batch-invariant kernels; report score sigma if not |
| Incident reproduction | 2 | Log version, TP, GPU, seed, cache mode per request |
| RL / distillation on rollouts | 3 | Batch-invariant kernels plus token-id transport |

---

## 11. Stress tests: where it gets worse

A design is only trustworthy after you push on it. Four pushes.

**Change the tensor-parallel degree.** Split a 4096-dimensional reduction across TP=2 and each rank sums 2048 terms locally, then an all-reduce adds 2 partials. At TP=4 each rank sums 1024 terms and the all-reduce adds 4. Different order, different bits — so *the same model on the same GPUs with a different TP degree is a different function*. Worse, collective libraries choose their algorithm by message size: ring for large payloads, tree for small ones, with thresholds. Since message size scales with batch size, **the all-reduce algorithm can change with batch size too**, which is the section-4 problem arriving through a second door. Rung 2 therefore requires pinning TP degree, and a serious rung-2 setup pins the collective algorithm as well.

**Hit the prefix cache instead of missing it.** Section 6 covered the mechanism; here is the operational shape. Request A arrives cold and prefills 1,000 tokens. Request B arrives with the same 900-token system prompt and hits the cache. B's first 900 tokens are not recomputed — they are read from blocks written under A's batch shape. With vLLM's default block size of 16 (per their anatomy post, 2025-09-05), a prefix of 907 tokens caches 56 complete blocks and recomputes the remaining 11 tokens, and that recomputed tail runs under *B's* batch shape. So B's numerics are a mixture of A's prefill conditions and B's own. This is not a defect in prefix caching — the vLLM V1 post reports the feature costs under 1% throughput even at a 0% hit rate, which is why it is on by default and should stay on. It is a statement that **cache state is part of your input** for reproducibility purposes. A rung-2 harness either warms the cache identically before every measurement or disables it.

**Turn on speculative decoding.** Speculation changes the verification shape: instead of one forward pass over one position, the target model runs over $k+1$ positions at once, so the token at position $t$ is computed inside a different-shaped kernel launch than plain decode would use. Rejection sampling makes the accepted tokens *distributionally* equal to what the target model would have produced — that is the correctness guarantee, and it is a real one — but distributional equality is not bitwise equality, and the number of tokens accepted at each step depends on the draft model's agreement, which itself varies with batch. A speculative engine and a non-speculative one running the same prompt will diverge, and two speculative runs at different batch sizes will diverge from each other. If you need rung 1 or above, turn speculation off, or accept that the guarantee you have is distributional. [The speculative decoding core-idea post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) works through the draft-and-verify contract that makes the distributional claim true.

**Move to different hardware.** An L4 and an A100 do not have the same SM count, so a shape-selected kernel fills them differently. FlashAttention and FlashInfer ship architecture-specific code paths; the vLLM Triton backend post (2026-03-04) reports Triton reaching 100.7% of FlashAttention 3's performance on an H100 for Llama 3.1 8B at batch 1 with a 500-token input and long decode, and being the *default* on AMD ROCm — three different backends across your fleet is three different reduction orders. Bitwise reproducibility across GPU architectures is, in practice, out of reach without giving up the tuned kernels entirely. If your requirement is written as "same answer everywhere", push back on it and find out whether the requirement is really "same answer on the eval box", which is achievable. [The attention backends deep dive](/blog/machine-learning/model-serving/attention-backends-deep-dive-flashattention-flashinfer) covers how much these implementations differ internally.

---

## 12. Case studies and real numbers

Four public results, each cited with its setup.

**Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (Horace He, 2025-09-10).** The reference treatment, and the source of the term used in this post's title. It defines batch invariance as fixing the reduction order regardless of kernel batch size, gives per-op strategies (data-parallel RMSNorm, fixed-tile matmul, fixed-split attention), and releases `batch-invariant-ops` with a vLLM integration example. The headline demonstration: 1,000 completions of "Tell me about Richard Feynman" from Qwen3-235B at temperature 0 produced 80 unique completions with a first divergence at token 103; with batch-invariant kernels, all 1,000 were identical. Performance on Qwen3-8B over 1,000 sequences: 26 s for default vLLM, 55 s for the unoptimized deterministic path, 42 s after improving the attention kernel. Read this one first.

**vLLM, *Bitwise-consistent train/inference determinism* (2025-11-10).** The production-scale follow-through, and the source of the crispest statement of the root cause: "Kernels for high batch sizes parallelize heavily on the batch dimension, while kernels for low batch sizes parallelize more within a single instance." The work targeted vLLM's fused SiLU MLPs and RMSNorms with added residuals, then implemented matching backward passes in vanilla PyTorch for TorchTitan. Result on Qwen3 1.7B, GSM8K: the bitwise RL run is 2.4× slower than the non-bitwise case, `kl_div` stays 0.0, and the model reaches higher total reward in fewer steps. Note that this post credits vLLM's own engineering and does not itself name the Thinking Machines work; the two are cited here separately because they are separate publications making compatible claims.

**LMSYS / SGLang, *Towards Deterministic Inference in SGLang and Reproducible RL Training* (2025-09-22).** The independent replication, and the most operationally detailed of the three. SGLang integrated batch-invariant kernels and made them work alongside chunked prefill, CUDA graphs, the radix cache and non-greedy sampling, exposing it as `--enable-deterministic-inference`. Supported attention backends and their strategies: FlashInfer with fixed split sizes, FlashAttention 3 with `num_splits` fixed to 1, and Triton (which extends the support to AMD). For sampling above temperature 0 they added a seeded hash function, `multinomial_with_seed`, with per-request seeds in the sampling arguments — the same structural idea as section 5's per-request generator. Cost on Qwen3-8B, H200 140GB, TP1: an average 34.35% slowdown across the FlashInfer and FA3 backends, with FlashInfer at +42.6% and FA3 at +27.2% for 1024-in/1024-out. Reproducibility result: identical rollout responses and loss values for the first iterations of GRPO training in the slime framework. That this replicates independently in a second engine is the strongest evidence that the mechanism is real and general.

**vLLM, *Agent Lightning* (2025-10-22).** The above-the-model case. Detokenizing at inference and retokenizing at training can produce different ids even when the strings match, because tokenization is not unique, tool-call serialization reformats whitespace, and chat templates differ across frameworks. The fix is `"return_token_ids": true` (v0.10.2+), which adds `prompt_token_ids` and `token_ids` to the response. Cite this one at anybody who is chasing kernel numerics before they have checked that their two frameworks agree on the token ids.

**vLLM, *Model Runner V2* (2026-03-24).** Relevant here for one design detail: the Triton sampler uses Gumbel-Max without materializing an explicit softmax, with a stateless in-kernel RNG. That is the section-5.1 construction shipped in a production engine, and stateless in-kernel RNG is the strongest form of the per-request generator idea — the random number for `(request, step, vocab index)` is *computed* from those coordinates rather than read from a stream whose position depends on who drew before you.

---

## 13. When to reach for this (and when not to)

**Do this always, because it is free.** Per-request `torch.Generator` seeded from a stable hash of the request. fp32 in the sampler. Explicit stable sort and a documented tie-break. Log the engine version, GPU model, TP degree, sampling parameters and seed with every request. If a request carries a client seed, honour it end to end. Total cost: under 1% of a decode step by the arithmetic in section 9, and it removes the three cheapest classes of "unreproducible" report.

**Do this when a machine consumes your output.** Batch-invariant kernels — via `batch-invariant-ops`, or SGLang's `--enable-deterministic-inference`, or whatever your engine exposes by the time you read this. The trigger is not "we want determinism", it is "something downstream diffs or differentiates our tokens". Evals, golden-trace CI, RL rollouts, distillation datasets. Budget 30% to 60% latency by the cited numbers, and validate against your own workload because all three published figures are for 8B-class models on single GPUs.

**Do not do this for a chat product.** Nobody is diffing your bytes. The 30% you would spend is real money and real p99, and you would be buying a property with no consumer. If a customer asks for reproducibility, ask what they will do with it — the answer is usually "compare two runs", which you can serve better by giving them a pinned single-tenant endpoint for their eval rather than degrading everyone's throughput.

**Do not write batch-invariant kernels yourself.** This is the one place in this series where I will tell you plainly to take the library. `nanoserve` is a teaching engine and writing its own RMSNorm is educational; writing a batch-invariant fixed-split attention kernel that is within 40% of FlashAttention is a serious multi-week project, and both He's library and SGLang's integration exist and are maintained. Build the harness, take the kernels.

**Do not chase bitwise parity across GPU architectures.** It is achievable in the sense that you can force every reduction to a fixed order on every device, and it is unachievable in the sense that you will have thrown away the tuned kernels on all of them. Fix the eval box instead.

---

## Key takeaways

1. **The sampler is innocent.** `argmax` and a seeded `multinomial` are pure functions. If your output changed, the logits changed.
2. **Floating-point addition is not associative**, and you can prove it in bf16 with three numbers: ${(256 + 1) + 1 = 256}$ while ${256 + (1 + 1) = 258}$.
3. **Reduction order is a performance decision, not a semantic one.** Sequential and tree summation of a 4096-term dot product differ in typical error by roughly $18\times$, and neither is wrong.
4. **The mechanism behind the ticket is batch-size-dependent kernel selection**: low-batch kernels parallelize within an instance, high-batch kernels parallelize across the batch, and they sum in different orders. In a continuously batched server, your logits depend on other users' traffic.
5. **A $10^{-5}$ perturbation is invisible until it lands on a near-tie**, and near-ties are common — inverting the cited first-divergence-at-token-103 result gives roughly one flippable step in a hundred. Short answers look reproducible; 1000-token answers essentially never are.
6. **`torch.use_deterministic_algorithms(True)` does not give you batch invariance.** It fixes the algorithm for a given shape; it does not fix the shape.
7. **Run the sampler in fp32.** Derived cost for Llama-3.1-8B at batch 64 on an A100: about 0.5% of a decode step. It eliminates manufactured ties for free.
8. **Give every request its own generator**, seeded from a hash that includes the request id, so no request's tokens depend on its position in the batch and two clients sending `seed=42` do not collide.
9. **Name your rung** — repeatable, batch-invariant, run-to-run reproducible, or equal to the trainer — and buy only the lowest one your consumer needs. Evals need rung 1; RL needs rung 3; a chat product needs rung 0.
10. **Ship the divergence harness before the fix.** One prompt, three batch sizes, print the first divergent step index and the top-2 logit gap at that step. A tiny gap is numerics; a large gap is a bug in your masking or position ids.

---

## Further reading

- Horace He and Thinking Machines Lab, [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025-09-10) — the definition of batch invariance, the per-op strategies, and the `batch-invariant-ops` library.
- vLLM, [*Bitwise-consistent train/inference determinism*](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) (2025-11-10) — the root-cause statement, the SiLU/RMSNorm work, and the 2.4×-slower RL run with `kl_div` at 0.0.
- LMSYS / SGLang, [*Towards Deterministic Inference in SGLang and Reproducible RL Training*](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) (2025-09-22) — `--enable-deterministic-inference`, backend-by-backend strategies, and the measured 34.35% average slowdown.
- vLLM, [*Agent Lightning: return token IDs to prevent retokenization drift*](https://vllm.ai/blog/2025-10-22-agent-lightning) (2025-10-22) — why strings are the wrong transport between a sampler and a trainer.
- vLLM, [*vLLM Triton Attention Backend Deep Dive*](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) (2026-03-04) — split-KV decode, heuristic-gated split selection, and why no single configuration dominates.
- [PyTorch: Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) — the authoritative list of knobs, including `CUBLAS_WORKSPACE_CONFIG` and which ops have no deterministic implementation.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) for the scoreboard, [the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) for the decoding knobs whose numerics this post audits, [writing a continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) for the loop that makes batch composition vary in the first place, [the tokenizer boundary](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) for the encode/decode round trip, and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) for where this sits in the final decision tree.
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) — the measurement discipline (warmup, `torch.cuda.synchronize()`, CUDA events, locked clocks, steady state) that everything in section 7 assumes.
