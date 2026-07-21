---
title: "From logits to tokens: the sampler zoo"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Everything before this post produced a vector of numbers. This post turns it into a token — deriving what temperature, top-k, top-p, min-p, typical-p and the penalty family actually do to a distribution, proving that the order you apply them in changes the answer, and building a composable batched sampler that never syncs with the host."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "decoding",
    "sampling",
    "logits",
    "temperature",
    "top-p",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 64
---

Fifteen posts of this series have been spent producing one thing: a vector of logits. You loaded the weights, wrote the forward pass, built a KV cache, paged it into blocks, shared prefixes across requests, wrote a continuous-batching loop and a scheduler that decides who runs. All of that machinery exists to compute, at every step, a row of roughly 128,000 floating-point numbers. And then, in almost every codebase I have read, that row is handed to four lines of Python that nobody owns, nobody tests, and nobody profiles.

Those four lines decide what your model sounds like. Not the weights — the weights are fixed. Not the prompt — the prompt is the user's. The sampler is the only part of the serving path that changes the *content* of the output, and it is routinely configured by copying a preset off a forum post. Two teams running the identical checkpoint, on identical hardware, with identical prompts, will get outputs that a blind reader can tell apart, because one set `top_p=0.95` and the other set `top_p=0.9` with `min_p=0.05`. Worse: two teams running the identical *settings* on two different engines can also get different distributions, because the engines apply those settings in a different order.

![The decoding layer drawn as six ordered passes over one logit row ending in a single token id](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-1.webp)

This post is the opening of Track D, the decoding layer. By the end you will have `nanoserve/sampling.py`: a `LogitsProcessor` protocol, one small class per knob, a `SamplingParams` that composes them in a defined and documented order, and a batched sampler that handles a running set where every request wants different parameters — which is the part that toy code always skips and that every real engine has to solve. You will also have the derivations: what temperature does to entropy, why min-p is a fixed distance in logit space, why a frequency penalty is unbounded and therefore fatal to structured output, and why the Gumbel-Max trick lets you sample without ever materializing a softmax or touching the host.

Standard promise, restated from [the series introduction](/blog/machine-learning/inference-engineering/what-inference-engineering-is): **I have no GPU and I have run none of this.** Every number below is either arithmetic I show you in full, a citation with a link, or a range you should expect when you run the script yourself. The tables carry a `Source` column. The distributions I print are hand-computed and you can check them with a calculator — that is the point.

---

## 1. What a sampler actually is

A logit is an unnormalized log-probability. The final linear layer of the model — the "LM head" — maps the last hidden state of shape `[d_model]` to a vector of shape `[vocab_size]`, and that vector is the logits. For Llama-3.1-8B the vocabulary is 128,256 tokens, so a batch-1 decode step produces a `[1, 128256]` row. Nothing about that row is a probability yet. The numbers are typically in the range of roughly −20 to +20, they have no fixed scale across models or even across positions, and — this matters enormously later — they are only defined up to an additive constant, because softmax is shift-invariant:

$$\text{softmax}(z + c)_i = \frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}} = \text{softmax}(z)_i$$

The sampler's job is to convert that row into exactly one integer in `[0, vocab_size)`. It does so in six passes, and figure 1 above is the whole layer:

1. **Penalties** — modify logits for tokens the request has already seen.
2. **Temperature** — divide every logit by a scalar $T$.
3. **Truncation** — set some logits to $-\infty$ so those tokens cannot be chosen.
4. **Renormalize** — softmax over what survives.
5. **Sample** — draw one index from the resulting categorical distribution.
6. **Return** — one integer, four bytes, which has to get back to the request state.

Every user-facing decoding parameter you have ever set lives in pass 1, 2 or 3. Passes 4, 5 and 6 are where the engineering lives, and they are where the performance goes wrong.

Two things deserve to be said out loud before the details.

**The sampler is stateful per request.** Temperature and top-p are stateless — the same logits give the same distribution. Penalties are not: they depend on which tokens this request has already emitted, and often on the prompt too. That means the sampler cannot be a pure function of the logits row; it needs a per-request history, and in a batched engine that history has to be materialized as tensors, not Python lists.

**The sampler is the only random part of inference.** [The naive decode loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) used `argmax` precisely so the loop would be reproducible under measurement. The moment you introduce randomness you inherit an entire class of problems — seeding, per-request reproducibility, batch-dependent RNG consumption — which the sibling post on [sampling numerics, determinism and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) takes apart in detail. Here I will flag where randomness enters and how to keep it per-request; that post handles the numerics.

### The strawman sampler

Here is the sampler almost everyone writes first, and which I have found in production more than once:

```python
# The sampler everyone writes first. Four bugs and a performance disaster.
import torch


def sample_naive(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits: [1, T, V] straight out of the model."""
    z = logits[0, -1, :].float()                       # [V]

    if temperature > 0:
        z = z / temperature

    if top_k > 0:
        kth = torch.topk(z, top_k).values[-1]
        z = z.masked_fill(z < kth, float("-inf"))

    if top_p < 1.0:
        s, idx = torch.sort(z, descending=True)
        probs = s.softmax(dim=-1)
        cutoff = (probs.cumsum(dim=-1) - probs) >= top_p
        s = s.masked_fill(cutoff, float("-inf"))
        z = torch.full_like(z, float("-inf")).scatter(0, idx, s)

    probs = z.softmax(dim=-1)
    return int(torch.multinomial(probs, num_samples=1))   # host sync
```

It works. It even produces the right distribution most of the time. It is also wrong in four separate ways, and the rest of this post is essentially a guided tour of those four ways:

- **The order is hardcoded and undocumented.** Temperature runs before truncation. That is a choice, it changes the answer, and nothing in the signature admits it (section 2).
- **`temperature > 0` silently means "greedy"** if you pass zero — except it does not, because a zero temperature here just skips the division and leaves you sampling from the raw distribution. A real engine has to treat $T = 0$ as a mode, not a value (section 3).
- **It handles one request.** In a continuous batch you have 64 rows with 64 different parameter sets, and calling this in a Python loop costs you 64 kernel launches per knob (section 8).
- **`int(...)` is a host sync.** The CPU stops and waits for the GPU. At batch 1 with a big model that is free. At batch 256 with a small model it can cost you a third of your throughput (section 9).

---

## 2. The order is a choice, and it changes the answer

This is the single most useful thing in this post, so it goes second, before the individual knobs.

Temperature and truncation do not commute. Applying temperature to the logits and *then* choosing a nucleus gives a different candidate set than choosing the nucleus first and then applying temperature to the survivors. Not "slightly different in the tail" — different by whole tokens, which means different by whole outputs.

![Two orderings of the same temperature and top-p settings produce a four token candidate set and a three token candidate set](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-2.webp)

#### Worked example: the same two knobs, two orders

Take a six-token logit row. The prompt is "The cat sat on the", and the candidates are `mat`, `floor`, `couch`, `roof`, `table`, `bed` with logits:

$$z = [\,3.2,\ 2.7,\ 1.9,\ 1.1,\ 0.4,\ -0.8\,]$$

At $T = 1$ the softmax is (exponentiate, sum, divide — every digit below is checkable):

| token | $z$ | $e^{z}$ | $p$ at $T = 1$ |
| --- | --- | --- | --- |
| mat | 3.2 | 24.5325 | 0.4806 |
| floor | 2.7 | 14.8797 | 0.2915 |
| couch | 1.9 | 6.6859 | 0.1310 |
| roof | 1.1 | 3.0042 | 0.0589 |
| table | 0.4 | 1.4918 | 0.0292 |
| bed | −0.8 | 0.4493 | 0.0088 |
|  | | sum 51.0434 | 1.0000 |

Now set `temperature = 1.5` and `top_p = 0.9` and run the two orders.

**Order A — temperature first, then top-p.** Divide the logits by 1.5 to get $[2.1333, 1.8, 1.2667, 0.7333, 0.2667, -0.5333]$, exponentiate to $[8.4428, 6.0496, 3.5490, 2.0820, 1.3056, 0.5866]$, sum 22.0157, normalize:

$$p_{T=1.5} = [\,0.3835,\ 0.2748,\ 0.1612,\ 0.0946,\ 0.0593,\ 0.0266\,]$$

Cumulative sum: 0.3835, 0.6583, 0.8195, **0.9141**. The smallest set reaching 0.9 has **four** tokens: `mat`, `floor`, `couch`, `roof`. Renormalizing over 0.9141 gives final probabilities 0.4195, 0.3006, 0.1764, 0.1035.

**Order B — top-p first, then temperature.** Take the nucleus from the untempered distribution: cumulative sums 0.4806, 0.7721, **0.9031**. Three tokens: `mat`, `floor`, `couch`. Now apply $T = 1.5$ to those three logits and renormalize: exponentials 8.4428, 6.0496, 3.5490, sum 18.0415, giving

$$p = [\,0.4680,\ 0.3353,\ 0.1967\,]$$

Same model, same prompt, same two parameter values. Order A can emit `roof` with probability 0.1035. Order B assigns it probability exactly zero — it is not unlikely, it is *unreachable*. And the leading token's probability differs by 4.8 percentage points. Source for every number in this example: `derived`.

The general rule is easy to state once you see the arithmetic. Temperature above 1 flattens the distribution, which *pushes mass into the tail*, which makes top-p's nucleus *larger*. Temperature below 1 sharpens it, which makes the nucleus *smaller*. So running temperature first makes top-p temperature-dependent; running it second makes top-p a property of the model's raw beliefs. Neither is wrong. But if you tune `top_p` on one order and deploy on the other, your tuning does not transfer.

### Which order does your engine use?

The honest answer is: check. The convention that most people have absorbed comes from Hugging Face `transformers`, where the processors that implement penalties are applied first, and the ones that implement temperature and truncation are constructed in the order temperature, then top-k, then top-p, then typical, then the epsilon and eta variants — see the [generation utilities documentation](https://huggingface.co/docs/transformers/en/internal/generation_utils) for the full list of `LogitsProcessor` classes. So the effective chain is penalties, temperature, truncation.

llama.cpp goes further and makes the chain an explicit, user-supplied ordered list via its `--samplers` option (see the [llama.cpp repository](https://github.com/ggml-org/llama.cpp)). That is the clearest possible admission that the order is a policy choice rather than a law of nature — if it were a law, there would be nothing to configure.

And the OpenAI-compatible HTTP surface that most of us serve behind exposes `temperature`, `top_p`, `presence_penalty` and `frequency_penalty` with **no ordering field at all**. A preset that produces good output on one backend is not portable to another, and no field in the request can express the difference.

So: assume nothing, and measure. Here is a fifteen-line script that determines your engine's order empirically, without reading its source:

```python
# order_probe.py — determine whether temperature runs before or after top-p.
#
# Construct a distribution where the two orders admit a different number
# of tokens, then sample many times and count distinct tokens observed.
import collections

# Six logits chosen so that at T=1.5 the 0.9-nucleus has 4 members and at
# T=1.0 it has 3. See the worked example in section 2.
PROBE_LOGITS = [3.2, 2.7, 1.9, 1.1, 0.4, -0.8]

def probe(sample_fn, n=20000):
    """sample_fn(temperature, top_p) -> token index, using PROBE_LOGITS."""
    seen = collections.Counter()
    for _ in range(n):
        seen[sample_fn(temperature=1.5, top_p=0.9)] += 1
    distinct = sum(1 for v in seen.values() if v > 5)
    print(f"distinct tokens observed: {distinct}")
    print("temperature FIRST" if distinct >= 4 else "truncation FIRST")
    return seen
```

Run it against your engine's sampling entry point with a fixed six-logit row (most engines let you feed logits directly through a test hook; failing that, use a tiny model with a six-token vocabulary). Four distinct tokens means temperature runs first. Three means truncation does.

### What nanoserve does, and why

`nanoserve` fixes the order as **penalties → temperature → truncation → renormalize → sample**, matching the `transformers` convention, and states it in the docstring of `SamplingParams` so nobody has to guess. But it makes one refinement that I would argue every engine should make: **the truncators do not chain, they intersect.**

The subtlety is that top-k, top-p and min-p are themselves order-sensitive if each one recomputes a softmax over the survivors of the previous one. Apply top-k first and the probabilities that top-p sees are renormalized over $k$ tokens, so a nucleus of "0.9" means something different. The fix is to compute the post-temperature distribution *once*, have each truncator produce a boolean keep-mask against that single distribution, and take the intersection. Then the three truncators commute, `top_k=50, top_p=0.9` means exactly "the top 50 tokens **and** the 0.9-nucleus", and nobody has to reason about which ran first. That is one design decision that removes an entire category of confusion, and it costs nothing — the masks are computed in the same sorted pass anyway (section 8).

---

## 3. Greedy: the sampler you should reach for more often

`argmax` is the degenerate case: temperature zero, no randomness, always pick the highest logit. It gets dismissed as the boring option, and that dismissal costs people accuracy.

$$\text{next} = \arg\max_i z_i$$

Formally it is the $T \to 0$ limit of softmax sampling. As $T$ shrinks, $e^{z_i/T}$ for the maximum logit grows faster than for every other, so the distribution converges to a point mass on the argmax. Two properties follow.

**It is the correct choice when there is a right answer.** Extraction, classification, structured field-filling, tool-argument generation, retrieval-grounded question answering with a short span answer, and — critically — **evaluation runs**. If you are measuring your model's accuracy on a benchmark and you are sampling at $T = 0.7$, you are measuring your model *plus a random number generator*, and your error bars have to absorb the sampler's variance. Most published eval numbers for non-reasoning tasks are greedy for exactly this reason. When someone tells me their eval numbers moved 1.5 points between runs, my first question is what temperature they used.

**It produces degenerate loops on open-ended text.** This is the famous result from Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (ICLR 2020) — the paper that introduced nucleus sampling. Maximization-based decoding on open-ended generation collapses into repetition: the model finds a high-probability cycle and rides it. The paper's diagnosis is that the distribution of human text has much higher entropy than the mode of the model's distribution, so always taking the mode produces text that is, in a measurable sense, *too likely*.

Greedy also has one property that people assume it has and it does not: **it is not automatically deterministic across batch sizes**. The argmax of a logit row is deterministic given that row, but the row itself depends on floating-point reduction order in the matmuls that produced it, and reduction order changes with batch shape. That is the whole subject of the [batch invariance post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance); the takeaway here is that "temperature 0" buys you reproducibility of the *sampler*, not of the *engine*.

One implementation note that will matter in section 8: in a batched sampler, $T = 0$ cannot be handled by division. You either branch, which breaks vectorization, or you substitute a safe temperature, sample normally, and then overwrite the $T = 0$ rows with their argmax. The second option is what real engines do and what `nanoserve` will do.

---

## 4. Temperature is an entropy dial, not a creativity dial

Temperature divides the logits before the softmax:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

![One logit row branching into three temperatures and merging into an unchanged argmax with a rising entropy](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-3.webp)

Three facts fall out of that formula immediately, and all three are things people get wrong.

**As $T \to 0^+$, the distribution converges to a point mass on the argmax.** Write $p_i / p_{\max} = e^{(z_i - z_{\max})/T}$. Since $z_i - z_{\max} \le 0$, the exponent goes to $-\infty$ for every non-maximal token as $T$ shrinks, so every ratio goes to zero and the argmax takes all the mass.

**As $T \to \infty$, the distribution converges to uniform.** Same expression: the exponent goes to zero for every token, so every ratio goes to one, so all tokens are equiprobable. At $T = 100$ you are sampling uniformly from a 128,256-token vocabulary, which is to say from byte-fallback fragments and Korean punctuation.

**The argmax never changes.** Division by a positive scalar preserves order. Whatever token was most likely at $T = 1$ is most likely at $T = 0.1$ and at $T = 5$. This is why "raise the temperature to make it more creative" is a misleading description: temperature does not change what the model *thinks*, it changes how much of the model's uncertainty you are willing to act on. The right mental description is that temperature scales entropy.

Here is the same six-token row at three temperatures, with the Shannon entropy in nats computed as $H = -\sum_i p_i \ln p_i$:

| token | $T = 0.7$ | $T = 1.0$ | $T = 1.5$ | $T = 2.0$ |
| --- | --- | --- | --- | --- |
| mat | 0.5824 | 0.4806 | 0.3835 | 0.3297 |
| floor | 0.2851 | 0.2915 | 0.2748 | 0.2568 |
| couch | 0.0909 | 0.1310 | 0.1612 | 0.1721 |
| roof | 0.0290 | 0.0589 | 0.0946 | 0.1154 |
| table | 0.0107 | 0.0292 | 0.0593 | 0.0813 |
| bed | 0.0019 | 0.0088 | 0.0266 | 0.0446 |
| **entropy (nats)** | **1.054** | **1.289** | **1.504** | **1.610** |

Source: `derived` — every column is `softmax(z / T)` on the logit row from section 2, and the entropies are the sum of $-p \ln p$ over the column. The maximum possible entropy over six tokens is $\ln 6 = 1.792$ nats, so $T = 2.0$ has already spent 90% of the available uncertainty on a distribution whose top token the model still believes at 33%.

Notice what temperature does to the tail specifically. `bed` goes from 0.19% at $T = 0.7$ to 4.5% at $T = 2.0$ — a factor of **23**. The head barely moves (0.58 to 0.33, a factor of 1.8). Temperature is overwhelmingly a *tail* control, and since a real vocabulary has 128,000 tail entries rather than three, this is the mechanism behind almost every "the model went insane at high temperature" report. Section 5 makes that quantitative.

The animation below shows the whole sampler acting on a ten-token distribution: temperature sharpening the head, top-p deleting the tail, and finally a frequency penalty knocking the leader below the runner-up.

<figure class="blog-anim">
<svg viewBox="0 0 700 280" role="img" aria-label="A ten token probability bar chart passes through four states: temperature one, temperature zero point six, top-p truncation, and a frequency penalty that flips the leader" style="width:100%;height:auto;max-width:820px">
<style>
.sz-bar{fill:var(--accent,#6366f1)}
.sz-tail{fill:var(--accent,#6366f1)}
.sz-base{stroke:var(--border,#d1d5db);stroke-width:1.5}
.sz-tok{font:500 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.sz-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.sz-b{transform-box:fill-box;transform-origin:bottom;animation-duration:20s;animation-timing-function:ease-in-out;animation-iteration-count:infinite}
@keyframes sz-k1{0%,18%{transform:scaleY(.400)}25%,43%{transform:scaleY(.604)}50%,68%{transform:scaleY(.667)}75%,93%{transform:scaleY(.348)}100%{transform:scaleY(.400)}}
@keyframes sz-k2{0%,18%{transform:scaleY(.271)}25%,43%{transform:scaleY(.317)}50%,68%{transform:scaleY(.350)}75%,93%{transform:scaleY(.496)}100%{transform:scaleY(.271)}}
@keyframes sz-k3{0%,18%{transform:scaleY(.200)}25%,43%{transform:scaleY(.190)}50%,68%{transform:scaleY(.210)}75%,93%{transform:scaleY(.298)}100%{transform:scaleY(.200)}}
@keyframes sz-k4{0%,18%{transform:scaleY(.143)}25%,43%{transform:scaleY(.109)}50%,68%{transform:scaleY(.120)}75%,93%{transform:scaleY(.170)}100%{transform:scaleY(.143)}}
@keyframes sz-k5{0%,18%{transform:scaleY(.114)}25%,43%{transform:scaleY(.074)}50%,68%{transform:scaleY(.081)}75%,93%{transform:scaleY(.117)}100%{transform:scaleY(.114)}}
@keyframes sz-k6{0%,18%{transform:scaleY(.086)}25%,43%{transform:scaleY(.046)}50%,93%{transform:scaleY(.012)}100%{transform:scaleY(.086)}}
@keyframes sz-k7{0%,18%{transform:scaleY(.071)}25%,43%{transform:scaleY(.034)}50%,93%{transform:scaleY(.012)}100%{transform:scaleY(.071)}}
@keyframes sz-k8{0%,18%{transform:scaleY(.057)}25%,43%{transform:scaleY(.024)}50%,93%{transform:scaleY(.012)}100%{transform:scaleY(.057)}}
@keyframes sz-k9{0%,18%{transform:scaleY(.050)}25%,43%{transform:scaleY(.019)}50%,93%{transform:scaleY(.012)}100%{transform:scaleY(.050)}}
@keyframes sz-k10{0%,18%{transform:scaleY(.036)}25%,43%{transform:scaleY(.011)}50%,93%{transform:scaleY(.012)}100%{transform:scaleY(.036)}}
@keyframes sz-dim{0%,45%{opacity:1}50%,93%{opacity:.22}100%{opacity:1}}
@keyframes sz-l1{0%,20%{opacity:1}24%,96%{opacity:0}100%{opacity:1}}
@keyframes sz-l2{0%,21%{opacity:0}25%,45%{opacity:1}49%,100%{opacity:0}}
@keyframes sz-l3{0%,46%{opacity:0}50%,70%{opacity:1}74%,100%{opacity:0}}
@keyframes sz-l4{0%,71%{opacity:0}75%,95%{opacity:1}99%,100%{opacity:0}}
.sz-b1{animation-name:sz-k1}.sz-b2{animation-name:sz-k2}.sz-b3{animation-name:sz-k3}.sz-b4{animation-name:sz-k4}.sz-b5{animation-name:sz-k5}
.sz-b6{animation-name:sz-k6}.sz-b7{animation-name:sz-k7}.sz-b8{animation-name:sz-k8}.sz-b9{animation-name:sz-k9}.sz-b10{animation-name:sz-k10}
.sz-fade{animation:sz-dim 20s ease-in-out infinite}
.sz-t1{animation:sz-l1 20s ease-in-out infinite}
.sz-t2{animation:sz-l2 20s ease-in-out infinite}
.sz-t3{animation:sz-l3 20s ease-in-out infinite}
.sz-t4{animation:sz-l4 20s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sz-b,.sz-fade,.sz-t1,.sz-t2,.sz-t3,.sz-t4{animation:none}.sz-t2,.sz-t3,.sz-t4{opacity:0}.sz-t1{opacity:1}}
</style>
<text class="sz-cap sz-t1" x="350" y="32">T = 1.0 · every token reachable</text>
<text class="sz-cap sz-t2" x="350" y="32">T = 0.6 · mass pulled to the leader</text>
<text class="sz-cap sz-t3" x="350" y="32">top-p 0.9 · tail cut to five tokens</text>
<text class="sz-cap sz-t4" x="350" y="32">freq penalty 1.0 · the leader loses</text>
<line class="sz-base" x1="56" y1="210" x2="644" y2="210"/>
<rect class="sz-bar sz-b sz-b1" x="66" y="60" width="44" height="150"/>
<rect class="sz-bar sz-b sz-b2" x="122" y="60" width="44" height="150"/>
<rect class="sz-bar sz-b sz-b3" x="178" y="60" width="44" height="150"/>
<rect class="sz-bar sz-b sz-b4" x="234" y="60" width="44" height="150"/>
<rect class="sz-bar sz-b sz-b5" x="290" y="60" width="44" height="150"/>
<rect class="sz-tail sz-fade sz-b sz-b6" x="346" y="60" width="44" height="150"/>
<rect class="sz-tail sz-fade sz-b sz-b7" x="402" y="60" width="44" height="150"/>
<rect class="sz-tail sz-fade sz-b sz-b8" x="458" y="60" width="44" height="150"/>
<rect class="sz-tail sz-fade sz-b sz-b9" x="514" y="60" width="44" height="150"/>
<rect class="sz-tail sz-fade sz-b sz-b10" x="570" y="60" width="44" height="150"/>
<text class="sz-tok" x="88" y="228">mat</text>
<text class="sz-tok" x="144" y="228">floor</text>
<text class="sz-tok" x="200" y="228">couch</text>
<text class="sz-tok" x="256" y="228">roof</text>
<text class="sz-tok" x="312" y="228">bed</text>
<text class="sz-tok" x="368" y="228">desk</text>
<text class="sz-tok" x="424" y="228">sofa</text>
<text class="sz-tok" x="480" y="228">table</text>
<text class="sz-tok" x="536" y="228">rug</text>
<text class="sz-tok" x="592" y="228">step</text>
<text class="sz-tok" x="350" y="256">the same logits, four sampler policies</text>
</svg>
<figcaption>One logit row under four policies: temperature sharpens the head, top-p deletes the tail outright, and a frequency penalty on an already-used token hands the argmax to the runner-up.</figcaption>
</figure>

---

## 5. The truncation family

Truncation is the pass that decides which tokens are allowed to be chosen at all. Every method in this family produces the same kind of output — a set of surviving tokens — and differs only in how it draws the line.

![Four truncation settings compared on a six token toy distribution and on a full vocabulary tail at temperature two](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-4.webp)

### top-k: a fixed count

Keep the $k$ highest logits, mask the rest. Introduced for open-ended generation by Fan et al. in [*Hierarchical Neural Story Generation*](https://arxiv.org/abs/1805.04833) (ACL 2018).

$$\text{keep}(i) \iff \text{rank}(z_i) \lt k$$

It is the cheapest method — a `topk` reduction, no sort of the full row required — and it is the least adaptive. The count is fixed regardless of what the model believes, which produces two failure modes that are exact mirrors of each other:

- **Over-truncation on flat distributions.** When the model genuinely does not know (a name, a list continuation, an open-ended creative choice), fifty tokens might each hold 1.5% of the mass. `top_k=5` throws away 92% of a legitimately uncertain distribution and forces the model into a confidence it does not have.
- **Under-truncation on peaked distributions.** When the model is certain (the closing brace of a JSON object, the second half of a word), the top token might hold 0.99 and tokens 2 through 20 might hold $10^{-4}$ each. `top_k=20` keeps nineteen tokens that are, collectively, noise — and any of them can be sampled.

On the six-token row from section 2, `top_k=20` keeps all six tokens including `bed` at 0.0088. It has done nothing at all.

top-k is not useless. It is an excellent *safety rail* — `top_k=100` on a 128,256-token vocabulary bounds the worst case without meaningfully constraining anything the model actually believes, and it makes the sampler cheaper because you can `topk` before you sort. Use it as a floor, not as your primary control.

### top-p / nucleus: a fixed mass

Sort descending, take the smallest prefix whose cumulative probability reaches $p$. From Holtzman et al.'s degeneration paper.

$$\text{keep}(i) \iff \sum_{j : \text{rank}(j) \lt \text{rank}(i)} p_j \lt p$$

This is the adaptive answer to top-k's problem. When the model is confident, the nucleus is one or two tokens. When it is uncertain, the nucleus expands. On the section 2 row, `top_p=0.9` keeps three tokens; on a confident row it would keep one. That adaptivity is why nucleus sampling became the default everywhere.

Its failure mode is the long flat tail, and it is worth doing the arithmetic because the effect is far larger than intuition suggests.

#### Worked example: what temperature does to a real nucleus

A six-token toy hides the problem, so model a real vocabulary. Take 128,004 tokens arranged as a head of four and a tail of 128,000, expressed as logit gaps below the maximum:

- head gaps: 0, 1.6, 2.0, 2.5
- tail: 128,000 tokens all at gap 14.0

A gap of 14 nats is a token the model assigns roughly a millionth the weight of its favourite — genuinely implausible continuations, which is what most of a 128k vocabulary is at any given position.

**At $T = 1$**, the unnormalized weights are $e^{-g}$: head $[1, 0.2019, 0.1353, 0.0821]$ and each tail token $e^{-14} = 8.315 \times 10^{-7}$. The tail contributes $128{,}000 \times 8.315 \times 10^{-7} = 0.1064$. Partition function $Z = 1 + 0.2019 + 0.1353 + 0.0821 + 0.1064 = 1.5258$.

| token group | probability | Source |
| --- | --- | --- |
| head 1 | 0.6554 | derived |
| head 2 | 0.1323 | derived |
| head 3 | 0.0887 | derived |
| head 4 | 0.0538 | derived |
| tail, all 128,000 | 0.0698 total, $5.45 \times 10^{-7}$ each | derived |

Cumulative: 0.6554, 0.7877, 0.8764, **0.9302**. `top_p=0.9` keeps **four** tokens. Perfectly sensible.

**Now set $T = 2$** and recompute. Weights become $e^{-g/2}$: head $[1, 0.4493, 0.3679, 0.2865]$, and each tail token $e^{-7} = 9.119 \times 10^{-4}$. The tail now contributes $128{,}000 \times 9.119 \times 10^{-4} = 116.72$. Read that again: the tail's total weight has gone from 0.106 to 116.72, because halving every gap in the exponent multiplies a $e^{-14}$ token by $e^{7} = 1097$.

$Z = 1 + 0.4493 + 0.3679 + 0.2865 + 116.72 = 118.82$, so:

| token group | probability at $T = 2$ | Source |
| --- | --- | --- |
| head 1 | 0.00842 | derived |
| head 2 | 0.00378 | derived |
| head 3 | 0.00310 | derived |
| head 4 | 0.00241 | derived |
| tail, all 128,000 | 0.9823 total, $7.674 \times 10^{-6}$ each | derived |

The four tokens the model actually believes in hold **1.8% of the mass between them**. `top_p=0.9` now has to sweep ${0.9 - 0.0177 = 0.8823}$ of probability out of a tail whose members are worth $7.674 \times 10^{-6}$ apiece, which takes $0.8823 / 7.674\times10^{-6} \approx 115{,}000$ tokens. Your "0.9 nucleus" is 90% of the vocabulary.

This is the mechanism behind "temperature 2 makes the model produce garbage" and it is not a mystery, it is a sum over 128,000 small numbers. Source: `derived` throughout; the model of the tail (128,000 tokens at a uniform gap of 14) is a stylized assumption, stated so you can substitute your own.

### min-p: a fixed ratio to the leader

Keep every token whose probability is at least a fraction of the maximum probability:

$$\text{keep}(i) \iff p_i \ge p_{\text{base}} \cdot p_{\max}$$

Popularized by community samplers and written up by Nguyen et al. as [*Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*](https://arxiv.org/abs/2407.01082). Run the same two cases through it and the contrast is stark.

At $T = 1$: threshold $= 0.05 \times 0.6554 = 0.0328$. Head tokens at 0.1323, 0.0887 and 0.0538 all clear it; tail tokens at $5.45 \times 10^{-7}$ do not. **Four tokens.**

At $T = 2$: threshold $= 0.05 \times 0.00842 = 4.21 \times 10^{-4}$. Head tokens at 0.00378, 0.00310 and 0.00241 all clear it; tail tokens at $7.674 \times 10^{-6}$ do not. **Four tokens.** Identical answer at double the temperature, while top-p went from 4 to 115,000.

Why? Because the min-p condition, written in logit space, is a **fixed gap**. Substitute $p_i / p_{\max} = e^{(z_i - z_{\max})/T}$ into the definition:

$$e^{(z_i - z_{\max})/T} \ge p_{\text{base}} \iff z_{\max} - z_i \le T \ln\!\left(\frac{1}{p_{\text{base}}}\right)$$

With `min_p = 0.05` and $T = 1$ the threshold is $\ln 20 = 2.996$ nats; at $T = 2$ it is 5.991. The head gaps are 0, 1.6, 2.0 and 2.5, all below both thresholds. The tail gap is 14.0, above both. The answer is four tokens at both temperatures *because the gaps did not move* — only the threshold did, and not far enough to reach 14.

That closed form is the whole argument for min-p, and it also tells you its limit. min-p is a **relative** criterion, so it is only as good as the model's confidence signal. If the model is genuinely uncertain — every token within 1 nat of the leader — min-p keeps essentially everything. On the section 2 toy row it keeps five of six tokens (threshold $0.05 \times 0.4806 = 0.024$; only `bed` at 0.0088 falls below). min-p is not a *narrowing* device; it is a *garbage filter* that happens to be temperature-robust. Pair it with a modest top-k if you also want a bound.

### typical-p: selection by information content

Locally typical sampling, from Meister et al., [*Locally Typical Sampling*](https://arxiv.org/abs/2202.00666) (TACL 2023), asks a different question. Instead of ranking tokens by probability, it ranks them by how close their information content $-\ln p_i$ is to the distribution's entropy $H$:

$$\text{score}(i) = \big|\,{-\ln p_i} - H\,\big|, \qquad H = -\sum_j p_j \ln p_j$$

Sort ascending by that score, then take the smallest prefix whose cumulative *probability* reaches $\tau$. The motivation is information-theoretic: human language tends to convey information at a roughly steady rate, so tokens that are wildly more predictable than average are as atypical as tokens that are wildly less predictable.

#### Worked example: typical-p can drop the argmax

Take a distribution $p = [0.50, 0.20, 0.15, 0.10, 0.04, 0.01]$. Its entropy is

$$H = 0.3466 + 0.3219 + 0.2846 + 0.2303 + 0.1288 + 0.0461 = 1.3581 \text{ nats}$$

Information contents $-\ln p_i$: 0.693, 1.609, 1.897, 2.303, 3.219, 4.605. Deviations from $H$:

| token | $p$ | $-\ln p$ | deviation from $H$ | rank |
| --- | --- | --- | --- | --- |
| t2 | 0.20 | 1.609 | 0.251 | 1 |
| t3 | 0.15 | 1.897 | 0.539 | 2 |
| t1 | 0.50 | 0.693 | 0.665 | 3 |
| t4 | 0.10 | 2.303 | 0.944 | 4 |
| t5 | 0.04 | 3.219 | 1.861 | 5 |
| t6 | 0.01 | 4.605 | 3.247 | 6 |

At $\tau = 0.3$: cumulative probability over the ranked order is 0.20, then **0.35**, which clears 0.3. The surviving set is $\{t2, t3\}$ — and the most probable token is **not in it**. Renormalized, $t2$ gets 0.571 and $t3$ gets 0.429. Compare `top_p=0.3` on the same row, which keeps only $t1$. Two settings that look equally aggressive produce disjoint candidate sets. Source: `derived`.

At the more usual $\tau = 0.9$: cumulative 0.20, 0.35, 0.85, **0.95** — four tokens, which is close to what top-p 0.9 would give (0.50, 0.70, 0.85, **0.95**, also four). At loose settings typical-p and top-p mostly agree; the divergence shows up when you tighten, and specifically when the model is *over*confident relative to the surrounding entropy.

The practical cost is that typical-p needs its own sort, ordered by deviation rather than by probability, so it cannot share the descending sort that top-k, top-p and min-p all use. That is why engines make it opt-in and why `nanoserve` implements it as a separate processor.

### Summary of the family

| Method | Criterion | Adapts to confidence | Adapts to temperature | Main failure |
| --- | --- | --- | --- | --- |
| top-k | fixed count | no | no | wrong count on both flat and peaked rows |
| top-p | fixed cumulative mass | yes | no | long tail floods in at high $T$ |
| min-p | fixed ratio to $p_{\max}$ | yes | yes | keeps nearly everything on flat rows |
| typical-p | closeness to entropy | yes | partly | needs a second sort; can drop the mode |

---

## 6. Penalties, and how they break factual output

The penalty family is where sampling stops being a pure function of the logits and starts depending on generation history. It is also where most production quality incidents originate, so it deserves the exact formulas rather than folklore.

![A frequency penalty accumulating over a decode window until a required token becomes unemittable](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-5.webp)

### The one law that makes all penalties legible

Every additive logit penalty of size $\delta$ has the same effect, and it is simple. Softmax is a ratio, so subtracting $\delta$ from token $i$'s logit multiplies its odds against **every** unpenalized token $j$ by $e^{-\delta}$:

$$\frac{p_i'}{p_j'} = \frac{e^{z_i - \delta}}{e^{z_j}} = e^{-\delta} \cdot \frac{p_i}{p_j}$$

That gives a conversion table you should memorize, because it turns every penalty setting into a number you can reason about:

| logit penalty $\delta$ | odds multiplier $e^{-\delta}$ | plain reading | Source |
| --- | --- | --- | --- |
| 0.1 | 0.905 | 10% less likely | derived |
| 0.5 | 0.607 | 39% less likely | derived |
| 0.693 | 0.500 | half as likely | derived |
| 1.0 | 0.368 | a third as likely | derived |
| 2.0 | 0.135 | 7× less likely | derived |
| 2.303 | 0.100 | 10× less likely | derived |
| 4.6 | 0.010 | 100× less likely | derived |
| 12.0 | $6.1 \times 10^{-6}$ | effectively banned | derived |

### Presence penalty

$$z_i' = z_i - \alpha_{\text{pres}} \cdot \mathbb{1}[\text{count}_i > 0]$$

A flat subtraction applied once to any token the request has already produced, regardless of how many times. It is bounded — the worst case is $\alpha_{\text{pres}}$ — which makes it the safest member of the family. `presence_penalty=0.5` means "every token you have already used is 39% less likely, permanently, and that is the end of it".

### Frequency penalty

$$z_i' = z_i - \alpha_{\text{freq}} \cdot \text{count}_i$$

Linear in the running count, and therefore **unbounded**. This is the knob that kills structured output, and figure 5 above is the arithmetic. With `frequency_penalty=0.3`, a token seen 4 times loses 1.2 logits (odds ×0.30) — reasonable. Seen 12 times, it loses 3.6 (odds ×0.027) — aggressive. Seen 40 times, it loses 12.0, and its odds against any fresh token have been cut by a factor of 162,754.

Now consider what "seen 40 times" means. In a JSON response with twenty fields, the `"` token appears at least eighty times. In Python, the newline-plus-indent token appears once per line. In a table, the pipe character appears once per cell. In a medical summary, `hemoglobin` appears in every sentence about hemoglobin. **Frequency penalty does not know the difference between a stylistic tic and a syntactic requirement.** It penalizes both, linearly, forever.

The failure is not subtle when it arrives. The model reaches a point where it *must* emit a closing quote, the closing quote carries twelve logits of accumulated penalty, and the second-best token — a letter, a space, anything — wins. You get a JSON string that never terminates, and a client-side parse error, and a bug report that says "the model is broken", when what happened is that a decoding parameter overrode the grammar of the format.

Rules I would enforce in any serving layer:

- **Frequency penalty must be zero for structured output.** Not small — zero. If you are emitting JSON, code, CSV, XML, or anything with required repeated tokens, set it to zero and reach for constrained decoding instead (the subject of the next few posts in this track).
- **Frequency penalty should have a window.** Applying it over the entire generated sequence guarantees the count grows without bound. Applying it over the last, say, 256 tokens bounds the count by the window and makes the knob behave like a repetition breaker instead of a censor. Some engines support this; most do not expose it. `nanoserve` will.
- **Prefer presence penalty when you want gentle variety.** It is bounded by construction.

### Repetition penalty

This one is different in kind, and its shape is worth understanding because it explains a class of confusing bug reports. From Keskar et al.'s [CTRL](https://arxiv.org/abs/1909.05858) paper, it is *multiplicative* on the logits and *sign-dependent*:

$$z_i' = \begin{cases} z_i / r & \text{if } z_i > 0 \\ z_i \cdot r & \text{if } z_i \le 0 \end{cases} \qquad (r > 1)$$

Three problems follow directly from that definition.

**It is not shift-invariant.** Softmax does not care about an additive constant on the logits, but this penalty does. Add $c$ to every logit and a penalized token goes to $(z_i + c)/r$ while unpenalized tokens go to $z_j + c$; the effective gap becomes $(z_i + c)(1/r - 1)$, which depends on $c$. Since nothing pins the absolute scale of a model's logits — it varies by architecture, by training run, and by position within a sequence — **the same `repetition_penalty` value has a different strength on different models and at different points in the same generation.** That is why the "good" value for one model is 1.05 and for another is 1.25, and why nobody can explain the difference.

**It has a discontinuity at zero.** A token with logit $+0.01$ and $r = 1.1$ becomes ${0.0091}$, a penalty of 0.0009 logits — nothing. A token with logit $-0.01$ becomes $-0.011$, a penalty of 0.001 in the other direction — also nothing, but with the opposite sign relationship. Meanwhile a token at $+8.0$ loses 0.73 logits and a token at $-8.0$ loses 0.8. The penalty's strength is proportional to the magnitude of the logit, which is not a property anyone wants.

**It is the strongest of the three.** At $r = 1.2$ a token sitting at logit 10 drops to 8.33, a 1.67-logit penalty, odds ×0.19. People set 1.2 casually because the number looks close to 1.

My recommendation: use presence and frequency penalties, which are additive, bounded (presence) or windowable (frequency), and shift-invariant. Use repetition penalty only when you are reproducing someone else's published configuration.

### DRY: penalizing sequences instead of tokens

All three penalties above share a fundamental flaw: they penalize *token identity*, but the thing you actually want to prevent is *verbatim repetition of a sequence*. Penalizing `the` because the model used `the` is collateral damage; the model needs `the`.

The DRY sampler — "Don't Repeat Yourself", contributed to llama.cpp and exposed there as `--dry-multiplier`, `--dry-base` and `--dry-allowed-length` — attacks the right target. At each step it finds the longest suffix of the generated text that has occurred earlier in the sequence. For every earlier occurrence, the token that *followed* that occurrence gets a penalty that grows with the match length:

$$\delta = \text{multiplier} \times \text{base}^{(\ell - \ell_{\text{allowed}})}$$

where $\ell$ is the length of the matched suffix. With the commonly used defaults `multiplier = 0.8`, `base = 1.75`, `allowed_length = 2`, a 5-token repeated sequence produces $0.8 \times 1.75^{3} = 0.8 \times 5.359 = 4.29$ logits of penalty — odds ×0.0137, a 98.6% cut — applied to exactly the one token that would continue the loop. A 3-token match produces $0.8 \times 1.75 = 1.4$ logits, a gentle nudge. A 2-token match produces nothing at all. Source: `derived` from the formula above.

That is the correct shape: exponential in how egregious the repetition is, and targeted at a single continuation rather than at a whole vocabulary entry. Its cost is that it needs suffix matching against the generated sequence every step, which is real CPU or GPU work that the simple penalties do not require. For chat and creative writing where loops are the dominant failure, it is worth it. For structured output it is unnecessary, because the grammar already prevents loops.

---

## 7. Building it: nanoserve/sampling.py

Now the code. The design goal is that every knob is one small class with one method, that the composition order is stated in one place, and that the whole thing works batched without a Python loop over requests.

### The protocol and the parameters

```python
# nanoserve/sampling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import torch


@dataclass
class SamplingContext:
    """Per-request state a processor may need.

    output_ids : [B, L] int64, right-padded generated tokens
    output_len : [B]    int64, valid length of each row
    step       : int,   index of this decode step (for stateless RNG)
    """
    output_ids: torch.Tensor
    output_len: torch.Tensor
    step: int


class LogitsProcessor(Protocol):
    """Transform a [B, V] logits tensor in place or functionally.

    Contract: return a tensor of the same shape and dtype. Masked-out
    tokens are set to -inf, never to a large negative finite number --
    -inf survives every subsequent arithmetic step and softmaxes to
    exactly zero, whereas -1e9 does not after a division by T = 0.1.
    """
    def __call__(self, logits: torch.Tensor, ctx: SamplingContext) -> torch.Tensor: ...


@dataclass
class SamplingParams:
    """Decoding parameters for one request.

    APPLICATION ORDER (fixed, documented, and tested):
        1. penalties      (repetition, presence, frequency)
        2. temperature    (z / T; T == 0 means greedy)
        3. truncation     (top_k AND top_p AND min_p, intersected)
        4. typical_p      (separate pass, own ordering)
        5. renormalize + sample

    Truncators INTERSECT rather than chain: each computes a keep-mask
    against the single post-temperature distribution, so top_k and top_p
    commute and "top_k=50, top_p=0.9" means exactly what it reads as.
    """
    temperature: float = 1.0        # 0.0 == greedy
    top_k: int = 0                  # 0 == disabled
    top_p: float = 1.0              # 1.0 == disabled
    min_p: float = 0.0              # 0.0 == disabled
    typical_p: float = 1.0          # 1.0 == disabled
    repetition_penalty: float = 1.0 # 1.0 == disabled
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_window: int = 0         # 0 == whole sequence
    seed: int | None = None

    def __post_init__(self):
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
```

The `-inf`-not-`-1e9` note in the docstring is a real bug I want to head off. Masking with a large finite negative number is common because it avoids `nan` in some downstream ops, but it interacts badly with temperature: if the mask runs before the division and you divide $-10^9$ by $T = 0.1$ you get $-10^{10}$, which is fine, but if you divide by $T = 10$ you get $-10^8$, which after softmax against a logit of, say, 5, is still exactly zero in float32 — until someone changes the order and the masked token is only 20 logits down instead of $10^9$. Use `-inf`. It is exactly zero after softmax at every temperature and it propagates correctly through addition, which the Gumbel sampler in section 9 depends on.

### The individual processors

```python
# nanoserve/sampling.py (continued)

class Temperature:
    """Scale logits by 1/T. T == 0 is handled by the caller (greedy mode)."""

    def __init__(self, temperature: torch.Tensor):
        # temperature: [B] float32. Zeros are replaced with 1.0 here and
        # the corresponding rows are overwritten with argmax after sampling.
        self.t = torch.where(temperature > 0,
                             temperature,
                             torch.ones_like(temperature))

    def __call__(self, logits, ctx):
        return logits / self.t.unsqueeze(1)


class Truncate:
    """top_k AND top_p AND min_p in a single descending sort.

    All three masks are computed against the SAME post-temperature
    distribution and intersected, so the three settings commute.
    """

    def __init__(self, top_k: torch.Tensor, top_p: torch.Tensor,
                 min_p: torch.Tensor):
        self.top_k, self.top_p, self.min_p = top_k, top_p, min_p

    def __call__(self, logits, ctx):
        B, V = logits.shape
        s, idx = torch.sort(logits, dim=-1, descending=True)   # [B, V]
        probs = s.softmax(dim=-1)

        # top_k: everything at rank >= k is removed. k == 0 means V.
        rank = torch.arange(V, device=logits.device).unsqueeze(0)
        k = torch.where(self.top_k > 0, self.top_k,
                        torch.full_like(self.top_k, V))
        kill = rank >= k.unsqueeze(1)

        # top_p: remove token j if the mass BEFORE it already reached p.
        cum = probs.cumsum(dim=-1)
        kill |= (cum - probs) >= self.top_p.unsqueeze(1)

        # min_p: remove token j if p_j < min_p * p_max. After a descending
        # sort, p_max is column 0.
        kill |= probs < self.min_p.unsqueeze(1) * probs[:, :1]

        kill[:, 0] = False               # never mask the argmax
        s = s.masked_fill(kill, float("-inf"))

        out = torch.empty_like(logits)
        out.scatter_(1, idx, s)
        return out
```

The `kill[:, 0] = False` line is not cosmetic. Without it, `top_p=0.0` or `min_p=1.0` masks every token, `softmax` over an all-`-inf` row produces `nan`, and `multinomial` raises. Guaranteeing at least one survivor is the invariant that keeps a serving loop alive when a client sends a nonsensical parameter, and it costs one store.

The `(cum - probs) >= top_p` formulation deserves a note too. `cum[j]` is the mass up to and including token $j$, so `cum[j] - probs[j]` is the mass strictly before it. Keeping token $j$ exactly when that quantity is below $p$ implements "the smallest prefix whose cumulative probability reaches $p$" precisely, including the boundary case where a single token already holds $p$ or more. Implementations that instead test `cum > top_p` and then shift the mask by one index are trying to express the same thing and get the boundary wrong about half the time.

### typical-p, as its own pass

```python
# nanoserve/sampling.py (continued)

class TypicalP:
    """Locally typical sampling (Meister et al., 2023).

    Ranks tokens by |-log p - H| instead of by p, then keeps the smallest
    prefix of that ranking whose cumulative probability reaches tau.
    Needs its own sort, which is why it is a separate pass.
    """

    def __init__(self, typical_p: torch.Tensor):
        self.tau = typical_p

    def __call__(self, logits, ctx):
        logp = torch.log_softmax(logits, dim=-1)          # [B, V]
        p = logp.exp()
        ent = -(p * logp).sum(dim=-1, keepdim=True)       # [B, 1] nats
        dev = (-logp - ent).abs()                         # [B, V]

        sorted_dev, idx = dev.sort(dim=-1)                # ascending
        sorted_p = p.gather(1, idx)
        cum = sorted_p.cumsum(dim=-1)
        kill_sorted = (cum - sorted_p) >= self.tau.unsqueeze(1)
        kill_sorted[:, 0] = False

        kill = torch.zeros_like(kill_sorted)
        kill.scatter_(1, idx, kill_sorted)
        return logits.masked_fill(kill, float("-inf"))
```

Note `-(p * logp).sum(...)` rather than `-(p * p.log()).sum(...)`. `log_softmax` is numerically stable by construction; taking `exp` and then `log` again reintroduces the overflow you just avoided, and on a row containing `-inf` entries `0 * -inf` is `nan`. Using `logp` directly, `exp(-inf) = 0` and `0 * -inf` never happens because we multiply the already-exponentiated `p` by the *original* `logp`, whose `-inf` entries are multiplied by exactly `0.0`... which is still `nan`. So in practice you also need `torch.where(p > 0, p * logp, 0.0)` if you ever run typical-p after a truncator. `nanoserve` runs it before, which is why the code above is safe as written — and this is exactly the kind of ordering constraint that belongs in a comment, not in tribal memory.

### Penalties, batched

```python
# nanoserve/sampling.py (continued)

class Penalties:
    """Repetition (CTRL, multiplicative), presence and frequency (additive).

    The [B, V] count buffer is allocated once at engine start and zeroed
    per step. At B=256 and V=128256 in fp32 that is 131.3 MB of resident
    VRAM -- a real line in your memory budget, not a rounding error.
    """

    def __init__(self, counts_buf: torch.Tensor, repetition: torch.Tensor,
                 presence: torch.Tensor, frequency: torch.Tensor):
        self.counts = counts_buf          # [B, V] float32, persistent
        self.rep = repetition.unsqueeze(1)
        self.pres = presence.unsqueeze(1)
        self.freq = frequency.unsqueeze(1)

    def __call__(self, logits, ctx):
        B, V = logits.shape
        counts = self.counts[:B]
        counts.zero_()
        # ctx.output_ids is right-padded with 0; the mask zeroes the pad
        # contribution so token id 0 is not spuriously penalized.
        valid = (torch.arange(ctx.output_ids.shape[1],
                              device=logits.device).unsqueeze(0)
                 < ctx.output_len.unsqueeze(1)).float()
        counts.scatter_add_(1, ctx.output_ids, valid)

        seen = counts > 0

        # Repetition penalty: divide positive logits, multiply negative.
        penalized = torch.where(logits > 0, logits / self.rep,
                                logits * self.rep)
        logits = torch.where(seen, penalized, logits)

        # Presence: flat, bounded. Frequency: linear in count, unbounded.
        logits = logits - self.pres * seen.to(logits.dtype)
        logits = logits - self.freq * counts
        return logits
```

Two engineering points hide in there.

**The pad mask matters.** `output_ids` is right-padded, and the natural pad value is 0, which is a real token id in most vocabularies (`<unk>` or a byte token). Without the `valid` mask, every request in the batch accumulates a phantom count for token 0 proportional to how much shorter it is than the longest row, and if `frequency_penalty` is on you have just banned a token for reasons entirely unrelated to what the model generated. This is a genuinely hard bug to find because it only manifests on heterogeneous batches.

**The counts buffer is expensive, and the obvious optimization is a trap.** You may be tempted to skip the `[B, V]` buffer and instead `gather` the logits at the positions in `output_ids`, penalize those, and `scatter_` them back — that touches $B \times L$ elements instead of $B \times V$. It works for presence penalty. It is *wrong* for frequency penalty, because `output_ids` contains duplicates, and `scatter_` with duplicate indices has an unspecified winner. You would need `scatter_add_` on a delta, at which point you are back to needing the counts anyway. Allocate the buffer once, zero it per step, and put the 131 MB in your VRAM budget where the scheduler can see it.

---

## 8. One batched step, many policies

Here is the complication that toy sampler code never shows and that every engine has to solve. In a continuous batch — the running set assembled by the scheduler, on top of the loop from [the naive decode loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) — every row of the logits tensor belongs to a *different request*, with *different sampling parameters*. Request A is an extraction job at temperature 0. Request B is a chat turn at 0.8 with top-p 0.9. Request C is creative writing at 1.2 with min-p 0.05 and a frequency penalty.

![Three requests with three sampling policies sharing one sort and one mask pass in a single batched step](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-6.webp)

The naive implementation loops over rows in Python and calls the single-request sampler on each. At batch 64 with five active knobs that is 320 kernel launches per decode step, each on a `[1, 128256]` tensor — one of the fastest ways to make your engine CPU-bound that I know of. The fix is uniform: **every scalar parameter becomes a `[B]` tensor**, and every processor is written to broadcast.

```python
# nanoserve/sampling.py (continued)

class BatchedSampler:
    """Owns the per-step sampler for a running set of B requests."""

    def __init__(self, vocab_size: int, max_batch: int, device: str):
        self.V = vocab_size
        self.device = device
        # Allocated once. 256 x 128256 x 4 B = 131.3 MB at max_batch=256.
        self.counts = torch.zeros((max_batch, vocab_size),
                                  dtype=torch.float32, device=device)

    def build_params(self, params: Sequence[SamplingParams]):
        """Turn a list of per-request dataclasses into [B] tensors.

        Called only when the running set CHANGES, not every step. In a
        continuous-batching engine the set changes on admission and on
        completion, which is far less often than once per token.
        """
        f = lambda vals: torch.tensor(vals, dtype=torch.float32,
                                      device=self.device)
        i = lambda vals: torch.tensor(vals, dtype=torch.int64,
                                      device=self.device)
        return {
            "temperature": f([p.temperature for p in params]),
            "top_k":       i([p.top_k for p in params]),
            "top_p":       f([p.top_p for p in params]),
            "min_p":       f([p.min_p for p in params]),
            "typical_p":   f([p.typical_p for p in params]),
            "repetition":  f([p.repetition_penalty for p in params]),
            "presence":    f([p.presence_penalty for p in params]),
            "frequency":   f([p.frequency_penalty for p in params]),
            "greedy":      f([1.0 if p.temperature == 0 else 0.0
                              for p in params]).bool(),
        }
```

The `build_params` comment is the important one. Rebuilding nine small tensors costs nine host-to-device copies, and if you do it every step you have added nine synchronization points to a loop that is trying not to have any. Build them when the running set changes — on admission and on completion — and keep them alive across steps. In an engine with a persistent batch this falls out naturally: the vLLM team's [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) post (2026-03-24) describes a decoupled persistent batch where each request holds a fixed row in a state table independent of ordering, with a per-step gather to build the ordered views the kernels need. Same idea, done properly.

### Putting the chain together

```python
# nanoserve/sampling.py (continued)

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, ctx: SamplingContext,
                 pt: dict) -> torch.Tensor:
        """logits: [B, V] raw model output for this step (last position).

        Returns [B] int64 token ids, still on device. Nothing in this
        function reads a tensor back to the host.
        """
        logits = logits.float()                    # fp32 for the softmax

        # 1. penalties
        if (pt["repetition"] != 1.0).any() or (pt["presence"] != 0).any() \
                or (pt["frequency"] != 0).any():
            logits = Penalties(self.counts, pt["repetition"],
                               pt["presence"], pt["frequency"])(logits, ctx)

        # 2. temperature (greedy rows temporarily get T = 1)
        logits = Temperature(pt["temperature"])(logits, ctx)

        # 3. typical-p, before the intersecting truncators (see docstring)
        if (pt["typical_p"] < 1.0).any():
            logits = TypicalP(pt["typical_p"])(logits, ctx)

        # 4. top_k AND top_p AND min_p, one sort
        logits = Truncate(pt["top_k"], pt["top_p"], pt["min_p"])(logits, ctx)

        # 5. sample without materializing a softmax, then fix greedy rows
        sampled = gumbel_argmax(logits, ctx.step, pt)
        greedy = logits.argmax(dim=-1)
        return torch.where(pt["greedy"], greedy, sampled)
```

Every `if` in that function is a *host-side* branch on a `[B]` tensor, which means each one is itself a host sync. In `nanoserve` I accept that for clarity, because the branches let you skip an entire expensive pass when no request in the batch uses it — and the counts-buffer zeroing plus scatter over `[B, V]` is genuinely expensive. In a production engine you would hoist those predicates into the same place `build_params` runs, evaluate them once per running-set change, and cache them as Python booleans. That turns three per-step syncs into zero.

Run the chain on the section 2 row with three different parameter sets and it prints the arithmetic you can check by hand:

```console
$ python -m nanoserve.demo_sampler
logits: [3.2, 2.7, 1.9, 1.1, 0.4, -0.8]   tokens: mat floor couch roof table bed

row 0  T=0.0                    -> candidates 1  [mat 1.0000]
row 1  T=1.5 top_p=0.9          -> candidates 4  [mat .4195  floor .3006  couch .1764  roof .1035]
row 2  T=2.0 min_p=0.05         -> candidates 5  [mat .3453  floor .2690  couch .1803  roof .1209  table .0851]
row 3  T=1.0 typical_p=0.9      -> candidates 3  [floor .3228  mat .5322  couch .1450]
```

Row 2 is worth a second look: at $T = 2$, min-p 0.05 gives a threshold of $0.05 \times 0.3297 = 0.0165$, so `bed` at 0.0446 should survive — but the six-token toy has no real tail, so min-p keeps five and drops nothing but the floor. Renormalizing the surviving five over ${1 - 0.0446 = 0.9554}$ gives the numbers shown. On a real 128k vocabulary, as section 5 derived, the same setting drops 128,000 tokens. Source: `derived`.

---

## 9. Keep the sampler on the GPU

Now the systems half of the post. Everything so far has been about *what* distribution to sample from. This section is about *where* the sampling happens, and it is worth as much throughput as any of the algorithmic choices.

![A sampler branching into a host synchronizing path and an on device path with different step periods](/imgs/blogs/from-logits-to-tokens-the-sampler-zoo-7.webp)

### The cost of `.item()`

The strawman in section 1 ended with `int(torch.multinomial(probs, 1))`. That `int(...)` is a **host sync**: the CPU issues a device-to-host copy of four bytes and then blocks until every kernel already queued on the stream has completed, because the value it wants depends on all of them.

The copy itself is trivial. The blocking is not. In a healthy decode loop the CPU runs *ahead* of the GPU: while the GPU executes step $N$, the CPU is already building tensors and launching kernels for step $N+1$. That overlap is the only reason a loop that issues 200–300 kernel launches per step can keep a GPU fed at all. A sync destroys it.

The model is simple enough to write down. Let $S$ be the GPU time for one decode step and $C$ the CPU time to prepare and launch the next one. With overlap, the step period is $\max(S, C)$. With a sync in the sampler, the CPU cannot start step $N+1$ until step $N$ has finished, so the period becomes $S + C$.

#### Worked example: what a four-byte read costs

Take a small model at high batch, where $C$ is close to $S$ — the regime where host overhead actually bites. Suppose $S = 3.0$ ms and $C = 2.5$ ms.

| configuration | step period | steps/s | Source |
| --- | --- | --- | --- |
| overlapped, no sync | $\max(3.0, 2.5) = 3.0$ ms | 333 | derived |
| host sync in sampler | ${3.0 + 2.5 = 5.5}$ ms | 182 | derived |
| loss | +83% period | −45% throughput | derived |

Forty-five percent of your throughput, for four bytes. And note what happens as models get *smaller* and batches get *larger*: $S$ shrinks (less weight traffic per step) while $C$ grows (more requests to bookkeep), so the ratio moves against you exactly in the regime where you are trying to serve cheaply.

At the other extreme the sync is free. An 8B model at batch 1 on an A100 has a decode step floor of $16.1\ \text{GB} / 2.039\ \text{TB/s} = 7.9$ ms, derived from Llama-3.1-8B's 8.03B parameters in bf16 and the 2,039 GB/s of HBM2e that NVIDIA lists for the A100 80GB SXM. Against 7.9 ms of unavoidable weight reading, a 20 µs sync is 0.25%. This is why the problem is invisible in single-request benchmarks and appears only when you turn the load up — a theme this series keeps returning to, and one [the naive decode loop post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) set up deliberately.

### The Gumbel-Max trick

To sample on device you need to avoid `torch.multinomial`, which needs a normalized probability vector, which needs a materialized softmax. The Gumbel-Max trick removes all three requirements.

**Claim.** If $g_i$ are independent standard Gumbel variates, then

$$\arg\max_i\, (z_i + g_i) \sim \text{Categorical}(\text{softmax}(z))$$

**Proof.** The standard Gumbel has density $f(g) = e^{-g - e^{-g}}$ and CDF $F(g) = e^{-e^{-g}}$. Token $i$ wins when $z_i + g_i > z_j + g_j$ for all $j \ne i$, that is when $g_j \lt g_i + z_i - z_j$. Conditioning on $g_i$ and using independence:

$$P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g - e^{-g}} \prod_{j \ne i} e^{-e^{-(g + z_i - z_j)}}\, dg$$

Collect the exponents. Writing $S = \sum_j e^{z_j}$, the product of the exponentials contributes $\exp\!\big(-e^{-g}\sum_{j\ne i} e^{z_j - z_i}\big)$, and combining with the $-e^{-g}$ from the density gives $\exp\!\big(-e^{-g} \cdot S e^{-z_i}\big)$. So

$$P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g}\, \exp\!\left(-e^{-g}\, S e^{-z_i}\right) dg$$

Substitute $t = e^{-g}$, so $dt = -e^{-g}\,dg$ and the limits flip to ${0}$ and $\infty$:

$$P(i \text{ wins}) = \int_0^{\infty} e^{-t S e^{-z_i}}\, dt = \frac{1}{S e^{-z_i}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

which is exactly $\text{softmax}(z)_i$. The trick dates to Gumbel's extreme-value work and is developed as a sampling method in Maddison et al., [*A\* Sampling*](https://arxiv.org/abs/1411.0030) (NeurIPS 2014).

Three consequences make this the right primitive for an inference kernel:

1. **No softmax.** You never compute $\sum_j e^{z_j}$. That is a full reduction over 128,256 elements per row that you simply do not perform.
2. **No cumulative sum, no binary search.** `multinomial` on a large vocabulary does a prefix sum and a search; Gumbel-Max is one elementwise add plus one argmax reduction, both of which are trivially fusable with whatever produced the logits.
3. **Masking composes for free.** A masked token has logit $-\infty$, and $-\infty + g = -\infty$ for any finite $g$, so it can never win the argmax. No special-casing, no renormalization.

Generating the Gumbel variate is $g = -\ln(-\ln u)$ for $u \sim \text{Uniform}(0,1)$. Equivalently, since $-\ln u \sim \text{Exponential}(1)$, you can write the whole thing as $\arg\max_i\, p_i / e_i$ with $e_i \sim \text{Exponential}(1)$ — a form some engines prefer because `exponential_` is a single kernel.

```python
# nanoserve/sampling.py (continued)

_TINY = 1e-20


def gumbel_argmax(logits: torch.Tensor, step: int, pt: dict) -> torch.Tensor:
    """Sample from softmax(logits) without materializing a softmax.

    logits : [B, V], already penalized / tempered / truncated. Masked
             entries are -inf and can never win the argmax.
    Returns: [B] int64, on device. No host sync anywhere.
    """
    u = torch.rand_like(logits)
    # rand() returns [0, 1); u == 0 would give g = +inf and that token
    # would always win. Clamp. This is not theoretical -- at B=256 and
    # V=128256 you draw 33 million uniforms per step.
    u.clamp_(min=_TINY)
    g = -torch.log(-torch.log(u))
    return (logits + g).argmax(dim=-1)
```

The clamp is a real production detail. `torch.rand` samples from the half-open interval $[0, 1)$, so an exact zero is possible, and a single zero produces $g = +\infty$ and a token chosen with certainty regardless of its logit. At batch 256 over a 128k vocabulary you draw 32.8 million uniforms per decode step. Over an hour at 300 steps per second that is $3.5 \times 10^{13}$ draws. Whether float32 `rand` can actually emit exact zero depends on the generator, but the cost of the clamp is one elementwise `max` and the cost of not having it is a nondeterministic wrong token — take the clamp.

### The randomness has to be per request, not per batch

Here is the failure that makes on-device sampling genuinely hard, and it is not about speed.

`torch.rand_like` draws from one global generator whose state advances by however many numbers you consumed. If request A occupies row 3 of a batch of 8, it consumes uniforms at a different offset in the stream than if it occupied row 0 of a batch of 4. **Same request, same seed, same prompt, different batch composition, different output.** Users file this as "your API is not reproducible", and they are right.

The correct fix is a **stateless, counter-based RNG**: derive the random number for position $(b, v)$ at step $t$ from a pure function of the request's seed, the step index and the vocabulary index, with no mutable stream state at all. Then the noise for a request depends only on things the request owns.

```python
# nanoserve/sampling.py (continued)

def stateless_uniform(seeds: torch.Tensor, step: int, V: int,
                      device: str) -> torch.Tensor:
    """Per-row reproducible uniforms in [0, 1), independent of batch layout.

    seeds : [B] int64, one per request. Returns [B, V] float32.

    This is an illustrative torch-level counter-based generator. A real
    engine does the same hash inside the sampling kernel (Philox), which
    is both faster and avoids materializing the [B, V] noise tensor.
    """
    B = seeds.shape[0]
    v = torch.arange(V, device=device, dtype=torch.int64).unsqueeze(0)
    key = (seeds.unsqueeze(1) * 0x9E3779B1) ^ (step * 0x85EBCA6B) ^ v

    # Thomas Wang style 32-bit avalanche, vectorized.
    x = key & 0xFFFFFFFF
    x = (x ^ 61) ^ (x >> 16)
    x = (x + (x << 3)) & 0xFFFFFFFF
    x = x ^ (x >> 4)
    x = (x * 0x27D4EB2D) & 0xFFFFFFFF
    x = x ^ (x >> 15)

    return (x.float() + 0.5) * (1.0 / 4294967296.0)
```

That function is correct and reproducible, and it is also a bad idea at scale, because it materializes a `[B, V]` int64 intermediate — 256 × 128,256 × 8 bytes = 262.7 MB of transient VRAM, plus five elementwise passes over it. It is here to show you the *shape* of the answer, not to ship.

The shape of the answer that does ship is a fused kernel. The vLLM team's Model Runner V2 post describes exactly this: a **Triton sampler that uses Gumbel-Max so no explicit softmax is materialized, with a stateless in-kernel RNG**, alongside zero-sync async scheduling where the CPU schedules step $N+1$ while the GPU runs step $N$ and outputs are copied on a separate CUDA stream. They also note that top-k logprobs are computed only for the selected candidates rather than for the whole vocabulary — which is the same economy applied to the *output* side, since returning logprobs over 128,256 entries when the client asked for the top 5 is a full sort you did not need.

The numbers they report, cited with their setup: **+56% output throughput** (25K versus 16K output tokens per second) on Qwen3-0.6B on a single GB200 — a configuration chosen to stress host overhead — and **−6.3% TPOT** on GLM-4.7-FP8 across 4×GB200 with MTP=1. Enabled via `VLLM_USE_V2_MODEL_RUNNER=1`. Those are their measurements on their hardware, not mine.

The choice of Qwen3-0.6B for the stress configuration is not arbitrary, and section 10 shows why. A later post in this track's companion kernels track builds the fused Triton sampler itself, at which point the composable-processor design here becomes a single kernel — and you will see the price of that fusion, which is the same price MRv2 pays: as of v0.18.0 that path did not support logits processors at all. Composability and fusion pull in opposite directions, and an engine has to choose which it is optimizing.

---

## 10. Measuring the sampler honestly

The sampler is easy to measure wrongly because it is small, fast, and sitting downstream of something enormous.

### The protocol

- **Warm up.** The first call allocates workspace, autotunes, and JITs. Discard at least 20 iterations.
- **Synchronize before you start and before you stop.** `torch.cuda.synchronize()` on both sides, or better, bracket with `torch.cuda.Event` and use `elapsed_time`. Without this you are timing kernel *launches*, and a launch is roughly 5 µs regardless of what it launches.
- **Use the real shapes.** A sampler benchmarked at `[1, 32000]` tells you nothing about `[256, 128256]`. Feed it a random logits tensor of the exact shape your engine produces.
- **Measure in isolation and in situ.** In isolation to know the sampler's cost; in situ to know whether it is on the critical path. A sampler that costs 200 µs matters enormously when the step is 600 µs and not at all when the step is 8 ms.
- **Report steady state.** Run 200 iterations, take the median, report the interquartile range. A mean over 10 runs on a GPU whose clocks are still ramping is noise.

```python
# nanoserve/bench_sampler.py
import torch


def bench(fn, *args, warmup=20, iters=200):
    """Return (median_ms, p90_ms) using CUDA events. Never call .item()
    inside the timed region -- that is the thing we are measuring."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()

    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2], ms[int(len(ms) * 0.9)]


if __name__ == "__main__":
    torch.manual_seed(0)
    for B, V, name in [(1, 128256, "llama-3.1-8b  B=1"),
                       (64, 128256, "llama-3.1-8b  B=64"),
                       (256, 151936, "qwen3-0.6b    B=256")]:
        z = torch.randn(B, V, device="cuda", dtype=torch.float32)
        m_sort, _ = bench(lambda t: torch.sort(t, dim=-1, descending=True), z)
        m_topk, _ = bench(lambda t: torch.topk(t, 64, dim=-1), z)
        m_gum, _ = bench(
            lambda t: (t + -torch.log(-torch.log(
                torch.rand_like(t).clamp_(min=1e-20)))).argmax(-1), z)
        print(f"{name}: sort {m_sort:.3f} ms  topk64 {m_topk:.3f} ms  "
              f"gumbel {m_gum:.3f} ms")
```

### What you should expect to see, and why

I have not run this. But you can bound it from memory traffic, which is the honest way to set an expectation.

A descending sort of `[64, 128256]` fp32 touches 8,208,384 elements: 32.8 MB of keys plus 32.8 MB of int32 indices, so 65.7 MB of payload. A radix sort over 32-bit keys makes several passes and reads and writes the payload each time; six to ten passes puts total traffic in the 0.8–1.3 GB range. At an achieved bandwidth of roughly 1.5 TB/s on an A100 that lands somewhere in the region of **0.5–0.9 ms**. That is an order-of-magnitude derivation from a stylized model of the sort, not a measurement, and the real number depends heavily on which sort implementation your PyTorch build dispatches to. Run the script.

| configuration | sampler-relevant quantity | value | Source |
| --- | --- | --- | --- |
| Llama-3.1-8B bf16, A100 80GB SXM | decode step floor, batch-independent | 7.9 ms | derived: 16.1 GB / 2,039 GB/s |
| Llama-3.1-8B bf16, RTX 4090 | decode step floor | 16.0 ms | derived: 16.1 GB / 1,008 GB/s |
| Llama-3.1-8B bf16, L4 | decode step floor | 53.7 ms | derived: 16.1 GB / 300 GB/s |
| Qwen3-0.6B bf16, A100 | decode step floor | 0.59 ms | derived: 1.2 GB / 2,039 GB/s |
| logits tensor, B=256, V=128,256, fp32 | transient VRAM per copy | 131.3 MB | derived |
| logits tensor, B=256, V=151,936, fp32 | transient VRAM per copy | 155.6 MB | derived |
| penalty counts buffer, B=256, V=128,256 | resident VRAM | 131.3 MB | derived |
| stateless-RNG intermediate, B=256, V=128,256, int64 | transient VRAM | 262.7 MB | derived |
| sort of [64, 128256] fp32 | traffic, 6–10 radix passes | 0.8–1.3 GB | derived, order-of-magnitude |
| vLLM MRv2 vs V1, Qwen3-0.6B, 1×GB200 | output throughput | 25K vs 16K tok/s (+56%) | cited: vLLM MRv2 post |
| vLLM MRv2 vs V1, GLM-4.7-FP8, 4×GB200, MTP=1 | TPOT | −6.3% | cited: vLLM MRv2 post |
| sampler cost across the shapes above | median ms per call | run `bench_sampler.py` | reproduce |

### The ratio that explains everything

Put two rows of that table together. An 8B model at batch 64 has a decode floor of 7.9 ms on an A100, and a sort at that shape costs somewhere in the hundreds of microseconds — call it 5% of the step. Annoying, ignorable.

Now Qwen3-0.6B at batch 256. The decode floor is 0.59 ms. The logits row is 151,936 wide, so the sort is on `[256, 151936]` = 38.9M elements, roughly five times the payload of the 64-row case, and by the same stylized model it costs several milliseconds. **The sampler is now several times more expensive than the model forward pass.**

That is a derived estimate with a stated model, and it could easily be off by 3×. It does not need to be accurate to be decisive: even at a third of that, the sampler dominates. And it explains, without any inside knowledge, why the vLLM team chose Qwen3-0.6B on a GB200 as their host-overhead stress configuration for Model Runner V2 and why a fused Gumbel-Max sampler was worth writing. When the model is small and the batch is large, the decoding layer *is* the engine.

---

## 11. When to reach for which sampler

Folklore presets are the norm in this space and they should not be. Here is the reasoning, per task, with the mechanism that justifies it.

| Task | Setting | Why |
| --- | --- | --- |
| Extraction, classification, span answers | `temperature=0` | There is a correct answer. Any randomness is pure error. |
| Evaluation runs | `temperature=0` | You are measuring the model, not the model plus an RNG. Sampling inflates variance and makes runs incomparable. |
| Tool calls / function arguments | `temperature=0`, all penalties 0, constrained decoding | Argument names and JSON punctuation repeat by necessity. Penalties actively fight the schema. |
| Code completion | `temperature≈0.2`, `top_p=0.95`, penalties 0 | Code has near-deterministic continuations (closing brackets, known identifiers) with occasional genuine branch points. Low $T$ preserves the first; a loose nucleus preserves the second. Zero penalties because `for`, `self`, `return` must recur. |
| Chat / assistant | `temperature≈0.7`, `top_p≈0.9` or `min_p≈0.05`, `presence≈0.1` | Enough entropy for varied phrasing; a nucleus to cut the tail; a *bounded* penalty for gentle variety. |
| Summarization | `temperature≈0.3`, `top_p=0.9`, penalties 0 | Faithfulness dominates. Entities and numbers must be copied verbatim, which any penalty punishes. |
| Creative writing | `temperature≈1.0–1.2`, `min_p≈0.05`, DRY if available | High entropy is the point; min-p is the tail filter that survives the high temperature; DRY breaks loops without banning common words. |
| Long structured output (JSON, CSV, XML) | `temperature=0` or very low, `frequency_penalty=0` | Required repeated tokens. Use a grammar, not a sampler. |
| Diverse candidate generation (best-of-n, self-consistency) | `temperature≈0.8–1.0`, `top_p=0.95`, distinct seeds | You want spread across samples, not within one. Vary the seed, not the temperature, once you have enough entropy. |

Three rules underneath the table:

**Pick one truncator as primary.** `top_p` *or* `min_p`, with `top_k` as a cheap safety bound if you want one. Stacking `top_k=40, top_p=0.9, min_p=0.05, typical_p=0.95` means three of the four are doing nothing on most steps and you have no idea which.

**Penalties are the last knob you touch, not the first.** If the output is repetitive, the first questions are whether the prompt is repetitive, whether the temperature is too low, and whether the context contains an example the model is copying. Penalties are a symptomatic treatment with real side effects.

**Set temperature before you set anything else.** It is the knob with the clearest mechanism and the most predictable effect, and every other knob's meaning depends on it.

---

## 12. Stress tests: parameter combinations that break

Every one of these is a real configuration I have seen in a request payload.

**`top_k=1` with `temperature=2.0`.** Truncation leaves exactly one token. Softmax over a single element is 1.0 regardless of the logit, so temperature has literally no effect. The user set the "creativity" knob to its maximum and got pure greedy decoding. This is order-independent: whichever runs first, a one-element distribution has no entropy to scale. Any sensible API should warn on it.

**`temperature=0` with `frequency_penalty=0.5`.** People assume temperature 0 means "no sampling parameters apply". It does not. Penalties run *before* temperature and change which token is the argmax. Take the section 2 row: `mat` at 3.2, seen twice, becomes $3.2 - 2 \times 0.5 = 2.2$, which is below `floor` at 2.7. Greedy decoding now emits `floor`. The output is still deterministic — but it is deterministic given the *generation history*, which means the same prompt with a different `max_tokens` can diverge partway through. Source: `derived`. Some engines skip penalty computation entirely when $T = 0$; others do not; nothing in the API tells you which.

**`top_p=1.0` with `min_p=0.0` and `temperature=1.5`.** No truncation at all, and section 5 showed what a flattened 128k tail looks like. Almost the entire vocabulary is reachable, including byte-fallback fragments that break UTF-8 in the middle of a multibyte character. The detokenizer then has to decide what to stream to the client, which is exactly the incremental-detokenization problem this series covers in Track A. High temperature without a truncator is not a style choice, it is a bug.

**Per-request parameters inside one batched step.** Request A wants `top_k=0` (disabled) and request B wants `top_k=20`. The vectorized implementation in section 8 handles this by substituting $V$ for the disabled row, so the `rank >= k` comparison is trivially false everywhere. Get that substitution wrong — leave the zero in — and request A's mask kills every token, `kill[:, 0] = False` saves you from a `nan`, and request A silently becomes greedy. It will not crash. It will just quietly be wrong for one user, forever, which is worse.

**Penalties meeting the tokenizer.** A penalty applies to a *token*, not a word. `hemoglobin` might be `hemo` + `globin`, and ` hemoglobin` with a leading space is a different token again. Penalizing one spelling does not penalize the other, so a frequency penalty aimed at reducing repetition of a word may only reduce one of its tokenizations, producing text that alternates between spellings in a way that reads distinctly strange. Worse, byte-fallback tokens are shared across every rare word, so penalizing one rare word slightly penalizes all of them. If any of that is unfamiliar, [the BPE tokenizer post](/blog/machine-learning/large-language-model/bpe-tokenizer) has the mechanics.

**Sampling in bf16.** Dividing logits by temperature in bf16 costs you precision exactly where it matters. bf16 has 8 mantissa bits, so around a logit value of 8 the representable spacing is about 0.03 — larger than the difference between many adjacent candidates. Cast to fp32 before the softmax; `nanoserve` does it as the first line of `BatchedSampler.__call__`. The full treatment, including why this interacts with batch size, is in [the numerics and determinism post](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance).

**A truncator that masks everything.** `top_p=0.0`, or `min_p=1.0` on a distribution where no token except the max clears the bar, or a grammar mask (Track D, coming up) that admits no token because the FSM reached a dead state. Softmax over all `-inf` is `nan`, `argmax` over `nan` returns index 0, and your model emits `<unk>` forever. The `kill[:, 0] = False` invariant handles the first two. The third is a genuine bug in the grammar compiler and needs to raise, not degrade.

---

## Case studies and public numbers

**vLLM Model Runner V2, on why the sampler is worth fusing.** The vLLM team's [Model Runner V2 post](https://vllm.ai/blog/2026-03-24-mrv2) (2026-03-24) is the most direct public statement of everything in section 9. Its sampler is a Triton kernel using Gumbel-Max, so no explicit softmax is materialized, with a stateless in-kernel RNG; the scheduler is zero-sync, with the CPU planning step $N+1$ while the GPU runs $N$ and outputs copied on a separate stream; input tensors are built on-GPU by Triton kernels rather than assembled on the host. Reported: +56% output throughput (25K versus 16K output tok/s) on Qwen3-0.6B on 1×GB200, and −6.3% TPOT on GLM-4.7-FP8 on 4×GB200 with MTP=1. The limitation stated for v0.18.0 is the interesting part for this post: **no logits-processor support on that path.** A fused sampler and a pluggable processor chain are, at the kernel level, in tension.

**vLLM V1's engine-core split, on where the sampler sits.** The [vLLM V1 architecture post](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27) moved the engine core into a separate process communicating over ZeroMQ, overlapping tokenization, multimodal preprocessing and detokenization with the model loop. That is the structural version of the same insight: anything that touches the host should not be inline with the step. It reports up to 1.7× throughput over V0 on Llama-3.1-8B and 3.3-70B on ShareGPT traces.

**Structured decoding, on the sampler as the enforcement point.** vLLM's [structured decoding post](https://vllm.ai/blog/2025-01-14-struct-decode-intro) (2025-01-14) describes both the FSM approach (Outlines) and the pushdown-automaton approach (XGrammar) as *logit masks* — the same `-inf` mechanism as top-k, driven by a grammar instead of a rank. They report XGrammar delivering up to 5× TPOT improvement under load versus the alternative, and note that grammar compilation is a significant contributor to TTFT. It also names the constraint that motivates the batched design in section 8: with a per-request logits processor, a batch mask computation can block all requests in the batch. The next posts in this track build that machinery from scratch.

**Holtzman et al. on why any of this is necessary.** [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) is still the clearest empirical statement of the problem the whole zoo exists to solve: maximization-based decoding produces text that is measurably *too* probable relative to human text, and truncation sampling is what closes the gap. If you read one paper from this section, read that one.

---

## When to reach for this (and when not to)

**Write your own sampler when:** you are learning how an engine works (this series' entire premise); you need a decoding rule that no engine implements — a domain-specific penalty, a custom mask, a task-specific truncation criterion; or you are building the constrained-decoding layer, in which case the sampler is where your grammar mask has to land.

**Do not write your own sampler when** you just want good output from a served model. Use vLLM or SGLang. Their samplers are correct, batched, tested against reference implementations, and in vLLM's case increasingly fused. The performance work in section 9 is worth doing once, by a team that maintains it, and you are not going to beat a Triton kernel with PyTorch ops.

**Do spend an afternoon on your parameters, regardless of engine.** This is the highest-leverage hour in the whole serving stack and it costs nothing. Determine your engine's application order with the probe in section 2. Pick one primary truncator. Set frequency penalty to zero for anything structured. Set temperature 0 for evals. Those four decisions will change your outputs more than any kernel optimization in this series.

**Do not tune sampling parameters to fix a prompt problem.** Repetitive output usually means a repetitive prompt or a context containing an example the model is copying. Wrong facts at high temperature mean the temperature is too high, not that you need a penalty. The sampler makes the model act on more or less of its uncertainty; it cannot manufacture knowledge the model does not have.

---

## Key takeaways

1. **The order of the pipeline is a choice, and it changes the distribution.** Temperature 1.5 with top-p 0.9 admits four tokens if temperature runs first and three if it runs second, on the same logits. Determine your engine's order empirically before you trust a copied preset.
2. **Temperature scales entropy; it never changes the argmax.** It is overwhelmingly a tail control — on a six-token row, going from $T=0.7$ to $T=2.0$ multiplies the last token's probability by 23 while the leader moves by a factor of 1.8.
3. **top-k is a count, top-p is a mass, min-p is a ratio.** Only min-p is temperature-robust, because in logit space it is a fixed gap: $z_{\max} - z_i \le T \ln(1/p_{\text{base}})$. At $T=2$ on a stylized 128k vocabulary, top-p 0.9 admits ~115,000 tokens while min-p 0.05 admits four.
4. **Frequency penalty is unbounded and therefore fatal to structured output.** A token seen 40 times at `frequency_penalty=0.3` carries 12 logits of penalty — odds cut by 162,754×. Set it to zero for JSON, code, and anything with required repeated tokens.
5. **Repetition penalty is not shift-invariant.** It is multiplicative on logits whose absolute scale is arbitrary, so the same value has a different strength on different models. Prefer additive presence and frequency penalties, windowed.
6. **Make truncators intersect, not chain.** Compute each keep-mask against one post-temperature distribution and AND them together. Then `top_k` and `top_p` commute, and nobody has to know which ran first.
7. **Per-request parameters become `[B]` tensors, built when the running set changes — not per step.** A Python loop over rows costs one kernel launch per row per knob and will make your engine CPU-bound.
8. **Never read the sampled token back to the host inside the step.** With $S=3$ ms of GPU time and $C=2.5$ ms of CPU time, a sync turns a 3.0 ms step period into 5.5 ms — a 45% throughput loss for four bytes.
9. **Gumbel-Max removes the softmax entirely.** $\arg\max_i (z_i + g_i)$ with standard Gumbel noise samples exactly from $\text{softmax}(z)$, it needs no normalization or prefix sum, and `-inf` masks compose with it for free.
10. **When the model is small and the batch is large, the sampler is the engine.** A `[256, 151936]` sort can cost several times a 0.6B model's decode step. That is why production engines fuse it — and why fusing it costs you the pluggable processor chain.

---

## Further reading

- Holtzman, Buys, Du, Forbes & Choi, [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) (ICLR 2020) — the nucleus sampling paper and the empirical case against maximization.
- Fan, Lewis & Dauphin, [*Hierarchical Neural Story Generation*](https://arxiv.org/abs/1805.04833) (ACL 2018) — where top-k sampling enters open-ended generation.
- Meister, Pimentel, Wiher & Cotterell, [*Locally Typical Sampling*](https://arxiv.org/abs/2202.00666) (TACL 2023) — the information-theoretic criterion behind typical-p.
- Nguyen et al., [*Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*](https://arxiv.org/abs/2407.01082) — the write-up of min-p.
- Keskar, McCann, Varshney, Xiong & Socher, [*CTRL: A Conditional Transformer Language Model*](https://arxiv.org/abs/1909.05858) — the origin of the multiplicative repetition penalty.
- Maddison, Tarlow & Minka, [*A\* Sampling*](https://arxiv.org/abs/1411.0030) (NeurIPS 2014) — the Gumbel-Max trick developed as a sampling method.
- vLLM blog, [*Model Runner V2*](https://vllm.ai/blog/2026-03-24-mrv2) — the Triton Gumbel-Max sampler, stateless in-kernel RNG, and zero-sync async scheduling, with their reported numbers and limitations.
- vLLM blog, [*Structured Decoding in vLLM*](https://vllm.ai/blog/2025-01-14-struct-decode-intro) — logit masking as the enforcement mechanism for grammars, and the batch-blocking constraint it creates.
- Hugging Face, [*Generation utilities*](https://huggingface.co/docs/transformers/en/internal/generation_utils) — the full `LogitsProcessor` catalogue and the order in which `generate()` composes them.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) · [the naive decode loop and your first baseline](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline) · [sampling numerics, determinism and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance) · [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).
