---
title: "Train-Inference Mismatch for LLMs: Great Loss, Broken Generation"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose the family of LLM bugs where training loss falls beautifully but generation produces loops, drift, or garbage — exposure bias, KV-cache divergence, decoding skew, and serve-time dtype shifts — and prove each fix with a cache-equivalence test."
tags:
  [
    "debugging",
    "model-training",
    "llm",
    "kv-cache",
    "finetuning",
    "deep-learning",
    "pytorch",
    "decoding",
    "quantization",
    "exposure-bias",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/train-infer-mismatch-for-llms-1.png"
---

The finetune was a triumph, right up until I asked it to write a sentence. Twelve thousand steps, a cross-entropy loss that fell from 1.9 to 0.78 and held there, a validation loss tracking it almost exactly, no spikes, no NaNs, a perplexity that would have looked at home in a paper. By every instrument I had been taught to trust, this was a healthy run. Then I loaded the checkpoint, prompted it with a simple instruction, called `model.generate`, and watched it produce: "The the the answer is is the the the the the the" for ninety-seven tokens until it hit `max_new_tokens` and gave up. No exception. No warning. A model that had, on paper, learned the task to a loss of 0.78 could not produce a single coherent sentence.

I did what everyone does first: I assumed the weights were broken. I re-ran the eval loop and got the same low loss. I re-loaded from a different checkpoint and got the same garbage generation. I lowered the learning rate and retrained, and got the same low loss and the same garbage. For most of a day I treated this as an *optimization* bug — surely a model that generates "the the the" has not actually learned anything, and the loss must be lying to me about that. It was not. The loss was telling the truth about exactly what it measured. The problem was that *what it measured was not what generation does*, and the gap between those two things is where this entire class of bug lives. Figure 1 is the whole story in one picture: training feeds the model the gold prefix at every step and asks it to predict one token; generation feeds the model its *own* previous outputs and asks it to predict the next. Those are different conditioning distributions, and cross-entropy only ever scored the first one.

![Two-column figure contrasting teacher-forced training that always conditions on the gold prefix against free-running generation that conditions on the model's own previous tokens and drifts](/imgs/blogs/train-infer-mismatch-for-llms-1.png)

This post is about the family of bugs where **training and inference disagree** — where the loss is genuinely low, the gradients were genuinely fine, the data was genuinely clean, and yet the deployed model generates something the training distribution would never have produced. It is one of the most disorienting bug classes in LLM work precisely because every instrument you would normally read says the run is healthy. The loss curve is your primary diagnostic for most training bugs, and here it is *structurally incapable* of seeing the problem. So the discipline has to change: instead of staring at the loss, you have to make training and inference compute the same thing, find the one place they diverge, and prove the convergence with a test rather than a vibe.

By the end you will be able to do six concrete things. First, explain *why* cross-entropy loss under teacher forcing cannot measure exposure bias — the gap between the conditioning distribution the loss sees and the one generation actually walks. Second, write and run a **cache-equivalence test** that decodes a sequence with and without the KV-cache and asserts the tokens match, which catches the single most common silent inference bug: a cached decode that diverges from a full forward pass. Third, recognize the signatures of the other five mismatches — decoding-param skew, padding and position drift, serve-time dtype and quantization shifts, chat-template skew, and EOS/stopping bugs — from how the generation fails. Fourth, build a **generate-during-training callback** that decodes a fixed probe set every few hundred steps so you see exposure-bias drift while you still can fix it, instead of at release. Fifth, write a serve-time audit that diffs the dtype, the format string, and the decoding config between training-eval and production. Sixth, know precisely when a broken generation points at a mismatch and when it points back at the weights. These bugs live mostly in the **model-code**, **numerics**, and **evaluation** corners of the [six places a bug hides](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — data, optimization, model code, numerics, systems, evaluation — and the master move is the same as always: bisect to the corner before you touch code.

A note on scope. This is the *inference-time* companion to several sibling posts. The padding and position mechanics get their full treatment in the [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms) post; the format string and role tokens are covered in [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs); the more general "which mode is my model in" trap is in [train vs eval mode bugs](/blog/machine-learning/debugging-training/train-eval-mode-bugs). Here the spine is the *train-inference gap itself*: I will derive why the loss cannot see it, give you the cache-equivalence test that nails the worst offender, and point at those siblings where their mechanism is the proximate cause rather than re-deriving them.

## 1. The science: why a perfect loss cannot see a broken generation

Start from what the loss actually computes, because the entire bug class follows from one detail most people never look at. A decoder-only language model is trained to maximize the likelihood of the next token given the *true* previous tokens. For a target sequence $y_1, y_2, \dots, y_T$, the per-token cross-entropy loss is

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_1, \dots, y_{t-1}\big).
$$

Read the conditioning bar carefully. At step $t$, the model is conditioned on $y_1, \dots, y_{t-1}$ — the **gold** prefix, the actual tokens from the training corpus. This is called **teacher forcing**: no matter what the model predicted at step $t-1$, the input to step $t$ is the ground-truth token $y_{t-1}$, not the model's own prediction. Every position is scored independently against the truth, and the prefix is always correct because it comes from the data, not from the model. That is what makes training efficient — a single forward pass over a length-$T$ sequence produces $T$ supervised predictions in parallel, because the entire gold sequence is available up front and the causal mask lets every position attend to its true predecessors at once.

Now look at generation. At inference there is no gold sequence — that is the whole point, we are generating it. So step $t$ must be conditioned on the model's *own* previous outputs:

$$
\hat{y}_t \sim p_\theta\big(\cdot \mid \hat{y}_1, \dots, \hat{y}_{t-1}\big).
$$

The conditioning bar now holds $\hat{y}_1, \dots, \hat{y}_{t-1}$ — tokens the model produced, which may differ from any gold sequence and, crucially, may not look like *anything* in the training distribution. This regime is called **free running** or **autoregressive decoding**. The model is now consuming a prefix it generated itself, and the loss never scored this regime even once. The loss conditioned on gold prefixes; generation conditions on model prefixes. They are different functions of the same weights.

### Exposure bias, made precise

The term for this gap is **exposure bias** (Ranzato et al., 2016; the term and analysis come from the sequence-level training literature, building on Bengio et al.'s 2015 scheduled-sampling work). The name is exact: during training the model is only ever *exposed* to gold prefixes, so it never learns how to behave when its prefix contains its own mistakes. To make this provable rather than hand-wavy, model a single decoding step as having some per-step error probability $\epsilon$ — the probability that, conditioned on a *correct* prefix, the model samples a token that takes it off the data manifold. Under teacher forcing this is all the loss ever sees: each step starts from a correct prefix, so the relevant error rate is exactly $\epsilon$ at every position, independently.

Free-running generation is different because errors are not independent across steps — they *condition the future*. Suppose at some step the model samples an off-distribution token. Now the prefix for the next step is no longer something the training distribution contains, so the model's error probability there is not $\epsilon$ but something larger, call it $\epsilon' > \epsilon$, because it is being asked to continue from a context it was never trained on. If that step also errs, the prefix drifts further, and $\epsilon''$ is larger still. The probability of producing a fully correct length-$T$ sequence under free running is therefore *not* $(1-\epsilon)^T$ — that would be the teacher-forced bound. It is bounded below by a product of growing failure probabilities, and once the prefix leaves the manifold there is often no path back. This is the geometric-compounding intuition behind the classic result that the gap between teacher-forced and free-running error grows with sequence length. The loss, which measures only the per-step teacher-forced $\epsilon$, is blind to the compounding entirely. A model can have a tiny per-step $\epsilon$ — a beautiful loss — and still almost never produce a clean long sequence, because the compounding is not in the objective. Figure 2 draws the compounding cascade: one slip moves the context off-distribution, which raises the per-step error rate, which makes the next slip more likely, which is how you get "the the the" or topic drift or an answer that starts coherent and dissolves by token 80.

![Branching graph showing one wrong sampled token moving the prefix off the training distribution, raising the per-step error probability, and cascading into loops or topic drift with a rare self-recovery branch](/imgs/blogs/train-infer-mismatch-for-llms-2.png)

To put a number on the compounding, take the crudest possible model of it. Suppose teacher-forced training drives the per-step error rate (the chance of leaving the manifold from an on-manifold prefix) down to $\epsilon = 0.01$ — one percent, an excellent number that corresponds to a very low loss. Under teacher forcing, every one of the $T$ steps starts from a gold prefix, so the expected number of errors over a length-$T$ sequence is just $\epsilon T$: at $T = 200$ that is two errors, and each is independent and recoverable because the *next* step is handed the gold token regardless. The teacher-forced world is forgiving by construction. Now switch to free running with the pessimistic-but-instructive assumption that *once you leave the manifold you do not come back* — the off-manifold error rate is effectively 1. Then the probability of generating a fully clean length-$T$ sequence is $(1-\epsilon)^T$, and at $\epsilon = 0.01$, $T = 200$ that is $0.99^{200} \approx 0.134$ — a model with a *one-percent* per-step error rate produces a clean 200-token generation only about thirteen percent of the time. Push to $T = 500$ and it is $0.99^{500} \approx 0.0066$ — essentially never. The loss reported $\epsilon = 0.01$ and felt great; the generation is broken most of the time, and *nothing in the loss told you*. The real world sits between these two bounds — models do partially recover, which is why large LLMs generate coherent long text at all — but the gap between $\epsilon T$ expected errors (teacher-forced) and a $(1-\epsilon)^T$ clean-sequence probability (free-running, no recovery) is the exposure-bias gap made arithmetic, and it is why a tiny per-step loss is not a promise of clean generation.

There is a clean way to state the whole thing. Teacher-forced loss measures $\mathbb{E}_{y \sim \text{data}}[-\log p_\theta(y_t \mid y_{<t})]$, an expectation over **gold** prefixes drawn from the data distribution. Generation quality depends on the model's behavior under **its own** prefix distribution $p_\theta(\hat{y}_{<t})$, which is a different measure. When these two prefix distributions match — when the model's free-running rollouts stay on the manifold the data lives on — the loss is a faithful proxy for generation, and most well-trained large models are in this regime most of the time. When they diverge — because of a true exposure-bias problem, or, far more often in practice, because a *bug* knocks the inference path off the training path — the loss and the generation come apart, and you are in this post.

That last sentence is the practical pivot, and it is worth being blunt about. Pure exposure bias — the model genuinely failing to recover from its own small errors despite a correct inference stack — is real but is *not* usually why your finetune generates garbage. Large pretrained models are remarkably robust to their own minor errors precisely because pretraining exposed them to enormous diversity. When a finetuned LLM generates garbage while showing a great loss, the overwhelmingly likely cause is a **mechanical mismatch**: the inference code is computing something different from what training optimized. The KV-cache diverged from a full forward pass. The serve-time dtype is not the training dtype. The chat template at inference does not match the one in training. The decoding parameters are wrong. The loss is honest; the inference path is buggy. The rest of this post is a tour of those mechanical mismatches, hardest-hitting first, each with the test that proves it.

> The provable point: cross-entropy under teacher forcing measures per-step error on *gold* prefixes, while generation walks the model's *own* prefix distribution. The loss is structurally blind to anything that makes those two prefix distributions diverge — exposure-bias compounding, or any inference-path bug. A great loss with broken generation is therefore not a contradiction; it is the expected signature of a train-inference mismatch.

## 2. The worst offender: KV-cache vs full-attention divergence

Of all the mechanical mismatches, the KV-cache bug is the one I have lost the most hours to, because it is the one that *should* be impossible. The KV-cache is supposed to be a pure speed optimization that changes nothing about the math. When it changes the math, you get a model that scores perfectly under the eval loop (which usually does a full forward pass with no cache) and generates subtly-then-grossly wrong text at serving time (which uses the cache). Understanding the bug requires understanding the equivalence the cache is supposed to preserve.

### The equivalence condition the cache must satisfy

In a decoder-only transformer, generating token $t+1$ requires the attention over all previous positions $1, \dots, t$. Naively, you would re-run the full forward pass over the entire prefix every step — that is the **full forward** or **uncached** path, and it costs $O(t^2)$ work to generate a length-$T$ sequence because step $t$ re-attends over $t$ positions. The **KV-cache** exploits the fact that, in a causal transformer, the keys and values for positions $1, \dots, t$ do not change when you append position $t+1$ — position $t$'s representation only attends *backward*, so adding a future token never alters past keys and values. So you cache the keys $K_{1:t}$ and values $V_{1:t}$ from previous steps, compute only the new query, key, and value for position $t+1$, append them to the cache, and attend the new query over the cached $K, V$. This drops the per-step cost to $O(t)$ and the total to $O(T^2)$ but with a tiny constant, because each step does one token of work instead of re-encoding the whole prefix.

The mathematical claim that makes this valid is an **equivalence condition**, and it is the assertion your test should check directly:

$$
\text{logit}_t^{\text{cached}} = \text{logit}_t^{\text{full}} \quad \text{for every position } t,
$$

up to floating-point tolerance. The logit the cached decode produces at position $t$ must equal the logit a single full forward pass over the same prefix would produce at position $t$. If that holds for every $t$, the cache is correct and generation with the cache is bit-for-bit (within float tolerance) the same as generation without it. If it fails at some position $t^\*$, the cache has a bug, and from $t^\*$ onward the cached generation walks a different path from the reference — which, because of the exposure-bias compounding from Section 1, often means it dissolves into garbage even though the *first* divergence was a tiny numerical difference. Figure 3 is the contrast: the full forward recomputes everything and is the reference; the cached decode reuses `past_key_values` and is correct only if its position bookkeeping and cache state are exactly right.

![Two-column figure contrasting a full forward pass that recomputes all attention as the reference logit against a cached decode that reuses past key values and diverges when position ids are off by one or the cache is not reset](/imgs/blogs/train-infer-mismatch-for-llms-3.png)

### The three ways the cache actually breaks

In practice the equivalence fails for one of three reasons, and they have distinct fingerprints.

**Position ids.** This is the most common one. Rotary position embeddings (RoPE) and learned absolute position embeddings both depend on the *position index* of each token. When you decode with a cache, the new token's position is not 0 — it is the current sequence length. If your code passes `position_ids` that restart at 0 for the new token (a classic bug when you hand-roll the decode loop, or when you set up a cache for a prefix and forget that the continuation positions must continue from the prefix length), the rotary phase applied to the new query and key is wrong, and the attention dot products are computed at the wrong relative angle. The result is an attention pattern that is subtly off, which produces a logit that is subtly off, which diverges from the full forward whose positions were correct. In modern Hugging Face `transformers`, this is managed by the `cache_position` tensor; older hand-written loops used an explicit `position_ids` argument that was easy to get wrong. The fingerprint: the cached and full-forward logits match for the *prefix* (where positions agree trivially) and start diverging at the *first generated token* — exactly where the position bookkeeping kicks in.

**Cache not reset between sequences.** The cache is stateful. If you reuse a model object across generations and the `past_key_values` from the previous prompt are not cleared, the new sequence attends over the *previous* sequence's keys and values. With Hugging Face `generate` this is handled for you, but the moment you write your own loop or use a low-level `model(input_ids, past_key_values=...)` call, a stale cache is a real risk. The fingerprint here is bizarre: generation quality depends on *what you generated before*, so the same prompt gives different outputs depending on session history, and a fresh process "fixes" it.

**Cache dtype and layout.** The cache stores keys and values; if it is allocated in a lower precision than the compute (a common memory optimization — an fp16 or even int8 KV-cache while the model computes in bf16), the cached attention is computed on rounded keys and values while the full forward uses full-precision ones. Usually this is a small, tolerable difference, but combined with long contexts and aggressive quantization it can push the cached path measurably off the full-forward path. The fingerprint: the divergence is small per step but grows with context length, and it shrinks if you raise the cache precision.

It is worth being precise about *why the training eval loop is blind to all three of these*, because that blindness is what lets the bug survive to production. A standard evaluation loop computes loss (or perplexity) by running a single forward pass over each full gold sequence with `use_cache=False` and the causal mask, scoring every position at once — exactly the teacher-forced computation from Section 1. That forward pass never instantiates a `past_key_values` object, never appends to a cache, never decodes step by step, and never advances a `cache_position` across generated tokens. So every one of the three cache failure modes — wrong continuation positions, stale cache state, lossy cache dtype — is *structurally outside* the code path the eval loop exercises. The eval number can be flawless precisely because it measures the one path (full forward, gold prefix) that has no cache and no autoregression. The cache bug lives only in the *other* path, the one you ship. This is the deepest reason the loss and the generation come apart here: they run different code, and the difference is exactly the buggy part.

A useful way to internalize the equivalence is to write out what the cached step actually computes. At decode step appending token at position $t$, the model computes a single new query $q_t$, key $k_t$, and value $v_t$ from the new token's embedding (plus its position), appends $k_t, v_t$ to the cached $K_{1:t-1}, V_{1:t-1}$, and computes attention as $\text{softmax}\!\big(q_t K_{1:t}^\top / \sqrt{d}\big) V_{1:t}$. Compare that to the full forward, which recomputes $q_t, k_{1:t}, v_{1:t}$ for *every* position from scratch and reads off the same attention at row $t$. The two are identical **if and only if** three things hold: the cached $K_{1:t-1}, V_{1:t-1}$ are the same values the full forward would compute for those positions (broken by a stale cache or a lossy cache dtype), the new $q_t, k_t$ carry the correct position $t$ (broken by a position-id bug — RoPE rotates $q_t$ and $k_t$ by an angle proportional to $t$, so a wrong $t$ rotates them wrong), and the causal mask is the same (it trivially is, since both attend over $1{:}t$). Reading the equivalence as "the cached step must reconstruct the same $q, k, v$ and the same history the full forward would" tells you exactly which of the three to check when the test fails.

The fix in all three cases is to make the cached path reproduce the full-forward path exactly: correct `cache_position` / `position_ids`, a fresh cache per sequence, and a cache precision that does not perturb the logits beyond your tolerance. But you cannot *fix* what you cannot *see*, and the loss will not show you any of this. You need a test that asserts the equivalence directly.

### The cache-equivalence test

This is the single most valuable diagnostic in this entire post. Generate a sequence two ways — once with the cache and once with a full forward pass per step — and assert the tokens match. If they diverge, you have a cache bug, and the position of the first divergence tells you a lot.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def cache_equivalence_test(model, tokenizer, prompt, max_new_tokens=64):
    """Greedy-decode a prompt twice: with the KV-cache and with a full
    forward pass each step. Assert the two token sequences are identical.
    A divergence localizes a KV-cache / position-id / cache-reset bug."""
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids

    # --- Path A: cached greedy decode -------------------------------------
    with torch.no_grad():
        cached = model.generate(
            **enc,
            do_sample=False,            # greedy: deterministic, no sampling noise
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,             # the fast path under test
            pad_token_id=tokenizer.eos_token_id,
        )
    cached_new = cached[0, input_ids.shape[1]:]

    # --- Path B: uncached greedy decode (full forward every step) ---------
    seq = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(seq, use_cache=False)        # recompute everything
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    uncached_new = seq[0, input_ids.shape[1]:]

    # --- Compare ----------------------------------------------------------
    n = min(len(cached_new), len(uncached_new))
    a, b = cached_new[:n], uncached_new[:n]
    if torch.equal(a, b):
        print(f"PASS: cached == uncached for all {n} tokens")
        return True
    first_diff = int((a != b).float().argmax().item())
    print(f"FAIL: diverged at generated token {first_diff}")
    print("  cached:  ", tokenizer.decode(a[max(0, first_diff-2):first_diff+3]))
    print("  uncached:", tokenizer.decode(b[max(0, first_diff-2):first_diff+3]))
    return False
```

Three details make this test trustworthy. First, it uses **greedy decoding** (`do_sample=False`, `num_beams=1`), because sampling introduces its own randomness that would make the two paths diverge for reasons unrelated to the cache; greedy is deterministic, so any divergence is a real numerical difference. Second, it compares **tokens**, not logits — token equality is the property you actually care about, and it tolerates the tiny float differences that do not flip an argmax while catching the ones that do. Third, it reports the **position of the first divergence**, which is your localization signal: divergence at the very first generated token screams "position id bug"; divergence that appears only at long context length suggests cache-dtype rounding; identical-then-suddenly-different across sessions suggests a stale cache.

A stricter variant compares logits directly with a tolerance, which catches sub-argmax drift before it ever flips a token and is the version I run in CI:

```python
def cache_logit_equivalence(model, input_ids, atol=1e-2):
    """Compare cached vs full-forward logits at the boundary token.
    Tighter than token-equality: catches drift before it flips an argmax."""
    model.eval()
    with torch.no_grad():
        # Full forward over the whole sequence -> logits at the last position
        full = model(input_ids, use_cache=False).logits[:, -1, :]
        # Cached: prime on all-but-last, then feed the last token with cache
        primed = model(input_ids[:, :-1], use_cache=True)
        cached = model(input_ids[:, -1:], past_key_values=primed.past_key_values,
                       use_cache=True).logits[:, -1, :]
    max_abs = (full - cached).abs().max().item()
    print(f"max |full - cached| logit diff = {max_abs:.2e}", 
          "PASS" if max_abs < atol else "FAIL")
    return max_abs
```

#### Worked example: the position-id bug that broke generation at token 31

Here is the run from the intro, made concrete. A 1.3B-parameter decoder-only model finetuned on an instruction dataset, RoPE positional encoding, trained and eval'd in bf16. Eval loss 0.78, validation perplexity 2.18, both healthy. The eval loop computed loss with a single full forward pass over each gold sequence — `use_cache=False`, every position scored at once — so it never exercised the cache. Generation used `model.generate(..., use_cache=True)`, the cached path.

I ran the cache-equivalence test on five prompts. All five reported the same thing: `PASS` for the prompt tokens, then `FAIL: diverged at generated token 31`. Token-for-token, the cached and uncached decodes agreed for the prompt and the first 31 generated tokens, then split. The stricter logit test at the boundary read `max |full - cached| logit diff = 4.1e-01` — four-tenths of a logit, more than enough to flip an argmax on a borderline token. The divergence at a *generated* token (not in the prompt) and the clean prefix pointed straight at position bookkeeping. The cause: a custom generation wrapper someone had written to inject a stop-sequence check was passing `position_ids` that it recomputed from `attention_mask.cumsum(-1) - 1` *without accounting for left-padding offset*, so once the running position crossed a particular boundary the cached query got the wrong rotary phase. Replacing the hand-rolled wrapper with stock `generate` (which derives `cache_position` correctly from the cache length) made the test read `max |full - cached| logit diff = 3.0e-03` and `PASS: cached == uncached for all 64 tokens`. The generation went from "the the the" to coherent instruction-following on the same checkpoint, the same weights, the same loss. Figure 7 shows that before→after: the test went from a FAIL at token 31 to a PASS across the whole sequence, and that test flip is the proof the fix worked — not the eyeballed text, the assertion.

![Two-column before and after figure showing the cache equivalence test diverging at token 31 with a large logit difference before the fix and matching all 128 tokens within tolerance after the position ids are corrected](/imgs/blogs/train-infer-mismatch-for-llms-7.png)

The lesson worth carrying: the eval loss could not have caught this in a hundred years, because the eval loss does a full forward pass and the bug only exists in the cached path. The cache-equivalence test caught it in under a minute, told me it was a position bug from *where* it diverged, and gave me a FAIL→PASS proof of the fix. Run this test in CI on every model you serve.

## 3. Sampling vs greedy: when the decoding params are the bug

Sometimes the cache is perfect, the dtype matches, the template is right — and the model still generates garbage, or generates beautifully under greedy decoding and degenerately under sampling, or the reverse. This is the decoding-parameter mismatch, and it is insidious because the *model* is genuinely fine; the bug is in the numbers you pass to `generate`. The most common version: you evaluated the model with one decoding config and you serve it with another, so the thing you measured is not the thing you ship.

### The science: how temperature and truncation reshape the distribution

The model outputs a logit vector $z \in \mathbb{R}^{|V|}$ at each step; decoding turns it into a token. The base transformation is the softmax with **temperature** $\tau$:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}.
$$

Temperature reshapes the distribution before you sample. As $\tau \to 0$, the softmax becomes an argmax (greedy decoding); the distribution collapses onto the single highest-logit token. As $\tau \to \infty$, it flattens toward uniform; every token becomes equally likely. At $\tau = 1$ you sample from the model's native distribution. The crucial fact for debugging is that temperature is applied *to the logits*, so it interacts multiplicatively with the logit *scale* — and the logit scale is exactly what serve-time dtype and quantization perturb (Section 5). A model that is well-calibrated at $\tau = 0.7$ in bf16 can be over-sharp or over-flat at the same $\tau = 0.7$ if quantization has shifted the logit magnitudes, which is one way two mismatches conspire.

On top of temperature sit the **truncation** methods. **Top-k** keeps only the $k$ highest-probability tokens and renormalizes, zeroing the tail. **Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability exceeds $p$ and renormalizes; it adapts the cutoff to the shape of the distribution, keeping more tokens when the model is uncertain and fewer when it is confident. The reason these exist is the **degeneration** problem (Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration"): pure sampling from the full distribution at $\tau = 1$ occasionally draws from the long, low-probability tail, and those low-probability tokens are exactly the off-manifold tokens that trigger the exposure-bias cascade from Section 1. Truncation removes the tail, so a single unlucky draw cannot knock the generation off the manifold. Greedy and low-temperature sampling avoid the tail by construction, which is why a buggy serving config often "works greedy, breaks sampled" — greedy never touches the dangerous tail.

There is a second, subtler degeneration: greedy and low-temperature decoding *over-favor* high-probability continuations, which produces **repetition loops** — "the the the," or a phrase that repeats forever. The mechanism is a positive feedback loop in the model's own probabilities: once a phrase appears, the model assigns it high probability to continue (it is consistent with the prefix), so greedy picks it again, which makes it more consistent, which raises its probability further. Concretely, suppose the model has learned that a sentence like "Please let me know if you have any questions." is a high-probability ending; once it emits that sentence, the prefix now *contains* that sentence, and the model — which has seen that exact sentence followed by *another* copy of itself essentially never in clean data but which finds it locally the most probable continuation given a context that already ends in it — picks it again. Greedy decoding has no mechanism to escape this basin because it always takes the local maximum, and the local maximum is self-reinforcing. Sampling escapes it because it occasionally picks a non-maximal token that breaks the cycle, which is one reason "fine when sampled, loops when greedy" is such a common pairing. This is why **repetition penalty** (which divides the logit of already-emitted tokens by a factor $> 1$ before the softmax) and **no-repeat-ngram** constraints (which set the probability of any n-gram already seen to zero) exist: they explicitly break the positive feedback. The diagnostic consequence: a model that loops under greedy but is fine when sampled does *not* necessarily have a weight bug — it may just need a repetition penalty or a touch of temperature, and the "fix" is a decoding-config change, not a retrain. The trap is the reverse mistake too: a model that is *fine greedy* but degenerate when sampled with `top_p=1.0` does not have a weight bug either — it has an un-truncated tail, and the fix is to set a sane `top_p`, not to retrain. In both directions, the decoding config is the suspect, and the test is to vary one decoding parameter at a time and watch the symptom move.

### The diagnostic: pin the GenerationConfig to the eval config

Before the code, it helps to have the decoding parameters and their failure directions in one table, because the bug is almost always "this parameter was set to a value that does the wrong thing," and knowing which direction each knob fails in tells you what to reach for.

| Parameter | What it does | Too low / off | Too high / on |
| --- | --- | --- | --- |
| `temperature` $\tau$ | scales logits before softmax | $\to 0$: greedy, may loop | $\gg 1$: flat, incoherent |
| `top_p` (nucleus) | keep smallest set with cumulative prob $\ge p$ | small $p$: repetitive, safe | $p=1.0$: full tail, degenerates |
| `top_k` | keep $k$ highest-prob tokens | small $k$: repetitive | $k=0$/large: tail leaks in |
| `repetition_penalty` | divide emitted-token logits by factor | $1.0$: loops unsuppressed | $\gg 1.2$: avoids valid repeats |
| `eos_token_id` | token(s) that stop generation | wrong/unset: never stops | n/a |
| `max_new_tokens` | hard cap on generated length | too small: answer cut off | too large: wasted compute |

The fix for decoding skew is discipline, not cleverness: the decoding config you evaluate with must be the decoding config you serve with, and both must be version-controlled. Hugging Face has a first-class object for this, `GenerationConfig`, and the bug almost always comes from leaving it implicit and letting it default differently in two places.

```python
from transformers import GenerationConfig

# Pin ONE generation config; use it in eval AND in serving.
gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=0,                       # disabled; rely on nucleus
    repetition_penalty=1.1,        # mild; suppresses loops
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
gen_cfg.save_pretrained("artifacts/")     # ships with the model

# Audit: assert the served config equals the eval config, field by field.
def assert_gen_config_matches(eval_cfg, serve_cfg, fields=(
        "do_sample", "temperature", "top_p", "top_k",
        "repetition_penalty", "eos_token_id")):
    mismatches = []
    for f in fields:
        a, b = getattr(eval_cfg, f, None), getattr(serve_cfg, f, None)
        if a != b:
            mismatches.append((f, a, b))
    if mismatches:
        for f, a, b in mismatches:
            print(f"MISMATCH {f}: eval={a!r} serve={b!r}")
        raise AssertionError("eval and serve decoding configs differ")
    print("decoding config matches between eval and serve")
```

#### Worked example: greedy is fine, sampling loops, and the eval lied

A 7B chat model finetuned for summarization. The offline eval — ROUGE on a held-out set — ran greedy decoding (`do_sample=False`) and scored well: ROUGE-L 41.2, which the team signed off on. Production served with `do_sample=True, temperature=1.0, top_p=1.0` (effectively *no* truncation), because someone had copied a default config that never set `top_p`. In production, roughly one summary in eight degenerated into a loop or wandered off-topic — the long-tail draws the unbounded sampling let through, exactly the degeneration mechanism. The greedy eval could not see it because greedy never samples the tail.

The audit caught it immediately: `MISMATCH do_sample: eval=False serve=True`, `MISMATCH top_p: eval=1.0 serve=1.0` (and `temperature` 0.0-effective vs 1.0). Pinning the served config to `do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1` — and re-running the *offline* eval under that exact config so the number reflected reality — dropped the loop rate from about 12% to under 1% and brought the honest ROUGE-L to 39.8. The two-point ROUGE drop from 41.2 was not a regression; it was the *truth*. The greedy 41.2 had been an optimistic number for a config nobody served. The rule that falls out: **evaluate under the decoding config you will serve**, and pin it as an artifact so the two cannot drift. Figure 4 collects this mismatch alongside the others into the symptom-test-fix matrix I keep next to the keyboard.

![Matrix figure mapping KV-cache, dtype and quantization, decoding parameter, and template mismatches to their generation symptom, the confirming test, and the fix](/imgs/blogs/train-infer-mismatch-for-llms-4.png)

## 4. Padding and position drift at inference

This mismatch deserves its own section because it is the most counterintuitive: the *batch shape* of your inference can change the output. The same prompt, generated alone, gives a clean result; generated inside a batch with other prompts of different lengths, it gives a different — sometimes broken — result. The culprit is padding, and specifically the interaction of padding with position ids and attention masks in decoder-only models.

### The science: why left-padding is mandatory for decoder-only generation

Decoder-only generation appends tokens to the *right* of the prompt. To batch prompts of different lengths, you must pad them to a common length. If you pad on the **right** (the natural choice, and the correct one for *training* with a causal mask), then the newly generated tokens are appended after the padding, and the model is asked to continue from a sequence that ends in pad tokens. The position of the first real generated token is now wrong — it sits at the padded length, not the true prompt length — and the attention mask has to skip the pad tokens in the middle, which the position ids do not naturally account for. The model conditions on a prompt whose "end" is a run of pad tokens, which is nothing like anything it trained on. The result: right-padded batched generation produces garbage for the shorter sequences in the batch while the longest sequence (which has no padding) is fine. The fingerprint is unmistakable once you know it — *the longest item in the batch is correct and the shorter ones are broken*, and generating each item alone (batch size 1, no padding) fixes everything.

The correct convention for decoder-only generation is **left-padding**: pad on the left so that every prompt *ends* at the same right-hand boundary, the real tokens are flush against the position where generation begins, and the generated tokens continue from a true token rather than from padding. The position ids must then be set so the pad positions do not consume real position indices — in practice, `position_ids = attention_mask.cumsum(-1) - 1` clamped at the pad positions, or simply letting `generate` handle it when you set `tokenizer.padding_side = "left"`.

Walk the position arithmetic, because it is the crux. Take a batch of two prompts, lengths 3 and 5, padded to length 5. Under **left-padding** the short prompt becomes `[pad, pad, t1, t2, t3]` and its attention mask is `[0, 0, 1, 1, 1]`; computing `position_ids = attention_mask.cumsum(-1) - 1` gives `[-1, -1, 0, 1, 2]`, and clamping the pad positions to 0 (or 1) means the *real* tokens `t1, t2, t3` get positions `0, 1, 2` — exactly the positions they would have alone — and the first generated token lands at position 3, flush against `t3`, which is what the model trained on. Under **right-padding** the short prompt is `[t1, t2, t3, pad, pad]` with mask `[1, 1, 1, 0, 0]`; the real tokens get positions `0, 1, 2` correctly, but the *next* token — the first generated one — is appended after the padding at index 5, so its position is 5, not 3, and worse, it attends over a context whose immediate predecessors are pad tokens. The model is being asked to continue a sequence that, from its point of view, ends in two padding tokens at the wrong absolute position. That is a context the training distribution never contained, and the generation for that row is correspondingly off. The longest row in the batch (length 5, no padding) has no such offset, which is exactly why *it* stays correct while the shorter rows break — the canonical fingerprint. This is the single most common batched-generation bug, and the fix is two lines:

```python
# Decoder-only generation MUST use left padding.
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token        # if no pad token exists

# Now batched generation is correct: real tokens are flush-right,
# generation continues from a true token, position ids line up.
enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
out = model.generate(**enc, max_new_tokens=128, use_cache=True)
```

The deeper mechanics — how the attention mask interacts with position ids under packing, why a wrong mask leaks across documents, the cumsum position trick in full — are the subject of the [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms) post, and I will not re-derive them here. The point for *this* post is that padding is a train-inference mismatch: training pads right (correct under the causal mask, where the loss masks the pads anyway), generation must pad left, and if your serving code inherited the training tokenizer's `padding_side="right"` without changing it, your batched generation is silently wrong while your single-example generation is fine.

### The diagnostic: batch-invariance test

The test mirrors the cache-equivalence test in spirit: generation should be **invariant to batching**. Generate a prompt alone and again inside a batch of mixed-length prompts; assert the output for that prompt is identical.

```python
def batch_invariance_test(model, tokenizer, prompts, max_new_tokens=48):
    """Generate prompts[0] alone, then inside the full batch.
    With correct left-padding the two outputs must match. A mismatch
    localizes a padding-side / position-id / attention-mask bug."""
    model.eval()
    device = next(model.parameters()).device

    # Solo generation (no padding involved)
    solo_enc = tokenizer(prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        solo = model.generate(**solo_enc, do_sample=False,
                              max_new_tokens=max_new_tokens, use_cache=True)
    solo_new = solo[0, solo_enc.input_ids.shape[1]:]

    # Batched generation (prompts[0] is padded alongside the others)
    batch_enc = tokenizer(prompts, return_tensors="pt",
                          padding=True).to(device)
    with torch.no_grad():
        batched = model.generate(**batch_enc, do_sample=False,
                                 max_new_tokens=max_new_tokens, use_cache=True)
    # row 0, skip its (left) padding + prompt to find the new tokens
    row0_prompt_len = int(batch_enc.attention_mask[0].sum().item())
    pad_len = batch_enc.input_ids.shape[1] - row0_prompt_len
    batched_new = batched[0, pad_len + row0_prompt_len:]

    n = min(len(solo_new), len(batched_new))
    ok = torch.equal(solo_new[:n], batched_new[:n])
    print(f"padding_side={tokenizer.padding_side}: "
          f"{'PASS' if ok else 'FAIL'} batch invariance over {n} tokens")
    return ok
```

Run this with `padding_side="right"` and it FAILs for the shorter prompts; flip to `"left"` and it PASSes. That FAIL→PASS is the proof, and it is far more reliable than reading the generations and guessing.

## 5. Serve-time dtype and quantization shift

You trained in bf16. You serve in fp16, or int8, or 4-bit. The weights are "the same" — same checkpoint — but the *computation* is in a different number system, and that shifts the distribution the model samples from. This is a train-inference mismatch in the **numerics** corner, and it is increasingly common because serving cost pushes everyone toward quantized inference while training stays in bf16.

### The science: how precision perturbs the logit distribution

The relevant difference between bf16 and fp16 is not the total bit width (both are 16 bits) but the *split* between exponent and mantissa. bf16 has 8 exponent bits and 7 mantissa bits — the same dynamic range as fp32 but coarser precision. fp16 has 5 exponent bits and 10 mantissa bits — finer precision but a much smaller range, with a maximum representable value around $6.5\times10^4$ and a smallest normal around $6.1\times10^{-5}$. A model trained in bf16 may have activations or attention scores whose magnitude exceeds the fp16 range; served in fp16, those overflow to `inf`, and the softmax over `inf` produces `nan`, and you get garbage or empty generations. This is the same overflow story as the [mixed-precision debugging](/blog/machine-learning/edge-ai/quantization-from-first-principles) considerations, but at *serve* time it manifests as a generation bug rather than a training NaN.

Quantization to int8 or 4-bit is a different and more interesting perturbation. Quantization maps a floating-point weight $w$ to an integer via a scale $s$: $\hat{w} = s \cdot \text{round}(w / s)$, introducing a rounding error bounded by $s/2$ per weight. These per-weight errors are small, but they *accumulate* through the matrix multiplies and, critically, they perturb the **logits** — the pre-softmax scores. A small logit perturbation $\Delta z$ changes the sampled distribution by an amount that depends on temperature: at low temperature, where the softmax is sharp, a small logit shift can flip which token is the argmax; at the decision boundary between two near-tied tokens, a quantization error of a tenth of a logit is enough to change the greedy output. So quantization does not usually *destroy* the model — it *nudges* the distribution, and that nudge compounds through autoregressive generation exactly like any other small error. A model that is coherent in bf16 and slightly-off in int8 is the common case; a model that is garbage in int8 usually has an outlier-activation problem that the quantization scheme handled badly (which is its own deep topic — see [LLM activation quantization](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache)).

The quantitative way to see the shift is to compare the logit distributions directly. Compute the logits for a fixed input under both precisions and measure the divergence:

$$
D_{\text{KL}}\big(p_{\text{bf16}} \,\|\, p_{\text{int8}}\big) = \sum_i p_{\text{bf16},i} \log \frac{p_{\text{bf16},i}}{p_{\text{int8},i}}.
$$

A KL of $10^{-3}$ nats is negligible — the distributions are essentially identical and generation will match. A KL of $10^{-1}$ nats or more means the distributions have meaningfully diverged and you should expect different, possibly degraded, generation. This KL diff per dtype is the serve-time analogue of the cache-equivalence test, and it belongs in your release audit.

There is a sharper way to think about *when* a small logit perturbation actually changes a generation, because most of the time it does not. Under greedy decoding the only thing that matters is which token is the argmax, so a logit perturbation $\Delta z$ changes the output *only if it reorders the top two logits*. If the gap between the top logit and the runner-up at some step is $g = z_{(1)} - z_{(2)}$, then a perturbation flips the argmax only when it shrinks that gap below zero, which requires $|\Delta z| \gtrsim g$. Most steps have a comfortable gap — the model is confident, $g$ is several logits wide, and a quantization error of a tenth of a logit cannot touch it. The dangerous steps are the *near-ties*, where $g$ is small and the model is genuinely uncertain between two tokens; there, a tenth-of-a-logit quantization error flips the output. So the per-step probability that quantization changes the greedy token is roughly the probability that the model was at a near-tie, which is exactly the fraction of "hard" decisions in the generation. This is why quantization degradation is so context-dependent: a generation full of confident, low-entropy steps (boilerplate, formatting) is nearly immune, while a generation full of genuine choices (reasoning, creative continuation) accumulates flips. And because each flip can launch the exposure-bias cascade from Section 1, a handful of flipped near-ties early in a long generation can dissolve the whole thing — which is the mechanism behind "int8 is fine on short factual prompts and falls apart on long reasoning chains."

### The diagnostic: dtype/quant audit

```python
import torch.nn.functional as F

def dtype_logit_audit(model_ref, model_quant, tokenizer, prompt):
    """Compare next-token distributions of a reference (bf16) model and a
    quantized/lower-precision serving model on the same input. Reports
    max abs logit diff, argmax agreement, and KL(ref || quant)."""
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        zr = model_ref(**enc.to(next(model_ref.parameters()).device)
                       ).logits[:, -1, :].float().cpu()
        zq = model_quant(**enc.to(next(model_quant.parameters()).device)
                         ).logits[:, -1, :].float().cpu()
    max_abs = (zr - zq).abs().max().item()
    argmax_same = (zr.argmax(-1) == zq.argmax(-1)).item()
    pr, pq = F.softmax(zr, -1), F.softmax(zq, -1)
    kl = (pr * (pr.clamp_min(1e-9).log() - pq.clamp_min(1e-9).log())).sum().item()
    print(f"max|dz|={max_abs:.3f}  argmax_match={argmax_same}  "
          f"KL(ref||quant)={kl:.4f} nats")
    return kl
```

#### Worked example: bf16 train, fp16 serve, empty generations

A model trained in bf16, deployed on hardware where the serving stack defaulted to fp16. Training loss healthy, eval (also run in bf16) healthy. In production, about 4% of requests returned empty or truncated generations, and logging showed `nan` in the logits for exactly those requests. The dtype audit on a triggering prompt read `max|dz|=inf` — the fp16 forward had overflowed. The cause: a handful of attention-score magnitudes in this model exceeded the fp16 max (~$6.5\times10^4$) on certain long inputs, overflowing to `inf`, and the softmax over `inf` produced `nan`, which propagated to an empty generation. Switching the serving dtype to bf16 (same range as training) eliminated the `nan`s entirely; the empty-generation rate went from 4% to 0%, with `max|dz|=2e-3` and `KL=3e-4` between the new bf16-serve and the bf16-train reference — statistically identical. The lesson: **serve in the dtype you trained in unless you have measured the divergence and decided you can tolerate it.** If cost forces quantization, audit the KL and the argmax-agreement on a probe set before you ship, and calibrate or use an outlier-aware scheme if the divergence is large. Figure 6 stacks the serving layers where this skew enters — the dtype cast is the bottom of the stack, right below the trained checkpoint, and a wrong cast poisons everything above it.

![Vertical stack figure showing the layers between a trained bf16 checkpoint and the emitted token where serve-time skew enters: dtype cast, chat template, KV-cache decode, and decoding parameters](/imgs/blogs/train-infer-mismatch-for-llms-6.png)

## 6. Chat-template and format skew at inference

If you finetuned a chat or instruct model, the model learned to respond to a *specific format* — a particular arrangement of role markers, special tokens, and a generation prompt. The most common and most maddening train-inference mismatch in instruct-finetuning is serving the model with a *different* format than it trained on. The model sees a prompt string it has never seen the shape of, and it responds out of distribution: it ignores the instruction, never stops, or rambles.

### The science: the format is part of the conditioning

During chat finetuning the training example is not the raw user message; it is the message wrapped in a template, for example (using a ChatML-like format):

```bash
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Summarize this article.<|im_end|>
<|im_start|>assistant
```

The model learns $p_\theta(\text{response} \mid \text{this exact wrapped string})$. The role tokens (`<|im_start|>`, `<|im_end|>`), the role names, the newlines, and crucially the trailing **generation prompt** (`<|im_start|>assistant\n` with nothing after it, signaling "your turn") are all part of the conditioning context. If at inference you feed the raw "Summarize this article." with no template, or with a *different* template (the Llama format instead of the ChatML format, say), you are conditioning the model on a string from a distribution it never saw, and the response distribution is correspondingly off. The two classic failure modes: (1) the model **ignores the instruction** or behaves like a base model, because the instruct-conditioning tokens are absent; (2) the model **never stops** — it does not emit EOS — because the `<|im_end|>` it learned to produce at the end of a turn is part of a template the inference path mangled, so the learned stopping behavior never triggers.

The fix is to use the model's *own* chat template for both training and inference, never to hand-format. Hugging Face tokenizers carry the template with the model:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this article."},
]
# add_generation_prompt=True appends the trailing assistant marker,
# the exact token sequence the model learned to continue from.
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
enc = tokenizer(prompt, return_tensors="pt").to(device)
out = model.generate(**enc, max_new_tokens=512, use_cache=True,
                     eos_token_id=tokenizer.eos_token_id)
```

The template skew is covered in depth in the [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs) post; the point here is that it is one face of the train-inference mismatch — the conditioning string differs between training and serving — and the diagnostic is the same as everything else in this post: **diff the exact string the model sees in training against the exact string it sees in serving.**

### The diagnostic: diff the train string against the serve string

```python
def template_skew_audit(tokenizer, messages, train_formatter):
    """Diff the exact prompt string used in training against the one
    apply_chat_template produces for serving. Any difference is the bug."""
    serve_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    train_str = train_formatter(messages)        # how training built the string
    if serve_str == train_str:
        print("PASS: train and serve prompt strings are identical")
        return True
    print("FAIL: prompt strings differ")
    # show the first divergent character region
    for i, (a, b) in enumerate(zip(train_str, serve_str)):
        if a != b:
            lo = max(0, i - 20)
            print(f"  first diff at char {i}:")
            print(f"    train: ...{train_str[lo:i+20]!r}")
            print(f"    serve: ...{serve_str[lo:i+20]!r}")
            break
    return False
```

### EOS, stopping criteria, and max_new_tokens

A closely related mismatch is **stopping**, and it deserves care because the same symptom — a model that never stops — has two completely different root causes that live on opposite sides of the train-inference boundary. If the loss masked or never saw the EOS token (a [loss-masking bug](/blog/machine-learning/debugging-training/loss-function-bugs)), or the template at inference does not present EOS where the model learned to emit it, the model never learns to stop and generation runs to `max_new_tokens` every time. The symptom is a model that produces a good answer and then *keeps going* — answering, then starting a new fabricated turn, then another. The confirming test is a clean two-way split. Decode a few generations *without* a stopping criterion (set the stop tokens aside) and inspect the raw token ids. **Case A: the EOS token appears in the raw ids but generation did not stop.** Then the model learned to emit EOS correctly and the bug is purely on the inference side — your `eos_token_id` in the `GenerationConfig` is wrong, unset, or points at a different token than the one the model emits (a common cause: the training template used a custom end-of-turn token like `<|im_end|>` as the *de facto* stop, but the `GenerationConfig` lists only the tokenizer's nominal `eos_token`, so the model dutifully emits `<|im_end|>` and generation sails right past it). The fix is to add the right stop token(s) to the generation config — a one-line inference fix, no retrain. **Case B: the EOS token never appears in the raw ids.** Then the model genuinely did not learn to stop, and that is an *upstream training* bug — the loss masked the EOS position, or the dataset never included the end-of-turn token in the labels, or sequence packing bled the stop token into a masked region. No amount of inference configuration fixes Case B; you have to fix the data/masking and retrain. This single test — *does EOS appear in the raw ids?* — routes you to an inference fix (Case A) or a training fix (Case B), which is exactly the kind of boundary the whole post is about. A too-small `max_new_tokens` is the opposite failure — the answer is cut off mid-sentence — and is the easiest of all to spot and fix; the tell is that the generation ends abruptly at *exactly* the same token count every time, which a content-driven EOS stop would never do.

## 7. Batched vs single generation, and putting it together

A few of the mismatches above interact specifically when you batch, beyond the padding issue of Section 4. Batched generation shares a `GenerationConfig` and a stopping policy across the batch, but sequences finish at different times; a correctly implemented `generate` keeps decoding finished sequences as padding and masks them out, but a hand-rolled loop that does not can let a finished sequence's pad tokens bleed into the still-running ones, or can apply the wrong attention mask as sequences complete. There is a second batching subtlety worth flagging: even with everything correct, batched and single-sequence generation can differ by *floating-point reduction order*. A matrix multiply over a batch may accumulate in a different order than the same multiply for a single row, and bf16/fp16 addition is not associative, so the logits can differ in the last bit or two. Almost always this is below the argmax-flip threshold and the tokens are identical — but on a genuine near-tie (recall the gap argument from Section 5), a last-bit difference can flip one token, and the exposure-bias cascade does the rest. This is the rare case where solo-vs-batch divergence is *not* a bug but a numerical artifact; you distinguish it from a real bug by its signature — a real padding/masking bug breaks the shorter rows systematically and grossly, while a reduction-order artifact flips at most an occasional near-tie token and only on a tiny fraction of prompts. The batch-invariance test from Section 4 is the catch-all here: if solo and batched generation agree token-for-token under greedy decoding, your batching is correct; if they diverge grossly and systematically for the shorter rows, you have a padding, masking, or finished-sequence-handling bug, and the position of divergence localizes it.

Now bisect a real broken generation end to end, because the individual tests are most powerful when sequenced. Figure 5 is the decision tree I walk, and it is the same bisection discipline the whole series preaches — narrow the suspect before you touch code.

![Decision tree figure that bisects a broken generation by asking whether greedy decode is clean, then whether cached equals uncached, then whether dtype and template match, localizing to one of the mismatch causes](/imgs/blogs/train-infer-mismatch-for-llms-5.png)

**Step 1 — is greedy clean?** Generate greedily (`do_sample=False`). If greedy is *also* broken, the problem is not sampling parameters — it is a deterministic mismatch (cache, dtype, template, or a genuine weight problem), so move on. If greedy is clean and only *sampling* breaks, the bug is in your decoding params (temperature/top_p/repetition_penalty), which is Section 3; pin the config and stop.

**Step 2 — does cached equal uncached?** Run the cache-equivalence test from Section 2. If it FAILs, you have a KV-cache or position-id bug; the divergence position tells you which. Fix and re-run the test to PASS. This step alone resolves the majority of "great loss, broken greedy generation" cases I have seen.

**Step 3 — does the serve-time dtype and template match training?** Run the dtype audit (Section 5) and the template-skew audit (Section 6). A large KL or argmax disagreement points at dtype/quant; a string diff points at template skew. Fix whichever fires.

**Step 4 — only now consider exposure bias or the eval set.** If greedy is clean, the cache is equivalent, the dtype matches, and the template matches, *and* generation is still subtly worse than the loss suggested, then you may genuinely be looking at exposure bias or an eval set that does not reflect generation. This is the rarest case and the only one that is not a mechanical bug; the remedy is generate-during-training monitoring (Section 8) and, if it is truly exposure bias, training-time interventions like scheduled sampling or sequence-level objectives — but verify the four mechanical causes first, because they are the cause 90+% of the time.

Here is the whole bisection as a lookup table — the one I keep open while debugging a generation, mapping each mismatch to the symptom that betrays it, the test that confirms it, and the fix that resolves it.

| Mismatch | Generation symptom | Confirming test | Fix |
| --- | --- | --- | --- |
| KV-cache / position id | clean prefix, garbage after token $N$ | cache-equivalence test FAILs at $N$ | correct `cache_position`; fresh cache per seq |
| Cache dtype | divergence grows with context length | logit diff rises with length | raise KV-cache precision |
| Decoding params | loops greedy, or random sampled | vary one param, watch symptom move | pin `GenerationConfig` to eval config |
| Left vs right padding | shorter batch rows broken, longest fine | batch-invariance test FAILs | `tokenizer.padding_side = "left"` |
| Serve-time dtype | fine bf16, `nan`/degraded fp16/int8 | logit KL per dtype, argmax disagree | serve train dtype, or calibrate quant |
| Chat-template skew | ignores instruction or never stops | diff train string vs serve string | use registered chat template both sides |
| EOS / stopping | answers then runs to `max_new_tokens` | does EOS appear in raw ids? | fix `eos_token_id` (A) or retrain (B) |

The table is also a statement of the post's thesis in tabular form: every row is a place where the inference computation can differ from what training optimized, every symptom is a way that difference shows up in the *generation* (never in the loss), and every test is a deterministic check that isolates one row. Work top to bottom on a broken generation and you will localize it.

#### Worked example: bisecting a finetune that "forgot how to talk"

A team brought me a Llama-style chat model finetuned on a customer-support dataset. Loss 0.71, validation loss 0.74, both clean. Generation: coherent for two sentences, then it would start a new "Customer:" turn and answer its own question, forever, until `max_new_tokens`. They had spent two days assuming catastrophic forgetting and were about to retrain with a lower learning rate.

I walked the tree. Step 1: greedy was *also* broken (same self-conversation), so it was not sampling. Step 2: the cache-equivalence test PASSed for 64 tokens — the cache was fine, not a position bug. Step 3: the template-skew audit FAILed instantly — training had wrapped examples with a hand-written f-string that used `\n\n### Assistant:\n` as the turn marker, but serving used `tokenizer.apply_chat_template`, which emitted the model's *registered* Llama chat format with `[/INST]` markers. The model had learned to stop at the *hand-written* marker and to emit it to start a new turn; served with the registered template, it never saw the marker it learned, so it never stopped and kept generating turns in its *training* format. The fix was not retraining — it was registering the hand-written format as the tokenizer's chat template (and, going forward, training with the registered template so the two could not drift). With the matching template, the model stopped correctly after one turn; the self-conversation vanished. Two days of "it forgot everything" were a thirty-second string diff. This is the entire thesis of the post: **the loss was honest, the inference path was wrong, and a diff localized it.**

## 8. The generate-during-training callback: don't trust the loss alone

Every diagnostic so far is run *after* training, at serving time. The deeper fix is to never let the loss be your only signal in the first place. The discipline is simple and high-leverage: **decode a fixed probe set during training and read the generations**, every few hundred steps, with the exact serving config. This surfaces exposure-bias drift, template problems, looping, and EOS failures while you can still act, instead of at release.

```python
from transformers import TrainerCallback

class GenerateProbeCallback(TrainerCallback):
    """Every `every_n_steps`, free-running-decode a fixed probe set with the
    SERVING generation config and log the results. Catches train-inference
    mismatch (loops, no-EOS, drift) that the falling loss cannot show."""
    def __init__(self, tokenizer, probe_prompts, gen_cfg, every_n_steps=500):
        self.tok = tokenizer
        self.probes = probe_prompts          # frozen at run start
        self.gen_cfg = gen_cfg               # the SERVING config
        self.every = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        model.eval()
        device = next(model.parameters()).device
        for p in self.probes:
            msgs = [{"role": "user", "content": p}]
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, generation_config=self.gen_cfg)
            gen = self.tok.decode(out[0, enc.input_ids.shape[1]:],
                                  skip_special_tokens=False)
            emitted_eos = self.tok.eos_token in gen
            print(f"[step {state.global_step}] eos={emitted_eos} | {gen[:120]!r}")
        model.train()                        # CRITICAL: restore train mode
```

Three things make this callback worth its cost. It uses the **serving generation config**, so it exercises the real inference path, not a teacher-forced one. It logs whether **EOS was emitted**, which catches the never-stops bug the instant the model regresses on it. And it restores `model.train()` at the end — forgetting that is its own [train/eval-mode bug](/blog/machine-learning/debugging-training/train-eval-mode-bugs) that would freeze BatchNorm-style stateful behavior and quietly degrade the rest of training. Figure 8 shows what this callback buys you: the loss is falling the whole time, but the probe set reveals loops starting at step 1500 and a no-EOS regression at step 2500 that the loss curve cannot show, and the recovery once the template and cache are fixed.

![Timeline figure showing a probe set decoded at increasing training steps where the loss keeps falling while the generations develop loops then a no-EOS failure before a template and cache fix lets the probe recover](/imgs/blogs/train-infer-mismatch-for-llms-8.png)

#### Worked example: the probe set that caught a regression 1,000 steps early

A 3B instruct finetune, probe set of 12 fixed prompts, decoded every 500 steps with the serving config. The loss fell monotonically from 1.6 to 0.7 across 5,000 steps — textbook. The probe log told a different story. At step 500 (loss 1.4) all 12 probes generated clean, EOS-terminated answers. At step 1500 (loss 1.0) three probes started repeating their final sentence — early looping. At step 2500 (loss 0.8) five probes ran to `max_new_tokens` with `eos=False` — the model was *unlearning* to stop, almost certainly because a sequence-packing change had started bleeding EOS handling across documents. The loss did not so much as twitch at any of these. Catching the no-EOS regression at step 2500 instead of at release saved a full retrain: the packing config was fixed, training resumed, and by step 3500 all 12 probes were clean and EOS-terminated again with the loss continuing down to 0.7. Without the probe callback, this ships, and someone discovers it in production a week later. **The loss is a necessary signal and an insufficient one; the probe set is what makes the generation path observable during training.**

## Case studies and real signatures

Four named patterns, each a real or well-documented signature of a train-inference mismatch.

**The left-padding-breaks-generation bug.** This is so common in the Hugging Face ecosystem that it is essentially folklore, and the library warns about it explicitly: decoder-only models default `padding_side` to `"right"` because that is correct for training, and batched generation with right padding produces garbage for shorter sequences while the longest sequence in the batch is fine. The fingerprint — *longest item correct, shorter items broken, batch-size-1 fixes it* — is the canonical example of a batch-shape-dependent generation bug, and the fix is the one-liner `tokenizer.padding_side = "left"`. It is the cleanest demonstration that *batching* is part of the inference computation, not a neutral wrapper around it.

**Exposure bias and scheduled sampling.** The train-inference gap was named and studied in the sequence-modeling literature: Bengio et al. (2015) introduced **scheduled sampling**, which during training occasionally feeds the model its own predictions instead of the gold token, annealing from fully teacher-forced toward partially free-running, precisely to reduce exposure bias. Ranzato et al. (2016) framed it as a train-test objective mismatch and proposed sequence-level training (MIXER). The practical takeaway for modern LLMs is nuanced: large pretrained models suffer *less* from pure exposure bias than the early RNN sequence models did, because pretraining exposes them to such diversity that their own minor errors usually stay on-manifold — which is exactly why, when a modern finetune generates garbage, you should suspect a mechanical mismatch (cache/dtype/template) *before* you blame exposure bias. The science is real; the diagnosis order matters.

**Nucleus sampling and neural text degeneration.** Holtzman et al. (2020) documented that maximum-likelihood-trained LLMs produce degenerate text (repetition, blandness) under naive decoding, and that the fix is at *decoding* time — top-p (nucleus) sampling truncates the unreliable tail. The relevance to this post: a generation that loops or degenerates is frequently a *decoding-config* bug, not a weights bug, and the historical record is that the field solved a large fraction of "the generations are bad" by changing the decoder, not the model. When your loss is fine and the generation degenerates, reach for the decoding config before you reach for a retrain.

**Quantization-induced distribution shift at serve time.** The mixed-precision and quantization literature (Micikevicius et al. 2018 for the fp16 range and loss-scaling mechanics; the weight-only and activation quantization work behind GPTQ/AWQ/SmoothQuant) establishes that lower-precision *inference* perturbs the logit distribution, and that the perturbation is tolerable when bounded and catastrophic when it isn't (fp16 overflow → `nan`; large-outlier activations quantized badly → distribution collapse). The serve-time fingerprint — fine in bf16, degraded or `nan` in fp16/int8, with the divergence measurable as a logit KL — is a numerics-corner mismatch, and the audit (compare logit distributions per dtype before shipping) is the standard defense.

## When this is (and isn't) your bug

A decisive section, because the whole skill is pointing the diagnosis at the right corner.

**It is a train-inference mismatch when:** the loss and validation loss are healthy and stable, there is no NaN during training, and the model nonetheless generates loops, drift, garbage, or never stops — *and* the badness depends on the inference path (greedy vs sampled, cached vs uncached, batched vs solo, one dtype vs another, one template vs another). The tell is that *changing how you generate* changes the output quality without touching the weights. If flipping `padding_side`, or fixing `cache_position`, or matching the serving dtype, or pinning the decoding config, or applying the right chat template moves the generation from broken to clean, the weights were always fine and the bug was in the inference path.

**It is NOT a train-inference mismatch (look elsewhere) when:** the *loss itself* is wrong — diverging, NaN, or plateaued at chance. A loss that never came down means the model genuinely did not learn, and no amount of inference fiddling will help; that is an [optimization or data bug](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), not a mismatch. Likewise, if the cache-equivalence test PASSes, the dtype audit shows a tiny KL, the template diff is identical, and greedy and sampled both generate the same garbage, then the inference path is faithful and the model truly produces that distribution — the problem is upstream in training (wrong loss masking, training on the prompt, a data problem). The diagnostic boundary is sharp: **if greedy-uncached-bf16-correct-template generation is also broken, stop blaming inference and go back to the training run.** Every test in this post is, at bottom, a way to decide which side of that boundary you are on.

A second boundary worth naming: do not confuse *exposure bias* (a genuine training-objective limitation) with an *inference bug* (a mechanical mismatch). Exposure bias is what is left after every mechanical mismatch is ruled out — it is the residual gap when the inference path is provably faithful and the model still can't quite hold a long generation together. In modern LLM finetuning that residual is small and rare; the mechanical mismatches are common. Spend your debugging budget on the cache, the dtype, the template, and the decoding config first.

## Key takeaways

- **A great loss with broken generation is the expected signature of a train-inference mismatch, not a contradiction.** Teacher-forced cross-entropy scores per-step error on *gold* prefixes; generation walks the model's *own* prefix distribution. The loss is structurally blind to the gap.
- **The cache-equivalence test is your highest-leverage diagnostic.** Greedy-decode with and without the cache and assert the tokens match. A FAIL localizes a KV-cache / position-id / cache-reset bug; the divergence position tells you which. Run it in CI.
- **Divergence at the first generated token is a position-id bug; divergence that grows with context is a cache-dtype bug; divergence that depends on session history is a stale cache.** Read the *where*, not just the *whether*.
- **Evaluate under the decoding config you serve, and pin it as an artifact.** Greedy eval + sampled serve is a lie; the offline number must reflect the served config (`do_sample`, `temperature`, `top_p`, `repetition_penalty`, `eos_token_id`).
- **Decoder-only generation requires left padding.** Right padding (correct for training) breaks batched generation for the shorter sequences while the longest is fine. The fix is `tokenizer.padding_side = "left"`. Confirm with a batch-invariance test.
- **Serve in the dtype you trained in unless you measured the divergence.** fp16 can overflow a bf16-trained model to `nan`; int8/4-bit nudges the logits. Audit the per-dtype logit KL and argmax-agreement on a probe set before shipping.
- **Use the model's registered chat template for both training and inference.** Template skew makes the model ignore instructions or never stop. Diff the exact training string against the exact serving string; any difference is the bug.
- **Decode a fixed probe set during training with the serving config.** The loss cannot show loops, drift, or a no-EOS regression; a generate-during-training callback can, hundreds of steps before release.
- **Bisect before you touch code:** greedy clean? → cached == uncached? → dtype/template match? Only then suspect exposure bias. The mechanical mismatches are the cause 90+% of the time.

## Further reading

- Bengio, Vinyals, Jaitly, Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (2015) — the original exposure-bias intervention.
- Ranzato, Chopra, Auli, Zaremba, "Sequence Level Training with Recurrent Neural Networks" (MIXER, 2016) — frames the train-test objective mismatch precisely.
- Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration" (2020) — nucleus (top-p) sampling and why naive decoding degenerates.
- Micikevicius et al., "Mixed Precision Training" (2018) — the fp16 representable range and loss-scaling mechanics behind serve-time dtype overflow.
- Hugging Face `transformers` documentation on `generate`, `GenerationConfig`, KV-cache (`use_cache`, `past_key_values`, `cache_position`), and `padding_side` for decoder-only generation.
- Within this series: the master decision tree, [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the capstone [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook); siblings [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms), [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs), [train vs eval mode bugs](/blog/machine-learning/debugging-training/train-eval-mode-bugs), and [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs).
