---
title: "Attention and Masking Bugs: The Future-Token Peek and Other Leaks"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the masking bugs that make a decoder cheat on train loss and collapse on generation, using the science of softmax masks and a future-invariance test that turns suspicion into a binary signal."
tags:
  [
    "debugging",
    "model-training",
    "attention",
    "masking",
    "llm",
    "nlp",
    "transformers",
    "finetuning",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/attention-and-masking-bugs-1.png"
---

The run looked like a miracle. You were finetuning a small decoder-only language model on a few hundred million tokens of clean text, you had set up the loss masking carefully, and after the first epoch your train loss had dropped to `0.05`. For a next-token language model, a cross-entropy of `0.05` means the model is assigning roughly $e^{-0.05} \approx 95\%$ probability to the correct next token, every token, on average. That is a perplexity of about `1.05`. No language model in the world has a perplexity of `1.05` on natural text; the entropy of English is famously around one bit per character, and even the best frontier models sit at a per-token perplexity well above two on held-out text. A perplexity near one is not a great model. It is a model that already knows the answer. And there is exactly one way a next-token model can already know the answer: it is reading it.

When you ran the same checkpoint on the validation set, the loss came back `8.1` — worse than random for a vocabulary of fifty thousand tokens would be about `10.8`, so the model had learned *something*, but the train–val gap was a chasm, not a gap. When you asked it to generate, it produced fluent-looking garbage that drifted off topic within ten tokens and then degenerated into repetition. You had the signature of one of the most insidious bug classes in deep learning: an **attention masking bug**. Somewhere in the path from raw attention scores to the softmax, a position in the sequence was allowed to look at a token it should not have been able to see — most likely the very next token, the one it is being trained to predict. The model was not learning to predict the future. It was copying it.

![A dataflow graph showing query times key scores being scaled, then a mask added before softmax, with a correct path producing honest attention weights and an off-by-one path producing leaked weights that drive train loss too low](/imgs/blogs/attention-and-masking-bugs-1.png)

This post is about that whole family of bugs: the future-token peek, the missing causal mask, the padding mask that is wrong or absent, the boolean-versus-additive confusion that silently inverts which tokens you keep, the mask that does not broadcast across heads, the all-masked row that detonates into a NaN, and the attention dropout you forgot to turn off at eval time. They live in the **model code** place of the six places a training bug can hide — data, optimization, model code, numerics, systems, evaluation — but they masquerade as every other place. A masking leak looks like a too-good optimizer. An all-masked row looks like a numerics bug. A padding leak looks like a data bug. The art is to recognize the signature, run the one test that isolates the cause, and fix it before you waste another GPU-day. By the end you will be able to take any decoder whose train loss looks suspiciously good or whose generation is broken, prove in under a minute whether it is peeking at the future, and install a unit test that makes the bug impossible to ship. Let us start with the science, because you cannot diagnose a mask without knowing exactly where it enters and what it does.

## 1. Scaled dot-product attention, and the one place the mask lives

Attention, stripped to its arithmetic, is a weighted average of value vectors where the weights come from a softmax over similarity scores. For a single query position attending over $T$ key positions, with queries $Q$, keys $K$, and values $V$ each of dimension $d_k$, the operation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

That $M$ — the additive mask — is the entire subject of this post. Everything else in the formula is benign. The query-key dot product $QK^\top$ produces a $T \times T$ matrix of raw similarity scores where entry $(i, j)$ is how much query position $i$ wants to attend to key position $j$. The scaling by $1/\sqrt{d_k}$ keeps the dot products from growing with dimension and saturating the softmax — that is a numerics safeguard, not a correctness one. The softmax turns each row of scores into a probability distribution over keys. The final matmul with $V$ takes the weighted average. None of those steps decide *which positions a query is allowed to see*. Only $M$ does.

The mask $M$ is a $T \times T$ matrix (or it broadcasts to one) added to the scores *before* the softmax. Its job is to make forbidden positions contribute zero weight. It does this by setting forbidden entries to $-\infty$ (in practice a large negative number like `-1e9` or `torch.finfo(dtype).min`) and allowed entries to `0`. After adding $M$, a forbidden score is $-\infty$, and $\text{softmax}$ sends $e^{-\infty} = 0$, so that key gets exactly zero weight. An allowed score is unchanged because we added zero. This is the key mechanical fact: **the mask is additive, it is applied once, immediately before the softmax, and the only two values that belong in it are `0` (keep) and `-inf` (block).** Every masking bug is some violation of that sentence — the wrong entry is `-inf`, the wrong entry is `0`, the values are `1`/`0` instead of `0`/`-inf`, the matrix has the wrong shape, or it is missing entirely.

Let us be fully explicit about why $-\infty$ is the right value to inject, because the choice is not arbitrary and the arithmetic of *almost* $-\infty$ is where several bugs live. After adding $M$, a single row of the masked scores is a vector $s = (s_1, \ldots, s_T)$ where allowed entries are the real (finite) scores and blocked entries are $-\infty$. The softmax of that row is

$$
\text{softmax}(s)_j = \frac{e^{s_j}}{\sum_{k=1}^{T} e^{s_k}}.
$$

For a blocked entry, $s_j = -\infty$ gives $e^{s_j} = e^{-\infty} = 0$ exactly, so that key receives exactly zero weight *and* it contributes exactly zero to the denominator. That second part is the subtle one: the mask does not merely zero the forbidden weight, it removes the forbidden key from the *normalization* entirely, so the remaining allowed weights renormalize to sum to one among themselves. That is the behavior we want — the attention distribution is a proper probability distribution over only the allowed keys.

Now watch what happens with the common buggy substitutes for $-\infty$. If you use a large negative number like `-1e9` in fp32, $e^{-10^9}$ underflows to `0`, so it behaves like true $-\infty$ and you are fine. But if your block value is only `-1e4` or `-100` — a real mistake people make to "avoid `inf`" — then the arithmetic matters. We have $e^{-100} \approx 3.7 \times 10^{-44}$, still effectively zero, but $e^{-10} \approx 4.5 \times 10^{-5}$ is *not* zero, and $e^{-1} \approx 0.37$ is not even small. A block value that is merely "somewhat negative" leaks a nonzero weight to the forbidden key, and because softmax normalizes, even a tiny leaked weight steals mass from the allowed keys and shifts every output vector. The rule that falls out of the math is sharp: **your block value must be negative enough that $e^{\text{block}}$ underflows to zero in your working dtype.** `torch.finfo(dtype).min` is the safe choice because it is the most-negative representable value and guarantees underflow; in fp16 that is about `-65504`, in bf16 about `-3.4e38`. Using `-1` or `-100` as the block value is the additive-magnitude bug we return to in Section 8, and the math above is exactly why it corrupts quietly instead of failing loudly.

A second piece of the science worth making rigorous is the *normalization pollution* from an unmasked pad key, because it explains why a padding bug shifts every output rather than just the pad position. Suppose a query attends over four real keys with raw scores $(2, 1, 0, 1)$ and one pad key that, unmasked, has score $1.5$. Without masking the pad, the softmax denominator is $e^2 + e^1 + e^0 + e^1 + e^{1.5} \approx 7.39 + 2.72 + 1.00 + 2.72 + 4.48 = 18.31$, and the pad key alone soaks up $4.48/18.31 \approx 24\%$ of the attention. That `24%` is not just wasted on a meaningless token — it is *subtracted* from the real keys, which now share only `76%` of the distribution. The output vector, a weighted sum of values, is therefore pulled `24%` of the way toward the pad token's value vector, which is garbage. Mask the pad (set its score to $-\infty$), and the denominator drops to $13.83$, the real keys reclaim all the mass, and the output is correct. The size of the corruption scales with how much padding you added, which is precisely why the symptom of a padding bug is *batch-dependence*: more padding means more stolen mass means a more corrupted output. We will turn that into a test in Section 7; for now, note that the math predicts the symptom before you instrument anything.

There are two conceptually distinct masks that both enter at this same point, and conflating them is itself a common bug. The **causal mask** enforces autoregression: in a decoder-only language model, query position $t$ may attend to key positions $0, 1, \ldots, t$ but not to $t+1, t+2, \ldots$, because at generation time those future tokens do not exist yet. The causal mask is purely a function of position; it is the same lower-triangular pattern for every sequence in the batch. The **padding mask** enforces that real tokens never attend to padding tokens — the filler tokens you add to make variable-length sequences into a rectangular batch. The padding mask depends on the actual data: which positions are padding differs from sequence to sequence. The full mask added before softmax is the union of both: a key is blocked if it is in the future (causal) *or* it is padding (pad). Get the union wrong and you have either a leak or a NaN.

Why does the leak hurt so specifically? Consider the next-token training objective. At each position $t$, the model produces a hidden state $h_t$, projects it to logits over the vocabulary, and is trained so that $\text{softmax}(\text{logits}_t)$ puts mass on the actual token at position $t+1$. The label for position $t$ is the input token at position $t+1$. Now suppose the causal mask is off by one and position $t$ is allowed to attend to position $t+1$. Then $h_t$ — the very vector used to predict token $t+1$ — has *read* token $t+1$. The model can learn the trivial function "copy the embedding of the token you just attended to into the logits," and it will, because gradient descent finds the easiest loss-reducing function and copying is far easier than modeling language. Train loss plummets toward zero. But at inference time, when you generate token by token, token $t+1$ does not exist when you are computing $h_t$ — that is the whole point of generation — so the model has nothing to copy and produces nonsense. The leak is a free lunch that exists only during training. That asymmetry between training and inference is the defining signature of this entire bug class, and it is why the figure above splits the post-softmax path into an honest branch and a leaked branch.

## 2. The bug taxonomy, with the mechanism for each

There are seven masking bugs that account for nearly every attention failure I have seen, and they are worth enumerating precisely because each one has a distinct mechanism and therefore a distinct confirming test. The taxonomy below is the map; the rest of the post fills it in. Notice that they sort cleanly by symptom: three of them make train loss too low, one detonates into a NaN, and two break generation while leaving train loss looking fine.

![A decision tree branching from an attention symptom on a decoder into too-low train loss, NaN at step one, and broken generation, each leading to its specific cause and confirming test](/imgs/blogs/attention-and-masking-bugs-3.png)

**Bug 1: causal-mask off-by-one.** The mask is almost right but shifted by one position, so query $t$ can see key $t+1$. This is the future-token peek. Mechanically, the model learns to copy the next token; train loss collapses toward zero, perplexity approaches one, and generation is broken because the future is unavailable at inference. This is the single most common and most damaging masking bug because the off-by-one is so easy to introduce: a `>` where you needed `>=`, a `triu` with the wrong `diagonal` argument, a shift applied to the wrong tensor.

**Bug 2: no causal mask at all.** A decoder built without any causal mask is fully bidirectional — every position sees every other position, including all future tokens. This is total leakage, even worse than the off-by-one. The signature is the same (train loss near zero, generation broken) but more extreme, and it is common when someone copies a `nn.TransformerEncoderLayer` (bidirectional by default) into a decoder, or forgets to pass `is_causal=True` to the attention call.

**Bug 3: padding mask wrong or missing.** Two failure modes hide here. First, real tokens attending *to* padding: if you do not mask pad keys, query positions blend in the (meaningless) value vectors of pad tokens, polluting their representations. Second, the subtler one — pad tokens attending *out* and their outputs polluting the batch through downstream operations or through loss if they are not masked there too. The signature is that outputs depend on how much padding you added, which they must never do, and that batching the same sentence with different neighbors changes its predictions.

**Bug 4: additive versus boolean mask confusion.** Frameworks disagree about mask conventions. Some take a boolean mask where `True` means "this position is valid, keep it"; others where `True` means "this position is masked, block it." Some take a float mask of `0`/`-inf` added to scores; some take `1`/`0` you are supposed to multiply by. If you pass a boolean keep-mask where a block-mask was expected, you invert which tokens are visible — you mask exactly the tokens you meant to keep. If you add a `1`/`0` mask instead of `0`/`-inf`, you nudge scores by one instead of blocking them, which barely masks anything (a `+1` to a logit changes a softmax weight only slightly). Both produce silent, subtle corruption rather than a clean failure.

**Bug 5: mask not broadcast over heads or batch.** Attention scores have shape `(batch, heads, T, T)`. The causal mask is naturally `(T, T)` and must broadcast across batch and heads; the padding mask is naturally `(batch, T)` and must broadcast across heads and query positions to `(batch, 1, 1, T)`. Get the broadcasting wrong — feed a `(batch, T)` mask where `(batch, 1, 1, T)` was needed, or a `(T, T)` mask that silently aligns to the wrong axis — and you mask the wrong positions, or one head behaves differently from the rest. The signature here is shape-dependent and easy to miss because the code does not crash; broadcasting "works" by aligning trailing dimensions, just not the way you intended.

**Bug 6: all-masked row produces a NaN.** If every key for some query is masked — which happens for fully-padded rows, or when a causal mask combines with padding to leave a position with no valid keys — then that row of scores is all $-\infty$. Softmax computes $e^{-\infty} = 0$ for every entry, sums them to `0`, and divides by zero: $0/0 = \text{NaN}$. The NaN then flows into the loss and poisons the whole model in one backward pass. This one masquerades as a numerics bug, which is why it cross-links to NaN hunting, but its root cause is the mask.

**Bug 7: attention dropout left on at eval.** Attention dropout randomly zeros entries of the attention weight matrix during training for regularization. If you evaluate or generate without switching to `model.eval()`, dropout stays active and your attention weights are randomly perturbed at inference, producing noisy, non-deterministic, slightly-degraded outputs. This is a train/eval-mode bug that happens to live in attention; the signature is that eval results change run to run and are a few points worse than they should be.

These seven sort into three symptom buckets, and that sorting is the first move in diagnosis. A suspiciously low train loss with collapsed validation points at bugs 1, 2, or 5 (leakage). A NaN at step one with no other explanation points at bug 6. Train loss that looks fine but generation that is broken or padding-dependent points at bugs 3, 4, or 7. Knowing the bucket from the symptom tells you which test to run next, and that is exactly what makes this a tractable, minutes-not-days diagnosis instead of a blind search.

## 3. The science of the leak: why train loss collapses and by how much

Let us make the leak quantitative, because "train loss gets low" is not precise enough to diagnose with confidence. We want to predict the *magnitude* so that when you see a number, you know whether it is consistent with a leak or with honest learning.

The next-token cross-entropy loss for a single position is $L_t = -\log p_\theta(x_{t+1} \mid x_{\le t})$, where $p_\theta$ is the model's predicted probability of the true next token. Averaged over a corpus, $\bar L = \frac{1}{N}\sum_t L_t$, and perplexity is $\text{PPL} = e^{\bar L}$. For a well-trained language model on natural English, per-token perplexity is bounded below by the entropy of the language itself. Shannon's classic estimates put the entropy of English at roughly one bit per character; converting bits-per-character to per-token cross-entropy depends on tokenization, but the practical floor for a strong model on held-out English is a per-token loss meaningfully above `1.5` (perplexity above ~4.5 for many tokenizers, lower for byte-level but never near `0`). The exact floor varies, but the order of magnitude is robust: **a healthy next-token train loss does not go below roughly `1.5`–`2.5` on real text, and it never approaches zero.**

Now contrast with the leak. If query $t$ can attend to key $t+1$, the model can implement the function: "look up the token at position $t+1$, embed it, and copy that embedding into the output logits." With residual connections and a single attention layer, this is learnable in a handful of steps. The predicted distribution at $t$ then puts almost all mass on the true token $t+1$, so $p_\theta(x_{t+1} \mid \cdot) \to 1$ and $L_t = -\log p_\theta \to 0$. The asymptotic train loss under a full causal leak is therefore approximately the floor imposed by the model's ability to copy, which for a high-capacity model is very close to zero — the `0.05` you saw. The gap between a healthy floor near `2` and a leaked floor near `0` is enormous and unambiguous. **If your next-token train loss is below `1.0`, and certainly below `0.5`, on real text, you almost certainly have a leak.** That single threshold is the cheapest leak detector you own, and it costs nothing — you are already logging train loss.

The validation behavior follows from the same mechanism but with the opposite sign. On data the model has memorized via copying during training, loss is near zero. On held-out data evaluated *with the same buggy mask*, the loss is *also* low, which is a trap — the leak helps on any data you teacher-force through it, because teacher forcing supplies the future tokens. This is why a buggy run can show a low *validation loss* too, as long as validation uses teacher forcing with the same leaky mask. The place the leak cannot hide is **generation**, where you decode one token at a time and the future genuinely does not exist. That is why generation is the gold-standard test: it removes the crutch. A model with a causal leak will have great teacher-forced loss and broken free-running generation, every time. We will turn this into a precise unit test in Section 6, but the science is already telling us where to look.

#### Worked example: reading the numbers on a leaked run

Suppose you finetune a 350M-parameter decoder with a vocabulary of 32,000 tokens. After 2,000 steps your dashboard reads: train loss `0.08`, validation loss (teacher-forced) `0.31`, and a quick generation sample that reads like a Markov chain that has lost its mind. Let us reason from the numbers. A vocabulary of 32,000 means uniform-random loss is $\ln 32000 \approx 10.4$. A strong finetune on this data would plausibly reach train loss around `2.0` and validation around `2.3`. You are at `0.08`, which is perplexity $e^{0.08} \approx 1.08$ — the model is `92%` confident of the correct token at every position. That is not learning; that is reading. The teacher-forced validation loss of `0.31` (perplexity ~1.36) confirms the leak helps on held-out data too, because validation also supplies the future. The broken generation is the tell that breaks the tie: the moment you remove teacher forcing, the crutch is gone. You have not trained a model; you have trained a copier. The fix will *raise* train loss to something like `2.0`, which feels like a regression on the dashboard but is actually the run finally becoming real. Anchor the cost in your head: if that run burned eight hours on an `8×A100` node at roughly `\$12` per GPU-hour, the leak just wasted on the order of `\$760` producing a model that cannot generate. Catching it at step 1 with a unit test is the cheapest insurance you will ever buy.

## 4. The causal mask, constructed correctly and incorrectly

The single highest-value thing to get right is the causal mask, so let us build it from scratch and examine the off-by-one. The correct causal mask for a sequence of length $T$ is lower-triangular: entry $(i, j)$ is "keep" if $j \le i$ (key at or before the query) and "block" if $j > i$ (key in the future). The diagonal must be *kept* — a position is always allowed to attend to itself.

![A grid showing a three by three causal mask where the diagonal and lower-left cells are kept and the upper-right cells above the diagonal are set to negative infinity](/imgs/blogs/attention-and-masking-bugs-2.png)

Here is the correct construction in PyTorch, written three ways so you can recognize the right pattern in any codebase you inherit:

```python
import torch

T = 5

# Way 1: triu of ones, shifted to keep the diagonal.
# triu(diagonal=1) is the STRICT upper triangle (j > i), which is exactly
# the set of future positions we must block. This is the correct argument.
mask_bool = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # True = block
# Convert to an additive float mask: 0 where keep, -inf where block.
additive = torch.zeros(T, T)
additive.masked_fill_(mask_bool, float("-inf"))

# Way 2: tril keep-mask, then invert for blocking. tril(diagonal=0) keeps j <= i.
keep = torch.tril(torch.ones(T, T, dtype=torch.bool))  # True = keep
additive2 = torch.zeros(T, T).masked_fill_(~keep, float("-inf"))

# Way 3: the built-in, which is the one to prefer in production.
additive3 = torch.nn.Transformer.generate_square_subsequent_mask(T)

assert torch.equal(additive.isneginf(), additive2.isneginf())
assert torch.equal(additive.isneginf(), additive3.isneginf())
print(additive)
```

The lethal off-by-one is using `diagonal=2` in the `triu` (which blocks only positions $j > i+1$, leaving $j = i+1$ visible — the future peek), or using `diagonal=0` in `triu` (which blocks the diagonal too, so a position cannot even attend to itself, a different and also-broken state). The most common real-world version is subtler: someone constructs the mask correctly but then **shifts the labels in the wrong direction or by the wrong amount**, which is functionally equivalent to a mask off-by-one because it changes which input position is paired with which label. The boundary between "attention masking" and "label shifting" bugs is thin; both let position $t$ inform the prediction of a token it should not have seen.

Here is the buggy construction next to a quick assertion that catches it:

```python
import torch

def make_causal_mask(T, diagonal=1):
    block = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=diagonal)
    return torch.zeros(T, T).masked_fill_(block, float("-inf"))

good = make_causal_mask(5, diagonal=1)   # correct
bug  = make_causal_mask(5, diagonal=2)   # OFF BY ONE: position t can see t+1

def assert_strictly_causal(additive_mask):
    """A correct causal mask keeps the diagonal-and-below (finite) and blocks
    everything strictly above the diagonal (-inf). Raise if either fails."""
    T = additive_mask.shape[-1]
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool))  # j <= i
    must_block = ~keep                                     # j > i
    finite = torch.isfinite(additive_mask)
    # Every kept cell must be finite; every future cell must be -inf.
    assert finite[keep].all(), "a position cannot attend to its own past"
    assert (~finite[must_block]).all(), "a future position is NOT blocked: LEAK"

assert_strictly_causal(good)             # passes
assert_strictly_causal(bug)              # AssertionError: a future position is NOT blocked: LEAK
```

That `assert_strictly_causal` is the first guardrail to add to any decoder. It runs in microseconds, it does not need a forward pass, and it converts the most expensive bug in this post into an exception at construction time. Notice it tests the *mask itself*, independent of the model — it is a pure correctness check on the data structure. That separation matters: a mask can be correct here and still be applied wrong (broadcast incorrectly, added at the wrong place, overwritten by a library default), which is why we also need the behavioral test in Section 6 that checks the model's actual outputs.

## 5. The before-and-after: what fixing the off-by-one does to the instruments

The whole series insists on before→after evidence, so let us make the fix concrete. You have the `0.08` train loss, the broken generation, and the `assert_strictly_causal` that just fired on `diagonal=2`. You change the `2` to a `1`, restart, and watch the instruments.

![A before and after comparison where the leaked mask gives train loss zero point zero five, validation loss eight, and incoherent generation, and the fixed mask gives train loss two point one, validation loss two point four, and coherent generation](/imgs/blogs/attention-and-masking-bugs-4.png)

The table below is the kind of side-by-side you should be able to produce for any masking fix. The numbers are representative of a real 350M-parameter finetune on a clean corpus; your exact values will differ, but the *direction and magnitude* of each change are the signature you are confirming.

| Instrument | Leaked (off-by-one) | Fixed (correct causal) | What the change confirms |
|---|---|---|---|
| Train loss @ 2k steps | `0.08` | `2.1` | Loss *rising* to a sane floor is correct — the crutch is gone |
| Train perplexity | `1.08` | `8.2` | A perplexity above ~4 is consistent with real language modeling |
| Val loss (teacher-forced) | `0.31` | `2.4` | Val now *tracks* train instead of being mysteriously low |
| Train–val gap | `0.23` | `0.3` | A small, honest gap replaces the leak-suppressed one |
| Generation coherence | broken @ ~10 tokens | coherent paragraphs | The decisive tell: free-running decoding now works |
| Future-invariance test | FAIL (delta `0.4`) | PASS (delta `0.0`) | The unit test from Section 6 flips green |

The counterintuitive line is the first one: **the fix makes train loss go up, and that is the correct outcome.** This trips up engineers constantly. They see the loss jump from `0.08` to `2.1` after a change and assume they broke something, when in fact they just stopped the model from cheating. The way to keep yourself honest is to never read train loss in isolation; read it against the floor you expect from the science (Section 3) and against generation quality. A train loss below the entropy floor is not a triumph, it is a confession.

#### Worked example: distinguishing a leak fix from a real regression

Two days later a teammate sees your commit, notices train loss went from `0.08` to `2.1`, and pings you that you "regressed the model by 25×." How do you prove it is a fix and not a regression in 60 seconds? You run three checks. First, generation: the old checkpoint produces incoherent drift, the new one produces coherent text — decisive on its own. Second, the future-invariance test (next section): old FAILs, new PASSes. Third, the entropy-floor argument: `0.08` is perplexity `1.08`, which is physically impossible for honest English modeling, so the old number was never real. Any one of these settles it; all three together are airtight. The lesson is that a dashboard number is meaningless without a model of what it *should* be — and for next-token loss, the floor from the language's entropy is that model. The cost framing makes it visceral: had you "trusted the dashboard" and shipped the `0.08` checkpoint, you would have shipped a model that cannot generate a sentence, and discovered it only in production.

## 6. The future-invariance test: turning suspicion into a binary signal

The mask assertion in Section 4 checks the mask data structure. But masks get applied wrong in ways the structure cannot reveal: the library ignores your mask and builds its own, the mask broadcasts to the wrong axis, a custom kernel applies it after softmax instead of before, or label shifting reintroduces the leak the mask prevented. To catch *all* of those, you need a test that checks the model's actual behavior, not its mask. The right test follows directly from the definition of causality: **the prediction at position $t$ must not depend on any input token at position $> t$.** If it does, the future has leaked in, no matter how the leak got there.

So the test is: take a batch, record the logits at every position, then perturb the input tokens strictly after some cut position $t$, run the forward pass again, and assert that the logits at positions $\le t$ are bit-identical. A correct causal model is *invariant* to its future; a leaking model is not.

![A before and after comparison of the future-invariance test where perturbing tokens after t changes the logits and FAILs on a leaking mask, and leaves the logits identical and PASSes on a correct mask](/imgs/blogs/attention-and-masking-bugs-6.png)

```python
import torch

@torch.no_grad()
def assert_future_invariance(model, input_ids, cut=None, atol=1e-5):
    """Causality test: logits at positions <= cut must be invariant to any
    change in input tokens at positions > cut. FAILs loudly on a leak."""
    model.eval()  # also rules out attention-dropout noise (bug 7)
    B, T = input_ids.shape
    cut = T // 2 if cut is None else cut

    logits_a = model(input_ids).logits          # (B, T, vocab)

    perturbed = input_ids.clone()
    # Replace every token strictly after `cut` with a different valid token id.
    vocab = model.config.vocab_size
    perturbed[:, cut + 1:] = (perturbed[:, cut + 1:] + 7) % vocab
    logits_b = model(perturbed).logits

    delta = (logits_a[:, :cut + 1] - logits_b[:, :cut + 1]).abs().max().item()
    leaked = delta > atol
    print(f"max logit delta on positions <= {cut}: {delta:.3e}  ->  "
          f"{'LEAK (FAIL)' if leaked else 'causal (PASS)'}")
    assert not leaked, (
        f"Future leak: changing tokens after {cut} moved logits at <= {cut} "
        f"by {delta:.3e}. The model is peeking at the future."
    )
```

Run this once after you build a decoder and once in CI on every model change. On a correct model the delta is `0.0` (or floating-point noise below `1e-5`); on the `diagonal=2` off-by-one it will be large and positive, because perturbing token $t+1$ changes the logits at $t$ that were copying it. This single test catches bugs 1, 2, and 5 (all the leakage bugs) and, because it calls `model.eval()`, incidentally rules out bug 7. It is the most valuable thirty lines of code in your training repo. I add it to the same test file as the overfit-one-batch check, because both are pre-flight sanity tests that should pass before a single real training step runs — see [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) for the companion sanity check that, combined with this one, rules in or out most model-code bugs in minutes.

There is a subtlety worth flagging: the test must use `model.eval()` and `torch.no_grad()`, and it must perturb to a *different valid token id* (the modulo keeps it in range; perturbing to an out-of-range id could trip an embedding-lookup error and mask the real signal). If your model uses sliding-window or chunked attention, the invariance still holds — position $t$ still must not see $t+1$ — so the test is valid for those architectures too, with the same assertion.

#### Worked example: running the future-invariance test on a tiny model

Let us walk the test through with concrete numbers so you know exactly what each outcome looks like, because "delta is large or small" is too vague to act on confidently. Build a two-layer decoder with vocabulary `100`, hidden size `64`, and a sequence of length `8`, then run the test with `cut = 3` (so positions `0,1,2,3` must be invariant to changes at positions `4,5,6,7`).

On a **correct** model, the run prints `max logit delta on positions <= 3: 0.000e+00 -> causal (PASS)`. The delta is exactly zero — not "small," but bit-identical — because the forward pass at positions `0..3` literally never reads positions `4..7`; the same arithmetic runs on the same inputs and produces the same floating-point bits. (If you compile the model or use a fused attention kernel, you may see `1e-7`-scale noise from non-associative floating-point reductions, which is why `atol=1e-5` is the threshold, not exact zero.)

On the `diagonal=2` **off-by-one** model, the run prints something like `max logit delta on positions <= 3: 4.13e-01 -> LEAK (FAIL)`. A delta of `0.41` in the logits is enormous — it means the prediction at position `3` shifted by half a logit when you changed token `4`, which is only possible if position `3` read token `4`. Now the diagnostic power: the delta is *localized*. If you compute the per-position delta instead of the max, you find it is large at positions `2` and `3` (which could see `3` and `4` respectively under the off-by-one) and zero at position `0` (which under `diagonal=2` could see position `1`, unaffected by your perturbation starting at `4` — so to fully exercise it you would lower `cut`). This localization is itself a diagnostic: the *position* where invariance first breaks tells you the *offset* of the leak. A leak at position `t` that responds to a perturbation at `t+1` is an off-by-one; a leak that responds to a perturbation anywhere ahead is a fully-bidirectional missing-mask bug. One test, run with a couple of `cut` values, distinguishes bug 1 from bug 2.

The numbers also calibrate your atol. A *correct* model gives `0` or sub-`1e-6` noise; a *leaking* model gives a delta on the order of the logit scale itself (tenths to whole units). There is no middle ground — you will never see a delta of `1e-3` from a real leak, because a leak that helps train loss must move logits by a lot. So the threshold is not delicate: anything above `1e-4` is a leak, full stop. That binary cleanliness is what makes this test trustworthy enough to gate CI on.

## 7. Padding masks: attending to PAD and the batch-dependence smell

Causal leakage is the dramatic bug; padding bugs are the quiet, pervasive ones. When you batch sequences of different lengths, you pad the short ones to the batch's max length with a `[PAD]` token, and you must ensure (a) real tokens never attend to pad keys, and (b) pad positions never contribute to the loss. Miss (a) and your real tokens blend in the value vectors of pad tokens; because softmax *normalizes over all keys*, even if pad tokens get small weight, they steal probability mass from real tokens and shift every attention output. Miss (b) and you train on predicting pad tokens, which wastes capacity and biases the model toward emitting padding.

The defining smell of a padding bug is **batch-dependence**: the model's prediction for a given sentence changes depending on what else is in the batch with it, because different batchmates produce different amounts of padding. A correct model is batch-invariant — a sentence's predictions must be identical whether it is batched alone or with a long neighbor that forces lots of padding. Here is the test:

```python
import torch

@torch.no_grad()
def assert_padding_invariance(model, tokenizer, sentence, atol=1e-4):
    """A real token's predictions must not depend on how much PAD follows it."""
    model.eval()
    ids = tokenizer(sentence, return_tensors="pt").input_ids
    real_len = ids.shape[1]

    pad_id = tokenizer.pad_token_id
    # Same sentence, padded to two different lengths.
    short = ids
    long = torch.full((1, real_len + 16), pad_id)
    long[:, :real_len] = ids
    attn_short = (short != pad_id).long()
    attn_long = (long != pad_id).long()

    out_short = model(short, attention_mask=attn_short).logits[:, :real_len]
    out_long = model(long, attention_mask=attn_long).logits[:, :real_len]

    delta = (out_short - out_long).abs().max().item()
    print(f"max logit delta over real tokens: {delta:.3e}  ->  "
          f"{'PAD LEAK (FAIL)' if delta > atol else 'pad-invariant (PASS)'}")
    assert delta <= atol, (
        f"Padding leak: adding 16 pad tokens moved real-token logits by "
        f"{delta:.3e}. Real tokens are attending to PAD."
    )
```

If this fails, the usual cause is that you built or relied on a causal mask but never combined it with a padding mask, so the pad keys are visible (causally they are in the past for later positions). The fix is to build the full additive mask as the union: block a key if it is future *or* padding. In Hugging Face `transformers`, you do this by passing `attention_mask` to the model — a `(batch, T)` tensor of `1` for real tokens and `0` for pad — and the model internally expands it to `(batch, 1, 1, T)` and combines it with the causal mask. The trap is forgetting to pass `attention_mask` at all, in which case the model assumes no padding and your pad tokens are fully visible. For decoder-only generation specifically, the direction of padding also matters enormously; left-padding versus right-padding interacts with position ids and causal masking in ways that deserve their own treatment, which the sibling post [attention mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms) covers in full.

| Padding failure | Mechanism | Confirming test | Fix |
|---|---|---|---|
| No `attention_mask` passed | Pad keys fully visible to real tokens | `assert_padding_invariance` FAILs | Pass `attention_mask=(ids != pad_id)` |
| Pad tokens in the loss | Model trained to predict PAD | Pad positions have nonzero loss | Set pad labels to `-100` (`ignore_index`) |
| Mask inverted (1=block) | Real tokens masked, pad kept | Row sums of mask are wrong | Use `1`=keep convention HF expects |
| Mask not broadcast to heads | One head ignores padding | Per-head attention differs on pad | Expand to `(B, 1, 1, T)` |

## 8. Additive versus boolean, and the silent inversion

The fourth bug is a convention mismatch, and it is silent because the wrong mask is still a *valid* mask — it just blocks the wrong things. There are three conventions in the wild, and mixing them is the bug:

- **Additive float mask**: `0.0` keeps, `-inf` (or `-1e9`) blocks. Added to scores before softmax. This is what raw scaled-dot-product attention expects.
- **Boolean keep-mask**: `True` keeps, `False` blocks. Used by some APIs (and by `scaled_dot_product_attention` when you pass a boolean `attn_mask`, where `True` means "participate").
- **Boolean block-mask**: `True` blocks, `False` keeps. Used by `nn.MultiheadAttention`'s `key_padding_mask` and `attn_mask`, where `True` means "do not attend."

The same boolean tensor means opposite things in `scaled_dot_product_attention` (True = keep) versus `nn.MultiheadAttention` (True = block). Pass the wrong one and you invert the mask: you keep exactly the tokens you meant to block and block the ones you meant to keep. The signature is bizarre — the model attends only to the future and padding and ignores the real past — and it usually produces a high, stuck loss rather than a low one, because you have destroyed the useful information. The second sub-bug is the magnitude error: building a `1`/`0` mask and *adding* it (instead of `0`/`-inf`), which only nudges scores by one logit and barely masks anything.

```python
import torch
import torch.nn.functional as F

q = torch.randn(1, 1, 4, 8)
k = torch.randn(1, 1, 4, 8)
v = torch.randn(1, 1, 4, 8)

# CORRECT for F.scaled_dot_product_attention: boolean attn_mask, True = KEEP.
keep = torch.tril(torch.ones(4, 4, dtype=torch.bool))      # True on/below diagonal
out_good = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

# Equivalent additive form: 0 keep, -inf block.
additive = torch.zeros(4, 4).masked_fill_(~keep, float("-inf"))
out_good2 = F.scaled_dot_product_attention(q, k, v, attn_mask=additive)
assert torch.allclose(out_good, out_good2, atol=1e-5)

# The easiest robust path: do not hand-build it at all.
out_builtin = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(out_good, out_builtin, atol=1e-5)

# BUG: passing the INVERTED boolean (True = block) to an API expecting True = keep.
out_bug = F.scaled_dot_product_attention(q, k, v, attn_mask=~keep)
print("inverted-mask output differs:", not torch.allclose(out_good, out_bug, atol=1e-3))
```

#### Worked example: the boolean inversion, traced through one row

Trace a single query's attention through an inverted boolean mask to see exactly how the corruption manifests, because the symptom ("loss stuck high") is easy to misattribute to the learning rate. Take query position `3` in a length-`4` causal sequence. The correct keep-mask for that row is `[True, True, True, True]` — position `3` may attend to all of `0,1,2,3`. Suppose the raw scores after scaling are `[0.5, 1.0, 2.0, 1.5]`. Correct softmax over all four gives weights roughly `[0.11, 0.18, 0.49, 0.29]`, a sensible distribution that leans on the most-similar keys.

Now invert the mask by accident — you pass `~keep`, so the row becomes `[False, False, False, False]` for position `3`. The API interprets `False` as "block," so *every* key for position `3` is set to $-\infty$. That is the all-masked-row case from Section 10: softmax of all $-\infty$ is `0/0 = NaN`, and the loss detonates. For positions where the inversion does not produce a fully-masked row — say position `1`, whose correct keep-mask `[True, True, False, False]` inverts to `[False, False, True, True]` — the model is now forced to attend *only to the future* (`2,3`) and forbidden from its real past (`0,1`). At training time the future is available, so this does not crash, but the model is attending to exactly the wrong tokens: it learns from positions it will not have at inference. The loss does not collapse (it is not a helpful leak — it is destroyed information) and does not cleanly NaN everywhere (only fully-masked rows NaN); it sits stuck at a high value while you waste hours blaming the optimizer. The discriminator is to **print the mask row sums**: a correct causal keep-mask has row sums `[1, 2, 3, 4]` (increasing), while the inverted one has `[4, 3, 2, 1]` (decreasing) — one glance tells you the mask is upside down. That five-second check would have saved the afternoon.

The defensive rule is simple and worth memorizing: **prefer `is_causal=True` over a hand-built mask whenever the attention is purely causal.** It removes the convention question entirely — the library builds the correct lower-triangular mask in the correct format internally. Hand-build a mask only when you need something non-standard (a custom padding combination, a prefix-LM block, a sliding window), and when you do, write the assertion that checks its values, and run the future-invariance test on the resulting model. The reason `is_causal=True` is safer is not that hand-built masks are hard; it is that the *convention* is easy to get backwards, and a backwards mask is silent.

## 9. The broadcast bug: when one head sees the future

Attention scores live in a four-dimensional tensor `(batch, heads, query, key)`, and the mask must align to it. The causal mask is `(query, key)` and broadcasts across batch and heads — that one is usually fine because the trailing two dimensions match. The padding mask is the trap: it is naturally `(batch, key)` because padding is per-sequence per-position, and it must be reshaped to `(batch, 1, 1, key)` so it broadcasts across heads and query positions. The classic error is reshaping it to `(batch, 1, key, 1)` or `(batch, key, 1, 1)`, which broadcasts the padding pattern across the *wrong* axis — you end up masking query positions by their padding status instead of masking key positions, or you mask a different head differently.

The science here is just broadcasting semantics. PyTorch aligns shapes from the trailing dimension and broadcasts size-1 dims. A `(batch, T)` mask added to `(batch, heads, T, T)` scores aligns the `(batch, T)` to the last two dims as `(_, _, batch, T)` — which is nonsense unless `batch == T`. It will not error; it will produce garbage, and if `batch` happens to equal `T` it will silently corrupt without even a shape mismatch to warn you. The guardrail is a shape assertion at the point of application:

```python
import torch

def apply_mask(scores, mask):
    """scores: (B, H, T, T). mask: additive, must broadcast to scores."""
    B, H, Tq, Tk = scores.shape
    # The mask, after any reshaping, must broadcast to (B, H, Tq, Tk).
    try:
        broadcast_shape = torch.broadcast_shapes(scores.shape, mask.shape)
    except RuntimeError as e:
        raise AssertionError(f"mask shape {tuple(mask.shape)} does not "
                             f"broadcast to scores {tuple(scores.shape)}: {e}")
    assert broadcast_shape == (B, H, Tq, Tk), (
        f"mask broadcasts to {broadcast_shape}, not {(B, H, Tq, Tk)} -- "
        f"likely wrong axis (pad mask must be (B,1,1,Tk))"
    )
    # Per-head sanity: every head must apply the SAME causal structure.
    return scores + mask

# Correct padding mask reshape: (B, T) -> (B, 1, 1, T)
B, H, T = 2, 4, 6
scores = torch.randn(B, H, T, T)
pad = (torch.rand(B, T) > 0.5)                       # True = keep
additive_pad = torch.zeros(B, T).masked_fill_(~pad, float("-inf"))
good_mask = additive_pad[:, None, None, :]           # (B, 1, 1, T)  CORRECT
apply_mask(scores, good_mask)                        # passes

bad_mask = additive_pad[:, None, :, None]            # (B, 1, T, 1)  WRONG AXIS
apply_mask(scores, bad_mask)                         # AssertionError or wrong broadcast
```

The broadcast bug is hard to see in code review because the reshape looks deliberate, so the assertion earns its place — it encodes the *intent* (`mask must broadcast to (B, H, Tq, Tk)`) as an executable check. When this bug is present, the future-invariance test from Section 6 also catches it, because a mis-broadcast padding mask typically lets some position see keys it should not. Two independent guardrails on the most error-prone reshape in the model is not overkill; it is proportionate to how silent and expensive the bug is.

#### Worked example: the head-dimension reshape that masks the wrong axis

Here is the version of the broadcast bug that does *not* trip a shape error, which is the dangerous one. Take a batch of `B = 6` sequences, `H = 8` heads, and length `T = 6`, so scores are `(6, 8, 6, 6)`. The padding mask is naturally `(B, T) = (6, 6)`. Notice the trap: `B` and `T` are *both* `6` here, a coincidence that is far more common than you would think because people debug with small square batches. You meant to reshape the pad mask to `(6, 1, 1, 6)` so it broadcasts across heads and query positions, masking *key* positions. Instead you write `additive_pad[:, None, :, None]`, producing `(6, 1, 6, 1)`. Because `B == T == 6`, this *broadcasts cleanly* against `(6, 8, 6, 6)` — no error — but it now masks *query* positions by their pad status instead of *key* positions. The result: pad query rows get masked (which is harmless, they are discarded) but real tokens still attend freely to pad keys (the leak you were trying to prevent is fully present), and the per-head behavior is intact only by luck.

The corruption is invisible in the loss for a while because it is a *partial* pad leak, and it is invisible in a shape check because the shapes broadcast. What catches it is the `assert apply_mask` from above — which compares the *broadcast result shape* to the intended `(B, H, Tq, Tk)` and would flag that the mask is masking the wrong axis even when `B == T`. The second catch is to *deliberately break the coincidence*: run your tests with `B = 3` and `T = 7` (unequal), where the bad reshape `(3, 1, 7, 1)` against `(3, 8, 7, 7)` now raises a clean broadcast error. The lesson is a debugging discipline worth internalizing: **never test masking code with a square batch where `batch == seq_len`,** because the most common broadcast bug hides exactly there. A `(3, 7)` test batch surfaces in milliseconds what a `(6, 6)` batch hides for a full run.

## 10. The all-masked row: a masking bug that detonates as a NaN

Bug 6 is the one that looks like it belongs in a different post. You build a perfectly correct causal mask, combine it with a perfectly correct padding mask, start training, and get `loss = NaN` at step 1. You go hunting in your numerics — checking for `log(0)`, for fp16 overflow, for bad labels — and find nothing, because the NaN was born in attention.

![A dataflow graph showing a fully padded query row whose keys are all masked to negative infinity, so exp gives zero for every entry, the denominator sums to zero, and the zero over zero division produces a NaN that becomes the loss](/imgs/blogs/attention-and-masking-bugs-7.png)

The mechanism is pure softmax arithmetic. Softmax for a row of scores $s_1, \ldots, s_T$ is $\text{softmax}(s)_i = e^{s_i} / \sum_j e^{s_j}$. If every $s_j = -\infty$ — which happens when a query position has *no* valid keys — then every $e^{s_j} = 0$, the denominator $\sum_j e^{s_j} = 0$, and each entry is $0/0 = \text{NaN}$. When does a query have no valid keys? The most common case is a *fully padded row*: a sequence position that is itself padding, whose causal mask blocks the future and whose padding mask blocks all the real keys, leaving nothing. Another case is a position at the very start of a packed sequence where document boundaries and causal masking conspire to leave no valid key. In every case, the all $-\infty$ row is the seed, and the NaN is the symptom one operation later.

This is precisely why the sibling post [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) lists "softmax of an all-masked row" among the canonical NaN sources: the NaN is real, but its *root cause is a masking bug*, and you fix it in the mask, not in the loss. There are two correct fixes. The robust one is to ensure no query row is ever fully masked — guarantee every position has at least one valid key (itself, via the diagonal of the causal mask, which is exactly why the diagonal must always be kept). Fully-padded *query* rows are still a problem because their output is undefined, but their output should never reach the loss (pad positions get `ignore_index = -100`), so the cleanest fix is to never let a fully-padded row's NaN propagate:

```python
import torch
import torch.nn.functional as F

def safe_softmax_attention(scores, additive_mask, query_valid):
    """scores: (B, H, T, T). additive_mask: 0/-inf. query_valid: (B, T) bool,
    True for real (non-pad) query positions. Prevents all-masked-row NaN."""
    masked = scores + additive_mask
    # Detect rows that are entirely -inf (no valid key) BEFORE softmax.
    all_blocked = torch.isneginf(masked).all(dim=-1, keepdim=True)  # (B,H,T,1)
    # For those rows, replace -inf with 0 so softmax returns a uniform (finite)
    # distribution instead of NaN. Their output is discarded downstream anyway.
    masked = masked.masked_fill(all_blocked, 0.0)
    weights = F.softmax(masked, dim=-1)
    # Zero out the contribution of fully-blocked rows for cleanliness.
    weights = weights.masked_fill(all_blocked, 0.0)
    assert torch.isfinite(weights).all(), "attention weights still non-finite"
    return weights
```

The diagnostic that points you here is a forward hook that asserts finiteness on the attention weights, so the NaN is named at the operation that produced it rather than discovered three thousand steps later as a flatlined dashboard:

```python
import torch

def attach_attention_nan_guard(model):
    """Fire at the first non-finite attention output, with the module named."""
    handles = []
    for name, module in model.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            def hook(mod, inp, out, _name=name):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    raise RuntimeError(f"non-finite attention output in {_name} "
                                       f"-- check for an all-masked (all -inf) row")
            handles.append(module.register_forward_hook(hook))
    return handles
```

The diagnostic discipline here is the same bisection the whole series preaches: a step-1 NaN with no data or LR explanation, in a model with padding, is an all-masked row until proven otherwise. Visualize the mask, find the fully-blocked rows, and you have your culprit. For the broader NaN-hunting method — bisecting by step and by layer, `torch.autograd.set_detect_anomaly`, and numerically-stable loss formulations — the dedicated NaN post is the reference.

## 11. Attention dropout at eval, and the train-versus-infer gap

The last bug is the mildest but the most embarrassing to ship. Attention dropout randomly zeros entries of the attention probability matrix during training, scaling the rest up to compensate, as a regularizer. It is supposed to be *off* at inference. If you evaluate or generate without calling `model.eval()`, attention dropout (along with all other dropout and BatchNorm-in-train-mode) stays active, so your attention weights are randomly perturbed every forward pass. The symptoms are: eval metrics that change run to run, generation that is non-deterministic even with greedy decoding and a fixed seed across separate calls, and a small but real degradation (a point or two of accuracy, a fraction of a point of loss) versus the same model in eval mode.

This is genuinely a train/eval-mode bug that happens to manifest in attention; the general treatment of `model.train()` versus `model.eval()` is its own subject. The fix is one line — `model.eval()` before any evaluation or generation, and `model.train()` before resuming training — but the *diagnosis* is what is worth internalizing. If your validation number jitters between runs of the same checkpoint, you have stochasticity at eval, and the two usual sources are dropout-left-on and a non-deterministic data order. Calling `model.eval()` and re-checking is a five-second test that rules one of them out. Note that the future-invariance test in Section 6 already calls `model.eval()`, so a model that passes that test in CI cannot ship with this bug present in the tested path — another reason to make that test part of your pipeline.

There is a deeper version of the train-versus-infer story for language models specifically: even with masking perfectly correct and dropout off, a model can behave differently in training (teacher-forced, full attention over the whole sequence) than in generation (autoregressive, with a KV cache). KV-cache bugs, position-id drift under incremental decoding, and sampling-versus-teacher-forcing discrepancies all live in that gap, and they are covered in [train infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms). The connection to masking is direct: a KV cache is just an incremental way of applying the causal mask, and a cache bug can reintroduce a leak (or break causality) that the training-time mask got right. If your model trains and evaluates correctly but generates incoherently, and the future-invariance test passes, the bug has moved from the mask to the cache.

## 12. The diagnostic flow: four checks before a single training step

Pulling it together, here is the order of operations. Run these four checks once when you build or inherit a decoder, and put the cheap ones in CI. They go from cheapest (no forward pass) to most behavioral, and each rules out a chunk of the taxonomy.

![A timeline of four ordered masking checks, from visualizing the mask and asserting its dtype and shape to perturbing the future and varying padding length, ending in training safely once all pass](/imgs/blogs/attention-and-masking-bugs-8.png)

1. **Visualize the mask and assert it is lower-triangular** (Section 4's `assert_strictly_causal`). No forward pass, microseconds. Rules out the off-by-one and missing-mask data-structure errors.
2. **Assert the mask's dtype, shape, and broadcast target** (Section 9's `apply_mask` assertion). Catches the broadcast bug and the additive/boolean confusion at the point of application.
3. **Run the future-invariance test** (Section 6). One forward pass, a perturbation, one more forward pass. Catches every leakage bug behaviorally, including ones the structural checks miss, and runs in `eval()` so it incidentally checks dropout.
4. **Run the padding-invariance test** (Section 7). Two forward passes of the same sentence at different pad lengths. Catches pad-leak and pad-in-loss bugs.

If all four pass, your masking is correct, and any remaining bad behavior lives elsewhere — in the data, the optimizer, the loss function, or the inference path. That is the bisection: these tests *remove the mask from suspicion* so you can stop staring at attention and look where the bug actually is. This is the same discipline as the overfit-one-batch test: a fast, decisive check that rules a whole class in or out before you spend hours in a debugger. The master frame — symptom to suspect to confirming test to fix — is laid out for the whole series in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and the complete decision tree and printable checklist live in the capstone [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

The full bug-to-test mapping, in one table you can pin above your desk:

![A matrix mapping six masking bugs to their mechanism, the confirming test that isolates each one, and the fix, so the symptom alone points to the next test to run](/imgs/blogs/attention-and-masking-bugs-5.png)

| Symptom | Likely masking bug | Confirming test | Fix |
|---|---|---|---|
| Train loss `< 1.0` on real text, val collapses | Causal off-by-one or no mask | Future-invariance FAILs | `is_causal=True`; diagonal `>=` not `>` |
| Train loss low, val low (teacher-forced), generation broken | Causal leak (any kind) | Generate; future-invariance FAILs | Correct causal mask |
| Output depends on batch composition | Padding mask missing/wrong | Padding-invariance FAILs | Pass `attention_mask`; block pad keys |
| Loss `= NaN` at step 1, padding present | All-masked row | `isnan` on attention weights | Guard fully-blocked rows; keep diagonal |
| Model attends to future/pad, loss stuck high | Inverted boolean mask | Print mask, check row sums | Match the API's True-means convention |
| Eval metrics jitter between runs | Attention dropout at eval | Re-run with `model.eval()` | Call `model.eval()` before eval/generate |

## 13. Beyond decoders: bidirectional models, vision, and time series

So far the running example has been a decoder-only language model, where causality is the central concern. The bug class changes shape in other architectures, and it is worth knowing how, because the *padding* bugs are universal even where causal bugs are not.

**Encoder / bidirectional models (BERT-style).** A bidirectional encoder has *no* causal mask by design — every token attends to every other token, which is correct for understanding tasks where the whole sequence is available. So the off-by-one and missing-causal-mask bugs do not apply; in fact, *adding* a causal mask to an encoder is the bug there, crippling it to half its context. But the padding mask is just as critical: a BERT encoder must still block pad keys, and the same `attention_mask` discipline applies. The all-masked-row NaN can still occur for a fully-padded sequence. The future-invariance test is *not* applicable (bidirectionality means the future legitimately matters), but the padding-invariance test is, and it is the one to run.

**Vision transformers.** A ViT splits an image into patches and runs bidirectional attention over them — no causal mask, like BERT. The masking bugs that bite here are different: a *patch-padding* mask if you handle variable-resolution images, and the masking in masked-image-modeling pretraining (MAE-style), where you mask out a fraction of patches and the bug is masking the wrong patches or leaking the masked patches' content through position embeddings or skip connections. The padding-invariance idea generalizes: a patch's representation should not depend on padding patches added to make a batch rectangular.

**Time-series transformers.** Forecasting models that predict the future from the past are *causal* in exactly the way a language decoder is, and they suffer the identical future-peek bug — and it is even more tempting to introduce, because a careless windowing or a normalization computed over the whole series (including future points) leaks the future without touching the attention mask at all. The future-invariance test applies directly and is the single best safeguard for a forecasting transformer: perturb the future inputs, and the prediction for the present must not move. Many a forecasting model with implausibly good backtest accuracy is a future leak wearing a respectable suit, and the test that catches it is the same one we built in Section 6.

The general principle across all of them: **identify what each position is *allowed* to see, then test that it sees only that.** For causal models, that is the future-invariance test. For padded batches of any architecture, that is the padding-invariance test. The architecture changes which test is meaningful; it does not change the discipline.

## 14. Case studies and real signatures

A few patterns recur often enough to name. These are well-known shapes of the bug rather than exact reproductions of any one team's logs; where I give a number it is representative of the order of magnitude you should expect, not a measured constant.

**The encoder-pasted-into-a-decoder.** A team builds a decoder by reusing PyTorch's `nn.TransformerEncoderLayer`, which is bidirectional by default, and forgets to pass a `src_mask`/`is_causal`. The model trains to a near-zero loss in a few hundred steps — far faster and far lower than any honest decoder — and generates word salad. The tell is the *speed and depth* of the loss drop: honest language modeling does not reach perplexity `1.1` in 300 steps. The future-invariance test FAILs with a large delta. The fix is one argument. This is the canonical bug 2, and it is common precisely because the encoder and decoder layer classes look interchangeable.

**Left-padding breaks decoder-only generation.** A widely-hit bug, well documented in the Hugging Face community, is that decoder-only models must be *left*-padded for batched generation, not right-padded. With right-padding, the pad tokens sit *after* the real prompt, the position ids and causal mask get the real tokens' positions wrong, and generation produces degraded or empty output even though training (which uses right-padding plus a loss mask) was fine. The signature is "training and single-example generation are fine, batched generation is broken," and it is a padding-plus-position bug rather than a causal-leak bug. It is covered in depth in the LLM padding sibling post; the takeaway here is that *which side you pad* is a correctness decision for decoders, not a cosmetic one.

**Confident overfitting that is really a leak.** In tabular and time-series settings, a model that posts implausibly good validation metrics is more often a data leak than a great model — the analog of the attention future-peek. The discipline is identical: when a metric is too good to be true, it is, and the test is to remove the suspected information channel (the future, the leaked feature, the visible neighbor) and confirm the metric drops to a believable value. The general data-leak treatment is in [data leakage the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer); the attention future-peek is the model-code-level instance of the same phenomenon, and it is worth seeing them as one family: *information that is present at training time but absent at inference time inflates training metrics and collapses in production.*

**The all-masked-row NaN under packing.** Teams that pack multiple documents into one sequence (to maximize token throughput) and apply a block-diagonal attention mask sometimes leave a position with no valid keys at a document boundary, producing the all $-\infty$ row and a step-1 NaN. The signature is a NaN that appears only with packing enabled and only on certain batches (the ones with awkward boundaries). The fix is guarding the all-masked row and ensuring every position can at least attend to itself. This is the masking root cause behind a NaN that, investigated naively, looks like a numerics problem.

**The fp16 block value that was not negative enough.** A more insidious case: a custom attention implementation that "avoids `inf`" by masking with a finite sentinel like `-1e4`, written when the model ran in fp32 and never revisited when the run moved to fp16. In fp32 `-1e4` underflows fine after the exponential; but the deeper issue surfaces when the mask is *added* and a real score happens to be large and positive — the masked score `score - 1e4` is still extremely negative, so this part is usually safe. The real failure is the *reverse* sentinel choice, `+0` for "keep" but a too-small magnitude for "block" combined with already-large logits in a deep network: the blocked keys retain a few percent of attention mass. The signature is maddening — the model trains, generation is *mostly* coherent but subtly degraded, and the future-invariance test shows a *small but nonzero* delta (say `3e-3`) instead of a clean zero or a clean large value. That intermediate delta is the fingerprint of a partial leak from an insufficiently-negative block value, and the fix is to switch to `torch.finfo(dtype).min`. The lesson generalizes: when the future-invariance delta is neither `0` nor order-one, suspect a soft mask, not a hard leak.

**The shift-by-one in the labels that mimics a causal leak.** A team correctly builds the causal mask but, when constructing the training labels, fails to shift them so that the label for position `t` is the input at `t` rather than `t+1`. Now the model is asked to predict the *current* token from a representation that includes the current token — a trivial copy through the residual stream, no attention leak required. Train loss collapses to near zero exactly as it does for a mask leak, and generation breaks the same way. The discriminating test is the one in the next section: future-invariance *passes* (the attention is genuinely causal) while train loss is still impossibly low, which points the finger squarely at the labels rather than the mask. This case is the reason the future-invariance test and the entropy-floor check are *both* in the pre-flight suite: together they separate the two most common impossibly-low-loss bugs.

## 15. When this is (and isn't) your bug

A masking bug has a sharp signature, and the discipline is to not blame attention when the symptom points elsewhere.

**It IS a masking bug when:** train loss on real text is implausibly low (below ~`1.0`, certainly below `0.5`) while generation is broken — that is a leak, almost always causal. Or the model's prediction for a sentence changes when you change its batchmates — that is padding. Or you get a step-1 NaN with padding present and no `log(0)`/LR explanation — that is an all-masked row. Or eval metrics jitter between identical runs — that is attention dropout (a train/eval-mode bug). The future-invariance test is the master discriminator: if it FAILs, you have a leak and should stop looking anywhere else.

**It is NOT a masking bug when:** train loss is *high* and stuck — a leak makes loss *low*, so a high stuck loss is an optimization or data problem, not a leak (run the overfit-one-batch test instead). A smooth loss curve that goes NaN at step 4,000 is numerics (overflow, `log(0)`), not masking — masking NaNs appear at step 1, deterministically, tied to padding. A model that trains, evaluates, *and* generates correctly but is simply not accurate enough is a capacity, data, or hyperparameter story, not a masking one — the mask is correct, you just need a better model. And if the future-invariance and padding-invariance tests both PASS, the mask is correct by construction; move on to the loss function, the data pipeline, or the inference path. The most important thing the four tests buy you is the *right to stop suspecting attention* — that is what bisection is for. A passing future-invariance test is permission to look elsewhere, and that permission is worth as much as catching the bug, because it stops you from rereading correct attention code for hours.

There is one genuinely confusing overlap worth calling out: a *loss-masking* bug (training on the prompt, or shifting labels wrong) can produce a low train loss that looks like a causal leak. The discriminator is the future-invariance test: a causal-mask leak FAILs it (the model's logits at $t$ depend on token $t+1$); a label-shift or prompt-masking bug PASSes it (the attention is causal; the *labels* are wrong). If train loss is suspiciously low but future-invariance PASSes, the bug is in the loss/labels, not the mask — see [the loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) for that branch. This is bisection at its finest: one test cleanly separates two bugs with the same symptom.

It is worth walking the full triage once as a decision procedure, because the value of these tests is that they execute in a fixed order and each one *eliminates* a branch. Start from the symptom. If **train loss is implausibly low**, run the future-invariance test. FAIL means a causal/broadcast leak — fix the mask and stop. PASS means the attention is honest, so the impossibly-low loss is coming from the labels or the loss reduction — check the label shift and the loss masking, not the attention. If **the symptom is broken generation with fine train loss**, run the padding-invariance test and check the padding side. PASS on padding-invariance plus correct padding side means the gap is in the inference path — the KV cache, position ids under incremental decoding, or sampling — not the mask. If **the symptom is a step-1 NaN**, check whether padding is present and whether any query row is fully masked; an all-masked row is the masking-flavored NaN, while a smooth-then-NaN curve much later is a numerics story for the NaN-hunting post. If **the symptom is eval jitter between identical runs**, call `model.eval()` and re-check; dropped jitter confirms attention dropout (or another train-mode module) left on.

Notice what this procedure refuses to do: it never has you *reading attention code* as the first move. Reading code to find a masking bug is the slowest possible path, because the bug is usually a one-character offset or a convention mismatch that the eye skips over precisely because it looks correct. The tests are faster *and* more reliable than reading, because they exercise the actual behavior. The single most common waste of an afternoon in this whole bug class is an engineer who *suspects* the mask, stares at the `triu` call for an hour convinced it is right (it is), and never runs the future-invariance test that would have located the leak in the *labels* in thirty seconds. Run the test first; read the code only after a test has narrowed the suspect to a specific line. That ordering — instrument, then read — is the entire discipline of this series compressed into one habit.

## 16. Key takeaways

- **The mask is one additive term before softmax**, with exactly two legal values: `0` (keep) and `-inf` (block). Every masking bug violates that sentence.
- **A next-token train loss below ~`1.0` on real text is a leak, not a triumph.** The language's entropy puts an honest floor around `1.5`–`2.5`; a perplexity near one means the model is reading the answer.
- **The future-invariance test is the master discriminator:** perturb tokens after $t$, and a correct decoder's logits at $\le t$ must be bit-identical. FAIL means a leak; it catches the off-by-one, the missing mask, and the broadcast bug in one shot.
- **Padding bugs smell like batch-dependence.** A sentence's prediction must not change with its batchmates; the padding-invariance test catches it. Always pass `attention_mask`, and set pad labels to `-100`.
- **An all-masked row is a masking bug wearing a NaN costume.** A row of all `-inf` makes softmax compute `0/0`; it shows up at step 1 with padding present. Guard fully-blocked rows and always keep the causal diagonal.
- **Prefer `is_causal=True` to a hand-built mask.** It removes the boolean-versus-additive convention question, which is the silent inverter.
- **Fixing a leak makes train loss go up, and that is correct.** Read train loss against the entropy floor and against generation quality, never in isolation.
- **A passing future-invariance and padding-invariance pair is permission to stop suspecting attention** and bisect to the loss, the data, or the inference path.
- **Bidirectional models (BERT, ViT) have no causal bug but the same padding bug;** causal time-series transformers have the identical future-peek as a decoder. Identify what each position may see, then test that it sees only that.

## 17. Further reading

- Vaswani et al., "Attention Is All You Need" (2017) — the original scaled dot-product attention and where the mask enters; the source of the $1/\sqrt{d_k}$ scaling and the masked self-attention in the decoder.
- PyTorch documentation: `torch.nn.functional.scaled_dot_product_attention` (the `is_causal` and `attn_mask` semantics, including the boolean-True-means-keep convention) and `torch.nn.Transformer.generate_square_subsequent_mask`.
- Hugging Face `transformers` documentation on `attention_mask`, padding side for decoder-only generation, and the `-100` `ignore_index` loss-masking convention.
- PyTorch documentation: `torch.autograd.set_detect_anomaly` and `register_forward_hook`, the tools behind the finiteness guard and NaN localization.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the symptom→suspect→test→fix decision tree, and the capstone [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
- Companion posts: [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test), [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs), [attention mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms), [the loss masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), and [train infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms).
