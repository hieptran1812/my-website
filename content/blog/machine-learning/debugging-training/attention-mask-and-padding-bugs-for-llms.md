---
title: "Attention Mask and Padding Bugs for LLMs: Left, Right, and the Pad That Leaks"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the padding-side, attention-mask, position-id, and packing bugs that quietly corrupt decoder-only LLMs, using the softmax-normalization math, a left-versus-right padding argument, and a batched-equals-unbatched test that turns suspicion into a binary signal."
tags:
  [
    "debugging",
    "model-training",
    "padding",
    "attention-mask",
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
image: "/imgs/blogs/attention-mask-and-padding-bugs-for-llms-1.png"
---

You finetuned a 7B decoder-only model. Training looked healthy: the loss came down smoothly to about `0.9`, the eval loss tracked it within a tenth, nothing exploded, no NaN, no warnings in the log. You shipped the checkpoint to a batched inference endpoint, sent it a batch of eight prompts of different lengths, and the output came back wrong in a very specific way. The longest prompt in the batch generated a perfectly reasonable answer. The shortest prompt generated fluent nonsense — grammatical, confident, and completely unrelated to the question. When you re-ran the *exact same short prompt by itself*, with a batch size of one, the answer was correct. Same weights, same decoding parameters, same prompt. The only thing that changed was the company it kept in the batch.

That is not a weights problem and it is not a sampling problem. A model whose output depends on its batch-mates is telling you, unambiguously, that something about how the batch is assembled is leaking into the computation. For decoder-only LLMs there is one overwhelmingly common cause of exactly this signature: the tokenizer is **right-padding** for generation when it must **left-pad**. The padding tokens you added to make the short prompt the same length as the long one ended up *after* the real text, so when the model reads "the last position" to predict the next token, it reads a pad, not the last real word. The fix is one line — `tokenizer.padding_side = "left"` — and the reason it works is a precise argument about positions, the KV-cache, and where the next-token logits come from. This post is about that argument and the whole family of bugs around it.

![A dataflow graph showing query times key scores entering a softmax with a padding mask, where an unmasked pad key leaks twenty-four percent of the attention mass and shifts the output vector while a blocked pad key keeps the mass on real keys](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-1.png)

These are the bugs that live in the **model code** and **data pipeline** places of the six places a training or finetuning bug can hide — data, optimization, model code, numerics, systems, and evaluation — and they are some of the most expensive precisely because they so rarely crash. A wrong padding side does not raise an exception; it returns confident garbage. A mask that does not match `input_ids` does not error; it shifts attention by a few tokens and lowers your loss in a way that looks like progress. A pad token leaking into the loss does not warn you; it just makes the model slightly better at predicting padding and slightly worse at everything else. By the end of this post you will be able to take any decoder-only run — training or generation — and in a few minutes prove whether its padding side is correct, whether the attention mask matches the tokens, whether the position ids account for padding, whether pad tokens are leaking into the loss, and whether packed sequences are bleeding across document boundaries. We will derive *why* each bug produces its signature from the softmax-normalization math and the autoregressive structure of generation, write the runnable check that confirms it, and show the before→after that proves the fix. This is the practical sibling to the general [attention and masking bugs](/blog/machine-learning/debugging-training/attention-and-masking-bugs) post; that one covers the causal-mask off-by-one and the future-token peek in the abstract, and this one covers what actually breaks when you batch real, variable-length text through a real tokenizer.

## 1. The arithmetic of a mask, and the two masks a decoder carries

Before we can debug padding, we have to be precise about where a mask enters the computation and what it does, because every bug in this post is a violation of that one mechanical fact. Attention, reduced to its arithmetic, is a weighted average of value vectors where the weights come from a softmax over similarity scores. For queries $Q$, keys $K$, and values $V$ of head dimension $d_k$, with an additive mask $M$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

The query-key product $QK^\top$ is a $T \times T$ matrix of raw scores, entry $(i,j)$ being how much query position $i$ wants key position $j$. The scaling keeps dot products from saturating the softmax. The softmax turns each row into a probability distribution over keys. The matmul with $V$ takes the weighted average. None of those steps decide *which keys a query is allowed to see*. Only $M$ does. The mask is a $T \times T$ matrix added to the scores **before** the softmax; its only two meaningful values are `0` (keep this key) and `-inf` (block this key, in practice a large negative number such as `torch.finfo(dtype).min`). After adding $M$, a blocked score is $-\infty$, the softmax computes $e^{-\infty} = 0$, and the key gets exactly zero weight — and, critically, contributes exactly zero to the denominator, so the remaining keys renormalize to sum to one among themselves. That last clause is the entire reason a padding bug shifts every output and not just the pad position, and we will use it constantly.

A decoder-only LLM carries **two** logically distinct masks that both enter at this point, and the entire art of this post is keeping them straight:

- The **causal mask** enforces autoregression. Query position $t$ may attend to keys $0, 1, \ldots, t$ but never to $t+1, \ldots$, because at generation time those future tokens do not exist yet. It is purely a function of position — the same lower-triangular pattern for every sequence in the batch.
- The **padding mask** enforces that no query attends to a padding token. Padding tokens are filler you add to make variable-length sequences into a rectangular `(batch, seq_len)` tensor. Unlike the causal mask, the padding mask depends on the data: which positions are padding differs from row to row.

The mask actually applied is the **union** of the two: a key is blocked if it is in the future *or* it is padding. In Hugging Face `transformers`, you usually do not build the $T \times T$ float mask by hand — you hand the model a 2D `attention_mask` of shape `(batch, seq_len)` where `1` marks a real token and `0` marks a pad, and the model expands it internally to the 4D additive mask `(batch, 1, seq_len, seq_len)`, combines it with the causal pattern, and adds it before each softmax. That 2D-to-4D expansion is where many bugs hide, because the 2D mask you pass *looks* fine while the 4D mask it expands to is wrong, or because the 2D mask does not actually correspond to your `input_ids`.

Why does an unmasked pad hurt so specifically? Let one query attend over four real keys with raw scores $(2, 1, 0, 1)$ and one pad key that, left unmasked, scores $1.5$. The softmax denominator is $e^2 + e^1 + e^0 + e^1 + e^{1.5} \approx 7.39 + 2.72 + 1.00 + 2.72 + 4.48 = 18.31$, so the pad key alone soaks up $4.48/18.31 \approx 24\%$ of the attention. That `24%` is not merely wasted on a meaningless token — it is *subtracted* from the real keys, which now share only `76%` of the distribution, and the output vector is pulled `24%` of the way toward the pad token's (garbage) value vector. Mask the pad and the denominator drops to $13.83$, the real keys reclaim all the mass, and the output is correct. The corruption scales with how much padding you added, which is exactly why a padding bug's hallmark symptom is **batch-dependence**: a sequence batched with longer neighbors gets more padding, steals more mass, and produces a more corrupted output than the same sequence batched with short neighbors or run alone. The first figure above shows that fork — the same scores feeding a softmax, with the pad either blocked or leaking — and the rest of the post is a tour of the ways the union mask goes wrong on real, variable-length, batched text.

Two refinements make the mechanism even sharper, and both predict a real bug. First, the *amount* of mass a pad steals depends on the pad token's score relative to the real scores, and that score is not random — it is the dot product of the query with the pad token's key vector, which after pretraining and finetuning can be *large*. A pad token that the model has seen millions of times tends to develop a key vector that several queries find moderately similar, so the leaked mass is often not a negligible one or two percent but a meaningful ten to thirty percent. You cannot reason "the pad will get little weight anyway, so a missing mask is harmless"; the arithmetic does not support that hope. Second, the leak is *nonlinear* in the number of pads. With $k$ pad keys each at score $s_{\text{pad}}$, the total stolen mass is $k \cdot e^{s_{\text{pad}}} / Z$ where $Z$ is the full denominator, so doubling the padding more than doubles the corruption once the pads start to dominate $Z$. This is the precise reason the batch-dependence is not gentle: a sequence padded to twice the length does not get twice the error, it gets a super-linear jump, which is why the bug presents as "fine on most batches, catastrophic on the batch with one very long member."

There is a subtle second-order effect worth naming because it confuses people who *do* mask attention but still see drift. The softmax is shift-invariant — adding a constant to every score leaves the distribution unchanged — but it is *not* invariant to which entries are in the normalization set. Removing a pad from the denominator (by masking it) rescales *all* the remaining weights, so masking is not a local edit to one entry; it changes the entire row's distribution. This is why you cannot "approximately" mask by zeroing the pad's weight *after* the softmax: zeroing post-softmax leaves the denominator polluted, so the real weights no longer sum to one and the output is shrunk toward zero by the fraction the pad used to hold. The mask must be applied *before* the softmax, in the score space, so the pad is excluded from normalization entirely. A surprising number of hand-rolled attention implementations get this wrong by multiplying a `0`/`1` mask into the post-softmax weights, which zeroes the pad weight but leaves every real weight too small — a quiet, uniform attenuation that looks like a slightly under-confident model rather than a bug.

## 2. The bug taxonomy, sorted by symptom

There are seven padding-and-mask bugs that account for nearly every decoder-only failure of this kind, and like all good taxonomies they sort by symptom, which is the first move in diagnosis. The figure below is the map; the rest of the post fills it in.

![A decision matrix mapping four padding bugs of wrong side, mask not matching input ids, pad in loss, and pad id equal to eos id, each to its symptom, its confirming check, and its fix](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-4.png)

**Bug 1: wrong padding side for the task.** Decoder-only batched *generation* requires left padding; *training* is normally right-padded. Getting it backwards breaks batched generation for the padded (shorter) sequences while leaving the longest sequence — which needs no padding — looking fine. The signature is the one in the intro: batched output depends on batch-mates, and batched disagrees with unbatched. We derive *why* in Section 3.

**Bug 2: the `attention_mask` does not match `input_ids`.** The mask is computed for a different length than the tokens (a truncation applied to one but not the other), or it is all-ones, ignoring the pad entirely, or it is stale from a previous example. The model then attends to pad tokens (the all-ones case) or masks the wrong positions (the length-mismatch case). The signature is a loss that is subtly off and attention that shifts with padding. Section 4.

**Bug 3: position ids do not account for padding or packing.** With left padding, the real tokens no longer start at position 0; if you let the model default to `position_ids = [0, 1, 2, ...]` across the whole row, the real tokens get the wrong rotary or learned positional encoding and generation drifts. With packing, each document should restart its position count. The signature is left-padded generation that is slightly worse than right-padded, or packed training that underperforms. Section 5.

**Bug 4: pad tokens leak into the loss.** If the pad positions are not set to the loss's `ignore_index` (`-100` in PyTorch cross-entropy and Hugging Face), the model is trained to predict pad tokens. Because pad is the single most frequent "token" in a heavily-padded batch, the model gets very good at predicting it, the average loss drops in a way that looks like progress, and capacity is wasted. The signature is a suspiciously low loss whose per-real-token loss is much higher. Section 6, cross-linked to [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug).

**Bug 5: packing without a block-diagonal mask (cross-document contamination).** To avoid wasting compute on padding, you concatenate many short documents into one long row. Under a plain causal mask, a token late in the row attends back into *earlier, unrelated documents*. The model learns to use context that will not exist at inference. The fix is a block-diagonal mask (or FlashAttention's variable-length / "varlen" path with `cu_seqlens`). The signature is a clean-looking loss that generalizes worse than padded training. Section 7.

**Bug 6: `pad_token_id == eos_token_id`.** Many tokenizers ship without a dedicated pad token and people set `pad_token = eos_token` to make the collator happy. If you then *also* train on the EOS positions (do not mask them), the model sees EOS followed by EOS followed by EOS in every padded row and learns that EOS predicts EOS — that stopping is cheap and repeatable. At inference it stops too early or emits runs of EOS. Section 8.

**Bug 7: 4D-versus-2D and additive-versus-boolean mask confusion.** You pass a 2D `(batch, seq_len)` mask where a 4D `(batch, 1, q, k)` mask was expected, or a boolean keep-mask where an additive `0`/`-inf` mask was expected, or a `1`/`0` multiplicative mask added instead of multiplied. The result is a mask that blocks the wrong positions or barely masks at all. Section 9.

These seven sort into three symptom buckets. Batched generation that disagrees with unbatched, or output that depends on batch-mates, points at bugs 1, 2, or 3 (the geometry of padding). A loss that looks too low for the data points at bugs 4 or 5 (something is leaking into or out of the loss). A model that stops too early or emits EOS runs points at bug 6. Knowing the bucket from the symptom tells you which test to run next, and that is what makes this a minutes-not-days diagnosis instead of a blind grid search over tokenizer settings.

## 3. Left versus right padding: the position argument, made rigorous

This is the most important section in the post, so let us build the argument from first principles rather than asserting "use left padding for generation" as a rule to memorize. The claim is precise: **decoder-only batched generation requires left padding because the next token is produced from the hidden state at the last position of the sequence, and only left padding guarantees that the last position holds a real token rather than a pad.**

![A two-column before and after comparison contrasting right padding where the last slot is a pad and the next-token logits come from a pad state against left padding where the last slot is the real final token and batched generation matches unbatched](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-2.png)

Start with how autoregressive generation works for a single sequence, no batch. You feed the prompt $x_0, x_1, \ldots, x_{n-1}$ through the model, which produces a hidden state at every position. To generate the next token, you take the hidden state at the **last** position, $h_{n-1}$, project it to vocabulary logits, and sample. You append the sampled token, and repeat. The defining fact is that *the next-token prediction always comes from the last position*. The model's `forward` returns logits of shape `(batch, seq_len, vocab)`, and generation reads `logits[:, -1, :]` — the slice at the final position — to get the distribution for the next token. Hold onto that `-1`; it is the whole bug.

Now batch two prompts of different lengths. A short prompt of 3 tokens and a long prompt of 7 tokens must become a rectangular `(2, 7)` tensor, so the short prompt needs 4 pad tokens. Consider **right padding** first:

```bash
row 0 (short):  [t0 t1 t2 P  P  P  P ]   positions 0..6
row 1 (long):   [u0 u1 u2 u3 u4 u5 u6]   positions 0..6
```

Generation reads `logits[:, -1, :]`, i.e. position 6 of every row. For the long prompt, position 6 is `u6`, the real last token — correct, the model predicts what comes after `u6`. For the short prompt, position 6 is a **pad token**. The next-token logits for the short prompt come from the hidden state of a pad token, four positions past where the real text ended. Even if you masked the pad keys correctly so the pad position cannot *attend* to anything meaningful, you are still reading the *output* at the pad position to decide the next real token, and that output is garbage. The model generates a first token that has nothing to do with `t0 t1 t2`. That is exactly the intro symptom: the short prompt in the batch produces nonsense while the long prompt is fine, because only the short prompt is being read at a pad position.

Now **left padding**:

```bash
row 0 (short):  [P  P  P  P  t0 t1 t2]   real positions need fixing (Section 5)
row 1 (long):   [u0 u1 u2 u3 u4 u5 u6]
```

Generation reads position 6 of every row. For the short prompt, position 6 is now `t2` — the real last token. The next-token logits come from the real final hidden state, exactly as in the unbatched case. The long prompt is unchanged. Both rows are read at a real token, batched generation matches unbatched generation, and the bug vanishes. That is the argument, and it is why every serious generation harness left-pads. The Hugging Face `generate` function literally warns you when `padding_side="right"` is set on a decoder-only model for exactly this reason: *"A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'`."* If you have ever seen that warning and ignored it, this is the bug it was protecting you from.

There is a second, equally important consequence beyond the `-1` slice: the **KV-cache and position alignment**. During incremental decoding, the model caches keys and values for already-seen positions and only computes the new token, so the cost per generated token is constant rather than growing with sequence length. The cache is a tensor indexed by position, and the new token is written at the next free slot. With right padding, the cache for the short row contains four pad positions sandwiched between the prompt and the first generated token: the prompt occupies slots `0..2`, the pads occupy slots `3..6`, and the first generated token wants to go at slot `7`, which is *after* four meaningless pad entries. So the generated tokens land at the wrong cache offsets, their position ids are computed relative to the padded length (slot `7`) rather than the real length (which should be slot `3`, right after `t2`), and every newly generated token attends back over a cache that has four pad entries wedged in the middle of the real context. Even if those pad entries are masked, the *positions* are now wrong: the first generated token thinks it is at position 7 when the real context is only 3 tokens long, so its rotary encoding is rotated by an offset of four. Left padding keeps the real tokens contiguous at the right edge — slots filled from the right — so newly generated tokens append cleanly at the next real position and their rotary angles follow the real tokens with no gap. Right padding does not just corrupt the first generated token; it corrupts the geometry of the entire generation loop, which is why the symptom is not "the first token is slightly off" but "the whole continuation is unrelated."

It is worth pausing on *why the library cannot just fix this for you transparently*. In principle a generation harness could detect right padding and internally re-index everything to the real lengths, and some do paper over part of it, but the fundamental obstacle is that the `-1` slice and the cache write are baked into the fast incremental-decoding path for throughput. Special-casing per-row real lengths inside the fused decode kernel would cost the very speed the cache exists to provide. So the ecosystem made the pragmatic choice: require left padding for generation, where the geometry is uniform across the batch (every row's real tokens end at the same right edge), and the fast path stays simple. That is the engineering reason the rule is a *requirement* and not merely a *recommendation* — it is load-bearing for the performance of every batched-generation system you will ever use.

Why is training the opposite — usually **right**-padded? Because training is teacher-forced: you feed the entire sequence at once and compute the loss at *every* position in parallel, not just the last. There is no `-1` slice; every real position contributes a loss term and every pad position is masked out of the loss (Section 6). Right padding keeps the real tokens at positions `0..n-1`, which is the natural alignment for the causal mask and for `position_ids = [0, 1, 2, ...]` with no shift. You *can* left-pad during training if you also fix position ids and the loss mask, but there is no benefit, and right padding is the convention every data collator assumes. The rule that falls out is sharp and worth memorizing: **train right, generate left.** Get it backwards in either direction and you break the corresponding phase.

#### Worked example: the batch-dependent prompt

You serve a chat model behind a batched endpoint with a default batch size of 16 and `padding_side="right"` left over from your training tokenizer. A user sends "What is 2+2?" — six tokens after templating. It happens to be batched with fifteen long document-summarization prompts of around 800 tokens each, so your six-token prompt gets padded to 800, meaning **794 pad tokens after it**. Generation reads position 799 — a pad — and produces a fluent paragraph about document structure, because the only signal at that position is whatever the pad's hidden state drifted to. The user re-sends the same prompt during a quiet minute; now it is batched with three other short prompts, padded to 40 tokens, read at position 39 — still a pad, still wrong, but *differently* wrong, because the corruption magnitude depends on how much padding there is. Your monitoring shows an intermittent, unreproducible "the model hallucinates on simple questions" bug that correlates with nothing you log, because what it correlates with — batch composition — is not in your logs. Switch to `padding_side="left"`, redeploy, and the six-token prompt is read at its real last token regardless of batch-mates. The "intermittent hallucination" disappears. If that endpoint served a million such requests before you caught it, on an `A100` at roughly `\$3` per GPU-hour, you did not lose money to compute — you lost it to every short prompt being wrong for weeks.

## 4. The attention mask that does not match the tokens

The second bug is more mundane and more common: the `attention_mask` you pass does not correspond to the `input_ids` you pass. There are three flavors, and all three are silent.

**Flavor A: the all-ones mask.** You build `input_ids` with padding but pass `attention_mask=None` or `attention_mask=torch.ones_like(input_ids)`. The model now attends to every position including the pads. Per Section 1, each unmasked pad steals softmax mass proportional to its score, and because every real query attends to the pads, every real token's representation is polluted. This is the most directly corrupting flavor because it reintroduces exactly the normalization pollution the mask exists to prevent. The signature: outputs depend on padding amount, and a sequence batched with more padding scores differently from the same sequence run alone.

**Flavor B: the length mismatch.** You truncate `input_ids` to 512 tokens but build `attention_mask` from the pre-truncation length, or vice versa. Now the mask's `1`s and `0`s land at the wrong positions — the mask says "real" where there is a pad and "pad" where there is real text. The model masks real tokens and attends to pads. This usually does not crash because the shapes still broadcast (both are length 512), so the mismatch is purely in the *values*, which no shape check catches. The signature: loss is wrong in a way that correlates with which examples got truncated.

**Flavor C: the stale mask.** In a hand-rolled collator you reuse a mask buffer across batches and forget to rebuild it, so a short batch carries a previous long batch's mask. Same value-mismatch failure as Flavor B.

The diagnostic for all three is the same and it is embarrassingly simple: **print the mask next to the input ids and verify that the `1`s are exactly the non-pad positions.** Here is the runnable check.

```python
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # see Section 8 for the caveat

batch = tok(
    ["What is 2+2?", "Explain the theory of relativity in one paragraph."],
    padding=True,
    return_tensors="pt",
)

ids = batch["input_ids"]
mask = batch["attention_mask"]

# The invariant: a position is attended (mask==1) iff it is NOT a pad token.
expected = (ids != tok.pad_token_id).long()
mismatches = (mask != expected).sum().item()
print("padding_side:", tok.padding_side)
print("input_ids:\n", ids)
print("attention_mask:\n", mask)
print("expected mask (ids != pad):\n", expected)
print("mask/ids mismatches:", mismatches)
assert mismatches == 0, "attention_mask does not match input_ids -- Bug 2"
```

If `mismatches` is anything but zero, you have Flavor A, B, or C, and the printed tensors tell you which: an all-ones `mask` against an `expected` with zeros is Flavor A; a mask whose `1`s are in different columns than `expected` is Flavor B or C. The assert is cheap enough to leave in your collator's unit test permanently. The reason this check is airtight is the invariant it encodes: the attention mask is, by definition, the indicator of "real token," and "real token" is by definition "not the pad id." Any deviation is a bug, full stop. The one subtlety is when `pad_token_id == eos_token_id` (Section 8): then `ids != pad_token_id` also marks genuine EOS tokens as pads, and you need a position-based length to disambiguate. We handle that case explicitly later.

It is worth being explicit about how the all-ones flavor (Flavor A) interacts with the math from Section 1, because it is the most damaging and the most common in hand-rolled code. With an all-ones mask the pad keys are fully visible, so each real query's softmax includes every pad in its normalization. If a row has $n$ real tokens and $k$ pads, and the pads carry typical scores, the fraction of mass lost to pads grows roughly as $k/(n+k)$ when scores are comparable — so a sequence that is `60%` pad loses well over half its attention mass to meaningless tokens, and its representation is dominated by pad value vectors. Worse, this happens at *every* layer and *every* head, compounding through depth: a small per-layer corruption becomes a large end-of-stack corruption after thirty-two layers. The model can partially adapt during training by learning pad value vectors that are near-zero, which masks the bug in the loss while leaving the representations fragile to any change in padding amount — exactly the batch-dependence that bites at inference. The practical upshot is that an all-ones mask is not "slightly suboptimal"; it is a structural leak that the model papers over just well enough to hide from your training curve.

One more trap inside this family: when you provide your own `attention_mask` but rely on the model's default `position_ids`, a correct mask is necessary but not sufficient. Masking the pad keys stops them polluting attention, but if the position ids are wrong (Section 5), the real tokens still get the wrong positional encoding. The mask and the position ids are two separate excludes, and fixing one does not fix the other — which is the whole point of the stack figure later: a pad has to be excluded at *three* stages, and people commonly fix one and assume they are done.

## 5. Position ids under padding and packing

This is the subtle one, the bug that survives even after you have fixed the padding side and the mask, because the position ids are a *third*, independent thing that has to account for padding. The figure for the pipeline of excludes makes the three-stage structure concrete.

![A vertical stack showing a pad token introduced by the collator that must then be excluded from the attention mask, the position ids, and the loss labels, with a miss at any stage corrupting the run](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-5.png)

The model needs to know each token's position to apply positional encoding — rotary embeddings (RoPE) in most modern LLMs, or learned absolute position embeddings in older ones. By default, many model `forward` implementations compute `position_ids = torch.arange(seq_len)` — `[0, 1, 2, ..., T-1]` — for every row, ignoring the mask. That default is correct for **right** padding because the real tokens occupy `0..n-1` and the pads occupy `n..T-1`; the real tokens get positions `0..n-1`, exactly right, and the pads get positions they will never use because they are masked out anyway.

But with **left** padding the default is wrong. The pads occupy `0..(T-n-1)` and the real tokens occupy `(T-n)..(T-1)`. If you let `position_ids = [0, 1, ..., T-1]`, the first real token gets position `T-n` instead of `0`, the second gets `T-n+1` instead of `1`, and so on — every real token's positional encoding is shifted by the amount of padding. For RoPE, where position enters as a rotation angle, this rotates every query and key by an offset that depends on how much padding the row happened to get, so the same real token at the same real position gets a different encoding depending on batch composition. That is the *same* batch-dependence signature as the padding-side bug, which is why people fix the padding side, see generation improve but not fully, and get stuck: there were two position bugs, not one.

The correct `position_ids` for a padded row count only the real tokens, starting from 0 at the first real token. Hugging Face's `generate` and most model `forward` paths now build this for you *if you pass the `attention_mask`*, via the cumulative-sum trick:

```python
import torch

def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Real tokens are numbered 0,1,2,... starting at the first real token.
    Pads get position 1 (a dummy, since they are masked out anyway).
    This matches the convention used inside transformers' decoder forward."""
    # cumulative count of real tokens, minus 1, clamped so pads do not go negative
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos = pos.masked_fill(attention_mask == 0, 1)
    return pos

mask = torch.tensor([
    [0, 0, 1, 1, 1],   # left-padded: two pads, three real tokens
    [1, 1, 1, 1, 1],   # no padding
])
print(position_ids_from_mask(mask))
# tensor([[1, 1, 0, 1, 2],     <- real tokens at positions 0,1,2 (correct)
#         [0, 1, 2, 3, 4]])
```

The real tokens in the left-padded row get positions `0, 1, 2` — exactly as if there were no padding — and the pad positions get a dummy value that never matters because they are masked. The rule: **if you left-pad, you must pass the `attention_mask` so the model can derive correct `position_ids`, or pass `position_ids` explicitly.** Relying on the default `arange` with left padding is the bug. The cheapest confirmation is to print the position ids the model will use and check that the first *real* token in every row gets position 0.

Packing adds a second position concern. When you concatenate documents A, B, C into one row to avoid padding, a naive `arange` numbers them `0, 1, 2, ..., (len_A + len_B + len_C - 1)` continuously, so document B's first token gets a position deep into the sequence rather than 0. Whether that matters depends on the positional scheme and whether you also fixed the attention mask (Section 7), but the clean convention is to **restart position ids at 0 for each packed document**, matching how the model will see each document at inference (alone, starting at position 0). Implementations that support packing — `trl`'s `SFTTrainer` with `packing=True` on recent versions, or FlashAttention varlen — handle this for you; hand-rolled packing usually does not, and that is a bug.

#### Worked example: left padding helped, but not enough

You switched a generation harness from right to left padding and watched a quality metric on a held-out set of short prompts jump from `41%` exact-match to `68%` — the padding-side fix working. But unbatched generation on the same prompts scored `74%`, so a `6`-point gap remained that should not exist: batched and unbatched ought to be identical for a deterministic decoder. You print `position_ids` for a batched run and find the first real token of a short, heavily-padded row sitting at position `93` instead of `0`, because the model defaulted to `arange` and you were passing the mask but an older model revision ignored it for position derivation. You pass `position_ids` explicitly via `position_ids_from_mask`. Batched exact-match rises to `74%`, closing the gap to unbatched exactly. The lesson: the padding side and the position ids are two separate excludes, and a partial fix leaves a residual gap that masquerades as "batching is just a little lossy" — it is not; it is a second bug.

## 6. Pad tokens leaking into the loss

Now move from attention to the loss, because a pad can corrupt the objective even when attention is perfectly masked. During teacher-forced training, the cross-entropy loss is computed at every position: at position $t$ the model predicts the token at $t+1$, and the loss term is $-\log p_\theta(x_{t+1} \mid x_{\le t})$. The pad positions have labels too — the next token after a real token might be a pad, and a pad's "next token" is another pad. If you do not exclude these, the model is trained to predict padding.

The mechanism for why this lowers your loss deceptively is worth making quantitative, because the symptom — "loss looks great" — is the opposite of alarming. Suppose a batch is `40%` padding on average (common when sequences vary a lot in length). Predicting the pad token is trivial: after one pad, the next is almost always another pad, so $p_\theta(\text{pad} \mid \text{pad}) \to 1$ within a few steps and the loss on pad positions goes to nearly `0`. The reported average loss is a length-weighted mix:

$$
\bar L = (1 - f)\,\bar L_{\text{real}} + f\,\bar L_{\text{pad}} \approx (1 - f)\,\bar L_{\text{real}} + f \cdot 0 = (1 - f)\,\bar L_{\text{real}}
$$

where $f$ is the pad fraction. With $f = 0.4$ and a true per-real-token loss $\bar L_{\text{real}} = 2.0$, the reported loss is $0.6 \times 2.0 = 1.2$. You see `1.2`, feel good, and the model is actually at `2.0` on the tokens you care about — and worse, gradient steps are being partly spent making it better at predicting pad, which is capacity you wanted spent on language. The signature: the loss is suspiciously low and *moves when you change the padding amount* (longer max length, more padding, lower reported loss), which a correct loss must never do.

The fix is `ignore_index`. PyTorch's `nn.CrossEntropyLoss` and `F.cross_entropy` take `ignore_index=-100`, and Hugging Face models use the same convention internally: any label equal to `-100` contributes nothing to the loss and nothing to the gradient. The job of the data collator is to set pad-position labels to `-100`. `DataCollatorForLanguageModeling` and `DataCollatorForSeq2Seq` do this for you; a hand-rolled collator often forgets. Here is the check that catches it.

```python
import torch
import torch.nn.functional as F

def loss_pad_audit(labels: torch.Tensor, pad_token_id: int, ignore_index: int = -100):
    """Confirm pad positions are excluded from the loss.
    `labels` is the [B, T] label tensor your collator produced."""
    n_total = labels.numel()
    n_ignored = (labels == ignore_index).sum().item()
    # Any label that still equals the pad id is a pad LEAKING into the loss.
    n_pad_leaking = ((labels == pad_token_id) & (labels != ignore_index)).sum().item()
    print(f"total labels        : {n_total}")
    print(f"ignored (-100)      : {n_ignored}  ({100*n_ignored/n_total:.1f}%)")
    print(f"pad tokens in loss  : {n_pad_leaking}")
    assert n_pad_leaking == 0, "pad tokens are leaking into the loss -- Bug 4"

# Compare reported loss with and without masking to quantify the deception.
def reported_vs_real(logits, labels, pad_token_id, ignore_index=-100):
    V = logits.size(-1)
    leaked_labels = labels.clone()
    leaked_labels[leaked_labels == ignore_index] = pad_token_id  # simulate the bug
    loss_leaked = F.cross_entropy(logits.view(-1, V), leaked_labels.view(-1))
    loss_clean = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                                 ignore_index=ignore_index)
    print(f"loss WITH pad in it : {loss_leaked.item():.3f}")
    print(f"loss masked (real)  : {loss_clean.item():.3f}")
```

Run `loss_pad_audit` on a batch from your real collator and `reported_vs_real` on a forward pass, and you will see the two numbers diverge by exactly the factor $(1-f)$ the math predicted. This is the cross-link to [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), which goes deeper on the related question of masking the *prompt* versus the *completion* — a different `-100` decision with the same machinery. For now the rule is: **count your `-100` labels; if pad positions are not among them, your loss is lying about how well you are doing on real tokens.**

There is a sharp connection back to attention here that is easy to miss. Masking a pad out of *attention* (Section 4) and masking it out of the *loss* (this section) are independent. You can mask attention perfectly and still train on pad labels, or mask the loss perfectly and still let pads pollute attention. The pad has to be excluded at both stages — and at the position-id stage too — which is exactly the three-stage structure of the stack figure above. A reviewer who checks the attention mask, sees it correct, and signs off has verified one of three excludes.

## 7. Packing and cross-document attention contamination

Padding wastes compute: if your batch is `40%` pads, `40%` of your matmul flops do nothing — you pay for the full rectangular `(batch, seq_len)` tensor but only `60%` of it carries signal. The waste is even worse than the pad fraction suggests for attention specifically, because attention is quadratic in sequence length: a sequence padded to length $T$ costs $O(T^2)$ in the attention matmul regardless of how many of those positions are real, so padding short sequences up to a long maximum is doubly expensive. If your data has a long tail of lengths — a few 2,000-token documents among many 100-token ones — naive padding to the max can spend the overwhelming majority of its attention flops on pad-against-pad scores that are immediately masked away. The standard fix is **packing** — concatenate many short documents end to end until you fill the context length, so almost every position is a real token and almost every flop does useful work. Packing can double or triple throughput; a corpus that was `55%` padding becomes near `0%`, turning a nine-hour finetune into a four-hour one. It also introduces one of the nastiest silent bugs in LLM training, and it is silent precisely because it *raises* throughput and lowers loss while corrupting what the model learns.

![A grid of a packed batch attention layout where document B attending back into document A is marked as cross-document leak while the block-diagonal cells confine each document to its own span](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-3.png)

The bug: when documents A, B, and C share one row under a **plain causal mask**, every token can attend to every earlier token in the row — including tokens in *earlier, unrelated documents*. A token in document C attends back into A and B. The model learns to predict C's tokens using A's and B's context, which is leakage of information that will never be present at inference, when C is generated on its own starting from position 0 with no A or B in its context. The model gets a free, fake context window during training that it cannot use at test time. The loss looks fine — even good, because cross-document context is sometimes weakly predictive — but the model generalizes worse and, in the worst case, learns spurious dependencies on whatever happened to precede each document in the packed stream.

The figure above shows the fix as a mask layout: each document attends only within its own block (the `A->A` and `B->B` cells, plus the causal lower triangle inside each block), and any attention from B back into A is blocked to `-inf`. That is a **block-diagonal causal mask** — block-diagonal across documents, causal within each block. Constructing the full $T \times T$ block-diagonal mask in memory is feasible for short context but expensive for long context (it is $T^2$), which is why production training uses **FlashAttention's variable-length ("varlen") path**. Instead of materializing the mask, you pass `cu_seqlens` — the cumulative sequence lengths marking document boundaries — and FlashAttention computes attention within each segment only, never across boundaries, in the fused kernel. The before→after figure makes the contrast concrete.

![A before and after comparison contrasting naive concatenation under a plain causal mask that bleeds context across documents against FlashAttention varlen with cumulative sequence lengths that confines attention to each document span](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-8.png)

Here is the conceptual shape of constructing the segment ids and the boundaries a varlen path consumes, plus a small dense block-diagonal mask you can use to verify correctness on a toy batch:

```python
import torch

def build_packing_metadata(seq_lengths):
    """Given the lengths of documents packed into one row, build the
    segment ids and cumulative sequence lengths FlashAttention varlen wants."""
    seg_ids = torch.cat([torch.full((L,), i) for i, L in enumerate(seq_lengths)])
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0)))
    return seg_ids, cu_seqlens

def block_diag_causal_mask(seg_ids):
    """Dense reference mask (for tests, not production): a key is allowed iff
    it is in the same document AND not in the future."""
    T = seg_ids.size(0)
    same_doc = seg_ids[:, None] == seg_ids[None, :]          # block-diagonal
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # no future
    allowed = same_doc & causal
    mask = torch.zeros(T, T)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask

seg_ids, cu_seqlens = build_packing_metadata([3, 2, 4])   # docs of length 3, 2, 4
print("segment ids:", seg_ids.tolist())                    # [0,0,0,1,1,2,2,2,2]
print("cu_seqlens :", cu_seqlens.tolist())                 # [0,3,5,9]
m = block_diag_causal_mask(seg_ids)
# Spot-check: position 5 (first token of doc 2) must NOT see position 0 (doc 0).
print("doc2 -> doc0 blocked?", torch.isinf(m[5, 0]).item())   # True
print("doc2 -> doc2 self ok?", (m[5, 5] == 0).item())         # True
```

The two-line confirmation that this matters: take a single document, run it alone, then run it packed *after* a long unrelated document, both with the **plain causal mask**, and compare the loss on the shared document. If the losses differ, the document is reading its packed neighbor — that is the contamination. Switch to the block-diagonal mask (or varlen with the right `cu_seqlens`) and the two losses become identical, which is the proof that the document is now isolated. The rule: **packing without a block-diagonal/varlen mask trains on context that will not exist at inference; the test is that a document's loss is invariant to what it is packed with.**

#### Worked example: the packing speedup that cost a point

You enable packing to cut a finetune from 9 hours to 4 — a real `2.25×` throughput win, because your data was `55%` padding before and is now near `0%`. Training loss comes down to `0.94`, a touch lower than the `0.98` you got with padding, which you read as "packing is strictly better." But your held-out instruction-following eval *drops* from `62%` to `58%`. You suspect the eval, the LR, the epochs — the usual suspects — and waste a day. The actual cause: you packed with a plain causal mask, so every completion was trained while attending to whatever random documents preceded it, and the model learned to lean on context that vanishes at inference. You run the invariance test — same document, packed versus alone — and the losses differ by `0.07`, confirming cross-document bleed. You switch `trl`'s `SFTTrainer` to its varlen/`flash_attention_2` packing path (or pass `cu_seqlens`), retrain, and the eval recovers to `63%` while keeping most of the speedup. The lower training loss was the leak helping the model cheat; the honest number was always a little higher.

## 8. When the pad token is the EOS token

Many tokenizers — GPT-2's, Llama's base tokenizer, and others — ship without a dedicated padding token, because the base model was pretrained on packed streams that never needed one. When you start batching for finetuning or generation, the collator needs *some* id to pad with, and the path of least resistance is `tokenizer.pad_token = tokenizer.eos_token`. That single line is fine *if and only if* you are disciplined about two downstream things; if you are not, it produces a model that stops too early or emits runs of end-of-sequence tokens.

The mechanism is the loss-leak of Section 6 wearing a costume. Set `pad_token_id == eos_token_id` and right-pad a training batch, and a short sequence looks like `[t0, t1, t2, EOS, EOS, EOS, EOS]` where the first `EOS` is the *real* end-of-sequence the model should learn to emit, and the rest are *padding* that happens to share the EOS id. If your loss masking is based on "set pad positions to `-100`" but you identify pad positions by `ids == pad_token_id`, you cannot distinguish the real terminal EOS from the padding EOS — they have the same id. Two failure modes follow:

- If you mask *all* EOS-id positions out of the loss (including the real one), the model **never learns to emit EOS**, so at inference it does not know when to stop and generates until it hits `max_new_tokens` — the "model won't shut up" bug.
- If you mask *none* of them (train on every EOS-id position), the model sees EOS followed by EOS followed by EOS in every padded row and learns that **EOS predicts EOS** — stopping is cheap and self-reinforcing — so at inference it stops too early, often immediately, or emits a short run of EOS.

Both are bad, and both come from the id collision making "real EOS" and "pad" indistinguishable by id alone. The clean fix is to use a **distinct** pad token so the two are separable, then mask only the pads:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Option A (preferred): add a dedicated pad token, distinct from EOS.
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<|pad|>"})
    # remember: model.resize_token_embeddings(len(tok)) after this

# Now pad positions and real EOS positions have different ids, so a
# label-masking rule keyed on the pad id keeps the real EOS in the loss.
assert tok.pad_token_id != tok.eos_token_id
```

If you cannot or do not want to add a token (it requires resizing the embedding matrix and the new row starts untrained), you can keep `pad_token == eos_token` but you must build the loss mask from **position**, not from token id — mask everything after the *first* EOS in each row, which preserves that first EOS in the loss while excluding the padding EOS after it. The position-based collator below does exactly that.

```python
import torch

def labels_with_first_eos_kept(input_ids, eos_id, ignore_index=-100):
    """When pad_id == eos_id, keep the FIRST eos per row in the loss
    (the real terminator) and mask every position after it (padding)."""
    labels = input_ids.clone()
    B, T = input_ids.shape
    for b in range(B):
        eos_positions = (input_ids[b] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 1:
            first_eos = eos_positions[0].item()
            labels[b, first_eos + 1:] = ignore_index  # mask the padding EOS
    return labels

ids = torch.tensor([[5, 6, 7, 2, 2, 2, 2]])   # eos_id = 2; one real, three pad
print(labels_with_first_eos_kept(ids, eos_id=2).tolist())
# [[5, 6, 7, 2, -100, -100, -100]]  -> real EOS at index 3 kept, rest masked
```

The confirming test: prompt the finetuned model and check that it emits exactly one EOS and stops, rather than running to `max_new_tokens` (the never-learned-EOS case) or stopping immediately (the EOS-predicts-EOS case). The rule: **if `pad_token_id == eos_token_id`, you must keep exactly the first EOS per row in the loss and mask the rest; the safest path is a distinct pad token.** This is also a [tokenization bug](/blog/machine-learning/debugging-training/tokenization-bugs) at heart — it is about special-token identity — which is why that sibling post and this one keep pointing at each other.

## 9. The mask API zoo: 2D vs 4D, additive vs boolean

The last family is about API confusion, because the ecosystem disagrees about what a "mask" is, and passing the right values in the wrong convention silently masks the wrong thing. There are four conventions you will meet:

| Convention | Shape | Meaning of the values | Where you see it |
| --- | --- | --- | --- |
| 2D key-padding mask | `(batch, seq_len)` | `1` = keep, `0` = pad (HF) | `transformers` `attention_mask` |
| 2D boolean (PyTorch MHA) | `(batch, seq_len)` | `True` = **ignore**, `False` = keep | `nn.MultiheadAttention` `key_padding_mask` |
| 4D additive float | `(batch, 1, q, k)` | `0` = keep, `-inf` = block | internal expanded mask |
| 4D boolean (SDPA) | `(batch, …, q, k)` | `True` = keep, `False` = block | `scaled_dot_product_attention` |

The two traps that produce the most silent corruption:

**Boolean polarity inversion.** PyTorch's `nn.MultiheadAttention` `key_padding_mask` uses `True` to mean "ignore this position" — the *opposite* of Hugging Face's `attention_mask`, where `1`/`True` means "keep." If you take a Hugging Face `attention_mask` and pass it straight into `nn.MultiheadAttention` as a `key_padding_mask`, you invert the meaning: you ignore exactly the real tokens and attend to exactly the pads. The model sees only padding. This usually does not crash because the shape is right; it just trains on garbage. Always check the polarity convention of the specific function you are calling.

**Additive vs multiplicative magnitude.** A `0`/`-inf` mask is *added* before softmax. A `1`/`0` mask is meant to be *multiplied* into the weights (or used to build the additive mask). If you *add* a `1`/`0` mask, you nudge blocked scores by `+1` instead of sending them to `-inf` — and `+1` to a logit barely changes a softmax weight, so the "blocked" keys are almost fully visible. This is the silent-leak cousin of the `-1e4` block-value bug: the mask is present, the code runs, and it masks almost nothing. The rule is the one from Section 1: an additive mask's block value must be negative enough that $e^{\text{block}}$ underflows in your dtype; `torch.finfo(dtype).min` is the safe choice.

PyTorch's `scaled_dot_product_attention` (SDPA) deserves a specific note because it is now the default fast path. It accepts `attn_mask` as **either** a boolean mask (`True` = keep) **or** an additive float mask, and it auto-detects by dtype. It also has an `is_causal=True` shortcut that applies the causal pattern without you materializing it — but `is_causal=True` and a custom `attn_mask` are mutually exclusive in spirit: if you pass both, you can double-apply or conflict. The safe pattern for a padded decoder batch is to build one additive mask that is the union of causal and padding and pass it as `attn_mask` with `is_causal=False`, or use `is_causal=True` for the causal part and a separate key-padding step — but never both casually. The minimal check is to feed a tiny known batch and assert the masked positions get zero weight:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, d = 1, 1, 4, 8
q = torch.randn(B, H, T, d)
k = torch.randn(B, H, T, d)
v = torch.randn(B, H, T, d)

# Block the last key for the FIRST query, additively.
attn_mask = torch.zeros(B, H, T, T)
attn_mask[0, 0, 0, 3] = float("-inf")

# Reproduce SDPA by hand to read the weights and confirm the block.
scores = (q @ k.transpose(-2, -1)) / (d ** 0.5) + attn_mask
weights = F.softmax(scores, dim=-1)
print("weight query0 -> key3 (should be 0):", weights[0, 0, 0, 3].item())
assert weights[0, 0, 0, 3].item() == 0.0, "mask did not block -- check polarity/value"
```

If that weight is not exactly `0.0`, your mask value or polarity is wrong before you ever touch a real model. The rule: **know your function's convention — 2D vs 4D, additive vs boolean, keep-True vs ignore-True — and unit-test that a blocked key gets exactly zero weight.**

## 10. A full bisection: a finetune that broke only when batched

Let me put the whole toolkit to work on one realistic run, because the point of this series is not to memorize seven bugs but to *bisect* — to narrow from a symptom to a suspect to a confirming test without touching code blindly. Here is the run. You finetuned a 7B instruction model with `trl`'s `SFTTrainer`, packing on, on a dataset of customer-support conversations. Training was uneventful: loss to `0.87`, eval loss `0.91`, no NaN, no warnings you noticed. You deployed to a batched endpoint and within an hour support flagged that the assistant "sometimes answers a completely different ticket." Reproduction was maddening — it depended on time of day.

Start with the six places. The symptom is not a NaN (rules out the loud end of numerics), not a high-and-stuck loss (rules out the obvious optimization and data-quality failures), and not present at training time. It is a *generation-time, batch-correlated* defect. That single observation — "correlates with time of day," which is a proxy for *traffic*, which is a proxy for *batch size and composition* — points the spotlight at the model-code/data-pipeline place where padding lives. Before reading any code, you form the hypothesis: the output depends on batch composition. The cheapest possible test confirms or kills it: run the same prompt batched with long neighbors, then run it alone.

```python
prompt = build_chat("My order #4471 never arrived, what do I do?")
out_batched = model.generate(**tok([prompt, long_a, long_b, long_c],
                                    padding=True, return_tensors="pt").to(dev),
                             max_new_tokens=64, do_sample=False)
out_alone = model.generate(**tok([prompt], return_tensors="pt").to(dev),
                           max_new_tokens=64, do_sample=False)
print("batched:", tok.decode(out_batched[0], skip_special_tokens=True))
print("alone  :", tok.decode(out_alone[0],  skip_special_tokens=True))
```

The batched output discusses a refund policy for a product the user never mentioned; the unbatched output correctly asks for the order's shipping email. **Batched disagrees with unbatched.** That is the master signature, and it has just told you the bug is padding geometry — not weights (same weights), not sampling (`do_sample=False`, deterministic), not the prompt format (same prompt). You have bisected from "sometimes wrong, correlates with traffic" to "padding geometry" in two `generate` calls.

Now localize *within* padding. First suspect, because it is the most common: padding side. Print it.

```python
print(tok.padding_side)   # -> 'right'
```

Right padding on a decoder-only model at generation time is Bug 1. You set `tok.padding_side = "left"`, re-run the batched-vs-unbatched test, and the gross "wrong ticket" failure is gone — both now discuss order #4471. But you do not stop, because the series discipline is to *measure* the fix, and the exact-match on a held-out batch of 200 support prompts reads `71%` batched against `76%` unbatched. A `5`-point residual gap on a deterministic decoder is a second bug. You print the position ids and find the first real token of a heavily-padded row at position `38`, not `0` — the older model revision was not deriving positions from the mask. You pass `position_ids` explicitly (the `position_ids_from_mask` helper from Section 5), and batched exact-match rises to `76%`, matching unbatched exactly. Two bugs, found by refusing to accept a partial fix.

Now the stress test, because a good debugger asks "what *else* is wrong that this symptom was hiding?" You enabled packing. Was the packing mask block-diagonal? You run the invariance test: the same support conversation, packed after a long unrelated ticket, versus alone, both at training time with the trainer's mask.

```python
loss_alone  = lm_loss(model, pack([conv]))
loss_packed = lm_loss(model, pack([long_unrelated_ticket, conv]))
print(loss_alone.item(), loss_packed.item())   # -> 2.02  1.94
```

They differ by `0.08`. The conversation's loss drops when it is packed after another ticket, which means it is *attending across the document boundary* into that ticket — Bug 5, cross-document contamination, hiding under a packing speedup. Your `0.87` training loss was partly the leak helping. You switch the trainer to the FlashAttention-2 varlen packing path so each document gets its own block, retrain, and the invariance test now reads `2.02` both ways while the held-out instruction-following eval ticks up two points. Three bugs — wrong padding side, mask-blind position ids, and cross-document packing — all under one symptom of "sometimes answers the wrong ticket," all found by bisecting from the symptom to the place to the specific bug, confirming each with a binary test before changing anything. That is the method; the seven bugs are just the map of where to look.

#### Worked example: putting numbers on the three-bug run

Tally the instruments across the bisection so the before→after is concrete. Before any fix: batched exact-match `41%`, unbatched `76%`, training loss `0.87`, same-doc-packed-vs-alone gap `0.08`. After left padding: batched `71%`. After mask-aware position ids: batched `76%` (gap to unbatched closed). After varlen packing: training loss rises to the honest `0.94`, the packing invariance gap collapses to `0.00`, and downstream eval rises from `58%` to `60%`. Notice that *two* of the three fixes made a headline number look *worse* — training loss went up, because it had been artificially low from the cross-document leak. That is the signature of an honest fix in this domain: the dashboard often regresses while the model improves, because the original number was inflated by the very bug you removed. If this finetune cost `\$180` in GPU time, the rerun cost another `\$180` — cheap against shipping a model that answered the wrong support ticket for every batched short prompt.

## 11. Case studies and real signatures

These are well-known patterns worth naming, because recognizing the signature is most of the diagnosis.

**The Hugging Face right-padding generation warning.** The `transformers` library emits, for decoder-only models with `padding_side="right"` at generation time, the explicit warning quoted in Section 3 telling you to set `padding_side="left"`. This is not a style nag; it is the library detecting Bug 1 for you. The signature it protects against is precisely the batch-dependent, short-prompts-break failure. If your generation harness suppresses warnings (many do, to clean up logs), you will not see it — which is an argument for *not* suppressing library warnings during bring-up. The fix is one line and the library is begging you to apply it.

**Packing and cross-document attention in production training.** The move from naive packing under a causal mask to **document-aware** packing (block-diagonal mask / FlashAttention varlen with `cu_seqlens`) is now standard in serious pretraining and instruction-tuning pipelines exactly because the cross-contamination measurably hurt downstream quality. Modern `trl` `SFTTrainer` packing, when paired with FlashAttention-2, supports the varlen path so packed documents do not attend across boundaries; older or hand-rolled packing did not, and the resulting models learned spurious cross-document dependencies. The signature, as in the worked example, is a packing speedup that comes with a small but real downstream-quality regression that disappears once attention is confined per document.

**`pad_token = eos_token` and the model that won't stop (or stops instantly).** The id collision in Section 8 is one of the most frequently reported finetuning footguns on community forums: a finetuned chat model that either generates until `max_new_tokens` every time (real EOS masked out of the loss) or replies with an empty/one-token response (EOS-predicts-EOS learned from padding). The tell that distinguishes it from other "won't stop" causes is that it appears the moment you switch to a padded batch collator with `pad_token = eos_token` and goes away when you add a distinct pad token or keep only the first EOS per row in the loss.

**The left-padding-improves-but-leaves-a-gap signature.** The Section 5 worked example — batched generation that gets much better with left padding but retains a few points of gap to unbatched — is a recurring real pattern whenever the position ids are not derived from the mask. It is worth naming because it is the case where people *think* they fixed the padding bug (they fixed one of two) and stop, leaving a residual quality gap they then misattribute to "batching is inherently a bit lossy." It is not; deterministic decoding must be batch-invariant, and any gap is a remaining bug.

| Symptom you observe | Most likely bug | Confirming test | Fix |
| --- | --- | --- | --- |
| Batched gen garbage on short prompts, unbatched fine | Right padding for generation (Bug 1) | Assert batched output == unbatched for each prompt | `padding_side="left"` for generate |
| Output depends on batch-mates, attention shifts with padding | Mask `!=` `input_ids`, or pads unmasked (Bug 2) | Print mask vs `(ids != pad_id)`; count mismatches | Rebuild mask from `ids != pad_id` |
| Left padding helped but batched `!=` unbatched still | Position ids not mask-aware (Bug 3) | Print `position_ids`; first real token must be 0 | Pass `attention_mask` or explicit `position_ids` |
| Loss suspiciously low; drops when max length grows | Pad tokens in the loss (Bug 4) | Count `-100` labels; compare masked vs leaked loss | Set pad-position labels to `-100` |
| Packing sped up training but eval regressed | Cross-document attention (Bug 5) | Same doc packed vs alone — losses must match | Block-diagonal mask / varlen `cu_seqlens` |
| Model won't stop, or stops instantly | `pad_id == eos_id` mishandled (Bug 6) | Is `pad_id == eos_id`? Are EOS positions masked? | Distinct pad token; keep first EOS in loss |
| Masked positions still get weight; model sees only pads | Mask API/polarity confusion (Bug 7) | Unit-test: blocked key weight must be exactly 0 | Match function's convention; use `finfo.min` |

## 12. The five-minute audit: a runnable pre-flight

Tie the diagnostics together into one ordered pass you can run on a single batch before you launch any decoder-only run. The order matters: it goes from the cheapest, most common bug (padding side) to the subtlest (packing), so you spend your attention in proportion to how often each bug bites.

![A timeline of five ordered checks for a decoder-only run covering padding side, the mask matching the ids, the position ids, the loss mask, and the block-diagonal packing mask](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-7.png)

```python
import torch
from transformers import AutoTokenizer

def padding_preflight(tokenizer, examples, mode="generate"):
    """One pass over a real batch that checks the five padding excludes.
    mode='generate' expects left padding; mode='train' expects right."""
    expected_side = "left" if mode == "generate" else "right"
    print(f"[1] padding_side = {tokenizer.padding_side} (expected {expected_side})")
    assert tokenizer.padding_side == expected_side, f"set padding_side={expected_side}"

    batch = tokenizer(examples, padding=True, return_tensors="pt")
    ids, mask = batch["input_ids"], batch["attention_mask"]

    # [2] mask matches ids (mind the pad==eos caveat)
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        mism = (mask != (ids != tokenizer.pad_token_id).long()).sum().item()
        print(f"[2] mask/ids mismatches = {mism}")
        assert mism == 0, "attention_mask does not match input_ids"
    else:
        print("[2] pad==eos: verify mask via length, not id (see Section 8)")

    # [3] position ids: first real token in each row must be position 0
    pos = mask.long().cumsum(-1) - 1
    pos = pos.masked_fill(mask == 0, 1)
    for b in range(ids.size(0)):
        first_real = (mask[b] == 1).nonzero(as_tuple=True)[0][0].item()
        assert pos[b, first_real].item() == 0, f"row {b} first real token pos != 0"
    print("[3] position ids OK: first real token at position 0 in every row")

    # [4] loss mask sanity (only meaningful in train mode)
    if mode == "train":
        labels = ids.clone()
        labels[mask == 0] = -100
        leaking = ((labels == tokenizer.pad_token_id) & (labels != -100)).sum().item()
        print(f"[4] pad tokens leaking into loss = {leaking}")
        assert leaking == 0 or tokenizer.pad_token_id == tokenizer.eos_token_id

    print("[5] packing: if you concatenate docs, pass cu_seqlens / block-diag mask")
    return batch

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<|pad|>"})
tok.padding_side = "left"
padding_preflight(tok, ["What is 2+2?", "Summarize the water cycle in two sentences."],
                  mode="generate")
```

Run that, and four of the five bugs cannot survive to step one; the fifth (packing) is a structural choice you confirm with the same-doc-packed-vs-alone invariance test. The whole pass takes seconds and a single batch. The before→after that proves it is worth stating in instrument terms: a run with Bug 1 shows batched generation exact-match far below unbatched (e.g. `41%` vs `74%`); after the fix they are equal. A run with Bug 4 shows a reported loss of `1.2` against a true per-real-token loss of `2.0`; after masking, the dashboard reads the honest `2.0` and the model actually improves on real tokens. A run with Bug 5 shows a same-document loss that changes by `0.07` depending on packing neighbors; after the block-diagonal mask, it is invariant. Each fix turns a number that *looked* like progress into one that is honest, which is the recurring theme of this entire series: your instruments were lying, and a precise test made them tell the truth.

The decision of which side to pad, and the position-id consequence that rides along with it, is captured in the tree figure — the task picks the side, and left-padding generation pulls in the position-id fix.

![A decision tree from whether the task is batched, branching to generation which requires left padding and a position-id fix, and training which uses right padding with a masked loss](/imgs/blogs/attention-mask-and-padding-bugs-for-llms-6.png)

## 13. When this is (and isn't) your bug

Padding and mask bugs have a specific fingerprint, and it is worth being decisive about when a symptom points *elsewhere* so you do not waste a day flipping the padding side on a bug it cannot fix.

**It IS a padding/mask bug when:** batched output disagrees with unbatched for the same input; output depends on batch composition; the symptom appears the moment you start batching variable-length sequences; left padding improves batched generation; the loss moves when you change the max sequence length; a model stops too early or never stops right after you set `pad_token = eos_token`; a packing speedup comes with a downstream regression. All of these are batch-geometry or token-identity signatures, and they are what this post fixes.

**It is NOT a padding/mask bug when:** unbatched generation is *also* broken (batch size 1 has no padding, so a padding bug cannot be the cause — look at the prompt format, the [chat template](/blog/machine-learning/debugging-training/attention-and-masking-bugs), sampling, or the weights). If train loss is high and *stays* high regardless of padding, that is an optimization or data problem, not a mask. If the loss is smooth then NaNs, that is numerics (an all-masked row can cause it, but the broader pattern is a learning-rate or fp16 issue covered elsewhere). If a single sequence run alone is wrong, padding is exonerated — the entire diagnostic value of "run it unbatched" is that batch size 1 removes padding from the picture. That single contrast — batched wrong, unbatched right — is the cleanest possible isolation, and it is the first test you should run when a decoder-only model misbehaves in a batched setting. Use the [taxonomy and decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) to confirm you are in the model-code/data place before you spend time here, and the [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full symptom→suspect→test→fix flow this post instantiates.

## 14. Key takeaways

- **Train right, generate left.** Decoder-only generation reads the last position for the next token, so the last position must be a real token — left-pad. Training is teacher-forced over all positions, so right-pad. Backwards in either direction breaks that phase.
- **Batched-disagrees-with-unbatched is the master test.** A deterministic decoder must give the same output batched or alone. If it does not, padding geometry is leaking into the computation — start with the padding side, then the mask, then position ids.
- **An unmasked pad steals softmax mass.** A pad key with a finite score takes a share of the attention distribution proportional to $e^{\text{score}}$ and subtracts it from real keys, shifting every output — which is why the symptom scales with how much padding you added.
- **The pad must be excluded three times.** From attention (the mask), from positional encoding (mask-aware `position_ids`), and from the loss (`-100`). Fixing one is not fixing the bug; reviewers who check only the attention mask have verified one of three.
- **Count your `-100` labels.** If pad positions leak into the loss, the model learns to predict padding, the reported loss reads $(1-f)$ times the real per-token loss, and a low number *looks* like progress while capacity bleeds away.
- **Packing needs a block-diagonal/varlen mask.** Concatenating documents under a plain causal mask lets later tokens read earlier unrelated ones; the test is that a document's loss is invariant to what it is packed with. FlashAttention varlen with `cu_seqlens` is the production fix.
- **`pad_token == eos_token` is a trap.** Keep exactly the first EOS per row in the loss and mask the rest, or add a distinct pad token; otherwise the model never learns to stop or learns to stop instantly.
- **Know your mask API.** 2D vs 4D, additive `0`/`-inf` vs boolean, keep-True vs ignore-True — passing right values in the wrong convention masks the wrong positions silently. Unit-test that a blocked key gets exactly zero weight.

## 15. Further reading

- Vaswani et al., "Attention Is All You Need" (2017) — the scaled dot-product attention and the masked-softmax mechanism every bug in this post is a violation of.
- Hugging Face `transformers` documentation — the `tokenizer.padding_side` attribute, the `attention_mask` contract, `DataCollatorForLanguageModeling`/`DataCollatorForSeq2Seq`, and the decoder-only right-padding generation warning.
- Dao et al., "FlashAttention" (2022) and the FlashAttention varlen / `cu_seqlens` API — the production path for document-aware packing without materializing a block-diagonal mask.
- PyTorch documentation — `torch.nn.functional.scaled_dot_product_attention` (mask conventions and `is_causal`), `nn.MultiheadAttention` `key_padding_mask` (note the inverted `True = ignore` polarity), and `nn.CrossEntropyLoss` `ignore_index`.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — why position ids enter as rotation angles, which is what makes left-padded `arange` position ids corrupt every real token's encoding.
- Within this series: [attention and masking bugs](/blog/machine-learning/debugging-training/attention-and-masking-bugs) (the general causal-mask and future-peek treatment), [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) (prompt-vs-completion `-100` masking), [tokenization bugs](/blog/machine-learning/debugging-training/tokenization-bugs) (special-token identity, the pad/eos collision), [train-infer mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms) (the broader works-in-training-breaks-in-generation gap), the [taxonomy and decision tree](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
