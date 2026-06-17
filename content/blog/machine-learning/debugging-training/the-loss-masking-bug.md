---
title: "The Loss-Masking Bug: Training on the Prompt and Other Wasted Runs"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Decode your label tensor, find the masking bug, and stop wasting finetune runs where the model trains on the prompt, never learns to stop, or quietly scores loss on nothing at all."
tags:
  [
    "debugging",
    "model-training",
    "finetuning",
    "llm",
    "nlp",
    "loss-masking",
    "huggingface",
    "trl",
    "pytorch",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-loss-masking-bug-1.png"
---

You started an instruction-tuning run on Friday. Loss came down from 2.6 to 1.9 over three epochs, the curve looked smooth, no NaNs, no spikes, the GPUs ran hot the whole time. By every dashboard you have, the run was healthy. Then you loaded the adapter, typed a question, and the model wrote back your question. It echoed the user's prompt, paraphrased it, and only then dribbled out a weak, half-finished answer that never stopped — it kept generating until it hit your `max_new_tokens` cap, trailing off into repeated tokens. You burned twelve GPU-hours and roughly \$30 of compute to teach a model to autocomplete prompts.

Nothing crashed because nothing was wrong with your optimizer, your learning rate, your model code, or your numerics. The bug was three lines deep in the data collator, in a single tensor you never looked at: `labels`. Somewhere between the chat template and the cross-entropy loss, the wrong tokens got marked as "score this" and the right tokens got marked as "ignore this." The loss you watched go down was the *average* loss over a mixture of tokens — most of which were the user's prompt, which the model can predict trivially because it can just copy them from the input. The completion, the part you actually cared about, was a rounding error in that average. The instruments lied because they were measuring the wrong thing.

This post is about that class of bug: **loss masking**. It is the single most common silent failure in supervised finetuning (SFT) of language models, and it is invisible to every instrument except one — the `labels` tensor itself, decoded. By the end you will be able to take any SFT run and, in under a minute, answer the only question that matters: *which exact tokens is this run computing loss on?* You will have a runnable label-mask decoder, a completion-only collator you understand line by line, a unit test that fails loudly when the mask is wrong, and a mental decision tree that routes you from a symptom (echoes the prompt, never stops, loss too low) straight to the masking failure that caused it.

![Diagram showing the supervised finetuning pipeline as a vertical stack from raw chat example through chat template, tokenizer, data collator setting labels to negative one hundred, the internal label shift, and cross-entropy that skips masked labels](/imgs/blogs/the-loss-masking-bug-1.png)

This is one of the six places a bug can hide in a training run — **data, optimization, model code, numerics, systems, evaluation** — and loss masking sits squarely in *data*, specifically in the label construction that the data pipeline owns. The reason it survives so long is that it does not announce itself in any of the other five places. Your gradients are finite, your LR is fine, your model graph is correct, your GPUs are saturated. The bisection that the rest of this series teaches — overfit one batch, read the instruments — needs one extra instrument here: *read the labels*. Let me show you exactly how, starting from the mechanism that makes the bug possible.

## 1. The mechanism: cross-entropy, `ignore_index`, and what a "label" really is

To debug loss masking you have to know precisely what the loss function does with each token. Language models are trained with **next-token prediction**: at every position in the sequence, the model outputs a probability distribution over the vocabulary, and the loss measures how surprised the model was by the token that actually came next. The loss for a single sequence of length $T$ is the average negative log-likelihood:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

That sum runs over every position $t$. The crucial question for our bug is: *does every position contribute?* If you compute loss naively, yes — every token in the sequence, prompt and completion alike, adds a term to that sum and therefore a term to the gradient. That is the default behavior, and for **pretraining** it is exactly what you want: you are teaching the model to predict every token of a giant corpus, so every token earns its loss.

For **instruction tuning** it is usually *not* what you want. Here a training example has two parts: a **prompt** (the system message, the user's instruction, any few-shot context) and a **completion** (the assistant's response you want the model to produce). You do not want to teach the model to generate users' questions — at inference time the user supplies the prompt, and the model's only job is the completion. Computing loss on the prompt spends gradient signal on a task you will never ask the model to do.

The mechanism that lets you exclude tokens is the `ignore_index` argument to cross-entropy. In PyTorch, `torch.nn.functional.cross_entropy` and `torch.nn.CrossEntropyLoss` both take `ignore_index`, which defaults to `-100`. Any position whose target label equals `-100` is **dropped from both the numerator and the denominator** of the loss. It contributes nothing to the sum and is not counted in $T$. Here is the contract, made concrete:

```python
import torch
import torch.nn.functional as F

# 5 positions, vocab size 10
logits = torch.randn(5, 10)
labels = torch.tensor([-100, -100, 3, 7, 2])  # first two ignored

loss = F.cross_entropy(logits, labels, ignore_index=-100)

# This is exactly equivalent to averaging over ONLY the 3 real labels:
mask = labels != -100
manual = F.cross_entropy(logits[mask], labels[mask])
assert torch.allclose(loss, manual)  # passes — denominator is 3, not 5
```

So `-100` is not a magic number; it is the **default `ignore_index`**, chosen because it is not a valid token id (token ids are non-negative). When you "mask" a token in SFT, you are setting its label to `-100`. The model still *attends* to that token (it is in `input_ids`, the model reads it as context), but the model is never *scored* on predicting it.

This gives us the precise definition of the loss-masking bug, in two halves:

1. **Masking the wrong tokens for loss.** You intended completion-only training but left the prompt's labels as real token ids, so the model is scored on predicting the prompt. Or the reverse — you masked too much and scored on nothing.
2. **Forgetting which tokens must be in the loss.** The EOS token must be a real label or the model never learns to stop; the completion's first token must be scored or the model never learns to start its answer; pad tokens must be masked or they inflate the loss with trivial predictions.

> The instrument for this entire class of bug is one line: **decode the tokens where `labels != -100`**. That decoded string should be *exactly* the completion plus EOS, nothing more and nothing less. Everything below is elaboration on that single check.

### Why training on the prompt dilutes the gradient

It is worth being quantitative about *why* training on the prompt is bad, because "it's just a bit of extra loss" undersells it. Consider a typical instruction example: a 180-token prompt and a 40-token completion. If you train on the full sequence, the loss is an average over 220 tokens, of which 180 — **82%** — are prompt. The gradient that updates your model is a weighted sum of per-token gradients, and the prompt tokens dominate that sum by sheer count.

Now, the prompt tokens are not adversarial; predicting them is a real (if useless) task. But they are *easy* to predict relative to the completion, because much of the prompt is boilerplate (system messages, role tokens, repeated phrasing) that the pretrained model already models well. So two things happen at once. First, the **magnitude** of the gradient is spread across 220 tokens instead of concentrated on the 40 you care about — the effective learning rate on the completion is diluted by roughly $40/220 \approx 0.18$, so you get about a fifth of the signal per optimizer step. Second, the model spends capacity *getting better at generating prompts*, which is not just wasted but can actively skew the model toward an "echo the input" behavior, exactly the symptom we opened with.

![Two-column before-and-after figure contrasting an unmasked run where one hundred eighty prompt tokens enter the loss against a masked run where the prompt is set to negative one hundred and all gradient lands on forty completion tokens plus EOS](/imgs/blogs/the-loss-masking-bug-2.png)

#### Worked example: the gradient dilution, in numbers

Take a batch of 8 examples, each with a 180-token prompt and a 40-token completion, on a model where the completion tokens have an average per-token loss of 2.0 and the prompt tokens (mostly boilerplate the pretrained model already knows) average 0.5.

With **no masking**, the reported batch loss is the token-weighted average:

$$\mathcal{L}_{\text{full}} = \frac{180 \cdot 0.5 + 40 \cdot 2.0}{220} = \frac{90 + 80}{220} \approx 0.77$$

With **prompt masked**, the loss is computed over completion tokens only:

$$\mathcal{L}_{\text{comp}} = \frac{40 \cdot 2.0}{40} = 2.0$$

Two consequences fall out of this. First, the **numbers are not comparable** — a masked run *starts at a higher loss* (2.0 vs 0.77) because it is no longer averaging in the easy prompt tokens. Engineers who "fix" the mask and then panic that "loss went up" are misreading this; the higher number is the *honest* number. Second, in the unmasked run, $80/170 \approx 47\%$ of the loss mass — and a similar share of the gradient — comes from the completion, but the *per-step update on completion behavior* is throttled because the optimizer is simultaneously chasing the 180 prompt tokens. You are paying for 220 tokens of compute to get 40 tokens of useful learning, and even that 40 is diluted. Over a 12-hour run, that is most of your money spent teaching the model nothing you will use.

### The gradient, made rigorous

It helps to see *why* the dilution is exactly proportional to the token fraction, because that proportionality is what lets you predict the effect before you run anything. The mean-reduced loss over a sequence with token set $S$ (the positions that are scored — for a masked run, just the completion) is

$$\mathcal{L} = \frac{1}{|S|}\sum_{t \in S} \ell_t, \qquad \ell_t = -\log p_\theta(x_t \mid x_{<t})$$

By linearity of the gradient, the parameter update direction is the average of the per-token gradients:

$$\nabla_\theta \mathcal{L} = \frac{1}{|S|}\sum_{t \in S}\nabla_\theta \ell_t$$

Now compare two runs on the *same* example. In the **masked** run, $S = C$ (the 40 completion positions) and the gradient is $\frac{1}{40}\sum_{t \in C}\nabla \ell_t$ — the pure completion direction. In the **unmasked** run, $S = P \cup C$ (220 positions), and the gradient is

$$\nabla_\theta \mathcal{L}_{\text{full}} = \frac{1}{220}\left(\sum_{t \in P}\nabla \ell_t + \sum_{t \in C}\nabla \ell_t\right) = \underbrace{\frac{40}{220}}_{0.18}\cdot\left(\tfrac{1}{40}\sum_{C}\nabla\ell_t\right) + \underbrace{\frac{180}{220}}_{0.82}\cdot\left(\tfrac{1}{180}\sum_{P}\nabla\ell_t\right)$$

The completion direction you actually want enters the update **scaled by 0.18** and is added to a prompt direction scaled by 0.82 that points somewhere unrelated. Two harms compound: the useful component is attenuated by the completion's token fraction, *and* it is perturbed by a large prompt component pulling the weights toward "be a better prompt generator." If the prompt and completion gradients happen to be roughly orthogonal (common, since they are different tasks), the effective step *along the completion direction* is about a fifth of what masking would give — so to first order you would need roughly $1/0.18 \approx 5\times$ as many steps to make the same progress on the completion, at five times the cost. That is the quantitative skeleton under "the prompt dilutes the gradient." The token fraction is not a vibe; it is the literal coefficient in front of the gradient you care about.

#### Worked example: how much does this cost in dollars and steps

Make it concrete with money. Suppose a clean, masked finetune reaches your target completion loss in 1,000 optimizer steps, and your setup runs at 4 steps/second on a single A100 you rent at \$2.00 per GPU-hour. The masked run finishes in 250 seconds of compute — call it \$0.14, ignoring data loading. Now run it unmasked. The completion gradient is attenuated by the 0.18 factor derived above, so to make comparable progress on the completion you need on the order of $1{,}000/0.18 \approx 5{,}500$ steps — *if* the prompt gradient were merely noise. In practice it is worse, because the prompt component actively pushes the model toward echoing input, so part of your "extra" steps are spent *undoing* that pull. Even taking the optimistic 5,500-step figure, that is 1,375 seconds and roughly \$0.76 — more than 5× the cost — to reach a *worse* model that also echoes prompts and may not stop. Scale those per-example numbers to a real 5,000-example, 3-epoch run on 8 GPUs and the gap is the difference between a \$30 run and a \$160 run that you then throw away. The masking fix is not a polish step; it is often a 5× efficiency multiplier hiding in three lines of the collator.

## 2. The internal label shift: the off-by-one that bites manual loss code

Before we fix any masking, you must understand a subtlety that trips up everyone who computes loss by hand: **the label shift**. Next-token prediction means position $t$ predicts the token at position $t+1$. So the "label" for the logits at position $t$ is `input_ids[t+1]`, not `input_ids[t]`.

Hugging Face causal LMs do this shift **internally**. When you pass `labels` to a model like `LlamaForCausalLM`, the model itself slices:

```python
# Inside transformers' CausalLMOutput loss computation (paraphrased):
shift_logits = logits[..., :-1, :].contiguous()   # drop the last position's logit
shift_labels = labels[..., 1:].contiguous()       # drop the first position's label
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

The consequence that matters for masking: **you pass `labels` aligned to `input_ids`, position for position, and the model handles the shift.** Your `labels` tensor should be the *same length* as `input_ids`, with `labels[i]` being either `input_ids[i]` (score this position) or `-100` (ignore it). You do **not** pre-shift the labels yourself. If you do, you double-shift, and every position now predicts the token two ahead — the model learns a garbled objective and loss plateaus high without ever erroring.

![Grid figure showing input ids on the top row and the internally shifted labels on the bottom row, with the assistant token Paris aligned so position t predicts token t plus one and prompt positions set to negative one hundred](/imgs/blogs/the-loss-masking-bug-3.png)

Let me make the alignment unambiguous. Suppose after tokenization a tiny example is:

| index | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `input_ids` token | `[BOS]` | `What` | `is` | `?` | `Paris` | `[EOS]` |
| role | prompt | prompt | prompt | prompt | completion | completion |
| `labels` (correct) | `-100` | `-100` | `-100` | `-100` | `Paris` | `[EOS]` |

You pass that 6-long `labels` tensor. Internally the model shifts: the logit at position 3 (the `?`) is scored against `labels[4]` = `Paris`, and the logit at position 4 (`Paris`) is scored against `labels[5]` = `[EOS]`. So the model is taught, "after the prompt ending in `?`, predict `Paris`; after `Paris`, predict `[EOS]` (stop)." That is exactly right. Notice the masking is on `labels` at the prompt *positions*, and because of the internal shift, the last prompt position (`?` at index 3) is what actually predicts the first completion token. **You must keep `labels[i] = -100` for prompt positions and `labels[i] = input_ids[i]` for completion positions, and let the shift do the rest.** This is the single most important alignment fact in the post; print it on your wall.

### When you compute loss manually

In a custom training loop where you call the model with `labels=None` and compute loss yourself — common in research code, RLHF reward models, and any time you need per-token loss — *you* are responsible for the shift, and now you can get it wrong in both directions:

```python
# CORRECT manual loss with shift and masking
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits                          # (B, T, V)

shift_logits = logits[:, :-1, :]                 # predict positions 1..T-1
shift_labels = labels[:, 1:]                     # targets are tokens 1..T-1

loss = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
```

Two failure modes here:

- **No shift** (`F.cross_entropy(logits.reshape(...), labels.reshape(...))`): every position is scored against *itself*, which is a trivial copy task — the model can achieve near-zero loss by learning the identity map at the output, and your "trained" model is useless. Symptom: loss crashes to near zero implausibly fast.
- **Double shift** (you pre-shifted `labels` *and* the model shifts, or you shift twice in your own code): every position predicts two tokens ahead, an impossible task on average, loss stays high and the model never improves coherently. Symptom: loss plateaus, generations are scrambled.

The fix is discipline: **either pass `labels` to the model and never shift, or compute loss yourself and shift exactly once.** Never both.

#### Worked example: tracing a double-shift loss that won't drop

A researcher ported an SFT loop from a tutorial into a custom trainer to add per-token loss logging. They kept the tutorial's manual shift *and* passed `labels` to the model, so the model shifted too. The symptom: train loss started at 6.1, fell to about 4.8 in the first few hundred steps, then sat there forever — no NaN, no spike, just a high plateau, and generations were word salad. They blamed the learning rate and tried five values; none helped, which is itself a tell (a pure LR problem usually responds *somehow* to a 10× change).

Trace one tiny sequence to see why no LR can save it. Take `input_ids = [BOS, A, B, C, EOS]` at indices 0–4, all of the completion scored. The model internally forms `shift_labels = labels[1:] = [A, B, C, EOS]` and scores its logits at positions 0–3 against them: position 0 (BOS) should predict A, position 1 (A) should predict B, and so on — correct. But the researcher *also* pre-shifted `labels` before passing them, so what reached the model was `[A, B, C, EOS, -100]`, and the *model's* internal shift turned that into `[B, C, EOS, -100]`. Now position 0 (BOS) is scored against B, position 1 (A) against C, position 2 (B) against EOS. The model is being asked to predict the token *two ahead*, which is genuinely unpredictable on average — there is no function of the prefix that reliably names the token-after-next better than chance for most positions. So the loss floors at the entropy of "token two ahead," which is high and irreducible. No learning rate fixes an *impossible* objective. The fix was deleting their manual shift (let the model own it); loss then dropped from 6.1 to 1.3 and generations became coherent. The general lesson: **when no learning rate helps and the loss floors high, suspect the objective itself — a shift or a mask — not the optimizer.**

## 3. The diagnostic that ends the debate: decode the labels

Everything above is theory. Here is the practice. The definitive test for any masking bug is to take one real batch, find the positions where `labels != -100`, and decode them back to text. What you see *is* what your model is being trained to produce.

```python
import torch

def decode_loss_targets(tokenizer, input_ids, labels):
    """Print exactly what the loss is computed on, per example.

    input_ids, labels: (B, T) tensors. labels uses -100 for ignored.
    """
    for b in range(input_ids.size(0)):
        ids = input_ids[b]
        lab = labels[b]

        loss_mask = lab != -100
        n_loss = int(loss_mask.sum())
        n_total = int((ids != tokenizer.pad_token_id).sum()) if tokenizer.pad_token_id is not None else ids.size(0)
        frac = n_loss / max(n_total, 1)

        # The tokens the loss is actually computed on:
        target_ids = ids[loss_mask]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=False)

        print(f"--- example {b} ---")
        print(f"loss tokens: {n_loss} / {n_total} non-pad  ({frac:.1%})")
        print(f"LOSS IS COMPUTED ON: {target_text!r}")
        # Does it end with EOS?
        has_eos = (target_ids[-1].item() == tokenizer.eos_token_id) if n_loss else False
        print(f"ends with EOS: {has_eos}")
```

Run it on one batch from your actual dataloader, not on a toy:

```python
batch = next(iter(trainer.get_train_dataloader()))
decode_loss_targets(tokenizer, batch["input_ids"], batch["labels"])
```

This one function answers nearly every masking question. Read the output against this checklist:

- **`LOSS IS COMPUTED ON:` should print exactly the assistant's response, plus the EOS token.** If you see the system prompt, the user's question, or role tokens like `<|user|>` in there, your **prompt is not masked**.
- **`loss tokens` fraction should be plausible** — for a 180-token prompt and 40-token completion, you expect roughly 18%. If it reads **100%**, nothing is masked (you are training on everything). If it reads **0%**, everything is masked (you are training on nothing, and your loss is `nan` or a meaningless constant).
- **`ends with EOS` should be `True`.** If it is `False`, the model is never scored on stopping, and at inference it will ramble.

![Matrix figure with four masking failure rows and columns for symptom, confirming test, and fix, mapping prompt not masked to prompt echo, template not found to zero or full loss fraction, missing EOS to never stopping, and unmasked padding to inflated loss](/imgs/blogs/the-loss-masking-bug-4.png)

I cannot overstate how much grief this 20-line function saves. People spend days tweaking learning rates and data ratios for a model that "won't learn the format," when a single decode would have shown that the loss was being computed on the *user's* turns the entire time. **Decode the labels before you debug anything else.**

#### Worked example: reading a real decode and routing the bug

Here is actual output from a broken run I want you to learn to read:

```bash
--- example 0 ---
loss tokens: 214 / 214 non-pad  (100.0%)
LOSS IS COMPUTED ON: '<|system|>\nYou are a helpful assistant.<|user|>\nWhat is the capital of France?<|assistant|>\nThe capital of France is Paris.'
ends with EOS: False
```

Two bugs at once, both visible in three lines. The fraction is **100%**, so nothing is masked — the loss includes the system prompt and the user turn; this run is training on the prompt. And `ends with EOS: False`, so even the completion that *is* scored never includes a stop token; the model will not learn to stop. The decision tree below routes both: a decode that "shows the prompt text" means *prompt not masked*, and a target span that does not end in EOS means *EOS not in labels*. You fix both in the collator, re-run the decode, and confirm:

```bash
--- example 0 ---
loss tokens: 11 / 214 non-pad  (5.1%)
LOSS IS COMPUTED ON: 'The capital of France is Paris.<|im_end|>'
ends with EOS: True
```

Now the loss is computed on exactly the assistant's text plus the end token. The fraction (5.1%) is low because this is a short completion behind a long template, which is expected. *This* is what a correctly masked example looks like.

![Tree figure that starts from decoding the labels where they are not negative one hundred and branches to shows prompt text leading to prompt not masked, shows nothing leading to template missed, and shows completion leading to a check for whether EOS is present](/imgs/blogs/the-loss-masking-bug-5.png)

### Log the loss-token fraction as a live instrument

The one-time decode is the diagnostic; the **loss-token fraction logged every step** is the alarm that catches a masking regression mid-run. It is one line in a custom loop or a callback in `Trainer`, and it costs nothing:

```python
class LossTokenFractionCallback:
    """Log the fraction of tokens contributing to loss each step.

    A sudden jump to ~100% means a mask broke (training on the prompt);
    a drop toward 0% means the response template stopped matching.
    """
    def on_step(self, batch, logger):
        labels = batch["labels"]
        scored = (labels != -100).float()
        valid = (batch["attention_mask"] == 1).float()
        frac = (scored.sum() / valid.sum().clamp(min=1)).item()
        logger.log({"loss_token_fraction": frac})
```

Pin a horizontal reference line on that chart at your expected fraction (computed from typical prompt/completion lengths) and you have a tripwire. If a new data source with a different template lands in epoch 2 and stops matching the collator, the fraction craters and you see it immediately, rather than discovering it weeks later from a bad model. What should the line sit at? It depends entirely on your prompt-to-completion ratio:

| Scenario | Typical loss-token fraction | Reads wrong if |
|---|---|---|
| Long system prompt, short answer | 3–10% | 0% (template miss) or 100% (no mask) |
| Balanced instruction + answer | 15–35% | jumps to ~100% mid-run |
| Long completion, short prompt | 50–80% | drops toward 0% |
| Multi-turn, only last turn scored | 5–15% | scores earlier assistant turns too |
| Pretraining / domain adaptation | ~100% (minus padding) | drops below ~90% unexpectedly |

The exact number is less important than its *stability*: a healthy run holds the fraction roughly constant; a masking bug shows up as a step change. Watching this one scalar is the cheapest insurance against the most expensive bug in finetuning.

## 4. Doing it right in `trl`: `DataCollatorForCompletionOnlyLM`

The modern, supported way to get completion-only masking in Hugging Face is `trl`'s `DataCollatorForCompletionOnlyLM`. It is a thin, clever wrapper around the standard language-modeling collator: it tokenizes your formatted examples normally, then **searches for a "response template"** — the token sequence that marks where the assistant's turn begins — and sets every label *before* that template to `-100`. Everything from the template onward keeps its real label and is scored.

```python
from transformers import AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    train_dataset=dataset,
    args=SFTConfig(max_seq_length=2048, packing=False),
    data_collator=collator,
)
```

This works beautifully — *when the template matches*. And the most common, most maddening bug in this entire post is that **the template does not match the tokenized stream**, and the collator fails *silently*.

![Graph figure showing the completion-only collator searching tokenized input ids for the response template, branching to template found at index k leading to masking tokens up to k, or template not found leading to the whole example masked and loss zero](/imgs/blogs/the-loss-masking-bug-7.png)

### Why the template silently fails to match

The collator does not search the *string* `"<|start_header_id|>assistant..."`; it searches the *token id sequence* you get from tokenizing that string. The trap is that **tokenization is context-dependent**: the token ids for a template can differ depending on what comes immediately before it. A leading space, a preceding newline, or a different BOS handling can cause the tokenizer to produce a different first token id for the template in isolation than it does inside the full example. When that happens, the collator's id-subsequence search finds *no match*.

What does it do when it finds no match? In older `trl` versions it would, depending on the path, either mask the **entire** example (every label becomes `-100`, so the example contributes *zero* loss) or mask **nothing** (the whole thing is scored, prompt included). Newer versions warn, but warnings scroll off in a long training log and nobody reads them. Either way the symptom is downstream: your loss is suspiciously low (lots of masked examples), or the model trains on prompts (nothing masked), and you have no idea why.

The robust fix is to give the collator the template **as token ids you extracted from a real tokenized example**, not as a raw string. The `trl` docs recommend this when you hit mismatches:

```python
# Tokenize the template the way it appears mid-sequence, then strip the
# leading BOS the tokenizer adds, so the ids match the in-context form.
response_template_ids = tokenizer.encode(
    "\n<|start_header_id|>assistant<|end_header_id|>\n\n",
    add_special_tokens=False,
)[2:]  # drop the first 1-2 ids that only appear at the seam

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,  # pass IDS, not a string
    tokenizer=tokenizer,
)
```

But never trust the fix blindly. **Always run the decoder from §3 on a batch built by this collator** and confirm the loss span is exactly the completion. If `loss tokens` reads 0% on every example, your template ids are still wrong.

#### Worked example: the template that masked everything

A team finetuning a Mistral model reported their SFT "did nothing" — three epochs, eval loss flat at 1.9 the whole time, generations unchanged from the base model. The loss curve was smooth and looked fine, just high. They suspected the learning rate (1e-5, reasonable), then the data (5,000 clean examples), then LoRA rank. All red herrings.

The decode took ten seconds:

```bash
loss tokens: 0 / 196 non-pad  (0.0%)
LOSS IS COMPUTED ON: ''
ends with EOS: False
```

**Zero** loss tokens on every example. Their `response_template` string tokenized to ids that never appeared in the stream because their chat template put a newline before `[/INST]` that shifted the first token id. So the collator masked *everything*, the loss was computed over an empty set on most examples, and the optimizer received essentially no gradient — the model never moved off the base weights. "Eval loss flat at 1.9" was the base model's loss, unchanged, for three epochs. They had spent \$200 of compute on a no-op. Passing the template as extracted token ids fixed it; the next decode showed `loss tokens: 31 / 196 (15.8%)` and the completion text, and the finetune finally took.

### When to train on the prompt anyway

Completion-only masking is the right default for instruction tuning, but it is not a universal law, and knowing the exceptions keeps you from "fixing" a mask that was correct. Train on the **full sequence** (prompt included) when:

- **You are continued-pretraining or domain-adapting**, not instruction-tuning. If the goal is to teach the model a new domain's language (legal text, a codebase, a new language), every token is signal and you want the standard `DataCollatorForLanguageModeling` with no prompt mask. There is no "prompt" and "completion" — it is all corpus.
- **The "prompt" is short and you want the model to learn the format too.** For some structured-generation tasks where the input is a brief, fixed schema, training on it is cheap and can help the model internalize the format. This is a judgment call; the cost is the dilution from §1, so it only makes sense when the prompt is a small fraction of the sequence.
- **You are training a base model from scratch.** Pretraining is next-token prediction over everything, full stop. Completion-only masking is a *finetuning* technique.

The decision rule: **mask the prompt when the prompt is something the user supplies at inference and the model should never generate.** Score the prompt when you genuinely want the model to learn to produce that text. Most chat/instruction finetuning is the former; most pretraining and domain adaptation is the latter. The two standard collators line up with these two cases:

| Collator | What it masks | Use when |
|---|---|---|
| `DataCollatorForLanguageModeling(mlm=False)` | Only padding (labels = input shifted) | Pretraining, continued pretraining, domain adaptation |
| `DataCollatorForCompletionOnlyLM` | Padding **and** everything before the response template | Instruction tuning, chat SFT — the common case |

If you reach for `DataCollatorForLanguageModeling` on an instruction-tuning task, you have *chosen* to train on the prompt — which is the §1 bug if you did not mean to. The collator is doing exactly what you told it; the bug is the choice. This is why the decode in §3 is the arbiter: it shows you what you actually configured, not what you intended.

## 5. The EOS bug: a model that never learns to stop

A completion-only mask can be perfect and you can *still* ship a model that rambles forever, because of a separate, sneaky requirement: **the EOS token must be in the labels and must be scored.** Stopping is a learned behavior. The model learns to emit EOS only if, during training, the EOS token was a real (non-`-100`) label that followed the completion. If your data formatting drops EOS, or appends it but then your mask covers it, the model is never taught that responses end, and at inference it generates until it hits the length cap.

There is a clean piece of science under "the model learns to stop." At every generation step the model emits a distribution over the vocabulary, and EOS is just one token in that vocabulary. The probability the model assigns to EOS at the end of a response, $p_\theta(\text{EOS} \mid \text{response})$, is shaped entirely by the training signal: gradient descent raises the probability of whatever token actually followed in the data. If EOS *never* followed a completion in your scored labels, the gradient never once pushed $p_\theta(\text{EOS})$ up at that context, so it stays at its (tiny) pretrained baseline. At inference, greedy or sampled decoding keeps picking non-EOS tokens because EOS has negligible probability, and the model runs until the length cap. The reason this is invisible in the loss curve is that EOS is a single token among hundreds in a completion — even if every EOS is missing from the labels, it changes the average loss by less than one token's worth, far below the noise floor of the curve. The behavior is catastrophic at inference but imperceptible in training loss. *That* asymmetry — a tiny loss effect, a total behavioral failure — is the signature of the EOS bug, and it is why you must check it explicitly rather than trust the loss.

There are three distinct ways to break this:

1. **EOS never added.** Your formatting function builds `prompt + completion` but forgets to append the tokenizer's EOS. The sequence has no stop token at all. (Many chat templates add EOS for you via the turn-end token; raw concatenation does not.)
2. **EOS added but masked.** You append EOS, but your masking logic marks the EOS position as `-100` — for instance, an off-by-one in your completion-span computation that stops one token short.
3. **The wrong EOS.** Chat models often use a turn-end token (`<|im_end|>`, `<|eot_id|>`) as the functional stop, distinct from the tokenizer's classic `</s>`. If you append `</s>` but the model is supposed to stop on `<|eot_id|>`, the generation config and the training disagree.

The diagnostic is built into the decoder from §3: the `ends with EOS` line. If it is `False`, you have one of these three. To distinguish them, also check whether the EOS id appears *anywhere* in `input_ids`:

```python
eos_id = tokenizer.eos_token_id
for b in range(input_ids.size(0)):
    ids = input_ids[b]
    lab = labels[b]
    in_input = (ids == eos_id).any().item()
    in_labels = ((lab == eos_id)).any().item()  # eos that is scored
    print(f"ex {b}: EOS in input={in_input}  EOS scored={in_labels}")
```

- `EOS in input=False` → case 1, you never added it. Fix the formatting to append EOS (or the correct turn-end token).
- `EOS in input=True` but `EOS scored=False` → case 2, you added it but masked it. Fix the span so the EOS label is real.
- Both `True` but the model still rambles → check case 3, that the EOS you scored matches the one in `generation_config.eos_token_id` at inference.

#### Worked example: the off-by-one that ate the EOS

A finetune produced perfectly good answers that simply never terminated — every response ran to the 256-token cap and then repeated. The completion-only mask looked right; the decode showed the correct completion text. But:

```bash
LOSS IS COMPUTED ON: 'The capital of France is Paris.'
ends with EOS: False
```

The completion was scored, but the EOS that should follow `Paris.` was not. The bug was in a hand-rolled span computation: `completion_end = prompt_len + completion_len`, used as `labels[prompt_len:completion_end]`. But `completion_len` was computed *before* appending EOS, so the slice stopped exactly at the EOS token and left its label `-100`. One character — changing `completion_len` to `completion_len + 1`, or simpler, computing the span from the actual tokenized length — fixed it. After the fix, `ends with EOS: True`, and the model's stop rate on a 200-example eval went from 39% to 99%. The before-and-after table at the end of this post records exactly this kind of multi-instrument swing.

## 6. Padding tokens leaking into the loss

The mirror image of "EOS must be scored" is "**pad must not be scored**." When you batch sequences of different lengths, the collator pads the short ones to the batch's max length with the pad token. Those pad positions are not real data; predicting them teaches the model nothing useful and, worse, *inflates* the loss denominator with trivially-predictable tokens, deflating your loss number and adding a spurious "predict pad after pad" signal.

The fix is mechanical: **every pad position must have label `-100`.** The standard `DataCollatorForLanguageModeling` and the `trl` collators do this for you when the tokenizer's `pad_token_id` is set correctly. The bug appears when:

- **`pad_token` is unset and you set `pad_token = eos_token`.** This is a *very* common workaround for models (like base Llama) that ship without a pad token. Now `pad_token_id == eos_token_id`, and naive masking logic that masks "where `input_ids == pad_token_id`" will *also mask your real EOS tokens* — reintroducing the never-stops bug from §5. Conversely, logic that scores EOS will also score pads.
- **You build labels by copying `input_ids` and only masking the prompt**, forgetting that the right-padding region is still real token ids in the labels.

This is the most insidious interaction in the post, so let me be explicit. If `pad_token_id == eos_token_id`, you cannot distinguish pad from EOS by token id alone; you must distinguish them by *position* (use the `attention_mask`: positions with `attention_mask == 0` are padding) or by setting a *distinct* pad token. The robust pattern:

```python
# Distinct pad token avoids the pad==EOS ambiguity entirely.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))

# When building labels, mask padding by the attention mask, not by id:
labels = input_ids.clone()
labels[attention_mask == 0] = -100   # all padding -> ignored
labels[prompt_mask] = -100           # prompt -> ignored
# completion + its EOS keep real labels
```

If adding a pad token is not an option (it changes the embedding matrix, which complicates merging/serving), then **mask by `attention_mask`, never by comparing to `pad_token_id`**, precisely so that real EOS tokens — which share the id — survive in the labels.

| Masking strategy | Pad masked? | Real EOS survives? | Safe when |
|---|---|---|---|
| Mask `input_ids == pad_token_id`, pad≠EOS | yes | yes | distinct pad token set |
| Mask `input_ids == pad_token_id`, pad==EOS | yes | **no — bug** | never (masks your EOS) |
| Mask `attention_mask == 0` | yes | yes | always — recommended |
| No pad masking | **no — bug** | yes | never (loss inflated) |

The decoder from §3 catches this too: if `loss tokens` is much *higher* than your completion length, or if the decoded target text ends in a run of pad tokens, your padding is leaking into the loss.

There is a second-order effect worth naming, because it explains why an unmasked-padding bug can make a model *worse* and not merely waste a little compute. When right-padding leaks into the loss, the model is scored on "predict pad given pad given pad ...," a trivially learnable pattern (after the first pad, every token is pad). The model learns it instantly and with high confidence. The danger is at the *seam*: the last real token is followed by the first pad, so the model is also taught "after the real content, predict pad." If your pad token is distinct and never appears at inference, this is harmless noise. But if `pad_token == eos_token` (the §6 workaround), then "predict pad after content" *is* "predict EOS after content" — which is accidentally the right lesson, masking the EOS bug — or, if your masking logic instead removed those, you have suppressed the only EOS signal you had. The two bugs interact in a way that can make the model's stop behavior depend on padding accidents in your batches. The robust escape is, again, to mask by `attention_mask` and use a *distinct* pad token, so padding and stopping are never entangled.

#### Worked example: the loss that looked too good because of padding

A team batched a dataset with wildly varying lengths (a few 1,800-token examples mixed with many 60-token ones) and right-padded to the batch max. They had masked the prompt correctly but built `labels` by cloning `input_ids` and masking only the prompt span — so the padding region kept real `pad_token` labels. Their train loss looked *great*: 0.4 after one epoch, far lower than comparable runs. They almost shipped it. The decode told the truth:

```bash
loss tokens: 1247 / 71 non-pad  (loss tokens far exceed real tokens)
LOSS IS COMPUTED ON: 'Sure, here is the answer.<|pad|><|pad|><|pad|>...(1180 pads)...'
```

The loss span was dominated by pad tokens — 1,180 of them in this batch — each contributing a near-zero loss term (predicting pad after pad is trivial). The 0.4 "loss" was the average of a handful of real completion losses (around 1.6 each) and a thousand pad losses (around 0.01 each), pulled down by the pads. The model was *fine* on the real content but the metric was a lie, and worse, the gradient was diluted across the pad predictions just like the prompt-dilution of §1. Masking the padding to `-100` (by `attention_mask`) lifted the reported loss to its honest 1.6 and concentrated the gradient on real tokens. The shipping decision flipped from "great, deploy" to "this is the same as the other runs, keep iterating" — which was the correct call.

## 7. Sequence packing: labels are fine, attention is the trap

To use GPUs efficiently, SFT pipelines often **pack** multiple short examples into one long sequence — concatenate example A, example B, example C until you fill the context window — instead of padding each to the max length. Packing is great for throughput, but it introduces two distinct concerns, and people conflate them.

![Two-column before-and-after figure showing naive packing where document B attends back to document A and contaminates the loss at the seam, versus block-diagonal masking where EOS marks each boundary and attention stays within a document](/imgs/blogs/the-loss-masking-bug-6.png)

**Concern one: the labels across the boundary.** When you pack, the label tensor is just the concatenation of the per-example labels, with `-100` on each example's prompt and real labels on each completion (plus each example's EOS). This part is usually *correct by construction* — each example contributes its own masked labels, and the EOS between documents is a genuine label the model should predict (it learns "this document ended"). So packing does *not* inherently break loss masking. The common misconception that "packing trains across document boundaries" is, for the *labels*, false: a position at the end of document A predicts document A's EOS, and the *first* position of document B predicts document B's first token — those are correct, independent objectives.

**Concern two: attention bleed across the boundary.** This is the real packing bug. By default, a packed sequence uses a single causal mask over its whole length, which means tokens in document B can *attend back to* document A. Document B's first token sees all of document A as context. This is semantically wrong — A and B are unrelated examples — and it pollutes the representations and therefore the gradient on B's completion. The fix is a **block-diagonal attention mask** (also called a "reset" or "document" mask) that prevents attention from crossing example boundaries, so each packed document attends only to itself.

This is exactly the territory of the padding-and-attention-mask post in this series, [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms), which goes deep on how masks misalign — read it for the attention side. For our purposes the key facts are:

- Modern `trl` `SFTTrainer` with `packing=True` and recent transformers supports passing `position_ids` and using FlashAttention's variable-length (`varlen`) path, which gives you the block-diagonal behavior *if you enable it*. Older setups silently let attention bleed.
- The bleed does **not** show up in the labels, so the §3 decoder *will not catch it*. You diagnose it separately: check whether the run sets `attn_implementation="flash_attention_2"` with packing and whether `position_ids` reset at each document boundary (they should restart at 0 for each packed example, not count monotonically across the whole pack).

```python
# Inspect a packed batch's position_ids — they should reset per document.
batch = next(iter(trainer.get_train_dataloader()))
print("has position_ids:", "position_ids" in batch)
if "position_ids" in batch:
    pids = batch["position_ids"][0]
    # A reset looks like ...,30,31,0,1,2,... at each doc boundary.
    resets = (pids[1:] < pids[:-1]).sum().item()
    print(f"document boundaries (position_id resets): {resets}")
```

If `position_ids` is absent or monotonic across the whole pack while you are packing, attention is bleeding across documents. The practical guidance: **if you pack, enable the variable-length attention path and verify `position_ids` reset**; if you cannot, prefer padding with the completion-only collator (you lose some throughput but avoid the bleed). And regardless — *run the label decoder*, because packing also makes it easy to lose an EOS at a boundary or double-count a template.

To see why the bleed matters quantitatively, think about what a single attention head computes for the first token of document B. With a plain causal mask over the whole pack, that token's query attends over the keys of *every* preceding token, including all of document A. The softmax over attention scores will put *some* weight on A's tokens — even if small, it is nonzero — so B's representation is a blend that includes A's content. During the backward pass, the gradient on B's completion loss flows back through those cross-document attention weights and nudges the model based on a context that will *never* occur at inference (at inference, B is its own request and never sees A). You are training the model on an input distribution you will not serve, which is a form of train–serve skew baked into the attention pattern. The block-diagonal mask sets the cross-document attention scores to $-\infty$ before the softmax, so their weight is exactly zero and the gradient cannot leak. The cost of the correct behavior is essentially free with FlashAttention's variable-length kernels, which is why "pack with varlen attention" is the modern recommendation and "pack with a plain causal mask" is a latent bug.

One more packing subtlety on the *label* side, since people over-rotate on it: when you concatenate documents, you usually insert each document's EOS between them, and that EOS *should* be a scored label — it teaches "this document ended," which is the same stopping signal of §5. The mistake is *dropping* the EOS at the join (so two documents run together with no boundary token) or *masking* it (so the boundary is invisible to the loss). Either way the model's sense of "where a response ends" degrades under packing specifically. The decode catches it: in a packed batch, you should see EOS appear at each document boundary in the scored labels. If your packed `LOSS IS COMPUTED ON:` is one long run with no EOS until the very end, your boundaries lost their stop tokens.

## 8. Special and template tokens counted in loss

A subtler variant: even with the prompt correctly masked, you can end up scoring **structural tokens** you did not mean to teach. Chat templates wrap content in role markers (`<|im_start|>assistant`, `<|eot_id|>`, `<|end_of_turn|>`). The completion-only collator masks everything before the *response template*, but the response template *itself* and the trailing turn-end token are right at the boundary, and exactly which of them fall inside the loss span depends on where the template match lands.

Usually you *do* want the turn-end token scored (it is the model's stop signal, the EOS of §5). But you usually do *not* want the `<|im_start|>assistant\n` header scored — that is structure the inference harness supplies, not something the model should generate. If your response template is set so the match lands *before* the header, those header tokens get real labels and the model spends capacity learning to emit role markers, which can cause it to hallucinate `<|im_start|>` mid-response at inference.

The diagnostic, again, is the decoder: look at the *first* tokens of `LOSS IS COMPUTED ON:`. If it starts with `assistant\n\n` or a role marker, your template boundary is off by a few tokens. Adjust the `response_template` to include the header, so the header is masked and the loss span starts at the assistant's actual first content token. This is a small effect compared to the prompt-masking bug, but on heavily-templated data it is the difference between a clean model and one that occasionally blurts out structural tokens.

| Token type | Should be in loss? | Why |
|---|---|---|
| System / user content | No | inference-time input, model never generates it |
| Assistant role header | No (usually) | harness supplies it; scoring it causes hallucinated markers |
| Assistant content | **Yes** | the actual thing you want generated |
| Turn-end / EOS | **Yes** | the learned stop signal |
| Padding | No | not real data; inflates loss |

## 9. A unit test that fails loudly when the mask is wrong

Decoding by eye is the right *first* move, but you do not want to re-decode by hand on every dataset change. Encode the §3 checks into a **unit test** that runs on a sample of your actual training data and asserts the invariants. This turns a silent, week-eating bug into a red CI check.

```python
import torch

def test_completion_only_masking(collator, tokenizer, examples, n=16):
    """Assert that for each example, the loss span is exactly the
    assistant completion + EOS, and nothing from the prompt."""
    batch = collator([collator.tokenizer(e) if not isinstance(e, dict) else e
                      for e in examples[:n]])
    input_ids, labels = batch["input_ids"], batch["labels"]

    for b in range(input_ids.size(0)):
        ids, lab = input_ids[b], labels[b]
        loss_mask = lab != -100

        # 1) SOMETHING is scored (not 0%, not 100%)
        frac = loss_mask.float().mean().item()
        assert frac > 0.0, f"ex {b}: nothing is scored (template not matched?)"
        assert frac < 1.0, f"ex {b}: everything is scored (prompt not masked?)"

        # 2) The scored span ends with EOS / turn-end
        scored_ids = ids[loss_mask]
        assert scored_ids[-1].item() in (tokenizer.eos_token_id, TURN_END_ID), \
            f"ex {b}: loss span does not end with a stop token"

        # 3) No prompt content is scored: decode and assert the user text
        #    does NOT appear in the scored span
        scored_text = tokenizer.decode(scored_ids, skip_special_tokens=False)
        assert "<|user|>" not in scored_text and "system" not in scored_text.lower(), \
            f"ex {b}: prompt/role tokens leaked into the loss span"

    print(f"PASS: completion-only masking verified on {min(n, input_ids.size(0))} examples")
```

Wire this into your test suite and run it whenever you change the chat template, the tokenizer, the collator, or the dataset format — the four things that break masking. The three asserts map one-to-one onto the three failure modes: nothing scored (template miss), everything scored (prompt leak), no stop token (EOS bug). When one fires, you know *which* bug before you even open the data.

This is the embodiment of a principle from the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): the cheapest place to catch a data bug is a deterministic assert on a few real examples, run before the expensive training starts. A masking test costs milliseconds and saves GPU-days.

## 10. The full before→after: what the instruments read

Let me put the whole thing together on one finetune so you can see every instrument move at once. The setup: a 7B base model, 5,000 instruction examples, LoRA finetune, 3 epochs. The buggy run trained on the full sequence (no prompt mask) and never appended EOS. The fixed run masked the prompt with a verified completion-only collator and included the turn-end token.

![Matrix figure comparing the buggy run and the fixed run across four instruments, with loss-token fraction dropping from eighty-two percent to eighteen percent, eval loss falling from one point nine four to one point three one, prompt-echo rate dropping, and stop-at-EOS rate rising to ninety-nine percent](/imgs/blogs/the-loss-masking-bug-8.png)

| Instrument | Buggy run | Fixed run | What it means |
|---|---|---|---|
| Loss-token fraction | 82% | 18% | Buggy run scored mostly prompt; fixed run scores the completion |
| Train loss (final) | 0.71 | 1.34 | Fixed run's *higher* number is honest (no easy prompt tokens) |
| Eval loss on completions | 1.94 | 1.31 | The metric that matches the task improves only after the fix |
| Prompt-echo rate (200 evals) | 37% | 2% | Model stops parroting the input |
| Stops at EOS (200 evals) | 39% | 99% | Model learns to terminate |
| Useful gradient share | ~18% | ~100% | Compute spent on the task you care about |

Read the trap in that table: the **buggy run has a *lower* train loss** (0.71 vs 1.34). If you only watched train loss, the buggy run looks *better*. That is the entire reason this bug survives — the headline instrument points the wrong way. The honest instrument is **eval loss computed on completions only** (which is masked the same way for both runs, so it is comparable), and it tells the true story: 1.94 buggy → 1.31 fixed. Then the behavioral metrics — prompt-echo rate and stop rate — confirm what the user actually experiences.

How would you *measure* these honestly in your own run? The eval loss must be computed with the *same* completion-only masking for both models so the denominator matches (otherwise you are comparing averages over different token sets — the §1 trap). Prompt-echo rate: generate on a held-out prompt set and count responses whose first 20 tokens have high overlap with the input. Stop rate: count generations that emit EOS before the length cap. All three are cheap to compute and all three move together when the mask is fixed, which is the signature of *the masking being the real bug* rather than a coincidence.

## 11. Case studies and real signatures

These are patterns I have seen repeatedly, distilled to the symptom and the one-line confirming test.

**The "SFT does nothing" no-op (response template mismatch).** A finetune where eval loss is flat across all epochs at the *base model's* loss, and generations are unchanged. This is the §4 silent-template-failure: the completion-only collator masked every token, the optimizer got near-zero gradient, the weights never moved. The tell is `loss tokens: 0%` on the decode. This is widely reported on the `trl` issue tracker and is the single most common SFT bug filed against the library — the documented fix is to pass the response template as token ids extracted from a real example, then verify with a decode.

**The prompt-echoing chatbot (no prompt mask).** A model that restates the user's question before answering and produces flabby completions. This is the opening story of this post and the §1 mechanism: training on the full sequence, gradient diluted across the prompt, model partly optimized to *generate prompts*. The tell is `loss tokens: ~100%` and the decoded span containing the user turn. Masking the prompt sharpens the finetune; in the §10 table it took prompt-echo from 37% to 2%.

**The model that won't shut up (EOS not learned).** Coherent answers that never terminate, running to the token cap and repeating. Two root causes converge here: EOS never appended (§5 case 1) or EOS masked out (§5 case 2 / §6 pad==EOS). The tell is `ends with EOS: False`. This is also a known interaction of the `pad_token = eos_token` workaround on base Llama-family models — set a distinct pad token or mask by `attention_mask`, and the model learns to stop.

**Cross-document bleed under packing.** A packed run where completions are subtly worse than the same data padded, with no label-level difference. This is the §7 attention bleed: labels are fine, but document B attended to document A. The tell is *not* in the decode — it is `position_ids` that fail to reset at boundaries, or `attn_implementation` not set to the varlen path. The fix is the block-diagonal mask; the symptom is quiet enough that many teams never notice they are leaving quality on the table.

**Multi-turn masking that scores the wrong turns.** A conversational dataset with several user/assistant turns per example, finetuned so the model gives stilted, repetitive replies and sometimes generates the *user's* next turn unprompted. The root cause is a multi-turn masking mistake: in a multi-turn conversation, you typically want to score *every assistant turn* and mask *every user turn*, but a naive single-`response_template` collator masks only up to the *first* assistant turn and scores everything after — including the later user turns. So the model is trained to generate user messages, which is why it sometimes continues the conversation by speaking as the user. The decode is unmistakable: `LOSS IS COMPUTED ON:` contains user turns from the middle of the conversation. The fix is per-turn masking — mask each user span individually and score each assistant span — which `trl`'s instruction-template support and the `DataCollatorForCompletionOnlyLM` with both an instruction *and* response template can express. The general signature: a multi-turn model that role-confuses is almost always a multi-turn masking bug, not a model-capacity problem.

A note on accuracy: the *direction* and *mechanism* of every case above is well-established and reproducible; the specific numbers in the worked examples and the §10 table are representative figures from this class of run, chosen to illustrate the relative movements (lower train loss but higher eval loss on the bug, echo rate collapsing on the fix), not measurements from one canonical published benchmark. Where a number is a known library behavior (the `-100` default `ignore_index`, the internal `logits[..., :-1]` / `labels[..., 1:]` shift), it is exact and you can read it in the PyTorch and transformers source.

## 12. When this is (and isn't) your bug

Loss masking is the right suspect when the symptom is about *what the model produces*, not *whether the run is numerically healthy*. Be decisive:

- **It IS a masking bug when:** the model echoes the prompt; it never stops; an SFT run leaves eval loss flat at the base model's value (no-op); train loss is *implausibly low* and generations are bad; or the model emits structural tokens (`<|im_start|>`) mid-response. All of these are about the *target tokens*, which is exactly what masking controls. **First move: decode the labels.**
- **It is NOT a masking bug when:** the loss goes to NaN (that is numerics — see the NaN-hunting track, not labels); the loss spikes then diverges (that is the learning rate or a bad batch); the LoRA adapter produces *identical* logits to the base model with no change at all (that is a PEFT graph problem — the adapter is not in the forward pass — not masking); or the model trains fine but fails only at inference with a *different* prompt format (that is chat-template skew, covered in [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs)). A masking bug specifically produces a model that *trained on the wrong tokens* — it does not produce instability.

The clean separation: if **overfit-one-batch passes** (the model can drive completion loss to near zero on a handful of examples *with masking applied*), your mask is probably fine and you should look elsewhere. If overfit-one-batch *with masking* fails or behaves strangely (loss won't drop, or drops to zero suspiciously fast), the mask is a prime suspect — decode it. And remember the deepest tell of this whole class: **a masking bug makes the headline loss look** *good* **while the model is bad.** Whenever the loss curve and the actual generations disagree, suspect the labels.

To place masking in the bisection frame the series uses: a masking bug lives in **data**, the first of the six places. The fastest bisection is to ask, in order, three questions whose answers each eliminate whole regions of the search. *Is something scored?* (loss-token fraction not 0% and not 100% — rules in/out the template-match and no-mask bugs). *Is the scored span the completion?* (decode it — rules in/out the prompt-leak and multi-turn bugs). *Does the span end in a stop token?* (rules in/out the EOS bug). Three reads, each ten seconds, and you have either localized the bug or definitively cleared masking and moved on to optimization or model code. That is the whole point of the decode-first discipline: it converts a vague "the finetune is bad" into a precise verdict on the one tensor that controls what the model learns. Contrast this with the expensive failure mode — sweeping learning rates and data ratios for days — which never touches the labels and therefore never finds a label bug. The instrument you do not read is the bug you do not find.

This also clarifies the relationship to the [loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) post, which covers the *other* half of "the loss optimizes the wrong thing": wrong reduction, logits-vs-probabilities, double-softmax, class-index off-by-one. Loss masking is the *which tokens* question; that post is the *which formula* question. Together they cover the two ways a perfectly-converging run can be optimizing something you did not intend.

## 13. A clean reference implementation

Pulling the correct pieces into one place, here is a masking setup I trust, with every guard from the post in it. Use `trl`'s collator in production; this manual version is for when you need to see and control every token.

```python
import torch

IGNORE = -100

def build_labels(tokenizer, prompt_text, completion_text, max_len=2048):
    """Completion-only labels with EOS scored and padding masked.

    Returns input_ids, attention_mask, labels (all 1-D, same length).
    """
    # Tokenize prompt and completion separately so we know the boundary.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    comp_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    # Append the stop token to the completion so it is learned and scored.
    eos = tokenizer.eos_token_id
    comp_ids = comp_ids + [eos]

    input_ids = (prompt_ids + comp_ids)[:max_len]
    attention_mask = [1] * len(input_ids)

    # Labels: -100 on the whole prompt, real ids on the completion+EOS.
    labels = [IGNORE] * len(prompt_ids) + comp_ids
    labels = labels[:max_len]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }

def collate(features, pad_id):
    """Right-pad a batch; pad positions get label -100 via attention_mask."""
    maxlen = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = maxlen - len(f["input_ids"])
        out["input_ids"].append(torch.cat([f["input_ids"], torch.full((pad,), pad_id)]))
        out["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        # Critical: padded label positions are IGNORE, not pad_id.
        out["labels"].append(torch.cat([f["labels"], torch.full((pad,), IGNORE)]))
    return {k: torch.stack(v) for k, v in out.items()}
```

The three guards that make this correct, each tracing to a section above: (1) EOS is appended to the *completion* and kept in labels (§5), (2) the prompt span is `IGNORE` so only completion+EOS is scored (§1), and (3) padding label positions are `IGNORE` rather than `pad_id`, masked by construction rather than by id comparison so a pad==EOS collision cannot eat your real EOS (§6). After building a batch with this, **run the §3 decoder and the §9 test** — the reference implementation is not a substitute for verifying the actual tensor.

If you are doing this as part of a broader, healthy finetune, the surrounding decisions — learning rate (often 1e-5 to 2e-5, far lower than pretraining), epochs (1–3, because SFT overfits fast), and catastrophic forgetting — are covered in [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it). Masking gets your *labels* right; that post gets the *optimization* around them right. And tokenization itself — the BOS/EOS/PAD handling, the double-BOS, vocab mismatches that make the response template fail to match in the first place — is the subject of [tokenization bugs](/blog/machine-learning/debugging-training/tokenization-bugs), which is the upstream cause of half the template-mismatch failures in §4.

A word on *why* I prefer tokenizing the prompt and completion separately (as `build_labels` does) over the clone-and-mask-a-span approach. When you tokenize the full string and then try to find the completion's start index, you are back in the same fragile token-id-matching game that breaks the `trl` collator in §4 — the boundary between prompt and completion can tokenize differently than either piece in isolation, and your span index drifts by a token or two, eating the first completion token or leaking the last prompt token. Tokenizing the two halves separately and concatenating sidesteps the search entirely: you *know* the prompt is the first `len(prompt_ids)` tokens because you built it that way. The cost is that you must ensure the concatenation tokenizes identically to how the model will see it at inference (no surprise merge at the seam), which you verify — of course — with a decode. For most chat templates this is clean; for a few aggressive byte-level tokenizers it is not, and there the collator's id-search, fed the correct template ids, is the more robust path. Either way, the verification is identical: decode the non-`-100` labels and read them.

## 14. Stress-testing the fix

Once your mask is correct, pressure-test it against the conditions that re-break it. This is the make-it-fail-small discipline applied to masking.

**What if the data format changes?** Add a new dataset with a different chat template (say, you mix Alpaca-style and ShareGPT-style examples). The response template that matched one will not match the other — back to the §4 silent failure, but now on *part* of your data, so `loss tokens` is fine on average and the bug hides in a subset. Defense: run the §9 unit test on a sample of *each* source, not just the first batch.

**What if the completion is empty or longer than the context?** Truncation can cut the completion so that EOS falls off the end (`max_len` clips it), reintroducing the never-stops bug for long examples only. Defense: assert that the *last non-pad label* is the stop token, and log how many examples got truncated past their EOS.

**What at fp16/bf16?** Masking is integer logic on labels, so precision does not affect *which* tokens are scored — but a near-empty loss span (few scored tokens) makes the per-step loss noisier, which can look like instability. If you see jumpy loss after a masking fix, check that you did not accidentally drop the loss-token count to near zero.

**What on multi-GPU?** With data-parallel training, each rank builds its own batch; a masking bug is identical across ranks (it is in the shared collator), so DDP does not hide or reveal it — but if you *shard* the dataset and one shard has a different format, you can get the per-source mismatch above on only some ranks. Defense: the same per-source unit test, and decode a batch from each rank during a smoke test.

**What if you switch from padding to packing?** Re-verify, because packing changes how EOS and boundaries land (§7). The labels are usually still correct, but the attention path is the new risk — check `position_ids` reset.

In every case, the verification is the same two tools: **decode the labels** (catches every label-side bug) and **inspect `position_ids` / `attn_implementation`** (catches the packing attention bleed the labels can't show). You are never guessing; you are reading the exact tensor the loss consumes.

## 15. Key takeaways

- **The definitive test for any masking bug is one line: decode `input_ids` where `labels != -100`.** That decoded string must be *exactly* the completion plus EOS. If it shows the prompt, you are training on the prompt; if it shows nothing, your template did not match; if it lacks EOS, the model won't stop.
- **`-100` is the default `ignore_index` of cross-entropy** — labels set to `-100` are dropped from both the loss sum and its denominator. Masking a token means it is read as context but never scored.
- **Hugging Face causal LMs shift labels internally** (`logits[..., :-1]` vs `labels[..., 1:]`): pass `labels` aligned to `input_ids` and never pre-shift. If you compute loss by hand, shift exactly once — never zero (trivial copy task) and never twice (predicts two ahead).
- **Training on the prompt dilutes the gradient** toward the completion by roughly the completion's token fraction (often ~5–20%), and a masked run *correctly* shows a *higher* loss than an unmasked one — the higher number is the honest one.
- **`DataCollatorForCompletionOnlyLM` fails silently when the response template's token ids don't match the stream.** Pass the template as extracted token ids, then *verify with a decode* — a 0% or 100% loss-token fraction is the tell.
- **EOS must be a real, scored label or the model never learns to stop.** Check `ends with EOS` on the decode; beware the `pad_token = eos_token` workaround that masks your real EOS.
- **Mask padding by `attention_mask == 0`, not by `input_ids == pad_token_id`** — the latter eats your EOS when pad and EOS share an id.
- **Packing keeps labels correct but lets attention bleed across documents** unless you use a block-diagonal mask; the labels won't show it — check `position_ids` reset at boundaries.
- **A masking bug makes the headline loss look good while the model is bad.** Whenever the loss curve and the generations disagree, suspect the labels first.
- **Encode the checks as a unit test** (something scored, not everything scored, ends with a stop token, no prompt tokens in the span) and run it whenever the template, tokenizer, collator, or data format changes.

## Further reading

- **PyTorch documentation — `torch.nn.CrossEntropyLoss` and `torch.nn.functional.cross_entropy`.** The authoritative source for `ignore_index` (default `-100`) and exactly how ignored targets are excluded from the average.
- **Hugging Face `transformers` source — the causal-LM loss in the model `forward`.** Read the `shift_logits = logits[..., :-1, :]` / `shift_labels = labels[..., 1:]` block to see the internal label shift first-hand.
- **`trl` documentation — `DataCollatorForCompletionOnlyLM` and `SFTTrainer`.** The completion-only masking recipe, the response-template-as-token-ids fix for silent mismatches, and the `packing` option.
- **Ouyang et al., 2022, "Training language models to follow instructions with human feedback" (InstructGPT).** The reference for the SFT-then-preference-tuning recipe that makes completion-only masking standard practice.
- **Rafailov et al., 2023, "Direct Preference Optimization."** Where masking and label alignment matter again for preference data; useful once your SFT masking is solid.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom → suspect → confirming-test → fix decision tree this post instantiates for the *data/label* branch.
- [Loss-function bugs](/blog/machine-learning/debugging-training/loss-function-bugs) — the sibling post on the *other* way a converging loss optimizes the wrong thing (reduction, logits-vs-probabilities, class-index off-by-one).
- [Finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) — the learning-rate, epochs, and catastrophic-forgetting decisions that surround a correctly-masked SFT run.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone checklist that puts label-decoding into the standard pre-flight for every finetune.
