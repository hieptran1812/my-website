---
title: "Tokenization Bugs: The Off-by-One That Corrupts Every Sequence"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to find the tokenizer bug that silently corrupts every training example before the model sees it, with a decode-the-batch audit, a special-token inspector, an embedding-resize check, and before-and-after evidence from finetunes that wouldn't stop or wouldn't converge."
tags:
  [
    "debugging",
    "model-training",
    "tokenization",
    "llm",
    "nlp",
    "finetuning",
    "deep-learning",
    "transformers",
    "huggingface",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/tokenization-bugs-1.png"
---

The finetune trained cleanly. The loss fell from 2.1 to 0.9 over two epochs, the curve was smooth, the eval perplexity looked reasonable, and you saved the adapter feeling good about the run. Then you loaded it for inference, typed a one-line question, and watched the model answer in three words and then keep going — for a paragraph, then a page, then until it hit the 2,048-token generation cap, repeating itself, drifting off-topic, never stopping. Nothing in the loss curve warned you. The gradients were healthy, the LR was sane, the data was clean. The model simply never learned where a response ends, because in every single training example the token that says "stop here" was missing or masked out. The optimizer faithfully minimized a loss computed over sequences that had no end, and so the model learned a language with no periods.

This is a tokenization bug, and it belongs to a family that is uniquely maddening because the corruption happens *before the model exists in the computation graph*. By the time a tensor of `input_ids` reaches the embedding layer, the damage is baked in: a duplicated beginning-of-sequence token has shifted every position by one, or the wrong tokenizer has mapped your words to a different model's vocabulary, or a freshly added chat token is pointing at an embedding row that was never initialized. The model trains on whatever you fed it, and it cannot tell you that what you fed it was nonsense. Every other failure mode in this series at least leaves a fingerprint on the instruments — a grad-norm spike, a NaN, a plateau. Tokenization bugs leave the instruments looking pristine while quietly poisoning the data at the source.

![A vertical stack showing raw text, the tokenizer, input ids, the embedding table, the model, and the loss, with the tokenizer layer flagged as the place a single wrong special token corrupts everything above it](/imgs/blogs/tokenization-bugs-1.png)

In the six-places framework that runs through this series — a bug hides in *data*, *optimization*, *model code*, *numerics*, *systems*, or *evaluation* — tokenization sits at the very bottom of the data layer, below everything you normally inspect. It is the first transformation applied to your text and the last one you think to check, which is exactly why it earns its own post. A tokenization bug masquerades as an optimization problem (the loss is "fine" but the model is broken), or a model-code problem (the model "won't stop" so you go hunting in the generation loop), or a data problem (you re-clean the dataset that was already clean). The unifying tell that pulls you back to the tokenizer is brutally simple and almost nobody does it: **decode the `input_ids` you actually feed the model and read them as text.** If you had decoded one batch, you would have seen the missing EOS, the doubled BOS, the truncated answer, the garbage byte — in seconds, before the run.

By the end of this post you will be able to take any LLM finetune that won't converge, won't stop generating, or produces garbage on new tokens, and localize the tokenization bug in minutes. You will know the seven ways a tokenizer betrays you — the double BOS where the tokenizer adds a beginning token and your template adds another, the missing EOS that gives the model no gradient to stop, the tokenizer-versus-checkpoint mismatch that maps your text to a stranger's vocabulary, the unresized embedding that points new tokens at uninitialized rows, the whitespace and leading-space conventions that make `"word"` and `" word"` tokenize differently, truncation that eats the answer, and the pad-token-equals-EOS confusion that leaks into your loss. More importantly, you will have the single highest-leverage check in the whole category: decode the batch, print the special-token ids, and compare `len(tokenizer)` to the embedding rows. Three lines of code that kill an entire class of bugs. Let us start where the off-by-one starts: the special tokens.

## 1. The science: subwords, special tokens, and why one wrong id shifts everything

To debug tokenization you need a working mental model of what a tokenizer actually is, so let us build one from first principles before we break it. A tokenizer is a deterministic, reversible map from a string to a list of integers and back. Modern LLMs use *subword* tokenization, almost always a flavor of **Byte-Pair Encoding (BPE)** or its byte-level cousin. The idea is simple and worth internalizing. You start with a vocabulary of individual bytes or characters, then you greedily merge the most frequent adjacent pair into a new symbol, over and over, until you have a vocabulary of the target size — 32,000 for Llama 2, around 128,000 for Llama 3, 151,936 for Qwen2. Common words like `" the"` end up as a single token; rare words like `"antidisestablishmentarianism"` get split into several subword pieces; and any string of bytes, including emoji and code, can always be represented because the base vocabulary includes every byte. That last property, **byte-level fallback**, is why a BPE tokenizer never truly fails to encode something — it falls back to raw bytes — and it is also the source of some of the subtlest whitespace bugs we will hit in section 5.

Alongside the learned subword vocabulary, every tokenizer carries a small set of **special tokens**: reserved ids with structural meaning rather than linguistic content. The four you must know cold are **BOS** (beginning-of-sequence), **EOS** (end-of-sequence), **PAD** (padding filler), and **UNK** (unknown, mostly vestigial in byte-level tokenizers because byte fallback means nothing is ever truly unknown). For Llama 2 the ids are concrete and worth memorizing as a reference point: BOS is id 1 (`<s>`), EOS is id 2 (`</s>`), and there is no separate PAD by default. These tokens are not decoration. The BOS gives the model a fixed anchor at position zero; the EOS is the token whose probability the model must learn to raise when a response is complete, and it is the literal signal generation uses to halt. If the model never sees EOS with a gradient on it during training, it never learns to predict EOS, and it never stops. That single fact explains the opening story entirely.

Now the off-by-one. A decoder-only language model trains on **next-token prediction**: at each position $t$, given tokens $x_0, x_1, \dots, x_t$, it predicts the distribution over $x_{t+1}$. The loss is cross-entropy between the model's prediction at position $t$ and the actual token at position $t+1$. In Hugging Face `transformers`, this shift is handled *inside* the model's forward pass — you pass `input_ids` and identical-length `labels`, and the model internally shifts the labels so that the logits at position $t$ are scored against `labels[t+1]`. The contract is that `input_ids[t]` and `labels[t]` describe the *same physical position* in the sequence; the model does the one-step shift for you. This is the crux of why a tokenization bug is an *off-by-one engine*. If anything perturbs the alignment between positions — an extra token spliced in at the front, a token dropped by truncation at the back — then every label after the perturbation describes the wrong position. The model is now being trained to predict, at each step, a token that does not follow the context it sees. It is being taught a systematically scrambled language.

Consider the cleanest example: the **double BOS**. Suppose the correct tokenization of "The cat sat" with a BOS prefix is `[1, 450, 6635, 3290]` — BOS, then the three content tokens. Next-token training pairs context `[1]` with target `450`, context `[1, 450]` with target `6635`, and so on; the model learns "after BOS comes 'The', after 'The' comes 'cat'." Now suppose a bug inserts a second BOS so the sequence is `[1, 1, 450, 6635, 3290]`. The model is now trained to predict, after seeing a single BOS, that the next token is *another BOS*. After two BOS, predict 'The'. The entire next-token map has been shifted and corrupted by one position. It is not catastrophic in the sense of producing a NaN — the loss still descends, because the corrupted map is still *learnable*, it is just the wrong map — which is precisely why it is so dangerous. The run looks healthy and the model is quietly internalizing a fiction.

Here is the deeper reason this matters more for finetuning than for pretraining. During pretraining on a trillion tokens, a small systematic offset gets averaged over enormous diversity and the model is robust to it; a stray duplicate special token is a rounding error in the data distribution. During finetuning on, say, ten thousand instruction examples, *every single example* carries the same structural bug — the double BOS is not random noise, it is a consistent, repeated corruption in 100% of your training signal. Systematic corruption does not average out; it gets *learned*. That asymmetry is why tokenization bugs are a finetuning specialty: the same mistake that pretraining shrugs off, finetuning amplifies into a behavioral defect.

There is one more piece of mechanism you need before we start breaking things, and it is the bridge from "wrong id" to "garbage output": the **embedding lookup**. After tokenization produces a list of integer ids, the model's very first operation is to use each id to index a row of the embedding matrix $E \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size and $d$ is the hidden dimension. The id is not a number the model does arithmetic on — it is a *pointer*. Token id 6635 means "go fetch row 6635 of $E$," and that row, a $d$-dimensional vector the model learned during training, *is* the model's representation of that token. This pointer semantics is the crux of why several tokenization bugs produce garbage rather than a graceful error. If the id is off by one (double BOS), the model fetches the embedding of the *wrong* word and processes a scrambled sentence. If the id points past the end of the table (unresized embedding), it fetches an uninitialized or out-of-bounds row — noise, or a crash. If the tokenizer that produced the id was trained against a *different* table (tokenizer mismatch), the id 6635 means "cat" to your tokenizer but the model's row 6635 was trained for some other word entirely, so every token is silently mistranslated. The embedding lookup has no error-correction: it faithfully retrieves whatever row the integer names, correct or not. That is why the only reliable verification is to check the integers themselves, by decoding them back to text.

It helps to see one BPE encode-and-decode round-trip concretely, because the merge mechanism explains the whitespace surprises later. Take the string `"unhappiness"`. A byte-level BPE tokenizer starts from individual bytes/characters and applies its learned merge list in order: it might merge `u+n → un`, then `h+a → ha`, then `p+p → pp`, building up `un`, `happi`, `ness` as the rank of each merge dictates, finally landing on something like `["un", "happiness"]` or `["un", "happ", "iness"]` depending on which merges its vocabulary learned. The exact split is not something you can predict by reading the word — it is whatever the merge ranks dictate — which is exactly why you must *tokenize and look* rather than assume. Decoding reverses this perfectly: concatenate the token strings (handling the space markers) and you get the original bytes back. The round-trip is lossless for in-vocabulary text, and the `audit_batch` round-trip check exploits exactly this property: any deviation between `decode(encode(text))` and `text` means something non-trivial happened — a special token was inserted, whitespace was normalized, or a byte fell back — and that something is worth understanding before it corrupts a run.

## 2. The double BOS: when the tokenizer and the template both add one

The double BOS is the most common tokenization bug in instruction finetuning, and it is almost always caused by the same collision: **the tokenizer adds a BOS, and so does your chat template or your manual string concatenation, so you get two.** Understanding the collision requires understanding `add_special_tokens`, the single most misunderstood flag in the Hugging Face tokenizer API.

When you call `tokenizer("The cat sat")`, the tokenizer by default sets `add_special_tokens=True`, which means it automatically prepends the BOS (and, for some tokenizers, appends EOS) according to the tokenizer's configured template. For a Llama tokenizer, `tokenizer("Hello").input_ids` returns `[1, 15043]` — note the leading `1`, the BOS, that *you did not put there*. This is convenient for single-shot encoding and it is exactly the trap. If you build your training string by hand — concatenating a system prompt, a user turn, and an assistant turn, perhaps inserting your own `<s>` markers because that is what the model card showed — and *then* call `tokenizer(full_string)`, the tokenizer prepends *its* BOS on top of *your* BOS. Two id-1 tokens at the front. The same thing happens when you use `tokenizer.apply_chat_template(...)` — which itself may add a BOS — and then pass the result to a trainer that tokenizes again with `add_special_tokens=True`. The fix is to know who is responsible for the BOS and let exactly one party add it.

![A two-by-three grid contrasting a correct single-BOS sequence in the top row against a buggy double-BOS sequence in the bottom row, where the duplicate beginning token pushes every real token one slot to the right](/imgs/blogs/tokenization-bugs-2.png)

The diagnostic is mechanical and takes one line. Decode the first example and count the special tokens.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# A common buggy pattern: a template already added <s>, then we tokenize again.
templated = "<s>[INST] What is 2+2? [/INST] 4</s>"
ids = tok(templated).input_ids
print(ids[:6])                       # -> [1, 1, 518, 25580, ...]  TWO 1s
print(tok.decode(ids[:6]))           # -> '<s><s>[INST] What'      double BOS, visible

# Count how many times the BOS id appears in the sequence.
bos_id = tok.bos_token_id            # 1 for Llama 2
print("BOS count:", ids.count(bos_id))   # -> 2  (should be exactly 1)
```

The moment you `decode` the ids, the bug is staring at you: `<s><s>`. The `count` of `bos_token_id` is the assertion you bake into your data pipeline so it never recurs. Two id-1 tokens means a double BOS; zero means a missing BOS (also a bug for models that expect one); exactly one is correct. The principled fix depends on which party should own the special tokens. If you are letting the chat template add structure, do *not* let the trainer re-add specials:

```python
# Let the chat template own the special tokens; do not double up.
messages = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}]

# apply_chat_template can return ids directly and adds the model's BOS once.
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
print(ids.count(tok.bos_token_id))   # -> 1, correct

# If you ever re-tokenize a templated STRING, disable the second BOS:
text = tok.apply_chat_template(messages, tokenize=False)
ids2 = tok(text, add_special_tokens=False).input_ids   # <-- the critical flag
print(ids2.count(tok.bos_token_id))  # -> 1, correct
```

The rule to memorize: **whoever builds the structured string owns the special tokens, and everyone downstream tokenizes with `add_special_tokens=False`.** If the chat template inserts the BOS, the trainer must not. If you concatenate by hand and add your own `<s>`, the tokenizer must not. Pick one owner and disable the rest.

It is worth being precise about what `add_special_tokens` actually controls, because the name suggests "add ALL special tokens" and that is not what it does. The flag toggles the tokenizer's **post-processor**, a small template that wraps your content tokens with the structural tokens the tokenizer was configured to add — for most causal LLM tokenizers that is just a leading BOS, sometimes a trailing EOS, occasionally neither. It does *not* control the special tokens that already exist *inside* your string. If your text literally contains the characters `<|im_start|>` and that string is registered as a special token, the tokenizer will encode it as the single special-token id regardless of `add_special_tokens`, because that flag governs the *wrapper*, not the *content*. This distinction is the source of the most confusing double-token bugs: people set `add_special_tokens=False` expecting it to suppress *all* special tokens, then are baffled when their in-string `<s>` still becomes a BOS id. The flag suppressed the *wrapper* BOS; the *literal* `<s>` in the text is content and is encoded as-is. The only way to be sure is, again, to decode and count.

A related subtlety lives in `apply_chat_template`. The chat template is a Jinja string stored on the tokenizer (`tok.chat_template`) that turns a list of role/content messages into a single formatted string with the model's exact turn markers. Whether it adds a BOS depends on how that specific model's template was written — some templates begin with `{{ bos_token }}`, some do not and rely on the tokenizer's post-processor to add it, and some add it twice by accident in the template itself (a bug that has shipped in real model repos). This is why you cannot reason about the BOS count from the code alone; you must apply the template to a real message list and count `bos_token_id` in the result. The `add_generation_prompt` argument adds a trailing "assistant:" turn-opener for *inference* (so the model knows to start generating a response) and should be `True` at serve time and `False` during training (where the assistant turn is the label, not a prompt to continue). Getting that argument backwards is its own train/serve skew: train with `add_generation_prompt=True` and you teach the model to predict its own turn-opener; serve with `False` and the model does not know it is its turn to speak.

#### Worked example: the double BOS that capped accuracy

A team finetunes Llama-2-7B on a 12,000-example instruction set with `trl`'s `SFTTrainer`, building the prompt with a manual template string that starts with `<s>`. They also pass the default tokenizer, which adds its own BOS. Every example has a double BOS. The run trains: loss falls 1.74 → 0.88 over three epochs, no NaN, grad-norm a healthy 1.2. On the held-out instruction-following benchmark the model scores **41%**, well below the 60%-ish the base recipe should hit. The team blames the LR, the data quality, the LoRA rank — three days of bisection.

The decode-the-batch test ends it in 30 seconds: `tok.decode(batch["input_ids"][0])[:20]` prints `'<s><s>[INST] What is'`. Two BOS. They re-tokenize the templated string with `add_special_tokens=False`, re-run, and the benchmark moves to **59%** with no other change. The before→after instrument readings: loss curve looks nearly identical in both runs (1.74→0.85 vs 1.74→0.88, indistinguishable on a screenshot), grad-norm identical, but benchmark accuracy 41% → 59%, an 18-point swing from deleting one duplicated integer per example. The loss curve, the one instrument everyone watches, was completely blind to the bug.

## 3. The missing EOS: a model that never learns to stop

If the double BOS is the most common tokenization bug, the missing EOS is the most *visible* in its consequences and the most often misdiagnosed. The symptom is unmistakable once you see it: the finetuned model generates a plausible answer and then refuses to terminate, continuing past the natural end of the response, often repeating or hallucinating, until generation hits `max_new_tokens`. People reach for repetition penalties, `min_new_tokens`, custom stopping criteria — all band-aids over a wound in the training data. The model does not stop because it was never taught to.

The science is exact. Generation halts when the model samples the EOS token; the decoding loop checks `next_token == eos_token_id` and breaks. For the model to *predict* EOS with meaningful probability at the end of a response, two things must be true during training: the EOS token must be *present* in the `input_ids` at the end of each example, and its position must be *unmasked in the labels* so it contributes to the loss. The loss at the final content position is cross-entropy between the model's prediction and the next token; if the next token is EOS and that position carries a real label (not `-100`), the model gets gradient pushing it to raise EOS probability there. Remove the EOS, or mask it with `-100`, and that gradient never flows. The model literally never receives a single training signal that says "this is where you stop," so at inference it assigns near-zero probability to EOS and runs forever.

Let us make the gradient argument quantitative, because the size of the effect explains why the model is so confidently non-stopping. Cross-entropy gives the clean gradient $\partial L / \partial z_i = p_i - y_i$ at the logits, where $p$ is the softmax of the logits and $y$ is the one-hot target. At the position where the answer ends, the target is EOS, so $y_{\text{EOS}} = 1$ and the gradient on the EOS logit is $p_{\text{EOS}} - 1$. Before training, $p_{\text{EOS}}$ for a random end-of-sentence context is tiny — EOS is a rare token in pretraining text relative to content words — so the gradient is close to $-1$, a strong push to *raise* the EOS logit. Over a few thousand examples this pushes the model to predict EOS confidently at the end of a response. Now remove the EOS from the labels: there is no position where the target is EOS, so the gradient on the EOS logit is *never* $p_{\text{EOS}} - 1$; it is only ever $p_{\text{EOS}} - 0 = p_{\text{EOS}} \ge 0$ at positions where EOS is a *wrong* answer, a push to *lower* the EOS logit. The net effect is brutal: with no positive EOS signal and a steady stream of negative ones (every content position says "EOS is not the answer here"), the model learns to *suppress* EOS, assigning it near-zero probability everywhere — including at the natural end of a response. The model does not merely fail to learn to stop; it actively learns *not* to stop. That is why repetition penalties and `min_new_tokens` band-aids fight a model that has been trained, with full gradient force, against the very token they need it to emit.

![A before-and-after comparison showing a training example with no EOS that gives the model no gradient to stop and runs to the length cap, versus one with EOS appended that teaches a clean stop in forty tokens](/imgs/blogs/tokenization-bugs-3.png)

There are two distinct ways the EOS goes missing, and you must check both. The first is that the EOS is simply **never appended** to the example — your formatting function builds `prompt + answer` and forgets the terminator. The second, subtler one is that the EOS *is* appended but is **masked out of the labels**, often because a loss-masking helper masks "non-content" tokens too aggressively and treats the EOS as structural rather than as a token to predict. The diagnostic distinguishes them: decode to see whether the id is present, then inspect the labels to see whether it is `-100`.

```python
import torch

eos_id = tok.eos_token_id            # 2 for Llama 2

# `batch` came straight from your collator / dataset.
ids    = batch["input_ids"][0]
labels = batch["labels"][0]

# (1) Is the EOS even in the input_ids?
has_eos = (ids == eos_id).any().item()
print("EOS present in input_ids:", has_eos)

# (2) If present, is its label position a real label (not -100)?
if has_eos:
    pos = (ids == eos_id).nonzero(as_tuple=True)[0]
    print("EOS positions:", pos.tolist())
    print("Labels at EOS positions:", labels[pos].tolist())
    # A value of -100 here means the EOS is masked out of the loss -> still won't learn to stop.
```

If `has_eos` is `False`, append it. If it is `True` but the labels at those positions are `-100`, unmask them. The fix for the first case is to make your formatting function explicitly add the EOS and confirm it survives tokenization:

```python
def format_example(prompt, answer, tok):
    # Build the assistant turn and append the model's EOS string explicitly.
    text = f"{prompt}{answer}{tok.eos_token}"     # tok.eos_token == '</s>'
    enc = tok(text, add_special_tokens=True)      # BOS added once, EOS now inside text
    # Assert the terminator survived.
    assert enc["input_ids"][-1] == tok.eos_token_id, "EOS got stripped or truncated!"
    return enc
```

A note on `trl`'s `SFTTrainer`: it has historically been a frequent source of this bug because, depending on version and config, it does not always append EOS for you, and the `DataCollatorForCompletionOnlyLM` masks the prompt — if your formatting function also failed to add EOS, you ship a model that cannot stop. The lesson is the same regardless of library: never trust that EOS was added; assert it.

#### Worked example: the chatbot that wouldn't shut up

A 7B model is finetuned for customer-support replies. In testing, every reply is correct for the first sentence and then spirals: it answers the question, then invents a second question and answers that, then a third, running to the 512-token cap on nearly every generation. The team adds a repetition penalty (helps cosmetically, hurts quality), then a hard `max_new_tokens=80` (truncates real answers), then writes a custom `StoppingCriteria` on a newline (breaks multi-paragraph replies). All band-aids.

The audit: `(batch["input_ids"][0] == tok.eos_token_id).any()` returns `False`. There is no EOS in any training example — the formatting function concatenated `instruction + response` and stopped. They add `tok.eos_token` to every response, confirm `labels[-1] != -100`, and re-finetune for the same one epoch. Now the model emits EOS and stops on its own: average generation length drops from **512 tokens (the cap, every time)** to **47 tokens (the natural answer length)**, and the awkward band-aids come out. The instrument that confirmed the fix was not the loss curve — which moved trivially — but the *generation-length histogram*, which collapsed from a spike at the cap to a sane distribution centered on real answer lengths.

## 4. The diagnostic that ends most tokenizer bugs: decode the batch

You have now seen three different tokenization bugs, and all three were caught by the same one move: decode what you actually feed the model and read it. This is the central thesis of the entire post, so let us make the diagnostic a first-class, reusable tool rather than an ad-hoc print statement. The principle is that the tokenizer config, the chat template, the collator, and the trainer's internal tokenization compose in ways no human can predict from reading the code; the only ground truth is the tensor that hits the embedding layer. Decode it.

![A decision tree routing a tokenizer symptom through a single decode-the-batch test that splits special-token bugs from vocab-mismatch bugs in under a minute](/imgs/blogs/tokenization-bugs-4.png)

Here is a complete batch auditor. Drop it in right before your training loop, run it once, and read the output with your eyes. It decodes the sequence, locates and labels every special token, shows exactly which positions are masked in the labels, and round-trips to catch encode/decode mismatches.

```python
import torch

def audit_batch(batch, tok, n=1, show_chars=200):
    """Decode and inspect what the model actually trains on. Run once, READ it."""
    specials = {
        tok.bos_token_id: "BOS", tok.eos_token_id: "EOS",
        tok.pad_token_id: "PAD", tok.unk_token_id: "UNK",
    }
    for i in range(min(n, batch["input_ids"].size(0))):
        ids    = batch["input_ids"][i].tolist()
        labels = batch.get("labels", None)
        print(f"\n===== example {i} | length {len(ids)} =====")

        # (1) The single most important line: read it as text.
        print("DECODE:", repr(tok.decode(ids))[:show_chars])

        # (2) Where do the special tokens actually land?
        for sid, name in specials.items():
            if sid is None:
                continue
            pos = [j for j, t in enumerate(ids) if t == sid]
            print(f"  {name} (id {sid}) at positions {pos}  count={len(pos)}")

        # (3) Which positions contribute to the loss?
        if labels is not None:
            lab = labels[i].tolist()
            n_unmasked = sum(1 for x in lab if x != -100)
            print(f"  labels: {n_unmasked}/{len(lab)} positions unmasked (rest are -100)")
            # Show the first few unmasked (label, decoded-token) pairs.
            preview = [(x, tok.decode([x])) for x in lab if x != -100][:8]
            print("  first unmasked labels:", preview)

        # (4) Round-trip: does encoding the decoded text reproduce the ids?
        rt = tok(tok.decode(ids), add_special_tokens=False)["input_ids"]
        print("  round-trip stable:", rt == [t for t in ids if t != tok.pad_token_id])

audit_batch(next(iter(train_dataloader)), tok, n=2)
```

Read the output as a checklist. Does the decoded text look like what you intended — the right prompt, the right answer, the right chat structure? Is there exactly one BOS at position 0 and exactly one EOS at the end? Are the unmasked label positions the *completion* tokens and not the prompt (for instruction tuning you train on the answer, not the question)? Does the EOS position carry a real label? Does the round-trip reproduce the ids? Five questions, all answered by reading one printout. I have caught double BOS, missing EOS, prompt-trained-instead-of-completion, left-vs-right truncation eating the answer, and a wrong-tokenizer mismatch all from this single function. It is the cheapest, highest-leverage diagnostic in LLM finetuning, and it costs less than a second of wall-clock time.

The companion to decoding the batch is auditing the special-token configuration itself, because half of these bugs originate in a tokenizer whose special-token ids or strings are not what you assume. This second auditor prints the full special-token map and flags the dangerous coincidences:

```python
def audit_special_tokens(tok):
    print("vocab size (len(tok)):", len(tok))
    for name in ["bos", "eos", "pad", "unk"]:
        s = getattr(tok, f"{name}_token", None)
        i = getattr(tok, f"{name}_token_id", None)
        print(f"  {name.upper():4} token={s!r:>12}  id={i}")
    # The classic trap: PAD == EOS means padding is indistinguishable from a real stop.
    if tok.pad_token_id is not None and tok.pad_token_id == tok.eos_token_id:
        print("  WARNING: pad_token_id == eos_token_id  (mask PAD in loss or generation may misbehave)")
    if tok.pad_token_id is None:
        print("  NOTE: no PAD token set; batching/padding will need one assigned")

audit_special_tokens(tok)
```

These two functions together — `audit_batch` and `audit_special_tokens` — are the practical core of this post. If you adopt nothing else, adopt the habit of running them once before every finetune. They turn a class of multi-day silent bugs into a 10-second visual inspection.

## 5. Whitespace, the leading space, and byte-level surprises

Not every tokenization bug involves a special token. A whole category lives in how subword tokenizers handle **whitespace**, and these bugs are subtle because they do not crash and do not look wrong until you compare token counts. The root fact is that byte-level BPE tokenizers (GPT-2, Llama, Mistral, Qwen all descend from this lineage) encode the *leading space* as part of the token. In GPT-2's vocabulary the space is rendered as `Ġ`; in the SentencePiece-derived Llama vocabulary it is rendered as `▁` (a lower-one-eighth block, U+2581). This means `"word"` and `" word"` (with a leading space) tokenize to *different* token ids. The version with the leading space is usually the "natural" mid-sentence form, because in running text almost every word is preceded by a space.

```python
# The leading space changes the tokenization.
print(tok("word", add_special_tokens=False).input_ids)    # e.g. [1734]
print(tok(" word", add_special_tokens=False).input_ids)   # e.g. [3186]  -- different id!

# Decode shows the space convention explicitly.
print(repr(tok.convert_ids_to_tokens(tok(" word", add_special_tokens=False).input_ids)))
# -> ['▁word']   the ▁ encodes the leading space
```

Why does this matter for training? Two ways. First, **concatenation seams.** If you build a prompt by joining strings without thinking about spaces — `system + user + assistant` — you can produce token sequences at the seams that never occur in natural text, where a word that should have a leading-space token instead gets a no-leading-space token because it landed right after a special token or another word with no separator. The model has to learn these unnatural seam tokens, which are pure noise. Second, **`add_prefix_space`.** The fast GPT-2-style tokenizers expose an `add_prefix_space` flag that controls whether a leading space is added to the *first* word of a string before tokenizing. If your training data was tokenized with one setting and your inference path with another, the first token of every input differs between train and serve — a silent train/infer skew that degrades quality in a way no metric on the training distribution will reveal.

The diagnostic is to tokenize a representative string both ways and look at the boundary tokens. More practically, the round-trip check in `audit_batch` catches the gross cases: if `decode(encode(text)) != text` up to special tokens, something about whitespace or byte handling is lossy. For Llama-family tokenizers, a notorious specific instance is that the SentencePiece tokenizer adds a leading `▁` (space) to the start of the string by default, so `tok.decode(tok("Hello").input_ids)` can come back as `" Hello"` with a leading space you did not have — usually harmless, occasionally the cause of a one-token offset when you slice sequences by character position.

The **byte-fallback** mechanism is the other whitespace-adjacent surprise. When a byte-level tokenizer hits a character or byte sequence not in its merged vocabulary — an unusual Unicode glyph, a malformed UTF-8 fragment, certain emoji — it falls back to encoding the raw bytes as a sequence of byte tokens (often rendered like `<0xE2>`). This is correct and reversible, but it means a single visible character can become three or four tokens, which interacts badly with truncation (you can truncate in the *middle* of a multi-byte character, producing an un-decodable fragment) and with length budgeting (your "100-character" answer might be 250 tokens). The defense is, again, to decode and look: if your decoded batch contains `<0x..>`-style byte tokens where you expected clean text, you have a byte-fallback situation worth understanding before it bites truncation.

| Symptom | Whitespace / byte cause | Confirming test | Fix |
| --- | --- | --- | --- |
| `"word"` ≠ `" word"` ids | leading-space convention (`▁` / `Ġ`) | tokenize both, compare ids | keep natural spacing at seams |
| First token differs train vs serve | `add_prefix_space` mismatch | tokenize same string both paths | match `add_prefix_space` everywhere |
| Un-decodable fragment at end | truncation split a multi-byte char | decode the truncated tail | truncate on token, re-validate decode |
| `<0xE2>`-style tokens appear | byte fallback on rare glyph | inspect `convert_ids_to_tokens` | expect it; budget length on tokens |
| Decoded text has stray leading space | SentencePiece prefix space | round-trip `decode(encode(x))` | account for it in char slicing |

#### Worked example: the leading-space skew that cost two points

A team trains a span-extraction finetune where they build inputs by string-formatting a template: `f"Question: {q}\nContext: {c}\nAnswer:{a}"` — note there is no space between `Answer:` and the answer `{a}`. In running text the answer would naturally begin with a leading space, so the model's learned representation of the first answer word is the *with-space* token (e.g. `▁Paris`). But because the template glued the answer directly onto the colon, the tokenizer produced the *no-leading-space* variant (`Paris`, a different, rarer id) for the first word of every answer. The model trained on these unnatural no-space first tokens, ids it had barely seen during pretraining. The run looked fine — loss descended normally — but exact-match accuracy sat about two points below a sibling run, and nobody could explain it. The audit: tokenize one example, `convert_ids_to_tokens`, and look at the answer's first token — it was `Paris`, not `▁Paris`. Adding a single space after the colon (`Answer: {a}`) restored the natural with-space token and recovered the two points. The lesson: a missing space at a concatenation seam is invisible in the source string and invisible in the loss curve, but it changes the token id of the word right after the seam, and the only way you ever see it is by converting ids to tokens and reading the space markers.

The meta-lesson of this section is that subword tokenizers are *not* a transparent map from words to ids. They are a learned, byte-aware, space-sensitive compression scheme, and the only way to know what they did to your text is to look at the tokens, not the source string. Two strings that look identical to you — differing only in a space you cannot see at a seam — can tokenize to different ids, and the model treats different ids as different words. This is why the discipline of *decoding and inspecting tokens* is not paranoia; it is the only ground truth in a system where the source string and the token stream are related by a learned, opaque function.

## 6. Truncation that eats the answer

Here is a bug that does not involve special tokens or whitespace at all, yet wastes runs constantly: **`max_length` truncation removes the part of the sequence you most needed to train on.** In instruction tuning you train on the *completion* — the answer — and the prompt is context. If your prompt is long and your `max_length` is modest, right-side truncation (the default) chops off the *end* of the sequence, which is exactly where the answer lives. You end up training on a prompt with no answer, or worse, an answer cut off mid-sentence with no EOS. The loss is computed over whatever survived, which may be all prompt and no completion — and recall that for completion-only training the prompt is masked with `-100`, so you can end up with an example that contributes *nothing* to the loss because every unmasked (answer) token was truncated away.

The science is just arithmetic but it bites because it is invisible. Say your `max_length=512`, your average prompt is 480 tokens (a few retrieved documents plus instructions), and your answers average 90 tokens. Then 480 + 90 = 570 > 512, so right-truncation discards the last 58 tokens — which are the back half of the answer including the EOS. The model learns truncated, terminator-less answers. The fix is a choice between three levers: raise `max_length`, truncate from the *left* (drop old context, keep the recent answer), or use a smarter packing strategy. The critical insight for instruction tuning is **truncation side**: for decoder-only generation you usually want to keep the most recent tokens, so `truncation_side="left"` protects the answer, whereas the default `"right"` protects the (less important) start of the prompt.

```python
# Make truncation visible and protect the completion.
tok.truncation_side = "left"          # keep the END (the answer + EOS) for instruction tuning

enc = tok(
    full_text,
    max_length=1024,                  # budget generously; measure your real distribution first
    truncation=True,
    add_special_tokens=False,         # the template already added BOS
)

# Diagnostic: how often is the answer being clipped?
n = len(enc["input_ids"])
if n >= 1024:
    tail = tok.decode(enc["input_ids"][-30:])
    print("Possibly truncated. Last 30 tokens decode to:", repr(tail))
    assert enc["input_ids"][-1] == tok.eos_token_id, "EOS lost to truncation!"
```

The honest way to size `max_length` is to *measure the length distribution* before you pick a number, rather than guessing. Tokenize the whole dataset once, histogram the lengths, and set `max_length` to a high percentile (say the 99th) so you keep nearly every full example while bounding memory.

```python
import numpy as np

lengths = [len(tok(format_example(ex), add_special_tokens=False)["input_ids"])
           for ex in dataset]
lengths = np.array(lengths)
for p in (50, 90, 95, 99, 100):
    print(f"  p{p:>3} length = {int(np.percentile(lengths, p))}")
print(f"  fraction over 512: {(lengths > 512).mean():.1%}")
print(f"  fraction over 1024: {(lengths > 1024).mean():.1%}")
```

If 12% of your examples exceed `max_length`, then 12% of your training signal is corrupted by truncation — a large enough fraction to visibly hurt the model, and a number you can only know by measuring. This is the same "look at your data before you train" discipline the data-debugging track preaches, applied to lengths.

There is a second, sneakier truncation failure that the percentile histogram alone will not catch: **truncation interacting with packing.** A common efficiency optimization is to *pack* multiple short examples into one fixed-length sequence to avoid wasting compute on padding. If packing concatenates examples and then truncates the pack to a fixed length, the final example in each pack can be cut mid-answer — and because packing already concatenated across document boundaries, the truncation point is now arbitrary with respect to your examples. The defense is to pack to *token* boundaries that respect example edges (drop the overflowing example to the next pack rather than splitting it), and to ensure each packed example still carries its own EOS so the model learns that documents end even inside a pack. The general principle: any operation that changes sequence length — truncation, packing, sliding windows — is a place where the EOS can vanish or the answer can be clipped, so the post-operation `audit_batch` decode is mandatory, not optional. Decode the *packed, truncated, collated* batch, the exact tensor the model trains on, not the pre-processing intermediate you assume is faithful.

A final note on the *measurement* of truncation damage: the honest before→after metric here is not the loss (which barely moves, because the surviving tokens still train normally) but a behavioral one — **answer completeness**. Sample fifty generations before and after the fix and count how many end with a complete sentence and an EOS versus how many trail off mid-thought. In the run that motivated this section, raising `max_length` from 512 to 1024 and switching to `truncation_side="left"` moved complete-answer rate from 71% to 98%, while the training loss was statistically indistinguishable between the two runs. The metric that lies (loss) said nothing; the metric that matters (completeness) said everything.

## 7. The tokenizer–checkpoint mismatch and the unresized embedding

We now reach the two structural bugs that are not about *content* tokens at all but about the *integrity of the vocabulary*: loading a tokenizer that does not match the model, and adding tokens without resizing the embedding. Both produce garbage, and both are easy to introduce when you assemble a pipeline from parts.

![A directed graph showing a tokenizer loaded against a checkpoint forking into a matching path and a mismatched path that either crashes on an index error or silently maps tokens to the wrong ids](/imgs/blogs/tokenization-bugs-8.png)

### The mismatch: a tokenizer from a different model

A model's embedding table is a lookup: id $i$ selects row $i$ of a matrix of shape `(vocab_size, hidden_dim)`. The mapping from text to id is the *tokenizer's* job, and the mapping from id to meaning is *baked into the trained embedding*. These two must come from the same training run. If you load `meta-llama/Llama-2-7b-hf` weights but a *different* tokenizer — a Mistral tokenizer, or a fine-tuned variant with an extended vocabulary, or even the right family but a different version — then the integer your tokenizer assigns to "cat" points at the embedding row that the *model* learned for some other word. The model receives systematically wrong inputs for every token. There are two failure modes, shown in the graph above. If the vocab sizes differ and the tokenizer emits an id $\ge$ the model's `vocab_size`, you get a hard **index-out-of-range crash** — annoying but honest, because it stops you immediately. The far more dangerous case is when the vocab sizes *match by coincidence* but the id-to-token assignment differs: no crash, just silently scrambled inputs, a model that trains to high loss and never makes sense.

The diagnostic is to compare the tokenizer to the model card and to assert the sizes agree:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # SAME repo!

emb_rows = model.get_input_embeddings().weight.shape[0]
print("len(tokenizer):", len(tok))           # 32000
print("embedding rows:", emb_rows)           # 32000
assert len(tok) <= emb_rows, "tokenizer can emit ids the embedding has no row for!"

# A stronger check: a known token must map to the id the model card documents.
assert tok.convert_tokens_to_ids("<s>") == 1, "BOS id is not what this checkpoint expects"
```

The rule is blunt: **load the tokenizer and the model from the same repository, every time.** When you cannot — because you deliberately extended the vocabulary — you must resize, which is the next bug.

The silent-mismatch case deserves emphasis because it is the one that wastes the most time. Imagine two tokenizers from the same family that happen to share a `vocab_size` of 32,000 but disagree on a few hundred id assignments — perhaps one reordered a block of added tokens, or one is a community re-export with a different merge order. There is no crash: every id your tokenizer emits is a valid row index into the model's embedding. But for the disagreeing tokens, the row the model fetches is the embedding of a *different* word. The model trains on inputs that are mostly correct (the shared assignments) and partly scrambled (the disagreeing ones), so the loss descends — slowly, to a higher floor than it should — and the model is subtly, pervasively confused. There is no single dramatic symptom, just a finetune that is worse than it should be for no visible reason. The only way to catch it is to assert that a handful of known tokens map to the ids the model card documents, which the `convert_tokens_to_ids` assertion above does. When in doubt, the cheapest possible check is to encode a fixed probe string with both the tokenizer you are using and the tokenizer from the model's own repo and assert the id lists are identical.

### The unresized embedding: new tokens pointing at nothing

When you add tokens — a new `<|im_start|>` chat token, special domain tokens, a pad token the base model lacked — you grow `len(tokenizer)` but the model's embedding table does not grow on its own. If the tokenizer now emits id 32004 but the embedding has 32000 rows, you index past the end of the table. Depending on the path this either throws an index error or, in some configurations, reads uninitialized or wrapped memory and produces garbage gradients. The fix is one function call, `model.resize_token_embeddings(len(tok))`, and you must call it *after* adding tokens and *before* training.

![A before-and-after comparison showing an added token id past the 32000-row embedding reading an out-of-range or uninitialized row, versus resizing to 32008 rows so the new tokens become trainable and learn](/imgs/blogs/tokenization-bugs-6.png)

```python
# Add new special tokens (e.g. chat-format tokens the base model lacked).
new_tokens = ["<|im_start|>", "<|im_end|>"]
num_added = tok.add_special_tokens({"additional_special_tokens": new_tokens})
print("added:", num_added, "new len(tok):", len(tok))   # e.g. added 2, len 32002

# CRITICAL: grow the embedding (and the tied output head) to match.
model.resize_token_embeddings(len(tok))
print("embedding rows now:", model.get_input_embeddings().weight.shape[0])  # 32002

# Optional but recommended: initialize new rows to the mean of existing embeddings
# so they start in a sensible region rather than at random.
with torch.no_grad():
    emb = model.get_input_embeddings().weight
    mean_vec = emb[:-num_added].mean(dim=0)
    emb[-num_added:] = mean_vec
    # If the output head is untied, do the same for it.
    if model.get_output_embeddings() is not None and \
       model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        out = model.get_output_embeddings().weight
        out[-num_added:] = out[:-num_added].mean(dim=0)
```

The science of *why* an unresized or freshly-resized embedding gives garbage is worth stating precisely, because it explains both the bug and the recommended fix. The embedding row for a token is the vector the model uses to *represent* that token everywhere downstream. A randomly initialized new row is a random vector with a norm comparable to the trained rows, pointing in a direction that means nothing — the model has no learned association for it, so its first few hundred steps of seeing that token produce large, noisy gradients as the model scrambles to assign it meaning, often destabilizing the run. Worse, with weight-tied embeddings (the input embedding and the output projection share the same matrix, common in Llama-family models), a garbage new row also corrupts the *logits* for that token, so the model both reads and writes the new token incorrectly. Initializing new rows to the *mean* of the existing embeddings sidesteps the worst of this: the new token starts as a generic, average-meaning vector near the bulk of the embedding cloud, and the model nudges it toward its real meaning from a stable starting point rather than from random noise. The before→after is stark: with random init, runs that add several new tokens often spike loss in the first 50 steps and sometimes diverge; with mean init plus a correct resize, the loss is smooth from step 1 and the new tokens acquire sensible embeddings within an epoch.

#### Worked example: the chat tokens that produced garbage

A team adapts a base (non-chat) Mistral-7B into a chat model by adding `<|im_start|>` and `<|im_end|>` to mark turns. They add the tokens to the tokenizer and start finetuning. The loss starts at 8.4 (versus a normal ~2.5 for this base), spikes to 19 at step 30, and the run NaNs at step 70-ish on some seeds. They suspect the LR, halve it, same story. The audit: `len(tok)` is 32002, `model.get_input_embeddings().weight.shape[0]` is 32000 — they never resized. The new tokens were emitting ids 32000 and 32001 into a 32000-row table. After `model.resize_token_embeddings(len(tok))` plus mean-initializing the two new rows, the run starts at loss 2.6 (sane), descends smoothly, and the chat tokens learn meaningful embeddings within the first epoch. The instrument that nailed it was a one-line assertion comparing `len(tok)` to the embedding row count — a check that takes microseconds and would have prevented two days of LR-blaming.

## 8. Fast versus slow tokenizers, and pad-equals-EOS

Two final traps round out the catalog, both subtle and both common.

**Fast versus slow tokenizers.** Hugging Face ships two implementations of most tokenizers: a "slow" pure-Python one and a "fast" Rust-backed one (the `tokenizers` library), selected by `use_fast=True/False`. They are *usually* identical but not always — there are documented historical cases where the fast and slow versions of the same tokenizer disagreed on edge cases (whitespace handling around special tokens, certain Unicode normalization, the treatment of the prefix space). If your data was preprocessed with one and your training or inference uses the other, you get a silent train/serve skew. The defense is to be explicit and consistent: pick `use_fast=True` (the default and the faster path), use it everywhere, and if you ever see an unexplained quality gap between two pipelines that should be identical, check whether one is fast and the other slow.

```python
fast = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
slow = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)

probe = "Hello,  world!  <s>edge</s>"   # double spaces + special-token strings
print("fast == slow:", fast(probe).input_ids == slow(probe).input_ids)
# If False, you have a fast/slow discrepancy to pin down before it skews train vs serve.
```

**Pad equals EOS.** Many base models ship without a dedicated PAD token, and the most common workaround is to set `tok.pad_token = tok.eos_token`, reusing the EOS id for padding. This is fine *if and only if* you handle the consequences. The first consequence: padding tokens must be masked out of the loss (set their label to `-100`) and masked out of attention (the attention mask must be 0 at pad positions); if you set PAD = EOS and then forget to mask, the model trains on pad-EOS tokens as if they were real, and worse, it learns that long runs of EOS are normal — which can make it *over*-predict EOS and stop too early, or under-predict it because the EOS signal is now diluted across both real stops and padding. The second consequence: at generation time, if PAD and EOS share an id, some stopping logic and some left-padding-for-batched-generation setups can behave surprisingly. The clean alternatives are to add a genuine new PAD token (and resize the embedding, per section 7) or to keep PAD = EOS but rigorously ensure pad positions are `-100` in labels and `0` in the attention mask.

```python
# The common workaround and its required guardrail.
if tok.pad_token is None:
    tok.pad_token = tok.eos_token       # reuse EOS id for padding

# ... in the collator, pad positions MUST be masked in the labels:
labels = input_ids.clone()
labels[attention_mask == 0] = -100      # do not compute loss on padding
# And the attention mask must already be 0 at pad positions (it is, if you padded correctly).

# Sanity: the *real* EOS that ends a response must remain unmasked so the model learns to stop,
# even though pad-EOS tokens after it are masked. Verify exactly one unmasked EOS per example.
```

The subtlety worth holding onto: when PAD = EOS, you have two *kinds* of EOS in the sequence — the one real terminator that ends the response (must be unmasked, the model must learn it) and the padding-EOS that fills the batch (must be masked). Conflating them is how you get a model that either never stops or always stops at length one. Keeping them straight is the whole game, and `audit_batch` shows you exactly which EOS positions carry real labels.

It is worth tracing precisely *how* the conflation produces the "always stops at length one" failure, because it is a perfect illustration of how a tokenization-level decision propagates into a behavioral pathology. Suppose you pad on the *right* with PAD = EOS and you *forget* to mask the pad positions in the labels. Now a short example padded to length 256 has its real answer in the first 40 tokens, a real EOS at position 40, and then 215 padding-EOS tokens from position 41 to 255, *all unmasked*. The model receives 215 gradient signals per example saying "predict EOS here," at positions whose context is itself a run of EOS tokens. It learns, overwhelmingly, that EOS follows EOS and that EOS is extremely probable — so at inference it emits EOS almost immediately, producing one-word answers. Conversely, if you pad on the *left* with PAD = EOS and the attention mask is wrong, the leading run of EOS tokens corrupts the position information and the model attends to padding as if it were content. Both pathologies trace to the same root: a padding token that is indistinguishable, by id, from a meaningful stop token, with masking that failed to tell them apart. The clean fix — a dedicated PAD token with its own id, resized into the embedding — removes the ambiguity at the source, which is always preferable to patching it downstream with careful masking you might forget on the next dataset.

## 9. The diagnostic flow: bisecting a tokenizer bug end to end

Let us put the whole method together on one realistic failing run and bisect it the way the series teaches: read the instruments, make-it-fail-small, and route the symptom through the cheapest discriminating test.

![A timeline of five ordered checks that decode the real batch, print special-token ids, count BOS and EOS, compare tokenizer length to embedding rows, and round-trip encode and decode before any GPU time is spent](/imgs/blogs/tokenization-bugs-7.png)

**The run.** You finetune Qwen2-7B on a 20,000-example instruction dataset with LoRA. The loss descends from 1.9 to 0.7, no NaN, grad-norm steady around 0.8 — every optimization instrument is green. At inference the model produces answers that are *correct in content* but never stop, running to `max_new_tokens=256` every time, and occasionally the first token of every answer is a stray special-token string. Two symptoms: won't-stop, and a weird first token.

**Step 1 — decode the batch.** Before touching the model, run `audit_batch(next(iter(train_dl)), tok)`. The decoded text reads `<|im_start|><|im_start|>user\n...`. Two `<|im_start|>` tokens at the front — a double special token, the chat-template analog of the double BOS. The "weird first token" symptom is explained: the template added the turn-start token and the trainer's tokenization added it again.

**Step 2 — count the special tokens.** The auditor reports `<|im_start|>` appears at positions `[0, 1]` (count 2) and `<|im_end|>` appears once near the end but — crucially — `tok.eos_token_id` appears **zero** times, and the labels at the final position are `-100`. So there are *two* bugs stacked: a duplicated turn-start token, and no unmasked EOS. The won't-stop symptom is now fully explained: the model never sees an EOS with a gradient on it.

**Step 3 — check the vocabulary integrity.** Run `audit_special_tokens(tok)` and compare `len(tok)` (151,646 after adding the two chat tokens) to `model.get_input_embeddings().weight.shape[0]`. They match — good, someone did resize. So this is *not* an unresized-embedding bug; the vocabulary integrity is intact. That clears one suspect and saves you from chasing it.

**Step 4 — fix and re-measure.** Two targeted fixes: re-tokenize the templated string with `add_special_tokens=False` so the turn-start token appears once, and append `tok.eos_token` to every assistant turn with its label left unmasked. Re-run `audit_batch`: now exactly one `<|im_start|>` at position 0, exactly one unmasked EOS at the end. Re-finetune. The loss curve is nearly identical to before (it was never the problem), but the *behavior* changes completely: average generation length drops from 256 (the cap) to 38, and the stray-first-token symptom is gone.

**Step 5 — stress test.** Now interrogate the fix the way a careful debugger does. *What if the bug were data, not tokenization?* If the dataset itself contained truncated or malformed answers, `audit_batch` would show garbled decoded text, not clean text with a missing EOS — the clean decode rules out a data-content bug. *What if it only failed at inference?* The decode-the-batch test operates on training data, so it cannot catch a train/serve template skew on its own; you also decode the *inference* prompt (`tok.apply_chat_template(msgs, add_generation_prompt=True)`) and confirm it matches the training format minus the answer. *What if the embedding had not been resized?* You already cleared that in step 3, but if it had been the suspect, the signature would have been a loss that started absurdly high (8+) and spiked early, not a clean descent — a different fingerprint entirely. Each stress-test question maps a possible alternative cause to a distinguishing instrument reading, which is the entire discipline of bisection.

## 10. Case studies and real signatures

Tokenization bugs are not academic; they are among the most-reported issues in open-source finetuning, and a few patterns recur often enough to have names. Each one maps to a most-likely cause, a cheap confirming test, and a one-line fix, and the matrix below is the lookup table you reach for the moment a finetune misbehaves in a way the loss curve cannot explain.

![A matrix mapping five tokenizer symptoms such as won't stop, first token always a beginning marker, and garbage on new tokens to their likely cause, a confirming test, and a one-line fix](/imgs/blogs/tokenization-bugs-5.png)

**The double-BOS in Llama instruction tuning.** This is the canonical case. The Llama 2 chat format uses `<s>[INST] ... [/INST]` and the tokenizer adds its own `<s>` (BOS, id 1). Countless community finetunes pasted the `<s>` from the model card into their template string *and* tokenized with the default `add_special_tokens=True`, producing two BOS tokens on every example. The signature is exactly what we saw: a clean loss curve and a benchmark several points below the recipe's published number, with no other anomaly. The fix — disable the second BOS — is documented in numerous post-mortems and is the first thing experienced practitioners check on a Llama finetune that underperforms by a suspicious, consistent margin.

**The missing-EOS "won't stop" bug in `trl`.** Because `SFTTrainer` historically did not always append EOS, and because completion-only collators mask the prompt, a formatting function that forgot the terminator would ship a model that generates correctly and never stops. This is one of the most-filed issues in the instruction-tuning ecosystem, and the giveaway is the generation-length histogram pinned at `max_new_tokens`. The fix is to explicitly append `tok.eos_token` and assert it survives, exactly as in section 3.

**The unresized embedding when adding chat tokens.** Teams adapting base models into chat models by adding `<|im_start|>`/`<|im_end|>` (the ChatML convention) routinely forget `resize_token_embeddings`. The signature is a loss that starts much higher than the base model's natural loss and frequently spikes or NaNs in the first ~50 steps. This is a documented gotcha across the Hugging Face and ChatML-adoption communities, and the one-line `len(tok) == embedding_rows` assertion catches it instantly.

**The tokenizer-version mismatch.** As model families ship updated tokenizers (added tokens, changed special-token handling), loading weights from one revision with a tokenizer from another produces either a crash (size mismatch) or silent id scrambling (same size, different mapping). The defense — load both from the same repository revision — is simple but easy to violate when assembling a pipeline from cached components or mixing a base model's weights with a derivative's tokenizer.

The thread connecting all four is that *the loss curve never warned anyone*. These bugs corrupt the data below the level the optimizer can observe, so the only honest detector is to inspect the tokens directly. That is why this post elevates `decode(input_ids)` from a debugging trick to a standing pre-flight check.

| Bug | Where it hides | Loss-curve signature | Instrument that catches it |
| --- | --- | --- | --- |
| Double BOS | data (tokenizer + template) | clean descent, looks normal | `count(bos_id)` per example |
| Missing EOS | data (formatting / masking) | clean descent, looks normal | generation-length histogram |
| Unresized embedding | model code (vocab vs table) | starts high, spikes / NaNs early | `len(tok)` vs embedding rows |
| Tokenizer mismatch | data (wrong tokenizer) | high loss or hard crash | tokenizer vs model-card diff |
| Truncation eats answer | data (max_length / side) | clean but answers incomplete | decode last 30 tokens, assert EOS |
| Pad == EOS unmasked | data (collator masking) | mild, stops too early/late | decode unmasked EOS positions |

## 11. When this is (and isn't) your tokenizer bug

Bisection is as much about clearing suspects as confirming them, so be decisive about when a symptom points *away* from tokenization.

**It is probably a tokenizer bug when** the model trains cleanly (smooth loss, no NaN, healthy grad-norm) but *behaves* wrongly at inference — won't stop, starts with a stray token, ignores chat structure — and the wrongness is *systematic*, identical across every example, rather than random. Systematic-behavior-despite-clean-training is the fingerprint of corruption in the data pipeline, and the tokenizer is the first stage of that pipeline. Decode the batch first.

**It is probably not a tokenizer bug when** the loss curve itself is sick. A smooth-then-NaN curve is numerics (fp16 underflow, log0), not tokenization — go to the NaN-hunting playbook. A loss that won't descend at all on a single overfit batch is a model-code or optimization bug — go to the overfit-one-batch test. A loss that descends but the *metric* and the loss disagree on the *training distribution* (not just at inference) is more likely a loss-function bug — wrong reduction, logits-vs-probabilities — than a tokenizer bug. And a model that trains and generalizes fine but is slightly worse than expected with no behavioral oddity is more likely a hyperparameter or data-quality issue than a structural tokenizer corruption, which tends to produce *qualitative* failures (won't stop, garbage tokens) rather than a few points of metric.

The decisive move when you are unsure is the cheapest one: **decode one batch.** It costs a second and it cleanly partitions the space. If the decoded text looks exactly like what you intended — right structure, one BOS, one unmasked EOS, the completion correctly unmasked, no byte-fallback garbage, the answer intact — then the tokenizer is *cleared* and you should stop blaming it and move up the stack to the model, the loss, or the optimizer. If it looks wrong, you have found your bug in seconds. Either way, the decode-the-batch test earns its place as the first thing you run on any LLM finetune that misbehaves.

This routing — *which* of the six places, and *which cheapest test* settles it — is the backbone of [the taxonomy and decision tree for training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and the full symptom-to-fix workflow is assembled in [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook). Tokenization bugs interact closely with three sibling failure modes in the LLM track: [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), where `-100` is applied to the wrong span (including masking the EOS we needed); [attention-mask and padding bugs for LLMs](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms), where pad tokens leak into attention or loss; and [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs), the train-versus-inference template skew that the double-special-token bug is one instance of. If your tokenizer is clean but the finetune still misbehaves, the broader recipe pitfalls live in [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it).

## 12. Building tokenization checks into the pipeline

The reason tokenization bugs survive so long is that nobody looks until something is on fire. The cure is to make the looking automatic — turn the ad-hoc decode into a standing assertion that runs on every dataset build, so a regression fails loudly at preprocessing time instead of silently at inference time three days later. Here is a compact validation pass to run once over a sample of the tokenized dataset before any GPU is reserved.

```python
def validate_tokenized_dataset(dataset, tok, model=None, sample=64):
    """Pre-flight: fail loudly on the common tokenizer bugs before training."""
    bos, eos = tok.bos_token_id, tok.eos_token_id
    problems = []

    # (1) Vocabulary integrity: tokenizer must not emit ids the embedding lacks.
    if model is not None:
        rows = model.get_input_embeddings().weight.shape[0]
        if len(tok) > rows:
            problems.append(f"len(tok)={len(tok)} > embedding rows={rows}: resize_token_embeddings!")

    for ex in dataset.select(range(min(sample, len(dataset)))):
        ids = ex["input_ids"]
        labels = ex.get("labels")

        # (2) Exactly one BOS (if the model uses one) and it is at position 0.
        if bos is not None:
            n_bos = sum(1 for t in ids if t == bos)
            if n_bos != 1:
                problems.append(f"BOS count {n_bos} (want 1); double/missing BOS")
            elif ids[0] != bos:
                problems.append("BOS present but not at position 0")

        # (3) At least one EOS, and the LAST real EOS is unmasked in labels.
        if eos is not None:
            eos_pos = [i for i, t in enumerate(ids) if t == eos]
            if not eos_pos:
                problems.append("no EOS in sequence: model won't learn to stop")
            elif labels is not None and all(labels[i] == -100 for i in eos_pos):
                problems.append("every EOS is masked (-100): no gradient to stop")

        # (4) Some unmasked label exists (the example actually trains on something).
        if labels is not None and all(l == -100 for l in labels):
            problems.append("ALL labels are -100: example contributes zero loss (truncated answer?)")

    if problems:
        from collections import Counter
        for msg, count in Counter(problems).most_common():
            print(f"  [{count:>3}x] {msg}")
        raise AssertionError(f"{len(problems)} tokenization problems found across {sample} examples")
    print(f"OK: {sample} examples passed BOS/EOS/label/vocab checks")
```

Wire this into your data-build step and into CI if you can. The four checks it encodes — vocabulary integrity, exactly-one-BOS-at-position-0, at-least-one-unmasked-EOS, and not-all-labels-masked — are precisely the four bugs that cost the most engineer-days in this post. The reason a *standing* check beats a one-time manual decode is that tokenizer behavior changes under you: a teammate bumps the `transformers` version, a new chat template ships with the next base model, someone swaps `use_fast`, or a dataset gets re-exported with different spacing. Any of these can silently reintroduce a bug you fixed last month, and only an assertion that runs on every build will catch the regression. Manual vigilance does not scale; encoded invariants do. A failed assertion at preprocessing time costs a second; the same bug discovered after a multi-hour run costs the run, the GPU-hours, and the day you spend mis-diagnosing it. At roughly \$2 to \$3 per GPU-hour for a single A100 and an 8-hour finetune, each silently-corrupted run you prevent is on the order of \$20 of compute plus the far larger cost of your time chasing a ghost. The check pays for itself the first time it fires.

## 13. Key takeaways

- **Decode the batch.** `tokenizer.decode(input_ids)` on one real training example is the single highest-leverage check in LLM finetuning. Read it as text and most tokenizer bugs are visible in seconds — before any GPU time.
- **Double BOS = clean loss, low score.** A duplicated beginning-of-sequence token (tokenizer adds one, template adds another) shifts the whole next-token map by one. The loss curve looks perfect; the benchmark sits a few points low. Count `bos_token_id` per example; it must be exactly one.
- **Missing EOS = won't stop.** If no unmasked EOS appears in the labels, the model gets zero gradient to predict a stop and runs to the length cap. The detector is the generation-length histogram pinned at `max_new_tokens`. Append `tok.eos_token` and confirm its label is not `-100`.
- **`add_special_tokens` has one owner.** Whoever builds the structured string (chat template or manual concat) adds the special tokens; everyone downstream tokenizes with `add_special_tokens=False`. Two owners means a double BOS.
- **Resize after adding tokens.** Growing `len(tokenizer)` does not grow the embedding. Call `model.resize_token_embeddings(len(tok))` and mean-initialize new rows. The catch is a one-line `len(tok) == embedding_rows` assertion; the symptom is a loss that starts high and spikes early.
- **Load tokenizer and model from the same repo.** A mismatched tokenizer maps your text to a stranger's vocabulary — a crash if sizes differ, silent garbage if they coincide.
- **Protect the completion from truncation.** Measure the length distribution, set `max_length` to a high percentile, and use `truncation_side="left"` for instruction tuning so the answer and its EOS survive.
- **PAD == EOS needs guardrails.** Reusing the EOS id for padding is fine only if pad positions are `-100` in labels and `0` in the attention mask, and the one real terminating EOS stays unmasked.
- **The loss curve is blind to tokenizer bugs.** They corrupt the data below the optimizer's view. Trust the decoded tokens, not the curve.

## 14. Further reading

- Sennrich, Haddow, and Birch, "Neural Machine Translation of Rare Words with Subword Units" (2016) — the paper that introduced BPE to neural sequence models and the foundation of modern subword tokenization.
- Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (2018) — the tokenizer underpinning Llama and many others, including the `▁` whitespace convention.
- Hugging Face `transformers` documentation, the tokenizer summary and the `PreTrainedTokenizer` API reference — the authoritative source for `add_special_tokens`, `truncation_side`, `apply_chat_template`, and `resize_token_embeddings`.
- Hugging Face `tokenizers` library documentation — the fast Rust tokenizers, byte-level BPE, and the fast-vs-slow behavior notes.
- The `trl` `SFTTrainer` documentation and issue tracker — the canonical source for the EOS-not-appended and completion-masking behaviors discussed in the case studies.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom-to-suspect-to-test decision tree this post instantiates for the data layer.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone that assembles every track, including the pre-flight tokenizer checks, into one workflow.
- [The loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) and [chat-template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs) — the two sibling LLM-track posts that pick up where tokenization ends.
