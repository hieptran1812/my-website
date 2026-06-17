---
title: "Chat Template and Formatting Bugs: Train-Serve Skew and the Model That Won't Stop"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose and fix the silent chat-template mismatches that make a finetuned LLM ignore instructions or ramble forever — by diffing the exact strings, checking EOS is in the labels, and confirming generation actually stops."
tags:
  [
    "debugging",
    "model-training",
    "chat-template",
    "llm",
    "finetuning",
    "deep-learning",
    "nlp",
    "transformers",
    "trl",
    "tokenization",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/chat-template-and-formatting-bugs-1.png"
---

You finetuned a 7B model on 40,000 high-quality instruction pairs. The loss curve is textbook: a clean descent from 1.9 to 0.7 over three epochs, no spikes, no NaNs. Eval perplexity on a held-out split looks fine. You merge the adapter, load it into vLLM, send the first real prompt — "Summarize this paragraph in one sentence" — and the model produces a summary, then keeps going. It writes a second summary. It starts a fictional dialogue. It invents a user turn and answers it. Four hundred tokens later it hits the generation cap mid-word and the server cuts it off. You try a different prompt. This time it ignores your instruction entirely and free-associates about the topic. The base model — the one you started from — handled both prompts perfectly.

Nothing crashed. No exception, no warning, no red line in the logs. The training run was, by every instrument you looked at, healthy. And yet the model is broken in a way that is both catastrophic (it won't follow instructions, it won't stop) and completely invisible to the loss curve. This is the signature of a **chat-template bug**: a mismatch between the exact string format the model was trained on and the exact string format it is served with, or a formatting mistake that silently removes the one token the model most needed to learn — the one that says "this turn is over."

This post is about that class of bug. It is one of the most common and most demoralizing failure modes in LLM finetuning precisely because every conventional instrument says you are fine. The fix is almost always trivial once you see it. The hard part is seeing it, because the bug lives in a layer most people never look at: the formatting layer between your list of `{"role": ..., "content": ...}` message dictionaries and the integer token IDs the model actually consumes. Figure 1 shows where that layer sits and why a defect in it is so hard to spot from the loss.

![Diagram showing the chat-template layer positioned between raw message dictionaries and the tokenized model input, with serving required to match training](/imgs/blogs/chat-template-and-formatting-bugs-1.png)

By the end you will be able to take any finetuned chat model that ignores instructions, rambles forever, echoes role headers, or works in your notebook but breaks in production, and localize the cause in minutes. The single most powerful move — the one diagnostic that subsumes most of the others — is to **render the exact formatted string your model trains on, render the exact string it is served with, and diff them character by character**. We will build that diff, plus an EOS-in-the-labels check and a generation-stops test, into copy-and-run code. We will also do the science: *why* a finetuned model is, mathematically, a distribution over exactly-the-format-it-saw, *why* a masked end token guarantees the model never learns to stop, and *why* a single missing newline can move the model off-distribution.

This is the [chat-template chapter](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) of the series' running frame: a training bug hides in one of six places — **data, optimization, model code, numerics, systems, or evaluation** — and you bisect to the right one before touching code. Chat-template bugs are a data-and-evaluation bug wearing a model-quality costume. The loss is fine because the loss only measures how well the model predicts the (mis)formatted training data. The model is broken because the thing you serve is not the thing you trained. Let's pull that apart.

## 1. The symptom: a healthy loss curve and a broken model

Start by naming the symptoms precisely, because each one points at a slightly different sub-cause and the differential diagnosis matters.

There are four recurring symptoms of a chat-template bug, and a good finetuner learns to recognize each one on sight:

1. **The model ignores instructions.** You ask it to "answer in JSON" or "be concise" and it does neither. It behaves more like a base completion model than an instruction-tuned one — it continues your text rather than responding to it. This is the classic **train-serve template skew**: the model learned to respond to one format and you are feeding it another, so from the model's point of view your prompt is gibberish in an unfamiliar layout, and it falls back to raw next-token continuation.
2. **The model won't stop generating.** It answers, then keeps writing — a second answer, an invented next turn, a runaway monologue — until it hits `max_new_tokens`. This is the **EOS-not-learned** bug: the end-of-turn token was never in the loss, so the model never raised its probability of emitting it, so at inference there is no token that ever triggers the stop condition.
3. **The model echoes role headers or repeats the turn structure.** Its output starts with something like `assistant` or `<|im_start|>assistant` or repeats `### Response:`. This means the generation prompt was handled inconsistently — usually `add_generation_prompt` was set wrong, or the template's assistant header leaked into the part the model is asked to produce.
4. **It works in my notebook but breaks in production.** Locally you call `apply_chat_template` and it is great; in the serving stack someone hand-formats the prompt, or the serving template differs by a newline, and the quality falls off a cliff. This is train-serve skew again, but now the skew is between two *serving* paths, and it is maddening because the model file is identical.

What all four share is the defining feature of this bug class: **the training loss does not move when the bug is present.** That is not a coincidence; it is a mathematical consequence of what the loss measures, and understanding it is the key to never being fooled again. Let's make it rigorous.

### Why the loss can't see it

A causal language model is trained to minimize, over its training distribution $\mathcal{D}$ of token sequences $x = (x_1, \dots, x_T)$, the average negative log-likelihood of each token given its predecessors:

$$\mathcal{L} = -\frac{1}{|\mathcal{M}|}\sum_{t \in \mathcal{M}} \log p_\theta(x_t \mid x_{<t})$$

where $\mathcal{M}$ is the set of positions that contribute to the loss (the unmasked positions — in instruction finetuning, typically the assistant-response tokens). The crucial observation is that $\mathcal{D}$ here is **the formatted training data**, in whatever format your pipeline produced. If your pipeline wraps every example as `User: {q}\nAssistant: {a}`, then the model is learning $p_\theta(\text{token} \mid \text{this exact layout})$. The loss measures fit to *that* distribution and nothing else.

So a finetuned model is, quite literally, a conditional distribution over **exactly the format it was trained on**. The loss going down means "the model has gotten good at predicting the next token *inside this specific string layout*." It says nothing about whether that layout is the one you will serve. If you train on layout $A$ and serve with layout $B$, you are evaluating $p_\theta(\cdot \mid \text{context in layout } B)$ — a region of input space the model saw rarely or never during finetuning. The model's behavior there is governed by whatever the *base* model's pretraining instilled, plus some bleed from finetuning, but the carefully learned instruction-following behavior is keyed to layout $A$. You are, in effect, prompting a different model than the one your loss curve described.

This is why the bug is invisible to every standard instrument. Grad norm is fine; the gradients are real, they are just teaching the model the wrong format's conditional. Loss is fine; it descends on the format you trained. Even held-out perplexity on a same-format validation set is fine, because that set shares the bug. The only instrument that catches it is one that compares the *training string* to the *serving string* — and almost nobody logs those.

### Why even a single newline matters

It is worth pausing on the claim that "a single missing newline" can degrade a model, because it sounds like superstition and it is not. The mechanism is concrete and lives in the tokenizer. A language model does not see characters; it sees token IDs, and the mapping from characters to IDs is *context-sensitive* under byte-pair encoding. The string `assistant\n` does not tokenize to "the `assistant` token, then the newline token" in isolation; the trailing newline often merges with the surrounding bytes into a token like `assistant\n` (one ID) or shifts how the *next* characters tokenize. Change `\n` to `\n\n`, or drop it entirely, and the boundary token IDs change. Now the model, which during finetuning always saw token ID sequence $S_A$ at the assistant boundary, is served sequence $S_B$ where $S_A \ne S_B$ at exactly the position where it must "decide" to start answering in the learned style.

How big is the effect? It depends on how far the perturbed prefix is from the trained one in the model's internal representation, which you cannot compute in closed form — but you can bound the intuition. The assistant-boundary tokens are the *most heavily conditioned-on* positions in the whole sequence: every token the model generates attends back to them. A perturbation there propagates forward through every subsequent step. Contrast that with a one-token perturbation deep in the middle of a long user message, which the model can mostly route around. The sensitivity is highest exactly where templates differ — at the role boundaries and the generation seam — which is why "small" formatting differences produce outsized behavioral changes. The corollary for debugging: when you diff strings, **pay closest attention to the bytes around the role headers and the assistant boundary.** A difference in the middle of the user content is usually harmless; a difference at the seam is usually the bug.

There is a second, statistical way to see why same-format validation can't save you. Suppose your finetuning format introduces a systematic bias $b$ (say, the model never learns to stop because EOS is masked). The validation loss is computed on the same biased format, so the bias is *baked into the reference distribution* the loss compares against — the loss does not penalize the model for being wrong about EOS because the validation labels also omit the gradient there. Formally, if the loss is taken over the same masked set $\mathcal{M}$ that excludes EOS, then both training and validation are blind to $\ell_{\text{EOS}}$, and no amount of validation tells you the model can't stop. The metric and the bug are correlated, so the metric's power to detect the bug is zero. This is the precise, quantitative reason you need an *out-of-band* check (a string diff, an EOS-in-labels assert, a generation-stops test) that does not flow through the same masked loss.

> The deeper lesson, which recurs across the whole series: **a metric that shares your bug cannot detect your bug.** Same-format validation perplexity is contaminated by exactly the formatting defect you are trying to find, so it will happily report health. To catch train-serve skew you need an out-of-band check — the literal string diff — that does not pass through the same code path. This is the same reason a [data leak](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) inflates your validation AUC: the validation set is downstream of the same contamination.

## 2. What a chat template actually is

Before we can debug a chat template we need to be precise about what it is, because vagueness here is where most bugs are born.

A **chat template** is a deterministic function that turns a list of role-tagged messages into a single string of tokens. Concretely, you have input like:

```python
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]
```

and the template renders it into a flat string that the tokenizer then encodes. Different model families use different conventions. A ChatML-style model (used by many Qwen and OpenAI-lineage models) renders the above roughly as:

```bash
<|im_start|>system
You are a terse assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

A Llama-3-style model uses a different set of special tokens — `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` (end of turn):

```bash
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a terse assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Paris.<|eot_id|>
```

A Mistral-instruct-style model uses `[INST]` and `[/INST]` brackets and a leading `<s>` BOS token, with no explicit role names at all:

```bash
<s>[INST] What is the capital of France? [/INST] Paris.</s>
```

Three things should jump out. First, the **special tokens differ by family** — `<|im_end|>` versus `<|eot_id|>` versus `</s>` all mean "turn is over," but they are different token IDs in different vocabularies, and using the wrong one is a real bug. Second, the **whitespace is load-bearing**: notice the blank line after the Llama header (that is two newlines, `\n\n`), the single space after `[INST]` in Mistral, the newline after `<|im_start|>system` in ChatML. These are not cosmetic. They are part of the byte sequence the model was pretrained and finetuned to expect, and changing them moves you off-distribution. Third, **the end-of-turn token is part of the assistant content during training** — the `<|im_end|>` / `<|eot_id|>` / `</s>` that closes the assistant turn is exactly the token the model must learn to emit to stop. If your formatting drops it or masks it, you get the won't-stop bug.

To make the family differences concrete and debuggable, here is a side-by-side of the moving parts that actually cause bugs when you mix them up. The point of this table is not to memorize formats — `apply_chat_template` does that for you — but to recognize *which knobs differ*, so when you read a `repr()` dump you know what "correct" looks like for that family.

| Property | ChatML (Qwen-style) | Llama-3-style | Mistral-instruct |
|---|---|---|---|
| Role/turn open tokens | `<|im_start|>role` | `<|start_header_id|>role<|end_header_id|>` | `[INST]` / no role |
| End-of-turn token | `<|im_end|>` | `<|eot_id|>` | `</s>` |
| BOS in template? | usually no explicit BOS | yes, `<|begin_of_text|>` | yes, `<s>` |
| Whitespace that bites | `\n` after role open | `\n\n` after header | single spaces inside brackets |
| Default system prompt? | none injected | none / version-dependent | none |
| Common skew failure | wrong end token if mixed with Llama | double BOS on re-encode | space-after-`[INST]` dropped |

The recurring lesson across the row labeled "end-of-turn token": if you finetune a Qwen base but copy a Llama tutorial's stop-token config, you will train the model to emit `<|im_end|>` while telling the serving stack to stop on `<|eot_id|>` — which is not even in Qwen's vocabulary the same way — and the model will never halt. The mismatch is mechanical and the table makes it visible.

In Hugging Face `transformers`, the template is not Python code you write; it is a **Jinja string stored on the tokenizer** as `tokenizer.chat_template`, and you invoke it with `tokenizer.apply_chat_template(messages, ...)`. The single most important rule in this entire post is: **use the tokenizer's built-in `apply_chat_template`, the same way, on both the training and serving sides, and never hand-roll the format string.** Almost every bug in this article is some variation of breaking that rule. Figure 2 makes the contrast concrete.

![Before-and-after comparison showing a hand-rolled training format that skews from the served template versus a single shared chat template applied identically on both sides](/imgs/blogs/chat-template-and-formatting-bugs-2.png)

### `apply_chat_template`, correctly

Here is the correct, idiomatic usage. The key flag is `add_generation_prompt`, which controls whether the rendered string ends *after* the assistant header (so the model continues from there) or includes the full assistant turn.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# For INFERENCE: end the string at the assistant header so the model
# generates the answer. add_generation_prompt=True appends the opening
# of the assistant turn (e.g. "<|im_start|>assistant\n").
prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(repr(prompt))
# '<|im_start|>system\nYou are a terse assistant.<|im_end|>\n
#  <|im_start|>user\nWhat is the capital of France?<|im_end|>\n
#  <|im_start|>assistant\n'

# For TRAINING: include the full assistant turn (the target), with its
# closing end-of-turn token. add_generation_prompt=False.
full = messages + [{"role": "assistant", "content": "Paris."}]
train_str = tok.apply_chat_template(
    full,
    tokenize=False,
    add_generation_prompt=False,
)
print(repr(train_str))
# '...<|im_start|>assistant\nParis.<|im_end|>\n'
```

Read those two `repr()` outputs carefully, because `repr()` is your best friend in this whole domain — it shows you the literal newlines (`\n`) and exact tokens that a normal `print` would hide. Notice the asymmetry: the inference string ends with `<|im_start|>assistant\n` and waits for the model to produce `Paris.<|im_end|>`. The training string includes `Paris.<|im_end|>` as the thing the model is supposed to predict. The **assistant-turn boundary** — where the prompt ends and the target begins — is exactly the seam where train and serve must line up. Figure 3 shows how a single template feeds both call sites and differs only in this flag.

![Dataflow graph showing one apply_chat_template function feeding both a training call with the generation prompt off and a serving call with it on, splitting at the assistant boundary](/imgs/blogs/chat-template-and-formatting-bugs-3.png)

## 3. The definitive diagnostic: diff the exact strings

If you remember one technique from this post, make it this one. Almost every chat-template bug — skew, wrong special tokens, missing newlines, double BOS, generation-prompt mistakes — is caught by a single check: **render the exact string your model sees during training, render the exact string it sees during serving, and diff them.** No exceptions, no clever inference. Just look at the two strings and find where they disagree.

The reason this works is the science from §1: the model is a distribution over exactly the format it trained on. So the question "is there a train-serve bug?" reduces to "do the two formats differ?" — and that is a literal string comparison. It is the most boring diagnostic imaginable and it is the most powerful, because it does not pass through the same buggy code path that produced the contaminated metrics.

Here is the diagnostic as runnable code. It takes a single example, renders it the way training does and the way serving does, and prints a character-level diff.

```python
import difflib
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# A representative example. Use a REAL row from your dataset.
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
answer = "Paris."

# --- How TRAINING formats this example (paste your real code here) ---
# The bug usually lives in THIS function. If you hand-roll, reproduce it.
def training_format(messages, answer):
    s = ""
    for m in messages:
        s += f"{m['role'].capitalize()}: {m['content']}\n"
    s += f"Assistant: {answer}"        # <-- no special tokens, no EOS!
    return s

# --- How SERVING formats this example (your inference path) ---
def serving_format(messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

train_str = training_format(messages, answer)
serve_str = serving_format(messages)  # serving stops before the answer

print("=== TRAIN string (repr) ===")
print(repr(train_str))
print("\n=== SERVE string (repr) ===")
print(repr(serve_str))

# Character-level diff of the shared prefix (the prompt portion).
print("\n=== DIFF (prompt portion) ===")
for line in difflib.unified_diff(
    serve_str.splitlines(keepends=True),
    train_str.splitlines(keepends=True),
    fromfile="serve", tofile="train", lineterm="",
):
    print(line, end="")
```

Run this and the bug is right there in the output: training uses `User:` / `Assistant:` plain-text headers with no special tokens, while serving uses `<|im_start|>user\n` and `<|im_end|>`. The two strings share almost nothing. The model trained on the left, gets served the right, and behaves like it has never seen an instruction in its life — because, in this layout, it hasn't.

The discipline this encodes: **never trust that two code paths produce the same format; prove it.** Print both with `repr()` so newlines and special tokens are visible. The most insidious version of this bug is not a wholesale format difference like the one above — it is a *single missing newline* or a `\n` versus `\r\n`, which a casual `print` renders identically but which the tokenizer encodes as different IDs. `repr()` and a character diff catch those; eyeballing does not.

#### Worked example: the hand-rolled format that cost a week

A team finetunes Qwen2.5-7B for a customer-support assistant. Their data-prep script, written before they adopted `apply_chat_template`, formats each example as `### Instruction:\n{q}\n\n### Response:\n{a}` — the old Alpaca format. The loss descends beautifully: 2.1 → 0.6 over two epochs. They deploy behind vLLM, which serves Qwen with its native ChatML template (`<|im_start|>` / `<|im_end|>`). In production the model ignores the system prompt, answers off-topic, and frequently continues past its answer. Their internal win-rate eval (judged by a stronger model) reads **24%** — worse than the un-finetuned base.

The on-call engineer spends a day suspecting the LoRA adapter (it merged fine), then the data quality (it is clean), then the LR (it is reasonable). On day two they run the string diff above on one example. The training string is `### Instruction:\n...\n\n### Response:\n...` with no special tokens. The serving string is `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`. They share zero structure. The fix is one line in data prep: replace the hand-rolled `f`-string with `tok.apply_chat_template(messages, tokenize=False)`. They re-run the finetune; win rate goes to **61%**. The week's lesson, taped above the desk: *diff the strings on day zero, not day seven.*

This worked example is the entire post in miniature: the loss lied, the standard suspects were innocent, and a five-line string diff localized the bug in seconds once someone finally ran it.

### The subtler skew: a single newline

The §3 worked example was a *wholesale* skew — two completely different formats — which is at least easy to see once you diff. The more insidious version is a near-match that differs by one character, because that one survives a casual visual review and only a `repr()` or a programmatic diff catches it. Here is the pattern I have seen waste the most time.

A team adopts `apply_chat_template` (good) but their serving stack, for historical reasons, builds the prompt slightly differently — it appends the assistant header without the trailing newline, so training ends every prompt with `<|im_start|>assistant\n` and serving ends with `<|im_start|>assistant` (no newline). To a human reading the logs these look identical; the newline is invisible. But the tokenizer encodes `assistant\n` and `assistant` as different final token IDs, so at generation time the model is asked to continue from a token it never saw at that position during training. The symptom is mild and confusing: the model mostly works but occasionally starts its answer with a stray space or newline, or its first word is subtly off, or its instruction-following is a few points worse than the offline eval suggested. Mild, format-shaped, and reproducible — the chat-template signature.

The diagnostic is the same string diff, but you must look at the *tail*, and you must use `repr()`:

```python
serve_tail = serving_format(messages)[-30:]
train_tail = training_format(messages, "")[:-0][-30:]  # prompt portion
print("serve tail:", repr(serve_tail))   # '...<|im_start|>assistant'      <- no \n
print("train tail:", repr(train_tail))   # '...<|im_start|>assistant\\n'   <- has \n
# The diff is a single '\n'. Invisible without repr(). Fatal to the seam.
```

The fix is to make serving call the identical `apply_chat_template(..., add_generation_prompt=True)` rather than reconstructing the header. The general principle, which I will keep hammering: **the seam between prompt and generation is the highest-sensitivity position in the sequence (§1), so a one-character difference *there* hurts far more than a one-character difference in the middle of the user message.** When you diff, weight the seam.

#### Worked example: the newline that cost three eval points

A retrieval-augmented assistant finetuned on Qwen2.5-7B passes its offline eval at **72%** answer-correctness but lands at **68%** in the production A/B, a gap nobody can explain — the model file is identical, the prompts look identical. Someone finally runs `repr()` on the production prompt and the offline prompt for the same query. The offline harness uses `apply_chat_template(..., add_generation_prompt=True)` and ends in `<|im_start|>assistant\n`. The production service hand-builds the prompt and ends in `<|im_start|>assistant` — one missing newline. They point production at `apply_chat_template`; the next A/B reads **72%**, matching offline. Four points of "model regression" was one byte. The honest way they confirmed it was the byte: a unit test asserting `production_prompt == offline_prompt` for a fixed query, which failed before the fix and passed after — no model retraining required, because the model was never the problem.

## 4. The model that won't stop: EOS and the science of stopping

The won't-stop bug deserves its own deep treatment, because it is the most confusing symptom (the model clearly learned *something* — it answers correctly, then keeps going) and because the science of *why* a model stops is genuinely illuminating.

### How a model stops at all

A causal LM never "decides to stop" in any deep sense. Generation is a loop: sample a token, append it, feed it back, repeat. The loop terminates when one of two things happens: you hit `max_new_tokens`, or the model emits a token that the decoder is configured to treat as a stop signal — the **EOS** (end-of-sequence) or, in chat models, the **end-of-turn** token (`<|im_end|>`, `<|eot_id|>`, `</s>`). The decoder checks each sampled token ID against `eos_token_id` (which can be a list) and halts when it matches.

So "the model stops" means exactly: **at the end of a coherent turn, the model assigns high enough probability to the end-of-turn token that it gets sampled (or, under greedy decoding, becomes the argmax).** That is a learned behavior. The base pretrained model has some weak prior toward ending documents, but instruction finetuning is where the model learns the much sharper behavior "after I finish answering a turn, emit the end-of-turn token." For the model to learn that, the end-of-turn token must appear in the training targets *and contribute to the loss*. If it does not, the model never gets a gradient pushing $P(\text{EOS} \mid \text{end of a complete answer})$ upward, and that probability stays at its near-zero pretraining baseline.

Let's make this rigorous, because it is the crux. In finetuning, the loss at the position right after the last content token of an assistant turn is

$$\ell_{\text{EOS}} = -\log p_\theta(\text{EOS} \mid \text{full turn so far}).$$

Minimizing this drives $p_\theta(\text{EOS} \mid \cdots) \to 1$ at turn boundaries — that is *literally* the model learning to stop. But this term is only in the sum $\mathcal{L} = \sum_{t \in \mathcal{M}} \ell_t$ **if the EOS position is in the unmasked set $\mathcal{M}$.** If your label-masking marks the EOS position as `-100` (PyTorch's `ignore_index`, the standard "don't compute loss here" sentinel), then $\ell_{\text{EOS}}$ is never computed, never backpropagated, and $P(\text{EOS})$ at turn boundaries is left untouched. The model becomes excellent at producing answers and completely unable to end them.

This connects directly to the [loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug): the same mechanism that correctly masks the *prompt* tokens out of the loss can, if you are one token too greedy, also mask the *EOS* token out of the loss. The boundary between "this is prompt, mask it" and "this is target, train on it" must fall *before* the EOS, not after. Figure 4 lays out the decision tree for diagnosing a non-stopping model.

![Decision tree for diagnosing a model that will not stop, branching into a training-side cause of an unlearned end token and a serving-side cause of a wrong stop configuration](/imgs/blogs/chat-template-and-formatting-bugs-4.png)

### The two flavors of "EOS not learned"

There are two distinct ways the end-of-turn token fails to be learned, and they need different fixes:

1. **The template omits the EOS entirely.** Your formatting code builds the assistant turn but never appends `<|im_end|>` / `<|eot_id|>` / the tokenizer's `eos_token`. There is no end token in the string at all, so there is nothing for the model to learn to emit. This is common with hand-rolled formats (the Alpaca `### Response:\n{a}` ends with the answer and a newline — no special end token). Fix: use `apply_chat_template`, which appends the correct end-of-turn token, or, if you must hand-roll, explicitly append `tok.eos_token`.

2. **The template includes the EOS but masking removes it from the loss.** The end token is in `input_ids`, but the label-construction logic sets it to `-100`. The model sees the token as input (so it is in-context during teacher forcing) but never gets a gradient to *produce* it. Fix: make sure the masking boundary includes the EOS position in the loss.

Both produce the identical symptom — the model never stops — so you must check both. The check is concrete and is the subject of the next section.

### Diagnostic: is EOS actually in the labels?

Here is the runnable check. It takes one fully-built training example — `input_ids` and `labels` after your collator has run — and verifies two things: the EOS token is present in `input_ids`, and its position is *not* masked in `labels`.

```python
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
eos_id = tok.eos_token_id              # for Qwen2.5 this is <|im_end|>
print("eos_token:", repr(tok.eos_token), "id:", eos_id)

# `batch` is one example AFTER your data collator. Pull a real one.
def check_eos_learnable(input_ids, labels, eos_id):
    input_ids = torch.as_tensor(input_ids)
    labels = torch.as_tensor(labels)

    eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) == 0:
        print("FAIL: no EOS token in input_ids — template omits the end token")
        return False

    # Is at least one EOS position actually IN the loss (label != -100)?
    learned = [(int(p), int(labels[p])) for p in eos_positions]
    in_loss = [p for p, lab in learned if lab != -100]
    print("EOS positions (idx, label):", learned)
    if not in_loss:
        print("FAIL: every EOS position is masked (-100) — model can't learn to stop")
        return False

    print(f"OK: EOS present and in loss at positions {in_loss}")
    return True

# Example usage with a tiny constructed batch:
# input_ids = [...]; labels = [...] from your collator
check_eos_learnable(batch["input_ids"][0], batch["labels"][0], eos_id)
```

There are two failure messages and they map exactly to the two flavors above. "No EOS in input_ids" means your template dropped it. "Every EOS position is masked" means your label logic ate it. Either way you now know which fix to apply, and you have a regression test you can keep forever. Figure 5 shows the before/after on the actual stopping behavior.

![Before-and-after comparison of EOS masked out of the loss versus kept in the loss, showing the end-of-turn probability rising and runaway generations dropping to near zero](/imgs/blogs/chat-template-and-formatting-bugs-5.png)

#### Worked example: from rambling to clean stops

A finetune of Llama-3-8B for summarization produces correct summaries that never end — every generation runs to the full 512-token cap, and **83%** of outputs are truncated mid-sentence by the server. Greedy decoding, so it is not a sampling problem. The team measures $P(\text{eot\_id})$ at the true end of a held-out summary: it is **0.001** — essentially the pretraining baseline, untouched by finetuning.

They run `check_eos_learnable` on a training example. Output: `EOS positions (idx, label): [(247, -100)]`. The `<|eot_id|>` is present in `input_ids` at position 247 but its label is `-100` — masked. Their custom collator masked everything from the assistant header onward *up to and including* the trailing end token, off by one position. They change the masking boundary to keep the EOS position in the loss (`labels[eos_pos] = input_ids[eos_pos]`), re-run one epoch, and re-measure: $P(\text{eot\_id})$ at turn end is now **0.94**, and the truncation rate drops from 83% to **0.4%**. The summaries stop cleanly. Nothing else in the pipeline changed — same data, same LR, same everything — because nothing else was wrong. One masked token was the entire bug.

The honest way to *measure* this fix, by the way, is not vibes. It is the truncation rate (fraction of generations that hit `max_new_tokens`) on a held-out eval set, plus the mean $P(\text{EOS})$ at the gold turn boundary. Both are cheap to log and both move sharply when you unmask EOS, which is exactly the kind of before→after evidence that distinguishes a real fix from a lucky reroll.

## 5. The generation prompt and the assistant-turn boundary

The third symptom — the model that **echoes role headers** or repeats the turn structure — is almost always a mistake with `add_generation_prompt`, so let's nail down what that flag does and how to get it wrong.

At inference, you want the rendered string to end *exactly* at the point where the model should start producing the assistant's answer. For ChatML that means ending with `<|im_start|>assistant\n` — the opening of the assistant turn, but none of its content. The model then generates `Paris.<|im_end|>`. The `add_generation_prompt=True` flag is what appends that trailing assistant header. If you set it `False` at inference (or forget to set it), the string ends after the *user* turn's `<|im_end|>`, and the model — left to its own devices — often re-emits the assistant header itself (`<|im_start|>assistant`) as part of its output, because that is what comes next in the format it learned. Now your output is polluted with role tags.

Conversely, during *training* you want `add_generation_prompt=False`, because the assistant turn (header *and* content *and* end token) is the target the model learns to produce. If you accidentally set `add_generation_prompt=True` during training data prep and then append the answer yourself, you can end up with a doubled assistant header (`<|im_start|>assistant\n<|im_start|>assistant\nParis.`), which the model dutifully learns to reproduce.

The rule is a clean asymmetry, and it is worth memorizing:

| Context | `add_generation_prompt` | Why |
|---|---|---|
| Training (building targets) | `False` | The full assistant turn is the target; you append nothing |
| Inference (building prompts) | `True` | End at the assistant header so the model generates the answer |

Here is a diagnostic that catches generation-prompt mistakes by inspecting the *tail* of each rendered string — the part that matters for this bug:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [{"role": "user", "content": "Hi"}]

infer = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
train = tok.apply_chat_template(
    messages + [{"role": "assistant", "content": "Hello!"}],
    tokenize=False, add_generation_prompt=False,
)

print("INFER tail:", repr(infer[-40:]))
# '...<|im_start|>assistant\\n'   <- ends AT the assistant header, good
print("TRAIN tail:", repr(train[-40:]))
# '...assistant\\nHello!<|im_end|>\\n'  <- includes content + end token, good

# Red flags to assert against:
assert infer.rstrip().endswith("assistant"), \
    "inference prompt should end at the assistant header (add_generation_prompt=True?)"
assert "<|im_start|>assistant" in train, "training string missing assistant turn"
assert train.count("<|im_start|>assistant") == 1, "doubled assistant header!"
```

The `assert train.count("<|im_start|>assistant") == 1` line is the one that catches the doubled-header bug, which otherwise produces a model that compulsively writes `assistant` before every answer in production. Cheap assert, expensive bug.

### Reproduce the serving format in a unit test

The most durable defense against all of this is to **reproduce the exact serving format in a unit test** that runs in CI. Not "format an example and look at it" — an actual assertion that the training format and the serving format agree on the shared prefix, and that the special tokens you expect are present. Here is the shape of that test:

```python
import pytest
from transformers import AutoTokenizer

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def render_train(tok, messages, answer):
    full = messages + [{"role": "assistant", "content": answer}]
    return tok.apply_chat_template(full, tokenize=False, add_generation_prompt=False)

def render_serve(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def test_train_serve_share_prefix(tok):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Ping?"},
    ]
    train = render_train(tok, messages, "Pong.")
    serve = render_serve(tok, messages)
    # The serving prompt must be an exact prefix of the training string.
    assert train.startswith(serve), (
        "train-serve skew!\nserve tail: %r\ntrain head: %r"
        % (serve[-60:], train[: len(serve) + 20])
    )

def test_end_token_present(tok):
    train = render_train(tok, [{"role": "user", "content": "Ping?"}], "Pong.")
    assert tok.eos_token in train, "training string is missing the end-of-turn token"
```

The `train.startswith(serve)` assertion is the unit-test form of the string diff from §3, and it is the single most valuable test you can add to an LLM finetuning repo. If serving is not a prefix of training, you have skew, full stop. Run it in CI and the class of bug that cost the team in §3 a week can never ship again.

## 6. Special tokens, the vocabulary, and double-BOS

We have been assuming the special tokens (`<|im_start|>`, `<|eot_id|>`, etc.) exist and behave. Several bugs live in *that* assumption, and they interact with the tokenizer in subtle ways.

### Are the role/turn tokens actually in the vocab?

A special token like `<|im_start|>` only works as a single atomic token if it is in the tokenizer's vocabulary as a special token. If it is not — for instance, if you switched base models, or added ChatML tags to a tokenizer that does not know them — then `<|im_start|>` gets tokenized as the *literal characters* `<`, `|`, `im`, `_`, `start`, `|`, `>` (seven-ish tokens of byte-level junk) instead of one clean control token. The model then has to learn the role structure from a noisy multi-token spelling, which it does poorly, and your beautiful template degrades into garbage in token space.

The check is direct: encode a special token and confirm it maps to a single ID that the tokenizer recognizes as special.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

for t in ["<|im_start|>", "<|im_end|>"]:
    ids = tok.encode(t, add_special_tokens=False)
    print(f"{t!r:>16} -> ids {ids}  ({'single token, OK' if len(ids)==1 else 'SPLIT into pieces — BUG'})")
# '<|im_start|>'  -> ids [151644]  (single token, OK)
# '<|im_end|>'    -> ids [151645]  (single token, OK)

# If a token splits, you must add it and resize embeddings:
# tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
# model.resize_token_embeddings(len(tok))
```

If a control token splits into pieces, you add it via `add_special_tokens` and call `model.resize_token_embeddings(len(tok))` so the model has an embedding row for the new ID. Forgetting the resize is its own bug — the embedding matrix and the vocabulary disagree on size, and either you crash on the out-of-range ID or (worse, with some setups) you silently index into a neighboring row.

### The double-BOS interaction

This one is the most common subtle interaction, and it ties back to [tokenization bugs](/blog/machine-learning/debugging-training/tokenization-bugs) generally. Many chat templates already include the beginning-of-sequence token (BOS, e.g. `<|begin_of_text|>` for Llama-3, `<s>` for Mistral) as part of the rendered string. The Jinja template emits it. So when you then call the tokenizer on that rendered string, you must pass `add_special_tokens=False` — otherwise the tokenizer *also* prepends a BOS, and you get **two BOS tokens** at the start of every sequence.

Two BOS tokens is not catastrophic the way a masked EOS is, but it is a real, measurable quality degradation: the model was pretrained with exactly one BOS at position 0, and a doubled BOS is an off-distribution prefix that perturbs the first several tokens of attention. On some models it noticeably weakens the first-token prediction; on instruction-tuned models it can subtly destabilize formatting adherence. The cruel part is that it is invisible unless you count.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [{"role": "user", "content": "Hi"}]

# apply_chat_template with tokenize=True handles special tokens correctly.
ids_ok = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

# The WRONG pattern: render to string, then re-encode WITH special tokens.
s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids_double = tok.encode(s, add_special_tokens=True)   # <-- adds a SECOND BOS

bos = tok.bos_token_id
print("bos_token_id:", bos)
print("correct path  BOS count:", ids_ok.count(bos) if isinstance(ids_ok, list) else sum(int(i==bos) for i in ids_ok))
print("double path   BOS count:", ids_double.count(bos))
# correct path  BOS count: 1
# double path   BOS count: 2   <-- the bug
```

Two defenses. First, prefer `apply_chat_template(..., tokenize=True)`, which tokenizes correctly and will not double the BOS. Second, if you must render to a string and re-encode (as some `trl` configurations do), pass `add_special_tokens=False`. And always, always add the BOS-count assertion to your unit test: `assert input_ids.count(tok.bos_token_id) <= 1`. Figure 6 collects these template bugs with their signatures and tests into one diagnostic matrix.

![Matrix mapping each chat-template bug to its symptom, a confirming string-level test, and the fix, covering skew, masked end token, missing generation prompt, and double beginning-of-sequence token](/imgs/blogs/chat-template-and-formatting-bugs-6.png)

### The tokenizer-version trap

One more special-token bug deserves a callout because it is sneaky and increasingly common: **the template that travels with the checkpoint disagrees with the template you trained against.** Chat templates are versioned artifacts. A model family ships an initial instruct release with one template, then a point release fixes a whitespace quirk or adds a tool-calling section to the template. If you finetuned against an older tokenizer snapshot but your serving stack pulls the *latest* tokenizer for the "same" model, the two templates can differ — a different default system prompt, an extra newline, a renamed special token — and you have train-serve skew with no obvious cause, because both sides genuinely loaded "the tokenizer for model X."

The defense is to **pin the tokenizer and template to an exact revision** and load that same revision on both sides. In `transformers` you pass `revision="<commit-or-tag>"` to `from_pretrained`, and you should save the tokenizer *into your finetuned checkpoint* so serving loads *your* template, not the upstream one:

```python
from transformers import AutoTokenizer

# Pin the EXACT revision so train and serve agree byte-for-byte.
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    revision="a1b2c3d",          # a specific commit, not the moving tag
)

# After finetuning, SAVE the tokenizer with your model so serving uses
# the same chat_template you trained with — it lives in tokenizer_config.json.
tok.save_pretrained("./my-finetuned-model")
# tokenizer_config.json now carries the chat_template field; serving loads it.

# Optional: hash the template so a CI test can detect upstream drift.
import hashlib
h = hashlib.sha256(tok.chat_template.encode()).hexdigest()[:12]
print("chat_template hash:", h)   # assert this matches your expected value in CI
```

That `chat_template` hash is a cheap drift detector: store the expected hash, assert it in CI, and you get a loud failure the day an upstream template change would otherwise have silently reintroduced skew. The principle is the same one running through this whole post — **make the format an explicit, pinned, version-controlled artifact instead of something each side reconstructs from memory** — but the tokenizer-version trap is the form it takes once your model is in production and the upstream repo keeps moving.

## 7. System-prompt handling and multi-turn formatting

Two more places template bugs hide: how you handle the **system prompt**, and how you handle **multi-turn** conversations. Both are subtler than the single-turn case and both produce confusing partial-failure symptoms.

### System-prompt mismatches

The system prompt is where the model's persona, format instructions, and guardrails live, and several bugs cluster around it. The most common: **training with a system prompt and serving without one (or vice versa).** If every training example had a system message and your serving path omits it, the model is again slightly off-distribution — it learned to condition on a system turn that is now absent, and the attention pattern it expects at the start of the sequence is disrupted. The reverse (no system at train, a system prompt at serve) is usually milder but can still cause the model to over- or under-weight the system instructions because it never learned how they interact with the rest of the format.

A trickier one: some templates handle a *missing* system message by injecting a **default** system prompt, and some do not. If your training data has no system message but the serving template silently inserts "You are a helpful assistant," then training saw no system turn and serving always has one — skew, created by the template's own defaulting logic. The only way to know is to render both and look.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

no_sys = [{"role": "user", "content": "Hi"}]
with_sys = [{"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"}]

print("NO system prompt:")
print(repr(tok.apply_chat_template(no_sys, tokenize=False, add_generation_prompt=True)))
print("\nWITH system prompt:")
print(repr(tok.apply_chat_template(with_sys, tokenize=False, add_generation_prompt=True)))

# If the "no_sys" rendering still contains a <|im_start|>system block,
# the template is injecting a DEFAULT system prompt — match that at train time.
```

If the no-system rendering still contains a `system` block, the template injects a default, and your training data must include that same default (or you must disable the injection) to stay on-distribution. The discipline is the same as always: render both paths, diff, eliminate the difference.

### Multi-turn: which turns get loss, and the running context

Multi-turn conversations add a dimension the single-turn case does not have: **which turns contribute to the loss?** The standard and almost-always-correct answer is that **only the assistant turns get loss; the system and user turns are masked context.** You do not want gradients teaching the model to *produce* user messages — that would teach it to hallucinate the human's side of the conversation, which is a real and ugly failure mode. So in a three-turn chat (system, user, assistant, user, assistant), the loss covers only the two assistant turns *including their end-of-turn tokens*, and everything else is `-100`.

Getting this masking right across a multi-turn example is fiddly, because you have to find the assistant spans in the tokenized sequence and mask everything else. The `trl` library's `SFTTrainer` with a chat-format dataset can do this for you via a `DataCollatorForCompletionOnlyLM` or its newer assistant-masking support, but you must configure it with the right response template — the exact token string that marks where the assistant turn begins. If that marker string does not match your template's actual assistant header (down to whitespace), the collator finds no assistant spans and either masks everything (loss is `nan` or zero) or masks nothing (you train on the prompt). Figure 7 shows the diff-the-strings workflow as the ordered diagnostic it should be.

![Timeline of the diagnostic workflow: render the train string, render the serve string, diff them, check the end token is in the labels, then generate and confirm the model stops](/imgs/blogs/chat-template-and-formatting-bugs-7.png)

Here is the `trl` configuration done correctly, with the response template that matches ChatML, plus a sanity print of the resulting labels:

```python
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# The response template MUST match the assistant header in the chat template,
# including the trailing newline. For ChatML that is "<|im_start|>assistant\n".
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tok,
)

# Sanity check ONE example BEFORE you launch a 12-hour run:
example = {"messages": [
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]}
text = tok.apply_chat_template(example["messages"], tokenize=False)
batch = collator([tok(text)])
labels = batch["labels"][0]
ids = batch["input_ids"][0]

# Print which tokens are in the loss (label != -100) — must be ONLY the answer + EOS.
kept = [(tok.decode([i]), int(l)) for i, l in zip(ids, labels) if l != -100]
print("tokens in loss:", kept)
# Expect: [('Paris', ...), ('.', ...), ('<|im_end|>', ...)]  <- answer AND end token
```

That final `print` is the multi-turn equivalent of the EOS check: it shows you, token by token, exactly what the model is being trained to produce. The correct output is the assistant content *plus the end-of-turn token* and nothing else. If you see user tokens in the kept list, you are training on the prompt. If you see the assistant content but *not* `<|im_end|>`, you are back to the won't-stop bug. One print, before the run, saves the run. Figure 8 shows the full multi-turn picture of which turns carry loss.

![Dataflow graph of a multi-turn chat showing system and user turns masked as context while only the assistant turns and their end tokens carry loss, with a branch showing that masking the end token breaks stopping](/imgs/blogs/chat-template-and-formatting-bugs-8.png)

#### Worked example: the multi-turn collator that masked everything

A team trains a multi-turn assistant with `trl`'s `DataCollatorForCompletionOnlyLM`. They copy a `response_template` from a tutorial that used Llama format: `" [/INST]"`. But their base model is Qwen with ChatML, whose assistant header is `<|im_start|>assistant\n`. The collator searches each tokenized example for `" [/INST]"`, finds it in *zero* examples, and — per its default behavior when the marker is absent — masks the **entire** sequence to `-100`. Every label is `-100`. The loss is computed over an empty set: it logs as `nan` on some versions, or stays flat at the initial value on others because there is no gradient at all.

The giveaway is that the loss *does not decrease* — it sits at, say, 2.3 forever, or shows `nan`. The team's instinct is "LR too low," and they crank the LR, which does nothing (there are no gradients to scale). The actual diagnostic is the `kept = [...]` print above: it returns `[]` — an empty list — meaning *no tokens are in the loss*. They fix the `response_template` to `"<|im_start|>assistant\n"`, re-run the print, see `[('Paris', ...), ('.', ...), ('<|im_end|>', ...)]`, and the loss finally descends. Total debug time once they printed the labels: thirty seconds. Time spent before that, blaming the optimizer: two days.

The transferable rule: **a loss that is flat or `nan` from step 1 in an SFT run is very often an everything-masked bug, not an optimization bug.** If [overfit-one-batch](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) cannot drive the loss down, and the labels print as all `-100`, stop touching the LR and fix the masking.

### Sequence packing and the cross-document EOS

There is one more formatting subtlety that bites teams chasing training throughput: **sequence packing**. To avoid wasting compute on padding, many SFT pipelines concatenate multiple short examples into one long sequence up to the context length — pack example A, then example B, then C, into a single 4096-token row. Packing is great for efficiency (no padding waste, higher tokens-per-second) but it introduces two formatting bugs that are easy to create and hard to see.

The first is the **cross-document attention bleed**. If you pack examples A, B, C into one sequence and use a plain causal mask, then the tokens of example B can attend back to the tokens of example A — the model "sees" the previous, unrelated conversation while predicting B. During training this teaches the model spurious dependencies across document boundaries; at inference, where each request is a fresh single sequence, those dependencies are absent, and you have a quiet train-serve mismatch baked into the attention pattern. The fix is a **block-diagonal (document) attention mask** that prevents attention from crossing the boundary between packed examples — modern `trl`/`transformers` packing supports this via position-id resets and `attention_mask` that zeroes cross-document attention. The diagnostic is to check whether your packing path actually resets attention at boundaries; if it does not, packing is silently changing what the model learns.

The second packing bug is directly about our topic: **the EOS at the document boundary**. When you pack A‖B‖C, the end-of-turn token at the end of A is now in the *middle* of the packed sequence, immediately followed by the start of B. If that EOS is in the loss (it should be), the model learns "after a complete answer, emit EOS" — good. But if your packing logic strips the trailing EOS when concatenating (a common "optimization" to save a token), then *none* of the packed examples teach stopping, and you are back to the won't-stop bug, now hidden inside the packing code rather than the collator. The check is the same `check_eos_learnable` from §4, run on a *packed* example: every document boundary inside the packed row should show an EOS token that is in the loss.

```python
# Diagnostic for packed sequences: count EOS tokens that are IN the loss.
def check_packed_eos(input_ids, labels, eos_id):
    import torch
    input_ids = torch.as_tensor(input_ids)
    labels = torch.as_tensor(labels)
    eos_pos = (input_ids == eos_id).nonzero(as_tuple=True)[0]
    in_loss = [int(p) for p in eos_pos if labels[p] != -100]
    print(f"packed seq: {len(eos_pos)} EOS tokens, {len(in_loss)} of them in loss")
    # In a row packed from k documents you expect ~k EOS tokens, all in loss.
    if len(in_loss) < len(eos_pos):
        print("WARNING: some document-boundary EOS tokens are masked out")
    return in_loss
```

The rule of thumb: **packing trades padding waste for two new ways to corrupt the format — cross-document attention and stripped boundary EOS — so if you turn packing on, re-run the string and EOS diagnostics on a *packed* example, not just a single one.** Many "my model regressed when I enabled packing" reports are exactly one of these two. If you do not need the throughput, padding is the simpler, safer default, and you sidestep both bugs entirely.

## 8. The full bisection: from symptom to root cause

Let's assemble the diagnostics into the series' standard move — **bisect to one of the six places before touching code** — for a concrete failing run, so the workflow is muscle memory. The order of the checks is not arbitrary; it is chosen to halve the suspect space at each step. The string diff comes first because it is the cheapest check (no GPU, one example) and it splits the two biggest sub-causes — skew versus EOS — in one shot. The EOS-in-labels check comes second because it splits the EOS branch into its two flavors. The serve-config check comes last because it only matters once you have confirmed the *training* side is correct. Each check is positioned to eliminate the largest remaining chunk of possibilities for the least effort, which is exactly what good bisection means: you are not testing hypotheses in the order they occur to you, you are testing them in the order that maximizes information per unit of work.

**The run:** Llama-3-8B finetuned on 50k instruction pairs. Symptom: in production the model gives a correct answer and then keeps generating an invented follow-up turn, every time. Loss descended cleanly 1.8 → 0.65.

**Step 1 — Which of the six places?** The loss is clean, so this is not an optimization or numerics divergence (no spike, no NaN). The model clearly learned the task (the answers are correct), so the model code and the data content are probably fine. The defect is *behavioral and format-shaped* (won't stop). That points squarely at **data formatting / evaluation** — specifically the chat template and the labels. We have bisected to two suspects in one breath without running anything.

**Step 2 — Diff the strings.** Render one training example and one serving prompt, `repr()` both, diff. They match on the shared prefix — good, no wholesale skew. So it is not symptom-1 (ignore-instructions). The model *does* follow instructions; it just won't stop. That rules skew out and points at EOS.

**Step 3 — Is EOS in the labels?** Run `check_eos_learnable` on a training example. Output: `EOS positions (idx, label): [(263, -100)]`. The `<|eot_id|>` is present but masked. **Found it.** The bug is in the label-construction (place: data/eval), not the model, not the optimizer, not the template's *string* (the EOS is in `input_ids`) — purely the masking boundary.

**Step 4 — Confirm with a test.** Before fixing, write the failing assertion: `assert any(labels[p] != -100 for p in eos_positions)`. It fails on the current pipeline. Good — we have a red test.

**Step 5 — Fix and verify.** Adjust the masking boundary to include the EOS position. The assertion goes green. Re-run one epoch. Measure the honest before→after on a held-out set:

| Instrument | Before (EOS masked) | After (EOS in loss) |
|---|---|---|
| $P(\text{eot\_id})$ at gold turn end | 0.002 | 0.96 |
| Truncation rate (hit max_new_tokens) | 79% | 0.6% |
| Mean generation length | 498 tokens | 64 tokens |
| Instruction win rate (LLM-judged) | 41% | 67% |
| Training loss (final) | 0.65 | 0.66 |

Note the last row: **the loss barely moved** (0.65 → 0.66) because the EOS term is a tiny fraction of the total loss, yet the model's *behavior* transformed. That is the entire moral of this post in one table: the loss is nearly blind to the bug that destroys the product. You cannot debug this class of failure from the loss; you debug it from the labels and the generated strings.

### Stress tests: where else could it hide?

Bisection is not done until you have asked "what if it were something else?" Here are the stress tests for this bug class:

- **What if the strings *did* differ (skew)?** Then symptom-1 (ignores instructions) would dominate, and the win rate would be near or below base. The fix is template alignment (§3), not EOS unmasking. The string diff distinguishes them.
- **What if it only fails in production, not in my eval?** Then your eval uses a different serving path than production. Reproduce the *production* serving format in your eval harness (the unit test of §5). Train-serve skew between two serving paths is the most common "works on my machine" LLM bug.
- **What if the EOS is learned but it still won't stop sometimes?** Check the decoder's `eos_token_id` config at serve time. If you trained the model to emit `<|im_end|>` (id 151645) but the serving stack is configured to stop only on the *base* `</s>` / `<|endoftext|>`, the model emits its end token and the decoder ignores it. This is the **wrong-stop-token** branch of Figure 4 — the model did its job and the serving config threw it away. Set `eos_token_id` (and/or `stop_token_ids` in vLLM) to include the end-of-turn token.
- **What if the loss is flat from step 1, not just won't-stop?** Then you likely masked *everything* (the §7 worked example), not just the EOS. Print the kept tokens; an empty list confirms it.
- **What if it's a base model, not a chat model?** A base (non-instruct) model legitimately won't stop on chat prompts because it was never instruction-tuned. That is not a bug; that is the base model. Make sure you actually finetuned on chat-formatted data with EOS in the loss.

This is the discipline: each symptom has a confirming test that *distinguishes* it from its neighbors, so you never fix the wrong thing. The string diff separates skew from EOS; the EOS check separates masked-EOS from omitted-EOS; the serve-config check separates a training bug from a serving bug.

## 9. Case studies and real signatures

These patterns are well-known enough in the open-source finetuning community that they have become folklore. A few worth knowing, framed accurately.

**The left-padding-breaks-generation signature.** Decoder-only models must be **left-padded** for batched generation, not right-padded. If you right-pad a batch at inference, the real tokens are followed by pad tokens, and the model's "next token" is computed at the position after the *padding*, not after your actual prompt — so generation begins from a meaningless position and produces garbage or refuses to follow the prompt. The fix is `tokenizer.padding_side = "left"` for generation. This is technically a [padding/attention-mask bug](/blog/machine-learning/debugging-training/attention-mask-and-padding-bugs-for-llms) rather than a pure template bug, but it co-occurs constantly with template skew because both live in the "how do I format the batch for serving" code, and both produce the "great training loss, garbage generation" signature. If your model is fine on single examples but breaks on batches, suspect padding side first.

**The `apply_chat_template` adoption inflection.** Hugging Face introduced `chat_template` as a tokenizer attribute and `apply_chat_template` as the canonical API specifically to kill the hand-rolled-format bug class. Before it, every finetuning repo invented its own format string, and train-serve skew was nearly universal — models trained in one repo and served in another almost never agreed on whitespace and special tokens. The standardization is the fix: when the *template travels with the tokenizer*, training and serving load the same template from the same checkpoint, and the skew disappears by construction. The lesson for your own work: store the template on the tokenizer, ship it with the model, and never let a serving stack reconstruct the format from memory.

**The "EOS not in completion" bug in early `trl`.** A recurring report in the `trl` ecosystem was models that would not stop after SFT, traced to the end-of-turn token not being included in the completion that the collator put into the loss. The community fix patterns — explicitly appending `eos_token`, or using assistant-masking that includes the turn's end token — are exactly the §4 fix. If you read `trl` issue threads about "model won't stop after SFT," the resolution is almost always "your EOS isn't in the labels." It is common enough that it is the first thing to check, not the last.

**The system-prompt default-injection skew.** Several popular instruct templates inject a default system prompt when none is supplied. Teams that trained on data with *no* system message, then served through a template that *adds* one, reported subtle instruction-following degradation that vanished when they either (a) added the same default system prompt to training data or (b) disabled the injection at serve time. The signature is "slightly off persona / weaker format adherence" rather than total failure, which makes it the hardest of these to catch — and the reason the render-both-paths habit from §7 matters.

**The "train on the prompt" waste.** A close cousin of the masked-EOS bug, with the opposite masking error: instead of masking too much (eating the EOS), the pipeline masks too little (or nothing) and computes loss over the *entire* sequence, including the user prompt. The model then spends a chunk of its gradient budget learning to *predict the user's question* — a task that is both useless (you never want the model to generate the human's turn) and actively harmful (it dilutes the instruction-following signal and can teach the model to continue prompts rather than answer them). The signature is a model that finetunes "fine" by the loss but is weirdly prone to restating or continuing the question, and a loss value that looks lower than peers (because predicting the often-repetitive prompt tokens is easier than predicting the answer). The fix is completion-only masking — exactly the `DataCollatorForCompletionOnlyLM` setup from §7 — and the confirming test is the `kept = [...]` print: if user tokens appear in the loss, you are training on the prompt. This is the [loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) seen from the formatting side: the masking boundary is part of the template's contract, and getting it wrong in either direction is a formatting failure.

**The "works in transformers, breaks in vLLM/TGI" signature.** A model that follows instructions and stops cleanly when you run `model.generate` in a notebook, but rambles or misbehaves under a production server like vLLM or TGI, is almost always a serving-config skew rather than a training bug. The two usual culprits: the serving stack applies its *own* chat template (which may differ from the one in your checkpoint), or its stop-token configuration does not include your trained end-of-turn token. Both are the wrong-stop / serve-skew branches we have already mapped. The diagnostic is to extract the *exact* prompt the server sends to the model (most servers can log it) and diff it against your `apply_chat_template` output, and to confirm the server's stop-token list includes your `eos_token_id`. The model is innocent; the serving glue is where these live.

The honest caveat on all of these: they are pattern descriptions and community-reported signatures, not precise benchmark numbers. The *mechanism* in each case is solid and reproducible; the magnitude of degradation depends on your model, data, and how far off-distribution the skew pushes you. When I quote a number like "win rate 24% → 61%" in a worked example, that is an illustrative-but-realistic figure for the magnitude of a wholesale-skew fix, not a measurement from a specific public benchmark. Measure your own — the framework here tells you *what* to measure (truncation rate, $P(\text{EOS})$ at turn end, win rate, the string diff) and *why* each one moves; the exact deltas are yours to record.

## 10. When this is (and isn't) your bug

A decisive section, because the worst outcome is fixing a template that was fine while the real bug sits untouched.

**It IS a chat-template / formatting bug when:**

- The loss descended cleanly but the model ignores instructions, rambles forever, or echoes role headers. Clean loss + format-shaped behavioral failure is the signature.
- The model works on single examples in your notebook but breaks in the serving stack, or breaks on batched generation. Train-serve skew or padding-side.
- The string diff between your training format and your serving format shows *any* difference — even one newline.
- `check_eos_learnable` reports the end token absent from `input_ids` or masked in `labels`.
- The model produces a correct answer and then a second invented turn. That is the won't-stop EOS signature almost every time.

**It is NOT a chat-template bug (look elsewhere) when:**

- The **loss spikes or goes NaN.** That is numerics or optimization, not formatting. Go to [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) — a template bug never makes the loss diverge; it keeps the loss suspiciously *healthy*.
- The model is **slow / GPU-underutilized.** That is a systems/throughput problem, unrelated to formatting.
- The answers are **factually wrong but well-formatted and properly stopped.** That is a data-content or capability problem, not a template problem. The template is about *form*, not *content* — if the form is right (it follows the structure, it stops cleanly) but the substance is wrong, you have a data or model-capacity issue.
- `overfit-one-batch` **fails** (you cannot drive the loss to ~0 on one batch). That is a deeper model/optimization/wiring bug. A template bug does not prevent overfitting a batch — it lets you overfit the *wrong format* perfectly.
- The model **stops correctly and follows instructions** but the *quality* is mediocre. That is a finetuning-recipe question (LR, data, epochs — see [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it)), not a formatting bug.

The clean discriminator: **chat-template bugs keep the loss healthy and break the *form* of the output (structure, stopping, instruction-following). They never make the loss diverge and they never make a well-formed-but-wrong answer.** Hold that line and you will route symptoms to the right place.

A second discriminator worth internalizing: chat-template bugs are reproducible and deterministic. The model fails *the same way every time* on the same prompt, because the skew is a fixed property of the format, not a stochastic training artifact. If your failure is intermittent — sometimes stops, sometimes does not, varying run to run — suspect sampling parameters or a [train-infer mismatch](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms) at decode time before you suspect the template. Deterministic, format-shaped failure points at the template; stochastic failure points at the decoder.

## 11. Building it right from day zero

Debugging is recovery; the better play is to make the bug structurally impossible. A few practices that, adopted up front, retire this entire class:

**One template, loaded from the checkpoint, on both sides.** Never hand-roll a format string. Always `tok.apply_chat_template`. Ship the template *with* the tokenizer (it lives in `tokenizer_config.json` as `chat_template`) so serving loads the identical template from the identical checkpoint. The skew bug requires two templates; if there is only ever one, there is nothing to skew.

**A CI test that asserts serve-is-a-prefix-of-train.** The `assert train.startswith(serve)` test from §5, run on every commit, makes wholesale skew un-shippable. Add the EOS-present and BOS-count assertions alongside it.

**Print the labels of one example before every long run.** A thirty-second `kept = [...]` print that shows exactly which tokens are in the loss. The answer plus its end token, nothing else. This single habit catches masked-EOS, train-on-prompt, and everything-masked — three of the most expensive bugs in this post — before you spend a GPU-hour. At, say, a few dollars per GPU-hour for an 8B finetune, a single avoided bad 12-hour run pays for the habit a thousand times over.

**A generation-stops smoke test.** After training, before deploying, generate on five held-out prompts with a generous `max_new_tokens` and assert that every generation ends with the EOS token *before* the cap. If any run hits the cap, the model has not learned to stop, and you catch it in seconds rather than in the first production incident.

```python
import torch

def assert_model_stops(model, tok, prompts, max_new_tokens=256):
    model.eval()
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        out = model.generate(
            ids, max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id, do_sample=False,
        )
        gen = out[0, ids.shape[1]:]
        stopped = (gen[-1].item() == tok.eos_token_id) and (len(gen) < max_new_tokens)
        assert stopped, (
            f"model did NOT stop on prompt {p!r} (len={len(gen)}, "
            f"last_tok={tok.decode([gen[-1].item()])!r}) — EOS not learned?"
        )
    print(f"OK: model stops cleanly on all {len(prompts)} prompts")
```

That smoke test is the runtime counterpart to the static EOS-in-labels check: one verifies the *training data* taught stopping, the other verifies the *model* learned it. Run both. Together they make the won't-stop bug a thing that simply cannot reach production.

### The unifying principle: make the format a contract

Step back and notice that every defense in this section is the same move applied at a different layer: **turn the format from an implicit assumption into an explicit, checkable contract.** The skew bug exists because two code paths *assume* they agree on the format; the fix is to *prove* they agree (the prefix assertion). The won't-stop bug exists because the labels *assume* the EOS is in the loss; the fix is to *assert* it is (the EOS check). The double-BOS bug exists because the tokenization *assumes* one BOS; the fix is to *count* them. The version-drift bug exists because serving *assumes* the upstream template is stable; the fix is to *hash and pin* it. None of these require cleverness — they require treating the format as a first-class, tested artifact rather than a string that gets reconstructed by convention in three different places.

This is why the bug class is simultaneously so common and so cheap to retire. It is common because formatting *feels* like trivial glue code that does not warrant a test — it is just string concatenation, what could go wrong? — and so nobody guards it. It is cheap to retire because the guards are all one-liners: a `startswith` assertion, an EOS-in-labels check, a BOS count, a template hash, a generation-stops smoke test. Five small tests, run in CI, and the entire category of "clean loss, broken model" failures you have been reading about cannot reach production. The cost-benefit is lopsided enough that not adding these tests is, in hindsight, the actual bug.

A final note on cost, since this series cares about the economics of a wasted run. An 8B SFT run is a multi-hour, multi-GPU job; at a few dollars per GPU-hour, a single bad run that has to be discarded because the EOS was masked is a real expense, and the *second* bad run (because the first fix was wrong) doubles it. The pre-run label print and the post-run generation-stops test cost seconds and catch exactly the bugs that turn one clean run into three discarded ones. The discipline pays for itself the first time it saves you a re-run, and it will.

## 12. Key takeaways

- **A finetuned model is a distribution over exactly the format it was trained on.** Serve a different format and you are prompting a model that never existed. The loss cannot see this because the loss only measures fit to the training format.
- **The definitive diagnostic is the string diff.** Render the exact training string and the exact serving string with `repr()`, diff them character by character. Any difference — even one newline — is a bug. This single check subsumes most others.
- **Use `apply_chat_template` identically on both sides.** Never hand-roll the format. Training uses `add_generation_prompt=False` (the full assistant turn is the target); inference uses `add_generation_prompt=True` (end at the assistant header so the model generates).
- **A model that won't stop has not learned its end token.** Either the template omitted the EOS or the labels masked it (`-100`). Check both with the EOS-in-labels diagnostic. The masking boundary must keep the EOS *in* the loss.
- **Clean loss + format-shaped failure = template bug; diverging loss = numerics.** A template bug keeps the loss healthy and breaks the *form* (stopping, instruction-following, structure). It never makes the loss spike or NaN.
- **Only assistant turns get loss in multi-turn chat — including their end tokens.** Mask system and user turns to `-100`; never train the model to produce the human's side. Print the kept tokens to verify.
- **Mind the double-BOS.** If the template already emits the BOS, tokenize with `add_special_tokens=False` or use `tokenize=True`, and assert at most one BOS per sequence.
- **Wrong stop token at serve time can throw away a model that learned to stop.** If training taught `<|im_end|>` but serving stops only on `</s>`, generation never halts. Configure `eos_token_id` / `stop_token_ids` to include the end-of-turn token.
- **A flat-or-NaN loss from step 1 is usually everything-masked, not a bad LR.** Print the labels; an empty kept-list means no tokens are in the loss, so cranking the learning rate does nothing.
- **Make it structural: one template from the checkpoint, a CI prefix-assertion, a label print before every run, and a generation-stops smoke test.** Four cheap habits retire the entire bug class.

## Further reading

- Hugging Face `transformers` documentation — **"Chat Templates"** and the `apply_chat_template` / `chat_template` API reference. The canonical source for template behavior, `add_generation_prompt`, and the Jinja format.
- Hugging Face `trl` documentation — **`SFTTrainer`**, `DataCollatorForCompletionOnlyLM`, and assistant-token masking. How to mask prompts and keep the completion (including EOS) in the loss.
- Hugging Face `tokenizers` / `transformers` tokenizer docs — special tokens, `add_special_tokens`, `eos_token` / `bos_token`, and `padding_side` for generation.
- PyTorch documentation — `CrossEntropyLoss` and `ignore_index` (the `-100` masking convention that underlies label masking).
- **Within this series:**
  - [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom → suspect → test → fix decision tree this post instantiates.
  - [The Training Debugging Playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone bisection method and printable checklist.
  - [Tokenization Bugs](/blog/machine-learning/debugging-training/tokenization-bugs) — BOS/EOS/PAD handling, the double-BOS, and vocab/tokenizer mismatches that this post builds on.
  - [The Loss-Masking Bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) — `-100` masking, prompt-vs-completion loss, and the off-by-one that eats the EOS.
  - [Finetuning an LLM Without Breaking It](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) — the LR / epochs / forgetting recipe questions that live next door to formatting.
  - [Train-Infer Mismatch for LLMs](/blog/machine-learning/debugging-training/train-infer-mismatch-for-llms) — KV-cache, sampling, and padding drift at decode time, the stochastic cousin of template skew.
