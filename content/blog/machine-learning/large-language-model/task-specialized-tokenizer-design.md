---
title: "Customizing tokenizers per task: a design playbook for agents, math, new languages, and code"
date: "2026-06-09"
publishDate: "2026-06-09"
description: "A per-workload playbook for tokenizer design — what to change in the vocabulary, the special tokens, the digit policy, and the whitespace rules when you build for agents, math, a new language, or code."
tags: ["tokenizer", "llm", "bpe", "agents", "tool-calling", "math", "multilingual", "fertility", "vocab-extension", "code", "fill-in-middle", "nlp"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 51
---

The most expensive bug I have ever shipped was three space characters. A code-completion model we had fine-tuned kept producing subtly wrong indentation inside nested Python blocks, and for two weeks the whole team chased the decoder, the sampling temperature, the training data — everything except the one component nobody thinks about. The model had been trained on a tokenizer whose vocabulary had no token for a run of spaces. Every level of indentation was four separate space tokens, every deeply nested function blew past the context window twice as fast as it should have, and the model never got a clean signal about where one indent level ended and the next began. The fix was not in the model. It was a dozen new entries in the vocabulary.

That is the recurring shape of tokenizer problems: they arrive disguised as model problems, sampling problems, data problems, or billing problems, and they are almost never debugged where they actually live. The reason is that most engineers inherit a tokenizer the way they inherit the system clock — it is just there, it does its job, and you don't think about it until something is visibly broken. But the tokenizer is the one layer of the entire stack the model cannot learn its way around. Whatever boundaries it draws, whatever it merges or splits, whatever it normalizes away — the model is stuck with that forever. Pretraining can teach a model to recover from a mediocre tokenizer, but it can never teach it to see a distinction the tokenizer erased.

![The tokenizer pipeline as four tunable dials](/imgs/blogs/task-specialized-tokenizer-design-1.webp)

The diagram above is the mental model for this entire post: tokenization is a short pipeline — decode bytes, normalize, pre-tokenize with a regex, merge into subwords, inject special tokens, look up embeddings — and it exposes exactly four dials. The first dial is **segmentation granularity**: how aggressively the merge algorithm fuses bytes into long pieces. The second is the **special-token namespace**: the reserved, unmergeable symbols the model uses to mark structure. The third is **normalization**: what gets silently rewritten before the model ever sees it. The fourth is **digit and whitespace policy**: the special-cased handling of the two character classes that break every naive tokenizer. There is no universal setting for these four dials. The right positions depend entirely on what the model will spend its time doing. An agent, a calculator, a Vietnamese chatbot, and a code assistant want four genuinely different tokenizers, and this post is the playbook for each.

If you want the general design space — algorithm families, vocab-size economics, the serving stack — that lives in the sibling post on [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm), and the raw algorithm internals are in [BPE Tokenizer](/blog/machine-learning/large-language-model/bpe-tokenizer). Here we assume you know what BPE is and go straight to the question those posts deliberately leave open: *given a task, which dials do I turn, and how far?*

> The tokenizer is the only non-learnable layer in an LLM. Every parameter above it has to live with whatever it decides. Choose it for the job, not for the demo.

## Why one tokenizer can't serve every task

The default move is to grab the tokenizer that ships with whatever base model you are building on — `cl100k_base` if you came up through the OpenAI ecosystem, the Llama-3 tokenizer if you are in the open-weights world — and never look at it again. For an English chat assistant that is genuinely fine. The trouble starts the moment your workload stops looking like English web prose, because every one of those popular tokenizers was *trained* on English-heavy web text and *tuned* for it. The vocabulary spends its 100,000-odd slots on the merges that compress that corpus best, and a slot spent on `" approximately"` is a slot not spent on a Thai syllable cluster, a LaTeX operator, or a 16-space indent.

Here is the assumption-versus-reality table I wish someone had shown me before my first multi-task deployment:

| Task | What a generic English tokenizer does | Why it hurts |
|---|---|---|
| **Agent / tool use** | Encodes `{`, `"`, `:`, role markers as ordinary text tokens | No reliable boundary between thought, tool call, and answer; parsing is regex-on-text, which breaks the moment the model emits a stray brace |
| **Math / arithmetic** | Groups digits into chunks of up to three, left to right | The same number gets different token boundaries depending on its length, so place value is not aligned to position and carries cross token edges |
| **New / low-resource language** | Falls back to bytes for unseen scripts and diacritics | Fertility of 3–5 tokens per word: 3–5× the cost, 3–5× the latency, and a shorter effective context for the same text |
| **Code** | Treats each space as its own token, splits identifiers arbitrarily | Indentation explodes token counts; no fill-in-middle support; `camelCase` and `snake_case` fragment inconsistently |
| **Structured / scientific** | Applies subword merges to SMILES strings, DNA, JSON keys | Merges invent spurious "words" out of chemically or syntactically meaningless byte runs |

Every row in that table is the same underlying failure: a vocabulary optimized for one distribution, applied to another. The clearest way to feel it is to take a single short string from each domain and watch where the boundaries fall.

![The same content, four very different tokenizations](/imgs/blogs/task-specialized-tokenizer-design-2.webp)

Look at what happens across the four tapes. The math expression `3.14+27182` shatters into digit chunks of two and three, so the model never sees a stable "this position is the ones place" signal. The tool call `{"city":"Hanoi"}` spends most of its tokens on JSON punctuation, and the city name itself gets split mid-word into `"Han` and `oi"`. The Vietnamese `Hà Nội` — two words, six characters — costs five tokens because the diacritic-bearing characters fall outside the English-tuned merges and decompose toward bytes. And the Python indentation becomes four separate, identical space tokens before the model reaches a single character of code. Same tokenizer, four pathologies. None of them are fixable downstream. All of them are fixable by turning one or two of the four dials.

## The four dials you actually turn

Before we go task by task, it is worth being precise about the four dials, because the rest of the post is just "for task X, set these dials to these positions." Each dial maps to a distinct stage of the pipeline in the first figure, and each one has a sensible default that you override only when the workload demands it.

| Dial | What it controls | Generic default | You change it when… |
|---|---|---|---|
| **Granularity** | How long the merged pieces get; vocab size; the regex that pre-splits text | byte-level BPE, ~100–128k vocab, English-tuned merges | The dominant content is a script or syntax the base merges don't cover well |
| **Special tokens** | The reserved, unmergeable symbols that mark structure | a handful: BOS, EOS, PAD, a few chat markers | You need machine-parseable structure — channels, tool calls, FIM, document boundaries |
| **Normalization** | What is rewritten before tokenization (casing, Unicode form, whitespace collapse) | light NFC, no case folding | The rewrite destroys task-relevant distinctions (math symbols, code whitespace) |
| **Digit / whitespace policy** | Special handling of digits and whitespace runs | digit grouping, single-space tokens | The model must do arithmetic, or read whitespace-significant code |

The first dial, **granularity**, is the one most people mean when they say "train a tokenizer." It is governed by the pre-tokenization regex (which decides what *can* be a token boundary) and the merge algorithm (which decides what *becomes* a single token within those boundaries). The GPT-2-style regex, for instance, splits on a fixed pattern of letters, numbers, and punctuation so that merges never cross those class boundaries. Change the regex and you change the entire shape of the vocabulary.

The second dial, **special tokens**, is the one this post will argue is the most under-used. A special token is a vocabulary entry that the pre-tokenizer is forbidden to split and the merge algorithm is forbidden to fuse with anything else. It always encodes to exactly one ID and always decodes back to exactly its string. That round-trip guarantee is what makes it the right tool for marking structure — and it is precisely what agents need.

The third dial, **normalization**, is the quiet one. Normalization runs before anything else and rewrites the input: Unicode normalization forms (NFC, NFKC) fold visually-identical characters together, case folding collapses `A` and `a`, whitespace normalization collapses runs of spaces. Each of these is a lossy transform, and "lossy" is exactly what you do *not* want when the lost information was the point — a half-fraction glyph in a math problem, or the exact indentation of a Python file.

The fourth dial, **digit and whitespace policy**, is technically a special case of granularity, but it deserves its own dial because digits and whitespace are the two character classes where the generic default is actively wrong for entire categories of task, and because the fix is a small, well-understood regex change rather than a full retrain.

One clarification before we proceed, because it trips people up: these four dials are largely *orthogonal* to the choice of merge algorithm. Whether you use BPE, WordPiece, or Unigram-LM, you still decide a granularity, a special-token namespace, a normalization policy, and a digit/whitespace policy — the algorithm determines *how* pieces are discovered from a corpus, while the dials determine *what the tokenizer is allowed to do* with the result. A Unigram tokenizer and a BPE tokenizer trained on the same corpus with the same digit policy will both split digits the same way, because that policy lives in the pre-tokenization regex, upstream of the merge algorithm entirely. So when this post says "turn the digit dial," it is a change you can make regardless of which algorithm family you inherited from your base model. The [sibling design post](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) covers the algorithm trade-offs in depth; here they are deliberately held constant so the task-specific moves stand out on their own.

With the dials named, we can go to work. We will take the four workloads in turn — agents, math, new languages, and code — and for each one, name the dials, show the code, and quantify the win.

## 1. Agents: the tokenizer is your control plane

> If your agent's protocol can be broken by the model typing a `}` in the wrong place, you built the protocol out of text. Build it out of tokens instead.

The defining property of an agent is that its output is not prose for a human — it is a *program* for a runtime. The model emits something, a harness parses it, and the parse result decides whether to call a tool, return an answer, or keep thinking. The entire reliability of the agent rests on that parse being unambiguous. And the single biggest mistake in agent design is making the parse depend on *text* — on the model dutifully producing well-formed JSON, or remembering to close a markdown code fence, or never emitting the literal string `Final Answer:` inside its reasoning.

The dial you turn for agents is the **special-token namespace**. Instead of marking structure with text that the model might fumble, you reserve dedicated, unmergeable tokens for every structural boundary the runtime cares about: the start and end of a message, the channel a message belongs to, the start of a tool call, the start of a tool result. Because these are real vocabulary entries, the model emits them as atomic decisions — a single sampling step picks the "I am now calling a tool" token — and the harness parses them by ID, not by string matching. There is no `}` that can break it because the boundary is not a `}`.

![An agent turn is a token stream wrapped by reserved tokens](/imgs/blogs/task-specialized-tokenizer-design-3.webp)

The figure shows the structure that OpenAI's Harmony format (shipped with the gpt-oss models) and, before it, ChatML made standard. Every message is a flat run of tokens wrapped by reserved control tokens. `<|start|>` and `<|end|>` bracket each message. A `<|channel|>` token routes the assistant's output into one of three streams: `analysis` for hidden chain-of-thought, `commentary` for tool calls, and `final` for the user-visible answer. A tool call ends with `<|call|>` instead of `<|end|>`; the final answer ends with `<|return|>`. The model never has to *describe* what it is doing in prose — it *is* in a channel, marked by a token, and the harness knows exactly how to route every span.

The practical consequences are large. Hidden reasoning can be stripped from the transcript by deleting everything between `<|channel|>analysis` and the next `<|end|>`, with zero risk of accidentally deleting part of the answer, because the boundaries are tokens and not a fragile `<thinking>...</thinking>` regex. A tool call can be detected the instant the `<|call|>` token is sampled, enabling the harness to stop generation and dispatch immediately. And because each control token is a single ID, the structural overhead is a handful of tokens per turn rather than the dozens that text-based markup costs.

Here is how you actually add a control namespace to an existing tokenizer, and the footgun that gets everyone the first time:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

## Reserve control tokens for a custom agent protocol. These must be additive
## special tokens: the pre-tokenizer will never split them, and BPE will never
## merge them with a neighbor.
control = ["<|tool_call|>", "<|tool_result|>", "<|thought|>", "<|action|>"]
n_added = tok.add_special_tokens({"additional_special_tokens": control})
print(n_added, "added; vocab is now", len(tok))

## Each control token encodes to exactly one id and round-trips exactly.
ids = tok('<|tool_call|>{"city":"Hanoi"}', add_special_tokens=False)["input_ids"]
print(ids[0], tok.decode([ids[0]]))   # -> the single reserved id, "<|tool_call|>"

## Round-trip determinism is the property your harness depends on. Test it.
s = '<|thought|>need weather<|tool_call|>get_weather<|tool_result|>22C<|action|>answer'
assert tok.decode(tok(s, add_special_tokens=False)["input_ids"]) == s
```

The footgun is the line that is *not* there yet. The moment you call `add_special_tokens`, the tokenizer's vocabulary is larger than the model's embedding matrix, and the very next forward pass will index out of bounds — or, worse, silently read garbage — unless you resize the model:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model.resize_token_embeddings(len(tok))   # MUST happen before any forward pass

## The new rows are randomly initialized. They have anomalously small, untrained
## norms until you fine-tune on data that actually uses the new tokens.
```

That last comment is the second-order trap, and it is important enough to be the section's gotcha.

### Second-order optimization: untrained reserved tokens are a live wire

When you add four control tokens and resize the embedding matrix, you create four embedding rows initialized to noise. If you then deploy without fine-tuning on data that uses them, those rows stay near their initialization — they have a much smaller norm than trained tokens because the optimizer never pushed them anywhere. A token with a tiny, untrained embedding behaves erratically: prompting the model to emit it, or feeding it in, produces undefined, often bizarre output. This is exactly the mechanism behind the famous glitch tokens, and we will see it again in the case studies. The discipline is simple but non-negotiable: every special token you reserve must appear in fine-tuning data, ideally hundreds of times, before it ships. If you are reserving a *budget* of tokens for future use — a common and sensible move, since you cannot add tokens later without another resize — leave them out of the model's reachable output space, or accept that they are landmines until trained.

There is one more agent-specific lever worth naming: **serialization format choice beats vocabulary tuning** for structured output. A model emitting a tool call in verbose JSON spends tokens on `{`, `"`, `}`, `:`, and every key name on every call. The same call in a terser format — positional arguments, or a compact key scheme — can be half the tokens. You do not always need a new tokenizer to make structured output cheap; sometimes you need a cheaper structure. Reserve special tokens for the *boundaries* that must be unambiguous, and choose a *payload format* that is cheap under whatever tokenizer you already have. The two moves compose.

### The token-budget math of structured output

It is worth putting numbers on the payload-format argument, because the savings are larger than most people expect. Take a single tool call with three arguments, expressed four ways — verbose JSON, minified JSON, a positional format the harness expands back into JSON, and a reserved-token-framed compact form. Under `cl100k_base`, the token counts come out roughly like this:

| Format | Example | Approx. tokens |
|---|---|---|
| Verbose JSON | `{ "name": "search", "args": { "query": "weather", "limit": 5 } }` | ~28 |
| Minified JSON | `{"name":"search","args":{"query":"weather","limit":5}}` | ~22 |
| Positional / compact | `search⟶weather⟶5` (unit-separator delimited) | ~8 |
| Reserved-token framed | `<\|tool_call\|>search weather 5` | ~6 |

Across a long agent trajectory with dozens of tool calls, the difference between 28 and 8 tokens per call is the difference between a transcript that fits the context window and one that overflows it — and it is pure overhead, paid on every turn, contributing nothing to the model's reasoning. The structural boundaries still want reserved special tokens so the harness can find each call unambiguously, but the *arguments* should ride in the cheapest serialization your schema allows. This is the clearest case where the right tokenizer move and the right protocol move are different moves that multiply rather than substitute.

One subtlety bites teams pairing custom tokens with constrained decoding. When you force the model's output to match a grammar — a JSON schema, a regex, a tool-call format — the grammar operates on *characters* but the model emits *tokens*, and a single token can straddle a grammar boundary. A token like `":` covers a quote and a colon and possibly the next character, so a rule like "the next character must be a digit" has to be translated into "which token IDs are consistent with a digit appearing next." That translation is exactly where token-aware constrained-decoding libraries earn their keep, and reserved single-token boundaries make it dramatically easier, because a boundary that is its own token cannot straddle anything.

## 2. Math and reasoning: every digit in its place

Arithmetic is where tokenizer design stops being an efficiency question and becomes a correctness question. A model can be coaxed past a high-fertility tokenizer with more training; it cannot reliably add two numbers if the tokenizer scrambles their place values. The dial here is **digit policy**, a sub-dial of granularity, and it is one of the cleanest examples of a small tokenizer change producing a large capability change.

The problem starts with the standard pre-tokenization regex used by the GPT-3.5/GPT-4 family (`cl100k_base`), which groups digits into runs of up to three, scanning left to right. So `1234567` pre-tokenizes into `123`, `456`, `7`. The last group holds the low-order digits, and the *length* of that last group depends on the total number of digits modulo three. The consequence is brutal for arithmetic: the token boundaries of a number depend on how long the number is, so the same place value lands at different token positions in different numbers.

![Digit grouping decides whether the model can add](/imgs/blogs/task-specialized-tokenizer-design-4.webp)

The figure makes the misalignment concrete. Under three-digit left grouping, `12345` becomes the two tokens `[123][45]`. To add it to another number, the model has to learn that the *ones* digit lives at the end of the second token — but for a number like `123456`, the ones digit lives at the end of a *three*-digit token. The "where is the ones place" question has no stable answer in token-space. Carries, which propagate from low place to high place, routinely cross a token boundary, so the model must internally re-derive place value from variable-length chunks on every single addition. Single-digit tokenization removes the problem at the root: `12345` becomes `[1][2][3][4][5]`, one digit per token, so place value is exactly token position counted from the right. Carries stay inside a single decoding step. This is precisely the choice the Llama family made — Llama tokenizers split every digit individually — and it is a meaningful part of why those models do arithmetic better than their fertility-optimized cousins.

You can watch the difference directly:

```python
import tiktoken
from transformers import AutoTokenizer

cl = tiktoken.get_encoding("cl100k_base")           # GPT-3.5/4: groups up to 3
llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # single digit

n = "12345"
print([cl.decode([t]) for t in cl.encode(n)])
## -> ['123', '45']        two tokens, boundary depends on length

print([llama.decode([t]) for t in llama(n, add_special_tokens=False)["input_ids"]])
## -> ['1', '2', '3', '4', '5']   one digit per token, place value = position

## The misalignment is visible the moment lengths differ:
for x in ["7", "70", "700", "7000"]:
    print(x, "->", [cl.decode([t]) for t in cl.encode(x)])
## 7    -> ['7']
## 70   -> ['70']
## 700  -> ['700']
## 7000 -> ['700', '0']   <- the ones digit is now alone in the second token
```

That last line is the whole problem in one example. The digit `7` is in the thousands place of `7000`, but the tokenizer puts it in a token with the hundreds and tens, while the actual ones digit (`0`) is stranded in its own token. The model cannot rely on any fixed relationship between token position and place value.

There are three ways to set the digit dial correctly, in increasing order of intrusiveness:

| Strategy | What it does | Cost |
|---|---|---|
| **Single-digit split** | Pre-tokenize so every digit is its own token | +1 token per digit (higher fertility on numbers) |
| **Right-to-left grouping** | Group digits in threes *from the right*, so the ones digit always ends a token | Keeps fertility low, aligns low-order digits |
| **Explicit separators** | Insert a delimiter (comma, space) every three digits before tokenizing | Cheap, prompt-side, works with any tokenizer |

Single-digit is the cleanest and is what you want if you are training from scratch and care about math. Right-to-left grouping is the compromise: by grouping from the right, the rightmost token always contains the ones, tens, and hundreds, so the most carry-active digits stay aligned across operands, while you keep the lower fertility of three-digit chunks. The third option is the one you can deploy today without touching the tokenizer at all — inserting thousands separators so the model sees `7,000` instead of `7000` nudges the boundaries toward place-value alignment, and it is a genuinely effective prompt-engineering trick for arithmetic-heavy workloads on a model you cannot retrain.

To see why this matters mechanically, walk a single addition through both tokenizations. Take `4528 + 947`. Under three-digit left grouping the operands tokenize as `[452][8]` and `[947]` — the first number's ones digit sits alone in a second token while the second number's ones digit sits at the *end* of a three-digit token. The model has to align the `8` of `[452][8]` with the `7` of `[947]`, even though those digits live at completely different offsets within their respective tokens, then propagate the carry from `8 + 7 = 15` leftward across a token boundary into `[452]`. Every one of those alignment-and-carry steps is something the model must reconstruct from context rather than read off a stable structure. Under single-digit tokenization the same sum is `[4][5][2][8] + [9][4][7]`, and "align the last token of each operand, carry into the second-to-last" is a fixed positional rule the model can learn once and apply everywhere. The arithmetic did not get easier; the *representation* did.

Digits are not the only math-specific concern. LaTeX and mathematical notation expose the **normalization** dial in a way that bites silently. Consider what NFKC normalization — the default in many SentencePiece setups — does to mathematical text. It maps the single glyph `½` to the three characters `1`, `⁄`, `2`; it folds full-width digits to ASCII; it rewrites various Unicode minus and multiplication signs. For ordinary prose, harmonizing these is helpful. For a math model, it means the input the model trains on is not the input the user typed, and an evaluation that depends on exact symbol handling can silently degrade. A short catalogue of what aggressive normalization does to math-relevant characters makes the risk concrete:

| Input glyph | After NFKC | Why it matters |
|---|---|---|
| `½` (U+00BD) | `1⁄2` (three chars) | A fraction typed as one symbol becomes three tokens with a fraction-slash |
| `x²` superscript | `x2` | Exponents collapse into ordinary digits, so `x²` and `x2` become indistinguishable |
| `１２３` full-width | `123` | Locale-specific digits are silently re-spaced, changing token boundaries |
| `−` (U+2212 minus) | `-` (hyphen) | Unicode minus folds to ASCII hyphen — usually fine, but not reversible on output |
| `ﬁ` ligature | `fi` | Harmless in prose, but any exact-match evaluation now mismatches the source |

The discipline for math models is to *lock normalization down*: use the lightest Unicode normalization that keeps your text valid (typically NFC, not NFKC), audit exactly what the normalizer does to a sample of real inputs, and never let casing or symbol folding touch the math.

Operators and notation are the other half of the math dial. A model doing symbolic math sees a steady diet of `\frac`, `\sum`, `\int`, `^`, `_`, `\left(`, and the rest of the LaTeX vocabulary, and a general tokenizer fragments these in unhelpful ways — `\frac` might land as `\`, `fra`, `c`, spreading its structural meaning across tokens that also appear in unrelated contexts. A math-tuned tokenizer earns dedicated tokens for the high-frequency LaTeX commands, so `\frac` is a single token whose embedding can specialize to "a fraction starts here." The same applies to the structural characters `^` and `_` that mark superscript and subscript: when they are stable single tokens, the model can reliably attach the following group to the right structural role. None of this is exotic — it is the granularity dial applied to a symbol vocabulary instead of a word vocabulary — but it is routinely missed, because the people training the tokenizer rarely sample their corpus for how the math actually tokenizes.

### Second-order optimization: fertility is not the metric, alignment is

The instinct after reading the above is to optimize digit fertility — to make numbers cost as few tokens as possible. That instinct is wrong for math. Single-digit tokenization *increases* fertility on numbers (five tokens for `12345` instead of two) and *improves* arithmetic, because the thing that matters is not how few tokens a number costs but whether the model can map token position to place value. You are deliberately spending more tokens to buy a cleaner representation. This is the general lesson of task-specialized tokenization in miniature: the right objective is not always "compress harder." For an agent it is "make boundaries unambiguous." For math it is "align structure to position." Compression is one goal among several, and for some tasks it is the wrong one to maximize. The blog post on [training LLMs for math](/blog/machine-learning/large-language-model/training-llm-for-math) goes deeper on the downstream training side of this; here the point is only that the tokenizer sets the ceiling.

## 3. A new or low-resource language

When you take a model trained mostly on English and point it at Vietnamese, Thai, Hindi, or any script the base vocabulary under-serves, the first thing that breaks is not accuracy — it is economics. The dial is **granularity**, and the symptom is *fertility*: the average number of tokens the tokenizer spends per word. A tokenizer with a fertility of 1.1 on English might have a fertility of 3 or more on Vietnamese, which means every Vietnamese request costs three times the tokens, runs three times slower per word, and fits a third as much text in the same context window. The model is not "bad at Vietnamese" in some deep sense; it is being charged a 3× tax at the door.

![Fertility: the token tax a tokenizer charges per language](/imgs/blogs/task-specialized-tokenizer-design-5.webp)

The heatmap shows the shape of the tax (the numbers are illustrative but the gradient is real). English sits near 1.0 token per word for every tokenizer — that is what these vocabularies were built for. Vietnamese, Thai, and Hindi climb to 3, 4, even 5 tokens per word under the older, smaller English-centric tokenizers like GPT-2's 50k and Llama-2's 32k. The two levers that bring the tax down are both visible in the table: a *bigger* vocabulary has room for more non-English merges (cl100k's 100k helps), and a *deliberately multilingual* training corpus spends those slots on the right languages (Llama-3's 128k, trained on far more non-English text, roughly halves the fertility of its predecessor). The difference between the worst cell (Hindi at 5.5) and the best (Hindi at 2.0) is not a different model — it is a different tokenizer in front of the same kind of model.

It is worth being precise about *why* the tax exists, because it is not that these tokenizers cannot represent the text. Modern byte-level tokenizers can represent anything, since every byte is in the vocabulary as a fallback. The tax is that the *merges* that would compress the target language are missing. When the merge table has no entry for a common Vietnamese syllable, the tokenizer falls back to the constituent bytes — and a single accented character in UTF-8 is two or three bytes, each its own token. Nothing is ever truly out-of-vocabulary in a byte-level tokenizer, which is a genuine robustness win, but "representable" and "efficiently represented" are very different properties, and fertility is exactly the measure of the gap between them.

Measuring fertility is the first thing you should do before any multilingual project, and it is a few lines:

```python
from transformers import AutoTokenizer

def fertility(tok, text):
    """Average tokens per whitespace-delimited word."""
    words = text.split()
    n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    return n_tokens / max(1, len(words))

samples = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank.",
    "vi": "Con cao nau nhanh nhen nhay qua con cho luoi ben bo song.",
}
for name in ["gpt2", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.1-8B"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(name)
    for lang, text in samples.items():
        print(f"  {lang}: fertility={fertility(tok, text):.2f}")
```

The reason to measure before building is that fertility translates directly into money and latency. If your Vietnamese traffic runs at fertility 3.0 where it could run at 1.6 after extension, you are paying roughly 1.9× the tokens on every request — and since both API billing and autoregressive decoding latency are linear in output tokens, that is 1.9× the cost *and* 1.9× the per-word generation time, on every Vietnamese request, indefinitely. At even modest traffic the extension project pays for itself in weeks. At zero Vietnamese traffic it never pays for itself at all. The number decides the project.

Once you have measured the tax and decided it is too high, you have two routes: train a new tokenizer from scratch on a corpus that includes your target language, or *extend* the existing one. The decision is usually not close:

| Decision | Retrain from scratch | Extend the base |
|---|---|---|
| Embeddings | All re-initialized; full pretrain required | Base rows preserved; only new rows trained |
| Compute | A full pretraining budget | A few billion tokens of continued pretraining |
| English ability | Must be re-acquired from scratch | Retained — base merges are unchanged |
| Risk | High; you are training a new model | Low; surgical, reversible, testable |
| When it wins | You are pretraining a new model anyway | You are adapting a strong existing model |

For almost every practical project — where you are adapting a strong released base model rather than pretraining a frontier one — extension is the right call, because retraining the tokenizer from scratch would invalidate every learned embedding and force a full pretraining run. Extension keeps the base model's knowledge intact and grafts the new language on. The mechanism is worth understanding precisely.

![Vocab extension grafts new pieces onto a frozen base](/imgs/blogs/task-specialized-tokenizer-design-6.webp)

The figure traces the dataflow. You start from two inputs: the base tokenizer (say, 128k pieces) and a corpus in the target language. You train an auxiliary BPE on the target corpus to discover the merges the base lacked — say, 16k new pieces. You keep *all* of the base merges (so existing text tokenizes identically and the base embeddings stay valid) and add only the new, non-duplicate pieces, producing a merged vocabulary of around 144k. Then comes the part that actually determines whether the extension works: you resize the model's input embedding and output projection to the new vocabulary size, which creates 16k new, randomly-initialized rows — and you *mean-initialize* each new row from the base-tokenizer segmentation of that piece's surface string. A new token for the Vietnamese piece `nh` is initialized as the average of the embeddings the base tokenizer would have used for `n` and `h`. This warm start is the difference between continued pretraining that converges in a few billion tokens and one that has to relearn those rows from noise.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import ByteLevelBPETokenizer

base = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

## 1. Train an auxiliary BPE on the target-language corpus.
aux = ByteLevelBPETokenizer()
aux.train(files=["vi_corpus.txt"], vocab_size=16000, min_frequency=3)

## 2. Add only the pieces the base tokenizer does not already have.
base_vocab = set(base.get_vocab())
new_pieces = [p for p in aux.get_vocab() if p not in base_vocab]
base.add_tokens(new_pieces)
print(f"added {len(new_pieces)} pieces; vocab {len(base)}")

## 3. Resize, then mean-init each new row from the base segmentation of its string.
old_emb = model.get_input_embeddings().weight.data.clone()
model.resize_token_embeddings(len(base))
emb = model.get_input_embeddings().weight.data

base_only = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # original
for piece in new_pieces:
    surface = piece.replace("Ġ", " ")          # byte-level space marker
    sub_ids = base_only(surface, add_special_tokens=False)["input_ids"]
    if sub_ids:
        new_id = base.convert_tokens_to_ids(piece)
        emb[new_id] = old_emb[sub_ids].mean(dim=0)   # warm start

## 4. Continue pretraining on the target corpus so the new rows specialize.
model.tie_weights()  # if the model ties input/output embeddings
```

After this, you run continued pretraining on the target-language corpus — typically a few billion tokens — to let the new rows move from "average of their sub-pieces" to genuinely useful representations. The full training-side treatment of this, including data mixtures and catastrophic-forgetting mitigation, is in [training LLMs to adapt to a new language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language). The tokenizer's job is only to set up the vocabulary and the warm start correctly; do that, and you can roughly halve the fertility tax for the target language while keeping the base model's English ability nearly intact.

The quality of the auxiliary BPE depends entirely on the quality of the target corpus, and two corpus issues sink more extension projects than any modeling mistake. The first is *register coverage*: if your corpus over-represents one style — say, formal written Vietnamese but not the diacritic-heavy informal style users actually type — the new merges will compress the former and leave the latter at byte-level fertility, and production traffic will not see the speedup the offline numbers promised. The second is *deduplication*: web-scraped corpora are full of near-duplicate documents, and an aux BPE trained on undeduplicated text spends its merge budget memorizing boilerplate — cookie banners, navigation chrome, licence headers — instead of language. Deduplicate aggressively before training the aux tokenizer, and sample the corpus to confirm it matches the distribution of text the model will serve. A tokenizer can only compress what it was shown.

### Second-order optimization: the embedding tax and the merge-boundary trap

Two non-obvious consequences deserve attention. The first is the **embedding tax**: every token you add is a new row in both the input embedding and the output projection, each of dimension `d_model`. For a model with `d_model = 4096`, adding 16k tokens adds 16,000 × 4096 × 2 ≈ 130M parameters just in the embedding tables, in fp16 that is ~260 MB of extra weights and a proportionally larger softmax to compute at every step. A bigger vocabulary lowers fertility but raises the per-step cost of the embedding and the final softmax; there is a real trade-off, and 16k is a reasonable extension precisely because it is large enough to matter and small enough not to dominate. Gemma's choice of a 256k vocabulary is the opposite end of that trade-off — superb multilingual fertility, but a heavy embedding tax that only makes sense at scale.

The second is the **merge-boundary trap**. When you add new pieces, you must be careful that they do not change how *existing* text tokenizes, or you silently break the base model's learned representations. Because you keep all base merges and add new pieces with lower priority, common English text still tokenizes exactly as before — but if a new piece happens to be a substring that the base would have tokenized differently in some context, you can get inconsistent segmentation. The safe discipline is to add new pieces only for sequences the base tokenizer was *fragmenting badly* (high-fertility target-language spans), verify on a held-out English set that fertility and segmentation are unchanged, and never add a piece that overlaps a frequent English merge.

## 4. Code and structured data

Code is where the **whitespace policy** and **special-token** dials both matter at once, and where the gap between a generic tokenizer and a tuned one is the most visceral. The opening story of this post was a whitespace bug; here is why it happens and how the right vocabulary fixes it.

![A code tokenizer changes the vocabulary, not just the corpus](/imgs/blogs/task-specialized-tokenizer-design-7.webp)

The left column is what a generic web tokenizer does to Python. Indentation is whitespace-significant in Python, and a four-space indent at four levels of nesting is sixteen leading spaces. A tokenizer with no whitespace-run tokens encodes that as sixteen separate space tokens before the model reaches a single keyword. Deeply nested code blows past the context window at twice the necessary rate, and the model gets a noisy, repetitive signal where it should get a clean "four levels deep" signal. The right column is the code-tuned tokenizer: it has explicit tokens for runs of whitespace, so a 16-space indent is one or two tokens; it treats indentation levels as merge units; it carries fill-in-middle sentinels so the model can be trained to infill, not just continue; and it has identifier-aware boundaries so `camelCase` and `snake_case` fragment consistently.

```python
import tiktoken

gpt2 = tiktoken.get_encoding("gpt2")          # no whitespace-run tokens
cl   = tiktoken.get_encoding("cl100k_base")   # added whitespace-run tokens

indent = "    " * 4                            # 16 spaces, 4 indent levels
print("gpt2 :", len(gpt2.encode(indent)), "tokens")   # ~16
print("cl100k:", len(cl.encode(indent)), "tokens")    # ~1-2

## On a realistic nested function the difference compounds:
code = "def outer():\n    def inner():\n        if x:\n            return 1\n"
print("gpt2 :", len(gpt2.encode(code)))
print("cl100k:", len(cl.encode(code)))
```

The whitespace win is the headline, but the **fill-in-middle (FIM)** sentinels are the part that is genuinely specific to the code task and impossible to add later without retraining. A plain language model is trained left-to-right: given a prefix, predict the next token. But code completion in an editor is almost never "continue from the end" — it is "fill in the gap between what is above the cursor and what is below it." FIM is a training transformation that teaches a left-to-right model to do exactly that, and it works entirely through special tokens. You split a document into prefix, middle, and suffix, then rearrange them into a sequence the model can be trained on with ordinary next-token prediction:

```python
## StarCoder / InCoder style FIM sentinels — these must be real special tokens.
PRE, SUF, MID = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"

def to_fim(prefix, suffix, middle):
    # "PSM" ordering: the model sees prefix and suffix, then predicts the middle.
    return f"{PRE}{prefix}{SUF}{suffix}{MID}{middle}"

## At training time, `middle` is the target; at inference, you feed everything up
## to and including <fim_middle> and let the model generate the gap.
print(to_fim("def add(a, b):\n    return ", "\n\nprint(add(2, 3))", "a + b"))
## <fim_prefix>def add(a, b):\n    return <fim_suffix>\n\nprint(add(2, 3))<fim_middle>a + b
```

The three sentinels are unmergeable special tokens, reserved exactly like the agent control tokens of section 1 — same dial, different purpose. The model learns that when it sees `<fim_middle>`, its job is to produce text that joins the prefix to the suffix. This is how every serious code model from InCoder and StarCoder onward supports editor infill, and it is fundamentally a tokenizer-and-training-data design, not a model-architecture one.

Identifier boundaries are the quieter half of the code dial. Programmers write the same concept as `getUserName`, `get_user_name`, and `GetUserName`, and a generic tokenizer trained on prose will split each of these differently and inconsistently — sometimes `get`, `User`, `Name`; sometimes `get`, `_`, `user`, `_`, `name`; sometimes a single merged blob if the exact string happened to be frequent in the prose corpus. Inconsistent identifier segmentation forces the model to learn that three different token sequences refer to the same naming concept, which wastes capacity and weakens transfer between naming conventions. Code-tuned tokenizers mitigate this two ways: by training on a code-heavy corpus, so the common identifier sub-pieces (`get`, `set`, `user`, `name`) earn their own stable tokens; and by choosing a pre-tokenization regex that treats case transitions and underscores as soft boundaries, so `camelCase` and `snake_case` decompose into the same underlying word pieces. It is a smaller win than whitespace runs or FIM, but it compounds across the millions of identifiers in a real codebase, and it is the kind of detail that separates a tokenizer trained on code from a general tokenizer that merely *saw* some code.

Structured data that is not code — JSON, YAML, markup — sits in between. The agent lesson applies: mark the *boundaries* with special tokens where you control the format, and choose a payload serialization that is cheap under your tokenizer. JSON is expensive precisely because its punctuation tokenizes poorly; if you are designing a protocol from scratch and tokens matter, a format with less syntactic overhead pays off on every single call.

There is one more category worth a paragraph for contrast, because it shows the principle generalizing past natural language entirely: **scientific sequence data**. SMILES strings for molecules, DNA and protein sequences, and similar domains look superficially like text but have no word structure for BPE to discover. Applying subword merges to a SMILES string invents spurious "tokens" out of chemically meaningless byte runs — a merge that fuses the end of one functional group with the start of another teaches the model a boundary that does not exist. The right move for these domains is usually the opposite of everything above: *less* merging, not more. Character-level tokenization, or a small hand-curated vocabulary of chemically meaningful units, or fixed k-mers for genomics, all beat a BPE trained to compress. The dial here is granularity turned all the way down. It is the same playbook — match the tokenizer to the structure of the data — applied to data whose structure happens to reward minimal segmentation.

### Second-order optimization: serialization beats vocabulary for output

The recurring temptation with structured tasks is to keep enriching the vocabulary — add tokens for common JSON keys, for frequent code idioms, for everything. Past a point this is the wrong lever. The cheaper win is usually to change what the model is asked to emit. A model that returns a compact, low-punctuation representation and lets the harness expand it into JSON spends a fraction of the tokens of one that emits verbose JSON directly, under *any* tokenizer. Reserve the vocabulary changes for the structural boundaries that must be unambiguous (the FIM sentinels, the message delimiters) and the high-frequency content that genuinely dominates your corpus (whitespace runs in code). For everything else, design the format to be cheap rather than teaching the tokenizer to compress an expensive one.

## The cross-cutting discipline: evaluate, budget, and serve

Whatever dials you turn, three practices separate a tokenizer that works in a demo from one that works in production. They cut across all four tasks, and skipping them is how tokenizer bugs reach users.

![Which dial to turn, keyed on the dominant workload](/imgs/blogs/task-specialized-tokenizer-design-8.webp)

The decision tree summarizes the whole playbook: identify the dominant workload first, then turn the matching dial. Agents reserve a control namespace and test round-trip determinism. Math splits digits and locks normalization. New languages measure fertility, then extend the vocabulary with a mean-initialized warm start. Code adds whitespace-run tokens and FIM sentinels. The tree is keyed on the *workload*, not the model size — a 7B math model and a 70B math model want the same digit policy, and a 7B agent and a 70B agent want the same control tokens.

The first practice is **evaluation**, and the only honest way to evaluate a tokenizer is to run it over text that looks like your actual traffic and measure three things: fertility (cost), round-trip fidelity (correctness), and special-token coverage (does every reserved token appear in training data). Here is a compact harness:

```python
def tokenizer_report(tok, samples):
    rows = []
    for name, text in samples.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        round_trip_ok = tok.decode(ids) == text
        fert = len(ids) / max(1, len(text.split()))
        rows.append({
            "sample": name,
            "tokens": len(ids),
            "fertility": round(fert, 2),
            "round_trip": round_trip_ok,
        })
    return rows

samples = {
    "english_chat": "Can you book me a table for two at 7pm?",
    "tool_call":    '<|tool_call|>{"name":"book","args":{"size":2,"time":"19:00"}}',
    "arithmetic":   "What is 48273 + 9156?",
    "vietnamese":   "Dat ban cho hai nguoi luc bay gio toi.",
    "python":       "def f(x):\n    return x ** 2\n",
}
for row in tokenizer_report(tok, samples):
    print(row)
```

Round-trip fidelity deserves emphasis: a tokenizer that does not decode back to exactly what it encoded is a latent bug that will surface as corrupted output, broken tool calls, or mangled multilingual text. Test it on adversarial inputs — strings with your special-token literals in them, unusual Unicode, trailing whitespace — not just clean prose.

The second practice is **reserved-token budgeting**. You cannot add tokens to a deployed model without resizing embeddings and (ideally) further training, so you reserve a budget of unused special-token slots up front. The discipline is to reserve generously but *guard* the unused ones — keep them out of the model's reachable output until they are trained — so that an untrained reserved token never becomes a glitch token in production.

The third practice is **train/serve consistency**: the tokenizer that served a request must be byte-for-byte the one that trained the model. This sounds obvious and is violated constantly — a normalization setting that differs between the training pipeline and the inference server, a special-token added on one side but not the other, a different Unicode form. The symptom is a model that scores well offline and mysteriously worse in production. The fix is to pin the exact tokenizer artifact (not "the same algorithm," the same *file*) and to include a round-trip assertion in your serving health check.

| Practice | What it catches | The cheap test |
|---|---|---|
| **Fertility eval** | Cost and latency regressions on real traffic | Tokens-per-word on a held-out traffic sample |
| **Round-trip eval** | Silent corruption, broken structure | `decode(encode(x)) == x` on adversarial inputs |
| **Reserved-token budget** | Glitch tokens, "can't add tokens later" | Count untrained reachable special tokens |
| **Train/serve pinning** | Offline-online score gaps | Hash the tokenizer file in both pipelines |

### Token healing and the partial-token problem

There is a failure mode specific to any system that builds a prompt by concatenating strings — which is every agent, every RAG system, every structured-output harness — and it produces bugs that look like model errors. The issue is that tokenization is not compositional: `tokenize(a + b)` is not generally `tokenize(a) + tokenize(b)`. If you build a prompt that ends in the middle of what would naturally be a single token — say your template ends with `http` and the model is meant to continue with `://example.com` — the model now has to predict a continuation conditioned on a token boundary that never appears in natural text, because in real text `http` and the following characters merge into different tokens. The model has been pushed off the manifold of token sequences it was trained on, and it often responds with degraded or strange output.

The fix is *token healing*: before generation, back up over the last token or two of the prompt and let the model re-generate them, constrained to be consistent with the original prompt's trailing characters. This re-aligns the boundary so the model sees a natural token sequence. Several inference stacks implement it, and the symptom it cures — a model that completes a partial URL, path, or identifier badly when the prompt ends mid-token — is nearly impossible to find if you do not know token healing exists. A closely related and even simpler footgun is *trailing whitespace*: a prompt that ends in a space tokenizes differently from one that does not, because that space would normally attach to the following word's token, so a trailing-space prompt is a classic source of "the model behaves worse than it should for no visible reason."

### A note on byte-level and tokenizer-free directions

It is reasonable to ask whether all of this dial-turning is a temporary burden — whether models will eventually operate on raw bytes and make tokenizer design obsolete. There is real research in this direction: byte-level models and architectures that learn their own segmentation (entropy-driven dynamic patching, hierarchical byte processing) aim to remove the fixed vocabulary entirely. The appeal is exactly the failure modes in this post — no fertility tax, no glitch tokens, no digit misalignment, no normalization surprises — because there is no vocabulary to mis-tune. The cost is that bytes make sequences several times longer, so these models spend architecture and compute to claw back the efficiency a good tokenizer provides for free. As of today, tokenizer-based models remain the production default by a wide margin, and even the byte-level approaches reintroduce task-specific structure — where to place patch boundaries — that rhymes with the dials here. The practical takeaway is not "wait for tokenizers to disappear." It is that the *questions* this post asks — where should boundaries fall for this task, and what structure must be unambiguous — outlive any particular tokenizer technology.

## Case studies from production

The patterns above are abstractions distilled from a decade of public incidents and a fair number of private ones. Here are nine concrete cases, each a tokenizer decision (or mistake) that shaped a real system. They are organized roughly by the dial they illustrate, and together they trace the same arc the field has walked.

![Frontier tokenizers drifted from generic to task-specialized](/imgs/blogs/task-specialized-tokenizer-design-9.webp)

The timeline shows that arc: from GPT-2's single generic web vocabulary in 2019, through Codex adding whitespace-run tokens for code in 2021, the FIM sentinels of InCoder and StarCoder in 2022, the digit- and code-aware `cl100k` in 2023, Llama-3's single-digit 128k vocabulary in 2024, and gpt-oss's Harmony channel tokens in 2025. Each step is a dial being turned for a task the previous generic vocabulary served badly. The cases below are the texture behind that line.

### 1. SolidGoldMagikarp and the untrained-token landmine

In early 2023, researchers probing the GPT-2/GPT-3 tokenizer found a cluster of tokens — `SolidGoldMagikarp`, ` petertodd`, and dozens of others — that made the models behave bizarrely: asked to repeat them, the models would dodge, insult the user, or emit unrelated text. The root cause was pure tokenizer hygiene. These tokens had been created by BPE merges over a training corpus that included scraped Reddit usernames and counting-thread artifacts, so they earned a vocabulary slot — but the *model's* training corpus barely contained them, so their embedding rows were never trained. They sat near their random initialization, with anomalously small norms, and feeding them to the model was feeding it noise it had never learned to interpret. The lesson for every team that adds special tokens: a vocabulary slot without training data is a live wire. This is exactly the second-order trap from the agent section, discovered the hard way at the scale of a frontier model, and it is why reserved-token budgeting and special-token coverage checks are non-negotiable.

### 2. The merged-digit arithmetic gap

A widely-reproduced finding across 2022–2023 was that models using three-digit-grouped tokenization (the GPT-3.5 family) made systematic errors on multi-digit arithmetic that models with single-digit tokenization (the Llama family) avoided, even at comparable scale. The errors were not random — they clustered around carries that crossed token boundaries and around numbers whose length was not a multiple of three. The same model that could write a sorting algorithm would misadd two five-digit numbers, because the tokenizer never gave it a stable place-value representation to compute over. The fix that the field converged on was the one in section 2: split digits individually. It costs more tokens per number and it makes the model better at arithmetic, which is the clearest possible demonstration that fertility is the wrong thing to optimize for math.

### 3. The right-to-left and comma tricks

Following the merged-digit findings, practitioners discovered two cheap mitigations that did not require retraining the base model. The first was inserting thousands separators: prompting the model with `48,273` instead of `48273` changes the pre-tokenization boundaries so that the comma-delimited groups align to place value, and measurably improves addition accuracy on models stuck with grouped digits. The second, used in some custom tokenizers, was grouping digits from the *right* rather than the left, so the rightmost token always holds the ones, tens, and hundreds — keeping fertility low while aligning the most carry-active digits across operands. Neither is as clean as single-digit tokenization, but both are deployable on a model you cannot retrain, and both make the same point: the tokenizer's digit boundaries, not the model's reasoning, were the bottleneck.

### 4. Llama-2 to Llama-3: the multilingual speedup

When Meta moved from Llama-2 to Llama-3, one of the least-discussed but most consequential changes was the tokenizer: from a 32k SentencePiece vocabulary to a 128k vocabulary based on tiktoken, trained on a far more multilingual corpus. The headline result that engineers noticed immediately was a roughly 2× reduction in token counts for many non-English languages — which is simultaneously a 2× cost reduction, a 2× latency improvement per word, and a 2× effective context extension for those languages, all from the tokenizer alone. The model got better at non-English text not only because it was trained on more of it, but because the tokenizer stopped charging a punitive fertility tax at the input. It is the cleanest production example of the fertility dial from section 3, turned at the scale of a flagship model.

### 5. Vietnamese vocabulary extension on Llama

Several Vietnamese-focused projects — PhoGPT, the VinaLLaMA and SeaLLM lineages, and others — followed the extension recipe from section 3 almost exactly: start from a Llama base, train an auxiliary BPE on a large Vietnamese corpus, merge the new pieces, mean-initialize the new embedding rows, and continue pretraining. The reported fertility improvements were substantial — often cutting Vietnamese token counts by a third or more — and crucially, English ability was largely preserved because the base merges were kept intact and the extension only *added* pieces. The teams that got this right verified the merge-boundary invariant (English text tokenizes unchanged) and warm-started the new rows; the teams that skipped the mean-init spent far more continued-pretraining compute getting the new tokens to converge. The recipe is boring and it works, which is exactly what you want from a tokenizer procedure.

### 6. Codex and the run-of-spaces problem

The original GPT-3 BPE vocabulary, optimized for web prose, had no tokens for runs of whitespace — and Python, the most-requested language for code generation, makes whitespace syntactically significant. When OpenAI built Codex, one of the necessary tokenizer changes was adding tokens for runs of spaces (2, 3, 4, and more), so that indentation did not explode token counts and the model could represent nesting depth cleanly. This is the change my opening war story was missing, productized: it is the difference between a code model that treats `            ` as twelve tokens and one that treats it as one. The whitespace dial from section 4, turned at the moment code generation became a serious product, and a permanent fixture of every code tokenizer since.

### 7. InCoder, StarCoder, and the FIM sentinels

Left-to-right language models cannot natively do editor infill — completing the gap between code above and below the cursor — because their training objective only ever predicts forward. InCoder and then StarCoder solved this with fill-in-middle: reserve three special tokens (`<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`), rearrange training documents so the model learns to produce the middle given the prefix and suffix, and you get a model that infills with ordinary autoregressive decoding. The entire capability lives in three vocabulary entries and a data transformation. It is the strongest case for the special-token dial outside of agents: a genuinely new capability — bidirectional-feeling completion — delivered through tokenizer design rather than a new architecture. Every code-completion product you use in an editor today depends on it.

### 8. ChatML, Harmony, and the channel protocol

The progression from raw text prompting to ChatML's `<|im_start|>`/`<|im_end|>` to gpt-oss's Harmony channel tokens is the agent dial maturing in public. ChatML's contribution was making role boundaries into special tokens, so "where does the system prompt end and the user turn begin" stopped being a fragile string convention. Harmony went further, adding a channel mechanism — `analysis`, `commentary`, `final` — so that hidden reasoning, tool calls, and user-visible answers are each marked by reserved tokens and routed by ID. The payoff is the one from section 1: a harness can strip chain-of-thought, detect a tool call the instant it starts, and reconstruct a clean transcript, all without a single regex over model text. Each generation of this protocol moved more of the agent's structure out of brittle text and into robust tokens.

### 9. SentencePiece normalization quietly breaking math

A recurring and maddening class of bug: a model scores worse on a math evaluation than expected, and the cause turns out to be NFKC normalization in a SentencePiece tokenizer silently rewriting the math. The single glyph `½` becomes three characters; full-width digits used in some locales fold to ASCII in ways that change spacing; various Unicode minus and multiplication signs get harmonized. Each rewrite is individually reasonable for prose and individually destructive for a math benchmark that depends on exact symbol handling. The teams that hit this discovered that the input the model trained and evaluated on was not the input anyone thought they were feeding it. The fix is the normalization dial from section 2: use the lightest Unicode normalization that keeps text valid, and audit exactly what the normalizer does to a sample of your real inputs before trusting any math number.

### 10. Mistral's tekken and the control-token redesign

When Mistral introduced its tekken tokenizer with the Nemo generation, the headline was efficiency — a tiktoken-based vocabulary roughly 30% more compact than the previous SentencePiece tokenizer on source code and many non-English languages — but the quieter and more durable change was a cleaner control-token namespace. Earlier Mistral models encoded instruction boundaries with text conventions like `[INST]` and `[/INST]`: ordinary tokens a user could accidentally type, with all the ambiguity that implies. The redesign moved toward reserved, unmergeable control tokens for the structural boundaries, the same direction ChatML and Harmony took. It is a representative example of a frontier lab treating the special-token namespace as a first-class design surface rather than an afterthought, and of the industry converging on a single principle: structure belongs in tokens, not in text.

### 11. Gemma's 256k vocabulary and the embedding tax

Gemma shipped with a 256k-token vocabulary, several times larger than the typical 100–128k, and it is the clearest production illustration of the embedding-tax trade-off from section 3. The large vocabulary buys excellent multilingual fertility — far fewer tokens per word across a wide range of languages — but it pays for that in embedding parameters: at 256k tokens and a few thousand hidden dimensions, the input embedding and output projection together account for a substantial fraction of a small model's total parameters, and the final softmax over 256k entries is a real per-step compute cost. The choice makes sense for a model family that prioritizes multilingual reach; it makes much less sense for a narrow English-only model, where those parameters would be better spent on depth. The lesson is that vocabulary size is not a "bigger is better" knob — it is a trade between input efficiency and the cost of carrying and computing over the embedding table, and the right point on that trade depends entirely on the workload.

### 12. The constrained-decoding boundary bug

A failure that surfaces repeatedly in agent and structured-output systems: a tool-calling integration works perfectly in testing and then produces malformed calls in production, intermittently, with no clear pattern. The root cause is frequently a token-boundary mismatch between the constrained-decoding grammar and the tokenizer. The grammar enforces, say, that a JSON string value is followed by a comma or a closing brace — a character-level rule — but the model emits tokens, and a token like `",` or `"}` satisfies the grammar in a way the naive character-level enforcement did not anticipate; or conversely, the only tokens consistent with the next required character get masked out, forcing the model onto a low-probability path that derails generation. Teams that hit this learn to use token-aware constrained-decoding libraries that reason about which token IDs are consistent with the grammar, rather than enforcing the grammar on decoded characters after the fact — and they learn, again, that reserving single-token boundaries for the structural delimiters removes an entire class of these bugs, because a delimiter that is its own token can never straddle a grammar boundary.

### 13. DeepSeek's bilingual vocabulary

DeepSeek's models were built for strong performance in both Chinese and English, and the tokenizer reflects that as a deliberate decision rather than an afterthought. Instead of taking an English-centric vocabulary and accepting a high fertility tax on Chinese, or taking a Chinese vocabulary and paying it on English, the tokenizer was trained on a balanced bilingual corpus so that both languages earn efficient merges within a roughly 100k-token budget. The result is competitive fertility on both scripts at once — which matters because every token saved on Chinese input is context budget and latency returned to the user, and at the scale of a bilingual product those savings compound across every request. It is the fertility dial of section 3 turned not for a single new language bolted onto an English base, but for two first-class languages from the start. The broader lesson is that "multilingual" is not a single setting: a tokenizer balanced for two specific languages will beat a generically multilingual one on exactly those two, and if you know your traffic is bilingual, designing the vocabulary for that pair is worth the effort.

## When to customize a tokenizer, and when to just use cl100k

Customizing a tokenizer is real work with real risk — every change is a potential train/serve mismatch, a potential glitch token, a potential merge-boundary regression. The decision tree earlier keys on workload; this section keys on whether to reach for the tools at all.

**Reach for a customized tokenizer when:**

- **You are building an agent or a structured-output system.** Reserved control tokens for channels, tool calls, and message boundaries are worth it almost unconditionally; the reliability gain over text-based protocols is large and the cost is a handful of vocabulary slots plus the discipline to train them.
- **The model will do arithmetic or symbolic math.** Single-digit tokenization (if training from scratch) or right-to-left grouping, plus locked-down normalization, is the difference between a model that can add and one that cannot. This is a correctness change, not an optimization.
- **Your dominant traffic is a language the base vocabulary under-serves.** If measured fertility on your real traffic is above ~2.5 tokens per word, vocabulary extension with mean-initialized embeddings will pay for itself in cost, latency, and context budget.
- **You are training a code model.** Whitespace-run tokens and FIM sentinels are not optional for a serious code model; they are the baseline, and they cannot be retrofitted without retraining.
- **You work in a scientific-sequence domain.** SMILES, DNA, protein, and similar data want *less* merging, not more — a deliberately minimal or character-level tokenizer beats a compression-optimized BPE.

**Skip the custom tokenizer when:**

- **You are building an English (or major-language) chat assistant** and the base model's tokenizer already has a fertility near 1 on your traffic. Spend your effort elsewhere; the tokenizer is not your bottleneck.
- **You can solve the problem in the prompt.** Thousands-separator insertion for arithmetic, or a compact output format the harness expands, often captures most of the win with none of the train/serve risk.
- **You cannot commit to training the tokens you add.** An untrained special token is worse than no special token. If you are not going to fine-tune on data that exercises every new token, do not add it.
- **The base model is frozen and you are API-only.** If you cannot resize embeddings and continue training, your tokenizer is fixed; work with it via prompting and serialization, not vocabulary surgery.
- **The fertility tax is small and the traffic is low.** Vocabulary extension has a fixed engineering and embedding-tax cost; below some traffic threshold it does not pay back. Measure first.

The thread through all of it is the one from the first figure: the tokenizer exposes four dials — granularity, special tokens, normalization, digit and whitespace policy — and a workload is a prescription for where to set them. The mistake is not turning the dials wrong; it is forgetting they exist, inheriting a vocabulary built for someone else's traffic, and then debugging the consequences everywhere except the one layer the model can never learn around.

## Further reading

- [Designing and choosing a tokenizer for your LLM](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) — the general design space: algorithm families, vocab-size economics, library choice, and the serving stack.
- [BPE Tokenizer: the foundation of modern language models](/blog/machine-learning/large-language-model/bpe-tokenizer) — the algorithm internals, worked merge traces, and byte-level BPE.
- [Training LLMs for math](/blog/machine-learning/large-language-model/training-llm-for-math) — the downstream training side of the digit-policy story.
- [Training LLMs to adapt to a new language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language) — data mixtures, continued pretraining, and forgetting mitigation for the extension recipe.
- The fill-in-middle training transformation (Bavarian et al., 2022) and the StarCoder technical report for the canonical FIM and code-tokenizer designs.
- The OpenAI Harmony response format documentation for the channel-token agent protocol used by gpt-oss.
