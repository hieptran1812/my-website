---
title: "The tokenizer boundary: incremental detokenization and the bugs that live at the edge of your engine"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Your engine speaks integers and your users speak text, and everything that goes wrong between them - broken UTF-8 in a live stream, a stop string leaked to the client, a prompt format that silently degrades quality - lives in one file you probably have not written yet."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "tokenizer",
    "detokenization",
    "streaming",
    "chat-templates",
    "transformers",
    "pytorch",
    "latency",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 57
---

There is a class of bug that only appears in production, only for some users, and only in the middle of a sentence. A support ticket arrives with a screenshot: the assistant's reply reads fine for two paragraphs and then, right where a currency symbol should be, there are three black diamonds with question marks in them. Nobody can reproduce it locally. The evaluation suite is green. The model is fine. The kernels are fine. The bug is in eleven lines of Python that call `tokenizer.decode()` once per generated token and concatenate the results.

That file — the one that turns a request into integers and turns integers back into a stream of text — is the subject of this post. Most people treat the tokenizer as preprocessing: a thing you call before the interesting part starts. It is not. It is a **boundary of your engine**, in the same sense that a syscall interface is a boundary of a kernel. Everything on the inside is integers. Everything on the outside is bytes that a human will read. The boundary is where the two representations meet, and it is where a shocking proportion of production incidents actually live: broken characters in a stream, a stop sequence leaked to the client before it was detected, a prompt hand-built with the wrong role markers that quietly costs you a chunk of model quality, a bill that is 40% higher for one language than another for the same document.

![Layered view of the tokenizer boundary showing the chat template, the encoder, the engine that handles only integers, the incremental detokenizer and the stop check before output reaches the client](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-1.webp)

By the end of this post you will have written `nanoserve/tokenizer.py`: a `PromptBuilder` that renders and encodes a chat request without letting a user inject a control marker, an `IncrementalDetokenizer` that emits only the newly *stable* prefix of the output each step and never emits a broken character, and a `StopChecker` that catches stop strings spanning several tokens without leaking any of them. You will also be able to answer three questions your finance team will eventually ask: why the same document costs more on one model than another at the same price per token, how much of your model's weights are the output projection, and why your GPU is idle 12% of the time at high concurrency when no kernel got slower.

This is post 4 of the [Inference Engineering series](/blog/machine-learning/inference-engineering/what-inference-engineering-is). We are still in Track A, still building the path from weights to a token. The previous post wrote the forward pass; this one writes the two ends of it. If you want the *training-side* view — how a BPE merge table is learned, how you choose a vocabulary size before training — the repo already has [how BPE tokenizers work](/blog/machine-learning/large-language-model/bpe-tokenizer) and [designing and choosing a tokenizer for an LLM](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm). I will not re-derive any of that. This post is entirely about the **serving-time contract**.

A note on numbers, repeated in every post of this series: I have no GPU and I have run nothing. Every quantity below is either derived from arithmetic I show you, cited from a public source I link, or written as a script for you to run with an expected direction rather than a fabricated value.

## 1. The engine speaks integers: what the boundary owns

Open any inference engine and follow a request inward. Somewhere near the HTTP handler, a list of chat messages becomes a single string. That string becomes a list of integers. From there until the very end, nothing in the engine knows anything about text. The scheduler sees sequence lengths. The KV cache sees block indices. The attention kernel sees tensors. The sampler sees a vector of 128,256 logits and returns one integer. Not one of those components could tell you whether the model is writing English, Python, or base64.

Then, at the far end, integers have to become text again — and not in one shot at the end, but *incrementally*, character by character, as they arrive, because the product is a chat interface and the user is watching.

The figure above is the shape of it. Five layers, all of them on the host CPU, all of them outside the part of the system you profile with Nsight. Reading top to bottom:

1. **The chat template** turns a list of role-tagged messages into one string with the exact control markers the model was post-trained on.
2. **The encoder** turns that string into token ids, with special-token handling that must distinguish markers *you* wrote from bytes a *user* typed.
3. **The engine** runs. Integers in, integers out.
4. **The incremental detokenizer** turns the growing id list into a growing string, holding back bytes that do not yet form a complete character.
5. **The stop check** holds back a few more characters, in case what just arrived is the beginning of a stop sequence.

Layers 1 and 2 run once per request and cost milliseconds. Layers 4 and 5 run once per *token per stream* and cost microseconds — which, multiplied by concurrency, is where the interesting performance story is. We will get there in section 10.

The reason to draw the boundary explicitly is that its two halves fail in completely different ways. Encode-side bugs are **security and quality** bugs: a marker injected by a user, a prompt format that does not match training, a double BOS token. Decode-side bugs are **correctness and latency** bugs: a half-formed character on the wire, a missing space, a stop string leaked. They deserve separate code, separate tests, and separate attention. Let us take them in order.

## 2. The encode side: what really happens to a request string

Start with the mechanism, because almost every encode-side bug follows from one property of byte-level BPE that people know abstractly but do not apply.

A modern byte-level BPE tokenizer (GPT-2 style, tiktoken style, Llama-3 style) runs three stages on an input string:

1. **Marker scan.** The input is searched for the literal text of any *special* or *added* token. Matches are cut out and replaced by their id directly. This happens **before** anything else.
2. **Pretokenization.** The remaining text is split by a regular expression into chunks — roughly words, runs of digits, runs of punctuation, and runs of whitespace. Merges are never allowed to cross a chunk boundary. This is why a token can be `" cost"` but never `"cost the"`.
3. **Merge.** Each chunk's bytes are mapped into a printable alias alphabet (the classic `bytes_to_unicode` trick: byte 0x20 becomes the character U+0120, so that every byte has a printable representation), and the ranked merge table is applied greedily until no merge applies.

Three consequences fall directly out of that pipeline, and each one is a production bug I have seen described in more than one incident write-up.

### The same string does not always tokenize the same way

Because merges are greedy and pretokenization is context-sensitive at the edges, `"Hello"` and `" Hello"` are different tokens with different ids, and

$$
\text{encode}(A) \mathbin\Vert \text{encode}(B) \neq \text{encode}(A \mathbin\Vert B)
$$

in general. Concatenating the ids of two separately-encoded strings is *not* the same as encoding their concatenation. The difference is usually confined to the tokens straddling the join, but "usually confined to one or two tokens" is exactly the kind of statement that turns into a bug report six months later.

Where this actually bites in a serving engine: **prefix caching**. If two requests share a system prompt as *text*, you would like them to share KV blocks. They only do if they share a prefix as *ids*. Encode the system prompt separately and splice, and you get a stable id prefix. Encode the whole rendered conversation as one string every time, and the boundary token between the system block and the user block can shift as the user's first character changes — which silently drops your cache hit rate. (The [prefix caching post](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) covers what you do with the shared prefix once you have it; the point here is that whether you *have* one is decided by the tokenizer.)

In practice, the marker-based chat templates rescue you: because every block starts with a special token that is matched in stage 1 and never participates in a merge, the id boundary between blocks is deterministic. This is one of several reasons to use the template rather than raw string concatenation.

### Special tokens are a code path, not data

![Dataflow showing request text entering a marker scan that either matches a literal control marker or falls through to byte-level pretokenization and BPE merges, with both paths converging on the final id stream](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-2.webp)

Stage 1 is the interesting one. The tokenizer scans the input for the *literal text* of every special token before it does anything else. If a user types the eight characters `<|eot_id|>` into your chat box, and you pass that string to an encoder with special-token parsing enabled, the tokenizer will happily emit the end-of-turn id. From the model's point of view, the user's turn ended right there — and whatever the user typed *after* it appears to the model to be a new turn, possibly with a `<|start_header_id|>system<|end_header_id|>` header that the user also typed.

That is prompt injection with no cleverness required. It does not need a jailbreak prompt or a role-play framing; it needs the user to type ten ASCII characters that your encoder treats as a control instruction. The figure above shows exactly where the fork is. The marker scan runs before the merge table; a matched marker either injects a turn boundary or gets escaped back into ordinary text, and a single encode flag decides which. Both outcomes land in the same id stream, where the model has no way to tell a marker a user typed from one your template wrote.

The fix has two halves, and you need both:

```python
# nanoserve/tokenizer.py
from transformers import AutoTokenizer

class SpecialTokenError(ValueError):
    pass

class PromptBuilder:
    """Renders chat messages to token ids without letting user text
    encode into a control marker."""

    def __init__(self, model_id: str):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # len(tok) includes added tokens; tok.vocab_size does NOT.
        self.n_vocab = len(self.tok)
        self.special_ids = set(self.tok.all_special_ids)
        for t in self.tok.additional_special_tokens:
            self.special_ids.add(self.tok.convert_tokens_to_ids(t))
        # Every literal marker string, longest first, so that a longer
        # marker is not shadowed by a shorter one that is its prefix.
        self.marker_strings = sorted(
            {getattr(t, "content", str(t))
             for t in self.tok.added_tokens_decoder.values()},
            key=len, reverse=True,
        )

    def sanitize(self, text: str) -> str:
        """Neutralize any literal control marker inside user content.
        A zero-width space inside the pipe pair keeps the text readable
        to a human while breaking the exact-match scan."""
        for m in self.marker_strings:
            if m in text:
                text = text.replace(m, m.replace("<|", "<\u200b|"))
        return text
```

The first half is *sanitizing message content before rendering*. The second half is *asserting after encoding*, because sanitization is a blacklist and blacklists rot when a model version adds a marker you did not know about:

```python
    def encode_chat(self, messages: list[dict], add_generation_prompt=True):
        clean = [
            {**m, "content": self.sanitize(m["content"])}
            for m in messages
        ]
        ids = self.tok.apply_chat_template(
            clean,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        # Whatever markers survive must be ones the TEMPLATE wrote.
        # Count them and compare against what the template should emit.
        return ids

    def encode_user_text(self, text: str) -> list[int]:
        """Encode a fragment that must contain NO control markers."""
        try:
            ids = self.tok.encode(
                text, add_special_tokens=False, split_special_tokens=True
            )
        except TypeError:
            # Older transformers has no split_special_tokens kwarg.
            ids = self.tok.encode(self.sanitize(text),
                                  add_special_tokens=False)
        bad = self.special_ids.intersection(ids)
        if bad:
            names = [self.tok.convert_ids_to_tokens(i) for i in bad]
            raise SpecialTokenError(f"user text encoded to markers: {names}")
        return ids
```

`split_special_tokens=True` is the flag that tells recent versions of `transformers` to treat special-token text as ordinary text and run it through the merge table. Check that your version has it; the `try/except` above is not defensive paranoia, it is the difference between a working guard and a silently absent one. And keep the assertion even when the flag works — it costs a set intersection over a list of a few thousand integers, which is nothing next to a forward pass, and it converts a security bug into a 400 response.

### Added tokens are not vocabulary tokens

There is a second population of special ids that is easy to miss. A tokenizer has:

- **Vocabulary tokens** — the ones the BPE trainer produced, reachable through the merge table.
- **Added tokens** — entries in `added_tokens_decoder`, matched by literal string in stage 1, never produced by merging.

Two facts about added tokens cause real bugs:

- `tokenizer.vocab_size` reports the **base** vocabulary and excludes added tokens; `len(tokenizer)` includes them. For Llama 3 the base BPE vocabulary is 128,000 entries and there are 256 reserved and special entries on top, giving 128,256 rows in the embedding matrix — which is what `config.vocab_size` says. Size any array off `tokenizer.vocab_size` and you will be 256 short, which produces either an `IndexError` or, worse, silent wraparound if you index without bounds checking. **Always build tables with `len(tokenizer)` and cross-check against `model.config.vocab_size`.**
- If *you* call `tokenizer.add_tokens(...)` without resizing and training the embedding matrix, the new ids index rows of randomly-initialized (or out-of-range) embeddings. The model does not error; it just produces nonsense whenever that token appears. This is a fine-tuning bug that arrives at the serving team as "the model went crazy on our custom tags".

### Should you trust `add_special_tokens`?

No — you should *decide* it, per call site, and write the decision down.

`tokenizer(text, add_special_tokens=True)` (the default) prepends whatever the tokenizer's post-processor says goes at the start, which for Llama-family models is a BOS token. Chat templates *also* emit BOS as part of the rendered string. Do both and you get two BOS tokens: `<|begin_of_text|><|begin_of_text|><|start_header_id|>...`. The model was never post-trained on that, generation quality drops slightly and unpredictably, and nothing anywhere raises an error.

The rule for `nanoserve`, and for any engine:

| Call site | `add_special_tokens` | Why |
| --- | --- | --- |
| `apply_chat_template(tokenize=True)` | not applicable | the template owns BOS |
| Encoding an already-rendered template string | `False` | BOS is already in the string |
| Encoding a raw completion prompt (no chat) | `True` | nothing else adds BOS |
| Encoding user text for splicing | `False` | it is a fragment, not a sequence |
| Encoding a stop string for id comparison | `False` | you want the bare tokens |

And then assert it. One line at startup catches every version bump that changes a default:

```python
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}],
    tokenize=False, add_generation_prompt=True,
)
ids_from_string = tok.encode(rendered, add_special_tokens=False)
ids_from_template = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}],
    tokenize=True, add_generation_prompt=True,
)
assert ids_from_string == ids_from_template, (
    "template/encoder disagree: check add_special_tokens"
)
bos = tok.bos_token_id
assert ids_from_template.count(bos) <= 1, "double BOS"
```

This assertion has caught more real problems for more teams than any test you will write about attention.

## 3. Chat templates are the prompt format

Here is the single most common quality bug in self-hosted LLM serving, and it has nothing to do with the engine at all:

```python
prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
```

This looks reasonable. It runs. The model answers. And it is wrong, in a way that costs you real capability and produces no error, no warning, and no anomaly in any metric you are collecting.

![Side-by-side comparison of a hand-concatenated f-string prompt that silently degrades quality against a template-rendered prompt with exact role markers and a generation prompt](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-3.webp)

The reason is simple once stated. Instruction-tuned models are trained on a *specific byte sequence* around each turn. The post-training data for Llama 3.1 does not contain the string `"User:"` as a role marker; it contains the token `<|start_header_id|>`, the token for `user`, the token `<|end_header_id|>`, two newlines, the content, and `<|eot_id|>`. Those markers are single tokens with dedicated embeddings that the model learned to condition on very strongly. Replace them with English words and you have moved the prompt off the distribution the model was aligned on. The model still responds — it is a language model, it will complete anything — but instruction-following, refusal behavior, tool-call formatting, and stop behavior all degrade by an amount nobody can predict and everybody notices eventually.

The correct move is to never write the format yourself. It ships with the checkpoint, as a Jinja template inside `tokenizer_config.json`, and you render it:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
print(tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True))
```

For a Llama 3.1 instruct checkpoint that prints something very close to this — read the exact strings out of *your* checkpoint rather than trusting mine, because they change between model versions:

```console
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 20 Jul 2026

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

Several things in that output are worth stopping on.

**The template injects content you did not write.** The two "Cutting Knowledge Date" and "Today Date" lines are emitted by the Jinja template itself, not by your system message. They are part of the format the model was tuned with. If you hand-build the prompt you lose them; if you cache rendered prompts across days, the date line changes and your prefix cache misses every midnight. That second one is a genuine operational subtlety: **your prefix cache hit rate can drop at a fixed time every day for a reason that lives in a Jinja template.**

**The generation prompt is a real thing.** `add_generation_prompt=True` appends `<|start_header_id|>assistant<|end_header_id|>\n\n` with no content and no `<|eot_id|>`. That trailing header is what tells the model "your turn now, in the assistant role". Forget it and the model may well continue the *user's* turn, writing a second user message — which is the classic "the model is talking to itself" symptom.

**The trailing newlines matter.** There are exactly two newlines after the assistant header, and the model's first generated token continues from there. If you strip trailing whitespace anywhere in your pipeline — a `prompt.strip()` in a logging wrapper, a YAML round-trip that trims — you have changed the prompt. Byte-exactness is the whole point.

### Tools and multi-part system blocks

Tool calling makes the template do more work. Templates that support tools take a `tools=` argument and render the JSON schemas into the system block in a model-specific layout:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
text = tok.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True)
```

Different model families place tool schemas in different positions, use different JSON formatting, and mark tool *results* with different roles (`tool`, `ipython`, `function`). Llama 3.1 has a `<|python_tag|>` marker for one of its tool-call styles. There is no cross-model standard, which is exactly why hand-building is hopeless and why the template is the only source of truth. If your engine exposes an OpenAI-compatible `/v1/chat/completions` endpoint, the translation from OpenAI's tool schema to the model's rendered format is a per-model adapter, and it belongs next to the template, not in your HTTP layer.

### Template drift

Templates change between model versions, and they change in ways that are invisible unless you diff them. A minor release can add a date line, reorder a system block, change whether a trailing newline is emitted, or add a new special token for a new capability. Your engine must therefore:

- **Pin the template to the checkpoint**, not to your codebase. Load it from `tokenizer_config.json` at model load. Never vendor a copy.
- **Snapshot-test the rendered output.** Store the rendered string for a fixed set of message lists as a golden file, and fail loudly on a diff at model upgrade. The diff is the release note you did not get.
- **Log the template hash** with every request batch, so that when quality shifts you can correlate it with a template change instead of blaming the sampler.

```python
import hashlib
tmpl = tok.chat_template or ""
template_hash = hashlib.sha256(tmpl.encode()).hexdigest()[:12]
```

Twelve hex characters in your startup log will one day save you a week.

## 4. The streaming problem: you cannot decode one token at a time

Now the heart of the post.

The engine produces one token id per step per sequence. The product needs to push text to a browser as it appears. The obvious implementation is:

```python
# WRONG. Do not ship this.
for tid in generate(...):
    yield tok.decode([tid])
```

It works for English prose most of the time, which is what makes it so dangerous — it survives your local testing and fails for a fraction of real traffic. There are three independent ways it is wrong.

### Failure 1: a character split across two tokens

Byte-level BPE has a vocabulary over **bytes**, not characters. Every byte value 0–255 has a token (that is what makes the tokenizer lossless and able to represent any input). When the merge table has no entry covering a particular multi-byte character, the tokenizer emits its bytes individually, or in fragments.

UTF-8 encodes a code point in one to four bytes ([RFC 3629](https://www.rfc-editor.org/rfc/rfc3629)): ASCII in one, Latin letters with diacritics and Greek and Cyrillic in two, most CJK characters and precomposed Vietnamese syllables in three, and everything in the supplementary planes — including emoji — in four. A three-byte character emitted as a byte token plus a two-byte token is two decode steps. Call `decode()` on the first one alone and you get U+FFFD, the replacement character, because a lone `0xE2` is not valid UTF-8. Concatenate the per-token decodes and you have permanently destroyed the character: the information about which bytes they were is gone.

<figure class="blog-anim">
<svg viewBox="0 0 700 300" role="img" aria-label="Five tokens arrive in sequence; the naive decoder prints replacement characters for the two byte tokens while the incremental decoder holds them in a buffer and prints one correct character once the sequence completes" style="width:100%;height:auto;max-width:820px">
<style>
.ad-chip{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ad-cl{font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ad-side{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ad-out{font:600 18px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.ad-bad{fill:#dc2626}
.ad-good{fill:var(--accent,#6366f1)}
.ad-buf{fill:var(--accent,#6366f1);opacity:.15}
.ad-buft{font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.ad-rule{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4}
@keyframes ad-s1{0%,3%{opacity:.10}8%,100%{opacity:1}}
@keyframes ad-s2{0%,23%{opacity:.10}28%,100%{opacity:1}}
@keyframes ad-s3{0%,43%{opacity:.10}48%,100%{opacity:1}}
@keyframes ad-s4{0%,63%{opacity:.10}68%,100%{opacity:1}}
@keyframes ad-s5{0%,83%{opacity:.10}88%,100%{opacity:1}}
@keyframes ad-hold{0%,42%{opacity:.12}48%,64%{opacity:1}70%,100%{opacity:.12}}
.ad-a1{animation:ad-s1 12s ease-in-out infinite}
.ad-a2{animation:ad-s2 12s ease-in-out infinite}
.ad-a3{animation:ad-s3 12s ease-in-out infinite}
.ad-a4{animation:ad-s4 12s ease-in-out infinite}
.ad-a5{animation:ad-s5 12s ease-in-out infinite}
.ad-ah{animation:ad-hold 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ad-a1,.ad-a2,.ad-a3,.ad-a4,.ad-a5{animation:none;opacity:1}.ad-ah{animation:none;opacity:.12}}
</style>
<text class="ad-side" x="14" y="66">tokens in</text>
<g class="ad-a1"><rect class="ad-chip" x="170" y="40" width="92" height="40" rx="8"/><text class="ad-cl" x="216" y="65">cost</text></g>
<g class="ad-a2"><rect class="ad-chip" x="270" y="40" width="92" height="40" rx="8"/><text class="ad-cl" x="316" y="65">sp</text></g>
<g class="ad-a3"><rect class="ad-chip" x="370" y="40" width="92" height="40" rx="8"/><text class="ad-cl" x="416" y="65">0xE2</text></g>
<g class="ad-a4"><rect class="ad-chip" x="470" y="40" width="92" height="40" rx="8"/><text class="ad-cl" x="516" y="65">0x82 0xAC</text></g>
<g class="ad-a5"><rect class="ad-chip" x="570" y="40" width="92" height="40" rx="8"/><text class="ad-cl" x="616" y="65">each</text></g>
<text class="ad-side" x="14" y="138">byte buffer</text>
<g class="ad-ah"><rect class="ad-buf" x="170" y="112" width="230" height="36" rx="8"/><text class="ad-buft" x="186" y="136">holding 1 of 3 bytes</text></g>
<line class="ad-rule" x1="14" y1="170" x2="686" y2="170"/>
<text class="ad-side" x="14" y="212">naive decode</text>
<text class="ad-out ad-a1" x="170" y="212">cost</text>
<text class="ad-out ad-bad ad-a3" x="224" y="212">&#xFFFD;</text>
<text class="ad-out ad-bad ad-a4" x="235" y="212">&#xFFFD;&#xFFFD;</text>
<text class="ad-out ad-a5" x="257" y="212">each</text>
<text class="ad-side" x="14" y="272">incremental</text>
<text class="ad-out ad-a1" x="170" y="272">cost</text>
<text class="ad-out ad-good ad-a4" x="224" y="272">&#x20AC;</text>
<text class="ad-out ad-a5" x="235" y="272">each</text>
</svg>
<figcaption>The same five tokens through two decoders: printing each token alone leaks three replacement characters, while buffering the incomplete UTF-8 sequence across steps 3 and 4 emits one correct character.</figcaption>
</figure>

The animation shows the shape of the fix: the correct decoder does not emit anything at step 3. It holds the byte. At step 4 the sequence completes and one character is released. The user sees a slightly bursty stream instead of a corrupted one, which is a trade every user will take.

Note *which* users hit this. English ASCII prose almost never triggers it. Text with currency symbols, dashes, quotation marks from a word processor, mathematical symbols, accented names, or any non-Latin script triggers it constantly. If your first market is English-only, this bug ships to production and waits.

### Failure 2: spaces appear and disappear

Byte-level BPE represents a leading space as part of the token. Internally the space byte 0x20 is aliased to the printable character U+0120, so a token that means `" cost"` prints as `Ġcost` when you look at the piece string. Concatenating pieces and mapping the alias back to bytes reproduces the spaces exactly.

But `tokenizer.decode([single_id])` does more than map bytes back. Two things it does will hurt you:

- **SentencePiece-derived tokenizers strip a leading space at sequence start.** The piece `▁cost` (U+2581 is SentencePiece's space marker) decodes to `"cost"` when it is the only piece, because at the start of a sequence the marker is treated as the artificial prefix space added during encoding. Decode each token in isolation and *every* word loses its leading space: you stream `Thecostofeachtoken`.
- **`clean_up_tokenization_spaces=True`** — the default on many tokenizers — post-processes the string, removing spaces before punctuation and around contractions. That is a *whole-string* transformation. Apply it per chunk and the result depends on where the chunk boundaries happen to fall, which means the same generation streamed in different chunk sizes produces different text. Two clients, same output ids, different strings on screen.

The rule: **inside a streaming server, always pass `skip_special_tokens` and `clean_up_tokenization_spaces` explicitly, and set the cleanup to `False`.** Do any prettification once, at the end, on the complete string, if you want it at all — and you probably do not, because it makes the streamed text and the final text differ.

### Failure 3: re-decoding everything is quadratic

The first fix people reach for is: keep all the ids, decode the whole list every step, and emit the difference.

```python
# Correct, but O(n^2).
prev = ""
for tid in generate(...):
    ids.append(tid)
    full = tok.decode(ids, clean_up_tokenization_spaces=False)
    yield full[len(prev):]
    prev = full
```

This is correct. It is also quadratic in output length, and the constant is not small — `decode` on a long list walks every token, builds a string, and allocates.

#### Worked example: the cost of full re-decode

Take a 2,000-token response. Full re-decode does

$$
\sum_{t=1}^{N} t = \frac{N(N+1)}{2}
$$

token-decodes, which for ${N = 2000}$ is 2,001,000. The incremental version does 2,000. That is a factor of 1,000 more work — **derived**, not measured — and it grows linearly with output length, so a 8,000-token reasoning trace is 4,000 times more work than the incremental path.

Multiply by concurrency and it stops being an abstraction. Sixty-four concurrent streams each doing a 2,000-token generation means the server performs 128 million token-decodes instead of 128 thousand, all on the CPU, all competing for the GIL with your engine's scheduling loop. This is a real mechanism by which a server that is fine at batch 4 falls over at batch 64 with the GPU sitting idle. Source for the arithmetic: the formula above. Source for what it costs on *your* box: run it, in section 10.

### The correct solution: a byte buffer plus a stable prefix

![Timeline of five decode steps in which the first two tokens emit text immediately, a byte token is held in a buffer, and the following token completes a three-byte sequence that is then emitted as one character](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-4.webp)

The clean formulation drops down to bytes. For a byte-level BPE tokenizer, detokenization *is* byte concatenation: every token id maps to a fixed byte string, and the output text is the UTF-8 decoding of their concatenation. Once you have that table, the incremental problem becomes the well-understood problem of **streaming UTF-8 decoding**: append bytes to a buffer, decode the longest prefix that forms complete code points, keep the remainder.

Building the byte table is the model-family-specific part:

```python
# nanoserve/tokenizer.py
import re
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

def build_byte_table(tok) -> list[bytes]:
    """Map every token id to the exact bytes it contributes.
    Handles byte-level BPE (GPT-2 / Llama-3 / tiktoken style) and
    SentencePiece byte-fallback vocabularies."""
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    table: list[bytes] = []
    for i in range(len(tok)):                # len(), not vocab_size
        piece = tok.convert_ids_to_tokens(i)
        if piece is None:
            table.append(b"")
            continue
        m = _BYTE_FALLBACK.match(piece)
        if m:                                # SentencePiece <0xNN>
            table.append(bytes([int(m.group(1), 16)]))
        elif all(c in u2b for c in piece):   # byte-level alias alphabet
            table.append(bytes(u2b[c] for c in piece))
        else:                                # SentencePiece text piece
            table.append(piece.replace("▁", " ").encode("utf-8"))
    return table
```

Two notes on that function. First, `len(tok)` and not `tok.vocab_size` — the added-token trap from section 2, and it matters here because the special ids are exactly the ones near the top of the range. Second, the byte-fallback branch must run *before* the alias branch, because `<0x41>` happens to consist entirely of characters that are also in the alias alphabet, and misclassifying it gives you the literal seven characters `<0x41>` in your output stream instead of the letter `A`. That is a bug you will find only by testing byte-fallback tokens explicitly, which is section 6.

Now the detokenizer. Python ships the hard part already — `codecs.getincrementaldecoder("utf-8")` is a stateful decoder that buffers incomplete sequences for you — but write it once by hand so you know what it is doing:

```python
# nanoserve/tokenizer.py
class IncrementalDetokenizer:
    """Emits only the newly-stable text each step.

    Invariants:
      * the returned string is always valid UTF-8
      * concatenating every return value equals the full decode
      * at most 3 bytes are ever held back
    """

    def __init__(self, byte_table: list[bytes], skip_ids: set[int] | None = None):
        self.table = byte_table
        self.skip = skip_ids or set()
        self.buf = bytearray()
        self.n_held_steps = 0          # observability, see section 10
        self.n_bad_bytes = 0

    def push(self, token_id: int) -> str:
        if token_id in self.skip:
            return ""
        self.buf += self.table[token_id]
        return self._drain()

    def _drain(self) -> str:
        n = len(self.buf)
        if n == 0:
            return ""
        # A code point is at most 4 bytes, so an INCOMPLETE tail is at
        # most 3 bytes. Four cuts therefore find the longest decodable
        # prefix, whatever the buffer length.
        for cut in range(n, max(-1, n - 4), -1):
            try:
                text = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            del self.buf[:cut]
            if cut != n:
                self.n_held_steps += 1
            return text
        # No cut decoded: the buffer STARTS with bytes that can never
        # form a character (a stray 0xff, a lone continuation byte).
        # Drop one and retry rather than wedging the stream forever.
        del self.buf[:1]
        self.n_bad_bytes += 1
        return "�" + self._drain()

    def flush(self) -> str:
        """End of stream: emit whatever is left, replacing bad bytes."""
        text = self.buf.decode("utf-8", errors="replace")
        self.buf.clear()
        return text
```

The `_drain` loop is the whole idea, and it is worth reading twice. A UTF-8 code point is at most four bytes, so an incomplete sequence at the end of the buffer is at most three bytes long. Trying cuts at `n`, `n-1`, `n-2`, `n-3` therefore finds the longest decodable prefix in at most four attempts, regardless of buffer size. **This is O(1) per token, not O(n).** No allocation proportional to output length, no re-decode, no GIL storm at concurrency.

The fallback below the loop is not decoration. If none of the four cuts decodes, the problem is not an incomplete *tail* but an invalid *head*: a stray `0xff`, or a continuation byte with nothing before it. That happens when a byte table is built wrong, when a model emits a byte token that cannot start a sequence, or when a request is resumed from a truncated state. Without the fallback, the buffer never drains and the stream silently stops producing text while the engine keeps burning GPU on it — a hang, not a crash, which is far worse to diagnose. Dropping one byte and retrying costs one replacement character and keeps the stream alive.

`flush()` exists because a generation can end mid-character: the model hit `max_tokens` in the middle of a three-byte sequence, or the user disconnected. At that point you have bytes that will never be completed, and holding them forever would truncate the response. `errors="replace"` is the honest ending — the character genuinely was cut off — and it is exactly one replacement character rather than three.

Using it in the decode loop is three lines:

```python
# nanoserve/engine.py (excerpt)
detok = IncrementalDetokenizer(byte_table, skip_ids=req.special_ids)
for tid in engine.step_stream(req):
    chunk = detok.push(tid)
    if chunk:
        yield chunk
yield detok.flush()
```

Note the `if chunk:` guard. Most steps produce text; some produce the empty string. Do not send an empty server-sent-event frame — some clients treat it as a keepalive, some as a terminator, and all of them waste a round trip.

### The portable alternative: an offset window

Building a byte table requires knowing your tokenizer family. If you want one code path across every tokenizer on the Hub, including ones with unusual piece encodings, there is a second approach that works purely through the public `convert_tokens_to_string` API. It is what vLLM does in [`detokenizer_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/detokenizer_utils.py): keep two indices into the token list, decode a small window twice, and use the *replacement character* as the signal that the text is not yet stable.

```python
def detokenize_incrementally(tok, all_ids, prefix_offset, read_offset):
    """Return (new_text, prefix_offset, read_offset).
    Decodes a small window, not the whole sequence."""
    prefix_text = tok.convert_tokens_to_string(
        tok.convert_ids_to_tokens(all_ids[prefix_offset:read_offset]))
    new_text = tok.convert_tokens_to_string(
        tok.convert_ids_to_tokens(all_ids[prefix_offset:]))
    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # Stable: advance both offsets and emit the delta.
        return new_text[len(prefix_text):], read_offset, len(all_ids)
    # Not stable yet (incomplete character): emit nothing, hold position.
    return "", prefix_offset, read_offset
```

The trick is subtle and worth spelling out. `prefix_offset` trails `read_offset` by a few tokens so that `convert_tokens_to_string` has enough left context to make correct spacing decisions — a piece's rendering can depend on whether it is at the start of the string, which is failure 2 above. Ending in U+FFFD means the window's last token was an incomplete character, so nothing new is emitted and the offsets stay put; the next token will complete it and the delta will appear then.

The cost is that it decodes a small window twice per token instead of appending bytes once. The benefit is that it needs no knowledge of the tokenizer family. Here is the comparison in full:

| Strategy | Valid UTF-8 | Correct spacing | Work per token | Needs family knowledge |
| --- | --- | --- | --- | --- |
| `decode([tid])` per token | no | no | O(1) | no |
| Full re-decode each step | yes | yes | O(n) | no |
| Offset window (vLLM style) | yes | yes | O(window) | no |
| Byte table + UTF-8 buffer | yes | yes | O(1) | yes |

For `nanoserve` we ship the byte-table version as the fast path and keep the offset-window version as a fallback selected at model load when the byte table cannot be built. That is roughly the structure every serious engine converges on.

## 5. Stop strings that span a token boundary

Now the failure mode that is genuinely irreversible, because it involves bytes that already left your process.

A request arrives with `stop: ["\n\nHuman:"]`. This is a completion-API idiom that is still everywhere: the client wants generation to end if the model starts hallucinating the next turn of a dialogue. Your job is to detect that string in the output and stop, without the string appearing in what the user receives.

The problem is that the model does not emit the string. It emits *tokens*, and the string is spread across three of them: a token for the two newlines, a token for `Human`, and a token for the colon. If you check "does this chunk equal a stop string" you never match. If you check "does the accumulated text contain a stop string" you match correctly — but by then you have already streamed `\n\nHuman` to the client, because you streamed each chunk as it arrived. The user watches the model start writing `Human:` and then the stream ends. Over HTTP with server-sent events there is no unsend.

![Two-row grid showing three tokens that together spell a stop string, with the client-visible row emitting nothing while the tail is a possible prefix and then finishing cleanly](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-5.webp)

The fix follows from one observation. If the longest stop string has length $L$ characters, then any occurrence of a stop string that has *begun* but not *finished* must start within the last ${L-1}$ characters of what you have accumulated. Everything before that is safe: no stop string starting there could still be incomplete. So the policy is: **release everything except the last ${L-1}$ characters.**

That is a hold-back buffer, and it is provably minimal — hold back fewer characters and there exists a stop string and a token split that leaks; hold back more and you add latency for nothing.

```python
# nanoserve/tokenizer.py
class StopChecker:
    """Detects stop strings that span token boundaries without
    leaking any part of them to the client.

    Holds back (max stop length - 1) characters. That is the minimum
    that is provably safe, and therefore the minimum added latency."""

    def __init__(self, stops: list[str], include_stop_str: bool = False):
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.include = include_stop_str
        self.pending = ""

    def push(self, text: str) -> tuple[str, bool]:
        """Returns (text_safe_to_emit, stopped)."""
        if not text:
            return "", False
        self.pending += text
        for s in self.stops:
            i = self.pending.find(s)
            if i >= 0:
                out = self.pending[: i + len(s)] if self.include \
                      else self.pending[:i]
                self.pending = ""
                return out, True
        if self.hold == 0:
            # No stops, or only 1-char stops: nothing to hold back.
            # This branch is NOT optional -- s[:-0] is the empty string
            # in Python, so falling into the slice below would swallow
            # the entire stream.
            out, self.pending = self.pending, ""
            return out, False
        if len(self.pending) > self.hold:
            out, self.pending = self.pending[:-self.hold], self.pending[-self.hold:]
            return out, False
        return "", False

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return out
```

Trace it against the figure. Suppose the stop string is `"\n\nHuman:"`, so ${L = 8}$ and `hold = 7`.

- The model emits `"...answer."`. No stop string is present; `pending` is longer than 7 characters, so everything but the last 7 goes out. Text flows normally.
- The model emits `"\n\n"`. `pending` is now the previous 7 held characters plus 2. Still longer than 7, so some older text is released — but the two newlines stay held.
- The model emits `"Human"`. `pending` ends with `"\n\nHuman"`, 7 characters. Nothing new is released.
- The model emits `":"`. `find` matches at some index; everything from that index onward is discarded, `stopped` is `True`, and the request finishes with `finish_reason: "stop"`.

The client never saw a single character of the stop string.

### The latency you pay

Hold-back is not free. With `hold = 7` the client's view of the stream trails the model by up to 7 characters — roughly two tokens of English at four characters per token. At a typical inter-token latency, that is on the order of tens of milliseconds of *perceived* extra delay before each visible update, and it makes the stream burstier: nothing, nothing, then a chunk.

The trade is explicit and worth stating as a rule:

| Stop configuration | Hold-back | Streaming feel |
| --- | --- | --- |
| No stop strings (EOS only) | 0 chars | perfectly smooth |
| Short stops, e.g. `"\n\n"` | 1 char | indistinguishable |
| `"\n\nHuman:"` | 7 chars | slightly bursty |
| A 60-char sentinel | 59 chars | visibly laggy |

The operational conclusion: **cap the length of client-supplied stop strings**, and cap their count. A request that supplies a 500-character stop string is asking you to buffer 499 characters before showing anything, and a request that supplies 200 stop strings turns your per-token `find` loop into a real cost. Both are trivially defensible with a validation check at the API layer — 4 stop strings of at most 32 characters is a limit nobody legitimate will notice.

One more subtlety: the `find` above scans all of `pending` every token. `pending` is bounded by `hold` plus one chunk, so it is small, and this is fine. If you ever change the code to accumulate the full output in `pending`, that scan becomes O(n) per token and you have reinvented the quadratic bug from section 4. Keep `pending` bounded.

### Ordering: detokenize, then stop-check

The two components compose in exactly one order:

```python
detok = IncrementalDetokenizer(byte_table, skip_ids=special_ids)
stopper = StopChecker(req.stop)
stopped = False
for tid in engine.step_stream(req):
    if tid in eos_ids:
        break
    text, stopped = stopper.push(detok.push(tid))
    if text:
        yield text
    if stopped:
        break
if not stopped:
    # End of stream. Bytes still in the detokenizer belong BEFORE the
    # characters the stopper is holding, so flush in that order --
    # and route the tail through the stopper, because the stop string
    # may complete on the very last byte.
    tail, stopped = stopper.push(detok.flush())
    if tail:
        yield tail
    if not stopped:
        yield stopper.flush()
```

Detokenize first, because stop strings are defined over *characters*, not tokens or bytes. Trying to match stop strings against token ids is a category error that fails the moment the tokenizer splits the string differently than you expected — which is the entire problem.

The flush ordering at the end is not cosmetic either. The detokenizer's held bytes were produced *before* the characters sitting in the stopper's `pending`, so `detok.flush()` must go through `stopper.push()` rather than being concatenated after `stopper.flush()`. Get that backwards and the last few characters of every truncated response come out in the wrong order — a bug that only shows up on generations that hit `max_tokens` mid-character, which is exactly the traffic nobody tests.

## 6. Testing the boundary: the cases that actually break

The nice property of this component is that it is pure, deterministic, and needs no GPU. You can test every nasty case in milliseconds, in CI, on a laptop. There is no excuse for not having these tests, and I have never seen a team that had them before their first incident.

Build a synthetic byte table so the tests do not depend on downloading a model:

```python
# tests/test_detok.py
import pytest
from nanoserve.tokenizer import IncrementalDetokenizer, StopChecker

# A fake vocabulary. Ids 2 and 3 split the three UTF-8 bytes of
# U+20AC; ids 4 and 5 split the four bytes of U+1F680.
FAKE = [
    b"cost",        # 0
    b" ",           # 1
    b"\xe2",        # 2  first byte of a 3-byte sequence
    b"\x82\xac",    # 3  remaining two bytes
    b"\xf0\x9f",    # 4  first half of a 4-byte sequence
    b"\x9a\x80",    # 5  second half
    b" each",       # 6
    b"\xff",        # 7  never valid UTF-8, anywhere
]

def test_multibyte_split_across_two_tokens():
    d = IncrementalDetokenizer(FAKE)
    assert d.push(0) == "cost"
    assert d.push(1) == " "
    assert d.push(2) == ""            # held, not emitted
    assert d.push(3) == "€"      # completed, emitted whole
    assert d.push(6) == " each"
    assert d.flush() == ""

def test_four_byte_split():
    d = IncrementalDetokenizer(FAKE)
    assert d.push(4) == ""
    assert d.push(5) == "\U0001f680"

def test_never_emits_replacement_mid_stream():
    d = IncrementalDetokenizer(FAKE)
    out = "".join(d.push(t) for t in [0, 1, 2, 3, 4, 5, 6])
    assert "�" not in out

def test_concat_equals_full_decode():
    ids = [0, 1, 2, 3, 4, 5, 6]
    d = IncrementalDetokenizer(FAKE)
    streamed = "".join(d.push(t) for t in ids) + d.flush()
    whole = b"".join(FAKE[t] for t in ids).decode("utf-8")
    assert streamed == whole      # the invariant that matters

def test_truncated_at_max_tokens():
    d = IncrementalDetokenizer(FAKE)
    d.push(0); d.push(2)          # ends mid-character
    assert d.flush() == "�"  # exactly one, at the very end

def test_invalid_byte_does_not_wedge_the_stream():
    d = IncrementalDetokenizer(FAKE)
    d.push(7)                     # 0xff can never start a sequence
    assert d.push(0) != ""        # stream keeps flowing
```

The fourth test is the important one. `test_concat_equals_full_decode` is the *specification*: streaming must produce exactly the string that a batch decode would. Every other test is a special case of it. Property-test it if you can — generate random id sequences from your real byte table and assert the invariant — and you will find the edge cases you did not think of.

Then the stop-string cases:

```python
def test_stop_string_spanning_three_tokens():
    s = StopChecker(["\n\nHuman:"])
    emitted = []
    for chunk in ["Here is the answer.", "\n\n", "Human", ":"]:
        text, stopped = s.push(chunk)
        emitted.append(text)
        if stopped:
            break
    out = "".join(emitted)
    assert "Human" not in out
    assert "\n\n" not in out
    assert out.startswith("Here is the answer")

def test_stop_string_inside_one_token():
    s = StopChecker(["END"])
    text, stopped = s.push("all done END now")
    assert stopped and text == "all done "

def test_no_stops_means_no_holdback():
    s = StopChecker([])
    text, stopped = s.push("stream me")
    assert text == "stream me" and not stopped

def test_single_char_stop_holds_nothing():
    # hold == 0 here. Guards the s[:-0] trap.
    s = StopChecker(["#"])
    assert s.push("abc") == ("abc", False)
    assert s.push("d#e") == ("d", True)

def test_near_miss_is_eventually_released():
    s = StopChecker(["\n\nHuman:"])
    s.push("text\n\nHum")            # held: looks like a prefix
    text, stopped = s.push("ans are great")
    assert not stopped
    assert "Hum" in text             # released once disambiguated
```

The last one catches the failure mode where a too-eager implementation drops held text on a near miss. `"\n\nHumans are great"` is not a stop string, and the client must eventually see every character of it.

Finally, a round-trip test against the *real* tokenizer, marked slow, run nightly:

```python
@pytest.mark.slow
def test_real_tokenizer_roundtrip():
    from transformers import AutoTokenizer
    from nanoserve.tokenizer import build_byte_table
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    table = build_byte_table(tok)
    corpus = load_fixtures()   # prose, code, math, many scripts, emoji
    for text in corpus:
        ids = tok.encode(text, add_special_tokens=False)
        d = IncrementalDetokenizer(table)
        streamed = "".join(d.push(i) for i in ids) + d.flush()
        assert streamed == tok.decode(
            ids, clean_up_tokenization_spaces=False), text[:40]
```

Put fixtures in that corpus that your product will actually see: a currency symbol, a name with diacritics, a code block with tabs, a paragraph in each language you support, a table, a URL, an emoji. Twenty fixtures, one assertion, and an entire category of production incident is closed.

## 7. Offsets, spans, and why highlighting is harder than it looks

Sooner or later a product requirement arrives that needs to map *tokens* back to *characters*. Three of them are common:

- **Logprob display.** The API returns a per-token logprob, and the UI wants to shade the text by confidence. That needs a character span per token.
- **Citation and grounding.** A RAG answer claims a span of a source document; you highlight it. That needs character offsets into the *prompt*.
- **Streaming syntax highlighting.** The client wants to tokenize the partial output as code. That needs stable character positions.

Fast tokenizers offer `return_offsets_mapping=True`, which gives you a `(start, end)` character span per token:

```python
enc = tok("The cost of each token", return_offsets_mapping=True,
          add_special_tokens=False)
for tid, (a, b) in zip(enc.input_ids, enc.offset_mapping):
    print(tid, repr("The cost of each token"[a:b]))
```

This works well for the encode direction on ordinary text. It has three limitations you must design around.

**It is encode-only.** Offsets come from the encoder's record of where each token came from in the input string. There is no such record for generated tokens — they were never in an input string. To get spans for *output* tokens you must build them yourself while detokenizing, which the byte-table design makes easy: track the running byte length and code-point length as you drain.

```python
class SpanTrackingDetokenizer(IncrementalDetokenizer):
    """Adds a per-token character span to the base detokenizer.
    A token that completes a character owns that character; a token
    that only contributed bytes gets a zero-width span."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.char_pos = 0
        self.spans: list[tuple[int, int]] = []

    def push(self, token_id: int) -> str:
        text = super().push(token_id)
        start = self.char_pos
        self.char_pos += len(text)
        self.spans.append((start, self.char_pos))
        return text
```

Read what that gives you honestly: for a character split across two tokens, the first token gets a zero-width span and the second gets the whole character. There is no honest alternative — the character does not belong to either token individually. A confidence-shading UI must therefore handle zero-width spans, and a naive implementation that assumes every token covers at least one character will produce off-by-one highlighting exactly on the non-ASCII characters. This is the same class of bug as failure 1, wearing a different hat.

**Offset mapping is unreliable across byte-fallback tokens.** When a character is emitted as several byte tokens, the encoder has to attribute character positions to tokens that individually correspond to *part* of a character. Different tokenizer implementations resolve this differently: some assign the full character span to every byte token (so spans overlap), some assign it to the first (so the rest are zero-width), some produce spans that do not tile the input at all. None of these are wrong exactly — the question has no clean answer — but it means you cannot assume offsets are non-overlapping, contiguous, or monotonic in width. Validate before you trust:

```python
def offsets_are_clean(enc, text) -> bool:
    prev_end = 0
    for a, b in enc.offset_mapping:
        if a < prev_end or b < a or b > len(text):
            return False
        prev_end = b
    return True
```

If that returns `False` on your corpus, your highlighting feature needs the byte-level path instead of the character-level one.

**Normalization can move the text.** Some tokenizers apply a normalizer (NFKC, lowercasing, whitespace collapsing) before pretokenization. Offsets are then relative to the *normalized* string, not the string your user sent, and the two have different lengths. If your tokenizer has a non-trivial normalizer, offsets into the original text require inverting a mapping that may not be invertible. Check `tok.backend_tokenizer.normalizer`; if it is not `None`, verify your assumption on real data before building a product on it.

The design rule that falls out of all three: **track byte offsets internally, convert to character offsets only at the API edge.** Bytes are unambiguous, they tile exactly, and they compose with the byte-table detokenizer for free. Characters are what the UI needs, but the conversion is a single well-defined operation at the boundary rather than an assumption spread through your code.

## 8. Tokenizer efficiency is an economic variable

Now for the part that reaches outside engineering.

Everyone knows tokens cost money. Fewer people internalize that **the number of tokens in a document is a property of the tokenizer, not of the document.** The same paragraph, priced at the same dollars per million tokens on two different models, costs different amounts. And the same 128k context window holds a different amount of *text* depending on the language.

![Tree showing how the same kilobyte of text produces different token counts depending on whether the training corpus taught merges for that script, ending in byte fallback for rare scripts](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-6.webp)

### The mechanism

The figure traces the causal chain. BPE learns merges from a training corpus. A merge exists for a byte pair only if that pair was frequent in that corpus. So:

- Text in the **dominant language of the tokenizer's corpus** gets long merges. Common English words are a single token; very common ones with a leading space are a single token including the space.
- Text in a **script that was rare in the corpus** has few or no merges covering it. In the limit, the tokenizer falls back to emitting one token per byte.
- Because non-Latin scripts and accented Latin characters take **two to three UTF-8 bytes per character**, byte fallback for those scripts costs two to three tokens per *character*.

That gives a hard, derivable bound. Let $c$ be the average UTF-8 bytes per character for a script. Then tokens per character lies in

$$
\frac{1}{M} \le \frac{\text{tokens}}{\text{character}} \le c
$$

where $M$ is the length in characters of the longest merge that applies. English prose with a well-fitted tokenizer sits near the bottom of its range; a script the tokenizer never saw sits at the top of a range whose ceiling is three times higher to begin with. **That is the whole mechanism**, and it is why the effect is large rather than marginal.

For calibration on the English end, OpenAI's documentation offers a widely used rule of thumb: for common English text, one token is roughly four characters, or about three-quarters of a word ([What are tokens and how to count them](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)). That is a *cited* approximation for English on a GPT-family tokenizer, not a universal constant, and it is exactly the number people mistakenly apply to every language.

The direction for other content is unambiguous even without a measurement:

| Content type | Direction vs English prose | Why |
| --- | --- | --- |
| English prose | baseline | matches the corpus; long merges |
| Source code | more tokens | indentation, punctuation runs, identifiers split |
| Accented Latin (e.g. Vietnamese) | more tokens | 2-3 bytes per character, few merges |
| Chinese, Japanese, Thai | more tokens | 3 bytes per character, no word-boundary merges |
| Base64 or hashes | far more tokens | maximum entropy, no structure to merge |
| Repeated boilerplate | fewer tokens | long merges cover it |

Note what is *not* in that table: numbers. I am not going to tell you that Vietnamese costs 2.3x English on Llama 3.1, because I have not measured it and neither has anyone whose measurement I can point you to for your corpus. What I will give you is the script.

### Measure it yourself

```python
# tools/token_cost.py
"""Measure tokens-per-byte for your own corpus on any tokenizer.

    python tools/token_cost.py meta-llama/Llama-3.1-8B-Instruct \
        corpus/en.txt corpus/code.py corpus/docs.md
"""
import sys, os
from transformers import AutoTokenizer

def profile(model_id: str, paths: list[str]):
    tok = AutoTokenizer.from_pretrained(model_id)
    rows = []
    for p in paths:
        text = open(p, encoding="utf-8").read()
        n_ids = len(tok(text, add_special_tokens=False).input_ids)
        n_bytes = len(text.encode("utf-8"))
        n_chars = len(text)
        rows.append({
            "file": os.path.basename(p),
            "chars": n_chars,
            "bytes": n_bytes,
            "tokens": n_ids,
            "bytes_per_token": n_bytes / n_ids,
            "chars_per_token": n_chars / n_ids,
            "tokens_per_1k_chars": 1000 * n_ids / n_chars,
        })
    return rows

if __name__ == "__main__":
    model_id, paths = sys.argv[1], sys.argv[2:]
    print(f"{'file':<20}{'tokens':>9}{'B/tok':>8}{'ch/tok':>8}{'tok/1kch':>10}")
    for r in profile(model_id, paths):
        print(f"{r['file']:<20}{r['tokens']:>9}"
              f"{r['bytes_per_token']:>8.2f}{r['chars_per_token']:>8.2f}"
              f"{r['tokens_per_1k_chars']:>10.1f}")
```

**Report `bytes_per_token`, not `chars_per_token`, when you compare across scripts.** Characters-per-token flatters scripts with information-dense characters: one Chinese character carries roughly as much meaning as an English word, so "1.2 characters per token" sounds bad and is not. Bytes-per-token has a hard floor of 1.0 — the pure byte-fallback case, derived from the fact that every byte has a token — and that floor makes it a fair yardstick. A tokenizer that gets 4.0 bytes per token on your corpus is doing four times better than the worst case; a tokenizer that gets 1.3 is barely doing anything.

Run the script across the model candidates you are considering, on *your* documents, and put the table in your model-selection doc. It takes ten minutes and it is the only version of this number that is true for you.

#### Worked example: the same document, two prices

Suppose your corpus profiles at ${B_1 = 3.9}$ bytes per token on model A and ${B_2 = 2.6}$ bytes per token on model B — a difference well within the range you should expect between a tokenizer fitted to your content and one that is not. A 40 KB document (40,960 bytes) then costs

$$
n_A = \frac{40960}{3.9} \approx 10500 \text{ tokens}, \qquad
n_B = \frac{40960}{2.6} \approx 15750 \text{ tokens}
$$

At an assumed price of \$0.50 per million input tokens for both models, that is \$0.00525 versus \$0.00787 per document — 50% more on model B **at the same advertised price**. Across a million documents a month, \$5,250 versus \$7,875. The arithmetic is derived; the two bytes-per-token figures are placeholders you must replace with output from `token_cost.py` on your data.

The lesson for procurement: **dollars per million tokens is not a comparable price** across models. The comparable quantity is dollars per million *bytes of your content*, which is dollars-per-token divided by bytes-per-token. Publish that number internally and the model-selection conversation gets much shorter.

#### Worked example: what a 128k window actually holds

The same mechanism reshapes the context window. Take a 128,000-token budget.

- At 3.9 bytes per token, that is 499,200 bytes. For ASCII English, bytes equal characters, so about 499,000 characters. At roughly 5.5 characters per word plus a space, that is about 77,000 words.
- At 2.6 bytes per token, that is 332,800 bytes. If the script averages 3 bytes per character, that is about 111,000 characters — and if the script's characters carry roughly a word each, the *semantic* content may still be large.
- In the byte-fallback limit of 1.0 bytes per token, 128,000 tokens is 128,000 bytes: about 42,600 characters of a three-byte script.

All three lines are derived from the bytes-per-token figure and the script's bytes-per-character; only the bytes-per-token inputs need measuring. The takeaway is not "context windows are smaller in other languages" as a slogan — it is that **your chunking and retrieval budgets are tokenizer-dependent constants and must be computed per model**, not hardcoded once in a config file and shared across deployments.

## 9. Vocabulary size and the lm_head you pay for

There is a reason vocabulary size is not simply set as large as possible, and it lives in the last layer of the model.

The output projection — the `lm_head` — maps the hidden state to one logit per vocabulary entry. It is a matrix of shape $d_{\text{model}} \times V$, and its memory in bytes is

$$
M_{\text{head}} = d_{\text{model}} \cdot V \cdot b
$$

where $b$ is bytes per parameter. For an untied model, the input embedding table has the same shape, so the pair costs $2 M_{\text{head}}$.

#### Worked example: 128k vocabulary versus 32k, on Llama-3.1-8B

Llama-3.1-8B has ${d_{\text{model}} = 4096}$ and ${V = 128256}$ (128,000 base BPE entries plus 256 reserved and special entries), with untied embeddings. So:

$$
M_{\text{head}} = 4096 \times 128256 \times 2 = 1{,}050{,}673{,}152 \text{ bytes} \approx 1.05 \text{ GB}
$$

That is 525 million parameters in the output projection, and another 525 million in the input embedding: **1.05 billion of the model's roughly 8.03 billion parameters, about 13%, are the vocabulary interface.** In bf16 the whole model is about 16.06 GB, of which 2.1 GB is embeddings and lm_head.

With a 32,000-entry vocabulary instead:

$$
M_{\text{head}} = 4096 \times 32000 \times 2 = 262{,}144{,}000 \text{ bytes} \approx 262 \text{ MB}
$$

a saving of about 788 MB on the head and the same again on the embedding table.

Now the interesting part: what does that buy at decode time? During decode the engine reads essentially every weight once per step, so a decode step's lower bound is

$$
t_{\text{step}} \approx \frac{W}{B_{\text{HBM}}}
$$

with $W$ the bytes actually read and $B_{\text{HBM}}$ the memory bandwidth. Crucially, the *input embedding table* is not fully read — decode looks up one row — while the *lm_head* is read in full to produce logits. So the decode-relevant weight bytes are about ${16.06 - 1.05 = 15.01}$ GB for the 128k model, and ${15.01 - 1.05 + 0.26 = 14.22}$ GB for a hypothetical 32k version.

NVIDIA lists 2,039 GB/s of HBM2e bandwidth for the A100 80GB SXM ([A100 product page](https://www.nvidia.com/en-us/data-center/a100/)). Then:

| Configuration | Decode bytes read | Lower-bound per step | Source |
| --- | --- | --- | --- |
| 128k vocab | 15.01 GB | 7.36 ms | derived |
| 32k vocab | 14.22 GB | 6.98 ms | derived |
| Difference | 0.79 GB | 0.38 ms, 5.2% | derived |

So the fat vocabulary makes every decode step about 5% slower. But Meta reports that Llama 3's 128k-token vocabulary "encodes language much more efficiently", yielding **up to 15% fewer tokens** than Llama 2's 32k SentencePiece vocabulary ([Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)). If a given piece of output text needs 15% fewer tokens, it needs 15% fewer decode steps. For the same *text*:

$$
T_{128k} = 0.85 N \times 7.36 = 6.26 N \text{ ms}, \qquad
T_{32k} = N \times 6.98 = 6.98 N \text{ ms}
$$

The larger vocabulary wins by about 10% on wall-clock time for the same output text, despite each step being slower. That is the trade in one line: **a bigger vocabulary makes each step slower and takes fewer steps, and the second effect usually dominates** — which is why the industry moved from 32k to 128k and beyond. (The 15% figure is Meta's, cited; the step times are derived from the bandwidth arithmetic above; combining them assumes the token reduction applies to output as well as input, which is an assumption I am flagging rather than hiding.)

There are two more costs of a large vocabulary that the bandwidth math misses.

**The logits tensor.** At decode you materialize a `[batch, V]` tensor. In fp32 with a batch of 64 and ${V = 128256}$ that is

$$
64 \times 128256 \times 4 = 32{,}833{,}536 \text{ bytes} \approx 32.8 \text{ MB}
$$

per step, allocated and freed every step. Fine. But if you ever compute logits for *all* prefill positions — which the naive implementation does, because that is what `transformers` returns by default — an 8,192-token prefill costs

$$
8192 \times 128256 \times 4 = 4{,}202{,}692{,}608 \text{ bytes} \approx 4.2 \text{ GB}
$$

in one allocation. That is a genuine and common source of out-of-memory errors on a 24 GB card, and the fix is to slice the hidden states to the last position *before* the lm_head, not after. Every real engine does this; make sure yours does.

**The sampling step.** Top-k, top-p and repetition penalties all operate over the full logit vector. Sorting 128,256 floats per sequence per step is four times the work of sorting 32,000. At batch 1 this is lost in the noise; at batch 128 on the CPU it is not, which is one of the reasons a fused GPU sampler matters. That is a later post in this series; the point here is that vocabulary size is one of its inputs.

The larger design lesson is that vocabulary size is a **cross-cutting** parameter. It sets your token bill, your effective context length, 13% of your weight memory, your logits allocation, and your sampling cost — and it is fixed at pretraining time, so by the time you are serving, it is a constraint rather than a knob. Knowing its consequences is what lets you predict them.

## 10. Where the tokenizer costs you latency

Everything in this post runs on the host CPU. That is the single most important operational fact about the tokenizer boundary, and it is why tokenizer cost is so often diagnosed as something else.

![Comparison table of five tokenizer cost sites showing where each runs, the condition under which it becomes significant, and the corresponding fix direction](/imgs/blogs/the-tokenizer-boundary-and-incremental-detokenization-7.webp)

The figure lays out the five sites. Read the middle column as the trigger condition — the load pattern under which each one stops being free.

### Encode, on the request path

Encoding happens before the request can be scheduled, so it is pure TTFT. For a short chat turn it is negligible. For a RAG request with 8,000 tokens of retrieved context it is a real, measurable slice of your time-to-first-token, and it is *serialized* ahead of prefill.

The `tokenizers` library is Rust and releases the GIL, so it parallelizes across threads better than most Python work — but only if you actually call it from a thread pool rather than inline in your async event loop. Encoding a long prompt inline in an `asyncio` handler blocks every other request on that loop for the duration. Push it to an executor:

```python
import asyncio
encode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

async def encode_async(builder, messages):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        encode_pool, builder.encode_chat, messages)
```

There is a second, larger win available: **encoding overlaps with nothing today, but it could overlap with the previous request's prefill.** If your engine has a queue, encode on admission rather than on dequeue, and the cost disappears into the queue wait that already existed.

### Detokenize, once per token per stream

This is the one that scales badly, because it happens once per token per *live stream*.

#### Worked example: the CPU budget for detokenization

Take 64 concurrent streams, each producing 50 tokens per second. That is

$$
64 \times 50 = 3200 \text{ detokenize calls per second}
$$

Now suppose each call costs $\tau$ microseconds of CPU. The fraction of one core consumed is ${3200 \tau \times 10^{-6}}$:

| Per-call cost | CPU consumed | Verdict | Source |
| --- | --- | --- | --- |
| 10 microseconds | 3.2% of a core | invisible | derived |
| 100 microseconds | 32% of a core | noticeable under GIL | derived |
| 300 microseconds | 96% of a core | a full core, GIL-bound | derived |
| Your value | run the script below | — | reproduce |

The incremental byte-buffer implementation should land near the bottom of that range: it appends a few bytes and tries at most four `decode` calls on a buffer of at most four bytes. The full re-decode implementation lands near the top and gets worse as sequences lengthen, because its per-call cost is proportional to sequence length — which is precisely the quadratic behavior from section 4 showing up as a CPU bill.

The reason a full core matters more than it sounds: in CPython, the parts of this that are pure Python hold the GIL, and your engine's scheduling loop is also Python. A detokenizer consuming a core is a detokenizer stealing scheduling time, and the symptom is a GPU that goes idle between steps for no visible reason. If you are profiling with [Nsight Systems](/blog/machine-learning/performance-engineering/nsight-systems-for-ai-services), this appears as gaps on the CUDA stream with nothing in flight — a CPU-side stall that looks like a GPU problem.

Measure yours honestly:

```python
# bench/bench_detok.py
import time, statistics
from transformers import AutoTokenizer
from nanoserve.tokenizer import build_byte_table, IncrementalDetokenizer

def bench(model_id: str, n_tokens: int = 2000, reps: int = 5):
    tok = AutoTokenizer.from_pretrained(model_id)
    table = build_byte_table(tok)
    ids = tok.encode(open("corpus/sample.txt").read(),
                     add_special_tokens=False)[:n_tokens]
    # Warm up: first call pays import and cache costs.
    d = IncrementalDetokenizer(table)
    for i in ids[:200]:
        d.push(i)
    times = []
    for _ in range(reps):
        d = IncrementalDetokenizer(table)
        t0 = time.perf_counter()
        for i in ids:
            d.push(i)
        times.append((time.perf_counter() - t0) / len(ids) * 1e6)
    return statistics.median(times)   # microseconds per token
```

Report the median of several repetitions, discard the first, and pin the process to a core if your machine has aggressive frequency scaling — the same discipline as any other microbenchmark, laid out in [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark). Then run the same benchmark against the naive full-re-decode implementation and confirm the ratio matches the ${N/2}$ prediction from section 4. If it does not, your byte table is wrong.

### Instrument it in production

Two counters make this whole layer observable, and they cost nothing:

```python
# Per request, on completion:
metrics.histogram("tok.encode_ms", encode_ms)
metrics.histogram("tok.detok_us_per_token", detok_us / n_out)
metrics.counter("tok.held_steps", detok.n_held_steps)
metrics.counter("tok.stop_holdback_chars", stopper.hold)
metrics.counter("tok.special_token_rejected", n_rejected)
```

`tok.held_steps` is the interesting one. It counts how often the detokenizer withheld output because a character was incomplete — which is a direct measure of how much non-ASCII content your traffic contains. If it is near zero, your byte-fallback code path has never run in production and is therefore untested by anything except your unit tests. If it is high, you have a large non-English user base and every claim in section 8 applies to your bill.

`tok.special_token_rejected` is a security signal. A sustained nonzero rate means someone is probing your endpoint with control markers. That belongs on a dashboard.

### The fix directions this series returns to

Three of the five costs in the figure have engine-level fixes that later posts develop:

- **Encode overlapping with scheduling** — do it on admission, in a thread pool, so it hides inside queue time. This connects to [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption).
- **Detokenization off the critical path** — batch the per-token detokenize work for all streams into one call per engine step, and run it on a separate thread from the scheduler. The API-server post in Track I builds this.
- **A fused GPU sampler that removes a host sync** — the reason your Python loop can even see the token id is a device-to-host copy. Sampling on the GPU and copying a small batch of ids once per step changes the shape of this whole layer. Track E covers it.

None of those are premature optimizations you should do today. Get correctness first: byte buffer, hold-back, tests. The performance work is worth doing when `tok.detok_us_per_token` says so, and not before.

## Case studies and real numbers

**vLLM's incremental detokenizer.** The offset-window algorithm in section 4 is a simplified version of what vLLM ships in [`detokenizer_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/detokenizer_utils.py). Two details of the production version are worth copying. First, it seeds `prefix_offset` a fixed number of tokens back from the end rather than at zero, which bounds the window regardless of sequence length — the O(1)-per-token property, obtained without a byte table. Second, it uses `new_text.endswith("�")` as the "not stable yet" test, which is a neat trick: rather than reasoning about UTF-8 byte structure, it asks the tokenizer to render and checks whether the renderer gave up. That works across tokenizer families where byte-level reasoning would need family-specific code.

**Llama 3's vocabulary change.** Meta's [Llama 3 announcement](https://ai.meta.com/blog/meta-llama-3/) states that the 128k-token vocabulary "encodes language much more efficiently" and cites up to 15% fewer tokens versus Llama 2. That is the cited half of the worked example in section 9; the memory and bandwidth arithmetic there is the derived half. Both halves are needed to conclude anything: the token reduction alone would suggest a bigger vocabulary is free, and the memory cost alone would suggest it is expensive.

**tiktoken's design.** OpenAI's [tiktoken](https://github.com/openai/tiktoken) is a byte-level BPE implementation in Rust with a deliberately small API surface, and it is worth reading precisely because it does *not* try to be a general tokenizer framework. `encode_ordinary` skips special-token scanning entirely, and `encode` takes explicit `allowed_special` and `disallowed_special` sets — with the default being to *raise* if disallowed special text appears in the input. That default is the right one and it is the opposite of what most HuggingFace code paths do. If you are designing an encode API for an engine, copy that: **make handling of special tokens in user text an explicit, required decision, not a default.**

**The chat template as an interoperability layer.** HuggingFace's [chat templating documentation](https://huggingface.co/docs/transformers/main/en/chat_templating) is the closest thing to a standard for this. The important design decision it records is that the template ships *with the checkpoint*, as data, rather than being code in a library — because the format is a property of the model's post-training, and only the people who did the post-training know it. Every engine that hardcodes a prompt format for a model family is making a bet that will eventually be wrong.

## When to reach for this (and when not to)

**Write your own incremental detokenizer when:** you are building an engine (this series' whole premise); you need byte-exact spans for a citation or highlighting feature; you serve non-Latin scripts at meaningful volume and need the byte-fallback path to be correct and fast; or your profile shows detokenization consuming real CPU.

**Do not write your own when:** you are calling vLLM, SGLang, or TGI through their APIs. They have already solved this, they have been beaten on by more traffic than you will send, and re-solving it is pure risk. Read their implementation, copy the ideas, do not fork the code.

**The middle case is the common one.** You are using a framework for the engine but writing your own API layer on top. In that situation you inherit correct detokenization but you *do not* inherit stop-string hold-back semantics, chat-template rendering, or special-token sanitization at your edge — because those depend on your API contract, not on the engine's. Section 2 and section 5 are yours to own even when section 4 is not.

**And regardless of what you use:** run the round-trip test from section 6 against your actual stack, with fixtures in every script you serve. It takes an hour. It is the single highest-value hour in this post.

## Key takeaways

1. **The tokenizer is a boundary of your engine, not preprocessing.** Everything inside is integers; both conversions are yours to get right.
2. **Never let user text encode into a control marker.** Sanitize message content before rendering, then assert that no special id survives encoding. A blacklist plus an assertion, because the blacklist will rot.
3. **Use `apply_chat_template`, always.** A hand-built prompt is off-distribution and degrades quality with no error, no warning, and no metric.
4. **Never decode tokens one at a time and concatenate.** Buffer bytes until they form complete UTF-8, and emit only the newly stable prefix — at most three bytes are ever held.
5. **Never re-decode the full sequence each step.** It is O(n²) in output length, which is a factor of 1,000 at 2,000 tokens and shows up as a saturated CPU core at concurrency.
6. **Hold back `max_stop_len - 1` characters.** It is the provably minimal buffer that prevents leaking a stop string that spans tokens, and it is why you should cap client-supplied stop-string length.
7. **Set `clean_up_tokenization_spaces=False` in a server.** Whole-string prettification applied per chunk makes streamed text depend on chunk boundaries.
8. **Track byte offsets internally, convert to characters at the API edge.** Offset mapping is encode-only and unreliable across byte-fallback tokens.
9. **Tokens per byte is a property of the tokenizer, not the document.** Dollars per million tokens is not a comparable price across models; dollars per million bytes of *your* content is.
10. **A bigger vocabulary makes each decode step slower and takes fewer of them** — for Llama-3.1-8B, about 13% of the parameters are the vocabulary interface, and the token reduction usually wins.
11. **All of it runs on the CPU**, which is why tokenizer cost is so often misdiagnosed as a GPU problem. Instrument `detok_us_per_token` and `held_steps` and you will see it.

## Further reading

- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) — the series intro and the layer map this post sits in.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that assembles `nanoserve` and benchmarks it honestly.
- [How BPE tokenizers work](/blog/machine-learning/large-language-model/bpe-tokenizer) — the training-time algorithm this post deliberately does not re-derive.
- [Designing and choosing a tokenizer for an LLM](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) — vocabulary sizing and corpus fit, from the pretraining side.
- [Prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) — what you do with a shared id prefix once the tokenizer gives you a stable one.
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark) — the measurement discipline behind `bench_detok.py`.
- [vLLM `detokenizer_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/detokenizer_utils.py) — the production offset-window detokenizer.
- [HuggingFace chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating) and [tiktoken](https://github.com/openai/tiktoken) — the two reference implementations of the encode side.
- [RFC 3629](https://www.rfc-editor.org/rfc/rfc3629) — UTF-8, the specification the whole streaming problem reduces to.
