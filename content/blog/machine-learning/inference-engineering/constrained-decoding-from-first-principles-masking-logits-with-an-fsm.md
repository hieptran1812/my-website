---
title: "Constrained decoding from first principles: masking logits with an FSM"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Build JSON mode by hand - compile a schema into a state machine, precompute which vocabulary tokens each state allows, and set every other logit to negative infinity so malformed output stops being possible rather than merely unlikely."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "decoding",
    "structured-output",
    "constrained-decoding",
    "json-schema",
    "tokenizer",
    "pytorch",
    "latency",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 58
---

There is a specific kind of production bug that makes engineers lose faith in language models. The endpoint is supposed to return a small JSON object — a classification label, a confidence, two extracted fields. It returns one, correctly, roughly all of the time. Then a request comes in at 3 a.m. where the model decides to be helpful, prefixes the object with `Sure! Here is the JSON you asked for:`, wraps it in a fenced code block, and adds a friendly sentence afterwards. `json.loads` throws. The retry succeeds. Nobody notices except the p99 latency chart, which now has a second hump.

The usual fix is a loop: parse, catch, re-prompt, try again, give up after three attempts. That loop is an admission that the system has no guarantee. It converts a correctness problem into a latency-and-cost problem, and it never quite goes away, because "the model usually complies" is a statement about a distribution, not about a contract.

There is a completely different answer, and it is the subject of this post. At every decode step, before the sampler runs, compute the set of tokens that could legally continue the output under a grammar, set every other logit to negative infinity, and sample as usual. The illegal token is not merely unlikely — it has probability exactly zero, because it is not in the support of the distribution any more. Malformed output becomes *impossible by construction*. That is a categorically stronger guarantee than asking nicely and retrying, and it costs you two things you will learn to measure precisely by the end of this post.

![Two ways of getting JSON out of a model shown side by side, with a free decode that must be parsed and retried on the left and a masked decode that cannot produce an unparseable string on the right](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-1.webp)

This is post 18 of the [Inference Engineering series](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and the third in Track D, the decoding layer. The previous post built `nanoserve/sampling.py` — the [sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) of temperature, top-k, top-p, min-p and the penalties. This post writes `nanoserve/grammar/`, which sits *in front* of all of them: a regex-to-DFA compiler, a token-level index, and a `GrammarLogitsProcessor` that plugs into the same chain. By the end you will have built JSON mode from nothing, and you will be able to answer the three questions that actually decide whether you can turn it on in production: how much does compiling the schema cost, how much does applying the mask cost per step, and what does the mask *not* protect you from.

A standing note for this series: I have no GPU and I have run nothing. Every number below is derived from arithmetic I show you, cited from a public source I link, or written as a script for you to run with an expected direction rather than a value I invented.

## 1. Three ways to get structured output, and only one is a guarantee

Before building anything, it is worth being precise about what the alternatives buy you, because "just use the library" hides a real design decision.

**Prompt and hope.** Put "respond only with JSON" in the system prompt. Cheap, zero engineering, no engine changes, and no guarantee whatsoever. The failure rate depends on the model, the prompt, the temperature, and the phase of the moon. It gets worse as the schema gets more complicated and it gets worse under distribution shift — which is to say, it gets worse exactly when you stop watching.

**Prompt, parse, retry.** Same as above, plus a validator and a loop. This *is* a guarantee, of a sort: the client eventually sees valid JSON or an error. But the guarantee is purchased with tokens and wall-clock. A retry re-runs the entire request: prefill again, decode again, bill again. If a fraction $f$ of requests need one retry, your mean cost per successful request is $(1+f)$ times the base cost and your tail is *at least* twice your base latency, because a p99 request is now a request that failed once and ran twice. The retry loop is the reason JSON-mode endpoints have bimodal latency histograms.

**Coerce after the fact.** Ask for JSON, then run a repair pass — strip code fences, find the outermost braces, fix trailing commas. This works surprisingly often and is worth having as a belt-and-braces layer. It is also unbounded in its failure modes, since "repair arbitrary text into the object I wanted" is not a well-defined operation. Repairing a truncated object requires guessing what the missing values were.

**Constrain the decode.** Compile the schema into a machine that knows, at every point, which characters may come next. Translate that into which *tokens* may come next. Mask the logits. Sample. The output is a member of the language by construction. There is no parse step that can fail, no retry, no repair, and no bimodal tail from re-running requests.

| Approach          | Guarantee                | Extra TTFT              | Extra per step          | Expressiveness       | Source                    |
| ----------------- | ------------------------ | ----------------------- | ----------------------- | -------------------- | ------------------------- |
| Prompt and hope   | none                     | zero                    | zero                    | anything you can ask | derived                   |
| Parse and retry   | eventual, at $(1+f)$ cost | zero                    | zero                    | anything you can ask | derived                   |
| Post-hoc repair   | best effort              | zero                    | zero                    | shallow fixes only   | derived                   |
| FSM logit mask    | by construction          | schema compile + index  | one mask per request    | regular languages    | derived; mechanism cited: vLLM structured decoding |
| Pushdown / CFG    | by construction          | grammar compile         | one mask per request    | context-free         | cited: XGrammar paper     |

The last row is the sequel; nested and recursive structures need a stack, which a finite-state machine does not have, and that is the subject of [grammar-based decoding with GBNF and XGrammar](/blog/machine-learning/inference-engineering/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar). This post stays in the regular world on purpose, because everything hard about constrained decoding — the token mismatch, the index, the memory, the batching — shows up already at the finite-state level, and it is much easier to see there.

One honest caveat before we start: a masked decode changes *what the model can say*, not *what the model knows*. Section 10 is entirely about that, and it is the part most write-ups skip.

## 2. The mechanism: a schema is a regular language

Start with the smallest possible piece: a JSON number.

The JSON grammar (RFC 8259, [the JSON data interchange format](https://www.rfc-editor.org/rfc/rfc8259)) defines a number as an optional minus sign, an integer part that is either `0` or a nonzero digit followed by more digits, an optional fraction, and an optional exponent. Written as a regular expression over characters:

```python
INT = r"-?(0|[1-9][0-9]*)"
NUM = INT + r"(\.[0-9]+)?([eE][-+]?[0-9]+)?"
```

A regular expression is a description of a *regular language*, and every regular language is recognised by a deterministic finite automaton: a finite set of states, a start state, a set of accepting states, and a transition function $\delta(s, c) \to s'$ that says which state you land in after reading character $c$ from state $s$. The key property, the one that makes all of this work, is that **the state is a complete summary of the past**. To know what may legally come next, you do not need the whole string produced so far. You need one integer.

![A state machine for a JSON number where the integer part branches into a fraction, an exponent, the accepting state and an unreachable dead state, and all three live paths merge back into accept](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-2.webp)

Read the figure as a contract per state. In `start`, the only legal characters are a minus sign or a digit. In `int part`, legal characters are more digits, a decimal point, an `e`, or whatever may follow a number in the enclosing grammar (a comma, a closing brace). In `fraction`, digits and `e`. That is the whole idea: **a state is a set of legal next characters**, and anything else leads to a dead state that we will simply never allow the model to enter.

Now compose. A quoted JSON string is also regular:

```python
HEX  = r"[0-9a-fA-F]"
CHAR = r'([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u' + HEX * 4 + r")"
STR  = r'"' + CHAR + r'*"'
```

Three alternatives inside the character class: an ordinary character (anything except a quote, a backslash, or a control byte), a two-character escape, or a `\u` escape with exactly four hex digits. Note that I spelled the four hex digits out by repetition rather than using a counted-repetition operator — the tiny regex dialect we are about to build has no `{4}`, and writing it out is one line and zero parser complexity.

And an object with a fixed set of required keys, in a fixed order, is regular too:

```python
BOOL  = r"(true|false)"
VALUE = {"string": STR, "integer": INT, "number": NUM, "boolean": BOOL}

def object_regex(props, ws=r"[ ]?"):
    """Fixed key order, all keys required. Every JSON Schema feature beyond
    this either fits the same shape or needs a pushdown automaton."""
    fields = [f'"{key}"{ws}:{ws}{VALUE[typ]}' for key, typ in props.items()]
    return "{" + ws + ("," + ws).join(fields) + ws + "}"
```

That covers a very large fraction of real API schemas: a flat object, known keys, scalar values, maybe an enum. `object_regex` for `{"label": "string", "score": "number"}` produces a pattern matching exactly the strings `{"label": "...", "score": ...}` with at most one optional space at each separator.

The `ws` parameter is worth pausing on. JSON allows arbitrary whitespace between tokens. If you model that faithfully as `[ \t\n\r]*`, you add a self-loop at every separator, and every self-loop state is another row in the index we are about to build. Restricting whitespace to "at most one space" is a real, deliberate narrowing of the language: the output is still valid JSON, it just is not *every* valid JSON. Almost every production constrainer makes some version of this choice, and it is one of the levers you pull when the state count gets out of hand.

### What "regular" cannot do

Fixed key order is a genuine limitation. If your schema says "these three keys, in any order", a DFA has to enumerate all orderings — ${3! = 6}$ paths through the machine, and $n!$ in general. Six is fine. Eight keys in any order is 40,320 paths and the compiler falls over. This is not a bug in the technique, it is the definition of "finite state": the machine has to remember which keys it has already emitted, and the only place it can store that is in the state.

Arbitrary nesting is the other wall. An object containing objects containing objects, to unbounded depth, is not a regular language — you need a stack to match braces. In practice a schema pins the depth, so you can unroll it, and the unrolled machine is regular again. But the state count multiplies with each level, and at some point you want the pushdown automaton instead.

## 3. The crux: the model emits tokens, the grammar reads bytes

Here is the part that most explanations wave at and then skip, and it is the entire engineering problem.

Everything in section 2 is a machine over **characters**. Your model does not emit characters. It emits **tokens** — integers indexing a vocabulary of subword pieces, each of which is a sequence of bytes. Llama-3.1-8B has a vocabulary of 128,256 entries. A single token may be one byte, or ten. It may be `{"`, spanning a brace and a quote — two different positions in the grammar. It may be `": "`, spanning a quote, a colon and a space, crossing from "end of key" to "start of value" in one step. It may be a lone continuation byte from the middle of a multi-byte UTF-8 character, which is not a character at all until the next token arrives — exactly the situation the [tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) built the incremental detokenizer to handle.

So the question "which tokens are legal right now" cannot be answered by looking up one character in $\delta$. It has to be answered like this:

> A token $t$ is legal from state $s$ if and only if the machine, started in $s$, can consume **every byte** of $t$ without dying. If it survives, the token also tells you your new state: whatever state the walk ended in.

![A walk of one four byte token through the machine one byte at a time, ending in a surviving state, next to a rejected token whose walk dies on its final byte](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-3.webp)

The figure shows the walk for the token `2.5e` starting in the `int part` state. Byte `2` keeps it in `int part`. Byte `.` moves it to `fraction`. Byte `5` stays. Byte `e` moves it to `exponent`. Four transitions, all survived, so the token is legal and its landing state is `exponent`. The token `2.5.5` fails on its fourth byte, because `fraction` has no outgoing edge for a second decimal point, so it is illegal and gets masked.

Three consequences fall out of this immediately, and they are the reason constrained decoding is an engineering problem rather than a five-line function.

**First: legality is per-state, not global.** The same token is legal from one state and illegal from another. `,` is legal after a complete value and illegal in the middle of a number. So the answer is a function of state, and there are as many answers as there are states.

**Second: the cost of computing one answer is proportional to token length, not to one.** Deciding whether token $t$ is legal from state $s$ costs $|t|$ transitions. Deciding it for the *whole vocabulary* from state $s$ costs $\sum_{t \in V} |t|$ transitions — call that sum $L$, the total byte length of the vocabulary. That is a large number, and you do not want to pay it inside the decode loop.

**Third: it does not depend on the request.** For a fixed grammar and a fixed tokenizer, the answer to "which tokens are legal from state $s$" is the same for every request, forever. So compute it once, store it, and look it up. That is the **index**, and it is the single idea that makes constrained decoding practical. It is exactly the construction described in Willard and Louf's [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), the paper behind Outlines: precompute a map from FSM state to the set of allowed vocabulary tokens, and the per-step work collapses to a table lookup.

### The byte-level detail that bites

Build the machine over **bytes**, not over Python characters. It is tempting to write the DFA over `str` and call `token.decode()` to walk it, and it works right up until a token contains an incomplete UTF-8 sequence — at which point `decode()` either throws or silently produces a replacement character, and you have just made a legality decision about the wrong string.

Byte-level BPE vocabularies genuinely contain such tokens. A three-byte character like an em dash or a currency symbol can be split across two tokens, so one of them holds the lead byte and the other holds continuations. If your grammar says "any character except quote, backslash, and control bytes", the byte-level version of that is "any byte except `0x22`, `0x5C`, and `0x00`–`0x1F`" — and a continuation byte in the range `0x80`–`0xBF` passes that test cleanly, which is what you want. The characters reassemble downstream in the detokenizer; the grammar never needs to know.

The cost of getting this wrong is subtle and awful: your masked output is valid JSON at the byte level but you have made a token illegal that should have been legal, so the model's non-ASCII text is quietly degraded. Nobody notices until a user writes in a language with accents.

## 4. Building the index: state to allowed-token bitmask

The index is a rectangular table with one row per DFA state and one column per vocabulary entry, holding one bit: legal or not. Alongside it sits a second table of the same shape holding the landing state.

Both tables are precomputed. At decode time, the entire per-step algorithm is:

1. Look up row `state` of the mask table.
2. Set every logit whose bit is zero to negative infinity.
3. Sample (with the full sampler chain from post 16 applied afterwards).
4. Look up `goto[state][sampled_token]` and store it as the request's new state.

Step 4 is one array read. Step 1 is one array read. There is no grammar evaluation in the loop at all.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="Eight token logits shown as bars; as the state machine advances through four states, illegal tokens collapse to negative infinity and the legal set shrinks and re-opens" style="width:100%;height:auto;max-width:820px">
<style>
.g1-bar{transform-box:fill-box;transform-origin:bottom}
.g1-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.g1-tok{font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.g1-hd{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.g1-sub{font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--accent,#6366f1);text-anchor:middle}
.g1-note{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.g1-inf{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:end}
@keyframes g1-p1{0%,22%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}25%,47%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}50%,97%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}100%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}}
@keyframes g1-p2{0%,22%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}25%,97%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}100%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}}
@keyframes g1-p3{0%,47%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}50%,72%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}75%,100%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}}
@keyframes g1-p4{0%,47%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}50%,97%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}100%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}}
@keyframes g1-p5{0%,22%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}25%,72%{transform:scaleY(1);fill:var(--accent,#6366f1);opacity:1}75%,100%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}}
@keyframes g1-p6{0%,100%{transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}}
@keyframes g1-f1{0%,22%{opacity:1}25%,97%{opacity:0}100%{opacity:1}}
@keyframes g1-f2{0%,22%{opacity:0}25%,47%{opacity:1}50%,100%{opacity:0}}
@keyframes g1-f3{0%,47%{opacity:0}50%,72%{opacity:1}75%,100%{opacity:0}}
@keyframes g1-f4{0%,72%{opacity:0}75%,97%{opacity:1}100%{opacity:0}}
.g1-a1{animation:g1-p1 12s ease-in-out infinite}
.g1-a2{animation:g1-p2 12s ease-in-out infinite}
.g1-a3{animation:g1-p3 12s ease-in-out infinite}
.g1-a4{animation:g1-p4 12s ease-in-out infinite}
.g1-a5{animation:g1-p5 12s ease-in-out infinite}
.g1-a6{animation:g1-p6 12s ease-in-out infinite}
.g1-s1{animation:g1-f1 12s ease-in-out infinite}
.g1-s2{animation:g1-f2 12s ease-in-out infinite}
.g1-s3{animation:g1-f3 12s ease-in-out infinite}
.g1-s4{animation:g1-f4 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.g1-a1,.g1-a3,.g1-a4,.g1-a5{animation:none;fill:var(--accent,#6366f1)}.g1-a2,.g1-a6{animation:none;transform:scaleY(.04);fill:var(--border,#d1d5db);opacity:.55}.g1-s1{animation:none;opacity:0}.g1-s2{animation:none;opacity:0}.g1-s3{animation:none;opacity:1}.g1-s4{animation:none;opacity:0}}
</style>
<g class="g1-s1"><text class="g1-hd" x="360" y="36">state s0 - only the opening brace is legal</text><text class="g1-sub" x="360" y="60">output so far: (nothing yet)</text></g>
<g class="g1-s2"><text class="g1-hd" x="360" y="36">state s1 - a key must open, or the object may close</text><text class="g1-sub" x="360" y="60">output so far: &#123;</text></g>
<g class="g1-s3"><text class="g1-hd" x="360" y="36">state s2 - inside a string, almost everything re-opens</text><text class="g1-sub" x="360" y="60">output so far: &#123;&quot;</text></g>
<g class="g1-s4"><text class="g1-hd" x="360" y="36">state s3 - after the colon, only a value may start</text><text class="g1-sub" x="360" y="60">output so far: &#123;&quot;n&quot;:</text></g>
<rect class="g1-bar g1-a1" x="60"  y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a2" x="140" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a3" x="220" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a4" x="300" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a5" x="380" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a4" x="460" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a3" x="540" y="90" width="56" height="130" rx="4"/>
<rect class="g1-bar g1-a6" x="620" y="90" width="56" height="130" rx="4"/>
<line class="g1-axis" x1="40" y1="220" x2="700" y2="220"/>
<text class="g1-inf" x="700" y="214">masked logits sit at -inf</text>
<text class="g1-tok" x="88"  y="242">&#123;</text>
<text class="g1-tok" x="168" y="242">&quot;</text>
<text class="g1-tok" x="248" y="242">The</text>
<text class="g1-tok" x="328" y="242">123</text>
<text class="g1-tok" x="408" y="242">&#125;</text>
<text class="g1-tok" x="488" y="242">true</text>
<text class="g1-tok" x="568" y="242">,</text>
<text class="g1-tok" x="648" y="242">0x82</text>
<text class="g1-note" x="360" y="276">8 of 128,256 vocabulary entries - bar height is the logit the sampler actually sees</text>
</svg>
<figcaption>As the machine walks from the opening brace into a key string and on to the first value, the set of survivable tokens shrinks, re-opens inside the string, and narrows again; every other logit is pinned at negative infinity before the sampler runs.</figcaption>
</figure>

Watch the third frame in that animation carefully, because it is the one that surprises people. Inside a string value, the allowed set is *enormous* — nearly the whole vocabulary, since a JSON string may contain any character except a quote, a backslash, or a control byte. The mask is not always restrictive. It is restrictive at the structural joints (braces, colons, commas) and almost transparent in the middle of free text. That shape has a direct performance consequence, which section 9 gets to: the mask costs the same whether it removes 128,000 tokens or three.

### How much does the index cost to store

One row is one bit per vocabulary entry. For a vocabulary of size $|V|$:

$$\text{bytes per state} = \left\lceil \frac{|V|}{8} \right\rceil$$

For Llama-3.1-8B, $|V| = 128{,}256$, so a row is $128{,}256 / 8 = 16{,}032$ bytes, or **15.7 KB per state**. Store the same information as a byte-per-token boolean array and it is 128,256 bytes, or **125 KB per state** — eight times more, for the same information.

![A three by three comparison of storing one allowed set as a boolean tensor, a packed bitmask, or a sorted list of token ids, with the bytes per state and the per step cost at batch sixty four](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-4.webp)

| Representation | Bytes per state (derived)     | Batch 64, one step   | Notes                                | Source  |
| -------------- | ----------------------------- | -------------------- | ------------------------------------ | ------- |
| bool tensor    | $\lvert V \rvert$ = 125 KB    | 7.8 MB               | fine if it lives on device already   | derived |
| int32 bitmask  | $\lvert V \rvert / 8$ = 15.7 KB | 1.0 MB             | the one you ship                     | derived |
| sorted id list | 4 bytes per legal token       | varies with state    | tiny at joints, huge inside a string | derived |

The sorted-id-list row is the interesting one. It is by far the smallest representation at a structural joint, where only two or three tokens are legal, and by far the largest inside a string, where 120,000 tokens are legal. A representation whose size depends on the data is a representation whose *per-step cost* depends on the data, which means jitter. The bitmask is the same 15.7 KB at every state, which makes it boring, predictable, and correct to ship.

## 5. The memory bill, and where it explodes

Multiply out.

$$M_{\text{index}} = |S| \cdot \left\lceil \frac{|V|}{8} \right\rceil \cdot 2$$

The factor of two is the `goto` table alongside the mask — except `goto` cannot be a bitmask, it holds a state id. Store it as int32 and it is ${4|V| = 513}$ KB per state, which is thirty-two times the mask and would dominate everything. So do not store it densely. The landing state is only needed for tokens that are *legal*, and at a structural joint that is a handful of entries. Store `goto` as a sparse dictionary per state, or better, as a single flat dictionary keyed by `(state, token)` — the number of entries is the number of legal `(state, token)` pairs, which is exactly the number of set bits in the mask table.

That leaves the mask table as the dominant term:

$$M_{\text{index}} \approx |S| \cdot 15.7\ \text{KB} \quad \text{for a 128k vocabulary}$$

![A layered view of index memory growing from a single state through a small object schema, a large schema, an enum heavy schema and finally a fleet of live schemas, with a bounded cache at the base](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-5.webp)

#### Worked example: index memory for a real schema

Take a classification endpoint: `{"label": string, "confidence": number, "reason": string}`, fixed key order, at most one space between tokens.

- The literal key names contribute one state per character: `"label"` is 7 characters including quotes, `"confidence"` is 12, `"reason"` is 8. That is 27 states, plus a colon and optional space after each (6 more) and a comma and optional space between fields (4 more).
- Two string values contribute the string machine: quote, in-string, escape, four hex-digit states, closing quote — about 8 states each, and they are shared if the compiler minimises, but assume not: 16.
- One number value contributes the machine from figure 2: about 6 states.
- Opening and closing brace, plus the accept state: 3.

Total by hand: roughly 62 states. Compile it with the code in section 6 and `len(dfa.delta)` reports **74** — hand-counting undershoots, because subset construction splits states you thought were shared. Take the compiler's number, always. Index memory: $74 \times 15.7\ \text{KB} = 1{,}158\ \text{KB}$, call it **1.1 MB**. That is nothing. Source: derived, from the byte-per-state formula and a state count printed by the compiler in section 6.2.

Now the same arithmetic for a schema with a large enum. An enum of $n$ fixed strings compiles to something close to a trie over those strings: the machine has to remember exactly which prefix it has emitted, because that determines which characters may follow. So the state count is bounded above by the total character count, $n \cdot \bar{c}$, and the fraction of that you actually pay depends entirely on how much prefix the values share. Section 11 compiles a real one — 5,000 product codes — and the compiler reports **10,588 states**, which at 15.7 KB each is $10{,}588 \times 15.7\ \text{KB} = 162\ \text{MB}$ for a single schema.

162 MB is survivable for one schema. It is not survivable for two hundred. A multi-tenant endpoint where every customer has their own schema, and several of them have big enums, will hold gigabytes of index — memory that comes directly out of your KV cache budget, which is the thing that actually sets your maximum batch size. This is the single most important operational fact about constrained decoding at scale, and it argues for exactly one design: **an LRU cache of compiled indices keyed by a hash of the canonical schema**, with a hard byte budget, and eviction that is allowed to happen.

## 6. Writing it: `nanoserve/grammar/`

Enough analysis. Here is the code, in four files.

### 6.1 A byte-level regex compiler

We need a regex engine, but a very small one: alternation, concatenation, `*`, `+`, `?`, character classes, and escapes. No backreferences, no lookahead, no counted repetition — none of which are regular anyway, or in the case of `{n}`, none of which we need.

```python
# nanoserve/grammar/regex.py
"""A byte-level regex -> NFA -> DFA compiler. Alphabet is 0..255."""
from dataclasses import dataclass

CLASSES = {
    "d": frozenset(range(0x30, 0x3A)),
    "w": frozenset(range(0x30, 0x3A)) | frozenset(range(0x41, 0x5B))
         | frozenset(range(0x61, 0x7B)) | {0x5F},
    "s": frozenset({0x20, 0x09, 0x0A, 0x0D}),
}
ANY = frozenset(range(0x20, 0x7F)) | frozenset(range(0x80, 0x100))


@dataclass
class Frag:
    start: int
    accept: int


class NFA:
    def __init__(self):
        self.moves: list[dict[int, set[int]]] = []
        self.eps: list[set[int]] = []

    def state(self) -> int:
        self.moves.append({})
        self.eps.append(set())
        return len(self.moves) - 1

    def move(self, s: int, byte: int, t: int) -> None:
        self.moves[s].setdefault(byte, set()).add(t)

    def epsilon(self, s: int, t: int) -> None:
        self.eps[s].add(t)
```

The parser builds the NFA directly, Thompson-construction style: every syntactic form becomes a fragment with one entry and one exit, wired together with epsilon edges.

```python
class _Parser:
    def __init__(self, pattern: str, nfa: NFA):
        self.p, self.i, self.n = pattern, 0, nfa

    def peek(self) -> str:
        return self.p[self.i] if self.i < len(self.p) else ""

    def eat(self) -> str:
        c = self.p[self.i]
        self.i += 1
        return c

    def alt(self) -> Frag:
        frags = [self.concat()]
        while self.peek() == "|":
            self.eat()
            frags.append(self.concat())
        if len(frags) == 1:
            return frags[0]
        s, a = self.n.state(), self.n.state()
        for f in frags:
            self.n.epsilon(s, f.start)
            self.n.epsilon(f.accept, a)
        return Frag(s, a)

    def concat(self) -> Frag:
        frags = []
        while self.peek() not in ("", "|", ")"):
            frags.append(self.repeat())
        if not frags:                      # the empty language element
            s = self.n.state()
            return Frag(s, s)
        for x, y in zip(frags, frags[1:]):
            self.n.epsilon(x.accept, y.start)
        return Frag(frags[0].start, frags[-1].accept)

    def repeat(self) -> Frag:
        f = self.atom()
        while self.peek() in ("*", "+", "?"):
            op = self.eat()
            s, a = self.n.state(), self.n.state()
            self.n.epsilon(s, f.start)
            self.n.epsilon(f.accept, a)
            if op in ("*", "?"):
                self.n.epsilon(s, a)       # skip the body entirely
            if op in ("*", "+"):
                self.n.epsilon(f.accept, f.start)   # loop back
            f = Frag(s, a)
        return f

    def atom(self) -> Frag:
        c = self.eat()
        if c == "(":
            f = self.alt()
            assert self.eat() == ")", "unbalanced parenthesis"
            return f
        if c == "[":
            return self.charclass()
        if c == "\\":
            bs, _ = self.escape()
            return self.byteset(bs)
        if c == ".":
            return self.byteset(ANY)
        assert ord(c) < 256, f"non-ascii literal {c!r} in pattern"
        return self.byteset(frozenset({ord(c)}))

    def escape(self):
        """Consume one escape body. Returns (byteset, single code or None)."""
        e = self.eat()
        if e in CLASSES:
            return CLASSES[e], None
        if e == "x":
            code = int(self.eat() + self.eat(), 16)
            return frozenset({code}), code
        return frozenset({ord(e)}), ord(e)

    def member(self):
        c = self.eat()
        if c == "\\":
            return self.escape()
        return frozenset({ord(c)}), ord(c)

    def charclass(self) -> Frag:
        neg = self.peek() == "^"
        if neg:
            self.eat()
        bs: set[int] = set()
        while self.peek() != "]":
            lo_set, lo = self.member()
            if lo is not None and self.peek() == "-" and self.p[self.i + 1] != "]":
                self.eat()
                _, hi = self.member()
                bs |= set(range(lo, hi + 1))
            else:
                bs |= lo_set
        self.eat()                          # closing bracket
        if neg:
            bs = set(range(256)) - bs
        return self.byteset(frozenset(bs))

    def byteset(self, bs: frozenset) -> Frag:
        s, a = self.n.state(), self.n.state()
        for b in bs:
            self.n.move(s, b, a)
        return Frag(s, a)
```

Then subset construction, which turns the nondeterministic machine into a deterministic one. This is textbook, and it is where the state count is decided.

```python
@dataclass
class DFA:
    start: int
    accept: frozenset
    delta: list          # state -> {byte: state}


def to_dfa(nfa: NFA, frag: Frag) -> DFA:
    def closure(seeds) -> frozenset:
        stack, seen = list(seeds), set(seeds)
        while stack:
            s = stack.pop()
            for t in nfa.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    start = closure({frag.start})
    ids = {start: 0}
    order = [start]
    delta: list[dict[int, int]] = []
    i = 0
    while i < len(order):
        cur = order[i]
        i += 1
        by_byte: dict[int, set[int]] = {}
        for s in cur:
            for b, ts in nfa.moves[s].items():
                by_byte.setdefault(b, set()).update(ts)
        row: dict[int, int] = {}
        for b, ts in by_byte.items():
            nxt = closure(ts)
            if nxt not in ids:
                ids[nxt] = len(order)
                order.append(nxt)
            row[b] = ids[nxt]
        delta.append(row)
    accept = frozenset(j for j, ss in enumerate(order) if frag.accept in ss)
    return DFA(start=0, accept=accept, delta=delta)


def compile_regex(pattern: str) -> DFA:
    nfa = NFA()
    return to_dfa(nfa, _Parser(pattern, nfa).alt())
```

Two things to notice, because they matter later. First, `delta[s]` is a dictionary containing only the bytes that have a transition — a missing key *is* the dead state, which makes the legality walk a single `.get()` per byte with no sentinel bookkeeping. Second, `to_dfa` visits states in id order and appends rows in the same order, so `delta[s]` is genuinely state `s`'s row. Get that wrong and you produce a machine that is subtly, silently, wrong — and a wrong grammar mask does not crash, it just constrains to the wrong language.

Run it on the number regex and print the state count:

```python
dfa = compile_regex(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?")
print(len(dfa.delta), sorted(dfa.accept))
```

```console
10 [2, 3, 6, 7, 9]
```

Ten states for a JSON number, five of them accepting — because `1`, `10`, `1.5`, `1e9` and `1.5e-9` are all complete numbers, and the machine has to be able to stop at each. This is the first sanity check on any grammar compiler you write: does the accepting set match the strings you believe are complete? Check it directly rather than by eye:

```python
def accepts(dfa, s: str) -> bool:
    cur = dfa.start
    for b in s.encode():
        cur = dfa.delta[cur].get(b)
        if cur is None:
            return False
    return cur in dfa.accept
```

### 6.2 Schema to regex

```python
# nanoserve/grammar/schema.py
import json

INT  = r"-?(0|[1-9][0-9]*)"
NUM  = INT + r"(\.[0-9]+)?([eE][-+]?[0-9]+)?"
HEX  = r"[0-9a-fA-F]"
CHAR = r'([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u' + HEX * 4 + r")"
STR  = r'"' + CHAR + r'*"'
BOOL = r"(true|false)"
META = set("\\|()[]*+?.")

SCALAR = {"string": STR, "integer": INT, "number": NUM, "boolean": BOOL}


def literal(s: str) -> str:
    """Escape a fixed string so the regex matches it verbatim."""
    return "".join("\\" + c if c in META else c for c in s)


def value_regex(spec: dict) -> str:
    if "enum" in spec:
        return "(" + "|".join(literal(json.dumps(v)) for v in spec["enum"]) + ")"
    return SCALAR[spec["type"]]


def schema_regex(schema: dict, ws: str = r"[ ]?") -> str:
    """A flat object with required, fixed-order keys. Deliberately narrow."""
    assert schema["type"] == "object", "only flat objects here"
    props = schema["properties"]
    fields = [
        f'{literal(json.dumps(k))}{ws}:{ws}{value_regex(v)}'
        for k, v in props.items()
    ]
    return "{" + ws + ("," + ws).join(fields) + ws + "}"
```

`json.dumps` on a key or an enum value gives you the quoted, escaped JSON form for free, which is exactly what the grammar needs to match. Then `literal` escapes the regex metacharacters inside it. This ordering matters: escape for JSON first, then for the regex.

```python
schema = {
    "type": "object",
    "properties": {
        "label": {"enum": ["spam", "ham", "unknown"]},
        "confidence": {"type": "number"},
    },
}
dfa = compile_regex(schema_regex(schema))
print(len(dfa.delta), "states")
for s in ['{"label": "spam", "confidence": 0.97}',
          '{"label":"ham","confidence":-1.5e-9}',
          '{"label": "other", "confidence": 1}',
          '{"label": "spam" "confidence": 1}']:
    print(accepts(dfa, s), s)
```

```console
59 states
True {"label": "spam", "confidence": 0.97}
True {"label":"ham","confidence":-1.5e-9}
False {"label": "other", "confidence": 1}
False {"label": "spam" "confidence": 1}
```

Fifty-nine states, so just under a megabyte of index at a 128k vocabulary by the formula in section 5. Note what the last two lines prove: a value outside the enum is rejected, and so is a missing comma. That is the guarantee, made concrete — those strings are not in the language, so no sequence of sampled tokens can produce them.

### 6.3 The index

Now the crux. For every state, walk every token.

```python
# nanoserve/grammar/index.py
import numpy as np


class GrammarDeadEnd(RuntimeError):
    """A reachable state from which no vocabulary token can continue."""


class TokenIndex:
    def __init__(self, allowed: np.ndarray, goto: dict, accept: frozenset, start: int):
        self.allowed = allowed              # [S, V] bool
        self.goto = goto                    # {(state, token): state}
        self.accept = accept
        self.start = start
        self.bitmask = np.packbits(allowed, axis=1, bitorder="little")  # [S, V/8] u8

    @classmethod
    def build(cls, dfa, token_bytes: list[bytes], eos_id: int,
              special_ids: frozenset = frozenset()) -> "TokenIndex":
        n_states, n_vocab = len(dfa.delta), len(token_bytes)
        allowed = np.zeros((n_states, n_vocab), dtype=bool)
        goto: dict[tuple[int, int], int] = {}
        for s in range(n_states):
            for tid, raw in enumerate(token_bytes):
                if tid in special_ids:
                    continue                # control markers are never grammar text
                cur, ok = s, True
                for b in raw:
                    nxt = dfa.delta[cur].get(b)
                    if nxt is None:
                        ok = False
                        break
                    cur = nxt
                if ok:
                    allowed[s, tid] = True
                    goto[(s, tid)] = cur
            # EOS is legal exactly where the grammar is complete.
            allowed[s, eos_id] = s in dfa.accept
            goto[(s, eos_id)] = s
        idx = cls(allowed, goto, dfa.accept, dfa.start)
        idx.check_reachable(eos_id)
        return idx

    def check_reachable(self, eos_id: int) -> set:
        """Walk the states a request can actually enter, one token at a time,
        and reject any that offer no legal continuation."""
        seen, stack = {self.start}, [self.start]
        while stack:
            s = stack.pop()
            live = np.flatnonzero(self.allowed[s])
            if not live.size:
                raise GrammarDeadEnd(
                    f"reachable state {s} allows no token; the grammar and the "
                    f"tokenizer cannot agree on any continuation"
                )
            for tid in live:
                t = self.goto[(s, int(tid))]
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        self.reachable = seen
        return seen
```

The reachability walk is not decoration. A DFA compiled from a regex contains states that are perfectly well-formed as *character* machines but that no sequence of *tokens* can ever land you in — the middle of a `\u` escape, for instance, when no vocabulary entry ends there. Checking "does every state allow at least one token" flags those and rejects a grammar that is actually fine. Checking "does every **reachable** state allow at least one token" is the question you meant, and it catches the real failure: a grammar that requires a character sequence your tokenizer cannot spell.

That inner triple loop is the entire compile cost, and it is worth writing down as a formula because it is what shows up in your TTFT:

$$C_{\text{build}} = |S| \cdot \sum_{t \in V} |t| \;=\; |S| \cdot L$$

where $L$ is the total byte length of the vocabulary. Measure $L$ for your tokenizer in two lines:

```python
tb = [t.encode() if isinstance(t, str) else t for t in vocab_bytes]
print(len(tb), sum(len(x) for x in tb))     # |V| and L
```

For a 128k byte-level vocabulary expect $L$ in the high hundreds of thousands of bytes. Taking $L \approx 600{,}000$ as a round figure and a 500-state schema, $C_{\text{build}} \approx 3 \times 10^8$ dictionary lookups. In pure Python, at an order-of-magnitude 100 ns per lookup, that is **about 30 seconds**. In a compiled implementation at a few nanoseconds per step it is **on the order of a second**. Both of those are order-of-magnitude estimates, clearly labeled as such — the point is not the exact value, it is that the value has three or four digits of milliseconds in it, and that number is going to land somewhere.

This is precisely the effect the vLLM team names in [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) (14 January 2025), where they describe FSM compilation as "a significant contributor to increased TTFT". It is not a subtle overhead you discover with a profiler. It is a visible stall at the front of the request.

Three ways to make it disappear, in increasing order of effort:

1. **Cache by schema hash.** The same schema compiles to the same index every time. A `sha256` over the canonically-serialised schema plus the tokenizer's identity is a perfect cache key. Section 7 has the code.
2. **Walk a vocabulary trie instead of a flat list.** Tokens share prefixes; walking `abc`, `abd` and `abe` separately repeats the `ab` walk three times. A trie over the vocabulary visits each distinct prefix once, and the per-state cost drops from $L$ to the trie's node count. Count it yourself before you build it — the saving depends entirely on your vocabulary:

   ```python
   trie = {}
   for raw in token_bytes:
       node = trie
       for b in raw:
           node = node.setdefault(b, {})
   def count(node): return 1 + sum(count(c) for c in node.values())
   print(count(trie), "trie nodes vs", sum(len(t) for t in token_bytes), "flat steps")
   ```

3. **Build lazily.** Most schemas never visit most of their states on a given request. Compute a state's row the first time a request lands in it, and memoize. The first request pays for the states it visits; every later request pays nothing. This turns a one-off multi-second stall into a handful of sub-millisecond hiccups spread across the first few requests.

### 6.4 The logits processor

```python
# nanoserve/grammar/processor.py
import torch


class GrammarLogitsProcessor:
    """Hard constraint. Runs FIRST in nanoserve's processor chain."""

    def __init__(self, index: TokenIndex, device: torch.device, debug: bool = False):
        self.index = index
        self.device = device
        self.debug = debug
        # [S, V] bool on device. 125 KB per state -- see section 9 for the
        # bitmask path you want when many schemas are live at once.
        self.masks = torch.from_numpy(index.allowed).to(device, non_blocking=True)
        self.last_retained: torch.Tensor | None = None

    def __call__(self, states: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """states: [B] int64 on device.  logits: [B, V] float32 on device."""
        allowed = self.masks.index_select(0, states)          # [B, V] bool
        if self.debug:
            # Retained probability mass: how much of the model's own belief
            # survives the mask. Costs an extra softmax -- debug only.
            with torch.no_grad():
                p = torch.softmax(logits.float(), dim=-1)
                self.last_retained = (p * allowed).sum(dim=-1)   # [B]
        logits.masked_fill_(~allowed, float("-inf"))
        return logits

    def advance(self, state: int, token_id: int) -> int:
        nxt = self.index.goto.get((state, token_id))
        if nxt is None:
            raise GrammarDeadEnd(f"token {token_id} illegal from state {state}")
        return nxt

    def is_done(self, state: int) -> bool:
        return state in self.index.accept
```

The `advance` raise is not defensive paranoia. If a token that the mask forbade is ever sampled, something upstream is broken — a stale mask, a request whose state was not updated after a preemption, an off-by-one in the batch index — and you want a loud crash, not a quietly malformed response. In the [continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) the running set is rebuilt every step; a grammar state that travels with the request rather than with the slot is the only version of this that survives preemption.

## 7. Plugging into the sampler chain, and the batched complication

Post 16 built a chain of logits processors. The grammar mask joins it, and **it must run first**.

```python
# nanoserve/engine/step.py
def sample_step(logits, batch, grammar, chain):
    """logits: [B, V]. batch: the running set, one entry per active request."""
    states = torch.tensor([r.fsm_state for r in batch],
                          device=logits.device, dtype=torch.long)
    logits = grammar(states, logits)         # 1. hard constraint, -inf on illegal
    for proc in chain:                       # 2. temperature, penalties, top-k/p
        logits = proc(logits, batch)
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    for r, tid in zip(batch, tokens.tolist()):
        r.fsm_state = grammar.advance(r.fsm_state, tid)
    return tokens
```

Order is not a style preference here, it is a correctness requirement, and the reason is worth spelling out.

Negative infinity is stable under everything the sampler chain does downstream. Dividing by a positive temperature leaves it at negative infinity. Adding a finite repetition penalty leaves it at negative infinity. A top-k selection picks the $k$ largest logits, and negative infinity is never among them as long as at least $k$ legal tokens exist. So masking first and sampling second always works.

Reverse the order and you have a live bug. Top-p keeps the smallest set of tokens whose cumulative probability reaches $p$ — computed on the *unmasked* distribution, that nucleus can consist entirely of tokens the grammar forbids. Apply the mask afterwards and every surviving logit is negative infinity. `softmax` of an all-negative-infinity row is `nan`, `multinomial` on a row of `nan` raises or returns garbage, and you have a request that dies in the sampler for reasons that look nothing like a grammar problem. Mask first.

There is one interaction that still needs care even in the correct order: **top-k with a small k at a tight joint**. If the grammar allows exactly two tokens and your chain runs `top_k=1`, you get greedy behaviour within the legal set, which is fine. But some implementations of top-k assume at least $k$ finite entries and will happily return an infinite one when there are fewer. Clamp $k$ to the number of legal tokens, or simply check for it once in a test.

### Every request is at a different state

This is the batching complication, and it is the reason a grammar processor cannot be a simple element-wise function.

In a continuously-batched engine, the running set at any step contains requests at wildly different points: one is emitting the opening brace, one is deep inside a string value, one is on its last comma, and three are not constrained at all. Their FSM states are unrelated integers, and each needs a different row of the mask table.

```python
allowed = self.masks.index_select(0, states)     # [B, V] bool, one gather
```

That single gather is the whole answer, and it is why the mask table is laid out as `[S, V]` rather than as a per-request object: the batched lookup becomes one contiguous device-side gather instead of B separate copies. The cost is a `[B, V]` boolean allocation per step — at batch 64 and a 128k vocabulary that is $64 \times 128{,}256 = 8.2$ MB of scratch, allocated and freed every step. Preallocate it. A steady-state decode loop should not be calling the allocator.

Requests with no schema still need a row. The cleanest handling is a synthetic "unconstrained" state whose mask is all ones, so the gather has no branch and the batch stays rectangular. It wastes 15.7 KB of index and buys you a decode loop with no special cases.

### Caching compiled indices

```python
# nanoserve/grammar/cache.py
import hashlib, json, threading
from collections import OrderedDict


class IndexCache:
    """LRU over compiled indices, bounded in bytes, keyed by schema + tokenizer."""

    def __init__(self, budget_bytes: int = 2 << 30):
        self._lru: OrderedDict[str, TokenIndex] = OrderedDict()
        self._bytes = 0
        self._budget = budget_bytes
        self._lock = threading.Lock()

    @staticmethod
    def key(schema: dict, tokenizer_id: str) -> str:
        canon = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{tokenizer_id}\x00{canon}".encode()).hexdigest()

    def get_or_build(self, schema: dict, tokenizer_id: str, builder) -> TokenIndex:
        k = self.key(schema, tokenizer_id)
        with self._lock:
            if k in self._lru:
                self._lru.move_to_end(k)
                return self._lru[k]
        index = builder(schema)                       # slow path, outside the lock
        size = index.bitmask.nbytes + index.allowed.nbytes
        with self._lock:
            self._lru[k] = index
            self._bytes += size
            while self._bytes > self._budget and len(self._lru) > 1:
                _, evicted = self._lru.popitem(last=False)
                self._bytes -= evicted.bitmask.nbytes + evicted.allowed.nbytes
        return index
```

Two details that are easy to get wrong. The canonical key must include the **tokenizer identity**, because the same schema compiles to a different index for a different vocabulary, and a cache that ignores that will hand a Llama index to a Qwen model and mask the wrong tokens. And `sort_keys=True` is doing real work: two JSON schemas that differ only in key order in the *schema document* describe the same language, and should hit the same cache entry.

The build must happen outside the lock, or one slow compile stalls every request in the process — which is the shape of the problem the vLLM post describes for Outlines' batch mask computation, where it notes the mask computation "blocks all requests" while it runs. Any single-threaded step in a shared component becomes a global stall under load.

## 8. The first cost: compile time lands on TTFT

TTFT — time to first token — is the interval between the request arriving and the first token reaching the client. Everything the engine does before the first forward pass is inside it.

Schema compilation is inside it. Index construction is inside it. And unlike prefill, neither of them uses the GPU at all, so they do not overlap with anything; they are dead host time in front of a request.

| Cost                        | Formula                                | Example value                    | Source                                     |
| --------------------------- | -------------------------------------- | -------------------------------- | ------------------------------------------ |
| Regex to DFA                | grows with pattern length and enum size | milliseconds for a flat object   | derived; measure with `time.perf_counter`  |
| Index build, cold           | $\lvert S \rvert \cdot L$ byte steps   | ~$3 \times 10^{8}$ at 500 states | derived, order of magnitude                |
| Index build, cached hit     | one dict lookup                        | microseconds                     | derived                                    |
| Index build, lazy per state | $L$ byte steps, once per visited state | ~600k steps per new state        | derived                                    |
| Effect on TTFT              | added directly, no GPU overlap         | named as significant             | cited: vLLM structured decoding post        |

The operational consequence is a rule about *when* you compile. There are exactly three sane policies:

**Compile at registration.** If schemas are registered ahead of time — a tool catalogue, a fixed set of endpoints — compile them at startup and never on the request path. This is the right answer whenever it is available, and it is available far more often than people assume.

**Compile on first use, asynchronously, and queue behind it.** The first request carrying a new schema triggers a build on a worker thread; that request waits, subsequent requests with the same schema wait on the same future rather than starting their own build. The naive version — every concurrent request for a cold schema starts its own compile — is a thundering herd that can take out the host.

**Compile lazily per state.** Build rows on demand. The cost is spread and no single request eats the whole bill.

What you must not do is compile synchronously on the request path with no cache, because then TTFT for a cold schema is the full build, and every request is cold.

#### Worked example: the TTFT arithmetic for a cold schema

Take the 74-state classification schema from section 5, serving Llama-3.1-8B on one A100 80GB, with a 700-token prompt.

- Prefill for 700 tokens on an 8B model is a few tens of milliseconds on an A100 — the exact figure depends on your attention backend and is something you should measure with the benchmark harness from [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark), not take from a blog post.
- Index build, cold, in pure Python: $74 \times 600{,}000 = 4.4 \times 10^7$ dictionary lookups. At an order-of-magnitude 100 ns each, roughly **4.4 seconds**.
- Index build, cache hit: a `sha256` over a few hundred bytes plus a dictionary lookup. **Microseconds.**

The ratio is the entire story. A cold build is two orders of magnitude larger than the prefill it precedes; a warm one is three orders smaller. Source: derived, from the state count you print with `len(dfa.delta)` and a per-lookup cost labeled as an order-of-magnitude estimate. Run it yourself with `time.perf_counter()` around `TokenIndex.build` and you will get your own number; expect it to be large and expect it to be dominated by the Python interpreter, which is why real implementations put this loop in Rust or C++.

## 9. The second cost: the mask, every step, for every request

The compile cost is paid once. The mask cost is paid on every decode step of every constrained request, forever, and it is the one that shows up as TPOT — time per output token — rather than as TTFT.

Break it into three pieces.

**Piece one: selecting the row.** One gather of `[B]` rows out of an `[S, V]` table. On device, this is a memory-bound kernel reading $B \cdot |V|$ bytes for the boolean layout. At batch 64 that is 8.2 MB read and 8.2 MB written. On an A100 with roughly 2 TB/s of HBM bandwidth (NVIDIA's [A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) lists 2,039 GB/s for the 80GB SXM part), $16.4\ \text{MB} / 2\ \text{TB/s} \approx 8\ \mu\text{s}$. Against a TPOT budget of 20 ms that is 0.04 percent. Negligible.

**Piece two: applying it.** `masked_fill_` over `[B, V]` floats: read 32.8 MB of logits at batch 64 in fp32, read the mask, write the logits back. Call it another 25 microseconds of bandwidth on the same GPU. Also negligible.

**Piece three: getting the mask to the GPU in the first place.** This is where the cost actually lives, and it is not a bandwidth problem, it is a *placement and serialisation* problem.

![Two paths for producing the per step bitmask, one building it once and broadcasting it to both GPU workers and one rebuilding it separately inside every worker](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-6.webp)

If the whole mask table lives on device, piece three is free — but section 5 showed that only works when the resident schema set is small. Once you have many live schemas, the table lives in host memory and the per-step masks have to be shipped.

$$\text{bytes per step} = B \cdot \left\lceil \frac{|V|}{8} \right\rceil$$

At batch 64 and $|V| = 128{,}256$: $64 \times 16{,}032 = 1{,}026{,}048$ bytes, almost exactly **1.0 MB per step**. PCIe 4.0 x16 runs 16 GT/s over 16 lanes with 128b/130b encoding, giving $16 \times 16 \times (128/130) = 252$ Gb/s, or **31.5 GB/s** in one direction. So 1.0 MB takes $1.0\ \text{MB} / 31.5\ \text{GB/s} \approx 33\ \mu\text{s}$ at theoretical peak, and you should expect a real pinned-memory copy to land somewhere in the 40–60 microsecond range. Against a 20 ms TPOT budget: a fifth of a percent. Still fine — *if the copy is asynchronous and the assembly is fast*.

Neither of those is free.

```python
# nanoserve/grammar/bitmask.py
import numpy as np, torch


def batch_bitmask(index: TokenIndex, states: list[int]) -> torch.Tensor:
    """[B, V/8] uint8, pinned, ready for a non-blocking H2D copy."""
    rows = index.bitmask[states]                 # numpy fancy index: B memcpys
    out = torch.from_numpy(np.ascontiguousarray(rows))
    return out.pin_memory()


def apply_bitmask_(logits: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Expand the packed mask on device and fill. packed: [B, V/8] uint8."""
    B, V = logits.shape
    shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1          # [B, V/8, 8]
    allowed = bits.reshape(B, -1)[:, :V].bool()
    return logits.masked_fill_(~allowed, float("-inf"))
```

The `batch_bitmask` path costs B memcpys of 15.7 KB each — pure `memmove`, no Python-level per-token work. That is the version that scales. The version that does not scale is any implementation that *computes* the allowed set per request per step: a Python loop over a schema, a set intersection, a fresh mask array. Sixty-four of those, serialised, at half a millisecond each, is 32 milliseconds — larger than the entire decode step, and it happens on the critical path with the GPU idle.

That is the failure the vLLM post names when it lists, among Outlines' limitations, that batch mask computation "blocks all requests". A shared, single-threaded mask builder is a lock that every request in the batch queues behind.

### Where the mask should actually be computed

There is a beautiful piece of scheduling available here, and it is what vLLM's v1 design does. The post describes the roadmap explicitly: move guided decoding to the scheduler level, compute the bitmask **in one process**, and broadcast it to the GPU workers.

Two wins come from that, and they are separate.

**The deduplication win.** With tensor parallelism, every worker holds a shard of the model but they all sample from the same logits. Computing the same bitmask in every worker is $TP$ times the work for one result. Compute it once in the scheduler, broadcast the bytes.

**The overlap win.** This is the subtle one. You know a request's next FSM state the instant you have sampled its token — which is *before* the next forward pass starts. So the mask for step $n{+}1$ can be assembled on the CPU while the GPU runs the forward pass for step $n{+}1$. A forward pass on an 8B model is on the order of milliseconds; the mask assembly is on the order of tens of microseconds. There is room by two orders of magnitude. Done right, the mask cost is not on the critical path at all — it is hidden entirely under a kernel you were running anyway.

Done wrong, the mask is assembled after the logits come back, on the same thread, and every microsecond of it is added directly to TPOT. Same total work, completely different latency.

This is the single most common way a JSON-mode rollout goes bad: the feature ships, p50 barely moves, and p99 quietly doubles overnight because the mask builder is a serialisation point that only bites once the batch is full. A later post in this series takes exactly that incident apart end to end, from the latency histogram back to the offending line. Instrument for it *before* you enable the feature, not after — the two costs to have on a dashboard from day one are cold-compile count and mask-assembly time per step.

| Where the mask is built    | Work per step            | On the critical path?     | Source                          |
| -------------------------- | ------------------------ | ------------------------- | ------------------------------- |
| In each TP worker          | $TP \times$ one bitmask  | yes                       | derived                         |
| Once, after logits return  | one bitmask              | yes                       | derived                         |
| Once, overlapped with forward | one bitmask           | no                        | derived; design cited: vLLM v1 roadmap |
| Whole table resident on device | zero                  | no                        | derived; needs small schema set |

## 10. What masking does not fix

Now the honest part, which is also the part that decides whether you should turn this on.

![A breakdown of JSON mode output into a syntax branch that the mask fully solves and semantics and distribution branches that it does not touch at all](/imgs/blogs/constrained-decoding-from-first-principles-masking-logits-with-an-fsm-7.webp)

### It guarantees syntax, not semantics

The mask is a proof about the *shape* of the string. It says the output parses, the keys are present, the number is a number. It says absolutely nothing about whether the number is right.

A schema-constrained model will cheerfully emit `{"label": "spam", "confidence": 0.97}` for a message that is obviously not spam. It will emit a required `"citation"` field containing a plausible-looking URL that does not exist, precisely *because* the grammar requires a string there and the model has to put something. Constrained decoding can even make hallucination worse in one specific way: a field the model has no information about cannot be omitted, so the model is forced to invent a value. An optional field the model can skip is often better engineering than a required field the model must fabricate.

The practical rule: the mask replaces your *parser*, not your *validator*. Keep the semantic checks — ranges, cross-field consistency, referential integrity against your own data — and run them on output that is now guaranteed to reach them in one piece.

### It reshapes the distribution

This is the more interesting cost, and it is real.

Let $p(t)$ be the model's next-token distribution and $A_s$ the allowed set at state $s$. Masking and sampling is exactly sampling from the renormalised conditional:

$$p'(t) = \frac{p(t) \cdot \mathbb{1}[t \in A_s]}{m_s}, \qquad m_s = \sum_{u \in A_s} p(u)$$

$m_s$ is the **retained mass**: how much of the model's own probability survives the mask. When $m_s$ is close to 1, masking is nearly a no-op — the model was going to produce a legal token anyway and you have changed almost nothing. When $m_s$ is small, you are sampling from the far tail of the model's belief, and the token you get is one the model considered unlikely. Do that repeatedly and the sequence drifts into a region the model has little reliable signal about, because every prefix it now conditions on is one it would not have written.

That is the mechanism behind the concern raised in [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) (Tam et al., 2024), which reports that format restrictions can degrade performance on reasoning tasks. The finding is contested — the counter-argument from the structured-generation community is that much of the measured gap comes from prompt and parsing differences rather than from the constraint itself — and post 20 in this series takes the accuracy question apart properly. For now, treat it as a live question and instrument for it rather than assuming either side is right.

Which is why `GrammarLogitsProcessor` exposes `last_retained`. Log the mean retained mass per request, in debug mode or on a sampled fraction of traffic, and you get a diagnostic that is genuinely actionable:

- **Mean $m_s$ near 1.** The model and the schema agree. The mask is insurance, not surgery.
- **Mean $m_s$ low.** The model is fighting your schema. Usually the fix is not in the engine: rename the keys to things the model would write naturally, reorder fields so the easy ones come first, loosen an enum, or put an example in the prompt. A schema the model wants to produce anyway costs you nothing.
- **A specific state with near-zero $m_s$.** You have found the exact joint where the constraint is doing violence. Print the state and the top unmasked tokens; the answer is usually obvious and usually a naming problem.

That last diagnostic is, in my view, the most useful thing you get out of writing the constrainer yourself instead of importing it. It is a direct measurement of the disagreement between your schema and your model, and there is no other way to see it.

### It does not save you tokens

A mild but common misconception: constrained decoding does not make the output shorter or the generation faster. The model still emits one token per step. If anything, forcing structure it would not have chosen can make the output *longer*, since the model has to spell out fixed keys character by character in tokens that are not its natural segmentation of them.

The exception is **jump-forward decoding**: when the machine's current state has exactly one outgoing path for the next several characters — the middle of a fixed key name, for instance — no model call is needed at all, because there is only one legal continuation. You can append those characters directly and skip the forward passes. SGLang's compressed-FSM work popularised this and it is a genuine throughput win on key-heavy schemas. Note, though, that the vLLM post lists "no jump-forward" among the limitations of a pure logit-processor design: a processor that only sees logits cannot skip a step, because by the time it runs, the step has already happened. Getting jump-forward requires the grammar to live at the scheduler level, which is the same architectural move as the bitmask broadcast.

## 11. Stress test: where it breaks

A technique is only as good as its behaviour at the edges. Four edges, and what each one does to the machinery above.

### A schema with a huge enum

The failure is state count, and it is quadratic in annoyance: the compile gets slow *and* the index gets big, and both are linear in states, so a 100× bigger enum is 100× both.

```python
codes = [f"SKU-{i:05d}" for i in range(5000)]
schema = {"type": "object",
          "properties": {"sku": {"enum": codes},
                         "qty": {"type": "integer"}}}
dfa = compile_regex(schema_regex(schema))
print(len(dfa.delta), "states",
      len(dfa.delta) * 128256 // 8 / 2**20, "MB of mask")
```

```console
10588 states 161.9 MB of mask
```

Ten thousand states and 162 MB of mask, for a schema whose *other* field is a single integer. That is memory taken directly out of your KV cache budget, which is what sets your maximum batch size. Three fixes, in order of preference:

1. **Do not put the enum in the grammar.** Constrain the field to a plain string and validate the value afterwards against a set. You lose the by-construction guarantee for that one field and you get a 40-state schema.
2. **Minimise the DFA.** Hopcroft's algorithm merges states with identical futures, and an enum of similar strings has enormous suffix sharing — all 5,000 SKU codes converge on the same "expecting more digits then a quote" states. Minimisation on this example should collapse it dramatically. Implementing it is a page of code and worth it if enums are your workload.
3. **Emit an index rather than a value.** Change the schema so the model picks an integer from `0` to `4999` and you look the SKU up yourself. This is often better for accuracy too, since a five-digit integer is a much easier thing for a model to commit to than a 9-character opaque string.

### Unicode in string values

Covered in section 3, but the failure mode deserves a test. A grammar built over Python `str` and walked by decoding each token will throw or corrupt on a token holding a partial multi-byte character. A grammar built over bytes handles it without knowing it happened.

```python
def test_partial_utf8_token_is_legal_inside_a_string():
    # 0xE2 0x82 0xAC is the euro sign. A byte-level BPE vocabulary can hold
    # the lead byte and the continuation bytes as two separate tokens.
    dfa = compile_regex(STR)
    vocab = [b'"', b"\xe2", b"\x82\xac", b"<eos>"]
    idx = TokenIndex.build(dfa, vocab, eos_id=3, special_ids=frozenset({3}))

    s1 = idx.goto[(dfa.start, 0)]          # after the opening quote
    assert idx.allowed[s1, 1]              # lead byte legal
    s2 = idx.goto[(s1, 1)]                 # now mid-character
    assert idx.allowed[s2, 2]              # continuation bytes still legal
    s3 = idx.goto[(s2, 2)]
    assert idx.allowed[s3, 0]              # and the string can close
```

The assertion that matters is the second one: from a state in the middle of a character, the continuation bytes must still be legal. If your character class was written over Unicode code points and then naively projected to bytes, they will not be, and you will have silently banned every non-ASCII character from your string fields.

### A token that jumps past the end

Consider a schema whose output ends `...}` and a vocabulary containing the token `}\n\n`. Walking that token from the state just before the final brace consumes `}`, reaches the accepting state, and then needs a transition for `\n` — which does not exist, so the token is illegal and gets masked. Correct, and it falls out of the construction for free.

But now consider the opposite arrangement: a grammar with a trailing `\s*`, and the same token. The walk succeeds, and the model has emitted trailing whitespace it may not have intended. Harmless for JSON. Not harmless if your grammar's tail is permissive in a way you did not think about — a `.*` at the end of a pattern means every token is legal forever and your constraint has evaporated.

The rule: **make the accepting state as tight as you can, and prefer to end the grammar at the last meaningful character**. Then EOS is the only legal continuation and there is nothing to overshoot into.

### The model wants EOS in the middle of the object

This one is guaranteed to happen and its handling defines whether your endpoint is production-grade.

`TokenIndex.build` sets `allowed[s, eos_id] = s in dfa.accept` — EOS is legal exactly where the grammar is complete, and masked everywhere else. Which means when the model decides it is done halfway through the object, the mask forbids it, and the model must emit something. It emits its best legal token, which may be an arbitrary continuation of a string it had finished saying.

There are three outcomes and you need to handle all three:

1. **The model recovers.** Most common. It emits a closing quote, a comma, the next key, and finishes properly. Nothing to do.
2. **The model loops.** It wanted to stop, cannot, and the highest-probability legal token is one that keeps the current string open. You get `"aaaaaaaa..."` until `max_tokens`. **Always set `max_tokens` on a constrained request.** A constrained decode has removed the model's ability to terminate itself, so the length bound is now the only stopping condition you have.
3. **The state has no legal token at all.** `GrammarDeadEnd`. This should be impossible if `check_reachable` passed at build time, because it walks every state a request can enter and rejects the grammar if any of them offers no continuation — which is a strictly better place to find out than the middle of a decode loop.

```python
def step_with_budget(req, grammar, logits, chain):
    if req.emitted >= req.max_tokens:
        if not grammar.is_done(req.fsm_state):
            # Truncated under a hard constraint: the output is a prefix of a
            # legal string, not a legal string. Say so; do not ship half an object.
            raise TruncatedUnderGrammar(req.id, req.fsm_state)
        return None
    ...
```

Never return a truncated constrained output as a success. The whole promise of this technique is that the client does not have to parse defensively; hitting `max_tokens` mid-object breaks that promise, and the client deserves an error rather than a half-object it will try to `json.loads`.

Repeated `TruncatedUnderGrammar` errors are also a signal, not just an error: they mean the retained mass is low and the model is fighting the schema. Correlate the two and you will usually find one bad field.

## 12. How to measure it honestly

Four measurements, and the order matters because each one isolates a different cost.

**Compile time, cold and warm.** Wrap `TokenIndex.build` in `time.perf_counter()`. Run it twice: once with a cold cache and once warm. Report both. The cold number tells you what a new schema costs a user; the warm number tells you what your cache is worth. Expect a ratio in the thousands.

**Per-step mask overhead, isolated.** CUDA events around the processor call, on a fixed batch, with warmup and a synchronize before you start the clock:

```python
def bench_mask(grammar, states, vocab, iters=200, warmup=20, device="cuda"):
    logits = torch.randn(len(states), vocab, device=device)
    st = torch.as_tensor(states, device=device)
    for _ in range(warmup):
        grammar(st, logits.clone())
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        grammar(st, logits.clone())
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters      # milliseconds per call
```

Three things this gets right and most quick benchmarks get wrong: the warmup, so you are not timing kernel compilation or the caching allocator's first-touch; the `synchronize()` before the first `record()`, so no earlier work leaks into the window; and the `clone()`, so you are not timing `masked_fill_` on an already-all-negative-infinity tensor, which is a different memory access pattern. Expect this to come out in the tens of microseconds on an A100 at batch 64 — bandwidth-bound, as derived in section 9 — and if it comes out in the milliseconds you have Python in the loop.

**End-to-end, under open-loop load.** Neither of the above tells you what a user sees. Drive the server with Poisson arrivals at a fixed rate, with and without a schema, and compare TTFT and TPOT distributions at p50 and p99. Open-loop matters here more than usual: a closed-loop harness with fixed concurrency hides queueing, and queueing is precisely where a serialised mask builder does its damage. If your p99 TPOT degrades while your p50 does not, you have a serialisation problem, not a bandwidth problem.

**Retained mass, in production.** Sample a small fraction of constrained requests with `debug=True` and log mean and minimum $m_s$. This is a quality metric, not a performance metric, and it is the one that tells you whether the constraint is helping or fighting.

A note on what *not* to report: tok/s at batch 1 with a schema tells you nothing about a server. The mask cost is per-request-per-step, so it scales with batch, while the model's decode cost at batch 1 is dominated by weight traffic and barely scales at all. A single-stream benchmark will show the mask as free. Under load it is not free, and load is what you are shipping.

## 13. Case studies and real numbers

**vLLM's structured decoding backends.** The vLLM team's [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) post (14 January 2025) is the best single account of what these systems cost in production, and it is the primary source for this post's cost claims. It describes the Outlines backend as a token-level FSM advancing one token per step — exactly the machine built here — and names two limitations directly: FSM compilation is a significant contributor to increased TTFT, and the batch mask computation blocks all requests. It also states the v1 direction: guided decoding moves to the scheduler level, the bitmask is computed in one process and broadcast to the GPU workers.

**XGrammar's throughput claim.** The same post reports XGrammar delivering up to a 5× improvement in TPOT under load, and notes that grammar compilation was moved from Python into C with pthread-based parallelism to get there. Two lessons, and the second is the one to internalise: the algorithmic idea in this post is language-agnostic, but the *implementation language* of the compile loop is a first-order performance decision. A Python triple loop over 128k tokens is not going to be competitive with anything. The [XGrammar paper](https://arxiv.org/abs/2411.15100) has the full construction.

**The Outlines index construction.** Willard and Louf's [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) is the paper that named the precomputed state-to-token index as the thing that makes constrained decoding cheap at decode time. Everything in section 4 is a restatement of that idea, with the memory arithmetic worked out for a modern 128k vocabulary.

**Backend selection is a real decision.** The vLLM post describes falling back from XGrammar to Outlines when XGrammar cannot express a grammar (in that version, no regex, and no complex JSON expressed via regex), and notes that lm-format-enforcer struggles in long-context settings. If you use a library rather than your own code, know which backend you are actually on — the guarantees, the compile cost, and the failure modes all differ, and the selection may be automatic and silent. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) in this repo covers how those choices surface in the engine's configuration.

## 14. When to reach for this, and when not to

**Reach for it when the output is consumed by a program.** Any time a downstream system parses the model's output — tool calls, extraction pipelines, classification endpoints, anything that feeds a database — the by-construction guarantee is worth its cost. The cost is bounded and measurable; a parse failure in production is neither.

**Reach for it when your retry rate is nonzero and your schema is stable.** The break-even is easy: if a fraction $f$ of requests retry, you are paying $(1+f)$ times the token cost and taking a doubled tail on those requests. A cached index costs microseconds of TTFT and tens of microseconds per step. Any $f$ above roughly a percent makes the mask cheaper, and the tail improvement is free on top.

**Do not reach for it when the output is prose for a human.** A grammar over free text buys nothing and can only distort.

**Do not reach for it when every request has a different, large schema.** The cache cannot help you, every request pays a cold compile, and the index memory competes with your KV cache. Constrain to a plain string and validate afterwards.

**Do not write it yourself for production.** Write it once, as we did here, to understand the mechanism, the memory, and the diagnostics. Then use XGrammar or Outlines, because they have the compile loop in a fast language, the DFA minimiser, and years of edge cases you have not thought of. What you gain from building it is the ability to read their configuration knobs and know exactly what each one is doing to your latency — and the retained-mass diagnostic, which you can add to any engine in twenty lines.

**Reach for a pushdown automaton instead when the schema is recursive.** Nested objects to unbounded depth, arrays of objects, anything with real recursion: a finite-state machine cannot express it and unrolling it explodes the state count. That is the next post.

## Key takeaways

- **Masking is a guarantee, retrying is a hope.** Setting illegal logits to negative infinity removes malformed output from the support of the distribution; no parse step downstream can fail.
- **The state is the whole summary of the past.** A DFA state, one integer per request, is all you need to know which characters may come next. This is why the technique is cheap at all.
- **The hard part is that models emit tokens and grammars read bytes.** A token is legal from a state only if the machine survives *every one of its bytes*, and the landing state is wherever the walk ended. Precompute that for every state and every token — that table is the whole technique.
- **Build the machine over bytes, never over decoded characters.** Byte-level vocabularies contain partial UTF-8 sequences, and decoding them to check legality either throws or silently bans non-ASCII text from your string fields.
- **A mask row is $\lceil \lvert V \rvert / 8 \rceil$ bytes** — 15.7 KB at a 128k vocabulary. Multiply by states, multiply by live schemas, and put a byte-budgeted LRU in front of it before the number surprises you.
- **Compile cost lands on TTFT with no GPU overlap.** Cache by a hash of the canonical schema plus the tokenizer identity, build outside the lock, and prefer compiling at registration to compiling on the request path.
- **Build the bitmask once per step and overlap it with the forward pass.** You know the next state the moment you sample the token, which is milliseconds before you need the mask. Rebuilding it per worker, after the logits return, is the same work at strictly worse latency.
- **The grammar mask runs first in the sampler chain.** Top-p computed before the mask can leave every legal token outside the nucleus, and an all-negative-infinity row softmaxes to `nan`.
- **Syntax is guaranteed; semantics is not.** Keep your validator. A required field the model has no answer for is a field the model must invent — an optional field is often the better schema.
- **Log the retained mass.** $m_s = \sum_{u \in A_s} p(u)$ tells you how much of the model's belief survives the mask, and a low value points at exactly which field of your schema the model disagrees with.
- **Always set `max_tokens` on a constrained request,** and error rather than return a truncated object. Masking EOS removes the model's ability to stop on its own.

## Further reading

- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) — Willard and Louf. The state-to-token index construction that this whole post is built on.
- [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100) — the pushdown-automaton successor, and the engineering that makes compilation fast.
- [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) — the vLLM team's account of backends, compile-time TTFT cost, batch-mask serialisation, and the v1 scheduler-level bitmask design.
- [RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259) — the grammar you are compiling, in three pages.
- [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) — Tam et al., on format restriction and reasoning performance. Read it alongside the counter-arguments; the question is not settled.
- [From logits to tokens: the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) — the processor chain this mask runs in front of.
- [Grammar-based decoding: GBNF, pushdown automata and XGrammar](/blog/machine-learning/inference-engineering/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar) — what to do when a finite-state machine is not enough.
- [The tokenizer boundary and incremental detokenization](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) — why the grammar has to be byte-level, in full.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the capstone that puts every piece of `nanoserve` back together.
