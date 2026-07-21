---
title: "Structured output in production: streaming JSON, tool calls, and the failures that survive a perfect grammar"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Your decoder can no longer emit invalid JSON, and your product still breaks - because a prefix is not a document, a tool call is a loop rather than a token, and three of the nastiest bugs in the business live outside the sampler entirely."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "structured-output",
    "tool-calling",
    "streaming",
    "json",
    "decoding",
    "agents",
    "vllm",
    "ml-systems",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 53
---

The two posts before this one bought you a guarantee. A finite-state machine or a pushdown automaton walks alongside the sampler, masks every token that would break the schema, and the string that comes out of your engine is now, by construction, syntactically valid JSON. It cannot fail to parse. You wrote the mask, you tested it against a nasty recursive schema, and it holds.

Then you shipped it, and the bug reports started anyway. The UI shows a spinner for six hundred milliseconds and then snaps the whole object into place, which is a worse experience than the free-text version it replaced. An agent loop calls a tool that was not in the request — it was in the conversation four turns ago — and your dispatcher returns a 404 into the model's context, where it sits and poisons the next three turns. A field validates against the schema and is still wrong: `"unit": "F"` when the user asked in Celsius, `"days": 0` when the schema said integer and meant positive integer. And once a week, some request comes back with a tool call whose id is `search:2`, and a parser deep in a serving stack raises `IndexError` and takes the whole request down.

None of those is a grammar bug. Every one of them lives in the space between "the tokens are legal" and "the product works". That space is what this post is about.

![Layered view of a streamed structured response showing byte-level holdback, token ids, the stable decoded prefix, the unbalanced JSON prefix, the tolerant parse and the finalized UI fields](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-1.webp)

By the end you will have written `nanoserve/structured/preview.py` — an incremental, tolerant JSON parser that turns every intermediate prefix into a renderable object and emits per-field completion events — and `nanoserve/structured/tools.py`, a tool-call loop with the failure paths made explicit rather than accidental. You will also have a decision rule for what to do when output is syntactically perfect and semantically wrong, an honest position on whether constraining hurts model quality, and a debugging technique, borrowed from a public vLLM investigation, that isolates prompt-layer bugs from model-layer bugs in about ten minutes.

This is post 20 of the [Inference Engineering series](/blog/machine-learning/inference-engineering/what-inference-engineering-is), and the third of three on constrained decoding. The [grammar-based decoding post](/blog/machine-learning/inference-engineering/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar) built the machinery that makes the syntax free. This one is everything the machinery does not buy you.

The usual note, repeated in every post of this series: I have no GPU and I have run nothing. Every quantity below is either derived from arithmetic shown in full, cited from a public source with a link, or written as a script for you to run with an expected direction rather than an invented value.

## 1. Three kinds of incomplete, stacked on top of each other

Start with the figure above, because the whole streaming problem is in it.

When a client is watching a structured response arrive, there are three different notions of "not finished yet" active at the same time, and they are not aligned:

1. **Bytes are incomplete.** A byte-level BPE tokenizer can split a multi-byte UTF-8 character across two tokens. The [tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) built an `IncrementalDetokenizer` whose entire job is to hold back the one to three bytes that do not yet form a complete character. Until they do, there is no text to give anyone.
2. **Tokens are incomplete.** The model has emitted 12 of the 27 tokens that will make up this object. The 12th token might be `":"` or `` "Han" `` or `` "],\n" `` — a token boundary has no relationship whatsoever to a JSON boundary.
3. **The JSON document is incomplete.** The decoded text so far is `{"city": "Hano` — a prefix of a valid document, and not itself a valid document. `json.loads` on it raises. It will keep raising for another 400 milliseconds.

Layer 1 is the tokenizer's problem and is already solved. Layer 2 is not really a problem so much as a fact — you receive text in irregular chunks, and the chunk boundaries are meaningless. **Layer 3 is the one nobody writes code for, and it is the one the user sees.**

The naive product implementation is to buffer the whole string, wait for the terminal event, call `json.loads` once, and render. That works, and it throws away every millisecond of the generation. Which brings us to the mechanism.

## 2. A JSON prefix is not a JSON document — until you close it

Here is the observation that makes streaming structured output tractable, and it is worth stating precisely because the sloppy version of it leads to bad code.

**Claim.** For any prefix $P$ of a valid JSON document, there exists a short suffix $S$ such that $P' \cdot S$ parses, where $P'$ is $P$ with at most one incomplete lexical token removed from its tail. Moreover $S$ is determined entirely by a single left-to-right scan of $P$ that tracks two things: the stack of open containers and whether the scan is currently inside a string.

The proof is basically the parser itself. JSON's grammar is a context-free grammar whose only recursion is through `{` and `[`; a valid prefix leaves a stack of unmatched openers, and appending their matching closers in reverse order discharges exactly that recursion. Everything else that can be unfinished is *lexical*, not structural: an unterminated string (close it with `"`), a partial number (`3.` or `1e`), a partial literal (`tru`, `nul`), a trailing separator (`,` or `:`) with nothing after it, or a key with a colon and no value yet. There are finitely many of those cases and each has an obvious repair — either complete it or drop it.

So the algorithm is: scan, then try a small ladder of increasingly aggressive repairs, and take the first that parses.

![Timeline of one JSON object arriving over seven decode steps with the open-container stack and the speculative closers appended at each step](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-2.webp)

The timeline above tracks a single object through its generation. At step 3 the buffer is `{"ci` — one open container, one open string. Close the string, close the brace, and you get `{"ci"}`, which does not parse, so the ladder drops the dangling key instead and yields `{}`. Correct: at step 3 you genuinely know nothing. At step 8 the buffer is `{"city": "Hanoi",` — the trailing comma gets trimmed, the brace gets closed, and the preview is `{"city": "Hanoi"}`. That is the first moment the client can render a real field, and it is *final*: nothing later in the stream can change it.

Here is the same thing in motion, which is the part a still frame cannot show — the closer riding the caret, and each field switching from provisional to final at the exact step its closing quote arrives.

<figure class="blog-anim">
<svg viewBox="0 0 660 210" role="img" aria-label="A JSON object arrives token by token while a speculative closing brace follows the caret, and each field turns from a dashed placeholder into a solid value the moment its closing quote arrives" style="width:100%;height:auto;max-width:820px">
<style>
.so-mono{font:16px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937)}
.so-cap{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.so-cover{fill:var(--background,#ffffff)}
.so-caret{stroke:var(--accent,#6366f1);stroke-width:2}
.so-brace{font:16px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--accent,#6366f1)}
.so-ghost{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 4}
.so-solid{fill:var(--accent,#6366f1);fill-opacity:.16;stroke:var(--accent,#6366f1);stroke-width:1.5}
.so-val{font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes so-rev{0%{transform:translateX(0)}85%,96%{transform:translateX(350px)}100%{transform:translateX(0)}}
@keyframes so-brace{0%,80%{opacity:1}88%,100%{opacity:0}}
@keyframes so-f1{0%,35%{opacity:0}40%,92%{opacity:1}97%,100%{opacity:0}}
@keyframes so-f2{0%,62%{opacity:0}67%,92%{opacity:1}97%,100%{opacity:0}}
@keyframes so-f3{0%,83%{opacity:0}88%,92%{opacity:1}97%,100%{opacity:0}}
.so-mv{animation:so-rev 10s ease-in-out infinite}
.so-bmv{animation:so-rev 10s ease-in-out infinite,so-brace 10s linear infinite}
.so-1{animation:so-f1 10s ease-in-out infinite}
.so-2{animation:so-f2 10s ease-in-out infinite}
.so-3{animation:so-f3 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.so-mv,.so-bmv,.so-1,.so-2,.so-3{animation:none}.so-mv{transform:translateX(350px)}.so-1,.so-2,.so-3{opacity:1}.so-bmv{opacity:0;transform:translateX(350px)}}
</style>
<text class="so-cap" x="28" y="26">decoded stream, one token per step</text>
<text class="so-mono" x="28" y="64">{"city":"Hanoi","unit":"C","days":3}</text>
<rect class="so-cover so-mv" x="28" y="44" width="640" height="30"/>
<line class="so-caret so-mv" x1="29" y1="44" x2="29" y2="74"/>
<text class="so-brace so-bmv" x="34" y="64">}</text>
<text class="so-cap" x="28" y="102">parsed fields</text>
<rect class="so-ghost" x="28" y="114" width="180" height="44" rx="8"/>
<rect class="so-ghost" x="224" y="114" width="180" height="44" rx="8"/>
<rect class="so-ghost" x="420" y="114" width="180" height="44" rx="8"/>
<rect class="so-solid so-1" x="28" y="114" width="180" height="44" rx="8"/>
<rect class="so-solid so-2" x="224" y="114" width="180" height="44" rx="8"/>
<rect class="so-solid so-3" x="420" y="114" width="180" height="44" rx="8"/>
<text class="so-val so-1" x="118" y="142">city: Hanoi</text>
<text class="so-val so-2" x="314" y="142">unit: C</text>
<text class="so-val so-3" x="510" y="142">days: 3</text>
<text class="so-cap" x="28" y="188">dashed = not yet closed · solid = closing quote seen, value is final</text>
</svg>
<figcaption>A speculative closing brace rides the caret so every prefix parses, and each field switches from dashed to solid at the exact step its closing quote arrives.</figcaption>
</figure>

### The scanner

The scanner is deliberately boring, and it must be *incremental* — carried across calls, folding in only the new characters. Section 4 shows what happens if it is not.

```python
# nanoserve/structured/preview.py
import json
import re
from dataclasses import dataclass, field

# a lexical token that could still be growing: 3 -> 34, tru -> true
_PARTIAL_TAIL = re.compile(r"[-+0-9.eEA-Za-z]+$")
# a key and its colon with no value yet:  {"a": 1, "b":
_DANGLING_KEY = re.compile(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$')
# a complete string in key position with no colon yet:  {"a": 1, "b"
_DANGLING_STR = re.compile(r',?\s*"(?:[^"\\]|\\.)*"\s*$')


@dataclass
class ScanState:
    """Everything a JSON prefix scan needs to remember between chunks."""

    stack: list = field(default_factory=list)  # '{' or '[' per open container
    in_string: bool = False
    escaped: bool = False
    chars: int = 0

    def advance(self, chunk: str) -> "ScanState":
        """Fold new characters into the state. Cost is O(len(chunk))."""
        for ch in chunk:
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
            elif ch == '"':
                self.in_string = True
            elif ch in "{[":
                self.stack.append(ch)
            elif ch in "}]":
                if self.stack:
                    self.stack.pop()
        self.chars += len(chunk)
        return self

    def closers(self) -> str:
        return "".join("}" if c == "{" else "]" for c in reversed(self.stack))
```

Note what is *not* in there: no key tracking, no value-position flag, no per-container state. Every extra piece of state is another place for the scanner to disagree with `json.loads`, and disagreements at this layer are exactly the class of bug that only shows up on one user's input. Keep the scanner minimal and let the real parser adjudicate.

### The repair ladder

```python
# nanoserve/structured/preview.py (continued)

def _candidates(buf: str, st: ScanState):
    """Progressively more aggressive repairs of a JSON prefix, cheapest first."""
    body = buf[:-1] if st.escaped else buf  # a lone trailing backslash
    if st.in_string:
        yield body + '"'  # 1. close the open string and keep it
    else:
        yield body  # 1. maybe the prefix is already well-formed
    if st.in_string:
        body = _DANGLING_STR.sub("", body)  # 2. or throw the open string away
        yield body
    trimmed = body.rstrip(" \t\r\n,:")
    yield trimmed  # 3. drop a trailing separator
    yield _DANGLING_KEY.sub("", trimmed)  # 4. drop a key with no value
    partial = _PARTIAL_TAIL.sub("", trimmed).rstrip(" \t\r\n,:")
    yield _DANGLING_KEY.sub("", partial)  # 5. drop a half-written literal


def preview(buf: str, st: ScanState):
    """Return the object a valid completion of `buf` would start with, or None."""
    tail = st.closers()
    for cand in _candidates(buf, st):
        try:
            return json.loads(cand + tail)
        except ValueError:
            continue
    return None
```

Two design points, both of which people get wrong.

**The ladder is ordered by information loss, not by likelihood.** Candidate 1 keeps everything; candidate 5 throws away a whole token. Trying the aggressive repair first would silently discard a `3` that was actually complete. Always take the least destructive repair that parses.

**`json.loads` is the oracle.** The scanner decides *what to append*; the standard library decides *whether it worked*. If a candidate parses, it is JSON — by definition, not by your regex's opinion. This is why the ladder can afford to be heuristic: a wrong guess costs one failed `json.loads` on a short string, not a corrupted render.

Run it, and every prefix has an answer:

```python
st = ScanState()
buf = ""
for chunk in ['{"ci', 'ty": "Han', 'oi", "un', 'it": "C", "days": 3', "}"]:
    buf += chunk
    st.advance(chunk)
    print(repr(buf), "->", preview(buf, st))
```

```console
'{"ci'                              -> {}
'{"city": "Han'                     -> {'city': 'Han'}
'{"city": "Hanoi", "un'             -> {'city': 'Hanoi'}
'{"city": "Hanoi", "unit": "C", "days": 3' -> {'city': 'Hanoi', 'unit': 'C', 'days': 3}
'{"city": "Hanoi", "unit": "C", "days": 3}' -> {'city': 'Hanoi', 'unit': 'C', 'days': 3}
```

Look at line 2 and line 4. Both are *correct previews* and both contain a value that is not final. `'Han'` will become `'Hanoi'`. `3` might become `30`. A client that renders those as settled facts will flicker, and a client that writes them to a database will be wrong. Which is the next problem.

## 3. From previews to field events: the contract that makes a UI easy

A preview object is a decent debugging tool and a bad API. The frontend does not want a new whole object 27 times; it wants to know **which fields are done**, so it can render each one once, in place, and never take it back.

The rule for finality falls straight out of the scan state, and it is pleasingly simple for the flat objects that dominate real schemas:

> A top-level key is **final** unless it is the last key in the preview *and* the scan is still inside its value.

"Still inside its value" is three conditions OR'd together: the scan is inside a string, or the container stack is deeper than the root object, or the buffer's tail is a run of characters that could still grow into a longer number or literal. Everything before the last key has already been closed by a comma, and JSON forbids revisiting it.

```python
# nanoserve/structured/preview.py (continued)

class StreamingObject:
    """Turns a text delta stream into per-field completion events."""

    def __init__(self):
        self.buf = ""
        self.st = ScanState()
        self.view = {}
        self.final = set()

    def _value_still_open(self, key) -> bool:
        if not self.view or key != next(reversed(self.view)):
            return False  # only the newest key can be unfinished
        if self.st.in_string or len(self.st.stack) > 1:
            return True
        return bool(_PARTIAL_TAIL.search(self.buf.rstrip()))

    def feed(self, delta: str):
        """Consume a decoded text delta; return a list of (event, key, value)."""
        self.buf += delta
        self.st.advance(delta)
        obj = preview(self.buf, self.st)
        if obj is None:
            return []
        events = []
        for key, value in obj.items():
            if key in self.final:
                continue
            if self._value_still_open(key):
                if self.view.get(key) != value:
                    events.append(("field.delta", key, value))
            else:
                self.final.add(key)
                events.append(("field.done", key, value))
        self.view = obj
        return events
```

That is the whole contract. `field.delta` means *this is a guess, render it greyed out or not at all*. `field.done` means *this will never change; commit it*. A client that ignores `field.delta` entirely is still correct — it just gets a coarser, chunkier experience — and that property is worth designing for, because half your clients will ignore it.

Wire it into the endpoint, downstream of the incremental detokenizer so the previewer never sees a broken character:

```python
# nanoserve/api.py
async def stream_structured(req, engine):
    so = StreamingObject()
    detok = IncrementalDetokenizer(engine.tokenizer)  # post 4
    token_ids = []
    async for step in engine.generate(req):
        token_ids.append(step.token_id)
        text = detok.push(step.token_id)  # "" while a character is incomplete
        if not text:
            continue
        for kind, key, value in so.feed(text):
            yield sse(kind, {"key": key, "value": value})
        yield sse("output.preview", {"object": so.view})
    yield sse(
        "output.done",
        {
            "object": so.view,
            "finish_reason": step.finish_reason,
            # section 10: hand back the ids so nobody has to re-tokenize
            "token_ids": token_ids,
        },
    )
```

The ordering matters. `detok.push` may return the empty string for a token, in which case there is no new JSON text and no event — the previewer must not be called with a partial character, or the scanner's `in_string` flag can flip on a replacement character that a later byte would have completed differently.

![Comparison of buffering a structured response to completion against emitting a tolerant preview at every step with time to first field on each side](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-3.webp)

#### Worked example: time to first field

Take the object the animation streams, in a realistic form:

```json
{"city": "Hanoi", "unit": "C", "days": 3, "alerts": ["heat advisory"]}
```

That is 69 characters. JSON tokenizes worse than prose because it is punctuation-dense — quotes, colons, braces and commas rarely merge into long tokens — so a Llama-3-class tokenizer lands near 2.5 characters per token on this kind of text rather than the ~4 you see on English. Call it 27 tokens, and check yours rather than trusting mine:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
s = '{"city": "Hanoi", "unit": "C", "days": 3, "alerts": ["heat advisory"]}'
print(len(s), len(tok.encode(s, add_special_tokens=False)))
```

The first field, `"city": "Hanoi"`, is complete at character 24 of 69, which is roughly token 8 of 27.

Now put a decode rate on it. For an 8B model in bf16 on a single RTX 4090 at low batch you should expect a time-per-output-token in the low tens of milliseconds; run the baseline benchmark from post 5 of this series and use your own figure. Taking 22 ms as the working number:

| Quantity                       | Value  | Source                                        |
| ------------------------------ | ------ | --------------------------------------------- |
| Object length                  | 69 chars | derived: counted                            |
| Tokens for the object          | ~27    | derived: 69 chars at ~2.5 chars/token         |
| Tokens to the first closed field | ~8   | derived: 24 of 69 chars                       |
| TPOT, 8B bf16, batch 1, 4090   | 22 ms  | reproduce: `bench.py` from post 5, expect 15–35 ms |
| Time to full object            | 594 ms | derived: 27 × 22 ms                           |
| Time to first field (TTFF)     | 176 ms | derived: 8 × 22 ms                            |
| Discarded by buffering         | 418 ms | derived: 594 − 176                            |

Four hundred and eighteen milliseconds of already-computed, already-correct information, thrown away because `json.loads` is all-or-nothing. On a larger object the gap widens roughly linearly with the number of fields, and for an object whose *first* field is the one the user is looking at — a title, a classification, a decision — buffering hides the answer behind the metadata.

**TTFF is the metric to put on the dashboard for any structured endpoint.** Time to first token is meaningless when the first token is `{`.

## 4. What the previewer costs, and how to measure it honestly

Everything above runs on the CPU, in the same process that is trying to keep a GPU fed. That is precisely the kind of cost that hides from `nvidia-smi` and shows up as a mysterious drop in tokens per second at high concurrency. So derive it before you ship it.

**The naive version is quadratic.** If you re-scan the entire buffer on every token — which is what you get if you write `preview(buf)` as a pure function of the string, the obvious first implementation — then over $N$ output tokens averaging $c$ characters each, the total character work is

$$
W_\text{naive} = \sum_{k=1}^{N} c \cdot k = \frac{c\,N(N+1)}{2} \approx \frac{c\,N^2}{2}.
$$

With the incremental `ScanState` above, each token folds in only its own characters:

$$
W_\text{inc} = \sum_{k=1}^{N} c = c\,N .
$$

The ratio is $N/2$. For a 400-token structured response that is a 200× difference in scanning work.

#### Worked example: the previewer's CPU budget at concurrency

Take a 400-token JSON output at 2.5 characters per token, so about 1,000 characters, and a Python character loop at roughly 30 ns per character — an order-of-magnitude figure for CPython, not a measurement; the point is the ratio, and you should replace it with your own number using the script below.

- Naive, per stream: $W = 2.5 \times 400^2 / 2 = 200{,}000$ character steps → about 6 ms per stream, but unevenly distributed: the *last* token alone re-scans 1,000 characters, costing about 30 µs.
- Naive, per engine step at batch 256 near the end of generation: $256 \times 30\ \mu s \approx 7.7$ ms of single-threaded Python, against a decode step budget of 22 ms. **Over a third of your step time, on the host, doing nothing but re-reading text you already read.**
- Incremental, per engine step at batch 256: $256 \times 2.5 \times 30\ \text{ns} \approx 19\ \mu s$. Rounding error.

That gap is the whole justification for carrying `ScanState` across calls. It is also a good illustration of a rule that runs through this whole series: the host-side work per token per stream is multiplied by concurrency, and concurrency is exactly the regime nobody tests in.

Measuring host-side cost honestly is easier than measuring GPU cost — no synchronization games, no CUDA events — but it has its own traps. This is the harness:

```python
# nanoserve/bench_preview.py
import time
import statistics
from nanoserve.structured.preview import ScanState, StreamingObject, preview

def sim_stream(obj_text, chunk=3):
    return [obj_text[i : i + chunk] for i in range(0, len(obj_text), chunk)]

def bench(obj_text, mode, reps=200):
    chunks = sim_stream(obj_text)
    per_chunk = []
    for _ in range(reps):
        so, buf, st = StreamingObject(), "", ScanState()
        for c in chunks:
            t0 = time.perf_counter_ns()
            if mode == "incremental":
                so.feed(c)
            else:  # rescan the world every time, the naive version
                buf += c
                preview(buf, ScanState().advance(buf))
            per_chunk.append(time.perf_counter_ns() - t0)
    return (
        statistics.median(per_chunk) / 1000,
        sorted(per_chunk)[int(0.99 * len(per_chunk))] / 1000,
    )
```

Four rules for making the number mean something:

- **Warm up.** The first few hundred iterations pay import cost, regex compilation and CPython's own warm-up on branch-heavy loops. Discard them.
- **Report p99, not the mean.** The naive version's cost grows monotonically within a stream, so its mean is dominated by short prefixes and hides the expensive tail. The tail is what collides with the decode step.
- **Measure per *chunk*, not per stream.** Your engine's deadline is per step. A total that looks fine spread over eight seconds can still blow a 22 ms budget at one moment.
- **Then measure it in the server, not the microbenchmark.** Host-side costs interact: the detokenizer, the previewer, the SSE serializer and the HTTP framing all share one event loop. The honest test is offered load against the real endpoint with open-loop arrivals, watching tokens per second at fixed concurrency — the methodology from [setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

One more cost worth naming: **event volume**. If you emit `output.preview` on every token, a 400-token response produces 400 SSE frames each containing the entire object so far. That is $O(N^2)$ bytes on the wire — the same quadratic, moved from CPU to network. Emit `field.done` events always, and `output.preview` at most every $k$ tokens or on a fixed time interval. The field events carry the meaning; the previews are a convenience.

## 5. The tool-call loop, end to end

Structured output stops being a formatting feature and becomes a control-flow feature the moment the object is a *tool call*. Now the string the model emits determines what code runs, and the result of that code goes back into the prompt. The loop has more surface area than people expect.

![Dataflow of a tool call showing the model emitting a call that is parsed and then either dispatched to execution or rejected with an error object, with both paths appending to the next prompt](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-4.webp)

Six things happen per iteration, and each is a place to be wrong:

1. **Render.** The tool schemas and the conversation history are turned into a prompt by the model's chat template. Section 6 is about what happens when this step is subtly wrong.
2. **Generate.** The model emits a call, typically as a name plus an `arguments` field that is *a JSON document inside a JSON string* — a nesting that surprises people the first time they see the escaping.
3. **Parse.** Extract the name, the arguments, and the call id. Section 6, bug 3, is about the id.
4. **Validate.** Is the name one of the tools declared *in this request*? Do the argument types match the schema? Section 11 is about the first half of that question.
5. **Execute.** Run the function. It can fail, time out, or return something enormous.
6. **Append.** Put the result back into the conversation as a tool message, and go to 1.

Here is the loop with all six steps visible and the failure paths made explicit:

```python
# nanoserve/structured/tools.py
import json
from dataclasses import dataclass

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


class ToolLoop:
    def __init__(self, engine, registry, max_iters=8, max_result_chars=8000):
        self.engine = engine
        self.registry = registry  # {name: (callable, json_schema)}
        self.max_iters = max_iters
        self.max_result_chars = max_result_chars

    def run(self, messages, tools):
        allowed = {t["function"]["name"] for t in tools}
        for it in range(self.max_iters):
            out = self.engine.chat(messages, tools=tools)
            if not out.tool_calls:
                return out  # the model answered instead of calling
            messages.append(out.assistant_message)
            for call in out.tool_calls:
                messages.append(self._dispatch(call, allowed))
        # the loop is a resource; failing to bound it is how you get a 400-turn
        # conversation and an OOM in the KV cache
        raise RuntimeError(f"tool loop exceeded {self.max_iters} iterations")

    def _dispatch(self, call: ToolCall, allowed: set) -> dict:
        if call.name not in allowed:
            # section 11: this is not a client error, it is a prompt problem
            return self._tool_msg(call, {"error": "tool_not_available",
                                         "available": sorted(allowed)})
        fn, schema = self.registry[call.name]
        ok, problem = validate(call.arguments, schema)
        if not ok:
            return self._tool_msg(call, {"error": "invalid_arguments",
                                         "detail": problem})
        try:
            result = fn(**call.arguments)
        except Exception as exc:  # a tool raising must not kill the request
            return self._tool_msg(call, {"error": "tool_failed",
                                         "detail": str(exc)[:400]})
        return self._tool_msg(call, result)

    def _tool_msg(self, call: ToolCall, payload) -> dict:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        if len(body) > self.max_result_chars:
            # truncating in the middle of JSON produces a string the model
            # will try to parse; say so explicitly instead
            body = body[: self.max_result_chars] + '\n... [truncated]'
        return {"role": "tool", "tool_call_id": call.id,
                "name": call.name, "content": body}
```

Three things in there are load-bearing and are the ones missing from most first drafts.

**Errors are formatted output, not exceptions.** Every failure path produces a well-formed tool message that goes back into the context. The model is remarkably good at recovering from `{"error": "invalid_arguments", "detail": "days must be >= 1"}` and completely helpless in the face of a dropped turn. Both exits in the figure merge at the same append step for exactly this reason: the error path is part of the generation contract, and it deserves the same care as the success path.

**The loop is bounded.** An unbounded tool loop is an unbounded context, and an unbounded context is an OOM in the KV cache — this is the failure mode the [eviction and preemption post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping) deals with from the engine side, and it is much cheaper to prevent from the API side.

**Results are truncated at a declared boundary.** Tool results are frequently the largest thing in the conversation. The vLLM team's Mooncake Store post (2026-05-06) reports statistics from 610 Codex and SWE-bench agentic traces that put numbers on this: an input-to-output token ratio of **131:1**, a median of **33 turns** per trace, roughly **2,242 tokens of context growth per turn**, and a **median context around 80K tokens by turn 30**. That growth is almost entirely tool results being appended. Truncate deliberately, or the scheduler will make the decision for you.

## 6. A taxonomy of real failures: the Kimi K2 tool-calling investigation

Now the part that is impossible to invent, and the reason this post exists in this series at all.

In October 2025 the vLLM team published [a debugging write-up on Kimi K2 tool-calling accuracy](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy). The setup: Kimi-K2-Instruct-0905 served on vLLM v0.11.0, evaluated on a tool-calling benchmark of over 1,200 attempts. Before the fixes, roughly **218 of those 1,200-plus calls succeeded — about 18%**. After three bug fixes, **1,007 succeeded**, which the post describes as a **4.4× improvement**. The post also reports completion rate at **99.925%** (3,997 of 4,000), Tool-Call F1 at **83.57%** (precision 81.96%, recall 85.24%), and schema accuracy at **76.00%**.

Sit with the shape of that for a moment. The model was fine the whole time. The grammar was fine. Eighty-two percent of tool calls failed because of three bugs, none of which were in the sampler, the mask, the kernel, or the weights.

![Taxonomy of one tool-calling symptom decomposing into a prompt-layer cause a serialization-layer cause and a parser-layer cause](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-5.webp)

What makes this a *taxonomy* rather than an anecdote is that the three bugs sit at three different layers, and each one is representative of a whole class.

### Bug 1 — the prompt layer: an argument that was silently dropped

vLLM inspected the signature of the tokenizer's `apply_chat_template` to decide which arguments to pass. Kimi's tokenizer hid some of those parameters behind `**kwargs`, so signature inspection did not see them, and **`add_generation_prompt=True` was never passed through**. The rendered prompt therefore lacked the assistant-turn tokens that tell the model it is now the assistant's turn to speak. The fix landed in [PR #27622](https://github.com/vllm-project/vllm/pull/27622).

The general lesson: **introspection-based argument passing is a silent-failure machine.** There is no error when a keyword argument is dropped — the function has a default, the default is used, and everything downstream is type-correct and semantically wrong. `**kwargs` is exactly the construct that defeats `inspect.signature`, and it is extremely common in tokenizer wrappers.

The symptom class is the worst one in this business: **the prompt was wrong and the model was blamed.** The output looks like a model that has forgotten how to call tools. Every instinct says to try a different model, tune the sampler, tighten the grammar. All of it is wasted, because the input was never what you thought it was.

The defensive pattern is to make the contract explicit and assert on the rendered artifact rather than trusting the plumbing:

```python
# nanoserve/structured/prompting.py
GEN_MARKER = "<|im_assistant|>"  # whatever your model's assistant-turn marker is

def render(tokenizer, messages, tools):
    text = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    # do not trust that the flag survived the call: check the artifact
    if not text.rstrip().endswith(GEN_MARKER):
        raise RuntimeError(
            "chat template did not append the generation prompt; "
            "the assistant turn marker is missing from the rendered text"
        )
    return text
```

Five lines, and the entire bug class becomes a loud startup failure instead of a 60-point accuracy drop.

### Bug 2 — the serialization layer: a Python list rendered into the prompt

The OpenAI-compatible message schema allows message content to be either a plain string or a list of typed content parts. vLLM normalized the former into the latter: a message with `content: ''` was rewritten to `[{'type': 'text', 'text': ''}]`. Kimi's Jinja chat template expected a string, and **rendered the list literally into the prompt** — Python's repr, brackets, quotes and all, dropped into the middle of a conversation. The fix was to type-check the content before rendering.

The general lesson: **a normalization that is invisible to the caller is visible to the model.** Every layer that rewrites a request for its own convenience is one more place where the prompt drifts from what the model was post-trained on. And unlike bug 1, this one is *right there in the text* — anyone who printed the rendered prompt would have seen `[{'type': 'text', 'text': ''}]` sitting in the conversation and known immediately.

Which is the practical takeaway, and it is embarrassingly simple: **log the fully rendered prompt in your development environment, and read it.** Not the message list. Not the request JSON. The exact string that gets tokenized.

### Bug 3 — the parser layer: a brittle split on a legal id

vLLM parsed tool-call ids with the equivalent of `function_id.split('.')[1].split(':')[0]`. That works on the format the parser's author had in front of them and raises `IndexError` on an id like `search:2`, which has no dot. Moonshot's API normalizes historical tool-call ids into a canonical form; vLLM did not, so ids from a conversation's history could reach the parser in a shape it had never been shown. The fix landed in [PR #27565](https://github.com/vllm-project/vllm/pull/27565).

The general lesson: **positional string surgery on an identifier is a crash waiting for a new client.** An id is an opaque token from an external system. The only safe operations on it are equality comparison and passing it along. If you must extract structure from one, the parse must be total — a regex that either matches or falls back, never an index into a split.

```python
# nanoserve/structured/tools.py (continued)
import re

_ID = re.compile(r"^(?P<ns>[A-Za-z0-9_]+)[.:](?P<name>[A-Za-z0-9_-]+)(?::(?P<seq>\d+))?$")

def parse_call_id(raw: str) -> dict:
    """Total function: every string maps to a record, nothing raises."""
    m = _ID.match(raw or "")
    if not m:
        # unknown shape is normal, not exceptional: keep the id verbatim
        return {"id": raw, "ns": None, "name": None, "seq": None}
    return {"id": raw, **m.groupdict()}
```

And normalize ids on the way *in*, at the request boundary, so that history from any client converges to one shape before it reaches anything that reads it. That is the actual root cause in the vLLM case: two systems disagreed about who owns normalization, so nobody did it.

### The pattern across all three

| Layer         | What broke                          | Why it was hard to see                          | The general defense                        |
| ------------- | ----------------------------------- | ----------------------------------------------- | ------------------------------------------ |
| Prompt        | Generation-prompt flag dropped by signature inspection | No error; a valid default was used | Assert on the rendered string, not the call |
| Serialization | Empty content normalized to a list, rendered literally | Visible only in the final prompt text | Log and read the exact tokenized string     |
| Parser        | Positional split on an id with no dot | Crashed only on ids from a specific client path | Total parsers; normalize at the boundary    |

Three layers, one symptom, zero overlap in the fixes. When a structured-output system underperforms, the useful first question is not "which knob" — it is **which layer**.

## 7. The debugging method worth stealing: bypass the server's templating

The vLLM post also describes *how* the team separated these, and the technique generalizes to any templated-prompt system. It is the single most useful thing in the write-up.

The problem with debugging a chat endpoint is that `/v1/chat/completions` does two things at once: it renders your messages into a prompt, and it generates from that prompt. When the output is bad, you cannot tell which half is at fault. So **take the rendering away from the server**: apply the chat template yourself, in your own process, where you can print it, and then post the resulting string to the raw `/v1/completions` endpoint, which does no templating at all.

If the output is now good, the bug is in the server's rendering. If it is still bad, the bug is in the model or the sampling. That is a clean bisection of a system that otherwise has no seam in it.

```python
# nanoserve/tools/bisect_prompt.py
"""Separate 'the prompt was wrong' from 'the model was wrong'."""
import json
import requests
from transformers import AutoTokenizer

BASE = "http://localhost:8000/v1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

messages = [...]  # your failing conversation
tools = [...]     # the tool schemas as sent

# 1. render it yourself, and LOOK AT IT
prompt = tok.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, tokenize=False
)
print("=== rendered prompt ===")
print(prompt)
print("=== end, %d chars / %d tokens ===" %
      (len(prompt), len(tok.encode(prompt, add_special_tokens=False))))

# 2. generate from that exact string, with no templating in the path
raw = requests.post(f"{BASE}/completions", json={
    "model": MODEL, "prompt": prompt, "max_tokens": 256, "temperature": 0.0,
}).json()["choices"][0]["text"]

# 3. generate the normal way, through the server's own rendering
chat = requests.post(f"{BASE}/chat/completions", json={
    "model": MODEL, "messages": messages, "tools": tools,
    "max_tokens": 256, "temperature": 0.0,
}).json()["choices"][0]["message"]

print("=== raw completion ===\n", raw)
print("=== chat completion ===\n", json.dumps(chat, indent=2))
```

Read the printed prompt character by character. That is where bug 2 would have jumped out. And the two outputs at the bottom answer the layer question directly:

| Raw completion | Chat completion | Conclusion                                          |
| -------------- | --------------- | --------------------------------------------------- |
| Good           | Bad             | The server's rendering is wrong — bugs 1 and 2 live here |
| Bad            | Bad             | The prompt you built is wrong, or the model cannot do the task |
| Good           | Good            | The bug is in parsing the output, not producing it — bug 3 lives here |
| Bad            | Good            | Your local template version differs from the server's |

Two extra refinements make this even sharper. Set `temperature=0` on both sides so the comparison is not sampling noise — and note that even at temperature zero the two paths can diverge for the batch-invariance reasons the determinism post in this track covers, so compare *behavior classes*, not exact strings. And diff the two prompts as **token ids**, not text, because whitespace differences that are invisible on screen are extremely visible to a tokenizer:

```python
a = tok.encode(prompt, add_special_tokens=False)
b = tok.encode(server_rendered_prompt, add_special_tokens=False)
import difflib
for line in difflib.unified_diff([str(t) for t in a], [str(t) for t in b], lineterm=""):
    print(line)
```

## 8. Valid JSON, wrong answer: coerce, retry, or repair

Grammar-constrained decoding guarantees syntax. It cannot guarantee semantics, and the gap is bigger than people assume. The vLLM Kimi K2 numbers make this concrete from a different angle: even after all three fixes, with 99.925% of requests completing, **schema accuracy was 76.00%** and Tool-Call F1 was 83.57%. Completing is not the same as being right.

The semantic failures you will actually meet:

- **Constraint violations the grammar cannot express.** `{"days": 0}` when the schema says integer and the business rule says positive. JSON Schema can express `minimum`, but your grammar compiler probably compiled only the *structural* subset — most do.
- **Enum near-misses.** `"celsius"` when the enum is `["C", "F"]`. A well-built grammar prevents this; a grammar built from a loose schema does not.
- **Cross-field inconsistency.** `{"start": "2026-08-01", "end": "2026-07-01"}`. No per-field constraint catches it.
- **Type-correct nonsense.** `{"city": "the city the user mentioned"}` — a string where a string belongs, containing an instruction rather than an answer.
- **Truncation.** The object hit `max_tokens` mid-generation. With a grammar this often *still* produces valid JSON, because the mask forced a close; you get a complete object missing half its fields, which is far more dangerous than a parse error.

Three responses, and they differ by three orders of magnitude in cost.

![Comparison table of coercion retry and repair across extra tokens added latency main risk and when to use each](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-6.webp)

**Coerce** — fix it in code. Lowercase the enum, clamp the integer, swap the reversed dates, parse the date string that came back as `01/08/2026`. Cost: zero tokens, microseconds. Risk: you are guessing at intent, and you will never find out you guessed wrong, because coercion produces no signal.

**Retry** — sample again, usually with a nudge in the prompt or a bumped temperature. Cost: a full generation.

$$
\mathbb{E}[\text{generations}] = \frac{1}{p}, \qquad
\mathbb{E}[\text{extra tokens}] = \left(\frac{1}{p} - 1\right) N_\text{out}
$$

where $p$ is the per-attempt validity rate. Risk: the tail. If $p = 0.92$ and $N_\text{out} = 400$, the mean extra cost is only $(1/0.92 - 1) \times 400 \approx 35$ tokens — under 9% — but 8% of requests take **two full generations**, which at 22 ms per token is 8.8 s instead of 4.4 s. Your p95 latency now sits on the retry path. Under load, a retry is also a *new admission*: it re-enters the queue, competes for KV blocks, and can trigger the preemption cascade described in the [admission control post](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse). The mean cost of retries is cheap; the variance is not.

**Repair** — send the bad output plus the validation error back to a model and ask for a corrected object. Cost: a new prefill over the original output plus repair instructions (call it 700 input tokens) plus a fresh generation. Risk: the repair model makes its own mistakes, and you now have two failure rates multiplying.

### The decision rule

Ordered, and the order is the whole point:

1. **If the fix is deterministic and lossless, coerce.** Case normalization, whitespace, number formatting, a date in a known alternative format, a unit conversion the schema fully specifies. These have exactly one correct answer and no ambiguity. Coerce, and **emit a metric** — `structured.coerced{field=...}` — because a coercion rate that climbs is your early warning that a prompt or model change has drifted.
2. **If the value is missing or self-contradictory and the output is short, retry** — but only if the validity rate is high enough that retries stay rare. Draw the line where the retry rate's contribution to p95 is acceptable: if $p \lt 0.9$, you have a prompt or schema problem, and adding retries is treating a systematic error as a random one.
3. **If the output is long and mostly right, repair.** Regenerating 2,000 tokens to fix one field is wasteful; a targeted repair call that returns only the corrected fields is much cheaper than either a retry or a full repair of the whole object.
4. **Never retry an unbounded number of times, and never retry inside the tool loop's iteration budget.** Two attempts total, then fail loudly with the validation error in the response. A silent third attempt is how a 4-second endpoint becomes a 30-second endpoint at p99.
5. **Push the constraint into the grammar wherever the grammar can hold it.** Every semantic check you can turn into a structural one costs zero at runtime. Enums are the big win here: compile the enum into the grammar as a literal alternation and the entire class of near-miss failures disappears at the mask.

Rule 5 deserves emphasis because it inverts the whole cost table. Coercion, retries and repairs are all *downstream* fixes. The grammar is the only place where a constraint costs nothing, because it is enforced by a mask that was going to run anyway.

## 9. Does constraining hurt quality? An honest answer

This is the question everyone asks and almost nobody answers carefully, so let me be explicit about what is mechanism and what is speculation.

### The mechanism by which it could hurt

Constrained decoding works by setting the logits of disallowed tokens to $-\infty$ and renormalizing over what remains. Three consequences follow, and all three are real:

**Renormalization over a subset changes the distribution, not just its support.** If the model's top choice is masked, the probability mass does not vanish — it is redistributed proportionally over the survivors. The token you sample is drawn from a distribution the model never produced. When the mask is tight (say, only three tokens allowed at a structural position) this is harmless: any of them continues the document. When the mask is tight at a *semantic* position — a value the model wanted to express differently — you are choosing among the model's less-preferred options with the top ones removed.

**A schema can forbid the model's natural reasoning path.** Many models produce noticeably better answers when they explain before concluding. A schema of the form `{"answer": "..."}` structurally forbids that: the first token after `{"answer": "` must be the answer itself. There is no room for the intermediate computation, and there is no mechanism by which the model can create room. This is a real and under-appreciated effect, and it has a real fix that costs nothing: **put a reasoning field first in the schema.** `{"reasoning": "...", "answer": "..."}` — because JSON objects are generated in order, the model gets to write its reasoning before it commits, and the schema still validates. Order the fields of a schema by *what the model needs to write first*, not by what reads nicely.

**Token-level masks interact badly with tokenization.** The token-versus-character mismatch from the FSM post applies directly here. A mask that is correct at the character level may force the model onto a token path it would never have taken — splitting a word into unusual pieces because the natural single token would have overshot a boundary. The model is now off the token distribution it was trained on, mid-word.

### The mechanism by which it helps

**Zero parse failures.** An unconstrained JSON-mode endpoint fails some fraction of the time — trailing commas, markdown fences around the object, a stray explanation before it. Every one of those is a total loss: a full generation billed and discarded, plus a retry. If the unconstrained parse-failure rate is $q$, constraining removes $q$ of your generations from the retry path outright. At $q = 0.05$ and 400-token outputs, that is a 5% token saving *before* counting the latency tail.

**No format drift over long conversations.** Unconstrained format compliance degrades with context length and with the number of prior turns. Constrained compliance does not degrade at all, because it is not a model behavior.

**Smaller models become usable.** A large part of what a bigger model buys you on structured tasks is format reliability. Take that away as a requirement and the quality gap on the *content* is often much narrower than the gap on the end-to-end success rate.

And on throughput, there is a citable number: the vLLM team's [structured decoding post](https://vllm.ai/blog/2025-01-14-struct-decode-intro) (2025-01-14) reports that the XGrammar backend delivers **up to 5× TPOT improvement under load** relative to the prior path, and notes that FSM compilation is a significant contributor to TTFT. So the *cost* of constraining is now mostly a one-time compilation cost, not a per-token one — which changes the calculus considerably from where it stood two years ago.

### What is not settled

**Whether constraining changes answer accuracy, for your task, is an empirical question with no universal answer, and I am not going to invent one.** The published claims on both sides tend to compare different things — constrained JSON against unconstrained free text, which conflates the format effect with the parse-failure effect — and the results depend heavily on schema shape, on whether a reasoning field is present, and on the model.

Here is what I would actually measure, and the design matters more than the tooling:

1. **Three arms, not two.** (a) unconstrained free text with a parser, (b) unconstrained JSON-mode with a parser, (c) grammar-constrained. Comparing only (a) and (c) tells you nothing, because it mixes format effects with parse-failure effects.
2. **Two metrics per arm.** *Conditional accuracy* — accuracy given the output parsed — and *end-to-end success rate* — accuracy over all requests, counting parse failures as failures. Constraining typically loses nothing or a little on the first and wins on the second. The second is the one your product experiences.
3. **Two schema variants in the constrained arm**: with and without a leading reasoning field. If the gap between them is large, your quality question was really a prompt-design question.
4. **Fixed sampling parameters and a fixed seed across arms**, with the caveats from the determinism post about batch-dependent numerics. Use greedy decoding for the accuracy comparison so sampling variance does not swamp the effect.
5. **A big enough n.** A 2-point accuracy difference on 200 examples is noise. Size the sample for the effect you care about detecting before you run anything.

If you run that on your own workload, you will have an answer that is worth more than every blog post on the subject, this one included.

## 10. Token ids, retokenization drift, and why agent loops care

There is a failure mode in agent and RL pipelines that looks like nondeterminism and is actually a lossy round trip.

Your engine generated a sequence of token ids. It detokenized them to text and handed you the text. Something downstream — a training loop, an evaluator, the next turn of an agent — takes that text and tokenizes it again. **The ids you get back are not guaranteed to be the ids the model emitted, even when the strings match exactly.**

The vLLM team's [Agent Lightning post](https://vllm.ai/blog/2025-10-22-agent-lightning) (2025-10-22) names three causes:

1. **Tokenization is not unique.** A string like `HAVING` can be produced by `H` + `AVING` or by `HAV` + `ING`. The BPE encoder picks one according to its merge ranks; the model may have emitted the other. Identical text, different ids.
2. **Tool-call serialization reformats whitespace.** Your loop parses a tool call into a dict and re-serializes it with `json.dumps`, which normalizes spacing after colons and commas. The text is now *different* from what the model wrote, and so is the tokenization.
3. **Chat templates differ across frameworks.** The framework that renders the training prompt is often not the framework that rendered the inference prompt, and small marker differences shift every token boundary after them.

Why this matters more than it sounds: the post notes that the resulting off-policy effect is "not even at the token level," so it **cannot be corrected through token-level importance sampling** — the standard correction for off-policy data does not apply, because the trajectory you are training on is not a re-weighting of the trajectory you sampled. It is a different trajectory that happens to print the same.

The fix is to stop round-tripping. Since **vLLM v0.10.2**, a request can set `"return_token_ids": true` and the response carries `prompt_token_ids` and `token_ids` alongside the text, backward-compatibly. Downstream consumers use the ids directly and never re-tokenize.

This is a good idea for any engine, `nanoserve` included, and it is nearly free — the ids are right there:

```python
# nanoserve/api.py (continued)
def build_response(req, out_ids, prompt_ids, text, finish_reason):
    body = {
        "text": text,
        "finish_reason": finish_reason,
    }
    if req.return_token_ids:
        body["prompt_token_ids"] = prompt_ids
        body["token_ids"] = out_ids
    return body
```

And a test that catches drift the moment it appears, which belongs in the CI of anything that feeds model output back into a model:

```python
# nanoserve/tests/test_no_retokenization_drift.py
def test_round_trip_is_lossless(tokenizer, engine):
    out = engine.generate(PROMPT, max_tokens=128, return_token_ids=True)
    reencoded = tokenizer.encode(out["text"], add_special_tokens=False)
    if reencoded != out["token_ids"]:
        i = next(k for k, (a, b) in enumerate(zip(reencoded, out["token_ids"]))
                 if a != b)
        raise AssertionError(
            f"drift at index {i}: re-encoded {reencoded[i]} "
            f"({tokenizer.decode([reencoded[i]])!r}) vs emitted "
            f"{out['token_ids'][i]} ({tokenizer.decode([out['token_ids'][i]])!r})"
        )
```

Run that once against a structured endpoint and you will likely find drift immediately, because JSON with its dense punctuation offers many near-equivalent merge paths — and because the tool-call re-serialization in cause 2 is something your own loop is doing.

Three practical rules follow:

- **Carry ids, not text, between stages of an agent loop** whenever both stages are yours.
- **Never re-serialize a tool call you are putting back into the prompt.** Append the model's own emitted string verbatim. Parse a *copy* for dispatch.
- **If you must re-tokenize, assert that it round-trips**, and treat a failure as a bug in the pipeline, not a curiosity.

## 11. Enforcement gaps: the tool the model should not have been able to call

The vLLM Kimi K2 post ends with an acknowledged limitation that is more interesting than any of the three bugs, because it is not a bug at all — it is a missing constraint.

Even after the fixes, the model would sometimes **call tools that were not present in the current request**. The post notes that Moonshot's own API prevents this with an "Enforcer" — a constrained-decoding layer that restricts generation to the declared tools — and that vLLM had no equivalent.

Understanding why this happens requires looking at what the prompt actually contains.

![Dataflow showing tool names from conversation history and tools declared in the current request merging into one prompt with a per-request mask as the only constraint on what the model may emit](/imgs/blogs/structured-output-in-production-streaming-json-and-tool-calls-7.webp)

A multi-turn conversation carries its own history. Turn 3 called `search_kb`. Turn 5 called `create_ticket`. Those calls and their results are in the message list, and the chat template renders them into the prompt, because the model needs them to understand what has already happened. So by turn 9, the prompt contains **seven tool names**: five from history plus the two declared now.

From the model's position, all seven are equally present in its context, and the tool-definitions block is just more text. It is a strong hint, not a constraint. Nothing in the token stream marks five of those names as expired. So the model does what a reasonable predictor does: it emits the name that best fits the situation, which is sometimes a name it saw four turns ago.

**Generalize it into a rule: your tool schema must be constrained per request, not per session.** If the constraint lives only in the prompt, it is advisory. Enforcement has to happen at the mask.

Concretely, in an engine:

```python
# nanoserve/structured/tools.py (continued)
def tool_choice_grammar(tools) -> str:
    """A GBNF-style grammar whose name alternation is exactly this request's tools.

    Rebuilt per request. A cached grammar keyed on the *session* is the bug.
    """
    names = " | ".join('"\\"%s\\""' % t["function"]["name"] for t in tools)
    return f'''
root      ::= "{{" ws "\\"name\\"" ws ":" ws name ws "," ws
              "\\"arguments\\"" ws ":" ws object ws "}}"
name      ::= {names}
'''  # object/ws rules elided; see the grammar post
```

Two implementation notes that matter more than the grammar text:

**Cache the compiled grammar on the tool set, never on the session.** Compilation is expensive — the vLLM structured-decoding post names FSM compilation as a significant TTFT contributor — so caching is necessary. Key the cache on a hash of the sorted tool names plus their schemas. Then a session that changes its tool set gets a different grammar automatically, and a session that does not gets a cache hit. Keying on `session_id` gives you a fast, wrong answer: the first turn's tools enforced forever.

**Handle the "no tool" case explicitly.** A grammar that forces a tool call means the model can never answer directly, which is usually wrong. The root rule needs an alternation between a tool call and free text, and that choice needs to be available at the first token — which is why `tool_choice` in the OpenAI-compatible API has three values (`auto`, `none`, `required`) and they compile to three different grammars, not to one grammar with a flag.

And even with all that, keep the runtime check from section 5's `_dispatch`. A grammar is a strong constraint; it is not an excuse to skip validation at the dispatch boundary, because the grammar can be disabled, mis-cached, or bypassed by a client that talks to your engine directly.

## 12. Stress tests

A structured-output implementation that works on a three-field object and two tools tells you very little. Here is what to push on, and what each one breaks.

### Sixty tools in one request

This is the normal condition for a mature agent, and it is expensive in two separate ways.

#### Worked example: what 60 tools cost

A moderately documented tool schema — name, description, four parameters with descriptions and types — runs around 450 characters of JSON. At roughly 3.5 characters per token for schema text, that is about 130 tokens per tool, so **60 tools is about 7,800 tokens of prompt** before the conversation even starts.

The KV cost, using the memory formula from earlier in the series for Llama-3.1-8B (32 layers, 8 KV heads, head dim 128, bf16):

$$
\text{bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{B} = 128\ \text{KiB}
$$

$$
7{,}800 \times 128\ \text{KiB} = 998{,}400\ \text{KiB} \approx 975\ \text{MiB}
$$

Just under a gigabyte of KV cache, permanently resident for the life of the conversation, holding tool definitions.

The prefill cost, using the standard $2 N_\text{params}$ FLOPs-per-token estimate: $2 \times 8 \times 10^9 \times 7{,}800 = 1.25 \times 10^{14}$ FLOPs. NVIDIA's A100 datasheet lists 312 TFLOP/s for dense bf16 on the A100 SXM; at a realistic 40% model FLOPs utilization for a prefill of this size, that is $1.25 \times 10^{14} / (1.25 \times 10^{14}) \approx 1.0$ second of prefill for the tool block alone.

| Quantity                       | Value      | Source                                          |
| ------------------------------ | ---------- | ----------------------------------------------- |
| Tokens per tool schema         | ~130       | derived: 450 chars at ~3.5 chars/token           |
| Prompt tokens for 60 tools     | ~7,800     | derived: 60 × 130                                |
| KV bytes per token, 8B bf16    | 128 KiB    | derived: 2·32·8·128·2 bytes                      |
| KV for the tool block          | ~975 MiB   | derived: 7,800 × 128 KiB                         |
| Prefill FLOPs                  | 1.25e14    | derived: 2 · 8e9 · 7,800                         |
| A100 SXM dense bf16 peak       | 312 TFLOP/s | cited: NVIDIA A100 datasheet                    |
| Prefill time at 40% MFU        | ~1.0 s     | derived: 1.25e14 / (312e12 × 0.4)                |
| Mask bytes per request per step | ~15.7 KiB | derived: ceil(128,256 / 8) bytes                 |

Two mitigations, and both are already in this series. **Prefix caching** turns that 1.0 s into near-zero for every request after the first, provided the tool block sits at the *front* of the prompt and is byte-identical across requests — which means sorting your tools deterministically and never interpolating a timestamp or request id into the block. That is the whole trick from [prefix sharing and radix trees](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write), applied to the one part of the prompt you fully control. And **tool retrieval**: select the 8 plausibly relevant tools per turn instead of shipping all 60, which cuts both costs by 85% at the price of a retrieval step that can be wrong.

The mask side is cheaper than people fear. A bitmask over a 128,256-token vocabulary is $\lceil V/8 \rceil = 16{,}032$ bytes, about 15.7 KiB per request per step. At batch 64 that is 1 MiB per step moved host-to-device — trivial against PCIe bandwidth. The cost of masking is grammar *compilation* and the per-step CPU work to advance 64 automata, not the mask transfer.

### Multiple tool calls in one turn

Modern models emit several calls in a single assistant turn. Three things break:

- **The streaming parser needs an array-of-objects mode.** Your `StreamingObject` handles one object; parallel calls arrive as a list, and a call is dispatchable the moment *its* object closes, not when the array does. Extend the finality rule one level: an array element is final when the scan pops back to the array's depth.
- **Dispatch order is not emission order.** If the calls are independent, run them concurrently and the loop gets much faster. If they are not — and the model has no way to tell you — you get a race. The safe default is sequential; concurrent dispatch should be opt-in per tool, declared by you, not inferred.
- **Results must come back in the order the ids expect.** Each result carries its `tool_call_id`; append them in the order the calls were emitted regardless of completion order, or the conversation reads as though the model called them in a different sequence than it did.

### A tool result that is itself JSON

Your search tool returns a JSON document. You put it in the `content` of a tool message, which is a JSON string inside a JSON message, which the template renders into a prompt. Three consequences:

- **Double escaping.** If you `json.dumps` an already-serialized string you get `"{\"a\": 1}"` in the prompt — visibly escaped, which the model handles poorly. Serialize exactly once.
- **The size explosion.** A 1,100-character result is roughly 318 tokens; at 33 turns that is over 10,000 tokens of results, consistent with the per-turn growth in the Mooncake agentic traces cited earlier. Truncate at a *structural* boundary — drop whole array elements, not characters — so what remains is still parseable.
- **Confusion with the model's own output format.** A tool result full of JSON braces sits in the same context as the JSON the model is being asked to produce. Wrapping results in an unambiguous delimiter, or summarizing large results into prose before appending, both help. Test it; do not assume.

### The user cancels mid-stream

A cancelled tool call is the nastiest case in this list, because state may already have escaped your process.

- **If the call has not been dispatched**, cancellation is clean: abort the sequence, free its KV blocks, drop the partial buffer.
- **If the call has been dispatched**, you may have already created the ticket, sent the email, charged the card. The client is gone; the side effect is not. **Tool execution must be idempotent, keyed on the tool call id**, so that a retried conversation does not double-execute. This is not an inference concern — it is a distributed-systems concern that inference dropped into your lap.
- **The partial object must not be committed.** Your `StreamingObject.view` at cancellation time contains speculatively-closed values. Anything written to durable storage must come from `field.done` events only. This is the entire reason the finality distinction exists rather than just streaming previews.
- **Free the resources.** A cancelled request that stays in the running set holds KV blocks and a compiled grammar. Wire cancellation through to the scheduler; the [continuous batching loop](/blog/machine-learning/inference-engineering/writing-a-continuous-batching-loop) needs to know.

### Streaming and cancellation, together

The vLLM team's [streaming requests and realtime API post](https://vllm.ai/blog/2026-01-31-streaming-realtime) (2026-01-31) describes a related contract from the other direction: input that arrives incrementally, where each chunk appends to a cumulative prompt that reuses cached KV, and the first chunk becomes a persistent "anchor request" that keeps the session's blocks from being evicted while the server waits for more input. The post notes the technique is limited to causal models, requires a model trained for streaming, and that vLLM does not yet preempt idle streaming sessions. If your product streams *both* directions — audio in, tools out — that idle-session pinning is the resource question to ask first.

## 13. Case studies and real numbers

Four public results worth knowing, each with its setup, because a number without a setup is decoration.

**Kimi K2 tool calling on vLLM** — [vLLM blog, 2025-10-28](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy). Kimi-K2-Instruct-0905 on vLLM v0.11.0. Successful tool calls went from about 218 of 1,200-plus attempts (~18%) to 1,007, described as a 4.4× improvement, after fixing three bugs: a chat-template argument dropped by signature inspection ([PR #27622](https://github.com/vllm-project/vllm/pull/27622)), an empty-content normalization rendered literally by the Jinja template, and a tool-call-id parser that raised `IndexError` on ids like `search:2` ([PR #27565](https://github.com/vllm-project/vllm/pull/27565)). Post-fix: completion rate 99.925% (3,997/4,000), Tool-Call F1 83.57% (P 81.96%, R 85.24%), schema accuracy 76.00%. Acknowledged limitation: the model still called tools absent from the current request, which Moonshot's API prevents with a constrained-decoding "Enforcer" that vLLM lacked.

**Structured decoding backends** — [vLLM blog, 2025-01-14](https://vllm.ai/blog/2025-01-14-struct-decode-intro). XGrammar's pushdown-automaton backend delivers up to 5× TPOT improvement under load versus the prior path; FSM compilation is a significant TTFT contributor. Backends compared: Outlines (token-level FSM, one token per step), XGrammar (PDA, the default where possible), lm-format-enforcer (fails in long context). Documented limitations at the time: XGrammar v0 was GBNF-only with no regex support, Outlines' batch mask blocked all requests in the batch, and the CFG mode could crash the engine. The v1 roadmap moved guided decoding to the scheduler level, computing the bitmask once and broadcasting it to GPU workers.

**Retokenization drift** — [vLLM blog, 2025-10-22](https://vllm.ai/blog/2025-10-22-agent-lightning). Detokenizing at inference and re-tokenizing at training can produce different ids even when the strings match, from non-unique tokenization, tool-call serialization reformatting whitespace, and chat-template differences across frameworks. The resulting off-policy effect is "not even at the token level" and cannot be corrected by token-level importance sampling. Fix: `"return_token_ids": true` in vLLM v0.10.2+, which adds `prompt_token_ids` and `token_ids` to the response.

**Agentic context growth** — [vLLM blog, 2026-05-06](https://vllm.ai/blog/2026-05-06-mooncake-store). Across 610 Codex and SWE-bench agentic traces: input-to-output token ratio 131:1, median 33 turns, roughly 2,242 tokens of context growth per turn, median context near 80K tokens by turn 30, inter-turn delay median 5.2 s with a P99 of 81.4 s. This is the shape of a tool loop as a serving workload: enormous prefill, tiny decode, long idle gaps between turns.

That last one reframes everything in this post. If your input-to-output ratio is 131:1, then **your structured-output endpoint is a prefill machine**, and the optimizations that matter are prefix caching on the tool block, keeping the tool definitions byte-stable, and not re-sending 60 schemas you do not need. The decode-side concerns — the mask, the previewer, the per-token CPU — are real but second-order at that ratio.

## 14. When to reach for this (and when not to)

**Build the tolerant streaming previewer when** a human is watching the object arrive and the first field is meaningful on its own. Extraction UIs, form fill, classification-with-explanation, anything where the object has more than about four fields. The 418 ms in the worked example is a real product difference, and the code is under 100 lines.

**Do not build it when** the consumer is another program. A backend service that receives your structured output does not care about previews; it wants the final object. Streaming previews to a machine consumer adds event volume, adds an $O(N^2)$ wire cost if you emit full previews, and buys nothing. Send the terminal object and be done.

**Do not build it when** the object is small. A three-field object at 22 ms per token completes in a few hundred milliseconds. Under about 300 ms, previewing is imperceptible and you have added a state machine to your hot path for nothing.

**Build the explicit tool loop with error results always** — this one has no exceptions. The version that raises on a bad tool call and drops the turn is strictly worse in every dimension, and it costs three lines to do properly.

**Use vLLM or SGLang rather than your own engine when** you need production tool calling. This series builds `nanoserve` to teach the mechanisms, and the mechanisms in this post are genuinely worth implementing yourself — the previewer and the tool loop are application-layer code that you should own regardless of engine. But the *grammar compilation cache*, the *per-step mask broadcast to GPU workers*, the *parser hardening across a dozen model families' tool-call formats*: that is a large, boring, high-value surface that a serving framework maintains for you. The Kimi K2 case study is a good demonstration of how much of it there is. Ship on a real engine and put your own code in the two layers around it.

**Reach for grammar constraints by default, but design the schema for the model, not for the reader.** Reasoning field first, enums compiled into the grammar, no field the model has to guess. Most of what gets blamed on constrained decoding is a schema authored in the wrong order.

**Do not use a grammar when** the task genuinely needs free-form reasoning and the structure is incidental. A model that must think for 500 tokens before it knows the answer should think for 500 tokens; wrap the structure around the reasoning rather than replacing it.

## 15. Key takeaways

- **A syntax guarantee is not a product guarantee.** Streaming, tool dispatch, semantic validity, and enforcement scope are four separate problems that a perfect grammar does not touch.
- **Every JSON prefix can be made parseable** by scanning for the open-container stack and appending its closers, with a short ladder of lexical repairs for the tail. Try the least destructive repair first and let `json.loads` be the oracle.
- **Distinguish provisional from final.** A top-level key is final unless it is the newest key and the scan is still inside its value. Emit `field.done` for final values and never write a provisional one anywhere durable.
- **Carry the scan state across chunks.** Re-scanning the buffer per token is $O(N^2)$; at batch 256 near the end of a 400-token generation that is a third of your step budget spent re-reading text.
- **Time to first field is the metric for a structured endpoint.** Time to first token is meaningless when the first token is a brace.
- **When structured output underperforms, ask which layer, not which knob.** The Kimi K2 investigation found a prompt bug, a serialization bug and a parser bug behind one symptom; the fixes had nothing in common.
- **Bisect a templated prompt by rendering it yourself and posting to the raw completions endpoint.** Good there and bad through chat means the server's rendering is at fault. Print the prompt and read it.
- **Never do positional string surgery on an identifier.** Total parsers with a fallback, and normalize ids at the request boundary so two systems never both assume the other did it.
- **Coerce, then retry, then repair, in that order** — and push every constraint you can into the grammar, because that is the only place a constraint is free.
- **Constrain tools per request, not per session.** History puts old tool names in the prompt, and a prompt is advice, not enforcement. Key the compiled-grammar cache on the tool set, never on the session.
- **Hand back token ids.** Text round trips are lossy in ways that break RL and agent loops, and returning `token_ids` costs nothing.
- **At an input-to-output ratio of 131:1, a tool loop is a prefill workload.** Prefix-cache the tool block, keep it byte-stable, and stop shipping 60 schemas when 8 will do.

## Further reading

- [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro) — the FSM and pushdown-automaton backends, their limits, and the TPOT numbers quoted above.
- [Kimi K2 Accuracy Improvements on vLLM](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy) — the three-bug investigation, the before-and-after tool-calling numbers, and the Enforcer limitation.
- [Agent Lightning: returning token ids](https://vllm.ai/blog/2025-10-22-agent-lightning) — retokenization drift, why importance sampling cannot fix it, and the `return_token_ids` API.
- [Mooncake Store: distributed KV cache for agentic workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) — the agentic trace statistics that describe what a tool loop looks like as a serving workload.
- [Streaming requests and the realtime API in vLLM](https://vllm.ai/blog/2026-01-31-streaming-realtime) — incremental input, anchor requests, and the idle-session pinning caveat.
- [Grammar-based decoding: GBNF, pushdown automata and XGrammar](/blog/machine-learning/inference-engineering/grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar) — the machinery that makes the syntax free, which this post assumes.
- [The tokenizer boundary and incremental detokenization](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) — the byte-level layer beneath the previewer.
- [Stop conditions, EOS handling and thinking budgets](/blog/machine-learning/inference-engineering/stop-conditions-eos-handling-and-thinking-budgets) — the next post, on how generation actually ends.
- [The inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — the series capstone, where every layer meets.
