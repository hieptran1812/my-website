---
title: "Stop conditions, EOS handling, and thinking budgets: knowing when to shut up"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Every output token costs a full pass over your weights, so the code that decides when generation ends is the highest-leverage code in your engine. Build a stopping-criteria stack that handles multiple end tokens, split stop strings, runaway loops, thinking budgets, and clients who hang up."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "decoding",
    "tokenizer",
    "reasoning",
    "latency",
    "throughput",
    "pytorch",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 55
---

There is a bug I have seen in every LLM serving stack I have ever touched, and it never shows up as a crash. The model finishes its answer at token 214. It writes a period, and then — because the engine was told to watch for the wrong integer — it keeps going. It writes a newline. It writes `Human:`. It hallucinates the next turn of the conversation, then answers itself, then does it again, until token 4,096, where `max_tokens` finally cuts it off mid-word. The client gets a response with `finish_reason: "length"` and a wall of garbage after the good part. Nobody pages anyone. The dashboard shows healthy throughput. The bill shows an extra 3,882 tokens per affected request.

That is one integer wrong in a config file, and it is a 19× cost multiplier on that request.

Stopping is where three problems meet that are usually treated separately. It is a **correctness** problem: an answer that runs past its turn is wrong even if every token in it was sampled correctly. It is a **cost** problem: the decode phase of inference is linear in output length and nothing else, so unnecessary tokens are the one waste that scales one-for-one with your GPU bill. And it is a **UX** problem: the difference between a stream that ends cleanly and one that ends by having its last eight characters retracted is the difference between a product and a demo. Most engines get one of the three right.

![A branching diagram of the five distinct causes that end a generation grouped by whether the model, the policy layer, or the client decides](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-1.webp)

The figure above is the whole surface area. Five causes; exactly one of them — the model emitting an end-of-turn id — is something the model does. The other four are policy that *you* write, and each one has a failure mode that is silent by construction. This post writes all of them. By the end you will have `nanoserve/stopping.py`: a `StoppingCriteria` stack composing end-token ids, stop strings with the hold-back buffer from [the tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization), a hard token ceiling, a repetition-cycle detector, and a thinking budget with budget forcing — plus cancellation wired into the step loop so a client who closes their laptop stops costing you money on the very next iteration rather than 28 seconds later. You will also be able to derive, from your GPU's bandwidth and your GPU-hour rate, exactly what 300 unnecessary tokens cost you, which turns out to be the argument that gets this work prioritized.

This is the last post in the decoding-layer track. [The sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) decided *which* token; [constrained decoding and structured output](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) decided which tokens were *legal*; this one decides *when there are no more*. If you have not read [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), that post frames the scoreboard — TTFT, TPOT, tokens per second, memory, goodput, dollars per million tokens — that everything here moves.

## 1. The bill: output length is the only axis that scales linearly

Start with the economics, because the economics decide how much of this post you should implement.

A decode step generates one token for every sequence in the batch, and to do it the GPU must read every weight in the model out of HBM. That is the decode floor derived in [the baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline): the step time cannot go below the weight bytes divided by the memory bandwidth,

$$
t_{\text{step}} \;\gtrsim\; \frac{B_{\text{weights}}}{\text{BW}_{\text{HBM}}},
$$

and for Llama-3.1-8B in bf16 that is 16.06 GB of weights against 1.008 TB/s on an RTX 4090 — 15.9 ms per step — or against the A100 80GB's 2.039 TB/s — 7.9 ms per step. Both derived, both independent of batch size, because the *same* weight read serves every sequence in the batch.

That last clause is the entire economic argument. Prefill work scales with input tokens but is amortized across the batch and across cache hits; decode work scales with output tokens and is amortized across *nothing*. One extra output token is one extra full pass over 16 GB. Three hundred extra output tokens on one request is 300 extra passes.

![A four-row comparison of chat, retrieval answering, code completion, and agent traffic showing the input to output ratio, what drives the cost, and which stop lever matters](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-2.webp)

Which lever matters depends on the shape of your traffic, and the shapes differ more than people expect. A retrieval-augmented answer might be 20 input tokens for every output token, so its cost lives in prefill and its stopping story is mostly "cap `max_tokens` and move on." An agent loop is the opposite kind of extreme, and here there is a real published number: the vLLM team's [Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) reports that across 610 Codex and SWE-bench agentic traces the aggregate input-to-output ratio is **131:1**, with a median of **33 turns** per trace and roughly 2,242 tokens of context growth per turn. Read that ratio the wrong way and you conclude output does not matter. Read it correctly and you notice *why* the input side is so large: it is large because the same conversation is re-sent 33 times, and it is re-sent 33 times because each turn generated output that became input. Prefix caching makes the input side cheap to re-read. Nothing makes the output side cheap.

#### Worked example: what 300 wasted tokens actually cost

Take the A100 80GB running Llama-3.1-8B in bf16, and assume for planning purposes a rate of **\$2.00 per GPU-hour** — substitute your own; the structure of the arithmetic is what matters, not my placeholder.

At batch 1 the step floor is 7.9 ms, so the card produces about 127 tokens per second. At batch 32 the weight read is amortized 32 ways, the step time is still roughly 7.9 ms until the batch grows large enough to become compute-bound, and aggregate throughput is therefore

$$
\frac{32\ \text{tokens}}{7.9\ \text{ms}} \;=\; 4{,}051\ \text{tok/s} \quad\text{(derived)} .
$$

One GPU-hour at that rate produces $4{,}051 \times 3600 \approx 14.6$ million output tokens, so the *hardware floor* on output cost, in dollars per million output tokens, is

$$
\frac{2.00}{14.6 \times 10^{6}\ \text{tokens}} \times 10^{6} \;\approx\; 0.137 \quad\text{(derived)} .
$$

So roughly 14 cents per million output tokens of pure hardware cost, at that assumed rate. Now the 300 wasted tokens: $300 \times 1.37 \times 10^{-7} \approx 4.1 \times 10^{-5}$ dollars per request. That is four thousandths of a cent, which sounds like nothing until you multiply. At one million requests per day it is **\$41 per day, about \$15,000 per year**, for a bug that is one entry in a JSON file. And the dollar figure understates it, because those 300 tokens also occupied a batch slot for $300 \times 7.9\ \text{ms} = 2.37$ seconds. At batch 32 that is 9,600 token-slots that could have served other requests — the goodput cost, which is the number your users feel.

#### Worked example: trimming an agent loop

Take the Mooncake trace shape as the workload: 33 turns per session. Suppose each turn generates 250 output tokens, giving 8,250 output tokens per session (this 250 is my assumption for the example, not a cited figure — the cited numbers are the 131:1 ratio and the 33-turn median). Cutting output by 20% through a thinking budget and a correct end-token set saves 1,650 tokens per session.

In money, at the derived floor: $1{,}650 \times 1.37 \times 10^{-7} \approx 2.3 \times 10^{-4}$ dollars per session. Trivial. In *time*, at 7.9 ms per token: $1{,}650 \times 7.9\ \text{ms} \approx 13$ seconds removed from a session's total latency. That is the number that matters. Agent sessions are latency-bound loops where a human or another system waits on each turn, and 13 seconds is the difference between a tool that feels responsive and one that people stop using. **The reason to care about stopping in agent workloads is wall-clock, not cents.**

| Workload | Where the money is | Where the latency is | First lever to pull |
| --- | --- | --- | --- |
| Chat, short in / long out | output tokens | output tokens | correct end-token set |
| RAG, long in / short out | prefill + KV memory | TTFT | `max_tokens` cap, prefix cache |
| Code completion | TTFT | TTFT | stop strings, tight ceiling |
| Agent loop, many turns | amortized by prefix cache | total output across turns | thinking budget |

## 2. EOS is not one token, and that is the bug

The most common stopping bug in production is a plural noun treated as singular.

A base language model trained on raw documents has one end marker: end of text. An instruction-tuned chat model has at least two, because it needs to express two different ideas. "This turn is over, hand control back to the user" is not the same statement as "this document is over." A model that only knows the second one will happily keep writing after the assistant's turn, generating the user's next message itself, because from its point of view the document has not ended. Meta's Llama-3.1 instruct models carry `<|end_of_text|>` and `<|eot_id|>` as separate special tokens for exactly this reason, plus `<|eom_id|>` for the end of a message that expects a tool result to come back — you can see all of them in the [model card's special-token list](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Qwen's chat models use `<|im_end|>`. Gemma uses `<end_of_turn>`. Every family solves the same problem with a different string.

![A flow where one sampled identifier reaches two different stop sets and the thin one misses the end of turn and keeps generating before both paths rejoin at the client](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-3.webp)

The engine does not see any of those strings. The engine sees integers. So the question "did the model just end its turn?" is the question "is this integer in a set I was given?", and if the set is wrong the model does not stop. There is no exception thrown, no warning logged, no test that fails. The output just gets longer and worse.

There are four places the set can come from, and they disagree:

1. `tokenizer.eos_token_id` — a single integer, usually the *document* end token. This is the one people use, and it is usually the wrong one for a chat model.
2. `generation_config.eos_token_id` — increasingly a **list**. This is the authoritative source when it exists, because it is what the model publisher intends `generate()` to use.
3. The chat template itself — whatever token the Jinja template emits at the end of an assistant turn. If the template ends the turn with `<|eot_id|>`, that is the token to stop on, regardless of what any config says.
4. Whatever the client passed in the request as extra stop tokens.

Resolve all four, union them, and log the result at startup:

```python
# nanoserve/stopping.py
"""Everything that decides a generation is over."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("nanoserve.stopping")

# Turn-ending markers used across the major open families. We probe for
# all of them because a checkpoint's config is not always complete, and a
# missing turn marker is a silent 10x cost bug, not a crash.
TURN_END_MARKERS = (
    "<|eot_id|>",        # Llama 3.x, end of turn
    "<|eom_id|>",        # Llama 3.x, end of message (tool call pending)
    "<|im_end|>",        # Qwen, ChatML
    "<end_of_turn>",     # Gemma
    "<|end|>",           # Phi
    "<|endoftext|>",     # GPT-2 lineage, also used as turn end by some
)


def resolve_end_ids(tokenizer, generation_config=None,
                    extra: list[int] | None = None) -> frozenset[int]:
    """Union every source that can legitimately end a turn.

    Order matters only for logging. A superset is safe: an id that is
    never sampled costs one integer comparison per step. A missing id
    costs thousands of decode steps.
    """
    ids: set[int] = set()
    sources: dict[str, list[int]] = {}

    gc_eos = getattr(generation_config, "eos_token_id", None)
    if isinstance(gc_eos, int):
        gc_eos = [gc_eos]
    if gc_eos:
        sources["generation_config"] = list(gc_eos)
        ids.update(gc_eos)

    if tokenizer.eos_token_id is not None:
        sources["tokenizer"] = [tokenizer.eos_token_id]
        ids.add(tokenizer.eos_token_id)

    unk = tokenizer.unk_token_id
    probed = []
    for marker in TURN_END_MARKERS:
        tid = tokenizer.convert_tokens_to_ids(marker)
        # convert_tokens_to_ids returns the unk id (or None) for tokens
        # this tokenizer does not have. Both must be filtered.
        if tid is not None and tid != unk:
            probed.append(tid)
    if probed:
        sources["probe"] = probed
        ids.update(probed)

    if extra:
        sources["request"] = list(extra)
        ids.update(extra)

    log.info("end ids resolved: %s from %s", sorted(ids), sources)
    return frozenset(ids)
```

Two design decisions in there worth defending.

**We take the union, not a priority order.** The asymmetry of the errors is brutal: including an id that this model never samples costs one integer comparison per decode step, which is unmeasurable. Excluding an id the model *does* sample costs the difference between 214 tokens and 4,096. When the costs are that lopsided, be greedy.

**We probe by string, not by id.** Hardcoding `128009` works until you serve a fine-tune whose author extended the vocabulary, at which point the id is something else and the string is the same. Probing is one dictionary lookup per marker at startup.

There is a third failure that the union does not fix, and it is upstream: **if the prompt is not rendered with the model's chat template, the model may never emit a turn-end token at all**, because nothing in its context looks like a turn it should end. This is not hypothetical. The vLLM team's [Kimi K2 accuracy post](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy) (2025-10-28) documents a bug where vLLM inspected the signature of `apply_chat_template` and silently dropped arguments that the model's template accepted through `**kwargs`, so `add_generation_prompt=True` never reached the template and the assistant-turn tokens were missing from the prompt. On Kimi-K2-Instruct-0905 the reported effect was that successful tool calls went from roughly 218 out of 1,200-plus to 1,007 once the template bugs were fixed — a change they describe as 4.4×. The stopping behaviour and the template are the same subsystem viewed from two ends, which is why [the tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) and this one keep referring to each other.

### Should the end token appear in the output?

Almost always no, and this is a one-line decision with a surprising downstream consequence.

The end token should not be in the *text* returned to the client — nobody wants `<|eot_id|>` in their chat bubble. But when you append the assistant's turn to the conversation history for the next request, the turn must be terminated, and the cleanest way to terminate it is with the exact token id the model generated. If instead you return a string, let the client re-render it through the chat template, and re-tokenize, you can get a different id sequence for the same visible text. The vLLM team's [Agent Lightning post](https://vllm.ai/blog/2025-10-22-agent-lightning) (2025-10-22) names this drift precisely: tokenization is not unique, so detokenizing at inference and retokenizing later may produce different ids even when the strings match, and they added `"return_token_ids": true` so callers can carry the exact ids forward.

For stopping, the practical consequence is about your prefix cache. A multi-turn conversation whose history is rebuilt from strings can hash differently than the ids you actually served, and a prefix cache that keys on token ids will miss. Keep the ids; strip the end token from the *rendered text* only.

## 3. Stop strings live in characters; generation lives in tokens

The second mechanism is client-supplied stop strings, and it has a mismatch at its heart: `stop: ["\n\nHuman:"]` is a statement about characters, while the engine produces tokens, and no token boundary is obliged to line up with the string.

[The tokenizer boundary post](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) derived the fix, so I will restate the result rather than re-derive it: if the longest stop string is $L$ characters, any partially-matched stop string must begin within the last $L-1$ characters of accumulated output, so releasing everything except the last $L-1$ characters is both **safe** and **minimal**. Hold back fewer and there exists a token split that leaks; hold back more and you add latency for nothing.

![A two column comparison of a stream with no hold-back leaking five characters of a stop string against a stream holding seven characters and cutting cleanly](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-4.webp)

What that post did not cover is the engine-side consequence, which is the interesting half: **hold-back converts a correctness requirement into a latency cost, and the exchange rate is set by the client.** Every character you withhold is a character the user cannot see yet. Here is the mechanism running:

<figure class="blog-anim">
<svg viewBox="0 0 700 260" role="img" aria-label="An engine stream produces four tokens spelling a stop string while the client stream receives only the first, and the withheld tail is discarded when the match completes" style="width:100%;height:auto;max-width:820px">
<style>
.s1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.s1-hit{fill:var(--accent,#6366f1);opacity:.16;stroke:var(--accent,#6366f1);stroke-width:1.5}
.s1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.s1-lane{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.s1-badge{font:700 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.s1-note{font:400 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.s1-hold{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5;stroke-dasharray:6 4}
@keyframes s1-e1{0%,8%{opacity:0}12%,92%{opacity:1}100%{opacity:0}}
@keyframes s1-e2{0%,22%{opacity:0}27%,58%{opacity:1}68%,92%{opacity:.16}100%{opacity:0}}
@keyframes s1-e3{0%,37%{opacity:0}42%,58%{opacity:1}68%,92%{opacity:.16}100%{opacity:0}}
@keyframes s1-e4{0%,52%{opacity:0}57%,58%{opacity:1}68%,92%{opacity:.16}100%{opacity:0}}
@keyframes s1-c1{0%,18%{opacity:0}23%,92%{opacity:1}100%{opacity:0}}
@keyframes s1-win{0%,22%{opacity:0}27%,60%{opacity:1}70%,100%{opacity:0}}
@keyframes s1-fin{0%,62%{opacity:0}68%,92%{opacity:1}100%{opacity:0}}
.s1-a1{animation:s1-e1 11s ease-in-out infinite backwards}
.s1-a2{animation:s1-e2 11s ease-in-out infinite backwards}
.s1-a3{animation:s1-e3 11s ease-in-out infinite backwards}
.s1-a4{animation:s1-e4 11s ease-in-out infinite backwards}
.s1-b1{animation:s1-c1 11s ease-in-out infinite backwards}
.s1-w{animation:s1-win 11s ease-in-out infinite backwards}
.s1-f{animation:s1-fin 11s ease-in-out infinite backwards}
@media (prefers-reduced-motion:reduce){.s1-a1,.s1-b1,.s1-f{animation:none;opacity:1}.s1-a2,.s1-a3,.s1-a4{animation:none;opacity:.22}.s1-w{animation:none;opacity:1}}
</style>
<text class="s1-lane" x="20" y="40">engine text</text>
<g class="s1-a1"><rect class="s1-box" x="20" y="55" width="150" height="52" rx="8"/><text class="s1-t" x="95" y="87">answer.</text></g>
<g class="s1-a2"><rect class="s1-box" x="182" y="55" width="86" height="52" rx="8"/><text class="s1-t" x="225" y="87">\n\n</text></g>
<g class="s1-a3"><rect class="s1-box" x="280" y="55" width="132" height="52" rx="8"/><text class="s1-t" x="346" y="87">Human</text></g>
<g class="s1-a4"><rect class="s1-box" x="424" y="55" width="60" height="52" rx="8"/><text class="s1-t" x="454" y="87">:</text></g>
<rect class="s1-hold s1-w" x="176" y="49" width="314" height="64" rx="10"/>
<text class="s1-note s1-w" x="333" y="132">held back: 7 chars could still start the stop string</text>
<text class="s1-lane" x="20" y="184">client stream</text>
<g class="s1-b1"><rect class="s1-hit" x="20" y="199" width="150" height="52" rx="8"/><text class="s1-t" x="95" y="231">answer.</text></g>
<g class="s1-f"><text class="s1-badge" x="360" y="231">match at 8 chars, drop the tail, finish_reason: stop</text></g>
</svg>
<figcaption>The engine sees the stop string arrive across four tokens; the client lane never receives any of it, because the withheld tail is discarded the moment the match completes.</figcaption>
</figure>

The `StopChecker` from the tokenizer post handles the common case correctly, but it has a bug that only appears when a request supplies more than one stop string, and it is worth fixing here because the fix is two lines and the symptom is bizarre. That implementation iterates the stop list in the order the client sent it and returns on the first `find` that succeeds. If the client sends `stop: ["END", "\n\nHuman:"]` and the output contains `"...\n\nHuman: ... END"`, the loop matches `"END"` at a *later* index than `"\n\nHuman:"` and truncates in the wrong place. The client asked for the earliest occurrence of any stop string; list order is not a tiebreak they know about.

```python
# nanoserve/stopping.py
class StopStrings:
    """Detects the LEFTMOST occurrence of any stop string, holding back
    the provably minimal tail so nothing leaks over SSE.

    Composes with IncrementalDetokenizer: feed it detokenized text, never
    tokens. Stop strings are defined over characters.
    """

    def __init__(self, stops: list[str], include_in_output: bool = False,
                 max_len: int = 64, max_count: int = 8):
        stops = [s for s in stops if s][:max_count]
        too_long = [s for s in stops if len(s) > max_len]
        if too_long:
            raise ValueError(
                f"stop string exceeds {max_len} chars: {too_long[0]!r}")
        self.stops = stops
        self.hold = max((len(s) for s in stops), default=1) - 1
        self.include = include_in_output
        self.pending = ""
        self.matched: str | None = None

    def push(self, text: str) -> tuple[str, bool]:
        """Returns (text safe to emit now, stopped)."""
        if not text:
            return "", False
        self.pending += text

        # Leftmost match across ALL stops, not first match in list order.
        best_i, best_s = None, None
        for s in self.stops:
            i = self.pending.find(s)
            if i >= 0 and (best_i is None or i < best_i):
                best_i, best_s = i, s
        if best_i is not None:
            end = best_i + (len(best_s) if self.include else 0)
            out, self.pending, self.matched = self.pending[:end], "", best_s
            return out, True

        if self.hold == 0:
            # No stops, or only single-character stops: hold nothing.
            # NOT optional -- s[:-0] is the empty string in Python, so
            # the slice below would swallow the entire stream.
            out, self.pending = self.pending, ""
            return out, False
        if len(self.pending) > self.hold:
            out = self.pending[: -self.hold]
            self.pending = self.pending[-self.hold :]
            return out, False
        return "", False

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return out
```

Note the validation in the constructor. A client that supplies a 500-character stop string is asking you to buffer 499 characters before showing anything, and a client that supplies 200 stop strings turns the per-token scan into real CPU work multiplied by your batch size. Both are trivially defensible at the API layer, and **eight stop strings of at most 64 characters is a limit no legitimate caller notices.** vLLM and every other production engine impose comparable limits; ours is explicit and raises a 400 rather than degrading quietly.

The latency cost is worth stating as a rule, because it is the one place where correct stopping makes the product feel worse:

| Stop configuration | Hold-back | What the user perceives |
| --- | --- | --- |
| End tokens only, no stop strings | 0 chars | perfectly smooth |
| `"\n\n"` | 1 char | indistinguishable |
| `"\n\nHuman:"` | 7 chars | slightly bursty, roughly 2 tokens of lag |
| A 64-char sentinel | 63 chars | visibly laggy, ~16 tokens of lag |

At a TPOT of 20 ms and roughly four characters per English token, 63 characters of hold-back is about 16 tokens, or **320 ms of added perceived latency before every visible update**. That is not a rounding error; it is the difference between a p50 TTFT that meets your SLO and one that does not, and it is entirely self-inflicted by an API that accepts arbitrary stop strings.

## 4. `max_tokens`, and telling the truth in `finish_reason`

`max_tokens` is the blunt instrument. It does not know anything about the content; it counts. Which is exactly why you need it: it is the only stop condition that cannot be defeated by a model behaving strangely, and it is therefore the one that bounds your worst case. Every other mechanism in this post is an optimization on top of a ceiling that must exist.

The problem with a ceiling is where it lands. Cutting a generation at an arbitrary token index truncates whatever was in progress:

- **Mid-sentence.** Annoying, recoverable, the client can display it.
- **Mid-JSON.** Fatal for a machine consumer. The response parses as nothing. This is why the [structured output post](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) treats truncation as a first-class case rather than an error path: a constrained decoder knows the grammar state, so it knows whether the output is closeable, and a client that receives `finish_reason: "length"` on a JSON-mode request must never attempt to parse the body.
- **Mid-character.** A three-byte code point split across two tokens, with the ceiling landing between them. The `IncrementalDetokenizer` from the tokenizer post handles this by holding at most three bytes and flushing with `errors="replace"` — one replacement character rather than a `UnicodeDecodeError` in your HTTP layer at 3 a.m.

The engineering discipline here is not clever truncation. It is honest reporting. `finish_reason` is a single string, it costs nothing, and it is the only channel through which a client can distinguish "the model was done" from "I cut it off." Get it wrong and every downstream retry policy is wrong.

```python
# nanoserve/stopping.py
class FinishReason(str, Enum):
    STOP = "stop"            # end token, or a stop string matched
    LENGTH = "length"        # max_tokens or a budget ceiling
    TOOL_CALLS = "tool_calls"  # ended on a tool-call boundary
    CANCELLED = "cancelled"  # client went away; not an OpenAI value
    ERROR = "error"          # engine fault, KV exhaustion, kernel abort


@dataclass(frozen=True)
class Outcome:
    reason: FinishReason
    detail: str = ""         # engine-side label for metrics only

    # Detail strings are for YOUR dashboards, not for the client. The
    # client contract is the five values above; the detail tells you
    # which of the four different things that produce "length"
    # actually fired.
```

Five values in the contract, and a `detail` field that never leaves your process. This split matters more than it looks. `finish_reason: "length"` can mean the request's own `max_tokens` was hit, or a server-side ceiling clamped the request's `max_tokens` down, or the KV cache ran out of room for this sequence, or a thinking budget consumed the whole allowance and left nothing for the answer. Those are four different operational problems with four different fixes, and if your metrics only carry the public string you cannot tell them apart.

The clamping case deserves its own note, because it is where servers most often lie. A client sends `max_tokens: 100000`. Your server's real ceiling is 4,096. There are two honest behaviours — reject the request with a 400, or clamp and return `finish_reason: "length"` — and one dishonest one, which is to clamp silently and return `finish_reason: "stop"`. The dishonest one is common because it makes the error rate look better, and it is poison: the client's retry logic sees a clean finish, accepts a truncated answer, and the failure surfaces three systems downstream as corrupt data.

**Rule: `finish_reason: "stop"` is a promise that the model chose to stop.** If anything else ended the generation, the reason is not `stop`.

```python
class MaxTokens:
    """The ceiling that bounds the worst case. Nothing overrides it."""

    def __init__(self, requested: int, server_cap: int):
        self.limit = min(requested, server_cap)
        self.clamped = requested > server_cap
        if self.clamped:
            log.warning("max_tokens clamped %d -> %d", requested, self.limit)

    def check(self, n_generated: int) -> Outcome | None:
        if n_generated >= self.limit:
            return Outcome(FinishReason.LENGTH,
                           "server_cap" if self.clamped else "request_cap")
        return None
```

## 5. Degeneration: when the model will never stop on its own

Set the end tokens correctly, cap the length, and you still get requests that run to the ceiling. Sometimes the model is genuinely writing a long answer. Sometimes it has fallen into a cycle and is emitting the same phrase forever.

![An ordered sequence of decode steps from the first token through a completed answer to repeated four-gram detection and finally the token ceiling with the wasted step count](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-5.webp)

This is a well-documented property of likelihood-maximizing decoding, not a defect in your engine. Holtzman et al.'s [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) is the standard reference: decoding that chases high-probability continuations produces text that is repetitive and degenerate in ways human text is not, and the paper's proposed fix is nucleus sampling — a change to the *sampler*, covered in [the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo), not a change to the stopping layer.

That framing matters for how you build the detector. **A loop detector is a safety net, not a fix.** If it fires often, your problem is greedy decoding, a temperature of zero on a model that needs sampling, a repetition penalty set to nothing, or a prompt that has driven the model into a corner. Treat a rising loop-detection rate as an alert on your sampling configuration, and never as a reason to relax the detector's threshold.

The detector still has to exist, because the alternative is paying for 4,096 tokens of the same sentence. The naive implementation — count n-gram frequencies over the whole output and stop when any n-gram exceeds a threshold — has too many false positives. Real text repeats. Code repeats `    return None` a dozen times legitimately; a markdown table repeats `| --- |`; a list of citations repeats an author's name. What is *not* legitimate is an exactly periodic tail: the last $k \cdot p$ tokens being one $p$-token phrase repeated $k$ times, with nothing else in between.

```python
# nanoserve/stopping.py
class CycleDetector:
    """Flags an exactly periodic tail: the last k*p tokens are the same
    p tokens repeated k times.

    Much stricter than n-gram frequency counting, which fires on
    legitimately repetitive text (code, tables, lists). Cost is O(sum of
    periods * repeats) integer comparisons per token -- a fixed constant,
    independent of output length, which is what you need at batch 64.
    """

    def __init__(self, periods: tuple[int, ...] = (1, 2, 3, 4, 8, 16, 32),
                 repeats: int = 4, min_tokens: int = 48):
        self.periods = periods
        self.repeats = repeats
        self.min_tokens = min_tokens
        self.period_hit = 0

    def check(self, out: list[int]) -> Outcome | None:
        n = len(out)
        if n < self.min_tokens:
            return None
        for p in self.periods:
            span = p * self.repeats
            if span > n:
                break
            tail = out[n - span :]
            # Is the tail exactly p-periodic?
            if all(tail[i] == tail[i - p] for i in range(p, span)):
                self.period_hit = p
                return Outcome(FinishReason.LENGTH, f"cycle_p{p}")
        return None
```

Three parameters, each with a real justification:

- **`periods`** stops at 32 because a cycle longer than 32 tokens repeated four times needs 128 tokens of evidence, by which point you have spent the money anyway. Longer periods also start colliding with legitimate structure.
- **`repeats = 4`** rather than 2 or 3 because two repeats is common in ordinary prose (`"very, very"`, a repeated list header) and three happens in code. Four identical consecutive repeats of the same phrase is essentially never intentional, and never intentional in a way that matters at the token level.
- **`min_tokens = 48`** because a short output that happens to be periodic — a list of five identical bullet stubs, a padding sequence — should not be truncated. The guard exists for runaways, and runaways are long by definition.

The cost is worth checking, because this runs per request per step. With those periods the inner loop does at most $ (1+2+3+4+8+16+32)\times 4 = 264 $ integer comparisons in the worst case, and typically far fewer because it breaks on the first period that does not match at index $p$. Section 11 turns that into a budget.

The honest failure mode: this detector cannot catch *semantic* loops. A model that paraphrases the same paragraph five different ways is looping in every sense that matters, and no token-level comparison will see it. Detecting that requires embeddings and a similarity threshold, which is a latency cost inside your decode loop and a source of false positives on legitimately repetitive documents. My position is: do not do it in the engine. Catch semantic loops at the agent framework level where you have turn boundaries to compare, and keep the engine's detector cheap and exact.

## 6. Cancellation: the request is gone, the GPU is not

Now the purest form of waste in the entire system. A client opens a stream, receives 200 tokens, and closes the tab. The socket is gone. The engine, which knows nothing about sockets, keeps generating tokens 201 through 4,096, writing KV for each one, occupying a batch slot that other requests are queued behind, and throwing every result away.

![A two row grid showing engine state and block occupancy across four moments from admission through a closed socket to the flag check and the release of all blocks](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-6.webp)

#### Worked example: the cost of a disconnect nobody noticed

A request with a 2,048-token prompt and a 4,096-token ceiling, on the A100 running Llama-3.1-8B. The client disconnects after token 500.

The KV footprint first. From [the memory math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache), Llama-3.1-8B in bf16 costs $2 \times 32\ \text{layers} \times 8\ \text{KV heads} \times 128\ \text{dims} \times 2\ \text{bytes} = 128$ KB per token. At the moment of disconnect the sequence holds $2{,}548 \times 128\ \text{KB} = 326$ MB. If the engine never notices, that grows to $6{,}144 \times 128\ \text{KB} = 786$ MB by the ceiling. At vLLM's default block size of 16 tokens — cited from the vLLM team's [anatomy of a high-throughput inference system](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) post (2025-09-05) — that is 384 blocks held for nothing.

The time cost: 3,596 remaining steps at the 7.9 ms decode floor is **28.4 seconds** of a batch slot. At batch 32, one abandoned request holds 1/32 of your serving capacity for half a minute. If your disconnect rate is 5% and your mean generation is 500 tokens with a 4,096 ceiling, a meaningful fraction of your fleet is generating tokens for browsers that closed. All figures here are derived from the per-token KV formula and the decode floor; run them against your own model dimensions.

The fix is structural, and it has two halves that fail independently.

**Half one: notice.** In an ASGI server the disconnect surfaces as the request task being cancelled, which raises `CancelledError` inside your streaming generator. That works only if the generator is actually suspended at an `await` the server can interrupt, and only if the server closes the generator so its `finally` runs. Some deployments sit behind a proxy that holds the upstream socket open after the downstream client leaves, in which case you learn nothing. So do both: handle cancellation *and* poll.

```python
# nanoserve/api.py
import anyio
from fastapi import Request as HttpRequest
from fastapi.responses import StreamingResponse


async def stream_completion(http: HttpRequest, req):
    """Bridge one HTTP stream to one engine request.

    Two independent paths mark the request cancelled:
      1. the ASGI server cancels this task -> CancelledError -> finally
      2. the poller notices http.is_disconnected() while we are idle
    Either one is enough; neither alone is reliable.
    """

    async def poll_disconnect():
        while not req.done:
            if await http.is_disconnected():
                engine.cancel(req.req_id, reason="client_disconnect")
                return
            await anyio.sleep(0.25)

    async def body():
        async with anyio.create_task_group() as tg:
            tg.start_soon(poll_disconnect)
            try:
                async for chunk in engine.subscribe(req.req_id):
                    yield sse(chunk)
                yield sse_done(req.finish_reason)
            finally:
                # Runs on normal completion AND on CancelledError.
                # Idempotent: cancel() on a finished request is a no-op.
                engine.cancel(req.req_id, reason="stream_closed")
                tg.cancel_scope.cancel()

    return StreamingResponse(body(), media_type="text/event-stream")
```

The 250 ms poll interval is a deliberate trade. Polling every step would add an `await` to the hot path for a condition that changes at human timescales; polling every 250 ms bounds the waste at roughly 32 decode steps on an A100, which is about 0.8% of a 4,096-token ceiling. Tighten it if your ceilings are small.

**Half two: act.** `engine.cancel()` must not block, must not touch the GPU, and must not try to interrupt a forward pass in flight. It sets a flag. The step loop reads flags at a point where releasing blocks is safe — which is the top of the step, before the batch is built.

```python
# nanoserve/engine.py  (the cancellation hook, added to step())
def cancel(self, req_id: str, reason: str = "cancelled") -> None:
    """Called from the HTTP layer, possibly from another thread.

    Sets a flag and returns. Never touches CUDA, never takes the
    scheduler lock, never waits for the current step.
    """
    req = self.by_id.get(req_id)
    if req is not None and req.state is not State.FINISHED:
        req.cancelled = True
        req.cancel_reason = reason


@torch.inference_mode()
def step(self) -> list[tuple[str, int, str | None]]:
    # (0) Reap cancellations BEFORE planning. A cancelled request must
    #     not be scheduled, must not be admitted, and must return its
    #     blocks to the pool this iteration -- the free blocks are what
    #     the scheduler needs to admit someone useful.
    for req in list(self.sched.running):
        if req.cancelled:
            self._retire(req, FinishReason.CANCELLED, req.cancel_reason)
    for req in list(self.sched.waiting):
        if req.cancelled:
            # Never ran, holds no blocks: just drop it. Free.
            self.sched.waiting.remove(req)
            req.state = State.FINISHED
            self.metrics["cancelled_before_admit"] += 1

    plan = self.sched.schedule()
    ...  # unchanged from the continuous-batching post


def _retire(self, req, reason: FinishReason, detail: str = "") -> None:
    req.finish_reason = reason.value
    req.finish_detail = detail
    req.state = State.FINISHED
    if req.seq is not None:
        req.seq.release()          # blocks return to the free pool NOW
        req.seq = None
    if req in self.sched.running:
        self.sched.running.remove(req)
    self.finished.append(req)
    self.metrics[f"finish_{reason.value}"] += 1
```

Reaping cancellations *before* `schedule()` rather than after is not cosmetic. The scheduler's admission decision depends on how many free blocks exist — that is the whole subject of [the admission control post](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse). Freeing the blocks after planning means the next request waits a full step longer than it needed to, every time, and under load those steps compound into queueing delay that shows up as TTFT.

Cancelling a request that is *waiting* rather than running is the best case in the entire post: it costs nothing, it happened before any GPU work, and it should be counted separately in your metrics. A high `cancelled_before_admit` rate means your queue is deep enough that clients time out before you reach them, which is a capacity signal, not a stopping signal.

## 7. Thinking budgets: capping the part the user never sees

Everything so far is about a model that produces one stream of tokens. Reasoning models produce two: an internal chain that works through the problem, and an answer. The internal chain is often much longer than the answer, it is usually hidden from the user, and it is charged for. Controlling its length is now a first-class serving knob, and it is the most consequential one in this post because the numbers are large — reasoning traces of thousands of tokens are routine where the answer is a hundred.

The mechanism is simpler than it sounds, and it is worth being precise about, because vendors describe it in product language and the engine-level reality is plain. The model has been trained to emit a delimiter that separates thinking from answering. Open-weight families make this visible: DeepSeek-R1 and Qwen3 both wrap the chain in `<think>` and `</think>` tags, and Qwen3's chat template exposes an `enable_thinking` flag that controls whether the template pre-opens the thinking block — see the [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B) for the template. So the engine's job is a three-state machine over a token stream: before the open tag, between the tags, after the close tag.

<figure class="blog-anim">
<svg viewBox="0 0 700 250" role="img" aria-label="A thinking budget bar fills to its cap, a close-thinking token is forced in, and an answer bar then grows to completion" style="width:100%;height:auto;max-width:820px">
<style>
.s2-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.s2-fill{fill:var(--accent,#6366f1);opacity:.75}
.s2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.s2-sm{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.s2-badge{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.s2-cap{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:5 4}
@keyframes s2-think{0%{transform:scaleX(0)}42%,92%{transform:scaleX(1)}100%{transform:scaleX(0)}}
@keyframes s2-ans{0%,50%{transform:scaleX(0)}78%,92%{transform:scaleX(1)}100%{transform:scaleX(0)}}
@keyframes s2-m1{0%,12%{opacity:0}17%,92%{opacity:1}100%{opacity:0}}
@keyframes s2-m2{0%,28%{opacity:0}33%,92%{opacity:1}100%{opacity:0}}
@keyframes s2-m3{0%,42%{opacity:0}46%,92%{opacity:1}100%{opacity:0}}
@keyframes s2-force{0%,44%{opacity:0}49%,92%{opacity:1}100%{opacity:0}}
@keyframes s2-fin{0%,80%{opacity:0}84%,92%{opacity:1}100%{opacity:0}}
.s2-t{transform-box:fill-box;transform-origin:left center;animation:s2-think 13s ease-out infinite}
.s2-a{transform-box:fill-box;transform-origin:left center;animation:s2-ans 13s ease-out infinite}
.s2-n1{animation:s2-m1 13s linear infinite backwards}
.s2-n2{animation:s2-m2 13s linear infinite backwards}
.s2-n3{animation:s2-m3 13s linear infinite backwards}
.s2-fx{animation:s2-force 13s linear infinite backwards}
.s2-fn{animation:s2-fin 13s linear infinite backwards}
@media (prefers-reduced-motion:reduce){.s2-t,.s2-a{animation:none}.s2-n1,.s2-n2,.s2-n3,.s2-fx,.s2-fn{animation:none;opacity:1}}
</style>
<text class="s2-lbl" x="24" y="46">thinking tokens</text>
<rect class="s2-track" x="24" y="60" width="520" height="40" rx="8"/>
<rect class="s2-fill s2-t" x="24" y="60" width="520" height="40" rx="8"/>
<line class="s2-cap" x1="544" y1="52" x2="544" y2="108"/>
<text class="s2-sm" x="586" y="86">cap 512</text>
<text class="s2-sm s2-n1" x="160" y="126">128 / 512</text>
<text class="s2-sm s2-n2" x="330" y="126">384 / 512</text>
<text class="s2-sm s2-n3" x="516" y="126">512 / 512</text>
<text class="s2-badge s2-fx" x="300" y="166">budget exhausted, force the close-thinking token</text>
<text class="s2-lbl" x="24" y="204">answer tokens</text>
<rect class="s2-track" x="24" y="216" width="300" height="30" rx="8"/>
<rect class="s2-fill s2-a" x="24" y="216" width="300" height="30" rx="8"/>
<text class="s2-badge s2-fn" x="480" y="238">finish_reason: stop</text>
</svg>
<figcaption>The thinking bar fills to its cap, the engine injects the close-thinking token rather than letting the chain continue, and the answer is generated inside the remaining allowance.</figcaption>
</figure>

### Budget forcing, and why it costs one extra step

The naive implementation of a thinking budget is to stop the request when the thinking token count hits the cap. That is wrong, and wrong in the most expensive possible way: you have paid for 512 tokens of reasoning and thrown away the answer, so the request produces nothing at all and the client retries.

The correct implementation is **budget forcing**: when the budget is exhausted, do not sample the next token — *inject* the close-thinking token, and let the model continue from a state where the chain is syntactically complete. The model then does what it was trained to do after a close tag, which is answer. This technique is described and evaluated in Muennighoff et al.'s [*s1: Simple test-time scaling*](https://arxiv.org/abs/2501.19393), which uses it in both directions: append the end-of-thinking delimiter to *shorten* reasoning, or suppress that delimiter and append a continuation word like "Wait" to *lengthen* it. The paper's framing is test-time compute control; the serving framing is a cost knob. Same mechanism.

There is a subtlety in the implementation that is easy to get wrong and produces beautifully confusing bugs. A forced token is not a token you write into the output list. It is a token you must **feed through the model**, because the next forward pass conditions on a KV cache that has to contain an entry for it. Skip the forward pass and the model generates the answer conditioned on a sequence whose cache stops one token short — it does not know it closed the thinking block. So budget forcing costs exactly one extra decode step per forced token, and that step is real work at the decode floor.

The clean way to express this in an engine is a forced-token queue on the request, checked by the sampler:

```python
# nanoserve/stopping.py
from collections import deque


class Phase(str, Enum):
    PRE = "pre"            # before the thinking block opened
    THINKING = "thinking"  # inside <think> ... </think>
    ANSWER = "answer"      # after the close tag


class ThinkingBudget:
    """Caps the reasoning chain and forces a clean transition.

    On exhaustion we inject the close-thinking token instead of ending
    the request. The injected token is fed through the model like any
    other -- the answer must condition on a cache that contains it.
    """

    def __init__(self, open_id: int, close_id: int, max_thinking: int,
                 min_answer: int = 64, starts_open: bool = False):
        self.open_id, self.close_id = open_id, close_id
        self.max_thinking = max_thinking
        self.min_answer = min_answer
        self.phase = Phase.THINKING if starts_open else Phase.PRE
        self.n_thinking = 0
        self.forced = False

    def observe(self, tid: int, forced_q: deque[int]) -> Outcome | None:
        """Called once per generated token, before it is streamed."""
        if self.phase is Phase.PRE:
            if tid == self.open_id:
                self.phase = Phase.THINKING
            return None

        if self.phase is Phase.THINKING:
            if tid == self.close_id:
                self.phase = Phase.ANSWER      # model closed it itself
                return None
            self.n_thinking += 1
            if self.n_thinking >= self.max_thinking and not self.forced:
                # Budget forcing: the NEXT step emits the close tag
                # instead of sampling. It still runs a forward pass.
                forced_q.append(self.close_id)
                self.forced = True
                self.phase = Phase.ANSWER
                log.info("thinking budget forced at %d tokens",
                         self.n_thinking)
            return None

        return None

    def reserve(self, max_tokens: int) -> int:
        """Answer tokens guaranteed to remain after the chain.

        Without this, max_tokens=600 and max_thinking=512 leaves 88
        tokens for the answer -- and a request that spends its whole
        allowance thinking returns nothing the user can read.
        """
        return max(self.min_answer, max_tokens - self.max_thinking)
```

Then in the sampler, the forced queue takes precedence:

```python
# nanoserve/engine.py  (excerpt from step(), sampling stage)
next_ids = []
for req, row in zip(scheduled, logits):
    if req.forced_ids:
        # Injected token: no sampling, but a full forward pass next
        # step so its KV entry exists. This is the cost of forcing.
        next_ids.append(req.forced_ids.popleft())
        req.metrics["forced_tokens"] += 1
    else:
        next_ids.append(self.sample_one(row, req.sampling))
```

### The softer alternative: mask instead of inject

Forcing a token is decisive but abrupt — the chain gets cut wherever it happened to be, sometimes mid-sentence, and the model has to recover. A gentler mechanism reuses the machinery from [constrained decoding](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls): as the budget approaches, apply a logit bias that raises the close-thinking token's probability, and when the budget is exhausted, *mask out everything except* the close token. The model then closes the block itself, on the next boundary it considers reasonable, and the transition reads naturally.

| Mechanism | Extra decode steps | Transition quality | Precision of the cap |
| --- | --- | --- | --- |
| End the request at the cap | 0 | no answer at all | exact |
| Inject the close token | 1 | abrupt, sometimes mid-clause | exact |
| Ramp a logit bias near the cap | 0 | natural | soft, overshoots |
| Mask to close-token only at the cap | 0 | clean, model picks the wording | exact |

The masking row is the one I would ship, and it is essentially free because the logits processor already exists in your stack. The one caveat is that masking assumes the close token is *reachable* from the current state — if a grammar constraint is also active and forbids it, the two constraints conflict and you must decide which wins. My rule: the budget wins, because a budget violation costs money continuously while a grammar violation costs one retry.

### Accounting: thinking tokens are output tokens

Two things follow from a thinking budget, and both are policy rather than code.

**Bill them.** Reasoning tokens are generated tokens. Each one cost a full pass over the weights, exactly like an answer token, and the derivation in section 1 does not care what the token means. Any accounting that excludes them is a subsidy. Commercial reasoning APIs report them as a separate line inside the completion-token count for exactly this reason; check your provider's current API reference for the field names, which move around.

**Do not stream them by default.** The chain is not the product, and streaming it to a UI that then hides it wastes bandwidth and leaks reasoning that the model was not asked to show. The `Phase` state machine above already gives you the switch: emit nothing while `phase is Phase.THINKING` unless the request explicitly asked for the trace. This composes with the hold-back buffer without interacting, because the phase check happens on token ids and the hold-back happens on characters.

The quality-versus-cost frontier is real and is workload-specific, and I am not going to give you a number for it because I have not measured one and neither has anyone in a way that generalizes. What I will give you is the shape: reasoning length has strongly diminishing returns, the returns fall off at different points for different task families, and the only way to find your cap is to sweep it against your own evaluation set. Sweep `max_thinking` over something like 128, 256, 512, 1024, 2048 on your eval, plot accuracy against mean output tokens, and pick the knee. That is a half-day of work that pays for itself in a week, and it is a measurement *you* can make honestly on your traffic, which is worth more than any published curve on someone else's.

## 8. The KV subtlety: the last token has no cache entry

There is a detail about how stopping interacts with the KV cache that only becomes visible when you build resumable or streaming sessions, and it is genuinely subtle. It is worth a section because it explains an otherwise baffling "+1 token" that shows up in real systems.

Recall the invariant from [the KV cache post](/blog/machine-learning/inference-engineering/why-recompute-is-fatal-writing-a-kv-cache): the forward pass for position $t$ writes the key and value for position $t$ into the cache and produces the logits that predict the token at position $t+1$. So after you sample token $t+1$, the cache contains entries for positions $0 \ldots t$ and **nothing for the token you just generated**. Its KV entry gets written by the *next* forward pass — the one that would generate token $t+2$.

Which means: if you stop right there, the generated token exists in your output list but not in the cache.

The vLLM team's [streaming requests and realtime API](https://vllm.ai/blog/2026-01-31-streaming-realtime) post (2026-01-31) makes this operationally concrete. In their design, a streaming session sends successive chunks of input to the same conversation, and each resumable request is issued with `max_tokens` set to 1 so that the request's only real work is computing KV for `prompt_token_ids`. The single generated token, they note, lacks a KV state and is discarded — and because it lacks one, discarding it is, in their words, "essentially free." Nothing has to be rolled back. The cache after the request is exactly the cache for the prompt, which is precisely the state the next chunk wants to extend.

The same post notes the complementary case: a model that must emit a stop token may need "+1 token to recompute the stop token before processing the new input chunk." That is the invariant again, from the other side. If the turn's end token is part of the conversation — and for a chat model it is, because the next turn's prompt template includes it — then its KV entry has to exist before the next chunk's tokens can attend to the correct history. It did not get written when it was generated, so it gets recomputed as the first token of the next chunk's prefill. One token of extra prefill per turn.

Three practical consequences.

**For streaming sessions**, keep the session's blocks pinned. The same vLLM post observes that holding the KV blocks across the idle gap between chunks avoids recomputation, and the reason it matters more than it looks is prefix-cache granularity: per the anatomy post, only *complete* blocks are cacheable, so a prefix that ends mid-block leaves `long_prefix_len % block_size` tokens to recompute on every resume. With the default 16-token block, an unlucky boundary costs up to 15 tokens of redundant prefill per chunk, on every chunk, forever. Pinning is cheaper than recomputing, until memory pressure says otherwise — and *that* trade is exactly the eviction policy from [the eviction and preemption post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping).

**For multi-turn agent loops**, the assistant turn you append to history must contain the end token, and it must contain it as the same id you generated. Section 2 covered the retokenization drift; here is the cache-level version of the same argument. If the next turn's prompt differs from the served ids by even one token, the prefix hash diverges at that block, and every block after it misses. On the Mooncake trace shape — 33 turns, roughly 80K tokens of context by turn 30 — a hash divergence at turn 3 means 30 turns of full prefill instead of cache hits. That is the difference between the 92.2% and 1.7% hit rates their post reports for two different configurations.

**For your own engine**, decide explicitly whether `output_ids` includes the terminating token, write it down, and make the API layer and the cache layer agree. My convention in `nanoserve`: `output_ids` **always** contains the end token, the rendered text **never** does, and the API returns both when `return_token_ids` is set. One rule, two consumers, no ambiguity.

## 9. Assembling the stack

Now compose. Six checks, one order, one owner of the finish reason.

![A layered stack of the six stopping checks from the cancellation flag down through the end token set, thinking budget, stop strings, loop detector, and token ceiling to the finish reason](/imgs/blogs/stop-conditions-eos-handling-and-thinking-budgets-7.webp)

The order is load-bearing, and here is the reasoning behind each position:

1. **Cancelled flag first.** It is one boolean, and if it is set nothing else matters — do not pay for a stop-string scan on a request nobody will read.
2. **End token ids second.** It is a set membership test on an integer, it is the most authoritative signal available, and it must beat every policy check so that a model that genuinely finished is reported as `stop` rather than as whatever policy would have fired on the same step.
3. **Thinking budget third**, because it does not end the request — it *transitions* it. Running it before the text-level checks means the close token never reaches the detokenizer as visible output.
4. **Stop strings fourth**, because they need detokenized text, which is the most expensive input in the stack.
5. **Loop detector fifth**, because it is the only check that can be wrong, and everything that can end the request correctly should get the chance first.
6. **`max_tokens` last**, because it is the backstop. If it fires, nothing else did.

```python
# nanoserve/stopping.py
class StoppingCriteria:
    """The whole stack. One instance per request; feed it one token per
    step along with the newly detokenized text.

    Returns (text safe to emit now, Outcome or None).
    """

    def __init__(self, end_ids: frozenset[int], max_tokens: MaxTokens,
                 stop_strings: StopStrings | None = None,
                 budget: ThinkingBudget | None = None,
                 cycles: CycleDetector | None = None,
                 include_end_token_in_text: bool = False):
        self.end_ids = end_ids
        self.max_tokens = max_tokens
        self.stops = stop_strings
        self.budget = budget
        self.cycles = cycles
        self.include_end = include_end_token_in_text

    def feed(self, req, token_id: int, text_delta: str,
             ) -> tuple[str, Outcome | None]:
        # 1. cancellation
        if req.cancelled:
            return "", Outcome(FinishReason.CANCELLED, req.cancel_reason)

        # 2. end tokens -- authoritative, and never streamed as text
        if token_id in self.end_ids:
            tail = self.stops.flush() if self.stops else ""
            return (tail if not self.include_end else tail + text_delta,
                    Outcome(FinishReason.STOP, f"end_id_{token_id}"))

        # 3. thinking budget -- transitions phase, may enqueue a forced
        #    token; does not itself finish the request
        if self.budget is not None:
            self.budget.observe(token_id, req.forced_ids)
            if self.budget.phase is Phase.THINKING and not req.stream_thinking:
                text_delta = ""      # generated, billed, not streamed

        # 4. stop strings -- character space, with hold-back
        stopped = False
        if self.stops is not None and text_delta:
            text_delta, stopped = self.stops.push(text_delta)
        if stopped:
            return text_delta, Outcome(FinishReason.STOP,
                                       f"stop_str:{self.stops.matched!r}")

        # 5. degenerate cycle
        if self.cycles is not None:
            hit = self.cycles.check(req.output_ids)
            if hit is not None:
                tail = self.stops.flush() if self.stops else ""
                return text_delta + tail, hit

        # 6. the ceiling
        hit = self.max_tokens.check(len(req.output_ids))
        if hit is not None:
            tail = self.stops.flush() if self.stops else ""
            return text_delta + tail, hit

        return text_delta, None
```

Two details that are easy to miss.

**Every terminal path flushes the stop-string buffer.** If the request ends for a reason other than a stop-string match, the characters being withheld are legitimate output and the client is owed them. Forget this and every response that ends on `max_tokens` or an end token is silently missing its last few characters — a bug that will be reported to you as "the model cuts off the last word sometimes" and will take you a week to find, because it depends on which stop strings the client happened to send.

**Thinking tokens are suppressed by blanking `text_delta`, not by skipping the detokenizer.** The detokenizer must see every token, because a multi-byte character can straddle the boundary between the last thinking token and the first answer token. Skip it for hidden tokens and you get a replacement character at the start of every answer.

Wiring it into the step loop is small, because all the work moved into the criteria object:

```python
# nanoserve/engine.py  (excerpt, the retire stage of step())
for req, tok in zip(scheduled, next_ids):
    req.output_ids.append(tok)
    delta = req.detok.push(tok)                 # incremental, UTF-8 safe
    emit, outcome = req.criteria.feed(req, tok, delta)
    if emit:
        req.emit(emit)                          # SSE queue
    if outcome is not None:
        self._retire(req, outcome.reason, outcome.detail)
```

Ten lines, and it now handles multiple end tokens, split stop strings, hidden reasoning with a forced transition, exact repetition cycles, a hard ceiling, and clients who left.

## 10. Stress test: the cases that break it

The best property of this whole component is that it is pure Python over integers and strings. No GPU, no model, no network. Every case below runs in milliseconds on a laptop and belongs in CI.

**A stop string that is a prefix of another.** The client sends `stop: ["</s", "</stop>"]`. Hold-back is sized from the longest, so 6 characters. The shorter string is a prefix of the longer, so it *always* matches first, and the longer one is dead code. That is correct behaviour — the client asked to stop on the earliest occurrence of either — but it is worth an explicit test, because a natural-looking "match the longest" implementation gets it backwards and holds output that should have terminated.

**Two stop strings where list order disagrees with position.** Covered in section 3; the leftmost-match fix. Test it directly, because the naive version passes every single-stop test.

**A stop string appearing inside a JSON string value.** The client is in JSON mode and sends `stop: ["\n\n"]` out of habit; the model generates `{"note": "line one\n\nline two"}`. Should it stop? **No.** The stop string is data inside a string literal, not a structural boundary, and stopping there produces unparseable output. There is no way for a character-level stop checker to know that. The fix comes from [constrained decoding](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls): when a grammar is active, the grammar knows when the document is complete, so the correct stop condition is "the automaton reached an accepting state," and character-level stop strings should be **disabled entirely** in structured mode. Encode that as a rule in your API layer: `response_format` and `stop` are mutually exclusive, and supplying both is a 400.

**`max_tokens` hit exactly at a multi-byte character.** The 4,096th token is the first two bytes of a three-byte code point. `IncrementalDetokenizer.flush()` emits one replacement character; the alternative is dropping the partial bytes entirely. Both are defensible; be deliberate and test which one you do. Whatever you choose, `finish_reason` is `length`, and a client parsing the body must handle a trailing replacement character.

**A client that disconnects during a 4k-token generation.** The test is not "does it stop" but "how many steps later." Assert the bound: with a 250 ms poll and a 7.9 ms step, no more than roughly 32 extra steps. Make it an assertion on a fake clock, not a wall-clock test, so it does not flake in CI.

**Batch interference.** One request's stop must not stall the others. The criteria stack runs per request per step, so its cost multiplies by batch size; section 11 gives the budget.

```python
# tests/test_stopping.py
from nanoserve.stopping import (StopStrings, CycleDetector, MaxTokens,
                                ThinkingBudget, FinishReason)
from collections import deque


def test_leftmost_match_wins_over_list_order():
    s = StopStrings(["END", "\n\nHuman:"])
    out, stopped = s.push("ok\n\nHuman: hi END")
    assert stopped and out == "ok"          # not "ok\n\nHuman: hi"


def test_prefix_stop_fires_first():
    s = StopStrings(["</s", "</stop>"])
    out, stopped = s.push("done</stop>")
    assert stopped and out == "done"        # matched the 3-char prefix


def test_nothing_leaks_across_a_token_split():
    s = StopStrings(["\n\nHuman:"])
    seen = ""
    for chunk in ["answer.", "\n\n", "Human", ":"]:
        out, stopped = s.push(chunk)
        seen += out
        if stopped:
            break
    assert seen == "answer." and "Human" not in seen


def test_flush_returns_held_text_when_ending_for_another_reason():
    s = StopStrings(["\n\nHuman:"])
    out, _ = s.push("the last words")
    assert s.flush() == "the last words"[len(out):]
    assert out + "the last words"[len(out):] == "the last words"


def test_cycle_detector_ignores_legitimate_repetition():
    d = CycleDetector()
    code = [10, 20, 30, 40, 10, 20, 31, 41] * 8      # near-periodic, not exact
    assert d.check(code) is None


def test_cycle_detector_catches_an_exact_loop():
    d = CycleDetector()
    out = list(range(60)) + [7, 8, 9] * 4
    hit = d.check(out)
    assert hit is not None and hit.detail == "cycle_p3"


def test_budget_forces_the_close_token_not_the_end():
    q = deque()
    b = ThinkingBudget(open_id=1, close_id=2, max_thinking=3,
                       starts_open=True)
    for tid in (50, 51, 52):
        assert b.observe(tid, q) is None
    assert list(q) == [2]                    # forced close, request lives on


def test_clamped_max_tokens_still_reports_length():
    m = MaxTokens(requested=100_000, server_cap=4096)
    assert m.clamped and m.limit == 4096
    out = m.check(4096)
    assert out.reason is FinishReason.LENGTH and out.detail == "server_cap"
```

Seven tests, no GPU, and they cover every failure I have described. The `test_cycle_detector_ignores_legitimate_repetition` case is the one worth keeping forever: it is the guard against someone loosening the detector's parameters and silently truncating structured output.

## 11. Measuring stopping honestly

Two things need measuring, and they need different techniques.

**The CPU cost of the stack.** This is host-side Python, not GPU work, so `torch.cuda.Event` is the wrong tool. Time it with `time.perf_counter()` around `criteria.feed()` alone, and compare against a budget rather than against nothing. The budget is derivable: the stack runs once per request per step, so at batch $B$ it adds $B \cdot c$ to the host side of each step, and the host has until the GPU finishes the step. On the A100 at the 7.9 ms decode floor,

$$
c_{\max} \;=\; \frac{7.9\ \text{ms}}{64} \;=\; 123\ \mu\text{s per request per step} \quad\text{(derived, batch 64)} .
$$

That is an enormous budget for a set membership test, a bounded string scan, and 264 integer comparisons — which is the point. If your stack is anywhere near 123 µs you have accidentally made something O(output length); the usual culprit is accumulating the full output in the stop-string `pending` buffer and rescanning it every token, which turns the whole thing quadratic. Keep `pending` bounded by `hold` plus one chunk and this stays a rounding error.

**The behaviour of the stack in production.** Four metrics, and they answer four different questions:

| Metric | What it tells you | Alert when |
| --- | --- | --- |
| `finish_reason` counts, by `detail` | which mechanism is actually ending requests | `length` share rises |
| Output-length histogram, split by `finish_reason` | whether the ceiling is doing the model's job | `length` bucket piles up at the cap |
| Cycle-detector fire rate | whether your sampling config is degenerate | any sustained nonzero rate |
| Cancellation rate, split before/after admission | client patience vs queue depth | before-admit share rises |

The first two together are the diagnostic that matters most. A healthy service ends the large majority of requests on `stop` with a smooth length distribution. A service where 30% of requests end on `length` with a spike at exactly the cap is a service where the model is not stopping and the ceiling is hiding it. That is the signature of the wrong end-token set, and it is visible from a dashboard without reading a single response body.

For the load-generation side of any measurement here, use the harness from [the baseline post](/blog/machine-learning/inference-engineering/the-naive-decode-loop-and-your-first-baseline): warm up, synchronize before timing GPU work, run open-loop with Poisson arrivals rather than closed-loop, and report p50 and p99 separately. Stopping bugs are heavily tail-weighted — the mean output length barely moves when 2% of requests run to the ceiling, while p99 latency doubles.

#### Numbers used in this post

| Quantity | Value | Source |
| --- | --- | --- |
| Llama-3.1-8B bf16 weight bytes | 16.06 GB | derived: parameter count × 2 bytes |
| A100 80GB HBM bandwidth | 2,039 GB/s | cited: [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) |
| Decode floor, A100 80GB | 7.9 ms/step | derived: 16.06 GB / 2.039 TB/s |
| Decode floor, RTX 4090 | 15.9 ms/step | derived: 16.06 GB / 1.008 TB/s |
| Aggregate tok/s at batch 32, A100 | 4,051 | derived: 32 / 7.9 ms |
| Hardware floor, cost per 1M output tokens | \$0.137 | derived at an assumed \$2.00/GPU-hour |
| Cost of 300 wasted tokens per request | \$4.1e-5 | derived from the line above |
| KV bytes per token, Llama-3.1-8B bf16 | 128 KB | derived: 2 × 32 × 8 × 128 × 2 bytes |
| KV held by an abandoned 6,144-token sequence | 786 MB | derived: 6,144 × 128 KB |
| Wasted time, disconnect at token 500 of 4,096 | 28.4 s | derived: 3,596 × 7.9 ms |
| Host budget per request per step, batch 64 | 123 µs | derived: 7.9 ms / 64 |
| vLLM default KV block size | 16 tokens | cited: [vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) |
| Agentic trace input:output ratio | 131:1 | cited: [vLLM Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) |
| Agentic trace median turns | 33 | cited: same |
| Kimi-K2 tool calls after template fixes | ~218 → 1,007 | cited: [vLLM Kimi K2 post](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy) |

Everything in that table is either arithmetic you can redo in thirty seconds or a link you can open. Nothing in it is a benchmark I ran, because I have no GPU and ran none.

## 12. Case studies and real numbers

**Streaming sessions and the token with no cache entry.** The vLLM team's [streaming requests and realtime API](https://vllm.ai/blog/2026-01-31-streaming-realtime) post (2026-01-31) describes a design where an audio or text stream is fed to the model in chunks, each chunk issued as a resumable request against a persistent "anchor" whose KV blocks stay resident. The token-handling detail is the one this post's section 8 is built on: `max_tokens` is set to 1 so each resumable request computes KV only for its `prompt_token_ids`; the single generated token has no KV state and is discarded, which they describe as essentially free; and a model that emits a stop token may need one extra token to recompute that stop token before the next input chunk is processed. The post also notes vLLM does not yet preempt idle streaming sessions, which is the honest cost of pinning: memory held while nothing is generated.

**A chat template bug reported as a tool-calling bug.** The [Kimi K2 accuracy investigation](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy) (2025-10-28) is the best public write-up I know of showing that stopping, templating, and tool calling are one subsystem. Three separate defects — a signature-inspection bug that dropped `add_generation_prompt`, a content-normalization bug that rendered an empty string as a list literal, and a tool-call-ID parser that crashed on IDs containing a colon — together produced a model that appeared to be bad at tool calling. After the fixes the post reports successful tool calls rising from roughly 218 of 1,200-plus to 1,007. The debugging method is worth stealing: they rendered the chat template externally and posted the resulting token ids to the raw completions endpoint, bypassing the server's templating entirely, which isolates a template fault from a model fault in one request. The post also notes that Moonshot's own service uses an "Enforcer" — constrained decoding — that vLLM did not have at the time, which is the same observation as section 10's: when a grammar is available, it is a better stop condition than any string.

**Agentic traffic is a stopping problem wearing a caching costume.** The [Mooncake Store post](https://vllm.ai/blog/2026-05-06-mooncake-store) (2026-05-06) reports, over 610 Codex and SWE-bench traces, an aggregate input-to-output ratio of 131:1, a median of 33 turns, roughly 2,242 tokens of context growth per turn, and about 80K tokens of median context by turn 30. Their result is a distributed KV pool that lifts the cache hit rate from 1.7% to 92.2% in their configuration (Kimi-2.5 NVFP4, one prefill and one decode instance over 12 GB200s). Read alongside section 1, the lesson for stopping is: caching makes the re-read of history cheap, which mathematically *increases* the share of remaining cost that is output generation. The better your prefix cache, the more your bill is decided by how many tokens you generate per turn.

**Budget forcing as a published technique.** Muennighoff et al.'s [s1](https://arxiv.org/abs/2501.19393) introduces budget forcing as a test-time scaling method: force-terminate the reasoning chain by appending the end-of-thinking delimiter, or extend it by suppressing that delimiter and appending a continuation token. The paper is about improving accuracy by spending more compute; a serving engine uses the identical mechanism in the other direction, to spend less. The important transfer is the mechanism's validity — appending the delimiter produces a model that answers, rather than a model that is confused — which is what makes a hard cap safe to ship.

## 13. When to reach for this (and when not to)

Not every deployment needs all six mechanisms. Here is where I would draw the lines.

**Always, no exceptions:** resolve the end-token set from `generation_config` and the chat template, and log it at startup. Set a server-side `max_tokens` ceiling and report `finish_reason` honestly. These two are twenty lines of code and they are the difference between a 214-token response and a 4,096-token one. There is no deployment small enough to skip them.

**Almost always:** cancellation. The code is small, it touches your HTTP layer rather than your kernels, and the waste it removes is total — abandoned generations produce nothing of value at all. The only reason to skip it is if you serve exclusively non-streaming batch jobs where nobody disconnects.

**When clients send stop strings:** the hold-back checker, with a hard cap on stop-string length and count. If your API does not expose `stop`, do not build this — it is the only mechanism here that costs latency, and paying latency for an unused feature is a bad trade.

**When you serve a reasoning model:** thinking budgets, with the masking variant if you already have a logits-processor pipeline. This is the highest-value item on the list for reasoning traffic and irrelevant for everything else.

**When your sampling is greedy or near-greedy:** the cycle detector. If you run at temperature 0.7 with a sensible repetition penalty, it will essentially never fire, and it is still worth the 264 comparisons as insurance.

**And when should you not write any of this?** If you are running vLLM or SGLang, most of it is already there and better tested than yours will be: multiple end-token ids, stop strings with proper hold-back, `finish_reason`, and cancellation on disconnect are all production features. Building your own makes sense for exactly two reasons — you are learning how the machine works, which is what this series is for, or you need a stop condition the engine does not expose, such as a domain-specific budget, an early-exit on a confidence signal, or a grammar-aware terminal state. **Everything else is a solved problem you should be consuming, not writing.** The value of having written it once is that when the production engine behaves strangely, you know exactly which of these six mechanisms is misconfigured, and that is worth the afternoon.

## 14. Key takeaways

1. **Output length is the only cost axis that scales one-for-one.** Every output token is a full pass over the weights, amortized across nothing. At the derived A100 floor that is about \$0.137 per million output tokens of pure hardware cost, and 300 wasted tokens per request at a million requests a day is roughly \$15,000 a year.
2. **EOS is a set, not a token.** Resolve it from `generation_config`, the tokenizer, and a probe of known turn-end markers; take the union; log it. A missing end id costs thousands of decode steps and throws no error.
3. **Stop strings are characters, generation is tokens.** Hold back the length of the longest stop string minus one — provably safe, provably minimal — and match the *leftmost* occurrence across all stops, not the first in list order.
4. **Cap the stop strings your API accepts.** Eight strings of at most 64 characters. A 500-character stop string is a request to buffer 499 characters before the user sees anything.
5. **`finish_reason: "stop"` is a promise that the model chose to stop.** Clamping `max_tokens` silently and reporting `stop` corrupts every retry policy downstream. Keep a private `detail` field so your metrics can distinguish four different causes of `length`.
6. **A loop detector is a safety net, not a fix.** Detect exact periodicity, not n-gram frequency, so legitimate repetition in code and tables survives. A rising fire rate is an alert on your sampler configuration.
7. **Reap cancellations at the top of the step, before scheduling.** The freed blocks are what the scheduler needs to admit somebody useful, and a disconnect noticed 28 seconds late is 3,596 forward passes thrown away.
8. **Budget forcing beats budget truncation.** Ending a request at the thinking cap yields no answer at all; injecting or masking to the close-thinking token yields a real one. Injection costs exactly one extra decode step, because the forced token still needs its KV entry.
9. **The generated token has no KV entry until the next forward pass.** That is why discarding a final token can be free, why a stop token may need to be recomputed on resume, and why streaming sessions must pin their blocks.
10. **Every terminal path must flush the hold-back buffer.** Forget it and every response ending on a ceiling or an end token silently loses its last few characters.

## Further reading

- vLLM, [Streaming Requests and the Realtime API](https://vllm.ai/blog/2026-01-31-streaming-realtime) — resumable requests, `max_tokens=1`, the discarded final token, and the extra token needed to recompute a stop token.
- vLLM, [Kimi K2 Accuracy: Debugging Tool Calling and Chat Templates](https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy) — three template bugs that looked like a bad model, and the external-templating trick for isolating them.
- vLLM, [Mooncake Store: Distributed KV Cache for Agentic Workloads](https://vllm.ai/blog/2026-05-06-mooncake-store) — the 610-trace agentic workload statistics used throughout section 1.
- vLLM, [Agent Lightning: returning token ids to prevent retokenization drift](https://vllm.ai/blog/2025-10-22-agent-lightning) — why the ids you served must be the ids you replay.
- Muennighoff et al., [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — budget forcing in both directions.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — why likelihood-maximizing decoding loops, and why the fix belongs in the sampler.
- Within this series: [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) for the scoreboard, [the tokenizer boundary](/blog/machine-learning/inference-engineering/the-tokenizer-boundary-and-incremental-detokenization) for incremental detokenization and the hold-back derivation, [the sampler zoo](/blog/machine-learning/inference-engineering/from-logits-to-tokens-the-sampler-zoo) for the knobs that cause degeneration, [structured output in production](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) for grammar-aware termination, and [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) for how the whole engine fits together.
