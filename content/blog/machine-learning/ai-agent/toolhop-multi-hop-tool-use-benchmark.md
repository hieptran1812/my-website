---
title: "ToolHop: Stress-Testing Multi-Hop Tool Use in LLMs"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into ToolHop, ByteDance's query-driven benchmark for multi-hop tool use, why the best model still clears under half its problems, and what its error taxonomy tells us about building agents that chain tools."
tags: ["llm", "ai-agent", "tool-use", "benchmark", "evaluation", "function-calling", "multi-hop", "bytedance", "agentic-llm"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

The single most over-claimed capability in the agent literature is "tool use." Every model card has a function-calling number. Every framework demo shows the model calling a weather API and getting the right answer. And almost all of those demos share one quiet property: they are **single-hop**. One query, one tool, one answer. The model fills a couple of arguments from the prompt, the tool returns, and the model paraphrases the return. That is a parameter-filling problem, and modern models are genuinely good at it.

The problem your production agent actually faces is **multi-hop**: the answer to the user's question requires calling tool A, feeding A's *return value* into tool B, feeding B's return into tool C, and only then producing an answer. Now the model has to carry state across calls, decide which tool comes next given what it just learned, parse a structured return it did not write, and recover when a call fails. This is where agents quietly fall apart in the field — not because they pick the wrong tool, but because they lose the thread across the chain.

![ToolHop query-driven data construction shown as a timeline: a seed multi-hop query feeds three construction stages that produce a verifiable instance](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-1.webp)

The diagram above is the mental model for the benchmark this post is about. [ToolHop](https://arxiv.org/abs/2501.02506) — "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use," ByteDance Research, accepted to ACL 2025 — takes that multi-hop chain seriously and builds an evaluation around it. Its headline finding is brutal and clarifying: across 14 models from 5 families, the best (GPT-4o) reaches only **49.04%** answer correctness under mandatory tool use. Everything below GPT-4o is worse, and several open models are far worse. If you have been assuming that a model with a strong single-hop function-calling score will chain tools reliably, ToolHop is the benchmark that should change your mind.

## TL;DR

> **What ToolHop is and why it matters.** ToolHop is a benchmark of **995 multi-hop user queries** backed by **3,912 locally executable tools** (ByteDance Research, ACL 2025). Each query genuinely requires chaining 3–7 dependent tool calls to answer. The dataset is built **query-driven** — starting from the multi-hop question and working backward to the tools — through three stages (Tool Creation → Document Refinement → Code Generation), which gives it five properties that ad-hoc tool benchmarks lack: diverse queries, *meaningful* tool interdependencies, locally runnable tools, detailed execution feedback, and *verifiable* answers checked by exact match against a gold entity.
>
> **The result that matters.** Even with mandatory tool access, the best model — GPT-4o — scores **49.04%**; the 14-model average is **32.12%**. Without tools, the same models average **19.83%**. So tools roughly *double* accuracy, yet the ceiling sits below half. Multi-hop tool use is unsolved.
>
> **What breaks.** The failures are not about tool *selection*. They are about chain mechanics: hallucinating an argument instead of waiting for a prior return (Qwen2.5 forced into parallel calls — up to **75.08%** of queries), unsupported parameter types (Gemini1.5 lacks union types), ignoring execution feedback (only GPT recovers — **+23.4 points** with detailed feedback vs. minimal), and premature termination (LLaMA3.1 cannot emit reasoning text and a tool call in the same turn).
>
> **Who should care.** Anyone building agents that chain tools, anyone choosing a model for an agentic product, and anyone writing an eval harness — ToolHop's harness design (real executors, detailed feedback, exact-match scoring) is a template worth copying. See its companions in this series: the [Model Atlas hub](/blog/machine-learning/bytedance-research-model-atlas), [PaSa](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent), and the [agent-trajectory evaluation](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) post.

This is a deep-dive: we will work through what makes multi-hop hard, how ToolHop is constructed (the construction method is the most reusable idea in the paper), what the full results table actually says once you read past the headline, a runnable multi-hop executor you can adapt, the error taxonomy in detail, and a critique section that says plainly what would change my mind about the benchmark's conclusions.

## 1. The mental model: a multi-hop query is a dependency DAG

Start with the shape of the problem, because the shape is the whole point. A single-hop tool query has one degree of freedom: which tool, and what arguments. A multi-hop query is a **directed acyclic graph** of tool calls where edges are data dependencies — the output of one call is the input of the next.

![A multi-hop query rendered as a tool-dependency DAG: query feeds tool A, whose output feeds tools B and C to a grounded answer, while a guessed argument branch skips a tool and reaches a wrong answer](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-2.webp)

Consider a concrete query of the kind ToolHop contains: *"What is the currency-exchange rate, on the date the author of [a given book] was born, between their home country's currency and the US dollar?"* Answering it is a chain:

1. **Hop 1** — `lookup_author(book)` returns an author name.
2. **Hop 2** — `nationality_of(author)` returns a country.
3. **Hop 3** — `currency_of(country)` returns a currency code, and `birthdate_of(author)` returns a date.
4. **Hop 4** — `fx_rate(currency, "USD", date)` returns the rate, which is the answer.

Each arrow in the diagram is a data dependency. The critical structural fact: **the argument to hop N is not in the user's prompt — it is the return value of hop N−1.** The model literally cannot know the country to pass to `currency_of` until `nationality_of` has run and returned. This is what "meaningful interdependency" means, and it is the property that separates ToolHop from benchmarks where each "tool call" can be answered straight from the question text.

The danger path in the diagram is the one that matters for the rest of this post. When a model, instead of calling `nationality_of` and *waiting*, guesses the country — because the author "sounds French," or because it pattern-matched the book title — the chain forks into the red branch. Every downstream call is now grounded on a hallucinated value, and the final answer is wrong even though the model called real tools with well-formed arguments. **A single broken hop poisons every answer after it.** Exact-match scoring against the gold entity catches this ruthlessly: a rate computed from the wrong currency is simply not the gold rate.

This DAG framing also explains why single-hop function-calling scores fail to predict multi-hop performance. A model can be excellent at "given this query and this tool, produce a well-formed call" and still be terrible at "given the *return* of the last tool, decide and execute the next call." The first is a translation task. The second is a stateful control problem. They are different skills, and ToolHop is the first benchmark in this series designed specifically to isolate the second one. If you want the broader landscape of how tool use evolved from single calls to orchestrated chains, the [advanced tool-use](/blog/machine-learning/ai-agent/advance-tool-use) post is the companion piece here.

### Why "tool selection" is the wrong thing to measure

Most early tool benchmarks scored **tool selection**: given a query and a menu of tools, did the model pick the right one? That is a retrieval/classification problem, and it conflates two failure modes that ToolHop deliberately separates. In a single-hop selection benchmark, a wrong answer almost always means a wrong tool pick. In a multi-hop chain, the model can pick *every* tool correctly and still fail, because the failure lives in the **arguments** — specifically, the arguments that come from prior returns. ToolHop's design forces you to look at the chain mechanics rather than the menu.

## 2. Single-hop vs multi-hop: the difference is a state-carrying loop

Let us make the contrast explicit, because the entire difficulty of ToolHop lives in this one structural change.

![Before-after comparison: single-hop tool use as one query to one tool to a final answer, versus multi-hop as a chain of 3-7 tools where each argument is a prior tool output and ignoring feedback breaks the chain](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-3.webp)

The left column is the solved problem. Single-hop tool use is a three-step, **loop-free** process: one query maps to one tool, the model fills the arguments from the query text, and the tool's return *is* the answer. There is no state to carry, no decision about what comes next, and no opportunity for an error to compound — if the tool returns, you are done. Frontier models, and most mid-tier open models, handle this well; it is why function-calling leaderboards look so healthy.

The right column is what ToolHop measures. The query maps to a **chain of 3–7 tools**. The argument to each call is the *output of a prior call*, not text the model can read off the prompt. And crucially, there is a loop: the model emits a call, the executor returns (possibly an error), and the model must read that return and decide the next call. Ignoring the feedback — treating an error as a dead end, or proceeding as if a failed call succeeded — breaks the chain.

This is the structural reason multi-hop is hard, and it is worth stating as an engineering principle: **single-hop tool use is a function from text to a call; multi-hop tool use is a closed-loop controller over an environment that talks back.** Closed-loop control is harder than open-loop translation for the same reason a self-driving stack is harder than a lane-keep assist: errors accumulate, and the system must observe and correct rather than fire-and-forget.

| Property | Single-hop | Multi-hop (ToolHop) |
|---|---|---|
| Tools per query | 1 | 3–7 (avg ≈ 4) |
| Where do arguments come from? | The user's prompt | Mostly prior tool returns |
| Control structure | Open loop | Closed loop (observe → decide → act) |
| Error behavior | Isolated | Compounding (one bad hop poisons downstream) |
| What a wrong answer usually means | Wrong tool / wrong arg | Broken hop, ignored feedback, or hallucinated arg |
| Scoring | Tool match or single answer | Exact match vs gold entity at chain end |
| Frontier model accuracy | High (often 80%+) | ≤ 49.04% (GPT-4o, mandatory) |

That last row is the whole story. The same models that ace single-hop function calling land below 50% the moment the arguments have to come from prior returns and the loop has to close. The drop is not a few points; it is the difference between a shippable capability and a research problem. For a grounding in how to *build* the closed-loop agent on the left-to-right side of this table, the [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide) guide is the hands-on companion.

## 3. Query-driven construction: build the tools backward from the question

Here is the part of the paper I think is most reusable, independent of the benchmark itself. The hard problem in building a tool-use benchmark is not collecting tools — it is collecting tools that have *meaningful interdependencies*. If you start with a pile of tools and try to write queries that chain them, you get one of two bad outcomes: either the queries are contrived (no real user would phrase it that way), or the "dependencies" are fake (the model could answer each sub-step independently from world knowledge). ToolHop inverts the process. It is **query-driven**: start from a real multi-hop query and *manufacture the tools the query needs*, in three stages.

### Stage 1 — Tool Creation

Take a multi-hop user query — one that a person might actually ask, requiring several pieces of looked-up information composed together. **Decompose it into atomic subqueries**, where each subquery is answerable by exactly one tool, and the subqueries form a dependency chain (the output of one is the input to the next). For each atomic subquery, create the tool that answers it. This is the step that guarantees interdependency *by construction*: the tools exist precisely because the query needs them in sequence, so the dependency edges are real, not retrofitted.

Because the decomposition is driven by the query, the per-query tool count tracks the query's actual hop depth. ToolHop's distribution is concrete: **428 queries need 3 tools, 353 need 4, 136 need 5, 10 need 6, and 68 need 7** — totaling the 995 queries. The mass sits at 3–4 hops, which is realistic: most useful multi-hop questions are a short composition, not a 10-step saga. But the 78 queries at 6–7 hops are where the chain-discipline failures concentrate, because every additional hop is another place for the loop to break.

### Stage 2 — Document Refinement

A tool that takes a single string argument is too easy — the model can pass almost anything and the schema gives no signal. So ToolHop *refines the tool documents* to make them realistic: it expands the parameter set and adds structured types. The measured effect is precise. Average parameter count rises from **3.49 before refinement to 5.91 after**, and roughly **12% of simple string parameters are replaced by more structured types** (enums, numbers, nested objects, unions). The count of *required* parameters changes little — refinement is about making the *surface area* of each tool realistic, not about making every call mandatory-heavy.

This stage is doing real work for the benchmark's validity. A tool with one loosely-typed string parameter cannot distinguish a model that genuinely parsed the prior return from one that pasted a blob and got lucky. Structured parameters with enums and unions force the model to actually understand what it is passing, and — as we will see in the error analysis — this is exactly where Gemini1.5 falls over, because it does not support union-type parameters at all.

### Stage 3 — Code Generation

This is the property that I think most benchmarks get wrong and ToolHop gets right: **every tool is backed by executable Python that returns real values.** The tools are not stubs that echo a canned string, and they are not LLM-simulated returns. They are *locally executable code*, so when the model calls `currency_of("France")`, a real function runs and returns `"EUR"` (or raises a real exception on a malformed argument). This makes the feedback channel real — the model sees what actually happened — and it makes answers *verifiable*: because the code is deterministic and the query's gold answer was fixed during construction, the harness can check correctness by exact match against a known entity, with no LLM-judge in the loop and no ambiguity about what "correct" means.

The payoff of all three stages together is a benchmark instance with five guaranteed properties — diverse queries (because they are seeded from varied real questions), meaningful interdependencies (Stage 1), locally executable tools (Stage 3), detailed feedback (Stage 3's real exceptions), and verifiable answers (deterministic code + predefined gold). That combination is rare, and it is why ToolHop's numbers are trustworthy in a way that LLM-judged or stub-based tool benchmarks are not. The dataset ships on Hugging Face under CC BY 4.0 with the evaluation code under Apache 2.0.

#### Why query-driven beats tool-driven, made concrete

It is worth dwelling on the inversion, because it is the kind of design decision that looks like a detail and is actually the whole game. Suppose you build a tool-use benchmark the obvious way: scrape a few thousand public API specs, drop them in a pool, and ask an LLM to "write a query that requires three of these tools." What you get is a query whose three tools are *plausibly* related but not *necessarily* dependent. Concretely, imagine the pool contains `get_weather(city)`, `get_population(city)`, and `get_timezone(city)`, and the generated query is "What is the weather, population, and timezone of Paris?" That query uses three tools — but they are **independent**. Each takes `city="Paris"` straight from the prompt. There is no hop. A model can fire all three in parallel and compose the results, and parallel calling is *correct* here. This is a single-hop query wearing a three-tool costume, and a tool-driven construction process produces these by the thousand without noticing.

ToolHop's query-driven process cannot produce that degenerate case, because the decomposition in Stage 1 *defines* the dependency. It starts from "what was the EUR→USD rate on the birthday of the author of *The Stranger*?" — a query whose answer is genuinely unknowable until you have resolved each prior hop — and manufactures the tools to match. The dependency edge between `nationality_of` and `currency_of` exists because the query's decomposition demands it, not because two scraped APIs happened to share a parameter name. This is the difference between a benchmark that *measures* chaining and one that merely *contains* multiple tool calls. When you read a tool-use leaderboard, the first question to ask is: were the queries built from the tools, or the tools from the queries? Only the latter guarantees the hops are real.

There is a second, subtler benefit. Because every tool was created to answer one atomic subquery of one real query, the tool pool has *no distractors by accident* and *every needed tool by construction*. The model is given exactly the tools the query needs (plus, in the full benchmark, the accumulated pool across queries acts as a realistic menu). This means a wrong answer is never excused by "the right tool wasn't available" — the chain was always completable. Every failure is therefore attributable to the model's chaining behavior, which is precisely what makes the error taxonomy in Section 8 clean enough to act on.

#### A full worked instance, end to end

Trace the running example all the way through, because seeing one instance materialize makes the three stages concrete. Start with the seed query: *"What was the EUR→USD exchange rate on the birthday of the author of The Stranger?"* The gold answer was fixed at construction time — say `1.0832` for a specific resolved date — so correctness later is a single exact-match check.

**Stage 1 decomposes it into a dependency chain of atomic subqueries**, each of which becomes one tool:

1. `author_of(book: str) -> str` — "The Stranger" → "Albert Camus"
2. `birthday_of(person: str) -> str` — "Albert Camus" → "1913-11-07"
3. `exchange_rate(base: str, quote: str, date: str) -> float` — ("EUR", "USD", "1913-11-07") → the rate

The edges are forced by the data flow: `birthday_of` cannot run until `author_of` returns, and `exchange_rate` needs the date from `birthday_of`. There is no prompt-level shortcut; the model must thread the values through.

**Stage 2 refines each tool's documentation** so the schema is unambiguous: parameter names, types, an enum for currency codes, a date format string, and a one-line description that does not leak the answer. This is what lets a model form a valid call without guessing the interface — and what makes a malformed call the model's fault rather than the benchmark's.

**Stage 3 backs each signature with executable Python** holding a lookup table or a small deterministic function, so `author_of("The Stranger")` truly returns `"Albert Camus"` and `exchange_rate("EUR", "USD", "1850-01-01")` raises a typed `ValueError` for an out-of-range date rather than silently returning garbage. Now the instance is complete: three dependent, executable, well-documented tools; a query that requires all three in order; and a single verifiable gold value. A model that returns `1.0832` is right; one that returns `1.08` or `"about 1.08 dollars"` is scored wrong after answer standardization; one that fires all three calls in parallel with `date` guessed from the prompt is wrong by construction, because the prompt never contained the date. The whole apparatus exists to make that last distinction — real chaining versus parallel guessing — measurable.

## 4. Anatomy of a ToolHop instance

It helps to see the layers of a single instance stacked up, because the layering is what makes the evaluation honest. The stack below is the layer cake that every ToolHop problem ships.

![Five-layer stack of a ToolHop instance: the multi-hop query on top, then tool-set JSON schemas, then executable Python with real return values, then the feedback channel of results and exceptions, then the single verifiable answer](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-6.webp)

Read the stack top to bottom:

- **Query** — one multi-hop information need. This is the seed of the whole instance; everything below it exists to make this query answerable and checkable.
- **Tool set** — the JSON schemas the model sees, 3–7 per query, each with the refined parameter set from Stage 2 (avg 5.91 params, with structured/union types). This is the *interface* the model reasons over.
- **Executable Python** — the implementation beneath each schema. This is the layer most benchmarks omit; ToolHop ships it, so calls produce real return values rather than stubs.
- **Feedback channel** — what comes back after a call: the return value on success, or a real exception (TypeError, KeyError, a value error from a bad enum) on failure. This is the signal the model is supposed to read and act on. Whether a model *uses* this channel is the single biggest differentiator across families, as the error analysis will show.
- **Verifiable answer** — one exact-match string (a standardized "objective entity"). Because the code is deterministic and the gold was fixed at construction, scoring is unambiguous: the model's final answer either equals the gold entity or it does not.

The reason to draw this as a stack rather than a flow is that the layers are *commitments the benchmark makes*, not steps in time. Remove the executable-Python layer and you have a benchmark that can only check "did the model produce a plausible-looking call," not "did the chain actually compute the right thing." Remove the verifiable-answer layer and you are back to LLM-judging, with all its noise. ToolHop's credibility rests on keeping every layer in the cake. This is the same discipline argued for in [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer): the more of the execution you can ground in real signals, the less your eval drifts.

## 5. Decomposing a query into a tool-call chain

Stage 1's decomposition deserves its own picture, because the *structure* of the decomposition is what determines hop depth and therefore difficulty.

![Tree decomposition of one user query into atomic subqueries, each backed by one tool whose output feeds the next, ending in a final-answer tool](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-5.webp)

The tree shows the four-hop example from Section 1, decomposed the way ToolHop's construction does it. The root is the user query. It splits into atomic subqueries — find the author, map author to nation, map nation to currency — and each subquery is realized by exactly one tool: `lookup_author()`, `nationality_of()`, `currency_of()`, and finally `convert()` which produces the final answer. The leaves are the tools; the internal nodes are the reasoning steps the model must perform to know *which* tool to call next.

The thing to internalize from this tree is that **the model never sees this tree.** It sees the query and the flat list of tool schemas. It has to *reconstruct* this decomposition on the fly, one hop at a time, using the return of each call to figure out the next. The tree is the gold structure; the model's job is to traverse it without a map, and to do so while the environment is talking back to it (sometimes with errors). A model that "knows" the answer from pretraining — say, it has memorized the author's nationality — is *penalized* under mandatory tool use, because it is supposed to ground each step in a real call, and a guessed-but-correct value still skips a tool the harness expects. This is a subtle but important design choice: ToolHop is measuring *grounded chaining*, not trivia recall.

This also clarifies why the construction is query-driven rather than tool-driven. If you started from the four tools and tried to write a query, you would have to invent a question whose answer happens to require exactly this chain — and you would have no guarantee the dependencies are tight (maybe `currency_of` could be answered without `nationality_of` if the model just knows the author's country). Starting from the query and decomposing guarantees the chain is the *minimal* path to the answer, so every hop is load-bearing.

## 6. The evaluation loop: model-then-executor, with real feedback

Now the dynamic view — how a single instance is actually run.

![The ToolHop evaluation loop unrolled as a pipeline: query plus schemas to model hop 1, to executor return, back to model hop 2, to executor, to final answer, to exact-match scoring against the gold entity](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-7.webp)

The loop unrolls into alternating **model** and **executor** turns. The model receives the query plus the 3–7 callable tool schemas. It emits a tool call (hop 1). The executor *actually runs the Python* and returns a real value — or a real error. That return re-enters the model's context, and only then can the model emit hop 2, whose arguments depend on hop 1's return. The cycle repeats until the model decides it has enough to emit a final answer, which is scored by exact match against the gold entity.

Three design decisions in this loop are worth dwelling on:

1. **Real execution, not simulation.** Because the executor runs code, the return values are correct and the errors are real. A model that passes a string where an enum is required gets a real value error, not a polite "please try again." This is what makes the feedback channel a genuine test of recovery rather than a scripted retry.
2. **Feedback re-enters before the next hop.** The model cannot batch all its calls up front; it must observe each return before deciding the next call (in the serial setting). This is the closed loop from Section 2, instantiated. Models that try to *parallelize* — emit hop 2 before hop 1 has returned — hallucinate hop 2's arguments, because they literally do not have the value yet.
3. **Exact-match scoring at the end.** No partial credit for "called the right tools but got the wrong final value." The benchmark scores the *outcome*, which is the right thing to score for an agent: a travel agent that books the wrong flight gets no credit for having queried the right APIs.

This loop is also a clean template for your own eval harness, which is the subject of the next section's code. If you are building an internal tool-use eval, copy this structure: real executors, feedback that re-enters the context, and outcome-based scoring. The [eval-agents](/blog/machine-learning/ai-agent/eval-agents) post goes deeper on harness design choices; ToolHop is a concrete, well-built instance of those principles.

### A runnable multi-hop tool-chain harness

Here is a compact but faithful executor that captures ToolHop's loop: real Python tools, feedback (including exceptions) fed back into the model, serial execution, and exact-match scoring. It is written against the Anthropic Messages API tool-use interface, but the structure is provider-agnostic — swap the `client.messages.create` call for your provider's tool-calling endpoint and the loop is identical.

```python
import json
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

#--- 1. The locally executable tools (ToolHop Stage 3 analog) ---------------
#--- Real functions with real returns and real exceptions — not stubs.
_AUTHORS = {"The Stranger": "Albert Camus"}
_NATION  = {"Albert Camus": "France"}
_CURRENCY = {"France": "EUR"}
_FX = {("EUR", "USD", "1913-11-07"): 1.0852}

def lookup_author(book: str) -> str:
    return _AUTHORS[book]                      # KeyError on unknown book

def nationality_of(author: str) -> str:
    return _NATION[author]

def currency_of(country: str) -> str:
    return _CURRENCY[country]

def fx_rate(base: str, quote: str, date: str) -> float:
    if quote not in {"USD", "EUR", "GBP", "JPY"}:   # enum-typed param
        raise ValueError(f"unsupported quote currency: {quote}")
    return _FX[(base, quote, date)]

REGISTRY = {fn.__name__: fn for fn in
            (lookup_author, nationality_of, currency_of, fx_rate)}

#--- 2. The JSON schemas the model sees (ToolHop Stage 2 analog) ------------
TOOLS = [
    {"name": "lookup_author",
     "description": "Return the author of a book.",
     "input_schema": {"type": "object",
                      "properties": {"book": {"type": "string"}},
                      "required": ["book"]}},
    {"name": "nationality_of",
     "description": "Return the home country of a person.",
     "input_schema": {"type": "object",
                      "properties": {"author": {"type": "string"}},
                      "required": ["author"]}},
    {"name": "currency_of",
     "description": "Return the ISO currency code of a country.",
     "input_schema": {"type": "object",
                      "properties": {"country": {"type": "string"}},
                      "required": ["country"]}},
    {"name": "fx_rate",
     "description": "Return the FX rate base->quote on a date (YYYY-MM-DD).",
     "input_schema": {"type": "object",
                      "properties": {
                          "base":  {"type": "string"},
                          "quote": {"type": "string",
                                    "enum": ["USD", "EUR", "GBP", "JPY"]},
                          "date":  {"type": "string"}},
                      "required": ["base", "quote", "date"]}},
]

#--- 3. The evaluation loop (ToolHop's model<->executor cycle) ---------------
def run_instance(query: str, gold: str, max_hops: int = 8) -> dict:
    messages = [{"role": "user", "content": query}]
    transcript = []
    for hop in range(max_hops):
        resp = client.messages.create(
            model="claude-opus-4-8[1m]",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        if not tool_calls:                       # model emitted a final answer
            final = "".join(b.text for b in resp.content if b.type == "text")
            return {"correct": final.strip() == gold,
                    "answer": final.strip(), "hops": hop, "trace": transcript}

        results = []
        for call in tool_calls:                  # serial-style execution
            try:
                out = REGISTRY[call.name](**call.input)
                payload, is_err = json.dumps(out), False
            except Exception as e:                # REAL feedback, not a stub
                payload, is_err = f"{type(e).__name__}: {e}", True
            transcript.append({"hop": hop, "tool": call.name,
                               "args": call.input, "ret": payload,
                               "error": is_err})
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": payload, "is_error": is_err})
        # Feedback re-enters the context BEFORE the next hop.
        messages.append({"role": "user", "content": results})
    return {"correct": False, "answer": None, "hops": max_hops,
            "trace": transcript, "note": "hit max_hops without answering"}

#--- 4. Score one instance --------------------------------------------------
q = ("On the day the author of 'The Stranger' was born, what was the "
     "EUR->USD exchange rate? Answer with the number only.")
print(run_instance(q, gold="1.0852"))
```

A few things this skeleton makes concrete that prose cannot. The `is_error` flag on the tool result is the **feedback channel** — when `fx_rate` is called with an out-of-enum currency, the model gets `ValueError: unsupported quote currency: ...` back in its context, and a *good* model corrects on the next hop while a *bad* model ignores it and either repeats the error or answers anyway. The serial loop (feedback appended before the next `client.messages.create`) is what prevents the parallel-call hallucination failure mode. And the exact-match check (`final.strip() == gold`) is ToolHop's scoring: no judge, no fuzzy matching, no partial credit. If you wanted to reproduce ToolHop's three settings, you would vary one thing: **mandatory** forces `tool_choice={"type": "any"}` so the model must call a tool, **free** leaves `tool_choice` unset so the model decides, and **direct** removes `tools` entirely so the model must answer from parametric knowledge alone.

## 7. The results: tools double accuracy, and the ceiling is still under half

Now the headline table, read carefully.

![Matrix of answer correctness for six representative models across the Direct, Mandatory, and Free settings, color-coded by accuracy tier](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-4.webp)

The matrix shows answer correctness for six representative models across the three settings. Read the columns first:

- **Direct** (no tools, answer from parametric knowledge): the models are clustered low, **18–27%**, averaging **19.83%** across all 14. This is the floor — what the model knows without grounding. That it is non-zero at all is mostly memorization of common entities.
- **Mandatory** (model must call tools): accuracy jumps. GPT-4o reaches **49.04%**, GPT-4-Turbo **47.94%**, Qwen2.5-72B **45.43%**, Claude3.5-Sonnet **39.90%**, Gemini1.5-pro **31.16%**, LLaMA3.1-70B a dismal **19.10%**. The 14-model average is **32.12%**.
- **Free** (model decides whether to call tools): broadly similar to mandatory, averaging **32.84%**. The notable movers: Claude3.5-Sonnet actually does *better* free (**45.23%**) than mandatory (**39.90%**), while Qwen2.5-72B drops (**38.29%** free vs **45.43%** mandatory).

Three readings of this table are load-bearing.

**First: tools help, a lot, but not enough.** Going from Direct to Mandatory lifts the average by **+12.29 points** (19.83% → 32.12%), and the GPT family gains the most — about **+23.59 points** on average. So the benchmark is not impossible; tool access is genuinely the difference between knowing and not knowing. But the absolute ceiling — 49.04% for the best model — means *more than half of multi-hop queries are answered wrong even by the strongest model with mandatory tool access*. That is the sentence to put on the slide.

**Second: the mandatory-vs-free gap is small and inconsistent.** A natural hypothesis is that forcing tool use (mandatory) helps models that would otherwise lazily answer from memory, while free hurts them. The data only partly supports this. For most models the two columns are within a few points, and for Claude3.5-Sonnet free is *better*. The interpretation: the dominant failure mode is not "the model declined to use tools" — it is "the model used tools but broke the chain." If declining-to-use-tools were the bottleneck, mandatory would dominate free everywhere. It does not.

**Third: family matters more than size.** Qwen2.5-72B (45.43% mandatory) beats Claude3.5-Sonnet (39.90%) and crushes LLaMA3.1-70B (19.10%), despite the LLaMA being a strong general model. The gap is not about parameter count; it is about how each family's *tool-calling interface and training* handle the closed loop. This is the bridge to the error analysis.

### The full 14-model picture

The six-row matrix is the representative slice; the full mandatory-setting numbers across all 14 models tell the family story more completely.

| Family | Model | Mandatory accuracy |
|---|---|---|
| LLaMA3.1 | 8B | 12.76% |
| LLaMA3.1 | 70B | 19.10% |
| Qwen2.5 | 7B | 9.85% |
| Qwen2.5 | 14B | 26.38% |
| Qwen2.5 | 32B | 25.03% |
| Qwen2.5 | 72B | 45.43% |
| Gemini1.5 | flash-002 | 29.35% |
| Gemini1.5 | pro-002 | 31.16% |
| Claude3.5 | Haiku | 38.09% |
| Claude3.5 | Sonnet | 39.90% |
| GPT | 3.5-Turbo | 35.38% |
| GPT | 4o-mini | 40.20% |
| GPT | 4-Turbo | 47.94% |
| GPT | 4o | **49.04%** |

Notice the non-monotonicities. Qwen2.5-14B (26.38%) slightly *outscores* Qwen2.5-32B (25.03%) — a clue that something other than raw capacity is at play, which the paper attributes to differing parallel-call rates. GPT-3.5-Turbo (35.38%) beats both LLaMA3.1-70B and every Qwen below 72B, despite being an old and small model — because its tool-calling interface and feedback-handling are mature. The lesson for model selection is blunt: **for a multi-hop agent, evaluate the family's tool-calling behavior directly; do not extrapolate from MMLU or from single-hop function-calling scores.** A weaker general model with a better tool-loop can beat a stronger one with a worse loop. For the broader provider-by-provider view of tool-calling maturity, see the [model-context-protocol](/blog/machine-learning/ai-agent/model-context-protocol) post, which covers how the calling interface itself shapes these outcomes.

### A worked example: why depth crushes accuracy

The exact-match, no-partial-credit scoring interacts with hop depth in a way that is worth making quantitative, because it explains the shape of the degradation. Model the chain as a sequence of independent hops, each succeeding with probability $p$. A query of depth $d$ succeeds only if *every* hop succeeds, so its success probability is

$$P_{\text{success}}(d) = p^{\,d}.$$

This is a deliberately simplified model — real hops are not independent, and a model that breaks hop 2 may correlate with breaking hop 4 — but it captures the dominant effect. Plug in a generous per-hop success rate of $p = 0.90$:

| Hop depth $d$ | Queries at this depth | $P_{\text{success}} = 0.9^{\,d}$ |
|---|---|---|
| 3 | 428 | 72.9% |
| 4 | 353 | 65.6% |
| 5 | 136 | 59.0% |
| 6 | 10 | 53.1% |
| 7 | 68 | 47.8% |

Even at a per-hop reliability of 90% — which would be excellent — a 7-hop query succeeds less than half the time, purely from compounding. Now invert the question: if GPT-4o's *overall* accuracy is 49.04% and the query mass sits at 3–4 hops, what per-hop reliability does that imply? Weighting by the depth distribution, an overall 49% corresponds to a per-hop success rate around **0.83–0.85** — meaning the best model gets roughly one in six *individual hops* wrong. That single number is more diagnostic than the headline accuracy: it says the bottleneck is per-hop reliability, and that the only way to move the overall number is to fix the per-hop failures the error taxonomy enumerates. Halving the per-hop error rate (0.85 → 0.925) would lift a 4-hop query from ~52% to ~73% — a far larger swing than any prompt-tuning at the chain level. **Depth is a multiplier on your per-hop bug rate; fix the per-hop bugs.**

### The invocation-error rate is a second axis

Answer correctness is the outcome metric, but ToolHop also reports **invocation error rate** — the fraction of queries where the model produced a malformed or failing tool call. The average is **18.72%**, but the spread is enormous: GPT-4o keeps it to **9.45%**, while LLaMA3.1-8B hits **41.61%**. That is a four-fold difference in how often a model even produces a *runnable* call. This metric matters because it separates two kinds of failure: a model can be wrong because it broke the chain logically (correct calls, wrong composition) or because it could not form valid calls at all. LLaMA3.1's combination of high invocation-error rate and low accuracy says its problem is upstream — it struggles to emit well-formed calls in the loop — while GPT-4o's low invocation-error rate but sub-50% accuracy says its problem is the harder one: it forms valid calls but still loses the multi-hop thread.

## 8. The error taxonomy: four ways a chain breaks

This is where ToolHop earns its keep as a *diagnostic* benchmark rather than just a leaderboard. Because the tools are real and the feedback is real, the authors can attribute failures to specific, mechanistic causes — and those causes map cleanly onto specific model-family weaknesses.

![Grid of the four failure modes: hallucinated argument, wrong tool or type, ignored feedback, and premature stop, each with its cause and the model family it bites hardest with measured frequencies](/imgs/blogs/toolhop-multi-hop-tool-use-benchmark-8.webp)

The grid lays out four distinct failure modes, each in its own column: the mode (red), the mechanism that causes it (amber), and the family it bites hardest, with a measured number (neutral/green). Walk through them.

### Error 1 — Hallucinated argument (the parallel-call trap)

The mode: the model passes an argument it *guessed* rather than one it *read from a prior return*. The mechanism: the model emits parallel tool calls — it tries to call hop 2 (and hop 3) in the same turn as hop 1, *before hop 1 has returned a value*. Since the value does not exist yet, the model invents it. This is the red branch from the Section 1 DAG, and it is the single most damaging failure because it produces well-formed, confident, and wrong calls.

The family it bites: **Qwen2.5**. The paper measures parallel-call rates of **70.1% for Qwen2.5-14B and 75.08% for Qwen2.5-32B** — i.e., on three-quarters of queries, the 32B model fires parallel calls without first processing prior results, "leading to hallucinations in parameter value assignments." This is almost certainly *why* Qwen2.5-32B (25.03%) underperforms Qwen2.5-14B (26.38%) and falls so far short of Qwen2.5-72B (45.43%): the larger-but-not-largest models lean harder into parallel calling and pay for it. The fix is not a bigger model; it is a serial calling discipline (the loop in our Section 6 code, where feedback re-enters before the next hop). If you are running Qwen2.5 in an agent, **disable parallel tool calls for multi-hop chains** — the parallelism is an optimization for *independent* calls, and multi-hop calls are by definition dependent.

### Error 2 — Wrong tool or unsupported parameter type

The mode: the model cannot produce a valid call because the tool's schema uses a parameter type the model's tool-calling interface does not support. The mechanism: ToolHop's Stage 2 refinement deliberately introduces structured types, including **union-type parameters** (a parameter that can be one of several types). The family it bites: **Gemini1.5**, which "lacks support for union-type parameters," so it simply cannot handle tool lists that include those structures. This is a hard interface limitation, not a reasoning failure — and it is exactly the kind of thing a benchmark with realistic, structured schemas surfaces that a benchmark with single-string-argument tools never would. Gemini1.5-pro's 31.16% reflects, in part, queries it could not even attempt validly because the schemas exceeded its type system.

The engineering lesson: **your tool schemas are part of your model-selection criteria.** If your domain naturally has union-typed parameters (e.g., an `id_or_name` field that accepts an integer or a string), Gemini1.5 will fail on them regardless of how smart it is. Either flatten your schemas to the model's supported type system or pick a model whose interface covers your types.

### Error 3 — Ignored feedback (the recovery gap)

The mode: a tool call fails, the executor returns a real error, and the model *ignores it* — treating the error as a dead end and either giving up or answering as if the call succeeded. The mechanism: the model does not incorporate the feedback channel into its next decision. This is the failure that the whole "executable tools with real feedback" design is built to detect, and the result is the most striking finding in the paper.

The family split: **only GPT recovers.** ToolHop's Table 5 compares performance on error-containing queries with *detailed* feedback (the real exception message) versus *minimal* feedback. GPT-4o scores **47.87%** with detailed feedback but only **24.47%** with minimal feedback — a **+23.4-point** swing. GPT-4-Turbo shows the same pattern (**29.31%** vs **12.07%**). The implication: GPT models *read the error and correct the call*, so giving them rich error messages roughly doubles their success on hard queries. Other families do not show this responsiveness — for them, detailed feedback barely moves the needle, because they are not using it. This is a genuine, measurable capability difference, and it is invisible to any benchmark that returns canned stubs instead of real exceptions.

The product takeaway is immediate and actionable: **if you are running GPT-family models, invest in rich, structured error messages from your tools** — type the exception, name the offending parameter, suggest the valid range. You will get roughly 20 points of accuracy back on the queries that hit errors. If you are running a family that does not use feedback, that investment is wasted, and you instead need to prevent the errors up front (stricter argument validation, retries with reformatted prompts).

### Error 4 — Premature termination

The mode: the model stops and answers before the chain is complete — it skips a hop. The mechanism the paper identifies for LLaMA3.1 is specific and architectural: these models "cannot output both natural language text and tool call instances simultaneously," which restricts chain-of-thought *during* tool use. Without the ability to reason in text alongside emitting a call, the model loses the scratchpad it needs to track where it is in the chain, and it terminates early. The family it bites: **LLaMA3.1**, whose 8B variant has the worst invocation-error rate (**41.61%**) and whose 70B variant scores just 19.10% — barely above its tool-free Direct score (18.79%), meaning tools are providing almost no lift because the model cannot run the loop.

This one is the hardest to work around from the outside, because it is a property of how the model's tool-calling was trained. The partial mitigations: force an explicit "reasoning" turn between tool calls (a separate text-only generation that the model then conditions on), or use a ReAct-style scaffold that interleaves explicit Thought/Action/Observation steps so the reasoning lives in the prompt structure rather than needing to coexist with the call in one generation.

### Why all four converge on the same scoreboard

The unifying point of the grid: four mechanistically distinct failures — a timing bug (parallel calls), an interface gap (union types), a recovery gap (ignored feedback), and an architectural limit (no text+call) — all surface as **the same observable**: a wrong final entity at exact-match time. This is exactly why a single accuracy number is insufficient for choosing a model, and why ToolHop's value is the *attribution*. Two models can both score ~31% and be failing for completely different, completely fixable reasons. The benchmark tells you *which* reason, which tells you *which* fix.

## 9. Case studies: reading the failure modes through specific models

Let me turn the taxonomy into named, concrete incidents — the way you would diagnose them if these were your production agents.

### Case 1 — Qwen2.5-32B fires three calls into the void

A 5-hop currency query. Qwen2.5-32B, in its default parallel-calling mode, emits `lookup_author`, `nationality_of`, *and* `currency_of` in a single turn. But `nationality_of` needs the author name that `lookup_author` has not yet returned, and `currency_of` needs the country that `nationality_of` has not yet produced. So the model fills those arguments with hallucinated guesses — "Camus" → guesses "France" directly, skipping the lookup. Sometimes the guess is even right, which is worse, because it masks the bug. At 75.08% parallel-call rate, this is not an edge case for the 32B model; it is the default behavior. **Diagnosis:** timing bug. **Fix:** force serial calls; the 72B variant, which parallelizes less, scores nearly 20 points higher (45.43%).

### Case 2 — Gemini1.5-pro can't even form the call

A query whose `currency_of` tool, after Stage 2 refinement, has a parameter typed as a union (`country: string | {iso2: string}`). Gemini1.5-pro receives the schema, has no representation for union types, and either drops the parameter or emits an invalid call. The hop fails not because the model reasoned wrong but because the *interface* could not express the call. Across the queries with union-typed parameters, this caps Gemini's achievable score regardless of reasoning quality. **Diagnosis:** interface gap. **Fix:** flatten the schema to a supported type, or select a different model for union-heavy domains.

### Case 3 — GPT-4o reads the error and recovers

A query where `fx_rate` is first called with `quote="Euro"` instead of the enum value `"EUR"`. The executor raises `ValueError: unsupported quote currency: Euro`. GPT-4o, on the next hop, *reads that message*, maps "Euro" to the enum `"EUR"`, and re-issues the call correctly — landing the gold answer. The same instance with minimal feedback ("call failed") gives GPT-4o nothing to correct against, and it either repeats the error or gives up. This is the **+23.4-point** detailed-vs-minimal gap made concrete. **Diagnosis:** recovery, working as intended. **Lesson:** rich errors are free accuracy for GPT-family agents.

### Case 4 — LLaMA3.1-70B answers after two of four hops

A 4-hop query. LLaMA3.1-70B calls `lookup_author` and `nationality_of`, gets the country, and then — unable to interleave a "now I need the currency" reasoning step with another tool call — emits a final answer that is just the country, or a guessed rate. It terminated one or two hops early. Because tools provided almost no lift over its Direct score (19.10% vs 18.79%), this premature-termination pattern is pervasive, not occasional. **Diagnosis:** architectural (no text+call in one turn). **Fix:** ReAct-style scaffold that externalizes the reasoning.

### Case 5 — Claude3.5-Sonnet does better when free than when forced

Sonnet scores 45.23% in the **free** setting but 39.90% under **mandatory**. The likely mechanism: when forced to call a tool on every turn, Sonnet sometimes calls a tool even when it has already gathered what it needs, and the spurious call introduces an error or a distraction that derails the final answer. When free, it calls tools when they are genuinely needed and answers when it is ready, producing a cleaner chain. **Diagnosis:** over-eager forced calling. **Lesson:** "mandatory tool use" is not universally better; for some models, letting the model decide produces better chains. Match the calling policy to the model.

### Case 6 — GPT-3.5-Turbo punches above its weight class

GPT-3.5-Turbo (35.38%) outscores LLaMA3.1-70B (19.10%) and every Qwen2.5 below 72B, despite being older and smaller than several of them. The reason is entirely about the tool loop: GPT-3.5's function-calling was trained to call serially, read returns, and recover from errors — the three things the other models struggle with. **Diagnosis:** none; this is the *control* case showing that a mature tool loop beats raw capability. **Lesson:** the tool-calling interface and its training are first-class model-selection criteria, not an afterthought.

### Case 7 — the 7-hop tail is where everyone breaks

The 68 queries that need 7 tools are the stress test within the stress test. Every additional hop multiplies the chance of a broken hop: if a model has a 90% per-hop success rate, a 3-hop query succeeds ~73% of the time but a 7-hop query only ~48% (the compounding table above). The exact-match, no-partial-credit scoring means one broken hop anywhere in the seven zeroes the instance. This is why even GPT-4o's 49.04% overall hides a steep degradation with depth — the short-chain queries are carrying the average, and the long tail is where the closed-loop discipline truly gets tested. **Lesson:** if your product's queries are deep (5+ hops), budget for substantially lower reliability than the headline number suggests, and design for graceful degradation.

### Case 8 — Qwen2.5-72B versus its smaller siblings is a calling-policy story

The within-family Qwen2.5 progression is the cleanest natural experiment in the table: 7B at 9.85%, 14B at 26.38%, 32B at 25.03%, 72B at 45.43%. If capacity alone drove the score you would expect a clean monotone climb, but 32B *drops below* 14B — a textbook non-monotonicity. The paper's parallel-call measurements explain it: 14B fires parallel calls on 70.1% of queries, 32B on 75.08%, and the extra parallelism converts directly into more hallucinated arguments. The 72B variant pulls ahead not because it is twice as large but because it parallelizes less and processes returns more serially. The actionable read: **for a Qwen2.5 deployment, the calling policy is a bigger lever than the checkpoint size in the 14B–32B range.** A 14B with parallel calls disabled may well beat a 32B with them on; the benchmark cannot tell you that exactly, but the mechanism strongly implies it, and it is cheap to test on your own tools.

### Case 9 — the "correct guess" that hides the bug

The most insidious failure on ToolHop is the one that *passes*. When Qwen2.5-32B fires parallel calls and guesses `country="France"` for Camus instead of calling `nationality_of`, it is sometimes *right* — Camus really was French — and the chain happens to land the gold answer. The instance scores correct, but the model never grounded the hop; it got lucky on a memorized fact. This matters for two reasons. First, it means the *reported* accuracy slightly overstates grounded-chaining ability: some of the green cells are memorization wearing a tool-call costume. Second, it is a trap for practitioners — an agent that "works" on your common cases (popular authors, major currencies) by guessing will fail silently on the long tail (obscure entities it has not memorized) precisely where you most needed the tool. ToolHop's design mitigates but cannot fully eliminate this; the only robust defense in production is to *verify the grounding*, not just the answer — log whether each hop's argument traces to a prior return, and alert when the model skips a hop it should have taken. This is the trajectory-level evaluation argument again: the final answer is necessary but not sufficient evidence that the chain was sound.

## 10. How ToolHop compares to other tool-use benchmarks

ToolHop is not the first tool-use benchmark, and placing it against its peers clarifies what it uniquely measures.

| Benchmark | Hop depth | Tool execution | Answer scoring | What it isolates |
|---|---|---|---|---|
| Single-hop function-calling suites | 1 | Often stubbed | Tool/arg match | Tool selection + arg filling |
| Tool-selection benchmarks | 1 | None | Did model pick right tool | Retrieval over a tool menu |
| Simulated multi-tool benchmarks | 2–5 | LLM-simulated returns | LLM-judge | Plausibility of the plan |
| **ToolHop** | **3–7** | **Real local Python** | **Exact match vs gold** | **Closed-loop chain mechanics** |

The two columns that set ToolHop apart are **tool execution** and **answer scoring**. Benchmarks with stubbed or LLM-simulated returns cannot test the recovery loop, because their "errors" are scripted and their "returns" are made up — so they cannot distinguish a model that reads feedback from one that ignores it, and they cannot catch a hallucinated argument that *happens to look plausible*. Benchmarks scored by an LLM-judge inherit the judge's noise and can be gamed by confident-sounding wrong answers. ToolHop's real executors and exact-match-against-gold scoring close both gaps. The cost is construction effort — building 3,912 executable tools backward from 995 queries is expensive — which is exactly why the query-driven construction method (Section 3) is the contribution that makes the rest possible.

There is a sibling worth naming explicitly: a query-driven, verifiable-by-construction philosophy also shows up in [PaSa](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent), the paper-search agent in this series, where the agent's actions are grounded in real retrieval rather than simulated steps. Both share the conviction that an agent benchmark is only as trustworthy as the realism of its environment. And both sit under the broader [Model Atlas](/blog/machine-learning/bytedance-research-model-atlas) umbrella of ByteDance Research work this series covers.

## 11. What this means for building real agents

Pull the threads together into engineering guidance, because the point of reading a benchmark is to change what you build.

**Evaluate the tool loop, not the model.** The strongest predictor of multi-hop success in ToolHop is not model size or general capability — it is the family's tool-calling behavior: serial vs parallel default, type-system coverage, and feedback responsiveness. Before you pick a model for an agent, run a ToolHop-shaped eval (real executors, real feedback, exact-match) on *your* tools. A model that tops MMLU can be near the bottom here.

**Default to serial calls for dependent chains.** The parallel-call trap (Error 1) is the most common and most damaging failure. Parallelism is an optimization for *independent* calls; multi-hop calls are dependent by definition. Disable parallel tool calling for chains where each argument comes from a prior return — the 20-point gap between Qwen2.5-72B and its smaller siblings is largely this.

**Make your errors rich, then check whether your model reads them.** For GPT-family models, detailed exceptions are worth ~23 points on error-containing queries. Type the exception, name the offending parameter, give the valid range. But verify your specific model *uses* feedback — for families that ignore it, the better investment is preventing errors with strict argument validation before the call leaves your harness.

**Match the type system to the model.** If your tools use union types, enums, or nested objects, confirm your model's interface supports them. Gemini1.5's union-type gap (Error 2) is invisible until a structured schema hits it; then it is a hard cap. Either flatten the schema or pick a model that covers your types.

**Externalize reasoning when the model can't interleave it.** For models that cannot emit text and a tool call in one turn (LLaMA3.1, Error 4), use a ReAct-style scaffold so the chain-of-thought lives in the prompt structure. Otherwise the model terminates early and tools provide almost no lift.

**Budget reliability by hop depth.** Per-hop success rates compound. A model at 49% overall is carried by short chains; deep (5–7 hop) queries are far less reliable. If your product's questions are deep, design for graceful degradation — checkpoint intermediate results, allow the user to confirm a hop, or break the task into verified sub-tasks.

**Pick the calling policy per model.** "Mandatory tool use" is not universally better — Claude3.5-Sonnet does better free than forced. Test both policies; some models produce cleaner chains when allowed to decide when to call.

### The economics of a sub-50% ceiling

A 49% best-case accuracy is not just a research curiosity; it dictates the architecture of any product built on multi-hop tool use, and the per-hop reliability number from Section 7 is the lever. Take the implied per-hop success rate of ~0.85 for a strong model. A naive single-shot agent on a 4-hop query succeeds about `0.85⁴ ≈ 52%` of the time — coin-flip territory, unshippable for anything users depend on.

The standard fix is retries, and the math says how many you need. If a full-chain attempt succeeds with probability `s = 0.52`, then `n` independent attempts succeed with probability `1 − (1 − s)ⁿ`: two attempts reach `77%`, three reach `89%`, four reach `95%`. So a product targeting "95% of 4-hop queries answered correctly" needs roughly a four-attempt budget around a 49%-class model — which means four times the token cost and four times the latency per hard query, plus the orchestration to detect a bad chain and trigger the retry. That is the real price tag the benchmark is quoting, and it is why the per-hop reliability number matters more than the headline: every point of per-hop reliability you buy (through serial calls, rich errors, and validation) compounds across the chain and shrinks the retry budget super-linearly.

There is a cheaper lever than retries, and it is the one most teams skip: **hop-level checkpointing**. If you verify each hop's output before feeding it forward — a cheap sanity check, a type assertion, a confirmation step — you convert a multiplicative failure into an additive one. Instead of `0.85⁴`, you pay `0.85` per hop but catch and re-run the *single* failing hop rather than the whole chain. The cost is one validation pass per hop; the benefit is that a 4-hop query's failure probability drops from `~48%` to roughly `4 × 0.15 × (cost of one re-run)`, because you almost never have to redo more than one hop. For deep chains this is dramatically cheaper than whole-chain retries, and it degrades gracefully: a hop that keeps failing surfaces a specific, debuggable error rather than an opaque wrong final answer. The benchmark's verdict — that chains break at the hop level — is therefore also a design prescription: spend your reliability budget at the hop boundary, not on the whole chain.

### Reading ToolHop as a contract test for your own agent

The most durable way to use ToolHop is not as a number to cite but as a *template* you instantiate on your own tools. The benchmark's five properties — query-driven construction, meaningful interdependency, executable tools, real feedback, exact-match scoring — are a specification for a trustworthy agent eval, and they transfer directly. Concretely, to build a ToolHop-shaped contract test for a production agent: take ten real user questions that genuinely require chaining your tools; for each, write down the gold answer and the dependency DAG by hand; wire your *actual* tool implementations (not mocks) behind the harness so feedback is real; and score by exact match against the gold, with no LLM-judge in the loop. Run your candidate model under both mandatory and free tool policies, log every malformed call, and bucket every failure into the four-mode taxonomy from Section 8.

What you get is not a leaderboard rank but a *diagnosis* specific to your stack: the per-hop reliability of your tools with your model, the invocation-error rate of your schemas, and the dominant failure mode you should fix first. That diagnosis is what tells you whether to switch models, flatten a union-typed schema, enrich your error messages, or add hop-level checkpoints — the exact decisions Section 11 lays out, now grounded in your own numbers rather than the paper's 2024 snapshot. A benchmark that converts into a reusable contract test for your own system is worth far more than one you only read about, and ToolHop's construction recipe is precisely what makes that conversion mechanical rather than aspirational.

Make this part of your release process, not a one-time experiment. Tool-calling behavior changes between model versions — a provider can silently alter its default parallel-call policy or its function-calling format in a point release — so a contract test that you re-run on every model upgrade is the only way to catch a regression before your users do. Pin the gold answers, version the tool implementations alongside your code, and treat a drop in per-hop reliability as a failing test that blocks the upgrade. The teams that ship reliable agents are the ones that turned multi-hop evaluation into continuous integration rather than a paper they once read.

## 12. Critique: where ToolHop is strong, and where I would push back

A benchmark this carefully built still has boundaries worth marking, and a good engineer reads benchmarks adversarially.

**What it gets right.** The query-driven construction is the real contribution and it is excellent: it solves the meaningful-interdependency problem that dooms tool-pile-first benchmarks, and the executable-tools + exact-match-gold combination gives numbers I trust more than any LLM-judged tool benchmark. The error attribution — tying failures to parallel calls, union types, feedback handling, and text+call limits — turns a leaderboard into a diagnostic, which is far more useful for practitioners than a single ranking. The 49.04% ceiling is a genuinely important, well-supported result.

**Where I would push back.** First, **exact-match-against-a-single-gold-entity is strict to a fault.** Some multi-hop queries have legitimately multiple correct surface forms ("EUR" vs "Euro" vs "€"), and a model that computes the right value but formats it differently is scored wrong. The paper standardizes answers into "objective entities" to mitigate this, but the standardization itself is a judgment call, and some fraction of the "errors" are formatting, not reasoning. This means the *true* reasoning accuracy is probably somewhat higher than the reported numbers — though the *relative* model ranking is likely robust to it.

Second, **the tools are synthetic and constructed by the benchmark authors.** They are realistic in shape (5.91 params, structured types) but they are not real-world APIs with the messiness of production: rate limits, pagination, partial failures, schema drift, ambiguous documentation. A model that aces ToolHop's clean executable tools might still struggle against a flaky third-party API. ToolHop measures the *reasoning and chaining* skill cleanly precisely because it controls the environment — but that control is also a gap between the benchmark and production.

Third, **the models tested are a 2024 snapshot.** GPT-4o, Claude3.5-Sonnet, Qwen2.5, LLaMA3.1, Gemini1.5 — these are the frontier as of the paper, but tool-calling is exactly the capability that has improved fastest since. The 49.04% ceiling is a 2024 ceiling; the *interesting* question is whether the newest models have closed it, and the benchmark is the right instrument to check, but the specific numbers will date quickly.

Fourth, **the dataset is English-centric and the queries skew toward look-up-and-compose factual chains** (authors, currencies, dates). Multi-hop tool use in the wild also includes deeply procedural chains (set up a resource, configure it, deploy, verify) where the "tools" mutate state rather than return facts. ToolHop's chains are mostly read-only fact composition; stateful, side-effecting tool chains are a different and arguably harder problem it does not cover.

**What would change my mind about the headline conclusion** — that multi-hop tool use is fundamentally unsolved? Three things, concretely. (1) If a current frontier model (a 2025-2026 release) clears, say, **75%+** on ToolHop's mandatory setting *with the same exact-match-against-gold scoring*, that would show the ceiling was a snapshot artifact, not a fundamental limit, and I would revise "unsolved" to "rapidly being solved." (2) If a careful re-scoring with multiple acceptable gold forms (addressing the formatting critique above) lifted the reported accuracies by more than ~10 points across the board, I would conclude the benchmark *overstates* the difficulty and the real gap is narrower than 49% implies. (3) If an ablation showed that simply forcing serial calls + a ReAct scaffold + rich errors — the three fixes from Section 11, applied uniformly — pushed every family above 60% without any model change, I would conclude the failures are *harness* failures, not model failures, and that "multi-hop tool use is unsolved" should be restated as "multi-hop tool *harnessing* is under-engineered." My current read is that the truth is a mix: some of the gap is real model limitation (the feedback-recovery gap is a genuine capability difference), and some is fixable harness design. ToolHop is the instrument that lets you tell which is which, which is the highest compliment I can pay a benchmark.

## 13. When to reach for ToolHop, and when not to

**Reach for it when** you are selecting a model for an agent that chains dependent tool calls, when you are building a tool-use eval harness and want a proven template (real executors, feedback re-entry, exact-match scoring), or when you need to *diagnose* why an agent fails rather than just measure that it does — the error taxonomy is the value. It is also the right benchmark to cite when someone points at a single-hop function-calling score and claims their agent "does tool use"; ToolHop is the counter-evidence that single-hop scores do not predict multi-hop reliability.

**Do not reach for it when** your agent's tools mutate state and have side effects (ToolHop is read-only fact composition), when your tools are messy real-world APIs whose failure modes are operational rather than logical (rate limits, pagination — ToolHop's executors are clean), or when your queries are single-hop (you are over-testing; a function-calling suite is the right size). And do not treat the specific 2024 numbers as current — re-run the benchmark against your candidate models rather than trusting the published table, because tool-calling is the fastest-moving capability in the field.

The enduring lesson, independent of any leaderboard snapshot, is the one the first figure encodes: **a multi-hop query is a dependency DAG, and an agent's job is to traverse it without a map while the environment talks back.** Single-hop tool use is open-loop translation; multi-hop is closed-loop control, and the gap between the two is the gap between a demo and a product. ToolHop is the first benchmark in this series built to measure that gap honestly — real tools, real feedback, real verifiable answers — and its verdict, that the best model still clears under half, is the most useful thing it could possibly tell you before you ship.

## References

- **Paper:** "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use" — ByteDance Research, ACL 2025 Main Conference. [arXiv:2501.02506](https://arxiv.org/abs/2501.02506) · [HTML v2](https://arxiv.org/html/2501.02506v2)
- **Dataset:** [bytedance-research/ToolHop on Hugging Face](https://huggingface.co/datasets/bytedance-research/ToolHop) — 995 queries, 3,912 executable tools; dataset CC BY 4.0, code Apache 2.0.

### Related reading on this blog

- [ByteDance Research Model Atlas](/blog/machine-learning/bytedance-research-model-atlas) — the hub for this series of ByteDance Research deep-dives.
- [PaSa: an LLM paper-search agent](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent) — a sibling query-driven, verifiable-by-construction agent.
- [Advanced tool use in LLMs](/blog/machine-learning/ai-agent/advance-tool-use) — how tool use evolved from single calls to orchestrated chains.
- [Evaluating agents](/blog/machine-learning/ai-agent/eval-agents) — harness-design principles ToolHop instantiates.
- [Evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) — why grounding the execution in real signals keeps evals honest.
