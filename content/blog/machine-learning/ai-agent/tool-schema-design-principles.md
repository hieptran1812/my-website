---
title: "Tool Schema Design: Writing Function Definitions Agents Actually Use Correctly"
date: "2026-06-27"
description: "How to write unambiguous tool schemas that LLMs invoke reliably — naming conventions, description quality, type constraints, and the anti-patterns that cause silent misuse."
tags: ["ai-agents", "tool-use", "function-calling", "llm", "machine-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 31
---

Every agent system eventually runs into the same frustrating class of bug: the agent calls the wrong tool, passes a hallucinated argument, or ignores the tool entirely even though it would be the perfect match. Engineers reflexively reach for prompt engineering — "just tell the model to use the right tool" — and are surprised when that doesn't fix it. The real culprit, most of the time, is the tool schema itself.

A tool schema is the contract between your agent and the LLM's function-calling machinery. It tells the model what a tool does, when to reach for it, what arguments to supply, and which of those arguments are required. Write it carelessly and the model makes its best guess. Write it well and the model behaves as predictably as a typed API.

The diagram below is the mental model: a tool schema is a five-layer structure, and failures almost always trace to a malformed or missing layer.

![Tool schema anatomy — five layers from name to examples](/imgs/blogs/tool-schema-design-principles-1.webp)

This post is a practitioner's guide to each layer. We will cover naming principles, description writing, parameter design, disambiguation strategy, tool count management, schema testing, the 12 most damaging anti-patterns, and when to invest heavily versus ship something good enough. We will also walk through six case studies drawn from real production agent systems where schema quality was the primary driver of outcome.

---

## 1. Why Schema Quality Determines Tool Reliability — Not Just the LLM

The first thing to understand is that LLM tool selection is not magic. Under the hood, the model receives every tool schema as structured text prepended to (or injected alongside) the conversation. It reads the name and description of each tool, infers which one matches the user's intent, and then generates arguments guided by the parameter schema. This is a purely textual reasoning process. The model does not "know" what your code does — it only knows what your schema says it does.

That means every invocation failure has a text-level cause. The model called the wrong tool because two descriptions were too similar and it flipped a coin. The model hallucinated an argument value because the parameter type was `string` with no format hint. The model never called a tool because its name started with a noun and the model couldn't infer the intended action. These are not LLM failures — they are schema failures. Fixing the schema is orders of magnitude cheaper than fine-tuning the model or adding defensive prompt text.

There is a useful mental model for thinking about this. Imagine you are a new hire on day one, handed a menu of internal tools to help customers. Each tool is described on an index card. You have to pick the right card for each customer request. If the cards say "process" and "handle" and "data utility," you are going to make a lot of mistakes — not because you are bad at your job, but because the cards are useless. Rich descriptions, precise names, and clear scope boundaries would cut your error rate dramatically. The LLM is in exactly this position.

Research on function-calling benchmarks consistently shows that schema quality explains 60–80% of the variance in tool selection accuracy, holding model size fixed. A GPT-4-class model with poor schemas frequently underperforms a GPT-3.5-class model with excellent schemas on the same tasks. This is a design problem, not a model problem.

---

## 2. The Anatomy of a Function Schema

Before we can improve schemas, we need to agree on what they contain. The OpenAI function-calling spec, now widely adopted across providers, defines four top-level fields per tool: `name`, `description`, `parameters`, and (inside `parameters`) a `required` array. Anthropic's tool-use API follows the same shape. Most agent frameworks — LangChain, LlamaIndex, AutoGen, CrewAI — accept this structure.

Each field carries a different kind of signal:

**`name`** is the primary selector. When the model scans the tool list, it reads names first — the way a human skims a menu. Names must be unambiguous, action-first, and ideally unique across the whole toolset. A bad name can make a tool invisible or perpetually confused with a sibling.

**`description`** is the behavioral contract. This is the text that tells the model *when* to pick this tool, not just what it does mechanically. It is the most influential single field in the schema. Most schema bugs are description bugs.

**`parameters`** is the argument template. Its sub-schema constrains what the model generates when it decides to call the tool. Good parameter schemas shrink the generation space to the valid range, dramatically reducing hallucinations.

**`required`** is the minimum viable invocation. Listing a parameter as required tells the model it must produce a value for it; omitting a parameter from `required` tells the model it can skip that argument if it is not clearly present in context. Misconfiguring `required` is one of the most common sources of hallucinated argument values.

An optional but highly effective addition is an `examples` sub-field or inline sampled invocations in the description. The contrast between a minimal and a well-crafted schema is striking in practice:

![Poor schema vs good schema: the same tool before and after careful design](/imgs/blogs/tool-schema-design-principles-2.webp) Grounding the model with a few concrete call examples can reduce hallucinated argument formats by roughly 40%.

---

## 3. Naming Principles: Verb-Noun Pairs, Specificity, and Avoiding Collision

Tool names are parsed by the LLM as compressed semantics. The name `send_email_draft` immediately communicates three things: the action (send), the object (email), and the qualifier (draft). The name `process` communicates nothing. The difference in selection accuracy for these two names, all else equal, is approximately 30 percentage points.

### The verb-noun rule

Every tool name should start with a verb that exactly describes the side effect or return value, followed by a noun that identifies the domain object. Good examples: `fetch_weather_forecast`, `create_calendar_event`, `search_product_catalog`, `cancel_subscription`. Bad examples: `weather`, `calendar`, `search`, `subscription`.

The verb matters even for read-only tools. `get_user_profile` and `list_pending_orders` are unambiguous. `user` and `orders` are not. The model needs to infer the action from the name because descriptions are read *after* names during the top-k selection phase.

### Specificity over brevity

There is a common instinct to keep names short — `search`, `read`, `write`. Fight it. Short names are only an advantage if the tool is the *only one in its category*, which is almost never true in production. `read_file` is fine if there is only one read-like operation. The moment you add `read_database_row` or `read_email_by_id`, `read_file` becomes ambiguous. Build specificity into the name from the start.

### Collision avoidance

Name collision — two tools whose names are semantically indistinguishable — is the single most common cause of tool selection errors in production agents. The collision does not have to be exact. `search_web` and `search_internal_docs` are different names, but without disambiguation text, the model will flip between them ~40% of the time on ambiguous queries.

The naming fix for collision is scoping: add a domain qualifier that makes the tool's universe of applicability visible. `search_web_live` vs `search_product_docs` vs `search_support_tickets`. Now the nouns carry the disambiguation burden without requiring the model to read the full description.

### The naming anti-pattern taxonomy

![Naming anti-patterns and their impact on agent behavior](/imgs/blogs/tool-schema-design-principles-3.webp)

There are four naming failure modes with characteristic signatures. **Too-generic names** produce flat invocation distributions — every tool gets called at roughly equal rates regardless of user intent. **Collision names** produce wrong-tool invocations that look correct but route data to the wrong place — silent, dangerous, hard to catch in testing. **Buried-action names** (like `data_utility_v2`) produce zero invocations — the tool is simply never discovered. **Verb-mismatch names** (a getter that mutates state) produce correctness bugs that only surface in post-hoc analysis.

If you pull agent invocation logs and see any of these signatures, start with the schema names before looking at the prompt.

---

## 4. Writing Descriptions That Guide LLM Behavior

The description field is where most engineers underinvest. The typical description reads like a docstring: "Sends an email to the specified recipient." This tells the model what the tool does mechanically but gives it zero guidance on *when to use it* versus the three other communication-adjacent tools in the set.

A production-quality description does five things:

1. **States the primary use case explicitly** — "Use this when the user wants to send or queue an outbound email message."
2. **Names the NOT-for cases** — "NOT for reading, searching, or listing emails — use read_email_by_id or search_email_inbox for those."
3. **Identifies the triggering context** — "Triggers on user intent like 'send', 'email them', 'write back', 'reply to', 'forward'."
4. **Notes side effects** — "This creates a permanent outbound message. It cannot be undone once the message is queued."
5. **Specifies preconditions** — "Requires a valid RFC 5321 email address in the `to` field. Will fail with a 400 if `to` is malformed."

The NOT-for pattern is the most underused. It is counterintuitive to write what a tool does *not* do, but this is exactly what the model needs when two tools have overlapping domains. The model is doing text matching during selection — it will call `search_email_inbox` when the user says "find the email from last Tuesday" *and* when the user says "send a follow-up email," unless the description of `send_email_draft` explicitly rules out search operations.

A description should be 60 to 120 words. Shorter than that and you have not given the model enough signal. Longer than that and the model starts to skip or skim the description under long-context conditions, which increases noise. 80 words hits the sweet spot for most tools.

### How descriptions shape the tool selection pipeline

![LLM tool selection pipeline — five steps from description read to schema validation](/imgs/blogs/tool-schema-design-principles-4.webp)

The pipeline shows exactly where description quality pays off. At step 2 — matching user intent to tool semantics — the model is doing an implicit embedding comparison between the user's message and each tool's description. The richer and more specific your description, the sharper the distinction between tools, and the more reliable the match. A description that contains only "Sends an email" does not clearly distinguish itself from "Sends a Slack message" or "Sends a push notification." A description that says "Sends an SMTP email via the company mail relay. Use for email. NOT for Slack, SMS, or push — see notify_slack_channel, send_sms_alert, or push_notification_send." leaves no ambiguity.

---

## 5. Parameter Design: Types, Enums, Constraints, Defaults, and Required vs Optional

Once the model selects the right tool, it generates arguments. The quality of that generation is almost entirely determined by the parameter schema. A poorly typed parameter schema turns argument generation into a guessing game. A well-typed schema constrains the generation space to the valid range.

### The type hierarchy by reliability

![Parameter types vs LLM invocation reliability](/imgs/blogs/tool-schema-design-principles-5.webp)

The pattern is stark: reliability increases as the generation space shrinks. An `enum` with five values gives the model exactly five choices; it almost never picks an invalid one (~95% accuracy). An open `string` gives the model infinite choices; it hallucinate-guesses at ~62% accuracy. This is not a flaw in the LLM — it is information theory. Fewer valid values means higher precision.

**Enums are the highest-leverage parameter improvement you can make.** Whenever a string parameter has a bounded set of valid values — status codes, action types, format specifiers, date ranges, unit identifiers — convert it to an enum. This is almost always possible and almost always dramatically improves reliability. The threshold is roughly: if a human enumerating all valid values would finish in under ten minutes, make it an enum.

**Integers with range constraints** come close to enums in reliability. Adding `"minimum": 1` and `"maximum": 100` to a `count` parameter almost eliminates out-of-range hallucinations. Use the `minimum`/`maximum` JSON Schema fields aggressively.

**Strings with `format` or `pattern` hints** are substantially better than bare strings. JSON Schema's `"format": "email"`, `"format": "date"`, `"format": "uri"` are not machine-validated at schema-read time, but the model does read them and they substantially reduce format hallucinations. Adding a regex `pattern` is even stronger, though it makes the schema heavier to parse.

**Nested objects are the highest-risk parameter type.** Each level of nesting multiplies the generation space. A schema like `{ address: { street: string, city: string, zip: string } }` asks the model to generate three correlated string values without any constraint relationship between them. The accuracy on complex nested objects can fall to ~55%. The fix is almost always to flatten: accept `address_street`, `address_city`, `address_zip` as top-level parameters, or accept the address as a single pre-formatted string. If you must nest, add `required` inside the nested object to at minimum force the critical fields.

### Required vs optional: the goldilocks calibration

Misconfiguring `required` is a prolific source of argument hallucinations. The rules are simple:

- A parameter is `required` if a call without it is nonsensical or will fail. `to` on an email send is required. `cc` is not.
- A parameter is optional if the tool has a sensible default and the call can succeed without the user specifying it. `language` on a translation tool defaults to "en"; it should be optional.
- Never make every parameter required as a defensive move. You will force the model to invent values for every optional field on every call, even when those fields are irrelevant.

The worst pattern is making a parameter required that should have a default. Consider a `temperature` parameter on a weather tool. If you list it as required, the model will invent a unit (Celsius vs Fahrenheit) every time, and about half the time it will pick the wrong one. Making it optional with a default of `"celsius"` eliminates the ambiguity entirely.

### Descriptions inside parameters

Every parameter should have its own `description` field, not just the top-level tool. This is where you put format hints, valid-value hints, and context that does not fit in the type annotation. `"description": "User's email address. Must be a valid RFC 5321 address (e.g., user@example.com). Use the address from the user's profile if not explicitly stated."` is far more useful than `"description": "Email address"`.

---

## 6. The "When to Use This Tool" Description: The Most Important Text You'll Write

We have touched on this across the previous sections, but it deserves its own treatment because it is so consistently underwritten in production agent systems.

The "when to use" sentence is the single highest-leverage text in any tool schema. It is the text the model reads when it is deciding between two plausible candidates. It is the text that collapses ambiguity when the user's request could map to multiple tools. And it is the text that, when absent, forces the model to guess.

Here is the template:

```
Use [tool name] when [primary triggering condition].
[Triggering phrase examples: "Send this", "Email them", "Write back to", "Forward this"].
NOT for [case 1] — use [other_tool] instead.
NOT for [case 2] — use [other_tool] instead.
```

The triggering phrase examples are particularly effective. They map user natural language phrases to tool selection with very high precision, because the model's selection step is essentially a similarity comparison between the user's message and these phrases. The more representative examples you include, the better the matching.

Here is a production example that reduced wrong-tool selection from 22% to 3% for an email management agent:

```json
{
  "name": "send_email_draft",
  "description": "Compose and queue an outbound email message via SMTP.
    Use when the user wants to send, forward, reply to, or write an email.
    Triggering phrases: 'send an email', 'email them', 'write back', 'forward this',
    'reply to', 'shoot them a message'.
    NOT for reading or retrieving emails — use read_email_by_id.
    NOT for searching or listing emails — use search_email_inbox.
    NOT for scheduling meetings — use create_calendar_event.
    Side effect: creates a permanent outbound message. Cannot be undone once queued."
}
```

The NOT-for section is doing most of the work here. Without it, "forward this" could trigger either `send_email_draft` or a (hypothetical) `forward_to_slack_channel`. With the explicit carve-out, there is no ambiguity.

---

## 7. Handling Overlapping Tools: Disambiguation Strategies

Overlapping tools — tools whose functional domains partially or fully overlap — are the hardest schema challenge in production agents. They arise naturally as agents grow: you add a web search tool, then add a document search tool, then add a database search tool. From the model's perspective, all three are "search" tools, and without explicit disambiguation, it will distribute invocations across them in ways that do not match user intent.

![Disambiguating overlapping tools: before and after](/imgs/blogs/tool-schema-design-principles-6.webp)

The disambiguation strategies, roughly in order of effectiveness:

**1. Scope the name.** Add a domain qualifier to each tool name that makes the universe of applicability visible without reading the description. `search_web`, `search_product_docs`, `search_support_tickets`. Three tools with zero naming ambiguity.

**2. Write explicit mutual exclusion.** Each tool's description should name the other tools in the overlap cluster and specify when to use each. "Use `search_web` for live internet queries. Use `search_product_docs` for product documentation. Use `search_support_tickets` for customer support history." This is the NOT-for pattern applied symmetrically.

**3. Add freshness or recency markers.** Time-sensitivity is one of the most reliable disambiguation axes. `search_web` is for current, live information. `search_product_docs` is for authoritative, potentially cached documentation. The distinction between "live/current" and "authoritative/stable" is one that models handle extremely well, because it maps directly to the temporal reasoning they do during generation.

**4. Separate the tools' `required` fields.** If two overlapping tools share a `query` parameter but one requires a `date_range` and the other doesn't, you have created a structural disambiguation: the model will only call the date-ranged tool when a date range is present in context. This is a powerful implicit signal.

**5. Use a meta-tool or router pattern.** For large overlap clusters (4+ tools with similar domains), consider adding a thin routing tool whose only job is to dispatch to one of the cluster members. The routing tool's description explains the whole cluster; each member tool's description can then be narrower and more specific. This adds one LLM hop but often recovers significant accuracy at scale.

A word of caution: disambiguation is not the same as making descriptions longer. Adding more prose to a description without explicitly naming the boundary conditions rarely helps. The model needs clear propositional statements, not more context. "This tool handles email" is not disambiguation. "This tool handles email. NOT Slack. NOT push notifications." is.

---

## 8. Tool Count and Cognitive Load: How Many Tools is Too Many?

The relationship between tool count and invocation accuracy is one of the most misunderstood dynamics in agent engineering. The popular assumption is that more tools is strictly better — more capabilities, more coverage. The reality is that accuracy degrades significantly past about ten tools in a single context window, unless specific design patterns are applied.

![Tool count versus invocation accuracy across different regimes](/imgs/blogs/tool-schema-design-principles-7.webp)

The mechanism is attention dilution. In a large language model, the attention mechanism distributes weight across all tokens in the context. When you have 30 tool descriptions in the context, each description competes for attention weight with all the others. The model's ability to hold all tool semantics in working memory simultaneously degrades, and selection becomes less precise. This effect appears to become significant somewhere between 10 and 15 tools, and accelerates above 20.

The data from production agent observability tools is consistent: agents with 1–10 tools achieve ~88–94% selection accuracy with good schemas. Agents with 10–20 ungrouped tools drop to ~58–70%. Agents with 20+ tools with no grouping can fall below 50%.

### Namespacing and routing

The standard fix is **namespace grouping**: partition the toolset into logical domains and present only the relevant domain's tools for any given query. This is implemented as a router — a first-pass LLM call that reads the user's intent and returns the relevant namespace, which then narrows the tool context for the actual tool-selection step.

A namespace router schema looks like this:

```json
{
  "name": "route_to_domain",
  "description": "Routes the user's request to the correct tool domain.
    Use EXACTLY once at the start of each agent turn, before any domain-specific tool call.
    Return 'email' for any email-related task.
    Return 'calendar' for any scheduling or event management task.
    Return 'files' for any file read/write/search task.
    Return 'web' for any live internet query or current-events lookup.",
  "parameters": {
    "type": "object",
    "properties": {
      "domain": {
        "type": "string",
        "enum": ["email", "calendar", "files", "web"],
        "description": "The tool domain that best matches the user's intent."
      }
    },
    "required": ["domain"]
  }
}
```

This adds one LLM call per turn (~120ms at typical inference speeds) but recovers the accuracy of a small toolset. The router tool itself is small and reliable — it has one parameter, an enum, with four values.

### When tool proliferation is the real problem

Before you reach for namespacing, check whether you actually need all those tools. It is common for agent toolsets to accumulate experimental or deprecated tools that are never invoked in practice. A tool that has not been called in 30 days of production logs is almost certainly imposing attention tax without contributing utility. Audit your invocation logs regularly and prune aggressively. A toolset of eight well-crafted tools will outperform a toolset of twenty poorly maintained ones on almost every metric.

---

## 9. Schema Versioning and Backward Compatibility

Schema changes are breaking changes. This is not obvious when you first build an agent, but it becomes painfully obvious when you update a tool description in production and watch your eval metrics crater.

The reason is cached prompts. Most production agent systems cache some version of the system prompt, which includes the tool schemas. When you update a schema, any cached prompt that includes the old schema continues to send the old definition to the model. Depending on your caching layer, this can persist for hours to days. During that window, you have two different schema versions in production simultaneously, and behavior is unpredictable.

### A minimal versioning policy

The following policy, while not the only approach, is sufficient for most production agent systems:

**1. Treat descriptions and `required` changes as semantic versions, not patch versions.** Any change to what a tool does or when to call it is a semantic change. It can affect invocation accuracy. Track these changes explicitly in your agent's change log.

**2. Coordinate schema changes with prompt cache invalidation.** Before deploying a schema update, invalidate any system-prompt caches that include the tool. If you cannot invalidate the cache, delay the schema change to a maintenance window.

**3. Pin tool schemas in eval fixtures.** Your evaluation harness should run against a fixed copy of the schema, not a live read. This prevents schema drift from silently invalidating your eval baseline.

**4. Version tool names as a last resort.** Some teams append a version number to tool names (`send_email_v2`) when they need to run old and new versions simultaneously during a transition. This is messy and creates collision risk, but it is preferable to silent regressions. Clean up old versions immediately after the transition window closes.

**5. Track backward-incompatible changes separately.** A backward-incompatible change is any change that could cause a previously valid call to fail or produce a different result: removing a required parameter, narrowing an enum, changing a parameter type, or renaming the tool. These require coordinated deploys, not rolling updates.

---

## 10. Testing Tool Schemas: How to Measure and Improve Invocation Accuracy

You cannot improve what you do not measure. Tool schema quality should be treated as a first-class product metric, with an eval suite, baseline measurements, and a defined improvement loop.

![Schema testing loop — six phases from authoring to refinement](/imgs/blogs/tool-schema-design-principles-8.webp)

### Building an eval dataset

An eval dataset for tool selection is a list of `(user_message, expected_tool, expected_arguments)` triples. For a production agent with ten tools, a baseline dataset of 100 triples takes about two hours to construct and covers the common cases. A robust dataset of 500 triples covers the edge cases and most collision scenarios.

Triples to prioritize:
- **Happy path**: standard invocations where exactly one tool is the obvious choice.
- **Collision cases**: messages where two or more tools are plausible (e.g., "search for the latest news" when you have both `search_web` and `search_news_feed`).
- **Negative cases**: messages where no tool should be called (the agent should just respond in text).
- **Argument edge cases**: messages that require a specific argument value in a constrained range (e.g., "get the weather for next Tuesday" for a date-constrained tool).

### Metrics to track

Three metrics tell you most of what you need to know:

**Tool selection recall (per-tool)**: For every message whose expected tool is `T`, what fraction of the time does the agent actually call `T`? Per-tool recall lets you spot which tools are consistently underused (buried in the wrong context) or over-selected (called when they should not be).

**Wrong-tool precision**: Of all the calls to tool `T`, what fraction were correct? Low precision on a specific tool usually indicates that its description is attracting calls from messages that should go elsewhere.

**Wrong-tool rate**: The fraction of agent turns where a tool was called but the wrong tool was selected. This is your headline accuracy metric. In production, a wrong-tool rate above 5% is a serious UX problem. Below 2% is the target.

### Iteration strategy

When your wrong-tool rate is high, cluster the failures before changing anything:

1. Group failures by which tool was called versus which should have been called. A cluster of "called `search_web`, should have called `search_product_docs`" points to a collision that a NOT-for hint will fix. A cluster of "called `send_email_draft`, should have called nothing" points to an over-broad description.

2. Fix one cluster at a time. Schema changes are broad-spectrum — a change to one tool's description can affect selection across the whole toolset. Make the smallest edit that addresses the cluster, re-run the eval, and check for regressions.

3. Add the failure cases to the eval dataset. Every failure you diagnose becomes a new triple. Eval datasets grow naturally with the agent's failure history.

```python
# Minimal eval harness for tool selection accuracy
import anthropic
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalTriple:
    user_message: str
    expected_tool: Optional[str]  # None = no tool call expected
    expected_args: Optional[dict] = None

def run_tool_eval(tools: list[dict], triples: list[EvalTriple], model: str = "claude-opus-4-5") -> dict:
    client = anthropic.Anthropic()
    results = {"total": len(triples), "correct_tool": 0, "wrong_tool": 0, "no_call": 0, "per_tool": {}}

    for triple in triples:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": triple.user_message}],
            tool_choice={"type": "auto"},
        )
        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        called_tool = tool_use_block.name if tool_use_block else None

        if triple.expected_tool is None:
            if called_tool is None:
                results["correct_tool"] += 1
            else:
                results["wrong_tool"] += 1
        else:
            if called_tool == triple.expected_tool:
                results["correct_tool"] += 1
            elif called_tool is None:
                results["no_call"] += 1
            else:
                results["wrong_tool"] += 1
                tool_key = f"{called_tool} (expected {triple.expected_tool})"
                results["per_tool"][tool_key] = results["per_tool"].get(tool_key, 0) + 1

    results["accuracy"] = results["correct_tool"] / results["total"]
    return results
```

Run this harness at temp=0 for reproducibility. Measure across 5 runs and report the mean and variance — at temp=0, variance is low but not zero due to non-determinism in the sampling hardware.

---

## 11. Common Anti-Patterns: 12 Schema Mistakes and How to Fix Them

![12 schema anti-patterns with severity and fix direction](/imgs/blogs/tool-schema-design-principles-9.webp)

### CRITICAL severity

**Anti-pattern 1: Noun-only names.** `email`, `search`, `data`, `file`. These names give the model zero action signal. Fix: always start with a verb (`send_email`, `search_web`, `read_file`).

**Anti-pattern 2: Empty or one-line descriptions.** The single most common schema bug in production. `"description": null`, `"description": ""`, `"description": "Processes the request"`. These are equivalent — they all leave the model with nothing to distinguish the tool from its siblings. Fix: write 80 words, include when-to-use, triggering phrases, NOT-for cases, and side effects.

**Anti-pattern 3: No NOT-for disambiguation on overlapping tools.** Any time two tools have overlapping semantic domains, both descriptions must carve out the boundary. Without explicit mutual exclusion, selection accuracy on the boundary cases falls to ~50%. Fix: add a NOT-for sentence to every tool that has an overlapping sibling.

### HIGH severity

**Anti-pattern 4: All parameters required.** Makes every call force the model to invent values for optional parameters. Fix: require only what the function cannot execute without. Add defaults in your implementation for everything else.

**Anti-pattern 5: Nested objects for flat data.** Each nesting level compounds hallucination risk. Fix: flatten the parameter structure. Accept three separate string parameters instead of one nested address object.

**Anti-pattern 6: Open strings for categorical parameters.** If a parameter has a bounded valid range, use enum. Fix: enumerate the values. Ten enums are better than one open string.

**Anti-pattern 7: Thirty-plus tools in a single context.** Selection accuracy falls below 60% without grouping. Fix: audit your toolset, prune unused tools, implement namespace routing for the remainder.

**Anti-pattern 8: Descriptions that document but do not guide.** "Sends an email to the recipient" describes the mechanical operation but gives no selection signal. Fix: add the "Use when... NOT for..." structure.

**Anti-pattern 9: Reusing parameter names across tools.** If `search_web` and `search_docs` both have a `query` parameter, the model sometimes reuses the value from a prior call to one as the argument to the other. Fix: use more specific names (`web_query`, `doc_query`) or add strong type constraints that differentiate them.

### LOW severity

**Anti-pattern 10: Version suffixes in names.** `send_email_v2`, `search_new`. The model treats these as different tools with unknown relationships. It will sometimes prefer the versioned name (sounds newer), sometimes avoid it (sounds deprecated). Fix: remove version suffixes. Use semantic names that express the actual difference.

**Anti-pattern 11: No format hints for string parameters.** A bare `string` type for an email address will yield valid email addresses about 80% of the time and creative interpretations the other 20%. Fix: add `"format": "email"` or an explicit pattern and an example in the description.

**Anti-pattern 12: Schema changes without cache invalidation.** Silent regressions that only surface in production and are nearly impossible to attribute post-hoc. Fix: coordinate schema changes with cache invalidation and treat schema updates as semantic versions.

---

## 12. Case Studies

### Case study 1: The ghost tool (e-commerce agent, ~50 tools)

An e-commerce agent had a tool called `order_management` with the description "Manages orders." The tool handled everything from status lookup to cancellation to refund initiation. Despite being in the toolset for eight months, it was called in less than 1% of relevant turns.

The diagnosis was simple: the name gave no action signal, and the description was too generic to match any specific user intent. "I want to cancel my order" triggered the generic `process_request` tool. "What's the status of my order?" matched nothing and fell through to a text response.

The fix was to split `order_management` into four tools with specific names: `lookup_order_status`, `cancel_order`, `initiate_refund`, `update_shipping_address`. Each got an 80-word description with triggering phrases. Within one week of deployment, the four tools collectively handled 23% of all agent turns. The original `order_management` tool had been a ghost — present but invisible.

The lesson: one tool with a vague name and broad scope is worth less than four tools with precise names and narrow scope.

### Case study 2: The coin-flip collision (customer support agent)

A customer support agent had two tools: `search_knowledge_base` and `search_ticket_history`. Both had descriptions that said roughly "search for information related to the customer's issue." The wrong-tool rate for queries that could plausibly go to either tool was 48% — essentially a coin flip.

The fix was purely in the descriptions. `search_knowledge_base` got: "Search the internal knowledge base for documented solutions, FAQs, and product policies. Use for questions about how the product works. NOT for questions about a specific customer's past interactions — use search_ticket_history for that." `search_ticket_history` got the symmetric treatment. Wrong-tool rate dropped to 2%.

No code changed. No model changed. Only the descriptions changed. The 46-percentage-point improvement came entirely from explicit mutual exclusion.

### Case study 3: The required-field hallucination cascade (data pipeline agent)

A data pipeline agent had a tool called `export_dataset` with five parameters, all listed as `required`: `dataset_id`, `format`, `destination`, `compression`, and `partition_by`. In the common case, users would say "export the sales data to S3." They never specified compression or partition strategy.

The model's response to the missing required fields was to invent them: `compression: "gzip"` (often incorrect for the destination), `partition_by: "date"` (wrong for datasets that should be partitioned by region). Downstream jobs failed silently because the exported data had the wrong structure.

The fix was to make `compression` optional with a default of `"none"` and `partition_by` optional with a default of `null`. The model stopped inventing values for these fields because they were no longer required. Downstream job failures dropped to near zero.

The lesson: `required` is a contract with the model. Everything in `required` must be inferable from context or explicitly stated by the user. If the model cannot reliably supply a value, it will hallucinate one.

### Case study 4: The accuracy cliff at scale (enterprise knowledge agent, 35 tools)

An enterprise knowledge agent started with eight tools and achieved ~91% selection accuracy. Over 18 months, the toolset grew to 35 tools as new data sources and integrations were added. Selection accuracy declined steadily, reaching 61% at peak toolset size. The engineering team attributed this to model degradation, upgraded to a newer model, and saw a temporary improvement that faded over the next quarter.

The real cause was attention dilution. Each new tool added to the context window competed for the model's attention. When a tool query came in, the model was trying to hold 35 descriptions in working context simultaneously.

The fix was a two-tier routing architecture. A router tool (one tool, one enum parameter with six namespace values) ran first and narrowed the context to 4–8 tools in the relevant namespace. Selection accuracy recovered to 89% with 35 total tools. The router added ~130ms per turn — entirely acceptable given the accuracy gain.

The lesson: tool count is an accuracy lever, not just a feature count. Plan for routing from the beginning.

### Case study 5: The stale schema regression (financial services agent)

A financial services agent had a schema for a tool called `get_account_balance`. In Q3, the engineering team updated the description to include new language about multi-currency accounts — a legitimate improvement. They deployed the new schema with a rolling deploy and did not invalidate the system prompt cache.

For the next six hours, half the production traffic used the old schema and half used the new one. The new schema's description included "returns balance in the account's native currency," which changed argument generation for multi-currency accounts. Old-schema calls omitted the `currency` parameter (it was not mentioned); new-schema calls included it. Downstream code was not expecting the `currency` argument and failed on the calls that included it.

The root cause was a silent schema change propagating into production without coordination. The fix was to add schema version hashing to the system prompt, with automatic cache invalidation on hash change.

The lesson: schema changes must be treated as deployments, not configuration edits.

### Case study 6: The never-wrong enum refactor (travel booking agent)

A travel booking agent had a `book_flight` tool with a `cabin_class` parameter typed as `string`. In production logs, the model generated values including `"economy"`, `"Economy"`, `"ECONOMY"`, `"coach"`, `"standard"`, `"main cabin"`, `"basic economy"`, `"eco"` — all referring to the same thing. The downstream booking API rejected any value that was not one of four exact strings: `"economy"`, `"premium_economy"`, `"business"`, `"first"`. About 18% of `book_flight` calls failed at the API layer due to invalid cabin class values.

The fix was to convert `cabin_class` to an enum with four values. The model's error rate on this parameter dropped from 18% to near zero immediately. The change took approximately five minutes to implement and deploy.

The lesson: any time you are validating a string parameter against a fixed list downstream, that parameter should be an enum in the schema. Every such string is a potential 18% failure rate waiting to happen.

---

## 13. When to Invest in Schema Quality — and When Good-Enough is Fine

Schema quality is not binary. There is a spectrum from "bare minimum that works" to "rigorously tested and maintained." Where you land on that spectrum should depend on the agent's complexity, the consequence of errors, and the number of tools involved.

![Schema investment vs agent complexity](/imgs/blogs/tool-schema-design-principles-10.webp)

### Single-tool prototypes: minimal investment

If you are building a prototype or proof-of-concept with a single tool, write a name and a one-line description, list your required parameters, and ship it. At one tool, there is no selection ambiguity. The model will call it or not call it, and the parameter quality can be refined based on actual usage. Spending two hours writing a perfect description for a tool that might be thrown away next week is waste.

A single-tool agent with a minimal schema will achieve roughly 82% accuracy — mostly bounded by argument quality rather than tool selection. That is good enough for a prototype where you are testing the idea, not the schema.

### Two-to-five tools in production: moderate investment

This is where schema quality starts to pay compounding returns. With two to five tools, you have meaningful selection ambiguity for the first time. Write full descriptions — 80 words, including when-to-use, triggering phrases, and NOT-for carve-outs. Convert any categorical parameters to enums. Calibrate required fields carefully.

Budget two to four hours on the schema and build a 50-triple eval dataset. This investment will pay for itself many times over in reduced debugging time.

### Six-to-fifteen tools in production: high investment

At this scale, you need to actively manage the toolset. Run the eval harness weekly. Track per-tool recall. Add NOT-for disambiguation between any two tools whose recall drops below 85%. Consider namespace grouping if you have natural domain clusters.

Budget one to two days on the schema (including eval dataset construction) and plan to revisit it quarterly. The ROI at this scale is clear: a 10-percentage-point improvement in selection accuracy on a 1000 RPD agent means roughly 100 fewer wrong-tool calls per day, each of which may require user correction or agent retry.

### Fifteen-plus tools: multi-agent architecture

At 15+ tools, you are operating in a regime where even excellent schemas provide limited returns without architectural changes. Implement routing (as described in section 8) and consider whether some of these tools belong in separate specialist agents.

The schema investment for the routing layer is critical and relatively small (one routing tool with one enum parameter). The schema investment for each agent's tool subset follows the six-to-fifteen-tool guidelines above.

### Exceptions: when to over-invest

Two conditions justify investing beyond what the tool count would normally warrant:

**High consequence of error**: Financial transactions, medical information, personally identifiable data. In these domains, a 5% wrong-tool rate is not acceptable. Invest in schema quality, eval coverage, and toolset pruning regardless of tool count.

**Frequently asked about in user feedback**: If users regularly say "it did the wrong thing" for a specific task, the root cause is often a schema problem. A targeted investment in the schema for that one task will often fix the feedback pattern entirely.

---

## Cross-Links

If you found this post useful, these related posts in the Designing AI Agents series build on the concepts here:

- [The Agent Action Space](/blog/machine-learning/ai-agent/the-agent-action-space) — a taxonomy of what agents can do and the design constraints that follow
- [Agent Loop Anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy) — how tool calls fit into the full think-act-observe cycle
- [Advanced Tool Use](/blog/machine-learning/ai-agent/advance-tool-use) — patterns for multi-step, parallel, and conditional tool invocation

---

## When to Reach for Schema Quality — When Good-Enough is Fine

To close the loop: schema quality is the primary lever for tool invocation reliability. It is not a model problem, not a prompt engineering problem, and not a fine-tuning problem in the first instance. It is a documentation problem — the same class of problem as writing a good API reference or a clear function docstring.

The specific investments that pay the most:

1. **Verb-noun names** (30 min, one-time): immediate impact on selection accuracy.
2. **80-word descriptions with NOT-for hints** (2 hrs for 5 tools): eliminates 80% of collision errors.
3. **Enum conversion for categorical parameters** (30 min per parameter): eliminates hallucinated argument values.
4. **Required field calibration** (1 hr for full schema): eliminates hallucinated optional argument values.
5. **Eval dataset + accuracy tracking** (2–4 hrs initial, ongoing): makes schema quality measurable and improvable.

The investments that can wait until you need them:

- Routing architecture (wait until you have 15+ tools or see accuracy below 75%)
- Schema versioning policy (wait until you have a second deployment environment)
- Exhaustive triggering phrase inventory (wait until you have eval failures to guide it)

A schema that took you four hours to write carefully will outperform a schema that took you twenty minutes, for the entire lifetime of the agent. That is one of the better return-on-investment curves in software engineering.
