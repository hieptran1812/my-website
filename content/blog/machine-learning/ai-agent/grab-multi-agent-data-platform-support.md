---
title: "From firefighting to building: inside Grab's five-agent system for data-platform support"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A detailed engineering read of Grab's multi-agent support system for their 15,000-table analytics warehouse — the architecture, the six production problems that nearly killed it, and the numbers the post carefully does not give you."
excerpt: "Grab's analytics data warehouse team spent 40% of its week answering the same questions. They replaced that with five specialized agents behind a router and a human gate. This is a close reading of what they built, why each piece is shaped the way it is, and what you would still have to figure out yourself."
tags:
  [
    "ai-agent",
    "multi-agent",
    "langgraph",
    "orchestration",
    "context-engineering",
    "human-in-the-loop",
    "data-engineering",
    "guardrails",
    "production-ml",
    "case-study",
  ]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 56
---

Most engineering blog posts about agents describe a system that works. Grab's [*From firefighting to building*](https://engineering.grab.com/from-firefighting-to-building) is more useful than that, because roughly half of it describes a system that did not work yet — six specific ways the first version fell over in production, and what they changed. That second half is the part worth reading twice.

The setup is a familiar kind of pain. Grab's Analytics Data Warehouse team owns more than 15,000 tables, serves over 1,000 monthly users, and their tables back about half of all data-lake queries in the company. Being that load-bearing has a cost: the team estimated it spent around 40% of its time — roughly two days a week — on repetitive requests. What does this column mean. Where does this metric come from. Why does this number look wrong. Can you add a field to this table.

None of those questions are hard. All of them are expensive, because answering any one requires opening four different systems and holding the answer together in your head. Their fix was a multi-agent system in front of Slack: one router, five specialized agents, four safety gates, and a human review step that is deliberately not a blocker.

![Grab ADW architecture: one Slack request splits into an enhancement lane and an investigation lane, converging on human review](/imgs/blogs/grab-multi-agent-data-platform-support-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. One Slack message enters through a FastAPI endpoint into a LangGraph orchestrator. A first decision splits the traffic: requests that want to *change* something go to the Enhancement Agent, which writes code and opens a merge request. Requests that want to *know* something go to a Classifier, which decides which of three read-only specialists to run and in what order, and a Summarizer reconciles what they find. Both lanes end at a human — an engineer merging an MR, or a reviewer approving an answer. Neither lane lets the model touch production on its own.

I want to do three things here: explain why each structural choice is shaped the way it is, work through the six production problems in enough detail that you could implement the fixes, and then be honest about what the post does not tell you — because there are several numbers missing whose absence changes how you should read the results.

## Why this post is worth reading closely

The genre of "we built an agent system" post has a predictable shape: architecture diagram, list of agents, screenshot of a happy path, closing paragraph about the future of work. Grab's post has that, but it also has a section titled *Optimizing the Architecture* that opens with the line that building the system was one challenge and making it production-ready was another. What follows is a list of six failures. That is unusual, and it is where the transferable engineering lives.

Here is where common assumptions about this kind of system diverge from what Grab actually reports.

| Assumption | The naive view | What the post actually describes |
| --- | --- | --- |
| The hard part is making the agent smart | Better prompts, bigger model, more tools | The hard part is keeping context small enough that the agent stays coherent across four hops |
| More tools means more capability | Register every API you have; the model will pick | Over 30 tools became a *performance bottleneck* — their descriptions are read on every single inference |
| Human review is a safety feature | Add an approval step, ship it | Blocking on review created a queue that left questions unanswered for long stretches; they had to remove the block and keep the review |
| Guardrails live inside the agent | Tell the model not to query PII | Guardrails live in front of and beneath the agent: a Classifier that refuses, a SQL validator, a timeout, and no commit rights |
| Feedback improves the system automatically | Collect thumbs-up/down and iterate | Annotations sat as passive records until they wired them into an offline eval set; collecting feedback and *using* it are different projects |
| One capable model beats five narrow ones | Fewer moving parts, fewer bugs | They chose five, accepted the latency cost explicitly, and justified it by debuggability and blast radius |

Every row in that table is a decision someone had to defend in a design review. The rest of this article is about the arguments behind them.

## The 40% tax: what "firefighting" actually costs

Start with the scenario the post opens with, because it is precise. It is Friday, 5:00 PM. A Slack message says the `vehicle_id` values in a production table look like gibberish. Now you are the on-call data engineer.

You do not know yet whether this is a bug. To find out you have to establish four independent facts: what the column is supposed to contain, where the values come from, whether the transformation that produces them is behaving, and whether anything broke in the pipeline recently. Each fact lives in a different system. Grab names three of them: Hubble for metadata and catalog, Genchi for data-quality observability and contract enforcement, and Lighthouse for pipeline execution tracking. Add the warehouse itself, GitLab, Airflow, and the Slack channels where outages get announced.

![The same five investigation steps, done manually versus mediated by agents](/imgs/blogs/grab-multi-agent-data-platform-support-2.webp)

The figure above makes the important point, and it is easy to miss: **the agents did not eliminate a single step**. The metadata still gets looked up. The lineage still gets traced. A query still runs. The logs still get checked. What changed is who holds the thread between those four lookups.

That distinction matters because it tells you what kind of problem this is. It is not a knowledge problem — the answers were always retrievable. It is a **context-assembly problem**. The expensive part of the manual workflow is the human context-switch: opening a new tab, remembering which table you were tracing, translating a column name from the catalog into a `WHERE` clause, deciding whether the Airflow run you are looking at is the one that produced the rows you queried.

> Repetitive support work is rarely repetitive because the questions repeat. It is repetitive because the *procedure* repeats, over a different noun each time. That is exactly the shape of work an agent can absorb.

Grab makes this observation explicitly: while the problems differed, the solving process remained consistent. That sentence is the whole thesis. If your support burden has that property — same procedure, different noun — you have a candidate. If every request genuinely requires new judgment, you do not, and no amount of orchestration will change that.

### Why documentation does not fix this

The obvious objection is that 15,000 tables with good documentation would not generate these questions. It is worth taking seriously, and then rejecting it, because the reasoning generalizes.

Documentation answers *what a column means*. It does not answer *why this specific value looks wrong today*, which requires joining the definition against the current data, the current code, and the current pipeline state. Three of those four are live. A catalog entry written six months ago cannot tell you that the source system changed its JSON payload format last Tuesday.

More practically: documentation has a maintenance cost proportional to the number of objects, and a value proportional to the number of *reads*. At 15,000 tables the maintenance cost is enormous and the read distribution is brutally long-tailed. The agent approach inverts the economics — instead of pre-computing answers for every table, you pay per question actually asked, and the answer is computed against live state.

This is the same argument as the one for retrieval over fine-tuning, applied to operational knowledge rather than domain knowledge. If you have internalized that argument for [retrieval-augmented agents](/blog/machine-learning/ai-agent/retrieval-augmented-agents), this is its operations-team cousin.

## The split that makes everything else possible: brain and hands

Before the agents, there is one architectural choice that everything else depends on. Grab describes it as decoupling *the brain* (the LLM) from *the hands* (the specialized agents and tools), and credits it with improved debuggability.

![Layered stack: human surface, orchestrator, LLM brain, tool layer, systems of record](/imgs/blogs/grab-multi-agent-data-platform-support-3.webp)

The value of this split is not architectural elegance. It is that it makes failures **attributable**. When the model is allowed to call a database driver directly, a bad answer and a dangerous query are the same class of bug, and you fix both by editing a prompt and hoping. When the model can only emit an *intent* that a validator turns into an *effect*, the two failure modes separate cleanly:

- The answer is wrong but nothing dangerous happened → the reasoning layer is at fault. Fix the prompt, the retrieved context, or the agent's scope.
- Something dangerous almost happened → the tool layer is at fault. Fix the validator. The prompt is irrelevant, because you should never have been relying on it.

That second bullet is the one people get wrong. A prompt instruction like "never query PII columns" is a *preference*, not a *guarantee*. It fails under prompt injection, under distribution shift, under a sufficiently unusual phrasing of a question. A validator that reads the parsed SQL and rejects any statement touching a column tagged as PII is a guarantee, because it does not involve the model at all.

Here is the shape in code. This is deliberately boring, which is the point:

```python
# effects.py — the only module allowed to touch a system of record.
# The LLM never imports this; it emits a ToolCall that the runtime routes here.
from dataclasses import dataclass
from typing import Any, Literal

import sqlglot
from sqlglot import expressions as exp


@dataclass(frozen=True)
class ToolCall:
    """What the brain is allowed to produce: an intent, never an effect."""
    name: str
    args: dict[str, Any]
    agent: Literal["data", "code_search", "oncall", "enhancement"]


class Refused(Exception):
    """Raised before any I/O happens. Carries a reason the agent can read back."""


def execute(call: ToolCall, *, catalog) -> Any:
    handler = REGISTRY.get(call.name)
    if handler is None:
        raise Refused(f"unknown tool {call.name!r}")
    if call.name not in SCOPES[call.agent]:
        # The On-call Agent cannot run warehouse queries. The Data Agent cannot
        # open merge requests. Scope is enforced here, not in the prompt.
        raise Refused(f"{call.agent} is not scoped for {call.name!r}")
    return handler(**call.args, catalog=catalog)
```

Two properties are worth calling out. First, `Refused` is raised *before* any I/O — the refusal is not a rollback, it is a precondition. Second, scope is a property of the `(agent, tool)` pair, not of the tool alone. The On-call Agent and the Data Agent both need to read metadata; only one of them needs to run a query against the warehouse. That asymmetry is only expressible because the agents are separate identities, which is the first concrete payoff of the multi-agent design.

Anthropic reached a structurally similar conclusion for a very different system, which I wrote about in [scaling managed agents: decoupling the brain from the hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands). The convergence is not a coincidence — it is what happens when a team runs an agent long enough to need to debug it under time pressure.

## Why five specialists beat one super-agent

Grab addresses the monolith-versus-fleet question directly, and their treatment is more honest than most, because they list the costs of their own choice.

![Matrix comparing one super-agent against five specialists across five axes](/imgs/blogs/grab-multi-agent-data-platform-support-4.webp)

Read the bottom two rows of that figure carefully. **The specialist fleet loses on latency and it loses on operating cost.** Sequential agent execution adds latency; coordination adds complexity; five prompts need five eval sets. Grab does not hide this. They state that they prioritized maintainability and accuracy over latency reduction, and they justify it with a sentence that is worth quoting:

> When you're replacing a multi-hour manual investigation, taking a few minutes for a precise answer is a massive leap in operational throughput.

That justification is load-bearing, and it is *conditional*. It holds when your baseline is a human taking hours. It does not hold if your baseline is an existing system answering in 200 ms, and it does not hold for an interactive product surface where a user is waiting. The architecture is correct for Grab's problem because of the baseline they were measured against. Copy the architecture without copying the baseline and you will have bought latency you cannot afford.

### The real argument for specialists

The debuggability argument is the strong one, and it deserves a sharper statement than "modular is better."

Consider what a bad answer looks like in each design. In a monolith, a wrong answer is produced by a single inference call that read a system prompt, 30-plus tool descriptions, some retrieved metadata, some query output, some source code, and some log lines. When it hallucinates a transformation that does not exist, your debugging surface is *the entire prompt*. You can add an instruction, but you cannot localize the fault, and any instruction you add competes for attention with every other instruction.

In the fleet, the same wrong answer arrives attached to a trace: the Classifier chose these three agents in this order, the Code Search Agent returned this transformation summary, the Summarizer wrote this sentence. The hallucination is in the Code Search Agent's output. You now have a bug with an owner, a reproducible input, and a prompt small enough to reason about. You can add that input to that agent's eval set without touching the other four.

That is the difference between a system you can improve and a system you can only rewrite. It compounds: after six months the monolith's prompt is an archaeological record of every incident, and nobody dares delete a line.

### When the monolith is still right

To be fair to the other side, the fleet is genuinely worse when:

- **Your domain has one shape.** If every question is "explain this SQL," splitting it across agents adds hops and buys nothing.
- **You cannot afford five eval sets.** The maintenance cost is real and recurring. A two-person team that ships one prompt well will beat the same team maintaining five badly.
- **Latency is in the user's critical path.** Three sequential inference calls is three times the p99 tail, and tails compound.
- **The sub-tasks are not separable.** If agent B needs to see agent A's raw evidence rather than its conclusion, you have not decomposed the problem, you have just added serialization overhead.

The decision is really about whether your problem has natural seams. Grab's does: data investigation, code analysis, and production health are genuinely different domains with different tools and different failure modes. That is why five works there. [Multi-agent topologies](/blog/machine-learning/ai-agent/multi-agent-topologies) covers the shapes this can take in more detail.

## Pathway A: the Enhancement Agent, semi-automated on purpose

The first lane handles requests that change something — add a column, modify aggregation logic. This is the lane where an agent can actually break production, and its design reflects that.

![The enhancement pipeline: JIRA ticket through validation, code generation, merge request, human review, merge](/imgs/blogs/grab-multi-agent-data-platform-support-5.webp)

The word Grab uses is *semi-automated*, and they are explicit that this is by design rather than by limitation: code changes to production pipelines require human judgment. The agent accelerates research, coding, and testing; humans approve final changes.

Walk the scenario they give. A user wants a `customer_segment` column added to a `rides` table, sourced from `user_profiles`. Traditionally, an engineer spends a meaningful chunk of an afternoon clarifying requirements, writing the change, and testing it. With the agent, the sequence is:

1. **Read the ticket.** The agent pulls the JIRA request and extracts the actual requirement — source table, target table, column name.
2. **Locate the pipeline.** It searches the codebase for the files that actually build the target table. This is the step that eats human time, because it requires knowing the repository's conventions.
3. **Run validations.** Three of them, and the order matters: does the requested column exist in the upstream source; does it already exist in the target; is the schema compatible and does it meet data-quality requirements.
4. **Generate the change.** Both the transformation code and the DDL, following the repository's existing DDL script standards.
5. **Open a merge request.** Never a direct commit.
6. **Enable a test run.** The user can trigger the pipeline on Airflow, optionally with a date range.

Step 3 is the interesting one, and it is where most naive versions of this fail. An agent that writes plausible code without validating its assumptions produces merge requests that look right and fail at runtime — which is *worse* than no automation, because a reviewer now has to reverse-engineer the agent's assumptions instead of writing the change themselves.

Here is the validation pattern in the shape it needs to take:

```python
# enhancement/validate.py — every precondition is checked against live metadata,
# never inferred from the model's memory of the schema.
from dataclasses import dataclass


@dataclass
class AddColumnRequest:
    source_table: str          # "analytics.user_profiles"
    source_column: str         # "customer_segment"
    target_table: str          # "analytics.rides"
    target_column: str


@dataclass
class Finding:
    level: str                 # "block" | "warn"
    message: str


def validate_add_column(req: AddColumnRequest, catalog) -> list[Finding]:
    findings: list[Finding] = []

    src = catalog.columns(req.source_table)
    if req.source_column not in src:
        findings.append(Finding(
            "block",
            f"{req.source_column} does not exist in {req.source_table}; "
            f"nearest matches: {catalog.fuzzy(req.source_table, req.source_column)[:3]}",
        ))
        return findings                      # cheap check first; stop on failure

    tgt = catalog.columns(req.target_table)
    if req.target_column in tgt:
        findings.append(Finding(
            "block",
            f"{req.target_column} already exists in {req.target_table} "
            f"(type {tgt[req.target_column].dtype}); this ticket may be a duplicate",
        ))

    col = src[req.source_column]
    if col.is_pii:
        findings.append(Finding(
            "block",
            f"{req.source_column} is classified PII; propagating it to "
            f"{req.target_table} needs a governance review, not a merge request",
        ))
    if col.null_rate > 0.20:
        findings.append(Finding(
            "warn",
            f"{req.source_column} is {col.null_rate:.0%} null upstream; "
            f"downstream aggregations will silently change",
        ))
    if not catalog.type_compatible(col.dtype, catalog.table(req.target_table).engine):
        findings.append(Finding(
            "warn", f"{col.dtype} needs an explicit cast for the target engine")
        )
    return findings
```

Note the PII check sitting inside the *enhancement* validator, not only inside the query validator. Grab lists flagging governance concerns — PII classification, SLAs, backward compatibility — as an explicit Enhancement Agent responsibility. This is the right place for it: propagating a PII column into a widely-read table is a policy decision, and the agent's job is to surface it loudly enough that a human makes it.

### The second-order effect: reviewers read differently

There is a consequence of this design that the post does not spell out but which anyone who has run a bot-authored MR queue will recognize. When a machine opens merge requests, human review changes character. A reviewer reading a colleague's diff is checking judgment; a reviewer reading a generated diff is checking *whether the generator understood the request*.

That is a different and in some ways easier task, but only if the MR carries the agent's reasoning with it. An MR body that lists the validations that ran, the files the agent searched, and the assumptions it made turns review into verification. An MR body that just contains a diff turns review into re-derivation, and re-derivation is slower than writing the change yourself. If you build this, the MR description is not documentation — it is the interface.

## Pathway B: the Classifier and three read-only specialists

The second lane handles questions. Four agents participate, and their division of labour is the clearest part of Grab's design.

| Agent | What it reads | What it does | What it returns | Its guardrail |
| --- | --- | --- | --- | --- |
| **Classifier** | The raw question | Extracts tables, scripts, and the actual ask; decides which specialists to run and in what order | A plan with reasoning and per-agent task descriptions | Detects PII requests and out-of-scope queries *before* anything runs |
| **Data Agent** | Table and column metadata | Builds and runs queries; validates schemas; retrieves samples with exploratory comments | Sample rows and schema facts | Every query passes a PII, DDL/DML, partition-filter and schema check |
| **Code Search Agent** | The transformation codebase | Traces a column back through multiple transformation steps; explains the logic in plain language | A transformation narrative plus any divergence from documentation | Read-only; snippets rather than whole files |
| **On-call Agent** | Slack, Airflow, DQ metrics | Searches for outage and delay announcements; checks pipeline health, logs, retries; validates null counts, duplicates, ranges | Incident notes and an initial RCA | Read-only across observability surfaces |
| **Summarizer** | Only the other agents' responses | Reconciles conflicting findings into one coherent, structured narrative | The answer a human reviews | Never sees raw tool output |

Two things in that table deserve elaboration.

**The Classifier is a security boundary, not just a router.** Grab lists guardrail violation detection — PII requests, out-of-scope queries — as a Classifier responsibility, and it is the *first* thing that runs. This placement is deliberate and correct. If refusal lives inside each specialist, you have four places to get it right and four places to regress. If it lives in front, you have one, and it runs before any tool is in scope. The cost is that a Classifier failure is a single point of failure, which is exactly why its precision matters more than any other component's — a point I will come back to.

**The Summarizer's restricted input is a feature.** It reads structured agent responses, not raw tool output. That sounds like a limitation, and it is the thing that makes the final answer trustworthy. A summarizer with access to raw query results and raw source files will confidently blend them, and when it is wrong you cannot tell which input misled it. A summarizer that can only read three conclusions can only be wrong in ways attributable to one of those three conclusions.

### Nobody talks to anybody

The structural rule underneath the whole investigation lane is that agents do not communicate with each other. Every handoff goes back through the orchestrator, which Grab describes as the Handoffs Pattern: the previous agent returns its response to a central orchestrator, which cleans context, prunes tokens, and invokes the next agent.

![Four hops, each returning through the orchestrator with a pruned response](/imgs/blogs/grab-multi-agent-data-platform-support-6.webp)

This is worth dwelling on because it is the difference between a system that survives a long conversation and one that does not. In a peer-to-peer design where agent A passes its output straight to agent B, the context grows monotonically and nobody owns trimming it. In a hub design, there is exactly one place where context is inspected, summarized, and re-injected — and it runs once per hop, deterministically.

The orchestrator's state, per the post, tracks three things: conversation and tooling history, execution tracking (which agents ran, current progress, execution steps), and structured agent responses passed to subsequent agents. That third field is the important one. It is the difference between "here is everything that happened" and "here is what agent 1 concluded."

```python
# orchestrator/state.py — the LangGraph state object.
# Each field has a different retention policy, which is the whole point.
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class InvestigationState(TypedDict):
    question: str                                   # never pruned
    plan: list[str]                                 # Classifier output; never pruned
    messages: Annotated[list, add_messages]         # summarized when over budget
    steps: list[dict]                               # {agent, tool, ms, ok} — audit trail
    responses: dict[str, dict]                      # agent_name -> structured conclusion
    tokens_used: int


def handoff(state: InvestigationState, next_agent: str) -> InvestigationState:
    """The only path between two agents. Prune, then invoke."""
    state = prune_to_budget(state, budget=MODEL_BUDGET[next_agent])
    payload = {
        "question": state["question"],
        "task": task_for(state["plan"], next_agent),
        # Conclusions, not transcripts. The next agent never sees raw tool output.
        "prior": {k: v["conclusion"] for k, v in state["responses"].items()},
    }
    result = AGENTS[next_agent].invoke(payload)
    state["responses"][next_agent] = result
    state["steps"].append({"agent": next_agent, "ok": result["ok"]})
    return state
```

The line that carries the most weight is `"prior": {k: v["conclusion"] ...}`. It says the On-call Agent does not get to read the Data Agent's sample rows. It gets the sentence "the values are valid UUIDs joinable with `dim_vehicles`." That is a lossy handoff *on purpose*, and the loss is what keeps the fourth hop as coherent as the first.

For the general form of this problem, [effective context engineering for AI agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) and [shared state and coordination](/blog/machine-learning/ai-agent/shared-state-and-coordination) go deeper on the alternatives.

## One full trace: "why is the vehicle ID unreadable?"

Grab walks their second scenario end to end, and it is a good scenario precisely because the answer is boring: nothing is broken. Systems that can only find bugs are systems that will invent bugs.

![Timeline of the seven-step vehicle ID investigation](/imgs/blogs/grab-multi-agent-data-platform-support-7.webp)

**Step 1 — the Classifier plans.** It determines all three specialists are needed and fixes the order: Data Agent, then Code Search, then On-call. It records its reasoning: verify the data format, trace the transformation logic, check for production incidents.

Stop here for a moment, because this step is doing more than it appears to. The Classifier has decomposed a vague complaint ("unreadable") into three falsifiable hypotheses: *the values are malformed*, *the transformation corrupted them*, and *an incident broke the pipeline*. The rest of the run is hypothesis elimination. A senior engineer would have done exactly this, silently, in about four seconds. Making it an explicit artifact is what allows the rest of the system to be auditable.

**Step 2 — the Data Agent investigates.** It pulls metadata to construct a valid query, queries the actual data, and confirms the values are UUIDs rather than corruption. It then searches the catalog for dimension tables and builds a join to test readability. Conclusion: the IDs are valid UUIDs, joinable with `dim_vehicles` to get human-readable names.

The second half of that step is the part worth noticing. The agent did not stop at "these are valid UUIDs" — it went looking for what the *user actually wanted*, which was readable names, and found the join that provides them. That is the difference between answering the question asked and answering the question meant.

**Step 3 — the Code Search Agent traces lineage.** It scans transformation and lineage logic and finds that the ID is a raw UUID lifted from a JSON payload from the source system. It then queries the source table for samples to confirm the pattern matches. Conclusion: the format comes from upstream; no Spark transformation introduced it.

Note the confirmation step. The agent formed a hypothesis from code and then checked it against data. That cross-check — code says X, does data agree — is what separates a lineage summary from an actual finding.

**Step 4 — the On-call Agent checks production health.** Airflow status, Slack incident channels, data-quality metrics. Conclusion: no incidents, pipeline running successfully, metrics within normal ranges.

**Step 5 — the Summarizer reconciles.** It has one user concern and three agent findings, and it produces a structured answer: the IDs are not corrupt, the format originates upstream, nothing is broken, and here is how to get readable names.

**Step 6 — it posts immediately, tagged unreviewed.** The answer goes to Slack right away with an explicit "unreviewed" label, awaiting engineer review. This is the speed-versus-quality resolution I will come back to in detail.

**Step 7 — the thread stays open.** Anyone can reply and re-enter the loop with the agents.

### What this trace tells you about the design

Three properties of this run are worth extracting, because they are the actual reusable lessons:

**The conclusion is negative, and it is supported.** "Nothing is broken" is the hardest answer for an agent to produce credibly, because it is the default output of a system that failed to look. Here it is supported by three independent eliminations from three different evidence sources. A single agent asserting "no issue found" is worth nothing; three agents each eliminating a different hypothesis is an argument.

**The plan is committed before the evidence arrives.** The Classifier decides the order up front. That means the plan cannot be retro-fitted to whatever the first agent happened to find — a failure mode where a system rationalizes toward the first plausible explanation it stumbled into.

**Every conclusion is separately checkable.** A reviewer who doubts the answer does not have to re-run the whole investigation. They can look at the Code Search Agent's claim about the source payload and check that one thing.

## Production problem 1: context explosion

Now the useful half of the post. Grab's first production problem is the one every multi-agent system hits: context accumulates rapidly, and without careful management the excess degrades performance.

The mechanism is worth being precise about, because "context grows" undersells it. With four sequential hops, hop 4's prompt naively contains: the original question, the Classifier's plan, three agents' full tool call transcripts, three agents' full tool *outputs* — which include query result sets and source files — plus its own tool schemas. The growth is not linear in a comfortable way; the tool outputs dominate and they arrive in large chunks.

<figure class="blog-anim">
<svg viewBox="0 0 680 300" role="img" aria-label="Two context budgets across four agent hops. Without pruning the prompt grows monotonically and crosses the model context limit; with orchestrator handoff pruning it climbs and resets after every hop, never reaching the limit." style="width:100%;height:auto;max-width:840px">
<style>
.g1-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.g1-bar{transform-box:fill-box;transform-origin:left center;fill:var(--accent,#6366f1)}
.g1-hot{fill:#e8654f}
.g1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.g1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.g1-tick{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.g1-cap{font:600 12px ui-sans-serif,system-ui;fill:#e8654f;text-anchor:middle}
.g1-limit{stroke:#e8654f;stroke-width:2;stroke-dasharray:6 5}
.g1-hop{stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes g1-grow{0%{transform:scaleX(.10)}22%{transform:scaleX(.40)}47%{transform:scaleX(.68)}72%{transform:scaleX(.94)}95%,100%{transform:scaleX(1.14)}}
@keyframes g1-saw{0%{transform:scaleX(.10)}20%{transform:scaleX(.40)}24%{transform:scaleX(.16)}44%{transform:scaleX(.44)}48%{transform:scaleX(.18)}68%{transform:scaleX(.46)}72%{transform:scaleX(.20)}92%{transform:scaleX(.50)}96%,100%{transform:scaleX(.14)}}
.g1-a{animation:g1-grow 12s ease-in-out infinite}
.g1-b{animation:g1-saw 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.g1-a{animation:none;transform:scaleX(1.14)}.g1-b{animation:none;transform:scaleX(.46)}}
</style>
<text class="g1-cap" x="560" y="34">model context limit</text>
<line class="g1-limit" x1="560" y1="44" x2="560" y2="212"/>
<text class="g1-lbl" x="16" y="80">accumulate</text>
<text class="g1-sub" x="16" y="98">every hop appended</text>
<rect class="g1-track" x="180" y="58" width="380" height="44" rx="6"/>
<rect class="g1-bar g1-hot g1-a" x="180" y="58" width="380" height="44" rx="6"/>
<text class="g1-lbl" x="16" y="172">prune on handoff</text>
<text class="g1-sub" x="16" y="190">orchestrator cleans</text>
<rect class="g1-track" x="180" y="150" width="380" height="44" rx="6"/>
<rect class="g1-bar g1-b" x="180" y="150" width="380" height="44" rx="6"/>
<line class="g1-hop" x1="275" y1="216" x2="275" y2="230"/>
<line class="g1-hop" x1="370" y1="216" x2="370" y2="230"/>
<line class="g1-hop" x1="465" y1="216" x2="465" y2="230"/>
<line class="g1-hop" x1="560" y1="216" x2="560" y2="230"/>
<text class="g1-tick" x="227" y="252">Data</text>
<text class="g1-tick" x="322" y="252">Code Search</text>
<text class="g1-tick" x="417" y="252">On-call</text>
<text class="g1-tick" x="512" y="252">Summarizer</text>
<text class="g1-sub" x="180" y="280">bar length = tokens carried into the next inference call</text>
</svg>
<figcaption>The same four-agent investigation, run twice. Appending each agent's raw output pushes the prompt past the context limit by the fourth hop; pruning at every handoff keeps the working set flat no matter how long the conversation runs.</figcaption>
</figure>

Grab's answer has four parts, and they are complementary rather than alternative.

**Token tracking with `tiktoken`.** They count tokens in real time for budget visibility. This sounds trivial and is not — you cannot manage what you do not measure, and "the context feels big" is not a signal you can build a policy on. The budget needs to be a number the orchestrator can branch on.

**Intelligent summarization.** When the token limit is exceeded, earlier messages are automatically summarized while retaining question-relevant information; recent messages and critical context remain unsummarized. The two carve-outs matter as much as the summarization: recency is preserved because the most recent exchange is usually the one being reasoned about, and "critical context" — the original question, the plan — is preserved because summarizing the question is how a long conversation drifts off topic.

**RAG context pruning.** Two named tactics: full code files are replaced by a smaller LLM extracting the relevant snippets, and database queries are filtered to return only the top relevant results. The first is the interesting one. Using a cheap model as a *compressor* in front of an expensive model is one of the highest-leverage patterns in this whole space, because the compression is semantic rather than positional — it keeps the function that matters and drops the 400 lines of imports and helper code around it.

**The Handoffs Pattern.** Pruning is a step in the graph, not a behaviour of an agent.

Here is the budget logic in the shape it needs to take. The specific numbers are illustrative; the structure is not:

```python
# orchestrator/prune.py
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def prune_to_budget(state: dict, budget: int, keep_recent: int = 4) -> dict:
    """Summarize the oldest messages until the working set fits the budget.

    Three tiers, in priority order:
      pinned   - question + plan; never touched
      recent   - last `keep_recent` messages; never summarized
      historic - everything else; collapsed oldest-first
    """
    pinned = ntok(state["question"]) + ntok("\n".join(state["plan"]))
    msgs = state["messages"]
    recent, historic = msgs[-keep_recent:], msgs[:-keep_recent]

    def total() -> int:
        return pinned + sum(ntok(m.content) for m in historic + recent)

    while total() > budget and historic:
        # Collapse the oldest window, not the whole history: repeated summarizing
        # of an already-summarized block is how detail silently evaporates.
        window, historic = historic[:6], historic[6:]
        digest = summarize_window(
            window,
            focus=state["question"],       # keeps the summary question-relevant
            max_tokens=budget // 12,
        )
        historic.insert(0, digest)

    if total() > budget:
        # Still over after collapsing everything: the tool outputs are the problem,
        # not the history. Fail loudly rather than truncating mid-evidence.
        raise BudgetExceeded(f"{total()} tokens against a {budget} budget")

    state["messages"] = historic + recent
    state["tokens_used"] = total()
    return state
```

The `raise` at the bottom is the part most implementations skip, and skipping it is how these systems produce confidently wrong answers. Silent truncation removes evidence without removing the model's willingness to conclude. If you cannot fit the evidence, the correct behaviour is to say so — either by failing the hop or by escalating to a human — not by answering from whatever survived the cut.

### The second-order problem: summarization is lossy in a direction

There is a failure mode here that Grab's post does not discuss and that you should plan for. Summarizers are systematically biased toward *narrative coherence*. When you compress "the query returned 4 rows, two of which had null `segment`" you very often get back "the query returned some rows." The null detail — the anomaly — is exactly what an investigation needs and exactly what a summarizer treats as noise.

The mitigation is to summarize with the question in focus, which Grab does. A stronger mitigation is to keep anomalies out of the summarizable tier entirely: when a tool detects something unusual — a null rate above threshold, a schema mismatch, a failed run — that finding gets pinned alongside the question rather than appended to the message history. Structured findings survive compression; prose findings do not.

## Production problem 2: thirty tools is a tax on every inference

The second problem is the one I expect most readers to find genuinely new. Grab reports that their initial design, with over 30 distinct tools, presented a significant performance bottleneck — because tool calling is part of agent prompts, forcing agents to process verbose tool descriptions and outputs.

![The inference payload before and after trimming, with the three trim levers](/imgs/blogs/grab-multi-agent-data-platform-support-8.webp)

The mechanism is easy to overlook because tool schemas do not feel like context. They are configuration; you register them once. But they are serialized into every single request. Thirty tools, each with a description, an argument list, per-argument descriptions, and examples, is a fixed cost paid on every hop of every investigation, before the model has read a word of the actual question.

And the cost is not only tokens. It is *attention*. A model choosing among 30 similarly-described tools is doing a harder discrimination task than one choosing among six, and the errors it makes are the expensive kind: calling a plausible-but-wrong tool, then reasoning over its output as though it answered the question.

Grab's three levers:

1. **Include only the portions required for the decision.** The tool description exists to help the model decide whether to call the tool — not to document the tool.
2. **Aggressively truncate verbose tool output.**
3. **Streamline descriptions for conciseness and actionability.**

Here is the before and after, which is where this becomes concrete:

```python
# BEFORE — reads like API documentation. Every word is paid for on every call.
{
    "name": "query_warehouse",
    "description": (
        "Executes a SQL query against the Analytics Data Warehouse. The warehouse "
        "contains over 15,000 tables organized into schemas by domain. Queries are "
        "executed via the Presto engine. Please note that queries should include "
        "partition filters where possible for performance reasons. The warehouse "
        "supports standard ANSI SQL with some extensions. Results are returned as "
        "a list of dictionaries. Large result sets may be truncated. See the "
        "internal wiki for the full list of supported functions and known "
        "limitations. Contact the ADW team if you encounter issues."
    ),
    "parameters": {
        "sql": {"type": "string", "description": "The SQL query string to execute against the warehouse."},
        "timeout_seconds": {"type": "integer", "description": "How long to wait before cancelling. Defaults to 60."},
        "max_rows": {"type": "integer", "description": "Maximum rows to return. Defaults to 1000."},
        "catalog": {"type": "string", "description": "Which catalog to query. Defaults to the main one."},
        "trace_id": {"type": "string", "description": "Optional trace identifier for observability."},
    },
}

# AFTER — one purpose line; only the arguments the model actually chooses.
# timeout, max_rows, catalog and trace_id are set by the runtime, not the model.
{
    "name": "query_warehouse",
    "description": "Run a read-only SELECT. Requires a partition filter.",
    "parameters": {
        "sql": {"type": "string", "description": "SELECT only; must filter on the partition column."},
    },
}
```

The rule that generalizes: **an argument the model should never choose does not belong in the schema.** Timeouts, row caps, catalog names, and trace IDs are runtime policy. Exposing them invites the model to set them, which is both a token cost and a safety hole — a model that can set `max_rows` can set it to a million.

Output truncation follows the same logic:

```python
def truncate_result(rows: list[dict], *, max_rows: int = 20, max_chars: int = 4000) -> dict:
    """What goes into context is a *description* of the result, not the result."""
    head = rows[:max_rows]
    blob = json.dumps(head, default=str)[:max_chars]
    return {
        "row_count": len(rows),               # the model needs the count...
        "sample": blob,                       # ...but not every row
        "truncated": len(rows) > max_rows,
        "columns": list(rows[0].keys()) if rows else [],
    }
```

Returning `row_count` while truncating `sample` is what lets the agent reason correctly about scale ("4.2 million rows matched") without carrying 4.2 million rows. Truncating without reporting the count is how an agent concludes "only 20 rows matched."

[Tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles) covers the design space here in more depth. The short version: treat your tool surface as a prompt, because it is one.

## Production problem 3: letting an LLM near production data

The third problem is the one that gets systems like this cancelled. Grab is direct about the risk: agents with database access and code generation can access sensitive PII, execute dangerous SQL, run expensive queries, and generate breaking code changes.

![Four safety gates narrowing what survives to reach production](/imgs/blogs/grab-multi-agent-data-platform-support-9.webp)

Their answer is four layers, and the design property that makes it work is that **each layer catches a class the others structurally cannot see**.

**Layer 1 — input classification.** The Classifier detects PII requests and out-of-scope queries before any agent executes. This catches intent. It is the only layer that can, because once a request has been decomposed into tool calls, the intent is gone — "show me the user table" and "debug this join" can produce identical SQL.

**Layer 2 — SQL validation before execution.** Every query is checked for PII column access against column metadata, DDL/DML operations, slow queries (missing partition filters, excessive date ranges), and schema validity. Grab notes something important here: the agent *lacks* `DELETE`, `DROP`, `TRUNCATE` and `UPDATE` access at the database level, and the check is an additional safeguard. That ordering is correct — the permission is the control, the validator is defence in depth. A validator alone would be a single parser bug away from disaster.

**Layer 3 — timeout protection.** Strict execution limits on all database queries. This catches what neither of the above can: a query that is semantically fine, correctly scoped, and accidentally scans a petabyte.

**Layer 4 — enhancement controls.** No direct commits to master; mandatory human review; test environment before production.

Here is layer 2 in the shape it needs to take. Parse, do not regex — a regex-based SQL guard is a false sense of security, because SQL has too many ways to spell the same thing:

```python
# guards/sql.py
import sqlglot
from sqlglot import expressions as exp

FORBIDDEN = (exp.Delete, exp.Drop, exp.Update, exp.Insert, exp.Alter, exp.Create)


class QueryRefused(Exception):
    pass


def check_query(sql: str, *, catalog, max_scan_days: int = 31) -> str:
    try:
        tree = sqlglot.parse_one(sql, read="presto")
    except Exception as e:
        raise QueryRefused(f"unparseable SQL: {e}")

    # 1. Statement type. Belt and braces on top of the DB grant.
    if isinstance(tree, FORBIDDEN):
        raise QueryRefused(f"{type(tree).__name__.lower()} is not permitted")

    # 2. Every referenced table must exist, and every column must resolve.
    #    This kills hallucinated schemas before they become a confusing error.
    for tbl in tree.find_all(exp.Table):
        name = f"{tbl.db or catalog.default_db}.{tbl.name}"
        if not catalog.exists(name):
            raise QueryRefused(f"unknown table {name}")

    # 3. PII columns, resolved through the catalog rather than by name matching.
    for col in tree.find_all(exp.Column):
        meta = catalog.resolve(col, tree)
        if meta and meta.is_pii:
            raise QueryRefused(f"{meta.qualified_name} is classified PII")

    # 4. Cost. A missing partition filter on a large table is a full scan.
    for tbl in tree.find_all(exp.Table):
        name = f"{tbl.db or catalog.default_db}.{tbl.name}"
        info = catalog.table(name)
        if info.is_partitioned and not _filters_on(tree, info.partition_column):
            raise QueryRefused(
                f"{name} is partitioned on {info.partition_column}; "
                f"add a filter on it"
            )
        if (span := _date_span_days(tree, info.partition_column)) > max_scan_days:
            raise QueryRefused(f"{span}-day scan on {name} exceeds the {max_scan_days}-day cap")

    return sql
```

Two design notes. **PII is resolved through the catalog, not by name matching.** A guard that blocks columns named `email` misses `contact_1` and blocks `email_campaign_id`. The classification has to be metadata that a data-governance process owns, and the guard's job is to read it. **The refusal messages are written for the agent to read.** `"add a filter on event_date"` is actionable; `"query rejected"` produces a retry loop of increasingly desperate rewrites.

### The gap all four layers share

Here is a critique the post does not make of itself. All four gates protect against the agent *doing* something harmful. None of them protect against the agent *saying* something wrong.

An answer that confidently misattributes a metric's source is not caught by input classification, SQL validation, timeouts, or merge-request review. It passes every gate, because every gate is about actions. The only control for a wrong answer is the human review layer — which is why the next section is not a nice-to-have, and why the Classifier's routing accuracy, which nothing validates, is the quiet risk in this architecture.

If your threat model includes users pasting untrusted content into the thread — a table description, a log line, an error message from an external system — then the input path deserves its own attention. [Prompt injection in agents](/blog/machine-learning/ai-agent/prompt-injection-in-agents) covers what that looks like when the injected text arrives through a tool result rather than a user message.

## Production problems 4 and 5: trust first, then speed

These two are the same problem seen from opposite ends, and the way Grab resolved the tension is the most product-minded decision in the whole post.

### Problem 4: the reviewer needs more than a yes/no

Grab's framing is that even with retrieval and guardrails, agents are not perfect, and hallucinations erode trust. Their answer is to route every summarized response to a human reviewer with five actions.

![Tree of the five reviewer actions and where each one lands](/imgs/blogs/grab-multi-agent-data-platform-support-10.webp)

The five: **Approve** posts the answer with a footnote indicating human review. **Reject** marks it incorrect, logs it for improvement, and does not post. **Refine** adds a prompt to improve the summary and regenerates with the extra guidance. **Re-route to sub-agents** sends the question to a specific agent with additional context. **Annotate** provides structured feedback saved to a database for continuous improvement.

The design insight is in the ratio: **only one of five actions publishes anything.** Four produce training signal. That is not an accident of the UI — it is the mechanism by which review becomes data collection rather than a cost centre.

Consider what a two-button review UI (approve / reject) would have produced. A reviewer who sees an almost-right answer has two bad options: approve something slightly wrong, or reject and write it themselves. Both destroy information. Approving teaches the system that a wrong answer was fine; rejecting throws away the 80% that was correct along with the reason it failed.

Refine and re-route convert that dead end into a cheap correction *plus* a record of what was missing. That is the difference between a review step that decays into rubber-stamping and one that stays engaged. Anyone who has watched an alerting channel go from carefully triaged to universally muted knows which failure mode is the default.

The general design pressure here — make the informative action cheaper than the uninformative one — is the core of [human-in-the-loop design](/blog/machine-learning/ai-agent/human-in-the-loop-design).

### Problem 5: the reviewer is also a bottleneck

Then the second half. Grab's initial design withheld AI-generated responses until authorized by the engineering team, and this introduced a bottleneck that could leave inquiries unresolved for extended periods.

Read that carefully, because it describes a system that had become *worse than the thing it replaced* on the dimension users care most about. The whole promise was faster answers. An answer that exists in a queue at 5:00 PM on Friday and gets approved on Monday morning is, from the user's side, indistinguishable from no answer at all — with the added insult that the machine already knew.

<figure class="blog-anim">
<svg viewBox="0 0 700 300" role="img" aria-label="Two review policies on the same question. Under block-until-approved the answer stalls in the review queue and only reaches the user after an engineer is free. Under post-then-review it reaches the user immediately tagged unreviewed, and an engineer reviews afterwards, flipping the tag to reviewed." style="width:100%;height:auto;max-width:840px">
<style>
.g2-lane{stroke:var(--border,#d1d5db);stroke-width:2}
.g2-gate{fill:var(--surface,#f3f4f6);stroke:#e8654f;stroke-width:2;stroke-dasharray:6 4}
.g2-end{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.g2-dot{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:center}
.g2-slow{fill:#e8654f}
.g2-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.g2-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.g2-mid{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.g2-tag{font:700 12px ui-sans-serif,system-ui;text-anchor:middle}
.g2-warn{fill:#e8654f}
.g2-ok{fill:var(--accent,#6366f1)}
.g2-late{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 4;fill:none}
@keyframes g2-block{0%{transform:translateX(0)}10%,68%{transform:translateX(105px)}84%,100%{transform:translateX(415px)}}
@keyframes g2-fast{0%{transform:translateX(0)}10%{transform:translateX(105px)}22%,100%{transform:translateX(415px)}}
@keyframes g2-tagA{0%,18%{opacity:0}24%,64%{opacity:1}70%,100%{opacity:0}}
@keyframes g2-tagB{0%,64%{opacity:0}70%,100%{opacity:1}}
.g2-p1{animation:g2-block 14s ease-in-out infinite}
.g2-p2{animation:g2-fast 14s ease-in-out infinite}
.g2-t1{animation:g2-tagA 14s linear infinite}
.g2-t2{animation:g2-tagB 14s linear infinite}
@media (prefers-reduced-motion:reduce){.g2-p1{animation:none;transform:translateX(105px)}.g2-p2{animation:none;transform:translateX(415px)}.g2-t1{animation:none;opacity:1}.g2-t2{animation:none;opacity:0}}
</style>
<text class="g2-lbl" x="16" y="70">block first</text>
<text class="g2-sub" x="16" y="88">nothing ships unreviewed</text>
<line class="g2-lane" x1="190" y1="96" x2="620" y2="96"/>
<rect class="g2-gate" x="310" y="76" width="150" height="40" rx="8"/>
<text class="g2-mid" x="385" y="101">review queue</text>
<rect class="g2-end" x="560" y="72" width="90" height="48" rx="8"/>
<text class="g2-mid" x="605" y="101">user</text>
<circle class="g2-dot g2-slow g2-p1" cx="190" cy="96" r="10"/>
<text class="g2-sub" x="190" y="140">the answer exists at second 30 — the user gets it hours later</text>
<text class="g2-lbl" x="16" y="196">post, then review</text>
<text class="g2-sub" x="16" y="214">tagged, not withheld</text>
<line class="g2-lane" x1="190" y1="222" x2="620" y2="222"/>
<rect class="g2-end" x="560" y="198" width="90" height="48" rx="8"/>
<text class="g2-mid" x="605" y="227">user</text>
<circle class="g2-dot g2-p2" cx="190" cy="222" r="10"/>
<text class="g2-tag g2-warn g2-t1" x="605" y="266">unreviewed</text>
<text class="g2-tag g2-ok g2-t2" x="605" y="266">reviewed</text>
<rect class="g2-end" x="230" y="252" width="170" height="34" rx="8"/>
<text class="g2-mid" x="315" y="274">engineer reviews later</text>
<path class="g2-late" d="M400 269 L520 269 L520 240 L554 240"/>
<polygon class="g2-slow" points="554,234 566,240 554,246"/>
</svg>
<figcaption>Both policies review every answer. Blocking puts the on-call engineer's calendar on the user's critical path; tagging moves the same review behind the answer, so time-to-answer stops depending on who is awake.</figcaption>
</figure>

The fix is one sentence and a lot of nerve: post responses without immediate human review, clearly marked as unreviewed, with all posts remaining reviewable and modifiable by on-call engineers.

What makes this work is that it separates two clocks that the blocking design had fused. **Time-to-answer** now depends only on the agents. **Time-to-verification** depends on the on-call rotation. Both still happen; neither gates the other. The `unreviewed` label is what makes the separation honest — the user knows exactly what they are holding, and can decide whether to act on it now or wait.

This is a pattern worth naming, because it recurs: when a quality gate becomes a queue, the move is usually to convert the gate into a **label** and let consumers decide. Canary releases do this. Search results marked "preliminary" do this. The gate stops being a valve and becomes metadata.

The precondition is that the failure is *recoverable* and *visible*. A wrong Slack answer that someone corrects an hour later costs an hour of a person's time. If your agent's mistakes are not recoverable — if the consumer of the answer acts irreversibly on it — you do not get to make this trade, and the queue stays. Notice that Grab made exactly this distinction: unreviewed *answers* ship, unreviewed *code* does not. The read path got a label; the write path kept the gate.

| | Blocking review | Post-then-review |
| --- | --- | --- |
| Time-to-answer | On-call availability | Agent latency (minutes) |
| Time-to-verification | Same as above | Independent, asynchronous |
| Failure cost | Zero wrong answers published | A wrong answer visible until corrected |
| Failure recovery | Not needed | Edit in thread; the label sets expectations |
| Reviewer incentive | Queue pressure → rubber-stamping | Reviews when free → engaged |
| Applies to | Irreversible actions (merges, writes) | Recoverable output (answers, drafts) |

## Production problem 6: annotations are worthless until something reads them

The last problem is my favourite, because it is the one almost everybody has and almost nobody names. Grab's phrasing: annotations were passive records without systematic analysis. The system had valuable information about successes and failures but was not learning from it.

That is an accurate description of most feedback systems I have seen. Collecting feedback feels like closing the loop. It is not; it is opening one end of it.

![The annotation table feeding five consumers, all converging on the next release](/imgs/blogs/grab-multi-agent-data-platform-support-11.webp)

Grab wired five consumers onto the same table, and the figure above is the point: an annotation is worth exactly as much as the number of things that read it.

1. **Automated evaluation.** Random annotations are pulled to create test cases for offline evaluation against real failure scenarios.
2. **Pattern analysis.** Specifically: is the Classifier consistently routing to the wrong agents; does a specific agent have quality issues; are certain query types prone to hallucination; do particular table schemas cause confusion.
3. **Quality metrics.** Annotation rate over time as a reliability signal and a regression detector.
4. **Targeted improvements.** Better prompt examples, earlier guardrails, agent-specific tooling.
5. **Training data.** Fine-tuning on domain patterns, improving few-shot examples, building regression suites from actual failures.

The four questions under pattern analysis are the most quietly valuable list in the post, because each maps to a different fix:

| Pattern | What it means | The fix |
| --- | --- | --- |
| Classifier routes to the wrong agents | Routing prompt is under-specified for a question class | Add that class to the Classifier's examples; add a routing eval set |
| One agent has quality issues | That specialist's prompt or tools are wrong | Fix one prompt; the other four are untouched |
| A query type is prone to hallucination | Retrieval is missing evidence for that shape of question | Add a tool or expand a scope — not a prompt change |
| Particular schemas cause confusion | Metadata is ambiguous, not the model | Fix the catalog; the agent was reading it correctly |

That last row is the one worth internalizing. Sometimes the annotation is telling you that your *data platform* has a bug, and no amount of prompt engineering will fix a column whose description says "misc field 3." An agent that answers questions about your metadata is also, unavoidably, an auditor of your metadata.

The annotation record has to be structured for this to work. Free-text "this was wrong" is unclusterable:

```yaml
# One annotation. The shape is what makes offline evaluation possible later.
annotation_id: a-2026-07-19-0143
thread: "#adw-support/1721380000.412"
question: "why does gmv_daily disagree with the finance dashboard for 2026-07-15?"

classifier:
  chose: [data, code_search]
  order: [data, code_search]
  reasoning: "compare two metric definitions; no incident signal in the question"
  verdict: incorrect                 # <- the routing itself was the failure
  should_have_chosen: [data, code_search, oncall]
  note: "a late-arriving partition on 07-15 was the actual cause; only the
         On-call Agent had visibility into the backfill"

agents:
  data:        { verdict: correct }
  code_search: { verdict: correct }

summary:
  verdict: incomplete
  reviewer_action: re_route          # approve | reject | refine | re_route | annotate
  reviewer: hiep.tran
  corrected_answer: "definitions match; 07-15 was backfilled at 03:10 the next day"

labels: [routing_miss, late_partition, metric_reconciliation]
promote_to_eval: true                # sampled into the offline suite
```

The field doing the most work is `classifier.verdict`. Both specialists were right; the answer was still wrong, because the plan was wrong. If your annotation schema only records whether the final answer was good, that failure is invisible — you will see "incomplete answer," conclude the Summarizer is weak, and tune the wrong component for a quarter.

This is also why [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) matters more for multi-agent systems than for single ones: with five components, the number of ways to be right for the wrong reason grows faster than your ability to notice.

## What the post does not say

I have been complimentary, and the post deserves it. But it is a company engineering blog, and it is worth being precise about which claims are load-bearing and unverifiable.

![What the post publishes, what it withholds, and why each gap matters](/imgs/blogs/grab-multi-agent-data-platform-support-12.webp)

**The impact numbers are ratios without denominators.** The post reports an order-of-magnitude reduction in resolution time, an effectively eliminated support backlog, and several full-time equivalents of reclaimed bandwidth. Those are real outcomes and I have no reason to doubt them. But an order of magnitude *from what*, measured *how*? Median or p95? Over which question mix? A tenfold reduction on questions that took four hours is transformative; a tenfold reduction on questions that took four minutes is a rounding error on a two-day-per-week problem. "Several FTEs" against a team size we are not told is similarly hard to convert into a payback estimate for your own team.

**No model, no cost.** We do not learn which model backs the agents, whether all five share one, or what a resolved question costs. For an architecture whose acknowledged weakness is latency and operating cost, that is the single most decision-relevant number, and it is absent. Five sequential inference calls on a frontier model is a very different business case from five calls where four are on a cheap model.

**The Classifier's accuracy is never quantified.** The Classifier decides which specialists run. If it under-routes, the answer is confidently incomplete — and, as the annotation example above shows, *every downstream agent will look correct*. The post names Classifier misrouting as something pattern analysis looks for, which tells us it happens, but gives no rate. This is the component whose failure is least visible and least gated.

**The rejection rate is missing, and the trust argument depends on it.** The human-in-the-loop story only means something in the context of how often reviewers reject. If it is 1%, the review is nearly ceremonial and the unreviewed-label policy is clearly right. If it is 20%, a meaningful share of published-then-corrected answers were wrong while visible, and the calculus is different. We are told the mechanism exists and not how often it fires.

**"Majority of standard inquiries" is doing a lot of work.** Bots autonomously handle the majority of standard user inquiries and a significant portion of common enhancement requests. The qualifiers *standard* and *common* are unbounded — they are defined by whatever the system handles well.

To be clear about what this is and is not: none of the above suggests the results are inflated. It means the post is a **design document, not a benchmark**, and it should be read as one. What transfers is the architecture, the six failure modes, and the reasoning. What does not transfer is any expectation about your own numbers.

There is one more gap that is architectural rather than editorial. The system's read path is well defended against dangerous *actions* and essentially undefended against wrong *answers* — the only control there is a human who may review after publication. For a support bot answering questions about table semantics, that is a reasonable risk posture. If the same answers fed an automated decision downstream, it would not be, and nothing in the architecture would tell you that you had crossed the line.

## Six failure modes, and which layer each one lives on

Grab's post names six problems they hit. Below are six *failure modes* — the concrete, symptom-level shapes those problems take when you are on the receiving end of a bad answer. To be explicit: these are not incidents Grab reported. They are the recurring failures I have watched teams hit while building systems of this shape, and I have anchored each one to the specific Grab design decision that prevents it, because that is what makes the decisions legible as engineering rather than as taste.

![Tree of six failure modes sorted by the layer that produced them](/imgs/blogs/grab-multi-agent-data-platform-support-14.webp)

The tree above is the diagnostic structure, and its shape is the argument: **a wrong answer looks identical from the outside no matter which layer produced it.** You cannot tell a routing miss from a compression loss by reading the answer. You have to ask six different questions, and the architecture either lets you ask them or it does not.

### 1. The confident summary of nothing

**Symptom.** A user asks where a metric comes from. The bot returns a fluent, well-structured answer naming a source table and a transformation — and the table does not exist. Not "no longer exists": never existed. The user, reasonably, believes it, files a ticket against the wrong pipeline, and burns two engineers' afternoon.

**The wrong first hypothesis.** The model hallucinated, so tighten the Summarizer's prompt. Add "only state facts supported by the agent outputs." This is the reflex, and it is almost always wrong.

**What is actually happening.** The Code Search Agent's lookup failed — an empty result, a timeout, a repository it lacked access to — and returned something like `no matches found for column gmv_daily`. That string went into the Summarizer's context as a *message*, alongside genuine findings. The Summarizer's job is to produce one coherent narrative from several inputs, and coherence is exactly the wrong objective when one input is an absence. It filled the gap.

**The fix.** Two parts, and the prompt is neither of them. First, a failed tool call must return a typed failure the orchestrator can branch on, not a string the model reads as evidence. Second, the Summarizer must be structurally incapable of seeing raw tool output — which is precisely the constraint Grab describes when they say the Summarizer combines *responses* into a coherent narrative. It reads three conclusions. An agent that concluded nothing contributes nothing, rather than contributing the *text of its own failure*.

**The lesson.** A summarizer will always resolve ambiguity toward fluency. Do not ask it to distinguish evidence from the absence of evidence; make absence unrepresentable in its input.

### 2. The routing miss nobody logged

**Symptom.** The answer is accurate, well-sourced, and incomplete in a way nobody notices for a month. A user asks why a daily metric disagrees with a finance dashboard. The Data Agent confirms both definitions match. The Code Search Agent confirms the transformation is correct. The answer says the numbers should agree. They still do not, because a partition arrived nine hours late and no agent ever looked at pipeline state.

**The wrong first hypothesis.** The Summarizer under-reported, or the Data Agent should have caught the freshness gap. Both plausible; both wrong.

**What is actually happening.** The Classifier decided the question needed two specialists, not three. Every agent it *did* call performed correctly. The failure happened before any of them ran, and — this is the vicious part — every per-agent quality metric you have will look perfect. Trace-level evaluation that scores each agent's output in isolation gives this run a clean bill of health.

**The fix.** The routing decision has to be a separately annotated, separately evaluated artifact. Grab's Classifier emits its reasoning and its task descriptions, which is the precondition; the annotation schema then has to record `chose`, `should_have_chosen`, and a verdict on the plan itself, independent of the verdict on the answer. That is the field I flagged as load-bearing in the annotation example earlier.

**The lesson.** In a routed system, **under-routing is the dominant silent failure**, and it is invisible to every metric that scores components rather than plans. It is also the failure mode with no safety gate: all four of Grab's layers guard against harmful actions, none against an incomplete investigation.

### 3. Only slow, never wrong

**Symptom.** The support bot's cost line quadruples in a week. No incident, no bad answers, no complaints. Someone eventually notices that a handful of investigations each triggered a multi-terabyte scan.

**The wrong first hypothesis.** The agent is writing bad SQL. It is not — the SQL is correct, well-formed, and semantically exactly what the question required.

**What is actually happening.** A user asked a question whose honest answer requires a wide scan: "has this column ever been null?" The agent wrote `SELECT count(*) ... WHERE col IS NULL` with no partition filter, because there is no correct partition filter for that question. The query was right. The query was also a full table scan on a partitioned table with four years of history.

**The fix.** This is the gap that neither intent classification nor SQL validity checking can close, which is exactly why Grab has a *third* layer. Their SQL validator explicitly checks for missing partition filters and excessive date ranges, and separately every query carries a strict execution timeout. Note that these are two different controls: the validator refuses a query that is *predictably* expensive, and the timeout kills one that turned out to be expensive despite looking fine. Neither subsumes the other.

**The lesson.** Cost is a third axis alongside correctness and safety, and it needs its own gate. The refusal message matters too: `"add a filter on event_date"` lets the agent adapt; a bare rejection produces a retry loop of increasingly creative rewrites, each of which also costs money.

### 4. The summarizer ate the anomaly

**Symptom.** Investigations that run long — four hops, or a threaded conversation with follow-ups — produce noticeably blander answers than short ones. Not wrong, just useless: "the pipeline appears to be operating normally" where a two-hop run on the same question would have said "row counts dropped 12% on the 15th."

**The wrong first hypothesis.** The model degrades with long context, so use a model with a bigger window. This buys a week.

**What is actually happening.** The context compaction is working exactly as designed and destroying the finding. When the token budget is exceeded, older messages get summarized. The Data Agent's observation that two of four sampled rows had a null `segment` is, to a summarizer, a detail — it compresses to "the query returned sample rows." The anomaly is the highest-information token in the transcript and the first thing a fluency-optimizing compressor discards.

**The fix.** Anomalies must not live in the summarizable tier. When a tool detects something unusual — a null rate over threshold, a schema mismatch, a failed run, a count delta — that finding is emitted as a *structured finding* and pinned alongside the question and the plan, which Grab describes as the critical context that remains unsummarized. Structured records survive compression; prose observations do not. Concretely: `{"kind": "null_rate", "column": "segment", "value": 0.5, "threshold": 0.1}` is a field the orchestrator can carry verbatim at negligible token cost.

**The lesson.** Summarization is lossy *in a direction*. It preserves narrative and discards outliers, which is the exact inverse of what an investigation needs. Design the pinned tier before you design the compressor.

### 5. The queue became a mute button

**Symptom.** Six weeks after launch, the approval rate is 98% and rising, median review time is under fifteen seconds, and the number of `refine` and `annotate` actions has fallen to roughly zero. The dashboard looks fantastic. The system has stopped improving.

**The wrong first hypothesis.** The agents got better. They did not; the reviewers gave up.

**What is actually happening.** Reviewing was made a blocking step, so a queue formed, and queue pressure changes reviewer behaviour in an entirely predictable direction. When a reviewer sees an almost-right answer with eleven more waiting, "approve" costs three seconds and "refine" costs three minutes. The economics of the review UI determine what reviewers do, and no amount of asking them to be thorough changes the economics.

**The fix.** Grab's is two-sided, and both sides are necessary. Remove the block — post the answer tagged `unreviewed`, so review pressure is no longer coupled to user-facing latency. And make the informative actions cheap: five actions rather than two, so that "this is 80% right and missing the freshness check" is one click plus a sentence rather than a decision to reject and rewrite.

**The lesson.** A review step is a piece of *mechanism design*, not a checkbox. If the uninformative action is cheaper than the informative one, you will converge on the uninformative one, and you will discover this three months later when your eval set has stopped growing.

### 6. A catalog bug wearing a model costume

**Symptom.** A specific family of questions — always about the same handful of legacy tables — produces confidently wrong answers. Every other table works. Three rounds of prompt engineering move the needle slightly and then stop.

**The wrong first hypothesis.** These schemas are unusual, so the model needs more examples of them. You add few-shot examples. It half-works, which is the worst outcome, because it keeps you on the wrong track for another sprint.

**What is actually happening.** Those tables have column descriptions like `misc_flag_3` and `amt` and `status_2`, several written years ago by someone who has left, at least one of which is now actively wrong because the semantics changed and the description did not. The agent read the metadata correctly and reported what it said. Your data platform has a bug and the agent is faithfully surfacing it.

**The fix.** Fix the catalog. This is the least satisfying sentence in the article and the correct one. The useful move is to make the failure legible: this is precisely the fourth question in Grab's pattern-analysis list — *do particular table schemas cause confusion?* — and its presence on that list tells you they hit it. When annotation clustering shows failures concentrated on a table rather than on a question type or an agent, the fix is upstream of the agent entirely.

**The lesson.** An agent that answers questions from your metadata is also an auditor of your metadata, whether you wanted one or not. The clustered-by-table failure signature is the single most valuable diagnostic the annotation table produces, because it is the one that points *outside* the system you are debugging.

## Steal this: a build order

If you have a support burden with the same shape — repeated procedure, different noun — here is the order I would build in. It is deliberately not the order the architecture diagram suggests, because the diagram shows the finished system and finished systems are terrible construction plans.

![Six build steps and what each one de-risks](/imgs/blogs/grab-multi-agent-data-platform-support-13.webp)

**1. Build one read-only tool behind a validator and a timeout.** No agent yet. Just the tool, the guard, and a test suite of queries the guard must refuse. This is unglamorous and it is the step that determines whether the project survives its first security review. You want to be able to say "the agent physically cannot do X" and point at a test.

**2. Ship one specialist on your narrowest real question.** Pick the question your team answers most often and understands best — for Grab that would be "where does this column come from." One agent, one prompt, that one tool. Run it against real Slack traffic in shadow mode and read every output yourself for a week. You are not measuring accuracy yet; you are learning what the failures look like.

**3. Add a Classifier that routes and refuses.** Once you have two agents you need routing, and the moment you have routing you should move refusal in front of it. Guardrails belong ahead of every agent, not replicated inside each one. Build the routing eval set at the same time as the router — it is much harder to reconstruct later.

**4. Add specialists only where the first one measurably fails.** This is the discipline that keeps the fleet from becoming a zoo. Every new agent should be justified by a logged failure class the existing agents could not handle. "It seemed cleaner" is not a reason; it is how you end up with five agents that each do 20% of one job.

**5. Build the review UI before the first write path.** Five actions, not two — approve, reject, refine, re-route, annotate. Do this while the system is still read-only, so that by the time you ask for permission to write, you have months of evidence about how often the thing is right.

**6. Wire annotations into an offline eval set.** Sample, cluster, and promote real failures into a suite you run before every prompt change. Until this exists, every improvement is a guess and every regression is a surprise.

The ordering principle is that **each step earns the trust required for the next one**. Teams that skip to step four — five agents, full tool surface, no review UI, no eval set — build something that demos brilliantly and gets quietly switched off in month two, when nobody can explain why it said what it said.

## Reach for this pattern when — and when not to

**Reach for it when:**

- Your support load is **procedurally repetitive over varying nouns**. Same investigation, different table. This is the single strongest signal.
- The evidence lives in **three or more systems** that a human currently joins by hand. Two systems rarely justifies the machinery; four almost always does.
- Your baseline is **hours, not milliseconds**. The multi-agent latency cost is only invisible against a slow baseline.
- The failure mode is **recoverable and visible** — a wrong answer in a thread that someone can correct, not an action nobody notices.
- You have **structured metadata to ground against**: a catalog, a lineage graph, quality metrics. Grab has Hubble, Genchi and Lighthouse. Without an equivalent, your agents are guessing eloquently.
- Someone will **own the eval set**. Not "we'll add evals later." Now.

**Skip it when:**

- Every request genuinely requires **new judgment**. If the procedure differs each time, there is nothing to absorb.
- **Latency is in a user's critical path.** Three sequential inference calls will not fit inside an interactive budget.
- You cannot **enumerate what the agent must refuse.** If you cannot write the refusal test suite, you are not ready to write the tool.
- Your **metadata is unreliable**. An agent reading a catalog full of "misc field 3" will produce confident nonsense, and you will spend the quarter blaming the model. Fix the catalog first; the agent will also tell you where it is broken, which is a genuinely useful side effect.
- **One agent already works.** Do not decompose a problem that has not yet shown you a seam. Split when a specific failure class demands it.
- The **consumer of the answer acts irreversibly on it**. Then the post-then-review trade is unavailable, the queue returns, and much of the throughput benefit goes with it.

The transferable core of Grab's post is smaller than the architecture and more durable: put refusal in front of capability, make every failure attributable to one component, keep the reviewer's informative actions cheaper than the uninformative ones, and treat the feedback table as a system with consumers rather than a log. The five agents are an implementation detail. Those four properties are the design.

## Further reading

- [From firefighting to building: How AI agents restored our team's core productivity](https://engineering.grab.com/from-firefighting-to-building) — the original post by Sneh Agrawal, Rishi Raj, Ayan Chatterjee, Wen Zhong Tan and Sai Reddy Kakumanu.
- [Designing multi-agent systems: patterns, case studies, pitfalls](/blog/machine-learning/ai-agent/designing-multi-agent-systems) — the general design space this instance sits in.
- [Scaling managed agents: decoupling the brain from the hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands) — the same split, reached independently for a very different system.
- [Effective context engineering for AI agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) — the techniques behind production problem 1.
- [Tool schema design principles](/blog/machine-learning/ai-agent/tool-schema-design-principles) — why 30 tools is a prompt problem.
- [Human-in-the-loop design](/blog/machine-learning/ai-agent/human-in-the-loop-design) — review UIs that stay engaged rather than decaying into rubber stamps.
- [Agent observability and tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing) — what you need instrumented before the annotation flywheel can spin.
