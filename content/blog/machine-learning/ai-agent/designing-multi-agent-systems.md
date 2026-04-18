---
title: "Designing Multi-Agent Systems: Patterns, Case Studies, Pitfalls, and the Interview Questions"
publishDate: "2026-04-18"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  [
    "ai-agent",
    "multi-agent",
    "orchestration",
    "architecture",
    "system-design",
    "agent-communication",
    "interview",
    "llm",
    "coordination",
    "case-studies",
  ]
date: "2026-04-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Multi-agent systems are seductive — specialist agents, emergent behavior, the appeal of a 'team of AIs.' They are also where most agent projects get stuck. This is a long, rigorous walk through the patterns that actually work (orchestrator-worker, supervisor, pipeline, debate, hierarchical, blackboard), with deep case studies of real production systems (Anthropic's research agents, Claude Code's single-agent philosophy, Devin, MetaGPT, ChatDev, AutoGPT, customer support bots), and an interview Q&A section covering what gets asked at senior level."
---

## Why This Article Is A Little Skeptical

Multi-agent systems are currently the most overused and underdelivered idea in applied LLM engineering. You will read marketing copy about "teams of AI specialists collaborating to solve your problems," and if you squint, some of those teams are three prompts in a loop that quietly make the product worse than calling one strong model directly.

There is a very good version of this technology, and there is a very bad version, and most systems that ship to production need to cross a high bar before the good version wins. The goal of this article is to get you to the good version, honestly:

It is organized so that you can read it end-to-end as a design reference, or skip to the case-study sections (Anthropic's research system, Claude Code, Devin, MetaGPT, AutoGPT, customer support bots, Constitutional AI) to see the patterns grounded in real products. The interview Q&A at the end is written for staff-level agent-systems interviews and is deliberately specific — generic answers will fail those interviews.

If you have read the companion pieces in this cluster on [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide), [long-running architecture](/blog/machine-learning/ai-agent/designing-long-running-agents-architecture-foundations), and [long-running reliability](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production), this one sits alongside them. Where those articles focus on how to make one agent do a complex task well, this one focuses on when (and how) to split that task across multiple agents without making it worse.

## Part 1: What "Multi-Agent" Actually Means, And What It Doesn't

Before we design anything, a working definition. The word "agent" is overloaded, and the word "multi-agent" adds another layer of fuzz.

A **multi-agent system** is one in which multiple LLM-driven components satisfy all three of these properties at once:

- **Distinct roles or responsibilities.** Not just "agent 1" and "agent 2" with the same prompt and same tools — each agent has a reason to exist that's different from the others.
- **A defined communication protocol.** They exchange information according to rules, not by sharing an implicit context.
- **Coordination toward a goal no single agent owns.** The final outcome emerges from the combination of agents' work, not from any one of them alone.

That definition excludes a lot of things that are *called* multi-agent systems but aren't, architecturally:

- **An agent that calls many tools.** Tools are not agents. A tool does not reason about when to run; the agent calling it does. GPT-4 calling a calculator is not multi-agent.
- **A chain of prompt templates.** A pipeline where step 1 extracts entities, step 2 summarizes, and step 3 formats is a pipeline. Whether you call each step an "agent" is a semantic choice, not an architectural one — none of those steps reasons about its peers.
- **A single agent with a delegation subcall.** If your main agent calls `summarize(text)` via a helper LLM and then continues, that helper did not make autonomous decisions. It's a function.
- **A chat interface in front of one model.** Whatever the UI looks like, if all tokens come from one model's context, it's one agent.

Why does the definition matter? Because the engineering problems are different. Single-agent engineering is about prompt quality, tool design, state management, and context engineering. Multi-agent engineering inherits all of those plus a new axis of failure: **coordination**. Coordination bugs are hard. They compound across agents the way reliability errors compound across steps in long-running agents. If you accidentally build a multi-agent system when you only needed a single agent with better tools, you paid the coordination tax for no reason.

## Part 2: When Single-Agent Wins (The Default You Should Argue Yourself Out Of)

Start single-agent. Every time. A well-designed single agent with good tools, good context engineering, and good state management solves surprisingly complex problems. The default answer to "should we go multi-agent?" is no, and you should need a specific reason to override it.

The reasons to stay single-agent are worth listing explicitly because they are the counterweight to the multi-agent enthusiasm you'll encounter in team discussions:

**One coherent view of the task.** A single reasoner with one context makes internally consistent decisions. It does not forget its own earlier conclusions when it re-enters a new "subagent" context. Consistency is cheap here and expensive everywhere else.

**Lower cost per request.** Every agent boundary costs tokens. Each sub-agent gets its own system prompt, persona, tool definitions, and often a restated version of the task. Even small multi-agent systems frequently use 3–10× the tokens of a single agent for the same work. Anthropic's own published multi-agent research system uses roughly 15× the tokens of a single-agent chat (more on this below).

**Easier to debug.** One trace to read instead of N. When something goes wrong, you look at one context, one tool-call log, one state object. Multi-agent bugs are often "the handoff between agent A and agent B loses information" — a bug that doesn't exist if there's no handoff.

**Reliability compounds favorably.** In a single-agent system, your chain of success probabilities multiplies over steps. In a multi-agent system, it multiplies over steps *and* over handoff validations. Every extra multiplicand eats your end-to-end success rate. A 95% step-success single agent with 20 steps succeeds 35% of the time. A 95% step-success, 90% handoff-reliability multi-agent with the same work split 4-ways barely gets to 20%.

**No implicit-state bugs.** Single agents can't have "I assumed the other agent did X." They're the only ones who could have done X.

A good mental test: when someone on your team proposes splitting a single agent into two, ask "what does each agent see, write, and decide?" If you can't answer in a single paragraph with precise boundaries, the split is premature.

### Case Study: Claude Code's Deliberate Single-Agent Philosophy

Anthropic's Claude Code is a notable counter-example to the industry-wide multi-agent trend. It is a coding assistant that:

- Runs on a **single reasoning loop** with Claude as the brain.
- Operates over arbitrarily long tasks using a **task list and scratchpad** (structured state), not sub-agents.
- Uses a curated toolset (file read/write, shell, grep, web fetch, etc.) plus skills (reusable prompt-and-resource bundles).
- Does *not* spawn specialized sub-agents for "the coder," "the tester," "the reviewer" roles.

The reasoning, which Anthropic has written about publicly, is pragmatic:

- Coding requires a coherent mental model of the codebase. Splitting that model across agents loses fidelity at every handoff.
- Dev loops are long (hours) and have many steps (dozens). Compounding handoff reliability loss across those steps would be catastrophic.
- Debuggability and user trust matter more than theoretical specialization gains. A single trace the user can read is worth more than a cleverer architecture they can't understand.
- Tools and skills provide the specialization benefits of multi-agent without its coordination cost. A `run_tests` tool is a specialist. A `code_review` skill is a specialist. Neither needs to be an agent.

The lesson is not "never use multi-agent." The lesson is that one of the most successful production coding agents in 2024–2026 deliberately chose single-agent, and measured it against multi-agent alternatives before doing so. It's an existence proof that many problems that feel like they need a team of agents actually don't.

### A Single-Agent Baseline Protocol — What to Measure Before Splitting

Before you argue for multi-agent, measure the single-agent baseline rigorously. Five numerical criteria worth tracking, each with a threshold that suggests a split might be justified:

| Metric | Threshold that hints at splitting | What it actually means |
| --- | --- | --- |
| Task success rate | < 70% on structured eval | Current agent is struggling end-to-end; more agents *might* help — or prompt/tool work will |
| p95 task latency | > 60s and user-facing UX cares | Sequential steps dominate; parallelism across sub-agents may reduce wall time |
| Cost per task | within budget → no urgency to split | Splitting increases cost; only worth it if cost isn't the constraint |
| Context window utilization | > 80% full across > 50% of tasks | Context pressure is real; sub-agent offloading may help |
| Per-step drift rate | steps where agent forgets goal > 10% | A long single context may be the cause; bounded sub-contexts may fix it |

If *none* of these thresholds are being crossed, multi-agent is almost certainly premature optimization — your single agent is within spec. Improve prompts, tools, and memory before adding coordination.

Trade-off of the protocol: measuring rigorously is work. Teams skip it because "we want to try multi-agent." That's how architectures ship without justification and then accumulate coordination cost no one budgeted for.

A specific anti-measure: do not use "vibes" or "the team thinks multi-agent feels right." Vibes are how you end up at 15× cost without the quality gain Anthropic's research system actually demonstrated.

## Part 3: When Multi-Agent Wins (The Five Specific Signals)

Multi-agent is the right answer only when at least one of these signals clearly applies, and you can name which one. Vague appeals to "better coordination" or "specialization" are not signals — they are marketing.

### Signal 1: The Task Has Cleanly Separable Subtasks With Self-Contained Contracts

A subtask qualifies as separable if you can write down:

- A short, precise input specification.
- A short, precise output specification.
- Both without having to reproduce the parent agent's full context.

Example of separable: "Given a PR diff, produce a structured list of review comments with severity, file, line, and category." Input: diff + project style guide. Output: `List[Comment]`. The sub-agent does not need to know the PR's history, the author's prior work, or the team's Q1 roadmap.

Example of not separable: "Help me decide whether to refactor the auth module." The decision depends on so many entangled factors (codebase state, team bandwidth, product roadmap, historical context) that any sub-agent would either need the parent's full context or would make a decision based on insufficient information.

### Signal 2: Specialization Produces Measurable Quality Gains

Multi-agent is worth its cost only if specialized agents beat a generalist. This is testable. Run your proposed architecture. Run the single-agent baseline. Compare on your own eval. If the single-agent performs within 10% of the multi-agent at a fraction of the cost, keep single-agent.

In my experience, specialization wins most reliably when:

- The subtasks require different **tool sets** (coding vs. financial analysis vs. search).
- The subtasks reward different **prompt styles** (creative vs. rigorous; verbose vs. concise).
- The subtasks benefit from different **model sizes** (cheap model for classification, strong model for synthesis).

### Signal 3: Context-Bounding Is a First-Class Goal

If the parent agent's context is crowded and a subtask naturally carries a lot of verbose intermediate state, splitting to a sub-agent lets that sub-agent reason in a fresh context and return a compact summary. You're using multi-agent as a form of context compression.

This is the core insight behind Anthropic's research multi-agent system: parallel subagents each research one aspect of a query in their own context window, returning compact summaries to the lead agent. Without that split, the lead's context would fill up with tens of thousands of tokens of raw web content before it could synthesize anything.

### Signal 4: Independent Parallelism Is Available And Valuable

Some tasks have naturally parallel subtasks. A research project with 10 independent sub-questions can fan out to 10 parallel investigations, each returning a finding, with total wall time ≈ the slowest investigation instead of the sum. For user-facing applications, this can be the difference between 40s and 4s response time.

Parallelism is only valuable if:

- The subtasks are actually independent (no sequential dependency).
- The wall-time reduction matters (it doesn't if users are OK waiting).
- The total token cost is worth the wall-time reduction (parallelism does not reduce cost, only latency).

### Signal 5: Adversarial Diversity Improves Quality

Some tasks benefit from opposing perspectives: a proposer and a critic; a writer and an editor; a code generator and a test runner. The interaction between agents finds flaws that any single agent would miss. This is what enables the debate and critic-proposer patterns (covered below) to outperform single-agent reasoning on hard problems.

The crucial nuance: **adversarial diversity requires a real signal to bounce off**. A proposer and a critic with no ground truth to check against will drift into synthetic plausibility. The pattern works best when one side has access to something the other doesn't — test results, a verifier, a human's opinion, a gold answer.

If none of these five signals apply to your problem, you do not need multi-agent. You need better tools, better prompts, better state management, or a better-trained single agent. Before you introduce a coordination layer, you should have failed at those first.

### Signal Combination Matrix — Which Signals Together Actually Justify Multi-Agent?

Signals rarely appear singly. The pairing determines whether the case is strong, weak, or mis-diagnosed.

| Signal pair | Verdict | Archetype |
| --- | --- | --- |
| Separable + Parallel | **Strong case** | Anthropic's research system (parallel subagents on decomposed questions) |
| Separable + Context-bounding | **Strong case** | Research agent where each sub-question's content would crowd the lead |
| Parallel + Specialization | **Strong case** | Code review: static analysis, security, coverage agents in parallel |
| Adversarial + Verifier | **Strong case** | Constitutional AI; proposer + critic + verifier loop |
| Separable alone (no parallelism, no context-bounding) | **Weak case** | Probably a pipeline, not multi-agent — cheaper |
| Specialization alone | **Weak case** | Usually solvable with per-stage tools or prompts, not separate agents |
| Parallelism alone (work isn't separable) | **Wrong fit** | You'll spend the parallelism on wasted duplicated work |
| Adversarial without a grounded critic | **Wrong fit** | Both sides drift into plausibility; quality doesn't rise |
| "Feels like a team would help" | **Not a signal** | Feelings aren't signals; measure |

Why single signals are usually insufficient: *one* signal rarely clears the coordination cost bar. Multi-agent's cost premium (2–15× tokens) needs *compounded* benefits. "Separable" alone gets you a pipeline, not multi-agent. "Parallel" alone, without separability, just lets you waste time in parallel. The strong-case rows all deliver two independent benefits (speed AND quality, or bounded context AND specialization) — that's what earns the premium.

The practical rule: if you can't name *two* signals that apply, keep the single-agent design and invest the engineering elsewhere. If you can, the next question is whether the *combination* of benefits is independently measurable — not "we think this will help" but "we predict 20% quality gain AND 3× latency reduction, and we will measure both."

## Part 4: The Canonical Architectural Patterns, With Case Studies

Most real multi-agent systems are one of seven patterns — or a modest composition of two. Learning these (and which case studies exemplify each) gives you a vocabulary for design conversations and a checklist for spotting misapplied architectures.

### Pattern 1: Orchestrator–Worker

By far the most common and most useful pattern in production. A central orchestrator plans, dispatches tasks to workers, and collates results. Workers are self-contained, receive narrow inputs, and return structured outputs.

```
                    ┌──────────────┐
                    │ ORCHESTRATOR │
                    │  (planner)   │
                    └──┬────┬───┬──┘
                       │    │   │
              ┌────────▼┐ ┌─▼──┐ ┌▼────────┐
              │Worker A │ │B   │ │Worker C │
              └─────────┘ └────┘ └─────────┘
                       │    │   │
                    ┌──▼────▼───▼──┐
                    │ ORCHESTRATOR │
                    │  (collator)  │
                    └──────────────┘
```

The orchestrator is the only agent with a global view. Workers never talk to each other. The communication contract is fixed: each worker has a declared input and output schema. The orchestrator can dispatch workers in parallel, sequentially, or in a DAG of dependencies.

Works well for: research assistants, code review systems, data enrichment pipelines with decision-making, agents that produce structured reports over many sources.

**Case Study — Anthropic's Multi-Agent Research System.** In mid-2025 Anthropic published a detailed engineering post about the multi-agent research system that powers their "Research" feature. The architecture is a textbook orchestrator-worker with a specific twist for long-horizon information gathering:

- A **lead research agent** receives the user's question. It decomposes the question into sub-questions and a research plan.
- The lead dispatches **3–5 subagents in parallel**, each assigned a specific sub-question and given access to web search, browsing, and retrieval tools.
- Each subagent runs in its own context window. It performs several rounds of search and reading. When it has enough material, it returns a **compact summary** of its findings, with citations.
- The lead collates the summaries, identifies gaps, and either dispatches another round of subagents or moves to synthesis.
- A final **citation agent** verifies every factual claim against sources in the evidence bundle and attaches citations.

Anthropic reported that on their internal research benchmark, this system improved over single-agent Claude Opus by approximately 90.2% — a remarkable gain. The cost? Roughly 15× more tokens than a single-turn chat exchange. The insight the post emphasizes is that for research specifically, the dominant cost constraint is not tokens but **model capability and parallelism**: a strong model with 15× tokens but running 5 investigations in parallel substantially outperforms the same model running once with a crammed context.

The key architectural lessons from that case study:

- Subagents worked because each had a **self-contained research assignment**, returned **structured findings**, and never had to know about each other.
- The lead's job was **planning and synthesis**, not doing research itself. Splitting "think about the question" from "find the information" paid off.
- **Parallelism** was table-stakes. Running subagents sequentially would have reproduced a single-agent's experience with no speed advantage.
- Prompt engineering dominated. Anthropic's engineers wrote extensively about how much time was spent tuning the orchestrator's decomposition prompt and the subagents' researcher persona.
- **Evaluation was hard.** Research quality is subjective; they built a rubric-based evaluation and spent substantial time calibrating LLM-as-judge agreement with human evaluators.

The counter-lesson: this architecture worked for research because research has the five signals (separable sub-questions, specialization via tool/context isolation, parallelism, context-bounding). It would not help for coding, where the task is not cleanly separable and a single coherent view of the codebase matters.

### When Orchestrator–Worker Breaks — Three Failure Scenarios

Even the best-fitting pattern has specific failure modes. For orchestrator–worker, three that recur in production:

**Scenario 1: Workers need global context the orchestrator didn't give them.**
Symptom: a worker keeps asking the orchestrator for clarification, or silently makes assumptions that contradict each other. Root cause: the orchestrator's decomposition assumed each subtask was self-contained; it wasn't. Example: a research system where two parallel subagents are each assigned a sub-question that's actually the same question framed differently — they return overlapping results, and the lead has to reconcile. Fix: richer decomposition prompts with explicit "subtasks must not overlap" constraints, and an orchestrator-level dedup/overlap detection step. If the task is fundamentally entangled, *the pattern is wrong* — a single agent with better tools beats a forced split.

**Scenario 2: Orchestrator's planning quality is the ceiling.**
Symptom: workers do great work individually; the final output is mediocre. Root cause: the orchestrator's plan is the weakest link — it allocated work poorly, missed a necessary subtask, or set ambiguous goals. Workers optimized within their allocated subtask, but optimizing the wrong subtask. Fix: invest disproportionately in the orchestrator (best model, most prompt iteration, explicit planning rubric); treat worker prompts as tuning once the plan is stable. Trade-off: you're paying the premium model price on every top-level call.

**Scenario 3: Parallel workers race on shared resources.**
Symptom: intermittent failures that don't reproduce locally. Root cause: two workers hit the same API rate limit, write to overlapping rows, or consume the same external resource. Fix: per-worker rate budgeting, write-partitioning by worker ID, and idempotency keys on anything that touches shared state. Trade-off: some of the parallelism you wanted goes away (serialization on hot resources); measure whether wall-time wins still justify the coordination.

General principle: orchestrator–worker is robust when workers are genuinely independent and the orchestrator's planning is competent. Break either of those and the pattern's benefits disappear faster than its costs.

### Pattern 2: Supervisor (Hub-And-Spoke)

A supervisor variant of orchestrator-worker where the hub is less a planner and more a **traffic cop**. Each request comes in, the supervisor classifies it, routes to the right specialist, and returns the result. The supervisor does not decompose or collate — it just routes.

```
          ┌─────────────┐
          │ SUPERVISOR  │
          │  (router)   │
          └──┬──┬──┬──┬─┘
             │  │  │  │
        ┌────▼┐┌▼┐┌▼┐┌▼────┐
        │ A  ││B││C││ D   │
        └────┘└─┘└─┘└─────┘
```

Works well for: customer support triage, developer-assistance systems with specialized backends (SQL helper, DevOps helper, docs search), e-commerce agents that split shopping / shipping / returns.

**Case Study — Modern Customer Support Agents (Intercom Fin, Ada, Zendesk AI).** The leading customer-support AI products in 2025–2026 use a supervisor pattern with remarkable consistency. The flow:

- User sends a message.
- A **classification agent** (often a small cheap model) tags the intent: billing / technical / refund / cancellation / general / escalate-to-human.
- Based on the tag, the request is routed to a **specialized agent** tuned for that domain:
  - Billing agent has access to the billing API and a policy-retrieval tool scoped to payment and invoicing policies.
  - Technical agent has access to the knowledge base, status page, and a diagnostic tool.
  - Refund agent has access to order history, refund eligibility rules, and a refund approval tool with dollar limits.
  - Cancellation agent has access to the subscription API and a retention-offer tool.
- If any specialist runs out of confidence or the request exceeds its scope, it routes back to the supervisor, which either picks a different specialist or escalates to a human agent.

What makes this work in production:

- **Strict scope per specialist.** The refund agent literally cannot call the subscription-cancellation tool. This is not just prompt discipline — it's enforced at the tool-permission layer. A prompt injection that tries to convince the refund agent to cancel a subscription fails at the tool boundary.
- **Structured inputs to specialists.** The supervisor passes not free text but a typed struct: `{user_id, intent, sentiment, prior_context, current_message}`. The specialist does not re-read the chat history from scratch.
- **Escalation as a first-class path.** The specialist can always hand back with a reason. The supervisor logs every escalation, which becomes training data for improving the router.
- **Observability per route.** Dashboards track success per specialist, per intent, per resolution time. When a specialist regresses, it's visible; when a classifier miscategorizes, it's visible.

The pitfalls that teams hit repeatedly:

- **Supervisor quality is the ceiling.** If routing is 85% accurate, 15% of users get the wrong specialist. Improving specialists helps only the 85% that routed correctly; improving the supervisor helps everyone.
- **Too many specialists creates a fragile routing space.** Five specialists with clear boundaries beat fifteen with overlapping domains.
- **Classification confidence matters.** Low-confidence classifications should default to a general-purpose fallback or human, not a random specialist.

The lesson: the supervisor pattern works best when the domain partitions cleanly and the cost of a misroute is low (you can always route again, or escalate).

### Pattern 3: Pipeline (Sequential Specialization)

A fixed sequence of agents, each refining or transforming the output of the previous. No runtime decisions about who runs next — the order is part of the system's design.

```
[Extractor] → [Summarizer] → [Critic] → [Rewriter] → [Verifier]
```

Works well for: content processing (blog post → social media assets), document understanding pipelines, RAG flows (retrieve → rerank → synthesize → cite), data enrichment.

**Case Study — Perplexity's Search and Synthesis Pipeline.** Perplexity (and similar search-first LLM products) run what is essentially a pipeline with multiple LLM stages. The flow, simplified:

- **Query understanding stage.** Parses the user question, detects the intent (factual, exploratory, shopping, navigational), decides on retrieval strategy.
- **Multi-query generation stage.** Generates several search queries covering different angles of the question.
- **Retrieval stage.** Executes web search; retrieves candidate documents.
- **Reranking / filtering stage.** Scores documents for relevance and quality; selects the most useful subset.
- **Synthesis stage.** Reads the selected documents, composes an answer with inline citations.
- **Citation verification stage.** Checks that every cited fact is actually supported by the cited source.

Whether you call this "multi-agent" or "a pipeline with LLM stages" is a definitional choice. The important architectural observation is that this pattern works because:

- The sequence of stages is **known in advance**. There's no runtime ambiguity about what comes next.
- Each stage has a **narrow, typed job**. Bugs in one stage are local.
- The pipeline is **highly parallelizable**. Multi-query generation produces N queries; retrieval runs N searches; reranking happens on the combined set. Wall time stays low.
- The pipeline is **testable per stage**. Query understanding, reranking, synthesis, and citation can all be tested separately.

Teams sometimes over-engineer this pattern by adding runtime branching ("if this stage fails, try that stage"). In practice, keeping the pipeline rigid and handling exceptions at the boundary is cleaner and easier to debug.

### Pattern 4: Debate / Critic-Proposer

Two (or more) agents take turns, one generating and one criticizing. Over a few rounds, quality tends to rise — the critic finds flaws, the proposer addresses them.

```
Proposer → Critic → Proposer → Critic → ... → Final
```

A variant is **jury/voting**: N independent proposers produce candidate answers, a judge picks the best or takes a majority vote.

Works well for: hard reasoning tasks, code generation with a verifier (tests), essay or analysis where quality matters more than speed, alignment training, red-teaming.

**Case Study — Constitutional AI (Anthropic).** Constitutional AI is not an inference-time multi-agent system per se — it's a training methodology — but it's the cleanest real-world application of the critic-proposer pattern and illustrates the preconditions for debate to work.

In Constitutional AI, during the training phase, Anthropic:

- Has Claude (as proposer) generate a response to a prompt that might elicit harmful output.
- Has Claude (as critic, with a different prompt including a "constitution" of principles) critique the response for violations.
- Has Claude (as reviser) rewrite the response to address the critique.
- Uses the final revision as the training signal.

The proposer-critic-reviser loop is entirely self-play. What makes it work:

- The critic has a **grounded reference** (the constitution) to criticize against. This is not drifty self-criticism; it's criticism against fixed principles.
- The loop runs a **small, fixed number of rounds**. It doesn't explore indefinitely.
- The outcome is a **concrete artifact** (revised text) that goes on to training, not a conversation transcript.
- Critique quality is measurable downstream via alignment evals.

The lesson: the debate pattern succeeds when the critic has a ground-truth-like reference to anchor on (a constitution, test suite, verifier model, human evaluator). It fails when both sides are improvising plausibility — in that regime, quality doesn't rise, it just becomes more verbose.

**Case Study — AlphaCode-style Generation + Selection.** DeepMind's AlphaCode (and later competitive coding systems) use a variant of the jury pattern: generate a large number of candidate programs (thousands to millions), then filter down using test execution and clustering. It's not chatty peer debate; it's massive parallel proposal plus algorithmic selection.

What this illustrates: the *real* signal in "debate" is the selection mechanism, not the chat. If your selection mechanism is a judge model arguing with the proposer, you're paying a lot of tokens for a weakly calibrated judge. If your selection mechanism is code execution against tests, you have strong signal and debate is just a way to get there.

### Pattern 5: Hierarchical (Orchestrator Of Orchestrators)

For truly large tasks, orchestrator-worker composes recursively. A top-level orchestrator delegates to mid-level orchestrators, who delegate to workers.

```
Top Orchestrator
 ├── Sub-Orchestrator (research phase)
 │    ├── Worker (source A)
 │    ├── Worker (source B)
 │    └── Worker (source C)
 ├── Sub-Orchestrator (synthesis phase)
 │    ├── Worker (draft)
 │    └── Worker (fact-check)
 └── Sub-Orchestrator (review phase)
      └── Worker (final polish)
```

Works well for: large research tasks (the Anthropic research system is technically a hierarchy of depth 2, lead + subagents), autonomous coding agents tackling multi-repository changes, long-horizon planning systems.

**Case Study — Devin (Cognition Labs).** Devin, announced in 2024, is the best-known attempt at an autonomous software engineering agent. Its architecture is a hierarchical arrangement:

- A **top-level planner** that takes a task (e.g., "fix bug in PR #42 and deploy") and produces a plan: clone repo, reproduce bug, find root cause, fix, test, commit, push.
- **Phase-level coordinators** that manage each plan step (each could itself be a small orchestrator, depending on complexity).
- **Worker-level agents** with tools: shell, file editor, browser, code search.
- Durable state between phases — the planner can re-read its earlier decisions, artifacts from prior phases, and the original task.

Devin's public demos in 2024 showed both striking successes (autonomously fixing real bugs in public repositories, navigating unfamiliar codebases) and spectacular failures (getting stuck in loops, producing broken code, misunderstanding requirements). Later reporting and community testing revealed a consistent pattern: Devin works well on **bounded, well-specified tasks with verifiable success criteria** (bug fixes with failing tests, small features with clear requirements) and struggles when success is ambiguous or the task requires deep codebase-wide judgment.

The lessons:

- Hierarchy genuinely helps for long multi-phase tasks. Without it, a flat agent loses track of the plan and drifts.
- **Depth should stay shallow.** Devin's effective architecture is depth 2 (planner + workers). Deeper hierarchies in agent demos consistently collapse under coordination cost.
- **Verifiable progress is essential.** Devin works on coding tasks partly because tests give it a signal. Tasks with no verifier are where it stumbles.
- **Human oversight UX matters.** The production version of Devin (and competing products like Cognition's successor systems) exposes the planner's plan and intermediate results to the user so that course correction is possible. Pure autonomy without oversight is where the demos looked worst.

### Pattern 6: Peer Network / Swarm

Agents communicate freely, with no central orchestrator. This is what many "multi-agent framework" demos look like — and it is, with rare exceptions, the pattern that fails most reliably in production.

```
Agent A ⇄ Agent B
  ⇅         ⇅
Agent C ⇄ Agent D
```

No single source of truth. Agents chat with each other in free text. Decisions are emergent rather than directed.

**Case Study — AutoGPT (2023).** AutoGPT, released in spring 2023, was the viral poster child for peer-agent systems. A main agent would spawn sub-agents, assign them tasks, have them report back, and continue indefinitely toward a user-supplied goal. It used unstructured text as the coordination medium and had no explicit termination criterion.

What people actually experienced:

- **Infinite loops.** An agent would get stuck repeating the same task, with no mechanism to detect or escape.
- **Runaway costs.** Without budgets, a goal like "start a business" could consume hundreds of dollars of API calls overnight without converging.
- **Coordination collapse.** Agents would hallucinate about other agents' completed work, produce fake "reports," and drift further from the original goal with each round.
- **Irreducible debuggability.** When it went wrong, you couldn't isolate which agent was responsible; the free-text transcript was a tangle.

AutoGPT was genuinely useful as a research artifact — it taught the community in a very public way that unstructured peer coordination does not work for long-horizon tasks. Most serious production systems since have adopted structured orchestrator-based patterns.

**Case Study — MetaGPT and ChatDev (2023).** These academic projects simulate a software company with role-based agents: Product Manager, Architect, Engineer, QA. Each agent has a prompt tuned for its role and a specific position in the workflow.

What makes them interesting:

- They impose **Standard Operating Procedures (SOPs)**: fixed handoff sequences with structured artifact types (PRD, design doc, code, test report). This partially turns the peer network into a pipeline, which is why it works at all.
- They produce impressive demos on **small, bounded problems** — simple games, basic CRUD apps.
- They fail to scale to realistic complexity. As soon as the project requires cross-role negotiation ("the architect's design conflicts with the engineer's constraints"), the free-text negotiation drifts and code quality collapses.

The useful takeaway: MetaGPT and ChatDev showed that peer networks work if you constrain them with enough structure that they become effectively hierarchical or pipeline patterns. The less structure, the more they fail.

**When peer networks actually make sense.** There are narrow cases: very short-lived multi-agent interactions with a clear terminating condition (e.g., a 3-round debate for a specific decision), or simulation environments where emergent behavior is the point (games, social simulations). For business applications with reliability requirements, almost never.

### Pattern 7: Blackboard

A shared structured data object (the "blackboard") that agents read from and write to, coordinated by a controller that decides which agent runs next based on the blackboard's state.

```
          ┌─────────────┐
          │ BLACKBOARD  │  ◀── shared structured state
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │ CONTROLLER  │  ◀── picks next agent based on state
          └──┬──┬──┬──┬─┘
             │  │  │  │
        ┌────▼┐┌▼┐┌▼┐┌▼────┐
        │ A  ││B││C││ D   │
        └────┘└─┘└─┘└─────┘
```

Classic AI pattern from expert systems, still useful in LLM-era systems. The blackboard is **structured** (JSON, schemas), not free text. Each agent reads only the part of the blackboard it needs, writes back its contribution, and the controller advances the system.

Works well for: incident investigation (different agents contribute observations about network, application, database, user reports), medical diagnosis simulations, legal-case analysis, document drafting where different agents contribute different sections.

**Case Study — Incident Response Agents.** Several enterprise incident-response platforms in 2025–2026 use a blackboard pattern. The blackboard contains sections like: `incident.summary`, `incident.timeline`, `incident.affected_services`, `incident.hypotheses`, `incident.evidence`, `incident.recommended_actions`. Specialized agents — network agent, app-logs agent, database agent, deployment agent — each write their findings into the relevant blackboard section. A controller agent reads the accumulating state, decides whether enough evidence exists to form hypotheses, and either spawns further investigation (into specific hypotheses) or produces the final incident report.

What makes this pattern successful:

- The blackboard structure **enforces contracts**. Each section has a schema. Agents can't just dump free text.
- Agents are **stateless** with respect to each other. They read the blackboard, do their job, write back. No agent needs to know who else is running.
- The controller has **global visibility** without being a bottleneck — it reads cheap JSON, not agent transcripts.
- **Human observability is excellent.** An on-call engineer can look at the blackboard at any time and see exactly what's known and what's been decided.

## Part 5: A Rigorous Comparison Table

With the case studies in mind, the patterns' tradeoffs become clearer:

| Pattern | Parallelism | Control | Debuggability | Cost | Example that works | Example that fails |
| --- | --- | --- | --- | --- | --- | --- |
| Orchestrator–Worker | high | medium | high | medium | Anthropic research | poorly-specified workers |
| Supervisor | low | high | high | low | Intercom Fin, Ada | too many specialists |
| Pipeline | low-medium | very high | very high | low | Perplexity | runtime branching |
| Debate / Critic | none | medium | medium | high | Constitutional AI | no grounded critic |
| Hierarchical | high | medium | medium | high | Devin (depth 2) | depth > 2 |
| Peer / Swarm | high | low | very low | high | — | AutoGPT, MetaGPT at scale |
| Blackboard | medium | medium | high | medium | incident response | unschema'd blackboard |

The column "Example that fails" matters more than the one that succeeds. For each pattern, failure modes have specific shapes, and avoiding them is more important than chasing the successes.

The practical default remains: **orchestrator-worker for decomposable tasks, pipeline for fixed transforms, supervisor for routing-shaped problems**. Promote to hierarchy when task size demands it. Use debate only when you have a grounded critic. Avoid peer networks until you have a specific reason.

### A Pattern Selection Decision Tree

A concrete decision procedure. Walk top-to-bottom; the first "yes" wins.

```
Is the sequence of stages known in advance and the same every time?
  ├── Yes → Pipeline
  └── No ↓

Is the task a routing problem (classify, send to the right specialist)?
  ├── Yes → Supervisor
  └── No ↓

Do subtasks have clean self-contained I/O contracts AND parallelism would reduce latency?
  ├── Yes → Orchestrator–Worker
  └── No ↓

Is the task iterative refinement with a grounded verifier (tests, rubric, reference)?
  ├── Yes → Debate / Critic-Proposer
  └── No ↓

Is the task so large that one orchestrator's context fills up just coordinating?
  ├── Yes → Hierarchical (cap depth at 2)
  └── No ↓

Does state accumulate from many contributors and need a shared source of truth?
  ├── Yes → Blackboard
  └── No ↓

Default: stay single-agent with better tools.
```

Three worked applications of the tree:

**(1) "Auto-generate marketing copy from a product spec" →** Sequence known (parse → draft → critique → revise → format)? Yes → **Pipeline**. Adding runtime routing would just be churn.

**(2) "Answer a research question using the web" →** Sequence known? No. Routing? No. Separable subtasks + parallelism valuable? Yes → **Orchestrator–Worker**. Exactly Anthropic's research system.

**(3) "Diagnose a production outage across services" →** Sequence known? No. Routing? No. Separable with clean contracts? Partially — different services give different signals, but hypotheses emerge from correlating them. Need a shared source of truth? Yes → **Blackboard** with service-specific investigator agents writing findings to structured sections.

The tree is not exhaustive; the patterns can compose (an orchestrator–worker whose workers each run a small pipeline). But if you can't place your problem on the tree, multi-agent probably isn't the right answer yet.

## Part 6: Communication Between Agents — Where Most Systems Die

The biggest single predictor of multi-agent success in production is not the pattern you choose but **how agents exchange information**. Free-text chat is the default, and the default is wrong.

### Rule 1: Structured Handoffs Or Bust

Every agent-to-agent handoff should be a **typed structured object**, not a chat transcript:

```json
{
  "from": "extractor",
  "to": "summarizer",
  "schema_version": "v2",
  "payload": {
    "entities": [{"name": "...", "type": "...", "confidence": 0.9}, ...],
    "quotes": [{"text": "...", "source_doc_id": "...", "page": 3}, ...],
    "notes": "..."
  }
}
```

The receiver reads the payload, validates against the schema, and proceeds. No reading of the sender's reasoning. No reinterpretation.

Why this matters: free-text handoffs fail in specific, recurring ways. The receiver re-interprets the sender's intent, drifting from the original task. Details get lost in summarization-of-summarization. Hallucinations at one stage are treated as facts at the next. Contracts are untestable because there's no schema. I've audited multi-agent systems where 60% of production errors were handoff misinterpretations, and every one of them disappeared after we moved to structured schemas.

**Case anti-pattern.** A team at a startup I know shipped a "research agent" with three agents: researcher, analyst, writer. The handoffs were free text. On manual inspection of failing outputs, we found: the researcher returned 4000 tokens of findings including asides, tangents, and contradictions; the analyst picked one plausible thread (not always the main one) and produced its analysis; the writer sometimes contradicted the researcher because it latched onto the analyst's framing. After switching to a structured handoff (researcher returns `{findings: [{claim, evidence, confidence}]}`, analyst returns `{conclusions: [{claim, supporting_findings, caveats}]}`), end-to-end accuracy improved by 31% on their internal eval, with no other changes.

### Rule 2: Shared State Is Schema'd And Central

If multiple agents need to see the same information, put it in a **blackboard** — a schema'd data structure — not a chat log. Each agent has a known read/write surface.

```
TaskBlackboard:
  goal: string                 ← immutable after creation
  plan: List[Subtask]          ← orchestrator-only write
  facts: List[Fact]            ← append-only, author-tagged
  decisions: List[Decision]    ← orchestrator-only write
  artifacts: Dict[str, ArtifactRef]  ← any agent can write
  current_state: StateEnum     ← controller-managed
```

This gives you:

- **Durable state** across agent invocations, decoupled from any one agent's context.
- **Testable schemas.** You can validate blackboard state against invariants (all decisions reference valid facts, every subtask has an owner, etc.).
- **Observability.** You can inspect the blackboard at any time and see exactly what the system knows.
- **Access control.** Not every agent can mutate every field. Writer-specific permissions prevent one agent from corrupting another's work.

### Rule 3: Narrow Each Agent's View

An agent should see:

- Its task (narrow, specific).
- The inputs needed for that task.
- Any retrieved context specifically relevant.

It should not see:

- The whole system's history (causes drift and excessive context).
- Other agents' reasoning traces (fosters conflation of their reasoning with its own).
- The full blackboard (irrelevant fields distract and inflate tokens).

Giving each agent narrow context reduces drift, reduces cost, and keeps reasoning focused. In the Anthropic research system, each subagent sees only its assigned sub-question and its own research; it does not see the lead's plan or other subagents' findings. This isolation is what lets the subagent run in a compact context and produce a focused summary.

### Rule 4: Communication Is Asynchronous-Capable By Design

Even if your current workload is synchronous, design so that agents can post-and-poll or be invoked by events. This enables parallelism, retries, durable execution, and long-horizon scheduling later. Systems that assume synchronous in-process communication require architectural rewrites to scale.

## Part 7: Task Decomposition And Assignment

### Decomposition: Who Decides The Plan?

Three approaches:

**Static routing (code-defined).** You hard-code the structure: "extractor first, then summarizer, then critic." Most production systems look like this underneath the AI-orchestration marketing. Its main advantage: predictability and testability. Its main limitation: it only handles problem shapes you anticipated.

**LLM-based routing.** A supervisor or orchestrator agent reads the task and decides which agent runs. Flexible; adds a new failure mode (the LLM's routing mistakes). Best used when the problem truly has runtime variability (e.g., customer support intent routing).

**Hybrid.** Static structure with LLM decisions at specific branch points. Usually the best of both: predictable where it matters, adaptive where it helps. For example, a pipeline that runs the same way except for a single "which specialist handles this?" decision at the start.

In my experience, teams over-use LLM routing when static would do. A rule of thumb: if you can describe the routing in three sentences of English, hard-code it.

### Assignment: Matching Task To Agent

Three mental models, useful in different stages of system maturity:

**Capability-based.** Each agent declares its capabilities (tools, domains, constraints). The orchestrator picks based on the match. Flexible but vague; breaks when capabilities overlap or are described ambiguously.

**Role-based.** Agents have fixed roles ("researcher," "writer," "critic"). Requests are routed by role. Simple; works well for small teams of specialized agents; rigid for edge cases.

**Contract-based.** Each agent has a declared input/output schema. The orchestrator picks the agent whose contract matches the current task. Scales better than role-based for complex systems with many agents.

Default to role-based until you outgrow it. Promote to contract-based when you find yourself inventing new roles frequently to accommodate slight variations.

## Part 8: The Failure Modes That Sink Multi-Agent Systems

These failure modes recur across multi-agent projects. If you're designing one, treat this as a pre-mortem checklist.

### Failure 1: Coordination Cost Exceeds Specialization Gain

You split a single agent into four agents. Token cost goes up 4×, latency goes up 2× (despite some parallelism), and the quality improvement is 5%. This is the most common and least acknowledged multi-agent failure mode.

The specific diagnostic: always maintain a **single-agent baseline**. Run both architectures on the same eval. If multi-agent's win is less than 2× its cost premium, it's not worth it. Re-invest the effort into better tools or better prompts for the single agent.

### Failure 2: Information Loss At Handoffs

Each handoff compresses context. By the 4th handoff, the final agent is acting on a distorted summary of the task. Symptom: the final output drifts from the user's original intent, in ways no individual agent would cause on its own.

Fix: anchor the **original task and any critical constraints** as visible to every agent, not just the first. A dedicated field in every handoff — `original_intent: string` or `hard_constraints: List[Constraint]` — goes a long way. The Anthropic research system pins the user's original question to every subagent's prompt for exactly this reason.

### Failure 3: Cascading Errors

Agent A makes a small mistake. Agent B compounds it into a medium mistake. Agent C builds a plan on the compounded mistake. The final error is huge; the first error was small and invisible in the logs.

Fix: validation at every handoff boundary. Agent B has a validator for its input: does the payload satisfy its schema? Does it satisfy business invariants (e.g., "sum of items equals declared total")? If validation fails, B refuses and signals the orchestrator. Better to halt early than to build on a rotten foundation.

### Failure 4: Inter-Agent Loops

Agent A asks B for X. B doesn't have what it needs, asks A to clarify. A restates the request. B asks again. The individual interactions all look reasonable; the system is stuck.

Fix: **orchestrator-level budgets** — total hops across the task, total tokens, wall time. Detect "we took K steps without making progress on the blackboard" and escalate or terminate. These budgets are non-negotiable for any production multi-agent system.

### Failure 5: Shared State Drift

Multiple agents writing to the same state without contracts produces a blackboard that's internally contradictory. Decision X references fact Y that doesn't exist; two agents write conflicting values to the same field without noticing. No single agent detects the contradiction; the system as a whole is broken.

Fix: schema'd state with invariants, append-only where possible, explicit reconciliation where not, every write tagged with its author and timestamp.

### Failure 6: Ambiguous Termination

When does the multi-agent system stop? If the answer isn't obvious, it's a bug. Systems without explicit termination keep running, cost money, and produce confused final outputs.

Fix: explicit termination conditions, owned by the orchestrator. Examples: "all subtasks in the plan are marked complete," "a dedicated completion agent signals done," "we hit the budget," "the user confirms." Never rely on "emergent" termination.

### Failure 7: Tool Permission Proliferation

If every agent has every tool, the attack surface for prompt injection, error, and unintended behavior is multiplied. A bug or injection in agent C can affect systems only agent B should touch.

Fix: **least-privilege per agent**. Each agent's tool set is declared explicitly and enforced at the tool-permission layer. This is one of the biggest security wins in multi-agent design, and it costs nothing except discipline.

### Failure 8: Observability Collapse

A single-agent trace is hard to debug. A multi-agent trace, without discipline, is near-impossible. If you can't answer "which agent did what, when, with what input and what output?" you can't debug production.

Fix: structured logging per agent invocation with correlation IDs, a unified trace viewer, and payload captures at every handoff. Without this, production debugging becomes speculation.

### Failure 9: The Scaling Trap

A multi-agent system that works at 10 requests per hour breaks at 1000 because coordination overhead doesn't scale linearly. Contention on shared state, bottlenecks at the orchestrator, token costs compounding — all of these hit at volume.

Fix: partition workload by task ID so each task's multi-agent session is independent. Orchestrators are per-task, not global. Shared state is per-task. Scale horizontally at the task level.

### Migration Paths — Evolving Multi-Agent Architectures

Architectures should evolve as the product does. Three common migration paths, with the signals that trigger them and the trade-offs of migrating vs rewriting.

**Pipeline → Orchestrator–Worker.** Trigger: the fixed stage sequence no longer fits — you're adding runtime `if` branches in the pipeline, and each branch is its own mini-pipeline. Migration steps: promote the branch decision to an orchestrator call, keep existing pipeline stages as workers, introduce a DAG representation for the plan. Trade-off: you now pay for planning on every request; worth it when >30% of requests need non-default branching. Alternative: stay pipeline and push routing into a pre-stage classifier — cheaper, limited to binary-ish decisions.

**Flat Orchestrator–Worker → Hierarchical.** Trigger: the orchestrator's context is filling up with routing/coordination overhead and its planning quality is degrading (measurable: per-task success drops as subtask count grows). Migration steps: split the orchestrator into a top-level planner and mid-level orchestrators per phase, cap total depth at 2. Trade-off: latency increases by one planning round per level, observability becomes harder (nested traces), and per-level planning cost compounds. Worth it when subtask counts exceed ~15 or coordination overhead is measurably degrading the top-level agent.

**Peer-chat → Structured blackboard.** Trigger: debuggability has collapsed; you can't answer "why did the system do X" without reading transcripts end-to-end. Migration steps: introduce a schema for the shared state, freeze the current peer message types into typed payloads, add a controller that decides the next agent based on blackboard state. Trade-off: loss of "emergent" behaviors (some of which were working) in exchange for debuggability and invariant checking. Worth it whenever the system has to ship to production; peer chat is only defensible in research.

General principle: **migration is usually cheaper than rewrite**. Plan the schema and state boundaries when you first build — even a pipeline benefits from typed payloads — and the upgrade to orchestrator–worker or hierarchical becomes a refactor rather than a rewrite. If you skipped typing, the migration cost exceeds the rewrite cost, and you end up rewriting anyway.

## Part 9: Evaluation For Multi-Agent Systems

Evaluation in multi-agent is the sum of single-agent eval concerns plus the multi-agent-specific ones.

### Per-Agent Evals

Each agent has a contract. You unit-test it. Given input X (sampled from realistic distribution), produce a valid Y that meets criteria Z. Run these in CI; catch regressions early, before they compound in end-to-end runs.

For stochastic agents, run each test case multiple times and report distributions (pass rate, variance) rather than single runs.

### Handoff Evals

At every agent boundary, assert the handoff:

- **Schema conformance.** Is the output well-formed?
- **Factual support.** Are claims in the output actually supported by claims in the input? (LLM-as-judge works here.)
- **Completeness.** Did the output preserve all critical elements of the input, or was something dropped?
- **No injection.** Does the output contain content that would be dangerous for the next agent to parse as instructions?

Handoff evals catch information-loss and compounding-error bugs that pure end-to-end evals miss.

### End-To-End Task Success

The final answer. Grade subjective tasks with rubrics, objective ones with machine checks. Report success rate, cost, and latency together — a system that gets 95% right at 10× the cost of a 90% alternative is usually not worth it.

### Coordination Metrics

Metrics that only exist because you're multi-agent:

- **Hop count distribution per task.** A well-behaved system has a bounded, stable distribution. Creeping hop count means drift or loops.
- **Inter-agent retry rate.** High retries indicate poor handoff contracts or brittle agents.
- **Replanning rate.** How often the orchestrator rewrites the plan. Too high = planning prompts need work.
- **Single-agent-simulated performance on the same tasks.** The ultimate sanity check: what happens if you collapse to one agent? This is not optional. It's the only way to prove multi-agent is earning its cost.

### Case Study: How Anthropic Evaluated Its Research System

From the public engineering post, Anthropic's team described their evaluation approach for the multi-agent research system in revealing detail:

- They built a **rubric-based scoring system** for research quality (accuracy, completeness, depth of investigation, citation quality).
- They used **Claude-as-judge** to score runs, calibrated against human evaluations on a sample. They explicitly measured and reported inter-rater agreement between the model judge and humans.
- They ran the **same research queries** through a single-agent baseline and the multi-agent system, and reported the delta (the 90.2% improvement number).
- They tracked **cost multiplier** (15× tokens) as a first-class metric alongside quality, so the tradeoff was explicit.
- They ran **iterative eval-driven development** — every prompt change was evaluated against the full suite before shipping.

The lesson: the evaluation system was itself substantial engineering, comparable in complexity to the multi-agent system it was evaluating. If you're serious about multi-agent in production, budget for that.

## Part 10: Observability And Debugging

### The Unified Task Trace

The non-negotiable observability artifact for multi-agent systems is a **unified task trace** showing, per task:

- Every agent invocation (which agent, with what input, producing what output).
- Every tool call within each agent.
- Every blackboard state change, with author and timestamp.
- Every handoff (from, to, payload, schema version).
- Wall-clock timing, token counts, and cost per step.

Modern tracing frameworks (OpenTelemetry with LLM extensions) capture this naturally if you instrument it. Most production multi-agent systems have purpose-built trace viewers on top that let engineers navigate a task's execution like stepping through a debugger.

### Debugging By Failure Category

A taxonomy I've found useful:

- **Wrong final answer but all agents "succeeded."** Almost always a handoff issue — information lost between agents. Inspect the shrinking payload hop by hop; find the first point where critical content disappeared.
- **Specific agent keeps failing.** Isolate and reproduce. Run that agent with the observed input in isolation. Debug as a single-agent problem.
- **Nondeterministic failure.** Usually either non-seeded sampling in one agent, race conditions in shared state, or subtle ordering dependencies. Pin seeds for reproduction; audit state-write ordering.
- **Creeping cost.** Look at hop count distribution over time. A well-behaved system has bounded hops; creeping cost means unbounded loops or plan drift. The orchestrator's planning prompt is usually the culprit.
- **Degrading quality after a prompt change.** Run the new prompts against the full eval suite. Regression in specific agents often cascades in surprising ways to end-to-end quality.

### The Replay Requirement

Your system must support **replaying** any failed trace by re-feeding recorded agent inputs and comparing outputs. Without replay, nondeterministic production bugs are basically undebuggable. Implementation-wise: record every agent's prompt, tool-result, and model response. Store hashes or full content depending on privacy constraints. Build a "replay agent X at step Y of task Z" command into your admin surface.

## Part 11: Framework Landscape (As Of Early 2026)

The specific libraries change fast; the abstractions they push are more durable. A brief field guide:

- **LangGraph.** Explicit graph of nodes with shared state. Good fit for orchestrator-worker and pipeline patterns. Forces you to be explicit about state and transitions, which is good discipline. Overhead of graph construction can be awkward for very dynamic systems.
- **AutoGen (Microsoft).** Conversation-based multi-agent. Powerful for debate and peer patterns; less opinionated about structure, which cuts both ways — easy to prototype, easy to build something unshippable.
- **CrewAI.** Role-based multi-agent with strong ergonomics for orchestrator-worker. Popular for research-style tasks; production reports are mixed — some teams ship, others find the abstractions too constraining.
- **OpenAI Agents SDK.** Multi-agent primitives with handoff as a first-class concept. Well-integrated with tool calling and structured outputs in the OpenAI ecosystem.
- **Anthropic's Claude Agent SDK.** Leans toward single-agent-with-tools architectures, consistent with Anthropic's philosophy in Claude Code. Supports sub-agents but as a secondary pattern.
- **Custom on top of a durable workflow engine (Temporal, Prefect, Restate).** Where most serious production systems end up. You lose some agent-specific ergonomics but gain replay, durability, observability, and operational maturity from a battle-tested engine.

A useful heuristic for picking a framework:

- **Prototyping, exploring the pattern:** LangGraph, CrewAI, or the Agents SDK. Low friction, quick iteration.
- **Production-critical, long-running, durability required:** durable workflow engine plus a thin agent layer you own. The ergonomics loss is worth the operational maturity.
- **Research or complex debate scenarios:** AutoGen, if you can tolerate its looseness.

Resist frameworks that push free-text peer chat as the primary abstraction for production unless your problem genuinely matches (rare).

### Framework Trade-off Matrix — Specific Scenarios

Concrete scenarios and the framework that tends to fit each in practice:

| Scenario | First choice | Why | What to watch for |
| --- | --- | --- | --- |
| Quick prototype, exploring pattern viability | LangGraph or Agents SDK | Fast iteration; explicit graph forces early discipline | Easy to ship a prototype and call it production — resist |
| Production with durability, replay, long-running | Custom thin layer over Temporal (or similar) | Durability and replay are core; agent ergonomics are secondary | Agent-specific tooling (tracing, cost attribution) you'll have to build yourself |
| Research / debate scenarios with many roles | AutoGen | Conversation-native; flexible for multi-party roles | Structure discipline is on you — easy to produce un-shippable systems |
| Enterprise compliance (audit logs, per-tenant isolation, PII) | Custom on durable workflow engine | Compliance hooks typically require your own layer | Months of work before value; don't pick this unless compliance is truly required |
| Mostly orchestrator-worker, modest scale | CrewAI or Agents SDK | Role-based ergonomics fit the pattern | Some teams hit constraints when the pattern doesn't quite fit — plan for migration |
| Task graph with many conditional branches | LangGraph | Explicit state machine handles branching clearly | Graph definition verbosity for very dynamic systems |

The meta-point: the framework question is usually *second*. First pick the architecture pattern (orchestrator–worker, pipeline, etc.) from the decision tree; then pick the framework that best supports that pattern plus your operational requirements.

Trade-offs named explicitly:
- **Pick for demo ergonomics** → fast ship, painful when production hits. Most common failure mode.
- **Pick for operational maturity (Temporal-based)** → slow ship, painful prototype iteration.
- **Pick for "standard in the industry"** → often means picking for team hireability over fit for your problem. Sometimes right (onboarding matters); often not (your system has its specific shape).

Concrete escape hatch: design your agent logic to be framework-portable (structured payloads, explicit state object, no framework-specific primitives leaking into business logic). Most teams end up re-platforming once — make that a refactor, not a rewrite.

## Part 12: Three Deep Case Study Dives

Three case studies, analyzed at architectural depth, to ground the patterns above in reality.

### Deep Dive: Anthropic's Multi-Agent Research System

We introduced this above. Here are the architectural and operational details that are most instructive.

**The system's specific architecture:**

- A **lead researcher** that receives the user's query. It can see the full query and any attached context. Its prompt is oriented around planning: decompose the question, identify what needs to be investigated, dispatch work.
- **3–5 parallel subagents** per research round. Each gets a focused research directive and access to web search, browsing, and retrieval tools. Each runs in its own context window.
- Subagents return **structured findings**: `{findings: [{claim: str, evidence: [source_refs], confidence: float}], gaps: [str]}`.
- The lead **synthesizes** findings across subagents, identifies what's still missing, and either dispatches another round or moves to composition.
- A **citation agent** in the final stage verifies every factual claim against sources in the evidence bundle and attaches citations. This agent is essentially a fact-checker.

**Key engineering decisions:**

- **Parallel tool calls within each subagent.** Subagents call multiple web-search or fetch operations in parallel, not sequentially. This was reported to 2–3× agent speed.
- **Strict output schemas.** Subagents can't just return prose; they return structured findings that the lead can reason about without re-parsing free text.
- **No subagent-to-subagent communication.** Everything goes through the lead. This prevents the drift and redundancy of peer networks.
- **Context isolation.** The lead's context stays clean because subagents absorb the verbose research content. This is the "context-bounding" signal in action.
- **Iteration in the lead, not in subagents.** The lead decides "we need another round." Subagents don't iterate; they run once and return.

**Operational lessons the team called out:**

- Prompt engineering was the single biggest lever. They iterated prompts extensively, both for the lead and for subagents.
- Evaluation was the second biggest lever. The quality rubric and LLM-judge calibration took substantial engineering.
- Cost had to be reasoned about explicitly. 15× tokens is only acceptable if the output is genuinely 15× more valuable. For research, it can be; for many other tasks, it isn't.
- Error modes were specific and required specific fixes. Subagents would sometimes go off-topic; the fix was tighter prompts. The lead would sometimes under-plan; the fix was a more structured decomposition prompt.

**When this architecture is the right answer:** when you have a task where the top-level reasoning benefits from global plan + parallel specialized investigation + final synthesis, and you have a budget for 10–20× the single-agent token cost. Research, investigation, due-diligence tasks all fit.

**When it's the wrong answer:** when the task requires tight coherence (coding across files), when global context outweighs parallelism (debugging), when the budget is tight and quality gains would need to be dramatic to justify cost.

### Deep Dive: Devin And The Autonomous Coding Agent Landscape

Devin was announced by Cognition Labs in March 2024 as "the first AI software engineer." The announcement demos showed Devin taking a task description, planning, writing code, running tests, and deploying — all autonomously. It became the flashpoint for a debate about whether coding agents should be autonomous or supervised.

**The architecture (reconstructed from public information):**

- A **top-level planner** that takes the task and produces a multi-step plan.
- A **main execution loop** (effectively an orchestrator) that works through the plan.
- **Specialized tools / sub-capabilities**: shell, file editor, browser for documentation, test runner, git operations.
- **Durable task state**: the plan, progress so far, encountered errors, intermediate artifacts.
- **Self-correction loops**: when a step fails, a reflection step tries to diagnose and revise.

Whether this is "multi-agent" depends on your definition. The sub-capabilities are LLM-driven in places but function more like tools than autonomous agents. The "planner" and "executor" are arguably two agents; Cognition has described them as modes of one reasoner.

**What worked in public usage:**

- Small, well-specified bug fixes in familiar codebases.
- Tasks with **clear success criteria** (tests, type checks).
- **Exploration tasks** where the agent's thoroughness compensated for some imprecision.

**What struggled:**

- Large refactors where codebase-wide judgment was needed.
- Ambiguous requirements where the agent's interpretation drifted from the user's intent.
- Long sessions where the agent would occasionally get stuck in unproductive loops.
- Situations where the agent's generated code had subtle bugs that passed tests but broke edge cases.

**The broader lesson about autonomous coding.** The initial industry reaction to Devin was polarized: some teams raced to build similar autonomous systems; others (including Anthropic with Claude Code) deliberately stuck with human-in-the-loop architectures. By 2025–2026, the mainstream view converged: **autonomous coding works in narrow verticals with strong verification (tests, lints, human review), and is unreliable as a general coding partner.** Claude Code's human-in-the-loop design, where the model proposes and the human approves, has become the more commonly shipped pattern. Devin's successors have moved toward this model as well.

The architectural takeaway: **reliability in coding agents is dominated by the verifier, not the planner**. An agent with a great planner and a weak verifier produces confident wrong code. An agent with an adequate planner and a strong verifier (tests, humans, static analysis) produces correct code eventually. Invest in verification before autonomy.

### Deep Dive: MetaGPT And ChatDev — The Role-Based Software Company

MetaGPT (2023) and ChatDev (2023, Tsinghua) both simulate a software company with role-based multi-agent: product manager writes requirements, architect designs, engineer codes, QA tests. Both use LLMs as the backbone for each role.

**The architecture (MetaGPT example):**

- Roles: Product Manager, Architect, Project Manager, Engineer, QA, Data Analyst.
- Each role has a **prompt template** defining its persona, responsibilities, and SOP (standard operating procedure).
- Handoffs follow a **fixed SOP**: user → PM → Architect → PM → Engineer → QA. Each role produces a typed artifact (PRD, design doc, code, test report).
- The SOP constrains free-form communication — in effect, the system is a pipeline with role-annotated stages.

**Why it works on simple tasks:**

- Small games, basic web apps, simple algorithms are within the complexity ceiling of LLMs in 2023–2024.
- The SOP imposes enough structure that the agents' coordination doesn't collapse.
- Each role is narrow enough that the prompt engineering can specialize it.

**Why it fails on realistic tasks:**

- Real software has cross-cutting concerns (security, performance, backwards compatibility) that don't map cleanly to roles.
- Cross-role negotiation in free text (e.g., "architect says use Postgres, engineer thinks MongoDB is better") drifts under the same conditions that sink any free-text multi-agent system.
- The verifier (QA) is typically another LLM without access to a real test environment. Without verified test execution, "QA approved" means only that the QA prompt generated approval text.
- Complexity beyond toy scale exposes the gap between "generate plausible-looking code" and "generate correct working system."

**The architectural lesson:**

- Role-based multi-agent works when the roles have genuine informational asymmetries (different tool access, different specialized prompts) and when coordination is constrained by SOPs with structured artifacts.
- It does not work as a general simulation of a team. Real teams rely on shared mental models, unspoken conventions, and high-bandwidth communication that free-text handoffs do not reproduce.

MetaGPT and ChatDev remain valuable academic artifacts: they demonstrated empirically, at scale, that peer-agent with minimal structure collapses, and that imposing SOPs recovers some of the lost ground. Production systems have taken the lesson and built structured orchestrator patterns, leaving the "simulated company" framing as demo material.

## Part 13: A Reference Architecture For Your Own Multi-Agent System

A reasonable default for most real-world multi-agent systems looks like this:

```
                 ┌───────────────────────┐
                 │   Task Intake + API    │
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │    Orchestrator        │  ◀── strong model;
                 │    (planning + dispatch)│     decomposes, dispatches,
                 └─┬────┬────┬───────────┘     replans, terminates
                   │    │    │
          ┌────────▼┐ ┌─▼──┐ ┌▼────────┐
          │Worker A │ │B   │ │Worker C │  ◀── narrow task,
          │(tools   │ │    │ │         │     typed I/O,
          │ subset) │ │    │ │         │     scoped tools
          └─────────┘ └────┘ └─────────┘
                   │    │    │
                   ▼    ▼    ▼
          ┌────────────────────────┐
          │    BLACKBOARD / STATE  │  ◀── structured schema,
          │  (durable, schema'd)   │     versioned writes,
          └────────────┬───────────┘     visible to ops
                       │
          ┌────────────▼──────────┐
          │   Verifier / Validator │  ◀── validates handoff payloads
          └────────────┬──────────┘     against schema + invariants
                       │
                       ▼
                 Final output

  Orthogonal, always-on:
    - Per-agent tracing with correlation IDs
    - Cost tracking per step
    - Safety layer (content, injection, tool-permission enforcement)
    - HITL approval gates for sensitive actions
    - Replay from any trace
```

Most production systems fit this template with specific patterns chosen for their worker layer (pipeline vs. parallel dispatch vs. hierarchical).

## Part 14: Interview Questions With Detailed Analysis

The format follows the other articles in this cluster: the question, what the interviewer is really testing, a strong answer shape with detail, and the follow-ups that usually come. These are written for staff-level agent-systems interviews where generic answers fail.

### Q1: "When Would You Use Multi-Agent Instead Of Single-Agent With Tools?"

What's being tested: whether you have an honest, specific threshold or whether you default to multi-agent enthusiasm.

Strong answer: start with the default — single-agent with good tools. Switch only when a specific signal applies: separable subtasks with clean contracts, measurable specialization gains, parallelism that meaningfully reduces wall time, context-bounding as a first-class goal, or adversarial diversity (critic-proposer) with a grounded verifier. Every agent boundary adds coordination cost and new failure surface. Multi-agent earns its keep only when the measured quality gain vs. single-agent exceeds the cost premium.

Follow-up: "Give an example where multi-agent clearly wins." Research tasks of the kind Anthropic's research multi-agent system handles — decomposable sub-questions, parallel investigation, final synthesis.

Follow-up: "Give an example where it doesn't." Coding tasks that require coherent codebase-wide reasoning. Claude Code's deliberate single-agent choice illustrates this: splitting would lose the mental model more than specialization would help.

Follow-up: "What's the most common mistake in the decision?" Using multi-agent when the task is really a pipeline — adding runtime orchestration complexity where a fixed sequence of stages would do.

### Q2: "Walk Me Through The Main Multi-Agent Architectures And When To Pick Each."

What's being tested: breadth of pattern fluency.

Strong answer: seven canonical patterns. Orchestrator-worker (central planner, parallel workers, structured results) — default for separable subtasks. Supervisor (routing-only hub, specialized backends) — for routing-shaped problems like customer support. Pipeline (fixed sequence of stages) — for known transforms where predictability matters. Debate / critic-proposer (two or more agents in adversarial cycle) — for quality-critical tasks where a grounded verifier exists. Hierarchical (orchestrator of orchestrators) — for very large tasks; cap depth at 2 in production. Peer / swarm — almost never in production; useful academically. Blackboard (shared structured state, controller) — for incremental assembly and investigation.

Pick based on: structure of the problem (fixed vs. variable sequence), need for parallelism, control-vs-autonomy tradeoff, presence of a verifier (for debate), and operational constraints. Default to orchestrator-worker or pipeline.

Follow-up: "When does a pipeline beat an orchestrator-worker?" When the stages are fully predictable and don't need dynamic routing. Perplexity's search pipeline is a better fit than a dynamic orchestrator for known stages.

Follow-up: "When does the supervisor pattern struggle?" When the domain partitions unclearly, when the supervisor's classifier is noisy, or when specialists have overlapping responsibilities. Symptoms: high misroute rate, frequent human escalation for miscategorized issues.

### Q3: "How Do Agents Communicate? What Goes Wrong With Free-Text Handoffs?"

What's being tested: have you shipped one of these for real.

Strong answer: structured typed handoffs, not free text. Shared state as a schema'd blackboard, not a global chat log. Agents see only the narrow context they need. Free-text handoffs cause drift at each hop, information loss through summarization, untestable contracts, and hard-to-debug failures. Structured schemas (JSON, Pydantic) catch most multi-agent bugs at the boundary.

Follow-up: "How do you migrate an existing free-text system?" Introduce a schema at each boundary, validate both sides, start failing loudly on contract violations, then iterate. Concrete path: pick the boundary with the highest observed drift, schema it first, measure the improvement.

Follow-up: "What content belongs in the handoff payload vs. the shared blackboard?" Payload is the specific material for the next step (targeted input). Blackboard is global state that many agents need to read (goal, plan, accumulated facts). If two agents need to coordinate on state, blackboard. If it's a linear hand-off, payload.

### Q4: "You Have A Five-Agent System Where The Final Answer Drifts From The Original Intent. What's Going Wrong And How Do You Fix It?"

What's being tested: classic-failure-mode diagnosis.

Strong answer: this is almost certainly handoff information loss. Each agent compresses context; the original intent gets washed out over successive compressions. Fix in three parts: pin the original goal and critical constraints as a fixed anchor visible to every agent (not just the first); make handoffs structured and minimal so the receiver can't paraphrase ambiguously; add a final-stage verifier that checks the output against the original goal before producing the final answer.

Follow-up: "If the anchor is already pinned and it still drifts, what next?" Inspect specific handoff payloads hop by hop. Find the first stage where critical content is missing or distorted. The bug is at that stage's schema or prompt. Fix localized, not globally.

Follow-up: "What if the drift is in one agent specifically?" Isolate and reproduce: run that agent with the observed input. Debug as a single-agent problem. Common fixes: tighten its prompt, constrain its output schema, reduce its context.

### Q5: "How Would You Handle Cascading Errors In A Multi-Agent System?"

What's being tested: understanding that multi-agent compounds reliability problems.

Strong answer: validate every agent's output against its contract at the handoff boundary. Refuse invalid inputs at the receiver side; escalate back to the orchestrator. The orchestrator detects cascading errors via repeated validator failures and replans or escalates. Maintain per-agent error taxonomies (transient, misuse, environmental, semantic) and handle each appropriately. Add verification at key milestones, not only at the final output. And always have orchestrator-level budgets (hops, tokens, time) that halt the system before errors compound without bound.

Follow-up: "How do you avoid infinite retry loops across agents?" Global hop and cost budget; if the same handoff fails K times, escalate or terminate. Also distinguish "retrying because transient" from "retrying because stuck" — the latter means a structural problem in the agent or its inputs.

Follow-up: "What's the hardest cascading error to catch?" A small semantic error early (e.g., misidentifying an entity) that every downstream agent treats as fact. The fix is post-step verification against ground truth when available; when not, a dedicated audit agent at the final stage.

### Q6: "Design A Multi-Agent System For Code Review."

What's being tested: end-to-end design muscle.

Strong answer: orchestrator-worker pattern. The orchestrator receives the PR and its diff, decomposes into specialized reviews, dispatches workers in parallel, collates comments, produces a final review.

Workers (parallel, scoped tools):
- **Static analyzer agent**: runs linters, type checkers, style tools; emits structured findings.
- **Semantic reviewer agent**: reads the diff with context from the broader codebase, evaluates logic, readability, correctness; emits structured comments.
- **Security agent**: runs SAST tools, looks for known patterns; emits findings with severity.
- **Test coverage agent**: checks test additions against the diff; flags gaps.

Each worker returns `{comments: [{file, line, severity, category, message, suggested_fix?}]}`. The **collator agent** deduplicates, resolves conflicts, ranks by severity, and produces the final review.

Blackboard holds the diff, file context (cached, reusable), accumulating comments by author, and final verdict.

Follow-up: "What if two agents flag conflicting things?" Route conflicts to a dedicated resolver, or surface to the human as a "reviewer disagreement" explicitly. Never silently drop one side.

Follow-up: "Would you use pipeline or orchestrator-worker here?" Orchestrator-worker because the workers are independent and parallel, and the orchestrator benefits from dynamic decisions (e.g., skip security agent if no relevant file types changed).

Follow-up: "What's the single biggest reliability risk?" Workers hallucinating comments that aren't supported by the diff. Mitigation: every comment must reference a specific file/line in the diff, validated by the collator against the actual diff hunks.

### Q7: "How Do You Evaluate A Multi-Agent System?"

What's being tested: depth of evaluation thinking.

Strong answer: layered evaluation. Per-agent unit tests (contract tests in CI). Per-handoff validation (schema + information preservation). End-to-end task success on a golden task suite. Coordination metrics (hops, replans, retries). And critically, a single-agent baseline running on the same tasks to prove multi-agent is earning its cost. Production canary and replay capability round out the stack.

For subjective tasks (research, writing, review), use LLM-as-judge with rubrics, calibrated against human evaluators. Report inter-rater agreement. Iterate the rubric.

Follow-up: "What's the single most important metric?" The delta between multi-agent and single-agent baseline on your eval, paired with the cost multiplier. If single-agent is close at a fraction of cost, you don't need multi-agent.

Follow-up: "How do you catch regressions from prompt changes?" Full-suite eval before every ship. Track per-agent metrics, not just end-to-end — a regression in one agent can cascade in unexpected ways and be easier to localize with per-agent tracking.

### Q8: "What Are The Common Anti-Patterns In Multi-Agent Design?"

What's being tested: experience with real deployments.

Strong answer: free-text peer chat as the coordination medium. Shared context without schema (global chat log). Too many agents with overlapping roles. Every agent has every tool (permission proliferation). No explicit termination condition. No single-agent baseline to justify the architecture. Hierarchies deeper than 2 levels. LLM-based routing where static routing would suffice. Handoffs without validators. Shared state without invariants. No per-agent observability.

Follow-up: "How do you prevent these when starting a new project?" Start single-agent; promote only when forced by signals; enforce structured handoffs from day one; write schemas before prompts; maintain a single-agent baseline in eval from the start.

Follow-up: "Which anti-pattern causes the most damage in production?" Free-text handoffs combined with no validators. Errors compound silently; observability is poor; debugging is guesswork. Fix this first if you find it.

### Q9: "How Do You Share State Between Agents Without It Becoming A Mess?"

What's being tested: data-modeling for multi-agent.

Strong answer: structured blackboard with a schema. Append-only fields where possible (facts, events, findings), mutable with care (task state, decisions, current phase). Every write tagged with agent ID, step ID, and timestamp. Each agent has declared read and write scopes — not access to everything. Reconciliation is explicit where needed; never rely on "the last writer wins." Invariants on the blackboard are validated; violations are errors, not warnings.

Follow-up: "How do you handle a fact two agents disagree on?" Record both, with sources and confidences. A resolver agent (or human escalation) picks the winner. Never auto-merge silently; that's how subtle bugs enter the system.

Follow-up: "What prevents schema sprawl?" Versioned schemas with deprecation deadlines. Every schema change goes through review. Fields are added conservatively; unused fields are removed promptly.

### Q10: "How Do You Prevent One Agent's Prompt Injection From Affecting The Whole System?"

What's being tested: security thinking for multi-agent.

Strong answer: each agent has a least-privilege tool set, enforced at the permission layer (not just in prompts). Cross-agent communication is structured payloads, not raw text that could carry injected instructions. Every handoff has a validator; payloads with instruction-like content get flagged. High-stakes actions go through a separate action filter regardless of which agent initiates them. Blackboard state is schema-validated. Credentials never enter any agent's prompt — they're accessed via a vault-backed proxy at the tool layer.

Follow-up: "Give an example of an attack that naive multi-agent would miss." Adversarial content reaches agent A (treated as data). A produces a free-text handoff summary that includes the injection payload. Agent B parses that handoff and treats it as instructions, calls a tool accordingly. Structured typed handoffs plus action filters prevent this.

Follow-up: "What's harder to defend: direct or indirect injection?" Indirect. The adversary doesn't talk to your system; content they authored flows through a tool (email, document, web page) into your agents. Defense requires tagging content by trust level throughout the system and bounding what low-trust content can cause.

### Q11: "How Do You Debug A Multi-Agent System When Something Goes Wrong?"

What's being tested: operational maturity.

Strong answer: start with the unified task trace. Walk through each agent's input and output. Identify the first step where the actual behavior diverges from expected. If the divergence is at a handoff, it's likely payload schema or information loss. If it's within an agent, isolate and reproduce that agent's step with its observed input. If it's at the blackboard, audit writes for ordering issues or schema violations. Use replay capability to re-run specific steps with the recorded inputs. Escalate to single-agent-isolated debugging once the failing step is identified.

Follow-up: "What's the single most useful debugging artifact?" The full payload of every handoff. Most multi-agent bugs are visible there if you can read them.

Follow-up: "What if the bug is intermittent?" Pin seeds. Record nondeterministic steps. Replay with seeds. If it's a race in shared state, audit state-write ordering; impose ordering if needed.

### Q12: "Orchestrator-Worker Versus Hierarchical — When Do You Promote To Hierarchy?"

What's being tested: sense of scale.

Strong answer: default is flat orchestrator-worker. Promote to hierarchy when: the number of parallel subtasks is large enough that one orchestrator is overloaded; subtasks have natural further decomposition; or orchestrator context fills up with routing/coordination overhead that crowds out its ability to reason. Cap depth at 2 in production unless there's a specific reason (and the specific reason should be measurable, not aesthetic). Every added layer costs latency, tokens, and debuggability.

Follow-up: "What's 'orchestrator overload'?" When context grows too large for the orchestrator to reason well, when routing becomes unreliable because there are too many destinations, or when coordination logic in the orchestrator becomes a noticeable bottleneck. Mid-level orchestrators let each stay focused.

Follow-up: "How do you debug a hierarchical system?" The trace has to be hierarchical too. Visualization matters: flame-graph style views of nested orchestrators beat flat traces.

### Q13: "How Do You Decide Task Assignment At Runtime?"

What's being tested: routing design.

Strong answer: start with static routing — hard-code where behavior is predictable. LLM-based routing only at branch points where the decision genuinely benefits from reasoning. Capability-, role-, or contract-based as maturity grows; start with role-based. Observability on routing decisions: log every decision and outcome so you can measure router quality. Add rules for patterns the LLM router gets wrong; don't just retrain endlessly.

Follow-up: "How do you improve routing over time?" Log every routing decision with the outcome. Periodically analyze: where is the router systematically wrong? Add rule-based overrides for those cases. Re-prompt or fine-tune the router for the rest. Measure improvement before shipping.

Follow-up: "What's a classic bug in LLM-based routing?" Overconfident classification. The router returns a class label without hesitation on inputs it should flag as uncertain. Fix: calibration, multi-sample voting, or a confidence threshold below which the default fallback is used.

### Q14: "Walk Me Through Picking A Multi-Agent Framework."

What's being tested: informed opinion.

Strong answer: consider the pattern you need (pipeline, orchestrator-worker, debate). Evaluate the framework's state-management primitives (structured, durable, testable?). Check its replay and observability primitives. Check integration with your existing infrastructure, especially durable workflow engines you might need later. Avoid frameworks that push free-text peer chat as the primary abstraction, unless your problem truly matches. At scale, most teams end up on a custom layer over a durable workflow engine — pick frameworks that don't lock you out of that migration path.

Follow-up: "Starting today, no existing infra, orchestrator-worker pattern — what do you pick?" LangGraph or the Agents SDK for prototyping. Plan to re-platform onto a durable workflow engine when production requires durability, replay, and strong operational maturity.

Follow-up: "What's the biggest framework-selection mistake?" Picking based on demo ergonomics rather than operational primitives. Looks great at prototype; pain in production. Weigh observability, replay, and state management heavily.

### Q15: "Your Multi-Agent System Is 30% Slower And 3× More Expensive Than Single-Agent, And Quality Is The Same. Your Team Wants To 'Add More Agents To Improve Quality.' What Do You Do?"

What's being tested: engineering judgment under team pressure.

Strong answer: push back with data. Multi-agent is not earning its cost. Diagnose first: is the single-agent missing a specific capability multi-agent provides? If yes, add that capability to the single-agent (better tool, better prompt, better context) — cheaper and more reliable than more agents. If multi-agent really is the right shape, the problem is not "more agents" — it's better handoffs, better specialization, better state, or better verification. Adding agents without diagnosis makes it worse.

Establish a measurement protocol: no new agent ships without a measured quality gain vs. cost and latency cost. Commit to rolling back if the gain doesn't materialize.

Follow-up: "What if leadership insists?" Negotiate a time-bounded pilot with defined success metrics and a rollback plan. Measure rigorously. Let the data win or lose the argument. If it loses, don't ship. If it wins, ship. This is more defensible than dogma in either direction.

Follow-up: "What's the most common misdiagnosis in this situation?" Confusing "the system doesn't feel smart" with "we need more agents." Often the real issue is weak evaluation: the system is actually fine on objective measures but feels weak because users are hitting specific edge cases. Fix: improve evals to surface those edge cases, fix them directly.

## Closing Thoughts

The honest summary of multi-agent systems is:

Most production systems don't need it. Get better tools, state, and context engineering for a single agent first. When you do need multi-agent, be ruthlessly structured about it: typed handoffs, schema'd shared state, narrow agent scopes, explicit budgets, per-agent observability, validators at every boundary. The patterns that work are the ones that impose structure; the patterns that fail are the ones that leave coordination to emergence.

Almost every multi-agent failure in production is a communication or coordination failure, not a model failure. The models do fine; the glue between them is where the bugs live. Frameworks and libraries help, but the hardest part is not code — it's deciding when a problem genuinely benefits from multiple agents and when it just looks like it should.

The case studies above trace that line empirically. Anthropic's research system is a great fit because research has the signals. Claude Code's single-agent choice is the right fit because coding has the opposite signals. Customer support uses the supervisor pattern because the domain partitions cleanly. AutoGPT failed because it had no structure. MetaGPT works within bounds because SOPs imposed structure after the fact. Devin sits in the middle — autonomous where tests verify, fragile where they don't.

If you can explain which of those case studies your system most resembles, you know whether multi-agent is right for your problem. If you can't, start single-agent and measure.

**Related reading**

- [Designing Long-Running Agents — Part 1: Architecture](/blog/machine-learning/ai-agent/designing-long-running-agents-architecture-foundations)
- [Designing Long-Running Agents — Part 2: Reliability & Production](/blog/machine-learning/ai-agent/designing-long-running-agents-reliability-production)
- [Building Effective Agents: A Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide)
- [Scaling Managed Agents: Decoupling the Brain from the Hands](/blog/machine-learning/ai-agent/scaling-managed-agents-decoupling-brain-from-hands)
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents)
- [Designing AI Companion and Assistant Agents](/blog/machine-learning/ai-agent/designing-ai-companion-assistant-agents)
