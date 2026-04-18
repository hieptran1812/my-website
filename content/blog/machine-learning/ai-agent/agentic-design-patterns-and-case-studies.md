---
title: "Agentic Design: Patterns, Principles, and Real-World Case Studies"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["ai-agent", "agentic-design", "llm", "tool-use", "multi-agent", "workflows", "architecture", "case-studies"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Building LLM-powered agents is less about prompting cleverness and more about system design. This guide walks through the core patterns — workflows, tool use, reflection, planning, multi-agent systems — and grounds each one in real-world case studies from Claude Code, Devin, Perplexity, AutoGPT, and Operator."
---

## What Is an "Agent," Really?

The word *agent* has been stretched to mean everything from a single ChatGPT response to autonomous robots. That's unhelpful. Let's start with a definition that works for building real systems.

**An agent is a system where an LLM dynamically decides what to do next, using tools, until a goal is met.**

The key phrase is *dynamically decides*. A hard-coded pipeline that calls an LLM at step 3 is a workflow. A system that loops — LLM → action → observation → LLM → action — and the LLM itself chooses the actions, is an agent.

Anthropic's framing, from their influential *Building Effective Agents* essay, draws the same line:

- **Workflows** — LLMs orchestrated through predefined code paths. The control flow is fixed.
- **Agents** — LLMs that direct their own control flow, using tools to observe and act.

Both are useful. Workflows are more predictable, cheaper, and easier to debug. Agents are more flexible and handle open-ended tasks better. The hard part of agentic design is knowing which to reach for and how to compose them.

This guide walks through the design patterns, the principles behind them, and the case studies that show what works (and what doesn't) in production.

## The Core Anatomy of an Agent

Before the patterns, the parts. Every agent is built from the same primitives:

```
┌─────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                        │
│  (loop: plan → act → observe → reflect → repeat)        │
└─────────┬──────────────────────────────┬────────────────┘
          │                              │
    ┌─────▼─────┐               ┌────────▼────────┐
    │    LLM    │               │     TOOLS       │
    │ (reasoner)│               │  (actions the   │
    └─────┬─────┘               │   agent can     │
          │                     │   take)         │
          │                     └────────┬────────┘
          │                              │
    ┌─────▼──────────────────────────────▼────────┐
    │                CONTEXT                       │
    │  (the information the LLM sees each step:    │
    │   system prompt, history, tool results,      │
    │   retrieved docs, scratchpad notes)          │
    └──────────────────────────────────────────────┘
```

Four moving parts:

1. **The LLM** — the reasoning engine. Turns context into a next action.
2. **The tools** — the set of things the agent can actually *do*: call an API, run code, search, write a file, send an email.
3. **The context** — everything the LLM sees on each step. System prompt + conversation history + tool outputs + retrieved information.
4. **The orchestrator** — the loop that wires it all together and decides when to stop.

Most "bad agent" problems are in one of these four. Weak reasoning? LLM problem. Can't do the task? Tools problem. Loses the plot? Context problem. Runs forever or gives up early? Orchestrator problem. Diagnosing which is wrong is 80% of agent engineering.

## The Core Design Patterns

Here are the patterns that keep showing up. Most real agents combine several.

### Pattern 1: The Tool-Use Loop (ReAct)

The simplest agent pattern. The LLM alternates between **thinking** and **acting**:

```
User: "What's the weather in Tokyo and should I bring a jacket?"

Thought: I need the current weather in Tokyo.
Action: get_weather("Tokyo")
Observation: 12°C, cloudy, light wind.

Thought: 12°C is cool. A light jacket would be comfortable.
Final Answer: It's 12°C and cloudy in Tokyo. Yes, bring a light jacket.
```

This is the **ReAct** pattern (Reasoning + Acting), and it's the foundation of almost every tool-using agent. The LLM emits a structured response that the orchestrator parses into either a tool call or a final answer.

**When to use:** any task where the agent needs information from or actions in the outside world.

**When to avoid:** tasks where you already know the tool sequence in advance. A hard-coded pipeline is cheaper and more reliable.

### Pattern 2: Prompt Chaining (Workflow)

Not every problem needs an agent. **Prompt chaining** decomposes a task into a fixed sequence of LLM calls where each step's output is the next step's input.

```
Input → LLM(step 1) → LLM(step 2) → LLM(step 3) → Output
```

Example: writing a blog post.

1. LLM drafts an outline
2. LLM expands each section
3. LLM rewrites for tone
4. LLM fact-checks claims

No dynamic decisions. No tool loops. Just a chain.

**When to use:** the task has clear stages you can enumerate. Each stage is non-trivial but predictable. You want determinism.

**When to avoid:** the stages depend on the input in ways you can't enumerate.

### Pattern 3: Routing

A router LLM inspects the input and dispatches it to the right specialized handler:

```
                  ┌─ billing_agent
                  │
User query → Router ─ technical_support_agent
                  │
                  └─ general_chat
```

Each downstream agent has a narrow prompt and narrow tools. The router is cheap and shallow. The specialists are expensive and deep. You save money and quality simultaneously because each specialist only sees queries it's well-suited for.

**When to use:** your workload has clearly distinct categories. Customer support. Multi-tool apps. Any "there are roughly 5 kinds of questions" situation.

### Pattern 4: Parallelization

Run multiple LLM calls concurrently, then combine their outputs. Two flavors:

- **Sectioning** — break the task into independent subtasks, run in parallel, combine. (E.g., summarize 20 documents separately, then merge.)
- **Voting** — run the same task multiple times, combine by voting or majority. Improves reliability on tasks where the model sometimes gets it wrong.

Parallelization is a **latency** optimization (reduce wall-clock time) and, in the voting case, a **quality** optimization (reduce variance).

### Pattern 5: Orchestrator-Workers

A "manager" LLM breaks a task into subtasks and dispatches them to "worker" LLMs. The manager doesn't do the work — it coordinates. Workers don't see the whole task — they only see what they need.

```
        ┌─ Worker A: "write the intro section"
        │
Manager ─ Worker B: "write the methods section"
        │
        └─ Worker C: "write the conclusion"
            │
            └─ Manager assembles final output
```

This looks like routing but is stronger: the manager can *decompose dynamically*, not just classify.

**Case in point:** Anthropic's multi-agent research system uses this pattern — a lead agent spawns subagents to search different aspects of a question in parallel, then synthesizes their findings.

### Pattern 6: Evaluator-Optimizer (Reflection)

One LLM generates output. A second LLM critiques it. The first revises. Loop until the critic is satisfied (or a budget is hit).

```
Generator → draft → Evaluator → feedback → Generator → revised → ...
```

The evaluator can be the same model with a different prompt ("You are a strict code reviewer. Find bugs."), a smaller cheaper model, or even a non-LLM check (does the code compile? does the JSON validate?).

**When to use:** tasks with clear quality criteria (code correctness, policy compliance, format validation). The gain from reflection is largest when the evaluator has high-signal feedback the generator can act on.

**When to avoid:** tasks where "quality" is vague. A reflection loop that says "try again, better" just wastes tokens.

### Pattern 7: The Autonomous Agent

The fullest form. An LLM with tools, running in a loop, deciding its own next step and its own stopping condition.

```
while not done:
    action = LLM(context)
    observation = execute(action)
    context = update(context, action, observation)
    done = action.is_final()
```

This is what most people mean by "agent." It's the most flexible and the hardest to get right — loops can go forever, context can fill with garbage, errors can cascade.

**Production autonomous agents almost always combine this core loop with:**

- Step budgets (halt after N steps)
- Context compaction (summarize old turns)
- Safety rails (can't delete files, can't spend money without approval)
- Human-in-the-loop checkpoints (ask before irreversible actions)

## The Principles Behind What Works

Beyond specific patterns, a handful of principles keep resurfacing in agents that work well in production. These are the ones I'd tattoo on my arm if I had to.

### Principle 1: Start With the Simplest Thing That Could Work

Before building a multi-agent system, try a single agent. Before a single agent, try a prompt chain. Before a chain, try one good prompt. Each step up adds complexity, tokens, and failure modes.

The default answer to "should I add more agents?" is no. The burden of proof is on complexity.

### Principle 2: Tools Are Your Product, Not the Prompt

A mediocre prompt with great tools beats a great prompt with mediocre tools, every time. Tool design is where most of the agent quality lives.

Good tools:

- Have names and descriptions a smart human could understand without docs
- Take few, well-named parameters
- Return results that are easy for the next LLM call to reason about (structured, concise, errors that explain themselves)
- Fail explicitly with actionable error messages ("API rate limit hit, retry in 30s") not cryptically ("HTTP 429")

Bad tools make agents loop, retry wrong calls, or give up. Upgrading your tools is almost always higher-leverage than upgrading your prompts.

### Principle 3: Context Is a Scarce Resource

Every token in the LLM's context competes with every other. An agent's context fills with:

- System prompt
- Conversation history
- Tool definitions
- Tool results (often verbose — API dumps, stack traces, file contents)
- Retrieved documents
- Scratchpad notes

Without discipline, this fills fast and quality degrades. Techniques that help:

- **Compaction** — summarize older turns once they're no longer needed verbatim
- **Structured note-taking** — the agent writes key facts to a scratchpad, then clears long tool output
- **Sub-agent delegation** — hand a sub-task to a fresh agent with only the needed context, return just the result
- **Retrieval** — don't stuff everything into context; search for what's relevant per step

See the companion article on [Effective Context Engineering](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) for the details.

### Principle 4: Design for Failure, Not for Success

Agents fail. Tools time out, APIs change, the LLM picks the wrong action, a user asks something ambiguous. The agents that work in production are the ones whose designers planned for failure.

Concretely:

- Every tool call must handle errors gracefully
- The orchestrator must have a max-step budget
- Irreversible actions (delete, send email, spend money) need confirmation
- Every deployed agent needs observability — traces, not just logs
- "What does this agent do on input X?" should be answerable before shipping

### Principle 5: Evaluate on Outputs, Not Vibes

"It seems better" is not a quality signal. Real agent evaluation needs:

- A **test set** of representative tasks with expected outputs or grading criteria
- **Automated metrics** where possible (did it call the right tool? did the output validate?)
- **LLM-as-judge** for subjective criteria (but with a ground-truth-aligned judge)
- **Regression tracking** — when you change the prompt, do old cases still work?

See the [Eval Agents](/blog/machine-learning/ai-agent/eval-agents) article for practical evaluation techniques.

### Principle 6: Keep Humans in the Loop Where It Matters

Full autonomy is overrated. The best production agents ship with well-placed checkpoints:

- Approve a destructive action before executing
- Confirm a plan before starting a long task
- Ask a clarifying question when ambiguous
- Escalate when stuck

This isn't a failure of the agent — it's correct design. Humans catch what models miss, and a five-second confirmation prevents a catastrophic mistake.

## Case Studies: What Real Agents Actually Look Like

Theory is cheap. Let's look at how real production agents are designed.

### Case Study 1: Claude Code — The Developer's Coding Agent

**The task:** help a developer write, refactor, and debug code in their terminal.

**The architecture:**

- **Single-agent loop.** Claude Code is fundamentally one agent in a ReAct loop, not a multi-agent system. It reads, edits, runs shell commands, and loops.
- **Rich tool set.** File read/write/edit, glob/grep, bash, web fetch, subagent spawn. Each tool has careful documentation in the system prompt.
- **Context engineering via subagents.** For open-ended searches or research tasks, Claude Code spawns a subagent with a fresh context. The subagent does the work and returns a concise summary. The main agent never sees the intermediate noise.
- **TODO tracking as scratchpad.** A built-in TaskCreate/TaskUpdate mechanism lets the agent externalize its plan. Crucial for long tasks — without it, the "plan" has to live in context, which decays.
- **Permission layer.** Bash commands and file edits go through a permission system. Destructive actions prompt the user. Safe actions run automatically.
- **Hooks.** User-configurable pre/post-tool hooks let the environment enforce invariants (run the formatter after every edit, block `rm -rf` entirely).

**The lessons:**

- A single well-designed agent with great tools beats a cluster of mediocre agents, for well-scoped tasks like coding.
- Subagents are used surgically — for context isolation, not for a hierarchy of control.
- Human-in-the-loop (permissions) is a feature, not a limitation.
- Structured task tracking outside of raw chat history is essential for long-horizon work.

### Case Study 2: Devin — The Autonomous Software Engineer

**The task:** take a GitHub issue, write a PR that fixes it, end to end.

**The architecture (as publicly described):**

- **Long-horizon autonomous loop.** Devin runs in a sandboxed VM with a shell, browser, and editor. It can run for hours on a single task.
- **Plan-first execution.** Devin begins each task by producing a plan, shown to the user, then executes against it.
- **Streaming transparency.** The user sees the shell, browser, and editor in real time. This makes the agent auditable — critical for a "hands-off" system.
- **Explicit replanning.** When Devin hits a blocker, it re-plans explicitly rather than just retrying. Backtracking is first-class.
- **Episodic memory.** Knowledge learned across tasks (repo layouts, common pitfalls) is stored and retrieved.

**The lessons:**

- Autonomous long-horizon agents need explicit planning and re-planning steps, not just a vanilla ReAct loop.
- Transparency is the price of autonomy — the more the agent does alone, the more the user needs to see what it's doing.
- Persistent memory matters when tasks repeat or relate. Without it, every run starts from scratch.

### Case Study 3: Perplexity / Search Agents — Tool-Use as the Product

**The task:** answer a user's question with cited sources from the web.

**The architecture:**

- **Narrow tool set.** Search the web, fetch a page, optionally search again. That's mostly it.
- **Workflow, not agent.** For most queries, Perplexity uses a largely fixed pipeline: rewrite the query → search → rerank → extract → synthesize. The dynamic-decision surface is small.
- **Parallel search.** Multiple queries run in parallel for broader coverage.
- **Citation as a first-class output.** Every claim in the final answer is grounded in a retrieved source. This isn't a flourish — it's how users verify correctness.

**The lessons:**

- Not every "agent" needs dynamic planning. If the flow is predictable, a workflow with LLM steps is simpler, faster, and more reliable.
- Narrow tools beat broad tools. Perplexity doesn't try to do everything — it does search, really well.
- Citations turn the LLM from "authority" into "researcher" — a much safer role to ship.

### Case Study 4: AutoGPT / BabyAGI — Lessons From the Hype Cycle

**The task:** given any goal, break it into subtasks and autonomously accomplish them.

**The architecture:**

- **Recursive task decomposition.** A task-creation agent generates subtasks; a task-execution agent executes them; a task-prioritization agent reorders.
- **Vector memory.** Past actions and observations stored in a vector DB, retrieved each step.
- **No humans in the loop.** Fully autonomous.

**What happened in practice:**

- For broad, open-ended tasks, AutoGPT loops forever, loses the plot, or burns tokens on trivial subtasks.
- It works on narrow well-defined tasks — but for those, simpler patterns (ReAct, workflows) are cheaper and more reliable.
- The "agent keeps spawning subtasks" dynamic frequently fails due to context management — the agent forgets its own plan.

**The lessons:**

- Autonomy is not a feature; it's a tradeoff. Every degree of autonomy you add is a degree of reliability you subtract, unless paired with new structure to compensate (planning, memory, checkpoints).
- "Given any goal, figure it out" is too unconstrained for current LLMs. Successful agents have narrower scope.
- Vector memory is not a substitute for a good context strategy — if the agent retrieves the wrong thing, it's just loud noise in the prompt.

### Case Study 5: Operator / Computer-Use Agents — Agents That Click

**The task:** operate a computer — move the mouse, type, click — to accomplish user goals in real software.

**The architecture:**

- **Multimodal LLM.** The agent sees screenshots of the screen and reasons about what to click.
- **Action vocabulary.** A tight set of actions: click at (x, y), type, scroll, wait. The agent emits these one at a time.
- **Perception-action loop.** After each action, take a new screenshot, re-plan.
- **Heavy safety rails.** Sensitive actions (payments, sending messages) require user confirmation. Certain sites are blocked.

**The challenges this pattern surfaces:**

- **Grounding is brittle.** The model has to identify the right button from pixels. UI changes break agents.
- **Latency is painful.** Screenshot + LLM call per step means each action takes seconds. Tasks that take humans 30 seconds take the agent 5 minutes.
- **Error recovery is hard.** If the agent clicks the wrong thing, the resulting state might be alien to it.

**The lessons:**

- Agents that operate unstructured environments (pixels, DOMs, real-world robotics) pay a huge tax versus agents that operate structured ones (APIs, filesystems).
- The right primitive matters. When an API exists, agents should use it. Computer-use is the last resort, not the default.
- Tight safety rails and human confirmation are non-negotiable when the agent can act on the open internet.

### Case Study 6: Multi-Agent Research Systems

**The task:** answer a deep research question ("what's the state of the art in X?"), with breadth and depth.

**The architecture (from Anthropic's multi-agent research blog post):**

- **Lead agent + subagents.** A lead agent plans the research, spawns subagents to investigate specific angles in parallel, and synthesizes their findings.
- **Parallelism is the point.** Five subagents exploring five angles simultaneously beats one agent exploring sequentially.
- **Each subagent has a narrow brief.** Its context is focused; it returns a concise synthesis.
- **Token costs are high.** Multi-agent systems use 10-20× more tokens than single-agent systems. They're worth it when the task genuinely benefits from parallel exploration.

**The lessons:**

- Multi-agent systems pay off when a task has naturally parallel structure (multiple angles, multiple sources) and the cost of synthesis is worth the breadth gain.
- They do not pay off for tasks that are fundamentally sequential (coding, debugging, step-by-step reasoning) — there, a single agent with a well-maintained context is better.
- The manager-worker split is a powerful way to do context engineering: each worker has a clean context, the manager has the summaries.

## A Decision Framework for Your Own Agent

When you're designing an agent, here's the order I'd suggest walking through:

**Step 1: Can a prompt solve it?** A single LLM call with good prompting, maybe some RAG. If yes, stop.

**Step 2: Can a workflow solve it?** Fixed sequence of LLM calls, maybe with some routing. If yes, stop. Workflows are cheaper and more predictable than agents.

**Step 3: Does it need a single agent?** Tool-using ReAct loop, scoped to one domain. Most real-world agent needs stop here. Claude Code, Cursor's agent mode, Perplexity — all largely single-agent.

**Step 4: Does it need multiple agents?** Only if the task has genuine parallel structure or deep decomposition. Budget for 10-20× higher token cost.

**Step 5: How do you handle failure?** Before shipping: budgets, confirmations, retries, escalation, observability, evals.

**Step 6: What's your feedback loop?** Real users on real tasks with real evals. You will be wrong about what breaks — let the evidence update you.

## Closing: The Quiet Truth of Agentic Design

The flashiest demos are full autonomy, multi-agent swarms, emergent behavior. The actual state of the art, in systems that ship and work, is closer to this:

- One or two agents, not twenty
- Great tools, carefully designed
- Workflows where workflows will do
- Explicit planning, explicit context management, explicit safety rails
- Humans in the loop at the right moments
- Boring, reliable, measured improvement

Agentic design is not about building cleverer prompts. It's about building systems whose behavior you can understand, measure, and improve over time. The patterns and principles above are the scaffolding. The case studies show what they look like under load.

Build the simplest agent that could work. Make it work well. Add complexity only when the data says you must.

## Further Reading

Related articles:
- [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents)
- [How to Build Effective Agents](/blog/machine-learning/ai-agent/how-to-build-effective-agents)
- [Eval Agents](/blog/machine-learning/ai-agent/eval-agents)
- [Advanced Tool Use](/blog/machine-learning/ai-agent/advance-tool-use)
- [Model Context Protocol](/blog/machine-learning/ai-agent/model-context-protocol)

External sources:
- Anthropic — *Building Effective Agents*
- Anthropic — *How We Built Our Multi-Agent Research System*
- Lilian Weng — *LLM-Powered Autonomous Agents*
- Cognition AI — *Don't Build Multi-Agents*
- OpenAI — *A Practical Guide to Building Agents*
