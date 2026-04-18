---
title: "Scaling Managed Agents: Decoupling the Brain from the Hands"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "AI Agent"
tags:
  ["ai-agent", "managed-agents", "anthropic", "claude", "infrastructure", "architecture", "context-engineering", "sandbox"]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Anthropic's Managed Agents service rebuilt itself around one idea: the brain (harness) and the hands (sandbox) should be separate, replaceable pieces. This article unpacks why the original coupled design broke down, what the decoupled architecture looks like, and the engineering lessons any team building long-horizon agents can steal."
---

## Why This Article Exists

Most writing on agents is about the model — picking prompts, choosing tools, engineering context. But once you try to ship an agent that runs for hours, hands off work between sessions, and survives crashes, you discover a second discipline hiding underneath: **agent infrastructure**.

Anthropic recently published a post titled [*Scaling Managed Agents: Decoupling the Brain from the Hands*](https://www.anthropic.com/engineering/managed-agents), where Lance Martin, Gabe Cemaj, and Michael Cohen describe how they rebuilt Managed Agents from a monolithic "everything in one container" design into a decoupled system of independent brains, hands, and sessions.

This article is my annotated walkthrough. I'll try to explain not just *what* they changed, but *why* each move is an actual engineering principle you can lift into your own agent stack — whether you're running one agent on a laptop or a fleet of them in production.

## The Setup: What Is a "Managed Agent"?

A **managed agent** is an agent you don't run yourself. You hand Anthropic a task and some tools, and the service handles everything else: spinning up a sandbox, driving Claude through the loop, checkpointing state, retrying on failure, handing back a result.

Think of it as the **serverless version of an agent**. You don't manage the container, the retry logic, the session log, or the credentials plumbing. You just get an endpoint that can think and act.

Three moving parts live inside any managed agent platform:

| Component | What it does | Analogy |
| --- | --- | --- |
| **Harness** | The loop that calls the model, parses tool calls, routes results | The **brain** |
| **Sandbox** | The environment where generated code and tool calls execute | The **hands** |
| **Session log** | The durable record of everything that happened | The **memory** |

The original design fused all three into a single long-lived container. The new design treats them as independent, composable primitives. Almost every interesting engineering decision in the post follows from that one structural choice.

## The Problem: Pet Infrastructure

When every session had its own container holding the harness, the sandbox, *and* the log, containers became **pets**, not cattle.

If you've run infrastructure before, you know the distinction — pets are unique, hand-nurtured machines you SSH into when something breaks. Cattle are interchangeable workers you kill and replace. Pets don't scale. Every pet requires attention.

Here's what "pet sessions" actually felt like, in the authors' words:

> "Understanding why a container was unresponsive required walking the WebSocket event stream by hand: was the harness stuck in a loop? Had a tool broken? Was the network down? Was the container even alive?"

Three specific pains fell out of this:

### 1. Lost Sessions on Container Death

Session logs lived *inside* the container. If the container crashed, OOMed, or got evicted, the log died with it. Hours of agent work could vanish. You couldn't retry from where it left off because "where it left off" was in the same grave as the failure.

### 2. Private Resources Were Hard to Reach

Because harnesses assumed they ran next to their tools, customers had to either:

- **Peer their network** with Anthropic's infrastructure (painful, involves security review), or
- **Run the harness locally**, giving up the "managed" part of managed agents.

Neither option fit a world where most enterprise tools live behind a VPN or an OAuth wall.

### 3. Harnesses Absorbed Model-Specific Quirks

This one is subtle but important. Claude Sonnet 4.5 exhibited what the team calls **context anxiety** — as the context window approached its limit, the model rushed to wrap up, sometimes ending a task prematurely. To fix it, engineers added **context resets** into the harness.

Then Claude Opus 4.5 shipped without context anxiety. Suddenly the reset logic became **dead weight** — a workaround for a problem the new model didn't have. But because the harness was coupled to the container and the session, ripping it out meant touching everything.

> **The lesson:** when your harness contains model-specific hacks, every model upgrade becomes a migration.

## The Decoupling Solution

The rebuilt architecture virtualizes the three components and gives each a narrow, replaceable interface. It looks like this:

```
          ┌─────────────────────────┐
          │        SESSION          │  ← durable, external memory
          │  events, context, logs  │
          └──────────▲──────────────┘
                     │ getEvents() / emitEvent()
                     │
       ┌─────────────┴─────────────┐
       │         BRAIN             │  ← stateless harness
       │   (harness + Claude)      │
       └──────────┬────────────────┘
                  │ execute(name, input)
                  │
       ┌──────────▼────────────────┐
       │          HANDS            │  ← sandboxes, tools, MCP
       │   (execution environment) │
       └───────────────────────────┘
```

Every arrow is a narrow, stable API. Every box is independently replaceable. Let's walk through each piece.

### 1. The Brain Becomes Stateless

The harness moved **out** of the container. It now calls every execution environment through a single uniform tool interface:

```
execute(name, input) → string
```

That's the whole contract. The harness doesn't care whether `name` routes to a bash sandbox, an MCP server, a mobile device, or another agent. It just calls `execute` and gets a string back.

Because the session log lives externally, the harness is now **stateless**. If it crashes mid-task, a new harness can pick up where it left off:

```python
# Recover a crashed session
harness = Harness.wake(session_id)
session = harness.get_session(session_id)
# Resume the loop with full context
harness.continue_from(session)
```

The harness writes every meaningful step to the session with `emitEvent(session_id, event)`. This small change unlocks a huge one: **harnesses are now cattle**. You can kill one and spawn another without losing anything.

One concrete payoff the post calls out: **time-to-first-token dropped ~60% at p50 and >90% at p95**. Why? Because sessions no longer had to wait for a container to boot before the model could start thinking. Containers spin up only when the model actually calls a tool.

### 2. The Hands Become Pluggable

Sandboxes, custom tools, and MCP servers all hide behind the same `execute` interface. From the brain's perspective, nothing is special about "the sandbox" — it's just one more tool.

```
tools = [
    bash_sandbox,        # runs shell commands
    file_editor,         # reads/writes files
    mcp_server_github,   # pulls PR data
    another_agent,       # delegates subtasks
]

brain.run(task, tools=tools)
```

This unlocks three things you couldn't do before:

- **Mix and match execution environments.** A single brain can drive a cloud sandbox, a phone, and a local MCP server in the same loop.
- **Brain-to-brain delegation.** One agent can hand off subtasks to other agents, because another agent is just another tool that implements `execute`.
- **Independent failure.** When a sandbox dies, only that sandbox dies. The brain keeps its context. The session log is intact. You retry the tool call, not the whole session.

This is the same trick Unix played with file descriptors — make everything look like a file, and suddenly `cat < network_socket > disk_file` works without anyone planning for it. Make everything look like `execute`, and suddenly agents can delegate to agents without anyone designing for that case.

### 3. The Session Becomes a Queryable Context Object

This is the most interesting piece, and the one most likely to change how you think about agent context.

Traditionally, a session log is a **record** — write-only, read-once when debugging. Managed Agents treat it as a **queryable context object**. The harness can:

- `getEvents(range)` — fetch any slice of the history
- **Rewind** to before a specific moment
- **Reread** earlier context it had dropped
- **Transform** events before feeding them back to Claude (e.g., summarize a long tool output, reorder for prompt cache hits)

```python
# Fetch only the last 10 events
recent = session.get_events(last=10)

# Or rewind to before a known-bad decision
good_state = session.get_events(until=checkpoint_id)

# Or transform for cache-friendliness
compact = summarize(session.get_events(range=old))
brain.continue_with(compact + recent)
```

The session **guarantees durability and availability**. The harness **implements the context strategy**. These are now two different problems with two different owners — and either can evolve without breaking the other.

> **The principle:** separate "what must never be lost" from "what should be shown to the model right now." They sound like the same thing. They are not.

If you're curious about the context-engineering side of this tradeoff, I wrote a companion piece on [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents) that goes deep on the in-context strategies (compaction, structured notes, sub-agents). Managed Agents is the infrastructure layer underneath those strategies.

## Security: Decoupling Is a Security Story, Not Just a Scalability Story

Here is a detail from the post that I think deserves more attention than it got.

In the coupled design, **credentials lived in the same container where Claude's generated code ran.** If a prompt injection tricked Claude into `cat ~/.aws/credentials` or `env | grep TOKEN`, those secrets were one `execute` call away.

The new design fixes this with two patterns worth stealing:

### Bundled Authentication (for git)

When a sandbox is initialized, a **per-repository, scoped access token** is cloned into the git remote's config. Git commands work normally. The token is scoped to one repo and never passes through the harness or the model's context.

```
sandbox_init()
  → clone repo with ephemeral, repo-scoped token
  → wire token into local git remote config
  → harness never sees the token
  → Claude never sees the token
```

Prompt-injecting `git push` into an attacker's fork fails because the token can't reach anything except the intended repo.

### Vault-Based Credentials (for OAuth / MCP)

For OAuth-backed custom tools, tokens sit in an **external vault**. Claude calls an MCP tool through a **dedicated proxy** that fetches credentials from the vault using a session-scoped token:

```
Claude → MCP proxy → vault → tool
           ▲
           └── session-scoped access, no creds in harness
```

The harness never handles the credential. The model never sees it. A prompt injection gets you at most the effect of one tool call, not a stolen secret.

> **The general rule:** credentials should not sit in the same execution scope as attacker-controllable code generation. If your agent runs model-generated bash next to your API tokens, you have a prompt-injection-shaped hole in your system.

## The Design Philosophy: "Programs As Yet Unthought Of"

The part of the post I found most quotable is the framing of *why* this decoupling was worth the rebuild:

> "Systems that have endured for decades succeed by virtualizing hardware into abstractions — process, file — general enough for programs that didn't exist yet."

Managed Agents is trying to be the **operating system of agents**. It picks narrow interfaces — `execute(name, input)`, `emitEvent`, `getEvents`, `wake` — and is deliberately *un*opinionated about what runs behind them.

The service guarantees four things:

1. You can **manipulate state** (session logs).
2. You can **compute** (sandboxes, tools).
3. You can **scale** across many brains and many hands.
4. You get **reliable, secure long-horizon operation**.

It makes **no assumption** about whether your tool is a Python REPL or an iPhone, whether your brain is Claude Code or a custom harness, whether your session log is accessed once or a thousand times. Those are programs yet unthought of. The interfaces don't care.

This is the same bet Unix made with files, and the same bet Kubernetes made with pods. It usually turns out to be a good bet.

## What This Means for Builders

Even if you'll never use Managed Agents directly, the architecture gives you a checklist for your own agent stack. I'd frame it as five questions:

### 1. Is your session log separable from your runtime?

If your agent crashes, do you lose context? If yes, move the session log **outside** the process. S3, a database, an append-only log — anywhere but the same container that might die.

### 2. Is your harness stateless?

Does your agent loop keep critical state in local variables, or does it emit events? If you can't kill the harness and spawn a new one that picks up mid-task, you're running pets.

### 3. Do all your tools share one interface?

If your harness has special-case code paths for "the sandbox" vs "other tools", you can't add a new execution environment without touching the harness. Collapse it to one interface and every new tool is a plugin, not a refactor.

### 4. Where do your credentials live?

If credentials are in the same scope as model-generated code, a prompt injection reaches them. Push them into a vault, scope them per-session, and proxy every tool call that needs them.

### 5. Do you treat the session log as queryable?

Most agents write logs and never read them during a session. The managed-agents view — that the log is a **queryable context store** you fetch slices from — is more powerful. It's the difference between "my agent has 200K tokens of context" and "my agent has a 200K token working set over an unbounded history."

## The Performance Wins in One Table

To put the concrete numbers in one place:

| Metric | Before (coupled) | After (decoupled) | Why |
| --- | --- | --- | --- |
| **p50 TTFT** | baseline | ~60% faster | no container boot before first token |
| **p95 TTFT** | baseline | >90% faster | same, but the long tail was dominated by container boot |
| **Session survivability** | dies with container | survives harness, sandbox, and network failures | state lives in external session |
| **Container count** | one per active session | one per active *execution* | many brains share few hands |
| **Model-specific harness hacks** | baked in | removable | context strategy is a harness choice, not a container property |

The last row is the quiet one, but it's the one I care most about. **Decoupling lets your infra age gracefully as models change.** Context anxiety is gone in Opus 4.5 — great, pull the hack out of the harness, leave everything else alone. In a coupled world that's a migration. In a decoupled world it's a diff.

## Closing Thoughts

The shape of the argument in this article is one I find myself returning to a lot:

- Start with a tightly coupled system because it's the fastest path to "it works."
- Hit scale, hit reliability demands, or hit *model change velocity*, and the coupling starts costing more than it saved.
- Break the system into **narrow, composable interfaces** and let each piece evolve independently.

For agents specifically, the three interfaces that seem to matter are:

- `execute(name, input) → string` — for hands
- `emitEvent` / `getEvents` — for memory
- `wake(sessionId)` — for recovery

If your agent stack has those three, you can probably survive a model upgrade, a container failure, and a prompt-injection attempt in the same afternoon. If it doesn't, you're one bad Tuesday away from finding out.

**Further reading**

- The original Anthropic post: [Scaling Managed Agents: Decoupling the Brain from the Hands](https://www.anthropic.com/engineering/managed-agents)
- My earlier notes on [Effective Context Engineering for AI Agents](/blog/machine-learning/ai-agent/effective-context-engineering-for-ai-agents)
- A more practical, builder-focused companion: [Building Effective Agents: A Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide)
