---
title: "Multi-Agent Topologies: Orchestrator-Worker, Peer-to-Peer, and Everything Between"
date: "2026-06-27"
description: "The major architectures for connecting multiple AI agents — orchestrator-worker, pipeline, peer-to-peer, hierarchical, and market-based — with tradeoffs, failure modes, and when to use each."
tags: ["ai-agents", "multi-agent", "architecture", "orchestration", "llm", "machine-learning", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

There is a seductive simplicity to the single-agent loop. One model, one context window, one chain of tool calls, one answer. You can read the whole trace in a single sitting. When it fails, you know exactly what broke. When it succeeds, you know why.

The moment you add a second agent, you are no longer building a model — you are building a distributed system. And distributed systems have a way of punishing overconfidence. The network partitions. Messages arrive out of order. One worker finishes in 200 ms; another takes 12 s; the result you needed has already expired. Two agents write conflicting conclusions to the same shared document. A coordinator floods a downstream worker with 40 simultaneous requests. The worker drops half of them silently.

None of these problems appear in single-agent logs, because none of them can occur with a single agent. They only emerge when you add the second one — and they get worse, nonlinearly, as you add the third, fourth, and fifth.

This post is about the architectural choices that determine which of those failure modes you will face. We will go through seven canonical multi-agent topologies — orchestrator-worker, sequential pipeline, parallel fan-out/fan-in, peer-to-peer mesh, hierarchical tree, blackboard/shared workspace, and market/bidding — compare them on six dimensions, map their failure signatures, and walk through the decision framework we use at production scale to choose between them.

![The orchestrator-worker topology: one controller dispatches to four specialized workers, then merges results.](/imgs/blogs/multi-agent-topologies-1.webp)

The diagram above is the mental model for the most common topology. Before we can reason about why it sometimes fails, we need to understand why we reach for multi-agent systems in the first place.

## 1. Why one agent isn't enough — and why naively adding more makes things worse

A single agent running against a 128k-token context window can, in principle, hold everything in mind at once. It can reason across a 50-page document, a tool call trace, and a conversation history without needing another agent's help. So when does that break down?

**Context window pressure.** The theoretical limit is not the practical limit. At 60–70k tokens of active context, most models start showing degraded attention on early material ("lost in the middle" effect, documented in empirical studies with a clear U-shaped recall curve). A research task that requires simultaneously reasoning about 200 retrieved passages, a multi-step tool call history, and a complex instruction set pushes well past that threshold.

**Latency from serial execution.** A single agent that calls four tools sequentially — each taking 2–3 seconds — runs in 8–12 seconds minimum. That is often too slow for interactive applications and completely unacceptable for workflows triggered by user actions. Parallelism requires multiple agents.

**Specialization quality.** A single generalist agent asked to simultaneously act as a Python debugger, a legal document analyzer, a SQL query planner, and a customer sentiment scorer will do all four tasks at average quality. Specialist agents — given task-specific system prompts, domain-specific tools, and narrow context — consistently outperform generalists on their respective tasks when measured against human raters.

**Fault isolation.** A single agent that hallucinates on step 7 of a 12-step task corrupts all subsequent steps. Multiple agents with discrete inputs and outputs create natural verification points; a bad result from one agent can be caught, retried, or flagged before it contaminates the rest of the pipeline.

**Tool proliferation.** A single agent asked to use 15 different tools simultaneously — a browser, a Python executor, a SQL engine, 4 external APIs, and 8 domain-specific retrievers — produces a system prompt that is hundreds of lines long. The model must navigate this massive tool namespace on every tool call, and empirically, models with 15 tools available make worse tool-selection decisions than those with 5 tools targeted at their specific domain. Multi-agent systems let you give each agent a small, curated tool set matched to its role.

**Budget and rate limit management.** A single agent that calls an expensive API 40 times in a task is difficult to budget. A multi-agent system where each worker has its own rate-limit budget, timeout budget, and token budget is dramatically easier to cost-engineer. You can run expensive workers on powerful models and cheap workers on fast, small models, right-sizing the compute for each step.

**Why naive scaling fails.** Given these pressures, the temptation is to add agents freely. Add an agent for every subtask. Have them all communicate with each other. That temptation leads to coordination hell. Every agent-to-agent message is a potential failure point. Every shared state object is a potential race condition. Every parallel branch is a problem for the merger that must reconcile conflicting results. The coordination overhead grows as O(N²) in the fully connected case, and you end up spending more compute on routing messages than on the actual task.

The solution is choosing the right topology for each task's dependency structure. The topology is not decoration — it is the primary architectural decision that determines what can go wrong.

### The topology-first design principle

Before you write a single line of multi-agent code, write out the dependency graph of the subtasks. Specifically, answer these questions:

- Are the subtasks independent, or does B depend on A's output?
- Is the dependency structure linear (A→B→C) or a DAG (A and B both feed C)?
- How many parallel branches are there, and how wide is each?
- Are the subtasks homogeneous (same model, same interface) or heterogeneous?
- What is the expected number of agents at steady state?
- Which agent failures are tolerable (task degrades gracefully) and which are fatal (task fails entirely)?

Write these answers before drawing a single architecture diagram. The topology that satisfies all six answers cleanly is almost always the right choice. The topology that "feels right" but leaves two or three of these questions unresolved will produce the 2 AM incident six weeks later.

A useful sanity check: if you cannot draw the dependency graph of your task's subtasks in under five minutes, the task is not yet decomposed clearly enough to build a multi-agent system. Clarify the decomposition first; the topology choice will then be obvious.

## 2. Topology 1: Orchestrator-Worker — one controller, N specialists

The orchestrator-worker pattern is the most widely deployed multi-agent topology in production AI systems as of 2026. One agent — the orchestrator — holds the high-level plan, manages the task queue, routes subtasks to the appropriate specialist, monitors results, handles retries, and assembles the final output. The worker agents have no awareness of the overall plan; they receive a single, scoped task and return a single result.

![Orchestrator-Worker: the orchestrator fans out to four specialist workers and collects results into a single merged output.](/imgs/blogs/multi-agent-topologies-1.webp)

### Structure

```
orchestrator → task_queue → [worker_A, worker_B, worker_C, worker_D]
                                 ↓           ↓           ↓           ↓
                              result_A    result_B    result_C    result_D
                                                ↓
                                         orchestrator.merge()
```

The orchestrator typically runs on a capable frontier model (GPT-4o, Claude 3.7 Sonnet, Gemini 1.5 Pro) because it needs strong planning and routing capability. Workers can run on smaller, cheaper models tuned for their specific domain — a code worker running DeepSeek-Coder-7B, a summarization worker running Mistral-7B-Instruct, a retrieval worker using an embedding model with no generation at all.

### Implementation sketch

```python
from anthropic import Anthropic
import asyncio
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    worker_type: str
    payload: str

@dataclass
class Result:
    task_id: str
    content: str
    success: bool

class Orchestrator:
    def __init__(self, client: Anthropic, workers: dict):
        self.client = client
        self.workers = workers  # {'web_search': WebSearchWorker(), ...}

    async def run(self, user_goal: str) -> str:
        # Step 1: decompose goal into tasks
        tasks = await self._plan(user_goal)

        # Step 2: dispatch tasks to workers, potentially in parallel
        results = await asyncio.gather(
            *[self._dispatch(task) for task in tasks],
            return_exceptions=True
        )

        # Step 3: merge results
        return await self._merge(user_goal, [r for r in results if isinstance(r, Result)])

    async def _plan(self, goal: str) -> list[Task]:
        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system="""You are a task planner. Given a goal, decompose it into
            discrete subtasks. Each subtask must specify: worker_type (one of:
            web_search, code_executor, summarizer, validator), and a clear payload.
            Return JSON: {"tasks": [{"id": "t1", "worker_type": "...", "payload": "..."}]}""",
            messages=[{"role": "user", "content": goal}]
        )
        import json
        data = json.loads(response.content[0].text)
        return [Task(**t) for t in data["tasks"]]

    async def _dispatch(self, task: Task) -> Result:
        worker = self.workers.get(task.worker_type)
        if not worker:
            return Result(task.id, f"Unknown worker: {task.worker_type}", False)
        try:
            content = await worker.execute(task.payload)
            return Result(task.id, content, True)
        except Exception as e:
            return Result(task.id, str(e), False)

    async def _merge(self, goal: str, results: list[Result]) -> str:
        context = "\n".join(f"[{r.task_id}]: {r.content}" for r in results)
        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system="Synthesize the worker results into a coherent final answer.",
            messages=[{
                "role": "user",
                "content": f"Goal: {goal}\n\nWorker results:\n{context}"
            }]
        )
        return response.content[0].text
```

### When to use it

- Tasks that decompose naturally into parallel, independent subtasks
- You need a single point of accountability for the task's success or failure
- Workers are heterogeneous (different models, different APIs, different rate limits)
- You need retries, timeouts, and budget tracking to live in one place

### Key tradeoffs

| Dimension | Verdict |
|---|---|
| Latency | Medium — workers run in parallel, but orchestrator adds a planning round-trip |
| Fault isolation | Low — orchestrator failure kills everything |
| Coordination overhead | Low — all messaging goes through one hub |
| Debuggability | High — trace one agent's decision log to understand all routing |
| Token cost | Medium — orchestrator sees all results for merging |

The orchestrator is both the topology's greatest strength and its Achilles' heel. A single orchestrator failure brings down the entire task. In practice, this means you need at minimum a restart-on-failure policy and, for critical pipelines, a hot standby orchestrator with shared state.

## 3. Topology 2: Sequential Pipeline — agents as stages in a directed chain

A pipeline topology connects agents in a linear sequence: the output of agent N becomes the input to agent N+1. There is no coordinator — each agent simply consumes a structured artifact and produces the next version of it.

![Sequential pipeline: each agent transforms the artifact and passes it forward — retrieval, reasoning, critique, and formatting in sequence.](/imgs/blogs/multi-agent-topologies-2.webp)

### Structure

```
input → Agent_A → artifact_v1 → Agent_B → artifact_v2 → Agent_C → artifact_v3 → Agent_D → output
```

The pipeline model maps directly onto assembly-line thinking: each agent specializes in one transformation. The retrieval agent fetches context. The reasoning agent drafts an answer using that context. The critique agent identifies factual errors or logical gaps. The formatting agent renders the final output in the required schema.

### Why pipelines work well for document processing

Pipelines excel when the task has a natural sequential dependency graph — when step B cannot meaningfully start until step A is complete, and step C depends on B's output, not A's raw input. Document pipelines, code review workflows, and report generation are classic fits.

The coordination overhead in a pipeline is nearly zero. Agent N does not know or care about agents before or after it. It reads an input, does its job, writes an output. The pipeline runner is stateless — it just moves artifacts from one queue to the next.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Artifact:
    version: int
    content: str
    metadata: dict

class PipelineStage:
    def __init__(self, name: str, system_prompt: str, model: str = "claude-haiku-4-5"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def process(self, artifact: Artifact, client) -> Artifact:
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.system_prompt,
            messages=[{
                "role": "user",
                "content": f"Input artifact v{artifact.version}:\n\n{artifact.content}"
            }]
        )
        return Artifact(
            version=artifact.version + 1,
            content=response.content[0].text,
            metadata={**artifact.metadata, f"stage_{self.name}": "completed"}
        )

class Pipeline:
    def __init__(self, stages: list[PipelineStage], client):
        self.stages = stages
        self.client = client

    def run(self, initial_content: str) -> Artifact:
        artifact = Artifact(version=0, content=initial_content, metadata={})
        for stage in self.stages:
            artifact = stage.process(artifact, self.client)
            # Optional: validate artifact before passing to next stage
            if not self._validate(artifact, stage):
                raise ValueError(f"Stage {stage.name} produced invalid artifact")
        return artifact

    def _validate(self, artifact: Artifact, stage: PipelineStage) -> bool:
        # Domain-specific validation — check artifact isn't empty, has required fields, etc.
        return bool(artifact.content.strip())

# Usage:
stages = [
    PipelineStage("retrieval", "Fetch relevant context from the document. Output a context block."),
    PipelineStage("reasoning", "Draft a response using the provided context. Be thorough."),
    PipelineStage("critique", "Identify factual errors, logical gaps, or missing citations. Output the corrected draft."),
    PipelineStage("formatting", "Format the final answer as structured JSON with 'summary', 'details', and 'citations' keys."),
]
```

### The hidden failure mode: error amplification

Pipelines have a nasty failure mode that doesn't exist in orchestrator-worker: **error amplification**. A mistake at stage 1 is presented to stage 2 as ground truth. Stage 2 reasons from that mistake, possibly making it worse. Stage 3 receives a doubly-degraded artifact. By stage 4, the original error may be completely unrecoverable. The final output looks coherent — it is a valid artifact — but it is wrong in a way that no single stage's output clearly telegraphs.

The mitigation is a validation step between each stage that checks structural properties of the artifact (not just whether it is non-empty). For high-stakes pipelines, you insert a "red team" stage that specifically looks for errors rather than improving the artifact.

### When to use it

- Task steps have strict sequential dependencies (B requires A's output)
- You want maximum simplicity and debuggability (trace the artifact through stages)
- Per-stage models can be tuned or fine-tuned independently
- You need a clear audit trail: every artifact version is a checkpoint

## 4. Topology 3: Parallel Fan-out / Fan-in — parallel workers with a merger

Fan-out/fan-in is the topology to reach for when the task's subtasks are **independent** and latency matters. An orchestrator splits the task into N parallel sub-problems, dispatches them simultaneously, and a merger assembles the results after all (or enough) workers complete.

![Fan-out / fan-in: the orchestrator fans three parallel workers out, then a merger aggregates their results.](/imgs/blogs/multi-agent-topologies-3.webp)

### Why this topology cuts latency

If each of N sub-tasks takes time $t_i$, a sequential pipeline takes $\sum t_i$ while a parallel fan-out takes $\max(t_i)$. For 5 workers each averaging 3 seconds with variance, the serial time is ~15 seconds and the parallel time is ~5 seconds — a 3× speedup. Real gains are even larger when workers call external APIs, since network I/O is the bottleneck and it parallelizes completely.

```python
import asyncio
from anthropic import AsyncAnthropic

class FanOutOrchestrator:
    def __init__(self):
        self.client = AsyncAnthropic()

    async def run(self, question: str, num_workers: int = 3) -> str:
        # Generate independent sub-questions
        sub_questions = await self._decompose(question, num_workers)

        # Fan out: run all workers concurrently
        tasks = [self._worker(sq) for sq in sub_questions]
        partial_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter failed workers
        valid_results = [
            r for r in partial_results
            if isinstance(r, str) and r.strip()
        ]

        # Fan in: merge
        return await self._merge(question, valid_results)

    async def _decompose(self, question: str, n: int) -> list[str]:
        response = await self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=f"Split the question into {n} independent sub-questions. Return as JSON list.",
            messages=[{"role": "user", "content": question}]
        )
        import json
        return json.loads(response.content[0].text)

    async def _worker(self, sub_question: str) -> str:
        response = await self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": sub_question}]
        )
        return response.content[0].text

    async def _merge(self, original: str, partials: list[str]) -> str:
        context = "\n\n---\n\n".join(f"Partial {i+1}:\n{p}" for i, p in enumerate(partials))
        response = await self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system="Synthesize the partial answers into a single coherent response.",
            messages=[{"role": "user", "content": f"Original question: {original}\n\n{context}"}]
        )
        return response.content[0].text
```

### The merger bottleneck

The fan-in merger is the hottest node in this topology. Every partial result passes through it. If the merger is slow, the latency advantage of parallel workers evaporates. If the merger fails, all the parallel work is lost. In production, you should:

1. Run a lightweight merger that only assembles, not re-reasons
2. Stream partial results into the merger as workers complete, rather than waiting for all workers before starting
3. Design partial result schemas so merging is O(N) concatenation, not O(N²) cross-comparison

Another important design choice: **what happens when some workers fail?** A naive implementation waits for all workers with `asyncio.gather()` and fails if any exception occurs. A resilient implementation uses `return_exceptions=True`, filters failed results, and merges the valid subset. For some tasks, 3 out of 4 partial results is good enough; for others (e.g., a tax calculation that requires all four jurisdictions), any missing result must trigger a retry.

### When to use it

- Sub-tasks are genuinely independent (no data dependency between workers)
- Wall-clock latency is a primary constraint
- You can define a clear, deterministic merge strategy
- Workers are homogeneous (same model, same interface)

## 5. Topology 4: Peer-to-Peer / Flat Mesh — agents communicate directly

In a peer-to-peer mesh, every agent can send messages directly to any other agent without routing through a central hub. There is no orchestrator and no fixed execution order. Each agent subscribes to a message bus or shared channel, announces its capabilities, and responds to requests from peers.

![Peer-to-peer mesh: six agents in a directed mesh, each capable of addressing peers directly — high connectivity, high coordination cost.](/imgs/blogs/multi-agent-topologies-4.webp)

### Why P2P is rare in production

The appeal is fault tolerance and flexibility. Without a central orchestrator, no single component failure kills the system. Agents can form dynamic coalitions: "I need a SQL writer and a data visualizer — who's available?" The agents that are up respond; the ones that are down are simply skipped.

The reality is that P2P coordination in LLM-based systems is extraordinarily difficult to reason about. Consider:

- **Message ordering**: Agent A sends a question to B; B responds with a question to C; C responds to A with a different question. A receives B's response and C's question in interleaved order. Which message represents the current state of the task?
- **Consensus on shared state**: If A and B both derive a "conclusion" from independent reasoning chains, and they contradict each other, who arbitrates?
- **Deadlock**: A is waiting for B's response to proceed; B is waiting for A's response to proceed. Neither progresses. This is a classic deadlock that distributed systems textbooks dedicate entire chapters to, and LLM agents have no native deadlock detection.

For these reasons, true P2P mesh topologies are mostly confined to research multi-agent systems (AutoGen's group chat mode, CAMEL's role-playing framework) rather than production pipelines. What you see more often in production is a **relaxed mesh** — a partial P2P topology where a subset of agents can communicate directly, but there is still a lightweight coordinator for task management and deadlock detection.

### Message protocol for a minimal P2P system

```python
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from typing import Callable

class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ANNOUNCE = "announce"  # capability announcement

@dataclass
class AgentMessage:
    sender_id: str
    recipient_id: str   # None = broadcast
    msg_type: MessageType
    content: str
    correlation_id: str  # for tracking query-response pairs
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

class MessageBus:
    """Minimal shared bus for P2P agent communication."""
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._capability_registry: dict[str, list[str]] = {}  # agent_id -> capabilities

    def register(self, agent_id: str, capabilities: list[str], handler: Callable):
        self._subscribers.setdefault(agent_id, []).append(handler)
        self._capability_registry[agent_id] = capabilities

    def find_capable_agents(self, capability: str) -> list[str]:
        return [aid for aid, caps in self._capability_registry.items() if capability in caps]

    async def send(self, msg: AgentMessage):
        if msg.recipient_id is None:
            # broadcast
            for handlers in self._subscribers.values():
                for h in handlers:
                    await h(msg)
        else:
            for h in self._subscribers.get(msg.recipient_id, []):
                await h(msg)
```

The moment you build this, you realize you need: timeouts, retry logic, message deduplication (in case of network issues), capability versioning (agents update their capabilities), and some form of task ownership tracking. That's a distributed system with all the usual problems. Most teams that start here end up retrofitting an orchestrator layer.

### When P2P makes sense

P2P is appropriate when:

- Agents are long-running services with persistent state (not ephemeral task runners)
- The task structure is genuinely dynamic — no static decomposition is possible
- Fault tolerance matters more than simplicity (e.g., a 24/7 monitoring mesh where individual agents can fail and recover)
- You have the engineering capacity to build and maintain the coordination infrastructure

## 6. Topology 5: Hierarchical / Tree — supervisor trees with sub-orchestrators

The hierarchical tree topology generalizes orchestrator-worker by adding multiple levels of orchestration. A root orchestrator manages sub-orchestrators; each sub-orchestrator manages a team of leaf workers. This mirrors how human organizations scale: a CEO doesn't manage 200 individual contributors directly — there are layers of management.

![Hierarchical tree: root orchestrator delegates to two sub-orchestrators, each managing 2-3 leaf workers — O(log N) coordination per level.](/imgs/blogs/multi-agent-topologies-5.webp)

### The span-of-control argument

A single orchestrator managing N workers faces a fundamental scaling problem: as N grows, the orchestrator's planning, routing, and monitoring complexity grows proportionally. An orchestrator managing 20 workers is trying to hold 20 task states, 20 result streams, and 20 retry states in a single context window. At some point, the orchestrator itself starts making poor decisions — losing track of which workers are still running, misrouting tasks, forgetting intermediate results.

The fix is bounded span-of-control: no single agent manages more than K direct reports (K = 3–6 in practice, based on analogy with human org design). An architecture with 2 sub-orchestrators each managing 5 workers gives you 10 leaf workers without any single agent managing more than 5.

### Sub-orchestrator specialization

The deeper win from hierarchical trees is that sub-orchestrators can be specialized. A root orchestrator receives a complex request ("Build a competitive intelligence report on Company X"). It decomposes this into: (a) financial data gathering and (b) product/market analysis. The financial sub-orchestrator manages a team of workers specialized in SEC filings, earnings call transcripts, and analyst reports. The product sub-orchestrator manages workers specialized in patent databases, job postings, and GitHub activity. Each sub-orchestrator can use domain-specific routing logic and domain-specific tools without the root needing to understand any of that detail.

```python
class HierarchicalOrchestrator:
    """Root orchestrator that delegates to domain sub-orchestrators."""

    def __init__(self, sub_orchestrators: dict[str, 'SubOrchestrator'], client):
        self.sub_orchestrators = sub_orchestrators  # e.g. {'financial': ..., 'product': ...}
        self.client = client

    async def run(self, goal: str) -> str:
        # Decompose goal into domain-scoped sub-goals
        assignments = await self._route_to_domains(goal)

        # Dispatch to sub-orchestrators in parallel
        results = {}
        sub_tasks = [
            self._delegate(domain, sub_goal, results)
            for domain, sub_goal in assignments.items()
        ]
        await asyncio.gather(*sub_tasks)

        # Root-level synthesis
        return await self._synthesize(goal, results)

    async def _route_to_domains(self, goal: str) -> dict[str, str]:
        """Decide which sub-orchestrators should handle which aspects."""
        response = await self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=f"""Route this goal to the available domain orchestrators.
            Available domains: {list(self.sub_orchestrators.keys())}.
            Return JSON: {{"domain": "sub_goal"}}""",
            messages=[{"role": "user", "content": goal}]
        )
        import json
        return json.loads(response.content[0].text)

    async def _delegate(self, domain: str, sub_goal: str, results: dict):
        orchestrator = self.sub_orchestrators.get(domain)
        if orchestrator:
            results[domain] = await orchestrator.run(sub_goal)
        else:
            results[domain] = f"No orchestrator for domain: {domain}"

    async def _synthesize(self, goal: str, results: dict) -> str:
        context = "\n\n".join(f"[{domain}]:\n{result}" for domain, result in results.items())
        response = await self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="Synthesize the domain-specific findings into a unified final answer.",
            messages=[{"role": "user", "content": f"Goal: {goal}\n\nDomain results:\n{context}"}]
        )
        return response.content[0].text
```

### Failure propagation in trees

The failure modes in hierarchical trees combine those of orchestrator-worker at each level. A sub-orchestrator failure affects its entire subtree but not sibling subtrees. This is a significant improvement in fault isolation over flat orchestrator-worker, where a single orchestrator failure kills everything.

However, trees introduce a new failure mode: **partial result synthesis**. If the financial sub-orchestrator succeeds but the product sub-orchestrator fails, the root orchestrator must decide: retry the failed subtree, synthesize with only partial results, or fail the entire task. This decision requires domain knowledge the root may not have. The mitigation is to define a minimum viability threshold per sub-orchestrator at task design time, not at runtime.

### When to use it

- The task decomposes into clearly bounded domains
- The total number of leaf workers exceeds 8–10
- Domain sub-orchestrators can have specialized routing and tooling
- You need fault isolation at the domain level

## 7. Topology 6: Blackboard / Shared Workspace — agents read and write a shared state object

The blackboard topology is conceptually different from the previous five. Rather than agents sending messages to each other, agents communicate indirectly through a shared mutable state object — the "blackboard." Any agent can read any part of the blackboard and write to any part it owns. Agents are triggered by changes to the blackboard rather than by direct messages.

![Blackboard / shared workspace: all agents read and write through a single versioned artifact — decoupled, but the blackboard becomes a coordination hotspot.](/imgs/blogs/multi-agent-topologies-6.webp)

### The key decoupling property

In orchestrator-worker, workers depend on the orchestrator (they receive tasks from it). In a blackboard, no agent depends on any other agent directly. The retriever writes facts to the blackboard; it does not know or care whether the reasoner will read them in 1 second or 10 minutes. The reasoner reads from the blackboard when ready; it does not know which agent wrote the facts. This temporal and structural decoupling is the topology's core property.

This maps onto a real-world analogy: a hospital whiteboard in a busy emergency department. A nurse writes a patient's vitals. A doctor reads the vitals and writes a diagnosis. A pharmacist reads the diagnosis and writes medication orders. No one is calling anyone directly — they are all reading and writing a shared artifact as they become available to do so.

### Implementation with version-stamped state

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any
import hashlib, json, time

@dataclass
class BlackboardEntry:
    key: str
    value: Any
    version: int
    author: str
    timestamp: float
    checksum: str

class Blackboard:
    def __init__(self):
        self._state: dict[str, BlackboardEntry] = {}
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list] = {}  # key -> list of (callback, min_version)

    async def write(self, key: str, value: Any, author: str) -> BlackboardEntry:
        async with self._lock:
            current = self._state.get(key)
            new_version = (current.version + 1) if current else 1
            serialized = json.dumps(value, sort_keys=True)
            checksum = hashlib.sha256(serialized.encode()).hexdigest()[:8]
            entry = BlackboardEntry(
                key=key, value=value, version=new_version,
                author=author, timestamp=time.time(), checksum=checksum
            )
            self._state[key] = entry
            await self._notify_watchers(key, entry)
            return entry

    async def read(self, key: str) -> BlackboardEntry | None:
        return self._state.get(key)

    async def read_all(self) -> dict[str, BlackboardEntry]:
        return dict(self._state)

    async def watch(self, key: str, callback, min_version: int = 0):
        """Trigger callback whenever key is updated past min_version."""
        self._watchers.setdefault(key, []).append((callback, min_version))

    async def _notify_watchers(self, key: str, entry: BlackboardEntry):
        for callback, min_version in self._watchers.get(key, []):
            if entry.version >= min_version:
                asyncio.create_task(callback(entry))

class BlackboardAgent:
    def __init__(self, agent_id: str, owned_keys: list[str], watched_keys: list[str], client):
        self.agent_id = agent_id
        self.owned_keys = owned_keys
        self.watched_keys = watched_keys
        self.client = client

    async def register(self, blackboard: Blackboard):
        for key in self.watched_keys:
            await blackboard.watch(key, self.on_update)

    async def on_update(self, entry: BlackboardEntry):
        """Override in subclass to react to blackboard changes."""
        raise NotImplementedError
```

### When blackboard shines: iterative refinement

The blackboard topology is particularly well-suited for tasks where the quality of the output improves through multiple passes by different agents, and where the order of passes is not fully determined in advance. Scientific literature review, legal document analysis, and multi-perspective debate systems all fit this pattern.

Consider a debate-style system: the "pro" agent writes an argument to the blackboard, the "con" agent reads it and writes a rebuttal, the "pro" agent reads the rebuttal and writes a counter-argument. This iterative refinement naturally terminates when a judge agent reads the full blackboard history and determines that the arguments have converged. No orchestrator needed — each agent simply reacts to the state of the shared workspace.

### The write-contention problem

The blackboard's decoupling property creates a new problem: write contention. If two agents both want to update the same key simultaneously, you have a classic distributed write conflict. Unlike orchestrator-worker, where the orchestrator serializes all state updates naturally, a blackboard with many writers needs an explicit concurrency control strategy.

Options in increasing order of complexity:
1. **Key partitioning**: each agent owns exactly the keys it writes; no two agents write the same key. Simple, but limits the topology's flexibility.
2. **Optimistic locking**: agents read a version number before writing and abort if the version has changed. Simple to implement, but requires retry logic.
3. **Transactional writes**: a lock-based transaction layer ensures atomic read-modify-write. Adds latency but prevents all races.

### Triggering strategies for blackboard agents

Blackboard agents can be triggered in two fundamentally different ways, and the choice has significant performance implications:

**Event-driven triggering**: Agents subscribe to specific blackboard keys and are awakened when those keys change. This is the most efficient model — agents are idle until the data they need is available. The implementation uses the `watch` pattern shown in the code above. The cost is that the trigger logic must be pre-defined: each agent must know which keys to watch at registration time.

**Poll-driven triggering**: Agents periodically check the blackboard for new information that is relevant to their domain. This is simpler to implement and handles cases where the set of relevant keys is not known in advance. The cost is wasted compute (agents checking an unchanged blackboard repeatedly) and latency (an agent may miss a new entry until its next polling cycle).

In practice, production blackboard systems use event-driven triggering for well-known, high-priority keys (e.g., "task status changed to failed") and poll-driven triggering for exploratory agents that are looking for any new information in a domain.

The version-stamp on each entry serves a dual purpose: it is both a conflict detection mechanism (optimistic locking) and a progress metric. An orchestrator watching the overall blackboard can determine task progress by counting entries above version 1, and can detect a stall (no new high-version entries for N seconds).

### When to use it

- Agents process at heterogeneous speeds and asynchronous scheduling is valuable
- Task structure is not fully known at dispatch time (emergent task graphs)
- You want multiple agents to contribute to a single artifact iteratively
- Auditability is critical — every version of every key is a checkpoint

## 8. Topology 7: Market / Bidding — agents bid on tasks by capability

The market topology is the most exotic of the seven. Rather than routing tasks to predetermined agents, a market-based system announces tasks to a pool of agents, collects capability scores or bids, and assigns each task to the highest bidder. Agents specialize through selection pressure rather than through hardcoded routing.

### Why markets are appealing in theory

The appeal mirrors economic market theory. In a traditional orchestrator-worker system, the orchestrator needs to know which workers are good at what — and that knowledge gets stale as workers change. In a market system, workers self-report their confidence on each task type. Over time, a worker that consistently wins bids for SQL tasks and consistently delivers good SQL will emerge as the de facto SQL specialist, without anyone hardcoding that routing.

```python
from dataclasses import dataclass
import random

@dataclass
class Bid:
    agent_id: str
    task_id: str
    confidence: float   # 0.0 – 1.0
    estimated_latency_ms: int
    estimated_cost_tokens: int

class MarketOrchestrator:
    def __init__(self, agents: list, client):
        self.agents = agents
        self.client = client
        self._performance_history: dict[str, dict] = {}  # agent_id -> {task_type: win_rate}

    async def run_task(self, task_id: str, task_description: str, task_type: str) -> str:
        # 1. Announce task to all agents, collect bids
        bids = await asyncio.gather(
            *[agent.bid(task_id, task_description, task_type) for agent in self.agents]
        )
        valid_bids = [b for b in bids if b is not None]

        if not valid_bids:
            raise RuntimeError(f"No agent bid on task {task_id}")

        # 2. Select winner based on composite score
        winner_bid = self._select_winner(valid_bids, task_type)

        # 3. Assign task to winner
        winner = next(a for a in self.agents if a.agent_id == winner_bid.agent_id)
        result = await winner.execute(task_description)

        # 4. Update performance history (for future bid weighting)
        self._update_history(winner_bid.agent_id, task_type, success=bool(result))
        return result

    def _select_winner(self, bids: list[Bid], task_type: str) -> Bid:
        """Score bids using confidence, latency, cost, and historical performance."""
        def score(bid: Bid) -> float:
            history_bonus = self._performance_history.get(
                bid.agent_id, {}
            ).get(task_type, 0.5)
            latency_penalty = bid.estimated_latency_ms / 10000.0
            cost_penalty = bid.estimated_cost_tokens / 100000.0
            return (bid.confidence * 0.5 + history_bonus * 0.3
                    - latency_penalty * 0.1 - cost_penalty * 0.1)

        return max(bids, key=score)

    def _update_history(self, agent_id: str, task_type: str, success: bool):
        h = self._performance_history.setdefault(agent_id, {})
        prev = h.get(task_type, 0.5)
        # Exponential moving average
        h[task_type] = 0.8 * prev + 0.2 * (1.0 if success else 0.0)
```

### The real-world problems

Market topologies face three production problems that make them rare in LLM systems:

1. **Starvation**: an agent with high historical performance wins all bids of a given type, starving newer agents from learning opportunities. The new agent can never improve its history because it never wins bids. This is a cold-start problem.

2. **Bid inflation**: agents learn to bid high confidence regardless of actual capability, because winning bids is rewarded. Without a truthful mechanism (where lying about capability has a cost), bids degenerate into noise.

3. **Silent drop**: an agent that wins a bid and then fails silently (timeout, exception, empty result) has wasted the task's time budget. The market needs a reputation/penalty mechanism to penalize failed winners, which adds significant complexity.

### When to use it

- Agent pool is large and dynamic (agents join and leave the pool)
- Task types are heterogeneous and hard to categorize statically
- You want the system to learn which agents are good at which tasks over time
- You have the engineering capacity to build a proper bidding mechanism with truthfulness incentives and failure penalties

For most production applications, the complexity cost of a market system is not worth it. Use orchestrator-worker with explicit routing rules, and update those rules manually as you learn which workers excel at which tasks.

### Where market topologies do work in practice

Despite the challenges, one real-world application domain genuinely benefits from market-style allocation: **heterogeneous compute pools**. When you have a fleet of workers with different hardware (some with GPUs, some CPU-only), different model sizes (7B, 13B, 70B), and different specializations (code, math, language), routing "find me the fastest available worker capable of this task" is a natural fit for a bid-based system where each worker self-reports its current queue depth, estimated latency, and task-type confidence. The "bid" is not a dollar amount — it is a tuple of (estimated latency, confidence, current load), and the dispatcher picks the best fit.

This is essentially how modern LLM inference serving systems (vLLM, SGLang) do request routing in multi-replica deployments: each replica reports its current KV-cache utilization and queue depth, and the load balancer picks the least-loaded replica that can handle the request. The key difference from a full market system is that the "task type" is simple (all requests are text generation) and the "capability" is binary (a replica can handle a request or it cannot). The moment you introduce heterogeneous task types and heterogeneous capabilities, the market complexity increases dramatically.

## 9. Choosing a topology: decision framework with 6 dimensions

The seven topologies each shine in a different part of the design space. The following decision framework walks through six dimensions that determine which topology fits a given task. Work through them in order — the first yes/no you reach is often determinative.

![Topology selection decision tree: four questions about task independence, dependency structure, scale, and control needs determine the right architecture.](/imgs/blogs/multi-agent-topologies-8.webp)

### Dimension 1: Task independence

**Question**: Can the task's subtasks execute in any order, with no data dependency between them?

- **Yes**: Go to Dimension 2. Fan-out/fan-in or market are candidates.
- **No**: The tasks have dependencies. Go to Dimension 3.

**How to test this**: Draw the dependency DAG. If any edge exists between leaf tasks (not through the root), the tasks are not independent.

### Dimension 2: Merge strategy clarity

**Question**: If tasks are independent, is the merge strategy deterministic and well-defined?

- **Yes, deterministic merge**: Fan-out/fan-in is the right choice. The merger is a pure function of the partial results.
- **No, dynamic allocation needed**: Market/bidding. Use this when the task pool is large and task types are unpredictable.
- **No clear merge needed**: Tasks produce truly independent artifacts (e.g., monitoring agents each writing to separate dashboards). Flat dispatch with no merger.

### Dimension 3: Dependency structure

**Question**: Is the dependency structure a linear chain (A→B→C) or a DAG (A→C, B→C)?

- **Linear chain**: Sequential pipeline. A linear dependency structure is the one case where pipeline's simplicity and debuggability make it the dominant choice.
- **DAG**: Go to Dimension 4.

### Dimension 4: Scale (number of leaf workers)

**Question**: How many leaf workers do you need?

- **≤ 8 workers**: Simple orchestrator-worker. One orchestrator can manage 8 workers without context overflow or routing degradation.
- **9–30 workers with clear domain boundaries**: Hierarchical tree. Sub-orchestrators per domain.
- **> 30 workers or highly dynamic**: Go to Dimension 5.

### Dimension 5: Coordination pattern

**Question**: Is the task better characterized by message-passing or shared state?

- **Message-passing**: Iterative back-and-forth, debate, critique, collaborative writing → Peer-to-peer (with caution) or hierarchical.
- **Shared state**: Multiple agents contributing to a single document, scientific analysis, legal review → Blackboard.

### Dimension 6: Operational requirements

**Question**: What are the operational constraints?

| Constraint | Favors |
|---|---|
| Strict latency SLA (< 3 seconds) | Fan-out/fan-in |
| Maximum debuggability | Pipeline or orchestrator-worker |
| Maximum fault tolerance | Hierarchical tree or P2P |
| Budget constraints (minimize token spend) | Pipeline (no orchestrator overhead) |
| Dynamic task pool (agents join/leave) | Market/bidding or P2P |
| Regulatory audit requirements | Pipeline (full artifact trail) or Blackboard (full version history) |

![Topology comparison across six dimensions: no single topology dominates — each trades off 2-3 dimensions against the others.](/imgs/blogs/multi-agent-topologies-7.webp)

The matrix above summarizes the tradeoffs across all seven topologies and six dimensions. No topology dominates; every choice involves real sacrifices. The key insight is that **debuggability and fault isolation trade off against each other**: topologies with high fault isolation (P2P, hierarchical) have distributed state that is harder to trace; topologies with high debuggability (pipeline, orchestrator-worker) have centralized coordination that is a single point of failure.

## 10. Failure modes specific to each topology (and mitigations)

Every topology has a characteristic failure signature. Understanding these signatures in advance lets you design mitigations before they fire in production.

![Failure modes matrix: each topology's primary and secondary failure signatures across five failure categories.](/imgs/blogs/multi-agent-topologies-9.webp)

### Orchestrator-Worker: the cascade failure

**Failure mode**: The orchestrator crashes, times out, or enters a reasoning loop mid-task. All in-progress workers have their results orphaned — they completed successfully, but no one is collecting the results.

**Mitigation**:
- Persist orchestrator state (task queue, in-progress results, routing decisions) to an external store before each state transition
- Implement a checkpointed orchestrator: on restart, it reads the last persisted state and resumes from the last completed task
- Set a budget limit: if the orchestrator has been running for more than X seconds or Y LLM calls, emit a partial result rather than continuing indefinitely

```python
# Pattern: persisted orchestrator state
class ResumableOrchestrator:
    def __init__(self, state_store, client):
        self.state_store = state_store  # Redis, DynamoDB, etc.
        self.client = client

    async def run(self, task_id: str, goal: str) -> str:
        state = await self.state_store.get(task_id) or {"phase": "plan", "results": {}}

        if state["phase"] == "plan":
            state["tasks"] = await self._plan(goal)
            state["phase"] = "dispatch"
            await self.state_store.set(task_id, state)

        if state["phase"] == "dispatch":
            remaining = [t for t in state["tasks"] if t["id"] not in state["results"]]
            new_results = await asyncio.gather(*[self._dispatch(t) for t in remaining])
            state["results"].update({r.task_id: r.content for r in new_results if r.success})
            state["phase"] = "merge"
            await self.state_store.set(task_id, state)

        if state["phase"] == "merge":
            result = await self._merge(goal, state["results"])
            await self.state_store.delete(task_id)
            return result
```

### Sequential Pipeline: error amplification

**Failure mode**: A factual error at stage 1 is presented as ground truth to stage 2. The error compounds through successive stages. The final output is confidently wrong.

**Mitigation**:
- Add an explicit validation stage after each substantive stage (not just a format check, but a semantic/factual check)
- For high-stakes pipelines, run a "red team" stage that specifically looks for errors rather than improving the artifact
- Log each artifact version and the diff between versions — anomalously large diffs often signal an upstream error was corrected downstream, which can flag the original corruption

### Fan-out / Fan-in: the slow worker problem

**Failure mode**: One worker out of four takes 10× longer than the others (due to an API timeout, a large input, a reasoning loop). The wall-clock latency for the whole task becomes dominated by this one slow worker, eliminating most of the parallelism benefit.

**Mitigation**:
- Implement a timeout and **speculative redundancy**: if any worker exceeds the 90th-percentile expected latency, launch a duplicate request to a different worker for the same sub-task. Use whichever result arrives first.
- Design sub-tasks to be roughly equal in complexity (difficulty calibration at decomposition time)
- For the tail case, implement **partial merge**: if N-1 of N workers have completed, merge what you have, mark the result as "partial", and continue asynchronously

### Peer-to-Peer: deadlock and split-brain

**Failure mode 1 — Deadlock**: Agent A is waiting for B's response to proceed; B is waiting for A's response to proceed. Neither has any way to detect this.

**Mitigation**: Assign a per-task timeout to every agent. If an agent hasn't received a required response within the timeout, it proceeds with a "no response received" annotation rather than blocking indefinitely. A watchdog process detects when no agent has made progress for N seconds and forces a task reset.

**Failure mode 2 — Split-brain**: Two agents derive contradictory conclusions. Without a designated arbitrator, the system has no mechanism to resolve the contradiction. Both conclusions propagate to downstream agents.

**Mitigation**: Designate one agent per topic as the "authority". When other agents derive conclusions on that topic, they submit them to the authority agent, which resolves conflicts and issues the canonical conclusion. This is a partial reintroduction of hierarchy into the mesh — which is the usual path from P2P back toward sanity.

### Hierarchical Tree: partial result synthesis

**Failure mode**: A sub-orchestrator in one domain succeeds while another fails. The root orchestrator must synthesize a final answer from incomplete domain results — but it does not have enough domain knowledge to know how incomplete the result is or which domain's contribution was most critical.

**Mitigation**: Each sub-orchestrator declares its result with a **confidence and criticality score**. The root's synthesis prompt includes: "The following domain results are available. Note that [domain X] failed; its typical contribution is [description]. Synthesize the best available answer, clearly flagging what is missing."

### Blackboard: write contention and staleness

**Failure mode**: Two agents both write to the same blackboard key nearly simultaneously. The second write overwrites the first. Agent C reads the blackboard and gets the second write, never seeing the first write's content. The first agent's work is silently lost.

**Mitigation**:
- Enforce key ownership (each key has exactly one writer)
- For keys with multiple contributors, use a list-append model (each agent appends to a list rather than overwriting)
- Version all writes and require readers to explicitly request a minimum version number (readers that want the latest facts request version ≥ current; readers that want a stable snapshot specify a pinned version)

### Market / Bidding: starvation and silent drop

**Failure mode 1 — Starvation**: High-performing agents win all bids. New or recovering agents never get tasks to build reputation from. The market converges to a monopoly.

**Mitigation**: Implement an exploration bonus: with probability $\epsilon$ (e.g., 10%), assign tasks to a random agent rather than the highest bidder. This is the multi-armed bandit explore-exploit balance.

**Failure mode 2 — Silent drop**: An agent wins a bid, crashes after starting, and returns no result. The task's time budget is consumed.

**Mitigation**: Require winning agents to post a "heartbeat" every N seconds. If the heartbeat stops, the market reassigns the task to the second-highest bidder. Deduct a reputation penalty from the silent-drop agent.

## 11. Dynamic topology switching: when to restructure at runtime

Static topology selection is the starting point, but sophisticated multi-agent systems can change topologies at runtime in response to observed conditions.

![Dynamic topology switching: a pipeline detects a stage-3 bottleneck at t=4s and switches to a fan-out, cutting latency 2× on the remaining work.](/imgs/blogs/multi-agent-topologies-10.webp)

### When to switch

Three conditions justify runtime topology switching:

1. **Bottleneck detection**: A monitoring agent detects that one stage in a pipeline is taking 5× longer than expected. The rational response is to switch that stage to fan-out, parallelizing the slow work across multiple sub-agents.

2. **Workload characteristic change**: A task that started as a sequential document analysis (pipeline) discovers partway through that one section requires deep cross-referencing across 200 documents (fan-out). The task's dependency structure has changed.

3. **Agent failure**: In a flat orchestrator-worker topology, if the orchestrator fails and a hot standby is not available, a graceful degradation is to promote one of the workers as a temporary sub-orchestrator — a spontaneous hierarchical tree.

### How to implement topology switching

```python
from enum import Enum

class TopologyState(Enum):
    PIPELINE = "pipeline"
    FAN_OUT = "fan_out"
    ORCHESTRATOR_WORKER = "orchestrator_worker"

class AdaptiveController:
    def __init__(self, initial_topology: TopologyState):
        self.current_topology = initial_topology
        self.stage_latencies: list[float] = []
        self.BOTTLENECK_THRESHOLD_FACTOR = 3.0  # stage is 3× slower than average

    def observe_stage_latency(self, stage_index: int, latency_s: float):
        self.stage_latencies.append(latency_s)

    def should_switch(self) -> tuple[bool, TopologyState | None]:
        if len(self.stage_latencies) < 2:
            return False, None

        avg = sum(self.stage_latencies) / len(self.stage_latencies)
        max_latency = max(self.stage_latencies)
        max_idx = self.stage_latencies.index(max_latency)

        if max_latency > avg * self.BOTTLENECK_THRESHOLD_FACTOR:
            if self.current_topology == TopologyState.PIPELINE:
                # Switch the slow stage to fan-out
                return True, TopologyState.FAN_OUT

        return False, None

    def execute_switch(self, new_topology: TopologyState, current_state: dict) -> dict:
        """Migrate in-flight task state from current topology to new topology."""
        self.current_topology = new_topology
        # Return the state bundle for the new topology's executor
        return {
            "topology": new_topology.value,
            "carried_state": current_state,
            "switch_reason": "bottleneck_detected",
            "switch_timestamp": time.time()
        }
```

### The cost of switching

Runtime topology switching adds significant complexity. You need to:

1. Serialize the current task state in a format the new topology can consume
2. Decide what to do with in-progress work in the old topology (abort? incorporate?)
3. Handle the race condition where the switch itself takes time, during which the old topology is winding down and the new one hasn't started yet

The switching cost is real — typically 1–3 additional LLM calls to replanning + state transfer overhead. Switching only makes sense when the expected savings from the new topology exceed this cost for the remaining work. For a task that is 90% complete, switching topologies is almost never worth it; for a task that has just hit a bottleneck at 20% completion, it usually is.

A practical heuristic: switch if `(remaining_work_fraction × latency_improvement_factor) > (switch_cost_in_seconds / remaining_work_seconds)`.

## 12. Production case studies

### Case study 1: Perplexity-style search synthesis

**Context**: A real-time web search assistant needs to answer factual questions by searching the web, reading multiple sources, and synthesizing a cited response — all within a 5-second target latency.

**Topology chosen**: Fan-out/fan-in.

**Rationale**: The query is decomposed into 3–5 independent search queries (one per sub-topic). Each search worker calls the search API, fetches the top-3 results, and returns a summarized context block. A merger agent synthesizes across the blocks and generates the final response with citations.

**What went wrong in v1**: The merger agent was running on the same large model as the orchestrator. When 5 search workers returned their results simultaneously, the merger's context window was flooded with ~15,000 tokens of search snippets. The merger started losing coherence on early citations. The fix was to add a **compression stage**: each worker's summarizer reduced its 3,000-token snippet block to a 400-token compressed summary before the merge. The merger's effective input dropped from 15,000 tokens to 2,000, and citation quality recovered.

**Throughput achieved**: 40 concurrent search tasks/second at p95 latency of 4.1 seconds (vs. 14.2 seconds for a sequential pipeline equivalent).

### Case study 2: Legal contract review

**Context**: A legal tech platform reviews commercial contracts for risk clauses. Contracts average 80 pages. Review requires checking 12 risk categories (indemnification, IP assignment, non-compete, data privacy, etc.) against a company-specific policy document.

**Topology chosen**: Sequential pipeline with fan-out within each stage.

**Structure**:
1. Stage 1 (Parser): Extract structured clause inventory from the contract
2. Stage 2 (Fan-out): 12 specialist agents check one risk category each, in parallel
3. Stage 3 (Merger): Combine 12 risk assessments into a unified risk report
4. Stage 4 (Writer): Generate executive summary and remediation recommendations

**What went wrong in v1**: The pipeline used a single Pipeline topology (linear). With 80-page contracts, stage 1 alone took 45 seconds to extract the clause inventory. Only then could stage 2 begin. Target SLA was 60 seconds total. Fix: added streaming between stages — stage 2's specialist agents began as soon as each section's clause inventory was extracted, rather than waiting for the full document. The stage 1 agent emitted structured events as it processed each contract section; stage 2 agents subscribed to these events and began processing immediately. Total time dropped from 65 seconds to 28 seconds.

**Lesson**: Pipelining stages does not mean you must wait for the full stage-1 output. Streaming fan-out — where stage 2 begins as stage 1 produces partial output — is a hybrid topology that captures benefits of both pipeline and fan-out.

### Case study 3: Multi-perspective debate system

**Context**: A research platform wants to evaluate product decisions by having AI agents argue multiple perspectives (technical feasibility, market fit, cost, regulatory risk) and then synthesize a recommendation.

**Topology chosen**: Blackboard.

**Structure**: Each perspective agent (technical, market, cost, regulatory) reads the current state of the shared argument document and adds its contribution. A moderator agent monitors the blackboard and detects when the argument has become repetitive (by comparing embedding distance between consecutive contributions). A synthesis agent is triggered when the moderator declares the debate concluded.

**What went wrong in v1**: The cost agent and the market agent both tried to claim ownership of "pricing strategy" — a topic that genuinely straddled both domains. With no key ownership defined, both agents wrote to the same blackboard section and overwrote each other. Fix: added a **claim-before-write** protocol. Before writing to a key, an agent must post a 5-second hold on that key. If another agent has already posted a hold on the same key, the second agent writes to a sibling key (e.g., `pricing_strategy_cost` and `pricing_strategy_market`) and the synthesis agent is instructed to merge them.

**Result quality**: The debate system produced recommendations rated by human evaluators as 23% higher quality than a single-agent analysis on the same decisions, at 4.7× the token cost.

### Case study 4: Code generation with self-repair

**Context**: An automated software engineering agent receives a feature specification and must produce working, tested code that passes a CI suite.

**Topology chosen**: Hierarchical tree with a dynamic feedback loop.

**Structure**:
- Root: Task manager (receives spec, manages overall progress)
- Sub-orchestrator A: Design team (architecture agent + interface definition agent)
- Sub-orchestrator B: Implementation team (frontend agent + backend agent + database agent)
- Sub-orchestrator C: QA team (unit test writer + integration test runner + static analyzer)

**The twist**: On CI failure, sub-orchestrator C reads the failing test output and routes the failure back to sub-orchestrator B's relevant agent. This creates a repair loop that operates within sub-orchestrator B — the root orchestrator only knows "CI failed; QA subtree triggered repair cycle N."

**Benchmark**: On the SWE-bench lite dataset, this hierarchical approach with an internal repair loop achieved 42% pass rate vs. 31% for a flat single-agent approach on the same set of tasks.

**Why the hierarchy mattered**: When the repair loop was implemented at the root level (root orchestrator reads CI failure, routes to workers), the orchestrator's context grew on every repair cycle, eventually triggering context overflow and losing track of the original specification. Moving the repair loop to the sub-orchestrator level bounded the context growth — each repair cycle only needed the failed tests and the relevant implementation files, not the full task history.

### Case study 5: Customer support triage

**Context**: A B2B SaaS company routes customer support tickets to specialist agents (billing, technical, account management, product feedback) and must reply within 4 hours. Ticket volume peaks at 500/hour.

**Topology chosen**: Orchestrator-worker with capability-based routing.

**Structure**: A triage orchestrator classifies each ticket into one of 8 categories, routes it to the appropriate specialist worker, monitors the worker's draft, and optionally escalates to a human agent if the worker expresses low confidence.

**The failure mode they hit**: The triage orchestrator was a single instance. At peak load (500 tickets/hour = 8.3/minute), the orchestrator became the bottleneck. A single triage LLM call took ~1.5 seconds; at 8.3 tickets/minute, the orchestrator had a utilization of 8.3 × 1.5 / 60 = 20.8% — fine. But the orchestrator also did result quality-checking, which added another 1.5 seconds per ticket. At 100% utilization, tickets began queuing.

**Fix**: Decomposed the orchestrator into two roles — a lightweight **classifier** (50ms, small model, text classification only) and a separate **quality reviewer** (1.5s, larger model, full review). The classifier ran in a thread pool with 10 parallel instances. The quality reviewer ran as a separate asynchronous stage. The bottleneck was eliminated, and the 4-hour SLA was maintained at 3× the original ticket volume.

### Case study 6: Financial report generation

**Context**: A fintech company generates quarterly performance reports for 500 client portfolios. Each report requires: fetching portfolio data, running 15 financial calculations, generating narrative commentary, and formatting a 20-page PDF.

**Topology chosen**: Fan-out at the portfolio level + sequential pipeline within each portfolio.

**Why not pure fan-out**: The 15 calculations within a portfolio have sequential dependencies (you need the beta calculation before you can compute alpha, etc.). Pure fan-out across all 500 portfolios was straightforward — they are entirely independent. Within each portfolio, a pipeline was the natural fit for the sequential calculation chain.

**The composition**: 500 portfolio fan-out workers, each running a 5-stage internal pipeline (data fetch → calculations → risk metrics → narrative → formatting).

**Key engineering decision**: The 500 workers would have overwhelmed both the LLM API rate limits and the portfolio data API. The fan-out was throttled to a rolling window of 30 concurrent workers at a time, with new workers launched as existing ones complete. This is a **sliding window fan-out**, not a full fan-out — a critical distinction. A naive full fan-out for 500 workers would hit 429 rate-limit errors and require complex retry backoff, negating the latency benefit. The sliding window kept utilization high without overload.

**Total runtime**: 500 portfolios × 45s average (pipeline) / 30 concurrent = 750 seconds total. A fully sequential approach would take 500 × 45s = 22,500 seconds. The 30× speedup met the business requirement of completing all reports before market open.

## 13. When to stay with a single agent / when to go multi-agent

The single most important advice in this entire post is this: **multi-agent is not an upgrade; it is a tradeoff**. You are trading simplicity, debuggability, and coordination-free execution for parallelism, specialization, and scale. Every one of the seven topologies in this post adds coordination complexity that a single agent simply does not have.

The right default is a single agent. Move to multi-agent when you have hit a specific, measurable limit of the single-agent approach.

### Stay with a single agent when:

**The task fits in a context window with headroom.** If your task requires 40,000 tokens of context and your model supports 128,000 tokens, the context pressure argument does not apply. Splitting into multiple agents adds overhead without necessity.

**Latency is not the constraint.** If the user can wait 30 seconds for a high-quality answer, and a single agent can produce it in 20 seconds, the parallel fan-out that would deliver the same answer in 8 seconds is not worth the engineering complexity.

**The task is not decomposable.** Many tasks are inherently sequential and holistic — writing a coherent essay, debugging a complex program, analyzing a nuanced ethical dilemma. Forcing these into a multi-agent topology fragments the reasoning in ways that degrade quality. A single agent reasoning across the full problem often outperforms a pipeline of agents each handling a "phase."

**The team lacks distributed systems expertise.** A multi-agent system is a distributed system. If your team does not have experience with message queues, distributed state, idempotency, retries, and timeout handling, a multi-agent system will produce incidents at 2 AM that no one knows how to debug. A single agent that fails is a single failure to diagnose. A multi-agent system that fails has 7 possible places to start looking, 12 interaction patterns to check, and 3 race conditions that only appear under load.

**The task is exploratory.** When you are still figuring out what the task even is, multi-agent architecture is premature optimization. Build the single-agent solution first. Let it fail in concrete, observable ways. Those failures will tell you exactly which multi-agent topology is warranted.

### Move to multi-agent when:

**You have measured context window degradation.** Run your single agent. Check whether it correctly recalls facts from the first 20% of context in a 100k-token run. If recall has dropped noticeably, context pressure is real and specialization is warranted.

**Latency is a measured bottleneck.** Profile the single-agent run. Find the serial stages that could be parallelized. If parallelizing them would meaningfully reduce latency for your SLA, the fan-out investment is justified.

**Specialization quality is measured and significant.** Run your generalist agent against your domain task. Run a specialist agent (better system prompt, domain-specific tools, maybe a fine-tuned model) against the same task. If the specialist's quality gain is meaningful, specialization at scale justifies the orchestration overhead.

**You need heterogeneous tools or APIs.** Some tasks inherently require mixing different types of tools (browser automation, database queries, code execution, external API calls). A single agent model handles all of these, but the context management for 5 different tool types can be messy. A multi-agent system with one agent per tool class produces cleaner context management and easier individual debugging.

**Independent fault isolation is a business requirement.** In regulated industries, you may need to demonstrate that the billing calculation agent and the customer data processing agent have no shared state and cannot affect each other's behavior. A multi-agent topology with strict isolation boundaries is the right solution — not for performance, but for compliance.

### The compounding cost of premature multi-agent architecture

There is a specific pattern that kills multi-agent projects before they ship: the team designs a 6-agent hierarchical system from day one, before any single-agent baseline exists, because "we know it will need to scale." They spend four weeks building orchestration infrastructure, message buses, and deployment pipelines. On week five, they discover that the core task — the thing the agents are supposed to do — does not actually decompose cleanly into the six assumed subtasks. Two of the agents are nearly redundant; one agent's output is never actually consumed. The architecture now needs a redesign, but four weeks of infrastructure work are now sunk costs that the team is reluctant to abandon.

The single-agent baseline is not just a starting point — it is a requirements-gathering tool. Running a single agent on real tasks tells you which parts of the task actually benefit from specialization (the agent struggles with SQL but excels at prose), which parts are actually sequential versus parallelizable (you thought it was parallel; it turns out step B always needs step A's output), and what the realistic token budget per task looks like (you assumed 2,000 tokens per task; it's actually 12,000, which changes the economic case for the architecture entirely).

Build the single-agent baseline. Run it on 50 real tasks. Document its failure modes. Then design the multi-agent architecture that addresses exactly those failure modes — not the ones you assumed you would have.

### The litmus test

Before committing to a multi-agent topology, answer these three questions:

1. **Which specific, measurable limit of the single-agent approach are we hitting?** If you cannot name it (context overflow, latency, specialization quality, fault isolation), you do not yet have a compelling case.

2. **Which topology addresses that limit with the least added complexity?** Orchestrator-worker before hierarchical tree; pipeline before blackboard; fan-out before market.

3. **Do we have the observability infrastructure to debug a distributed system?** Structured logging per agent, correlation IDs that span agent boundaries, and dashboards that show per-agent latency and error rates. Without these, multi-agent systems fail silently in production.

The sweet spot is not "as many agents as possible" — it is "as few agents as necessary to meet the measured requirements." Teams that get this right build systems that are simpler than they look. Teams that get it wrong build systems that are more complex than anyone can reason about.

The seven topologies in this post are not a menu from which you pick one and you are done. They are building blocks. Real production systems often compose them: a fan-out at the top level, sequential pipelines within each worker, a blackboard for shared intermediate state, and a hierarchical sub-tree for a domain that grew too large for flat orchestration. The topology you ship on day one will not be the topology you run on day 180. Design for evolution: clean interfaces between agents, explicit state contracts at each topology boundary, and observability from the start. The architecture that can adapt as you learn is always more valuable than the architecture that was "theoretically optimal" on the whiteboard.

### Case study 7: Real-time incident response

**Context**: A platform engineering team builds an automated incident response system. When a production alert fires, the system must simultaneously investigate multiple possible causes, draft a status page update, page the relevant on-call engineer, and begin remediation steps — all within 90 seconds.

**Topology chosen**: Parallel fan-out with a time-boxed merger.

**Structure**: An alert router (the orchestrator) receives the PagerDuty webhook. It fans out to four parallel workers:
1. **Diagnostics worker**: queries Datadog, CloudWatch, and application logs for the previous 5 minutes
2. **Correlation worker**: searches a vector database of past incidents for similar signatures
3. **Status worker**: drafts a status page update based on the alert title and severity
4. **Runbook worker**: fetches the relevant runbook steps from Confluence

The merger waits up to 30 seconds for all four workers. If any worker exceeds 30 seconds, the merger proceeds with what it has, flagging the missing sections.

**The time-box insight**: Without a hard time-box, the merger would wait for the slowest worker. Datadog and CloudWatch API calls occasionally take 25–30 seconds under load. The status page update and runbook fetch are typically done in 3–5 seconds — fast enough to publish even if the diagnostic data arrives late. Separating the "fast path" (status + runbook) from the "slow path" (diagnostics + correlation) and publishing the fast path immediately reduced median time-to-status-page from 87 seconds to 11 seconds, which dramatically reduced inbound support volume during incidents.

**What they got wrong initially**: The first version used a single status worker that waited for the diagnostics worker before drafting the status update, because the template read "Based on the following diagnostic data: [...]". The status page was always delayed until the slowest diagnostic call completed. The fix was to split the status update into two rounds: a first round from just the alert metadata (fast), and a second round that updated the status page with diagnostic context when the slow worker completed.

**Lesson**: Fan-out is most effective when you design for **tiered freshness** — ship what you have immediately, update it with richer data as it arrives. Architectures that require full fan-in before publishing anything waste the low-latency potential of the topology entirely.

### Production observability: the missing prerequisite

Every case study above has one thing in common: the engineering team built the observability infrastructure before (or very shortly after) the multi-agent system went to production. Without observability, multi-agent systems fail in ways that are nearly impossible to diagnose:

**Structured per-agent logging with correlation IDs.** Every log line emitted by any agent in any topology must include a `task_id`, `agent_id`, and `span_id`. These three fields are what allow you to reconstruct the full execution trace of a multi-agent task from distributed logs. Without them, you have a log pile with no way to determine which log lines belong to the same task invocation.

```python
import uuid
import structlog

class AgentLogger:
    def __init__(self, agent_id: str, task_id: str):
        self.log = structlog.get_logger().bind(
            agent_id=agent_id,
            task_id=task_id,
            span_id=str(uuid.uuid4())[:8]
        )

    def info(self, event: str, **kwargs):
        self.log.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        self.log.error(event, **kwargs)

    def latency(self, operation: str, latency_ms: float):
        self.log.info("latency", operation=operation, latency_ms=latency_ms)
```

**Per-agent latency histograms.** You need to know — per agent, per topology position — the p50, p90, and p99 latency. A worker that averages 1.2 seconds but occasionally hits 15 seconds (p99) will intermittently dominate the wall-clock latency of any topology it participates in. You cannot optimize what you cannot see.

**Token usage per agent and per task.** In orchestrator-worker and hierarchical tree topologies, the orchestrator's token consumption grows with task complexity. Token usage that grows super-linearly with task size is a leading indicator of a context-overflow failure waiting to happen. Alert on it.

**Cross-agent message tracing.** In any topology with inter-agent messaging (pipeline, P2P, blackboard), you need distributed tracing that shows the full message-passing graph for each task. OpenTelemetry with a W3C trace context propagated through every agent message gives you this with standard tooling.

The teams that successfully run multi-agent systems in production are not the teams with the most sophisticated topologies — they are the teams that can answer "why did that task fail at 2:13 AM" within five minutes of being paged.

---

## Cross-links

For the patterns that agents use to call other agents as tools, see [Agent as Tool Pattern](/blog/machine-learning/ai-agent/agent-as-tool-pattern). For the coordination and shared-state primitives that underpin multi-agent communication, see [Shared State and Coordination in Multi-Agent Systems](/blog/machine-learning/ai-agent/shared-state-and-coordination). For a comparison of orchestration frameworks (LangGraph, AutoGen, CrewAI) and how they map to the topologies in this post, see [Orchestration Frameworks Compared](/blog/machine-learning/ai-agent/orchestration-frameworks-compared). For the ReAct loop that individual agents typically run inside any of these topologies, see [ReAct Pattern Deep Dive](/blog/machine-learning/ai-agent/react-pattern-deep-dive).
