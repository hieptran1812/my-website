---
title: "Shared State and Coordination: How Multi-Agent Systems Stay in Sync"
date: "2026-06-27"
description: "How multiple agents share and coordinate over state — message passing, shared memory, consensus, conflict resolution, and the consistency guarantees you actually need in LLM-based systems."
tags: ["ai-agents", "multi-agent", "coordination", "shared-state", "concurrency", "llm", "machine-learning", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 39
---

When two agents work on the same problem without knowing what the other is doing, you don't get twice the throughput — you get a mess. One agent researches a topic the other already covered. Another writes a conclusion that contradicts the first agent's findings. A third schedules the same API call three times in parallel, blowing through rate limits. Without coordination, multi-agent systems are just parallel chaos generators dressed up with LLM logos.

This post is about the concrete mechanisms that prevent that: how agents share state, how they communicate, what happens when two agents write to the same key at the same time, and which consistency guarantees you actually need versus which ones you're paying for unnecessarily. We'll go from first principles through production-grade case studies, covering every major pattern in the design space.

For the topology-level view of how agents are arranged, see [Multi-Agent Topologies](/blog/machine-learning/ai-agent/multi-agent-topologies). For the taxonomy of what agents remember individually, see [Agent Memory Taxonomy](/blog/machine-learning/ai-agent/agent-memory-taxonomy). This post sits between those two: it's specifically about the coordination layer — the protocols and data structures that make agents a coherent system rather than a collection of independent processes.

---

## 1. The Coordination Problem: Why Agents That Don't Communicate Produce Incoherent Results

Let's start with a concrete failure before we discuss solutions.

Suppose you have three agents working on a report: a Researcher, a Writer, and a Fact-Checker. You want them to work in parallel to save time. Without any shared state:

- The Researcher fetches five sources and begins summarizing.
- The Writer, not knowing what the Researcher found, begins drafting from its own context window.
- The Fact-Checker, operating independently, fetches two of the same sources the Researcher already fetched.

The Writer finishes a paragraph. The Researcher writes a different version of the same paragraph into its own output. The Fact-Checker catches an error — but in the Researcher's version, not the Writer's version.

At merge time, you have three inconsistent drafts, duplicated network calls, and no obvious way to reconcile the conflicting text. The final output either picks one arbitrarily (silently dropping the others) or produces a Frankenstein concatenation that makes no sense.

This isn't a pathological example. It's the default behavior of any multi-agent system that doesn't explicitly solve coordination. The failure modes are:

**Duplicate work.** Multiple agents independently execute the same computation or I/O. In LLM systems this means wasted API tokens, redundant tool calls, and duplicated latency.

**Conflicting writes.** Two agents write different values to the same logical key. The final value depends on write ordering, which is nondeterministic in a distributed system.

**Stale reads.** Agent B reads state that Agent A has already invalidated. Agent B's downstream reasoning is based on a premise that is no longer true.

**Missing coordination points.** Agent C depends on a result that Agent A hasn't produced yet, but no one told Agent C to wait. It proceeds with a null or default and produces nonsense.

**Cascading inconsistency.** Agent A's error propagates to Agent B's output, which then becomes input to Agent C. By the time you see the wrong final answer, the root cause is three hops back and invisible.

The coordination problem is fundamentally a distributed systems problem. You have multiple processes (agents), each with their own local state, and you need to agree on a shared version of reality. The solutions are the same ones distributed systems engineers have been building for thirty years: message passing, shared memory, event buses, consensus protocols, locking, and CRDTs. The difference in LLM-based systems is that the "processes" are probabilistic, their I/O is expensive and slow, and they often need to handle natural language rather than typed data structures.

Understanding which coordination mechanism to use — and when you need none at all — is the central engineering skill for multi-agent system designers.

---

## 2. State Sharing Patterns: Message Passing, Shared Memory, Event Bus, Blackboard

There are four fundamental patterns for how agents share state. Each makes a different trade-off between coupling, latency, and consistency.

![Four state-sharing patterns showing message-passing, shared-memory, event-bus, and blackboard topologies](shared-state-and-coordination-1.png)

### Pattern 1: Message Passing (Point-to-Point)

Agents communicate directly, passing structured messages. Agent A knows Agent B's address and sends it a message. Agent B processes the message and optionally sends a reply.

**Coupling:** Tight. Agent A must know Agent B exists and understand its message schema.

**Latency:** Low for small systems. Grows as O(N²) as you add more agents and point-to-point connections.

**Consistency:** High per-pair. Each agent-to-agent channel can have sequence numbers, ACKs, and delivery guarantees. But global consistency requires every pair to be consistent.

**Best for:** Small systems (2–4 agents) with well-defined interfaces. Request/response patterns where one agent explicitly depends on another's output.

### Pattern 2: Shared Memory (Blackboard Variant)

All agents read and write to a central store. An agent that wants to share something writes it; agents that need it read it.

**Coupling:** Loose. Agents don't need to know each other's addresses, only the schema of the shared store.

**Latency:** Bounded by the store's read/write latency. Redis gives you sub-millisecond reads; a Postgres row gives you single-digit milliseconds.

**Consistency:** Determined by the store. A Redis cluster with strong consistency mode gives you linearizable reads. A NoSQL store with eventual consistency gives you... eventual consistency.

**Best for:** Medium systems (4–12 agents) where agents produce and consume results asynchronously. The shared store becomes the single source of truth.

### Pattern 3: Event Bus (Publish/Subscribe)

Agents publish events to a bus. Other agents subscribe to event types they care about. When an event fires, all subscribers receive it independently.

**Coupling:** Minimal. Publishers and subscribers don't know each other exist. You can add subscribers without modifying publishers.

**Latency:** Slightly higher than direct message passing due to bus routing overhead. Kafka can handle millions of events per second at ~10ms latency.

**Consistency:** Ordered delivery is a configuration choice. Kafka partitions give you ordered delivery within a partition. Across partitions, ordering is not guaranteed.

**Best for:** Large systems (8+ agents) where many agents need to react to the same events. Fan-out patterns where one event should trigger multiple downstream agents.

### Pattern 4: Blackboard (Cooperative Writing)

A specialized form of shared memory where agents cooperatively annotate a shared artifact (the "blackboard"). No agent owns the blackboard. Any agent can write to it and any agent can read from it. The blackboard accumulates partial results until the task is complete.

**Coupling:** Minimal coupling to the blackboard schema. Agents contribute what they know and consume what they need.

**Latency:** Same as shared memory. The bottleneck is usually write contention, not read latency.

**Consistency:** The challenge is that multiple agents writing concurrently will conflict. The blackboard pattern requires a conflict resolution strategy.

**Best for:** Tasks where the final result is built incrementally from multiple partial contributions (document assembly, research synthesis, multi-step planning).

---

## 3. Message Passing: Direct Agent-to-Agent Communication

Message passing is the oldest coordination primitive and the one most developers reach for first. In LLM-based systems it's often implemented as function calls between agents, or as structured prompts where Agent A passes context to Agent B's system prompt.

![Message passing flow: Agent A formats message, sends to bus, Agent B receives, processes, returns ACK](shared-state-and-coordination-2.png)

### Message Schemas

The most important thing about message passing isn't the transport — it's the schema. A message without a well-defined schema is just an unstructured string that might break its receiver.

A minimal message schema for a multi-agent system looks like this:

```python
from pydantic import BaseModel
from typing import Any, Literal
import uuid
from datetime import datetime

class AgentMessage(BaseModel):
    message_id: str = str(uuid.uuid4())
    sender_id: str
    recipient_id: str
    message_type: Literal["task", "result", "error", "ack", "query"]
    payload: dict[str, Any]
    correlation_id: str | None = None  # links replies to original requests
    sequence_num: int | None = None    # for ordered channels
    timestamp: datetime = datetime.utcnow()
    version: str = "1.0"
```

The `correlation_id` is critical for request/response patterns. When Agent B replies to Agent A's request, it includes the original `message_id` as `correlation_id`. Agent A uses this to match replies to outstanding requests.

The `sequence_num` allows the receiver to detect gaps and reorder out-of-order delivery. If Agent A sends messages 1, 2, 3 and Agent B receives them in order 1, 3, 2, it can buffer message 3 until message 2 arrives.

### Ordering Guarantees

In a distributed system, message ordering is not free. You have to choose your ordering model:

**FIFO (First-In-First-Out) per sender:** Messages from Agent A to Agent B are delivered in the order Agent A sent them. Messages from Agent C to Agent B may interleave with Agent A's messages.

**Total ordering:** All messages are delivered to all recipients in a globally consistent order. Expensive — requires distributed coordination.

**Causal ordering:** If Agent A sends message M1 which causes Agent B to send message M2, then any agent that receives M2 also receives M1 first. This is the weakest ordering that preserves cause-and-effect.

For most LLM-based systems, causal ordering is sufficient. You rarely need total ordering (which would serialize all agent activity), and FIFO per sender is often too weak (it doesn't preserve causality across agents).

### Implementing Message Passing with a Queue

In practice, direct agent-to-agent TCP connections don't scale. You want a message queue in between:

```python
import asyncio
from collections import defaultdict
from asyncio import Queue

class MessageBus:
    def __init__(self):
        self._queues: dict[str, Queue] = defaultdict(Queue)
        self._sequence: dict[str, int] = defaultdict(int)

    async def send(self, message: AgentMessage) -> None:
        self._sequence[message.sender_id] += 1
        message.sequence_num = self._sequence[message.sender_id]
        await self._queues[message.recipient_id].put(message)

    async def receive(self, agent_id: str, timeout: float = 30.0) -> AgentMessage:
        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"No message for {agent_id} in {timeout}s")

    async def send_and_wait(
        self,
        message: AgentMessage,
        sender_queue: Queue,
        timeout: float = 60.0
    ) -> AgentMessage:
        await self.send(message)
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                reply = await asyncio.wait_for(
                    sender_queue.get(),
                    timeout=deadline - asyncio.get_event_loop().time()
                )
                if reply.correlation_id == message.message_id:
                    return reply
                # Put back unrelated messages
                await sender_queue.put(reply)
            except asyncio.TimeoutError:
                break
        raise TimeoutError(f"No reply to {message.message_id}")
```

The key point is that `send_and_wait` correlates by `message_id`. Unrelated messages that arrive during the wait are put back into the queue rather than discarded.

### When Message Passing Breaks Down

Message passing fails at scale because it creates tight coupling between agents. If you have 8 agents and each pair needs to communicate, you have 28 potential message channels to manage. Add a new agent and you potentially need to wire it to all 8 existing agents.

The deeper problem is that message passing doesn't give you a single source of truth. If Agent A tells Agent B that the task status is "in_progress", and also tells Agent C the same, and then Agent A updates the status to "complete" but only notifies Agent B, Agent C is now operating on stale information. Message passing doesn't automatically propagate state changes — you have to build that logic explicitly.

This is why systems with more than 4–6 agents almost always graduate to a shared store or event bus.

---

## 4. Shared Memory: Key-Value Store, Vector Store, Document Store

Shared memory solves the point-to-point coupling problem. Instead of agents knowing each other's addresses, they all know the address of the store. State flows through the store, not between agents directly.

![Shared memory hierarchy: Agent-local memory, vector store, global KV store in a cache-miss chain](shared-state-and-coordination-3.png)

### The Memory Hierarchy

Shared memory in a multi-agent system is not a flat structure. Think of it as a hierarchy with different layers for different access patterns:

**Agent-Local Memory:** The agent's own context window, scratch notes, and in-flight variables. This is not shared. It's the fastest tier (zero network latency) but completely invisible to other agents.

**Vector Store:** Persistent semantic memory. Embeddings of past episodes, retrieved documents, and intermediate results. Shared across agents, but reads are fuzzy (top-K nearest neighbors) rather than exact key lookups. Eventual consistency is typical and acceptable. See [Episodic Memory and Vector Stores](/blog/machine-learning/ai-agent/episodic-memory-vector-stores) for the full design space.

**Global KV Store:** Authoritative shared state. Task assignments, agent registry, step outputs, plan state. Exact key lookups, strong consistency options, and explicit writes required. This is the coordination layer.

The canonical flow is: an agent first checks its local state, then the vector store (for semantic recall), then the global KV (for authoritative facts). Writes that need to be visible to other agents always go to the global KV.

### Choosing the Right Backend

**Redis** is the default for global KV in multi-agent systems. It provides:
- Sub-millisecond read/write latency
- Atomic operations (SET NX for compare-and-swap, INCR for sequence numbers)
- TTL-based expiry (automatically clean up stale task claims)
- Pub/Sub built in (so your KV store can also be your event bus)
- Cluster mode for horizontal scaling

```python
import redis.asyncio as redis
import json
from typing import Any, Optional

class SharedStateStore:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.client = redis.from_url(url, decode_responses=True)

    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        serialized = json.dumps(value)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def read(self, key: str) -> Optional[Any]:
        value = await self.client.get(key)
        return json.loads(value) if value else None

    async def compare_and_swap(
        self,
        key: str,
        expected: Any,
        new_value: Any
    ) -> bool:
        """Atomically update key only if current value matches expected."""
        async with self.client.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                current = await self.client.get(key)
                if json.loads(current) != expected:
                    return False
                pipe.multi()
                pipe.set(key, json.dumps(new_value))
                await pipe.execute()
                return True
            except redis.WatchError:
                return False

    async def claim_task(self, task_id: str, agent_id: str, ttl: int = 300) -> bool:
        """Atomically claim a task. Returns True if claim succeeded."""
        key = f"task:claim:{task_id}"
        result = await self.client.set(key, agent_id, nx=True, ex=ttl)
        return result is True
```

The `claim_task` method uses Redis SET NX (set if not exists) to implement a distributed lock. If two agents both try to claim the same task simultaneously, exactly one will succeed. The TTL ensures that a crashed agent releases its claim automatically.

**PostgreSQL** is the right choice when you need:
- Complex queries across shared state (which agent is idle? what tasks are blocked?)
- ACID transactions spanning multiple keys
- Audit logs (all writes are durable and ordered)
- JSON columns that let you store structured state alongside relational fields

```sql
-- Task state table
CREATE TABLE tasks (
    task_id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Atomic task claim via SELECT FOR UPDATE SKIP LOCKED
-- (each agent gets a different unclaimed task even with N concurrent agents)
BEGIN;
SELECT task_id, task_data
FROM tasks
WHERE status = 'pending'
LIMIT 1
FOR UPDATE SKIP LOCKED;

UPDATE tasks
SET status = 'in_progress', claimed_by = $1, claimed_at = NOW()
WHERE task_id = $2;
COMMIT;
```

`SKIP LOCKED` is the key idiom here. Without it, multiple agents competing for the same task would all block on the lock. With `SKIP LOCKED`, each agent moves on to the next available task, eliminating the thundering herd.

**Chroma / Pinecone / Weaviate** (vector stores) are the right backend for:
- Semantic search over past agent outputs
- Retrieving relevant context for an agent's current task
- Long-term episodic memory that agents can query with natural language

Vector stores are eventually consistent by design — indexing a new embedding typically takes 100–500ms to become queryable. This is acceptable for most LLM use cases because agents don't need sub-millisecond reads; they're already waiting on LLM inference.

---

## 5. Conflict Resolution: What to Do When Two Agents Update the Same State

Conflict resolution is what happens when two agents try to update the same piece of shared state simultaneously. Every multi-agent system has to choose a strategy. There are four main options:

![Conflict resolution strategies matrix: Last-Write-Wins, Merge, Lock, CRDT vs consistency, overhead, complexity, correctness](shared-state-and-coordination-4.png)

### Last-Write-Wins (LWW)

The simplest strategy: whichever write arrives last wins. If Agent A writes "summary: version 1" at time T1, and Agent B writes "summary: version 2" at time T2 > T1, the final value is "version 2".

**Pros:** Zero coordination overhead. Trivial to implement.

**Cons:** Lossy. If Agent A's write was more recent in wall-clock time but arrived later due to network delay, you lose it. Clocks in distributed systems are not synchronized, so "last write" is ambiguous.

**When to use:** Idempotent updates where losing a write is acceptable. Heartbeat timestamps, cache invalidations, presence indicators.

**When not to use:** Anything where every write carries unique information that must be preserved. A document summary where each agent contributes a different paragraph cannot use LWW.

### Semantic Merge

The application defines a merge function that combines two conflicting values into one. Git's three-way merge is the canonical example. For LLM systems, semantic merge might mean asking an LLM to combine two conflicting summaries.

```python
async def merge_summaries(
    base: str,
    version_a: str,
    version_b: str,
    llm_client
) -> str:
    """Use an LLM to semantically merge two conflicting summaries."""
    prompt = f"""Two agents wrote different summaries of the same material.
Base (what both started from): {base}

Agent A's version: {version_a}

Agent B's version: {version_b}

Produce a single merged summary that incorporates the unique insights from both versions.
Do not duplicate content. Keep the result concise."""

    response = await llm_client.complete(prompt)
    return response.text
```

**Pros:** Preserves information from both writes. Produces semantically coherent results.

**Cons:** Expensive (LLM call per conflict). Non-deterministic (two runs of the merge might produce different outputs). Requires detecting that a conflict occurred.

**When to use:** Document generation tasks where multiple agents contribute sections. Research synthesis where several agents find complementary facts.

### Locking (Mutex)

Before writing, an agent acquires a lock on the key. If another agent holds the lock, the first agent waits. After writing, the agent releases the lock.

```python
from contextlib import asynccontextmanager
import redis.asyncio as redis

class DistributedLock:
    def __init__(self, redis_client, key: str, ttl: int = 30):
        self.client = redis_client
        self.key = f"lock:{key}"
        self.ttl = ttl
        self.token = str(uuid.uuid4())

    @asynccontextmanager
    async def acquire(self, timeout: float = 10.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            acquired = await self.client.set(
                self.key, self.token, nx=True, ex=self.ttl
            )
            if acquired:
                try:
                    yield
                finally:
                    # Only release if we still hold the lock (not expired + reacquired)
                    current = await self.client.get(self.key)
                    if current == self.token:
                        await self.client.delete(self.key)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Could not acquire lock on {self.key}")

# Usage
async def safe_write(store, key: str, value: Any, agent_id: str):
    lock = DistributedLock(store.client, key)
    async with lock.acquire(timeout=15.0):
        current = await store.read(key)
        updated = merge(current, value, agent_id)
        await store.write(key, updated)
```

**Pros:** Strong consistency. No lost writes. The lock ensures only one agent updates the value at a time.

**Cons:** High latency. If Agent A holds the lock for 2 seconds (LLM inference is slow), Agent B waits 2 seconds doing nothing. Deadlock risk if multiple locks are acquired in different orders. Lock contention at high agent counts.

**When to use:** Critical state updates where correctness is more important than throughput. Plan updates, task status transitions, resource allocation.

### CRDTs (Conflict-Free Replicated Data Types)

CRDTs are data structures designed so that any two concurrent writes can be merged deterministically without conflicts. The most common CRDTs for multi-agent systems are:

**G-Counter (Grow-only counter):** Each agent has its own counter. The global count is the sum of all agent counters. Incrementing is always conflict-free.

**OR-Set (Observed-Remove Set):** A set where adds and removes are always conflict-free. Each element has a unique tag, so "remove X" only removes the specific instance of X that was added.

**LWW-Register with vector clocks:** A register (single value) where conflicts are resolved by comparing vector clocks rather than wall-clock timestamps.

```python
class GCounter:
    """A grow-only counter for N agents. Conflict-free by construction."""
    def __init__(self, agent_id: str, num_agents: int):
        self.agent_id = agent_id
        self.counts = [0] * num_agents
        self.agent_index = hash(agent_id) % num_agents

    def increment(self, amount: int = 1) -> None:
        self.counts[self.agent_index] += amount

    def value(self) -> int:
        return sum(self.counts)

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge another counter's state. Always safe."""
        merged = GCounter(self.agent_id, len(self.counts))
        merged.counts = [
            max(a, b) for a, b in zip(self.counts, other.counts)
        ]
        return merged
```

**Pros:** No locking required. Scales to any number of agents. Guarantees eventual consistency.

**Cons:** Only works for specific data structures. Not all state can be modeled as a CRDT. Design complexity is high.

**When to use:** Progress counters, completion flags, tag sets, vote tallies. Any state that naturally accumulates rather than being replaced.

---

## 6. Eventual Consistency vs Strong Consistency: What LLM-Based Systems Actually Need

The distributed systems literature has a lot to say about consistency models. For LLM-based systems, the practical question is: which agents need to see which updates immediately, and which ones can tolerate lag?

### What Strong Consistency Buys You

Strong consistency (linearizability) means that after any write completes, every subsequent read anywhere in the system sees that write. There is a single global ordering of operations.

The cost is high: to achieve this, every read must check with a quorum of nodes before returning. In a Redis cluster or Postgres primary/replica setup, this means network round-trips for every read.

### Where You Actually Need It

In a multi-agent system, strong consistency matters for:

**Task assignment:** Two agents must not both believe they own the same task. This is an atomic compare-and-set operation. Redis SET NX gives you this.

**Planner state:** If the orchestrator updates the plan (adds a step, changes dependencies), all workers must see the update before acting on it. Otherwise Worker A might execute step 4 before Worker B has finished step 3 — which step 4 depends on.

**Final output writes:** The last write to the "final result" key must be visible to the consumer. You don't want to hand back a stale intermediate result.

### Where Eventual Consistency Is Fine

**Vector store retrieval:** An agent querying past episodes for context can tolerate a few seconds of lag. It doesn't matter if a very recent episode isn't indexed yet.

**Observation logging:** Writes to an audit log or trace store don't need to be immediately visible to other agents.

**Heuristic metadata:** Agent-maintained scores, confidence values, and priorities can lag behind without affecting correctness.

**Search results:** If one agent cached a search result that another agent also wants, a slightly stale cache is fine — the worst case is a redundant API call.

### The Causal Consistency Sweet Spot

For many inter-agent dependencies, causal consistency is the right model. It says: if Agent B read Agent A's write and then wrote something itself, any agent that reads Agent B's write must also see Agent A's write.

This preserves cause-and-effect without requiring global ordering. It's implementable with vector clocks:

```python
from typing import Dict

class VectorClock:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.clock: Dict[str, int] = {}

    def tick(self) -> Dict[str, int]:
        self.clock[self.agent_id] = self.clock.get(self.agent_id, 0) + 1
        return dict(self.clock)

    def update(self, other_clock: Dict[str, int]) -> None:
        for agent, time in other_clock.items():
            self.clock[agent] = max(self.clock.get(agent, 0), time)
        self.tick()

    def happens_before(self, other: Dict[str, int]) -> bool:
        """Returns True if self.clock happened before other."""
        return all(
            self.clock.get(agent, 0) <= time
            for agent, time in other.items()
        ) and any(
            self.clock.get(agent, 0) < time
            for agent, time in other.items()
        )
```

With vector clocks, when Agent B reads Agent A's value (which carries Agent A's clock), Agent B can propagate Agent A's causal history in its own writes. Any reader of Agent B's output can verify that they've also seen Agent A's write.

---

## 7. Turn-Taking and Locking: Preventing Simultaneous Conflicting Writes

Some state transitions need to be atomic — not just in the database sense, but in the logical sense. When an agent claims a task, marks it in-progress, executes it, and marks it complete, those four transitions should look atomic to observers. If another agent reads the state between "in-progress" and "complete", it should not make decisions based on that intermediate state.

![Turn-taking via token ring: Agent 1 holds token, writes state, releases token, Agent 2 acquires token](shared-state-and-coordination-7.png)

### Distributed Locks in Practice

The Redis SETNX lock pattern described earlier is the foundation. In production you need a few additional behaviors:

**Lease renewal.** If the operation holding a lock takes longer than expected (LLM inference can be slow), the lock might expire and another agent can acquire it. The holding agent should renew the lock before expiry:

```python
import asyncio

async def holding_agent_with_renewal(lock: DistributedLock, operation):
    async with lock.acquire():
        renewal_task = asyncio.create_task(
            renew_lock_periodically(lock, interval=lock.ttl // 2)
        )
        try:
            result = await operation()
        finally:
            renewal_task.cancel()
    return result

async def renew_lock_periodically(lock: DistributedLock, interval: float):
    while True:
        await asyncio.sleep(interval)
        current = await lock.client.get(lock.key)
        if current == lock.token:
            await lock.client.expire(lock.key, lock.ttl)
```

**Lock ordering.** If agents acquire multiple locks, they must always acquire them in the same order to prevent deadlock. Define a global key ordering (alphabetical works) and enforce it:

```python
async def acquire_multiple_locks(keys: list[str], agent_id: str):
    sorted_keys = sorted(keys)  # always same order
    locks = []
    try:
        for key in sorted_keys:
            lock = DistributedLock(redis_client, key)
            await lock.__aenter__()
            locks.append(lock)
        yield
    finally:
        for lock in reversed(locks):
            await lock.__aexit__(None, None, None)
```

**Fairness.** A naive spin-lock lets high-frequency agents starve low-frequency ones. Use a queue-based approach where agents register their intent to write and are served in order:

```python
async def enqueue_write(key: str, agent_id: str, value: Any, store) -> None:
    """Add write to a per-key queue. A single worker drains the queue."""
    queue_key = f"write_queue:{key}"
    entry = {"agent_id": agent_id, "value": value, "timestamp": time.time()}
    await store.client.rpush(queue_key, json.dumps(entry))

async def drain_write_queue(key: str, store) -> None:
    """Process writes in FIFO order."""
    queue_key = f"write_queue:{key}"
    while True:
        entry_json = await store.client.lpop(queue_key)
        if not entry_json:
            break
        entry = json.loads(entry_json)
        await store.write(key, entry["value"])
```

### Token Ring for Sequential Agents

For workflows where agents must execute in sequence (not just "don't conflict," but "strictly ordered"), a token ring provides a clean primitive. Only the agent holding the token may proceed to the next phase.

The token ring is simpler than a general distributed lock and easier to reason about: there is exactly one token, it moves around a fixed set of agents in a fixed order, and the system's total state advances one step per token holder.

In practice, implement the token as a Redis key whose value is the current token holder's agent ID:

```python
async def pass_token(current_holder: str, next_holder: str, store) -> bool:
    """Pass the write token to the next agent."""
    async with store.compare_and_swap_context("write_token") as (current, cas):
        if current != current_holder:
            return False  # we don't hold the token
        return await cas(next_holder)
```

---

## 8. Event-Driven Coordination: Agents Reacting to Each Other's Outputs

Message passing and shared memory are both pull-based from the receiver's perspective: an agent decides to check for new messages or read the store. Event-driven coordination inverts this: when state changes, interested agents are notified.

![Event-driven coordination timeline: Agent A emits event, event bus routes to subscribers B and C, both react concurrently](shared-state-and-coordination-6.png)

### Why Event-Driven?

In a system where agents are waiting for each other's results, the naive approach is polling:

```python
# Polling approach — wastes CPU and introduces latency
while True:
    result = await store.read(f"task:{task_id}:result")
    if result is not None:
        break
    await asyncio.sleep(0.5)  # 500ms latency penalty
```

Polling introduces artificial latency (up to one poll interval) and wastes compute on agents sitting idle. Event-driven coordination eliminates both:

```python
# Event-driven approach — zero polling latency
async def wait_for_result(task_id: str, event_bus) -> Any:
    async for event in event_bus.subscribe(f"task:{task_id}:complete"):
        return event.payload["result"]
```

### Event Bus Implementation

A minimal event bus built on Redis Pub/Sub:

```python
import redis.asyncio as redis
import asyncio
import json
from typing import AsyncGenerator, Any

class EventBus:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.publish_client = redis.from_url(url)
        self.url = url

    async def publish(self, channel: str, payload: Any) -> None:
        message = json.dumps({
            "channel": channel,
            "payload": payload,
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.publish_client.publish(channel, message)

    async def subscribe(
        self,
        channel: str,
        timeout: float | None = None
    ) -> AsyncGenerator[dict, None]:
        sub_client = redis.from_url(self.url)
        async with sub_client.pubsub() as pubsub:
            await pubsub.subscribe(channel)
            start = asyncio.get_event_loop().time()
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
                if timeout and (asyncio.get_event_loop().time() - start) > timeout:
                    break
```

### Event Patterns

**Fan-out:** One event triggers many agents. An orchestrator publishes "task_set_complete" and all downstream agents react simultaneously.

```python
async def orchestrator_publish_done(task_set_id: str, bus: EventBus):
    await bus.publish("task_set:complete", {
        "task_set_id": task_set_id,
        "completed_at": datetime.utcnow().isoformat()
    })
    # All subscribers (writer, critic, formatter, etc.) receive this simultaneously
```

**Fan-in:** Many agents produce events; one agent waits for all of them.

```python
async def wait_for_all_agents(agent_ids: list[str], bus: EventBus) -> list[dict]:
    tasks = [
        asyncio.create_task(
            anext(aiter(bus.subscribe(f"agent:{aid}:done")))
        )
        for aid in agent_ids
    ]
    results = await asyncio.gather(*tasks)
    return [r["payload"] for r in results]
```

**Conditional fan-out:** Only some agents react to an event based on routing rules.

```python
async def conditional_subscriber(
    agent_id: str,
    bus: EventBus,
    filter_fn
) -> AsyncGenerator[dict, None]:
    async for event in bus.subscribe("task:assigned"):
        if filter_fn(event, agent_id):
            yield event
```

### Ordering in Event-Driven Systems

The risk with event-driven coordination is out-of-order processing. Agent B might receive event E2 (task updated) before event E1 (task created) if the events travel through different paths.

Solutions:
1. **Sequence numbers on events.** Consumers buffer out-of-order events and release them in order.
2. **Topic partitioning.** All events for the same task_id go to the same Kafka partition, guaranteeing FIFO within that task.
3. **Idempotent consumers.** Design consumers to be correct regardless of order. If Agent B is checking a task's current state from the store rather than computing state from events, ordering doesn't matter.

Option 3 is the most robust for LLM-based systems: events are notifications ("something changed"), and agents re-read the authoritative state from the store rather than reconstructing state from the event stream.

---

## 9. Global vs Local State: What Each Agent Should Own vs What Should Be Shared

One of the most consequential design decisions in a multi-agent system is the boundary between what each agent owns privately and what goes into shared state. Get this wrong and you'll either have agents constantly fighting over global state or agents drifting out of sync with each other.

### The Ownership Principle

A piece of state should be owned by exactly one agent or by the shared store. Shared ownership — where multiple agents believe they can update the same state independently — is the source of most coordination bugs.

**Good partition:**
- Agent A owns its planning scratchpad. Only Agent A reads and writes it.
- The shared store owns the task completion ledger. Any agent can claim a task, but the store enforces mutual exclusion.
- Agent B owns its in-progress output buffer. When Agent B is done, it writes the final output to the shared store.

**Bad partition:**
- Agents A and B both write to "current_plan" without coordination. → Conflict.
- The shared store holds Agent A's ephemeral scratch variables. → Unnecessary contention.
- Agents share a mutable "running context" string that all of them append to simultaneously. → Race conditions.

### What Should Be Global

**Task registry:** The canonical list of tasks, their statuses, and who owns them. This must be globally consistent to prevent duplicate work.

**Plan state:** The current plan structure, step ordering, and dependency graph. All agents executing the plan must see the same plan.

**Final outputs:** Completed work product (generated text, computed values, retrieved documents) that downstream agents need to consume.

**Agent registry:** Which agents are alive, what their capabilities are, and whether they are idle or busy.

**Coordination primitives:** Locks, semaphores, sequence numbers, and barrier states.

### What Should Be Local

**Working memory:** Everything an agent needs to execute its current step: fetched documents, prompt templates, intermediate computations, LLM API responses.

**Retry state:** How many times an agent has attempted a step, what errors it has seen. Other agents don't need this.

**Cached reads:** An agent can cache values it has read from the global store. Cache invalidation happens via event subscription or TTL.

**Hypothesis state:** For agents that explore multiple options before committing, the unexplored branches are local until the agent decides.

### The Minimal Footprint Rule

Every time you are tempted to write a value to global state, ask: does any other agent need to read this? If the answer is no, keep it local.

The cost of global state is real: every global write is a coordination point. It consumes network bandwidth, competes with other agents for store capacity, and potentially triggers events that cause other agents to do work they don't need to do.

---

## 10. Observability of Shared State: Debugging Multi-Agent Coordination Failures

Multi-agent systems fail in ways that are qualitatively different from single-agent systems. A bug might not manifest in any individual agent's logs — it lives in the interaction between agents over shared state.

![Debugging coordination failure pipeline: observe inconsistent output, trace timeline, identify conflicting writes, replay, fix](shared-state-and-coordination-9.png)

### The Coordination Log

The single most important observability primitive is a structured log of every read and write to shared state, with agent ID, timestamp, and value:

```python
import structlog

log = structlog.get_logger()

class ObservableStateStore(SharedStateStore):
    async def write(self, key: str, value: Any, ttl: int | None = None, agent_id: str = "unknown") -> None:
        await super().write(key, value, ttl)
        log.info("state.write", key=key, value=value, agent_id=agent_id, ttl=ttl)

    async def read(self, key: str, agent_id: str = "unknown") -> Any:
        value = await super().read(key)
        log.info("state.read", key=key, value=value, agent_id=agent_id)
        return value

    async def compare_and_swap(self, key: str, expected: Any, new_value: Any, agent_id: str = "unknown") -> bool:
        result = await super().compare_and_swap(key, expected, new_value)
        log.info("state.cas", key=key, expected=expected, new_value=new_value,
                 agent_id=agent_id, succeeded=result)
        return result
```

This log lets you reconstruct exactly what happened, in what order, from which agent's perspective.

### Timeline Reconstruction

When you have a coordination bug, the first debugging step is to reconstruct the timeline. Parse the coordination log and sort all events by timestamp:

```python
def reconstruct_timeline(log_entries: list[dict]) -> list[dict]:
    """Sort events by timestamp and annotate with causal relationships."""
    sorted_entries = sorted(log_entries, key=lambda e: e["timestamp"])

    # Annotate each read with the write it saw
    last_write: dict[str, dict] = {}
    for entry in sorted_entries:
        if entry["event"] == "state.write":
            last_write[entry["key"]] = entry
        elif entry["event"] == "state.read" and entry["key"] in last_write:
            entry["read_from_write"] = last_write[entry["key"]]["agent_id"]
            entry["write_age_ms"] = (
                entry["timestamp"] - last_write[entry["key"]]["timestamp"]
            ) * 1000

    return sorted_entries
```

### Identifying Conflict Signatures

Common patterns in coordination logs that indicate bugs:

**Concurrent write pattern:** Two writes to the same key with timestamps within a few hundred milliseconds of each other, from different agents, where the second write doesn't incorporate the first.

```python
def find_concurrent_writes(timeline: list[dict], window_ms: float = 500) -> list[tuple]:
    writes_by_key: dict[str, list[dict]] = {}
    for entry in timeline:
        if entry["event"] == "state.write":
            writes_by_key.setdefault(entry["key"], []).append(entry)

    conflicts = []
    for key, writes in writes_by_key.items():
        for i, w1 in enumerate(writes):
            for w2 in writes[i+1:]:
                gap_ms = (w2["timestamp"] - w1["timestamp"]) * 1000
                if gap_ms < window_ms and w1["agent_id"] != w2["agent_id"]:
                    conflicts.append((key, w1, w2, gap_ms))
    return conflicts
```

**Stale read pattern:** An agent reads a value, another agent writes a new value, and then the first agent writes based on its stale read.

**Missing dependency pattern:** An agent reads a key before any agent has written it, getting a null, and proceeds with incorrect default behavior.

### Distributed Tracing

For production systems, wire every agent action to a distributed trace. OpenTelemetry integrates naturally with Python async code:

```python
from opentelemetry import trace

tracer = trace.get_tracer("multi-agent-system")

async def traced_agent_step(agent_id: str, step_name: str, fn):
    with tracer.start_as_current_span(
        f"{agent_id}.{step_name}",
        attributes={"agent.id": agent_id, "step.name": step_name}
    ) as span:
        try:
            result = await fn()
            span.set_attribute("step.status", "success")
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("step.status", "error")
            raise
```

In a distributed trace, you can see exactly which agent's action caused which downstream action — the causal graph is explicit.

---

## 11. Case Studies: Coordination Bugs in Production Multi-Agent Systems

### Case Study 1: The Duplicate Research Problem

**System:** A research assistant with four agents — a Planner, two Researchers, and a Writer.

**Failure:** Both Researchers fetched the same five web pages. The Writer received two nearly identical context blocks and produced a summary that repeated every fact twice.

**Root cause:** The Planner assigned "search: topic X" to both Researchers in parallel, but the task assignment was a string in the Planner's prompt context, not a claimed entry in a shared task registry. Both Researchers thought they had independent, unique assignments.

**Fix:** Introduce a task registry in Redis. The Planner writes search tasks to the registry in a `pending` state. Each Researcher claims a task (using SET NX) before executing it. If a claim fails (another Researcher claimed it first), the agent picks a different unclaimed task.

**Lesson:** Any time two agents might execute the same work, there must be a shared exclusion mechanism. Prompt-level task assignment is not sufficient.

---

### Case Study 2: The Race Condition in Plan Updates

**System:** An orchestrator with six worker agents executing a multi-step plan stored in Redis.

**Failure:** Two workers both completed their steps and both tried to update the plan's "next_step" field simultaneously. Worker A wrote "step 4" and Worker B wrote "step 5". The final value was "step 5" (LWW), but step 4 had never been marked as the next step, so its dependencies were never triggered.

**Root cause:** Plan updates were bare SET calls without coordination. Both workers read the current state, computed the next step independently, and wrote without checking that the other write had happened.

**Fix:** Replace bare SET with optimistic concurrency control. Each plan update includes a version number. The write only succeeds if the current version matches the expected version. If not, the agent re-reads and retries:

```python
async def advance_plan(plan_id: str, completed_step: str, agent_id: str):
    for _ in range(5):  # retry up to 5 times
        plan = await store.read(f"plan:{plan_id}")
        if plan is None:
            raise ValueError(f"Plan {plan_id} not found")

        if completed_step not in plan["pending"]:
            return  # already processed

        plan["pending"].remove(completed_step)
        plan["completed"].append(completed_step)
        plan["version"] += 1

        success = await store.compare_and_swap(
            f"plan:{plan_id}",
            expected={"version": plan["version"] - 1},
            new_value=plan
        )
        if success:
            return
        # Retry on CAS failure
        await asyncio.sleep(0.1)

    raise RuntimeError(f"Failed to update plan after 5 retries")
```

**Lesson:** Plan state is exactly the kind of shared state that requires optimistic or pessimistic locking. Bare writes to plan state are a time bomb.

---

### Case Study 3: The Stale Context Hallucination

**System:** A code generation system with a Planner, a Coder, and a Reviewer. The Reviewer reads the Coder's output from the shared store.

**Failure:** The Reviewer occasionally reviewed an older version of the code. The Coder had written two revisions; the Reviewer read the first one and approved code that had already been superseded.

**Root cause:** The Reviewer started as soon as it received the "coder_done" event. But the event was published from a Redis Pub/Sub channel, while the actual code was stored in a different Redis key. There was a brief window between the event publish and the key write. The Reviewer read the key during this window and got the previous revision.

**Fix:** Write-then-publish, never publish-then-write. The correct order is:
1. Write the new code to the Redis key.
2. Publish the "coder_done" event.

Any subscriber that receives the event after the key write will always see the latest version.

```python
async def publish_result(key: str, value: Any, channel: str, store, bus):
    await store.write(key, value)        # Write first
    await bus.publish(channel, {"key": key})  # Notify second
```

**Lesson:** Write-then-publish is a fundamental ordering invariant. Violating it creates windows where readers see stale state.

---

### Case Study 4: The Deadlock Between Critic and Writer

**System:** A content generation pipeline where a Writer and a Critic held locks on different keys and then each tried to acquire the other's key.

**Failure:** The system froze. Both agents were holding a lock and waiting for the other to release.

**Root cause:** The Writer locked "draft" then tried to lock "critique_history". The Critic locked "critique_history" then tried to lock "draft". Classic deadlock.

**Fix:** Enforce global lock ordering. All agents must acquire locks in alphabetical order of key name. "critique_history" comes before "draft" alphabetically, so both agents must try to acquire "critique_history" first. Now there is no cycle:

```python
LOCK_ORDER = sorted  # alphabetical ordering for all lock acquisitions

async def acquire_locks(keys: list[str]):
    for key in LOCK_ORDER(keys):
        await acquire_lock(key)
```

**Lesson:** If agents acquire multiple locks, you must define and enforce a global ordering. Otherwise deadlock is a matter of when, not if.

---

### Case Study 5: The Thundering Herd on Task Completion

**System:** Eight worker agents all subscribed to a "batch_complete" event. When the batch completed, all eight agents woke up, all read the same "pending_tasks" queue, and all tried to claim the first item.

**Failure:** Seven of the eight claim attempts failed (Redis SET NX is atomic, only one wins). The seven failed agents immediately retried, creating a spike of Redis load that briefly starved other operations.

**Root cause:** All agents had identical behavior on the event: immediately try to claim a task. With N agents competing, N-1 will fail and immediately retry.

**Fix:** Add jitter to retry delays. Each agent waits a random interval before retrying a failed claim:

```python
import random

async def claim_with_jitter(task_id: str, agent_id: str, store, max_attempts: int = 10):
    for attempt in range(max_attempts):
        if await store.claim_task(task_id, agent_id):
            return True
        # Exponential backoff with jitter
        base_delay = 0.1 * (2 ** attempt)
        jitter = random.uniform(0, base_delay * 0.5)
        await asyncio.sleep(base_delay + jitter)
    return False
```

Alternatively, use `BLPOP` instead of polling. The Redis `BLPOP` command blocks until an item is available in a list. With N agents all doing `BLPOP` on the same task queue, Redis serializes them automatically — only one agent gets each task.

**Lesson:** Event-driven wakeup of many agents simultaneously produces thundering herd. Either add jitter or use a queue primitive that handles distribution natively.

---

### Case Study 6: The Missing Barrier in Parallel Subtask Execution

**System:** A planning agent dispatched six parallel subtasks and then used the results to generate a final answer.

**Failure:** Occasionally the final answer was generated before all six subtasks had written their results. The incomplete result set produced a wrong final answer.

**Root cause:** The orchestrator checked a counter in Redis ("completed_count") after dispatching subtasks. But the counter increment and the result write were not atomic. An agent could increment the counter before writing its result:

```python
# WRONG - counter incremented before result is available
async def complete_subtask(task_id: str, result: Any, store):
    await store.write(f"subtask:{task_id}:count", "done")
    await store.incr("completed_count")  # counter up...
    await store.write(f"subtask:{task_id}:result", result)  # ...result not yet
```

**Fix:** Write result first, then increment counter:

```python
# CORRECT - result guaranteed visible when counter increments
async def complete_subtask(task_id: str, result: Any, store):
    await store.write(f"subtask:{task_id}:result", result)
    await store.incr("completed_count")  # result visible before counter
```

Better: use a barrier primitive that only opens when all expected results are present:

```python
async def barrier_wait(expected_count: int, results_key_prefix: str, store, timeout: float = 300):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        count = int(await store.read("completed_count") or 0)
        if count >= expected_count:
            # Verify all results are actually present
            results = []
            for i in range(expected_count):
                result = await store.read(f"{results_key_prefix}:{i}:result")
                if result is None:
                    break
                results.append(result)
            if len(results) == expected_count:
                return results
        await asyncio.sleep(0.5)
    raise TimeoutError("Barrier timed out")
```

**Lesson:** Completion signals and result availability are separate events. Always verify that the actual data is present, not just that a counter reached a threshold.

---

### Case Study 7: The Event Ordering Bug in Document Assembly

**System:** A document assembly pipeline where four agents each contributed a section. The sections were supposed to appear in order: introduction, analysis, conclusion, references.

**Failure:** The final document occasionally had sections in the wrong order. Analysis sometimes appeared before introduction.

**Root cause:** Agents published "section_complete" events as they finished. The assembler concatenated sections in the order it received events. Network latency made event arrival order nondeterministic.

**Fix:** Never rely on event arrival order for sequenced assembly. Use explicit ordering:

```python
# Each section is written with its position
async def write_section(position: int, content: str, doc_id: str, store):
    await store.write(f"doc:{doc_id}:section:{position}", content)

# Assembler reads sections in explicit order
async def assemble_document(doc_id: str, expected_sections: int, store) -> str:
    sections = []
    for i in range(expected_sections):
        while True:
            section = await store.read(f"doc:{doc_id}:section:{i}")
            if section is not None:
                sections.append(section)
                break
            await asyncio.sleep(0.1)
    return "\n\n".join(sections)
```

**Lesson:** Ordering-sensitive assembly must use explicit sequence positions. Event-driven notification can tell you something is ready; it cannot tell you where it goes.

---

## 12. Coordination Overhead: When Adding Coordination Costs More Than It Saves

Coordination isn't free. Every shared state access, every lock acquisition, every event subscription adds latency and complexity. At some point, the coordination overhead exceeds the cost of the inconsistency you're trying to prevent.

![Coordination overhead matrix: pattern vs agent count vs messages per step vs total overhead](shared-state-and-coordination-10.png)

### Measuring Coordination Cost

The cost of coordination has several components:

**Latency:** A Redis SET NX takes ~0.3ms on a local network. An LLM inference step takes 1–30 seconds. Coordination is 0.001% of total step time. At this ratio, aggressive locking is cheap.

But if your agents are doing frequent small updates (updating a counter after every token generated), the coordination cost becomes significant relative to the computation cost.

**Concurrency reduction:** Every lock reduces parallelism. If Agent A holds the "plan" lock for 5 seconds, Agents B through H are all blocked. Eight agents become effectively one agent.

**Cognitive complexity:** Every coordination mechanism is a thing that can fail. A distributed lock can expire. A CAS can fail and require retry logic. An event can be lost. The more coordination you add, the more failure modes you introduce.

### The Overhead Matrix

At small scale (2–4 agents), all patterns have acceptable overhead. Message passing is slightly more latent than shared memory but the absolute numbers are tiny.

At medium scale (5–8 agents), message passing starts to strain. N×(N-1) possible channels means up to 56 potential connections at 8 agents. Shared memory or event bus is better.

At large scale (9+ agents), message passing is untenable. Event bus is the only pattern that scales sublinearly. Blackboard patterns with high write frequency create contention storms. Shared memory with careful partitioning (different agents own different key namespaces) is the other viable option.

### When to Skip Coordination

Some workloads are naturally embarrassingly parallel. If your agents are each processing independent chunks of a large document, with no dependency between chunks and the final step is a simple concatenation, you don't need any inter-agent coordination during the main processing phase.

Skip coordination when:
- Agent outputs are independent (no agent reads another's intermediate state)
- The final aggregation is a simple join or concatenation
- Losing one agent's work is acceptable (you can re-run it)

Add coordination when:
- Two agents might do the same work (add a task registry with mutual exclusion)
- Agent B depends on Agent A's specific output (add explicit dependency tracking)
- The final result requires combining all agents' work without losing any

### Right-Sizing Consistency

Over-engineering consistency is as bad as under-engineering it. A system where every read requires a quorum check on a three-node Redis cluster, every write goes through a two-phase commit, and every agent step is wrapped in a distributed trace is expensive to operate and slow to run.

The right-sizing heuristic:
1. **Identify the state that must be strongly consistent** (task claims, plan state, final outputs). Apply strong consistency there.
2. **Identify the state that can tolerate stale reads** (vector store results, cached search results, agent heartbeats). Use eventual consistency there.
3. **Identify the state that agents own privately** (working memory, in-flight context). Use no consistency mechanism — it's not shared.

Most LLM-based multi-agent systems have far more local state than global state. The global state should be a thin layer of coordination primitives, not a fat shared database of everything every agent is thinking.

### The Coordination Complexity Budget

Every coordination mechanism you add to your system has a maintenance cost. When a lock deadlocks at 3am, you need an engineer who understands the locking protocol. When a CAS loop starts spin-locking under load, you need to debug it without halting the entire system.

Set a coordination complexity budget. A reasonable budget for most LLM-based systems:
- One shared state store (Redis or Postgres)
- One event bus (Redis Pub/Sub or Kafka for larger systems)
- At most two or three lock-protected code paths
- No more than one consistency level per state namespace

If you find yourself adding a fourth lock type or a second distributed store to solve a new problem, stop and ask whether the problem can be solved architecturally instead — by separating agents so they don't share state at all.

---

## Putting It Together: A Practical Coordination Architecture

Here is a battle-tested coordination architecture for a medium-complexity LLM multi-agent system (4–12 agents):

**Infrastructure:**
- Redis 7+ with cluster mode for high availability
- Structured JSON logging for all state reads and writes
- OpenTelemetry distributed tracing

**Global state namespaces:**
- `task:*` — Task registry with SET NX claims
- `plan:*` — Plan state with CAS-based optimistic writes
- `result:*` — Final outputs, write-once, read-many
- `agent:*` — Agent registry and heartbeats (TTL-based)

**Event channels:**
- `task:claimed` — Published when an agent claims a task
- `task:complete` — Published when an agent completes a task (write-then-publish)
- `plan:updated` — Published when the plan structure changes
- `agent:idle` — Published when an agent becomes available

**Consistency model:**
- Task claims: strong (SET NX)
- Plan updates: optimistic concurrency (CAS with version numbers)
- Result reads: eventual (subscribers poll after event, with write-then-publish guarantee)
- Agent heartbeats: eventual (TTL-based expiry)

**Lock usage:**
- Exactly two lock-protected operations: plan updates and resource allocation
- All locks with TTL and renewal logic
- Global key ordering enforced on multi-key acquisitions

**No coordination:**
- Agent working memory
- LLM inference context
- Cached reads (TTL < 5 seconds, acceptable staleness)

This architecture handles the coordination requirements of most LLM multi-agent systems without unnecessary complexity. It scales from 4 agents to ~20 before requiring architectural changes.

---

## Summary

Multi-agent coordination is a distributed systems problem. The failure modes — duplicate work, conflicting writes, stale reads, missing barriers — are well-understood from thirty years of distributed database research. The difference in LLM-based systems is that agents are slow, expensive, and probabilistic, which changes the tradeoff calculus but not the fundamental mechanisms.

The practical summary:

1. **Use a task registry with mutual exclusion** to prevent duplicate work. Redis SET NX is sufficient.

2. **Write-then-publish** every time you combine a state write with an event notification. Never publish-then-write.

3. **Choose the right consistency level per state type.** Task claims and plan state need strong consistency. Vector store queries and cached reads don't.

4. **Use event-driven coordination** above 4–6 agents. Polling and direct message passing don't scale.

5. **Enforce lock ordering** if you acquire multiple locks. Alphabetical key order eliminates deadlock.

6. **Add structured logging to every state access.** Coordination bugs are invisible without a timeline of who read and wrote what, when.

7. **Right-size coordination overhead.** Most state in a multi-agent system is local. Only the coordination layer needs distributed consistency.

For the topology layer that determines which agents exist and how they're structured, see [Multi-Agent Topologies](/blog/machine-learning/ai-agent/multi-agent-topologies). For the memory layer that determines what individual agents remember, see [Agent Memory Taxonomy](/blog/machine-learning/ai-agent/agent-memory-taxonomy) and [Episodic Memory and Vector Stores](/blog/machine-learning/ai-agent/episodic-memory-vector-stores). The coordination layer described here is the connective tissue between them — the protocols that make a collection of agents into a coherent system.
