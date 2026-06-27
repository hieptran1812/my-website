---
title: "Stateful Agent Deployment: Managing Sessions, Persistence, and Restarts"
date: "2026-06-27"
description: "How to deploy agents that maintain state across requests and restarts — session management, state serialization, checkpoint/resume, horizontal scaling, and the operational patterns that keep stateful agents reliable."
tags: ["ai-agents", "deployment", "stateful", "sessions", "production-ml", "llm", "machine-learning", "mlops"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

Deploying a stateless REST API is an exercise in arithmetic: add instances, divide load, done. Deploying a stateful agent is an exercise in distributed systems. The agent carries a conversation history, a set of completed tool calls, a scratchpad of intermediate results, and a progress marker saying "I finished step 3 of 7." None of that fits in a URL. All of it has to survive network partitions, process restarts, and the daily reality that your cloud provider will reclaim your pod at 3 AM without warning.

The diagram above is the mental model we will build this post around: agent state is not one thing. It is a layered stack of five components, each with a different size, a different eviction strategy, and a different durability requirement.

![Agent state components — five layers with different lifetimes and eviction strategies](/imgs/blogs/stateful-agent-deployment-1.webp)

This post is about every layer of that stack: how to represent it, where to store it, how to survive failures, how to scale horizontally without corrupting it, and how to keep it from growing until it chokes your context window. We will go deep on the engineering — serialization formats, Redis key design, checkpoint schemas, migration strategies — and close with eight real-world case studies from production deployments.

## 1. The Stateful Agent Problem

A stateless API lives an enviable life. Every request is complete: it arrives with all the context needed to produce a response, the handler runs, the response returns, and the process forgets everything. The load balancer is free to route the next request to a completely different instance. Auto-scaling is trivial because instances are interchangeable.

Agents break this contract in four ways.

**Conversation continuity.** The user's follow-up question ("now make it more formal") only makes sense in the context of the previous turns. The agent cannot re-derive that context from the request alone — it needs to retrieve stored history from somewhere.

**Long-running operations.** A "research and write a 5,000-word report" task takes minutes to hours. No HTTP request can be held open that long. The agent must be able to resume from a checkpoint if the process dies at minute 7.

**Tool call idempotency.** When the agent calls an external API — creates a calendar event, sends a Slack message, charges a credit card — re-executing that tool call on resume would cause double-execution. The agent must track which calls already completed so it can skip them on retry.

**Cost of re-derivation.** Re-running all 40 previous LLM calls to reconstruct the state is not an option. At $0.01 per 1K input tokens and a 200K-token context, that is $2 per re-derivation, plus latency measured in minutes.

The fundamental tension is this: **stateful agents need to behave like a coherent, continuous process, but they run on infrastructure that is discrete, preemptible, and unreliable.** Every design decision in stateful agent deployment is an attempt to bridge that gap.

### Why Agents Are Harder Than Databases

You might say: "databases are stateful and we know how to run those." True, but agent state has properties that make it much harder:

**Variable and unbounded size.** A user session in a web app stores a user ID and a CSRF token — maybe 512 bytes. An agent session stores conversation history, tool outputs, scraped web pages, retrieved documents, and intermediate computations. It starts at a few KB and grows to megabytes over a long task.

**Heterogeneous content.** Agent state mixes structured data (tool call results as JSON), semi-structured data (conversation turns as message arrays), and unstructured data (web page text in working memory). No single storage primitive handles all of this optimally.

**Semantic coupling to the model.** The state is not just bytes — it is the input to the next LLM call. The format of the state, the ordering of messages, the exact representation of tool results — all of this affects model behavior. Migrating state schemas is not just a data engineering problem; it can change what the model does.

**Checkpoint granularity.** A database checkpoints every write at WAL level. An agent's meaningful checkpoint granularity is "after completing this step," because there is no natural sub-step atomicity in an LLM reasoning chain. Getting this wrong means you either replay too much work or you corrupt intermediate reasoning.

## 2. What Agent State Contains

Before we can manage state, we need to be precise about what it is. Agent state is not a blob — it is a structured collection of five distinct components, each with different lifetimes, sizes, and eviction pressures.

### Conversation History

The most obvious component: the sequence of messages that make up the dialogue. In the OpenAI message format, this is an array of `{ role, content }` objects. For a multi-step task, this includes not just user/assistant turns but also tool call invocations and their results.

Conversation history is the **highest-pressure component for eviction** because it grows with every turn and directly occupies LLM context window tokens. A 128K-token context window fills up in roughly 60–80 turns of a tool-heavy agent. Once you hit ~75% of the context limit, performance degrades because the model's attention can no longer reach far-back context effectively.

Typical size: 5 KB for a 10-turn conversation, growing to 1–2 MB for a tool-heavy 100-turn session.

### Working Memory

The agent's scratch space: variables set during execution, retrieved documents, intermediate computation results, scraped web pages. This is what the agent needs to have "in front of it" to do its current task, but it is not part of the formal conversation history.

Working memory is often the **largest single component** — a research agent retrieving 20 web pages accumulates 2–5 MB of retrieved text. It is also the most aggressively pruneable: once a retrieved document has been synthesized into the agent's reasoning, the raw text can often be discarded.

Typical size: 50 KB to 5 MB depending on task type.

### Tool Call History

A log of every external tool call the agent has made: function name, arguments, result, timestamp, and latency. This serves two purposes: (1) the model can reference what tools have already been called to avoid redundant calls, and (2) the checkpoint system uses this log to determine which calls are safe to skip on resume.

Tool call history is **medium-pressure for eviction**: it grows with every tool call but individual entries are small (~1–5 KB each). After 100 tool calls you have 100–500 KB of history. Entries older than N steps can usually be evicted from the context window (though kept in the durable log).

Typical size: 10–500 KB for a full task.

### Progress Markers

Structured data describing the agent's progress through the current task: the step index, which sub-tasks have been completed, the current goal, any user-defined milestones. This is what enables checkpoint-resume — when a process dies, the new process reads the progress markers to determine where to resume.

Progress markers are **low-pressure for eviction** because they are small and structured. They are the most critical data: losing progress markers means the agent cannot resume and must start over.

Typical size: 1–20 KB.

### Metadata

Session-level metadata: session ID, user ID, timestamps (created, last active, expires), configuration parameters (temperature, model, tool set), and audit fields. This is structural state about the session, not content state.

Metadata is **never evicted** from the durable store — it must persist for the session lifetime. It is tiny and cheap.

Typical size: ~1 KB.

## 3. Session Management

A session is the container for all of this state. Every agent interaction belongs to a session, and the session lifetime determines how long the state is retained.

![Session lifecycle — five phases from creation to cleanup with configurable timeouts](/imgs/blogs/stateful-agent-deployment-2.webp)

### Session IDs

The session ID is the primary key for all state lookups. Generation requirements:

- **Globally unique.** Collisions are catastrophic — two users sharing a session ID means state leakage.
- **Unpredictable.** Sequential IDs (`session-1`, `session-2`) allow enumeration attacks.
- **URL-safe.** The session ID will appear in headers, query parameters, and log lines.

The standard approach is a UUID v4 (128-bit random) encoded as base64url: `session_7K9mP3xQ2wVnBjLs`. This is 22 characters, URL-safe, and has a collision probability of ~1 in 10^38 per billion sessions.

```python
import uuid
import base64

def generate_session_id() -> str:
    raw = uuid.uuid4().bytes
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"session_{encoded}"

# e.g. "session_7K9mP3xQ2wVnBjLs"
```

### Session Lifecycle

Every session passes through five phases:

**CREATE**: Triggered by the first API call. The server allocates a session ID, initializes the state object (empty conversation history, empty tool history, fresh progress marker), and writes it to the state backend. Returns the session ID to the client in a `X-Session-Id` response header or as a cookie.

**ACTIVE**: The session is receiving requests within the `idle_timeout` window. Every request reads the current state, runs the agent step, writes the updated state, and extends the TTL.

**IDLE**: No request has arrived for `idle_timeout` (e.g., 5 minutes). The state is still in the backend but no compute is running. The session is "warm" — the next request will load the state and resume.

**EXPIRE**: The session's absolute TTL has been reached, or the client explicitly called `DELETE /sessions/{id}`. The state is marked stale (soft delete) and will be garbage collected.

**CLEANUP**: An async background job runs periodically to hard-delete expired state and reclaim storage. This runs minutes to hours after expiry to allow for late-arriving retries.

### Session Cookie vs. Header Transport

Once the session ID is generated, the client needs to carry it on subsequent requests. Two approaches:

**HTTP cookie:** the server sets a `Set-Cookie: session_id=<value>; HttpOnly; SameSite=Strict; Secure` header on session creation. The browser (or SDK) sends it automatically on subsequent requests.

Pros: automatic expiry handling in the browser, cannot be accessed by JavaScript (`HttpOnly`), CSRF-protected with `SameSite=Strict`.
Cons: does not work for non-browser clients (mobile apps, CLI tools, server-to-server) without custom cookie handling.

**`X-Session-Id` header:** the server returns the session ID in a response header on creation. The client stores it and sends it on subsequent requests.

Pros: works for all client types, explicit in API contracts.
Cons: client must implement storage, no automatic expiry handling.

For agent APIs used by programmatic clients (SDKs, automation scripts), headers are more natural. For browser-based chat interfaces, cookies are cleaner. Many production systems support both: the server accepts the session ID from either a cookie or a header, with the header taking precedence.

```python
def extract_session_id(request: Request) -> str | None:
    """Extract session ID from header (preferred) or cookie."""
    # Header takes precedence — allows SDK to override a browser cookie
    header_id = request.headers.get("X-Session-Id")
    if header_id:
        return header_id
    
    # Fall back to cookie
    return request.cookies.get("session_id")
```

### Timeout Configuration

Two orthogonal timeouts:

| Timeout | Purpose | Typical value |
|---|---|---|
| `idle_timeout` | How long to keep state after the last request | 5–30 min (interactive) / 24h (async tasks) |
| `abs_timeout` | Maximum session lifetime regardless of activity | 24h–7 days |
| `checkpoint_interval` | How often to write a progress checkpoint | Every major step or every N minutes |

The idle timeout prevents abandoned sessions from consuming storage. The absolute timeout is a safety valve — a session that has been receiving requests for 30 days is almost certainly a bug.

### Session Resumption vs. Session Creation

When a client sends a request with a session ID, the server must decide: is this a valid session to resume, or should it create a new one?

```python
async def get_or_create_session(session_id: str | None, user_id: str) -> Session:
    if session_id:
        session = await state_backend.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if session.user_id != user_id:
            raise SessionOwnershipError("session belongs to different user")
        if session.is_expired():
            raise SessionExpiredError(session_id)
        return session
    
    # No session ID — create new
    new_id = generate_session_id()
    session = Session(
        id=new_id,
        user_id=user_id,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    await state_backend.set(new_id, session)
    return session
```

The ownership check is critical: if you skip it, you have a session hijacking vulnerability.

## 4. State Serialization

Storing state requires serializing it to bytes. The choice of serialization format affects storage size, deserialization speed, forward/backward compatibility, and debuggability.

### What to Serialize

The first question is not "what format?" but "what to serialize at all." The wrong answer is "everything" — this leads to bloated state objects that are slow to read, write, and transmit.

**Always serialize:**
- Session metadata (ID, user ID, timestamps, config)
- Progress markers (step index, completed tasks)
- Conversation history (the message array)

**Selectively serialize:**
- Tool call results: serialize the structured result, not the raw HTTP response. A 500 KB web page scraped as a tool response should be summarized before storage.
- Retrieved documents: store the URL + content hash, not the raw text. Re-fetch if needed, or keep in a separate content store.
- Intermediate computations: serialize the final result of each step, not every intermediate value.

**Never serialize:**
- Open file handles, network connections, or thread state
- Python function objects or closures
- Large binary data (images, PDFs) — store a reference to blob storage instead

### JSON vs. Binary

The two dominant choices are JSON (or JSONL) and a binary format like MessagePack or Protocol Buffers.

| Property | JSON | MessagePack | Protobuf |
|---|---|---|---|
| Human-readable | Yes | No | No |
| Size vs JSON | 1x | ~0.5–0.7x | ~0.3–0.5x |
| Deserialization speed | ~200 MB/s | ~600 MB/s | ~1200 MB/s |
| Schema enforcement | No | No | Yes |
| Cross-language | Yes | Yes | Yes |
| Debugging ease | Easy | Hard | Hard |

For most agent deployments, **JSON with gzip compression** is the right choice. The developer experience advantage — being able to read state with `redis-cli GET session:xyz | jq` — is worth the 30% size penalty versus MessagePack. The gzip compression brings JSON sizes down to within ~15% of binary formats anyway.

Use MessagePack or Protobuf only when:
- You have sessions with state > 100 KB and serialization/deserialization is measurable in your latency profile (add a timer to confirm before switching)
- You need schema enforcement for audit or compliance reasons

A practical benchmark to run before committing to a binary format:

```python
import json, msgpack, time, gzip

state_dict = session.model_dump()  # your actual state object

# JSON + gzip
t0 = time.perf_counter()
for _ in range(1000):
    encoded = gzip.compress(json.dumps(state_dict).encode())
json_gzip_ms = (time.perf_counter() - t0)

# MessagePack
t0 = time.perf_counter()
for _ in range(1000):
    encoded = msgpack.packb(state_dict)
msgpack_ms = (time.perf_counter() - t0)

print(f"JSON+gzip: {json_gzip_ms:.1f} ms/1000 iters")
print(f"MessagePack: {msgpack_ms:.1f} ms/1000 iters")
```

For typical 50–100 KB state objects, both formats complete in under 0.5 ms. The difference is rarely worth the debugging penalty of binary-opaque state. Reserve the binary format for the rare sessions where state routinely exceeds 500 KB.

### Serialization Schema

A clean state schema pays for itself when you need to migrate later (section 8). Keep it flat with explicit version fields:

```python
from pydantic import BaseModel
from typing import Optional
import datetime

class MessageEntry(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    timestamp: datetime.datetime

class ToolCallEntry(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict
    result: dict | str | None
    status: str  # "success" | "error" | "pending"
    started_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

class ProgressMarker(BaseModel):
    current_step: int
    total_steps: Optional[int] = None
    completed_step_ids: list[str]
    current_goal: Optional[str] = None
    last_checkpoint_at: datetime.datetime

class AgentSessionState(BaseModel):
    schema_version: int = 2  # bump on breaking changes
    session_id: str
    user_id: str
    created_at: datetime.datetime
    last_active_at: datetime.datetime
    expires_at: datetime.datetime
    
    messages: list[MessageEntry]
    tool_history: list[ToolCallEntry]
    progress: ProgressMarker
    working_memory: dict  # key → value, caller-defined
    config: dict
```

The `schema_version` field is load-bearing — you will use it in section 8 to route old state through migration code paths.

### Compression in Practice

A 100-turn agent session with full tool history can produce 200–500 KB of uncompressed JSON. Gzip compression typically reduces this to 60–120 KB — a 3–4× reduction that pays for itself in Redis memory savings and network transfer.

Compress at write time, decompress at read time, transparently in the backend wrapper:

```python
import gzip

class CompressedStateBackend:
    def __init__(self, backend: StateBackend):
        self._backend = backend
    
    async def save(self, state: AgentSessionState) -> None:
        raw = state.model_dump_json().encode("utf-8")
        compressed = gzip.compress(raw, compresslevel=6)
        
        # Tag with a magic byte so we know it's compressed
        tagged = b"\x1f\x8b" + compressed  # gzip magic bytes already present
        await self._backend.set_raw(state.session_id, tagged)
    
    async def load(self, session_id: str) -> AgentSessionState | None:
        raw = await self._backend.get_raw(session_id)
        if raw is None:
            return None
        
        # Decompress if compressed (gzip magic bytes)
        if raw[:2] == b"\x1f\x8b":
            decompressed = gzip.decompress(raw)
        else:
            decompressed = raw  # legacy uncompressed sessions
        
        return AgentSessionState.model_validate_json(decompressed)
```

Compression level 6 (Python default) gives ~90% of maximum compression at ~30% of the CPU cost. Level 9 is rarely worth it for session state, where compression ratios plateau quickly beyond level 6.

### Size Management at Serialization Time

Before writing state to the backend, apply size budgets:

```python
MAX_MESSAGES = 50
MAX_TOOL_HISTORY = 100
MAX_WORKING_MEMORY_KB = 512

def enforce_state_budget(state: AgentSessionState) -> AgentSessionState:
    # Keep the last N messages (but always keep the system prompt)
    system_msgs = [m for m in state.messages if m.role == "system"]
    non_system = [m for m in state.messages if m.role != "system"]
    if len(non_system) > MAX_MESSAGES:
        non_system = non_system[-MAX_MESSAGES:]
    state.messages = system_msgs + non_system
    
    # Keep the last N tool calls
    if len(state.tool_history) > MAX_TOOL_HISTORY:
        state.tool_history = state.tool_history[-MAX_TOOL_HISTORY:]
    
    # Enforce working memory budget
    wm_size = len(json.dumps(state.working_memory).encode())
    if wm_size > MAX_WORKING_MEMORY_KB * 1024:
        # Evict oldest entries first (assumes dict insertion order)
        while wm_size > MAX_WORKING_MEMORY_KB * 1024 and state.working_memory:
            oldest_key = next(iter(state.working_memory))
            del state.working_memory[oldest_key]
            wm_size = len(json.dumps(state.working_memory).encode())
    
    return state
```

Apply this before every write to the state backend. It is cheaper to evict aggressively at write time than to hit a storage or context limit at read time.

## 5. State Backends

The state backend is where your session data actually lives. Choosing the wrong backend is one of the most common and costly mistakes in stateful agent deployment.

![State backend comparison — Redis, PostgreSQL, and DynamoDB across latency, durability, cost, query flexibility, and ops overhead](/imgs/blogs/stateful-agent-deployment-3.webp)

### Redis: The Low-Latency Workhorse

Redis is the default choice for interactive agent sessions — those with a human in the loop expecting sub-second response times.

**What Redis does well:**
- Sub-millisecond read/write latency (p50 ~0.5 ms, p99 ~2 ms on a well-sized instance)
- Native TTL support — sessions expire automatically without a background GC job
- Atomic operations for update-and-extend-TTL in a single round trip
- Redis Cluster handles horizontal partitioning transparently

**What Redis does badly:**
- State lives in RAM — 1 KB per session × 1 million sessions = 1 GB of RAM just for keys, before values
- AOF persistence has a lag — a crash during the ~5 ms fsync window can lose the last write
- Limited query expressiveness — you cannot query "all sessions for user X" without a secondary index you build yourself

**Key design for agent sessions in Redis:**

```
session:{session_id}              → JSON blob of full state
user_sessions:{user_id}           → ZSET of {session_id: last_active_timestamp}
session_lock:{session_id}         → string (for distributed locking during updates)
```

The `user_sessions` sorted set enables the "list all sessions for a user" query that the bare session store cannot answer. Keep it updated every time you write session state.

**Update with optimistic locking:**

```python
async def update_session(redis: Redis, session_id: str, 
                         updater: Callable[[AgentSessionState], AgentSessionState]) -> AgentSessionState:
    """Atomically read-modify-write session state using Redis transactions."""
    key = f"session:{session_id}"
    
    async with redis.pipeline(transaction=True) as pipe:
        while True:
            try:
                await pipe.watch(key)
                raw = await pipe.get(key)
                if raw is None:
                    raise SessionNotFoundError(session_id)
                
                state = AgentSessionState.model_validate_json(raw)
                updated = updater(state)
                updated = enforce_state_budget(updated)
                
                pipe.multi()
                pipe.set(key, updated.model_dump_json(), 
                         exat=int(updated.expires_at.timestamp()))
                await pipe.execute()
                return updated
            except redis.WatchError:
                continue  # retry on concurrent modification
```

**Memory sizing for Redis:** budget 5–15 KB per active session (state JSON + Redis overhead). For 100K concurrent sessions, that is 500 MB to 1.5 GB. If your workload exceeds this, shard across multiple Redis instances by hashing the session ID.

### PostgreSQL: The Durable Option

When your sessions need to outlive Redis's in-memory budget, or when you need complex queries over session data, PostgreSQL is the right answer.

PostgreSQL stores session state as `JSONB` columns, which gives you document-store flexibility with relational durability (WAL-based, ACID).

```sql
CREATE TABLE agent_sessions (
    id              TEXT        PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    schema_version  INTEGER     NOT NULL DEFAULT 2,
    state           JSONB       NOT NULL,
    
    -- Partial index for active sessions query
    CONSTRAINT sessions_not_expired CHECK (expires_at > created_at)
);

CREATE INDEX idx_sessions_user_active 
    ON agent_sessions (user_id, last_active_at DESC)
    WHERE expires_at > NOW();

CREATE INDEX idx_sessions_expires 
    ON agent_sessions (expires_at)
    WHERE expires_at < NOW() + INTERVAL '7 days';
```

The `JSONB` type lets you query into the state without deserializing it in application code:

```sql
-- Find all sessions where the agent is currently on step 3
SELECT id, user_id, state->'progress'->'current_step' as step
FROM agent_sessions
WHERE state->'progress'->>'current_step' = '3'
  AND expires_at > NOW();
```

**PostgreSQL read latency** is 2–10 ms for a simple primary-key lookup, versus Redis's sub-ms. For interactive agents, this adds perceptible latency. Mitigate with a read replica and connection pooling (PgBouncer, PgCat), or cache the state in Redis and write-through to PostgreSQL for durability.

The hybrid pattern — Redis as a hot cache with PostgreSQL as the durable store — is the production standard for agents that need both fast interactive response and durability:

```python
async def load_session(session_id: str) -> AgentSessionState:
    # Check Redis hot cache first
    raw = await redis.get(f"session:{session_id}")
    if raw:
        return AgentSessionState.model_validate_json(raw)
    
    # Fall back to PostgreSQL
    row = await pg.fetchrow(
        "SELECT state FROM agent_sessions WHERE id = $1", session_id
    )
    if row is None:
        raise SessionNotFoundError(session_id)
    
    state = AgentSessionState.model_validate(row["state"])
    
    # Warm the cache for subsequent requests
    await redis.set(f"session:{session_id}", state.model_dump_json(),
                    ex=300)  # 5-minute cache TTL
    return state
```

### DynamoDB: The Serverless Option

DynamoDB is the right choice when you need ops-free scalability and can live with its constraints.

**DynamoDB strengths:**
- Zero ops: no clusters to manage, no replication to configure
- Auto-scaling from 0 to millions of sessions without intervention
- Multi-AZ durability with strong read consistency available
- On-demand pricing means you pay per request, not for idle capacity

**DynamoDB constraints:**
- 400 KB item size limit — this is a hard ceiling. A session with a 200-turn conversation can easily exceed this.
- No rich query language — you get primary key lookups and simple range scans, nothing like SQL
- Global secondary indexes are eventually consistent by default
- Per-request pricing becomes expensive at high sustained throughput (Redis is cheaper above ~10K req/s)

For the 400 KB limit: split large state into multiple items keyed by `(session_id, chunk_index)`, or store the conversation history in S3 and keep only a reference in DynamoDB.

```python
DYNAMO_MAX_ITEM_SIZE = 380_000  # leave headroom below the 400 KB limit

def split_state_for_dynamo(state: AgentSessionState) -> list[dict]:
    """Split state into DynamoDB-sized chunks if needed."""
    full_json = state.model_dump_json().encode()
    
    if len(full_json) <= DYNAMO_MAX_ITEM_SIZE:
        return [{
            "PK": f"SESSION#{state.session_id}",
            "SK": "STATE#0",
            "chunk_count": 1,
            "data": full_json.decode(),
        }]
    
    chunks = []
    for i, offset in enumerate(range(0, len(full_json), DYNAMO_MAX_ITEM_SIZE)):
        chunk = full_json[offset:offset + DYNAMO_MAX_ITEM_SIZE]
        chunks.append({
            "PK": f"SESSION#{state.session_id}",
            "SK": f"STATE#{i}",
            "chunk_count": len(range(0, len(full_json), DYNAMO_MAX_ITEM_SIZE)),
            "data": chunk.decode("latin-1"),  # bytes as latin-1 for storage
        })
    return chunks
```

## 6. Checkpoint and Resume

For long-running agent tasks — research tasks taking 10–60 minutes, code generation tasks spanning hundreds of LLM calls — you need a checkpoint-resume system. The agent must be able to die at any point and resume from the last consistent state without repeating completed work.

![Checkpoint and resume flow — agent checkpoints after each step, new process resumes from last checkpoint on failure](/imgs/blogs/stateful-agent-deployment-4.webp)

### What Makes a Good Checkpoint

A checkpoint is a snapshot of the agent state at a point where resumption is safe. "Safe" means:

1. **All side effects up to this point are committed.** If step 3 sent a Slack message and step 4 updated a database row, the checkpoint after step 4 guarantees both those effects happened. A resume that starts after the checkpoint will not re-execute them.

2. **The checkpoint is self-contained.** The new process must be able to load only the checkpoint and resume, without needing to replay earlier steps or access data that has since been evicted.

3. **The checkpoint is consistent.** Partial writes are worse than no write — a checkpoint written halfway through step 3 will corrupt the resume logic. Use atomic writes.

### Checkpoint Granularity

The trade-off: fine-grained checkpoints mean less work to replay on failure; coarse-grained checkpoints mean less checkpoint overhead during normal execution.

**After each major step** is the standard in production. "Major step" means a semantically meaningful unit: "finished retrieving all sources," "finished drafting section 2," "finished calling the payment API." This typically corresponds to one LLM completion call plus its associated tool calls.

For tasks with external side effects (API calls, file writes, database mutations), checkpoint granularity should match the side-effect granularity — one checkpoint per irreversible action.

### The Checkpoint Implementation

```python
import asyncio
from contextlib import asynccontextmanager

class AgentExecutor:
    def __init__(self, session: AgentSessionState, state_backend: StateBackend):
        self.session = session
        self.backend = state_backend
    
    async def run_step(self, step_id: str, step_fn: Callable) -> Any:
        """Run a step with automatic checkpoint before and after."""
        # Skip if already completed (idempotency)
        if step_id in self.session.progress.completed_step_ids:
            print(f"Skipping step {step_id} — already completed")
            return self.session.working_memory.get(f"step_result:{step_id}")
        
        # Mark step as in-progress
        self.session.progress.current_step = int(step_id.split("_")[-1])
        await self.backend.save(self.session)  # pre-step checkpoint
        
        try:
            result = await step_fn(self.session)
            
            # Commit step completion
            self.session.progress.completed_step_ids.append(step_id)
            self.session.working_memory[f"step_result:{step_id}"] = result
            self.session.progress.last_checkpoint_at = datetime.utcnow()
            await self.backend.save(self.session)  # post-step checkpoint
            
            return result
        
        except Exception as e:
            # State is NOT updated — the pre-step checkpoint is still current
            # The supervisor can resume from this step safely
            raise AgentStepError(step_id, e) from e
    
    async def run_task(self, steps: list[tuple[str, Callable]]):
        """Run a multi-step task with checkpoint/resume support."""
        for step_id, step_fn in steps:
            await self.run_step(step_id, step_fn)
```

The key property: if the process dies at any point inside `run_step`, the state stored in the backend is either the pre-step checkpoint (if it dies before completion) or the post-step checkpoint (if it completes). The resume logic reads `completed_step_ids` to determine which steps to skip.

### Idempotency of Tool Calls

Tool calls that have side effects must be idempotent or deduplicated. Two patterns:

**Idempotency keys:** pass a unique key to external APIs that support them (Stripe, most REST APIs). The API deduplicates by key.

```python
async def call_tool_idempotent(tool_name: str, args: dict, 
                                 call_id: str) -> dict:
    # Check if we already have a result for this call_id
    existing = next(
        (tc for tc in self.session.tool_history if tc.call_id == call_id), 
        None
    )
    if existing and existing.status == "success":
        return existing.result
    
    # Make the call with idempotency key
    result = await tool_registry.call(
        tool_name, 
        {**args, "idempotency_key": call_id}
    )
    
    # Record in history
    self.session.tool_history.append(ToolCallEntry(
        call_id=call_id,
        tool_name=tool_name,
        arguments=args,
        result=result,
        status="success",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    ))
    
    return result
```

**Tool call history deduplication:** before executing a tool, check the history. If the same call (same function name, same arguments) appears with a success status, return the cached result. This is particularly important for read-heavy tools (web search, database queries) where re-execution is safe but wasteful.

## 7. Horizontal Scaling

A single-instance stateful agent works fine for development. Production systems serving thousands of concurrent sessions need multiple instances — and that is where state management gets complicated.

![Horizontal scaling with sticky sessions — load balancer routes by session-id hash to agent instances, all backed by shared Redis](/imgs/blogs/stateful-agent-deployment-5.webp)

### The Fundamental Challenge

If instance A holds an in-memory cache of session X's state and the load balancer routes the next request for session X to instance B, instance B needs to fetch state from the shared backend — incurring a ~1–10 ms round trip that would not have been necessary if the request had gone to instance A.

Two strategies for handling this:

### Sticky Sessions (Session Affinity)

Route all requests for a session to the same instance. The load balancer computes a hash of the session ID and consistently routes to the same backend.

```nginx
upstream agent_instances {
    hash $http_x_session_id consistent;  # consistent hashing
    server agent1:8080;
    server agent2:8080;
    server agent3:8080;
}
```

With consistent hashing, adding or removing a server only remaps ~1/N of the sessions (versus 100% remapping with simple modulo). This is critical for deployments with instance churn.

**Strengths of sticky sessions:**
- Requests hit the in-memory cache > 95% of the time after the first request
- Reduces backend reads significantly for high-traffic sessions

**Weaknesses:**
- When an instance dies, all its sticky sessions must be rehomed. The redirect latency is one request (the request that discovers the instance is dead, then fails over).
- Uneven traffic distribution if sessions have very different activity levels
- Sticky routing doesn't survive client-side load balancer switches (when the client is itself load-balanced)

The mitigation for instance death: always write state to the shared backend after every step. The in-memory cache is an optimization, not the source of truth. When the sticky instance dies, the new instance loads state from the backend with one extra round trip.

### Stateless Instances with Shared Backend

The alternative is to abandon in-memory caches entirely: every request reads state from Redis, runs the step, writes state back to Redis. No per-instance state at all.

This sounds wasteful — and it is, ~2–10 ms of extra Redis round trips per request. But for workflows where each agent step takes 2–30 seconds of LLM time, 10 ms is invisible. And it gives you:

- True horizontal scale-out: any instance can serve any session
- Zero per-instance state to reason about during deployments
- Rolling deploys with no session disruption

```python
@app.post("/sessions/{session_id}/messages")
async def handle_message(session_id: str, request: MessageRequest,
                          redis: Redis = Depends(get_redis)):
    # Load state from Redis — no in-memory cache
    state = await load_session(redis, session_id)
    
    # Run agent step
    result = await agent.run_step(state, request.message)
    
    # Write updated state back to Redis
    await save_session(redis, state)
    
    return {"response": result.response, "session_id": session_id}
```

**My recommendation:** start with stateless instances backed by Redis. It is operationally simpler and the performance overhead is negligible for LLM-heavy workloads. Add in-memory caching only when profiling shows Redis reads are measurably contributing to P99 latency.

### Distributed Locking

When two requests for the same session arrive simultaneously (possible in async event-driven architectures), you need to serialize writes. Redis provides a clean distributed lock primitive:

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def session_lock(redis: Redis, session_id: str, timeout: float = 5.0):
    """Distributed lock for session writes using Redis SET NX."""
    lock_key = f"session_lock:{session_id}"
    lock_value = generate_session_id()  # unique value to detect ownership
    
    acquired = await redis.set(lock_key, lock_value, nx=True, ex=int(timeout))
    if not acquired:
        raise SessionConflictError(f"Session {session_id} is locked by another request")
    
    try:
        yield
    finally:
        # Only release if we still own the lock (prevents releasing a lock we didn't set)
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await redis.eval(script, 1, lock_key, lock_value)

# Usage
async with session_lock(redis, session_id):
    state = await load_session(redis, session_id)
    updated = await agent.run_step(state, request.message)
    await save_session(redis, updated)
```

The Lua script for lock release is important: it is atomic (the get-then-delete is a single Redis operation) and prevents releasing a lock that has expired and been re-acquired by another request.

## 8. State Migration

Your agent's schema will change. New fields get added. Old fields get renamed. The state format that version 1.0 of your agent wrote is different from what version 2.0 expects.

![State migration strategies — four change types with their downtime requirements, rollback risk, and effort](/imgs/blogs/stateful-agent-deployment-8.webp)

Unlike a relational database where you can run an `ALTER TABLE` and all rows are immediately migrated, agent state migration must be incremental — you cannot take the state backend offline while millions of sessions are mid-conversation.

### The Four Change Types

**Adding a field** is the easiest. Write the new field with a safe default; old readers that don't know about the field simply ignore it. Old writers don't write the field; new readers get the default.

```python
class AgentSessionState(BaseModel):
    # ... existing fields ...
    
    # Added in schema_version=3
    reasoning_trace: list[str] = Field(default_factory=list)
    # ^ Pydantic's default_factory means existing sessions get an empty list
```

Zero downtime, zero rollback risk. Deploy new code, done.

**Renaming a field** requires a dual-write window. In schema_version=3, write both the old name and the new name. Readers in version 3 read from the new name first, fall back to the old name. After all sessions have been touched by version 3 code (tracked via the `schema_version` field), deploy version 4 that drops the old name.

```python
class AgentSessionState(BaseModel):
    schema_version: int = 3
    
    @model_validator(mode="after")
    def migrate_renamed_field(self):
        # Migrate step_count → current_step during deserialization
        if hasattr(self, "_old_step_count") and not self.progress.current_step:
            self.progress.current_step = self._old_step_count
        return self
```

**Removing a field** requires a grace period: stop writing the field first, then remove it from the schema after sufficient time for old sessions to expire or be touched by new code.

**Restructuring the schema** is the most dangerous change. It typically requires a migration batch job and a planned deployment window:

1. Deploy new code that can read both the old and new schema (`schema_version` branching)
2. Run a batch job to migrate all existing sessions to the new schema
3. Deploy code that only reads the new schema
4. Monitor for `schema_version=old` reads; investigate any that appear

### Lazy Migration Pattern

For large state stores, batch migration jobs are risky — they create write load spikes and must be carefully rate-limited. The lazy migration pattern migrates on read:

```python
async def load_and_migrate_session(session_id: str) -> AgentSessionState:
    raw = await redis.get(f"session:{session_id}")
    data = json.loads(raw)
    
    # Apply migrations based on schema version
    version = data.get("schema_version", 1)
    
    if version < 2:
        data = migrate_v1_to_v2(data)
    if version < 3:
        data = migrate_v2_to_v3(data)
    
    state = AgentSessionState(**data)
    
    # Write back the migrated version if it changed
    if version < CURRENT_SCHEMA_VERSION:
        await redis.set(f"session:{session_id}", state.model_dump_json(),
                        exat=int(state.expires_at.timestamp()))
    
    return state
```

This spreads migration load across normal traffic patterns and requires no maintenance window. The downside: old-format sessions persist until they are accessed. If you need to enforce "no old sessions" by a deadline, combine lazy migration with a batch job that processes sessions that have not been accessed recently.

### Schema Version Tracking

Track which schema version each session is using in the session metadata itself (`schema_version` field), and in a metrics counter per version. When `schema_version < CURRENT` reads drop to zero, you can safely remove the migration code path.

```python
metrics.counter("agent.session.schema_version", 
                tags={"version": str(state.schema_version)}).increment()
```

## 9. Session Isolation

In a multi-tenant deployment — a SaaS product where different users, companies, or teams share the same agent infrastructure — state isolation is a hard security requirement.

![Session isolation — flat key space causes cross-tenant state leakage; namespaced keys prevent it entirely](/imgs/blogs/stateful-agent-deployment-6.webp)

### The State Leakage Attack Surface

Without explicit isolation, several leakage vectors exist:

**Key collision:** if session IDs are not globally unique (for example, sequential integers from separate ID generators in a distributed system), two tenants can be assigned the same session ID, and one reads the other's state.

**Missing ownership check:** if the server does not verify that the requesting user owns the session being loaded, any user with a guessed or enumerated session ID can read another user's conversation history, tool history, and working memory.

**Shared working memory namespaces:** if working memory keys like `retrieved_doc_1` are not namespaced by session, two concurrent sessions in the same process can accidentally share working memory via the same key.

**Log and trace leakage:** if conversation content is logged to a shared logging system without redaction, an operator querying logs for one session inadvertently sees another user's content.

### Namespace Everything

The defense is to namespace every key by user and session:

```python
# Redis key convention
def session_key(user_id: str, session_id: str) -> str:
    return f"u:{user_id}:s:{session_id}"

def user_sessions_key(user_id: str) -> str:
    return f"u:{user_id}:sessions"

# These are different keys even if session IDs were somehow equal
key_user_a = session_key("user-alice", "abc123")  # u:user-alice:s:abc123
key_user_b = session_key("user-bob", "abc123")    # u:user-bob:s:abc123
```

**Enforce namespace at the middleware layer,** not in business logic. A middleware class that wraps the state backend and injects the user context into every key:

```python
class NamespacedStateBackend:
    def __init__(self, backend: StateBackend, user_id: str):
        self._backend = backend
        self._user_id = user_id
    
    def _namespace_key(self, session_id: str) -> str:
        return f"u:{self._user_id}:s:{session_id}"
    
    async def get(self, session_id: str) -> AgentSessionState | None:
        return await self._backend.get(self._namespace_key(session_id))
    
    async def set(self, session_id: str, state: AgentSessionState) -> None:
        if state.user_id != self._user_id:
            raise SessionOwnershipError("cannot write state for different user")
        await self._backend.set(self._namespace_key(session_id), state)
```

This makes it impossible for a request handler to accidentally access another user's state, regardless of what session ID the client claims.

### Tenant-Level Isolation Boundaries

For B2B SaaS deployments where isolation must be at the company (tenant) level, not just the user level:

```python
def session_key(tenant_id: str, user_id: str, session_id: str) -> str:
    return f"t:{tenant_id}:u:{user_id}:s:{session_id}"
```

You can then use Redis keyspace notifications or access control lists to limit a tenant's service account to only keys under their tenant prefix — even an operator querying Redis will be constrained to their tenant's keyspace.

For the most sensitive deployments (healthcare, finance), consider **physical isolation**: separate Redis instances per tenant, separate database schemas, separate Kubernetes namespaces. The cost is higher operational complexity; the benefit is that a configuration error in one tenant cannot affect another.

## 10. Failure Recovery

Even with checkpoints and durable state, things fail. Processes die. Network partitions occur. The state backend becomes unavailable. Production-grade stateful agents need an explicit failure recovery path.

![Failure recovery flow — supervisor detects agent death within one health-check interval and restores session to last checkpoint](/imgs/blogs/stateful-agent-deployment-7.webp)

### What Failure Looks Like

**Process death (SIGKILL, OOM):** the process is terminated mid-step. The in-memory state of that step is lost. The last committed checkpoint in the state backend is the recovery point.

**Network partition:** the agent cannot reach the state backend for N seconds. In-progress steps may have been executed (side effects committed to external systems) but not checkpointed. On recovery, you must detect uncommitted side effects.

**State backend failure:** Redis cluster goes down, DynamoDB has an availability event. In-flight writes are lost. The last successfully written state is the recovery point — which may be several steps behind if you lost the backend mid-task.

**Infinite loop or stall:** the agent enters a reasoning loop, making LLM calls indefinitely. The process is alive but not making progress. This requires a timeout + forced checkpoint + resume with a modified context ("you've been trying this for 30 minutes, try a different approach").

### The Supervisor Pattern

A supervisor process monitors agent processes and coordinates recovery:

```python
import asyncio
from datetime import datetime, timedelta

class AgentSupervisor:
    def __init__(self, state_backend: StateBackend, 
                 dead_letter_queue: Queue):
        self.backend = state_backend
        self.dlq = dead_letter_queue
    
    async def watch_session(self, session_id: str, timeout: timedelta):
        """Watch a running session and trigger recovery on failure."""
        deadline = datetime.utcnow() + timeout
        
        while datetime.utcnow() < deadline:
            await asyncio.sleep(10)  # health check interval
            
            state = await self.backend.get(session_id)
            if state is None:
                return  # session was cleaned up normally
            
            if state.progress.last_checkpoint_at < datetime.utcnow() - timedelta(minutes=5):
                # No checkpoint in 5 minutes — agent may be stuck or dead
                await self.trigger_recovery(session_id, state)
                return
    
    async def trigger_recovery(self, session_id: str, 
                               state: AgentSessionState):
        """Attempt to recover a stuck or dead session."""
        # Send to dead letter queue for retry
        await self.dlq.enqueue({
            "type": "session_recovery",
            "session_id": session_id,
            "last_step": state.progress.current_step,
            "completed_steps": state.progress.completed_step_ids,
            "triggered_at": datetime.utcnow().isoformat(),
        })
        
        # Update state to mark recovery in progress
        state.working_memory["recovery_attempt_at"] = datetime.utcnow().isoformat()
        await self.backend.save(state)
```

### Dead Letter Queue for Task Recovery

When an agent session fails, the task should not be silently dropped. Put it in a dead letter queue (DLQ) for retry:

```python
async def process_recovery_from_dlq(message: dict):
    session_id = message["session_id"]
    
    # Load state from backend
    state = await state_backend.get(session_id)
    if state is None:
        logger.error(f"Recovery failed: session {session_id} not found in backend")
        return
    
    # Validate state integrity
    if state.schema_version != CURRENT_SCHEMA_VERSION:
        state = await migrate_state(state)
    
    # Resume from last checkpoint
    executor = AgentExecutor(state, state_backend)
    try:
        await executor.resume_task()
    except Exception as e:
        logger.error(f"Recovery of {session_id} failed after resumption: {e}")
        await notify_user(state.user_id, session_id, "Your task failed and could not be recovered.")
        raise
```

### Compensating Transactions

When a step has committed side effects (created a file, sent an email, charged a card) and then fails before checkpointing, you have two choices:

1. **Accept the duplicate:** if the side effect is idempotent or observable (the user can see the created file), accept that it happened and design the resume logic to skip that step.

2. **Compensate:** implement a compensating transaction — a reverse operation that undoes the side effect. (For a file creation: delete the file. For a database insert: delete the row by its generated ID.)

Most production systems choose option 1 for simplicity. True compensating transactions are complex and error-prone. Design your tools to be idempotent where possible (using idempotency keys with external APIs) and document "this step may have already happened" in your recovery runbook.

## 11. State Size Management

The most insidious production problem with stateful agents is state growth. A research agent that runs for 2 hours can accumulate 5–20 MB of state. At that size, loading the state takes 50–200 ms, serializing it takes another 50 ms, and feeding the conversation history to the LLM consumes the majority of a 128K context window.

![State size growth and pruning strategies — raw growth hits context limits; summarization and sliding window keep size bounded](/imgs/blogs/stateful-agent-deployment-9.webp)

### The Three Pruning Strategies

**Sliding window:** keep only the last N turns of conversation history. Simple to implement, effective at bounding size. The downside is abrupt information loss — if the user said something important in turn 1 and the window is 20, that information is gone by turn 21.

```python
SLIDING_WINDOW = 20

def apply_sliding_window(messages: list[MessageEntry]) -> list[MessageEntry]:
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    
    if len(non_system) <= SLIDING_WINDOW:
        return messages
    
    # Keep system prompt + last N turns
    return system_msgs + non_system[-SLIDING_WINDOW:]
```

**Summarization:** when the context window exceeds a threshold (e.g., 75%), use the LLM itself to summarize the earliest turns into a compact representation. The summary replaces the original turns.

```python
async def summarize_and_prune(state: AgentSessionState, 
                               llm: LLM,
                               threshold: float = 0.75) -> AgentSessionState:
    token_count = count_tokens(state.messages)
    max_tokens = llm.context_window * threshold
    
    if token_count <= max_tokens:
        return state
    
    # Identify turns to summarize (keep the most recent N tokens worth)
    to_keep_tokens = int(max_tokens * 0.4)  # keep 40% as-is
    recent_msgs, old_msgs = split_by_token_budget(state.messages, to_keep_tokens)
    
    if not old_msgs:
        return state
    
    # Summarize the old messages
    summary_prompt = f"""Summarize the following agent conversation history concisely, 
    preserving key decisions, completed work, and any information that will be needed
    to continue the task:
    
    {format_messages(old_msgs)}
    """
    
    summary = await llm.complete(summary_prompt)
    
    # Replace old messages with summary
    summary_msg = MessageEntry(
        role="system",
        content=f"[CONVERSATION SUMMARY - turns 1-{len(old_msgs)}]\n{summary}",
        timestamp=datetime.utcnow(),
    )
    
    state.messages = [summary_msg] + recent_msgs
    return state
```

**Selective eviction from working memory:** rather than blanket eviction, use a relevance score to decide what to keep. Recent additions and items that have been referenced in the last N LLM calls get higher scores; old, unreferenced items get evicted first.

```python
def evict_working_memory(
    working_memory: dict, 
    access_log: list[str],
    budget_kb: int = 512
) -> dict:
    """Evict working memory entries by LRU, keeping total size under budget."""
    size_kb = len(json.dumps(working_memory).encode()) / 1024
    
    if size_kb <= budget_kb:
        return working_memory
    
    # Build LRU order — last access time from access_log
    access_times = {}
    for entry in access_log:
        access_times[entry] = access_log.index(entry)  # last occurrence = most recent
    
    # Sort keys by last access time (oldest first)
    sorted_keys = sorted(
        working_memory.keys(),
        key=lambda k: access_times.get(k, 0)
    )
    
    evicted = dict(working_memory)
    for key in sorted_keys:
        if size_kb <= budget_kb:
            break
        entry_size = len(json.dumps({key: working_memory[key]}).encode()) / 1024
        del evicted[key]
        size_kb -= entry_size
    
    return evicted
```

### Monitoring State Size

Instrument your state backend writes to track state size distributions:

```python
async def save_session(state: AgentSessionState) -> None:
    serialized = state.model_dump_json()
    size_bytes = len(serialized.encode())
    
    # Metrics
    metrics.histogram("agent.session.state_size_bytes", size_bytes,
                      tags={"user_tier": state.config.get("tier", "free")})
    metrics.histogram("agent.session.message_count", len(state.messages))
    metrics.histogram("agent.session.tool_history_count", len(state.tool_history))
    
    # Alerts
    if size_bytes > 1_000_000:  # 1 MB
        logger.warning(f"Large session state: {state.session_id} is {size_bytes/1024:.0f} KB")
    
    await backend.set(state.session_id, serialized)
```

Alert when sessions exceed 500 KB. Investigate when they exceed 2 MB. A session exceeding 5 MB is almost always a bug — a tool call returning a full database dump, or a loop that failed to evict old results.

## 12. Case Studies

### Case Study 1: Customer Support Agent with 24-Hour Sessions

A B2B SaaS company deployed a customer support agent that handles multi-day support tickets. A user opens a ticket on Monday, the agent helps with initial diagnosis, and the user may return on Tuesday with more information.

**Challenge:** the 24-hour session requirement meant Redis alone was insufficient — Redis state would be lost if the instance was restarted for maintenance. The company also needed to query "all open tickets for customer X" — impossible with a pure key-value store.

**Solution:** hybrid storage. All active sessions are in Redis with a 30-minute idle TTL. A background job (running every 5 minutes) writes any session touched in the last 5 minutes to PostgreSQL. The PostgreSQL row includes a `jsonb` state column and a `last_active_at` indexed column for cross-customer queries.

When a session expires from Redis (user was idle for 30 minutes), the PostgreSQL row remains as the durable record. On the user's return, the handler checks Redis first (miss), then loads from PostgreSQL, writes back to Redis, and the session is warm again.

**Results:** Redis memory usage dropped 70% compared to the original 24-hour Redis TTL approach. PostgreSQL query latency for "list tickets for customer X" is ~5 ms. Session resumption after a Redis miss takes ~30 ms.

**Lesson:** the hybrid approach is not a premature optimization — it is the correct tool for requirements that span a hot cache and a durable store.

### Case Study 2: Code Generation Agent with Checkpoint-Resume

A developer tool company built a code generation agent that could generate entire applications from a specification — a 20–60 minute task involving 50–200 LLM calls and multiple external tool executions (filesystem writes, test runner calls, linting passes).

**Challenge:** at 20–60 minutes per task, process crashes were a regular occurrence (Kubernetes pod evictions, node preemptions, spot instance reclamations). Without checkpoint-resume, a crash at minute 50 meant starting over from scratch. Users were understandably angry.

**Solution:** every "file written" and every "test passed" event triggered an immediate checkpoint write to PostgreSQL. The progress marker included a list of generated file paths and their content hashes, plus the list of passed tests. On resume, the agent diffed the current filesystem against the checkpoint and re-generated only files that were missing or corrupted.

The team added an important optimization: rather than storing generated file contents in the session state (multi-MB), they stored the file paths and Git commit hashes. The actual files lived in an ephemeral workspace (a mounted volume) that survived pod evictions but not node replacements. For node-replacement recoveries, the files were regenerated from the LLM using the same step IDs (cache hits in the LLM provider's caching layer).

**Results:** task recovery rate improved from "always start over" to >92% of crashes recovering within 30 seconds. The remaining 8% were cases where the PostgreSQL backend was also under stress during the incident.

**Lesson:** don't store large generated artifacts in the session state. Store references (file paths, commit hashes, content-addressable storage keys) and regenerate from the checkpoint markers.

### Case Study 3: Financial Analysis Agent Under Strict Isolation

A fintech company deployed an AI agent that had access to multiple customers' portfolio data. Regulatory requirements mandated that no data from customer A could be visible to customer B's session — even in the same process.

**Challenge:** the default multi-tenant model (namespace by user ID) was not sufficient. The regulators required physical isolation: customer A's data must not be co-resident in memory with customer B's data, even momentarily.

**Solution:** separate Redis instances per large customer tier. Each enterprise customer got a dedicated Redis instance, a dedicated PostgreSQL schema, and agent instances that processed only that customer's sessions (via Kubernetes label selectors). The shared infrastructure handled only the "free" and "basic" tiers, using strict namespace isolation and separate Redis keyspace permissions.

The tool registry was also scoped: before the agent could call any tool, a middleware layer checked that the tool's data source was in the permitted set for the current tenant. A bug that granted a customer access to a wrong tool would fail at the data access layer, not just at the session boundary.

**Results:** audit passed. Physical isolation eliminated the class of "impermissible data access" bugs entirely — you cannot accidentally leak data across a process boundary that doesn't exist.

**Lesson:** for regulated industries, namespace isolation is necessary but not sufficient. Physical isolation (separate infrastructure) is the only way to make a hard guarantee.

### Case Study 4: Research Agent with Aggressive State Pruning

A research tool startup built an agent that performed multi-day research tasks — gathering information over 4–8 hours, then producing reports. The natural state size (all retrieved documents, all web pages, all LLM reasoning traces) was reaching 15–25 MB per session.

**Challenge:** 20 MB of state serialized to Redis on every step update took 200–300 ms. With steps averaging 3 seconds of LLM time, this overhead was 7–10% of per-step latency — noticeable and growing as sessions aged.

**Solution:** a tiered storage model. "Hot" working memory (documents referenced in the last 10 LLM calls, intermediate results of the current step) stayed in the session state blob — capped at 500 KB. "Warm" memory (documents from earlier in the session) was stored in a content-addressable cache (Redis with a content hash as key) with a 24-hour TTL. "Cold" memory (documents not referenced in the last 30 minutes) was moved to S3.

The session state blob held only references (S3 URIs, content hashes) to warm and cold memory. When the LLM needed to reference a warm document, the retrieval step fetched it from the warm cache (~5 ms). Cold retrieval from S3 took ~100 ms but was rare.

**Results:** median session state blob dropped from 18 MB to 340 KB — a 98% reduction. State serialization overhead dropped from 250 ms to ~4 ms. The agent could now run sessions 10× longer before any degradation.

**Lesson:** working memory belongs in a content-addressed cache, not in the session state blob. The session blob should hold references and metadata; the heavy content should live in a purpose-built store.

### Case Study 5: High-Throughput Session Migration Without Downtime

A mature ML platform company needed to migrate 8 million active agent sessions from schema v2 to schema v3. The v3 schema restructured the progress marker format: `current_step: int` became `current_step: { step_id: str, index: int, total: int }`. This was a breaking structural change, not a simple field addition.

**Challenge:** 8 million sessions across 40 Redis clusters. A batch migration job running at 10K sessions/second would take 13 minutes — during which any read of a not-yet-migrated session by new code would fail.

**Solution:** lazy migration in the read path, with a deployment strategy that ensured the old schema was always readable by both old and new code during the transition.

Phase 1 (day 1): deploy code that writes v3 format but reads both v2 and v3. All new sessions are v3. All reads of v2 sessions trigger lazy migration in-place (read, convert, write-back).

Phase 2 (week 2): run a background migration job at 500 req/s to convert any sessions that have been untouched (i.e., haven't been lazily migrated by normal traffic). This targets sessions with no recent activity.

Phase 3 (week 4): monitor `schema_version=2` counter in metrics. When it drops below 100 sessions/day (expired edge cases), deploy code that drops v2 support entirely.

**Results:** zero downtime. Total elapsed calendar time: 28 days. Total engineering time: 2 days for implementation, 26 days of waiting for old sessions to naturally expire or be migrated.

**Lesson:** lazy migration is almost always the right strategy for live systems. The cost is keeping the migration code for longer, but the benefit is zero-risk deployment.

### Case Study 6: Sticky Sessions Failure Mode

An agent platform serving enterprise customers used nginx sticky sessions with the `ip_hash` directive for session affinity. This worked fine for months.

Then they onboarded a large enterprise customer whose employees all exited through a single corporate NAT gateway — a single external IP. All of that customer's sessions were routed to one agent instance, which was now handling 40% of total traffic. The instance OOMed under the load.

**The diagnosis** took 2 hours. The `ip_hash` directive was invisible in normal metrics — the load appeared balanced by IP distribution, but not by session count or memory consumption.

**Solution:** switch from `ip_hash` to `hash $cookie_session_id consistent`. The session ID, not the client IP, became the affinity key. This distributes load based on session identity, which is uncorrelated with the client's network topology.

Add a metric: `agent_instance_session_count` — the number of active sessions being served by each instance. Alert when any instance exceeds 2× the mean. This catches imbalance from any cause.

**Lesson:** `ip_hash` sticky sessions are a trap for enterprise deployments with NAT gateways. Always use session-ID-based affinity.

### Case Study 7: Redis Cluster Failure and Session Recovery

A team running 4,000 concurrent agent sessions on a 3-shard Redis cluster with AOF persistence experienced a shard failure at 2 AM. The primary of shard 2 died of a hardware fault. The replica promoted within 20 seconds, but the last ~5 seconds of writes were lost (the AOF fsync interval was 1 second, but the replica replication lag was ~5 seconds).

**Impact:** ~400 sessions (those actively writing to shard 2 in the 5-second window) had their last checkpoint lost. Their state regressed to the version from 5 seconds earlier.

**Of those 400:**
- 350 had checkpoints within the last 30 seconds. They resumed at the step before the last one (one step of duplicated work — acceptable with idempotent tool calls).
- 40 had been mid-step when the failure occurred. On resume, the agent re-ran the step and the idempotency key in the external API deduplicated the tool call. No double-execution.
- 10 had been mid-step on a non-idempotent tool (a tool that returned unique data on each call — an RNG query). These 10 got slightly different data on the re-run. Not a corruption, but a behavioral difference.

**Post-mortem improvements:**
1. Reduce Redis replica lag to < 1 second (by moving from async replication to semi-sync: the primary waits for at least 1 replica to ack before confirming the write).
2. Mark all non-idempotent tool calls in the tool registry. The checkpoint system writes a pre-call checkpoint specifically before non-idempotent calls.
3. Add `Redis Sentinel` with 3 nodes to reduce failover time from 20 seconds to 5 seconds.

**Lesson:** Redis cluster replication lag is the true durability bound, not the AOF fsync interval. Design for the replica lag, not the fsync.

### Case Study 8: The Summarization Loop Bug

A team shipped a new summarization feature: when conversation history exceeded 60% of the context window, the agent would summarize the oldest 40% of messages. This worked correctly in testing.

In production, a class of sessions started summarizing every 2–3 steps, instead of every 20–30 steps. Investigation revealed the cause: the token counter was using the tokenizer for the wrong model. The deployed model (GPT-4o) has a context window of 128K tokens with a compact tokenizer, but the counter was using the GPT-3.5 tokenizer, which produces ~30% more tokens for the same text. Every session looked like it was at 78% context utilization when it was actually at 60%.

The agent was spending 15% of its step time doing unnecessary summarizations. Worse, each summarization threw away information — the summary was lossy. Sessions where the real context usage was fine were progressively losing early conversation context.

**Fix:** validate the tokenizer against the deployed model at startup, fail fast if they don't match.

```python
def validate_tokenizer_model_match(tokenizer: Tokenizer, llm: LLM):
    test_text = "The quick brown fox jumps over the lazy dog."
    tokenizer_count = tokenizer.count_tokens(test_text)
    llm_count = llm.count_tokens(test_text)  # via the model's token counting API
    
    if abs(tokenizer_count - llm_count) / llm_count > 0.05:
        raise ConfigurationError(
            f"Tokenizer mismatch: local={tokenizer_count}, model={llm_count} "
            f"for '{test_text}'. Update tokenizer to match {llm.model_id}."
        )
```

**Lesson:** the context window percentage used to trigger summarization is computed relative to the model's tokenizer, not a general-purpose one. This is not obvious, and the failure mode is subtle — the agent still works, just worse. Validate your tokenizer against your model in CI.

## 13. When to Go Stateless

Not every agent needs stateful deployment. Before building out session storage, Redis clusters, and checkpoint infrastructure, ask whether the simpler architecture would serve your needs.

![Stateless vs stateful architecture decision tree — based on task duration, user continuity, and scale requirements](/imgs/blogs/stateful-agent-deployment-10.webp)

### The Case for Stateless

**Single-turn tasks with complete context:** if every request to the agent includes all necessary context (a document to analyze, a code snippet to explain, a question to answer), there is no state to carry between requests. Deploy as a stateless HTTP endpoint, scale horizontally, done.

**Short conversation horizons:** if users never have conversations longer than 5–10 turns, you can pass the full conversation history in the request and store it on the client. The client is your state backend. This is the architecture used by most consumer AI products — the mobile app stores the message array and sends it with every request.

**Ephemeral tasks with acceptable re-derivation cost:** if re-running from scratch takes 2 seconds and costs $0.01, and crash rates are < 0.1%, the expected re-derivation cost is $0.00001 per task. You can just absorb crashes by retrying.

**Prototype and development:** stateful infrastructure is operational complexity. For a prototype, store everything in SQLite on the development machine and worry about production infrastructure later.

### The Stateless Trap

The failure mode is building a stateless architecture for a use case that is secretly stateful, then papering over the gaps with client-side state management and discovering too late that the client state is inconsistent, too large to transmit efficiently, or insecure.

Signs that you need to go stateful:

- Conversation histories are growing beyond 50 KB per session
- Users are experiencing 5+ minute tasks that are failing to complete
- The "context" passed in each request contains data that should not be on the client (tool call results, internal reasoning steps, retrieved documents)
- You are implementing custom client-side state sync logic to handle concurrent requests

Stateful infrastructure is the right investment when at least two of these are true: tasks run longer than 5 minutes, users return to conversations after > 30 minutes away, or you need resumability across process restarts.

## 14. Observability and Debugging Stateful Sessions

Stateless services are easy to debug: every log line is self-contained, every request is independent. Stateful agents are much harder — a misbehaving agent in step 12 may have been caused by something that happened in step 3. Without structured observability, debugging requires replaying hours of logs in your head.

### Session-Scoped Logging

The foundational practice: every log line emitted during an agent's execution must include the session ID and the current step index. Without this, correlating logs across a multi-step task is nearly impossible.

```python
import structlog

def get_session_logger(session_id: str, step_index: int) -> structlog.BoundLogger:
    return structlog.get_logger().bind(
        session_id=session_id,
        step_index=step_index,
        agent_version=os.environ.get("AGENT_VERSION", "unknown"),
    )

# Usage
log = get_session_logger(state.session_id, state.progress.current_step)
log.info("tool_called", tool_name="web_search", args={"query": query})
log.info("tool_completed", tool_name="web_search", result_size_bytes=len(result))
```

With session-scoped logging, you can reconstruct the full timeline of any session by filtering on `session_id`. This is the minimum needed to debug production incidents.

### Step-Level Spans

For latency analysis, instrument each agent step as a distributed trace span. The trace captures: step duration, LLM call duration within the step, tool call durations, and state read/write durations.

```python
from opentelemetry import trace

tracer = trace.get_tracer("agent")

async def run_step_with_tracing(session_id: str, step_id: str, 
                                 step_fn: Callable) -> Any:
    with tracer.start_as_current_span(f"agent.step.{step_id}") as span:
        span.set_attribute("session.id", session_id)
        span.set_attribute("step.id", step_id)
        
        start = time.monotonic()
        try:
            result = await step_fn()
            span.set_attribute("step.status", "success")
            return result
        except Exception as e:
            span.set_attribute("step.status", "error")
            span.set_attribute("step.error", str(e))
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            span.set_attribute("step.duration_ms", duration_ms)
```

The trace hierarchy: a root span per session task, child spans per step, grandchild spans per LLM call and tool call. This gives you waterfall views of where time goes across the full task.

### Session State Inspection

In production, you will occasionally need to inspect the live state of a running session — to debug why an agent is stuck, to verify a specific checkpoint was written, or to manually advance a stuck session's step counter.

Build a read-only admin API over the state backend:

```python
@app.get("/admin/sessions/{session_id}/state")
async def inspect_session(session_id: str, 
                           admin: AdminUser = Depends(require_admin)):
    state = await state_backend.get(session_id)
    if state is None:
        raise HTTPException(404)
    
    return {
        "session_id": state.session_id,
        "user_id": state.user_id,
        "schema_version": state.schema_version,
        "message_count": len(state.messages),
        "tool_call_count": len(state.tool_history),
        "current_step": state.progress.current_step,
        "completed_steps": state.progress.completed_step_ids,
        "last_checkpoint_at": state.progress.last_checkpoint_at,
        "state_size_bytes": len(state.model_dump_json().encode()),
        # Deliberately NOT returning conversation history — PII
    }
```

Omit conversation history and working memory from admin endpoints — they contain user PII. Return structural metadata only.

### Alerting on Session Health

Key metrics to alert on:

| Metric | Alert threshold | What it means |
|---|---|---|
| `session.state_size_bytes` p99 | > 2 MB | State growing too large, pruning may be failing |
| `session.step_duration_ms` p99 | > 30 s | Steps taking too long, possible stall |
| `session.checkpoint_age_minutes` max | > 10 min | Checkpointing not working, failure won't recover |
| `session.schema_migration_count` | > 0 | Sessions being migrated (expected during rollout) |
| `session.recovery_trigger_count` | > 0 | Sessions being recovered (unexpected, investigate) |
| `state_backend.error_rate` | > 0.1% | Backend degradation, risk of data loss |

Set these alerts before shipping to production. The `checkpoint_age_minutes` alert is particularly important: a session that hasn't checkpointed in 10 minutes is either stuck or not checkpointing at all, and the next crash will cause maximum data loss.

## 15. Session Expiry and Storage Lifecycle

Sessions do not live forever, and managing their lifecycle correctly prevents storage leakage and ensures users get predictable behavior when they return to old conversations.

### Expiry Strategy

There are two philosophically different approaches to session expiry:

**Eager expiry:** expire sessions after a fixed absolute TTL (e.g., 7 days), regardless of activity. Simple to reason about, easy to implement with Redis TTL or a PostgreSQL scheduled job. The user might return on day 8 and find their session gone — this is a UX decision, not a technical failure.

**Lazy expiry with renewal:** sessions have a base TTL but it is extended on each access (a "sliding window TTL"). An active user can maintain a session indefinitely. Implemented by updating the `expires_at` field on every write.

```python
IDLE_TTL = timedelta(hours=24)
MAX_SESSION_AGE = timedelta(days=30)

async def renew_session_ttl(state: AgentSessionState) -> AgentSessionState:
    now = datetime.utcnow()
    max_expiry = state.created_at + MAX_SESSION_AGE
    
    # Extend by idle TTL, but never past the absolute maximum
    new_expiry = min(now + IDLE_TTL, max_expiry)
    state.expires_at = new_expiry
    return state
```

The `MAX_SESSION_AGE` cap prevents sessions from living forever even for active users — important for storage budgeting and for ensuring stale state eventually gets cleaned up.

### Soft Delete and Grace Period

Rather than immediately deleting state on expiry, use a soft-delete pattern:

1. When `expires_at` passes, mark the session as `status: expired` but don't delete the state yet.
2. Keep it for a configurable grace period (e.g., 48 hours).
3. A background job deletes expired-and-past-grace-period sessions.

The grace period serves two purposes: it allows users to "undo" accidental session closures (by requesting session restoration within the grace period), and it provides a buffer for any delayed requests that arrive after the session has "officially" expired.

### Storage Budget Accounting

With thousands or millions of sessions, you need to account for storage usage:

```python
async def compute_storage_usage(tenant_id: str) -> dict:
    """Compute total storage used by a tenant across all sessions."""
    sessions = await pg.fetch(
        """
        SELECT 
            COUNT(*) as session_count,
            SUM(octet_length(state::text)) as total_state_bytes,
            AVG(octet_length(state::text)) as avg_state_bytes,
            MAX(octet_length(state::text)) as max_state_bytes
        FROM agent_sessions
        WHERE tenant_id = $1 
          AND expires_at > NOW()
        """,
        tenant_id
    )
    
    return {
        "tenant_id": tenant_id,
        "active_session_count": sessions[0]["session_count"],
        "total_state_mb": sessions[0]["total_state_bytes"] / (1024 * 1024),
        "avg_state_kb": sessions[0]["avg_state_bytes"] / 1024,
        "max_state_kb": sessions[0]["max_state_bytes"] / 1024,
    }
```

Alert when any tenant's total state storage exceeds their tier limit. This is both a billing signal and a health signal — a tenant with 50 GB of session state likely has a runaway agent or a bug in their state cleanup logic.

## The Operational Minimum

If you take nothing else from this post, take these five rules:

1. **Write state to the backend after every major step, not at the end of the task.** The end of the task is when crashes and preemptions happen.

2. **Namespace every key by user ID.** `session:{id}` is a multi-tenant security vulnerability. `u:{user_id}:session:{id}` is not.

3. **Always check session ownership before loading.** The session ID in the request header is user-controlled input. Verify it belongs to the authenticated user.

4. **Version your state schema from day one.** A `schema_version: int = 1` field in your state model costs nothing and saves days of work when you need to migrate.

5. **Set a state size budget and enforce it at every write.** A 500 KB budget per session is generous for most use cases and keeps serialization fast. Don't let sessions grow unbounded and discover the problem at 2 MB.

For deeper context on related challenges: if you need to understand how agents handle very long-running async tasks, see [async long-running agents](/blog/machine-learning/ai-agent/async-long-running-agents). For the observability layer that makes stateful session debugging tractable, see [agent observability and tracing](/blog/machine-learning/ai-agent/agent-observability-and-tracing). For the context window management that determines when state pruning must trigger, see [context window management](/blog/machine-learning/ai-agent/context-window-management).

Stateful agents are a distributed systems problem. The patterns above — checkpointing, namespacing, lazy migration, supervised recovery — are the same patterns that distributed databases use to survive real-world conditions. The difference is that your state is semantically coupled to an LLM's reasoning, which makes the correctness requirements both harder and more interesting. Good luck shipping them.
