---
title: "Async and Long-Running Agents: Building Tasks That Take Minutes or Hours"
date: "2026-06-27"
description: "How to build agents that run asynchronously for minutes or hours — task queues, polling vs webhooks, progress reporting, timeout handling, and the UX patterns that make long-running agents feel responsive."
tags: ["ai-agents", "async", "long-running", "task-queues", "production-ml", "llm", "machine-learning", "system-design"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 38
---

There is a wall that every team building AI agents eventually hits. The agent feels fast in the notebook. You hand it a real task — draft a competitive analysis, run a multi-step code review, synthesize a week of documents — and the HTTP connection sits at "loading" for two and a half minutes before the browser decides you must be dead. You are not dead. You just discovered the synchronous ceiling.

The ceiling is exactly where HTTP was never designed to live: holding an open request-response cycle for the duration of a task that calls an LLM six times, spawns five tool calls, and fetches three external APIs. Every layer between the user's browser and your agent has a timeout opinion. The browser has one (usually two minutes). The reverse proxy has one (Nginx defaults to 60 seconds). The load balancer has one. None of them care that your agent is 80% done.

The fix is not "add a longer timeout." The fix is to stop confusing submission with execution. The diagram above is the mental model: the user's action and the agent's work happen on different timelines, connected by a lightweight task ID.

![Sync vs async agent: the request-response ceiling](/imgs/blogs/async-long-running-agents-1.webp)

This post is a complete engineering guide to async and long-running agent systems. We cover the full stack: why sync breaks, how the async architecture is structured, which queue backend to pick and why, how callers track tasks, the three notification patterns (polling, webhooks, SSE) and when to use each, how to stream progress to users in real time, how to layer timeouts so nothing hangs silently, how to retry without duplicating work, how to handle priority so urgent tasks are never queued behind a batch job, and how to scale the worker pool. We close with six case studies from production systems and a concrete decision framework for when async is worth the added complexity.

## 1. The Synchronous Ceiling: Why Request-Response Breaks for Agent Tasks

Let us be precise about what breaks. HTTP is synchronous at the transport layer: the client opens a TCP connection, sends a request, and waits. Servers, proxies, and load balancers treat a connection that has received no response bytes as stalled and cut it at their configured timeout. This is the right default for API calls that should finish in milliseconds. It is the wrong default for an agent that is chaining together six LLM calls and a web search.

The specific failure modes are:

**Gateway timeout (502/504).** Your load balancer or reverse proxy kills the upstream connection and returns a 5xx to the client. The agent might be 90% done on the worker — the caller will never know.

**Client timeout.** The browser or mobile app gives up and shows the user an error. The agent finishes anyway (or does it? the worker might not even know the caller disconnected), and the result is lost.

**Memory pressure from held connections.** Each synchronous in-flight agent task holds one worker thread (or one async task slot) plus one open connection. At ten concurrent users running five-minute tasks, you are holding fifty slots open. The math gets ugly fast.

**No progress visibility.** The user stares at a spinner. They have no idea if the agent is on step 1 or step 8. Retry triggers more duplicate work.

**Cascading retries.** The client sees a timeout and retries. The first instance is still running. You now have two identical agent runs consuming compute and LLM budget.

There is also a subtler cost: the connection is not free. On the server side, a synchronous framework holds a thread (or a coroutine) per request. For a Python FastAPI app, that means your thread pool is exhausted by ten users each running three-minute tasks. For an async framework like asyncio, you are still holding an open file descriptor and a live coroutine, which is cheaper but not free.

The solution is conceptually simple: separate submission from execution. The caller submits a task and immediately gets back a small acknowledgment. The task runs on a worker. The caller later retrieves the result. This is the design of every reliable background-work system, from database batch jobs to email delivery. The difference for AI agents is that tasks are stateful, non-deterministic, and expensive to restart — which creates requirements that generic job queues do not always satisfy out of the box.

## 2. The Async Agent Architecture: Task Submission, Queue, Worker, Result Retrieval

The canonical async agent architecture has five components.

![Async agent architecture: submit to queue to result](/imgs/blogs/async-long-running-agents-2.webp)

**API gateway (submit endpoint).** Accepts the incoming request from the caller, validates the payload, assigns a `task_id` (typically a UUIDv4 or a KSUID for sortability), writes the task to the queue, and returns `HTTP 202 Accepted` with `{"task_id": "...", "status_url": "/tasks/{id}"}` in under 50 ms. The gateway does not know and does not care how long the task will take. Its job is to receive and acknowledge.

**Task queue (durable broker).** Holds tasks until a worker picks them up. The key property is durability: if the broker process restarts, tasks already enqueued must not be lost. Redis (with RDB/AOF persistence), Amazon SQS, and Temporal all satisfy this. An in-memory queue backed by a Python list does not.

**Worker pool (agent executors).** N processes (or threads, or containers) that pull tasks from the queue, execute the agent, and write the result to the result store. Workers are stateless across tasks; all state lives in the queue entry, the result store, and any checkpoints the worker writes as it proceeds.

**Result store.** Holds task status and output. A Redis hash or DynamoDB item works well for status (`{status: "running", pct: 43, started_at: ...}`). Large outputs (long reports, generated files) go to object storage (S3, GCS) with the result record holding a pointer. Set a TTL — 24 to 72 hours is standard — because callers who never retrieve a result should not accumulate indefinitely.

**Status API (retrieval endpoint).** `GET /tasks/{id}` returns the current status plus the result if complete. This is the polling target. If you support webhooks or SSE, this same endpoint also serves as the source of truth for reconnecting clients.

Here is the minimal Python implementation using FastAPI and Redis Queue:

```python
# requirements: fastapi uvicorn redis rq python-ulid
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from redis import Redis
from rq import Queue

app = FastAPI()
redis_conn = Redis(host="localhost", port=6379, decode_responses=True)
task_queue = Queue("agent_tasks", connection=redis_conn)

@app.post("/tasks", status_code=202)
async def submit_task(payload: dict):
    task_id = str(uuid.uuid4())
    # Write initial status before enqueue to avoid race condition
    redis_conn.hset(f"task:{task_id}", mapping={
        "status": "pending",
        "created_at": "2026-06-27T00:00:00Z",
    })
    job = task_queue.enqueue(
        "workers.run_agent",
        task_id=task_id,
        payload=payload,
        job_id=task_id,          # idempotency: same ID = same job slot
        job_timeout=600,         # 10-minute hard limit
        result_ttl=86400,        # keep result 24h
        failure_ttl=86400,
    )
    return {
        "task_id": task_id,
        "status": "pending",
        "status_url": f"/tasks/{task_id}",
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    data = redis_conn.hgetall(f"task:{task_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return data
```

```python
# workers/run_agent.py — executed by an rq worker process
from redis import Redis
import time

redis_conn = Redis(host="localhost", port=6379, decode_responses=True)

def run_agent(task_id: str, payload: dict):
    """Entry point called by the RQ worker."""
    try:
        redis_conn.hset(f"task:{task_id}", mapping={"status": "running", "pct": 0})

        # Step 1: Plan
        _update_progress(task_id, 10, "planning")
        plan = _call_llm_plan(payload)

        # Step 2-5: Execute each plan step
        for i, step in enumerate(plan.steps):
            pct = 10 + int(80 * (i + 1) / len(plan.steps))
            _update_progress(task_id, pct, f"step {i+1}/{len(plan.steps)}")
            step.execute()

        # Step 6: Synthesize
        _update_progress(task_id, 90, "synthesizing")
        result = _synthesize(plan)

        redis_conn.hset(f"task:{task_id}", mapping={
            "status": "completed",
            "pct": 100,
            "result": result.to_json(),
        })
    except Exception as exc:
        redis_conn.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "error": str(exc),
        })
        raise  # let RQ handle retry policy

def _update_progress(task_id: str, pct: int, step: str):
    redis_conn.hset(f"task:{task_id}", mapping={"pct": pct, "current_step": step})
    # Optionally publish to pub/sub for SSE consumers
    redis_conn.publish(f"task:{task_id}:progress", f"{pct}:{step}")
```

This pattern — submit returns in under 50 ms, worker updates status atomically, caller polls the status endpoint — handles the basic case. Everything else in this post is about making it production-grade.

## 3. Task Queues: Redis Queue, Celery, SQS, Temporal — Choosing the Right Backend

Not all queue backends are equal for agent workloads. The four I've seen used most in production have very different tradeoffs.

![Task queue backend comparison: RQ vs Celery vs SQS vs Temporal](/imgs/blogs/async-long-running-agents-3.webp)

### Redis Queue (RQ)

RQ is the simplest option. It uses Redis as its broker and stores job state in Redis hashes. A worker process runs `rq worker queue_name` and polls Redis with `BLPOP` (blocking left pop). Latency from enqueue to pickup is sub-millisecond on a local Redis. Failure handling is basic: you configure `job_timeout` and `max_retries`; jobs that exceed the timeout get moved to a `failed` queue.

The durability problem: RQ depends on Redis. If you use Redis with persistence disabled (no AOF or RDB), tasks in-flight when the Redis process crashes are gone. If you use Redis with AOF persistence, you get durability at a latency cost. Most small teams run RQ against a Redis with `appendfsync everysec`, which is a one-second durability window — acceptable for agent tasks but not for financial transactions.

RQ is the right choice when: your team is small, your task volume is low (< 1,000/hr), you want to minimize dependencies, and you can tolerate occasional task loss on Redis restart.

### Celery with Redis or RabbitMQ

Celery is the most widely deployed Python task queue. It separates the worker library (Celery) from the broker (Redis, RabbitMQ, or SQS) and result backend (Redis, database, or S3). This separation lets you swap components: use RabbitMQ for the broker (durable message queues with exchange routing) and Redis for the result backend (fast status reads).

Celery gives you built-in support for:
- Task routing (send compute-heavy tasks to GPU workers, quick tasks to CPU workers)
- Canvas workflows (chains, groups, chords — useful for multi-step agent pipelines)
- Rate limiting per task type
- ETA and countdown scheduling

The operational cost is medium: you need to maintain the broker and keep Celery worker processes healthy. Celery workers can be configured with concurrency (threads or processes per worker), and you typically run multiple worker processes behind a supervisor like systemd or Kubernetes Deployments.

A Celery task for an agent looks like:

```python
from celery import Celery
from celery.utils.log import get_task_logger
import redis

app = Celery("agent_tasks", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")
app.conf.task_acks_late = True          # ACK only after completion (at-least-once)
app.conf.task_reject_on_worker_lost = True  # re-queue if worker dies mid-task
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.task_time_limit = 600          # hard kill after 10 min
app.conf.task_soft_time_limit = 540     # SIGTERM after 9 min (graceful)

logger = get_task_logger(__name__)
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.task(bind=True, max_retries=3, default_retry_delay=5,
          name="agent_tasks.run_research_agent")
def run_research_agent(self, task_id: str, query: str, context: dict):
    """Long-running research agent. Retries up to 3 times on transient errors."""
    r.hset(f"task:{task_id}", mapping={"status": "running", "pct": 0,
                                        "celery_task_id": self.request.id})
    try:
        # ... agent execution ...
        result = _execute_research(task_id, query, context)
        r.hset(f"task:{task_id}", mapping={"status": "completed", "pct": 100,
                                            "result": result})
        return {"task_id": task_id, "status": "completed"}
    except TransientError as exc:
        logger.warning(f"Task {task_id} transient error: {exc}. Retrying.")
        raise self.retry(exc=exc)
    except PermanentError as exc:
        r.hset(f"task:{task_id}", mapping={"status": "failed", "error": str(exc)})
        return {"task_id": task_id, "status": "failed"}
```

### Amazon SQS

SQS is the zero-ops choice for teams running on AWS. SQS is a managed service — no servers to provision or maintain, 99.99% availability SLA, unlimited queue depth, and pay-per-request pricing. For most teams, the operational savings outweigh the latency cost.

The mechanism is different from Redis. SQS uses *visibility timeouts* instead of explicit ACKs. When a worker picks up a message, SQS makes it invisible to other workers for the configured visibility timeout (default 30 seconds, configurable up to 12 hours). If the worker completes successfully, it deletes the message. If the worker dies, the visibility timeout expires and SQS re-delivers the message to another worker. This gives you at-least-once delivery by default.

The key SQS-specific considerations for agent tasks:

1. **Set visibility timeout > your longest task.** If your research agent takes up to 20 minutes and you set a 30-second visibility timeout, SQS will re-deliver the message to another worker while the first worker is still running. Set visibility timeout to `max_task_duration × 1.5`.

2. **Use SQS FIFO queues for strict ordering.** Standard SQS guarantees at-least-once but not in-order. If your agents have ordering requirements (e.g., tasks that depend on previous results), use FIFO with deduplication IDs.

3. **Dead Letter Queue (DLQ) is built-in.** After N failed delivery attempts, SQS moves the message to the configured DLQ. Wire up a CloudWatch alarm on the DLQ depth.

```python
import boto3
import json
import uuid

sqs = boto3.client("sqs", region_name="us-east-1")
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123/agent-tasks"

def submit_agent_task(query: str, context: dict) -> str:
    task_id = str(uuid.uuid4())
    message = {
        "task_id": task_id,
        "query": query,
        "context": context,
        "submitted_at": "2026-06-27T00:00:00Z",
    }
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId="default",           # required for FIFO
        MessageDeduplicationId=task_id,     # idempotency
    )
    return task_id

def worker_loop():
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,             # long polling — cheaper than short polling
            VisibilityTimeout=1200,         # 20-minute visibility window
        )
        messages = response.get("Messages", [])
        for msg in messages:
            task = json.loads(msg["Body"])
            receipt_handle = msg["ReceiptHandle"]
            try:
                run_agent(task["task_id"], task["query"], task["context"])
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
            except Exception as e:
                # Let visibility timeout expire → SQS re-delivers
                # Optionally update status to "retrying"
                pass
```

### Temporal

Temporal is the most powerful — and most complex — option. It is a workflow orchestration engine that provides durable execution: if a worker dies mid-task, Temporal replays the workflow from the beginning using the event history, transparently resuming execution at exactly the step where the worker left off. This is not retry — it is true resumption.

For agent tasks, Temporal's primitives are compelling:

- **Activities** are individual executable steps (one LLM call, one tool call). Each activity has its own retry policy with exponential backoff.
- **Workflows** are durable orchestrators that call activities. If the workflow worker restarts, Temporal replays the event history to get the workflow back to its current state.
- **Signals** let external callers send messages to a running workflow (e.g., "cancel" or "add context").
- **Queries** let external callers read workflow state synchronously without affecting execution.
- **Timers** let you schedule delayed actions within workflows without external cron infrastructure.

The operational cost is real: you need to run the Temporal server (or pay for Temporal Cloud). The self-hosted server is a multi-component system (frontend, history, matching, worker services) backed by Cassandra or PostgreSQL. Temporal Cloud removes this burden at $0.02/workflow action.

```python
# temporal_worker/agent_workflow.py
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from datetime import timedelta

@activity.defn
async def call_llm(prompt: str, step: str) -> str:
    """Single LLM call. Retried automatically on transient failures."""
    return await llm_client.generate(prompt)

@activity.defn
async def run_tool(tool_name: str, args: dict) -> dict:
    """Single tool execution. Retried automatically."""
    return await tool_registry.execute(tool_name, args)

@workflow.defn
class ResearchAgentWorkflow:
    def __init__(self):
        self._progress = 0
        self._step = "pending"

    @workflow.query
    def get_progress(self) -> dict:
        return {"pct": self._progress, "step": self._step}

    @workflow.run
    async def run(self, query: str) -> str:
        self._step = "planning"
        plan = await workflow.execute_activity(
            call_llm,
            f"Create a research plan for: {query}",
            "planning",
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3,
                                     initial_interval=timedelta(seconds=2)),
        )
        self._progress = 20

        # Execute each research step
        results = []
        steps = parse_plan(plan)
        for i, step in enumerate(steps):
            self._step = f"research step {i+1}"
            result = await workflow.execute_activity(
                run_tool, step.tool, step.args,
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )
            results.append(result)
            self._progress = 20 + int(60 * (i + 1) / len(steps))

        self._step = "synthesizing"
        synthesis = await workflow.execute_activity(
            call_llm,
            f"Synthesize these research results: {results}",
            "synthesis",
            start_to_close_timeout=timedelta(seconds=60),
        )
        self._progress = 100
        return synthesis
```

The decision rule: use Temporal when your workflows have complex branching logic, your tasks run for more than 30 minutes, you cannot afford to re-run from scratch on failure, or you need precise per-step retry policies with different backoff curves per activity type. For shorter tasks with simpler flows, the operational overhead is not justified — RQ or Celery will serve you better.

## 4. Task IDs and Status APIs: How Callers Track Long-Running Tasks

The task ID is the contract between the caller and the system. Every decision you make about the status API flows from this contract.

**Task ID format.** UUIDv4 is safe and unguessable, which matters if status URLs are publicly routable. KSUID (K-Sortable Unique Identifier) adds time-sortability — `ksuid01J2...` encodes a timestamp prefix, so you can list recent tasks without a secondary sort key. ULID is similar. Pick one and be consistent across your API.

**Status schema.** A minimal status record:

```json
{
  "task_id": "01J2K3MNPQRSTVWXYZ0123456",
  "status": "running",
  "pct": 43,
  "current_step": "web_search_3",
  "created_at": "2026-06-27T10:30:00Z",
  "started_at": "2026-06-27T10:30:02Z",
  "updated_at": "2026-06-27T10:31:15Z",
  "estimated_completion": "2026-06-27T10:33:00Z",
  "result": null,
  "error": null
}
```

On completion:

```json
{
  "task_id": "01J2K3MNPQRSTVWXYZ0123456",
  "status": "completed",
  "pct": 100,
  "created_at": "2026-06-27T10:30:00Z",
  "started_at": "2026-06-27T10:30:02Z",
  "completed_at": "2026-06-27T10:32:47Z",
  "result": {
    "type": "research_report",
    "content_url": "https://storage.example.com/results/01J2K3M.json",
    "summary": "The report covers 14 sources across 3 categories..."
  },
  "error": null
}
```

**The status enum.** These are the states a task transitions through:

![Task status state machine: pending to terminal states](/imgs/blogs/async-long-running-agents-4.webp)

`SUBMITTED` → `PENDING` → `RUNNING` → one of `COMPLETED / FAILED / RETRYING / CANCELLED / TIMED_OUT`. The key rule: the four terminal states (`COMPLETED`, `FAILED`, `CANCELLED`, `TIMED_OUT`) are irreversible. Once a task enters a terminal state, the status must never change again. Callers poll until they see a terminal state, then stop.

**Result storage for large outputs.** Agent outputs can be large — a multi-page research report, a code diff, a dataset. Store large results in object storage and put a signed URL in the status record:

```python
import boto3
import json

s3 = boto3.client("s3")
RESULT_BUCKET = "agent-results"

def store_large_result(task_id: str, result: dict) -> str:
    """Store result in S3, return signed URL valid for 24h."""
    key = f"results/{task_id}.json"
    s3.put_object(
        Bucket=RESULT_BUCKET,
        Key=key,
        Body=json.dumps(result),
        ContentType="application/json",
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": RESULT_BUCKET, "Key": key},
        ExpiresIn=86400,
    )
    return url
```

**Idempotent status reads.** `GET /tasks/{id}` must be safe to call repeatedly with no side effects. Never make a status read trigger a side effect like re-running the task or sending a notification. Those happen via explicit `POST /tasks/{id}/retry` or `POST /tasks/{id}/cancel` endpoints.

## 5. Polling vs Webhooks vs SSE: Notification Patterns for Async Results

Once a task is submitted, the caller needs to know when it's done. There are three patterns, each with distinct tradeoffs.

![Notification patterns: polling vs webhook vs SSE](/imgs/blogs/async-long-running-agents-5.webp)

### Polling (Client Pull)

The simplest pattern. The client periodically calls `GET /tasks/{id}` until it sees a terminal status. Implement exponential backoff to avoid thundering-herd problems when many tasks complete around the same time:

```typescript
// TypeScript client with exponential backoff polling
async function pollTaskResult(taskId: string): Promise<TaskResult> {
  const BASE_DELAY_MS = 1000;     // start at 1s
  const MAX_DELAY_MS = 30_000;    // cap at 30s
  const MAX_DURATION_MS = 30 * 60 * 1000; // give up after 30 min

  const startTime = Date.now();
  let attempt = 0;

  while (Date.now() - startTime < MAX_DURATION_MS) {
    const resp = await fetch(`/api/tasks/${taskId}`);
    if (!resp.ok) throw new Error(`Status fetch failed: ${resp.status}`);
    const task = await resp.json();

    if (task.status === "completed") return task.result;
    if (task.status === "failed") throw new Error(`Task failed: ${task.error}`);
    if (task.status === "cancelled") throw new Error("Task was cancelled");
    if (task.status === "timed_out") throw new Error("Task timed out");

    // Exponential backoff with jitter
    const delay = Math.min(
      BASE_DELAY_MS * Math.pow(1.5, attempt) + Math.random() * 500,
      MAX_DELAY_MS
    );
    await new Promise(resolve => setTimeout(resolve, delay));
    attempt++;
  }
  throw new Error("Polling timeout exceeded");
}
```

Polling is robust under network interruptions — if the client disconnects and reconnects, it just resumes polling. The server has no per-client state. The cost is latency: with a 5-second polling interval, a task that completes at second 4.9 appears to the user at second 9.9. For background tasks where a few seconds of lag is acceptable, polling is often the right call.

### Webhooks (Server Push)

Webhooks flip the direction: the server POSTs to the client when the task completes. The caller provides a callback URL at task submission time:

```python
# Webhook payload sent by the server on task completion
{
  "event": "task.completed",
  "task_id": "01J2K3MNPQRSTVWXYZ0123456",
  "timestamp": "2026-06-27T10:32:47Z",
  "result": { "summary": "...", "content_url": "..." },
  "signature": "sha256=abc123..."  # HMAC-SHA256 of payload
}
```

The server implementation:

```python
import hmac
import hashlib
import httpx
import asyncio

WEBHOOK_SECRET = "your-signing-secret"

async def deliver_webhook(task_id: str, callback_url: str, payload: dict,
                          max_retries: int = 5):
    """Fire-and-forget webhook with exponential retry."""
    body = json.dumps(payload)
    signature = hmac.new(
        WEBHOOK_SECRET.encode(), body.encode(), hashlib.sha256
    ).hexdigest()

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    callback_url,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Agent-Signature": f"sha256={signature}",
                        "X-Task-Id": task_id,
                    },
                )
                if resp.status_code < 300:
                    return  # success
                # 4xx = don't retry; 5xx = retry
                if resp.status_code < 500:
                    break
            except httpx.RequestError:
                pass  # network error, retry
            await asyncio.sleep(2 ** attempt)  # 1, 2, 4, 8, 16 seconds
    # If all retries fail, log and move on — the caller can still poll
```

Webhooks are ideal for server-to-server integrations (CI/CD pipelines, enterprise integrations, Slack apps). They are the wrong tool for browser clients because browsers cannot listen on a port. The critical requirement is that the callback URL must be publicly routable and the caller must be running a server. For users behind firewalls or mobile apps, this is not always possible.

Always include an HMAC signature. The receiving server should verify it before processing the payload. Without this, any party who knows the webhook URL can forge completion events.

### Server-Sent Events (SSE)

SSE is the best choice for browser-native real-time progress reporting. The client opens an `EventSource` connection to a streaming endpoint; the server pushes events as they happen. Unlike WebSockets, SSE is unidirectional (server to client), uses plain HTTP/1.1, and the browser handles reconnection automatically.

```python
# FastAPI SSE endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json
import redis.asyncio as aioredis

app = FastAPI()
redis_pool = aioredis.ConnectionPool.from_url("redis://localhost:6379")

@app.get("/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """Stream task progress events via SSE."""
    async def event_generator():
        r = aioredis.Redis(connection_pool=redis_pool)
        pubsub = r.pubsub()

        # Send current state first (catch-up for reconnecting clients)
        current = await r.hgetall(f"task:{task_id}")
        yield f"data: {json.dumps(current)}\n\n"

        if current.get("status") in ("completed", "failed", "cancelled", "timed_out"):
            return  # already terminal, no need to subscribe

        await pubsub.subscribe(f"task:{task_id}:progress")
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                pct, step = message["data"].decode().split(":", 1)
                event_data = {"pct": int(pct), "step": step, "task_id": task_id}

                # Check if terminal
                current = await r.hgetall(f"task:{task_id}")
                event_data["status"] = current.get("status", "running")
                yield f"data: {json.dumps(event_data)}\n\n"

                if current.get("status") in ("completed", "failed", "cancelled", "timed_out"):
                    break
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})  # disable Nginx buffering
```

```javascript
// Browser EventSource client
const source = new EventSource(`/tasks/${taskId}/stream`);

source.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateProgressBar(data.pct);
  updateStepLabel(data.step);

  if (["completed", "failed", "cancelled", "timed_out"].includes(data.status)) {
    source.close();
    handleTerminalState(data);
  }
};

source.onerror = () => {
  // Browser auto-reconnects after 3 seconds — no action needed
  console.log("SSE connection lost, reconnecting...");
};
```

One critical detail: set `X-Accel-Buffering: no` in the response headers if your app runs behind Nginx. Nginx buffers upstream responses by default, which means SSE events pile up in the Nginx buffer instead of flowing to the client. This setting disables buffering for this response only.

## 6. Progress Reporting: Streaming Intermediate Results, Percentage Complete, ETA

Real-time progress reporting is what separates an agent that feels alive from one that shows a spinner for three minutes. The architecture has three components that communicate via Redis:

![Progress reporting flow: agent events to client UI](/imgs/blogs/async-long-running-agents-6.webp)

**Emitting progress from the worker.** The worker writes progress events after each significant step. The key discipline: do not emit on every token or every microsecond. Emit at semantic checkpoints — after each LLM call, after each tool call, after each retrieval. Aim for 5–15 progress events per task.

```python
class ProgressTracker:
    """Thread-safe progress tracker for agent workers."""

    def __init__(self, task_id: str, redis_client, total_steps: int):
        self.task_id = task_id
        self.r = redis_client
        self.total = total_steps
        self.current = 0
        self._start_time = time.time()

    def step(self, description: str):
        self.current += 1
        pct = int(100 * self.current / self.total)
        elapsed = time.time() - self._start_time
        eta_seconds = None
        if self.current > 0:
            per_step = elapsed / self.current
            remaining = self.total - self.current
            eta_seconds = int(per_step * remaining)

        # Atomic update: persist to hash AND publish to pub/sub
        pipe = self.r.pipeline()
        pipe.hset(f"task:{self.task_id}", mapping={
            "pct": pct,
            "current_step": description,
            "eta_seconds": eta_seconds or "",
        })
        pipe.publish(f"task:{self.task_id}:progress",
                     f"{pct}:{description}:{eta_seconds or ''}")
        pipe.execute()
```

**ETA computation.** A naive linear extrapolation from steps completed / steps total works for the first 60% of the task. For the last 40%, use a sliding window average over the last three steps instead of the total elapsed average, because later steps (synthesis, formatting) are often slower than early steps (planning):

```python
def estimate_eta(self) -> int | None:
    """Sliding-window ETA in seconds."""
    if len(self._step_durations) < 2:
        return None
    # Use last min(3, completed) step durations
    window = self._step_durations[-min(3, len(self._step_durations)):]
    avg_step = sum(window) / len(window)
    remaining = self.total - self.current
    return int(avg_step * remaining)
```

**Streaming partial results.** For long research tasks, users often want to see intermediate output before the full task completes. The cleanest pattern is to write partial results to a dedicated key and let the SSE stream signal "partial result available":

```python
def emit_partial_result(task_id: str, section: str, content: str, r):
    """Write a completed section to Redis for streaming to the client."""
    r.rpush(f"task:{task_id}:partials", json.dumps({
        "section": section,
        "content": content,
        "timestamp": time.time(),
    }))
    r.publish(f"task:{task_id}:progress",
              f"partial:{section}:{len(content)} chars")
```

The client EventSource handler can then fetch the partial list and display each completed section as it arrives, instead of waiting for the entire report.

## 7. Timeout Handling: Per-Step Timeouts, Task-Level Deadlines, Graceful Cancellation

Timeouts must be nested correctly. The inner timeout must always be shorter than the outer one, or the outer layer cuts the wire before the inner layer can trigger its recovery behavior.

![Nested timeout layers: step to task to client connection](/imgs/blogs/async-long-running-agents-7.webp)

The three layers, from inner to outer:

**Step timeout (30–120 seconds per LLM call or tool call).** This is the most important layer and the most commonly omitted. Every individual LLM call or tool invocation should have an explicit timeout. An LLM API that stops responding mid-stream will block your worker thread indefinitely without this:

```python
import asyncio

async def call_llm_with_timeout(prompt: str, timeout_seconds: int = 60) -> str:
    """Call LLM with a per-call timeout. Raises asyncio.TimeoutError on expiry."""
    try:
        return await asyncio.wait_for(
            llm_client.generate_async(prompt),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise StepTimeoutError(
            f"LLM call exceeded {timeout_seconds}s timeout"
        )

async def run_tool_with_timeout(tool: str, args: dict, timeout: int = 30) -> dict:
    try:
        return await asyncio.wait_for(
            tool_registry.execute_async(tool, args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise StepTimeoutError(f"Tool {tool!r} exceeded {timeout}s")
```

**Task-level deadline (5–30 minutes for the entire task).** The worker checks a deadline before each step. If the deadline has passed, the task exits cleanly and marks itself `TIMED_OUT` rather than being killed mid-step:

```python
import time
from dataclasses import dataclass

@dataclass
class TaskContext:
    task_id: str
    deadline: float  # Unix timestamp

    def check_deadline(self, step: str):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TaskDeadlineExceeded(
                f"Task {self.task_id} exceeded deadline at step {step!r}"
            )
        if remaining < 30:
            # Warn but don't abort yet — last step might still succeed
            logger.warning(
                f"Task {self.task_id} has {remaining:.0f}s remaining, "
                f"currently at {step!r}"
            )

def run_agent(task_id: str, payload: dict, deadline_seconds: int = 300):
    ctx = TaskContext(
        task_id=task_id,
        deadline=time.time() + deadline_seconds,
    )
    try:
        ctx.check_deadline("planning")
        plan = call_llm_with_timeout(payload["query"])

        for step in plan.steps:
            ctx.check_deadline(f"step:{step.name}")
            result = run_tool_with_timeout(step.tool, step.args)

        ctx.check_deadline("synthesis")
        final = synthesize(results)
        mark_completed(task_id, final)

    except TaskDeadlineExceeded as exc:
        mark_timed_out(task_id, str(exc))
```

**Client connection timeout (longer than task deadline).** This applies to any synchronous fallback path (e.g., waiting for a webhook). It should be set longer than the task deadline so that the task has time to timeout gracefully and report the result before the client gives up.

**Graceful cancellation.** Users cancelling tasks mid-execution is a common requirement. The cleanest implementation uses a Redis flag that the worker checks between steps:

```python
def is_cancelled(task_id: str, r) -> bool:
    return r.hget(f"task:{task_id}", "status") == "cancelled"

def run_agent_with_cancellation(task_id: str, payload: dict):
    for step in plan.steps:
        if is_cancelled(task_id, r):
            logger.info(f"Task {task_id} cancelled at step {step.name}")
            return  # exit cleanly — status already set by cancel endpoint

        result = execute_step(step)
```

The cancel endpoint:

```python
@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    current = redis_conn.hget(f"task:{task_id}", "status")
    if current in ("completed", "failed", "timed_out", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Task is already {current}")
    redis_conn.hset(f"task:{task_id}", "status", "cancelled")
    return {"task_id": task_id, "status": "cancelled"}
```

## 8. Retry and Resumption: Re-Queuing Failed Tasks Without Duplicating Work

The naive retry strategy — "if it fails, run it from the start" — doubles cost on every failure, and for a 20-minute task that fails at step 19 of 20, it is devastating. The right strategy is checkpointed resumption.

![Idempotent retry: checkpoint-based task resumption](/imgs/blogs/async-long-running-agents-8.webp)

**Write a checkpoint after each step.** A checkpoint is a small record of which steps have been completed and what they returned. Store it in Redis or a database. On resumption, the worker reads the checkpoint and skips completed steps:

```python
import json

class CheckpointedAgent:
    """Agent that writes a checkpoint after each step and resumes from it."""

    def __init__(self, task_id: str, r):
        self.task_id = task_id
        self.r = r
        self.checkpoint_key = f"task:{task_id}:checkpoint"

    def load_checkpoint(self) -> dict:
        raw = self.r.get(self.checkpoint_key)
        return json.loads(raw) if raw else {"completed_steps": {}}

    def save_checkpoint(self, step_name: str, result: any):
        checkpoint = self.load_checkpoint()
        checkpoint["completed_steps"][step_name] = result
        self.r.set(self.checkpoint_key, json.dumps(checkpoint), ex=86400)

    def delete_checkpoint(self):
        self.r.delete(self.checkpoint_key)

    def run(self, steps: list):
        checkpoint = self.load_checkpoint()
        completed = checkpoint["completed_steps"]

        results = {}
        for step in steps:
            if step.name in completed:
                # Resume from checkpoint — skip this step
                results[step.name] = completed[step.name]
                continue

            # Execute and checkpoint
            result = step.execute()
            results[step.name] = result
            self.save_checkpoint(step.name, result)

        # All steps done — clean up checkpoint
        self.delete_checkpoint()
        return results
```

**Idempotency keys prevent duplicate execution.** If the worker crashes after completing a step but before writing the checkpoint, the step will be re-run on the next attempt. Make each step idempotent: the same input always produces the same output, and a second execution of a completed step is safe. For LLM calls, this usually means using a deterministic seed (if the API supports it) or checking if a downstream side effect (like a file write) already happened before re-executing.

**Retry policies by error type.** Not all errors deserve the same retry strategy:

```python
from enum import Enum

class RetryPolicy(Enum):
    NO_RETRY = "no_retry"            # permanent errors
    IMMEDIATE = "immediate"          # temporary blips, retry once
    EXPONENTIAL = "exponential"      # transient errors, exponential backoff
    LONG_WAIT = "long_wait"          # rate limits, wait 60+ seconds

RETRY_POLICIES = {
    # LLM API errors
    "rate_limit": RetryPolicy.LONG_WAIT,        # wait for quota reset
    "context_length": RetryPolicy.NO_RETRY,     # input too large, fix it
    "timeout": RetryPolicy.EXPONENTIAL,         # might be temporary
    "server_error_5xx": RetryPolicy.EXPONENTIAL,

    # Tool errors
    "tool_not_found": RetryPolicy.NO_RETRY,     # misconfiguration
    "tool_network_error": RetryPolicy.EXPONENTIAL,
    "tool_auth_error": RetryPolicy.NO_RETRY,    # bad credentials

    # Agent logic errors
    "plan_parse_error": RetryPolicy.IMMEDIATE,  # might succeed with different seed
    "synthesis_error": RetryPolicy.EXPONENTIAL,
}
```

**Dead Letter Queues.** After `max_retries` failures, move the task to a DLQ instead of silently dropping it. The DLQ is a separate queue for failed tasks that you can inspect, re-process manually, or alert on:

```python
import redis

def move_to_dlq(r: redis.Redis, task_id: str, error: str):
    """Move a permanently failed task to the dead letter queue."""
    task_data = r.hgetall(f"task:{task_id}")
    task_data.update({"status": "failed", "error": error,
                      "failed_at": "2026-06-27T10:32:47Z"})

    pipe = r.pipeline()
    pipe.hset(f"task:{task_id}", mapping=task_data)
    pipe.lpush("dlq:agent_tasks", task_id)
    pipe.execute()
```

## 9. Priority Queues: Handling Urgent vs Background Agent Tasks

Not all tasks are equal. A user-triggered research query while the user is actively waiting has a very different SLA than a nightly batch summary job. Mixing these into a single queue means batch jobs starve interactive tasks.

![Priority queue worker pool: urgent vs background tasks](/imgs/blogs/async-long-running-agents-9.webp)

The implementation uses multiple queues with dedicated or shared workers:

```python
# Celery configuration with priority routing
from celery import Celery

app = Celery("agent")
app.conf.task_queues = {
    "high_priority": {"exchange": "high_priority", "routing_key": "high"},
    "default": {"exchange": "default", "routing_key": "default"},
    "low_priority": {"exchange": "low_priority", "routing_key": "low"},
}
app.conf.task_routes = {
    "agent.interactive_research": {"queue": "high_priority"},
    "agent.report_generation": {"queue": "default"},
    "agent.nightly_batch": {"queue": "low_priority"},
}

# Start workers with priority weight
# High priority: 2 dedicated workers + shared pool overflow
# Default: shared pool (2-20 workers)
# Low priority: shared pool, lowest scheduling weight
```

For Redis-backed systems, implement weighted random selection across queues. The worker checks `high_priority` first, then `default`, then `low_priority`:

```python
import redis
import time

r = redis.Redis()

def dequeue_with_priority(queues: list[str], weights: list[float],
                           timeout: int = 1) -> tuple[str, dict] | None:
    """
    Dequeue from priority-weighted queues.
    queues: ["high_priority", "default", "low_priority"]
    weights: [10, 3, 1]  — high is 10x more likely to be checked
    """
    # Weighted random selection — but for strict priority, always check
    # high first. Weighted random only for medium/low to allow starvation
    # prevention.
    import random

    # Always drain high-priority queue first
    msg = r.brpop(queues[0], timeout=0.01)
    if msg:
        return queues[0], json.loads(msg[1])

    # Then weighted random among remaining queues
    total = sum(weights[1:])
    pick = random.uniform(0, total)
    cumulative = 0
    for q, w in zip(queues[1:], weights[1:]):
        cumulative += w
        if pick <= cumulative:
            msg = r.brpop(q, timeout=timeout)
            if msg:
                return q, json.loads(msg[1])
    return None
```

**SLA enforcement.** Assign each queue an SLA and alert when tasks exceed it:

| Queue | SLA | Dedicated workers | Alert threshold |
|---|---|---|---|
| `high_priority` | P99 < 2s queue time | 2 reserved | queue depth > 5 |
| `default` | P99 < 30s queue time | 0 reserved (shared) | queue depth > 50 |
| `low_priority` | P99 < 10min queue time | 0 reserved (shared) | queue depth > 500 |

Monitor the P99 queue time (time from enqueue to pickup) for each queue and page if it exceeds the SLA for more than two minutes.

## 10. Concurrency Management: How Many Agent Workers to Run, Autoscaling

The right number of workers depends on task duration, memory per worker, and GPU availability if your workers run local models.

**Worker sizing.** Each agent worker needs:
- CPU: ~0.5–1 core for orchestration (the heavy lifting is at the API endpoint, not the worker)
- Memory: ~512 MB–2 GB depending on context held in memory during execution
- Network I/O: one open connection per LLM call + one per tool call

A t3.medium (2 vCPU, 4 GB RAM) can comfortably run 3–4 worker processes. A c5.xlarge (4 vCPU, 8 GB RAM) can handle 8–10.

**Autoscaling.** The correct scaling metric is queue depth (tasks waiting), not CPU utilization. CPU utilization for LLM-orchestration workers is low — the workers spend most of their time waiting for API responses. Queue depth spikes immediately when demand exceeds capacity:

```python
# Prometheus metrics for autoscaling decisions
from prometheus_client import Gauge
import redis

r = redis.Redis()

queue_depth = Gauge("agent_queue_depth", "Number of tasks waiting",
                    ["queue_name"])
active_workers = Gauge("agent_active_workers", "Number of active workers",
                       ["queue_name"])

def report_metrics():
    for queue in ["high_priority", "default", "low_priority"]:
        depth = r.llen(f"rq:queue:{queue}")
        queue_depth.labels(queue_name=queue).set(depth)
```

For Kubernetes, use KEDA (Kubernetes Event-Driven Autoscaler) to scale worker deployments based on Redis queue depth:

```yaml
# keda-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: agent-worker-scaler
spec:
  scaleTargetRef:
    name: agent-worker-deployment
  minReplicaCount: 2      # always keep 2 warm for low latency
  maxReplicaCount: 20
  cooldownPeriod: 120     # wait 2 min before scaling down
  triggers:
    - type: redis
      metadata:
        address: redis:6379
        listName: "rq:queue:default"
        listLength: "5"   # 1 worker per 5 tasks in queue
```

**Worker process vs thread vs async.** For pure API-orchestration agents (all LLM calls go to remote APIs), use async workers — they can handle multiple in-flight requests efficiently with one thread. For agents that run local models or do CPU-heavy processing, use process-based workers to avoid the GIL. Celery's `--pool=prefork` uses process pools; `--pool=gevent` uses greenlets for I/O-bound work.

**The 10x rule.** Design your worker capacity to handle 10x your expected peak load. Agent workloads are bursty — a marketing campaign that goes viral, a product launch that triggers simultaneous research requests from 200 users — and spinning up new workers takes 20–60 seconds. Pre-warm enough capacity to absorb a 10x spike for at least five minutes while autoscaling catches up.

## 11. UX Patterns for Async Agents: Keeping Users Informed Without Overwhelming

The best async agent UX is one where users forget the task is running asynchronously. Three patterns achieve this.

**Optimistic submission.** The moment the user submits, show a confirmation with a task ID and a progress indicator. Do not wait for the queue acknowledgment to update the UI — the enqueue latency is under 50ms, but even 50ms of blank screen reads as broken. Show the "submitted" state immediately, then reconcile with the server response.

```typescript
async function submitAndTrack(query: string) {
  // Optimistic update — show pending immediately
  const optimisticId = crypto.randomUUID();
  setTasks(prev => [...prev, {
    id: optimisticId,
    status: "submitting",
    query,
    pct: 0,
  }]);

  try {
    const resp = await fetch("/api/tasks", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query}),
    });
    const {task_id, status_url} = await resp.json();

    // Replace optimistic entry with real task ID
    setTasks(prev => prev.map(t =>
      t.id === optimisticId ? {...t, id: task_id, status: "pending"} : t
    ));

    // Start SSE stream
    startProgressStream(task_id);
  } catch (err) {
    setTasks(prev => prev.filter(t => t.id !== optimisticId));
    showError("Failed to submit task. Please try again.");
  }
}
```

**Meaningful progress steps.** Showing "43% complete" is better than a spinner. Showing "Searching academic databases for recent papers on X" is better than "43% complete." Every emitted progress event should carry a human-readable step description, not just a percentage. The step descriptions are your opportunity to make the agent feel intelligent and in-motion.

Bad progress messages:
- "Processing..."
- "Step 3 of 7"
- "Running"

Good progress messages:
- "Searching 3 academic databases for papers on quantum error correction"
- "Reading and summarizing 14 papers (8 remaining)"
- "Writing introduction section based on 6 key findings"

**Background mode.** For tasks that run longer than 5 minutes, give users a way to leave and come back. Store the task state in the browser's local storage and in the user's account. When they return, the status API restores the current state and the SSE stream reconnects:

```typescript
function TaskManager() {
  const [tasks, setTasks] = useState(() => {
    // Restore from localStorage on mount
    const saved = localStorage.getItem("pending_tasks");
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    // Persist to localStorage on change
    localStorage.setItem("pending_tasks", JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    // Re-subscribe to SSE for any in-progress tasks
    tasks.filter(t => !["completed", "failed", "cancelled"].includes(t.status))
         .forEach(t => subscribeToProgress(t.id));
  }, []);
}
```

**Email/push notification for long tasks.** If a task runs longer than 10 minutes, email or push notification is often the best UX — the user can do other things and get notified when the result is ready. Implement this as a webhook from your task system to your notification service:

```python
async def on_task_completed(task_id: str, user_id: str, result: dict):
    """Called when a long task completes. Send notification if user is offline."""
    user = await get_user(user_id)
    task_duration = result["completed_at"] - result["started_at"]

    if task_duration > 300:  # 5 minutes — user probably isn't watching
        await notification_service.send_email(
            to=user.email,
            subject=f"Your research task is complete",
            body=render_completion_email(task_id, result),
        )
```

## 12. Case Studies: Async Agent Patterns in Production

### Case Study 1: Research Agent at a Consulting Firm

A consulting firm built an internal research agent that takes a client question, searches 15 internal databases, reads and synthesizes 20–50 documents, and produces a 5-page structured report. Task duration: 8–25 minutes.

**The sync failure.** The initial implementation used a synchronous endpoint with a 10-minute HTTP timeout. Analysts experienced frequent 504 errors, especially for longer queries. The support burden was 40% of the team's week — every error required a manual re-run.

**The async solution.** They moved to Celery with RabbitMQ (already in their stack for other services). Each task got a unique task ID displayed in a dashboard. Analysts submitted tasks and monitored them in a simple table view that polled every 10 seconds. Completed reports were emailed automatically.

**Key decision: polling, not SSE.** The analysts were not sitting watching the progress bar. They were doing other work and checking back. The 10-second polling interval was invisible in practice, and polling required no server-side streaming infrastructure. They added SSE only six months later, for a specific use case where senior partners wanted live step updates on high-stakes queries.

**Result:** 504 errors dropped from 35% to 0.2% (the remaining 0.2% were genuine worker crashes, not timeout issues). The team reclaimed the support time.

### Case Study 2: Code Review Agent at a SaaS Company

A SaaS company built an agent that reviews pull requests: reads the diff, checks for security vulnerabilities, suggests refactors, and writes inline GitHub comments. Task duration: 2–8 minutes per PR.

**The priority queue problem.** Initially, all PRs entered a single queue. When a junior developer opened a 200-file refactor PR, it took 8 minutes and blocked the queue. Senior developers' small, urgent PRs waited behind it. The team missed a critical security fix review because the PR sat in queue for 45 minutes.

**The solution.** They implemented priority routing based on PR size. PRs with fewer than 20 changed files went to a `high_priority` queue with two dedicated workers. Larger PRs went to a `default` queue. They also added a `critical` tag (set by team leads) that bypassed queue depth entirely and used a third reserved worker.

**Checkpoint insight.** The agent crashed mid-review on large PRs if the LLM rate limit was hit. Without checkpoints, a crash on file 180 of 200 meant re-reviewing files 1–180. They added file-level checkpointing — the agent wrote a completed file list to Redis after each file. On restart, it skipped already-reviewed files. Average wasted work on retry dropped from 78% to 4%.

### Case Study 3: E-commerce Product Description Generator

An e-commerce platform built an agent that generates SEO-optimized product descriptions from raw product data. They process 50,000 new products per day. Task duration: 30–90 seconds per product.

**The throughput challenge.** At 50,000 tasks per day, they need 35 tasks per minute. Each task takes 30–90 seconds. They need at minimum 35 × 90 / 60 = 52 concurrent workers to keep up.

**SQS FIFO + spot instances.** They chose SQS FIFO (for deduplication — the same product ID should not be processed twice) and ran workers on EC2 Spot Instances. Spot instances are 70% cheaper than on-demand but can be reclaimed with two minutes' notice.

**The reclaim problem.** A spot instance reclaim while a worker is generating descriptions for product 1234 meant that product never got its description. They solved it with SQS's visibility timeout: set to 300 seconds (5× the longest task). When a spot instance is reclaimed, it gets a SIGTERM with a 120-second grace window — the worker catches SIGTERM, cleans up in-flight work, and NACKs any tasks it cannot finish. SQS re-delivers those tasks after the visibility timeout.

**Result:** 99.94% of products got descriptions on first delivery. The 0.06% that were mid-flight during reclaim were delivered within 5 minutes.

### Case Study 4: Financial Document Analysis Agent

A fintech company built an agent that analyzes 10-K filings and produces risk summaries. Task duration: 15–45 minutes per filing, driven by long-document chunking.

**The Temporal decision.** They originally used Celery but hit persistent problems with complex multi-step workflows. The agent had eight distinct phases: download, extract text, chunk, embed, retrieve context, draft section by section (6 sections), compile, review. If phase 7 (section 5 of 6) failed, Celery had no way to resume from exactly that step — it restarted the whole job.

After two months of fighting Celery workflow state management, they migrated to Temporal. The migration took three weeks. The payoff: zero restart-from-scratch failures in the following six months. Temporal's event history meant a failed activity retried exactly once from where it failed, not from the beginning.

**Signal-based human review.** They used Temporal Signals to implement a human-in-the-loop step. After the draft was generated, the workflow paused and sent a notification to a reviewer. If the reviewer approved via the API, the workflow received a signal and proceeded to the compile step. If they rejected with edits, the workflow received a signal with the edits and re-ran the draft step with the updated context. This would have been very complex to implement correctly in Celery.

### Case Study 5: Multi-Agent Orchestrator

An AI startup built a product where a coordinator agent breaks a user task into subtasks and dispatches them to specialized subagents (search, code, analysis). Each subagent runs asynchronously and reports results back to the coordinator.

**The fan-out problem.** When the coordinator dispatched 8 subtasks, it needed to wait for all of them before synthesizing. Naive implementation: the coordinator polled all 8 status endpoints in a loop. This was fragile — if any subtask timed out, the coordinator did not notice for up to 30 seconds.

**Async gather with partial results.** They used Python's `asyncio.gather()` with individual per-subtask timeouts. If a subtask timed out, the coordinator continued synthesis with the results from the other subtasks, marking the missing subtask as "unavailable" in the report:

```python
async def coordinate(task_id: str, subtasks: list[dict]) -> dict:
    """Run all subtasks concurrently, collect results with timeout."""
    async def run_subtask(subtask: dict):
        sub_id = await submit_task(subtask)
        return await poll_task_result(sub_id, timeout=300)

    results = await asyncio.gather(
        *[run_subtask(s) for s in subtasks],
        return_exceptions=True,  # don't raise on individual subtask failure
    )

    successful = {
        subtasks[i]["name"]: r
        for i, r in enumerate(results)
        if not isinstance(r, Exception)
    }
    failed = {
        subtasks[i]["name"]: str(r)
        for i, r in enumerate(results)
        if isinstance(r, Exception)
    }

    return await synthesize(successful, failed)
```

**Result:** Users got partial results in cases where one subtask failed, instead of a total failure. The coordinator's timeout was set tighter than individual subtask timeouts — 5 minutes for the coordinator, 3 minutes for each subtask — so the outer task always returned something within the user's patience threshold.

### Case Study 6: Overnight Batch Summarization Agent

A news aggregator runs an agent each night that reads 500 articles, clusters them by topic, and produces daily briefings. Task duration: 60–90 minutes total.

**The simplicity principle.** Unlike the other case studies, this team deliberately chose not to use async worker infrastructure. Their solution: a Python script run by cron, writing output to a file, with a simple "last completed run" status in a database table. No Redis, no queue, no SSE.

The reasoning: the task runs once per night, has no SLA requirement tighter than "done by morning," has a single user (the system itself, whose output is consumed by the next pipeline stage), and the operational complexity of maintaining a queue backend was not justified.

This case study exists to illustrate the principle from section 13: async infrastructure is a powerful tool, but it is not always the right tool. Overnight batch jobs with no interactive user waiting for results are exactly the kind of workload where async infrastructure adds complexity without adding value.

## 13. When Sync Is Fine / When to Go Async: The Latency Threshold Decision

The decision tree is straightforward:

![Sync vs async decision tree: choosing your pattern](/imgs/blogs/async-long-running-agents-10.webp)

**The latency thresholds:**

| Expected duration | Recommendation |
|---|---|
| < 2 seconds | Sync. Always. Any async overhead (queue write, worker pickup, poll round-trip) would be longer than the task itself. |
| 2–10 seconds | Sync with streaming. Return a streaming response with chunked transfer encoding. The connection stays open but progress flows to the client. This gives you the responsiveness of streaming without the infrastructure of async queues. |
| 10–30 seconds | Gray zone. If user-facing: async + progress bar. If background/server-to-server: sync with a 30-second timeout is workable. |
| 30 seconds–10 minutes | Async mandatory. Use a task queue with polling or SSE. |
| > 10 minutes | Async + email/push notification. The user has left the page; you need an out-of-band notification channel. |

**When sync is fine:**

- The task is genuinely fast (< 2 seconds P99, not just P50)
- The caller is a server, not a browser, and can hold a connection for 30 seconds
- The task has no retry/resume requirement — failure means "tell the user and let them try again"
- You have fewer than 10 concurrent users at peak
- You want to ship in a day, not a week

**When async is mandatory:**

- P99 task latency > 30 seconds
- The task calls multiple external APIs, each of which can be slow or flaky
- You need per-step retry with exponential backoff (retry the failed LLM call, not the entire task)
- You need to cancel tasks mid-execution
- You have more than 50 concurrent users
- The task output is large enough that delivering it in a single HTTP response is impractical
- You need to show progress to users in real time

**The hidden cost of async.** Every async system adds operational surface: you need to monitor queue depth, manage worker health, handle DLQ accumulation, maintain result store TTLs, and think about what happens when a worker dies mid-task. The systems I have seen that used async prematurely paid this cost without getting the benefit. The systems that went async reluctantly — only when they hit the sync ceiling — tended to have clean, minimal implementations that were easy to operate.

The best async agent system is the simplest one that reliably handles your actual workload. Start with sync, instrument your P99 latency, and migrate to async exactly when the numbers tell you to.

---

For more on observability across async agent systems, including how to trace a request from submission to completion across multiple queue hops and worker processes, see [observability and tracing for AI agents](/blog/machine-learning/ai-agent/agent-observability-and-tracing). For how to layer cost caps and circuit breakers on top of the async architecture so a single runaway task cannot exhaust your LLM budget, see [circuit breakers and cost caps](/blog/machine-learning/ai-agent/circuit-breakers-and-cost-caps). For the stateful persistence patterns that long-running agents need — checkpoints, session state, and memory across task restarts — see [stateful agent deployment](/blog/machine-learning/ai-agent/stateful-agent-deployment).
