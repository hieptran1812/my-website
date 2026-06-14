---
title: "Work Queues and Competing Consumers: Celery, Sidekiq, and Task Distribution"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The task queue is the most common message queue in production. Learn the competing-consumers pattern, the full task lifecycle, prefetch and fair dispatch, retries and scheduling and priority, chaining and workflows, why every task must be idempotent, and how to size a worker pool with real numbers across Celery, Sidekiq, BullMQ, SQS, and Kafka."
tags:
  [
    "message-queue",
    "task-queue",
    "celery",
    "sidekiq",
    "background-jobs",
    "kafka",
    "rabbitmq",
    "sqs",
    "distributed-systems",
    "event-driven",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/work-queues-competing-consumers-task-distribution-1.webp"
---

Open the codebase of any web application that has survived contact with real traffic and you will find a task queue. It might be called Celery, or Sidekiq, or Resque, or BullMQ, or a hand-rolled loop over an SQS queue, but it is there, and it is doing the same thing in every one of them: it is taking work that does not need to finish before you reply to the user, peeling it off the request path, and handing it to a pool of background workers to grind through on their own time. Sending the welcome email. Resizing the uploaded image into six thumbnails. Generating the monthly PDF report. Re-indexing the search document. Charging the card, calling the third-party API, warming the cache. None of these belong in the 80 milliseconds you have to render an HTTP response, so they get enqueued as **tasks**, and a separate fleet of processes — the **workers** — pulls them off and executes them.

This is, by an overwhelming margin, the most common real-world use of a message queue. Long before a team needs Kafka for event streaming or a fancy pub/sub fan-out, they need to move `send_email()` off the web request. The pattern that makes it work has a name that is older than any of these tools: **competing consumers**. You put N workers on one queue, and the broker hands each task to exactly one of them. Add workers and throughput goes up; remove workers and it goes down. There is no broadcast, no fan-out, no "every consumer sees every message" — that is pub/sub, and it is a different pattern for a different problem. A work queue is the opposite: each unit of work must be done **once**, by **one** worker, and the system's job is to spread the work evenly across however many workers you happen to be running. Figure 1 is the whole idea in one picture, and the rest of this post is the engineering that makes it survive production.

![One work queue feeding multiple competing workers where each task is delivered to exactly one worker](/imgs/blogs/work-queues-competing-consumers-task-distribution-1.webp)

By the end of this post you will be able to size a worker pool from a target task rate, explain exactly why a misconfigured prefetch lets one worker hoard a batch and starve the others, set retries with backoff that do not turn into a retry storm, chain tasks into a workflow without losing your mind, and — the lesson that costs the most money when learned the hard way — write your tasks so that running them twice does no harm. Because at-least-once delivery is the default, and at-least-once means a task **will** run twice eventually, and a task that charges a card twice is a task that should have been idempotent.

## 1. Why background jobs belong off the request path

Start with the request you are actually trying to make fast. A user submits a form to sign up. What has to happen before you can return `200 OK` and let them see the "welcome" page? You have to write the user row to the database. That's it. That's the only thing the user is waiting on. Everything else — sending the confirmation email, provisioning their default workspace, adding them to the CRM, kicking off a "day 1" onboarding drip, warming a recommendations cache — is work that *results from* the signup but does not need to *block* the signup. If you do all of it inline, your signup endpoint's latency is the **sum** of every downstream call, including the email provider's API, which on a bad day takes four seconds and on a very bad day times out entirely. Now your signup is broken because Mailgun is having a moment.

The whole point of pushing work off the request path is to decouple the latency and the failure domain of the response from the latency and failure domain of the side effects. This is the load-leveling and decoupling argument that the queue exists to serve, covered at length in the companion post on [message queues for async decoupling and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling). Here we care about the specific shape that decoupling takes when the messages are *units of work*: a producer (the web request) drops a small task description onto a queue and immediately returns; a worker picks it up later and does the heavy lifting.

There are three concrete wins, and it is worth being precise about each because they drive different design decisions.

**Latency.** The response returns as soon as the durable enqueue completes. Enqueuing to Redis is sub-millisecond; enqueuing to RabbitMQ with a publisher confirm is a couple of milliseconds; enqueuing to SQS over the network is single-digit to low-double-digit milliseconds. Compare that to the hundreds of milliseconds or multiple seconds the actual work might take. You have converted a variable, fat, failure-prone latency into a small, predictable one.

**Failure isolation.** If the email provider is down, the *email task* fails and gets retried — the signup does not. The user is already in the database, already logged in, already happy. The failure is contained to the one piece of work that depends on the flaky dependency, and the queue gives you a place to park and retry that work without involving the user at all.

**Throughput shaping (load leveling).** A flash sale or a viral moment can spike your request rate 10x for ninety seconds. If every request triggered six synchronous downstream calls, your downstream services would fall over. With a queue in front, the spike lands as a pile of enqueued tasks; the worker pool drains them at its own steady rate. The queue absorbs the burst and the workers see a smooth flow. This is the difference between a system that bends and one that breaks.

### What belongs on the queue and what does not

The decision rule is simple to state and surprisingly easy to get wrong: **anything the user is not literally waiting to see the result of should go on the queue.** Sending an email — queue it. Resizing an avatar — queue it, and show a placeholder until it's done. Computing the total the user is staring at on the checkout page — that is *not* a background job, that is the response, do it inline.

The grey area is "the user kind of wants to see this but it's slow." Generating a 40-page PDF report is the classic example. You cannot make the user stare at a spinner for thirty seconds while a synchronous request churns — the load balancer will time it out at 30 or 60 seconds and the connection will die anyway. The pattern here is **asynchronous request-reply**: enqueue the report-generation task, return a `202 Accepted` with a job ID immediately, and let the client poll a status endpoint (or open a websocket, or wait for a push notification) until the result backend says "done, here's the URL." The work is on the queue; the *user experience* is a progress bar. We will return to result backends in section 4, because that polling endpoint reads from one.

### The hidden cost: you now have two systems

Pushing work off the request path is not free, and the principal-engineer move is to be honest about what it costs you before you reach for it reflexively. The instant you introduce a task queue, you have gone from one system (the web app) to *three*: the web app, the broker, and the worker fleet. That is three things to deploy, three things to monitor, three things that can be the cause when something is broken at 3am. The work is no longer synchronous, which means it can fail *after* you've already told the user "got it" — so you need a story for surfacing those delayed failures (a notification, a status that flips to "failed," a support-visible error). The work is no longer ordered relative to the request, so you can no longer assume the email goes out before the user can possibly log in again. And the work is now *eventually* consistent with the request: there's a window, usually small but occasionally large under backlog, where the signup row exists but the welcome email hasn't been sent and the workspace hasn't been provisioned. Your product and your code have to tolerate that window.

None of this is a reason *not* to use a task queue — the wins are overwhelming and the alternative (synchronous everything) is worse in almost every dimension. But it is a reason to be deliberate. A surprising amount of work that teams shove onto queues didn't need to be there: a fast, reliable, sub-50-millisecond operation that you queued "to be safe" now carries the full operational weight of the async machinery for no latency benefit, plus the eventual-consistency window, plus the duplicate-execution risk. The bar for queuing should be "this is slow, flaky, or bursty, and the user doesn't need its result inline" — not "this happens after a request." Plenty of post-request work is genuinely fine to do synchronously.

### The contract: small, serializable, self-contained tasks

There is a discipline to *what* you put on the queue that pays for itself many times over. A good task is **small** (a description of work, kilobytes not megabytes), **serializable** (its arguments survive a round-trip through JSON or whatever your broker speaks — which means primitives and IDs, never live objects, open file handles, or database connections), and **self-contained** (everything it needs is in its arguments or fetchable from them). The most common rookie mistake is passing a fat object — a fully-loaded ORM model, a request body, a blob — as a task argument. It bloats the broker, it can fail to serialize, and worst, it captures a *stale snapshot*: the object you serialized at enqueue time may be out of date by the time the worker runs it minutes later. Pass the ID; let the worker fetch the *current* state. The task `update_search_index(doc_id=5567)` re-reads document 5567 when it runs and indexes whatever it is *now*; the task `update_search_index(doc={...20KB of stale JSON...})` indexes whatever it *was*. The first is correct and cheap; the second is a bug and a bloat in one.

## 2. The competing-consumers pattern

Competing consumers is the load-balancing pattern for messaging. One queue, many consumers, and the broker's contract is: **each message goes to exactly one consumer.** The consumers "compete" for messages — whoever is free grabs the next one. It is the messaging equivalent of a single line at the bank feeding multiple tellers, which is, not coincidentally, also the queueing-theory-optimal way to run a multi-server queue (one shared line beats one line per teller, every time, because no customer gets stuck behind a slow transaction while a teller sits idle next door).

Contrast this sharply with **publish/subscribe**, where each message is delivered to *every* subscriber. Pub/sub is for "tell everyone this happened" — an order was placed, and the inventory service, the email service, and the analytics service each independently react. Competing consumers is for "this needs doing once" — resize this one image. If you find yourself wanting each task done once but you've reached for a pub/sub topic, you've got the wrong tool, and you will discover this when you scale up and suddenly every image gets resized three times because you have three subscribers. The three messaging models — queue, pub/sub, and log — and exactly when each applies are laid out in [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models); a work queue is squarely the first model.

The beautiful property of competing consumers is **horizontal scalability with zero coordination.** The workers do not talk to each other. They do not elect a leader, partition the work among themselves, or maintain a shared assignment table. Each one independently asks the broker "give me a task" and the broker, holding the single shared queue, hands out tasks one at a time. To double your throughput you start more workers; to halve it you stop some. The broker is the coordination point, and because it already serializes access to the queue, the workers need no coordination of their own. Figure 2 makes the scaling visceral: one worker drains a backlog in a slow serial trickle, while a pool drains the same backlog in parallel.

![Comparison of a single worker draining a backlog serially against a worker pool draining it in parallel](/imgs/blogs/work-queues-competing-consumers-task-distribution-2.webp)

### The one rule competing consumers must enforce

For "exactly one consumer" to hold, the broker has to prevent two workers from grabbing the same task. The mechanism differs by system but the idea is universal: when a worker takes a task, the broker makes that task **invisible** or **reserved** so no other worker can also take it. In RabbitMQ, an unacknowledged message delivered to one consumer is not delivered to another. In SQS, a received message becomes invisible for the duration of its *visibility timeout*. In a Redis-backed queue like Sidekiq or Celery, the worker uses an atomic pop (`BRPOPLPUSH` into a per-worker in-flight list, or a `ZPOPMIN` on a sorted set) so the operation that removes the task and the operation that claims it are one indivisible step. The atomicity is what makes the pattern safe; without it, two workers polling at the same millisecond would both get the same task.

The catch — and it is the catch that drives half of this post — is what happens when a worker takes a task and then **dies** before finishing. The task was made invisible, but the worker that held it is gone. The broker cannot tell "still working" from "crashed." So after a timeout, it makes the task visible again and another worker picks it up. That is *at-least-once* delivery, and it is the reason a task can run twice. Hold that thought; it is the spine of section 8.

### The ordering tax you pay for parallelism

Competing consumers buys you throughput, but it spends something to do it: **ordering**. The moment you have N workers pulling from one queue, the tasks finish in *whatever order they finish* — task B, enqueued after task A, can complete before A if B's worker is faster or A's worker hit a slow task. For most background work this is completely fine: it does not matter whether user 5's welcome email or user 6's goes out first. But the instant your tasks have a *per-entity ordering requirement* — "apply these three balance updates to account 5567 in the order they were issued" — naive competing consumers will reorder them and corrupt your state, because update 2 might run on worker A while update 1 is still running on worker B.

There are two honest answers, and you should know both. The first is **don't put ordered work on a fan-out queue** — if account 5567's updates must be serial, they aren't independent units of work and the competing-consumers pattern is the wrong fit; route them through something that preserves per-key order. The second, when you *do* need both parallelism across entities and order within an entity, is **partition by key**: hash the entity ID to one of K queues (or Kafka partitions), so all work for account 5567 always lands on the same queue and is therefore processed serially by one worker, while accounts 5566 and 5568 land on other queues and run in parallel. You get parallelism *across* keys and order *within* each key. This is exactly the model Kafka enforces with partitions, and it's why "Kafka as a task queue" caps parallelism at the partition count — the partition is the unit of both ordering and parallelism. The deep treatment of how partitioning trades global order for per-key order and scalable parallelism is in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees); for a task queue the rule is: **a plain work queue gives you no ordering, so either your tasks must be order-independent or you must partition by key to recover per-key order.**

## 3. Task lifecycle: enqueue, reserve, execute, ack

Every task, in every one of these systems, moves through the same five states. The names differ — "reserve" vs "receive" vs "lease" vs "fetch," "ack" vs "delete" vs "commit" — but the shape is identical, and once you can see it, every task queue you ever touch becomes legible. Figure 3 is that lifecycle.

![The five-state task lifecycle moving from enqueue through reserve, execute, acknowledge, and store result](/imgs/blogs/work-queues-competing-consumers-task-distribution-3.webp)

**Enqueue.** The producer serializes a task — its name and its arguments — into a message and writes it to the broker. Critically, the message is *small*: it is a *description* of work, not the work's data. You enqueue `resize_image(image_id=98123, size="thumb")`, not the 4 MB JPEG. The worker fetches the image from object storage by ID. Keeping task payloads small keeps the broker fast and cheap; a queue is not a file store. If you find yourself wanting to stuff a megabyte into a task, put the megabyte in S3 and enqueue the S3 key — this is the *claim-check* pattern.

**Reserve.** A worker pulls the task and the broker marks it in-flight: reserved, invisible, leased, not-yet-acked — pick your broker's word. While reserved, no other worker can take it. The reservation has a clock attached: a visibility timeout (SQS), a lease duration, or in RabbitMQ the implicit "until the channel acks or the connection drops." This clock is the safety valve that recovers tasks from dead workers.

**Execute.** The worker runs your code. This is where the actual side effect happens — the email is sent, the image is resized, the row is written. This is also where everything that can go wrong goes wrong: the code throws, the third-party API times out, the worker runs out of memory, the pod gets OOM-killed, the deploy rolls and SIGTERMs the process mid-task.

**Ack.** On success, the worker tells the broker "done" — ack in RabbitMQ, `DeleteMessage` in SQS, removing the task from the in-flight list in a Redis queue, committing the offset in Kafka. The ack is what makes the task's removal *permanent*. Until the ack, the task is recoverable; after the ack, it is gone. The entire durability story hinges on **acking only after the work is truly complete.** Ack before you do the work (or, equivalently, configure auto-ack) and a crash mid-execution loses the task silently — that is at-most-once. Ack after, and a crash mid-execution re-runs the task — that is at-least-once. The full mechanics of ack, nack, requeue, and the visibility timeout are dissected in [push vs pull and acknowledgements](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read); for a task queue the rule compresses to one line: **ack last.**

**Result (optional).** If anyone cares about the outcome — a status, a return value, a "done" flag — the worker writes it to a *result backend*. Many tasks don't need this; fire-and-forget jobs like "send email" just ack and move on. But the async request-reply pattern from section 1 needs it, because the polling endpoint reads the result from there.

Here is the lifecycle as real Celery code, so the abstract states become method calls:

```python
# tasks.py — Celery, the canonical Python task queue
from celery import Celery

app = Celery("jobs", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")

@app.task(
    bind=True,
    acks_late=True,            # ack AFTER the task body returns, not before
    max_retries=5,
    default_retry_delay=2,     # base seconds, grows on retry
)
def resize_image(self, image_id, size):
    try:
        img = storage.fetch(image_id)        # execute: real side effect
        thumb = img.resize(SIZES[size])
        storage.put(f"{image_id}/{size}", thumb)
        return {"image_id": image_id, "size": size, "ok": True}  # -> result backend
    except TransientError as exc:
        # reserve again later with backoff instead of failing outright
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

```python
# producer side — enqueue from the web request and return immediately
result = resize_image.delay(image_id=98123, size="thumb")
return {"job_id": result.id, "status_url": f"/jobs/{result.id}"}, 202
```

The `acks_late=True` is the single most important flag in that snippet. By default Celery acks a task the moment a worker *receives* it (early ack), which means a worker crash mid-resize silently drops the task. Setting `acks_late=True` moves the ack to after the function returns, so a crash re-queues the task. The cost is that you now have at-least-once semantics and must make the task idempotent — which is exactly the bargain section 8 is about.

### The reservation clock and the heartbeat problem

The reserve state deserves more than a passing mention, because the *clock* attached to a reservation is where the subtlest task-queue bugs live. When a worker reserves a task, the broker starts a timer. If the worker acks before the timer expires, all is well. If the timer expires first, the broker assumes the worker died and re-makes the task available. The whole correctness of "exactly one worker per task" rests on that timer being longer than the task's real execution time — and shorter than the time you're willing to wait to recover a task from a genuinely dead worker. Those two constraints pull in opposite directions, and tuning the timer is a real engineering decision, not a default to copy-paste.

Different systems set the clock differently, and the differences matter operationally. RabbitMQ ties the reservation to the *channel*: a delivered-but-unacked message stays reserved until the consumer acks or the TCP connection drops, at which point it's immediately requeued — there's no fixed timeout, the liveness signal is the connection itself. That's elegant when connections are stable but means a hung worker that holds its connection open (deadlocked, not crashed) holds its task forever, which is why RabbitMQ added an optional *consumer timeout* (default 30 minutes) to reclaim tasks from stuck-but-connected consumers. SQS uses an explicit, per-message **visibility timeout** with a default of 30 seconds and a maximum of 12 hours, and gives you `ChangeMessageVisibility` to extend it mid-flight. Redis-backed queues like Sidekiq use a *reliable fetch* variant (a per-worker in-flight list plus a recovery sweep) so that if a worker process vanishes, a janitor process notices the orphaned in-flight entry and re-queues it.

The pattern that saves you across all of them is the **visibility heartbeat**: for tasks whose duration is variable or genuinely long, don't set a single huge timeout and hope — set a modest timeout and *extend* it periodically while the task runs. A 10-minute video transcode under a 30-second visibility timeout should extend its lease every 20 seconds, so that the lease always stays ahead of "now" while work is happening, but expires quickly (within ~30 seconds) the moment the worker actually dies. This gives you fast recovery on real failures *and* no spurious redelivery on slow-but-healthy tasks — the best of both. The anti-pattern is the single fixed 12-hour timeout: it never spuriously redelivers, but a task on a worker that died at minute one isn't recovered until hour twelve.

```python
# SQS worker: extend the lease while the task runs, so a healthy long
# task is never redelivered but a dead worker is recovered in ~30s
import threading

def process_with_heartbeat(sqs, queue_url, msg, handle):
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(20):              # every 20s
            sqs.change_message_visibility(     # push the lease 30s into the future
                QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"],
                VisibilityTimeout=30)
    t = threading.Thread(target=heartbeat, daemon=True); t.start()
    try:
        handle(msg)                            # the long task runs here
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])  # ack
    finally:
        stop.set(); t.join()
```

### Where the task body actually runs

One more thing worth making explicit about the execute state, because it shapes everything about how you write task code: the worker runs your task body in a process or thread it controls, *not* in your web framework's request context. There is no HTTP request object, no logged-in user from the session, no database transaction inherited from the controller — the task starts cold. Everything the task needs must be in its arguments (which is why you pass `image_id`, not the image, and `user_id`, not the user object) or fetchable from its arguments. This is a feature, not a limitation: it forces tasks to be self-contained, which is exactly what makes them retryable and idempotent. A task that depends on ambient request state is a task that can't be safely re-run on a different worker an hour later, which defeats the entire purpose. Write task bodies as pure functions of their arguments plus durable stores, and the rest of this post's guarantees fall into place.

## 4. The task-queue ecosystem (Celery, Sidekiq, BullMQ, SQS, Kafka)

There is no shortage of task queues, and the honest truth is that most of them are good and the right choice is usually dictated by your language and your existing infrastructure more than by any feature checklist. Still, they make genuinely different tradeoffs. Figure 4 lays the major systems against the features that actually matter in operation.

![Feature matrix comparing Celery, Sidekiq, BullMQ, SQS, and Kafka across broker, scheduling, priority, workflows, result backend, and ops effort](/imgs/blogs/work-queues-competing-consumers-task-distribution-4.webp)

**Celery (Python)** is the 800-pound gorilla of Python background jobs. It is a *framework*, not just a library: it brings task definitions, a result backend, a scheduler (Celery Beat) for periodic tasks, and a rich workflow layer (the "canvas" — chains, groups, chords). It speaks to either RabbitMQ or Redis as a broker, with RabbitMQ being the more robust choice and Redis the simpler one. Celery is powerful and, famously, *fiddly*: it has a large surface area, a lot of footguns around prefetch and acks (the defaults are not the safe ones), and a reputation for mysterious behavior under load. When configured correctly it is rock solid; the "correctly" is doing a lot of work in that sentence, and most of this post's tuning advice is aimed squarely at it.

**Sidekiq (Ruby)** is the gold standard for Ruby on Rails background jobs and, in my experience, the best-engineered of the bunch from an operability standpoint. It is Redis-backed, multi-threaded (one process runs many threads, which is efficient because Rails jobs are usually I/O-bound), and the defaults are sane. Sidekiq's killer feature is its dead-set and retry handling: failed jobs go into a retry set with exponential backoff and jitter *by default*, and after exhausting retries land in a "dead" set you can inspect and re-drive from a web UI. Priorities are done with weighted queue ordering. The open-source version is excellent; Sidekiq Pro and Enterprise add batches (its workflow primitive), reliable fetch, and rate limiting.

**Resque (Ruby)** is Sidekiq's predecessor — Redis-backed, but process-per-job and fork-based rather than threaded. It is simpler and more memory-isolating (a leaky job can't poison a long-lived process because each job runs in a fresh fork) but much less efficient. Most new Ruby projects pick Sidekiq; Resque persists in older codebases and where the fork-per-job isolation is genuinely wanted.

**BullMQ (Node.js)** is the modern standard for Node background jobs, Redis-backed, and notable for a clean TypeScript API and first-class support for delayed jobs, repeatable (cron) jobs, rate limiting, and **flows** — its DAG-based workflow primitive that lets a parent job depend on the completion of a tree of children. If you are in the Node ecosystem, BullMQ is the default answer, and its flow support makes it punch above its weight for orchestration.

**Amazon SQS** is not a task-queue framework — it is a managed *broker* — but it is used as the backbone of an enormous number of task queues because it is operationally trivial: no servers to run, near-infinite scale, pay per request. You bring your own worker loop (or use a thin framework on top). SQS gives you the broker primitives — visibility timeout, redrive to a dead-letter queue, delay queues up to 15 minutes — but *not* the higher-level features: no native priority, no built-in chaining, no result backend, no scheduler beyond the 15-minute delay cap. You assemble those yourself or do without. The trade is the lowest possible ops effort for the least built-in functionality.

**Kafka as a task queue** is the controversial one, so let me be blunt: Kafka is a *log*, not a task queue, and using it as one means living with the consequences of that. Within a consumer group, partitions are assigned to consumers, so your parallelism is capped at the partition count — 12 partitions means at most 12 concurrent workers on that topic, full stop, no matter how many worker processes you start. There is no per-message ack and no per-message redelivery: you commit *offsets*, and a single poisoned message can wedge a partition because you cannot ack message 100 while message 99 is stuck. People do build task queues on Kafka (the durability and replay are genuinely attractive, and the throughput is enormous) but they pay for it with rigid parallelism and DIY everything-else. The internals of why — log segments, consumer groups, partition assignment — are covered in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) and [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing). Use Kafka as a task queue when you are *already* on Kafka and the work maps cleanly onto partitions; reach for a real task queue otherwise.

Here is a rough decision table to anchor the choice:

| Situation | Reach for |
| --- | --- |
| Python web app, need full workflows + scheduling | Celery (RabbitMQ broker) |
| Ruby on Rails, want the smoothest ops | Sidekiq |
| Node/TypeScript app | BullMQ |
| Want zero servers to operate, simple jobs | SQS + a thin worker |
| Already all-in on Kafka, work maps to partitions | Kafka consumer group |
| Tiny app, a few jobs, hate dependencies | Redis list + a hand-rolled loop |

## 5. Prefetch, concurrency, and fair dispatch

This is the section that, if you internalize nothing else, will save you the most production pain — because prefetch is the single most misunderstood knob in task queues, and the wrong setting produces a failure mode (one worker hoarding while others sit idle) that is maddening to diagnose because it looks like a capacity problem when it is a *distribution* problem.

### Three distinct knobs

People conflate these constantly. They are not the same thing.

**Concurrency** is how many tasks a *single worker process* runs at once. A Sidekiq process with 10 threads runs 10 jobs concurrently. A Celery worker with `--concurrency=8` and the prefork pool forks 8 child processes. This is your per-process parallelism, and you size it to the *nature* of the work: CPU-bound work wants concurrency near the core count (more than that just thrashes the scheduler); I/O-bound work, which spends most of its time waiting on network or disk, wants concurrency much higher than the core count because the threads are mostly asleep.

**Prefetch** (the `prefetch_count` in RabbitMQ, `worker_prefetch_multiplier` in Celery, the `maxInFlight`/batch size elsewhere) is how many tasks a worker is allowed to *hold in its local buffer* beyond the ones it is actively running. It exists to hide network latency: if a worker had to round-trip to the broker for every single task, it would spend its life waiting on the network between tasks. Prefetch lets it grab a few ahead so there's always work in the local buffer. Prefetch is a *throughput optimization* and a *fairness liability* in the same breath.

**Pool size** is how many worker processes (and therefore how many machines or pods) you run. This is your horizontal scale, the knob from sections 2 and 9.

### The fair-dispatch problem

Here is the trap. Suppose a worker's prefetch is set high — say 100, or worse, unlimited. The worker connects, the broker sees a hungry consumer, and shoves 100 tasks into that worker's local buffer immediately. Now those 100 tasks are *reserved by that worker* and invisible to everyone else. If three of those tasks happen to be slow (a 30-second video transcode each) and the other 97 are fast, the worker chews through them in buffer order, and meanwhile the *other* workers — who would happily have taken the fast tasks — are sitting idle because there's nothing left in the queue: it's all hoarded in worker A's buffer. You have a fleet of idle workers and a growing latency on tasks that should have been instant. This is **head-of-line blocking** at the dispatch layer, and it is exactly the scenario figure 9 contrasts.

![Before-and-after comparison of an over-high prefetch causing one worker to hoard tasks versus a tuned low prefetch spreading work evenly](/imgs/blogs/work-queues-competing-consumers-task-distribution-9.webp)

The fix is **fair dispatch**: set prefetch *low* — `prefetch_count=1` in the classic RabbitMQ formulation, or `worker_prefetch_multiplier=1` in Celery for slow, uneven tasks. With prefetch 1, a worker holds exactly one task: the one it is working on. It does not get the next one until it acks the current one. So a slow task occupies *only its own worker*; the other workers keep pulling fast tasks off the still-populated queue and draining them. No idle workers, no hoarding, even distribution. Figure 6 shows fair dispatch on a timeline: the three fast tasks ack and exit while the one slow task finishes alone on its worker, never blocking anyone.

![Timeline showing fair dispatch with prefetch one where three fast tasks finish quickly while one slow task occupies only its own worker](/imgs/blogs/work-queues-competing-consumers-task-distribution-6.webp)

The catch with prefetch 1 is that you pay a broker round-trip between every task, so on *fast, uniform* tasks (a 2 ms task with a 1 ms round-trip) prefetch 1 cuts your throughput by a third because the worker is idle one third of the time waiting on the network. That's why the right setting depends on your task profile: **high prefetch for fast, uniform, short tasks** (amortize the round-trip), **low prefetch for slow, variable, long tasks** (preserve fairness). The pathology is using a high prefetch on a queue with a mix of fast and slow tasks — that is when one slow task in a hoarded batch wrecks your tail latency.

#### Worked example: quantifying prefetch starvation

Make it concrete. You have **4 workers** and a queue with **400 tasks**: 388 fast tasks at **10 ms** each and 12 slow tasks at **5 seconds** each, randomly interleaved. Total work is 388 × 0.01 s + 12 × 5 s = 3.88 s + 60 s = **63.88 seconds of CPU**, which across 4 workers is an ideal wall-clock of about **16 seconds** if work is spread evenly.

Now set **prefetch = 100** (each worker grabs a quarter of the queue, 100 tasks, into its local buffer up front). By the luck of the draw, worker A's 100-task buffer contains 4 of the slow tasks and worker B's contains 5. Worker A must chew through its buffer in order; it has 4 × 5 s = 20 s of slow work plus ~1 s of fast work it cannot offload to anyone — its buffer is reserved. Worker B has 5 × 5 s = 25 s. Meanwhile workers C and D, with only 1–2 slow tasks each, finish their buffers in ~6–11 s and **go idle** — the queue is empty because all 400 tasks are hoarded across the four buffers. Wall-clock is now set by the unluckiest worker: **~25 seconds**, with two workers idle for the back half. You bought a **56% slowdown** (25 s vs 16 s) and burned half your fleet, purely from hoarding.

Set **prefetch = 1** instead. Each worker holds one task and pulls the next only when it acks. The 12 slow tasks get spread across the 4 workers as they come up — roughly 3 slow tasks per worker, 15 s of slow work each — and crucially, while one worker is stuck on a 5 s task, the other three are rapidly draining 10 ms tasks off the still-full queue. No worker goes idle until the very end. Wall-clock lands near the ideal **~16 seconds**. The only cost is one broker round-trip per task; at 10 ms tasks with a 1 ms round-trip that's a ~9% overhead, far cheaper than the 56% you lost to hoarding. The lesson: **on mixed workloads, low prefetch is not a throughput sacrifice, it's a throughput *gain* because it keeps every worker busy.**

## 6. Retries, scheduling, and priority

A task queue that only runs tasks once and gives up on failure is a toy. Production task queues are defined by what they do *around* the happy path: retrying transient failures, scheduling work for later, and ordering urgent work ahead of routine work. Figure 7 is the taxonomy of these features.

![Tree taxonomy of task-queue features showing scheduling, priority, retries, and chaining hanging off the core dispatch loop](/imgs/blogs/work-queues-competing-consumers-task-distribution-7.webp)

### Retries with backoff

When a task fails because of a *transient* problem — a timeout calling a third-party API, a database deadlock, a momentary network blip — the right move is to try again in a moment, not to give up. But retrying *immediately* is a mistake, and retrying immediately *in a tight loop across your whole fleet* is a catastrophe: if the downstream is down and 10,000 tasks all retry every 100 ms, you have built a denial-of-service attack against your own dependency and a CPU-melting hot loop against your own broker. That is a **retry storm**, and it is one of the most common ways a task queue takes down the very service it depends on.

The discipline is **exponential backoff with jitter**: wait 1 s, then 2 s, then 4 s, then 8 s, each retry doubling the delay, with a random jitter added so that a thundering herd of simultaneously-failed tasks does not all retry at the same instant. After a bounded number of attempts — five is a common default — give up and route the task to a **dead-letter queue** for human inspection rather than retrying forever. The full theory of backoff schedules, DLQs, and retry budgets is the subject of [dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff); the one-paragraph version is: *bound the attempts, grow the delay exponentially, add jitter, and DLQ the rest.*

A subtle but critical point: you should only retry **idempotent** failures, or failures where you know the side effect did *not* happen. If a task charges a card and then the worker crashes after the charge succeeded but before the ack, retrying that task charges the card *again*. This is the exact reason section 8 exists — backoff retries multiply the duplicate-execution problem, so the safety of retries is downstream of the idempotency of tasks.

```python
# Celery: explicit exponential backoff with jitter and a retry ceiling
@app.task(bind=True, acks_late=True, max_retries=5)
def call_payment_api(self, charge_id):
    try:
        gateway.capture(charge_id)            # idempotent on charge_id (see section 8)
    except TransientGatewayError as exc:
        # 2**retries seconds, plus random jitter, capped — and DLQ after 5
        delay = min(2 ** self.request.retries, 60)
        raise self.retry(exc=exc, countdown=delay, jitter=True)
```

### Scheduling: delayed and periodic tasks

Two flavors. **Delayed tasks** run *once*, at a future time: "send this reminder in 24 hours," "expire this cart in 30 minutes," "retry this in 8 seconds." In Celery you pass `apply_async(countdown=30)` or an `eta=`; in BullMQ you pass a `delay`; in SQS you set a per-message `DelaySeconds` (capped at 15 minutes — for longer delays SQS users typically write to a DynamoDB table with a TTL or use a scheduler). The broker holds the task invisible until its time arrives, then makes it available. Note that a Redis-backed delayed task usually lives in a *sorted set* keyed by scheduled timestamp, and a background thread sweeps the set moving due tasks into the live queue — which is why "delayed" jobs in Sidekiq and BullMQ are slightly approximate (they fire within a poll interval of their target, not to the microsecond).

**Periodic tasks** run on a recurring schedule: "every night at 2am, generate the daily report," "every 5 minutes, refresh the leaderboard." Celery Beat is a separate scheduler process that enqueues these on a cron-like schedule; BullMQ has repeatable jobs; Sidekiq uses the sidekiq-cron or sidekiq-scheduler gems. The one operational gotcha: the scheduler is usually a *single* process, and if you run two of them (say, during a rolling deploy where old and new overlap) you get *double-scheduled* tasks. Periodic schedulers need leader election or a single-instance guarantee, and this is another place idempotency saves you.

### Priority queues

Sometimes work is not equal. A password-reset email must go out *now*; a marketing digest can wait an hour. Priority queues let urgent tasks jump ahead. There are two implementations, and the difference matters:

**Separate queues per priority** is the robust approach, and it's what Sidekiq does: you define a `critical` queue and a `default` queue and a `low` queue, and you configure workers to drain them in weighted order (e.g. check `critical` 6 times for every time you check `low`). This is predictable, debuggable, and avoids starvation if you use *weighted* rather than *strict* ordering. RabbitMQ and most production setups favor this.

**A single priority queue** where each message carries a priority number (RabbitMQ supports `x-max-priority`; Celery can map priorities onto this) is simpler to enqueue into but trickier in practice: strict priority can *starve* low-priority tasks forever if high-priority work never lets up, and priority queues interact badly with prefetch (a worker that prefetched 10 low-priority tasks won't preempt them when a high-priority task arrives — the high-priority task waits behind the prefetched batch, which is yet another reason to keep prefetch low when you care about priority).

The practical recommendation: **use separate weighted queues, and keep prefetch low on priority-sensitive queues.** Don't rely on in-queue priority numbers to do real-time preemption — they can't, because the prefetched batch is already committed to the worker.

### Rate limiting: when *not* draining fast is the goal

There is a fourth feature that sits alongside scheduling and priority and is easy to overlook until a third-party API rate-limits you into oblivion: **task rate limiting**. Sometimes the constraint is not "drain as fast as possible" but "drain *no faster* than 100 calls per second, because that's the partner API's ceiling and exceeding it gets us throttled or banned." A task queue with 50 workers will happily fire 500 concurrent calls at that API and earn you a wall of 429s. Rate limiting caps the *consumption* rate independent of how many workers you run.

Celery offers a per-task `rate_limit` (e.g. `"100/s"`); Sidekiq Enterprise and BullMQ have first-class rate limiters; SQS users typically implement a token bucket in Redis. The subtlety is that a rate limit must be *global across the fleet*, not per-worker — limiting each of 50 workers to 2/s to get 100/s total is fragile because it breaks the moment you scale the fleet. The robust implementation is a shared token bucket (a Redis key the whole fleet draws tokens from) so the limit holds regardless of worker count.

```python
# Global rate limit via a shared Redis token bucket — holds across the fleet
@app.task(acks_late=True)
def call_partner_api(payload):
    if not redis_token_bucket("partner-api", rate=100, per=1.0).acquire():
        # no token now: reschedule this task a moment later instead of hammering
        raise call_partner_api.retry(countdown=0.5)
    partner.send(payload)
```

This pattern — reschedule rather than block when you can't get a token — keeps workers free to do *other* work instead of sleeping on a held reservation, which matters when the rate-limited tasks share a queue with unconstrained ones. Like priority, rate limiting interacts with prefetch: a worker that prefetched ten rate-limited tasks will reschedule them one by one, so again, keep prefetch low on rate-limited queues.

## 7. Task chaining and workflows

Real work is rarely a single task. "Process this uploaded video" is: transcode it to three resolutions, generate a thumbnail, extract a transcript, and *then* — only after all of those finish — mark the video ready and notify the user. That is a **workflow**: a graph of tasks with dependencies, where some run in sequence, some run in parallel, and some run only after a fan-out completes.

Celery's "canvas" gives this a clean vocabulary that the other systems mirror:

- A **chain** runs tasks in sequence, passing each result to the next: `chain(transcode.s(vid), watermark.s(), publish.s())`. Task 2 starts when task 1 acks, task 3 when task 2 acks.
- A **group** runs tasks in parallel: `group(resize.s(img, s) for s in ["sm","md","lg"])` fans out three resizes that run concurrently across the worker pool.
- A **chord** is a group with a callback that fires *after every task in the group completes*: `chord(group(...))(finalize.s())`. This is the fan-in primitive, and it is the one with the sharpest edges.

```python
# Celery canvas: fan out three resizes, then notify once all three finish
from celery import chord, group

workflow = chord(
    group(resize_image.s(img_id, s) for s in ["sm", "md", "lg"])
)(mark_ready_and_notify.s(img_id))
# the callback mark_ready_and_notify runs ONCE, after all 3 resizes ack
```

The chord's sharp edge is the fan-in. To know that "all three resizes finished," *something* has to count completions, and that something is the result backend: each resize, on completing, increments a counter, and when the counter hits the group size, the callback fires. This means **chords require a result backend** (you cannot do fan-in without somewhere to track the count), and it means the result backend becomes load-bearing — if it's a single Redis instance and it falls over, your chord callbacks never fire and your workflows hang half-done. At scale, chord-heavy workflows put real pressure on the result backend, and "why are my chords not completing" almost always traces back to result-backend contention or expired result keys (results have a TTL; if a slow group member finishes after the others' results expired, the count is wrong).

My honest advice on workflows: **keep them shallow.** A chain of three is fine. A chord of a group of ten is fine. A deeply nested DAG of chords-of-chains-of-groups is a debugging nightmare — when it gets stuck (and it will), reconstructing *which* sub-task didn't complete across three levels of fan-out is brutal. For anything genuinely complex — multi-day human-in-the-loop processes, long-running sagas, anything needing visual inspection of state — graduate to a real workflow engine (Temporal, AWS Step Functions, Airflow) that is *built* for durable orchestration with first-class state inspection, rather than bending your task queue's chaining primitives past their comfort zone. Task-queue workflows are for *short, automatic* DAGs; workflow engines are for *long, stateful, observable* ones.

There is also a simpler-than-canvas pattern worth naming: **the task that enqueues the next task.** Instead of declaring the whole DAG up front, each task does its piece and, at the end, enqueues its successor with the data it produced. This is less elegant but radically easier to reason about and debug — every step is an independent, retryable, idempotent task, and the "workflow" is just the chain of enqueues. For many pipelines this beats the canvas precisely because it has no fan-in coordination to go wrong.

## 8. Why tasks must be idempotent

This is the section that separates engineers who have run task queues in production from those who have not. The defining property of almost every task queue — Celery with `acks_late`, Sidekiq, SQS, Kafka, BullMQ — is **at-least-once delivery**: a task is guaranteed to run *at least* once, which is a polite way of saying it might run *more* than once. And the engineering consequence is absolute: **your tasks must be idempotent, meaning running them twice produces the same result as running them once.** Figure 8 places idempotency where it belongs — a property of the worker layer, not the broker.

![Stack diagram of task-queue layers showing broker, worker, concurrency, and result store with concurrency as the throughput-tuning layer](/imgs/blogs/work-queues-competing-consumers-task-distribution-8.webp)

### Why duplicates are inevitable, not rare

It is tempting to treat duplicate execution as a freak event you can ignore. It is not. Here are the *routine* ways a task runs twice, none of them exotic:

- **The ack is lost.** The worker finishes the task and sends the ack, but the network drops the ack packet or the broker connection blips before the ack lands. The broker, never having seen the ack, redelivers the task after the visibility timeout. The work is done; it gets done again.
- **The worker crashes after the side effect, before the ack.** The card is charged, then the pod is OOM-killed. No ack was sent. Redelivery. The card is charged again.
- **The visibility timeout is too short.** A task legitimately takes 90 seconds but the visibility timeout is 60. At second 60 the broker decides the worker is dead and redelivers to a *second* worker while the first is still happily running. Now *two* workers run the same task simultaneously. (This one is insidious because it's a config bug, not a crash, and it produces concurrent duplicates, not just sequential ones.)
- **A retry after a partial failure.** The task did half its work, threw on the second half, and got retried. The first half runs again.
- **A rebalance** (in Kafka) moves a partition to a new consumer that re-processes uncommitted offsets.

Every one of these is normal operational reality. At-least-once is not a bug; it is the *only* delivery guarantee you can get cheaply, because the alternative — at-most-once — silently *loses* tasks on the same failures, which is almost always worse. The full taxonomy of why exactly-once is a near-myth and at-least-once is what you actually build on is in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once). So you accept at-least-once and you defend against duplicates with idempotency.

### How to make a task idempotent

The core technique is a **deduplication key**: a stable identifier for the *unit of work* (not the message — the message ID changes on redelivery; the *work* does not), checked against a store of "already done" keys before doing the side effect. The full design space — natural keys, the inbox table, conditional writes, TTLs on the dedup store — is the subject of [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). The patterns that matter most for tasks:

**Use a natural idempotency key from the domain.** Don't invent a UUID per task attempt — derive a key from the work itself. "Send welcome email to user 98123" has the natural key `welcome-email:98123`. "Charge order 5567" has the key `charge:order:5567`. Before sending or charging, do a conditional insert of that key into a dedup table; if it's already there, skip — the work was already done.

```python
# Idempotent task: a dedup key makes redelivery a no-op
@app.task(acks_late=True)
def send_welcome_email(user_id):
    dedup_key = f"welcome-email:{user_id}"
    # atomic: insert only if absent; returns False if the key already existed
    if not dedup_store.insert_if_absent(dedup_key, ttl=86400):
        return "already sent, skipping"     # redelivery lands here harmlessly
    email_provider.send(template="welcome", to=user_lookup(user_id))
    return "sent"
```

**Push idempotency into the downstream when you can.** The cleanest dedup is the one you don't have to implement. Stripe's charge API takes an `Idempotency-Key` header; pass your stable key and Stripe itself collapses duplicate charges. A database upsert (`INSERT ... ON CONFLICT DO NOTHING`) is naturally idempotent. A "set status to shipped" is idempotent because setting it twice is the same as setting it once. Whenever the *operation itself* is naturally idempotent, you get duplicate-safety for free and don't need a dedup store at all.

**Make the dedup check and the side effect atomic where possible.** The dangerous gap is between "check if done" and "do it" — if a duplicate arrives in that window, both copies pass the check and both do the work. For database side effects, do the dedup and the write in one transaction. For external side effects you can't transact with, the conditional insert *before* the side effect (insert-then-act, accepting that a crash between them re-runs the act) is the standard compromise.

Figure 9 earlier contrasted prefetch settings; the idempotency analogue is the difference between a naive consumer that double-charges on redelivery and an idempotent one where the dedup key turns the duplicate into a harmless no-op. The rule to tattoo on the inside of your eyelids: **if a task has a side effect, assume it will run twice, and make twice equal once.**

## 9. Scaling the worker pool

The whole promise of competing consumers is that you scale by adding workers. Let's make that quantitative, because "add more workers" is only useful if you can compute *how many*. Figure 5 is the architecture you're scaling — producers into a broker, a worker pool, and a result backend — and the worker pool is the part that grows and shrinks.

![Grid architecture showing producers feeding a broker, a worker pool of workers, and a result backend that the caller polls](/imgs/blogs/work-queues-competing-consumers-task-distribution-5.webp)

### The worker-sizing formula

The math is Little's Law dressed up for task queues. To keep up with an arrival rate without the queue growing without bound, your *service rate* must meet or exceed your *arrival rate*:

> workers_needed ≈ (arrival_rate × time_per_task) / concurrency_per_worker

Or even simpler, in terms of total concurrent execution slots: you need enough slots so that `slots × (1 / time_per_task) ≥ arrival_rate`. Let's run it.

#### Worked example: sizing a worker pool

You need to handle **5,000 tasks per minute**, and each task takes **300 ms** of wall-clock time. First, normalize to a rate: 5,000 / 60 = **83.3 tasks/second** arriving.

How many tasks can *one execution slot* (one thread or one process running tasks serially) handle? One task every 0.3 s = **3.33 tasks/second per slot**. So the number of concurrent slots you need is 83.3 / 3.33 = **25 slots** to *exactly* keep up. But "exactly keep up" means zero headroom — any burst, any slow task, any GC pause and the queue starts backing up. You always provision for the *peak*, not the average, and you leave headroom. A common rule is to size for **2× the average** so you can absorb bursts and drain backlogs: **50 slots**.

Now translate slots into workers, which depends on per-worker concurrency. If these are I/O-bound tasks (calling APIs, waiting on the DB) and you run **10 threads per worker** (Sidekiq-style), you need 50 / 10 = **5 worker processes**. If they're CPU-bound and you run Celery prefork with **4 processes per worker** on 4-core boxes, you need 50 / 4 ≈ **13 worker machines**. Same task rate, very different fleet, entirely because of the concurrency-per-worker number — which is why getting concurrency right (section 5) is upstream of getting fleet size right.

A sanity check from the other direction: with 50 slots at 300 ms each, your *drain rate* is 50 × 3.33 = 166 tasks/second = **10,000 tasks/minute**, double the 5,000/min arrival. So a backlog of, say, 30,000 tasks left over from an outage drains in 30,000 / (10,000 − 5,000) per minute = **6 minutes** of catch-up, because while you're draining the old backlog new tasks keep arriving at 5,000/min and you only net 5,000/min of progress against the backlog. If you needed to drain faster, you'd temporarily scale to more workers. That last calculation — *catch-up time = backlog / (drain_rate − arrival_rate)* — is the one to keep in your head during an incident, because it tells you whether scaling out will clear the backlog in minutes or hours.

### Autoscaling on queue depth, not CPU

The biggest mistake in scaling worker pools is autoscaling on **CPU utilization**, the default metric for web servers. It is the wrong signal for workers. A worker pool that is I/O-bound (most are) can be at 15% CPU while a backlog of 100,000 tasks piles up — the workers are *waiting on the network*, not burning CPU, so CPU-based autoscaling never triggers and your backlog grows unbounded while your dashboard shows a sleepy fleet.

Scale on **queue depth** (or its derivative, queue *latency* — how long the oldest task has waited) instead. The signal you care about is "is work piling up faster than we can drain it," and queue depth answers that directly. Concretely: target a queue depth (or an "oldest message age") threshold, and add workers when you exceed it, remove workers when you're comfortably below. SQS exposes `ApproximateNumberOfMessagesVisible` and the message-age metric precisely for this; Kubernetes setups use KEDA with a Redis-list-length or SQS-queue-depth trigger. The cleanest target metric of all is **oldest-task-age**: "no task should wait more than 60 seconds before a worker starts it" is a crisp SLO that maps directly to a scaling rule.

### Graceful shutdown is non-negotiable

When you scale *down* — or when you deploy, which kills every worker and starts new ones — you must drain in-flight tasks gracefully. A worker that gets SIGTERM and exits immediately abandons the task it was running. If it had already acked, the task is *lost*; if `acks_late`, the task is redelivered (and runs twice, hence idempotency again). The right behavior on SIGTERM is: **stop pulling new tasks, finish the ones in flight, ack them, then exit.** Every good task queue supports this — Celery has a warm-shutdown on the first SIGTERM, Sidekiq has a configurable timeout to finish jobs, BullMQ has `close()`. Set the timeout generously (longer than your longest task) and make sure your orchestrator's termination grace period (Kubernetes `terminationGracePeriodSeconds`) is longer still, or the orchestrator SIGKILLs the worker mid-drain and you're back to abandoned tasks. A deploy that doesn't drain gracefully is a deploy that re-runs a slice of your tasks every single time you ship — a silent, recurring idempotency test you didn't sign up for.

## Case studies and war stories

### The unbounded prefetch that idled a fleet

A team running Celery with the default `worker_prefetch_multiplier=4` and `--concurrency=8` had an effective prefetch of 32 tasks per worker. Their queue carried a mix of sub-second tasks and occasional multi-minute report generations. During a busy period, the long reports got hoarded into a few workers' 32-task buffers, those workers wedged on the long tasks, and the rest of the fleet went idle because the short tasks were all locked in the busy workers' buffers. The dashboard showed low CPU across the fleet and a growing queue — the classic "we have capacity but tasks are slow" head-scratcher. The fix was two lines: route the long reports to a *separate* queue with its own dedicated workers, and set `worker_prefetch_multiplier=1` on the mixed queue. Queue latency dropped by an order of magnitude with *zero* added capacity. The lesson: **separate slow and fast work onto different queues, and keep prefetch low on anything with variable task durations.** Mixing wildly different task durations on one queue is the root cause of more "mysterious" task-queue latency than anything else.

### The retry storm that DDoSed an internal service

A payment-reconciliation task called an internal ledger service. The ledger service had a bad deploy and started returning 500s. The reconciliation tasks were configured to retry *immediately* with no backoff, `max_retries` effectively unbounded. Within two minutes, tens of thousands of failed tasks were retrying in a tight loop, and the retry traffic — far exceeding normal load — kept the ledger service pinned under load even after the bad deploy was rolled back, because the retry herd never let it recover. It took manually purging the retry queue to break the cycle. The fixes were textbook: exponential backoff with jitter, a hard `max_retries` of 5, a circuit breaker that stops calling a failing dependency entirely for a cool-down window, and a dead-letter queue so exhausted tasks parked for inspection instead of looping. The lesson: **retries without backoff and without a ceiling are a self-inflicted denial-of-service waiting for a downstream blip to trigger them.** This is exactly the poison-message and retry-storm containment problem covered in [poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment).

### The double-charge from a short visibility timeout

An SQS-backed worker processed payment captures. The visibility timeout was set to 30 seconds — a copy-pasted default — but a small fraction of captures, the ones hitting a slow payment processor, took 45 seconds. For those, SQS made the message visible again at second 30, a *second* worker received it, and both workers captured the *same payment* — the customer was charged twice. The bug was invisible in testing because test captures were fast; it only manifested on the slow tail in production. Two fixes, layered: raise the visibility timeout well above the *p99.9* task duration (and extend it dynamically via SQS's `ChangeMessageVisibility` for long-runners), and — the real fix — make the capture idempotent by passing the order ID as the payment gateway's idempotency key, so even a true concurrent duplicate collapses to one charge. The lesson: **a visibility timeout shorter than your slowest task produces *concurrent* duplicate execution, and the only durable defense is idempotency, not timeout tuning.** Timeout tuning reduces the probability; idempotency makes the duplicate harmless.

### The chord that never completed

A media-processing pipeline used a Celery chord: a group of N transcode tasks with a callback to publish the video. Under load, some chords simply never fired their callback — videos stuck in "processing" forever. The cause was the result backend: results had a default TTL, and on busy days a straggler transcode finished *after* its sibling results had already expired from Redis, so the chord's completion counter could never reach N. Compounding it, the single Redis result backend was saturated and dropping writes. The fixes: a much longer result TTL (longer than the worst-case chord span), a dedicated and replicated result backend separate from the broker, and — strategically — replacing the deepest chords with the simpler "each task enqueues the next" pattern that needs no fan-in counter. The lesson: **chords and other fan-in workflows make the result backend load-bearing; treat it as a production database, not a cache, and keep workflow DAGs shallow.**

## When to reach for this (and when not to)

**Reach for a task queue when** you have work that results from a request but doesn't need to block it: sending notifications, processing uploads, generating documents, calling slow third-party APIs, running periodic maintenance, fanning out a piece of work across a pool. This is the default tool for *background jobs*, and "is this a background job?" has a default answer of "yes, queue it" for anything the user isn't synchronously waiting on. If you're building a web app of any size, you will need one, and you should reach for the idiomatic one for your language — Celery for Python, Sidekiq for Ruby, BullMQ for Node — rather than rolling your own, because the retry, scheduling, and dead-letter machinery you'll inevitably need is genuinely hard to get right and these tools have already paid that cost.

**Reach for SQS plus a thin worker when** your ops appetite is small and your needs are simple: at-least-once delivery, retries via redrive, delays up to 15 minutes, and a dead-letter queue cover an enormous fraction of real use cases with literally zero servers to operate. You give up native priority, chaining, and a result backend, but for fire-and-forget jobs you don't miss them. This is the lowest-total-cost-of-ownership option and an excellent default for teams already on AWS.

**Do not reach for a task queue when** the work must complete before you can answer the user — that's just synchronous code, and dressing it up as an async task you immediately wait on adds latency and complexity for nothing. Do not use a task queue as a *database* (don't enqueue tasks you'll query by attribute later — that's a job table in your DB, queried with SQL). And do not use it as a *long-running stateful workflow engine*: when your "workflow" needs to span days, wait on human approval, survive process restarts with full state inspection, and let you ask "where is order 5567 in the pipeline right now," you have outgrown task-queue chaining and want Temporal, Step Functions, or Airflow. Task queues are for *short, automatic, retryable units of work*; the moment the work becomes long-lived and stateful, the right tool changes.

**Be cautious using Kafka as a task queue.** It works, and the durability and replay are real advantages, but you inherit partition-capped parallelism (parallelism = partition count, period), no per-message ack or redelivery, and DIY scheduling, priority, and dead-lettering. Use it when you're *already* on Kafka and the work maps cleanly onto keys and partitions; choose a purpose-built task queue otherwise. The infrastructure you'd run *anyway* should drive this — adding a whole Kafka cluster *just* to run background jobs is using a freight train to deliver a pizza.

## Key takeaways

- **Move off the request path anything the user isn't waiting to see.** Enqueue a small task description, return immediately, and let workers do the heavy lifting. You convert fat, flaky latency into a small, predictable one and isolate failures to the task.
- **Competing consumers = N workers, one queue, each task to exactly one worker.** Throughput scales with worker count and needs zero coordination between workers because the broker serializes access to the shared queue. This is *not* pub/sub.
- **Ack last.** The task is recoverable until the ack and gone after it, so acknowledge only after the work truly completes. With Celery, that means `acks_late=True` — and accepting at-least-once in return.
- **Prefetch is a fairness liability.** High prefetch lets one worker hoard a batch and starve idle peers; on mixed-duration workloads, low prefetch (1–2) is a throughput *gain*, not a sacrifice, because it keeps every worker busy. Separate slow and fast work onto different queues.
- **Retries need exponential backoff, jitter, a hard ceiling, and a dead-letter queue.** Retries without backoff are a self-inflicted denial-of-service against your own dependencies waiting for a blip to trigger them.
- **Every task with a side effect will run twice — make twice equal once.** At-least-once is the default and duplicates are routine (lost acks, crashes before ack, short visibility timeouts, retries). Defend with a natural idempotency key or a naturally-idempotent downstream operation.
- **Keep workflows shallow.** Chains of three and single chords are fine; deeply nested DAGs are a debugging nightmare and make the result backend load-bearing. For long, stateful orchestration, graduate to a real workflow engine.
- **Size workers from the task rate, scale on queue depth, and drain gracefully.** Concurrent slots needed ≈ arrival_rate × time_per_task; provision ~2× for bursts; autoscale on queue depth or oldest-task-age, never CPU; and finish in-flight tasks on SIGTERM or every deploy silently re-runs a slice of your tasks.

## Further reading

- [Push vs pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — the ack, nack, requeue, and visibility-timeout machinery a task queue is built on.
- [Dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff) — the full theory of backoff schedules, retry budgets, and where exhausted tasks go to die.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the design space for dedup keys, inbox tables, and conditional writes that make duplicate execution harmless.
- [Delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — why at-least-once is what you actually build on and exactly-once is mostly a story.
- [Poison messages and retry storms: containment](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — how a single bad task or a retry herd takes down a fleet, and how to stop it.
- [Queue vs pub/sub vs log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — where the competing-consumers work queue sits among the messaging models.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — why Kafka's partition model caps task-queue parallelism and what you trade for its durability.
- [Celery documentation](https://docs.celeryq.dev/) and [Sidekiq best practices](https://github.com/sidekiq/sidekiq/wiki/Best-Practices) — the canonical operational guidance for the two most common task queues.
