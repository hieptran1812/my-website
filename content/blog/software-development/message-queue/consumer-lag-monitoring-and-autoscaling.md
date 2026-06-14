---
title: "Consumer Lag: Monitoring, Alerting, and Autoscaling"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Consumer lag is the single most important health signal in a message system, and most teams alert on it wrong. Define it precisely, learn why time-lag beats records-lag, alert on the derivative and the retention cliff instead of an absolute threshold, tell a stalled consumer from a slow one, and autoscale on lag up to the hard partition ceiling."
tags:
  [
    "message-queue",
    "consumer-lag",
    "observability",
    "autoscaling",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "monitoring",
    "keda",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/consumer-lag-monitoring-and-autoscaling-1.webp"
---

There is exactly one number that tells you whether a message system is healthy, and it is not throughput, not broker CPU, not disk usage, not the consumer's own success rate. It is consumer lag: the count of records that have been written to the log but that your consumers have not yet processed. Throughput can look perfect while you are quietly drowning. Broker CPU can be flat while a single stuck consumer loses you a day of orders. The consumer can report a hundred percent success rate on every message it touches while falling further and further behind on the messages it has not gotten to yet. Lag is the one signal that integrates all of those into a single answer to the only question that matters in steady state: are we keeping up?

And almost every team alerts on it wrong. The default instinct is to pick a number — lag over one hundred thousand, page someone — and that alert will do two things, both bad. It will fire constantly during normal traffic bursts that drain on their own in thirty seconds, training your on-call to ignore it. And it will stay silent during the slow, steady fall-behind that actually kills you, because a consumer leaking five thousand records a second of capacity will sit under your threshold for an hour before crossing it, by which point you may already be staring down the retention cliff. The absolute-threshold alert is the smoke detector that screams when you make toast and stays quiet when the wiring melts inside the wall.

![A linear pipeline showing log-end-offset of 8.4 million minus committed-offset of 8.2 million yielding a lag of 200 thousand records that have not yet been processed](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-1.webp)

This post is about doing it right. We will define lag precisely — per partition, as log-end-offset minus committed-offset (figure 1) — and then spend most of our time on the two refinements that turn lag from a vanity metric into an operational instrument. The first is time-lag: not how many records you are behind but how many seconds of wall-clock time you are behind, because one million records of backlog might be two seconds or two hours depending on your throughput, and only the seconds tell you how urgent it is. The second is the derivative: not the value of lag but its rate of change, because steady lag is harmless and growing lag is the only thing that ever needs a human. By the end you will be able to instrument lag with the right tools, write alerts that fire only when something is actually wrong, distinguish a healthy queue from a slow consumer from a fully stalled one, autoscale consumers on lag with KEDA, and know the exact moment when scaling stops helping because you have hit the partition ceiling. This post builds directly on [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing), which explains the offsets we are about to subtract, and on [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling), which explains how to make a single consumer faster before you reach for more of them.

## 1. Why lag is the metric that matters

Start with what lag actually measures, because the word gets used loosely. Lag is a measure of *work not yet done*. Every record a producer writes is a unit of work waiting for a consumer. The broker's job is to hold that work durably until a consumer claims it; the consumer's job is to claim it and finish it. Lag is the size of the in-between: the work that has been accepted by the system but has not yet been completed. In that sense it is the queue depth of your entire pipeline, and queue depth is the canonical health signal of every system that does asynchronous work, from a CPU run queue to a restaurant kitchen's ticket rail.

The reason lag is uniquely valuable is that it is the *only* metric that catches a whole class of failures that every other metric misses. Consider what you would see on a dashboard during a slow consumer fall-behind. Producer throughput: normal, because producers do not care whether anyone is reading. Broker CPU and disk: normal or even slightly elevated, because the broker is happily accepting writes. Consumer error rate: zero, because the consumer is succeeding on every single message it processes — it is just processing them too slowly. Consumer CPU: maybe high, maybe not, depending on whether the bottleneck is CPU or a downstream database. Every per-component metric looks fine. The only thing that reveals the problem is the gap between what was written and what was read, and that gap is lag. Lag is an *integral* metric: it accumulates the difference between two rates over time, so it surfaces a sustained imbalance that instantaneous rate metrics paper over.

### Lag vs the metrics people reach for instead

Teams that have not yet been burned tend to alert on the wrong things. They alert on consumer throughput dropping, which is noisy because throughput legitimately drops when there is simply less traffic — a quiet Sunday is not an incident. They alert on consumer error rate, which catches messages that fail loudly but completely misses the slow consumer that succeeds on everything it touches. They alert on end-to-end latency measured by a synthetic probe, which is closer but expensive to build and only samples the path the probe takes. Lag dominates all of these because it is cheap to compute, it is already maintained by the broker as a byproduct of offset tracking, and it directly answers the keeping-up question without any inference.

There is a deeper reason lag is the right top-level signal, and it ties back to the fundamental inequality from [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling). If producers write at rate *P* and your consumer group processes at aggregate rate *C*, then lag's derivative is exactly *P* minus *C*. When *C* is greater than or equal to *P*, the derivative is zero or negative and lag is flat or draining. When *C* falls below *P*, the derivative goes positive and lag climbs without bound. Lag is therefore the *running integral of the capacity deficit*. Watching lag is watching whether your capacity inequality holds, in real time, without having to measure *P* and *C* separately. That is why a single number can carry so much: it is the time-integral of the one inequality that governs the whole system.

### What lag does not tell you

Lag is necessary but not sufficient, and it is worth being precise about its blind spots so you do not over-trust it. Lag tells you records are waiting; it does not tell you *why*. A lag of one million could mean a slow consumer, a traffic spike, a stalled consumer, or a rebalance in progress — four causes with four different fixes, which is the entire subject of section six. Lag also says nothing about correctness: a consumer can be perfectly caught up while silently corrupting every record it processes. And lag on a healthy steady-state system has a nonzero floor — there is always some small amount of in-flight work — so a lag of zero is actually a mild warning sign that your consumers are over-provisioned and idle. Lag is the headline metric, but it is the start of an investigation, not the end of one. We will spend section six on exactly how to read it.

### Lag as the universal queue-depth signal

It is worth seeing that lag is not a Kafka quirk but an instance of a pattern that recurs everywhere work is buffered between a producer and a consumer. A thread pool has a task queue, and the depth of that queue is lag. A web server has a connection backlog, and its depth is lag. A CPU has a run queue, and its length — the load average — is lag. A kitchen has a rail of order tickets, and the number hanging there is lag. In every one of these the same physics holds: depth is the integral of arrival rate minus service rate, depth growing means the system is overloaded, and depth in *time* (how long until the oldest item is served) is more actionable than depth in *count*. When you understand consumer lag you understand queueing in general, which is why the instinct it builds transfers directly to capacity-planning any asynchronous system. The message broker just makes the queue explicit and the depth measurable, where a thread pool hides it inside a runtime. That explicitness is a gift: most systems make you infer their backlog, while a message system hands it to you as a first-class number, and the discipline of this post — watch the trend, convert to time, alert on the velocity — applies to every queue you will ever operate, broker or not.

## 2. Defining lag: offsets and the formula

To compute lag you need two offsets per partition, and the precision of the definition matters because people conflate them and then misread their dashboards. The first is the **log-end-offset** (LEO): the offset of the next record the broker will assign, which is one past the last record currently in the partition. If a partition holds records at offsets 0 through 8,399,999, its log-end-offset is 8,400,000. The LEO advances every time a producer appends. The second is the **committed offset**: the offset the consumer group has durably recorded as "processed up to here," stored in Kafka's internal `__consumer_offsets` topic. If the group has committed offset 8,200,000, it is asserting that records 0 through 8,199,999 are done and that 8,200,000 is the next one to fetch.

Lag is the difference:

```
lag(partition) = log-end-offset − committed-offset
```

In our example, lag is 8,400,000 minus 8,200,000, which is 200,000 records (figure 1). Those are the records the consumer has not yet committed as processed. Total lag for the consumer group is the sum of per-partition lag across every partition the group owns. That sum is the headline number, but you must never lose the per-partition breakdown, because the sum hides the single most dangerous failure mode in the whole topic: one partition stuck at a huge lag while every other partition is at zero. Summed, that might look like moderate lag spread thinly; per partition, it is a screaming red flag that one partition's consumer is dead.

### Committed offset vs current position

There is a subtle distinction that trips people up: the *committed* offset is not the same as the consumer's *current position*. The position is where the consumer's fetch pointer currently is in memory — it has fetched up to there and may be processing those records right now. The committed offset is where it has durably *recorded* its progress, which lags the position by however many records sit between commits. If you commit every five seconds, the committed offset can be up to five seconds of records behind the position. Lag computed against the committed offset is therefore slightly pessimistic: it counts records that may already be processed but not yet committed. That is usually fine and even desirable for monitoring — committed offset is the durable, crash-safe truth, and it is what the broker exposes — but be aware that a consumer committing infrequently will show a sawtooth lag pattern that dips at each commit, and that sawtooth is an artifact of commit cadence, not a real fluctuation in work done. The mechanics of when and how often to commit are covered in [Consumer offset commit strategies and failure modes](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes).

### The same idea outside Kafka

The offset-subtraction formula is specific to log-based brokers like Kafka and Pulsar, where every record has a monotonic position and "lag" is a literal arithmetic difference. In a classic broker like RabbitMQ there are no offsets; the analog is **queue depth**, the number of messages sitting unacknowledged in a queue, reported directly by the broker as `messages_ready`. The intuition is identical — depth is work not yet done — but the measurement is a gauge the broker maintains rather than a subtraction you compute. In SQS the analog is `ApproximateNumberOfMessagesVisible`, the count of messages available for receipt. Across all three the principle holds: find the size of the backlog, watch its trend, and convert it to time. The rest of this post uses Kafka's offset framing because it is the most explicit, but everything about derivatives, time-lag, the retention cliff, and autoscaling applies with trivial translation to queue depth on any broker. For the RabbitMQ-specific architecture, see [RabbitMQ acks, confirms, durability, and quorum queues](/blog/software-development/message-queue/rabbitmq-acks-confirms-durability-quorum-queues).

## 3. Records-lag vs time-lag

Here is the single most important reframing in this post, and the one most teams have never made: **records-lag is nearly useless on its own, and time-lag is what you actually want.** Records-lag is the raw count — one million records behind. Time-lag is the wall-clock answer — how long ago was the record the consumer is currently processing produced, or equivalently, how many seconds of backlog do you have at the current consumption rate. The reason time-lag dominates is that a record count has no inherent urgency. One million records of lag is a catastrophe on a low-volume topic and a non-event on a firehose, and the only way to tell which you are looking at is to divide by the throughput.

![A two-panel comparison contrasting a records-lag of one million which looks scary but lacks context against a time-lag that divides by throughput to reveal either two seconds behind or two hours behind](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-9.webp)

Figure 9 makes the point concretely. The same lag of one million records is two seconds behind on a topic consuming five hundred thousand records a second, and two hours behind on a topic consuming one hundred forty records a second. Two seconds is fine. Two hours might be a business emergency — if that topic feeds fraud detection, you are approving fraudulent transactions for two hours. The raw count of one million is identical in both cases and tells you nothing about which situation you are in. Only the division by throughput — the conversion to time — carries the urgency. Alerting on records-lag is alerting on a number whose meaning changes by four orders of magnitude depending on context you have not included.

#### Worked example: converting records-lag to time-lag

Let us do the conversion that should be on every lag dashboard. A topic has a consumer group reporting a total lag of 2,000,000 records. The group is currently consuming at 50,000 messages per second. The time-lag is:

```
time-lag = records-lag / consumption-rate
         = 2,000,000 records / 50,000 records per second
         = 40 seconds
```

So the consumer is forty seconds behind in wall-clock terms. Now ask the operational question: is forty seconds a problem? That depends entirely on the topic's purpose. If this is a clickstream feeding a daily analytics rollup, forty seconds is invisible — nobody will ever notice. If it is the topic behind a real-time pricing engine where stale prices mean money lost on every trade, forty seconds is a five-alarm fire. The same 2,000,000-record number, the same forty seconds, lands completely differently against the SLO. This is why your alert thresholds must be in *seconds of time-lag* tied to each topic's freshness requirement, not in a one-size-fits-all record count. A pricing topic might page at five seconds of time-lag; an analytics topic might not page until thirty minutes.

### How to actually measure time-lag

There are two ways to get time-lag, and they differ in fidelity. The cheap approximation is records-lag divided by the current consumption rate, exactly as above. It is easy because both numbers are already on your dashboard, but it is an estimate: it assumes the consumption rate will hold steady while you drain, and it uses the *current* rate rather than the rate at which the lagging records were produced. The precise method is to look at the **timestamp** of the record at the consumer's committed offset and subtract it from now. Kafka records carry a timestamp; the consumer's lag in time is `now − timestamp(record at committed offset)`. This is exact — it directly measures how stale the oldest unprocessed record is — but it requires the exporter to fetch that record's timestamp, which not all tools do out of the box. Kafka's `kafka-consumer-groups` tool does not give you time-lag directly; you need Burrow or a custom exporter that reads timestamps, or you compute the approximation from rate. Most teams use the rate-based approximation for dashboards and reserve the timestamp method for the precise SLO measurement, and that is a reasonable split.

### Why the distinction is operationally load-bearing

The records-versus-time distinction is not academic pedantry; it changes what you build. If you alert on records-lag, you will set a threshold that is wrong for at least half your topics — too sensitive for the high-volume ones (constant false pages) and too lax for the low-volume ones (real problems slip under). You will also be unable to write a single alert rule that works across topics, because the right record threshold for a firehose is a thousand times the right threshold for a trickle. Time-lag normalizes all of that: a five-second freshness SLO means the same thing on every topic regardless of volume, so you can write one alert rule, parameterized per topic by its SLO, and have it be correct everywhere. Time-lag is the unit that makes your alerting portable and your SLOs comparable. Records-lag is an implementation detail you convert away from as early as possible.

### Which throughput to divide by

There is one subtlety in the time-lag conversion that bites people, and it is worth getting right: *which* throughput do you divide records-lag by? There are two candidates and they answer different questions. Divide by the **consumption rate** (how fast the consumer is draining) and you get "how long until I catch up, if production stopped right now" — the drain-time estimate. Divide by the **production rate** (how fast records were written) and you get "how old is the oldest unprocessed record" — the staleness estimate, which is what an SLO usually means. In steady state these rates are equal and it does not matter. But during an incident they diverge sharply: if production is 55,000/s and consumption is 50,000/s, dividing one million records of lag by 50,000 gives 20 seconds of drain time, while the staleness of the oldest record is better approximated against the production timeline. For SLO and retention-cliff purposes, the *staleness* interpretation is the one that matters — you care how old the data is, because that is what determines whether you breach freshness and how close you are to the cliff. The cleanest way to sidestep the ambiguity entirely is the timestamp method from earlier: read the actual timestamp of the record at the committed offset and subtract from now. That measures staleness directly with no rate division at all, which is why timestamp-based time-lag is the gold standard and rate-based time-lag is the convenient approximation. Know which one your tool reports, because confusing drain-time for staleness can make a cliff look further away than it is.

## 4. How to measure lag (tools and trade-offs)

You have several ways to get lag out of Kafka, and they trade off freshness, cost, and operational burden. Knowing which to reach for saves you from either flying blind or drowning in a custom-monitoring side project.

![A four-layer stack showing the raw offset layer at the bottom feeding an exporter layer, then a dashboard layer, then an alert layer that produces a page](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-8.webp)

Figure 8 shows the layering. At the bottom is the raw offset data — committed offsets in `__consumer_offsets` and the log-end-offset on each partition leader. Every tool above is just a way to read those two numbers, subtract them, store the result over time, and act on it. The choice of tool is a choice of how much of that pipeline you build versus buy.

### The tools, from quick to production

**`kafka-consumer-groups --describe`** is the built-in CLI and your first stop for an ad-hoc check. It prints per-partition current-offset, log-end-offset, and lag for a group, right now, on demand:

```bash
kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --describe \
  --group payments-processor
```

```
GROUP             TOPIC      PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
payments-processor payments  0          8200000         8400000         200000
payments-processor payments  1          9100050         9100120         70
payments-processor payments  2          7700000         7700000         0
```

This is perfect for incident triage — you can immediately see that partition 0 has 200,000 lag while partitions 1 and 2 are essentially caught up, which tells you the problem is isolated to one partition (a likely stalled or slow consumer for that assignment, not a group-wide capacity issue). What the CLI cannot do is store history, compute derivatives, or alert. It is a flashlight, not a smoke detector. Run it during an incident; do not build monitoring on it.

**Kafka Exporter** (the `danielqsj/kafka_exporter` Prometheus exporter) is the most common production choice. It connects to the cluster, reads consumer-group offsets and log-end-offsets on a schedule, and exposes them as Prometheus metrics like `kafka_consumergroup_lag` per group, topic, and partition. You scrape it with Prometheus, store the series, and write PromQL alerts and Grafana dashboards on top. The trade-off is that it polls on an interval (typically every fifteen to thirty seconds), so your lag resolution is that interval — fine for almost everything, but it means you will not see sub-second lag spikes. It also does not natively compute time-lag; you derive that in PromQL from rate, or you switch tools.

**Burrow** (LinkedIn's consumer-lag monitor) is the most sophisticated open option and the one that takes time-lag seriously. Rather than a raw threshold, Burrow evaluates a *sliding window* of each consumer's offset commits and classifies the consumer's status as OK, WARN, STALL, STOP, or ERR based on whether the committed offset is advancing and whether lag is growing. This is exactly the stalled-versus-slow distinction from section six, computed for you. Burrow's insight — that the *pattern* of offset movement matters more than the instantaneous lag value — is the single best idea in lag monitoring, and even if you do not run Burrow you should steal its evaluation logic for your own alerts.

**JMX metrics** expose lag from the consumer's own side: `records-lag-max` and `records-lag` per partition are emitted by the Kafka consumer client. The advantage is that these are computed by the consumer in real time, so they are fresh and free of polling lag. The disadvantage is that they only exist while the consumer is running — if the consumer is dead, there is no JMX to scrape, and a dead consumer is precisely when you most need to know its lag. JMX is great for fine-grained consumer-side dashboards but must be paired with a broker-side source (exporter or Burrow) that keeps reporting even when the consumer is gone.

**Cloud metrics** — AWS MSK, Confluent Cloud, and managed Kafka services — expose lag as a first-class metric (`MaxOffsetLag`, `EstimatedTimeLag`, `consumer_lag`) through CloudWatch or the provider's metrics API. If you are on managed Kafka, start here; it is already wired and often includes an estimated time-lag. The trade-off is granularity and cost: cloud metrics are often one-minute resolution and you pay per metric, so high-cardinality per-partition lag across many topics can get expensive.

### A comparison table

| Tool | Resolution | History | Time-lag | Works when consumer is dead | Ops burden |
|------|-----------|---------|----------|----------------------------|-----------|
| `kafka-consumer-groups` CLI | on-demand | none | no | yes | none |
| Kafka Exporter + Prometheus | 15–30s poll | yes | via rate calc | yes | low |
| Burrow | sliding window | short | yes (status) | yes | medium |
| JMX `records-lag` | real-time | via collector | no | no | low |
| Cloud (MSK/Confluent) | ~1 min | yes | often yes | yes | none |

The pragmatic stack for most teams is Kafka Exporter into Prometheus into Grafana into Alertmanager, with Burrow's evaluation logic reimplemented as PromQL rules, and the CLI in your back pocket for incident triage. That gives you history, derivatives, per-partition breakdown, and alerting, all on infrastructure you already run for everything else. We will write the actual alert rules in section five.

### The end-to-end monitoring stack

Stepping back, the production lag pipeline is a chain of five components, and it is worth seeing the whole path at once because each link transforms the signal a little.

![A grid showing the lag monitoring pipeline flowing from the consumer group that commits offsets to a Kafka exporter computing log-end-offset minus committed, into Prometheus scraping every fifteen seconds, then a PromQL query taking the derivative, an alert rule combining rate and cliff, and finally a KEDA scaler that adds or removes pods](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-5.webp)

Figure 5 traces it. The **consumer group** commits offsets to `__consumer_offsets` as it works — that is the raw truth. The **exporter** reads those committed offsets plus each partition's log-end-offset and computes the subtraction, exposing `lag = LEO − committed` as a metric. **Prometheus** scrapes that metric on a fifteen-second interval and stores it as a time series, which is what gives you history and lets you compute derivatives at all. A **PromQL query** takes the derivative of the stored lag series — `deriv()` — turning a position into a velocity. The **alert rule** combines that velocity with the retention-cliff time-lag check and decides whether to page. And finally the **KEDA scaler** reads the same lag metric and adjusts the pod count, closing the loop back to the consumer group. Notice that the exporter, Prometheus, and the query are pure read-and-transform stages — they do not touch the consumer — while only the scaler at the end acts back on the system. That separation is healthy: your observability path is read-only and cannot itself cause an incident, and the one component that *can* change behavior (the scaler) is isolated at the end where its blast radius is contained to pod count. When you debug a lag-monitoring problem, walk this chain left to right: is the exporter reading the right group, is Prometheus scraping the exporter, is the query computing the derivative over the right window, is the rule firing, is the scaler acting? Each link can fail independently, and the failure looks different at each stage — a missing exporter target shows as no data, a wrong query window shows as a derivative that is too jumpy or too smooth, a misconfigured scaler shows as pods that do not move when lag does.

The reason to build the full chain rather than stopping at a dashboard is that a dashboard requires a human to be looking. The whole value of the alert and scaler stages is that they act *without* a human in the loop at 3am. A dashboard is for understanding an incident you already know about; the alert is for learning about one you do not. Build through to the alert at minimum, and to the scaler if your traffic varies enough to justify it.

## 5. Alerting: absolute vs rate vs the retention cliff

This is where most teams go wrong, so we will be opinionated. There are three things you can alert on, and the ranking is clear: alert on the **rate of change** and the **retention cliff**; use **absolute thresholds** only as a coarse backstop, if at all.

![A before-and-after comparison showing an absolute lag-over-100k alert that pages on harmless bursts and trains on-call to ignore it, versus a rate-of-change alert that fires only on a sustained climb and means a real problem](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-4.webp)

### Why absolute thresholds are noisy

An absolute threshold — page when lag exceeds 100,000 — fails in both directions, as figure 4 shows. It is too sensitive to bursts: real traffic is spiky, and a momentary surge will push lag past any fixed line for a few seconds before consumers drain it. Each of those is a false page. After a week of 3am pages that resolved themselves before the on-call even opened a laptop, the team mutes the alert or raises the threshold so high it is useless. And it is too lax for the slow leak: a consumer losing capacity steadily will sit just under the threshold for a long time, doing real damage, and only trip the alarm when lag is already enormous and the situation already dire. A single fixed number cannot distinguish a harmless spike from a dangerous trend, because it ignores time entirely. Lag is a position; danger is in the velocity.

### Alert on the derivative

The actionable signal is the rate of change of lag. If lag is *growing steadily* over several minutes, your consumers cannot keep up — *C* has fallen below *P* — and that is always worth a human's attention regardless of the absolute value. A burst that spikes lag and then drains has a derivative that goes positive then negative and nets to zero; it never triggers a derivative alert. A real fall-behind has a sustained positive derivative; it triggers immediately, while lag is still small and you have time to act. In PromQL the rule is built on `deriv()` over a window:

```yaml
# Alert when lag is growing steadily for 10 minutes.
- alert: ConsumerLagGrowing
  expr: deriv(kafka_consumergroup_lag_sum[10m]) > 1000
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Lag on {{ $labels.consumergroup }} growing > 1000 rec/s for 10m"
    description: "Consumers are falling behind; C < P. Investigate or scale."
```

The `deriv()` function fits a least-squares line to the lag series over the window and returns its slope in records per second. A positive slope sustained for the `for` duration means a real, persistent fall-behind. Tune the slope threshold to your topic's normal jitter — a noisy topic might need `> 5000` to avoid firing on routine wobble — but the structure is the point: you are alerting on velocity, not position. This one change eliminates the vast majority of false pages and catches real problems an hour earlier than any absolute threshold would.

### Alert on the retention cliff

The derivative alert tells you consumers are falling behind. The retention-cliff alert tells you that you are about to *lose data*, and it is the alert that must never be missed. Here is the danger: Kafka retains records for a configured window (say six hours) and then deletes them, regardless of whether any consumer has read them. If your lag in *time* approaches the retention window, the oldest unprocessed records are about to be deleted out from under your consumer — permanent, unrecoverable data loss. The retention cliff is the deadline, and the alert is: time-lag is approaching the retention window. This is fundamentally a time-lag alert, which is one more reason time-lag matters: you cannot express "approaching the retention cliff" in units of records, only in units of time, because retention itself is configured in time.

```yaml
# Alert when time-lag exceeds 70% of the retention window.
# retention = 6h = 21600s; 70% = 15120s.
- alert: ApproachingRetentionCliff
  expr: |
    (kafka_consumergroup_lag_sum / on(consumergroup)
     rate(kafka_consumergroup_current_offset_sum[5m]))
    > 15120
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "{{ $labels.consumergroup }} time-lag > 70% of retention"
    description: "Records will be deleted before consumption. Imminent data loss."
```

This rule divides records-lag by the consumption rate to get time-lag in seconds and fires when it crosses seventy percent of the retention window. Seventy percent gives you a margin to react — to scale consumers, to extend retention temporarily, to do something — before records start dropping at one hundred percent. A retention-cliff alert is `severity: critical` and pages immediately; a derivative alert is `severity: warning` and can go to a slower channel during business hours, because growing lag is a problem you have time to fix while a retention cliff is a deadline you cannot move.

#### Worked example: the retention cliff deadline

Now the headline calculation. A topic has six hours of retention and is consumed at 50,000 messages per second. Lag is currently growing at 5,000 records per second — consumers are processing 50,000/s but producers are writing 55,000/s, so the deficit is 5,000/s. The question every incident commander asks: how long until data loss?

First, what does the retention window hold in records? At 50,000 records per second of consumption — which roughly tracks production in steady state — six hours of retention is:

```
retention capacity = 50,000 rec/s × 6 h × 3600 s/h
                    = 50,000 × 21,600
                    = 1,080,000,000 records (1.08 billion)
```

That is the buffer. The oldest record in the partition is up to 1.08 billion records before the newest. Data loss happens when the consumer's committed offset falls so far behind the log-end-offset that the broker deletes the oldest unread record — that is, when lag (in records) reaches the retention capacity, because the consumer's read position would then be pointing at a record that has already been deleted.

Suppose lag right now is 1,000,000 records and growing at 5,000 records per second. The gap to the cliff is:

```
gap = retention capacity − current lag
    = 1,080,000,000 − 1,000,000
    = 1,079,000,000 records

time to cliff = gap / growth rate
              = 1,079,000,000 / 5,000
              = 215,800 seconds
              ≈ 60 hours
```

So at this growth rate you have roughly sixty hours before data loss — plenty of time, but only because the retention buffer is enormous relative to the deficit. The danger sharpens if the deficit is larger or the buffer smaller. Suppose instead retention is only one hour (not six) and the deficit is 20,000 records per second on the same 50,000/s consumer:

```
retention capacity = 50,000 × 3600 = 180,000,000 records
time to cliff = (180,000,000 − 1,000,000) / 20,000
             = 179,000,000 / 20,000
             = 8,950 seconds
             ≈ 2.5 hours
```

Now you have two and a half hours, and the alert had better fire well before that. This is why the retention-cliff alert uses time-lag against a percentage of the window: it abstracts away the exact record arithmetic and gives you a single, topic-agnostic deadline signal. The lesson the math teaches is that the retention window is your safety buffer, and a short retention window plus a large deficit can put you hours — not days — from irreversible loss. Teams that run tight retention to save disk are running a smaller safety margin than they realize.

## 6. Healthy lag vs growing lag vs a stalled consumer

Lag is a number, but its *shape over time* is the diagnosis. Three shapes matter, and confusing them leads to the wrong fix. We will name them precisely and then show how to tell them apart automatically.

![A timeline showing lag starting steady around 40k records, then a deploy slowing the consumer, then lag growing at 5k per second, reaching one million and 20 seconds behind, and finally nearing the retention cliff at six hours](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-2.webp)

Figure 2 walks the progression. Most of the time lag should be **steady** — a flat, nonzero line bouncing around a small floor (here ~40,000 records) as the consumer keeps pace with producers, draining each commit's worth of records as fast as they arrive. Steady lag is healthy. It is *not* zero, and it should not be: a lag that sits at exactly zero means your consumers are idle most of the time, which means you are over-provisioned. A small steady lag means consumers are working at a comfortable load with headroom. The floor's height is set by your commit interval and batch size — commit every five seconds at 50,000/s and you will see a steady lag oscillating up to ~250,000 records purely from commit cadence, with no actual fall-behind. Learn your topic's healthy floor; it is the baseline against which everything else is judged.

### Growing lag

When lag's line tilts upward and *keeps* tilting — a sustained positive derivative — consumers cannot keep up. *C* has fallen below *P*, whether because the consumer got slower (a bad deploy, a slow downstream, GC pauses) or producers got faster (a traffic spike, a backfill, a thundering herd). Growing lag is the classic problem and the one autoscaling is designed for: it is a capacity deficit, and the fix is more capacity — scale out, scale up, or tune the consumer. The defining feature is that the committed offset *is still advancing*; the consumer is making progress, just not fast enough. You can watch the committed offset climb; it is simply climbing slower than the log-end-offset.

### The stalled consumer

The worst shape is the most deceptive: lag is *flat or even slightly growing, and the committed offset is not advancing at all.* This is a **stalled** consumer. It looks superficially like steady lag if you only watch the lag number, but the tell is that the committed offset has frozen — the consumer is not processing anything. A stall is worse than growing lag because growing lag is a consumer that is at least making progress, while a stall is a consumer doing nothing: stuck on a poison message that throws on every retry, blocked on a downstream call that never returns, deadlocked, or evicted from the group and not rejoined. Under a stall, lag will eventually grow as producers keep writing, but the early, decisive signal is the *frozen committed offset*, not the lag value. If you alert only on lag magnitude, a stall on a low-volume topic can hide for a long time, because lag grows slowly when production is slow — and meanwhile zero records are being processed.

The single most important diagnostic, then, is not lag itself but **the rate of change of the committed offset.** A healthy consumer's committed offset advances at roughly the production rate. A slow consumer's advances, just too slowly. A stalled consumer's does not advance at all. This is precisely Burrow's insight: evaluate the *consumer's offset progress*, not just the lag gap. The rule that catches a stall is:

```yaml
# Alert when the committed offset has not advanced for 5 minutes
# but lag is nonzero -- the consumer is stalled, not idle.
- alert: ConsumerStalled
  expr: |
    rate(kafka_consumergroup_current_offset_sum[5m]) == 0
    and kafka_consumergroup_lag_sum > 0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "{{ $labels.consumergroup }} committed offset frozen with lag > 0"
    description: "Consumer is stalled (not advancing), not merely slow. Restart/investigate."
```

The `and kafka_consumergroup_lag_sum > 0` clause is essential: an offset that is not advancing because there is *nothing to process* (lag is zero, the topic is quiet) is perfectly healthy and must not page. A frozen offset *with work waiting* is a stall. That conjunction — not advancing AND work waiting — is the precise signature of a stalled consumer, and it is the alert that catches the failure mode that lag-magnitude alerts miss entirely.

### A signal-summary matrix

It helps to lay out what each signal reveals so you reach for the right one.

![A matrix showing four lag signals -- absolute lag, rate of change, time-lag, and stalled consumer -- against what each tells you and its value for alerting, where rate-of-change and time-lag are high-value and stalled is critical](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-3.webp)

Figure 3 summarizes the diagnostic toolkit. Absolute lag tells you the backlog size but is noisy and scale-dependent — a poor alarm. Rate of change tells you whether you are falling behind and is the best early-warning signal. Time-lag tells you how stale you are in seconds and maps directly to your SLO. And the stalled-consumer signal — a frozen committed offset with work waiting — is the critical, worst-case detection that the other three miss. A complete lag-monitoring setup watches all four: the magnitude for context, the derivative for early warning, the time-lag for SLO and the retention cliff, and the offset-advance rate for stalls. No single one is sufficient; together they tell you not just that lag exists but which of the four causes you are facing.

## 7. Autoscaling consumers on lag

If growing lag is a capacity deficit, the obvious response is to add capacity automatically. This is what autoscaling on lag does: watch lag, and when it grows, add consumer instances; when it drains, remove them. The dominant tool in the Kubernetes world is **KEDA** (Kubernetes Event-Driven Autoscaling), which has a first-class Kafka scaler that reads consumer-group lag and drives the number of consumer pods up and down.

![A directed acyclic graph showing lag rising at 5k per second triggering KEDA to scale from four to eight pods, then a check of whether partitions are free, branching to either draining the backlog and returning to steady lag or hitting the ceiling with idle pods](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-6.webp)

Figure 6 shows the loop as a directed acyclic graph — and it is deliberately *not* a cycle, because a feedback loop drawn as a graph would have a cycle, but the operational reality routes to distinct outcomes. Lag rises, KEDA scales the deployment from four to eight pods, and then comes the decisive branch: *are there free partitions for those new pods to own?* If yes, the extra pods get partition assignments, drain the backlog, and lag settles back to its steady floor. If no — if you already have as many pods as partitions — the new pods join the group, get zero partitions, and sit idle burning money while lag keeps growing. That branch is the entire subject of section eight, and it is why autoscaling on lag is necessary but not sufficient.

### A KEDA ScaledObject for Kafka lag

Here is a real KEDA configuration that scales a consumer deployment on lag:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: payments-consumer-scaler
spec:
  scaleTargetRef:
    name: payments-consumer        # the Deployment to scale
  minReplicaCount: 2               # never below 2 (HA)
  maxReplicaCount: 12              # never above partition count (12)
  cooldownPeriod: 300              # wait 5m before scaling down
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: kafka:9092
        consumerGroup: payments-processor
        topic: payments
        lagThreshold: "50000"      # target ~50k lag per replica
        activationLagThreshold: "1000"  # scale from 0 only above 1k lag
```

The key parameter is `lagThreshold`: KEDA treats it as the *target lag per replica* and scales so that total lag divided by replica count stays near it. If total lag is 600,000 and the threshold is 50,000, KEDA wants 600,000 / 50,000 = 12 replicas. As lag drains, the desired replica count falls, and after the `cooldownPeriod` the deployment scales back down. The `maxReplicaCount` of 12 is set equal to the partition count — this is the most important line in the whole config, and section eight explains why anything above it is wasted.

### The asymmetry of scaling up versus down

Scaling up and scaling down must not be symmetric, and getting this wrong causes thrashing. Scale **up fast**: when lag is growing, you want capacity now, so a short or zero stabilization window on the up direction is right — every second you delay, lag grows. Scale **down slow**: when lag drains, do not immediately remove pods, because (a) the drain might be temporary and lag could climb again, and (b) every scale-down triggers a consumer-group rebalance, which briefly *pauses* consumption while partitions are reassigned, momentarily making lag worse. That is the purpose of the `cooldownPeriod: 300` — wait five minutes of sustained low lag before removing a pod. The rebalance cost is real: each scale event stops the world for the group for the duration of the rebalance, covered in [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing). An autoscaler that flaps up and down every thirty seconds spends more time rebalancing than consuming, and its lag will be *worse* than if it had not scaled at all. Aggressive up, patient down, and a cooldown long enough to amortize the rebalance cost — that is the stable configuration.

### Autoscaling and backpressure are the same problem from two ends

Autoscaling on lag is one half of a control system; the other half is backpressure, covered in [Backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control). Backpressure slows the *producers* when consumers fall behind; autoscaling speeds up the *consumers*. Both respond to the same signal — a growing backlog — and a robust system uses both: autoscale consumers to add capacity, and apply backpressure upstream when you have hit the autoscaling ceiling and capacity can no longer grow. The two are complementary controls on the same lag signal, and thinking of them together — one pushes the deficit down by raising *C*, the other by lowering *P* — is the right mental model for keeping a pipeline stable under load it was not provisioned for. When autoscaling runs out of room, backpressure is the only remaining lever, and that handoff happens exactly at the partition ceiling.

### The economics of scaling on lag

Autoscaling is, at bottom, an economic decision, and it pays to make the trade-off explicit. The alternative to autoscaling is static provisioning for peak: run enough consumers, all the time, to handle your busiest moment. That is simple and never falls behind, but you pay for peak capacity twenty-four hours a day even though peak might last two hours. If your peak is three times your average, static provisioning for peak means roughly three times the consumer cost of provisioning for average. Autoscaling lets you provision for average and burst to peak only when lag demands it, so your bill tracks actual load rather than worst-case load. On a deployment of meaningful size — say a fleet that costs tens of thousands of dollars a month at peak — the savings from autoscaling are real money, and that is the case where it earns its operational complexity.

But autoscaling is not free, and the costs are easy to undercount. Every scale event triggers a rebalance, and rebalances briefly pause consumption, so a fleet that scales frequently spends a fraction of its time not consuming — pure overhead that eats into the very capacity you scaled to add. There is also a *reaction-lag* cost: the autoscaler observes lag on a fifteen-to-thirty-second scrape, decides, schedules a pod, the pod starts, joins the group, triggers a rebalance, and only then begins consuming — easily a minute or two from "lag started growing" to "new capacity online." During that window lag grows unchecked, so autoscaling always runs a little behind the load it chases. This is why you keep a `minReplicaCount` above the bare minimum: starting from a warm baseline of consumers means the autoscaler only has to add a few pods to handle a spike, rather than cold-starting the whole fleet, which would put you minutes behind at the worst possible moment. The honest framing is that autoscaling trades a steady, predictable over-provisioning cost for a variable cost plus reaction lag plus rebalance churn. For variable traffic with a big peak-to-average ratio, that trade is worth it. For flat traffic, it is not — you pay the complexity and the rebalance churn for savings that never materialize, and you would be better off statically provisioning for peak plus headroom and never touching the autoscaler again.

## 8. The partition ceiling and when scaling stops helping

Here is the hard limit that catches every team that autoscales naively: **a consumer group can have at most as many actively-consuming instances as the topic has partitions.** Each partition is assigned to exactly one consumer in the group. If you have twelve partitions and twelve consumers, every consumer owns one partition and the group is fully parallel. Add a thirteenth consumer and it owns *nothing* — it joins the group, gets zero partitions, and idles. Scaling past the partition count does not add throughput; it adds idle pods that cost money and, worse, trigger a rebalance every time they join or leave. The partition count is the ceiling on useful parallelism, full stop.

This is the branch in figure 6 that leads to "ceiling hit: idle pods." When your autoscaler scales up but lag keeps growing and the new pods show zero assigned partitions, you have hit the ceiling, and *no amount of further scaling will help.* This is the single most common autoscaling failure: lag is growing, the autoscaler dutifully adds pods, lag keeps growing, the autoscaler adds more pods, and the on-call engineer watches twenty pods consume what twelve partitions can deliver, paying for eight idle replicas while the incident continues unabated. The autoscaler did exactly what it was told; the constraint is physical and the autoscaler cannot see it.

#### Worked example: hitting the partition ceiling

A topic has 12 partitions. Each consumer instance can process 5,000 records per second. The group is currently keeping up at a production rate of 50,000 records per second with 10 active consumers (each owning roughly one partition, processing 5,000/s, for 50,000/s aggregate). Now production doubles to 100,000 records per second — a Black Friday spike, a backfill, a viral event.

The group needs aggregate capacity *C* of at least 100,000 records per second to keep up. At 5,000 per consumer:

```
consumers needed = required rate / per-consumer rate
                 = 100,000 / 5,000
                 = 20 consumers
```

The autoscaler computes that it needs 20 consumers and scales the deployment to 20 pods. But the topic has only 12 partitions. The math of what actually happens:

```
useful consumers   = min(desired, partitions) = min(20, 12) = 12
idle consumers      = 20 − 12 = 8
achievable rate     = 12 × 5,000 = 60,000 rec/s
deficit             = 100,000 − 60,000 = 40,000 rec/s
```

So even after scaling to 20 pods, the group processes only 60,000 records per second — capped by the 12 partitions — against 100,000 produced. Lag grows at 40,000 records per second, *with eight idle pods burning money the whole time.* Scaling did nothing past the twelfth pod. The only fixes are: raise per-consumer throughput (each consumer needs to process 100,000 / 12 ≈ 8,334 records per second instead of 5,000 — tune the consumer per [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling)), or *increase the partition count* so more pods can do useful work. Repartitioning is the structural fix, and it is not free — covered in [Partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) — because adding partitions to a keyed topic changes the key-to-partition mapping and can break ordering for in-flight keys. This worked example is the entire reason the partition count is a *capacity-planning* decision made up front, not a runtime knob: you must provision enough partitions for your peak, because at peak the partition count is the hard ceiling on how much capacity autoscaling can ever give you.

### How to set the partition count for autoscaling headroom

The rule that falls out of the math: set your partition count to your peak required parallelism, with headroom. If your peak production rate is *P_peak* and a single well-tuned consumer does *C_one*, you need at least *P_peak / C_one* partitions to be able to scale to that load, and you should provision more so the autoscaler has room and so a single slow consumer does not become the bottleneck. Over-partitioning has costs — more open file handles, more rebalance churn, more metadata — so it is a trade-off, not a free lunch; the tuning math is in [Partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning). But the principle for autoscaling is simple: the autoscaler's `maxReplicaCount` should equal the partition count, and the partition count should be sized for peak. Setting `maxReplicaCount` above the partition count is strictly worse than setting it equal — it lets the autoscaler create idle pods that do nothing but trigger rebalances. The ceiling is real; bake it into your configuration.

### The taxonomy of causes, and which the autoscaler can fix

Pulling sections six and eight together: autoscaling only fixes *one* of the four lag causes.

![A taxonomy tree rooting at rising lag and branching into slow consumer, traffic spike, stalled consumer, and rebalance storm, with leaves showing that tuning fixes a slow consumer, autoscaling fixes a spike, and a restart fixes a stall](/imgs/blogs/consumer-lag-monitoring-and-autoscaling-7.webp)

Figure 7 is the decision tree you run during an incident. A **traffic spike** (production jumped, *P* rose) is the one autoscaling is built for — add pods, drain, done, as long as you are under the partition ceiling. A **slow consumer** (*C* fell — bad deploy, slow downstream, GC) is fixed by *tuning*, not scaling; adding more slow consumers just multiplies the slowness, and if you were already at the partition ceiling, scaling does literally nothing. A **stall** (frozen offset) is fixed by a *restart or fixing the poison message* — autoscaling makes it worse, because the new pods inherit the same poison message and stall too, and meanwhile the rebalances from scaling churn the group. A **rebalance storm** (consumers thrashing in and out, perhaps *caused* by an over-aggressive autoscaler) is fixed by *stabilizing the group*, which often means scaling *down* and lengthening the cooldown. The critical operational lesson: do not reflexively autoscale on lag. Diagnose the cause first — read the offset-advance rate, check for partition skew, look at the deploy timeline — and only autoscale when the cause is genuinely a capacity deficit from rising load. Autoscaling the other three causes ranges from useless to actively harmful.

## 9. A lag SLO and runbook

Put it all together into something operable: a lag SLO and the runbook that hangs off it. An SLO turns "watch lag" into a precise, measurable commitment, and the runbook turns an alert into a sequence of actions instead of a panic.

### Defining the SLO

State the SLO in **time-lag**, because time-lag is the unit that maps to user-visible freshness, as section three argued. A good lag SLO looks like:

> 99% of the time, time-lag on the `payments` topic stays below 30 seconds; it never exceeds 70% of the retention window (4.2 hours of a 6-hour retention).

Two clauses, two purposes. The first clause (30-second time-lag, 99% of the time) is the *freshness* SLO — it is what your downstream consumers and their users actually experience, and it is the one you report to stakeholders. The second clause (never above 70% of retention) is the *data-loss* guardrail — it is binary and non-negotiable, because crossing it means losing records. The first is a soft target with an error budget; the second is a hard line. Both are expressed in time, both are derived from the topic's purpose and retention config, and both are computable from the same lag and rate metrics you already collect.

### The error budget and what it buys

The 99% freshness target gives you an error budget: one percent of the time, time-lag may exceed 30 seconds. Over a 30-day month that is about 7.2 hours of permitted staleness. That budget is *spendable* — it is the slack that lets you do a rolling deploy that briefly pauses consumers, run a backfill that temporarily widens lag, or absorb a traffic spike before the autoscaler catches up, all without breaching the SLO. When you have burned the budget, you freeze risky changes until it recovers. This is standard SLO discipline applied to lag, and it works exactly as it does for request latency: the budget converts a binary pass/fail into a managed resource that aligns reliability work with actual user impact. A team with a healthy lag error budget can move fast; a team that has burned it knows to stop and fix the consumer before shipping anything else.

### The runbook

The runbook is what the on-call engineer runs when an alert fires. Here is the structure, keyed to which alert fired:

**If `ConsumerLagGrowing` fired** (sustained positive derivative): the cause is a capacity deficit. First, check *which* of the four causes (figure 7). Pull `kafka-consumer-groups --describe` and look at per-partition lag. If lag is spread evenly across partitions and the committed offsets are all advancing, it is a genuine capacity deficit — a slow consumer or a traffic spike. Check the deploy timeline: did a deploy land just before lag started growing? If yes, it is a slow consumer; roll back or fix the regression. If no deploy and production rate jumped, it is a traffic spike; confirm the autoscaler is scaling and that you are under the partition ceiling. If you are at the ceiling, the only moves are to tune per-consumer throughput or to add partitions.

**If `ConsumerStalled` fired** (frozen committed offset with lag > 0): one partition or the whole group has stopped processing. Find the stuck partition from the per-partition describe — the one whose current-offset is not moving. Check the consumer logs for that partition's owner: a poison message throwing on every retry, a downstream call hanging, a deadlock. The fix is to unblock or skip the poison message (see [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment)) and restart the stuck consumer. *Do not autoscale* — new pods inherit the poison message.

**If `ApproachingRetentionCliff` fired** (time-lag > 70% of retention): this is the emergency. You are about to lose data. Buy time first: *temporarily increase the retention window* on the topic (`kafka-configs --alter --add-config retention.ms=...`) to push the cliff further out — this is the single fastest action and it stops the bleeding. Then attack the lag: scale consumers if under the partition ceiling, tune if at it. The order matters: extend retention first (seconds, stops data loss), then fix capacity (minutes). Never start with the slow fix while records are dropping.

```bash
# Emergency: push the retention cliff out from 6h to 24h to stop data loss.
kafka-configs.sh --bootstrap-server kafka:9092 \
  --entity-type topics --entity-name payments \
  --alter --add-config retention.ms=86400000
```

### Forward pointer: when lag spikes

The runbook above is the steady-state version. When lag *spikes* suddenly — not the slow leak but the sharp vertical climb — there is a denser, more tactical field guide to follow, with the full incident-response sequence for diagnosing and containing a lag spike under time pressure. That is the subject of the companion post, [debugging consumer lag spikes: a field guide](/blog/software-development/message-queue/debugging-consumer-lag-spikes-field-guide), which picks up exactly where this runbook leaves off and walks a real spike from page to resolution.

## Case studies and war stories

Abstract rules land harder when attached to real incidents. Here are four, each teaching one of this post's lessons.

### The alert that cried wolf

A payments team set a single absolute lag alert: page when total lag exceeds 50,000. The topic carried bursty traffic — settlement batches that dumped 80,000 records in a few seconds every hour on the hour. Each batch pushed lag past 50,000 for about ten seconds while consumers drained it, firing the page. Every hour, around the clock, the on-call got paged for an incident that resolved itself before they could open a laptop. Within a week the team had muted the alert. Two weeks after that, a bad deploy slowed the consumer by 30%, lag began a slow steady climb, and *the muted alert never fired* — they discovered the problem four hours later when a downstream team complained about stale data, by which point time-lag was over an hour. The lesson: an absolute threshold on bursty traffic trains your team to ignore the alarm, so it is silent precisely when a real, gradual problem appears. They replaced it with a `deriv() > 1000 for 10m` rule and the false pages stopped immediately while the next slow regression was caught in eleven minutes.

### The stall that hid behind healthy lag

A logistics company processed shipment events on a low-volume topic — maybe 50 records per second. One consumer hit a malformed record that threw a deserialization exception on every poll, stalling that partition completely. Because the topic was low-volume, lag grew at only 50 records per second; it took *hours* to reach any absolute threshold anyone had set, and the lag number looked unremarkable the whole time. Meanwhile zero shipment events were being processed for that partition's keys — an entire region's shipments silently frozen. The team only noticed when a customer escalated about a shipment stuck "in transit" for six hours. The fix that prevented recurrence was the `ConsumerStalled` alert: frozen committed offset with lag greater than zero, firing in five minutes regardless of how slowly lag was growing. The lesson: on low-volume topics, lag magnitude is a terrible stall detector because lag grows too slowly to trip a threshold; you must watch the committed offset's advance rate, not the lag value.

### Autoscaling into the partition ceiling

A media company autoscaled its video-transcoding consumers on lag with KEDA, `maxReplicaCount` set to 50, on a topic with 16 partitions. A viral event tripled the upload rate. Lag climbed, KEDA scaled to 50 pods, lag *kept climbing*, KEDA stayed at 50, and the on-call watched 34 idle pods consume nothing while lag grew for two hours and the cloud bill for the consumer deployment tripled. The 16 partitions could deliver only what 16 consumers could process; the other 34 pods owned zero partitions and idled. The post-incident fix had two parts: set `maxReplicaCount` to 16 (equal to the partition count, so the autoscaler stops creating idle pods), and raise the partition count to 48 so the topic could actually parallelize to its peak load. The lesson, the hardest one in this post: scaling past the partition count is not just useless, it is actively expensive and obscures the real problem. The partition count is the ceiling, and your autoscaler must know it.

### The retention cliff that ate a day of events

A fintech ran a 4-hour retention on an audit topic to save disk. A consumer bug stalled processing overnight. By the time anyone noticed in the morning, time-lag had crossed the 4-hour retention window, and the broker had deleted the oldest unread records — permanently. They lost roughly an hour of audit events, which for a regulated fintech was a compliance incident, not just an engineering one. There had been no retention-cliff alert; the only lag alert was an absolute threshold that the slow overnight growth had not yet tripped. The fix was twofold: a `severity: critical` retention-cliff alert at 70% of the window, and a longer retention (24 hours) on the audit topic specifically, because the cost of a day of disk was trivially less than the cost of a compliance incident. The lesson: short retention is a small disk saving and a large risk multiplier, and the retention-cliff alert — expressed in time-lag, the only unit that can express it — is the one alert you must never be without on a topic where data loss matters.

## When to reach for this (and when not to)

Lag monitoring is not optional; every message system needs it. But the *sophistication* of your lag setup should match the stakes.

**Always do the minimum**: a Kafka Exporter (or cloud lag metric) into a dashboard, a per-partition lag view, and a derivative-based growing-lag alert. This is cheap, runs on infrastructure you already have, and catches the common case. Even a hobby project benefits from a single `deriv() > threshold` alert; it is two lines of PromQL and saves you from discovering a stuck consumer days later.

**Add time-lag and a retention-cliff alert** when data loss has real consequences — payments, audit logs, anything regulated, anything where a deleted unread record is a business incident rather than a missing analytics row. The retention-cliff alert requires time-lag, which requires either Burrow, a timestamp-aware exporter, or a rate-based PromQL approximation. It is more setup, but on a topic where loss matters it is non-negotiable.

**Add a stalled-consumer alert** (frozen offset with lag > 0) when you have low-volume topics or topics where a single stuck partition has outsized business impact. On a firehose, a stall trips your growing-lag alert quickly anyway; on a trickle, only the offset-advance alert catches it in time. If any of your topics carry low-volume-but-high-importance events, build this alert.

**Add autoscaling** when your traffic is genuinely variable — real peaks and troughs — and the cost of provisioning for peak at all times is significant. Autoscaling on lag with KEDA pays off when there is a meaningful gap between average and peak load. If your traffic is flat, skip autoscaling; statically provision for the load plus headroom and save yourself the rebalance churn. And *never* autoscale without setting `maxReplicaCount` to the partition count and provisioning partitions for peak — autoscaling without respecting the ceiling is worse than not autoscaling at all.

**When to deliberately keep it simple**: if your topic is low-stakes (analytics, non-critical telemetry), flat-traffic, and generously retained, a single growing-lag alert is genuinely enough. Do not build Burrow, time-lag dashboards, and KEDA for a topic where a few minutes of staleness costs nothing. Match the instrument to the stakes; over-monitoring a low-stakes topic is wasted effort that dilutes attention from the topics that matter.

## Key takeaways

- **Lag is the integral of the capacity deficit**: its derivative is exactly producer rate minus consumer rate, so watching lag is watching whether *N×C ≥ P* holds in real time. It is the one metric that catches a slow consumer when every per-component metric looks healthy.
- **Lag equals log-end-offset minus committed-offset, per partition.** Always keep the per-partition breakdown; the summed total hides a single stuck partition that the per-partition view screams about.
- **Time-lag beats records-lag.** One million records is two seconds or two hours depending on throughput; only the seconds carry urgency. Express SLOs and alerts in seconds of time-lag, not record counts, so one rule works across topics of any volume.
- **Alert on the derivative, not the absolute value.** A `deriv()` alert catches a real fall-behind while lag is still small and ignores harmless bursts that drain on their own. Absolute thresholds page on bursts and stay silent on slow leaks — the worst of both.
- **The retention cliff is the deadline that loses data.** Alert at 70% of the retention window, expressed in time-lag, `severity: critical`. In an emergency, extend retention first (seconds) and fix capacity second (minutes).
- **A stalled consumer is worse than a growing one.** The signature is a frozen committed offset with lag greater than zero — the consumer is doing nothing, not merely going slowly. Alert on the offset-advance rate, not the lag magnitude, to catch it on low-volume topics.
- **Autoscaling fixes exactly one of four lag causes** — a traffic spike. Slow consumers need tuning, stalls need a restart, rebalance storms need stabilizing. Diagnose before you scale; autoscaling a stall makes it worse.
- **The partition count is the hard ceiling on useful consumers.** A group can have at most one active consumer per partition; everything beyond that idles and burns money. Set `maxReplicaCount` equal to the partition count and provision partitions for peak load.
- **Scale up fast, scale down slow.** Every scale event triggers a rebalance that briefly pauses consumption; a flapping autoscaler rebalances more than it consumes. Use a long cooldown on scale-down to amortize the rebalance cost.

## Further reading

- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — the offsets this post subtracts and the rebalance cost every scale event pays.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — how to raise per-consumer throughput *C* before reaching for more consumers, and the partition-ceiling math in depth.
- [Backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) — the other half of the control system: slow producers when autoscaling runs out of room at the ceiling.
- [debugging consumer lag spikes: a field guide](/blog/software-development/message-queue/debugging-consumer-lag-spikes-field-guide) — the tactical incident-response sequence for a sudden lag spike, picking up where this runbook ends.
- [Consumer offset commit strategies and failure modes](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes) — why committed offset lags the position and how commit cadence shapes the lag sawtooth.
- [Partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) — sizing the partition count for peak so the autoscaling ceiling is high enough.
- [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — the most common cause of a stalled consumer and how to contain it.
- [KEDA documentation: Kafka scaler](https://keda.sh/docs/scalers/apache-kafka/) — the official reference for autoscaling consumers on lag in Kubernetes.
- [Burrow](https://github.com/linkedin/Burrow) — LinkedIn's consumer-lag monitor and the source of the sliding-window evaluation logic this post recommends.
