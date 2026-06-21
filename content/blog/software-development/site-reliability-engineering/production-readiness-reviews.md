---
title: "Production Readiness Reviews: The Gate That Stops the 3am Page Before It Starts"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn launching a service from a leap of faith into a gated, checkable bar — a production readiness review that proves the thing can be operated, observed, and recovered at 3am by someone who never wrote a line of it."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "production-readiness-review",
    "prr",
    "launch-readiness",
    "on-call",
    "slo",
    "runbooks",
    "operational-readiness",
    "error-budget",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/production-readiness-reviews-1.png"
---

There is a specific kind of weekend that every on-call engineer remembers. A team shipped a new service on a Friday afternoon. The demo had been flawless — the feature worked, the product manager clapped, somebody ordered pizza. By Saturday at 2am the service was paging. Not because it was *broken*, exactly, but because nobody could tell whether it was broken. There was no dashboard. The single alert that did exist fired on CPU usage, which was high because the service was *busy*, which is what you want a service to be. There was no runbook, so the person holding the pager — who had never seen this code — spent forty minutes reading source on their phone trying to work out whether a queue depth of 8,000 was normal. It was not normal. The downstream database had quietly fallen over an hour earlier and the service had been retrying into the void ever since, each retry making it worse. By Sunday night the team had been paged forty times, had restarted the service eleven times, and had learned nothing they couldn't have learned in a one-hour review *before* the launch.

That weekend is entirely preventable, and the thing that prevents it is a **production readiness review** — a PRR. A PRR is a structured review that a service must pass before it reaches production traffic, before it goes generally available, and before SRE accepts the pager for it. It is the gate. It asks one deceptively simple question and refuses to let the service through until the answer is yes: *can this be operated, observed, and recovered at 3am by someone who did not build it?* Notice what that question is **not**. It is not "does it work in the demo." A demo proves the happy path executes once, while a stranger awake at the worst hour is staring at a screen. Readiness is the difference between those two things, and a PRR makes the difference explicit, written down, and gateable.

![A layered stack diagram showing how a production readiness review verifies every reliability layer was built before launch](/imgs/blogs/production-readiness-reviews-1.png)

This post is the checklist that ties the whole series together. Every prior track taught one layer of running a reliable system — defining SLIs and SLOs, building observability, setting burn-rate alerts, writing runbooks, planning capacity, designing for failure, testing your restores. The PRR is the gate that verifies *every one of those layers was actually done for this specific service* before it is allowed to carry real users and a real pager. It sits at the top of the reliability stack and asks, layer by layer, "is this true here?" By the end of this post you will be able to: run a PRR that catches a launch disaster two weeks early instead of two days late; write a PRR checklist where every item is a yes-or-no gate, not a vibe; state a hard "ready for on-call" bar that protects your rotation; right-size the review so a tiny internal tool gets a thirty-minute checklist and the payments path gets a deep multi-week one; and negotiate the exceptions that keep a PRR a useful gate instead of a bureaucratic moat teams route around. Let us build it.

## 1. What a PRR actually is, and why it is a gate

A production readiness review is a structured, written evaluation of whether a service is operationally ready for production. The concept comes most famously from Google's SRE practice, where it formalized the moment a service "graduates" — the moment it transitions from being a thing the development team runs by hand to a thing that SRE will help operate and carry a pager for. Google's SREs would not accept on-call responsibility for a service that had not passed a PRR, and that single policy is what made the PRR a *gate* rather than a polite suggestion.

The distinction between a gate and advice is the whole point, so let me be sharp about it. **Advice** is a document you email to a team that they read, nod at, and ignore under deadline pressure. **A gate** is a checkpoint with the authority to say *no*. A real PRR can block a launch. It can refuse the pager. It can tell a team, "you are not going to production on Friday, because you have no SLO, your only alert is on CPU, you have zero runbooks, and you have never run this under load — close those four gaps and come back." That refusal is not obstruction. It is the system working. The cost of those four gaps does not disappear when you skip the review; it just moves to the worst possible place, which is the first weekend the service is live, paid for by whoever is holding the pager.

Why does making it a gate work, when advice does not? Because of incentives and timing. Before launch, the gaps are cheap to close — writing an SLO is an afternoon, writing three runbooks is a day, wiring burn-rate alerts is a few hours. After launch, the same gaps cost a paged weekend, an angry incident review, a damaged on-call rotation, and sometimes a real outage with real users. A gate forces the cheap work to happen during the cheap window. It converts a probabilistic future disaster into a definite present checklist. That is a trade every team should take, and most teams will not take it voluntarily — which is exactly why it has to be a gate.

The contrast between a gate and a document is worth making explicit, because organizations slide from one to the other without noticing:

| Property | PRR as advice | PRR as a gate |
| --- | --- | --- |
| Can it say "no"? | No — it is read and ignored under deadline | Yes — it can block GA and refuse the pager |
| When do gaps get closed? | After launch, on a paged weekend | Before launch, in the cheap window |
| Who owns the gaps? | Diffuse — "we should fix that someday" | A named owner with a deadline |
| What happens under schedule pressure? | The advice is skipped first | The gate holds, or grants a tracked exception |
| Net effect on reliability | A false sense of having "reviewed" it | The cheap work actually happens before users arrive |

The right-hand column is the only one that changes outcomes, and the difference between the columns is entirely about *authority and timing*, not about the quality of the checklist. A brilliant checklist with no teeth is the left column.

### Who runs it and when

A PRR is run *jointly* by the SRE (or the platform/reliability function, or simply a senior engineer playing that role on a smaller team) and the team that owns and built the service. This is not SRE inspecting the developers like a customs officer. It is two parties with a shared interest sitting down with a checklist: the owning team knows how the service is supposed to behave, and the reviewer knows what "operable at 3am" actually requires because they have been the person awake at 3am. The output is co-owned. If the review finds a gap, it is the owning team's gap to close, but the reviewer's job to verify it is closed.

The *when* has three natural checkpoints, and you can use one, two, or all three depending on your organization:

- **Before first production traffic.** The earliest gate. Even if the service is launching to 1% of users behind a flag, you want the observability and the rollback path to exist before any real user touches it.
- **Before general availability (GA).** The service has been soaking in production at low exposure and is about to go to 100% of users or get publicly announced. This is the heaviest review for most services because GA is the point of maximum blast radius.
- **Before SRE takes the pager.** The "graduation" review. Even a service that has been in production for a while, run by its dev team, must pass this bar before a dedicated SRE rotation will accept responsibility for it. You do not hand SRE a service with no SLOs, no runbooks, and a noisy alert set and call it their problem.

On a smaller team without a dedicated SRE function, all three collapse into "the senior engineer who carries the pager reviews the service before we flip it on." The roles are smaller; the gate is the same.

How the review actually runs matters as much as the checklist. The most effective format I have used is asynchronous-first: the owning team fills in the scorecard (section 6 has the template) *before* any meeting, attaching evidence — links to the dashboard, the alert rule, the load-test report, the restore-drill log. The reviewer reads it cold and flags the gates that look thin. Then a short live session — thirty minutes for a standard review — walks only the flagged gates, because the green ones with solid evidence do not need discussing. This inverts the failure mode where a PRR is a two-hour meeting in which someone reads a checklist aloud and everyone says "yep." The async-first format puts the burden of *evidence* on the owning team and the burden of *judgment* on the reviewer, which is the right split. A reviewer who walks in having already read the evidence asks sharp questions; a reviewer hearing it for the first time in the room rubber-stamps.

## 2. The PRR checklist, grouped by category

A PRR is only as good as its checklist, and a good checklist has one property above all others: **every item is a yes-or-no gate**, not a discussion topic. "Is the service observable?" is a discussion topic. "Are the four golden signals — latency, traffic, errors, saturation — emitted and visible on a dashboard?" is a gate. You can answer the second one with a yes or a no, and if the answer is no, you know exactly what to fix. Write your checklist in the second style, always.

![A matrix mapping each production readiness checklist category to its pass bar and the reliability track that earns it](/imgs/blogs/production-readiness-reviews-2.png)

Here is the checklist, grouped by category. Each category ties back to a track in this series — because, again, the PRR is the gate that verifies every prior track was done *for this service*. I will give the gates first and then the reasoning behind a few that people get wrong.

**Observability** (the eyes — see the *monitor-the-user-not-just-the-server* and *dashboards-that-tell-the-truth* posts):

- [ ] The service's SLIs are defined and instrumented (a request-success ratio, a latency percentile — not just "it emits some metrics").
- [ ] The four golden signals are emitted: latency, traffic, errors, saturation.
- [ ] A dashboard exists that shows the SLIs and golden signals, and it is linked from the service's runbook directory.
- [ ] Structured logs are emitted with a request/trace ID, and they are queryable.
- [ ] Distributed tracing is wired so a slow request can be followed across service boundaries.

**SLOs and error budget** (the target — see *choosing-slis-that-reflect-user-pain* and *the-error-budget-the-currency-of-reliability*):

- [ ] An SLO is set *and agreed* with the service owners and, where relevant, the product owner. Agreement is part of the gate — an SLO nobody signed up to is not an SLO.
- [ ] The error budget that the SLO implies is computed and understood (a 99.9% SLO is 43.2 minutes of budget per 30-day month).
- [ ] Alerting is SLO/burn-rate-based, not cause-based — you page on the budget burning, not on every CPU spike.

**Alerting and on-call** (the trigger — see *alerting-that-doesnt-cry-wolf* and *designing-a-humane-on-call*):

- [ ] Every paging alert is *actionable*: a human woken by it can do something about it.
- [ ] Every paging alert links to a runbook.
- [ ] The service is in an on-call rotation with named owners — not "whoever notices."
- [ ] Escalation is defined: if the primary does not ack in N minutes, who gets paged next?

**Runbooks** (the response — see *runbooks-that-survive-3am*):

- [ ] The top failure modes (typically the three to five most likely or most damaging) each have a runbook.
- [ ] Each runbook is *3am-grade*: a stranger can follow it half-asleep, with exact commands, not "investigate the issue."

**Capacity** (the headroom — see *capacity-planning-and-forecasting*):

- [ ] The service has been load-tested and its saturation limit is known (the point where latency or errors degrade).
- [ ] Capacity is planned for expected peak plus a margin.
- [ ] Autoscaling is configured with sane minimums, maximums, and a target utilization.

**Resilience** (the failure design — see *timeouts-retries-and-backoff-done-right*, *circuit-breakers-bulkheads-and-load-shedding*, and *graceful-degradation-and-fallbacks*):

- [ ] Every outbound call has a timeout. (Defaults of "infinite" are the single most common readiness failure.)
- [ ] Retries use backoff *and jitter*, with a bounded retry budget — not naive immediate retries.
- [ ] Circuit breakers or load-shedding protect against a failing dependency.
- [ ] Dependencies are mapped, and for each one the question "what happens when it is down?" has a designed answer (fail open, fail closed, serve stale, degrade).

**Data** (the durability — see *backups-that-actually-restore* and *disaster-recovery-and-business-continuity*):

- [ ] Backups exist and the *restore has been tested*. A backup you have never restored is a hope, not a backup.
- [ ] RPO and RTO are stated (how much data you can lose, how fast you must be back).
- [ ] If the service is in scope for disaster recovery, the DR plan exists and has been exercised.

**Deploy** (the change safety — see *deploying-safely-progressive-delivery*):

- [ ] Deploys are progressive (canary or blue-green), not a single all-at-once cutover.
- [ ] Automated rollback is wired, triggered by the SLIs (roll back when error rate or latency regresses).
- [ ] There is a tested, documented manual rollback path as a fallback.

**Security and compliance basics**:

- [ ] Secrets are in a secret manager, not in code or environment files committed to the repo.
- [ ] Authn/authz is enforced on every endpoint that needs it.
- [ ] Data handling meets the relevant compliance bar (PII handling, retention, audit logging) if applicable.

That is a comprehensive checklist. It is also, deliberately, *too much* for a small internal tool — which is the subject of section 6. But first, let us be precise about the two categories people most often get wrong: alerting and resilience.

### Why "alert on the budget, not the cause"

A cause-based alert fires on a *possible cause* of a problem: CPU above 80%, memory above 90%, disk filling, a queue growing. The trouble is that most causes are not problems. CPU at 85% on a busy, well-tuned service is *correct*. You will get paged for it, find nothing wrong, and learn to ignore the alert — and then the one time CPU at 85% really does precede a meltdown, you sleep through it because the alert cried wolf forty times first. A symptom-based, SLO-based alert fires on *user pain*: the success ratio is dropping, latency is up, the error budget is burning. Those are always worth waking for, because by definition a user is being hurt. The PRR gate "alerting is SLO/burn-rate-based" exists to keep new services from being born with a noisy, ignorable, cause-based alert set. We will quantify exactly how much noise this removes in the worked example below.

The evidence the reviewer should demand for the SLO and alerting gates is the actual rule, not a promise. Here is what a reviewer wants to *see* — a Prometheus recording rule that computes the SLI as a ratio of good events over total events, and a multi-window, multi-burn-rate alerting rule built on it:

```yaml
# Recording rule: the SLI as a good-over-total ratio (the gate wants THIS, not "we have metrics")
groups:
  - name: tax_calculator_slo
    rules:
      - record: job:slo_errors:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="tax-calc",code=~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="tax-calc"}[5m]))
      - record: job:slo_errors:ratio_rate1h
        expr: |
          sum(rate(http_requests_total{job="tax-calc",code=~"5.."}[1h]))
          /
          sum(rate(http_requests_total{job="tax-calc"}[1h]))
```

```yaml
# Alerting rule: fast burn (page now) AND slow burn (page on a slow leak)
groups:
  - name: tax_calculator_burn_alerts
    rules:
      - alert: TaxCalcSLOFastBurn
        # 14x burn over 5m AND 1h confirms a genuine fast burn, not a blip
        expr: |
          job:slo_errors:ratio_rate5m > (14 * 0.001)
          and
          job:slo_errors:ratio_rate1h > (14 * 0.001)
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Tax-calc burning the 30-day budget in ~2 days"
          runbook: "https://runbooks/tax-calc/slo-fast-burn"
```

A reviewer reads that, confirms the SLI is a real ratio (not a proxy like CPU), confirms the alert thresholds are derived from the SLO ($14 \times 0.001$ is the 14x burn rate against the 99.9% budget), and confirms the `runbook` annotation points at a real page. Three gates ticked, with evidence on screen. Notice the gate is not "do you have alerts" — it is "show me the rule and let me read the threshold."

### Why every outbound call needs a timeout

This is the gate I have seen blocked more than any other, because it is so easy to forget and so catastrophic to skip. A call with no timeout will, when its dependency hangs, hold a thread or connection *forever*. Under load, every request hits the hung dependency, every request grabs a thread and never lets go, and within seconds your entire thread pool is exhausted — the service is down not because *it* failed but because it could not give up on a dependency that did. This is the classic cascading failure, and the system-design series covers the architecture of it in *cascading-failures-circuit-breakers-and-bulkheads*. The PRR gate is the operational backstop: before this service touches production, prove that every outbound call gives up in bounded time.

Here is the resilience layer expressed as runnable artifacts. A timeout and a bounded, jittered retry in a typical config:

```yaml
# Envoy-style route config: a hard timeout plus a bounded retry policy
route:
  cluster: payments-backend
  timeout: 2s                 # never wait forever
  retry_policy:
    retry_on: "5xx,connect-failure,reset"
    num_retries: 2            # bounded — not "keep trying"
    per_try_timeout: 800ms
    retry_back_off:
      base_interval: 100ms    # exponential backoff
      max_interval: 1s        # capped so it cannot grow unbounded
    # jitter is added automatically by the base/max backoff window
```

And a circuit breaker so a sick dependency stops getting hammered:

```yaml
# Envoy circuit-breaker thresholds: shed load to a struggling dependency
circuit_breakers:
  thresholds:
    - priority: DEFAULT
      max_connections: 256
      max_pending_requests: 64    # fail fast instead of queueing forever
      max_requests: 256
      max_retries: 16             # cap concurrent retries — kills retry storms
```

A reviewer reads these, checks that the timeout is finite, the retries are bounded, and the breaker exists, and ticks three gates. That is the practice: not "are you resilient?" but "show me the config, and let me read the numbers."

### Why an untested restore is not a backup

The data gates contain the one item I have seen go wrong with the worst consequences: a backup that has never been restored. Backups are easy to *configure* and quietly easy to break. The backup job runs nightly, reports success, and fills a bucket with files — and then the night you actually need it, you discover the files are encrypted with a key you rotated, or they are a logical dump of a schema that no longer matches, or the restore takes nine hours when your RTO is one, or the backup has silently been backing up an empty replica for three months. None of that shows up until the restore, which is why the gate is not "do backups exist" but "*has the restore been run, end to end, and timed*?" The evidence a reviewer demands is the log of an actual drill:

```bash
# Restore drill the PRR data gate requires — run it, time it, paste the output
# 1. Pull the most recent backup into an isolated restore target
aws s3 cp s3://backups/tax-calc/latest.dump /restore/latest.dump
# 2. Restore into a scratch database and TIME it (this is your real RTO)
time pg_restore --clean --no-owner -d tax_calc_restore /restore/latest.dump
# 3. Verify the data is actually usable, not just present
psql -d tax_calc_restore -c "SELECT count(*), max(updated_at) FROM tax_rates;"
# 4. Record the measured RTO and the freshness gap (your real RPO)
#    e.g. restore took 11m -> RTO met; newest row 6m old -> RPO ~6m
```

The numbers that fall out of this drill are the *real* RPO and RTO — not the ones in the design doc, the ones the system actually achieves. The PRR gate "RPO and RTO are stated" should mean *measured*, not *aspired*. A team that runs this drill and discovers the restore takes nine hours has learned something priceless on a quiet afternoon instead of during a real data-loss incident with users watching.

## 3. Launch readiness is not code-complete

The single most useful mental shift a PRR forces is the separation of *code-complete* from *launch-ready*. They feel like the same milestone — the feature works, so we can ship it — and they are not even close.

![A before-and-after figure contrasting a service that only works in the demo with one that a stranger can operate at 3am](/imgs/blogs/production-readiness-reviews-8.png)

Consider what "works in the demo" actually proves. It proves the happy path executes once, in a controlled environment, with the author present and watching. It proves nothing about what happens when the database is slow, when a dependency returns 503s, when traffic is 10x the demo, when a deploy goes bad, or when the author is asleep and a stranger is holding the pager. Launch readiness is precisely the set of properties that the demo cannot show you, and a PRR is the structured way to check them.

The litmus test I use, and the one I recommend you adopt as the spine of your whole review, is this: **can someone who did not build this service operate, observe, and recover it at 3am?** Break that into its three verbs and you have the heart of the checklist:

- *Operate*: can a stranger find the dashboard, read the SLIs, and tell whether the service is healthy? (Observability.)
- *Observe*: when something is wrong, can they tell *what* is wrong from the signals — not by reading source? (Golden signals, tracing, logs.)
- *Recover*: can they take a documented action that fixes it — a rollback, a runbook step, a failover — without paging the author? (Runbooks, rollback, failover.)

If the honest answer to any of those is "well, they'd have to ask the author," the service is not ready, no matter how well the demo went. The author is a single point of failure, and SREs do not accept single points of failure into a rotation. The PRR's job is to make that single point of failure visible *before* it costs a 3am phone call to someone who is on vacation.

There is a cultural payoff here too. When teams internalize that launch readiness is a separate, gated milestone, they start building for it from the beginning — they instrument the SLIs while writing the feature, they write the runbook as they discover the failure modes, they wire the rollback before they wire the deploy. The PRR stops being a thing done to them at the end and becomes a checklist they are working toward from the start. That is the difference between a PRR that teams resent and one they thank you for.

## 4. The PRR as a gate, and the negotiated exception

A gate that can only say "yes" or "no" is too brittle to survive in a real organization. There will be a service that is 90% ready, has one documented gap, and has a launch date driven by a regulatory deadline or a marketing commitment that genuinely cannot move. A pure binary gate forces a bad choice: block a launch the business needs, or wave through a gap and pretend it is fine. The release valve that keeps the gate both real *and* usable is the **negotiated exception with a deadline**.

![A flow diagram of the PRR decision showing the three outcomes of pass, time-boxed exception, and hard block until gaps close](/imgs/blogs/production-readiness-reviews-4.png)

A PRR ends in one of three outcomes:

1. **Pass.** Every gate is green. GA is approved.
2. **Time-boxed exception.** A *minor* gap is documented, the risk is explicitly accepted by a named owner, and a deadline to close it is committed. The launch proceeds with the gap, and the gap is tracked like any other reliability work.
3. **Block.** A *critical* gap exists. GA does not happen until the gap is closed and the service comes back through review.

The skill is in the line between minor and critical, and there is a principle for it: **a gap is critical if it would prevent recovery or hide an outage; a gap is minor if it only makes operations slightly more annoying.** No SLO and cause-based-only alerts? Critical — you cannot even *tell* if you are having an outage. No runbook for the top failure mode? Critical — you cannot recover at 3am. Untested restore on a stateful service? Critical — you could lose data permanently. A missing dashboard panel for a secondary metric, or a runbook for the fourth-most-likely failure mode? Minor — annoying, not dangerous. Exception with a two-week deadline, ship it.

The exception format matters enormously, because an exception without a deadline is just a permanent gap dressed up as a plan. Write it like this, and track it where you track everything else:

```yaml
# PRR exception — tracked as a real work item, not a verbal "we'll get to it"
exception:
  id: PRR-EXC-2026-0142
  service: checkout-tax-calculator
  gate_failed: "Runbook exists for top-3 failure modes"
  detail: "Runbooks for db-down and dependency-timeout exist;
           the cache-stampede runbook is not yet written."
  severity: minor          # does not block recovery of the top-2 failures
  risk_accepted_by: "jordan.lee (service owner)"
  approved_by: "sre-oncall-lead"
  deadline: "2026-07-04"   # hard date — the gap closes or GA is revisited
  tracking: JIRA-8821
```

Notice the discipline: a named human accepts the risk, a named reviewer approves the exception, and there is a hard deadline with a tracking ticket. When the deadline arrives, the exception is closed or the launch's GA status is genuinely reconsidered. An exception that quietly expires unmet is how a PRR rots into theater. Hold the line on deadlines, and the exception mechanism stays honest.

A note on power dynamics, because this is where PRRs live or die politically. The gate has teeth only if the organization backs it. If a VP can override a "block" with a phone call every time, the gate is decoration. The way to keep it real without being a tyrant is to make the *block* rare and the *reasoning* undeniable: block only on critical gaps, document exactly why, state exactly what closing it requires, and offer the exception path for everything else. When you block a launch, you should be able to say, "if this ships as-is, here is the specific outage I expect and the specific reason no one will be able to fix it at 3am." That is a hard argument to override, and it should be.

## 5. "Ready for on-call" — the bar SRE requires before taking the pager

There is a special, sharper version of the PRR that deserves its own treatment: the moment SRE accepts the pager for a service. This is the strongest form of the gate, because the person enforcing it is the person who will be woken up. The acceptance bar is therefore non-negotiable in a way that even a GA review sometimes is not — you can ship to users with a documented gap, but you cannot ask another human to be responsible for a service that cannot be operated.

![A matrix of the ready-for-on-call acceptance criteria showing the required state and the condition that gets a service rejected](/imgs/blogs/production-readiness-reviews-7.png)

The "ready for on-call" acceptance criteria are a tight subset of the full PRR — the items without which the pager is just a noise machine attached to a stranger:

- **SLO defined and agreed.** Without an SLO there is no definition of "down," so there is nothing to alert on and no way to know if the page was justified. *Reject if: no SLO, no target.*
- **Alerts are actionable and SLO/burn-rate-based.** The on-call must be able to do something about every page, and the pages must correlate with user pain. *Reject if: alerts fire on CPU and other causes, crying wolf.*
- **Every paging alert links a runbook.** A page that says "errors elevated" with no next step is cruelty to the on-call. *Reject if: any page has no documented next action.*
- **The service is on a rotation with escalation.** Not "the author gets a Slack message." A real rotation with a defined escalation path so a missed ack reaches a backup. *Reject if: a lone hero with no backup.*

The reason this bar is non-negotiable is an empathy argument dressed as an engineering argument. Accepting the pager means committing to wake up for this service, and a person can only sustainably do that if the pages are real, rare, and recoverable. Hand someone a service with no SLO, alerts on every CPU twitch, no runbooks, and no backup, and you have not given them a responsibility — you have given them a guaranteed burnout and a service that will, in the end, be operated badly. SRE refusing the pager for such a service is not bureaucracy. It is the only thing protecting the rotation, and a healthy rotation is what protects the users. Refuse the pager, hand back the gaps, and let the team close them. Then accept it, and mean it.

It helps to make concrete what separates a service that is genuinely ready for on-call from one that merely *runs*. The difference is almost always whether the on-call signal points at user pain and whether the on-call response is documented:

| Readiness signal | A service that is NOT ready for on-call | A service that IS ready |
| --- | --- | --- |
| Definition of "down" | None — no SLO, just "it feels slow" | An agreed SLO with an error budget |
| What pages you | CPU, memory, disk — causes, not pain | The budget burning — a real user symptom |
| Page volume | 35+ a week, most non-actionable | A handful a week, all actionable |
| Response to a page | Read source, guess, restart | Open the linked runbook, follow exact steps |
| If the primary misses the ack | Nothing — the page is lost | Escalates to a named backup in minutes |
| MTTR | Tens of minutes, source-reading | Single-digit-to-low-double-digit minutes |

The left column is a service that will burn out whoever carries it; the right column is a service a human can sustainably own. The "ready for on-call" gate exists to make sure every service crossing into a rotation is in the right column.

A useful artifact here is the explicit acceptance sign-off — the thing both parties put their name on:

```yaml
# Ready-for-on-call acceptance record
acceptance:
  service: payments-gateway
  reviewed: "2026-06-18"
  slo: { target: "99.95% success", window: "28d", agreed_by: "product + eng" }
  alerts:
    paging: 3            # all burn-rate based, all link a runbook
    cause_based_paging: 0  # gate: must be zero
  runbooks: { failure_modes_covered: 5, format: "3am-grade, exact commands" }
  rotation: { schedule: "follow-the-sun", escalation: "2-tier, 10-min ack" }
  accepted_by: "sre-team-checkout"
  pager_effective: "2026-06-25"
```

When SRE signs that record, the service has truly graduated. Everything before it was the dev team running their own thing; everything after is a shared, gated, operable responsibility.

## 6. Right-sizing the PRR so teams do not route around it

Here is how a good PRR program dies: it becomes a three-month bureaucratic gauntlet that every team learns to dread, so they start launching things "temporarily" without one, "just this once," until the un-reviewed services outnumber the reviewed ones and the whole program is a fiction. A PRR that is too heavy for the service it gates does not make systems more reliable — it makes the review *avoided*, which makes systems *less* reliable. The cure is to right-size the review to the risk.

![A decision tree for right-sizing a production readiness review by classifying the service risk into low, medium, and high](/imgs/blogs/production-readiness-reviews-5.png)

The principle is proportionality: **the depth of the review should scale with the blast radius of the service.** Classify the service first, then pick the review:

- **Low risk — internal, small blast radius.** An internal dashboard, a low-traffic admin tool, a batch job whose failure delays a report by an hour. *Light PRR: a 30-minute checklist.* You still want the basics — does it have an owner, a dashboard, an alert that pages someone, and a way to restart it? — but you do not need a load test or a multi-region DR plan for a tool five people use.
- **Medium risk — user-facing but recoverable.** A feature in the product whose failure annoys users but does not lose money or data and can be rolled back. *Standard PRR: a one-week review.* The full checklist, but proportionate: real SLOs, burn-rate alerts, runbooks for the top failures, a load test to known peak, progressive deploy.
- **High risk — money or data, irreversible.** Payments, anything that writes data you cannot reconstruct, anything in a regulatory path. *Deep PRR: multi-week, including a DR exercise, a chaos game day, and a security review.* Here you earn the heavy process, because the cost of getting it wrong is a financial loss or permanent data loss, not an annoyed user.

The contrast is the point. The same organization should run a thirty-minute checklist for the internal dashboard and a multi-week review for the payments service, and *both are correct*. Applying the payments review to the dashboard is the bureaucracy that kills the program; applying the dashboard checklist to payments is the negligence that causes the disaster. Right-sizing is how you avoid both.

A practical way to encode this is a one-line risk classifier at the top of every PRR, so the depth is chosen explicitly rather than by who is in the room:

```python
def prr_tier(service):
    """Pick the PRR depth from the service's blast radius."""
    if service.handles_money or service.data_loss_irreversible:
        return "DEEP"      # multi-week, DR exercise, chaos, security review
    if service.user_facing and service.recoverable:
        return "STANDARD"  # 1-week, full checklist, load test, canary
    return "LIGHT"         # 30-min checklist: owner, dashboard, alert, restart

# Examples
# payments-gateway      -> DEEP
# product-search        -> STANDARD
# internal-metrics-tool -> LIGHT
```

The classifier is not magic — it is a forcing function that makes a team *say out loud* how risky the service is, which is itself a useful conversation. A team that wants to claim "LIGHT" for something that handles money has to argue it past the reviewer, and that argument rarely survives contact.

### The PRR scorecard as an artifact

The review itself wants a single artifact that records the outcome: a scorecard with every gate, its status, and the evidence. This is what you fill in during the review and what you keep afterward as the record of "this service was ready on this date." Keeping it as code (YAML in the service's own repo) means it lives next to the thing it describes and shows up in code review when the service changes — which is how continuous readiness gets enforced in practice.

```yaml
# prr-scorecard.yaml — lives in the service repo, reviewed on every major change
service: checkout-tax-calculator
tier: STANDARD            # from prr_tier(): user-facing + recoverable
reviewed: "2026-06-18"
reviewers: ["sre-checkout", "team-payments"]
gates:
  observability:
    slis_instrumented:    { status: pass, evidence: "dashboard/tax-calc" }
    golden_signals:       { status: pass, evidence: "4/4 emitted" }
    logs_traces:          { status: pass, evidence: "trace-id on all logs" }
  slo_budget:
    slo_agreed:           { status: pass, evidence: "99.9%/300ms, signed by PM" }
    alerts_burn_rate:     { status: pass, evidence: "multi-window rule, no CPU" }
  alerting_oncall:
    actionable_paging:    { status: pass, evidence: "3 pages, all link runbooks" }
    rotation_escalation:  { status: pass, evidence: "2-tier, 10-min ack" }
  runbooks:
    top_failures_covered: { status: pass, evidence: "3 of top-3, 3am-grade" }
  capacity:
    load_tested:          { status: pass, evidence: "tested 2x peak; sat at 1.6x" }
    autoscaling:          { status: pass, evidence: "min 4 / max 40 / 70% target" }
  resilience:
    timeouts_breakers:    { status: pass, evidence: "2s timeout, breaker config" }
    deps_mapped:          { status: pass, evidence: "3 deps, fallbacks designed" }
  data:
    restore_tested:       { status: pass, evidence: "drill 2026-06-15, RTO 11m" }
  deploy:
    progressive_rollback: { status: pass, evidence: "Argo canary, auto-rollback" }
  security:
    secrets_authz:        { status: pass, evidence: "vault + authz on all routes" }
outcome: PASS             # PASS | EXCEPTION | BLOCK
exceptions: []            # populated when outcome is EXCEPTION
ga_approved: "2026-06-25"
```

The value of writing it down is twofold. First, it forces a status on every gate — there is no "we sort of looked at it." Second, it becomes the diff baseline for continuous readiness: when the service changes its datastore six months later, the `restore_tested` gate's status reverts to `unknown` in the pull request, and the reviewer knows exactly what to re-check.

## 7. Worked examples: the disaster caught, and the right-sized review

Two scenarios make the abstract concrete. The first shows the PRR earning its keep by catching a launch disaster early. The second shows it scaling correctly so the process does not become a tax on small things.

#### Worked example: the PRR that caught a launch disaster

A team is two weeks from the GA of a new `checkout-tax-calculator` service — a customer-facing service in the checkout path, so a STANDARD-to-DEEP tier. They file the PRR confident it is a formality. The review finds four critical gaps:

1. **No SLO.** There is no agreed target, so there is no definition of "the service is having a bad time." *Critical.*
2. **Alerts on CPU only.** The single alert pages when CPU exceeds 80%. The service is CPU-heavy by design, so this alert will fire constantly during normal peak traffic and never during an actual outage of the kind that matters. *Critical.*
3. **Zero runbooks.** If it breaks at 3am, the on-call reads source. *Critical.*
4. **Never load-tested.** Nobody knows the saturation point. The team *thinks* it handles peak, but "thinks" is not a number. *Critical.*

A team without a PRR ships this on Friday and lives the weekend from the opening of this post: forty pages, eleven restarts, forty minutes of reading source on a phone, no learning. With the PRR, GA is **blocked**, and the team spends the two weeks closing the gaps:

- They set an SLO of **99.9% of tax calculations succeed within 300ms**, agreed with the product owner. That implies an error budget of **43.2 minutes per 30-day month** — let me show the arithmetic, because the agreement should be on a number, not a feeling. A 30-day month is $30 \times 24 \times 60 = 43{,}200$ minutes. A 99.9% SLO permits $0.1\%$ of that to be bad: $43{,}200 \times 0.001 = 43.2$ minutes. So the budget is 43.2 minutes of "calculations failing or too slow" per month. Now everyone knows what they are protecting.
- They replace the CPU alert with a **multi-window, multi-burn-rate** alert on the SLO. Here is the burn-rate math the alert is built on. *Burn rate* is how fast you are spending the budget relative to spending it evenly: $\text{burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}$. With a 99.9% SLO, $1 - \text{SLO} = 0.001$. If the service is currently failing 1.4% of requests, the burn rate is $\frac{0.014}{0.001} = 14\times$. A 14x burn rate exhausts the entire 30-day budget in $\frac{30 \text{ days}}{14} \approx 2.1$ days — so a 14x burn sustained is an emergency worth a page. The alert fires fast on a fast burn and slowly on a slow burn, which is exactly the symptom-based behavior the PRR gate demands.
- They write **three 3am-grade runbooks** for the top failure modes: the tax-data dependency is down (serve last-good rates, degrade gracefully), the database is slow (shed load, check connection pool), and a deploy regressed (auto-rollback should have fired; manual rollback command included).
- They **load-test to 2x expected peak** and discover the saturation point is at 1.6x peak — *below* the 2x they assumed. They add autoscaling and a connection-pool limit so the service sheds load gracefully instead of falling over. This single finding is the one that would have caused the worst of the weekend, and the load test caught it on a Tuesday afternoon instead.

![A timeline of a blocked launch showing the gaps being closed over two weeks until GA is approved with only two pages](/imgs/blogs/production-readiness-reviews-6.png)

GA happens two weeks later than the original Friday. It is a **non-event**: two pages total in the first week, both actionable, both resolved from the runbook in under fifteen minutes, zero restarts, zero source-reading-at-3am. The measured before→after is stark. Without the PRR: ~40 pages in the first weekend, MTTR of 40+ minutes (reading source), eleven restarts, and a service whose saturation point was discovered *in production under real load*. With the PRR: 2 pages in the first week, MTTR ~12 minutes (runbook-driven), zero restarts, saturation point known and engineered around. The two weeks of "delay" bought the team a launch they did not have to live through.

#### Worked example: a right-sized PRR for a low-risk tool

The same organization is launching an `internal-metrics-tool` the same month — a small internal dashboard, used by about a dozen engineers, whose worst-case failure is that those engineers cannot see a chart for an hour. Running the `checkout-tax-calculator`'s deep review on this would be absurd: a multi-week review, a DR exercise, and a chaos game day for a tool whose blast radius is a dozen mildly inconvenienced colleagues. That is exactly the bureaucracy that makes teams route around the PRR.

So it gets a **LIGHT PRR — a 30-minute checklist**:

- Does it have a named owner? *Yes — the platform team.*
- Is there a dashboard or at least a health check showing it is up? *Yes — a `/healthz` endpoint and one Grafana panel.*
- Is there an alert that pages someone if it is fully down for, say, 15 minutes? *Yes — a single low-urgency alert to the team channel, not the pager. A dozen engineers losing a dashboard for an hour does not warrant a 3am phone call.*
- Is there a way to restart or roll it back? *Yes — a one-line `kubectl rollout undo` documented in the README.*

Thirty minutes, four questions, done. The tool launches. Notice that it does *not* have a formal SLO, a multi-burn-rate alert, three 3am runbooks, or a load test — and that is correct, because the risk does not justify the cost. The process scaled with the blast radius. The payments-grade rigor went to the payments-grade service, and the dashboard got dashboard-grade rigor. That is a PRR program that survives, because teams trust it to be proportionate.

The contrast between these two reviews — same organization, same month, 30 minutes versus multiple weeks — is the single most important thing to get right about PRRs after "make it a real gate." A gate that is always heavy gets avoided. A gate that scales with risk gets used.

#### Worked example: a graduation handoff that the on-call bar rejected

A third scenario shows the "ready for on-call" gate doing its specific job. A `notifications-service` has been running in production for eight months, operated by the team that built it. That team is overloaded and wants the central SRE rotation to take the pager. They request the graduation review expecting a hand-wave, because "it's been fine in production."

The review against the ready-for-on-call bar finds the service is *running* but not *operable by a stranger*. Concretely: the on-call data tells the story. Over the prior month the service paged **35 times**, of which the team estimates **30 were non-actionable** — alerts on a queue-depth gauge that spikes harmlessly during a nightly batch. There is an SLO written on a wiki page, but it was never agreed with the product owner and no alert is derived from it. There are no runbooks; the original author resolves every page by Slack. There is no escalation; pages go to one person's phone.

SRE declines the pager and hands back four items. Watch the arithmetic on the alert noise, because it is the whole case in numbers. The current state is 35 pages a month, ~30 non-actionable — a *signal-to-noise ratio* of roughly $\frac{5}{35} \approx 14\%$. The fix is to delete the queue-depth alert and replace it with a burn-rate alert on a real SLO. The team agrees an SLO of **99.95% of notifications delivered**, derives a multi-window burn-rate alert, and writes three runbooks. The next month: **4 pages, all actionable** — a signal-to-noise ratio of $100\%$, and an absolute volume cut from 35 to 4, an **89% reduction**. Now SRE accepts the pager, because the rotation is being handed a service that pages rarely, pages truthfully, and tells the on-call exactly what to do. The graduation gate did not obstruct the handoff — it made the handoff *safe to accept*. A rotation that absorbed the un-fixed version would have inherited 30 false pages a month and quietly started ignoring all of them, which is how a real outage eventually sleeps through the night.

## 8. The PRR as the series capstone-in-miniature, and continuous readiness

Step back and look at what the checklist actually is. Each category is a track in this series. The PRR is not a new body of knowledge — it is the **verification layer** that asks, for one specific service, "did you actually do all the things the rest of this series taught?"

- The *define* layer (SLIs, SLOs, error budgets) is checked by the Observability and SLO gates.
- The *measure* layer (metrics, logs, traces, dashboards) is checked by the Observability gates.
- The *budget and alert* layer (burn-rate alerting) is checked by the Alerting gates.
- The *respond* layer (on-call, runbooks, incident response) is checked by the Alerting/On-call and Runbooks gates.
- The *engineer-for-failure* layer (timeouts, retries, breakers, capacity, graceful degradation) is checked by the Resilience and Capacity gates.
- The *recover* layer (backups, restore drills, DR) is checked by the Data gates.
- The *change-safely* layer (progressive delivery, rollback) is checked by the Deploy gates.

This is why the PRR is the natural place to talk about the whole series at once. Every other post answers "how do I build this layer well?" The PRR answers "how do I *verify*, gate-by-gate, that this layer was built for this service before it ships?" If you have read the rest of the series, the PRR checklist should read like a table of contents — and that is intentional.

There is a useful way to think about the relationship: the rest of the series is the *curriculum*, and the PRR is the *exam*. You can attend every lecture on observability, SLOs, alerting, runbooks, capacity, resilience, and recovery — and still ship a service that has none of it, because knowing how to do a thing and having actually done it for a specific service are different states of the world. The exam closes that gap. It does not teach anything new; it refuses to let a service graduate until it can demonstrate, with evidence, that the curriculum was applied. An organization that teaches reliability brilliantly but never gates on it will still ship un-operable services, because under deadline pressure the un-gated good practice is always the first thing cut. The gate is what converts knowledge into consistent practice.

But a single review at launch is not enough, because services change. The version you reviewed at GA is not the version running six months later — it has new dependencies, new endpoints, more traffic, a different architecture. **Readiness is continuous, not a one-time stamp.** The principle of *continuous readiness* says: re-review on a major change, not only at the original launch. Concretely:

- **Re-PRR on a major architecture change** — a new datastore, a new critical dependency, a sharding change, a move to a new region.
- **Re-PRR on a significant traffic change** — the service that was reviewed at 1,000 requests per second and is now doing 50,000 has a different saturation point, a different capacity plan, and possibly different failure modes. The old load test is a lie now.
- **Re-PRR on an ownership change** — when a service changes teams, the new owners must clear the "ready for on-call" bar before they carry the pager, because runbooks and tribal knowledge do not transfer automatically.
- **Periodic lightweight re-review** — even with no major change, a once-a-year quick pass over the high-risk services catches the slow rot: the dashboard that broke, the runbook that references a command that no longer exists, the alert threshold that drifted out of relevance.

The continuous version does not have to be heavy. For most services it is a 30-minute "is everything in the checklist still true?" pass, triggered by a change or a calendar. The heavy review is reserved for the genuinely significant changes. The goal is to make sure the green checkmarks from launch day are still green today, because a service that *was* ready a year ago and has quietly drifted is exactly the service that pages you on a Saturday with a runbook that no longer works.

The mechanism that makes continuous readiness real rather than aspirational is tying it to the change that triggers it. If the scorecard lives in the service repo (as in the section-6 template), you can wire a lightweight check: a pull request that touches the datastore config flips the `restore_tested` gate to `unknown` and requires a reviewer to re-confirm it before merge. A pull request that adds a new outbound dependency flips `deps_mapped` and `timeouts_breakers` to `unknown`. This turns continuous readiness from "someone remembers to re-review" — which never happens — into "the change itself surfaces the gates it invalidated." It is the same trick that makes a good test suite valuable: the safety check rides along with the change instead of depending on human memory. You will not catch everything this way, but you will catch the big architectural drifts, which are the ones that turn a once-ready service into a 3am surprise.

## 9. Stress-testing the PRR: what if the review itself fails?

A good engineer does not just describe the happy path of their own process — they stress-test it. So let me turn the PRR on itself and ask the uncomfortable questions.

**What if the PRR is just a rubber stamp?** This is the most common failure mode. The review happens, everyone says "looks good," nothing is actually checked, and the gate provides false confidence — which is worse than no gate, because people *trust* a rubber stamp. The defense is the yes/no gate format from section 2. A reviewer cannot rubber-stamp "is the restore tested?" if the gate requires them to *see the output of an actual restore drill*. Make the evidence concrete — show me the dashboard, show me the alert rule, show me the load-test report, show me the restore log — and the rubber stamp becomes impossible. A gate is only as real as the evidence it demands.

**What if a critical incident reveals a gap the PRR missed?** It will happen — no checklist is complete, and reality finds gaps. This is not a failure of the PRR concept; it is feedback for the checklist. The blameless postmortem (covered in *the-blameless-postmortem*) should ask, "should the PRR have caught this?" and if the answer is yes, the checklist gets a new gate. The PRR checklist is a living document that *learns from every incident*. The cascading-failure outage that no PRR gate would have caught becomes, the next quarter, a new "every outbound call has a timeout, demonstrated" gate. The review gets smarter over time precisely because incidents feed back into it.

**What if the team genuinely cannot hit the deadline on an exception?** Then the exception is escalated, not silently extended. The named risk-acceptor and the reviewer have a conversation: is the gap now blocking enough to warrant pulling the feature, throttling its traffic, or accepting a longer-but-still-bounded extension with a re-acceptance of the risk? The one thing that must not happen is the deadline passing unremarked. A tracking ticket with a hard due date and an owner is what prevents the silent extension. If exceptions routinely blow their deadlines, that is itself a signal — either the deadlines are unrealistic or the gates are miscalibrated, and the program needs tuning.

**What if two teams disagree on whether a gap is critical or minor?** The owning team will almost always argue "minor" (they want to ship); the reviewer should err toward "critical" on anything that touches recovery or outage-detection. The tiebreaker is the litmus test: *does this gap prevent a stranger from recovering the service at 3am, or does it only make their night slightly more annoying?* If it prevents recovery, it is critical, full stop, and the gate holds. If it is annoyance, it is an exception. Putting the disagreement on that single axis resolves most of them quickly.

**What if the service is already in production and was never reviewed?** This is the legacy-service problem, and most organizations have a backlog of un-reviewed services running in prod. You do not block a live service that is already serving users — that would cause the outage you are trying to prevent. Instead you run a *retroactive PRR*: review it as it stands, find the gaps, and file them as exceptions with deadlines, prioritized by risk. The highest-risk un-reviewed services get reviewed first. Over a quarter or two the backlog of un-reviewed prod services shrinks to zero, and from then on the gate is enforced at launch.

**What if the PRR slows everything down so much that the org's velocity craters?** This is the legitimate fear behind every "do we really need a process?" objection, and the honest answer is that a *badly-sized* PRR absolutely will crater velocity. The defense is the entire premise of section 6: the review's weight must track the service's risk. If your light path is genuinely light — a thirty-minute checklist for the dozens of low-risk services that are the bulk of any fleet — then the heavy review lands only on the few services that genuinely warrant it, and the aggregate drag on velocity is small. Measure it: if teams complain the PRR is slow, look at how many *light* reviews are being run as *standard* or *deep* ones. Almost always the fix is not "weaken the gate" but "stop over-classifying low-risk services." A PRR program that has cratered velocity is a PRR program that forgot to right-size, not proof that gating is wrong.

**What if the reviewer is the bottleneck — only one person can run a deep review?** This is a real scaling problem as the fleet grows. The answer is to make the checklist and the scorecard good enough that *most* of the review is self-service: the owning team fills in the scorecard with evidence, automated checks confirm the mechanical gates (does every paging alert have a `runbook` annotation? does every outbound call have a configured timeout?), and the human reviewer's time is spent only on judgment calls — is this SLO actually agreed, is this runbook actually 3am-grade, is this risk classification honest. The more of the checklist you can verify with a linter, the less the human reviewer is the bottleneck, and the more their scarce judgment goes to the gates that genuinely need it.

## War story: the Google PRR model and a Friday-ship cascade

The PRR as a formal, gating institution is most associated with **Google's SRE practice**, documented in the Google SRE Book and SRE Workbook. The core idea Google formalized is the one this whole post is built on: SRE will not accept on-call responsibility for a service that has not passed a production readiness review. The PRR is the contract between the developers who build a service and the SREs who help run it — it defines what "operable" means and refuses the pager until that bar is met. Google's reviews are deep for high-risk services and lighter for low-risk ones, and the principle of an *error budget* — which the rest of this series treats as the currency of reliability — sits at the heart of the SLO gate. The reason this model has been so widely copied is that it solved the exact problem in this post's opening: it stopped services from being thrown over the wall to operations in an un-operable state. (The specifics of any one company's process vary; the model and its reasoning are what generalize, and that is what I have presented here.)

The contrasting war story is the un-reviewed Friday ship, and you can find a version of it in almost every engineering org's history because the pattern is so common it is nearly a law. A service launches without a readiness review. It has no SLO, so nobody can tell whether it is healthy. Its only alert is cause-based — CPU, or memory, or disk — so it either cries wolf or stays silent during the real failure. It has no runbook, so the on-call reads source at 3am. It was never load-tested, so its saturation point is discovered in production under real traffic. And it has no timeout on a critical dependency, so when that dependency hiccups, the service exhausts its thread pool and falls over — a cascading failure that takes down more than the original problem.

Trace that cascade in slow motion, because the mechanism is what the PRR gate is designed to stop. The dependency — say, a downstream rates API — starts responding slowly, taking eight seconds instead of eighty milliseconds. Each incoming request to our service makes a call to that API and, with no timeout, waits the full eight seconds holding a worker thread. The service has a pool of, say, two hundred threads. At even modest traffic, two hundred concurrent requests is reached in seconds, every thread is parked waiting on the slow dependency, and the service can no longer accept *new* requests — including health checks. The load balancer marks it unhealthy and pulls it, which concentrates traffic on the remaining replicas, which fill their thread pools faster, and the whole fleet topples one replica at a time. The original problem was one slow dependency; the outage is a total service collapse. A single PRR gate — "every outbound call has a finite timeout, demonstrated in config" — breaks this chain at step one: with a two-second timeout the threads are released, the service sheds the slow calls, returns degraded responses, and stays up. That one unchecked box is the difference between a degraded service and a dead fleet. Every one of those is a PRR gate. Every one of those is a few hours of cheap pre-launch work that, skipped, becomes a paged weekend. The system-design series dissects the architecture of that cascade in *cascading-failures-circuit-breakers-and-bulkheads*; the PRR is the operational gate that would have demanded the timeout config before the service ever shipped. The two posts are the two ends of the same lesson — design for failure, then *verify you did* before launch.

![A before-and-after figure contrasting a Friday ship with no PRR against a launch gated by a production readiness review](/imgs/blogs/production-readiness-reviews-3.png)

The measured difference between those two launches is the entire argument for the PRR in one comparison. The un-reviewed Friday ship: ~40 pages over a weekend, MTTR measured in tens of minutes of source-reading, repeated restarts, a saturation point found the hard way. The gated launch: a handful of actionable pages, MTTR in single-digit-to-low-double-digit minutes from a runbook, zero restarts, a saturation point known in advance. The work was the same work. The only difference is *when* it was done — in a one-week review, or on a weekend that the on-call engineer will remember for years.

## How to reach for this (and when not to)

A PRR is one of the highest-leverage processes in reliability engineering, but like every practice in this series it has a cost, and the cost is real: it is time, it is a gate that can slow a launch, and run badly it becomes bureaucracy that teams resent and route around. Here is when to reach for it and when to keep it light or skip it.

**Reach for a full PRR when:**

- A service is about to take *real production traffic* for the first time, especially customer-facing traffic.
- A service is about to go GA or be publicly announced — the point of maximum blast radius.
- SRE or any dedicated rotation is about to accept the pager. This is the one case where the bar is non-negotiable: never accept on-call for a service that cannot clear the "ready for on-call" criteria.
- A service handles money, writes irreversible data, or sits in a regulatory path. Earn the heavy review here.
- A service undergoes a major architecture or traffic change, or changes owning teams. Continuous readiness means re-reviewing on significant change.

**Keep it light or skip the heavy version when:**

- The service is a low-risk internal tool whose worst-case failure is a minor inconvenience to a handful of people. A 30-minute checklist is right; a multi-week review is bureaucracy. Do not make a tiny internal dashboard pass a payments-grade review — that is precisely how the program gets abandoned.
- The change is genuinely minor — a copy tweak, a config nudge that does not touch dependencies, capacity, or failure modes. Re-running a full PRR for a trivial change trains people to treat the gate as noise.
- You would be blocking a *live* service that is already serving users. Never cause an outage in the name of preventing one — run a *retroactive* review and file the gaps as deadlined exceptions instead of pulling the service.

And a warning on over-process: a PRR that takes three months and requires sign-off from six committees is not a reliability practice, it is a moat, and teams will dig under it. The measure of a healthy PRR program is not how thorough the heaviest review is — it is whether teams *use* it, which means it has to be proportionate, fast for low-risk things, and trusted to say "yes" quickly when a service is genuinely ready. Make the light path genuinely light. Reserve the heavy path for the services that earn it. The goal is operable services, not paperwork.

## Key takeaways

1. **A PRR is a gate, not advice.** It can block a launch and refuse the pager until the gaps are closed. The cost of those gaps does not vanish when you skip the review — it moves to the first weekend on call, paid by the on-call engineer.
2. **Launch-ready is not code-complete.** The demo proves the happy path runs once. Readiness proves a stranger can operate, observe, and recover the service at 3am without the author awake. Those are different milestones; the PRR checks the second.
3. **Every checklist item is a yes-or-no gate, backed by evidence.** Not "is it observable?" but "are the four golden signals emitted and on a dashboard — show me." A gate is only as real as the evidence it demands; concrete evidence is what kills the rubber stamp.
4. **The checklist is the whole series, verified for one service.** Observability, SLOs and error budget, alerting, runbooks, capacity, resilience, data, deploy, security — each gate maps to a track. The PRR is the verification layer over everything else.
5. **"Ready for on-call" is a hard, non-negotiable bar.** Never hand a rotation a service with no SLO, cause-based alerts, no runbooks, and no escalation. Refusing that pager protects the rotation, and a healthy rotation protects the users.
6. **Right-size the review to the risk.** A 30-minute checklist for an internal tool, a multi-week review for payments — both correct, in the same organization. A gate that is always heavy gets avoided; a gate that scales with risk gets used.
7. **Exceptions need a named owner and a hard deadline.** A documented minor gap can ship with a time-boxed exception and a tracking ticket. An exception with no deadline is a permanent gap pretending to be a plan; hold the line on deadlines or the gate rots into theater.
8. **Block only on critical gaps, and make the reasoning undeniable.** A gap is critical if it prevents recovery or hides an outage; minor if it only makes operations annoying. When you block, name the specific outage you expect and the specific reason no one could fix it at 3am.
9. **Readiness is continuous.** Re-review on a major architecture or traffic change, on an ownership handoff, and periodically for high-risk services. The green checkmarks from launch day must still be green today.
10. **The checklist learns from incidents.** When a postmortem finds a gap the PRR missed, add a gate. A living checklist that absorbs every outage gets smarter over time and is the closest thing reliability engineering has to institutional memory.

## Further reading

- *[Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset)* — the intro map for this series and the define→measure→budget→respond→learn→engineer loop the PRR verifies.
- *[The reliability maturity model](/blog/software-development/site-reliability-engineering/the-reliability-maturity-model)* — where a PRR program fits as an organization matures from reactive firefighting to gated, continuous readiness (sibling Track G post).
- *[Building an SRE culture and team](/blog/software-development/site-reliability-engineering/building-an-sre-culture-and-team)* — the people side of the gate: who runs the review, how the dev/SRE contract works, and how to keep the gate from becoming a moat (sibling Track G post).
- *[Choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain)* — the Observability and SLO gates depend on having SLIs that actually track what hurts users.
- *[Alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf)* — the reasoning behind the "SLO/burn-rate alerts, not cause-based" gate.
- *[Runbooks that survive 3am](/blog/software-development/site-reliability-engineering/runbooks-that-survive-3am)* — what a "3am-grade runbook" gate actually requires.
- *[Capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting)* — the load-test and saturation-limit gates.
- *[Deploying safely: progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery)* — the progressive-deploy and automated-rollback gates.
- *[Backups that actually restore](/blog/software-development/site-reliability-engineering/backups-that-actually-restore)* — why "the restore has been tested" is a gate and an untested backup is not a backup.
- *Google SRE Book and SRE Workbook* (Google) — the canonical treatment of production readiness reviews, the launch checklist, and the error-budget model.
- The system-design series' *cascading-failures-circuit-breakers-and-bulkheads* — the architecture behind the resilience gates the PRR enforces operationally.
