---
title: "Chaos Engineering: Breaking on Purpose"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Your redundancy, failover, retries, breakers, and fallbacks are a hypothesis until you break the system on purpose and watch — this is how to run that experiment safely."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "chaos-engineering",
    "fault-injection",
    "game-days",
    "resilience",
    "blast-radius",
    "observability",
    "incident-response",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/chaos-engineering-breaking-on-purpose-1.png"
---

At 03:11 on a Tuesday, the primary database for a payments service failed over to its replica. The runbook said this took about thirty seconds. The architecture diagram had a clean little arrow labeled "automatic failover." Two engineers had reviewed the design and signed off. Everyone believed it worked.

It took eleven minutes. The replica had been promoted, but the application's connection pool cached the old primary's IP for the duration of a TTL nobody had checked, and every request hammered a dead socket until the pool finally expired and re-resolved. The error budget for the quarter — gone in one night. The postmortem found that the failover path had **never once been exercised** between the day it was built and the night it was needed. It was a hypothesis dressed up as an architecture diagram.

Here is the uncomfortable truth at the center of this post: **you do not actually know whether your redundancy, your failover, your retries, your circuit breakers, and your fallbacks work until you break the system on purpose and watch.** Every resilience mechanism you have ever built is, until tested, a belief. A reassuring belief. A well-reviewed belief. But a belief. And beliefs about distributed systems have a habit of being wrong in exactly the ways that hurt most, discovered at exactly the hour you are least equipped to handle them.

Chaos engineering is the discipline of finding the failure mode in a controlled experiment, in daylight, with the whole team watching — before the failure mode finds you at 3am. It is the experiment that turns "we think it works" into "we have evidence." This post is about how to run that experiment without becoming the outage you were trying to prevent. We will apply the scientific method to reliability, control the blast radius so a surprise stays small, run game days that test the system *and* the team, build a fault-injection toolkit, and walk through two worked examples where breaking on purpose paid for itself in a single afternoon.

![A five-layer stack showing chaos engineering as the scientific method applied to reliability, from steady state through hypothesis, fault injection, observation, and learning](/imgs/blogs/chaos-engineering-breaking-on-purpose-1.png)

If you are arriving here cold, start with the series map, [reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), which frames the whole loop this series follows: define reliability with SLIs and SLOs, measure it with observability, spend an error budget, reduce toil, respond to incidents, and learn from them. Chaos engineering sits squarely in the "engineer the fix and prove it" part of that loop. It is how you close the gap between the resilience you designed and the resilience you actually have.

## 1. The thesis: untested resilience is a hypothesis, not a fact

Let me make the thesis as sharp as I can, because everything else follows from it.

When you add a circuit breaker, you are making a claim: *if a downstream dependency gets slow, this breaker will trip and we will serve a fallback instead of piling up threads.* When you deploy across three availability zones, you are making a claim: *if one zone goes dark, the other two absorb the traffic with no user-visible impact.* When you configure retries with backoff, you are making a claim: *transient errors will be smoothed over, and we will not amplify a small problem into a retry storm.*

Every one of those is a falsifiable statement about how the system behaves under a condition that, by definition, does not happen during normal operation. And here is the trap: the only time those conditions *do* happen is during an incident — the worst possible moment to discover your claim was false. You built the safety net for the fall, and the first time you find out whether the net holds is mid-fall.

Software is not like a bridge, where you can compute the load-bearing capacity from material properties and trust the math. Distributed systems are full of emergent behavior: a timeout that is slightly longer than a dependency's recovery time turns a blip into a brownout; a retry policy that looks reasonable per-client becomes a thundering herd in aggregate; a "graceful" degradation path that nobody exercised silently throws an unhandled exception. You cannot reason your way to confidence here. You have to *test* it, and the only honest test is the real failure, injected on purpose.

This is why I am allergic to the phrase "highly available" on a slide. Available how? Tested against what? An untested HA configuration is not high availability — it is a sincere wish with a YAML file attached. The way this discipline works is that it refuses to take resilience on faith. It treats every resilience mechanism as a hypothesis and demands an experiment before it will call the mechanism real.

> **The inversion you must internalize:** a chaos experiment that "fails" — where the steady state does *not* hold — is a **success**. You just found a bug that would otherwise have surfaced during a real outage, and you found it safely, in daylight, with everyone watching and an abort button in your hand. The experiments that worry me are the ones that always pass. They usually mean the fault was too small or the SLI too coarse to notice the damage.

## 2. The scientific method, applied to reliability

Netflix, who coined "chaos engineering" with Chaos Monkey and later formalized it in the *Principles of Chaos Engineering*, framed it explicitly as the scientific method. That framing is not decoration. It is the thing that separates chaos engineering from "let's randomly break stuff and see what happens," which is just vandalism with extra steps.

The method has four steps. Get all four right and you have an experiment. Skip one and you have a mess.

### Step 1 — Define the steady state

The **steady state** is a measurable definition of "the system is healthy." Crucially, it is defined in terms of **outputs the user cares about**, not internal implementation details. CPU at 40% is not a steady state — users do not feel CPU. Steady state is your **SLI** (Service Level Indicator): the request success rate, the p99 latency, the throughput in successful checkouts per minute. (If "SLI" is new, it is a single number that measures one dimension of user-visible health, usually a ratio of good events over total events in a rolling window. The series post [choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) goes deep on picking them.)

For a checkout service, a reasonable steady state might be: **availability ≥ 99.5% of requests succeed AND p99 latency ≤ 300ms**, measured over a five-minute rolling window. That is a number you can watch on a dashboard. When you inject a fault, that number is the thing you stare at. Did it move? By how much? For how long? Did it recover?

The steady state is the entire experiment's truth oracle. Without it, you are injecting faults and squinting at logs hoping to feel whether something is wrong. With it, you have a binary, defensible answer: the steady state held, or it did not.

In practice, the steady state is a query you can run, not a vibe. If you measure with Prometheus, the success-rate SLI is a ratio of good events over total events over a rolling window, and the latency SLI is a histogram quantile. Here are the two PromQL expressions you would put on the game-day dashboard and watch during the experiment:

```promql
# Steady-state SLI 1: request success rate over a 5-minute rolling window.
# "Good" = any non-5xx response. This is the number we predict will HOLD.
1 - (
  sum(rate(http_requests_total{job="checkout-api", code=~"5.."}[5m]))
  /
  sum(rate(http_requests_total{job="checkout-api"}[5m]))
)

# Steady-state SLI 2: p99 latency over a 5-minute window from a histogram.
# We predict this stays <= 0.300 (300ms) while the fault is injected.
histogram_quantile(
  0.99,
  sum(rate(http_request_duration_seconds_bucket{job="checkout-api"}[5m])) by (le)
)
```

Those two expressions ARE the experiment's pass/fail line. Before you inject anything, you put them on a Grafana panel, draw the SLO thresholds as horizontal lines (0.995 and 0.300), and watch. When the fault goes in, your eyes are on these two numbers and nothing else. If both stay on the right side of their lines, the steady state held; if either crosses, you have found a weakness — and your pre-committed abort condition references the very same queries. The discipline of building these *first* is what separates an experiment from breaking things and hoping.

### Step 2 — Form a hypothesis

State, *before* you inject anything, what you expect to happen. The canonical form is:

> "If we **kill one instance** / **add 300ms of latency to dependency X** / **blackhole an availability zone**, the steady state will **hold** — the system will absorb the fault and the SLI will stay within its bounds."

Write it down. Out loud, in the experiment doc. The reason this matters is that a hypothesis is **falsifiable** and **specific**: it commits you to a prediction. If you do not state the prediction first, you will rationalize whatever happens after the fact ("well, a little latency bump is fine, I guess"). The hypothesis is the contract: here is what I claim, here is the condition, here is the SLI that will tell us if I was right.

Notice that a good hypothesis predicts the steady state will *hold*. You are not predicting a failure — you are predicting **resilience**. You believe your breaker, failover, or fallback will absorb the fault. The experiment tests that belief.

### Step 3 — Inject the fault

Now you introduce a real-world failure mode: kill a node, partition the network, add latency, exhaust CPU or disk, blackhole a dependency, fail a zone. Real faults, not simulated ones. The whole point is to subject the actual production (or production-like) system to the actual stress, because the bugs you are hunting live in the gap between the model and reality.

The faults you choose should be ones that *will* happen in the wild: instances die, networks get slow and lossy, dependencies time out, disks fill, zones go offline. You are not inventing exotic failures; you are rehearsing the ordinary ones before they arrive uninvited.

### Step 4 — Observe and decide

Watch the steady state. Two outcomes:

- **The SLI held.** Congratulations — you now have *evidence* that this resilience mechanism works under this fault. Not a belief. Evidence. Record it; it is the most valuable artifact the experiment produces.
- **The SLI broke.** Congratulations — you just found a weakness *before* it found you, in a controlled window you can abort, with the people who can fix it standing right there. This is the success case dressed as a failure.

That is the loop. Define normal, predict resilience, break it for real, watch the number. Everything else in this post is about doing those four steps *safely* and *usefully* — which mostly comes down to controlling the blast radius and turning the exercise into something the team learns from.

This is the same epistemic discipline that good production debugging uses. The debugging series post [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) makes the case that you debug by forming a falsifiable hypothesis and designing the cheapest experiment that could prove you wrong. Chaos engineering is that same move, run *proactively* instead of reactively: you do not wait for the bug to page you; you go looking for it with an experiment.

## 3. Blast-radius control: the cardinal safety rule

Here is where most teams' chaos ambitions go to die — either because they were too scared to ever inject a real fault, or because their first experiment took down production and the program got banned. The bridge between those two failure modes is **blast-radius control**, and it is the single most important operational skill in this discipline.

**Blast radius** is the scope of potential harm if the experiment goes worse than expected. (Jargon note: blast radius is just "how much can this break and how many users does it touch.") The cardinal rule is: **minimize the blast radius, and never run an experiment you cannot stop.** Every other safety practice is a corollary of that rule.

![A control-flow graph showing blast-radius control, starting tiny with an armed abort button, expanding scope only after a clean run, and fixing weaknesses before re-running](/imgs/blogs/chaos-engineering-breaking-on-purpose-3.png)

Concretely, here is what blast-radius control looks like in practice:

1. **Start tiny.** One instance, not the fleet. One percent of traffic, not all of it. One pod, one customer cohort, one canary. The first experiment should be almost embarrassingly small. If killing a single instance reveals a weakness, you do not need to kill ten to learn it.
2. **Staging first, then production.** Run the experiment in a staging or pre-production environment that mirrors prod as closely as you can afford. You will catch the obvious failures there for free. But understand the limitation: staging rarely has production's traffic shape, data volume, or dependency latencies, so a clean staging run is necessary, not sufficient. Eventually you have to run it where the bugs actually live — but you earn that privilege.
3. **Have an abort button — and test it first.** Before you inject anything, you must be able to instantly halt the experiment and restore normal operation. Kill the chaos process, roll back the network rule, re-add the instance. The abort path is itself a thing you should verify *before* the experiment, because an abort button that does not work turns your controlled experiment into an uncontrolled outage.
4. **Run during business hours, with the team watching.** This is counterintuitive to people who think "don't touch prod during the day." But the entire premise is to break things *when you are best equipped to respond* — caffeinated, staffed, alert, with the experts in the room — not at 3am when one exhausted on-call engineer is alone. You are deliberately moving the failure from the worst time to the best time.
5. **Expand only as confidence grows.** A clean run at 1% earns you 10%. A clean run on one instance earns you a zone. You ratchet scope upward, never leaping. Each expansion is a fresh, smaller experiment with its own hypothesis.
6. **Define the abort condition up front.** Not "we'll stop if it feels bad." A specific SLI threshold: "if error rate exceeds 5% OR p99 exceeds 1 second for more than 60 seconds, abort immediately." When that line is pre-committed, nobody has to make a judgment call mid-panic.

The reason this matters so much is that the value of chaos engineering and its danger come from the *same source*: you are injecting real faults into real systems. Blast-radius control is what lets you keep the value while capping the danger. A surprise inside a 1% blast radius with a working abort button is a learning. The same surprise across 100% of traffic with no abort is a Sev1 and a banned program.

#### Worked example: sizing a blast radius you can defend

Suppose your checkout service does 12,000 requests per minute and your error budget for the month is 0.5% (a 99.5% SLO over 30 days gives you about **216 minutes** of total allowable error-time, since `(1 − 0.995) × 30 × 24 × 60 ≈ 216`). You want to run a latency-injection experiment.

If you target 1% of traffic — 120 requests per minute — and your worst case is that *all* of those error out for the 5 minutes it takes to notice and abort, that is `120 × 5 = 600` failed requests. Against `12,000 × 5 = 60,000` requests in that window, that is a 1% error rate for five minutes, which burns roughly 5 minutes of budget at a 2× rate. Annoying, recoverable, and a tiny fraction of your 216-minute monthly budget.

Now suppose you skipped blast-radius control and ran the same fault at 100% of traffic, and it took you 11 minutes to abort (because, surprise, the abort path had a bug). That is potentially `12,000 × 11 = 132,000` requests affected, burning over half your monthly error budget in a single experiment — the exact failure mode from this post's opening story, except *you* caused it. The arithmetic is the argument: blast radius is the difference between a 5-minute learning and a budget-blowing self-inflicted outage.

## 4. Hope versus evidence: what you actually gain

Let me dwell on the payoff, because if you cannot articulate the value, you will never get the time to do this.

![A before-and-after comparison contrasting untested resilience that fails at 3am against chaos-validated resilience where a bug is found safely in daylight](/imgs/blogs/chaos-engineering-breaking-on-purpose-2.png)

Before chaos engineering, your resilience claims live in the realm of **hope**. The failover diagram says it works. The breaker config looks right. The retry policy seems sane. Everyone nods in the design review. And then the real failure arrives and you discover the connection pool cached the old IP, the breaker threshold was set so high it never trips, the retry policy lacks jitter and creates a synchronized stampede, the fallback path throws because it was never exercised. Hope is not a strategy, and a design review is not a test.

After chaos engineering, those same claims live in the realm of **evidence**. You killed an instance during a game day and watched the load shed to peers with the SLI unmoved — you *know* failover works because you saw it. You added 5 seconds of latency to a dependency and watched the breaker trip and the fallback serve within 200ms — you *know* the breaker works because you watched it trip. The difference between "we think it works" and "we have evidence it works" is the entire game.

There is a softer benefit that compounds over time: chaos engineering builds **organizational nerve**. Teams that regularly break their systems on purpose stop being terrified of failure. They have seen the failover happen. They have watched the breaker trip. They have run the runbook under pressure when it was a drill. So when the real thing comes, it is familiar, not novel. Calm under fire is mostly the absence of surprise, and chaos engineering is how you spend your surprises in advance, on your own schedule.

| Property | Untested resilience (hope) | Chaos-validated resilience (evidence) |
| --- | --- | --- |
| When the failover path is first exercised | During the real outage, at 3am | During a game day, at 10am, on purpose |
| Who is present when it fails | One exhausted on-call engineer | The whole team, caffeinated, watching |
| Can you stop it? | No — it is a real outage | Yes — armed abort button, pre-committed condition |
| What you learn | Public, expensive, after the damage | Private, cheap, before any user impact |
| Confidence in the mechanism | A reviewed belief | Measured evidence, dated and recorded |
| Effect on team nerve | Trauma | Familiarity, calm |

## 5. The fault-injection toolkit

You need real tools to inject real faults. Here is the practical landscape, organized by what kind of fault you are introducing. Pick the tool by the resilience pattern you want to validate, not the other way around.

![A matrix mapping fault types to their tools, the resilience patterns each validates, and the risk level of running each fault](/imgs/blogs/chaos-engineering-breaking-on-purpose-4.png)

**Instance / process faults** — kill a node, kill a process, reboot a host. This is where chaos engineering started: Netflix's **Chaos Monkey**, part of the broader **Simian Army** (which also had Chaos Gorilla for whole-zone failures and Latency Monkey for latency injection). On Kubernetes, `kube-monkey` randomly kills pods, and `kubectl delete pod` or a plain `kill -9` does it manually. The fault is simple; what it validates is profound: does your replica set actually have N-1 capacity, does the load balancer drain the dead instance, does failover happen?

**Network faults** — latency, packet loss, partition, blackhole. The Linux kernel's traffic-control subsystem, driven by `tc` with the `netem` (network emulation) queueing discipline, is the bedrock tool and costs nothing. It adds latency, jitter, loss, and corruption to an interface. `iptables` can drop packets to a specific destination to simulate a blackhole or partition. Higher up the stack, **Chaos Mesh** (a CNCF project) and **Gremlin** (commercial) wrap these in declarative, scoped, abortable experiments designed for Kubernetes.

**Resource exhaustion** — CPU, memory, disk, I/O, file descriptors. `stress-ng` is the swiss-army knife for hammering CPU and memory. cgroup limits let you constrain a container's resources to simulate a noisy neighbor. Filling a disk to 95% (carefully, on a non-critical volume) tests whether your "disk almost full" alert fires and whether the app degrades gracefully or crashes.

**Dependency faults** — make a downstream service slow, error, or vanish. Blackhole the route to a dependency with `iptables`, inject latency in a service mesh sidecar (Istio and Envoy can add fault injection — fixed delays and HTTP error codes — declaratively), or use Gremlin/Chaos Mesh to target a specific service-to-service edge. This is the fault that validates breakers, timeouts, retry budgets, and fallbacks.

**Clock skew** — push a node's clock forward or backward. Underrated and brutal: clock skew breaks token expiry, certificate validation, distributed-lock leases, and any logic that assumes monotonic time across nodes. `libfaketime` or a controlled `date` adjustment in a sandbox exposes these.

Here is the single most useful command in the entire toolkit, because it costs nothing and works on any Linux box. It adds 300ms of latency (with 50ms of jitter) to all egress traffic on `eth0`:

```bash
# Add 300ms ± 50ms latency to all outbound traffic on eth0
sudo tc qdisc add dev eth0 root netem delay 300ms 50ms distribution normal

# Add 5% packet loss on top (run as a single replace to combine)
sudo tc qdisc change dev eth0 root netem delay 300ms 50ms loss 5%

# THE ABORT BUTTON — remove all emulation, restore normal networking instantly
sudo tc qdisc del dev eth0 root
```

That last line is your abort button for any `tc netem` experiment, and you should run it once *before* the experiment to confirm it works (it will harmlessly error if no qdisc exists, which is itself a useful check). Notice the discipline baked into even this tiny example: you know exactly how to stop before you start.

For team-friendly, scoped, declarative experiments, here is a **Chaos Mesh** spec that injects 5 seconds of latency into 50% of traffic hitting a `payments` dependency, scoped to a single namespace, with a fixed 10-minute duration so it self-terminates even if you forget:

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: payments-latency-experiment
  namespace: checkout
spec:
  action: delay
  mode: fixed-percent
  value: "50"                  # blast radius: only 50% of matched pods
  selector:
    namespaces:
      - checkout
    labelSelectors:
      app: checkout-api
  direction: to
  target:
    mode: all
    selector:
      namespaces:
        - payments
      labelSelectors:
        app: payments-api
  delay:
    latency: "5000ms"
    jitter: "200ms"
    correlation: "50"
  duration: "10m"             # auto-aborts after 10 minutes — the safety net
```

The `duration` field is doing safety work: even if every human walks away from their desk, the experiment ends itself. The `value: "50"` and the namespace selector are blast-radius controls baked into the spec. This is the difference between a tool built for chaos engineering and a raw `tc` command — the safety rails are first-class.

| Tool | Fault types | Scope control | Best for |
| --- | --- | --- | --- |
| `tc netem` | Latency, jitter, loss, corruption | Per-interface, manual | Cheap single-host network experiments |
| `iptables` | Blackhole, partition | Per-rule, manual | Simulating a dead dependency or split |
| `stress-ng` | CPU, memory, I/O exhaustion | Per-process | Saturation and noisy-neighbor tests |
| `kube-monkey` / Chaos Monkey | Kill instance / pod | Random, label-scoped | Continuous instance-death resilience |
| Chaos Mesh | Network, pod, stress, IO, time | Declarative, percent + selector | Scoped, abortable Kubernetes experiments |
| Gremlin | Full catalog | Declarative, fine-grained, audited | Enterprise game days with safeguards |
| LitmusChaos | Pod, network, resource | Declarative, CRD-driven | GitOps-native chaos in CNCF stacks |
| Istio / Envoy fault injection | Delay, abort (HTTP codes) | Per-route, percent | Mesh-level dependency fault testing |

A note on choosing: do not buy Gremlin for your first experiment. Start with `tc netem` and `kill -9` on a single staging host — they are free, they teach you the fundamentals, and they make you feel the danger directly, which is good calibration. Reach for Chaos Mesh or Gremlin when you are running scoped experiments in production and you need the safety rails, audit trail, and abort orchestration that a purpose-built tool provides.

Two more raw artifacts worth having in your back pocket, because they cost nothing and cover the two most common faults after latency. First, a resource-exhaustion experiment with `stress-ng` — pin four CPU cores at 90% load for two minutes to see whether your autoscaler reacts and whether latency degrades gracefully or cliffs:

```bash
# Saturate 4 CPUs to ~90% for 120s, then it self-terminates.
# Watch: does the HPA add a pod? does p99 degrade smoothly or cliff?
stress-ng --cpu 4 --cpu-load 90 --timeout 120s --metrics-brief

# Memory pressure variant: allocate 80% of a 2GiB cgroup for 120s
stress-ng --vm 2 --vm-bytes 80% --timeout 120s
```

Second, an `iptables` dependency blackhole — drop every packet to a downstream so it looks completely gone, the cleanest way to test whether a fallback actually serves:

```bash
# Blackhole all traffic to the payments dependency (10.0.5.20:8443).
# This is the fault: payments looks dead. Does the fallback serve?
sudo iptables -A OUTPUT -d 10.0.5.20 -p tcp --dport 8443 -j DROP

# THE ABORT BUTTON — delete the exact rule to restore connectivity.
sudo iptables -D OUTPUT -d 10.0.5.20 -p tcp --dport 8443 -j DROP
```

Again, notice the pattern: every fault command comes with its paired abort command, and you confirm the abort works before you arm the fault. That habit — never inject a fault whose reversal you have not already verified — is the muscle memory that keeps a chaos practice from becoming a chaos incident.

## 6. Writing a chaos-experiment spec

Before any fault goes in, the experiment lives as a written spec. This is not bureaucracy; it is the artifact that forces all four steps of the scientific method to be explicit *before* the adrenaline of injection. A good spec has five fields, and if you cannot fill all five, you are not ready to run.

```yaml
chaos_experiment:
  id: "CE-2026-041"
  title: "Checkout survives loss of one replica (N-1 capacity)"
  owner: "checkout-team"

  # 1. STEADY STATE — the measurable normal, as a query, not a vibe.
  steady_state:
    sli_success_rate: 'success rate >= 99.5% over 5m (PromQL above)'
    sli_latency: 'p99 <= 300ms over 5m (histogram_quantile above)'
    dashboard: "grafana.internal/d/checkout-sli"

  # 2. HYPOTHESIS — what we predict, stated as resilience HOLDING.
  hypothesis: >
    Killing one of three checkout-api replicas will hold the steady
    state: the two survivors absorb the load, success rate stays
    >= 99.5%, and p99 stays <= 300ms throughout.

  # 3. FAULT — the exact injection, the real-world failure we rehearse.
  fault:
    type: "instance termination"
    target: "1 random pod, app=checkout-api, namespace=checkout"
    method: "kubectl delete pod <name>"

  # 4. BLAST RADIUS — scope and the line we will not cross.
  blast_radius:
    scope: "exactly 1 of 3 replicas; staging-equivalent prod"
    environment: "prod-canary cluster, business hours only"
    abort_condition: "error rate > 5% OR p99 > 1s sustained 60s"
    auto_timeout: "20m hard stop, then re-scale regardless"

  # 5. ABORT — the verified way to stop, tested BEFORE injection.
  abort_procedure:
    command: "kubectl scale deploy/checkout-api --replicas=3"
    verified_before_run: true
    expected_recovery: "steady state restored within 90s"

  result:                       # filled in AFTER the run
    held: null                  # true = evidence; false = weakness found
    findings: ""
    action_items: []            # every finding gets a ticket + owner + date
```

The two halves of this spec — everything above `result`, written before; the `result` block, filled after — are the experiment's full record. The pre-written half makes the prediction falsifiable and the safety explicit. The post-written half turns the run into a durable artifact: a dated, defensible statement of evidence (`held: true`) or a tracked weakness (`held: false` plus action items). Keep these specs in version control next to your runbooks. After a year you will have a library of dated evidence about exactly which failure modes your system survives — which is worth more than any architecture diagram, because it is true.

## 7. Game days: testing the system and the team at once

A **game day** is a scheduled, planned chaos exercise where the team injects a realistic failure and practices the response. It is the single highest-leverage chaos practice for most organizations, because it tests two things at once that a purely automated experiment cannot: the **system's** resilience *and* the **team's** incident response.

![A timeline of a game day from kickoff through fault injection, the page firing, a capacity surprise, abort, and the debrief that files action items](/imgs/blogs/chaos-engineering-breaking-on-purpose-5.png)

This dual nature is why game days matter so much for an SRE practice. The automated experiment answers "does the breaker trip?" The game day answers "does the breaker trip, *and* does the on-call engineer get paged, *and* do they find the right runbook, *and* does the runbook actually work, *and* can two people coordinate on the incident bridge?" It exercises everything downstream of the fault, including the humans, the alerting pipeline, the runbooks, and the communication channels. It is, in effect, a fire drill for your incident response — and like a fire drill, the point is to find out that the exit is blocked *before* the building is on fire.

There are two flavors, and you want both in your rotation:

- **Tabletop game day.** No real fault is injected. The team gathers and walks through a scenario verbally: "It is 2am, the primary region just went dark, what does each of you do?" People talk through the runbook, find the gaps ("wait, who has the authority to fail over to the secondary region?"), and surface missing documentation. Cheap, zero risk, great for onboarding and for testing decision-making and escalation paths. The weakness: people *say* they would do the right thing, but saying is not doing.
- **Live game day.** A real fault is injected (with full blast-radius control) and the team responds for real. The on-call gets a real page. They open a real incident channel. They run the runbook against a real degraded system. This is far more valuable — and far more revealing — because it surfaces the gaps that only appear under real conditions, like a runbook step that references a dashboard that was renamed six months ago.

| Dimension | Tabletop game day | Live game day |
| --- | --- | --- |
| Fault injected | None — verbal scenario | Real fault, controlled blast radius |
| What it tests | Decision-making, escalation, comms | The system *and* the response, for real |
| Risk | Zero | Low if blast radius is controlled |
| Reveals | Missing docs, unclear ownership | Broken runbooks, untested code paths, real bugs |
| Cost to run | One hour, a meeting room | Prep, abort tooling, a watching team |
| Best for | Onboarding, new scenarios, low maturity | Proving resilience once patterns exist |
| Weakness | People *say* the right thing, untested | Needs observability and a tested abort path |

Run tabletops to find the gaps in your *thinking* and your documentation; run live game days to find the gaps in your *system* and your code. A healthy practice does both: tabletop a new failure scenario to shake out the obvious holes cheaply, fix those, and only then graduate the same scenario to a live exercise where the remaining, subtler bugs hide.

A live game day is the natural follow-on to [the anatomy of an incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident): it is a manufactured incident that lets your team rehearse the lifecycle — detect, declare, triage, mitigate, diagnose, resolve — with the stakes turned down and an abort button available. If your runbooks have never been run by anyone other than their author, a game day is how you find out which steps are fiction. The companion practice of writing runbooks that hold up is covered in [runbooks that survive 3am](/blog/software-development/site-reliability-engineering/runbooks-that-survive-3am); a game day is the proving ground for them.

### A game-day runbook (the agenda that keeps it safe and useful)

Here is the structured agenda I run. Treat it as a checklist; the structure is what keeps a game day from turning into either a non-event or an actual incident.

```yaml
game_day:
  title: "Q3 Checkout Replica-Loss Game Day"
  date: "2026-06-23"
  time: "10:00-12:00 local (NOT during a deploy freeze or peak traffic)"
  facilitator: "on-call lead (drives the clock, holds the abort button)"
  participants:
    - on-call engineer (responds as if real — does NOT know exact timing)
    - service owner (observes, does not rescue)
    - incident commander (practices coordination)
    - scribe (records the timeline for the debrief)

  prerequisites:
    - steady-state SLI dashboard is up and shared on screen
    - abort procedure tested and confirmed working BEFORE injection
    - stakeholders notified: "game day in progress, alerts may be drills"
    - on-call schedule clear, no concurrent real incident in flight

  hypothesis: >
    Killing 1 of 3 checkout-api replicas will hold steady state:
    success rate stays >= 99.5% and p99 stays <= 300ms over a 5m window.

  blast_radius:
    scope: "one replica in the checkout namespace, staging-equivalent prod"
    abort_condition: "error rate > 5% OR p99 > 1s sustained 60s -> abort"
    max_duration: "20 minutes then auto-restore regardless"

  agenda:
    - "10:00 kickoff: review hypothesis, abort condition, roles"
    - "10:15 inject: facilitator kills one replica (kubectl delete pod)"
    - "watch: did the page fire? did on-call ack? did SLI hold?"
    - "respond: on-call runs the runbook AS IF real"
    - "10:30 abort/restore: re-scale, confirm steady state recovers"
    - "11:00 debrief: what held, what broke, action items + owners"

  success_is_not: "the experiment passing"
  success_is: "every gap we found gets a ticket with an owner and a date"
```

The line `success_is_not: "the experiment passing"` is the cultural heart of the practice. A game day where everything works flawlessly and you find nothing is *suspicious*, not satisfying — it usually means the fault was too gentle or the SLI too coarse. A game day where you find three real gaps is a great game day. You came to find bugs; finding them is winning.

## 8. Where to run it: the maturity path

You do not start a chaos practice by killing pods in production. You climb a maturity ladder, and the rung you are on is determined by your prerequisites and your track record.

![A decision tree showing where chaos should run, gated by maturity, from staging experiments through scheduled game days to automated continuous production chaos](/imgs/blogs/chaos-engineering-breaking-on-purpose-6.png)

There are two hard **prerequisites** before you inject your first fault anywhere:

1. **You need observability and good SLIs.** Chaos engineering without monitoring is not an experiment — it is just breaking things and hoping you notice. You cannot define a steady state if you cannot measure the system's health, and you cannot tell whether the steady state held if you cannot watch the SLI in real time. If your dashboards are thin, fix that first; the series posts [monitor the user, not just the server](/blog/software-development/site-reliability-engineering/monitor-the-user-not-just-the-server) and [dashboards that tell the truth](/blog/software-development/site-reliability-engineering/dashboards-that-tell-the-truth) are the place to start. This is the one rule I will not bend on: no observability, no chaos.
2. **You need the resilience patterns actually built.** Chaos engineering *validates* resilience; it does not create it. If you have no failover, no breaker, and no fallback, then of course killing a node takes you down — you have not built anything to absorb it. Build the redundancy, the breakers, and the fallbacks first; the experiment comes after, to prove they work. (Those patterns are the subjects of the sibling posts on [redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works), `circuit-breakers-bulkheads-and-load-shedding`, and `graceful-degradation-and-fallbacks` in this track.)

With those in place, the ladder looks like this:

- **Rung 1 — Staging experiments.** Run faults in a production-like staging environment. Tiny blast radius, no real users at risk, fast iteration. You will catch the obvious bugs here: the abort path that does not work, the breaker that is misconfigured, the missing N-1 capacity in a smaller-but-representative cluster. A clean run here is a prerequisite for the next rung, not a substitute for it.
- **Rung 2 — Scheduled game days in production.** Once staging is clean and your team is comfortable, graduate to scheduled, supervised game days against production (or a production canary), with the full blast-radius regimen: small scope, abort button, business hours, team watching. This is where most mature teams live, and for many services it is the right ceiling.
- **Rung 3 — Automated, continuous chaos in production.** The endgame, and not everyone needs it. Tools like Chaos Monkey run *automatically* and *continuously* in production, killing instances during business hours, all the time, at a low rate. The genius of continuous chaos is that it makes resilience a *property that cannot silently rot*: if someone introduces a change that breaks failover, a Chaos Monkey kill exposes it within hours instead of months later during a real outage. It turns resilience into a continuously-enforced invariant rather than a thing you test once and hope stays true. But it demands real maturity: rock-solid observability, well-exercised runbooks, and a team that genuinely is not afraid of instance death because they have seen it a thousand times.

The mistake I see most often is teams trying to leap straight to Rung 3 because it sounds impressive, without the observability or the patterns to survive it. Start at Rung 1. Earn each rung. The ladder is not a formality; it is the blast-radius principle applied across time.

### Chaos and the error budget

There is a clean way to think about how much chaos you can afford, and it runs through the error budget — the currency that ties this whole series together. Recall the nines: a 99.9% SLO over 30 days gives you about **43.2 minutes** of error budget per month, because $(1 - 0.999) \times 30 \times 24 \times 60 = 43.2$. A 99.95% SLO gives you about 21.6 minutes; a 99.99% SLO gives about 4.3 minutes. That budget is what you are allowed to spend on *everything* unreliable — bad deploys, real incidents, and, yes, chaos experiments.

The disciplined view is that a chaos experiment is a **planned, deliberate withdrawal** from the error budget, and a small one if your blast radius is controlled. From the earlier worked example, a 1%-blast-radius experiment that errors for 5 minutes burns only a tiny slice of a 43.2-minute budget — a withdrawal you make on purpose, in daylight, to *buy evidence*. Contrast that with an unplanned 3am outage that blows the whole budget at once and buys you nothing but a postmortem. This reframes the cost question entirely: you are not asking "can we afford to break it?" You are asking "would we rather spend a few budget-minutes on purpose to find the bug, or spend the whole budget by accident when the bug finds us?" When you have budget to spare, spend a little of it on chaos; when the budget is already exhausted by real incidents, that is a signal to *pause* discretionary chaos and fix what is actually burning it. The error budget tells you both how much chaos you can run and when to stop. (The full mechanics are in [the error budget: the currency of reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability).)

## 9. What chaos actually validates

Let me close the loop on the rest of this track, because chaos engineering is the *capstone test* for every resilience mechanism this series describes. A single well-designed fault injection validates a whole stack of patterns at once — and, crucially, validates them *together*, in combination, which is exactly where they tend to interact in surprising ways.

![A fan-out graph showing one injected fault validating failover, circuit breakers, retry budgets, fallbacks, and the on-call response, all converging on a single verdict](/imgs/blogs/chaos-engineering-breaking-on-purpose-8.png)

Here is what each fault is really testing:

- **Kill an instance → does failover actually cut over?** Does the load balancer drain the dead node, does the replica set reschedule, does the surviving capacity absorb the load (N-1 capacity), or does the cutover stall on a cached IP like the opening story? This validates everything in [redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works).
- **Add latency to a dependency → does the circuit breaker trip?** When the downstream gets slow, does the breaker open at the right threshold and stop the bleeding, or does it never trip (threshold too high) and let threads pile up until the whole service stalls? This validates the breaker, timeout, and bulkhead work in `circuit-breakers-bulkheads-and-load-shedding`.
- **Make a dependency flaky → does the retry budget hold?** Do retries with backoff and jitter smooth over the transient errors, or do retries-without-jitter create a synchronized retry storm that amplifies a small problem into a self-inflicted outage? (The math of retry amplification — where a 3× retry policy turns a 10% failure rate into a 30% load surge on the struggling dependency — is covered in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right).)
- **Blackhole a dependency → does the fallback serve?** When the dependency is fully gone, does the service serve degraded-but-useful results (cached data, a default, a reduced feature set), or does it throw an unhandled exception because the fallback path was never exercised? This validates `graceful-degradation-and-fallbacks`.
- **Saturate CPU or spike traffic → does the autoscaler scale?** Does the horizontal pod autoscaler add capacity fast enough, or does it lag behind the load curve and let latency blow out before the new pods are ready and warm?
- **Inject any fault → does the on-call get paged and know what to do?** Does the alert fire on the right symptom, route to the right person, and link to a runbook that actually works? This is the part automated experiments miss and game days catch.

The combinatorial point is worth stressing: these mechanisms interact. A breaker that trips correctly might still cause a problem if the fallback it triggers is itself overloaded, or if the breaker's recovery probe creates a thundering herd when the dependency comes back. You cannot find those interaction bugs by testing each mechanism in isolation. Only an integrated fault injection — break one real thing and watch the whole system respond — surfaces them. That is the unique value chaos engineering adds over unit-testing your resilience config.

### The retry-amplification principle (why a dependency-latency experiment is so revealing)

The retry-budget check deserves a closer look, because the math explains why injecting latency is one of the most dangerous *and* most informative faults you can run. When a dependency gets slow, clients time out and retry. If your client retries up to $r$ times, then for every one original request the dependency may receive up to $1 + r$ requests during the slow period. The **retry-amplification factor** is therefore roughly $1 + r$: a policy of "retry twice" can triple the load on a dependency at precisely the moment it is least able to handle it.

Now layer in the cascade. Suppose the dependency is at 100% capacity and starts shedding 20% of requests. Clients see those failures and retry. With $r = 2$ retries and no jitter, the offered load can climb toward $1 + r = 3\times$ the original — turning a dependency that was merely *full* into one that is *catastrophically overloaded*, which causes more failures, which causes more retries, in a self-reinforcing spiral. This is the thundering herd, and the chilling part is that every client is behaving "correctly" by its own local logic; the disaster is emergent.

There are exactly two fixes, and a latency-injection experiment tests both at once. First, **backoff with jitter**: spreading retries out over randomized intervals desynchronizes the herd so the dependency sees a smear of load instead of synchronized waves. Second, a **retry budget**: a hard cap that says "retries may not exceed, say, 10% of total request volume," so that no matter how bad things get, the amplification is bounded at $1.1\times$ rather than $3\times$. When you inject 5 seconds of latency and watch the offered load on the dependency, you find out immediately whether your retry policy has jitter and a budget — or whether you have built a retry storm with a config file. (The full derivation lives in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right); the point here is that the experiment makes the amplification *visible* in a controlled window instead of during a real cascade.)

## 10. Worked example one: the game day that found a hidden capacity flaw

Let me walk through a complete experiment end to end, because the discipline only makes sense when you see it run. This is a composite of game days I have run; the numbers are illustrative but the shape is exactly real.

**The setup.** A checkout service runs three replicas behind a load balancer. Each replica sits at about 70% CPU under normal peak load. The team believes — reasonably, on paper — that losing one replica is survivable: "we have three, we can lose one." That belief had never been tested. It was a hypothesis wearing the costume of an obvious fact.

**Step 1 — steady state.** The team defines the SLI: success rate ≥ 99.5% and p99 latency ≤ 300ms over a 5-minute rolling window. The dashboard is on the big screen.

**Step 2 — hypothesis.** "If we kill one of the three replicas, the steady state will hold. The two survivors will absorb the traffic and the SLI will stay within bounds."

**Step 3 — inject.** At 10:15, the facilitator runs `kubectl delete pod checkout-api-7f8b9-x2k4l`. One replica gone. The clock starts.

**Step 4 — observe.** And here is the surprise. The two surviving replicas now have to absorb the load that three were handling. Each was at 70% CPU; now each is being asked to do 150% of its previous share. CPU pins at 100%, request queues back up, and **p99 latency jumps from 250ms to over 4 seconds.** The steady state breaks badly. At 10:25 the abort condition trips (p99 > 1s sustained), and at 10:30 the facilitator restores the replica. p99 settles back to 250ms within a minute.

![A before-and-after diagram showing the N-1 capacity bug, where three replicas could not absorb a loss but four right-sized replicas held the SLI steady](/imgs/blogs/chaos-engineering-breaking-on-purpose-7.png)

**The finding.** The cluster had no **N-1 capacity**. "We have three replicas" was true; "we can survive losing one" was false, because each replica was already running too hot to absorb a third of its peers' load. With three replicas at 70%, the math is brutal: losing one means the survivors each need to handle `(3 × 70%) / 2 = 105%` of a single replica's full capacity — which is, by definition, impossible without latency exploding.

**The fix.** Right-size for N-1. To survive the loss of one replica with each remaining replica staying under, say, 70% CPU, you need enough replicas that `(N − 1)` of them can carry the full load at 70%. The arithmetic: at peak the fleet needs `3 × 70% = 210%` of one replica's capacity. To carry that across `N − 1` replicas at 70% each, you need `(N − 1) × 70% ≥ 210%`, so `N − 1 ≥ 3`, meaning **N = 4**. Moving to four replicas drops the steady-state CPU to about 52% each, and losing one leaves three replicas each at about 70% — exactly at the design ceiling, SLI held.

#### Worked example: re-running to confirm the fix

After right-sizing to four replicas, the team re-ran the *same* experiment a week later — small blast radius again, because confidence is earned per-configuration, not transferred. At 10:15 they killed one of four replicas. The three survivors absorbed the load: each rose from ~52% to ~70% CPU, comfortably within the design ceiling. p99 latency moved from 250ms to 260ms — a blip well inside the 300ms SLI bound. The steady state **held**. The hypothesis was now *evidence*: this service survives the loss of one replica, proven, dated, and recorded.

**The payoff, stated honestly.** The team found and fixed a capacity flaw in a controlled 20-minute exercise during business hours, with the abort button bringing the system back in under a minute. The alternative — discovering this during a real instance death at 3am, with one on-call engineer watching p99 climb to 4 seconds and not understanding why — would have been a multi-hour Sev1, a chunk of the error budget, and a postmortem that found the same thing the game day found, except after the customer impact instead of before it. The cost of the experiment: one morning and one extra replica's worth of cloud spend. The cost of *not* running it: a real outage you could not abort.

## 11. Worked example two: proving the circuit breaker works

The second example is the happy path — the experiment where the steady state *holds* and you walk away with evidence instead of a bug. These are just as valuable, because "we have evidence the breaker works" is a thing you can put in a design review and a thing that lets you sleep.

**The setup.** A product-page service calls a `recommendations` dependency to show "customers also bought." The team has built a circuit breaker (using resilience4j, configured in the service mesh) that is *supposed* to trip when recommendations gets slow, serving an empty recommendations block as a fallback so the product page still renders. They *think* it works. The config looks right. But "looks right" and "works" are different claims, and only one of them is evidence.

**Step 1 — steady state.** SLI: product-page success rate ≥ 99.9% and p99 ≤ 400ms. Importantly, "success" means *the product page renders*, not "recommendations are present" — degraded-but-rendered is still a success by this definition, which is exactly the point of the fallback.

**Step 2 — hypothesis.** "If we add 5 seconds of latency to the recommendations dependency, the circuit breaker will trip within its rolling window, the fallback will serve an empty recommendations block, and the product-page steady state will hold. Users will see a page without recommendations, not a slow page or an error."

**Step 3 — inject.** Using the Chaos Mesh `NetworkChaos` spec from earlier (5 seconds of latency, 50% of traffic to the `recommendations` service, 10-minute auto-abort), the team injects the latency at 10:15 with the breaker's metrics on screen.

**Step 4 — observe.** The recommendations calls start timing out. The breaker's failure counter climbs. Within about 20 seconds — after the breaker sees enough failures in its sliding window to cross the 50% failure threshold — the breaker **opens**. Now calls to recommendations short-circuit instantly and return the fallback (empty block) without waiting on the 5-second timeout. The product-page p99 dips briefly to ~600ms during the 20-second window where calls were still timing out, then drops right back to ~280ms once the breaker is open and short-circuiting. Success rate never drops below 99.9%. The steady state **held.**

**The finding.** Two pieces of evidence, both valuable. First, the breaker works — it tripped, the fallback served, the page kept rendering. Second, and more subtle: there was a ~20-second window during which the breaker had not yet accumulated enough failures to trip, and during that window p99 climbed to 600ms. That is a real, measured detail you would never have known from reading the config. It tells you the breaker's sliding-window size and failure threshold trade detection speed against false trips — and now you have data to tune that trade-off deliberately instead of guessing.

**The payoff.** "We think the breaker works" became "we have evidence the breaker trips in ~20 seconds and the fallback holds the SLI, with a measured ~600ms p99 excursion during the trip window." That is a sentence you can defend. It is the difference between hoping the recommendations dependency's next real outage is survivable and *knowing* it is, because you already survived it on purpose.

#### Worked example: the half-open recovery probe

There is a sequel to this experiment that is worth running, because the breaker has a *second* behavior that is just as untested: what happens when the dependency comes *back*. Most breakers, after the open period, move to a **half-open** state and let a small number of probe requests through to test whether the dependency has recovered. If those probes succeed, the breaker closes and full traffic resumes; if they fail, it re-opens.

The subtle bug hides here. If the breaker lets *all* paused traffic through the instant it half-opens — instead of a trickle of probes — then the recovering dependency gets slammed with the full backed-up load the moment it shows the first sign of life, and it falls right back over. That is a thundering herd of the breaker's own making. To test it, the team extended the experiment: inject 5 seconds of latency for 90 seconds, then *remove* the fault and watch the recovery. The first run revealed exactly this bug — recovery was spiky, with the dependency briefly re-failing as the breaker dumped held traffic on it. The fix was to configure the half-open state to admit a limited number of concurrent probe calls and ramp traffic gradually. The re-run showed a smooth recovery with no re-failure. Two experiments, one config, and a breaker whose trip *and* recovery are both proven — none of which the config file alone would ever have told them.

## 12. War story: Chaos Monkey and the origin of the discipline

The canonical real-world story here is Netflix, and it is worth telling accurately because it is frequently mythologized.

When Netflix migrated to Amazon's cloud around 2010–2011, they faced a new reality: in the cloud, instances *will* fail, unpredictably and without warning, and there is nothing you can do to prevent it. The traditional response would have been to treat instance failure as an exceptional event to be avoided. Netflix made the opposite, more profound bet: if instance failure is inevitable, then the system must be *built* to tolerate it as a routine event — and the only way to guarantee that tolerance does not silently rot is to *cause* instance failure constantly, in production, so that any regression in resilience is caught immediately.

So they built **Chaos Monkey**: a service that randomly terminates production instances during business hours. Not at 3am — during business hours, deliberately, so that when something broke, engineers were at their desks to see it and fix it. The effect over time was that Netflix engineers stopped building services that could not survive an instance death, because such services would not survive contact with Chaos Monkey. Resilience became a *forced* property, continuously enforced, rather than an aspiration that drifts.

Netflix later expanded this into the **Simian Army**: Chaos Monkey for instances, **Chaos Gorilla** for taking down an entire availability zone, and **Chaos Kong** for failing an entire AWS region. Chaos Kong is the most striking — Netflix regularly evacuated whole regions on purpose, in production, to prove their multi-region failover worked. Because they practiced it constantly, when a real regional issue hit, the failover was a familiar, rehearsed maneuver rather than a terrifying first-time event. They had spent their surprises in advance.

The lesson is not "go run Chaos Kong." Most organizations should never run Chaos Kong, and that is fine. The lesson is the *principle*: failure is not an exceptional event to be prevented; it is a routine event to be tolerated, and the only way to know you tolerate it is to cause it deliberately and watch. Netflix turned that principle into a continuously-running invariant. You can adopt the same principle at whatever rung of the maturity ladder fits your risk and your scale.

What makes the Chaos Kong story genuinely instructive is the asymmetry it exposes. Regional failover is the most expensive resilience mechanism a team can build and the *least* likely to be exercised by accident — a real region failure might happen once every few years, which means a regional failover path can sit untested for years and rot completely in the meantime. DNS TTLs drift, capacity assumptions in the secondary region go stale, a new dependency gets added that only lives in the primary region, and nobody notices because the path is never walked. By evacuating a region on a schedule, Netflix forced that most-fragile, least-exercised path to be walked regularly, so its rot was caught in days rather than discovered during the once-in-three-years real event. The general rule falls right out of this: **the resilience mechanisms most worth chaos-testing are precisely the ones that are most expensive and least often exercised in normal operation** — regional failover, backup restores, the cold-standby promotion — because those are the ones most likely to have silently broken since the last time anyone looked. The path you exercise every second you can trust; the path you exercise once a decade you cannot.

A second, cautionary story rounds out the picture: the **thundering-herd / retry-storm** failure mode that has caused real cascading outages across the industry. The pattern is that a brief blip in a dependency causes every client to retry simultaneously; without jitter, those retries synchronize into waves that hammer the recovering dependency and prevent it from ever recovering — a self-reinforcing outage. This is precisely the kind of emergent, interaction-driven failure that you cannot find by reviewing any single component's config, and exactly the kind that a dependency-latency chaos experiment surfaces immediately. You inject latency, watch the retry volume spike, see the dependency get hammered, and realize your retry policy lacks jitter — in a controlled window, before it becomes a real cascading failure. (The architecture-level treatment of cascading failures and how to design against them lives in the system-design series; this series tests whether those designs hold up under real fault injection. For the broader testing context, the system-design post [testing distributed systems: chaos and load](/blog/software-development/system-design/testing-distributed-systems-chaos-and-load) frames chaos testing alongside load testing as the two ways you validate a distributed system before production does it for you.)

## 13. How to reach for this (and when not to)

Chaos engineering is powerful, but it has a cost and a set of prerequisites, and there are situations where running it is reckless or simply not worth it. Here is my decisive guidance.

**Reach for chaos engineering when:**

- You have **real resilience mechanisms** (failover, breakers, fallbacks, autoscaling) that are currently *unproven*. Untested resilience is exactly what this discipline exists to validate.
- You have **observability and good SLIs** so you can define a steady state and watch it. This is non-negotiable.
- The system is **important enough** that an undiscovered failure mode would cause real pain — user-facing services, revenue paths, anything with an SLO.
- Your team is **mature enough** to respond calmly and abort cleanly. Start with tabletop game days if they are not yet.

**Do NOT reach for chaos engineering when:**

- You have **no observability**. Chaos without monitoring is just breaking things. Fix the monitoring first; there is no exception to this.
- You have **not built the resilience patterns yet**. If you have no failover, killing a node will obviously take you down — you have nothing to test. Build the patterns, *then* validate them. Chaos engineering proves resilience exists; it does not create it.
- You **cannot stop the experiment**. If there is no working abort path, do not inject the fault. An unstoppable experiment is an outage with a fancy name.
- The system is a **low-stakes internal batch job** where a failure costs nothing and nobody is paged. The effort of a chaos program is not free; spend it where unreliability actually hurts. Do not gold-plate a cron job that runs once a night and can simply re-run.
- You are tempted to run it **at 3am or during peak traffic** to "make it realistic." No. The realism you want is in the *fault*, not in the *timing*. Inject the realistic fault during business hours with the team watching. Deliberately running chaos at the worst time defeats the entire purpose, which is to move the failure to the best time.
- The organization will treat a discovered weakness as a **failure to be punished** rather than a bug found safely. Chaos engineering is only safe inside a blameless culture; if finding a weakness gets someone blamed, people will quietly avoid running experiments and you lose the whole benefit. (This ties directly to [the blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) — the same culture that makes postmortems honest makes chaos experiments worth running.)

The meta-rule: chaos engineering is an investment, and like any reliability investment it should be sized to the cost of the unreliability it prevents. A payments service that loses revenue and trust on every outage justifies a serious chaos program. An internal dashboard that a handful of people refresh twice a day does not. Spend your effort where the error budget — and the user pain — actually lives.

## 14. Stress-testing your chaos practice

Before you call your chaos practice mature, pressure-test it against the hard questions, the same way you would stress-test any operational decision:

- **What if the dependency is down for two hours, not two minutes?** Your fallback might serve fine for a short blip but degrade unacceptably over a sustained outage (stale cache becomes dangerously stale). Run a *long-duration* fault, not just a momentary one, to find the difference between "survives a blip" and "survives an outage."
- **What if the on-call is asleep and does not ack the page?** A live game day at 10am tests the response with everyone alert. But the real failure mode is the unacked page at 4am. Test your *escalation* path — does the page escalate to a secondary on-call after N minutes of no ack? A game day can deliberately have the primary not respond to verify the escalation fires. (The humane-on-call mechanics are in [designing a humane on-call](/blog/software-development/site-reliability-engineering/designing-a-humane-on-call).)
- **What if two incidents overlap?** Your chaos experiment is itself an "incident." What happens if a real incident starts *during* the game day? You need a pre-agreed rule: real incident takes priority, abort the experiment immediately, restore, and stand down. This is why you never run a game day during a deploy freeze or an active incident.
- **What if the abort button does not work?** This is the nightmare. Mitigate it by *testing the abort path before every experiment*, by giving every experiment an automatic max-duration timeout (the `duration: "10m"` in the Chaos Mesh spec), and by starting at a blast radius small enough that even a failed abort is survivable. Defense in depth applies to your safety mechanisms too.
- **What if the experiment reveals a weakness you cannot fix quickly?** Good — that is information. File it with an owner and a date, raise its priority based on the real risk it represents, and *do not run the larger-blast-radius version* until it is fixed. A known weakness with a tracked fix is infinitely better than an unknown one waiting for 3am.

The practice that survives these questions is one where safety is layered (small scope + abort button + auto-timeout), where the experiment is treated with the same seriousness as a real incident, and where every finding becomes a tracked action item with an owner. That last part is what separates a chaos *practice* from a chaos *stunt*: the stunt breaks something impressive and moves on; the practice closes the loop by fixing what it finds and re-running to prove the fix.

## 15. Key takeaways

1. **Untested resilience is a hypothesis, not a fact.** Your failover, breakers, retries, and fallbacks are beliefs until an experiment turns them into evidence. The design review is not the test.
2. **Apply the scientific method: steady state, hypothesis, inject, observe.** Define a measurable normal (the SLI), predict that it will hold, inject a real fault, and watch. The SLI is your truth oracle.
3. **A "failed" experiment is a success.** Finding a weakness in a controlled window — in daylight, with an abort button — is the entire point. The experiments that always pass should make you suspicious, not happy.
4. **Control the blast radius above all else.** Start tiny, run in staging first, arm a tested abort button with a pre-committed condition, run during business hours with the team watching, and never run an experiment you cannot stop.
5. **Game days test the team, not just the system.** They exercise the page, the on-call, the runbook, and the incident response together — the parts an automated experiment cannot reach. Run tabletop drills to start, live game days as you mature.
6. **You need observability and the patterns first.** Chaos without monitoring is just breaking things; chaos validates resilience but does not create it. Both are hard prerequisites.
7. **Climb the maturity ladder: staging, then game days, then continuous prod chaos.** Earn each rung. Continuous chaos turns resilience into an invariant that cannot silently rot, but it demands real maturity.
8. **Chaos validates the whole resilience stack together.** One fault tests failover, breakers, retry budgets, fallbacks, autoscaling, and on-call response at once — including the interaction bugs you cannot find by testing components in isolation.
9. **Size the investment to the pain.** Run a serious program where unreliability costs real money and trust; do not gold-plate a low-stakes batch job.
10. **It only works inside a blameless culture.** If a discovered weakness gets someone blamed, people stop running experiments and you lose everything. Find bugs, fix them, re-run, repeat.

## 16. Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series map; where chaos engineering fits in the define → measure → budget → respond → learn → engineer loop.
- [Redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) — the failover patterns whose claims chaos experiments validate (and whose untested standbys they expose).
- [The anatomy of an incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident) — the incident lifecycle a live game day rehearses with the stakes turned down.
- [Timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — the retry-amplification math behind the thundering-herd failure mode chaos surfaces.
- The sibling Track E posts `circuit-breakers-bulkheads-and-load-shedding` and `graceful-degradation-and-fallbacks` — the breaker and fallback mechanisms a latency-injection experiment proves out.
- [Hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) — the same falsification discipline, run reactively in debugging; chaos engineering runs it proactively.
- [Testing distributed systems: chaos and load](/blog/software-development/system-design/testing-distributed-systems-chaos-and-load) — the system-design framing of chaos testing alongside load testing.
- *Principles of Chaos Engineering* (principlesofchaos.org) and the Netflix tech blog on the Simian Army — the canonical sources for the discipline and Chaos Monkey/Gorilla/Kong.
- The Google SRE Workbook, chapters on testing reliability and on canarying releases — the complementary "test before you ship" practices that pair with production chaos.
