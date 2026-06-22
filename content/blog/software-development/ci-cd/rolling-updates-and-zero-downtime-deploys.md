---
title: "Rolling Updates and Zero-Downtime Deploys: The Half-Dozen Knobs That Decide Whether a Release Drops Requests"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Zero-downtime deploys are a solved problem in Kubernetes, but only if you configure readiness probes, graceful shutdown, surge and unavailability, and a disruption budget correctly. This is the field guide to every knob and its failure mode."
tags:
  [
    "ci-cd",
    "devops",
    "kubernetes",
    "zero-downtime",
    "rolling-update",
    "readiness-probe",
    "graceful-shutdown",
    "pod-disruption-budget",
    "deployment",
    "reliability",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/rolling-updates-and-zero-downtime-deploys-1.png"
---

A team I worked with shipped a perfectly ordinary web service. Green build, green tests, image signed and scanned, `kubectl rollout` reported success in about ninety seconds. The dashboard said the deploy was clean. And yet, like clockwork, every single release produced a forty-second smear of `502 Bad Gateway` and `connection reset` errors in the edge logs, somewhere around 2% of all requests for the duration of the roll. Nobody noticed for months because the deploys happened at 3 a.m. and the error rate was "within noise." Then the team moved to deploying thirty times a day in business hours, and suddenly that 2% smear was happening during peak traffic, paging the on-call engineer, and quietly eroding a chunk of the error budget on every push. The deploy *looked* done. It was not done. It had never been done.

That gap — between "the rollout reported success" and "not a single client noticed we shipped" — is the entire subject of this post. The good news, and I want to say this plainly up front, is that **zero-downtime deployment is a solved problem on Kubernetes**. You do not need a service mesh, a sidecar, or a clever proxy to ship a new version of a stateless HTTP service without dropping a request. The platform gives you everything you need out of the box. The bad news is that it gives you everything as a set of independent knobs, each defaulting to a value that is *fine for a demo and wrong for production*, and each with its own quiet failure mode. Get all of them right and your deploys are invisible. Get any one of them wrong and every deploy is a mini-outage that hides inside your noise floor until traffic grows enough to make it a page.

This post is the field manual for those knobs. We are going to walk the four things that must happen for a request never to be dropped during a rollout — **bring up new pods, prove they are ready, shift traffic to them, and drain the old pods gracefully** — and for each one we will find the knob that controls it and the bug that bites you when the knob is wrong. By the end you will be able to read a `Deployment` spec and say, with confidence, "this will drop requests on every deploy, and here is the exact line that does it." Figure 1 lays out the mechanism we are about to take apart: how a single `kubectl apply` of version two becomes a careful, capacity-guarded handoff from the old `ReplicaSet` to the new one.

![Diagram of the Kubernetes rolling update mechanism showing the old ReplicaSet scaling down and the new ReplicaSet scaling up while a readiness gate and the maxSurge and maxUnavailable guards keep the Service endpoints full of healthy pods](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-1.png)

This sits exactly one layer below progressive delivery. A rolling update is the **baseline** every other deploy strategy builds on — canary and blue-green, which we will tie to at the end, are just smarter ways of *controlling* the same primitive of bringing up new pods and shifting traffic. If your rolling update drops requests, your canary will drop requests too; it will just drop fewer of them. So we fix the baseline first. This is the toolchain layer of the commit-to-production spine: the *deploy* step, engineered so that the *operate* step that follows it is boring. For the wider mental model of where this fits, see the series intro, [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model); for the reliability theory of *why* gating a deploy on health is the safe move, the SRE post on [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) is the companion piece.

## 1. What "zero downtime" actually means (and what it does not)

Let us define the target precisely, because "zero downtime" gets thrown around loosely and the loose version is achievable while the precise version takes work.

The loose version is "the service stays up." That is easy: run more than one replica and the service is up even while one pod is being replaced. The precise version — the one your users actually experience — is **"no client request receives an error or a dropped connection that it would not have received if no deploy were happening."** That is a much stronger claim. It means:

- No request is routed to a pod that cannot yet serve it (a pod that is still booting, still loading its model, still warming its connection pool, still running migrations).
- No in-flight request is severed when its pod is told to shut down.
- No new request is routed to a pod that has *begun* shutting down and will refuse it.
- The total serving capacity never dips so low that healthy pods get overwhelmed and start shedding load.

Notice that three of those four are about *individual requests*, not about *the service*. A service can be "100% up" by any external uptime monitor that pings once a minute, while simultaneously dropping 2% of real user requests during every deploy. The uptime monitor and the user are measuring different things. Zero-downtime engineering is about the user's measurement.

There is an important honesty here: **zero downtime is only fully achievable for workloads where a request can be safely retried or rerouted.** A stateless HTTP API is the easy case and the one we will spend most of our time on. A long-lived WebSocket or gRPC stream is harder — you cannot finish an infinite stream during a thirty-second grace period, so "zero downtime" for those becomes "graceful reconnection," which is a client-side contract as much as a server-side one. A stateful singleton (a leader, a database primary) is harder still and usually wants a different strategy entirely. We will note where the techniques stop applying. But for the enormous middle of the world — stateless services behind a load balancer — every dropped request during a deploy is a *self-inflicted* wound, and every one of them is fixable with configuration you already have access to.

The unit we will measure throughout is **dropped requests per release**: count the 5xx responses and connection resets attributable to the deploy window, divided by total requests in that window. A clean deploy is `0`. Our opening team was at roughly 2%. We are going to walk that to zero.

A note on *how to measure it honestly*, because this is the part most teams get wrong and it is the reason deploy-time errors hide for months. The errors a deploy causes do not look like a single clean spike; they look like a low-amplitude smear that blends into normal noise. To see them you must (a) tag every error with whether it falls inside a deploy window — annotate the deploy time on your `Deployment` and join it against your error metrics — and (b) measure from the *client edge*, not from inside the cluster, because the most damaging errors (connection refused, connection reset) often never reach your application logs at all; they happen at the connection layer, before any handler runs, and only the load balancer or the client sees them. A common false sense of safety comes from looking at application-level request logs and seeing nothing wrong — of course you see nothing, the dropped requests never made it to a handler to be logged. The honest signal is at the ingress or the external load balancer: 5xx-by-upstream and TCP resets, bucketed by deploy window. If you only have application logs, you are measuring the requests that *succeeded enough to be logged* and missing exactly the ones you are trying to eliminate.

## 2. The rolling update mechanism: how Kubernetes swaps pods

Start with the primitive. When you change the pod template of a `Deployment` — a new image tag, a changed environment variable, anything that alters the pod spec — Kubernetes does **not** delete all the old pods and create new ones. That would be a hard cutover with a guaranteed gap. Instead, the default `RollingUpdate` strategy creates a **new `ReplicaSet`** for the new pod template and then performs a careful dance: it scales the new `ReplicaSet` up a little, waits for those pods to become ready, scales the old `ReplicaSet` down a little, and repeats until the new one is at full size and the old one is at zero.

A `ReplicaSet`, if the term is new to you, is the controller that owns a set of identical pods and keeps exactly *N* of them running. A `Deployment` is a higher-level object that manages `ReplicaSet`s for you precisely so it can do this rolling handoff: at any moment during a roll there are *two* `ReplicaSet`s, the old one shrinking and the new one growing, and the `Deployment` controller is the choreographer adjusting both.

The choreography is bounded by two numbers, and these are the first two knobs. Both live under `spec.strategy.rollingUpdate`:

- **`maxSurge`** — how many pods *above* the desired replica count may exist during the roll. This is your *capacity headroom*. With `replicas: 4` and `maxSurge: 1`, the controller may run up to five pods briefly, so it can bring a new pod fully up *before* it needs to take an old one down.
- **`maxUnavailable`** — how many pods *below* the desired count may be unavailable during the roll. This is your *capacity floor*. With `maxUnavailable: 0`, the controller is forbidden from ever having fewer than four ready pods; it must surge a new ready pod *before* it removes an old one.

These two knobs together set the *shape* of the roll. The defaults are `maxSurge: 25%` and `maxUnavailable: 25%`, which for four replicas rounds to one each. That is a reasonable default for cost-sensitive workloads, but read what `maxUnavailable: 25%` actually permits: during the roll, your service may be running at 75% of its capacity. If you are sized so that 75% capacity cannot absorb peak load, the deploy itself becomes a load-shedding event — the remaining pods get overwhelmed and start returning errors, and you blame the new version when the cause was the deploy strategy.

For true zero downtime the canonical setting is **`maxUnavailable: 0` and `maxSurge: 1` (or higher)**. This says: never let capacity dip, always pay for a little extra capacity during the roll instead. The cost is real — you need enough cluster headroom (or a fast enough autoscaler) to schedule the surge pods — but it is the only setting that guarantees the capacity floor never drops.

Figure 1 above shows the resulting flow: the `Deployment` of v2 spawns the new `ReplicaSet` scaling 0→4 and instructs the old one to scale 4→0, with `maxSurge` providing the brief 5-pod peak and `maxUnavailable: 0` forbidding any gap, while the readiness gate ensures only proven-ready pods ever join the Service.

It is worth contrasting the `RollingUpdate` strategy with its only alternative, `Recreate`, because the contrast makes the design intent obvious. With `strategy.type: Recreate`, Kubernetes deletes *all* the old pods first, waits for them to be gone, and only then creates the new ones. That guarantees no two versions run at once — useful for a workload that genuinely cannot tolerate v1 and v2 coexisting (a schema-incompatible migration, a singleton that must not double-run) — but it also guarantees a *gap*: there is a window with zero pods, and every request in that window is dropped. `Recreate` is a deliberate trade of availability for version-exclusivity. For a stateless service you almost never want it; the entire reason `RollingUpdate` exists is to avoid that gap. Knowing `Recreate` is there, though, clarifies what `RollingUpdate` is buying you: overlapping versions in exchange for continuous availability. (If your two versions truly cannot overlap, the answer is usually not `Recreate` — it is to make them *able* to overlap via backward-compatible schema changes, the discipline of [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations), so you can keep rolling.)

There is one more mechanic to internalize about the two `ReplicaSet`s: the old one does not get *deleted* when the roll completes — it gets *scaled to zero* and kept around as a revision. This is what makes `kubectl rollout undo` fast: rolling back is just scaling the old `ReplicaSet` back up and the new one down, the exact same rolling dance in reverse, with the same `maxSurge`/`maxUnavailable` guards. The `Deployment` keeps the last `revisionHistoryLimit` (default 10) `ReplicaSet`s parked at zero replicas precisely so any of those revisions can be re-promoted without a rebuild. This is the same "build once, promote everywhere" idea the series rests on, expressed at the runtime layer: the artifact you roll back to is the *exact* one you tested and shipped before, not a rebuild that might drift.

Here is a `Deployment` with the strategy set for zero downtime. We will add probes and shutdown handling to this same spec as we go, building it up piece by piece.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
  labels:
    app: checkout
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # one extra pod may exist during the roll
      maxUnavailable: 0    # never run below 4 ready pods -> no capacity gap
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:v2
          ports:
            - containerPort: 8080
```

#### Worked example: the arithmetic of surge and unavailable

Suppose `replicas: 10`, peak load needs at least 8 ready pods to stay under your latency SLO, and a fresh pod takes 25 seconds to become ready. Compare three strategies for a full roll:

- `maxUnavailable: 25%, maxSurge: 25%` (defaults, rounds to 2 and 2): the controller may drop to 8 ready pods and surge to 12. At peak that 8-pod floor is *exactly* your minimum — one slow pod and you breach the SLO. The roll proceeds in batches of ~2, so roughly $\lceil 10/2 \rceil = 5$ waves; at 25s per wave plus scheduling, the roll takes on the order of 2–3 minutes.
- `maxUnavailable: 0, maxSurge: 2`: the floor is pinned at 10 ready pods (no SLO risk at all), the ceiling is 12. The controller surges 2, waits for ready, retires 2, repeats — about 5 waves, similar wall-clock time, but capacity *never* dips. Cost: you must be able to schedule 2 extra pods.
- `maxUnavailable: 0, maxSurge: 50%` (5): floor pinned at 10, ceiling 15, only ~2 waves, fastest roll. Cost: 5 extra pods of headroom for the duration.

The lesson is that `maxUnavailable: 0` is *free of risk* and `maxSurge` is the dial that trades cluster cost for roll speed. If you have headroom, raise `maxSurge` to make rolls fast; if you are tight on cluster capacity, keep `maxSurge` low and accept a slower roll — but keep `maxUnavailable` at 0 either way, because that is the knob that protects the user. Setting `maxUnavailable` above zero to "speed up the deploy" is the single most common way teams reintroduce a capacity gap they did not mean to.

| Strategy | Capacity floor | Cluster cost during roll | Roll speed | Use when |
| --- | --- | --- | --- | --- |
| `maxUnavailable: 25%, maxSurge: 25%` | 75% (risky at peak) | none extra | medium | cost-sensitive, off-peak, generous SLO |
| `maxUnavailable: 0, maxSurge: 1` | 100% | +1 pod | slow | tight cluster, true zero-downtime |
| `maxUnavailable: 0, maxSurge: 50%` | 100% | +50% pods | fast | headroom available, want quick rolls |

## 3. Readiness probes: the linchpin of zero downtime

Here is the single most important sentence in this post: **a pod is not sent traffic until its readiness probe passes.** Everything about zero downtime rests on this one fact, and the number-one zero-downtime bug in the world is the absence — or the dishonesty — of a readiness probe.

To see *why* readiness is the gate, you need a quick picture of how traffic actually reaches a pod. A `Service` is a stable virtual IP that fronts a changing set of pods. Behind it, Kubernetes maintains an `EndpointSlice` object — the live list of pod IPs that are *eligible to receive traffic*. The kubelet on each node reports each pod's readiness; when a pod's readiness probe passes, its IP is *added* to the `EndpointSlice`, and when readiness fails (or the pod begins terminating) its IP is *removed*. Every component that routes traffic — kube-proxy programming iptables or IPVS rules on each node, the in-cluster DNS, the ingress controller, an external cloud load balancer — watches that `EndpointSlice` and updates its own routing tables to match. So "a pod gets traffic" literally means "the pod's IP is in the `EndpointSlice`," and *that* is governed entirely by the readiness probe. The readiness probe is not a suggestion the load balancer consults; it is the mechanism that puts a pod into, or removes it from, the set of addresses traffic can land on. This is also why the endpoint-removal race in section 5 exists: removal from the `EndpointSlice` is fast, but the *propagation* of that removal out to every routing component is eventually consistent.

Kubernetes has three probes, and they are constantly confused. Figure 2 lays out exactly what each one asks and what happens when it fails, because the confusion is the source of half the deploy bugs in the field.

![Matrix comparing the Kubernetes readiness, liveness, and startup probes by the question each asks, the action taken on failure, and the role each plays in a zero-downtime deploy](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-2.png)

- **Readiness probe** answers *"am I ready to serve traffic right now?"* When it fails, the pod is **removed from the Service endpoints** — no traffic is routed to it — but the container is *not* restarted. This is the gate. A pod that has started but not yet finished warming up reports "not ready," stays out of the load-balancer rotation, and joins it only when it can actually serve. During a rollout, this is what stops a still-booting pod from receiving (and failing) requests.
- **Liveness probe** answers *"am I hung and need to be killed?"* When it fails enough times, Kubernetes **restarts the container**. This is for deadlocks and unrecoverable states — the app is running but wedged. Liveness has *nothing to do with traffic routing*; a pod can be "live" (not restarted) while "not ready" (not serving), which is exactly the state of a healthy pod that is busy or warming up.
- **Startup probe** answers *"is the app still booting?"* While it has not yet succeeded, the liveness and readiness probes are *disabled*. This exists so a slow-booting app (one that loads a large model, runs migrations, or warms a cache) is not killed by an impatient liveness probe before it has finished starting. Once the startup probe passes once, it never runs again and the other two take over.

The mental model worth burning in: **readiness controls traffic, liveness controls restarts, startup controls when liveness arms.** Mixing these up causes specific, diagnosable failures. The most damaging is having *no readiness probe at all*.

A practical aside on *how* probes check, because the mechanism choice has zero-downtime consequences. Each probe can be an `httpGet` (Kubernetes makes an HTTP request and treats 2xx/3xx as success), a `tcpSocket` (it just checks the port opens), an `exec` (it runs a command in the container and checks the exit code), or a `grpc` probe (it calls the gRPC health-checking protocol). The trap is the `tcpSocket` readiness probe: a port opening is *not* the same as the app being able to serve — a framework can bind the port before it has loaded its routes or warmed its pool, so a `tcpSocket` readiness probe is a textbook "probe that lies." For readiness, prefer `httpGet` against a real readiness endpoint that exercises the serving path, or `grpc` for gRPC services. Reserve `tcpSocket` for cases where you genuinely have nothing better. And note that probes run *from the kubelet on the pod's node*, on the schedule you set — they are not free; an `httpGet` readiness probe every second against an endpoint that does a database round-trip is a surprising amount of database load across a large fleet, which is why caching the dependency check inside the readiness handler matters.

### What happens with no readiness probe

When a container has no readiness probe, Kubernetes considers it ready the moment the container *process starts* — not when the application is listening, not when it can serve, just when the process exists. So during a rollout, the new pod is added to the Service endpoints essentially immediately, while the application inside is still booting: still binding the port, still loading config, still establishing its database pool. Traffic arrives. The application is not listening yet, so the kube-proxy or load balancer gets a connection refused, which surfaces to the client as a `502` or a reset. For the few seconds each new pod takes to actually start serving, it is a black hole for whatever fraction of traffic gets routed to it.

That is precisely the bug from our opening story. Figure 3 contrasts the two worlds side by side: without the probe, the cold pod is added to endpoints immediately and eats requests; with it, the pod is held out of rotation until it can genuinely serve.

![Before and after comparison showing that without a readiness probe a cold pod is added to the Service endpoints immediately and returns 502 errors, while with a readiness probe the pod is held out of rotation until the probe passes and zero requests are dropped](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-3.png)

Here is the same `checkout` deployment with all three probes added. Read the timing numbers carefully — they are knobs in their own right.

```yaml
spec:
  template:
    spec:
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:v2
          ports:
            - containerPort: 8080
          # Startup: give a slow boot up to 30 x 2s = 60s before liveness arms.
          startupProbe:
            httpGet:
              path: /healthz/startup
              port: 8080
            periodSeconds: 2
            failureThreshold: 30
          # Readiness: gate traffic. Checks downstream deps the request needs.
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: 8080
            periodSeconds: 5
            timeoutSeconds: 2
            successThreshold: 1
            failureThreshold: 3
          # Liveness: restart only a truly hung process. Generous + slow.
          livenessProbe:
            httpGet:
              path: /healthz/live
              port: 8080
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 6
```

Three endpoints, three meanings, and the discipline that makes this work is **keeping them genuinely distinct**:

- `/healthz/live` returns `200` as long as the process is not deadlocked. It should be *cheap and dependency-free* — it must not check the database, because if the database blips you do *not* want every pod to fail liveness and restart-loop in unison, turning a transient dependency blip into a full self-inflicted outage. Liveness checks *the process*, nothing downstream.
- `/healthz/ready` returns `200` only when the pod can actually serve a real request: the HTTP server is listening, the connection pool is warm, and the critical downstream dependencies it *needs* are reachable. This one *may* check dependencies, but carefully — see the next subsection.
- `/healthz/startup` returns `200` once the app has finished its one-time boot work (migrations applied, caches loaded, port bound). It exists so the long boot does not trip liveness.

### The readiness probe that lies

The second-most-common readiness bug is a probe that *passes too early* — a probe that lies. The classic version is a readiness endpoint that returns `200` as soon as the web framework is up, *before* the application has loaded the data or warmed the connections it needs to serve a real request. The pod reports ready, joins the rotation, gets real traffic, and fails it — because "the framework is up" is not the same as "I can serve." You have a readiness probe and you still drop requests, which is maddening to debug because the obvious culprit (no probe) is present.

The fix is to make the readiness endpoint *check what a real request depends on*, not what is convenient to check. If a request to this service needs a database connection, the readiness endpoint should verify it has a live connection from the pool (a fast `SELECT 1`, cached for a second or two so the probe itself does not hammer the database). If it needs a downstream service, it should confirm that client is initialized. The principle: **the readiness probe should pass if and only if the next real request would succeed.** A probe that checks less than that is a probe that lies, and a lying probe is barely better than no probe at all.

There is a subtle counter-balance here, and it is worth stating because over-correcting causes a different outage. A readiness probe that checks a *non-critical, shared* dependency can cause **correlated readiness flapping**: if every pod's readiness checks the same downstream service and that service has a 200ms hiccup, every pod simultaneously reports not-ready, the Service endpoint list goes empty, and your *entire* service goes dark for the duration of a blip that any single request could have tolerated. So the rule has a refinement: readiness should check the dependencies a request *cannot proceed without*, and should *not* check optional or gracefully-degradable ones. For everything degradable, serve a fallback instead of failing readiness — which is exactly the discipline covered in the SRE post on [graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks). Readiness is a binary "can I serve at all"; degradation handles "can I serve fully."

## 4. Health-check tuning: the probe that restart-loops a healthy pod

The probe *timing* knobs are their own source of incidents, and the failure modes are not obvious from the field names. Let us go through them, because the defaults will eventually bite a slow-but-healthy service.

Each probe has these dials:

- `initialDelaySeconds` — how long to wait after the container starts before probing at all. (Largely obviated by a proper `startupProbe`, which is the better tool for slow boots.)
- `periodSeconds` — how often to probe.
- `timeoutSeconds` — how long a single probe may take before it counts as a failure.
- `failureThreshold` — how many consecutive failures before the action (remove-from-endpoints for readiness, restart for liveness) fires.
- `successThreshold` — how many consecutive successes before "passing" (must be 1 for liveness and startup).

The dangerous combination is an **aggressive liveness probe on a slow-but-healthy service**. Picture a service that, under load, occasionally takes 3 seconds to respond to its liveness endpoint because the event loop is briefly busy — but it is not hung, it is *working*. If the liveness probe has `timeoutSeconds: 1` and `failureThreshold: 3` with `periodSeconds: 5`, then three slow responses in fifteen seconds — entirely possible under a load spike — and Kubernetes *restarts the container*. Now you have removed a busy-but-healthy pod from a service that was already under load, which makes the remaining pods busier, which makes *their* liveness probes slower, which restarts *them*. This is a **liveness-induced restart cascade**: the probe meant to protect you tears the service down precisely when it is under the most stress. The fix is to make liveness *generous and cheap*: a high `failureThreshold`, a dependency-free endpoint, and timing that only fires on a genuine multi-second hang, never on a transient busy period.

A small comparison of how each probe should be tuned:

| Probe | Endpoint cost | Timing posture | What failure should mean |
| --- | --- | --- | --- |
| Readiness | may check critical deps (cached) | moderate, responsive | "do not send me traffic right now" |
| Liveness | cheap, process-only, no deps | generous, slow to fire | "I am genuinely deadlocked, restart me" |
| Startup | the boot-completion signal | long total budget | "I am still booting, do not arm liveness" |

The deeper point: **liveness is a loaded gun pointed at your own pods.** A liveness probe that fires too easily does not improve availability; it *reduces* it, because most "unhealthy" pods are actually "busy" or "waiting on a transient dependency," and restarting them is worse than leaving them alone. A safe default is to have *no* liveness probe at all for many services and rely on readiness plus the process crashing on its own when truly broken — and to add liveness only when you have a *specific, known* deadlock it cannot recover from. Readiness keeps a struggling pod out of rotation without killing it; liveness should be reserved for the rare unrecoverable hang.

## 5. Graceful shutdown: the other half of zero downtime

Bringing pods up correctly is half the job. The other half is taking pods *down* correctly, and this is where even teams with perfect readiness probes still drop requests — because the shutdown path has its own sequence, its own race, and its own knob.

When the rolling update decides to retire an old pod (or a node is drained, or you scale down), Kubernetes does the following, and the order and the *simultaneity* matter enormously:

1. The pod is marked `Terminating`.
2. **Two things happen in parallel, at the same instant:** (a) the pod is removed from the Service endpoints (so *new* traffic stops being routed to it), and (b) the container receives a `SIGTERM` signal and, if defined, its `preStop` hook fires.
3. Kubernetes waits up to `terminationGracePeriodSeconds` (default **30**) for the container to exit on its own.
4. If the container has not exited by then, it receives `SIGKILL` — a hard, ungraceful kill.

Figure 4 shows this lifecycle on a timeline, with the critical detail that step 2's two halves are concurrent, not sequential.

![Timeline of the pod termination lifecycle showing that SIGTERM and endpoint removal fire at the same instant, followed by a preStop sleep that lets the load balancer catch up, then in-flight request draining, then process exit or a SIGKILL at the end of the grace period](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-4.png)

There are two distinct bugs hiding in this sequence, and you usually have both.

### Bug one: the app ignores SIGTERM and drops in-flight requests

If your application does not catch `SIGTERM`, the default behavior of most runtimes is to terminate the process more or less immediately — and any requests it was in the middle of serving are severed mid-response, surfacing to clients as truncated bodies or connection resets. Even if no *new* traffic is arriving (endpoint removal handled that), the requests that were already in flight when `SIGTERM` arrived get dropped.

The fix is **graceful shutdown in the application**: catch `SIGTERM`, stop accepting new connections, let in-flight requests finish, then exit. Every mature HTTP framework supports this. Here is the pattern in Go, which makes the control flow explicit:

```go
package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	srv := &http.Server{Addr: ":8080", Handler: newHandler()}

	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	}()

	// Block until we get SIGTERM (what Kubernetes sends on pod termination).
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	<-stop

	// Give in-flight requests a bounded window to finish. Must be < the pod's
	// terminationGracePeriodSeconds, leaving room for the preStop sleep too.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// Shutdown stops accepting new connections and waits for active ones to drain.
	if err := srv.Shutdown(ctx); err != nil {
		// Drain timed out; some connections were force-closed.
		os.Exit(1)
	}
}
```

The equivalent in Node.js with Express is `server.close(callback)`, which stops accepting connections and fires the callback once existing ones drain; in Python, a Gunicorn or Uvicorn worker handles `SIGTERM` and finishes in-flight requests if you set a sensible `--graceful-timeout`. The shape is always the same: **trap the signal, stop the front door, drain the room, leave.** The one budget you must respect: your in-app drain timeout must be *shorter* than `terminationGracePeriodSeconds`, or `SIGKILL` will interrupt your graceful drain and you are back to dropping requests.

### Bug two: the endpoint-removal race

This is the subtle one, and it survives even a perfect SIGTERM handler. Recall that endpoint removal and `SIGTERM` fire *at the same instant*. But endpoint removal is not instantaneous across the cluster — it is *eventually consistent*. The sequence is: the pod is removed from the `Endpoints`/`EndpointSlice` object, the API server notifies every kube-proxy and every ingress controller and every external load balancer, and *each of those* updates its own routing tables. That propagation takes time — typically a few hundred milliseconds to a couple of seconds, occasionally longer for an external cloud load balancer.

During that propagation window, the pod is `Terminating`, has *already* received `SIGTERM`, may have *already* begun shutting down its server — and yet some load balancers still have it in their routing table and are still sending it new requests. Those new requests arrive at a server that is closing its front door, and they get refused. You did everything right in the app and you *still* drop requests, because the app shut down *faster* than the cluster could stop routing to it.

The standard fix is a **`preStop` hook with a small sleep**. The `preStop` hook runs *before* `SIGTERM` is delivered to the main process (Kubernetes runs `preStop`, and only when it completes does it send `SIGTERM`). A `preStop` that just sleeps for a few seconds creates exactly the delay needed: endpoint removal has already been *triggered* (it happens in parallel with the start of termination), so during the `preStop` sleep the propagation completes — every load balancer learns the pod is gone — and only *then* does the app receive `SIGTERM` and begin draining. By the time the app stops accepting connections, nothing is routing to it anymore.

Here is the full pod spec for graceful shutdown, combining the `preStop` sleep and the grace period:

```yaml
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 45   # total budget before SIGKILL
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:v2
          lifecycle:
            preStop:
              exec:
                # Wait for LB/endpoint propagation BEFORE the app gets SIGTERM.
                # The app then has (45 - 10) = 35s to drain in-flight requests.
                command: ["sh", "-c", "sleep 10"]
```

The budget arithmetic must hold together: `terminationGracePeriodSeconds` (45) must exceed the `preStop` sleep (10) *plus* the app's worst-case in-flight drain time (which we set to 20s in the Go handler, leaving comfortable margin). If `preStop` sleep + app drain exceeds the grace period, `SIGKILL` interrupts you mid-drain. A good rule: `gracePeriod = preStopSleep + maxRequestDuration + buffer`.

An alternative to the `preStop` sleep, if you prefer not to add fixed latency to every shutdown, is to have the app **keep serving for a few seconds after receiving SIGTERM** — flip readiness to fail (which speeds endpoint removal) but *keep accepting* connections briefly, then drain. This is more code but avoids a blunt sleep. For most teams the `preStop sleep 5-10` is simpler, robust, and worth the few seconds of added shutdown time. Either way you are solving the same race: **the load balancer must learn the pod is gone before the pod stops answering.**

#### Worked example: how many requests a missing SIGTERM handler drops

Put numbers on the shutdown bug so its scale is concrete. Suppose `checkout` serves 2,000 requests/second across 6 pods — roughly 333 req/s per pod — with an average request duration of 150ms, so at any instant each pod has about $333 \times 0.15 \approx 50$ requests in flight. During a roll with `maxSurge: 2`, the controller retires 2 old pods at a time. If those pods exit *immediately* on `SIGTERM` (no graceful handler), each one severs its ~50 in-flight requests. Over a full roll of 6 pods retired in 3 waves of 2, that is $6 \times 50 = 300$ severed requests per deploy. At 30 deploys/day that is 9,000 dropped requests *every day*, each one a real user seeing a connection reset — and none of it shows up in `kubectl rollout status`, which reported success every time. Add a `SIGTERM` handler that drains for up to 20 seconds and those 300 in-flight requests finish normally; the count drops to whatever was still arriving during the endpoint-removal window, which the `preStop` sleep then takes to zero. The arithmetic is the argument: graceful shutdown is not a nicety, it is the difference between zero and thousands of dropped requests a day, and the busier the service the larger the number.

## 6. Pod Disruption Budgets: surviving voluntary disruptions

Everything so far protects you during a *deploy* you initiated. But pods also go away for reasons you did not initiate from a `Deployment` change — and the most dangerous of these is a **node drain**, when a cluster operator (or an autoscaler, or a managed-Kubernetes upgrade) cordons a node and evicts every pod on it so the node can be rebooted, patched, or removed.

Kubernetes distinguishes two kinds of disruption, and the distinction is the whole point of this section. Figure 5 lays out the taxonomy.

![Tree diagram classifying pod disruptions into voluntary disruptions such as node drains and cluster upgrades that a PodDisruptionBudget can bound, and involuntary disruptions such as node out-of-memory events and hardware loss that only replica count and spread can mitigate](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-5.png)

- **Voluntary disruptions** are deliberate, controlled evictions: draining a node for maintenance, a rolling cluster upgrade, scaling down a node pool, rebalancing. These go through the **Eviction API**, and the Eviction API *respects* a `PodDisruptionBudget`.
- **Involuntary disruptions** are the things you cannot schedule: a node kernel-panics, runs out of memory, loses power, or its underlying hardware fails. These do *not* go through the Eviction API and a `PodDisruptionBudget` does *not* protect against them — for those you need enough replicas and good anti-affinity spread so that losing one node does not lose too many pods.

A **`PodDisruptionBudget` (PDB)** is a small object that says, in effect, "you may voluntarily evict my pods, but never below this availability floor." It has two mutually-exclusive ways to express the floor:

- `minAvailable` — at least this many (or this percentage of) pods must remain available. `minAvailable: 80%` of 10 replicas means at most 2 may be evicted at once.
- `maxUnavailable` — at most this many (or percentage) may be unavailable. `maxUnavailable: 2` is equivalent for 10 replicas.

When a node drain tries to evict a pod that would breach the PDB, the Eviction API *refuses the eviction* and the drain *blocks and waits* until enough replacement pods have been scheduled and become ready elsewhere — at which point the eviction is allowed and the drain proceeds. The PDB turns a "kill them all now" into a "kill them as fast as availability allows."

### The bug: the cluster upgrade that killed the service

Without a PDB, here is the classic incident. A managed-Kubernetes upgrade rolls the node pool: it drains node 1 (evicting the 3 pods of your 4-replica service that happened to land there — because you also lacked anti-affinity spread), then node 2, and so on. When node 1 drains, three of your four pods vanish at once, the service is at 25% capacity, and either it sheds load catastrophically or, if those three pods held a quorum-based component, it drops below quorum and goes fully down. Nobody ran a deploy. Nobody touched your service. The *cluster* upgraded itself on schedule and took your service down as collateral.

Figure 8 shows the difference a PDB makes during exactly this scenario: the drain hits the eviction API, the PDB check blocks evictions that would breach the floor until pods reschedule, and the drain proceeds safely — versus the no-PDB path where everything is evicted at once and the service drops below quorum.

![Diagram of a node drain reaching the eviction API where a PodDisruptionBudget with minAvailable eighty percent blocks evictions until pods reschedule and the drain proceeds with zero downtime, contrasted with the no PDB path that evicts all replicas at once and drops the service below quorum](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-8.png)

Here is the PDB for our `checkout` service, plus the anti-affinity that handles the *involuntary* side a PDB cannot:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: checkout-pdb
spec:
  minAvailable: 80%        # at most 20% of replicas evicted at once
  selector:
    matchLabels:
      app: checkout
---
# In the Deployment template: spread pods across nodes so no single node
# (or its involuntary failure) takes too many replicas with it.
spec:
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: checkout
```

Worth being precise about the *interaction* between a PDB and a drain, because the timing is where the protection actually lives. When `kubectl drain node-1` runs, it first cordons the node (marks it unschedulable so nothing new lands there), then calls the Eviction API once per pod on that node. For each eviction, the API server checks: would removing this pod breach any PDB selecting it? If not, the eviction is allowed and the pod begins its normal graceful-termination sequence (the same `SIGTERM` + `preStop` + grace-period dance from section 5 — a drain is just another way to trigger it). If the eviction *would* breach the PDB, the API server returns a `429 Too Many Requests` and the drain *retries* that eviction, backing off, until the budget allows it — which happens once the `Deployment` controller has scheduled a replacement pod somewhere else and that replacement has passed its readiness probe. So the PDB does not just cap how many pods go down; it *paces* the drain to the speed at which replacements come ready. A drain of a node with two of your pods, under `minAvailable: 80%` of 5, evicts one pod, waits for its replacement to be `Ready` elsewhere, then evicts the second. The node operator sees the drain take a bit longer; your users see nothing.

This is also why a PDB is *useless without enough replicas or schedulable capacity elsewhere*. If you have `replicas: 5` and a `minAvailable: 80%` (floor 4) but your cluster is full and a replacement pod cannot be scheduled, the eviction blocks *forever* waiting for a replacement that never comes, and the drain hangs — the PDB did its job (it refused to breach availability) but the *outcome* is a stalled cluster operation. The PDB assumes the orchestrator can reschedule the evicted pod somewhere; give it the headroom (or autoscaler) to do so, or the protection turns into a stall.

One sharp edge worth naming: a PDB that is *too strict* can **block a drain forever**. If you set `minAvailable: 100%` (or `maxUnavailable: 0`), then *no* pod may ever be voluntarily evicted, and a node drain on a node hosting that pod will hang indefinitely, stalling your cluster upgrade and frustrating whoever is trying to patch the node. The PDB must leave *room* for the drain to make progress — `minAvailable` should be strictly less than `replicas`. With `replicas: 4` and `minAvailable: 80%` the floor rounds to 4, which actually *does* block all evictions; for 4 replicas you usually want `minAvailable: 75%` (floor 3, one evictable at a time) or `maxUnavailable: 1`. **Always check that `replicas - minAvailable >= 1`**, or your "protection" becomes a deadlock. Use percentages with care on small replica counts because of the rounding.

## 7. Putting it together: the complete zero-downtime spec

We now have all four steps and all their knobs. Figure 6 stacks them in the order they must occur: bring up new pods (`maxSurge`), prove they are ready (`readinessProbe`), shift traffic without a gap (`maxUnavailable: 0`), drain the old pods gracefully (`SIGTERM` + `preStop`), and guard the whole thing against cluster operations (PDB).

![Stack diagram of the four zero-downtime deploy steps from top to bottom, bring up new pods with maxSurge, prove ready with a readiness probe, shift traffic with maxUnavailable zero, and drain old pods with SIGTERM and preStop, with a fifth layer for the pod disruption budget guarding the cluster](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-6.png)

Here is the full, production-shaped manifest with every knob set, annotated so you can read each line back to the step it protects:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # STEP 1: surge headroom, fast roll
      maxUnavailable: 0    # STEP 3: never dip below 6 ready pods
  selector:
    matchLabels:
      app: checkout
  template:
    metadata:
      labels:
        app: checkout
    spec:
      terminationGracePeriodSeconds: 45   # STEP 4: drain budget
      topologySpreadConstraints:          # involuntary-disruption spread
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: checkout
      containers:
        - name: checkout
          image: ghcr.io/acme/checkout:v2
          ports:
            - containerPort: 8080
          startupProbe:                   # STEP 2: protect slow boot
            httpGet: { path: /healthz/startup, port: 8080 }
            periodSeconds: 2
            failureThreshold: 30
          readinessProbe:                 # STEP 2: gate traffic (the linchpin)
            httpGet: { path: /healthz/ready, port: 8080 }
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          livenessProbe:                  # generous, process-only
            httpGet: { path: /healthz/live, port: 8080 }
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 6
          lifecycle:
            preStop:
              exec:
                command: ["sh", "-c", "sleep 10"]   # STEP 4: win the endpoint race
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits: { memory: "512Mi" }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: checkout-pdb
spec:
  minAvailable: 80%        # STEP 5: bound voluntary disruptions
  selector:
    matchLabels:
      app: checkout
```

This is the whole solution for a stateless HTTP service. There is no magic in it — every line maps to one of the four steps plus the disruption guard, and you can defend each one to a reviewer. The art is not in knowing exotic features; it is in setting these ordinary knobs to non-default values *on purpose*.

## 8. Worked example: walking 2% dropped requests to zero

Let us return to the opening story and fix it precisely, measuring at each step, because the value of this is in the *attribution* — knowing which knob bought which improvement.

#### Worked example: root-causing and fixing a 2% deploy-time error rate

**The symptom.** A `checkout` service, 6 replicas, deploying ~30×/day. Edge logs show a consistent burst of `502` and `connection reset` errors for ~40 seconds on every deploy, totaling about **2% of all requests in the deploy window**. Over a day that is 30 bursts; the change-failure-rate metric is creeping up and the error budget is bleeding.

**Step 1 — measure honestly first.** Before touching anything, we instrument: a recording rule that counts 5xx-by-source and connection resets, tagged with the deploy timestamp from the rollout annotation. We confirm the errors are *tightly* correlated with the rollout window and split them by cause. About 60% are `connection refused` style 502s on *brand-new* pods (cold pods getting traffic) and 40% are `connection reset` on *terminating* pods (in-flight requests severed). That split tells us we have *both* the readiness bug and the shutdown bug.

**Step 2 — add the readiness probe.** The deployment had `livenessProbe` but no `readinessProbe` — a classic mix-up where the team thought "we have a health check" without realizing the health check they had did not gate traffic. We add a `/healthz/ready` probe that confirms the server is listening and the DB pool has a live connection. Next deploy: the cold-pod 502s vanish. We are now at roughly **0.8%** dropped, all on the terminating side.

**Step 3 — set `maxUnavailable: 0`.** It was on the default 25%, so during the roll capacity dipped to ~75% and at peak the remaining pods occasionally shed load. We pin `maxUnavailable: 0, maxSurge: 2`. The intermittent peak-time load-shed errors disappear. Still ~0.6% on termination.

**Step 4 — add SIGTERM handling.** The app was a Node service that exited immediately on `SIGTERM`, severing in-flight requests. We add `server.close()` on `SIGTERM` with a 20-second drain. Next deploy: the in-flight resets drop sharply but do *not* fully vanish — there is still a thin band of resets in the first ~1 second of each pod's termination.

**Step 5 — add the `preStop` sleep.** Those residual resets are the endpoint-removal race: the app was shutting down *faster* than the external load balancer could stop routing to it. We add `preStop: sleep 10` and bump `terminationGracePeriodSeconds` to 45. Next deploy: **0 dropped requests.** Clean.

The before/after, shown in Figure 7:

| Fix applied | Dropped requests / release | What it eliminated |
| --- | --- | --- |
| Baseline (no probe, defaults, no SIGTERM) | ~2.0% | — |
| + readiness probe | ~0.8% | cold-pod 502s |
| + `maxUnavailable: 0` | ~0.6% | peak-time load-shed |
| + SIGTERM handler | ~0.1% | most in-flight resets |
| + `preStop` sleep + grace period | **0%** | the endpoint-removal race |

![Before and after comparison of a single release, showing the before state with no readiness probe and no SIGTERM handler dropping two percent of requests over a forty second window, and the after state with a probe, maxUnavailable zero, a SIGTERM handler, and a preStop sleep dropping zero requests](/imgs/blogs/rolling-updates-and-zero-downtime-deploys-7.png)

The DORA payoff is concrete: change-failure rate for `checkout` deploys went from "every deploy is a minor failure" to clean, and because deploys stopped consuming error budget, the team could deploy *more* freely — deploy frequency rose because the cost-per-deploy fell to zero. That is the virtuous loop the whole series is about: making a deploy *safe* is what makes a deploy *frequent*.

#### Worked example: the node-drain that broke quorum

**The symptom.** A `coordinator` service, 5 replicas, ran fine for months. During a routine managed-Kubernetes minor-version upgrade, it went fully unavailable for ~90 seconds. No deploy had been triggered.

**The root cause.** The upgrade drained the node pool one node at a time. Two of the five `coordinator` pods had been scheduled onto the *same* node (no spread constraint). When that node drained, both pods were evicted *simultaneously* — and because the service used a quorum-based leader election needing 3 of 5, dropping to 3 was survivable, but the *next* node drained a third pod before the first two had rescheduled and rejoined, briefly leaving 2 of 5: **below quorum**. The service stopped electing a leader and went dark.

**The fix.** Two parts. First, a PDB with `minAvailable: 80%` (floor 4 of 5), so the eviction API blocks the drain from taking more than one `coordinator` pod down at a time and *waits* for the replacement to become ready before allowing the next eviction. Second, a `topologySpreadConstraint` with `maxSkew: 1` so the five pods land on five different nodes — addressing the *involuntary* risk the PDB cannot. The next cluster upgrade rolled through with zero `coordinator` disruption: each node drain blocked politely on the PDB, waited ~25 seconds for the rescheduled pod to pass readiness, then proceeded.

The lesson stacks on the first example: a clean *deploy* and a clean *cluster operation* are different problems with different knobs. The rolling-update + probe + graceful-shutdown trio protects deploys; the PDB + spread protects cluster operations. You need both, and most teams discover the second one the hard way during their first managed-cluster upgrade.

## 9. The deploy that looks done but isn't

A theme worth pulling out on its own: the gap between **`kubectl rollout status` says success** and **the service is actually serving cleanly**. These are not the same claim, and conflating them is how the opening team went months without noticing their 2% smear.

`kubectl rollout status deployment/checkout` returns success when the new `ReplicaSet` has the desired number of *ready* pods and the old one is at zero. That is a real signal — it means readiness probes passed — but it tells you nothing about whether requests were dropped *during* the transition, nothing about whether the readiness probe is *honest*, and nothing about the shutdown path. A deploy can report "successfully rolled out" while having severed every in-flight request on every retired pod, because rollout status does not watch client traffic; it watches pod readiness.

The discipline that closes this gap is to **measure the deploy from the client's side, not the cluster's side.** Concretely:

```bash
# Trigger and wait for the rollout to report done...
kubectl set image deployment/checkout checkout=ghcr.io/acme/checkout:v2
kubectl rollout status deployment/checkout --timeout=120s

# ...then ALSO assert the client-side truth: zero 5xx attributable to the deploy.
# This query (PromQL-style) is what actually proves zero downtime.
#   sum(rate(http_requests_total{job="checkout",code=~"5.."}[2m]))
# should stay flat across the deploy window. If it spikes, the rollout
# "succeeded" but the deploy did not.
```

A robust pipeline gates the deploy on *both*: rollout status (the cluster's view) *and* an error-rate check over the deploy window (the client's view). If you only check rollout status, you are checking the easy half. The harder, more honest signal is the one your users feel — and gating on it is the bridge to progressive delivery, where the canary's *entire purpose* is to make "is it actually serving cleanly" an automated, traffic-weighted gate rather than a thing you eyeball after the fact.

A second "looks done but isn't" trap: **the readiness probe that passes during the rollout but the app fails its first *real* request** — because the probe checks a trivial endpoint while real requests hit a code path the probe never exercises. Rollout status is green, every pod is "ready," and the first burst of genuine traffic 502s anyway. The only defense is an honest readiness probe (section 3) plus client-side measurement. The cluster's view of "ready" is only as good as the probe you wrote.

It helps to lay out the spectrum of deploy-validation signals from weakest to strongest, because "we waited for the rollout" sits near the bottom and teams routinely mistake it for the top:

| Signal | What it proves | What it misses |
| --- | --- | --- |
| `kubectl get pods` shows Running | the process started | nothing about readiness, traffic, or correctness |
| `kubectl rollout status` succeeds | new pods passed readiness, old RS at zero | dropped requests during the swap; probe honesty; shutdown drops |
| Synthetic smoke test post-deploy | one happy-path request now works | the deploy *window* errors (it ran *after* the roll) |
| Client-side error rate flat across deploy window | no requests dropped during the roll | whether v2 is *logically* correct |
| Canary analysis on real traffic | v2 serves correctly under real load at small blast radius | (this is the strongest gate; it is progressive delivery) |

The honest target for a *rolling* deploy is the fourth row — flat client-side error rate across the deploy window — which is achievable with the knobs in this post and no extra tooling. The fifth row is the job of progressive delivery, layered on top. The mistake is shipping on the second row and calling it done.

## 10. War story: the probes that took down more than they saved

Two real-shaped incidents to make the failure modes concrete, because the knobs in this post fail in ways that are counterintuitive — the safety mechanism is often what causes the outage.

**The liveness restart cascade.** A payments team set an aggressive liveness probe — `periodSeconds: 5, timeoutSeconds: 1, failureThreshold: 2` — on a service whose health endpoint checked the database. During a brief database failover (a planned ~8-second blip), every pod's liveness probe failed twice in a row, so Kubernetes restarted *every pod simultaneously*. The database recovered in 8 seconds, but now the entire fleet was cold-starting at once, with empty connection pools, all hitting the just-recovered database with a thundering herd of pool initialization. The service was down for ~4 minutes — *six times longer* than the database blip that triggered it. The fix had three parts: make liveness *dependency-free* (check the process, never the database), make it *generous* (`failureThreshold: 6`, slow to fire), and move the dependency check to *readiness* (so a database blip makes pods temporarily not-ready — removed from rotation, not restarted — and they rejoin automatically when the database returns). The principle the team adopted: **liveness restarts; readiness reroutes. When in doubt, reroute.**

**The PDB deadlock.** A platform team, having been burned by a node-drain outage, set `PodDisruptionBudget` with `minAvailable: 100%` on their critical services — reasoning that "we never want any of these pods to go down." The next cluster upgrade *stalled completely*: the node drain could not evict a single pod (any eviction would breach the 100% floor), so the drain hung, the upgrade timed out, and the cluster was left half-upgraded with cordoned nodes accumulating un-evictable pods. The "maximum protection" setting was actually a deadlock. The fix was to set `minAvailable` to a value that *leaves room for progress* (here, `maxUnavailable: 1`, allowing the drain to take one pod at a time while the replacement reschedules). The lesson: **a PDB must permit the drain to make progress; protection that blocks all motion is not protection, it is a freeze.**

Both stories share a structure that is worth internalizing: the mechanisms in this post are *guardrails*, and a guardrail set too tight does not just fail to help — it actively causes the failure it was meant to prevent. An over-eager liveness probe manufactures outages. A too-strict PDB freezes the cluster. The skill is not "turn the safety up to maximum"; it is "set each knob to the value that protects the user while leaving the system room to operate."

## 11. How rolling fits with progressive delivery

A rolling update is the *floor*, not the ceiling, of deploy safety. It answers "swap the pods without dropping requests," but it does *not* answer "is version two actually correct?" During a rolling update, once a new pod passes its readiness probe, it gets full production traffic — and a readiness probe only proves the pod can *serve*, not that the new code is *correct*. If v2 has a logic bug that returns wrong answers (but valid `200`s), a rolling update will happily roll it out to 100% of traffic, because every pod is "ready." Rolling update protects against *infrastructure* failure during a deploy; it does nothing about *application* regressions.

That is the gap progressive delivery fills, and it is the subject of the sibling post **`progressive-delivery-in-the-pipeline-canary-and-blue-green`** (planned in this series). The relationship, stated cleanly:

| Concern | Rolling update | Canary / blue-green |
| --- | --- | --- |
| No dropped requests during swap | yes (with the knobs in this post) | yes (builds on rolling) |
| Bounded blast radius if v2 is buggy | no — 100% traffic once ready | yes — 1% / 5% / 25% steps |
| Automated correctness gate (error rate, latency) | no | yes — analysis halts the roll |
| Instant rollback | re-roll the old RS (minutes) | flip traffic back (seconds) |
| Extra cost | +`maxSurge` pods briefly | +full second environment (blue-green) or +analysis tooling (canary) |

The crucial point for an engineer building the pipeline: **canary and blue-green are built on top of the same primitives covered here.** A canary deploy *still* brings up new pods, *still* needs them to pass a readiness probe before traffic, *still* needs graceful shutdown when the old version is retired, and *still* wants a PDB. Argo Rollouts and Flagger — the tools that implement canary in the pipeline — wrap the rolling update with traffic-shifting and automated analysis, but they do not replace the four steps; they *orchestrate* them more carefully. So if your rolling update drops requests, your canary will too. **Fix the floor first.** Then layer progressive delivery on top for the correctness gate and the smaller blast radius. The reliability theory of *why* a traffic-weighted, SLO-gated rollout is the safe way to ship is covered in the SRE post on [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery); the fleet-level view of choosing between blue-green, canary, and feature flags across many services lives in the microservices post on [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags). This post is the layer beneath both: get the rolling primitive right, and everything above it inherits the cleanliness.

This also composes with the *delivery* objects you wire into the pipeline — the `Deployment`, `Service`, and `PodDisruptionBudget` are exactly the Kubernetes objects the sibling post `kubernetes-for-delivery-the-objects-that-matter` (planned) covers, and the GitOps controller that applies these manifests is what turns "I edited a knob" into "the cluster reconciled to the new spec."

## 12. Stress-testing the design

A good design survives the awkward questions. Let us pose the ones that actually happen in production and reason through each.

**What if two pods are starting at once and they all fail readiness?** With `maxUnavailable: 0`, the old pods stay in rotation until the new ones pass readiness, so a stuck rollout *stalls* rather than *drops*. The new pods sit `not ready`, the old ones keep serving, and `kubectl rollout status` hangs until your `--timeout` fires. That is the correct, safe behavior: a bad new version cannot displace a working old one. Your pipeline should treat the rollout timeout as a *failure* and roll back — `kubectl rollout undo` re-promotes the old `ReplicaSet`, which is still around. The deploy fails *closed*, not open.

**What if the readiness probe depends on a downstream that is down during the deploy?** Then every new pod fails readiness, the roll stalls (per above), and the old pods — which presumably *also* now can't reach the downstream — are still serving, possibly also degraded. This is the case for distinguishing "critical" from "degradable" dependencies in readiness (section 3): if the downstream is truly required, stalling is correct; if it is degradable, readiness should *not* check it and the app should serve a fallback. Do not let a non-critical dependency's outage block your deploys.

**What if `SIGKILL` fires before the app finishes draining?** Then you drop the requests still in flight at the 45-second mark. The mitigation is to size `terminationGracePeriodSeconds` above your worst-case request duration (plus the `preStop` sleep), and — for genuinely long requests like large uploads or streaming — to accept that those *will* be interrupted and design the client to retry. There is no setting that lets you finish an arbitrarily long request during a bounded grace period; for unbounded streams, "graceful reconnection" is the real contract.

**What if the cluster has no spare capacity for the `maxSurge` pods?** Then the surge pods sit `Pending` (unschedulable), the roll stalls, and you have effectively coupled your deploy to your autoscaler's ability to add a node in time. The fix is either to keep headroom for the surge, to use a fast node autoscaler with over-provisioning (a low-priority "balloon" pod that gets evicted to make room instantly), or — if you genuinely cannot surge — to accept `maxUnavailable: 1` and a brief, *controlled* capacity dip instead of an uncontrolled one. The point is to choose the trade-off, not have it chosen for you by a `Pending` pod at 2 a.m.

**What if two deploys race — someone pushes v3 while v2 is still rolling?** The `Deployment` controller handles this cleanly: it abandons the in-progress v2 roll and starts rolling toward v3 (creating a third `ReplicaSet`), still honoring `maxSurge`/`maxUnavailable` throughout. No requests are dropped because the same guards apply to the new roll. What you *can* hit is a brief moment with three `ReplicaSet`s; the old ones scale to zero and are eventually garbage-collected per `revisionHistoryLimit`. The defense at the pipeline level is to *serialize* deploys of the same service (a concurrency group in your CI) so you are not chasing a moving target — but even unserialized, the platform does not drop requests.

**What if the `preStop` sleep is too short for a slow external load balancer?** Then you reintroduce the endpoint-removal race for the slow LB. The sleep must exceed the *worst-case* propagation time of your *slowest* routing layer — for an external cloud LB that can be several seconds. Measure it: watch for connection resets on terminating pods and lengthen the sleep until they vanish. Ten seconds is a safe default for most setups; cloud LBs occasionally want fifteen.

## 13. How to reach for this (and when not to)

Every knob in this post has a cost, and a senior engineer's job is to know when *not* to turn it. Decisive guidance:

**Always set, for any production HTTP service:** a readiness probe (this is non-negotiable — its absence is the number-one zero-downtime bug), `maxUnavailable: 0`, a `SIGTERM` handler in the app, and a `preStop` sleep. These are cheap, they compose, and skipping any one of them drops requests. There is no "small enough" service that benefits from omitting a readiness probe.

**Add a PDB and spread constraints** once you run on a cluster that gets *operated on* — any managed Kubernetes that does automatic node upgrades, any cluster with a node autoscaler that removes nodes, any environment where someone might `kubectl drain` a node. If your nodes are never drained (rare, and probably a smell), you can defer the PDB. The moment you are on EKS/GKE/AKS with auto-upgrade, set it *before* your first upgrade, not after the outage.

**Be conservative with liveness probes.** This is the one knob where the default instinct ("more health checking is safer") is *backwards*. Many services are better off with *no* liveness probe at all, relying on readiness to reroute around struggling pods and on the process crashing when truly broken. Add liveness only for a *specific, known, unrecoverable* deadlock, and make it dependency-free and generous. An aggressive liveness probe is a self-DDoS waiting for a dependency blip.

**Where this does not apply:** stateful singletons (a leader, a primary) want failover and connection-draining at the application/proxy layer, not a naive rolling update — rolling a stateful set is a different and more delicate operation. Long-lived streams (WebSocket, gRPC streaming) cannot be "drained" in a bounded grace period; their zero-downtime story is client-side graceful reconnection. And if you are a three-person startup on a PaaS (Render, Fly, Railway, App Engine) that does rolling deploys for you, you may not be writing these manifests at all — the platform sets the knobs, and your job is to ensure your *app* handles `SIGTERM` and exposes an honest readiness endpoint, because those two are *yours* no matter who runs the orchestrator.

**The one thing to internalize:** zero-downtime deployment is configuration, not architecture. You do not need a mesh, a fancy gateway, or a rewrite. You need the four steps set correctly on an ordinary `Deployment`. The reason teams drop requests on deploy is almost never that they lacked the *capability* — it is that the defaults are wrong and nobody changed them on purpose.

## Key takeaways

- **A pod gets no traffic until its readiness probe passes** — this single fact is the linchpin of zero downtime, and a missing or dishonest readiness probe is the number-one cause of deploy-time 502s.
- **Set `maxUnavailable: 0`** for true zero downtime; it pins the capacity floor at 100% and never drops a request to a capacity gap. Use `maxSurge` to trade cluster cost for roll speed.
- **Readiness reroutes, liveness restarts, startup defers liveness.** Check critical (not degradable) dependencies in readiness; keep liveness cheap, generous, and dependency-free — or omit it. An aggressive liveness probe causes restart cascades during dependency blips.
- **Graceful shutdown is the other half:** catch `SIGTERM`, stop accepting new requests, drain in-flight ones, exit — and the in-app drain timeout must be shorter than `terminationGracePeriodSeconds`.
- **The endpoint-removal race is real:** endpoint removal and `SIGTERM` fire simultaneously, so a `preStop` sleep (5–15s) is what lets the load balancer stop routing before the app stops answering.
- **A PodDisruptionBudget protects only *voluntary* disruptions** (node drains, cluster upgrades); set `minAvailable` strictly below `replicas` so the drain can make progress, and add topology spread for the *involuntary* failures a PDB cannot cover.
- **"Rollout succeeded" is the cluster's view, not the client's** — gate deploys on a client-side error-rate check over the deploy window, not just `kubectl rollout status`.
- **Rolling update is the floor, not the ceiling:** it prevents infrastructure-level drops but not application regressions. Canary and blue-green build on these same primitives — fix the rolling baseline first, then layer progressive delivery for the correctness gate.

## Further reading

- [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model) — the series intro and the commit-to-production spine this post's deploy step sits inside.
- [Deploying Safely with Progressive Delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery) — the SRE companion on why an SLO-gated, traffic-weighted rollout is the safe way to ship; the reliability theory behind the toolchain here.
- [Graceful Degradation and Fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — how to keep serving when a dependency is down, which is exactly the discipline that decides what a readiness probe should and should not check.
- [Deployment Strategies: Blue-Green, Canary, and Feature Flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) — the fleet-level view of choosing among deploy strategies across many services.
- The Kubernetes documentation on Deployments, probes (`Configure Liveness, Readiness and Startup Probes`), pod termination lifecycle, and `PodDisruptionBudget` — the canonical reference for every knob in this post.
- The companion siblings `kubernetes-for-delivery-the-objects-that-matter` and `progressive-delivery-in-the-pipeline-canary-and-blue-green` (in this series) — the objects you wire into the pipeline, and the layer that adds a correctness gate on top of the rolling baseline.
