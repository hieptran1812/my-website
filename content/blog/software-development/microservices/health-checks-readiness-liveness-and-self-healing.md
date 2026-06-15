---
title: "Health Checks: Readiness, Liveness, and Self-Healing Without Self-Harm"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How the platform knows a service instance is healthy and heals it automatically — and the surprisingly dangerous ways health checks turn a 30-second dependency blip into a fleet-wide outage."
tags:
  [
    "microservices",
    "health-checks",
    "kubernetes",
    "liveness-probe",
    "readiness-probe",
    "self-healing",
    "resilience",
    "distributed-systems",
    "software-architecture",
    "backend",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/health-checks-readiness-liveness-and-self-healing-1.webp"
---

At 02:14 on a Tuesday, ShopFast's order service was completely down. Not slow — *down*. The on-call engineer's phone showed forty pods in `CrashLoopBackOff`, all restarting at once, all printing the same line on boot: `dialing postgres... timeout`. The database itself had recovered minutes ago; its managed failover had taken about thirty seconds and was long finished. Yet the order service stayed dead for another twelve minutes, recovering only when every pod finally clawed its way through a cold start with empty caches and a stampede of connections. The post-mortem found the cause in four lines of YAML written months earlier by a well-meaning engineer who wanted the health check to be "thorough": the **liveness probe** queried the database. When the database blipped, every pod failed its liveness check simultaneously, Kubernetes dutifully restarted all of them, and a thirty-second dependency hiccup became a self-inflicted, fleet-wide outage that the platform's own "self-healing" actively made worse.

This is the central, counterintuitive truth of health checking, and it is the thing this post exists to teach: **the mechanisms that heal your system are the same mechanisms that can destroy it, and the difference is almost entirely in whether you understood what each probe is actually for.** A health check is not a single idea. It is at least three different questions the platform asks — *is this process wedged?*, *can this instance serve traffic right now?*, *has this slow-booting app finished starting?* — and each question has a completely different correct answer and a completely different action attached to it. Conflate them and you build a machine that amputates a healthy limb because a fingernail is dirty.

By the end of this post you will be able to design health endpoints that a platform can trust: a shallow `/livez` that only restarts a genuinely-wedged process, a deeper `/readyz` that gates traffic and powers graceful shutdown, and a startup probe that gives a slow boot the time it needs. You will be able to wire all three into a Kubernetes `Deployment` correctly, write the graceful-drain dance that lets a rolling deploy drop *zero* requests, and recognize the failure modes — restart storms, readiness flapping, the too-aggressive liveness probe that kills a busy-but-healthy instance, the thundering herd on recovery — before they page you. The figure below is the map of the whole article: three probes, three jobs, one kubelet turning their answers into actions.

![A stacked map of liveness readiness and startup probes each with a distinct job feeding the kubelet that turns answers into actions](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-1.webp)

This post closes Track 4 of the series — the resilience track. We have built [timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) to survive a slow dependency, [idempotency](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) to make retries safe, [partial-failure handling and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) to keep serving when a dependency is gone, and [rate limiting and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) to protect a service from being overwhelmed. All of those are things a service does *for itself*. Health checks are different: they are the contract between a service and the *platform* — the language the service uses to tell Kubernetes, the load balancer, and the autoscaler the truth about its own condition, so the platform can heal it automatically. Get the contract right and the platform is your tireless 24/7 SRE. Get it wrong and the platform becomes the most efficient outage-amplifier you have ever built. This post sets up the [Kubernetes essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) and [observability](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) posts that follow.

## Why a service has to tell the platform how it feels

Start with the problem from first principles. You have forty copies of the order service running across a cluster. Some are healthy. One has hit a deadlock and stopped processing requests, though its process is technically still alive — the operating system sees a running PID, the TCP port is still open, but nothing comes back. Another just started ten seconds ago and has not yet opened its database connection pool or warmed its caches, so if you send it a request it will return an error or hang. A third was perfectly healthy until two seconds ago, when a deploy sent it a `SIGTERM`; it is now trying to finish the eighteen requests it has in flight before it exits.

The platform — Kubernetes, the load balancer, the service registry — has to decide, continuously, two things about each of these forty instances. **Which ones should receive traffic?** (Not the deadlocked one, not the still-booting one, not the one that is draining.) And **which ones should be killed and replaced?** (The deadlocked one, yes; the still-booting one, absolutely not; the draining one, only after it finishes.) The platform cannot answer either question by looking from the outside. A process can hold an open TCP port and still be completely wedged. A process can be running and busy and perfectly healthy. The *only* entity that knows the difference is the service itself, from the inside.

So the service publishes the truth about its own condition through HTTP endpoints — or gRPC health methods, or a command the platform runs — that the platform polls on a timer. This is the **health check**. It is the service's self-report, and the platform acts on it mechanically and without sympathy. That mechanical, unsympathetic action is exactly what makes health checks powerful and dangerous in equal measure. The platform will do *precisely* what your probe tells it to, at machine speed, across the whole fleet at once. If your probe lies — if it reports "I am dead" when it merely means "a dependency is briefly unreachable" — the platform will execute that lie forty times in parallel.

A junior engineer's instinct is that a health check should be *comprehensive*: check everything, the database, the cache, the downstream payment service, the message broker, so that "healthy" really means *everything works*. That instinct is exactly backwards for one of the three probes, and getting it right requires understanding that the three probes are not three levels of the same check. They are three different checks answering three different questions, and the rest of this post is about keeping them distinct.

### The first principle: a health check is a control signal, not a status report

Reframe what a health check *is*. It is tempting to think of it as a status report you'd read on a dashboard — a green light that means "all good." But that is the wrong model, and the wrong model is what causes the disasters. A health check is a **control signal**: an input to an automated control loop that takes an *action*. The question to ask of every line you put in a health check is not "is this part of the system working?" but "**do I want the platform to take this probe's action when this line fails?**"

For a liveness probe, the action is "restart this pod." So the only thing you should check in a liveness probe is something for which "restart this pod" is the correct, helpful response. A deadlocked event loop? Restarting fixes it — put it in. A database that is briefly unreachable? Restarting does *not* fix it (the new process can't reach the database either) and restarting all forty pods at once makes everything worse — so it must *not* be in the liveness probe. The action defines the contents. This single reframing prevents the most expensive health-check mistakes in the industry.

## The three probes, defined precisely

Let me define each probe in plain terms, with the ShopFast order service as the running example, before we touch any YAML.

**Liveness** answers: *is this process wedged — deadlocked, stuck in an infinite loop, out of usable memory, event loop blocked — such that the only repair is to kill it and start fresh?* The action on failure is **restart the container**. Because the action is so violent and so total, the cardinal rule of liveness is that it must be **shallow**: it must check *only the process itself*, and it must never check anything a restart cannot fix. For ShopFast, `/livez` should answer one question — "is my request-handling loop responsive right now?" — and nothing more. It should not touch the database, the payment service, Redis, Kafka, or any other process. The moment a liveness probe checks a dependency, a dependency outage becomes a restart storm, and the restart storm is worse than the original outage.

**Readiness** answers a completely different question: *can this instance successfully serve a request right now?* The action on failure is **remove this pod from the load balancer** (and the inverse on success: add it back). Critically, **a readiness failure does not restart the pod.** It just stops sending it traffic until it reports ready again. Because the action is gentle and reversible, readiness is *allowed* to be deeper: it may check that mandatory dependencies are reachable, that the connection pool is established, that caches are warm enough to serve. For ShopFast, `/readyz` legitimately checks "is my Postgres connection pool up, and have my warm caches loaded?" — because if those aren't true, this pod *should* be temporarily removed from rotation while its healthy siblings carry the load. Readiness is also the mechanism behind graceful shutdown: a pod about to exit flips its readiness to *failing* so the load balancer drains it before it dies.

**Startup** answers: *has this slow-booting application finished its initial startup?* It exists for one reason: to hold the liveness probe off during boot. A service that takes twenty seconds to load a large model, build an in-memory index, or run migrations would, without a startup probe, fail its liveness probe during those twenty seconds and get killed before it ever finished booting — an infinite restart loop. The startup probe runs first; until it passes, the liveness and readiness probes don't run at all. Once it passes, it hands over to liveness and readiness and never runs again. For ShopFast's order service, which warms a product-pricing cache and takes about twenty seconds to boot, the startup probe gives it generous time, and only after boot does the tight, fast liveness probe take over.

Here is the decision matrix that you should internalize and, frankly, paste into your team's runbook. Notice that the *depth* and the *failure action* are exactly inverted between liveness and readiness — that inversion is the whole game.

![A decision matrix comparing liveness readiness and startup probes across question asked failure action depth dependency checking and blast radius](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-2.webp)

Read the matrix column by column. Liveness asks "wedged?", restarts on failure, must stay shallow, must *never* check downstream dependencies, and its failure blast radius is the most severe (it kills pods). Readiness asks "ready to serve?", drops the pod from the load balancer on failure, is *allowed* to be deep, *may* check mandatory dependencies, and its failure blast radius is gentler (it sheds traffic but the pod survives). Startup asks "done booting?", just waits and rechecks, only signals boot completion, checks only itself, and its blast radius is the mildest (it merely delays the start). If you remember one thing from this post, remember that the "Check downstream deps?" row says **NEVER** under liveness and **mandatory deps** under readiness. That single row, obeyed, would have prevented ShopFast's 02:14 outage entirely.

## Building the endpoints: shallow `/livez`, deeper `/readyz`

Before we write code, it's worth pinning down what "wedged" actually means, because liveness exists to detect *exactly that one condition* and nothing else. A process is wedged when it is technically alive — the OS scheduler still lists it, its TCP port is still bound — but it can no longer make progress on real work and *no amount of waiting will fix it.* The classic shapes: a **deadlock** where two goroutines or threads each hold a lock the other needs, so both block forever; an **event-loop block** in a single-threaded runtime (Node.js, a Python asyncio app) where some synchronous CPU-bound call has frozen the loop and no callbacks fire; a **resource exhaustion spiral** where the heap is so full the process spends nearly all its time in garbage collection and effectively stops serving; or a **livelock** where threads spin retrying a failing operation and never advance. What all of these share is the property that defines liveness's job: **a fresh process would not have the problem.** Restart the deadlocked process and the new one starts with no locks held. That is the entire and only justification for liveness restarting a pod — and it is precisely why a dependency outage does *not* qualify, because a fresh process cannot reach the down dependency any better than the old one could. If restarting wouldn't help, it doesn't belong in liveness.

Let's write them. We'll do ShopFast's order service in Go, but the shape transfers directly to any language. Start with the liveness endpoint, because the discipline is *what you leave out*.

A correct liveness handler does essentially nothing expensive. The most honest liveness check is: "if this HTTP handler can run and return 200, then by definition the process is not wedged — the request-handling loop is responsive." For many services that is genuinely enough. If you want slightly more, you can check a single in-process signal that a restart would fix: a heartbeat that a background worker updates, or a flag a watchdog sets when it detects the event loop has been blocked too long.

```go
// internal/health/livez.go
package health

import (
	"net/http"
	"sync/atomic"
	"time"
)

// lastBeat is updated by the main work loop every iteration. If it goes
// stale, the process is genuinely wedged and a restart is the right fix.
var lastBeat atomic.Int64

func Beat() { lastBeat.Store(time.Now().UnixNano()) }

// LivezHandler is SHALLOW on purpose. It checks ONE thing a restart can fix:
// is the work loop still beating? It NEVER touches the DB, cache, or any
// downstream service. A dependency outage must not be able to fail this.
func LivezHandler(maxStale time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		last := time.Unix(0, lastBeat.Load())
		if time.Since(last) > maxStale {
			// Work loop has not advanced; the process is wedged.
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("wedged: work loop stale\n"))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	}
}
```

Look at what is *not* there: no `db.Ping()`, no Redis call, no HTTP call to the payment service. That absence is the entire correctness argument. If you are tempted to add "just a quick DB check" to liveness, stop and ask the control-signal question: do I want all forty pods to *restart* when the DB blips? No. Then it does not belong here. The most common safe liveness probe of all is even simpler — an endpoint that returns 200 unconditionally, which proves the HTTP server itself is responsive. That is a perfectly defensible liveness probe for a stateless service, and it can never cause a dependency-driven restart storm because it depends on nothing.

Now the readiness endpoint, which is allowed to be deeper because its only power is to gate traffic. For ShopFast, "ready" means the Postgres pool is established and the pricing cache has loaded. If either is not true, this pod should sit out of rotation until it is — its siblings will carry the load.

```go
// internal/health/readyz.go
package health

import (
	"context"
	"net/http"
	"time"
)

type ReadyChecker struct {
	DB          interface{ PingContext(context.Context) error }
	CacheWarmed func() bool
	draining    atomic.Bool // set true during graceful shutdown
}

// StartDraining flips readiness to failing so the LB drains this pod.
func (rc *ReadyChecker) StartDraining() { rc.draining.Store(true) }

// ReadyzHandler MAY be deep: it checks mandatory deps. Failing it only
// removes the pod from the load balancer; it does NOT restart the pod.
func (rc *ReadyChecker) ReadyzHandler(w http.ResponseWriter, r *http.Request) {
	// 1) During shutdown, report not-ready immediately so traffic drains.
	if rc.draining.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("draining\n"))
		return
	}
	// 2) Mandatory dependency: the DB pool must be usable to serve orders.
	ctx, cancel := context.WithTimeout(r.Context(), 800*time.Millisecond)
	defer cancel()
	if err := rc.DB.PingContext(ctx); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("db pool not ready\n"))
		return
	}
	// 3) Warm caches must be loaded before this pod takes traffic.
	if !rc.CacheWarmed() {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("cache warming\n"))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ready\n"))
}
```

Notice three deliberate choices. The readiness check has a *tight timeout* (800ms) on its dependency check — a readiness probe that hangs is itself a problem, because a hung probe counts as a failure and removes the pod. It checks only **mandatory** dependencies — the database, because you literally cannot serve an order without it. It does *not* check the payment service, because the order service is designed to degrade gracefully when payment is down (queue the order, retry payment async — see [graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation)); a payment outage should not pull all order pods out of rotation. And it has a `draining` flag that lets graceful shutdown flip readiness to failing on demand, which we'll wire up shortly.

### Mandatory versus optional dependencies: the readiness judgment call

The hardest design decision in readiness is *which dependencies are mandatory*. The test is brutal and clear: **if this dependency is down, can this instance still do useful work?** If the answer is no, it's mandatory — put it in readiness, because a pod that can't do useful work should not get traffic. If the answer is yes (because you degrade gracefully), it's *optional* — keep it out of readiness, because pulling pods from rotation over an optional dependency just concentrates the same failing traffic onto fewer pods.

For ShopFast's order service: the Postgres pool is mandatory (no orders without it). The payment service is optional (orders queue and pay async). The recommendation service is optional (the page renders without recommendations). The shipping-rate service is optional (use a cached default rate). So `/readyz` checks Postgres and nothing else downstream. This is the same reasoning as a circuit breaker's: you isolate the failure to the smallest possible blast radius. A readiness probe that checks every dependency turns *any* dependency outage into a *full* outage — the exact opposite of what microservices resilience is supposed to buy you.

## How the kubelet turns probe answers into actions

The platform component that runs the probes is the **kubelet** — the agent on each Kubernetes node. It probes every pod on its node on a timer and routes each result down a different control path. The figure below shows the branch, and the branch is the most important picture in this post.

![A graph showing the kubelet branching liveness results into a restart path and readiness results into a load balancer endpoint update path](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-4.webp)

Trace the two paths. The liveness result, when it fails enough times in a row, flows to **restart the container** — the kubelet kills the process and starts a fresh one, which boots and (after warmup) rejoins. The readiness result flows to the **Endpoints object** — a Kubernetes resource that lists which pod IPs are currently eligible to receive traffic. When readiness passes, the pod's IP is in the Endpoints list; when it fails, the IP is removed. The Service (and the load balancer behind it) routes only to IPs in the Endpoints list. So a readiness failure quietly removes a pod from rotation; it never touches the process. These are two separate control loops with two separate actions, and the *only* loop that can restart you is the liveness loop. This is precisely why a dependency check belongs (if anywhere) in readiness and *never* in liveness: a readiness failure during a dependency blip just shifts traffic to healthy siblings; a liveness failure during a dependency blip kills the herd.

If you've read the [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) post, you'll recognize the Endpoints object as the live phone book: readiness is what writes and erases entries in it. Health checks and service discovery are two halves of the same machine — the registry holds *who exists*, and readiness holds *who can serve right now*.

There is a subtle timing property worth internalizing here, because it explains a whole class of "why did traffic hit a pod that was clearly not ready" confusions. The two loops run on independent clocks. The liveness loop probes on its period, counts failures, and acts. The readiness loop probes on its own period, updates the Endpoints object, and then — separately — the Service's data plane (kube-proxy's iptables or IPVS rules, or an external load balancer's pool) reads the Endpoints object and updates its routing. That last hop is *eventually consistent*: there is a propagation delay of anywhere from a few hundred milliseconds to a couple of seconds between "the Endpoints object changed" and "every load balancer instance has stopped routing to that IP." For *adding* a pod, this delay is harmless — a slightly late arrival of traffic to a freshly-ready pod is fine. For *removing* a pod, this delay is the source of the endpoint-propagation race that drops requests on shutdown, which is exactly why graceful shutdown needs that `preStop` drain we'll build later. The lesson: readiness is not an instantaneous switch on the load balancer; it is a signal that propagates, and production-grade shutdown has to account for the propagation window.

### Why not just let the load balancer figure it out itself?

A fair question from a junior: load balancers have always done their own "active" health checks — an HAProxy or an ELB will poll a backend's `/health` and stop routing to it if it fails, with no Kubernetes probes involved at all. Why does the platform also probe? The answer is that these are two *layers* of the same idea, and they cover different failure shapes. The platform's readiness probe is the *authoritative, control-plane* signal — it's what Kubernetes uses to decide membership in the Endpoints object, drive rolling updates, and feed autoscaling. The load balancer's own check (or a service mesh's *passive* health checking, which we'll touch on) is a *data-plane*, last-mile safety net: even if the control plane is slow to update, the data plane can notice a backend returning errors on *real* traffic and eject it temporarily. Modern meshes like Istio and Linkerd do exactly this with **outlier detection** — they watch the live error rate per upstream and eject a host that's failing real requests, then probe it back in, all without any explicit health endpoint. The practitioner's takeaway: configure your readiness probe well (the authoritative signal), and let the mesh's passive outlier ejection be the belt-and-suspenders that catches the failures readiness doesn't, like a pod that passes `/readyz` but is actually returning 500s on the real endpoint.

## The cardinal sin, drawn: deep liveness causes a restart storm

We've named the rule. Now let's make the failure visceral, because seeing the two outcomes side by side is what makes engineers stop adding "thorough" checks to liveness. Consider ShopFast's order service under a thirty-second database failover, first with the anti-pattern (liveness checks the DB) and then with the correct shallow liveness.

![A before and after comparison of a deep liveness probe causing a fleet wide restart storm versus a shallow liveness probe that leaves pods running](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-3.webp)

On the left, the anti-pattern. The database blips for thirty seconds. Every `/livez` probe across all forty pods queries the database, and every one fails at the same instant. The kubelet on each node, seeing three consecutive failures, restarts every pod. Now you have forty cold processes booting with empty caches, all dialing the database at once, and the outage outlives the original blip by minutes. On the right, the correct design. The same database blip happens, but `/livez` is shallow — it passes the entire time, because the work loop is still beating. `/readyz` fails (it checks the DB), so the pods are quietly removed from the load balancer; requests that arrive get a clean 503 or are retried by the client. When the database recovers thirty seconds later, `/readyz` passes again, the pods rejoin the load balancer, and there were *zero restarts*. The blip cost you thirty seconds of degraded service instead of twelve minutes of total outage.

#### Worked example: the cascading-restart outage, with the recovery math

Let's put real numbers on ShopFast's 02:14 incident so the cost is concrete. The fleet: 40 pods, each handling ~50 requests per second, so 2,000 RPS total. Each pod warms a pricing cache on boot that takes 20 seconds to load and a connection pool that takes ~3 seconds to establish — call it 23 seconds to fully ready. The database failover lasts 30 seconds.

With the **deep liveness probe** (probe every 10s, fail after 3 misses): all 40 pods fail liveness within ~10 seconds of the blip starting, and Kubernetes begins restarting them around T+30s. But now 40 pods are booting at once, each opening a fresh pool against a database that just recovered and is itself cold — and each pod's 20-second cache warm hits the database hard. The thundering herd of 40 simultaneous cold starts means the database, marginal already, struggles, slowing every boot. In practice the fleet took ~12 minutes to fully recover. During those 12 minutes the service served essentially 0 successful requests. At 2,000 RPS, that is **2,000 × 60 × 12 ≈ 1.44 million failed requests** from a 30-second database blip — a roughly **24× amplification** of the outage by the platform's own self-healing.

With the **shallow liveness probe**: 0 pods restart. During the 30-second blip, `/readyz` fails, pods leave the load balancer, and the ~60,000 requests that arrive in those 30 seconds get a fast 503 (which well-behaved clients retry with backoff). When the DB recovers, all 40 pods — still warm, caches intact, pools intact — flip back to ready within a probe cycle. Recovery time: **one readiness interval, ~5 seconds.** Failed requests: ~60,000 fast 503s, most of which clients successfully retried. The difference between the two designs is the difference between a 12-minute Sev-1 and a blip nobody outside the on-call channel ever noticed — and it is *entirely* in four lines of YAML.

The timeline below traces the deep-liveness version event by event, because the *order* of events is what makes it so insidious — the platform's restart action lands at T+30s, just as the database is recovering at T+35s, so the self-healing fires *exactly* when the original problem is already resolving, and replaces a recovering fleet with a cold one.

![A six event timeline showing a thirty second database blip becoming a twelve minute outage because the liveness probe checked the database](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-5.webp)

Walk the events left to right. At T+0 the database fails over and blips for thirty seconds. By T+10s every `/livez` probe has queried the database and failed, all forty at once. By T+30s each pod has hit three consecutive failures and the kubelet restarts the entire fleet. At T+35s the database recovers — but the pods are already gone, mid-restart. From T+55s onward, forty cold processes boot simultaneously with empty caches, each dialing the just-recovered database, a thundering herd that slows every boot. Only at around T+12m have the caches re-warmed and the service recovered. The cruelest detail is the near-coincidence of T+30s (restart fires) and T+35s (database heals): the self-healing mechanism did its maximum damage at the precise moment the underlying problem was fixing itself. A shallow liveness probe would have simply ridden out the thirty seconds and been serving again at T+35s.

## Wiring all three probes into a Kubernetes Deployment

Here is the correct configuration for ShopFast's order service, with all three probes separated by job. Read the comments — every value has a reason.

```yaml
# k8s/order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 4
  template:
    spec:
      # Give the pod up to 30s to drain in-flight requests on shutdown.
      terminationGracePeriodSeconds: 30
      containers:
        - name: order-service
          image: shopfast/order-service:1.42.0
          ports:
            - containerPort: 8080  # traffic
            - containerPort: 8081  # health endpoints (separate port)

          # STARTUP: holds liveness off during a ~20s cache warm + pool init.
          # 10 * 3s = up to 30s of boot time before liveness can ever fire.
          startupProbe:
            httpGet: { path: /startupz, port: 8081 }
            periodSeconds: 3
            failureThreshold: 10        # 30s total boot budget
            timeoutSeconds: 2

          # LIVENESS: shallow, fast, restarts only a genuinely wedged process.
          # Generous failureThreshold so a single slow tick never kills a busy
          # but healthy pod. NEVER checks a dependency.
          livenessProbe:
            httpGet: { path: /livez, port: 8081 }
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3         # 30s of being wedged before restart
            successThreshold: 1

          # READINESS: deeper, gates traffic. Fails fast (1 miss) so a draining
          # or unhealthy pod leaves the LB quickly; recovers after 2 successes
          # to avoid flapping back in on a single lucky probe.
          readinessProbe:
            httpGet: { path: /readyz, port: 8081 }
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 1
            successThreshold: 2

          lifecycle:
            preStop:
              # Give the LB a moment to notice readiness flipping before the
              # process gets SIGTERM. Closes the "killed mid-request" window.
              exec: { command: ["sleep", "5"] }
```

Several production-grade choices are encoded here. The health endpoints live on a **separate port** (8081) from traffic (8080) so that you can firewall health checks to the kubelet only, and so that load-shedding logic that throttles port 8080 never accidentally throttles a health probe. The `startupProbe` gives a 30-second boot budget (10 × 3s) so the slow cache warm never trips liveness. The `livenessProbe` has `failureThreshold: 3` — meaning a pod must look wedged for 30 consecutive seconds before it's restarted, which is deliberately forgiving so that a single GC pause or a brief CPU spike on a busy-but-healthy pod never triggers a needless restart. The `readinessProbe` fails after a *single* miss (so a draining pod leaves rotation fast) but requires two successes to rejoin (so it doesn't *flap* back in on one lucky probe). And the `preStop` hook sleeps 5 seconds, which closes a notorious race we'll discuss next.

### The startup endpoint, briefly

The startup endpoint can be as simple as a flag set once boot finishes:

```go
// internal/health/startupz.go
package health

import (
	"net/http"
	"sync/atomic"
)

var booted atomic.Bool

// MarkBooted is called once, after migrations run, the DB pool is established,
// and the pricing cache has finished its initial warm load.
func MarkBooted() { booted.Store(true) }

func StartupzHandler(w http.ResponseWriter, r *http.Request) {
	if !booted.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}
```

Once `/startupz` returns 200, the kubelet stops probing it forever and begins running liveness and readiness. The beauty of the startup probe is that it lets you keep the *liveness* probe tight (period 10s, threshold 3 = react to a real wedge in 30s) without that tightness murdering a slow boot — the two concerns are cleanly separated, which is the recurring theme of this entire post.

## Graceful shutdown: draining without dropping requests

Now the other half of readiness's job — and the half juniors almost always get wrong, dropping requests on every single deploy without realizing it. When Kubernetes deploys a new version, it sends each old pod a `SIGTERM` and waits up to `terminationGracePeriodSeconds` before sending `SIGKILL`. The naive service hears `SIGTERM` and immediately calls `os.Exit(0)`, instantly dropping every in-flight request — and there are always in-flight requests on a busy service. The correct service performs a careful four-step dance, and readiness is the linchpin.

The four steps, in order: (1) on `SIGTERM`, **flip readiness to failing** so the load balancer stops sending new requests; (2) **wait** for the load balancer to actually notice and drain the route (this is what the `preStop` sleep buys you, because endpoint propagation is *not* instantaneous); (3) **finish the in-flight requests** (call the HTTP server's graceful `Shutdown`, which stops accepting new connections but lets active ones complete); (4) **exit cleanly**. Here is the wiring, tying back to the `ReadyChecker.StartDraining()` we defined earlier.

```go
// cmd/order-service/shutdown.go
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func runWithGracefulShutdown(srv *http.Server, ready *health.ReadyChecker) {
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	<-stop // block until the platform asks us to stop

	// STEP 1: fail readiness so /readyz returns 503 and the LB drains us.
	ready.StartDraining()

	// STEP 2: wait for the load balancer to remove us from rotation.
	// Endpoint propagation is eventually-consistent; without this sleep,
	// the LB may still route to us after we stop accepting connections.
	// (The preStop hook in the manifest does the same job from outside.)
	time.Sleep(5 * time.Second)

	// STEP 3: stop accepting new connections; let in-flight requests finish.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		// Past the grace window with requests still running: forced close.
		_ = srv.Close()
	}
	// STEP 4: return; process exits 0 with zero dropped requests.
}
```

The subtle, expensive bug that this code avoids is the **endpoint-propagation race**. When a pod fails readiness, the kubelet updates the Endpoints object, then the Service's iptables/IPVS rules update, then (if you use one) the external load balancer's pool updates. That chain takes a few hundred milliseconds to a few seconds. If the pod stops accepting connections the instant it gets `SIGTERM`, requests routed during that propagation window hit a dead listener and fail. The `preStop` sleep (and the matching `time.Sleep` in code) holds the process open *while still serving* during the propagation window, so by the time it actually stops accepting connections, the load balancer has already drained it. This is the difference between a deploy that drops zero requests and one that drops a few hundred every single time — silently, because nobody's watching the error rate during a routine deploy.

![A six event timeline of a graceful rolling deploy that fails readiness drains in flight requests then exits cleanly dropping zero requests](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-6.webp)

The timeline above shows the full sequence on a rolling deploy: SIGTERM arrives, readiness flips to failing, the kubelet removes the pod from Endpoints, the propagation window passes while in-flight requests drain, the server closes cleanly, and the new pod is ready — zero requests dropped across the whole transition.

#### Worked example: requests dropped per rollout, gated vs ungated

ShopFast deploys the order service ~6 times a day. The fleet is 40 pods, 2,000 RPS, average request duration 120ms. A rolling deploy replaces pods in batches; assume each old pod has, on average, ~6 requests in flight at any instant (50 RPS × 0.12s ≈ 6).

**Ungated deploy** (no readiness flip, no preStop, immediate `os.Exit`): each retiring pod drops its ~6 in-flight requests *plus* whatever the load balancer routes to it during the ~1-second propagation window before it's removed — at 50 RPS that's another ~50 requests hitting a dead listener. Call it ~56 dropped per pod. Across 40 pods per rollout: **~2,240 failed requests per deploy.** At 6 deploys/day that's ~13,400 user-facing errors per day, *entirely from deploys*, invisible unless you correlate error spikes with rollout timestamps. (This is exactly the kind of "mystery" 0.1% error rate that haunts teams.)

**Readiness-gated deploy** (flip readiness, 5s preStop drain, graceful `Shutdown`): the pod leaves rotation before it stops serving, the propagation-window requests still land on a live listener, and the ~6 in-flight requests complete during the drain. Dropped per pod: **0.** Across 40 pods, 6 deploys/day: **0 failed requests from deploys.** The fix costs you a `sleep 5` and twenty lines of shutdown code, and it converts ~13,400 daily errors into zero.

![A before and after comparison of an ungated rolling deploy dropping requests at startup and shutdown versus a readiness gated rollout dropping zero](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-8.webp)

### Why the startup side of a deploy also drops requests

The shutdown side gets all the attention, but a rolling deploy has *two* windows where an ungated service bleeds requests, and the startup window is the one juniors never think about. When a new pod comes up, Kubernetes adds it to the Endpoints object as soon as the *container* is running — but if you have no readiness probe (or a trivial one that returns 200 before the app is actually ready), the load balancer starts routing real traffic to that pod while its connection pool is still empty and its caches are still cold. Those early requests either error out (no DB pool yet) or run pathologically slow (every request is a cache miss against a cold cache, so they all hit the database, which is now being hammered by a cold pod). For ShopFast, a cold pod with an unwarmed pricing cache might run at 10× its normal latency for the first few seconds and return errors until the pool establishes — and if the rollout is bringing up several pods at once, you get a cluster of cold pods all degrading their share of traffic simultaneously. The readiness probe closes this window from the other side: it holds the pod *out* of the Endpoints object until `/readyz` confirms the pool is up and the cache is warm, so the load balancer never routes a single real request to a pod that can't serve it well. This is why a *good* readiness probe is not just about shedding traffic from sick pods — it's equally about *withholding* traffic from not-yet-healthy ones. The deploy choreography that drops zero requests needs both halves: readiness gating new pods *in* only when ready, and readiness draining old pods *out* before they die.

There's a second-order consequence worth naming for the senior reader: this interaction between cold starts and readiness is *why* the `maxSurge` / `maxUnavailable` knobs and the cache-warm time interact with your deploy duration. If each new pod takes 23 seconds to warm and you replace 25% of a 40-pod fleet per batch (10 pods, `maxSurge: 25%`), each batch takes ~23 seconds of warm time before those pods count as ready and the next batch begins. So the full rollout takes roughly `(40 / 10) × 23s ≈ 92 seconds` — and that's *with* readiness gating. Without readiness gating it would appear "faster" but would be dropping requests the whole time. The right mental frame: readiness gating doesn't slow your deploy down so much as it makes the deploy *honest* about how long it actually takes to safely roll, which is exactly the time the new pods need to be genuinely ready. Optimizing deploy speed, then, often means optimizing *boot time* (lazy-load the cache, parallelize pool establishment, ship a smaller image) rather than weakening the readiness gate — weakening the gate just hides the cost in dropped requests instead of paying it honestly in deploy duration.

## Self-healing: what the platform does automatically

Health checks are the *input*; self-healing is the *output*. Once the platform can trust your probes, it repairs the system without a human. But "self-healing" is not one mechanism — it's a toolbox of distinct remedies, each triggered by a different signal and operating at a different layer. Knowing which is which is how you reason about *why the platform did that* at 3am.

![A tree of self healing mechanisms split into instance level repair and fleet level capacity each driven by a different health signal](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-7.webp)

The tree above splits self-healing into two families. **Instance repair** fixes one bad pod: a *liveness* failure restarts a wedged container; a *readiness* failure drops a pod from the load balancer (no restart); a *node* that goes `NotReady` for too long triggers the node controller to evict its pods and reschedule them elsewhere, and a node-auto-repair / cluster-autoscaler may replace the failed machine entirely. **Fleet capacity** right-sizes the herd: the *Horizontal Pod Autoscaler* (HPA) adds or removes replicas based on load, and a *deployment controller* (or a progressive-delivery tool like Argo Rollouts or Flagger) automatically *rolls back* a deploy whose new pods fail their health checks. Each of these reads a different signal. If you don't know which signal drove an action, you can't debug it — so let's look at the two most consequential: autoscaling and automated rollback.

The **node-level** healing deserves a moment because it's the layer juniors forget and seniors rely on. A pod can be perfectly healthy and still go dark because the *machine* underneath it died — a kernel panic, a hardware fault, a network partition that isolates the node, a disk filling up. The kubelet on that node stops sending heartbeats to the control plane; after the node-monitor grace period (40s by default) the node is marked `NotReady`, and after the pod-eviction timeout (5 minutes by default) the node controller evicts the pods and the scheduler recreates them on healthy nodes. This is why `replicas: 4` across a *spread* of nodes matters: if all four order pods landed on the same node and that node died, you'd have a five-minute outage waiting for reschedule; spread across nodes (with pod anti-affinity or topology-spread constraints), a single node failure costs you one of four pods and the load balancer routes around it instantly via readiness. The five-minute eviction default is deliberately patient — a brief network blip between a node and the control plane should *not* trigger mass pod rescheduling (that would be a node-level restart storm) — but it means node-level healing is your *slow* safety net, while readiness-driven traffic routing is your *fast* one. The fast loop (readiness) protects users in seconds; the slow loop (node eviction) restores capacity in minutes.

### Autoscaling: the HPA control loop

The Horizontal Pod Autoscaler is a closed feedback loop. It observes a metric (CPU utilization, or a custom metric like requests-per-second), compares it to a target, and adjusts the replica count to drive the observed value toward the target. The desired-replicas formula is, in essence:

```
desiredReplicas = ceil( currentReplicas × (currentMetric / targetMetric) )
```

So if you're running 4 pods at 90% CPU against a 60% target, the HPA computes `ceil(4 × 90/60) = ceil(6) = 6` and scales to 6. Crucially — and this ties directly back to readiness — **the HPA only counts pods that are Ready** when computing the current metric, and a pod that has just been added is given time to become ready before its (initially low) utilization drags the average down and triggers a counterproductive scale-*down*. A broken readiness probe sabotages autoscaling: if pods never report ready, the HPA can't measure them and scaling stalls; if pods flap ready, the metric oscillates and the HPA thrashes.

![A graph of the horizontal pod autoscaler feedback loop comparing observed load to a target and scaling replicas that only count when ready](/imgs/blogs/health-checks-readiness-liveness-and-self-healing-9.webp)

Here is a real HPA manifest for ShopFast's order service, scaling on both CPU and a custom RPS-per-pod metric, with the stabilization windows that prevent thrashing:

```yaml
# k8s/order-service-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 4         # never below 4 (handles a single-pod failure cleanly)
  maxReplicas: 40        # cap so a traffic spike can't bankrupt us
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 60 }
    - type: Pods
      pods:
        metric: { name: http_requests_per_second }
        target: { type: AverageValue, averageValue: "50" }  # 50 RPS/pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30   # react to spikes quickly
      policies:
        - type: Percent
          value: 100                   # at most double per 30s
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300  # scale down slowly (avoid flapping)
      policies:
        - type: Percent
          value: 25                    # shed at most 25% per minute
          periodSeconds: 60
```

The asymmetry between `scaleUp` and `scaleDown` is the production wisdom here. Scaling *up* is fast (a 30-second window, allowed to double) because being under-provisioned during a traffic spike costs you a real outage. Scaling *down* is slow (a 300-second window, at most 25% per minute) because being briefly over-provisioned costs you a little money, but scaling down too aggressively means you immediately have to scale back up on the next request wave — *flapping*, which both wastes resources and repeatedly subjects new pods to cold starts.

#### Worked example: HPA scaling ShopFast 4 → 40 for a flash sale

ShopFast runs a flash sale at noon. Baseline traffic is 200 RPS across 4 pods (50 RPS/pod, right at target). At 12:00:00 traffic jumps to 2,000 RPS — a 10× spike.

At the spike, each of the 4 pods is suddenly handling 500 RPS against a 50 RPS/pod target. The HPA computes `ceil(4 × 500/50) = ceil(40) = 40` and wants 40 pods. But the `scaleUp` policy caps growth at "double per 30s," so the trajectory is staged: **4 → 8 (12:00:30) → 16 (12:01:00) → 32 (12:01:30) → 40 (12:02:00, hitting the cap).** It takes about two minutes to reach full capacity. During those two minutes the existing pods are overloaded — which is exactly why you *also* need [load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) to protect them while the fleet scales, because autoscaling is not instantaneous and the spike is. Note also that each new pod has a ~23-second cold start before its readiness passes and it actually absorbs load, so the *effective* capacity lags the pod count by ~23 seconds. This is why pre-scaling before a *known* event (a scheduled `minReplicas` bump to 40 at 11:55) beats reactive scaling for a flash sale: you eat the cold starts before the traffic arrives, not during it. When the sale ends and traffic falls back to 200 RPS, the slow `scaleDown` window walks the fleet back from 40 to 4 over ~15 minutes, never flapping.

### Automated rollback: health checks gate the deploy

The last self-healing mechanism is the one that prevents a bad release from becoming an outage: a deploy that watches the *new* pods' health and rolls back automatically if they're unhealthy. A vanilla Kubernetes rolling update already does a primitive version of this — it brings up new pods and waits for them to pass readiness before terminating old ones, so a new version that *never* becomes ready (e.g. it crashes on boot) will stall the rollout rather than replace healthy pods with broken ones. The `maxUnavailable` and `maxSurge` knobs control how aggressively it proceeds:

```yaml
# In the Deployment spec — controls a safe rolling update.
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0    # never drop below desired capacity during deploy
      maxSurge: 25%        # bring up 25% extra new pods, then retire old ones
  minReadySeconds: 15      # a new pod must stay Ready 15s before it "counts"
```

`maxUnavailable: 0` guarantees you never lose capacity during a deploy, and `minReadySeconds: 15` is a quiet but important safety: a new pod must remain *continuously* ready for 15 seconds before the rollout considers it good and proceeds to retire an old pod. This catches the nasty case where a new version passes readiness once, then immediately crashes — without `minReadySeconds`, the rollout would march forward replacing every old pod with a flapping new one. For real progressive delivery — canary, blue-green, automated rollback on elevated error rates — you reach for the [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) covered later in the series, where a controller watches golden-signal metrics on the canary and aborts the rollout if they degrade.

## The failure modes of self-healing (and how to defuse each)

Self-healing is a set of feedback loops, and every feedback loop can oscillate, overshoot, or run away. Here are the ones that page you, with the cause and the fix for each.

**Restart storms.** Cause: liveness checks a shared dependency, so a dependency blip fails liveness on all instances at once and restarts the entire fleet (the ShopFast 02:14 outage). Fix: keep liveness shallow; *never* check a dependency in liveness. If you must check a dependency at all, do it in readiness, where the action is "drop from LB" not "restart."

**Readiness flapping.** Cause: a readiness check is too sensitive (e.g. it fails on a single slow dependency call), so pods oscillate in and out of the load balancer, and traffic thrashes between a shrinking and growing pool. Fix: require multiple consecutive successes to rejoin (`successThreshold: 2`+), give the dependency check a sane timeout, and consider a "degraded but serving" state (below) instead of binary not-ready.

**The too-aggressive liveness probe.** Cause: a tight `failureThreshold` (e.g. 1) plus a short timeout means a single GC pause, a brief CPU spike, or a momentarily slow probe response kills a busy-but-perfectly-healthy pod — and under load, when probes are slowest, you kill your *busiest* pods, making the overload worse. Fix: be generous with liveness — `failureThreshold: 3`+ and a `timeoutSeconds` comfortably above your p99 probe latency. Liveness should react to a *genuine, sustained* wedge (tens of seconds), not a transient.

**The thundering herd on recovery.** Cause: when a dependency recovers (or a fleet restarts), all instances reconnect and re-warm simultaneously, hammering the just-recovered dependency and possibly knocking it down again. Fix: jittered reconnect/backoff, staggered cache warming, connection-pool ramp-up, and — the structural fix — *don't restart the whole fleet at once in the first place* (which loops back to shallow liveness). A herd that never died can't stampede on recovery.

**The health-check thundering herd itself.** Cause: too-frequent probes (period 1s) across thousands of pods generate real load, and a deep readiness probe that does a DB query on every tick can put meaningful query load on the database just from health checking. Fix: probe periods of 5–15s are plenty; cache the result of an expensive readiness check for a couple of seconds so a burst of probes doesn't multiply into a burst of dependency calls.

**Liveness/readiness deadlock during overload.** Cause: under heavy CPU load, the probe endpoint itself starves and times out, the kubelet declares the pod dead and restarts it, which removes capacity and worsens the overload on the survivors — a positive feedback loop into total collapse. Fix: run health endpoints on a separate, lightly-loaded path; ensure the probe handler never competes with request handling for the same starved resource; and pair this with load shedding so the pod stays responsive enough to answer probes even when it's rejecting work.

### Stress-testing the design

A senior never ships a health-check design without running the three questions the kit demands of every resilience post. Let's run them against ShopFast.

*A downstream blips — do all instances restart?* With shallow liveness: **no.** A payment or recommendation outage touches nothing in `/livez`. A *database* blip fails `/readyz` (correctly, since the DB is mandatory) and pulls pods from rotation, but it does not restart them, and they rejoin the instant the DB recovers. Zero restarts. The design passes.

*A slow boot — does liveness kill it during startup?* With a startup probe granting a 30-second boot budget: **no.** Liveness doesn't run until `/startupz` passes, so the 20-second cache warm completes safely. Without the startup probe, the same liveness config (period 10s, threshold 3) would kill the pod at ~30s — but that 30s window is consumed *by the boot itself*, so a marginally slow boot would trip it and loop forever. The startup probe is what makes the tight liveness probe safe. The design passes.

*A deploy mid-request — are requests dropped?* With the readiness-flip + preStop + graceful `Shutdown` dance: **no.** The pod leaves rotation before it stops serving, the propagation race is closed by the 5-second drain, and in-flight requests complete. Zero dropped. The design passes. (Without graceful shutdown: ~2,240 dropped per deploy, as computed above. The design fails.)

## Designing a good health endpoint: degraded vs unhealthy

One more refinement that separates a senior's health endpoint from a junior's: the recognition that "ready" is not always binary. A service can be in three states, not two: **healthy** (serve everything), **degraded** (serve a reduced set — e.g. the order service can take orders but can't show personalized recommendations because that dependency is down), and **unhealthy** (can't do its core job — DB is down — so drop from rotation). The mistake is to model degraded as unhealthy and pull a perfectly-serving pod from rotation because an *optional* dependency is down.

The clean design: `/readyz` returns 200 (ready) as long as the *core* job is doable — it checks only mandatory dependencies. A *separate* `/healthz` or `/health/detail` endpoint, consumed by dashboards and humans (not by the readiness gate), reports the full picture including degraded subsystems. This keeps the traffic-gating signal honest and narrow while still giving your observability stack the rich detail it needs.

```go
// internal/health/detail.go — for humans/dashboards, NOT for the readiness gate.
type Detail struct {
	Status string            `json:"status"` // "healthy" | "degraded" | "unhealthy"
	Checks map[string]string `json:"checks"`
}

func (rc *ReadyChecker) Detail() Detail {
	d := Detail{Status: "healthy", Checks: map[string]string{}}
	// Mandatory: DB down => unhealthy (this is what /readyz also reflects).
	if rc.pingDB() != nil {
		d.Status = "unhealthy"
		d.Checks["postgres"] = "down"
	} else {
		d.Checks["postgres"] = "up"
	}
	// Optional: recommendations down => degraded, but STILL ready to serve.
	if !rc.recsReachable() {
		if d.Status == "healthy" {
			d.Status = "degraded"
		}
		d.Checks["recommendations"] = "down (degraded, still serving)"
	} else {
		d.Checks["recommendations"] = "up"
	}
	return d
}
```

The principle: the readiness *gate* must be narrow (mandatory deps only), while the *observability* of health can be rich. Don't let a beautiful detailed health report leak into the traffic-gating decision, or you'll pull serving pods over a recommendation outage. This connects directly to the [golden signals and SLOs](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) post — the detail endpoint feeds dashboards; the SLO is measured on actual served requests, not on probe results.

## Optimization: tuning probes for production

Once the design is correct, the tuning is where you buy reliability and cut waste. The knobs and their effects, with concrete guidance:

**`periodSeconds`** — how often the kubelet probes. The trade-off is detection latency vs probe load. For liveness, 10s is a good default (you detect a wedge within `period × failureThreshold` = 30s, which is fine for a sustained wedge). For readiness, 5s gives faster drain on shutdown and faster recovery on dependency healing. Going below ~2s rarely helps and adds real load at scale (thousands of pods × frequent probes). **Measure:** the kubelet's probe latency; if a deep readiness probe runs longer than its period, it can't keep up.

**`timeoutSeconds`** — how long before a probe counts as failed. Set it comfortably above your probe's p99 latency. A liveness timeout of 1s on a service whose probe p99 is 800ms under load will produce *false* liveness failures exactly when the service is busiest — the worst possible time to restart a pod. Rule of thumb: `timeoutSeconds ≥ 2 × probe-p99`, and keep the probe handler so cheap its p99 is single-digit milliseconds.

**`failureThreshold` / `successThreshold`** — the hysteresis. Liveness wants a *high* failure threshold (3+) so a transient never restarts a healthy pod. Readiness wants a *low* failure threshold (1) so a draining/unhealthy pod leaves rotation fast, but a *higher* success threshold (2) so it doesn't flap back in. Startup wants a high failure threshold (boot-budget ÷ period) and a `successThreshold` of 1.

**Separate concerns physically.** Health endpoints on a dedicated port (8081), so request-side load shedding never throttles probes; readiness-check results cached for ~2s so a probe burst doesn't multiply into a dependency-call burst; the liveness probe touching only in-process state so it stays single-digit-millisecond fast under any dependency condition.

**Degraded mode** as discussed: keep readiness narrow so optional-dependency failures don't shed capacity. The measurable win: during a recommendation-service outage that lasts 10 minutes, a *narrow* readiness keeps all 40 order pods serving (orders succeed, recommendations are absent) — error rate on the core order flow stays at baseline. A *wide* readiness that checked recommendations would pull all 40 pods, taking the *entire* order flow down over an *optional* dependency — converting a cosmetic degradation into a Sev-1. The optimization is, once again, mostly about *what you leave out*.

#### A note on probe protocol choice

`httpGet` is the default and right choice for most services. `tcpSocket` (does the port accept a connection?) is weaker — it proves the listener is up but not that the app behind it can serve, so it's a poor readiness check (it'll route traffic to a pod whose app is wedged behind an open socket). `exec` (run a command in the container) is the most flexible but the most expensive — forking a process every few seconds across thousands of pods adds real overhead, and a slow `exec` probe under CPU pressure is a classic false-liveness source. For gRPC services, use the standard gRPC health-checking protocol (`grpc_health_v1`) via the `grpc` probe type rather than bolting on an HTTP endpoint. Prefer `httpGet`; reach for the others only with a specific reason.

#### Make your probes observable

A health-check failure that you only discover by noticing pods restarting is a health-check failure you'll debug far too slowly. The optimization that pays for itself in the first incident is *instrumenting the probes themselves*. Emit a metric every time a readiness probe flips state (`readiness_state{pod, ready}`) and every time the kubelet restarts a container (`kube_pod_container_status_restarts_total`), and alert on their *rates*, not their absolute values. A single restart is noise; a restart *rate* climbing across the fleet is the leading indicator of a restart storm in progress — and if you catch it in the first thirty seconds you can roll back the bad probe config or scale the dependency before it cascades. Likewise, graph the *readiness flap rate*: pods entering and leaving rotation should be near-zero in steady state; a sudden flap rate is readiness oscillating, which tells you a dependency is marginal or a threshold is too twitchy before users feel it. The detailed `/health/detail` endpoint from earlier feeds your dashboard the *why* (which subsystem is degraded); these fleet-level probe metrics feed your *alerts* the *that-it's-happening*. This is the bridge to the [SLOs and golden signals](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) post: probe state changes are a golden-signal input, and a well-instrumented health system turns "we found out when the pager went off" into "we saw the flap rate climb and intervened before the SLO burned."

## Case studies

**The cascading-restart pattern is real and recurring.** The single most documented health-check anti-pattern across the industry is the liveness probe that checks a dependency, and it has caused real outages at real companies. The mechanism is always the same as ShopFast's 02:14 incident: a shared dependency degrades, every instance fails liveness simultaneously, the orchestrator restarts the entire fleet, and a transient dependency blip becomes a prolonged total outage amplified by cold-start thundering herds. The lesson, articulated repeatedly in production post-mortems and in Kubernetes guidance itself, is blunt: a liveness probe must depend on *nothing but the local process*. The order-of-magnitude amplification (a 30-second blip becoming a 10+ minute outage) is consistent across the public write-ups, and it's the reason the "liveness ≠ readiness" rule is now treated as gospel.

**Kubernetes probe best practices evolved through painful lessons — the addition of the startup probe.** Early Kubernetes had only liveness and readiness probes. Teams running slow-booting applications (large JVM services, ML model servers, apps with heavy migrations) discovered a nasty bind: to give a slow boot enough time, they had to set a huge `initialDelaySeconds` on liveness — but that same delay then made the running service slow to detect a *real* wedge, because the only knob was shared between "wait for boot" and "react to a wedge." Setting it short killed slow boots in a restart loop; setting it long meant a genuinely-wedged production pod sat dead for minutes. The Kubernetes project added the **startup probe** (stable in 1.18) specifically to decouple these two concerns: the startup probe owns the boot budget, then hands cleanly to a *tight* liveness probe. This is a textbook case of the platform evolving to encode a hard-won operational lesson — the same "separate the concerns" principle this whole post argues for, baked into the API.

**The graceful-drain win.** Teams that adopt the readiness-flip-then-drain pattern on shutdown report the same result: deploy-time error rates that were a persistent, mysterious 0.05–0.5% drop to *zero*. The pattern is now standard enough that it's documented in the Kubernetes "termination of pods" guidance, and progressive-delivery tools (Argo Rollouts, Flagger) and service meshes (Istio, Linkerd — see the [service mesh post](/blog/software-development/microservices/service-discovery-and-load-balancing) for how meshes own the endpoint update) build it in. The lesson is that the endpoint-propagation race is *real and silent* — it doesn't show up in functional tests because tests don't deploy under live traffic — and the cheap `preStop` sleep plus graceful `Shutdown` is the canonical fix. Companies running hundreds of deploys a day cannot tolerate dropping requests on each one; for them, this isn't optional polish, it's a hard requirement of independent deployability.

**Self-healing as the substitute for human toil.** The broader industry story — Google's SRE practice being the canonical articulation — is that at the scale of hundreds or thousands of services and tens of thousands of pods, you *cannot* have humans deciding which instances to restart and which to drain. The platform must do it from health signals, which means the health signals must be *trustworthy* and the actions must be *correct*. Netflix's chaos engineering exists precisely to verify that the self-healing loops actually heal (and don't self-harm) under real failure injection — you don't *know* your liveness probe is shallow enough until you blip a dependency in production and watch the fleet *not* restart. The maturity arc of a microservices org runs straight through this post: junior teams write probes that look thorough and cause restart storms; senior teams write probes that look almost suspiciously simple and never amplify an outage.

## Trade-offs: the depth dial and what it costs

Every choice in health checking is a dial between two failure modes, and there is no setting that avoids both — you are always choosing *which* way you'd rather be wrong. Making this explicit is what separates a thoughtful configuration from cargo-culted YAML, so here is the decision framing for each major knob, with the cost of each direction named.

**Probe depth (liveness).** Shallow vs deep. The cost of *too shallow* is that a genuinely broken pod — one whose request handling is failing but whose event loop still ticks — passes liveness and never gets restarted, so it lingers as a "zombie" returning errors until readiness or outlier detection catches it. The cost of *too deep* is the restart storm. Given that asymmetry, the correct bias is unambiguous: **err shallow.** A lingering zombie pod is a bounded, single-instance problem that readiness and the load balancer's passive checks will catch; a restart storm is an unbounded, fleet-wide outage. You would rather occasionally fail to restart one bad pod than occasionally restart all forty good ones. This is why "the senior's liveness probe looks suspiciously simple" — the simplicity is a deliberate bias toward the cheaper failure.

**Probe depth (readiness).** Narrow (mandatory deps only) vs wide (all deps). The cost of *too narrow* is that a pod with a broken optional dependency stays in rotation and serves degraded responses — usually acceptable, because that's exactly what graceful degradation is for. The cost of *too wide* is that an *optional* dependency outage sheds your *entire* fleet from rotation, converting a cosmetic degradation into a total outage. Again the bias is clear: **err narrow.** Put only the dependencies you genuinely cannot serve without in the readiness gate.

**Probe frequency.** Frequent vs sparse. Frequent probes (1–2s) detect a wedge or a recovery faster but generate real load at scale and amplify any per-probe dependency cost. Sparse probes (15–30s) are cheap but slow to react. The sweet spot for most services: liveness 10s, readiness 5s, startup 3s. The detection latency this buys (30s to restart a wedge, 5s to drain on shutdown) is fine for the vast majority of services, and the load is negligible.

**Failure threshold (hysteresis).** Twitchy vs sluggish. A low threshold reacts fast but false-positives on transients (a GC pause, a CPU spike); a high threshold ignores transients but is slow on real failures. The resolution is to set it *per probe according to the cost of acting*: liveness, whose action is expensive (restart), gets a high threshold (3+); readiness's *fail* direction, whose action is cheap and reversible (drop from LB), gets a low threshold (1), but its *recover* direction gets a higher success threshold (2) to avoid flapping. The hysteresis is asymmetric on purpose.

The unifying rule across all four dials: **the more violent the action a probe triggers, the more conservative and shallow that probe should be.** Liveness triggers the most violent action (restart), so it is the shallowest and most forgiving. Readiness triggers a gentle, reversible action (traffic gating), so it can be deeper and twitchier. Startup triggers the gentlest action (just wait), so it can be the most patient. The decision matrix figure earlier in this post is, at its heart, this single principle rendered as a table — and if you remember nothing else, remember to match probe aggressiveness *inversely* to action severity.

## When to reach for what (and when not to)

Health checks aren't optional in an orchestrated environment — if you run on Kubernetes, you *will* configure probes, and the only question is whether you configure them correctly. So the "when to use" here is really "which probe, how deep, and what to leave out."

**Always run a liveness probe, and keep it shallow.** Even a trivial "return 200" liveness probe is worth having, because it lets the platform restart a genuinely-wedged process. Just never let it depend on anything external. If you're unsure whether to add a check to liveness, the default is *don't*.

**Run a readiness probe whenever the service has a non-trivial boot or mandatory dependencies.** A stateless service with no dependencies and an instant boot can arguably skip a deep readiness check (a trivial 200 suffices), but the moment it has a connection pool to establish, a cache to warm, or a graceful-shutdown story, readiness earns its keep. Keep it to *mandatory* dependencies only.

**Run a startup probe whenever boot takes more than a few seconds**, or whenever boot time varies (cold caches, migrations). It's the clean way to keep liveness tight without murdering slow boots. A fast-booting stateless service can skip it.

**Reach for HPA when load varies** and the work is horizontally scalable (stateless or with externalized state). Don't autoscale a stateful singleton, and don't autoscale on a lagging metric without a stabilization window unless you enjoy flapping.

**When NOT to over-engineer:** if you're running a single instance of a hobby service behind a process manager, the elaborate three-probe + graceful-drain + HPA machinery is overkill — a `systemd` restart-on-failure is plenty. This machinery earns its complexity at *fleet* scale, where automated, correct self-healing is the only thing that lets a small team operate many services. The cost of getting it *wrong* (restart storms, dropped deploys) also scales with the fleet, which is exactly why the discipline matters more the bigger you get.

## Key takeaways

1. **A health check is a control signal, not a status report.** For every line in a probe, ask "do I want the platform to take this probe's *action* when this line fails?" — not "is this part of the system working?"
2. **Liveness restarts; readiness gates traffic; startup buys boot time.** Three probes, three different questions, three different actions. Conflating them is the root cause of most health-check outages.
3. **Liveness must be shallow and must NEVER check a dependency.** A dependency check in liveness turns a 30-second blip into a fleet-wide restart storm — a ~24× outage amplification, self-inflicted by the platform's own self-healing.
4. **Readiness may be deep, but only for *mandatory* dependencies.** Checking an optional dependency in readiness converts a cosmetic degradation into a full outage by shedding all your capacity. Use a "degraded but serving" state instead.
5. **Graceful shutdown is readiness's other job.** Flip readiness to failing, wait out the endpoint-propagation race with a `preStop` drain, finish in-flight requests with a graceful `Shutdown`, then exit. This converts hundreds of dropped requests per deploy into zero.
6. **Be generous with liveness, fast-to-fail and slow-to-recover with readiness.** A high liveness `failureThreshold` stops transients from killing busy-healthy pods; a low readiness `failureThreshold` with a higher `successThreshold` drains fast without flapping.
7. **Self-healing is a toolbox, not one thing.** Liveness→restart, readiness→drop-from-LB, node-controller→reschedule, HPA→scale, deploy controller→rollback. Know which signal drives which action, or you can't debug it at 3am.
8. **Every feedback loop can run away.** Restart storms, readiness flapping, thundering herds, and probe-starvation deadlocks are all self-healing loops oscillating. The structural fix for most of them is keeping liveness shallow so the fleet never dies all at once.
9. **The senior's probe looks suspiciously simple.** Juniors write "thorough" probes that cause outages; seniors write narrow probes that never amplify a failure. Simplicity in a liveness probe is not laziness — it's the correctness argument.

## Further reading

- *Building Microservices* (Sam Newman) — the chapters on deployment, resilience, and operating services at scale frame why the platform contract matters.
- *Microservices Patterns* (Chris Richardson) — the "Observable services" and health-check API patterns, with the readiness/liveness distinction.
- *Site Reliability Engineering* (Google) — the canonical argument for automated self-healing over human toil, and the discipline of trustworthy signals.
- Kubernetes documentation: "Configure Liveness, Readiness and Startup Probes" and "Termination of Pods" — the authoritative reference for probe semantics and the graceful-drain sequence.
- This series: [Anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) (where graceful shutdown and health endpoints first appear), [Service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) (the Endpoints object readiness writes to), [Resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads), [Handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation), and [Rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding).
- Coming next in the series: [Kubernetes for microservices: the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials), [Deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags), and [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices).
