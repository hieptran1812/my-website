---
title: "Service Discovery and Load Balancing: How Service A Finds Service B"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "In a fleet where pods come and go every minute, hardcoding a host is a bug waiting to page you. Learn how a service registry, client-side and server-side discovery, and health-aware load balancing let one service find and spread load across the many ephemeral instances of another, and what breaks when the registry goes stale."
tags:
  [
    "microservices",
    "service-discovery",
    "load-balancing",
    "distributed-systems",
    "software-architecture",
    "backend",
    "kubernetes",
    "service-mesh",
    "resilience",
    "consul",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/service-discovery-and-load-balancing-1.webp"
---

It is 11:58 on the morning of ShopFast's biggest flash sale of the year, and an engineer is staring at a graph that does not make sense. The inventory service is healthy. Every one of its pods reports green. CPU is comfortable, memory is comfortable, the database behind it is barely breathing. And yet the order service — the thing customers actually touch when they hit "buy" — is throwing 503s at a steady drip, roughly one request in twenty, with no pattern anyone can see. The on-call engineer pulls up the order service's logs and finds the smoking gun: a stream of `connection refused` errors, all pointed at a single IP address, `10.2.4.31`. That IP belonged to an inventory pod. Belonged. The pod was killed forty seconds ago by the autoscaler doing a routine rebalance, but the order service never got the memo. It is still, dutifully, sending one out of every twenty requests to a corpse.

This is the problem at the heart of every microservices deployment, and it is so fundamental that juniors usually do not even see it as a problem until it pages them. In a monolith, "where is the inventory code" has a trivial answer: it is in the same process, you call a function, the address is resolved at link time and never changes. The moment you split inventory into its own service running on its own pods, that question — *where is service B, right now?* — becomes a live, constantly-changing thing. Pods are ephemeral. The autoscaler adds them during the sale and removes them after. A deploy rolls every pod to a new IP. A node fails and Kubernetes reschedules its pods elsewhere. A health check trips and a pod is quietly pulled. The set of places where "the inventory service" actually lives is a moving target that can turn over completely in minutes, and any caller that assumes otherwise is the engineer staring at the graph at 11:58.

![A topology diagram showing the order service asking a registry for live inventory endpoints, the load balancer picking among healthy pods while a starting pod and a dead pod sit to the side](/imgs/blogs/service-discovery-and-load-balancing-1.webp)

So the two questions this post answers are deceptively simple and endlessly deep. First, *discovery*: how does the order service find out which inventory pods exist right now, without anyone editing a config file? Second, *load balancing*: given that there are N of them — and N is anywhere from 4 to 40 today — how does it spread requests across them so that no single pod melts, slow pods do not drag down the tail, and a pod that dies mid-request does not take a customer's order with it? We will build the answer the way the industry built it: start with the naive hardcoded host and watch it fail, introduce the service registry and the lifecycle of an instance registering and dying, then split the two great schools — client-side discovery, where the caller does the picking, and server-side discovery, where a proxy does it for them. We will work through the load-balancing algorithms (round-robin, least-request, latency-aware, consistent hashing) with actual numbers showing when each wins, wire health checks into the balancer so it only ever routes to ready pods, and then stress-test the whole thing: a pod dies mid-request, the registry goes stale, the registry itself partitions. By the end you should be able to look at any service-to-service call in your fleet and answer the question the 11:58 engineer could not: *how does this caller know who is alive, and what happens the instant one of them dies?*

This post sits in the communication track of the series, right after [the API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) and built directly on the foundations in [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — specifically fallacy five, "topology doesn't change," which is the fallacy this entire post is dedicated to defeating.

## The naive approach and exactly how it dies

Let us start where every team starts, because the failure is instructive. The first version of the order service calls inventory like this:

```yaml
# The naive config: a hardcoded host. Works in the demo, dies in production.
inventory:
  endpoint: "http://10.2.4.31:8080"
  # ...or, slightly less naive, a single DNS name:
  # endpoint: "http://inventory.internal:8080"
```

The hardcoded IP version dies the first time that pod is rescheduled, which in Kubernetes can be *minutes* into its life. There is no recovery — the order service will hammer a dead address until someone redeploys it with a new IP, which is insane. So teams reach for the DNS name version, and DNS feels like it should solve everything: `inventory.internal` resolves to an IP, the IP can change, problem solved. But plain DNS has three sharp edges that bite hard in a microservices fleet.

First, **DNS gives you one answer at a time, or a list you do not control how to pick from.** A single A-record returns one IP; if inventory has 20 pods, a single A-record sends *all* your traffic to one of them. Round-robin DNS returns the list in rotated order, but the client picks the first one, and clients cache aggressively, so you get terrible distribution where some pods are slammed and others idle. Second, **DNS caching defeats the whole point.** The order service's HTTP client, the JVM, the OS resolver, and intervening caches all cache the resolved IP for the record's TTL — and the JVM historically cached DNS *forever* by default. When a pod dies, its IP can live in a dozen caches for the full TTL, which is exactly the 40-second stale window that produced the 503s in the opening story. Third, **DNS has no concept of health.** A DNS record does not know that `10.2.4.31` stopped responding ten seconds ago; it will keep handing it out until something updates the record, and the something is exactly the machinery we are about to build.

The deeper lesson is that "where is service B" is not a static fact you can bake into config or even into a long-lived DNS record. It is a *query against a constantly-updated source of truth* — and that source of truth is the service registry. Everything else in this post is built on top of it.

## The service registry: the live phone book

A **service registry** is a database whose entire job is to answer one question: *which instances of each service are alive and ready right now?* It is a key-value store, indexed by service name, where the value is a list of healthy endpoints — IP, port, and metadata like the zone the instance runs in or the version it is serving. The registry is to a microservices fleet what DNS is to the public internet, except it updates in seconds instead of hours and it understands health.

The thing that makes a registry work is its **lifecycle protocol**: how an instance gets *into* the list and, far more importantly, how it gets *out* of the list when it dies. This is where the design subtlety lives, because instances rarely die politely. A graceful shutdown can deregister itself; a kill -9, an OOM, a node failure, or a network partition cannot. The registry must detect a vanished instance without the instance's cooperation, and it does that with leases and heartbeats.

![A graph showing a pod booting and self-registering, the registry holding a fifteen second lease, the pod heartbeating every five seconds through a readiness health check, branching to either staying in rotation or being reaped when the lease expires](/imgs/blogs/service-discovery-and-load-balancing-8.webp)

The lifecycle works like this. When an inventory pod boots, it **registers** itself with the registry: "I am inventory, I'm at 10.2.4.31:8080, I'm in zone us-east-1a, I'm serving version v3.2." The registry stores this with a **time-to-live lease** — say 15 seconds. The pod must then **renew** the lease by sending a **heartbeat** before it expires, typically every 5 seconds, giving it three chances to renew before the lease lapses. As long as the heartbeats keep coming, the pod stays in the list. The instant the pod dies — gracefully or not — the heartbeats stop, the lease expires within 15 seconds, and the registry **reaps** the entry. No human, no config edit, no redeploy: the dead pod removes itself by failing to renew. This is the self-healing property that makes the whole thing operationally sane at the scale of hundreds of services.

There is a second axis to the lifecycle that is easy to miss but decides who is responsible when registration goes wrong: *who* does the registering. In the **self-registration** pattern, the service instance registers and heartbeats itself — the inventory pod, on boot, calls the registry's API and starts a background heartbeat loop. This is simple and keeps the registry ignorant of how instances are managed, but it couples every service to the registry's client API and means a bug in one service's registration code can corrupt its own discovery. In the **third-party registration** pattern, a separate component — a registrar, a sidecar, or the orchestrator itself — watches instances come and go and registers them on their behalf. Kubernetes uses third-party registration: the control plane (not the pod) populates the EndpointSlice based on the pod's readiness, so application code does nothing. The trade-off is that third-party registration adds a moving part (the registrar must itself be reliable) but removes registration logic from every service. The industry trend, again, is toward third-party registration via the platform, because "the application shouldn't have to know how it's discovered" is the same impulse that drives the service mesh.

There are two broad families of registry, and the difference between them is a genuine architectural fork.

**Strongly-consistent registries (Consul, etcd, ZooKeeper).** These are built on a consensus algorithm — Raft for Consul and etcd, ZAB for ZooKeeper — that guarantees every reader sees the same registry state. The price is that writes require a quorum, so the registry cannot accept registrations during a network partition that splits the quorum. This is the CAP-theorem trade-off made concrete: these systems choose consistency over availability, which is the right call for things like leader election and configuration but has a subtle cost for service discovery that we will return to in the partition stress-test. The [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) deep-dive carries the full treatment of why consensus-backed stores refuse writes during a quorum-splitting partition; here the practitioner point is that a CP registry will refuse to register a fresh, healthy pod if the cluster has lost quorum, which during an incident is the worst possible moment to lose the ability to bring capacity online.

**Availability-first registries (Netflix Eureka).** Eureka deliberately chose the other side of the trade-off. It is an AP system: each Eureka server replicates registrations to its peers on a best-effort basis, and during a partition each server keeps serving whatever it last knew, even if that data is stale. Eureka would rather hand you a slightly-out-of-date list (which your client-side health checks can compensate for) than refuse to answer. It even has a "self-preservation mode" that stops reaping entries when it sees a sudden mass of heartbeat failures, on the theory that a sudden cliff of failures is more likely a network problem on Eureka's side than 90% of your fleet genuinely dying at once. This is a profound design choice that we will unpack in the case studies.

Here is a concrete Consul registration with a health check, which is the most common standalone-registry setup outside Kubernetes:

```json
{
  "service": {
    "name": "inventory",
    "id": "inventory-10-2-4-31",
    "address": "10.2.4.31",
    "port": 8080,
    "meta": { "version": "v3.2", "zone": "us-east-1a" },
    "checks": [
      {
        "http": "http://10.2.4.31:8080/health/ready",
        "interval": "5s",
        "timeout": "2s",
        "deregister_critical_service_after": "30s"
      }
    ]
  }
}
```

That `deregister_critical_service_after: 30s` line is the reaping policy spelled out: if the readiness check fails continuously for 30 seconds, Consul stops advertising this instance and eventually removes it entirely. The `interval: 5s` is the heartbeat cadence. Tuning these two numbers is one of the most consequential decisions in the whole system, and we will do the math on it in a worked example, because the gap between "pod dies" and "registry notices" is precisely the window during which callers route into a void.

There is a real tension hiding in those two numbers, and it is the same tension that runs through every failure detector in distributed systems. Make the lease TTL short and the heartbeat fast, and you detect a dead pod quickly — but you also risk *false positives*: a healthy pod that suffers a 4-second stop-the-world garbage-collection pause, or a transient network blip that drops two heartbeats, gets wrongly reaped, yanked out of rotation, and its traffic dumped onto its neighbors at the worst possible moment. Make the TTL long and the heartbeat slow, and you never false-positive a healthy pod — but a genuinely dead pod lingers in the rotation, bleeding requests, for the full TTL. This is the *failure-detector accuracy-versus-speed trade-off*, and there is no setting that wins both. The practical resolution is the one we keep arriving at: do not rely on the active heartbeat TTL as your primary defense against dead pods at all — set it to a reasonable, conservative value (so you do not false-positive healthy pods) and lean on *passive* outlier detection, which reacts to real request failures in milliseconds without the false-positive risk of an aggressive heartbeat. The heartbeat reaps the pod that vanished silently; the passive detector handles the pod that is failing requests *right now*. Together they cover both, and neither has to be tuned dangerously aggressive.

## Kubernetes: the registry you already have

If you run on Kubernetes, you already have a service registry and you may not have realized it, because Kubernetes built discovery into the platform so thoroughly that it feels like ambient magic. There is no separate registry to install for basic discovery; the machinery is the control plane itself.

Here is how the pieces fit. When you define a **Service**, you give it a stable name and a stable virtual IP (the ClusterIP) that never changes for the life of the Service. Behind that Service, a **selector** matches a set of pods by label. As pods come and go, the Kubernetes control plane continuously updates an **Endpoints** object (and, at scale, **EndpointSlices**) that lists the live pod IPs backing that Service. The kubelet runs each pod's **readiness probe**, and — this is the crucial part — a pod is only added to the Endpoints list when its readiness probe passes, and is removed the moment the probe starts failing. That readiness gate *is* the health-aware reaping we described, built into the platform.

```yaml
# A Kubernetes Service plus the readiness probe that gates membership in it.
apiVersion: v1
kind: Service
metadata:
  name: inventory
spec:
  selector:
    app: inventory          # matches pods carrying this label
  ports:
    - port: 8080
      targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inventory
spec:
  replicas: 4               # scales 4 -> 40 during the flash sale
  template:
    metadata:
      labels:
        app: inventory
    spec:
      containers:
        - name: inventory
          image: shopfast/inventory:v3.2
          ports: [{ containerPort: 8080 }]
          readinessProbe:           # <- gates Endpoints membership
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 3
            failureThreshold: 2     # 2 failures -> removed from Service
```

The order service does not query a registry API at all in this model. It just calls `http://inventory:8080`, and Kubernetes' in-cluster DNS resolves `inventory` to the Service's stable ClusterIP. From there, the actual load balancing across pods happens at the node level in `kube-proxy` (or, increasingly, in eBPF dataplanes like Cilium), which rewrites the destination from the ClusterIP to one of the live pod IPs from the EndpointSlice. So the readiness probe, the Endpoints/EndpointSlice list, and the kube-proxy rewrite together form a complete server-side discovery system that the caller never sees.

It is worth pausing on why this matters for the ShopFast flash sale specifically. When the autoscaler takes inventory from 4 pods to 40, each of those 36 new pods boots, starts its container, and the kubelet begins running its readiness probe. With `initialDelaySeconds: 5` and `periodSeconds: 3`, a new pod is not even probed for the first 5 seconds and needs a passing probe after that before it joins the Service's EndpointSlice. Only then does kube-proxy start sending it traffic. This staged admission is exactly what you want: the order service's calls keep flowing to the 4 already-warm pods, and the new pods are added to the rotation one by one *as each becomes ready*, not all at once in a half-initialized state. The flip side — and a real production gotcha — is termination: when a pod is deleted, Kubernetes sends it `SIGTERM` and *simultaneously* begins removing it from EndpointSlices, but those two events race. For a few hundred milliseconds, a terminating pod can still be in some node's routing table while it has already started shutting down, which is why production pods use a `preStop` hook with a small sleep (commonly `sleep 5`) to keep serving in-flight requests until the deregistration has propagated. Skip that, and a routine deploy drops a sliver of requests every time a pod rolls — the same dead-pod-routing problem from the opening story, self-inflicted on every deploy.

The reason **EndpointSlices** exist — and why they matter for the "scale 4 to 40" story and especially for fleets with thousands of pods — is a scale problem in the original Endpoints design. The old Endpoints object was a single object listing *every* pod IP for a Service. When a Service had thousands of endpoints, every single pod or endpoint change rewrote that entire giant object and pushed the whole thing to every node watching it. A Service with 5,000 endpoints across 5,000 nodes, churning during a deploy, could generate gigabytes of control-plane traffic and melt the API server. EndpointSlices shard that list into chunks of (by default) 100 endpoints each, so a single pod change only rewrites and re-pushes one small slice, not the whole list. This is a quiet but enormous scalability fix, and it is exactly the kind of thing that separates "works for my 4-pod demo" from "works for a fleet," which is the journey this whole series is about.

## Client-side versus server-side discovery: the great fork

Once you have a registry, there are two fundamentally different places you can put the logic that turns "the list of live inventory pods" into "the specific pod this request goes to." This is the single most important architectural decision in service discovery, and the two answers have different costs, different failure modes, and different operational owners.

![A before and after comparison contrasting client-side discovery where the caller queries the registry and picks an instance against server-side discovery where the caller hits one virtual IP and a proxy picks the instance](/imgs/blogs/service-discovery-and-load-balancing-2.webp)

**Client-side discovery.** The caller is smart. The order service embeds a discovery client — a library — that watches the registry, keeps a local cache of inventory's live endpoints, and runs the load-balancing algorithm itself to pick a pod for each request. Then it connects directly to that pod. There is no intermediary. This is the Netflix model: Eureka is the registry, and Ribbon (now largely succeeded by Spring Cloud LoadBalancer) is the client-side library that picks. gRPC's built-in load balancing is also client-side: a gRPC channel can be configured to resolve a service name to a list of backends and balance across them internally, opening one subchannel per backend and picking per-call.

The advantage is that there is **no extra network hop** — the caller talks straight to the chosen pod — and the caller has **full, per-call control** of the algorithm, can do clever things like latency-aware picking with first-hand latency data, and avoids a centralized proxy that could become a bottleneck or a single point of failure. The cost is that the discovery and balancing logic now lives in *every caller*, in *every language*. If your fleet is polyglot — Go, Java, Python, Node — you need a correct, well-maintained discovery client for each, and they must all agree on the registry protocol, the health semantics, and the algorithm. That is a real maintenance and consistency burden, and it is the single biggest reason the industry has drifted toward the next option.

**Server-side discovery.** The caller is dumb. The order service just sends its request to a single stable address — a load balancer, a Kubernetes Service ClusterIP, or a mesh sidecar — and *that* component queries the registry and picks the pod. The caller never knows there are 40 pods; it knows one address. Kubernetes' Service model is server-side discovery. A cloud load balancer in front of a service is server-side discovery. A service mesh sidecar is server-side discovery.

The advantage is that the discovery logic lives in *one place*, maintained by the platform team, and works for **any client in any language** because the client just makes an ordinary network call. A Python service and a Rust service get identical, correct, health-aware balancing for free. The cost is the **extra network hop** through the proxy, which adds a little latency (typically sub-millisecond for in-cluster proxies, but non-zero), and the proxy is a component that must itself be made highly available so it does not become the single point of failure for everything behind it.

Here is the gRPC client-side configuration, because it is the cleanest illustration of the client-side model in code:

```go
// gRPC client-side load balancing: the channel resolves the service name to a
// list of backends and balances across them itself - no proxy in the path.
conn, err := grpc.NewClient(
    "dns:///inventory.default.svc.cluster.local:8080",
    grpc.WithDefaultServiceConfig(`{
        "loadBalancingConfig": [{"round_robin":{}}],
        "healthCheckConfig": {"serviceName": "inventory"}
    }`),
    grpc.WithTransportCredentials(insecure.NewCredentials()),
)
// The channel keeps one subchannel per resolved backend, runs the gRPC health
// protocol against each, and round-robins live subchannels per RPC.
```

Note the subtle but important detail: `dns:///` with the *headless* Service form. A normal Kubernetes Service hands you one ClusterIP (server-side balancing in kube-proxy), which defeats gRPC's client-side balancer because the channel sees a single address and opens a single long-lived HTTP/2 connection to it — and since gRPC multiplexes many requests over one connection, server-side L4 balancing pins all of them to one backend. To get gRPC client-side balancing to actually spread load, you point it at a **headless Service** (`clusterIP: None`), whose DNS returns *all* the pod IPs, so the gRPC channel sees the full list and can balance across them itself. This exact gotcha — "my gRPC traffic is all hitting one pod" — is one of the most common real-world load-balancing bugs in Kubernetes, and it stems directly from the mismatch between L4 connection-level balancing and L7 request-level balancing over a multiplexed protocol.

The matrix below makes the fork explicit, so you can decide it on purpose rather than by accident.

![A decision matrix comparing client-side and server-side discovery across extra network hop, polyglot client support, caller complexity, registry coupling, and per-call control](/imgs/blogs/service-discovery-and-load-balancing-4.webp)

In practice, modern fleets often land on a hybrid: **server-side at the platform layer** (Kubernetes Services for the baseline, so every language works), with **client-side balancing for the specific protocols that need it** (gRPC with a headless Service, or a service mesh sidecar that gives you L7 client-side-quality balancing without putting the logic in the application). The service mesh, which we will get to, is essentially "server-side discovery from the application's point of view, client-side-quality balancing from the network's point of view" — it resolves the false choice by moving the smart logic into a sidecar that every language gets for free.

## The request path: what actually happens on a call

Before we get to algorithms, it helps to see the full path a single balanced request takes, because each layer is a place where things can go right or wrong. Whether the logic lives in a client library or a proxy, the same four stages run.

![A vertical stack showing a request resolving the service name to an endpoint list, filtering to ready pods only, applying the least-request algorithm, drawing a warm pooled connection, and reaching a chosen pod under fifty milliseconds](/imgs/blogs/service-discovery-and-load-balancing-3.webp)

First, **resolution**: turn the service name into a list of candidate endpoints, pulled from the registry's current view. Second, **health filtering**: drop any endpoint that is not currently ready — failing its health check, recently ejected by outlier detection, or draining. This is the step that means the difference between routing around a dead pod and routing into it. Third, **algorithm selection**: among the healthy candidates, pick one according to the load-balancing policy. Fourth, **connection**: hand the request to a connection from a pool — ideally a warm, already-established keep-alive connection so you do not pay TCP and TLS handshake cost on every request. Then, and only then, does the request touch the pod.

That third stage — picking one — is where the load-balancing algorithm lives, and the choice of algorithm has consequences out of all proportion to how simple it sounds.

## Load-balancing algorithms: round-robin is not the answer you think it is

The naive intuition is that load balancing means "spread requests evenly," and round-robin does exactly that: pod 1, pod 2, pod 3, pod 1, pod 2, pod 3, perfectly even. The problem is that *spreading requests evenly is not the same as spreading load evenly*, and the gap between those two is where tail latency comes from. Let us walk the algorithms from simplest to smartest.

**Round-robin** hands each successive request to the next pod in rotation. It is trivial to implement, stateless, and perfectly fair *if every request costs the same and every pod is equally fast*. Both of those assumptions are usually false. Requests vary — one inventory lookup hits cache, the next triggers a slow database scan. Pods vary — during the flash sale the autoscaler adds new pods on different node types, some pods share a noisy node with a memory-hungry neighbor, some are cold and still JIT-compiling. Round-robin sends the same number of requests to the slow pod as to the fast one, so the slow pod's queue grows, its latency climbs, and because it is still getting its "fair share" of new requests, it never recovers. Round-robin is blind to the one thing that matters: which pods are actually keeping up.

**Least-connections (L4) / least-request (L7)** picks the pod with the fewest in-flight requests right now. This is a massively better default, because in-flight count is a real-time proxy for "how busy is this pod." A slow pod accumulates in-flight requests (they are taking longer to drain), so its count rises, so least-request *stops sending it new work* until it catches up. It self-corrects exactly where round-robin self-destructs. The cost is that the balancer must track in-flight counts per backend, which is trivial for a single balancer but requires care in client-side balancing where each client only sees its own in-flight counts, not the global picture. Envoy and most modern proxies actually use a clever approximation called **power-of-two-choices** (P2C): instead of scanning all N backends for the absolute minimum (O(N) per request, expensive at scale), pick two backends at random and send to the less-loaded of the two. P2C gets you nearly all the benefit of least-request at O(1) cost, and it has a beautiful theoretical property: random-two-choices reduces the maximum load exponentially compared to pure random, which is why it is the default in Envoy and many others.

**Latency-aware (EWMA / least-time).** Least-request uses in-flight count as a proxy for slowness; latency-aware algorithms measure slowness directly, tracking an exponentially-weighted moving average (EWMA) of each backend's recent response latency and biasing traffic toward the faster ones. This is the most sophisticated common option and it shines when backends are genuinely heterogeneous — different instance types, different zones with different network latency, a backend with a degraded disk. The cost is complexity and the need for a good decay constant: weight recent latency too heavily and you get oscillation (a pod gets slow, traffic flees, it gets fast, traffic floods back, it gets slow), weight it too lightly and you react too slowly to a degrading pod. Finagle (Twitter's RPC library) popularized latency-aware balancing at scale, and Envoy supports it via its `LEAST_REQUEST` with active-request biasing and through its adaptive options.

**Consistent hashing** is a different beast entirely — it does not try to balance load evenly, it tries to send *the same key to the same pod every time*. You hash some request attribute (a user ID, a cart ID, a cache key) and map it onto a ring of pods, so request for user 12345 always lands on the same pod as long as that pod is alive. Why would you want that? **Cache affinity.** If each inventory pod caches the SKUs it has recently served, then sending all requests for a given SKU to the same pod means that pod's cache stays hot for that SKU and the hit rate soars; spreading the same SKU across all 40 pods means each pod sees only 1/40th of the requests for it and the cache is cold everywhere. Consistent hashing is also how you get *sticky sessions* when a backend holds per-user state. The "consistent" part is the clever bit: when a pod is added or removed, only the keys that mapped to *that* pod get remapped, not the entire keyspace — which is exactly the property that makes it survivable in a fleet where pods come and go. The full mechanism, including virtual nodes and the math of why only K/N keys move, is the subject of the [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) deep-dive, and everything there about distributing data across nodes applies identically to distributing *requests* across service instances.

The trade-offs across these four are not subtle, and they pull in different directions, so here they are side by side.

![A decision matrix comparing round-robin, least-request, latency-aware, and consistent-hash load balancing across hot-instance avoidance, cache affinity, simplicity, and cost on rebalance](/imgs/blogs/service-discovery-and-load-balancing-5.webp)

The senior summary: **default to least-request (or P2C least-request), because it self-corrects against heterogeneous pods at almost no cost.** Reach for **latency-aware** when your fleet is genuinely heterogeneous and you have measured that least-request is not enough. Reach for **consistent hashing** only when you specifically want cache affinity or session stickiness, and accept that you are trading even load distribution for hit-rate. Use **plain round-robin** only when you have proven your pods and requests are homogeneous — which, in a real autoscaling fleet, they almost never are.

#### Worked example: least-request versus round-robin under heterogeneous pods

Let us put numbers on why round-robin loses. ShopFast's inventory service has 10 pods. Nine of them are healthy and serve a request in 20ms. One of them — call it the straggler — is on a noisy node and serves requests in 200ms (10x slower), but it is not *failing*, so health checks pass and it stays in rotation. Total incoming load is 1,000 requests per second, evenly arriving.

Under **round-robin**, each pod gets exactly 100 requests per second. The nine healthy pods handle 100 RPS at 20ms with ease — at 20ms each, a single pod can serve 50 requests per second per concurrent slot, so 100 RPS needs only a couple of concurrent slots, no queueing, latency stays ~20ms. The straggler, though, also gets 100 RPS, but at 200ms per request it can only serve 5 requests per second per concurrent slot. To handle 100 RPS at 200ms it needs 20 requests in flight *constantly*, and if its concurrency is capped lower than that, requests queue. Queueing makes the effective latency balloon: requests wait behind 200ms predecessors. The straggler's served requests routinely hit 400ms, 600ms, worse. Since the straggler serves 10% of all traffic, **roughly 10% of your requests are 10x-plus slower than they should be** — and that 10% is exactly your p90-and-worse tail. Your p99 is now dominated entirely by the straggler. Round-robin took one slow pod and let it poison a tenth of your traffic.

Under **least-request**, the picture inverts. The straggler's requests take 200ms, so its in-flight count climbs — it accumulates 200ms-long requests faster than it drains them. The balancer sees the high in-flight count and *stops routing new requests to it* until it catches up. In steady state, least-request settles each pod to roughly equal *in-flight* counts, which means the straggler ends up receiving far fewer requests per second (proportional to its speed) while the fast pods soak up the difference. The straggler might end up serving ~15 RPS instead of 100, and the nine fast pods split the remaining ~985 RPS, about 109 each — still trivial for a 20ms pod. The result: the straggler is no longer a bottleneck, the slow tail it created largely disappears, and your **p99 drops from "dominated by the 200ms straggler" to roughly the fast pods' own tail, perhaps 30–40ms.** Same hardware, same straggler, one algorithm change, and the user-visible tail is cut by an order of magnitude. This is why "just use round-robin" is junior advice and "least-request by default" is the senior reflex.

#### Worked example: consistent-hash cache hit-rate versus round-robin

Now the cache-affinity case. ShopFast's inventory pods each keep an in-memory LRU cache of SKU availability, holding the 10,000 hottest SKUs. The catalog has 1,000,000 SKUs, but traffic is Zipfian — the top 10,000 SKUs account for, say, 80% of all lookups. There are 20 pods.

Under **round-robin**, every SKU's requests are spread evenly across all 20 pods. So each pod sees only 1/20th of the requests for any given hot SKU. From a single pod's perspective, the "hot 10,000 SKUs" it sees are diluted: it caches what it sees, but it sees a thin, random slice of the global traffic, so its 10,000-entry cache is constantly churning across the full hot set times the dilution. The effective cache hit rate is mediocre — say each pod manages a 60% hit rate, because its local cache cannot hold a coherent hot set when it only sees a fraction of each SKU's requests. That means 40% of lookups miss cache and hit the database. At 1,000 RPS that is 400 database queries per second of avoidable load.

Under **consistent hashing** keyed on SKU, every request for a given SKU goes to the *same* pod. Now each pod is the sole owner of roughly 1/20th of the SKU keyspace — about 50,000 SKUs — and within its slice, the hot SKUs concentrate. Each pod's 10,000-entry cache now holds the hottest SKUs *of its slice*, and because every request for those SKUs comes to it, the cache stays hot. The hit rate jumps to perhaps 95%. Now only 5% of lookups miss, which at 1,000 RPS is 50 database queries per second — an **8x reduction in database load** purely from routing the same key to the same pod. The cost you accept: when a pod dies or is added, its slice of SKUs remaps to other pods, which see a cold cache for those keys for a few seconds (a brief miss spike), and load is slightly less even than round-robin because hot SKUs concentrate on their owners. For a read-heavy cache-fronted service, that trade is overwhelmingly worth it — which is exactly the same logic that makes consistent hashing the backbone of [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).

## Health checks: only route to pods that can actually serve

Everything above assumes the balancer knows which pods are healthy. That assumption is doing enormous work, and getting it right is the difference between a balancer that routes around failure and one that routes into it. There are two complementary kinds of health checking, and you want both.

**Active health checks (probes).** The balancer (or the registry) periodically *asks* each pod "are you ready?" — an HTTP GET to `/health/ready`, a gRPC health-check RPC, a TCP connect. If the probe fails some number of times, the pod is marked unhealthy and dropped from the rotation. Kubernetes readiness probes are active checks. The strength of active checks is they catch a pod that is broken but not receiving traffic (a freshly-started pod whose dependencies are not connected yet); the weakness is they cost a steady drumbeat of probe traffic and they have *lag* — the probe interval plus the failure threshold is how long a dead pod stays in rotation before active checks notice.

**Passive health checks (outlier detection / ejection).** Instead of asking, the balancer *watches the real traffic* and ejects a pod that starts misbehaving — returning 5xx errors, timing out, resetting connections. This is the killer feature, because it reacts to the *actual* requests your users are making, with zero probe lag: the moment a pod returns its fifth 5xx in a row, it is ejected. Envoy calls this **outlier detection**, and it is the single most important production-grade load-balancing feature most teams under-use. Here is the difference it makes, drawn as the routing-into-a-dead-pod scenario versus the health-aware one:

![A before and after comparison showing a stale registry that keeps routing to a dead pod and fails twelve percent of calls versus a health-aware balancer that trips ejection after five errors, ejects the pod for thirty seconds, and reroutes to live pods](/imgs/blogs/service-discovery-and-load-balancing-6.webp)

It is worth being precise about the distinction between **readiness** and **liveness**, because conflating them causes real outages. A **readiness** check asks "should this pod receive traffic right now?" — failing it removes the pod from the load balancer but does *not* kill it. A **liveness** check asks "is this pod broken beyond recovery and in need of a restart?" — failing it kills and restarts the pod. The classic catastrophe is putting a dependency check (e.g. "can I reach the database?") in the *liveness* probe: when the database has a blip, every pod fails liveness simultaneously, Kubernetes kills and restarts the entire fleet at once, the restarts hammer the recovering database, and a 30-second database blip becomes a 20-minute full outage. Dependency health belongs in *readiness* (drop me from rotation while the dependency is down, but do not kill me), and only truly-unrecoverable conditions belong in liveness. This distinction, and the broader self-healing model, is the subject of the upcoming [health checks: readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) post — for load balancing, the one rule to carry is: *the balancer must consult readiness, and readiness must be cheap, local, and dependency-aware without being dependency-fatal.*

Here is an Envoy upstream cluster configured with both least-request balancing and passive outlier detection — the production-grade configuration the opening incident was missing:

```yaml
# Envoy cluster: least-request balancing plus outlier detection (passive health).
# A pod that returns 5 consecutive 5xx is ejected for 30s and rerouted around.
clusters:
  - name: inventory
    connect_timeout: 1s
    lb_policy: LEAST_REQUEST           # P2C least-request by default
    least_request_lb_config:
      choice_count: 2                  # power-of-two-choices
    type: STRICT_DNS                   # resolve headless Service to all pod IPs
    load_assignment:
      cluster_name: inventory
      endpoints:
        - lb_endpoints:
            - endpoint:
                address:
                  socket_address: { address: inventory, port_value: 8080 }
    outlier_detection:
      consecutive_5xx: 5               # 5 server errors -> eject
      interval: 10s                    # evaluation window
      base_ejection_time: 30s          # ejected for 30s, then probed back
      max_ejection_percent: 50         # never eject more than half the fleet
    circuit_breakers:
      thresholds:
        - max_connections: 1024
          max_pending_requests: 256    # bound the queue, shed beyond it
          max_requests: 1024
```

And here is the NGINX equivalent for teams not on Envoy, showing least-connections plus passive health via `max_fails`:

```nginx
# NGINX upstream: least-connections plus passive health checks.
# A backend that fails 3 times in 10s is taken out of rotation for 20s.
upstream inventory {
    least_conn;
    server 10.2.4.31:8080 max_fails=3 fail_timeout=20s;
    server 10.2.4.32:8080 max_fails=3 fail_timeout=20s;
    server 10.2.4.33:8080 max_fails=3 fail_timeout=20s;
    keepalive 64;                       # warm pooled connections
}
server {
    location /inventory/ {
        proxy_pass http://inventory;
        proxy_next_upstream error timeout http_502 http_503;  # retry next pod
        proxy_connect_timeout 1s;
        proxy_read_timeout 2s;
    }
}
```

The `max_ejection_percent: 50` and the bounded `max_pending_requests` are not decoration — they are the guardrails that keep health-aware balancing from turning into a self-inflicted outage, which is the failure mode we turn to next.

## The failure modes, and the math behind each

Now we stress-test. A senior does not trust a design until they have walked it through its failure modes with numbers, so let us do exactly that for the three that bite service discovery hardest.

### Stale registry entries: routing to dead pods

This is the opening incident. A pod dies, but the registry has not yet noticed, so the balancer keeps routing a share of traffic to a dead address. The size of the damage is entirely determined by the **detection window** — the gap between "pod dies" and "registry/balancer stops routing to it" — and the share of pods that are dead.

#### Worked example: a 30-second-stale registry and the error math

ShopFast's inventory has 10 pods serving 1,000 requests per second total, so each pod gets 100 RPS under even distribution. One pod gets OOM-killed at T+0. The registry's detection window is determined by its config: a heartbeat-based registry with a 15-second TTL and a 5-second heartbeat reaps the dead pod after the lease lapses — worst case the pod died right after its last heartbeat, so detection takes up to TTL = 15 seconds. But the *caller's* cache of the endpoint list adds more: if the order service refreshes its registry snapshot every 15 seconds, in the worst case the caller routes to the dead pod for the registry's 15s reaping *plus* the caller's 15s cache TTL = up to 30 seconds of stale routing.

During those 30 seconds, the dead pod still appears to be 1 of 10 pods, so it receives 1/10th of traffic = 100 RPS. Every one of those requests fails (connection refused or timeout). That is 100 failed requests per second for 30 seconds = **3,000 failed requests**, which against a total of 30,000 requests in that window is a **10% error rate** for half a minute. If the failures are timeouts rather than fast connection-refused, it is worse: each timing-out request also holds a caller thread or connection for the timeout duration, which can starve the caller's connection pool and cascade. This is the precise arithmetic behind the opening story, and it tells you exactly which knobs reduce the damage: shorten the registry TTL (faster reaping), shorten the caller's cache TTL (faster propagation), and — most powerfully — **add passive outlier detection so the caller ejects the dead pod after a handful of errors regardless of what the registry says.** With outlier detection set to eject after 5 consecutive errors, the dead pod is ejected after ~5 failed requests — a fraction of a second, not 30 seconds — cutting the 3,000 failures down to a handful. Passive health is what collapses the stale-registry window from 30 seconds to milliseconds, which is why it is non-negotiable in production.

### A pod dies mid-request: the retry dance

The stale-registry math covers *new* requests. But what about the request that was *in flight* on the pod at the instant it died? Picture the flash-sale timeline: pods scaling 4 to 40, and at T+4 minutes pod 17 is killed mid-request by a node failure.

![A timeline showing a flash sale starting with four inventory pods, the autoscaler adding pods, forty pods registered, pod seventeen dying mid-request, five errors triggering ejection, and a retry landing on a live pod](/imgs/blogs/service-discovery-and-load-balancing-7.webp)

The request that was on pod 17 gets a connection reset or a timeout. The caller must decide: **retry, or fail?** This is where service discovery and resilience meet, and the rules are precise. You may *safely* retry only if the request is **idempotent** — a read (`GET /inventory/sku/12345`) is trivially safe to retry; a non-idempotent write (`POST /reserve-stock`) is *not* safe to blindly retry, because the original might have succeeded before the connection dropped, and a retry would double-reserve. The right pattern: make writes idempotent with an idempotency key so retries are safe (the [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) post covers this in depth), then configure the balancer to retry the failed request *on a different pod* — Envoy's `retry_on: reset,connect-failure` with `host_selection_retry_max_attempts` ensures the retry does not land on the same dead pod. With that in place, pod 17 dying mid-request becomes invisible to the user: the request fails on pod 17, the balancer's outlier detection ejects pod 17 after a few such failures, the in-flight request is retried on a live pod, and the customer's order goes through with maybe 30ms of extra latency. Without it, that customer gets a 503 and an abandoned cart during your biggest sale of the year. The difference is entirely in the configuration.

There is a subtle trap here: **retries amplify load on an already-struggling fleet.** If inventory is slow because it is *overloaded*, every timeout-and-retry adds another request to the overloaded fleet, and you get a retry storm that turns a brownout into an outage. The defenses are a **retry budget** (cap retries at, say, 10% of total requests, so retries can never more than slightly amplify load) and **backoff with jitter**. Retries belong to the resilience toolkit — the full treatment is in [resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — but the load-balancing-specific rule is: *retry on a different host, with a budget, only for idempotent operations.*

### The cold-start problem: routing to a just-born pod

The flash-sale scale-out has a failure mode that is the mirror image of routing to a dead pod: routing to a pod that is *too young*. When the autoscaler adds 36 new inventory pods to go from 4 to 40, those pods are not instantly ready. A freshly-started JVM is JIT-compiling; its connection pools to the database are empty; its caches are cold. If the readiness probe passes the moment the HTTP server binds, the balancer will start slamming a stone-cold pod with its full 1/40th share of traffic, and that pod will be 5–10x slower than a warm one for the first 30–60 seconds — exactly the heterogeneous-pod problem from the worked example, except self-inflicted by adding capacity.

There are two production fixes. First, **slow start / warm-up ramping**: Envoy's `slow_start_window` and many balancers' equivalent ramp a newly-added pod's traffic share from near-zero up to full over a configured window (say 60 seconds), so a cold pod gets a trickle while it warms and only reaches full load once it is warm. Second, **a readiness probe that means it**: do not pass readiness until the pod has actually warmed — connection pools primed, a cache pre-fill done, the JIT given a few warm-up requests. The combination means the flash-sale scale-out adds capacity that *helps* instead of capacity that briefly hurts. This is the kind of second-order detail that separates a fleet that scales smoothly from one where every scale-up event causes a latency blip.

### Registry partition: the source of truth splits

The nastiest failure is when the registry itself has trouble. Consider a network partition that splits a Consul (CP) cluster so it loses quorum. Because Consul chooses consistency, it now *refuses writes* — no new pod can register, no dead pod can be deregistered. During a flash sale, that means the autoscaler's new pods cannot join the rotation (you cannot bring capacity online at the exact moment you need it), and dead pods cannot be reaped. Reads may still work from the existing data, so the system limps on the stale view. This is the CP cost made concrete: the registry trades availability-of-writes for consistency, and a partition during a scaling event is the worst time to pay that bill.

An AP registry like Eureka makes the opposite trade: during a partition, each Eureka server keeps serving its last-known list and keeps accepting registrations, even if servers temporarily disagree. New pods *can* join (in whichever partition they reach), and the system stays available — at the cost that two callers might briefly see different views of the fleet. Eureka's self-preservation mode goes further: if it suddenly sees most heartbeats fail (which usually means a network problem on Eureka's side, not a mass pod death), it *stops reaping* and keeps the full list, betting that serving slightly-stale entries (which client-side health checks will route around) is better than wrongly reaping a healthy fleet. This is a genuinely senior design insight: **for service discovery specifically, availability of the registry usually matters more than its consistency, because callers can compensate for stale entries with their own health checks, but they cannot compensate for a registry that refuses to tell them anything.** The stale entry costs you a few ejected requests; the unavailable registry costs you the ability to discover anything at all. This is why Netflix, running at enormous scale where partitions are routine, chose AP — and why Kubernetes' own model, where the kubelet readiness gating happens locally per-node and EndpointSlices propagate eventually, leans availability-first in practice too.

## Optimization: making discovery and balancing production-grade

The baseline works. Now make it fast, cheap, and resilient — with numbers, because an optimization you cannot measure is a guess.

**Connection pooling and keep-alive.** The single biggest per-request waste in a naive setup is establishing a fresh TCP connection (and TLS handshake, if you are doing mTLS) on every call. A TLS handshake is one or two extra round trips — at a 1ms in-datacenter RTT that is 1–2ms, but at cross-zone 5ms RTT it is 5–10ms *per request* of pure handshake overhead. Pooling warm keep-alive connections amortizes that to near-zero: the first request pays the handshake, the next thousand reuse the connection. For a service doing 1,000 RPS, moving from connection-per-request to a pool of 64 warm connections can drop p50 by several milliseconds and slash CPU spent on TLS. Set `keepalive` in your upstream (we did in the NGINX config) and size the pool to your concurrency. The win is measurable directly: watch the connection-establishment rate drop from ~1,000/s to near zero and p50 latency fall by the handshake cost.

**Locality / zone-aware routing.** This is the highest-dollar-impact optimization and most teams leave it on the table. In a cloud deployment spanning three availability zones, a request that crosses zones pays both *extra latency* (cross-AZ RTT is often 1–2ms versus ~0.1ms same-AZ) and *real money* (cross-AZ data transfer is billed, commonly around \$0.01 per GB *each way*). If your order-service pods balance blindly across all inventory pods regardless of zone, roughly two-thirds of traffic crosses a zone boundary (with three zones, only 1/3 stays local). Zone-aware / topology-aware routing tells the balancer to *prefer same-zone backends* and only spill to other zones when local capacity is insufficient or unhealthy. Kubernetes supports this via `Topology Aware Routing` (the `service.kubernetes.io/topology-mode: Auto` annotation) and Istio via locality load balancing.

#### Worked example: the dollars and milliseconds of zone-aware routing

ShopFast's order and inventory services each run 30 pods across three AZs, 10 per zone. Order-to-inventory traffic is 5,000 RPS, each request carrying ~4KB of response data, so ~20 MB/s of inter-service traffic, ~52 TB/month. With *blind* balancing, ~2/3 of that crosses a zone boundary: ~35 TB/month of cross-AZ transfer. At ~\$0.01/GB each way (~\$0.02 round trip in many clouds), that is roughly \$700/month *just in cross-AZ egress for this one service pair* — and a real fleet has dozens of such pairs, so the bill scales into real money. Latency-wise, the cross-zone hops add ~1–2ms to two-thirds of requests, fattening the tail. With **zone-aware routing**, ~90%+ of traffic stays in-zone (it only crosses when a zone is short on healthy capacity), cutting cross-AZ transfer by roughly 5–10x — call it \$700/month down to ~\$100/month for this pair — and removing the cross-zone latency from the common path so p99 tightens. The catch you must respect: zone-aware routing can *unbalance* load if zones have uneven capacity (if zone A has 10 inventory pods but receives traffic from 20 order pods, those 10 get overloaded), so it must be paired with a spill-over rule and enough per-zone capacity. Measured win: cross-AZ transfer bytes (down ~80–90%), cross-AZ \$ (down proportionally), and same-zone request fraction (up from ~33% to ~90%) on a dashboard.

**Outlier detection tuning.** We covered why outlier detection matters; tuning it is where the production craft lives. Too aggressive (eject after 1 error) and a transient blip ejects half your fleet, concentrating load on the survivors and potentially cascading. Too lax (eject after 50 errors) and a dead pod bleeds requests for too long. The `max_ejection_percent` cap is your safety net — never eject so many pods that the survivors get overwhelmed. A sane starting point: eject after 5 consecutive 5xx, base ejection 30s with exponential backoff on repeated ejection, max ejection 50%. Then measure: the metric to watch is "requests routed to a backend that then errored" — it should be a tiny fraction, and it tells you whether your ejection is fast enough.

**Right-sizing the detection window.** The stale-registry math told us the detection window drives the error count during a pod death. The optimization is to shrink it: a shorter registry TTL and faster heartbeats reap faster (at the cost of more heartbeat traffic and more sensitivity to transient network blips — set the TTL too short and a brief GC pause makes a healthy pod miss a heartbeat and get wrongly reaped). The sweet spot balances "reap dead pods fast" against "do not reap healthy pods that hiccup," and the safest lever is to lean on *passive* health (which reacts to real errors in milliseconds) rather than pushing active TTLs dangerously short.

## How a service mesh moves all of this into the sidecar

By now you have noticed a pattern: client-side discovery is powerful but requires a correct, well-maintained library in every language, and getting all of round-robin-versus-least-request, outlier detection, retries-on-a-different-host, connection pooling, zone-aware routing, and mTLS right *in every service's code, in every language* is a genuinely hard, never-finished job. A **service mesh** is the industry's answer to that burden: it takes all of this logic out of your application and puts it into a **sidecar proxy** — typically Envoy — that runs alongside every pod and intercepts all its network traffic.

![A vertical stack showing application code making a plain localhost call, a per-pod sidecar proxy, a control plane pushing endpoints, mesh load balancing with outlier ejection, and the destination pod reached over a mutual TLS hop](/imgs/blogs/service-discovery-and-load-balancing-9.webp)

The application code becomes blissfully dumb: it makes a plain call to `http://inventory:8080`, the sidecar transparently intercepts it, and *the sidecar* does the discovery (it watches the registry — in Istio's case via the control plane pushing endpoint updates over the xDS protocol), the health-aware least-request balancing, the outlier ejection, the retry-on-a-different-host, the connection pooling, and the mTLS encryption. Every language gets identical, correct, production-grade balancing for free, configured declaratively by the platform team, with zero application code. This is server-side discovery from the application's point of view (it just makes a local call) but client-side-quality L7 balancing from the network's point of view (the sidecar is *on the same host* as the app, so there is no centralized-proxy single point of failure, and it has per-request L7 visibility).

The cost is real and worth naming: a sidecar per pod is extra CPU, memory, and a small per-hop latency (typically sub-millisecond, but it is two extra proxy traversals per call); the control plane is a new critical dependency you must operate; and the whole thing adds significant operational complexity that a small fleet does not need. The honest senior position — covered fully in the upcoming [service mesh: Istio, Linkerd, when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) post — is that a mesh earns its keep once you have enough services, in enough languages, that maintaining discovery and resilience logic per-service has become a tax larger than the mesh's overhead. Below that threshold, Kubernetes' built-in Services plus per-language gRPC balancing is plenty. The mesh is not a starting point; it is a scaling decision.

Here is the Istio side of the same outlier-detection and locality config, to show how declarative it becomes — the application that this governs has *no* discovery or balancing code at all:

```yaml
# Istio DestinationRule: the same least-request + outlier detection + locality
# routing, declared once by the platform team, applied to every caller of
# inventory in any language - zero application code.
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: inventory
spec:
  host: inventory.default.svc.cluster.local
  trafficPolicy:
    loadBalancer:
      simple: LEAST_REQUEST
      localityLbSetting:
        enabled: true            # prefer same-zone backends, cut cross-AZ cost
    connectionPool:
      http:
        http2MaxRequests: 1024
        maxRequestsPerConnection: 100
      tcp:
        maxConnections: 256      # warm pooled connections
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

This is the same Istio shape used in the [inter-service communication](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) post's resilience example, because discovery, balancing, and resilience all live in the same data-plane proxy and are configured together — which is exactly the point of a mesh.

## Case studies: how real fleets solved discovery and balancing

These are real, named, and instructive. Where I give a number, treat it as order-of-magnitude — the lesson is in the shape of the design, not the exact figure.

**Netflix: Eureka and Ribbon, the AP client-side pioneer.** Netflix runs one of the largest microservices fleets in the world, across tens of thousands of instances on AWS, and they built the canonical client-side-discovery stack. **Eureka** is their service registry, and crucially it is an *AP* system — Netflix decided early that in a fleet of that size, network partitions and instance churn are not exceptional events but the constant background weather, and a registry that refuses to answer during a partition (a CP system losing quorum) was unacceptable. So Eureka prioritizes availability: it serves stale data during partitions and uses self-preservation mode to avoid mass-reaping during network hiccups. **Ribbon** was the client-side load balancer that paired with it, embedding the endpoint-watching and balancing logic directly in every (JVM) service. The lesson Netflix teaches is the AP-for-discovery insight: *availability of the registry beats consistency, because callers can route around stale entries with their own health checks but cannot route around a registry that is down.* The modern Netflix and broader Java ecosystem has largely moved past Ribbon (it is in maintenance mode, succeeded by Spring Cloud LoadBalancer and increasingly by meshes), but the architectural lesson is permanent.

**Lyft and Envoy: outlier detection born from operational pain.** Envoy was created at Lyft to solve exactly the problems in this post at scale, and its **outlier detection** (passive health checking that ejects misbehaving hosts based on real traffic) is one of its most influential contributions. The motivating insight was that active health checks have lag and cost, and a pod can be *passing* its `/health` probe while *failing* real requests (a poisoned cache, a bad deploy, a degraded dependency). Passive ejection reacts to what users actually experience, in milliseconds, with no probe traffic. Lyft's experience — that a single bad host in a large fleet, left in rotation, disproportionately damages the tail — is why outlier detection is on-by-default-worthy and why Envoy's power-of-two-choices least-request became a de-facto standard balancing policy. The lesson: *watch the real traffic, not just a synthetic probe, and eject fast.*

**Kubernetes EndpointSlices: solving discovery at fleet scale.** As Kubernetes clusters grew to thousands of nodes and Services with thousands of endpoints, the original single-Endpoints-object design became a control-plane bottleneck — any endpoint change rewrote and re-pushed the entire giant object to every watching node, generating enormous traffic during deploys and churn. **EndpointSlices** (graduated to GA in Kubernetes 1.21) shard the endpoint list into ~100-endpoint chunks so a single pod change touches only one small slice. This is the same kind of scalability fix as Kafka partitioning a topic or a database sharding a table: take a single hot object that everything writes to and shard it so writes spread out. The lesson: *the data structures of your discovery system are themselves a scaling concern; a design that works for 4 pods can melt your control plane at 4,000.* The full Kubernetes treatment is in the upcoming [Kubernetes for microservices: the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) post.

**Consul at HashiCorp-shop scale: CP discovery with health checks.** Many enterprises outside Kubernetes (and bridging VMs, multiple clusters, and bare metal) standardized on Consul as a CP registry with rich health checking, often paired with `consul-template` to render upstreams into NGINX/HAProxy configs, or Consul Connect for a mesh. The lesson Consul teaches is the flip side of Netflix's: a CP registry gives you a *single consistent view* that is invaluable for cross-datacenter discovery and for use cases (leader election, locking) where AP would be wrong — but you must design for the partition case where writes are refused, typically by keeping enough headroom that you are not depending on registering new capacity *during* a partition. The lesson: *choose CP or AP for your registry deliberately, knowing that the choice is a CAP trade-off and that for pure service discovery, AP is usually the safer default.*

The common thread across all four: discovery and balancing are not "set the load-balancer policy and forget it." They are a system with a lifecycle, a consistency model, a scaling profile, and a set of failure modes that you must reason about explicitly — exactly like a database or a message broker, which is why this post leans on the same CAP, partitioning, and consistent-hashing fundamentals those systems do.

## When to reach for what (and when not to)

Here is the decisive recommendation, because a framework that does not tell you what to do is just trivia.

**If you are on Kubernetes and just need services to find each other:** use the built-in Service + readiness probe model. Do not install Consul or Eureka for basic discovery — Kubernetes already is your registry, and the readiness-gated EndpointSlice model is correct, health-aware, and free. Add a headless Service for any gRPC clients that need real client-side balancing. This covers the overwhelming majority of teams and you should not reach further until you have a concrete reason.

**If you have heterogeneous pods or a slow-pod problem:** switch your balancing from round-robin to least-request (it is usually a one-line config change), and turn on passive outlier detection. These two changes fix most tail-latency-from-load-balancing problems and cost almost nothing. Do this *before* reaching for anything fancier.

**If you need cache affinity or session stickiness:** use consistent hashing keyed on the affinity attribute, and accept the slightly-less-even load and the cold-cache spike on rebalance. Measure the cache hit-rate improvement to confirm it is worth it.

**If you are running across availability zones and the cloud bill matters:** turn on zone-aware / topology-aware routing, paired with a spill-over rule and enough per-zone capacity. This is high-dollar-impact and underused.

**If you have many services in many languages and maintaining discovery/resilience logic per-service has become a tax:** adopt a service mesh, knowing it adds a per-pod sidecar, a control plane to operate, and real complexity. The mesh is a scaling decision, not a starting point — and if you have fewer than, say, a dozen services all in one language, it is almost certainly premature.

**And reach for none of this — just call a stable name directly — when** you have a single instance of the downstream and no scaling, or when the two things should not have been separate services at all. If splitting them only bought you a load-balancing problem between two things that always deploy together, the [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) was the right call and discovery is a cost you did not need to pay.

The meta-rule: **lean on the platform's built-in discovery first, fix balancing with least-request and outlier detection second, and add a mesh only when the per-service tax exceeds the mesh's overhead.** Every step up that ladder buys you something and charges you something; name both before you climb.

## Key takeaways

- **"Where is service B" is a query, not a constant.** Pods are ephemeral; their IPs turn over in minutes. Never hardcode a host and never trust plain DNS with its caching and health-blindness — resolve against a live registry whose source of truth updates in seconds.
- **The registry's lifecycle is the whole game.** Instances register on boot, renew a TTL lease via heartbeats, and are *reaped* when heartbeats stop — self-healing with no human in the loop. The gap between "pod dies" and "registry notices" is your stale-routing window, and it is the number that drives your error rate during a pod death.
- **On Kubernetes you already have a registry.** Service + selector + readiness probe + EndpointSlices is a complete, health-aware, server-side discovery system. EndpointSlices exist to keep it from melting the control plane at fleet scale.
- **Client-side discovery saves a hop but taxes every language; server-side centralizes the logic at the cost of a hop.** Modern fleets hybridize: platform Services for the baseline, headless Services for gRPC client-side balancing, a mesh sidecar to get client-side-quality balancing for free in every language.
- **Round-robin is blind to load; least-request self-corrects.** Round-robin sends the same share to a 10x-slower straggler and lets it poison your tail; least-request (P2C) stops feeding the slow pod and cuts p99 by an order of magnitude. Make least-request your default.
- **Consistent hashing buys cache affinity, not even load.** Routing the same key to the same pod can lift a cache hit-rate from 60% to 95% and cut downstream DB load ~8x — at the cost of less-even distribution and a cold-cache spike when pods change.
- **Health-aware balancing means passive, not just active.** Active probes have lag; passive outlier detection ejects a misbehaving pod after a handful of real errors in milliseconds, collapsing the stale-registry window from 30 seconds to nearly nothing. It is non-negotiable in production. Keep dependency checks in readiness, never liveness.
- **A pod dying mid-request is survivable if writes are idempotent and retries go to a different host with a budget.** Without that, it is a 503 for a customer; with it, it is 30ms of extra latency they never notice.
- **For service discovery specifically, registry availability usually beats consistency.** Callers can route around stale entries with their own health checks, but they cannot route around a registry that refuses to answer — which is why Netflix chose AP and why CP registries must keep capacity headroom for the partition case.
- **A service mesh moves all of this into the sidecar.** It resolves the client-side-versus-server-side dilemma by giving every language correct, declarative, health-aware balancing for free — but it is a scaling decision with real overhead, not a default.

## Further reading

- **Chris Richardson, *Microservices Patterns*** — the service-discovery chapter is the clearest practitioner treatment of the client-side-versus-server-side fork and the registry's role; Richardson's microservices.io catalogs the discovery patterns by name.
- **Sam Newman, *Building Microservices* (2nd ed.)** — the service-discovery and load-balancing sections, plus the broader argument for leaning on the platform.
- **The Envoy documentation on load balancing and outlier detection** — the definitive reference for P2C least-request, locality-aware routing, and passive health, written by the team that built it at Lyft.
- **The Kubernetes documentation on Services, EndpointSlices, and Topology Aware Routing** — the authoritative source for the built-in discovery model and the scale fixes.
- **The Netflix Tech Blog posts on Eureka and Ribbon** — the rationale for AP discovery at scale and the client-side balancing model that defined a generation.
- In this series, read next: [health checks: readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) for the health signals that feed the balancer, [service mesh: Istio, Linkerd, when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) for moving all of this into the sidecar, and [Kubernetes for microservices: the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) for the platform that gives you discovery out of the box. For the mechanism behind the algorithms, [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) and [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) are the deep dives this post builds on.
