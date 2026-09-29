---
title: "Service Discovery from DNS to Registries to xDS: What Clients Know During a Partition"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Trace how a client learns its next endpoint, detect stale answers, and choose a deliberate failure policy when the control plane disappears."
tags:
  [
    "networking",
    "distributed-systems",
    "service-discovery",
    "dns",
    "consul",
    "etcd",
    "envoy",
    "xds",
    "health-checks",
    "control-plane",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-1.webp"
---

Imagine a request to `payments` failing while the payments process is healthy. The client can reach its IP address. A direct `curl` returns a response. The production client still returns `no healthy upstream`. That error is usually read as a server verdict. Often it is a statement about the client's *view* of the server: which endpoints it learned, when it learned them, whether it accepted an update, and what it did when the source of updates stopped answering.

![A path map locating service discovery before the connection to a backend](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-1.webp)

The diagram above is the mental model. Discovery runs before, or in parallel with, the data-plane connection. Its output becomes an input to routing. A good endpoint behind a stale or empty view is invisible to the caller; a dead endpoint in a stale view can keep receiving calls. To place this boundary alongside the rest of a request, start with [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). To understand the DNS cache path specifically, see [DNS caching layers and stale answers](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers).

The operative question is not merely "where is this service?" It is "which set of endpoints did this client believe was valid at this instant, and why?" Static files, DNS, a service registry, and xDS all answer the first question. They make different promises about the second. During a partition, those promises are more important than their normal-path lookup latency.

## 1. Split the control plane from the data plane

The **data plane** is the path that carries a user request to a backend. The **control plane** is the path that tells the client or proxy which backends exist and which are eligible. In the simplest deployment, a file on disk is the control plane. With DNS, a resolver and authoritative server distribute an address set. With Consul, etcd, or xDS, a long-lived subscription may push changes into a local view. These systems can fail separately. A packet to the backend may flow while a packet to the registry does not, and the reverse is possible.

Consider a caller with a cached set of two endpoints. It sees one backend fail a request. The discovery stream remains connected but has not published a removal. Four statements are now independently testable: the backend process is healthy, the backend is reachable from the caller, the discovery authority considers it healthy, and the caller's current local view includes it. Conflating those statements is how teams spend an incident restarting an application that was never the problem.

The endpoint record itself is also richer than a hostname. An address and port determine where a TCP connection begins. A service identity determines *which* logical dependency is being requested. Priority, locality, weight, metadata, and health status may determine whether the endpoint is eligible for a particular caller. A registry can store these attributes. DNS A and AAAA records carry addresses; SRV records carry a target and port, priority, and weight. The fact that a source can represent metadata does not mean the client actually reads or honors it. The usable contract is the intersection of what the source publishes and what the consumer implements.

Measure the two planes separately. `curl` or `nc` to a known backend answers whether a data path and listener work. `dig` to the resolver, an HTTP query to the registry, or Envoy's configuration dump answers whether the control path has a usable view. A successful `ping` to a node answers neither application readiness nor discovery freshness. The [microservices service discovery overview](/blog/software-development/microservices/service-discovery-and-load-balancing) covers the architectural pattern. Here we will stay with the bytes, state transitions, and failure semantics on the wire.

> A healthy backend is no use to a caller that cannot name it; a fresh name is no use if the route to it is broken.

### A useful vocabulary for one endpoint

Call the authoritative endpoint set $A(t)$ and the set currently used by client $i$ $C_i(t)$. These symbols are an **explanatory model**, not a protocol equation. At a given instant, `C_i` may omit a new endpoint, include a removed one, or contain the right addresses with old weights. A control-plane message changes `C_i`; it does not carry a production request to the backend. A health check is one possible source of an update to `A`, but a failed health check does not prove that every client would fail a real request.

The model exposes three distinct delays. Detection delay is the time until an observer decides an endpoint changed state. Publication delay is the time until the authority records that decision. Consumption delay is the time until a caller receives and applies it. For a particular change, a useful approximation is:

$$
T_{\mathrm{visible}} \approx T_{\mathrm{detect}} + T_{\mathrm{publish}} + T_{\mathrm{consume}}.
$$

This is a budgeting model. Some stages overlap; retries and batching can make the tail much longer than the sum of typical values. It tells us what to instrument, not a guaranteed convergence bound. A dashboard that shows only fast DNS server responses measures one small part of this chain.

## 2. The progression from a file to DNS to a registry to xDS

![A matrix comparing static configuration, DNS, a registry, and EDS or xDS](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-2.webp)

Static configuration is attractive because it is explicit. A caller reads `payments=10.77.0.2:8080` at startup. There is no runtime dependency on a name server. There is also no automatic way for the caller to learn that a replacement now lives elsewhere. Someone or something must rewrite and reload the file. A deployed file may remain useful during a control-plane outage precisely because it is stale. It can also direct every caller at an address that has been reassigned. The failure policy is implicit unless the owner states when the file may be used and how it is replaced.

DNS moves the mapping behind a name. A client asks for `payments.example`, and one or more caches may answer. The authoritative record's time to live, or TTL, is a cache lifetime instruction, not a promise that every live connection will move when the TTL ends. [RFC 1035, published in November 1987](https://www.rfc-editor.org/rfc/rfc1035), defines the TTL field for resource records. The application may hold a connection pool, a runtime may cache an answer, a forwarder may cache it, and a resolver may return a changed list while the client keeps using its previous socket. The DNS post in this series follows those layers in detail. For discovery, remember that updating the record and changing traffic are different events.

Suppose, as a **derived illustrative scenario**, the resolver caches an address for 30 seconds, the application refreshes its own endpoint set every 20 seconds, and the next request arrives immediately after a backend replacement. If the application refresh and DNS cache happen in the worst order, a client may need approximately 30 seconds for the resolver's old answer to expire and up to another 20 seconds for its next application refresh, before connection establishment and retry time. The bound is a model, not a guarantee: connection reuse may last longer, a resolver may impose its own policy, and a failed lookup may be negatively cached. Lowering the authoritative TTL to 5 seconds only attacks one term. It can increase lookup load without changing an application that never re-resolves.

A registry separates service membership from the DNS record. Instances register an address, port, identity, and optional health information. Consumers query or watch the registry. [Consul's discovery documentation](https://developer.hashicorp.com/consul/docs/discover) describes a catalog populated by service registration and health checks. Consul can expose the result through DNS or HTTP. The DNS interface is convenient for an existing client but does not erase caching behavior between the agent and the application. Its [DNS configuration documentation](https://developer.hashicorp.com/consul/docs/discover/dns/configure) says the default response TTL is zero, which avoids downstream DNS caching in compliant clients but increases query work. A zero TTL also cannot force an application to stop using an existing TCP connection.

etcd is a lower-level key-value store rather than a complete service-discovery product. A controller can store endpoint records under a prefix, attach leases to ephemeral records, watch revisions, and materialize a local endpoint view. This gives precise state semantics, at the cost of building registration, health policy, resync, and client behavior yourself. [etcd's v3 API documentation](https://etcd.io/docs/v3.6/learning/api/) distinguishes linearizable range reads, which reflect current cluster consensus, from member-local serializable reads that can be stale. Its watches stream changes after a revision. This is a good toolkit for controllers; it is not a guarantee that an arbitrary caller is instantly in sync.

Envoy's Endpoint Discovery Service, or EDS, distributes endpoint assignments to proxies through the broader xDS configuration protocol. The application can send to a local or nearby proxy while that proxy consumes control-plane updates. [Envoy's dynamic configuration overview](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/operations/dynamic_configuration) explicitly distinguishes cluster discovery, endpoint discovery, listeners, and routes. EDS can carry attributes DNS cannot conveniently express for proxy load balancing, such as locality and canary information. Moving discovery into a proxy moves the state and its observability into the proxy as well. The application still needs a policy for the proxy's own absence or failure.

| Mechanism | Update carrier | Client-side state | Failure question | Source |
| --- | --- | --- | --- | --- |
| Static file | Deployment or reload | Parsed file and connections | How long can an old file be used? | Design model derived here |
| DNS | Queries and cached resource records | Resolver cache, runtime cache, connections | Which cache still has the old answer? | [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035) and [Consul DNS](https://developer.hashicorp.com/consul/docs/discover/dns/configure) |
| Consul registry | Agent query, HTTP, or Consul DNS | Last query result or watch state | Which read mode and leader view supplied it? | [Consul consistency modes](https://developer.hashicorp.com/consul/api-docs/features/consistency) |
| etcd-backed controller | Range read plus revisioned watch | Materialized snapshot and revision | Can the watch resume, or must it reload? | [etcd v3 API](https://etcd.io/docs/v3.6/learning/api/) |
| EDS and xDS | Management server response over a stream | Proxy's accepted resources | What version did this proxy accept? | [Envoy xDS protocol](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol) |

The comparison is not a maturity ladder. A stable single-host dependency may be best served by a file. A DNS name may be enough when instances change slowly and callers retry safely. A registry or proxy becomes useful when eligibility, locality, rapid churn, and policy need explicit distribution. It also becomes another system whose overload can change the behavior of every application that depends on it.

### DNS can carry membership without carrying the entire contract

An A or AAAA lookup returns addresses, not an application-ready promise. An SRV lookup can return target names and ports, but the caller must explicitly ask for SRV and implement its priority and weight behavior. A system that publishes SRV while its client asks only for A has not gained SRV semantics. Likewise, a resolver may rotate answer order, but a caller that always picks the first address can produce a skewed traffic distribution. A caller that holds one long-lived connection may ignore rotations entirely. Observe the *selected endpoint*, not just the order of records printed by `dig`.

DNS has a second subtle boundary: a record can be removed at the authoritative server while a recursive resolver is still allowed to return its previously cached copy. [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035) defines the TTL as the cache lifetime associated with a resource record. That TTL does not specify how a language runtime should manage an already resolved address or how long a connection pool can keep a socket. The relevant time to remove traffic is therefore a chain of events: record publication, recursive cache expiry, runtime lookup, endpoint selection, old connection retirement, and eventual retry. To diagnose a specific client, sample at those boundaries in that order. If the authoritative record is new and the recursive answer old, the DNS cache is the first divergence. If both DNS answers are new and the application dials the old IP, the application's resolver or connection state is the first divergence.

Negative answers deserve equal attention. A service that temporarily has no healthy instances may produce a negative response. A forwarder or client can cache that negative answer, so even when an instance returns, callers can continue to believe there is no service. Consul's [DNS configuration guide](https://developer.hashicorp.com/consul/docs/discover/dns/configure) specifically discusses negative-response caching and its effect on recovery. It recommends checking the operating system and forwarders, rather than assuming the TTL of positive service records controls the result. In an incident, record the DNS response code and authority section along with the answer section. "No A record" is too imprecise to distinguish a deliberately empty service from a cached lookup failure.

Lowering a TTL is therefore a focused tool. It increases query frequency and reduces one potential cache delay if clients honor the shorter value. It cannot make a dead process healthy, make a watch consumer accept a rejected xDS resource, or force an established connection to reconnect. Before a planned endpoint migration, identify the actual consumer behavior in a staging environment: resolve, connect, hold a socket, change the record, and time the next successful request to the new endpoint. This experiment measures the whole path, including connection reuse, instead of treating an authoritative TTL as the migration completion time.

### Registries centralize truth and decentralize copies

The word "registry" can suggest a single current list. In operation there is an authoritative state, replicas that may lag, agents with local state, and clients with their own accepted snapshots. A Consul catalog entry may be updated by an agent health check and then replicated. A DNS query against a Consul agent can use a stale read. A client can then cache the response or retain a connection. That is a sequence of copies, not one indivisible database read. Consul's [consistency documentation](https://developer.hashicorp.com/consul/docs/concept/consistency) describes anti-entropy between agent local state and the global catalog; its [API consistency guide](https://developer.hashicorp.com/consul/api-docs/features/consistency) describes leader and follower read choices.

For any reported "registry latency," ask which operation was timed. An HTTP GET served from a follower can be quick while writes are waiting on consensus or storage. A watcher can keep its TCP connection open while the update it needs is queued behind a slow producer. A client can receive a change quickly but reject it during validation. The useful measurements are write commit latency, replica or last-contact lag, delivery lag to the consumer, last accepted version, and endpoint selection in a real request. These measurements correspond to the detection, publication, and consumption stages of our explanatory model. A single query-duration histogram cannot replace them.

The same separation prevents a dangerous repair. If the registry is overloaded by polling, replacing polling with streams can reduce repeated full reads. But a stream can create many subscriptions and a large fanout for each change. Roblox's report is a concrete warning that the workload shape matters. A feature that saves work on one axis may move contention to another axis under high update churn. Before changing client behavior fleet-wide, replay a workload that includes normal change rate, bursts of registrations, failure-driven health changes, and mass reconnect after a control-plane restart. Measure both the management system and the clients' accepted update age.

## 3. Health is an observation from a particular path

![A graph showing active probes and passive request observations reaching different conclusions](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-3.webp)

An **active check** creates traffic specifically to test an endpoint. It can make a TCP connection, perform an HTTP request, or call a gRPC health method. A **passive check** observes requests that would have happened anyway: failures, resets, timeouts, or unusual latency. Consul documents HTTP, TCP, gRPC, and TTL checks in its [health-check guide](https://developer.hashicorp.com/consul/docs/register/health-check/vm). Envoy calls its passive mechanism [outlier detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier): a proxy compares upstream behavior with expectations and may eject an endpoint from its local healthy set. These are different observation paths, so agreement is useful evidence and disagreement is a diagnostic clue.

An active HTTP probe to `/health` from a registry agent can pass while user requests fail because the probe avoids authentication, a downstream database, or the source network of a subset of callers. A passive proxy can see a high failure rate for one caller's requests while a regional probe still succeeds. The inverse happens when a probe path is broken but application traffic on another path works. Neither mechanism is a universal oracle. Name the path, request shape, timeout, and failure threshold before calling a check "health."

The timing cost matters. Suppose an active probe runs every 10 seconds, takes at most 2 seconds to time out, and requires three consecutive failures. In a simple worst-phase **derived model**, a failure that begins just after a successful probe can wait almost 10 seconds for the first failed probe, then roughly 20 seconds for two more starts, and up to 2 seconds for the final timeout: approximately 32 seconds to decision, before publication and client consumption. Real implementations differ in whether thresholds count starts or completions, whether intervals jitter, and whether an agent delays deregistration. The useful act is to measure the actual sequence in logs. Setting the interval to one second to chase a shorter bound multiplies probe traffic and can make a sick control plane sicker.

Passive checks avoid that steady probe cost, but need user traffic to observe a failure. A cold endpoint with no requests can remain apparently healthy. A low-traffic endpoint may take a long time to cross a consecutive-failure threshold. And when the failure is a shared downstream service, ejecting every endpoint can produce an empty pool rather than a recovery. A cautious policy limits ejection and uses active probing to admit an endpoint again. The [health, readiness, and liveness post](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) owns the application-level contract for readiness; the network question here is which observer's result reaches which caller, and when.

Health can also be scoped differently. A registry may say an instance is globally unhealthy; a proxy can find that it is unreachable only from one zone. Globally withdrawing an endpoint because one observer has a partition can amplify a local network failure. Locally ejecting it may be safer for that caller, but it makes endpoint views intentionally different across proxies. That is a valid design if the operator can explain it and observe it. Use a direct request from the affected client path, a check result from the health observer, and the proxy's current endpoint set to separate these cases.

### Do not confuse a liveness signal with membership truth

A lease or TTL registration says the writer refreshed a timer. It does not prove the service can complete the target RPC. Conversely, missed keepalives can result from a path between the service and registry while the service still answers its callers. This distinction becomes acute during a partition. Automatically removing the instance from a global catalog can be correct for clients on the far side of the partition and incorrect for clients on the near side. If the service holds a unique writer role, a mere health check is especially dangerous as a failover trigger. Discovery names candidates; a separate authority or fencing mechanism must protect exclusive ownership.

## 4. Watches are state machines, not magic push notifications

![A state machine for taking a snapshot, applying watch events, and resynchronizing after a gap](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-4.webp)

Polling a registry repeatedly is simple but spends reads when nothing changes and creates a polling interval between publication and consumption. A watch replaces many queries with a long-lived stream of changes. The correct client still needs an initial state, an ordering point, reconnect behavior, and a response to missed history. "The connection is open" does not mean "the view is complete." A stalled stream can leave a perfectly reachable proxy serving an old set.

A robust etcd-backed controller follows a specific pattern. It reads the endpoint prefix and records the response revision. It then watches from the next revision, applies changes in order, and tracks the last applied revision. On reconnection, it resumes from that point if history still exists. If etcd reports that the requested revision has been compacted, the client discards the assumption of continuity, takes a new snapshot, and starts a new watch. These are the semantics described by [etcd's v3 Range and Watch APIs](https://etcd.io/docs/v3.6/learning/api/). A watch event is not a substitute for the snapshot: it says what changed, not necessarily the entire current membership set.

The same discipline applies even when the protocol hides the revision details. Track the last successful update, the version accepted by the consumer, and the age of the local view. Alert on a growing age when the service is expected to change. Avoid an alert that fires only when the stream is disconnected: a stream can be connected and not progress because a server is overloaded or an update was rejected. A client that silently handles a reconnect by clearing its endpoint set can cause an outage that a last-known-good policy would have avoided. A client that keeps every old endpoint forever can route to a retired address. Both are policies, whether chosen deliberately or by default.

### A revision is not a wall clock

A monotonically increasing revision lets the client detect ordering and gaps. It does not directly state how many seconds old the endpoint is. A wall-clock timestamp needs synchronized clocks and a definition of where the timestamp was recorded. Observe both when possible: revision gap to identify missed updates, and elapsed time since accepted update to bound the practical age of the view. Even then, an unchanged endpoint set can legitimately produce no update for a long time. A heartbeat or progress message helps distinguish quiet data from a stalled stream, but only if the protocol and consumer implement it. etcd's API offers progress notifications with a server-selected cadence; it does not promise a fixed timer to the application.

The state must survive updates atomically. If a controller writes a local file or proxy configuration, it should build the next complete set, validate it, and swap it into use without exposing a half-built list. A dropped event or failed parse should leave the prior accepted version available while raising a visible error. The useful operational question is "which exact version is serving requests now?" not "did the control plane send a message?"

## 5. xDS adds explicit acceptance to endpoint distribution

EDS answers which endpoints belong to a cluster, while CDS can define the cluster and LDS or RDS can change listeners and routes. Envoy's [xDS protocol specification](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol) describes a management server response with a version and nonce, followed by an Envoy request that acknowledges or rejects that response. A nonce ties the ACK or NACK to the specific response. The version describes the accepted resource state for that type. This means the control plane can know whether a proxy considered an update valid. It does **not** prove the proxy has successfully applied every resource or that a packet can reach the endpoint. Envoy's own protocol documentation explicitly limits what an ACK signifies.

The boundary is useful in incident response. If the management server publishes version `v8` but a proxy continues to report `v7` with an error detail, inspect the rejected resource. Restarting the backend will not fix a schema or reference error. If the proxy accepted `v8` but traffic still goes to an old address, inspect the proxy's active configuration, cluster host set, connection reuse, and whether another resource type supplies the route. If all proxies have the new endpoint but connects fail, discovery has succeeded and the data path is the next layer to investigate.

State of the world xDS sends a resource set for a type. Delta xDS sends additions and removals, avoiding retransmission of a large set for a small change. The [Envoy xDS documentation](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol) explains both variants and notes that a new stream begins with the client's most recently accepted version. In either variant, the operator still needs to decide what a proxy does when the management server is unavailable. Cached accepted configuration can keep traffic flowing, but it can be wrong for a service whose endpoint set changed. A TTL on an xDS resource can expire it; an unbounded last-known-good resource remains available but can outlive its safety window. Choose this per dependency, not as a single fleet-wide slogan.

Bootstrap deserves attention. A proxy needs enough static configuration to contact its management server. If that management server is discovered only through the same dynamic configuration it serves, the system has a circular dependency. Envoy's [initialization documentation](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/operations/init.html) describes bounded initial fetch behavior and resource warming. A process being up does not imply it has received the first endpoint set. Check readiness and accepted resources separately at startup, especially after a full regional restart when caches are cold.

## 6. A partition makes the policy visible

When a client loses the control plane but retains the data path, the client has only information from before the cut. It can continue with its last accepted endpoints, reject new requests until it can refresh, or use a bounded cache for a while and then fail closed. No protocol removes that choice. A short TTL can make the choice happen sooner. A watch can deliver updates sooner while the stream works. Neither can tell an isolated client about an endpoint change that occurs beyond the partition.

The sole animated figure shows the sequence that static snapshots tend to hide: the consumer begins with a valid view, the control link breaks, and the authority changes the endpoint set while the consumer keeps its old copy. The moving update stops at the partition. The data link may still work. Whether the old endpoint is safe depends on what changed and the service's contract.

<figure class="blog-anim">
<svg viewBox="0 0 900 300" role="img" aria-label="The authority advances from endpoint generation one to two during a partition, while the isolated client keeps generation one" style="width:100%;height:auto;max-width:900px">
<style>
.sd16-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.sd16-label{font:600 19px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.sd16-small{font:15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.sd16-line{stroke:var(--border,#d1d5db);stroke-width:4}.sd16-cut{stroke:#dc3545;stroke-width:5;stroke-dasharray:10 7}.sd16-update{fill:var(--accent,#6366f1)}
@keyframes sd16-send{0%,22%{transform:translateX(0);opacity:0}28%{opacity:1}60%{transform:translateX(265px);opacity:1}65%,100%{transform:translateX(265px);opacity:0}}
@keyframes sd16-new{0%,20%{opacity:0}30%,100%{opacity:1}}
.sd16-moving{animation:sd16-send 12s ease-in-out infinite}.sd16-newer{animation:sd16-new 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sd16-moving{animation:none;opacity:0}.sd16-newer{animation:none;opacity:1}}
</style>
<rect class="sd16-box" x="35" y="65" width="245" height="165" rx="12"/>
<rect class="sd16-box" x="620" y="65" width="245" height="165" rx="12"/>
<text class="sd16-label" x="157" y="105">Authority</text>
<text class="sd16-label" x="742" y="105">Isolated client</text>
<text class="sd16-small" x="157" y="145">generation 1</text>
<text class="sd16-small sd16-newer" x="157" y="184">generation 2 published</text>
<text class="sd16-small" x="742" y="145">generation 1 cached</text>
<text class="sd16-small" x="742" y="184">old endpoint still used</text>
<line class="sd16-line" x1="280" y1="145" x2="620" y2="145"/>
<line class="sd16-cut" x1="560" y1="96" x2="560" y2="210"/>
<circle class="sd16-update sd16-moving" cx="300" cy="145" r="12"/>
<text class="sd16-small" x="450" y="260">control update cannot cross the partition</text>
</svg>
<figcaption>The authority publishes a new generation, but the isolated client keeps routing with its last accepted endpoint set.</figcaption>
</figure>

For a read-only cache that tolerates a brief stale replica, continuing to use the old endpoint may preserve useful service. For a payment writer whose identity or authorization has changed, the same behavior can violate correctness. Fail closed can be a good safety decision, but it makes a control-plane partition an application outage even while backend sockets work. There is no universal setting that maximizes both availability and current membership truth under a partition. State the acceptable age and the consequence of an old endpoint before selecting the behavior.

Consul makes this trade-off concrete. Its [consistency-mode documentation](https://developer.hashicorp.com/consul/api-docs/features/consistency) states that Consul DNS uses stale reads by default, allowing a non-leader server to answer. The HTTP API offers modes with different consistency and load trade-offs. Headers such as `X-Consul-LastContact`, `X-Consul-KnownLeader`, and `X-Consul-Effective-Consistency` help a caller assess the result. A stale read can keep answering during a leader problem; it cannot promise that a removed endpoint is absent. A consistent read may be unable to answer when the quorum path is unavailable. The [Consul DNS configuration guide](https://developer.hashicorp.com/consul/docs/discover/dns/configure) warns that forcing stale reads back to the leader under overload can worsen the overload.

This gives us a more useful policy table than "DNS versus Consul". For each dependency, decide whether old membership is acceptable, how long, and whether a local passive failure should eject an old endpoint even when the global authority cannot be reached. If the endpoint provides exclusive write ownership, discovery freshness alone is insufficient. Require a lease, fencing token, or transaction check at the stateful authority. The [resilience-patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) owns retry and circuit-breaker policy; this post owns the endpoint set those mechanisms act on.

| Situation | Candidate policy | What it buys | Failure it admits | Source |
| --- | --- | --- | --- | --- |
| Read-only service, old endpoints remain useful | Bounded last-known-good | Requests can continue across a control-plane cut | May omit new replicas or send to a removed one | Derived design choice |
| Exclusive writer or authorization-sensitive target | Require a fresh authoritative decision | Avoids acting on an old writer or policy | Control-plane partition becomes unavailability | Derived design choice |
| Local path to one endpoint fails, other paths work | Local passive ejection with probe-based recovery | Avoids a known-bad path for that proxy | Different proxies can have different healthy sets | [Envoy outlier detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier) |
| Registry follower lags leader | Accept or reject stale read by service | Trades read throughput for freshness | Follower answer can lag; leader path can overload or fail | [Consul consistency modes](https://developer.hashicorp.com/consul/api-docs/features/consistency) |

### Calculate a stale-use budget explicitly

Suppose, as a **derived example**, an endpoint rotates every 12 hours and the service owner permits a cached endpoint for 120 seconds after loss of updates. That is a policy, not a probability of safety. The client records `last_good_at` when it accepts a complete valid set. If the control plane fails at time $t_0$, it may use that set while $t - t_0 \leq 120\,\mathrm{s}$, provided local connection attempts still succeed. After the bound, it stops routing or switches to a documented fallback. If the business requirement instead says an old writer must never be used after failover, even 120 seconds is unacceptable. The rule is determined by the cost of wrong routing, not the convenience of a configuration knob.

It is easy to choose a budget that no part of the system enforces. A DNS TTL is not an end-to-end age limit. A client might refresh the DNS record but reuse a socket. An xDS resource TTL matters only if the management server and client use it as intended. A registry lease may expire at the server while a disconnected client keeps a previous snapshot. Instrument the age where traffic is selected: in the caller's resolver, connection pool, or proxy, not just at the authority.

### Worked example: an old replica and an old writer

Take two services with the same endpoint update mechanism. A search service reads an index replica. A ledger service writes to one designated leader. Both clients lose their control-plane connection while the data-plane sockets remain open. The search service's old replica may still return a slightly older document set. If that behavior is explicitly within its contract, a bounded last-known-good set can be useful. The ledger client faces a different risk. If the former writer was demoted after the partition, an old address is not merely a stale performance choice. It can conflict with the current writer. The right safeguard lives at the write authority: a fencing token, term, transaction check, or equivalent rejection of commands from the old owner. Discovery helps find a candidate; it cannot by itself make an exclusive writer safe.

This example is a design scenario, not a report of either system in production. It illustrates why an organization-wide setting such as "serve stale for ten minutes" is underspecified. Ten minutes of stale search reads might be tolerable for one product and unacceptable for another. One stale ledger write may be unacceptable at any duration. The client should carry a dependency-specific classification: whether stale membership is permissible, the maximum age, whether local failure observations can remove endpoints, and what happens when the eligible set becomes empty. Put those decisions in configuration that can be inspected during an incident. A hidden default in a library will make the same outage look different across languages and client versions.

Now add a location change. Suppose the old endpoint remains reachable from zone A but not zone B. The global catalog marks it healthy because a probe in zone A succeeds. A caller in zone B should prefer its local passive evidence and stop selecting that endpoint, while not necessarily deleting the record globally. The control plane's membership set and the caller's eligible set may intentionally differ. Conversely, if a global authorization change removes the endpoint, allowing a local proxy to keep using it because its TCP connection still works may violate policy. The correct precedence between global membership and local health follows the service's safety contract. It cannot be inferred from one green probe.

### The control plane has a capacity budget

An endpoint update can fan out. If a service has 5,000 proxy subscribers and one endpoint changes, the management layer may need to encode, queue, and deliver that change to many consumers. That subscriber count is an **illustrative input**, not a claim about any deployed system. In a simple, derived upper-bound model with a 2 kB response per subscriber, one full response to each of 5,000 subscribers is about 10 MB of payload before framing and replication. If changes arrive ten times per second and every change causes a full response, the model reaches about 100 MB/s of outbound payload. Delta updates, batching, subscriptions by resource, and compression can greatly change that cost; producer CPU and lock contention may dominate first. The arithmetic matters because "push" is not free. It exchanges periodic read load for fanout load linked to change rate.

The same model explains why reconnect storms are dangerous. A control plane restarting after a partition may face thousands of clients asking for initial state at once, not the ordinary steady change rate. Warm caches, bounded client concurrency, randomized reconnect backoff, and incremental rollout can reduce the spike. A last-known-good client can often keep serving while retrying the control stream with backoff, provided its stale-use window has not expired. A fail-closed client may not have that option, so its reconnect path must be capacity-tested as a critical availability path. The [timeouts, retries, and backoff post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers the general retry policy. Here, count subscriptions and update bytes as part of the discovery system's own workload.

Do not infer that a successful delivery means a successful rollout. A management server may publish an update, the stream may transfer its bytes, and the proxy may NACK it. Or the proxy may ACK individual resources and still fail a later application step, as the xDS documentation cautions. Track at least three states separately: desired version at the source, delivered response at the transport, and accepted active version at each consumer. A rollout controller should not declare completion solely because it wrote a new registry value. It should check adoption across the consumers that matter, then sample real requests to verify the data path. This is especially important when replacing a whole region of endpoints and old connections may outlive the discovery change.

### Separate change propagation from request success

A useful incident timeline records five moments: when the backend changed, when the health observer detected it, when the authority committed the new set, when the affected consumer accepted it, and when that consumer's first request used the new set. The gaps localize delay. If the authority commit is prompt but the consumer is late, look at a cache, stream, watch, or NACK. If the consumer accepted promptly but requests still target old addresses, look at connection pooling or routing configuration. If requests target the new address and fail, discovery has completed and a listener, route, or application dependency is now the suspect.

That timeline should include the *negative* direction as well as the positive one. Removing a dead endpoint and adding its replacement have different risks. Removing too slowly wastes requests on a failed address. Adding too early sends traffic to a process that is listening but not ready. A rolling deployment can make both errors in sequence. Readiness checks, endpoint publication, and connection draining should form a coordinated change: the old endpoint stops taking new traffic, existing work drains under a deadline, and the new endpoint is published only after the real request path is ready. The exact order depends on the load balancer and application contract, but the observations above let an engineer verify what actually happened rather than relying on deployment labels.

## 7. Roblox: when the discovery dependency failed first

![A timeline of Roblox's October 2021 outage with streaming contention and BoltDB freelist amplification separated](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-5.webp)

Roblox's [owner postmortem, published in January 2022](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021), reports a 73-hour outage from October 28 through October 31, 2021. This is a case about a control plane that had become a broad dependency, not a case where a DNS TTL alone was too high. The postmortem says services relied on Consul for the locations of dependencies. Nomad and Vault also relied on Consul. At 13:37 on October 28, engineers noticed degraded Vault performance and high CPU on one Consul server, before player impact. By 16:35, online players had dropped to half their normal level. When Consul could not keep up, services struggled to connect to peers, schedule containers, and retrieve secrets.

The incident did not have one neat trigger. Roblox had upgraded from Consul 1.9 to 1.10 and had gradually enabled a streaming feature to reduce CPU and network cost of distributing updates. The owner report says it was enabled on a traffic-routing backend on October 27 at 14:00, one day before the outage. Under Roblox's unusual combination of many streams and high churn, that feature caused contention and poor performance. Separately, Roblox's write workload exposed a pathological BoltDB freelist cost in the Raft log store. The BoltDB problem was not the same bug as the streaming contention, and it was not simply a corrupt current-state database. The owner report says BoltDB stored the Raft log, not Consul's current state.

That distinction matters because the first recovery attempts targeted plausible but incomplete explanations. The team replaced hardware, including machines with more CPU and faster disks. It restored a snapshot and controlled access with `iptables`. When internal service-discovery and health-check traffic returned, Consul degraded again even without user traffic. The owner report says the median KV write latency that was normally below 300 ms had risen to about 2 seconds in this incident. These are Roblox's reported measurements in that environment, not general Consul thresholds. It also says the team reduced health-check frequency from 60 seconds to 10 minutes while lowering internal load. The action bought headroom but did not, by itself, solve the root causes.

The BoltDB mechanism is a precise warning against looking only at logical write size. Roblox's postmortem explains that deleted log pages remained in the file as reusable free pages, tracked in a freelist. Under the observed workload, the log store was 4.2 GB with 489 MB of actual data and 3.8 GB of free space. Its freelist was 7.8 MB, containing nearly a million free page IDs. The owner report says appending 16 kB or less of raw data could cause that 7.8 MB freelist to be rewritten. A tiny logical update therefore induced much larger storage work. The postmortem attributes slow Raft writes and leader consistency trouble to this behavior. None of those numbers should be applied to a different Consul cluster without measuring its storage and write pattern.

By 54 hours into the outage, Roblox reported that streaming had been disabled and the team had a process to prevent slow leaders from remaining elected. Consul was stable enough to begin the application return. Cache deployment and scheduler state then became recovery blockers. Some services had been shut down or scaled down. Roblox brought players back gradually with DNS steering; at 16:45 on October 31, 73 hours after the start, the owner report says all players had access. The incident illustrates how recovery time can include rebuilding dependent data-plane capacity after the control-plane mechanism is fixed. A registry returning fast again is not the same as every caller having a healthy, correctly sized backend set.

The transferable guardrail is to load-test the control plane for the *shape* of the workload: number of subscribers, change frequency, write frequency, health-check churn, and reconnect bursts. Isolate workloads where one cluster carries discovery, scheduling, and secrets. Monitor accepted endpoint age and control-plane write latency, not merely process liveness. Keep a documented last-known-good or fail-closed policy per dependency. Most of all, rehearse a controlled return from a fully down state: a recovered registry can be overwhelmed by the clients that faithfully reconnect to it.

### Evidence ledger for this case

| Field | Verified record | Source |
| --- | --- | --- |
| Case | Roblox service outage involving Consul, Nomad, Vault, and dependent services | [Roblox owner report](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021) |
| Event date | October 28–31, 2021 | [Roblox owner report](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021) |
| Source owner | Roblox, with HashiCorp collaboration described in the owner report | [Roblox owner report](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021) |
| Mechanism | Consul streaming contention under high subscription churn, plus a separate BoltDB Raft-log freelist performance pathology | [Roblox owner report](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021) |
| Verified numbers | 73-hour outage; normal median KV operation below 300 ms versus around 2 seconds during degradation; 4.2 GB log store, 489 MB actual data, 3.8 GB free space, 7.8 MB freelist; these are Roblox's reported context-specific figures | [Roblox owner report](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021) |
| Transfer lesson | Test subscriber churn and write amplification; observe last accepted state; avoid shared control-plane blast radius; rehearse warm recovery | Derived from the cited incident |

## 8. Diagnose the view before changing the source

![A decision tree separating endpoint-view age, accepted version, and data-path reachability](/imgs/blogs/service-discovery-from-dns-to-registries-to-xds-6.webp)

When a service appears absent, first ask whether the caller has any endpoints. If it has none, inspect the source query and the consumer's acceptance state. A DNS `NXDOMAIN` is different from an empty healthy set, and both are different from a client that never queried. If the caller has endpoints, compare its exact addresses, ports, and version to the authority's set. A difference locates the problem on the control path. If they match, test TCP and application requests to each endpoint from the affected caller path. That puts the problem on the data path or backend.

For DNS, record the queried name, resolver address, answer, TTL, and time. `dig +noall +answer @RESOLVER payments.example A` shows the DNS response from that resolver; it does not reveal the application's runtime cache. Query the application's actual resolver path if possible. For a registry, record the read mode and last-contact metadata. For an etcd-backed controller, record snapshot revision, last applied watch revision, and any compaction or reconnect error. For xDS, record response nonce, ACK or NACK, accepted version, and active host set. These are the discriminating fields. "Control plane healthy" is too vague to close the investigation.

| Observation at affected caller | Next measurement | Likely boundary | Source |
| --- | --- | --- | --- |
| No endpoints | Query authority and inspect consumer error or NACK | Publication or acceptance | [Envoy xDS](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol) |
| Old endpoints | Compare last accepted version, TTL, revision, and update age | Distribution or cache | [etcd API](https://etcd.io/docs/v3.6/learning/api/), [Consul consistency](https://developer.hashicorp.com/consul/api-docs/features/consistency) |
| Current endpoints, connect failure | Direct connect from caller network and route inspection | Data plane | Derived diagnostic procedure |
| Connect works, request fails | Send the real request shape and inspect proxy or app response | Backend or request policy | Derived diagnostic procedure |
| One zone fails while another works | Compare views and direct reachability by zone | Local partition or local health observation | [Envoy outlier detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier) |

The read-only command for a production host should identify the resolver before asking it questions. `cat /etc/resolv.conf` may be only one part of the runtime's behavior; a container, JVM, or proxy may use its own cache. `ss -tn dst <backend-address>` can show whether traffic still uses an existing socket. A registry API query should include status headers, not only the JSON body. A proxy's active configuration matters more than its management server's desired configuration. The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) ties this boundary back to route, socket, and packet evidence across the full request path.

### What to change after locating the boundary

If the control plane is healthy and only one consumer is old, repair that consumer's subscription or cache, with a controlled reload if necessary. If many consumers show the same old version, inspect publication and management-server load. If the authority is partitioned, choose the documented stale-use policy instead of repeatedly forcing reads that cannot succeed. If data paths fail with a current endpoint set, fix routing, listener, or backend readiness before changing discovery TTLs. A TTL change cannot open a blocked TCP route.

Avoid converting all health failures into global deregistration. One proxy's passive ejection can be a local response to local packet loss. A registry-wide removal changes every caller's view. In a zonal partition those have different blast radii. The [load-balancing post](/blog/software-development/system-design/load-balancing-from-l4-to-l7) discusses where algorithms sit in the architecture. The wire-level test here is simpler: which endpoint did this particular caller select, and could a SYN and an application request complete from its location?

### A small read-only production runbook

Start on the affected caller, not on a random healthy workstation. Record its source network, container identity, resolver configuration, and proxy process. From that same context, resolve the service name and query the discovery API if one is exposed. Do not paste an access token into a shared terminal transcript; use the existing authenticated client and retain only the response fields needed to establish state. The result should include the endpoint addresses and ports, the read mode or resource version where supported, and the time of the query. Compare it with the authority's desired set. A discrepancy is evidence about update distribution; identical sets move the investigation toward routing and application readiness.

Next, attempt a connection to each selected endpoint from that caller. A successful TCP connection proves only that the handshake completed. Send the smallest safe representative application request and record its status and timing. If a sidecar is present, compare a request through the sidecar with a direct request to the backend. If the direct path succeeds while the sidecar reports no healthy upstream, inspect the sidecar's active host set, local outlier ejections, and last xDS acceptance. If both paths fail, examine the route and listener before rewriting discovery. A capture can confirm SYN retransmissions or resets, but capture only the relevant host and port and treat payloads as sensitive.

Then look backward through time. Identify the last accepted endpoint update at the consumer, the last successful publication at the control plane, and the backend's actual readiness transition. If the consumer stopped advancing after a watch reconnect, check for compaction and a missing resnapshot. If Envoy NACKed a version, read `error_detail` and the previously accepted version. If a Consul read is stale, inspect last-contact information and the selected consistency mode. If the application still chooses an old address despite a current DNS answer, inspect its resolver cache and connection pool. Each branch has a concrete next measurement. None requires a blind restart as the first move.

Only after the first divergence is known should the team change a TTL, health threshold, watch consumer, or proxy rollout. Change one boundary at a time and record the predicted observation. For example, shortening an active probe interval should shorten detection time but increase check traffic. It should not change the time for an already accepted xDS resource to be applied. Flushing an application cache should change the chosen endpoint if the authority is current; it should not repair a blocked route. This prediction gives the on-call engineer a falsifiable test and prevents a discovery incident from becoming an unplanned series of control-plane mutations.

## Run it yourself

### Question

Can a caller continue to send data-plane requests to a previously learned endpoint while its control-plane path is blocked, and can that caller learn a new endpoint during the block? This small experiment models a cached endpoint list. It does not implement Consul or xDS, and it makes no claim about their performance.

### Preconditions

Use the Linux namespace setup from [post 1](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url): `c` at `10.77.0.1` on `c0`, `s` at `10.77.0.2` on `s0`. Run in a disposable Linux VM or a container with network namespace and packet-filter privileges. macOS needs a Linux VM as described there. The commands below need `ip`, `python3`, `curl`, `iptables`, `ss`, and root or equivalent capabilities. `netserver` should already be serving `10.77.0.2:8080`; if not, start it with the stable series command shown below. Do not apply the `iptables` rule to a production interface. This experiment adds only a rule in namespace `c` for TCP port `9090` and removes that exact rule in Reset.

```bash
set -euo pipefail
sudo ip netns list
sudo ip netns exec c ip -br addr show c0
sudo ip netns exec s ip -br addr show s0
sudo ip netns exec c ip route get 10.77.0.2
sudo ip netns exec c ss -tn
sudo ip netns exec c iptables -S OUTPUT
command -v python3 curl iptables ss
python3 --version
curl --version | head -n 1
```

Read: `c0` should carry `10.77.0.1/30`, `s0` should carry `10.77.0.2/30`, and the route should leave through `c0`. A qdisc check is useful if a prior lab added impairment: `sudo ip netns exec c tc qdisc show dev c0`. This lab expects no special loss or delay rule. Run `sudo ip netns exec s ss -lnt` and confirm `10.77.0.2:8080` is listening. If it is not, launch `netserver --listen 10.77.0.2:8080` in namespace `s` in a separate terminal.

### Baseline

Create an experiment-owned catalog file. The Python HTTP server is a deliberately simple control plane. The existing `netserver` on port `8080` is the data plane. Use a temporary directory created for this lab; the commands print its path and PID so cleanup is exact.

```bash
set -euo pipefail
LAB_DIR=$(mktemp -d /tmp/service-discovery-netlab.XXXXXX)
printf '{"service":"netserver","endpoint":"10.77.0.2:8080","generation":1}\n' > "$LAB_DIR/endpoints.json"
sudo ip netns exec s python3 -m http.server 9090 --bind 10.77.0.2 --directory "$LAB_DIR" > "$LAB_DIR/catalog.log" 2>&1 &
CATALOG_PID=$!
printf 'LAB_DIR=%s CATALOG_PID=%s\n' "$LAB_DIR" "$CATALOG_PID"
sleep 1
sudo ip netns exec c curl --fail --silent --show-error http://10.77.0.2:9090/endpoints.json
sudo ip netns exec c curl --fail --silent --show-error --output /dev/null --write-out 'data_http=%{http_code} connect_s=%{time_connect}\n' http://10.77.0.2:8080/echo
```

Read: the JSON `endpoint` is `10.77.0.2:8080` and `generation` is `1`. The second command's `data_http` should be a successful response code supported by the base `netserver` and `connect_s` should be a small local-VM duration. If the base server uses a different response for `/echo`, use its known endpoint and record the HTTP code rather than assuming a particular body. On an unimpaired local veth pair, `time_connect` is normally well below one second, but treat this as a qualitative reachability check, not a benchmark.

### Apply one change

Block only the control-plane port from namespace `c`. Keep the data-plane port untouched. Then change the catalog file to represent a new generation. This is a controlled partition of one port, not a simulation of quorum loss or Consul internals.

```bash
set -euo pipefail
sudo ip netns exec c iptables -I OUTPUT 1 -d 10.77.0.2 -p tcp --dport 9090 -j REJECT
printf '{"service":"netserver","endpoint":"10.77.0.2:8080","generation":2}\n' > "$LAB_DIR/endpoints.json"
sudo ip netns exec c iptables -S OUTPUT | head -n 4
```

### Compare

```bash
set -euo pipefail
sudo ip netns exec c curl --silent --show-error --max-time 2 --output /dev/null --write-out 'control_http=%{http_code} exit=%{exitcode}\n' http://10.77.0.2:9090/endpoints.json || true
sudo ip netns exec c curl --fail --silent --show-error --output /dev/null --write-out 'data_http=%{http_code} connect_s=%{time_connect}\n' http://10.77.0.2:8080/echo
sudo ip netns exec s curl --fail --silent --show-error http://10.77.0.2:9090/endpoints.json
```

Read: the control-plane request from `c` should have a nonzero curl exit status and no successful HTTP status, while the data-plane request still returns the same successful class of response as baseline. The catalog queried locally in `s` should show `generation:2`, proving the authoritative file changed although `c` could not retrieve it. `REJECT` commonly fails immediately; a `DROP` rule would instead wait for the client timeout. Scheduler and VM variance affect `connect_s`; the expected qualitative result is control request failure alongside data request success. A caller retaining generation 1 can keep using the old endpoint here because the endpoint did not change. If that address were retired, the same stale behavior could become wrong. That is the policy trade-off, not a universal endorsement of stale answers.

### Reset

```bash
set -euo pipefail
sudo ip netns exec c iptables -D OUTPUT -d 10.77.0.2 -p tcp --dport 9090 -j REJECT
sudo kill "$CATALOG_PID"
rm -rf "$LAB_DIR"
sudo ip netns exec c iptables -S OUTPUT | head -n 4
```

The rule deletion is scoped to the destination and port this experiment added. The temporary catalog server and directory are scoped by the printed PID and path. If the shell was restarted, locate the exact PID with `sudo ip netns exec s ss -lntp '( sport = :9090 )'` before stopping it. A safe production translation is read-only: query the resolver or registry from the affected caller, inspect the accepted proxy version, then make a direct request to a known backend from the same source network. Do not install packet filters, delete routes, or restart a registry merely to perform this diagnosis. If capturing packets, scope the filter to the affected name-server or registry connection and remember that application captures can contain credentials or payloads.

## Key takeaways

- Discovery produces a local endpoint view. Diagnose the view at the caller that failed, not only the authority that published it.
- Static files, DNS, registries, and xDS change how updates travel. None removes the need for a policy when updates stop.
- Active probes and passive request outcomes observe different paths. A healthy check and a failed user call can both be true.
- A watch needs a snapshot, ordering point, gap handling, and resynchronization. An open stream alone is not proof of a current view.
- xDS ACK and NACK make configuration acceptance visible, but acceptance is not proof of data-plane reachability.
- Under a partition, last-known-good availability and fresh-membership correctness trade against each other. Set a per-service stale-use budget and enforce it where traffic is selected.
- Roblox's 2021 outage shows that the discovery system itself can become the critical dependency, and that restoring it may be only the first stage of recovery.

## Further reading

- [Roblox Return to Service, January 2022](https://about.roblox.com/newsroom/2022/01/roblox-return-to-service-10-28-10-31-2021), the incident owner's detailed account.
- [Consul consistency modes](https://developer.hashicorp.com/consul/api-docs/features/consistency) and [Consul DNS configuration](https://developer.hashicorp.com/consul/docs/discover/dns/configure), for read freshness and cache behavior.
- [etcd v3 API](https://etcd.io/docs/v3.6/learning/api/), for Range, revision, Watch, and compaction semantics.
- [Envoy xDS protocol](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol), for version, nonce, ACK, NACK, and resource distribution.
