---
title: "DNS in Production: TTLs, Caching Layers, and Stale Answers"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Trace a stale address through application, host, and recursive caches, then measure the real recovery time of a DNS change."
tags:
  [
    "networking",
    "distributed-systems",
    "dns",
    "dns-caching",
    "ttl",
    "service-discovery",
    "failover",
    "incident-response",
    "observability",
    "split-horizon",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-1.webp"
---

The authoritative DNS record now points at the replacement endpoint. The status dashboard is green. One application process still calls the old address, while a new process on the same machine connects to the new one. A shell query says the record has a short time to live. Waiting that many seconds changes nothing. This is a familiar incident because the TTL on a DNS answer is only one timer in a much longer chain.

The latency ladder below places name resolution before connection establishment. The diagram is our opening mental model: a cache hit can make the DNS rung almost invisible, while a stale hit can make the remainder of the request fail at TCP or TLS. The visible error may therefore appear *after* the layer that made the wrong choice. We will follow a name from the application to its authoritative owner, distinguish actual DNS caches from connection reuse, and calculate recovery as a bounded sequence of events rather than as a slogan about TTL.

![Latency ladder highlighting the DNS decision before connection setup](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-1.webp)

The same request path is introduced in [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). Here we stay at the name-to-address decision. The sibling post on [DNS as a distributed database](/blog/software-development/networking/dns-the-distributed-database-your-request-starts-in) owns the recursive walk through root, top-level-domain, and authoritative servers. Our concern is what happens after a valid answer exists and a production client continues to act on yesterday's answer.

## The question is which component owns the address

**Senior rule: first identify who made the connection decision.** A fresh `dig` result proves what one DNS server returned to one query. It does not prove what an existing application process used for a socket. We need to identify the exact process, address family, name, resolver route, and time of the connection attempt.

A hostname is a lookup input. An IP address is the output a caller may use for a later `connect()`. Once the socket has an address, changing DNS cannot redirect that socket. A persistent HTTP connection, database session, or gRPC channel can continue talking to the old endpoint without issuing another query. This behavior is often called a DNS cache during an incident, but it is connection reuse. Killing a DNS cache will not close those sockets. Conversely, forcing reconnection can expose a stale application address cache that was hidden while the old connections remained healthy.

There are two clocks worth keeping separate. The DNS record's TTL is a duration attached to an RRset, a set of records of one name, type, and class. The client process also has a clock for how long it retains an address, if it retains one at all. The connection pool has a third clock for how long it retains established sockets. The pool's lifetime is not conveyed in DNS packets. [RFC 1035, November 1987](https://www.rfc-editor.org/rfc/rfc1035.html) defines the wire TTL in seconds. [RFC 2181, July 1997](https://www.rfc-editor.org/rfc/rfc2181.html) clarifies that records in one RRset should share a TTL and that implementations may cap unusually large received TTLs. Neither standard makes an application refresh an address on every request.

A common faulty experiment is `dig api.example.com`, followed by a claim that the application must already know the new address. `dig` is its own DNS client. It normally sends a DNS query according to its chosen server and options. The application might use `getaddrinfo()`, Java's `InetAddress`, an internal asynchronous resolver, an HTTP client with a pooled connection, or a service-discovery sidecar. Those routes need separate evidence. Even `getent ahosts` and `dig` can disagree without either command being broken, because `getent` follows the host's name-service switch policy while `dig` asks DNS directly.

Before touching a production cache, record the symptom in terms of endpoints. Ask the application to log the hostname and selected remote IP for a failed attempt. On Linux, `ss -tnp` can show established remote socket addresses and owning processes when permissions permit. Ask the authoritative server for the RRset, ask the configured recursive server separately, and compare from the same network view as the process. If those two DNS answers match but the process uses something else, focus on the process or an existing connection. If the recursive answer differs, look downstream at caching, forwarding, policy, and reachability.

| Observation | Most useful next measurement | What it can establish | Source |
| --- | --- | --- | --- |
| `dig` shows new IP, existing process connects to old IP | Log the process's lookup result and inspect `ss -tnp` | Distinguishes address reuse from a persistent socket | Derived here from the lookup and `connect()` boundary |
| Direct authoritative answer is new, configured resolver answer is old | Query both servers with `dig @server name A` and inspect TTL | Locates stale answer at or beyond the recursive path | [RFC 1035, November 1987](https://www.rfc-editor.org/rfc/rfc1035.html) |
| Two sites receive different answers at the same time | Query each site's configured resolver and compare authority or policy | Suggests intentional DNS views or propagation discrepancy | [BIND 9 Administrator Reference Manual, views](https://bind9.readthedocs.io/en/latest/reference.html#view-statement-definition-and-usage) |
| New process works, existing process fails | Compare chosen remote IP and pool lifetime | Points toward process memory or old sockets | Derived here from process lifetime |

Notice what the table does not do. It does not use `ping` as a service-health test. An address can answer ICMP while its HTTPS path is unavailable, and some healthy services never answer ICMP. A DNS answer can be syntactically correct while the selected endpoint is wrong for this client. Diagnosis needs both the control-plane answer and a connection attempt to the returned address.

## Five places an answer can persist

**Senior rule: enumerate caches by owner, not by how often a shell query changes.** There is no universal five-stage DNS stack installed on every machine. The useful model is five possible ownership boundaries between a caller and authoritative DNS: application memory, runtime address caching, a local host cache, a recursive resolver, and an upstream forwarder. Some paths skip a boundary. Some add multiple proxies. The diagram marks them as potential state, not a promise that every lookup visits all five.

![Five possible cache boundaries from application decision to authoritative DNS](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-2.webp)

**Application memory** is the broadest boundary. A service can resolve a hostname during startup and retain the resulting address in a configuration object, client constructor, connection target, or custom map. The lifetime may be a deployment lifetime. A hand-written cache might honor the DNS TTL, impose a shorter cap, impose a longer floor, refresh asynchronously, or never refresh. None of those policies can be inferred from an authoritative response. In a runbook, find the call that converts the name to an address and the call that later creates the socket. If the name is converted only at startup, the deploy or restart procedure is part of DNS failover.

**Runtime caching** is distinct from an application's own map. Java provides the canonical example because `InetAddress` caches successful and unsuccessful name resolutions inside the runtime. Oracle's [Java networking properties documentation](https://docs.oracle.com/en/java/javase/15/docs/api/java.base/java/net/doc-files/net-properties.html) describes the security properties `networkaddress.cache.ttl` and `networkaddress.cache.negative.ttl`. A negative value means indefinite caching, zero disables that cache, and positive values are durations in seconds. The documented default for successful lookups depends on the runtime's security configuration and implementation. Never copy a claimed universal Java default from a decade-old incident note into a current runbook. Read the property in the actual runtime, and account for the HTTP client's own connection pool separately. Oracle also states these are security properties rather than ordinary `-D` system properties, an operational distinction that has caused ineffective fixes.

**Local host caching** is conditional. The glibc name-service functions route lookups according to `/etc/nsswitch.conf`; a plain glibc resolver call is not, by itself, proof of a persistent process-wide DNS cache. A host may use `nscd`, `systemd-resolved`, `dnsmasq`, or another local service. `systemd-resolved` explicitly provides a caching stub resolver and can be reached through its APIs or through the host's resolver configuration, as its [project documentation](https://wiki.freedesktop.org/www/Software/systemd/resolved/) explains. An application that bypasses the host resolver, including some language libraries, may skip this local cache. Inspect `getent hosts name`, `/etc/nsswitch.conf`, `/etc/resolv.conf`, and `resolvectl status` where available before assuming the path.

**Recursive resolver caching** is the shared state most operators first think of as DNS caching. A recursive server receives a query, uses its own cached RRsets if valid, or resolves upstream and retains the result for its TTL subject to its policy. A response TTL from a cache is normally the remaining lifetime, not the original configured lifetime. Querying twice and seeing the TTL count down is evidence of caching at that server. A TTL that appears to reset may mean the cache refreshed, the queries hit different resolver instances, or the reply was synthesized by policy. You need resolver identity and logs to choose between them.

**Forwarder caching** exists when a local recursive or stub server forwards requests to another shared DNS service. The forwarder may independently cache or serve policy answers. Enterprise networks, VPNs, cloud VPC resolvers, and Kubernetes clusters commonly make the route less direct than one stub to one authoritative server. We should not presume that every forwarder caches, or that two forwarders share one cache. Capture the configured upstream addresses and query a specific server with `dig @address`, rather than treating a load-balanced resolver name as one stateful box. The [DNS resolution walk](/blog/software-development/networking/dns-the-distributed-database-your-request-starts-in) describes what the recursive service eventually does when it has no usable result.

This five-place model is a diagnostic map, not an additive TTL formula. A process cache can outlive a resolver's TTL because it does not consult the resolver again. A host cache may receive a newly refreshed answer while a process remains stale. A forwarder may not be on the path at all. If several serial caches honor decreasing wire TTLs exactly, their expiry windows overlap rather than simply adding their original TTLs. If each layer reassigns its own fixed lifetime, their windows can extend past the authority's intended lifetime. We must inspect actual policies before calculating a deadline.

### Negative answers have a separate timer

A missing name can be sticky too. NXDOMAIN means that a queried name does not exist. NODATA means the name exists but has no RRset of the requested type. [RFC 2308, March 1998](https://www.rfc-editor.org/rfc/rfc2308.html) defines caching of these negative answers using the SOA record in the authority section. Its negative caching TTL is determined from the lesser of the SOA record TTL and its MINIMUM field when the authoritative server constructs the response. A newly created name can therefore remain absent at a caching resolver after the record is published. Asking only for an A record can also hide the fact that an AAAA question had a different answer or negative cache key.

The operational lesson is not to set every TTL to zero. It is to distinguish a positive stale address from a cached negative result. Use `dig name A` and `dig name AAAA` against the same resolver; inspect `status`, `ANSWER`, `AUTHORITY`, and the SOA TTL. If an app reports `UnknownHostException` while `dig` now succeeds, Java's negative address cache is another candidate. A failed lookup and a failed TCP connection are not the same class of event, even when the product reports both as "service unavailable."

### Serve-stale is intentional, bounded staleness

A short authoritative TTL does not always imply that an expired RRset disappears immediately. [RFC 8767, March 2020](https://www.rfc-editor.org/rfc/rfc8767.html) defines a serve-stale mode in which a recursive resolver may use expired data when it cannot refresh from authority. The design trades freshness for continued resolution during an authoritative outage. It recommends a positive TTL of thirty seconds on the stale response and a configurable maximum stale retention. That thirty-second TTL on the client-facing answer is not evidence that the authoritative zone recently published it. It may be a stale answer deliberately served after a failed refresh.

This exception matters during failover. If the old address has become unsafe or unreachable and the authority is unavailable, stale serving can keep sending clients toward it. If the authority is temporarily unreachable but the old endpoint remains healthy, stale serving can prevent a wider outage. Neither outcome can be judged from TTL alone. Inspect resolver serve-stale settings, refresh failure metrics, and authoritative reachability. Treat a stale answer as a policy decision with an operational reason, not as a mysterious violation of the DNS standard.

## CNAME chains give one lookup several lifetimes

**Senior rule: do not read one TTL from a multi-record answer and call it the lifetime of the name.** A CNAME record maps an alias to another name. An A record maps a name to an IPv4 address, and an AAAA record maps a name to an IPv6 address. A client asking for the address of an alias may receive an alias record and a target address in one response, but these are separate RRsets. They can have different TTLs, separate authoritative owners, and separate cache histories.

![CNAME and target address record comparison with independent expiration](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-3.webp)

Consider a controlled example, not a claim about a public service. Suppose `api.example.test` has a CNAME to `blue.example.test` with a 300-second TTL. The target has an A record with a 30-second TTL. At time zero a resolver obtains both. Thirty seconds later it may need to refresh the target address while retaining the alias. It need not ask for the alias again until its own TTL expires. If the operator changes only the CNAME target to `green.example.test`, a resolver that still has the alias can keep following `blue.example.test` until the alias RRset expires, even if that target's A RRset refreshes repeatedly. The short A TTL did not make the alias switch fast.

Reverse the numbers. Give the alias a 30-second TTL and target A a 300-second TTL. After the alias expires, a fresh alias might point somewhere else, but a client that directly retains the old target address or a process that cached the final resolved IP does not necessarily care. These examples show why CNAME changes and address changes have different recovery envelopes. The simple derived model for a resolver that independently honors both TTLs is: changing the alias is gated by the alias's *remaining* lifetime, while changing the target address is gated by the target address RRset's remaining lifetime. A chain with more aliases creates more independent choices. This is an explanatory model for a particular caching behavior, not a DNS protocol equation.

[RFC 1034, November 1987](https://www.rfc-editor.org/rfc/rfc1034.html) specifies how name-server processing follows an alias when a query asks for another type. [RFC 2181, July 1997](https://www.rfc-editor.org/rfc/rfc2181.html) clarifies the CNAME alias's relationship to other records and the treatment of RRsets. The exact response content depends on resolver state and authoritative boundaries. One packet can include a whole chain, part of a chain, or a CNAME that causes a subsequent query. Do not use answer-section order as a guarantee that all records were fetched at the same instant.

A useful `dig` session asks for each link separately:

```bash
dig @192.0.2.53 api.example.test CNAME +noall +answer
dig @192.0.2.53 blue.example.test A +noall +answer
dig @192.0.2.53 blue.example.test AAAA +noall +answer
```

The address `192.0.2.53` is a documentation placeholder in this explanatory snippet, not a server the reader should expect to reach. Replace it with the resolver from the affected host. Read the owner name, type, TTL, and data on each line. Then query the authoritative servers for the alias zone and the target zone. If the target belongs to another provider, the other provider owns the target's address policy. A low TTL on your alias gives you control over when clients reconsider the target name, but not over how that target's own address data is maintained.

There is a further diagnostic trap with CNAMEs during outages. A negative answer can attach to the final name in a chain. A response for the original alias may then contain a valid CNAME followed by an NXDOMAIN or NODATA condition at the target. Treating the first record as proof that the service exists is wrong. Inspect the final queried name, response status, SOA, and A/AAAA result. [RFC 2308, March 1998](https://www.rfc-editor.org/rfc/rfc2308.html) calls out CNAME chains when defining the queried name for negative caching. During a migration, create and verify the target before publishing the alias to it. The order of operations is part of availability.

## DNS failover has a recovery path, not a single TTL

**Senior rule: publish time and client recovery time are different measurements.** DNS-based failover is attractive because it works for clients that already know the service hostname. It can also be disappointingly slow because changing an RRset is only one step between detecting a failure and establishing a useful new connection. The timeline below separates those steps.

![Failover timeline from detection through DNS publication and client reconnect](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-4.webp)

<figure class="blog-anim">
<svg viewBox="0 0 760 220" role="img" aria-label="Authoritative answer changes immediately while a cached answer remains old until its remaining TTL expires" style="width:100%;height:auto;max-width:860px">
<style>
.dns14-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.dns14-text{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dns14-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dns14-line{stroke:var(--border,#d1d5db);stroke-width:4}
.dns14-clock{fill:var(--accent,#6366f1)}
@keyframes dns14-sweep{0%{transform:translateX(0)}100%{transform:translateX(600px)}}
@keyframes dns14-old{0%,69%{opacity:1}72%,100%{opacity:0}}
@keyframes dns14-new{0%,69%{opacity:0}72%,100%{opacity:1}}
.dns14-moving{animation:dns14-sweep 10s linear infinite}
.dns14-old{animation:dns14-old 10s linear infinite}
.dns14-new{animation:dns14-new 10s linear infinite}
@media (prefers-reduced-motion:reduce){.dns14-moving,.dns14-old,.dns14-new{animation:none}.dns14-moving{transform:translateX(600px)}.dns14-old{opacity:0}.dns14-new{opacity:1}}
</style>
<rect class="dns14-box" x="25" y="20" width="320" height="90" rx="10"/>
<rect class="dns14-box" x="415" y="20" width="320" height="90" rx="10"/>
<text class="dns14-small" x="185" y="48">authoritative answer</text>
<text class="dns14-text" x="185" y="79">new address at t = 0</text>
<text class="dns14-small" x="575" y="48">recursive cache answer</text>
<text class="dns14-text dns14-old" x="575" y="79">old address</text>
<text class="dns14-text dns14-new" x="575" y="79">new address</text>
<line class="dns14-line" x1="80" y1="161" x2="680" y2="161"/>
<circle class="dns14-clock dns14-moving" cx="80" cy="161" r="11"/>
<text class="dns14-small" x="80" y="197">change</text>
<text class="dns14-small" x="500" y="197">remaining TTL</text>
<text class="dns14-small" x="680" y="197">refresh</text>
</svg>
<figcaption>The authority changes first; a compliant recursive cache can keep the old answer until its remaining TTL expires, then refresh.</figcaption>
</figure>

A useful explanatory model is

$$
T_{\mathrm{usable}} = T_{\mathrm{detect}} + T_{\mathrm{publish}} + T_{\mathrm{observe}} + T_{\mathrm{reconnect}} + T_{\mathrm{ready}}.
$$

Here `detect` is the elapsed time until the failure detector declares the old endpoint unhealthy. `publish` includes control-plane work until the authoritative servers return the intended answer. `observe` covers the remaining usable cache lifetimes and the caller's next lookup opportunity. `reconnect` covers closure or failure of old sockets and a fresh connection attempt. `ready` is time until the replacement endpoint can actually serve the request, including any health or state dependency. This sum is an operational accounting model, not an equation in an RFC. Some steps overlap, so a measured timeline can be shorter than a naive sum, but no one should promise recovery by citing only a record TTL.

Take an illustrative, derived scenario. A monitor polls every 10 seconds and requires three consecutive failed polls. If the endpoint fails just after a healthy poll and each poll completes immediately, declaration takes about 30 seconds; if the first failed poll happens almost immediately, it takes about 20 seconds. Suppose publication takes 5 seconds, a recursive cache has up to 60 seconds remaining on its positive answer, and a client refreshes its address every 120 seconds independently of DNS. Ignoring overlap and assuming it attempts a new connection immediately after refreshing, a conservative *illustrative* upper calculation is roughly `30 + 5 + 60 + 120 = 215 seconds`, before endpoint readiness. That is not a universal bound: a client could retain an address indefinitely, an existing socket could remain open, serve-stale could extend resolver behavior, and detector or publication failures may have no fixed bound. The point of the arithmetic is to force each timer onto a page where it can be measured.

The same scenario also has an optimistic path. A client with no address cache and no persistent connection may query a resolver that happens to expire just after publication. Its DNS observation delay can approach zero. Some users recover quickly while others wait minutes. A median lookup time can look normal throughout. To describe a failover promise, measure the distribution of client recovery by runtime, resolver population, network, and connection strategy. State which percentile the promise refers to and which failures it excludes.

| Stage | Controlled example | How to verify | Source |
| --- | ---: | --- | --- |
| Detection | 20–30 s for three immediate failed polls at 10 s spacing | Monitor event timestamps | Derived here from polling assumptions |
| Publication | 5 s assumed | Query each authoritative server directly | Illustrative assumption, not a measured provider limit |
| Recursive cache | 0–60 s remaining | Query configured resolver repeatedly and read answer TTL | Derived here for a 60 s positive TTL; [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html) |
| Client address memory | Up to 120 s in this example | Instrument actual lookup calls and chosen remote IP | Illustrative application policy |
| Existing connection | Unbounded without an application policy | Inspect socket remote IP and pool retirement behavior | Derived here from socket lifetime |

What can we control? Lowering a positive TTL *before* a planned migration can reduce a resolver's remaining lifetime after caches have already adopted the lower value. Lowering it at the moment of an incident does not rewrite answers already cached with the former TTL. A shorter TTL increases refresh traffic and dependence on the authoritative service. It may raise cold lookup latency if caches miss more often. It does not fix an indefinitely cached application address. Plan a connection-draining mechanism, expose the selected remote IP in logs, and rehearse a migration with a representative long-lived client.

Failures also come in different directions. DNS can return an old but reachable endpoint while the application wants the new one. It can return a new endpoint before that endpoint is ready. It can return NXDOMAIN because a record disappeared briefly. A resolver can time out because authority is unreachable, or serve stale data under [RFC 8767](https://www.rfc-editor.org/rfc/rfc8767.html). These are not interchangeable. The first calls for cache and connection inspection. The second calls for readiness and rollout sequencing. The third calls for negative-cache inspection. The fourth calls for authoritative reachability and resolver policy. A single "DNS failed" counter loses the distinction required to choose a safe action.

### Why a low TTL can make the incident worse

A shorter TTL asks caches to refresh more often. In healthy operation, that can move clients to a new address sooner. During an authoritative failure, it can move more clients from a usable cached answer into refresh attempts. Serve-stale changes this trade-off by retaining a previously known answer, but only when configured and when its conditions apply. A short TTL is therefore a policy choice about freshness, query load, and dependence on the DNS control plane. It is not an availability guarantee by itself.

A second-order failure appears when traffic moves faster than the replacement system can accept it. DNS can change where new connections go, but existing connections to the old site may drain slowly. If the old site fails abruptly, many clients may reconnect together. That can overload TLS termination, load balancers, or backend pools. The [TCP handshake cost](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you) becomes visible as a burst of SYNs and connection establishment. The [load-balancing design discussion](/blog/software-development/system-design/load-balancing-from-l4-to-l7) covers how a balancing tier allocates traffic once the connection reaches it. DNS chooses an earlier boundary: the destination to contact at all.

## Split-horizon: different answers can both be current

**Senior rule: always name the resolver and network view.** Split-horizon DNS means a name may return different data depending on the querying client's source, resolver, interface, policy, or view. An internal service name might resolve to a private address inside a corporate network and to no public address outside it. A public hostname might resolve to a private endpoint from a VPC and a public edge address from the internet. Neither answer is necessarily stale. [BIND 9's view mechanism](https://bind9.readthedocs.io/en/latest/reference.html#view-statement-definition-and-usage) is one documented implementation of client-dependent responses.

![Decision tree for distinguishing split DNS views from stale cached answers](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-6.webp)

An incident chat often compares the wrong measurements. One engineer runs `dig` on a laptop behind a VPN. Another runs it from a Kubernetes pod. A third asks a public resolver from home. Their outputs differ, so the team declares propagation broken. Before changing records, record the client's network namespace or host, `/etc/resolv.conf`, the exact resolver address, query type, fully qualified name, and whether a VPN or search suffix is active. Compare authoritative answers for the relevant view, not one public answer treated as the universal truth.

The shortest useful test matrix has the affected process environment on one axis and the server queried on the other. From each environment, query its configured resolver and, where policy allows, the relevant authoritative server. If a private authoritative view is reachable only inside a network, a public vantage point cannot directly verify it. A private answer appearing only in one view is expected. The anomaly is a process receiving an answer inconsistent with its intended view or retaining an old answer after that view has changed.

`dig +short` is convenient for a quick glance but poor evidence in this situation. It discards header flags, status, TTLs, and authority information. Use `+noall +answer +authority +comments` for a compact record that preserves the information needed to distinguish an answer from a cached denial. Query A and AAAA separately. If the application uses a short name, include the search-domain expansion as an explicit fully qualified query. The [Kubernetes DNS post](/blog/software-development/networking/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout) owns the `ndots` and search-list multiplication that can make one apparent lookup into several wire questions.

Split-horizon also affects disaster recovery. If only the public view changes during failover, internal callers may continue to use the private address. If internal DNS changes first, external users may still reach the old edge. This is not automatically a problem; it can be an intentional staged rollout. It becomes a problem when the rollout plan says "change the DNS record" without naming every view and proving each with an explicit query. A safe deployment checklist identifies zone, view, record type, authoritative servers, and expected answer before any mutation. A safe incident record captures the same fields after a mutation. The [service-discovery and load-balancing overview](/blog/software-development/microservices/service-discovery-and-load-balancing) discusses how clients use discovered endpoints above this wire-level decision.

## Two 2021 incidents that locate the DNS dependency

**Senior rule: use an incident to identify a failure boundary, not to smuggle in a claim about cache behavior the report did not measure.** The two cases below both affected name-dependent availability, but the public statements do not provide per-client TTL distributions. They cannot prove that a particular application cache caused or prolonged the incidents. Their value is the operational lesson that DNS is a production service with a change path and its own failure modes.

![Case matrix contrasting Salesforce and Akamai DNS-related disruptions](/imgs/blogs/dns-in-production-ttls-caching-layers-and-stale-answers-5.webp)

### Salesforce, 11 May 2021: a DNS network incident after an emergency fix

[Salesforce's public update](https://www.salesforce.com/jp/news/stories/salesforce-update-21/) says that multiple services became unavailable on 11 May 2021 and that systems were restored on 12 May in Japan time. The company attributed the root cause to the implementation of an emergency fix that triggered a software problem and led to a DNS network incident. The published statement is the owner evidence for the trigger and affected layer. It does not publish enough detail to reconstruct each resolver, RRset, TTL, or client cache. We should not infer those from the word "DNS."

For this post, the important mechanism is the separation between a service's application health and the name-resolution path a user must traverse to reach it. A DNS network incident can prevent a caller from obtaining a usable destination before any TCP handshake or application request begins. If the incident also affects a status surface that relies on the same naming path, observers may lose their normal source of situational awareness. The general guardrail is to make authoritative DNS changes reviewable and reversible, and to maintain an independently reachable status route. That is a design inference from the dependency, not a claim that Salesforce used or lacked a particular guardrail.

The incident also cautions against assuming that lowering a customer-facing TTL would have solved it. The public statement identifies an emergency fix and software problem in the DNS network path. It does not say that user-side stale positive answers were the cause. In an analogous incident, the first discriminating test would be direct authoritative queries from multiple vantage points, followed by the configured recursive answers and process-selected remote IPs. A healthy answer at authority with a stale recursive result suggests one branch of diagnosis. A failure at authority suggests a different branch. The distinction is what lets an incident team choose a rollback or resolver mitigation with evidence.

### Akamai, 22 July 2021: a DNS component of secure edge delivery

[Akamai's July 22, 2021 incident statement, updated July 23](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved), reports that at 15:45 UTC a software configuration update triggered a bug in its Secure Edge Content Delivery Network and affected that network's DNS component. Some customer websites became unavailable; the disruption lasted up to an hour; a rollback restored service. Akamai explicitly clarified on July 23 that the impact was isolated to a DNS component of Secure Edge Content Delivery Network, rather than a general outage of its Edge DNS product. That scope correction matters. Calling this "Akamai Edge DNS went down" would be stronger than the owner report supports.

The trigger was a configuration update, the immediate failure was a software bug in a DNS component, the user-visible outcome was website unavailability, and the recovery action was rollback. The transferable control is to rehearse configuration rollback and monitor resolution through the same service path customers use. Direct authoritative answers, edge-specific names, and end-user connection attempts answer different questions; a healthy generic DNS probe cannot certify every product-specific mapping path. This incident supplies no public evidence that a CNAME TTL, Java address cache, or split-horizon view was decisive. The case belongs here because it shows the dependency at the DNS control boundary, not because every DNS caching mechanism appeared in one event.

Put the cases beside our failure model. Salesforce's statement points to an emergency change that produced a DNS network incident. Akamai's statement points to an update that exposed a software bug in a specific DNS component. Both show that names can be a common point of failure before application request handling. Neither establishes a universal recovery time. For any actual deployment, we still need timestamps for failure detection, authoritative publication, resolver observation, client reconnection, and useful service. Those are the terms in the explanatory recovery model, and each must be observed locally.

## A practical diagnostic order during an incident

Start with the exact hostname the application used. Avoid a human-friendly alias if code actually calls a different name. Record the query type and whether the process can select IPv6. An application may receive both A and AAAA results and try one first; one address family can fail while the other works. Record the remote IP from a failing connection. A DNS packet capture is optional at first because an existing connection or in-process cache can cause no DNS packet at all. Absence of packets is a useful clue only after the capture point and time window are known.

Next, get the authoritative answer for the exact RRset. `dig +trace` can help locate delegation from a public view, but a private view or corporate forwarding policy can make it the wrong path. Find the authoritative servers for the *affected view*. Ask each directly if reachable. Record answer, TTL, status, and whether the server is authoritative. If authoritative servers disagree, do not troubleshoot a client cache yet. Verify publication and zone distribution. If they agree, query the actual recursive resolver configured for the affected workload. Do not substitute your laptop's resolver.

Then test the process boundary. Restarting a production process to "clear DNS" before logging its selected address destroys the best evidence and can create a reconnect surge. Prefer a read-only diagnostic endpoint or targeted log field that reports the name, resolved addresses, selected address, resolver mode, and connection reuse state. If no such field exists, add it to the next release. A packet capture on the host can confirm actual DNS questions, but it cannot prove what a runtime cached internally. `ss -tnp` can reveal remote addresses for existing TCP sockets, subject to permissions. Compare new and old processes. If only old processes fail, ask whether their address cache or connection pool is tied to process lifetime.

Finally, measure the user's path. A name can resolve and TCP can connect while TLS validation fails because the replacement endpoint is not configured for the hostname. An HTTP success check should use the same `Host` header, SNI name, and authentication context as the real client where possible. The [service-to-service security post](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) owns the authentication and trust policy; here the point is simply that DNS recovery is not application recovery. The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) puts this diagnostic order back into the end-to-end path.

| If this evidence appears | Prefer this next action | Avoid assuming | Source |
| --- | --- | --- | --- |
| Authority new, recursive old with decreasing TTL | Wait or inspect resolver policy and stale mode | That the authoritative update failed | [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html), [RFC 8767](https://www.rfc-editor.org/rfc/rfc8767.html) |
| Authority and recursive new, process dials old IP | Inspect runtime cache and pool state | That another DNS publish will help | Derived here from the process boundary |
| Resolver returns NXDOMAIN with SOA | Inspect negative cache lifetime and record creation sequence | That positive TTL controls the failure | [RFC 2308](https://www.rfc-editor.org/rfc/rfc2308.html) |
| Answers differ by source network | Verify intended DNS view and resolver selection | That one answer must be stale | [BIND 9 views](https://bind9.readthedocs.io/en/latest/reference.html#view-statement-definition-and-usage) |
| Name resolves, TLS fails | Check SNI, certificate, and endpoint readiness | That DNS is still the only blocker | Derived here from the connection path |

Do not flush a shared recursive cache as a reflex. It changes many clients' behavior and may create a burst against authority. Do not lower production TTLs during an incident without understanding the old cached TTLs and query-load consequence. A targeted process restart can be reasonable after evidence confirms an unbounded process cache, but plan for simultaneous reconnection and scope it to affected clients. The [timeouts and retries discussion](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers the wider application policy; DNS only determines one destination decision in that retry loop.

## Make the cache lifetime observable

**Senior rule: a cache you cannot interrogate is part of the incident, not just part of the optimization.** It is reasonable for a production service to cache addresses. It is unreasonable to promise a short failover while hiding the chosen address, lookup time, and refresh policy. The minimal useful telemetry is deliberately small: the original hostname, query family, returned address set, selected remote IP, lookup outcome, time of last successful lookup, and whether the connection was new or reused. A log line for every request would be excessive. A sampled diagnostic event on lookup and a metric for connection failures by endpoint are usually enough to place the failure.

The refresh policy belongs in configuration and in a deployment test. For Java, read the effective security property in the running runtime and verify the HTTP client's behavior with a controlled address change. For a Go service, inspect whether code uses `net.Dialer` with a hostname at each new connection or resolves once and dials a retained IP. For a proxy, inspect whether its upstream cluster resolves names periodically, at startup, or when a connection fails. The name of the programming language does not settle the question because a library can wrap, replace, or bypass its default resolver. The test is behavioral: change a test authority, observe the next DNS query, observe the selected IP, and observe the next useful request.

One precise way to discuss a cache is to identify its **entry key**. A DNS RRset cache keys by name, type, and class, though implementations may add view and policy context. An application map might key only by hostname and accidentally combine A and AAAA results. A connection pool commonly keys by scheme, hostname, port, TLS settings, and transport options; it may reuse a socket whose original address no longer matches current DNS. A host service can cache a negative result separately from a positive one. If engineers say "clear the cache" without naming key and owner, they are not yet describing an operation that can be reviewed or scoped.

Timestamped evidence is much more useful than a single snapshot. Save a short sequence of answers from the affected resolver with their remaining TTLs. If TTL decreases from one query to the next and the old address remains, that supports a currently valid cached RRset. If the TTL repeatedly jumps back, a load-balanced resolver fleet or refresh operation may be involved. If a nominally expired answer reappears with a short positive TTL while authority is failing, check serve-stale. If the answer changes by source network but remains stable within each network, check DNS views. These observations are clues, not final proof; logs or configuration at the resolver settle the mechanism.

Use cautious language when assigning causality. A client's failed connection at `10:01:00` and a stale resolver answer at `10:02:00` do not prove the earlier connection used that answer. The process could have retained an IP, used another resolver, or reused a socket. Join events by process identity, hostname, selected remote IP, and time. During a live incident, adding those fields can be more valuable than lowering the TTL again. The [observability design post](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers the broader logging architecture. Here the required signal is narrow: make the name-to-address decision visible at the component that actually dials.

### A cache ledger for a planned change

Before a planned migration, write down each participating lifetime in a small ledger. Include the published A and AAAA TTLs, every CNAME link TTL, negative TTL, known resolver caps or floors, runtime address-cache policy, connection pool maximum age, and endpoint drain period. Mark an unknown as unknown, not as zero. A TTL value from `dig` against an authoritative server is only one row. If the service has internal and public views, repeat the ledger per view. If the endpoint has different IPv4 and IPv6 readiness, repeat the test by family.

For example, assume an illustrative service has a CNAME TTL of 300 seconds, target A TTL of 30 seconds, runtime positive cache of 60 seconds, and maximum pooled connection age of 600 seconds. A change to the target's A record could become visible to a recursive resolver after up to 30 seconds of remaining target TTL, but the runtime may not ask that resolver for up to another 60 seconds after its own last lookup. A fresh connection might therefore lag the authoritative change by roughly 90 seconds under a conservative serial assumption. Existing pooled connections may continue for their configured lifetime, up to 600 seconds from creation, even after the name resolves to the new address. If instead the change is to the CNAME link, its 300-second remaining TTL replaces the 30-second target TTL in the relevant branch of the calculation. These are derived examples, not a promised upper bound: scheduling, serve-stale, failures, and a pool that renews indefinitely change the result.

The calculation is more honest when we carry the *remaining* TTL at the change instant. If a resolver cached a 300-second CNAME 290 seconds earlier, it has about ten seconds left, not 300. If it cached the record one second earlier, it has almost the full window. Different recursive caches entered the cycle at different times, which explains staggered recovery even when all obey the same rule. For a preplanned migration, lower the TTL ahead of the cutover and wait at least the *old* TTL for prior cache entries to age out before relying on the lower value. Verify from representative recursive resolvers; do not assume a control-plane UI update instantly changes their existing entries.

| Migration element | Example policy | Failure if omitted | Source |
| --- | --- | --- | --- |
| Alias RRset | 300 s positive TTL | Alias change can lag target address refresh | Derived example using [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html) |
| Target A RRset | 30 s positive TTL | Target address update remains cached briefly | Derived example using [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html) |
| Runtime address cache | 60 s assumed | Process keeps a final IP after DNS has refreshed | Illustrative application policy |
| Connection pool | 600 s maximum assumed | Existing sockets can outlive every DNS cache in the ledger | Illustrative pool policy |
| Negative cache | Read SOA TTL and MINIMUM | A newly published name may still appear absent | [RFC 2308](https://www.rfc-editor.org/rfc/rfc2308.html) |

The ledger is also a rollback tool. If the new address is broken and the authority is changed back, the same caches can hold the bad address until their remaining TTLs expire. A rollback operation in the DNS control plane is not the same as rollback completion for clients. If the old endpoint was immediately destroyed, some clients have no viable address during that window. Keep the old endpoint serving through the observed drain interval where feasible, then remove it after measured connections and lookups have moved. This is the wire-level reason a safe cutover overlaps old and new capacity.

## Design the failure detector for what DNS can actually change

A DNS failover controller can publish a different destination. It cannot repair a client that never re-resolves, terminate a stuck socket in someone else's process, or make an unready replacement healthy. That boundary should shape the health check. A DNS controller that marks a site healthy because its nameserver responds is checking its own plumbing, not the application's usefulness. A controller that checks only TCP may miss TLS or HTTP failures. A deep request check may depend on a backend whose temporary error should not trigger a global traffic move. Pick a check that matches the failover objective and state what it does not cover.

Detection also needs hysteresis. A single failed probe can be a transient network loss. Waiting for several failures reduces false moves but lengthens real recovery. The earlier three-poll example makes the trade-off visible without claiming a universal optimum. If probes run every ten seconds and require three consecutive failures, detection is roughly twenty to thirty seconds under the stated immediate-response assumption. If each failed probe itself waits five seconds for a timeout, the elapsed time can be longer depending on whether polling is scheduled from start or completion. The controller's actual timestamps, not the configured interval alone, settle the detection delay.

After publication, verify from more than one resolver. The controller should confirm that each intended authoritative server returns the new RRset, then monitor representative recursive resolvers and client probes. Do not confuse a resolver's TTL countdown with evidence that its chosen IP is usable. Send a real request to the returned destination with the hostname preserved for TLS and HTTP. Record both the DNS answer and the request result. If the new site fails and the old one still works, a rollback may be safer than waiting for every cache to expire into a broken address. If both sites fail, repeated DNS switches add uncertainty without creating capacity.

These controls connect to service discovery more broadly. A registry or xDS control plane may push endpoint changes to clients, while DNS clients usually pull on lookup. Push can shorten observation time for connected clients, but only if the client uses and correctly processes updates. Pull can be resilient and simple, but its refresh behavior is often opaque. The [service discovery from DNS to registries to xDS](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds) post compares those mechanisms. The invariant for this post is narrower: name the component that holds an endpoint and measure when it replaces it.

### What a rehearsed failover should report

A useful rehearsal report does not end at "record changed." It gives a timeline with detection time, authoritative publish time, first recursive observation by each sampled resolver, first new connection by each client cohort, and first successful request to the replacement. It records the fraction of clients still using the old address after each interval and identifies whether those are stale lookups or old sockets. It states the exact DNS view and address family. It keeps synthetic probes separate from real traffic. This is how an asserted recovery objective becomes a measurable claim.

Run the rehearsal during a maintenance window with a reversible test name before relying on a production hostname. Keep the old and new endpoints simultaneously capable of serving the test request. Lower TTLs ahead of time if that is part of the plan, and verify the lower TTL from representative resolvers before cutover. Intentionally leave one long-lived client running to test the worst behavior. If it never moves, the missing step is a client refresh or connection-drain policy, not another authoritative DNS edit. A successful rehearsal should include a rollback and the time clients take to observe that rollback. The failure path is bidirectional.

The same report should call out resolver outages separately from endpoint outages. During an endpoint outage, fresh DNS answers may be needed to redirect clients. During a DNS authority outage, previously cached answers may preserve connectivity, especially with serve-stale, but new names or uncached clients can fail. A policy that improves one scenario can worsen the other. Lower TTLs help prompt endpoint movement in the first scenario; longer retention can help continuity in the second. No single TTL optimizes both. Choose from measured failure modes and document the trade-off.

### Why your diagnostic tool may take a different path

A DNS incident becomes confusing when the tool used to inspect it is treated as equivalent to the application. `dig` sends DNS questions and displays DNS records. `getent ahosts` asks the system's host database through name-service configuration. A Java process can use its own `InetAddress` cache. A Go process can use either its pure Go resolver or the platform resolver depending on build and environment, while an HTTP client may reuse a connection after either resolver has finished. A proxy can have its own upstream discovery policy. Therefore an agreement between two `dig` invocations is evidence about those two queries, not about the address used by every process.

Inspect the route in layers. Read `/etc/nsswitch.conf` for the `hosts:` line to see whether local files, DNS, or a resolver integration has priority. Read `/etc/hosts` because it can override or supplement DNS without creating a DNS packet. Read `/etc/resolv.conf` inside the actual container or namespace, not on the host that launches it. If `systemd-resolved` is active, inspect its status and per-link DNS configuration. A VPN can install a domain-specific route to a resolver without changing the answer a public resolver returns. A Kubernetes pod can have a cluster DNS service and search suffixes unlike the node. The exact path is deployment-specific; the command is useful only when it runs in the same network and name-service context as the failing workload.

Check whether a result is already an address literal. Many client libraries accept either a hostname or an IP in the same configuration field. If a service was configured with an IP after a prior emergency, no DNS change will move it. A log that prints only the original configuration key may conceal a separate resolved IP stored in memory. The decisive observation is the remote address in the actual socket. Conversely, a log line saying "resolved api.example.test" may be emitted once at startup, not once per connection. Include a timestamp and lookup count before trusting it as evidence of refresh.

Do not use a public resolver as a universal referee. It may be outside the relevant split-horizon view, use a different ECS policy, or be blocked from private authority. Querying it can be useful as a *comparison* if the name is public, but the affected resolver is the primary observation point. Even two queries to the same resolver IP may reach different backends behind anycast or a load balancer. If a TTL seems to jump unpredictably, record resolver identifiers where available or ask the operator for fleet logs. The evidence should preserve exactly where each query came from and went.

### A compact evidence record

During a real outage, save one line per observation with UTC time, process or host, network view, query name, record type, resolver IP, response status, answer address, remaining TTL, and connection result. For a connection result, record the selected remote IP and whether the socket was reused. This is enough to reconstruct which boundary first diverged. It also lets a later reviewer distinguish a failed publication from a valid cached answer, a process-held address, a split view, and a healthy lookup followed by an application failure. Without this record, teams often change DNS repeatedly and then attribute recovery to the last change, even if an earlier cache simply expired.

## Run it yourself

### Question

Can a recursive DNS cache continue returning an old address after its authoritative zone has changed, and does the answer change after the cached TTL expires? This lab isolates that one claim. It does not simulate Java's runtime cache, a connection pool, or serve-stale.

### Preconditions

Use a disposable Linux environment with `ip`, `dig`, `python3`, and root or `CAP_NET_ADMIN` for namespace creation. The commands use the series' canonical `c` and `s` namespaces, `c0` and `s0` interfaces, and `10.77.0.0/30` addresses from [post #1's netlab setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). Run the setup first and confirm the names. The lab adds a small DNS authority and caching forwarder in the server namespace on separate loopback ports, so it changes no production resolver file. `dnslib` is a Python package prerequisite; install it in a disposable virtual environment before the timed experiment. The commands below assume `dnsmasq` version with `--server` and `--cache-size` options, and `dig` from BIND utilities. Record `dnsmasq --version` and `dig -v` on your machine. If `dnsmasq` is unavailable, do not reinterpret these commands as a public DNS benchmark.

This example uses a synthetic `ttl-lab.test` name. The authority runs on `10.77.0.2:5300`; the cache runs on `10.77.0.2:5301`. The address values `192.0.2.10` and `192.0.2.20` are documentation addresses. They are not reachable endpoints, and no HTTP call is needed to demonstrate caching. Stop any prior lab processes on these exact ports before beginning. The test is isolated to the namespaces and files named here.

### Baseline

```bash
set -euo pipefail
sudo ip netns list | grep -E '^c([[:space:]]|$)'
sudo ip netns list | grep -E '^s([[:space:]]|$)'
sudo ip netns exec c ip -4 addr show dev c0
sudo ip netns exec s ip -4 addr show dev s0
sudo ip netns exec c ip route get 10.77.0.2
sudo ip netns exec s ip -4 route
sudo ip netns exec s ss -lun '( sport = :5300 or sport = :5301 )'
dig -v
dnsmasq --version | head -1
python3 -c 'import dnslib; print(dnslib.__version__ if hasattr(dnslib,"__version__") else "dnslib installed")'
```

The `ss` output should contain no listener on the two lab ports. A command that finds an existing listener means the environment is not clean; investigate it rather than killing an unrelated process. Create an isolated directory and an authoritative answer file. The Python server reads that file on every query, so changing the file is a controlled authoritative mutation.

```bash
set -euo pipefail
LAB_DIR="$(mktemp -d /tmp/dns14.XXXXXX)"
printf '%s\n' "$LAB_DIR" > /tmp/dns14-active-dir
printf '%s\n' '192.0.2.10' > "$LAB_DIR/address"
cat > "$LAB_DIR/auth.py" <<'PY'
import pathlib, socketserver, sys
from dnslib import DNSRecord, RR, A, QTYPE, RCODE
address_file = pathlib.Path(sys.argv[1])
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        payload, sock = self.request
        request = DNSRecord.parse(payload)
        reply = request.reply()
        reply.header.aa = 1
        if str(request.q.qname).lower() == 'ttl-lab.test.' and request.q.qtype == QTYPE.A:
            reply.add_answer(RR('ttl-lab.test.', QTYPE.A, rdata=A(address_file.read_text().strip()), ttl=20))
        else:
            reply.header.rcode = RCODE.NXDOMAIN
        sock.sendto(reply.pack(), self.client_address)
class Server(socketserver.ThreadingUDPServer):
    allow_reuse_address = True
Server(('10.77.0.2', 5300), Handler).serve_forever()
PY
PYTHON_BIN="$(command -v python3)"
sudo ip netns exec s "$PYTHON_BIN" "$LAB_DIR/auth.py" "$LAB_DIR/address" > "$LAB_DIR/auth.log" 2>&1 &
printf '%s\n' "$!" > "$LAB_DIR/auth.pid"
sudo ip netns exec s dnsmasq --no-daemon --no-resolv --no-hosts --bind-interfaces \
  --listen-address=10.77.0.2 --port=5301 --cache-size=100 \
  --server=/ttl-lab.test/10.77.0.2#5300 --log-queries \
  > "$LAB_DIR/cache.log" 2>&1 &
printf '%s\n' "$!" > "$LAB_DIR/cache.pid"
sleep 1
sudo ip netns exec c dig @10.77.0.2 -p 5300 ttl-lab.test A +noall +answer
sudo ip netns exec c dig @10.77.0.2 -p 5301 ttl-lab.test A +noall +answer
```

Read the final field in each A answer: it should be `192.0.2.10`. Read the numeric TTL field immediately before `IN A`: the direct authority answer is 20 seconds, while the cached answer starts near 20 and counts down. Scheduler timing may make the first cached TTL 18–20 seconds. The `dnsmasq` log can show a forwarded first query and a cache hit later. These are expected lab ranges, not measurements from a production provider.

### Apply one change

```bash
set -euo pipefail
LAB_DIR="$(cat /tmp/dns14-active-dir)"
printf '%s\n' '192.0.2.20' > "$LAB_DIR/address"
sudo ip netns exec c dig @10.77.0.2 -p 5300 ttl-lab.test A +noall +answer
```

Only the authoritative answer file changed. The direct query should now return `192.0.2.20` with TTL 20. The cache has not been flushed and its configuration has not changed.

### Compare

```bash
set -euo pipefail
sudo ip netns exec c dig @10.77.0.2 -p 5301 ttl-lab.test A +noall +answer
sleep 22
sudo ip netns exec c dig @10.77.0.2 -p 5301 ttl-lab.test A +noall +answer
```

Read the final address field and the TTL field on both lines. Immediately after the treatment, expect the cached answer to remain `192.0.2.10` if less than twenty seconds elapsed since the cache's first fill. After a 22-second wait, expect `192.0.2.20` with a refreshed TTL around 18–20 seconds. If the first cached answer has already expired because setup or reading took too long, rerun the baseline with a newly created lab directory and change the authority promptly. The qualitative result is the important one: the authoritative server can be new while the recursive cache is still old. Exact seconds vary with command scheduling and dnsmasq's TTL accounting. Inspect `cache.log` for whether the first post-change answer was served locally and whether the later query was forwarded.

### Reset

```bash
set -euo pipefail
LAB_DIR="$(cat /tmp/dns14-active-dir)"
sudo kill "$(cat "$LAB_DIR/cache.pid")" "$(cat "$LAB_DIR/auth.pid")"
rm -f /tmp/dns14-active-dir
rm -rf -- "$LAB_DIR"
```

The reset stops only the two processes whose PIDs this lab recorded and removes only its temporary directory. Do not delete the shared `c` and `s` namespaces; post #1 owns their teardown. In production, use read-only `dig @configured-resolver hostname A +noall +answer +comments`, a direct query to an authorized authoritative server, resolver metrics, and application connection logs. Do not run this lab's DNS server or cache on a production interface. The experiment demonstrates one cache boundary; it does not establish a global upper bound on real client recovery.


The practical standard is a falsifiable statement, not a preferred TTL. If a team says that a hostname change reaches clients within one minute, ask for a continuously running client, a direct authoritative timestamp, a configured resolver timestamp, a chosen remote-IP timestamp, and a successful request timestamp. If they cannot produce all five, they have measured only publication. If they can, the comparison points directly to the slow boundary. A 30-second DNS RRset can coexist with a ten-minute socket, and a ten-minute RRset can coexist with a client that refreshes a different address family sooner. Record the behavior that users experience.

This matters even when there is no incident. A deployment system may drain the old site based on the authoritative change time and deallocate it too early. A synthetic monitor may query a public resolver while most customers use enterprise forwarders. A test client may open a new process for every request, hiding the long-lived service process's cache. The fix is to keep one representative process alive during a planned cutover and compare it with a new process from the same host and network view. That paired test isolates process lifetime without relying on folklore about the language runtime.

## Key takeaways

DNS answers are inputs to connection decisions, and several independent owners may retain those inputs. A record TTL is the recursive cache's clue about one RRset, not a lease on every application process or socket. A CNAME chain creates several lifetimes. A negative answer has its own SOA-derived lifetime. A split DNS view can make two different answers correct at the same instant. Serve-stale may intentionally return expired data when authority cannot be refreshed.

During an incident, ask what name the process used, what remote IP it chose, which resolver and view it asked, and what the authoritative server says for that view. Then measure the actual recovery path: detection, publication, cache observation, reconnection, and replacement readiness. If those boundaries are visible, "DNS propagation" stops being a catch-all explanation and becomes a testable timeline.

## Further reading

- [RFC 1035, Domain Names: Implementation and Specification, November 1987](https://www.rfc-editor.org/rfc/rfc1035.html), for TTL and DNS message fields.
- [RFC 2308, Negative Caching of DNS Queries, March 1998](https://www.rfc-editor.org/rfc/rfc2308.html), for NXDOMAIN and NODATA caching.
- [RFC 8767, Serving Stale Data to Improve DNS Resiliency, March 2020](https://www.rfc-editor.org/rfc/rfc8767.html), for the exceptional stale-answer policy.
- [Oracle Java networking properties documentation](https://docs.oracle.com/en/java/javase/15/docs/api/java.base/java/net/doc-files/net-properties.html), for JVM positive and negative address caches.
- [BIND 9 views reference](https://bind9.readthedocs.io/en/latest/reference.html#view-statement-definition-and-usage), for one implementation of split DNS answers.
