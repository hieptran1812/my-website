---
title: "DNS: The distributed database your request starts in"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Trace a name from a stub resolver through root, TLD, and authority, then diagnose caches, negative answers, truncation, and TCP fallback."
tags: ["networking", "distributed-systems", "dns", "name-resolution", "recursive-resolver", "authoritative-dns", "dns-caching", "edns", "tcp-fallback"]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/dns-the-distributed-database-your-request-starts-in-1.webp"
---

A request can fail before it has a destination. The service process has not sent a SYN, negotiated TLS, or put an HTTP byte on the wire. Yet the user sees the same apparent symptom as a dead server: the page does not load. That first ambiguity is why the [series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) puts DNS at the left edge of the request's latency ladder. The opening figure lights that segment. It does not assign made-up milliseconds to the others: actual timing depends on cache state, resolver placement, and whether the connection is reused.

![The request latency ladder with DNS highlighted before TCP, TLS, request, server work, first byte, and transfer.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-1.webp)

We usually say a hostname "resolves to an IP address." That phrase hides a distributed lookup over separately operated databases, an expiration policy, and several places where an answer can be absent even though the application is healthy. The practical engineering question is not simply whether a name resolves. It is **who answered, from which data, at what age, over which transport, and with which response code**. Those fields tell us whether to inspect a local cache, a recursive resolver, a delegation, an authoritative server, or the network between them.

This post follows one ordinary address lookup, explains the wire decisions that make it work, and ends with commands that isolate one controlled transport change. The [next DNS post](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers) owns application and operating-system cache layers, failover policy, and stale answers in production. Here we build the protocol model that makes those behaviors intelligible.

## The first boundary: looking up a destination is not sending to it

Suppose a process wants `www.example.com`. It asks a local name-resolution API for an address. The operating system may satisfy that call locally, or it may send a DNS question to a configured recursive resolver. If the recursive resolver has a valid cached answer, it replies from cache. Otherwise it walks the namespace until an authoritative server supplies an answer or an authoritative negative result. The process can then open a connection to the returned address. A resolver is consulted before that connection; it is not a proxy through which the later application packets pass.

![A branched path map separating the DNS control lookup through a recursive resolver from the later client-to-edge application data path.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-2.webp)

That separation is operationally useful. If `dig` returns an address promptly while `curl` stalls in connection establishment, the slow phase is no longer the DNS walk. Conversely, if `curl` reports a large `time_namelookup` and its `time_connect` is close to that value, the delay occurred before the TCP connection completed. Remember that curl's timestamps are cumulative from the start of the transfer. A difference between adjacent fields is a phase estimate, not the raw field interpreted in isolation. Connection reuse and application DNS caches complicate the picture; a repeated request may never repeat the DNS exchange at all.

DNS is a distributed database in a specific sense: authority for different subtrees is delegated to different servers, and recursive resolvers cache records under explicit time limits. It is not one globally synchronized table, nor a database that gives a transactional snapshot across all record types. An A answer and an AAAA answer can originate from different cache states. A CNAME may require another lookup. A parent may point to an old child nameserver while the child zone has changed. We should expect partial views during change and design probes that reveal which layer we sampled.

The terms matter. A **stub resolver** is the client-side component that asks for an answer. A **recursive resolver** agrees to pursue the answer on the client's behalf. An **authoritative server** is responsible for a zone's data, not for searching every other zone for the client. A **referral** tells a resolver which nameservers know more about a delegated part of the namespace. The terminology and recursive versus iterative behavior are specified in [RFC 1034, November 1987](https://www.rfc-editor.org/rfc/rfc1034.html) and clarified by [RFC 9499, March 2024](https://www.rfc-editor.org/rfc/rfc9499.html).

| Question | Useful field or command | What it separates |
| --- | --- | --- |
| Did this resolver answer? | `dig @RESOLVER name A` and `SERVER` | The configured resolver from another one |
| Was the response authoritative? | `flags`, especially `aa` | Authority from recursive or cached service |
| Did recursion run? | `rd` request and `ra` response flags | A recursion request from a server's offered capability |
| Was the name absent? | `status: NXDOMAIN`, or `NOERROR` with no requested type | Name error from existing name without that RR type |
| Is the answer old? | Record `TTL` on successive queries | Cache countdown from a newly fetched answer |
| Was UDP answer truncated? | `flags` containing `tc` | A complete UDP response from one needing retry |

Do not turn one field into a diagnosis. `aa` tells you about the answering server's authority for that answer, not whether the whole network is healthy. A low TTL can mean an intentionally short published TTL or an almost expired cached answer. A quick response can be a cached answer while authoritative servers are currently unavailable. The command must be aimed at the component whose claim you want to test.

## The delegation walk: asking who knows, then asking what they know

A cold recursive resolver starts with root hints, which give it addresses for root nameservers. It does not ask the root for the final address and expect the root to own every domain. The root normally returns a referral toward the relevant top-level domain, such as `.com`. The resolver follows that referral to a `.com` nameserver, which can refer it to the nameservers for `example.com`. Only a server authoritative for the relevant zone can finally answer the address question for `www.example.com`, or report that the requested name or RR type is absent.

![A referral stack showing root, TLD, and child-zone authority with delegation NS records and needed glue addresses.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-3.webp)

The animation shows why this is a walk. Each step gives the resolver a *place to ask next*. It does not copy the whole worldwide name database into one response. A warm recursive resolver can skip some steps because cached delegation records and addresses remain valid. A resolver can also ask different questions at intermediate levels when it uses QNAME minimization. In the clean mental model, we show the referral path; in a real trace, do not assume that every upstream packet contains the original full hostname. [RFC 9156, November 2021](https://www.rfc-editor.org/rfc/rfc9156.html) describes this privacy-preserving variation.

<figure class="blog-anim">
<svg viewBox="0 0 800 230" role="img" aria-label="A recursive resolver walks from a root referral to a TLD referral to an authoritative answer, then returns the address" style="width:100%;height:auto;max-width:900px">
<style>
.dns13-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.dns13-label{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dns13-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dns13-path{stroke:var(--border,#d1d5db);stroke-width:3;fill:none}
.dns13-active{fill:var(--accent,#6366f1);opacity:.24}
@keyframes dns13-walk{0%,18%{transform:translateX(0)}28%,45%{transform:translateX(190px)}55%,72%{transform:translateX(380px)}82%,100%{transform:translateX(570px)}}
.dns13-walker{animation:dns13-walk 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.dns13-walker{animation:none;transform:translateX(570px)}}
</style>
<path class="dns13-path" d="M115 108H685"/>
<rect class="dns13-box" x="20" y="58" width="170" height="100" rx="12"/>
<rect class="dns13-box" x="210" y="58" width="170" height="100" rx="12"/>
<rect class="dns13-box" x="400" y="58" width="170" height="100" rx="12"/>
<rect class="dns13-box" x="590" y="58" width="170" height="100" rx="12"/>
<rect class="dns13-active dns13-walker" x="20" y="58" width="170" height="100" rx="12"/>
<text class="dns13-label" x="105" y="96">root</text>
<text class="dns13-small" x="105" y="126">TLD referral</text>
<text class="dns13-label" x="295" y="96">TLD</text>
<text class="dns13-small" x="295" y="126">zone referral</text>
<text class="dns13-label" x="485" y="96">authoritative</text>
<text class="dns13-small" x="485" y="126">address RRset</text>
<text class="dns13-label" x="675" y="96">resolver</text>
<text class="dns13-small" x="675" y="126">answer to stub</text>
<text class="dns13-small" x="400" y="204">One resolver follows referrals; the stub waits for its final response.</text>
</svg>
<figcaption>The highlighted state advances only when the resolver learns where to ask next; the root and TLD provide referrals, and the authoritative server supplies the answer.</figcaption>
</figure>

The phrase "root, TLD, authoritative" is a teaching path, not a promise of exactly three packets. A resolver may query multiple servers after a timeout, fetch the address of a nameserver named in a referral, validate DNSSEC data, chase a CNAME, or stop early with a cached answer. A zone can also have delegated subzones below the registered domain. If `team.example.com` is separately delegated, the resolver needs another referral to reach its authority. Counting packets without recording cache state or delegations produces misleading performance conclusions.

A referral usually puts **NS resource records** in the authority section and may include corresponding address records in the additional section. Those address records are called **glue** when they break the circular dependency of reaching a nameserver whose name is inside the zone being delegated. Imagine the `.com` servers delegate `example.com` to `ns1.example.com`. If they gave only that name, a cold resolver would need to resolve `ns1.example.com` before it could query the `example.com` servers, but the latter is precisely where it is trying to go. Glue provides an address for that in-domain nameserver so the walk can continue. It is a bootstrap hint associated with the delegation, not permission to ignore authoritative data forever. The current referral requirements are set out in [RFC 9471, September 2023](https://www.rfc-editor.org/rfc/rfc9471.html).

The root does not hold the application IP address. The TLD usually does not either. This matters when triaging failure. A correctly operating root can refer us to nameservers that are unreachable. A correctly operating TLD can refer us to nameservers whose child zone is misconfigured. A final authoritative answer can be correct at one nameserver and inconsistent at another during a rollout. Test each boundary separately. A single `dig +trace` is excellent for a view of delegation, but it is not a replacement for asking each listed authoritative server and comparing their answers.

Here is a safe read-only exploration using a domain reserved for documentation. Results on a public network vary over time; inspect the fields, not a copied address value.

```bash
# On a host with BIND dig installed. These are read-only public DNS queries.
dig example.com A +trace
dig example.com NS +noall +answer
dig www.example.com A +noall +answer +authority +additional
```

`+trace` starts at a root server and follows delegations using iterative queries. Its output shows servers and the sections returned along the path. The command is a client-side diagnostic, not proof that the recursive resolver configured by your application followed exactly the same path. To inspect that resolver, query its address explicitly and record `SERVER`, `status`, `flags`, the question, and the answer TTL. If the process uses a different resolver than the shell, inspect the process runtime and operating-system configuration first. The [production DNS cache post](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers) takes that handoff further.

## An answer is a typed record, not a string-to-address map

The wire unit is a **resource record** (RR). It has an owner name, type, class, TTL, and type-specific data. The question asks for a particular name, type, and class. The response may include an answer section, an authority section, and an additional section. Those sections are distinct; a nameserver named in authority and an address placed in additional are not the same claim as an authoritative A record in the answer section. [RFC 1035, November 1987](https://www.rfc-editor.org/rfc/rfc1035.html) defines the core message and RR format.

The records that service engineers meet most often are these:

| Type | What it carries | Operational consequence |
| --- | --- | --- |
| A | IPv4 address for an owner name | One possible connection destination, not a health check |
| AAAA | IPv6 address for an owner name | A client can choose a different address family and path |
| CNAME | Alias from one owner name to another canonical name | Resolution may continue at the target; the target's RRset has its own TTL |
| NS | Nameservers for a zone or a delegation | Tells resolvers where to ask next; not an application endpoint |
| SOA | Zone origin and control fields, including the negative-cache minimum | Provides authority context and bounds a cacheable negative result |
| MX | Mail exchanger preference and target | Used by mail delivery, not normal browser address selection |
| TXT | One or more text strings | Carries many application conventions; type alone does not validate meaning |

Two common mistakes follow from flattening these types into a single lookup result. First, an A record and an AAAA record are distinct questions. An `AAAA` NODATA response says nothing about whether the A answer exists. Second, a CNAME is not an instruction for the HTTP client to connect to the literal alias target as text. The resolver follows the alias and returns the relevant address data. Each RRset involved has a TTL, so an alias can stay cached while the address RRset beneath it expires, or vice versa. A partial change can therefore make different clients see different effective destinations without any broken DNS server.

A **RRset** is the group of records with the same owner, class, and type. When a name has several A records, the group is cached as an RRset. DNS does not certify that each address is healthy, that a client will distribute requests evenly, or that all recursive resolvers will observe a change at once. Those are application and operations questions. The [service discovery and load balancing post](/blog/software-development/microservices/service-discovery-and-load-balancing) owns the architecture choice; this post owns how its DNS answers reach a client. If a connection to one returned IP fails, compare the answer set and connection target before blaming the authoritative zone.

## CNAME chains, address families, and why one lookup becomes several

A CNAME does not contain an IP address. It says that the owner is an alias for another canonical name. The resolver still needs the requested address type at that target. A response can include both the CNAME and the target address in one message, or the resolver may need another query. Either way, the alias RRset and target A or AAAA RRset can have different TTLs. A change at the target may become visible while the alias remains cached. A change to the alias may wait on its own cache lease even if the target address has a shorter TTL.

For example, imagine `www.example.com` has a CNAME to `front.example.net`. The alias has a TTL of 600 seconds and the target A RRset has a TTL of 60 seconds. A resolver that caches both at 10:00 can refresh the target address around 10:01 while retaining the alias until around 10:10, under a simple TTL-following model. If the operator changes the alias at 10:02, clients of that resolver can continue following the old target for several more minutes. The ten-minute and one-minute values in this example are assumptions for arithmetic, not observations of these documentation domains. The takeaway is not that CNAMEs are slow. It is that *each cached RRset has its own lease*.

![A DNS record matrix separating address, alias, delegation, mail, text, and zone metadata from the response section where each commonly appears.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-4.webp)

A and AAAA add another branch. A dual-stack client may ask for both types and select an address based on local policy and reachability. A healthy A answer does not prove that the AAAA path is healthy; a missing AAAA answer does not make the name nonexistent when A exists. If an incident affects only some clients, record which address family each client selected and which endpoint it tried. Use explicit `dig name. A` and `dig name. AAAA` queries, then test connections to the returned addresses separately. Do not collapse those into a single "DNS works" checkbox.

Multiple A or AAAA records do not make DNS a complete load balancer. The resolver can return a set, but the client decides which address to use and when to retry. Existing connections can stay pinned to a destination while new lookups see changed data. Health checks and traffic steering policies live above the core RR format. The [load balancing post](/blog/software-development/system-design/load-balancing-from-l4-to-l7) covers the architectural choices; at this layer, verify the RRset and the actual destination IP selected by the client.

This is also why query-count claims need a named workload. A browser navigation can require A and AAAA for one name, then more names for assets or redirects. A CNAME can add a target lookup, and a cold resolver can need delegations. A warm cache may collapse most of that work. Saying "DNS costs three queries" from the root/TLD/authority teaching diagram is as misleading as saying "HTTP costs one packet." The diagram establishes the authority sequence. Captures and resolver logs establish the count for a particular request.

### The flags and sections worth reading

Run `dig +comments +noall +answer +authority +additional example.com A` and examine its status line. `NOERROR` says the DNS operation completed without a DNS error code. It does not promise that an answer of the requested type is present. `NXDOMAIN` says the queried name does not exist according to the responder's view. `SERVFAIL` says the server failed to complete the query; the cause might be upstream reachability, validation, or server failure, and the status alone cannot identify which. The `aa` flag marks an authoritative answer. `rd` records a recursion-desired request bit, while `ra` reports the responder's recursion availability. `tc` reports truncation. Each is a clue tied to one response from one server.

When diagnosing a referral, ask with `+norecurse` and examine authority and additional, not only answer. When diagnosing a negative result, record the SOA in authority. When diagnosing a suspected stale result, ask the recursive resolver twice a few seconds apart and compare the TTL countdown. Then ask each authoritative server directly and compare its answer and serial or RRset. An unchanged TTL on repeated cached replies might mean an answer was refreshed, a different server answered, or a layer in front of your probe rewrote the response. Capture the `SERVER` field and avoid claiming a cause from TTL alone.

A DNS name also has a **class**, almost always `IN` for Internet use, and a type. This is why a negative result must be scoped precisely. "The domain does not resolve" collapses the actual evidence. Write down the full question: `www.example.com. IN AAAA`, the server queried, the returned status, and whether an SOA appeared. That makes the next experiment obvious. If A succeeds and AAAA is NODATA, look at address-family selection. If both A and AAAA produce NXDOMAIN through one resolver but authority has them, investigate cache state or forwarding. If the authoritative servers disagree, the resolver may be an innocent messenger.

## TTL is a lease on reuse, not a propagation stopwatch

The TTL attached to an RR is the maximum time a cache may reuse that data before consulting the source again. It is measured in seconds on the wire. A resolver generally decrements remaining TTL as an RRset ages. A TTL of zero means the record should be used only for the current transaction and not cached, per [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html). The simple model is useful: if a resolver caches an A RRset with an initial TTL of 300 seconds at time $t_0$, the remaining TTL at time $t$ is approximately $\max(0,300-(t-t_0))$ seconds. This is an explanatory model. Real resolvers apply policy, and clients may have additional caches.

Here is the first worked example. An operator publishes an A RRset with a TTL of 300 seconds, and a recursive resolver fetches it at 12:00:00. The operator changes the authority at 12:01:00. Under the simple model, the resolver can serve its old cached value for about 240 more seconds, until 12:05:00. That is 300 seconds of permission from when the resolver fetched the old RRset, not 300 seconds from the operator's update. Another resolver that fetched at 11:59:30 would expire earlier. The arithmetic is derived here; it is not a measurement of a particular provider. If the operator reduced the TTL to 30 seconds at 12:00:30, the old cached 300-second RRset would not retroactively shrink. Lower a TTL before a planned move, and wait for previously cached long TTLs to age out. Even then, application caches and connection reuse can extend observed behavior beyond a DNS TTL.

The second worked example is a capacity trade-off, again an explanatory model rather than a DNS protocol equation. Assume one recursive resolver receives 600 requests per minute for a name, all clients use the same RRset, requests arrive evenly, and the resolver fetches once per expiration. At a 60-second TTL, the idealized upstream fetch rate is about one per minute; at a 10-second TTL, about six per minute. Across 1,000 independent resolvers under the same assumptions, that becomes about 1,000 versus 6,000 upstream fetches per minute. This is a sixfold change in that model, not a prediction for any public authoritative service. Real resolvers coalesce misses, prefetch, cap TTLs, serve stale answers by policy, and distribute requests unevenly. The point is causal: shorter cache life generally increases opportunities for freshness and raises upstream query load. Choose it with both sides visible.

A numeric table needs provenance, so keep the assumptions beside the numbers:

| Published TTL | Ideal fetches per resolver per minute | Ideal fetches across 1,000 resolvers per minute | Source |
| --- | ---: | ---: | --- |
| 60 seconds | 1 | 1,000 | Derived here: 60 / TTL, then multiply by 1,000 |
| 10 seconds | 6 | 6,000 | Derived here: 60 / TTL, then multiply by 1,000 |

This model assumes continuously demanded records. A rarely requested record might expire unused and not be fetched again until a later request. It also assumes the authority is healthy at expiration. If authority fails, some resolvers may return an error and others may serve expired data under an explicit stale-answer policy, described in [RFC 8767, March 2020](https://www.rfc-editor.org/rfc/rfc8767.html). A successful stale response can preserve availability but conceal the fact that authority has become unreachable. That is why an incident check should include direct authoritative queries as well as user-facing recursive queries.

TTL is also not a guarantee about connection movement. A client can keep an existing TCP or QUIC connection after the DNS RRset has expired. A process can cache an address independently of the system resolver. A load balancer can continue receiving traffic at an old address. DNS controls what an address lookup can return; it does not revoke a socket. The next post explores those cache layers and honest DNS failover time. Here, hold onto one invariant: **a TTL bounds reuse by a cache following that TTL, not every downstream effect of the answer**.

## A missing answer can be cached too

The surprising outage pattern is a name that is created correctly at authority but still appears absent to some clients. Positive answers have TTLs, and negative answers can also be cached. [RFC 2308, March 1998](https://www.rfc-editor.org/rfc/rfc2308.html) defines how a cacheable negative answer carries an SOA record in the authority section. Its negative caching TTL is the minimum of the SOA RR's TTL and the SOA MINIMUM field. A resolver that cached a negative result can keep returning it until that negative TTL expires even after the operator adds the record at authority.

We must distinguish two negative shapes. **NXDOMAIN** means the queried name does not exist. **NODATA** is a successful `NOERROR` response for an existing name without the requested type. The cache keys differ: RFC 2308 treats an NXDOMAIN result by name and class, while NODATA is scoped by name, type, and class. This is why an `AAAA` NODATA answer should not be generalized to "this hostname is missing." Check the question and response code together. A CNAME chain can add more nuance, since the final target may be missing while the alias itself exists.

As a third worked example, assume an authoritative server returns NXDOMAIN at 09:00:00 with an SOA TTL of 900 seconds and SOA MINIMUM of 120 seconds. Under RFC 2308, the negative-cache TTL is $\min(900,120)=120$ seconds. If the operator creates the name at 09:00:30, a resolver that cached the negative response at 09:00:00 may keep returning it until about 09:02:00. That is about 90 seconds after the change, derived from the stated times. A different resolver that never saw the negative answer may return the new positive record immediately. An old negative answer and a new positive answer can coexist across resolvers without a contradictory authoritative database.

| Observation | Meaning to test | Next query |
| --- | --- | --- |
| Resolver says NXDOMAIN with SOA | A name error may be cached | Ask an authoritative server directly for the same name and type |
| Resolver says NOERROR without the requested RR type | The name may exist but lack this type | Ask for A and AAAA separately; inspect CNAME and authority |
| Authority has the new RR but resolver says NXDOMAIN | Negative cache or a different delegation path | Compare SOA, server, and remaining negative TTL |
| Both authority and resolver say NXDOMAIN | Published zone or name may still be wrong | Check exact name, trailing dot, zone, and nameserver set |

This is a protocol-level reason to create a name before giving it to clients, especially for a planned rollout. If clients query too early, they can seed negative caches. Lowering a positive A TTL does not lower the SOA-based negative TTL. The operational policy belongs in the next post, but the mechanism is here: absence is data with cache behavior. A runbook that only checks whether an A record exists now at the zone editor misses the cached historical answer that users are receiving.

Negative caching is also a protection for the rest of the system. Without it, every lookup for a typo or nonexistent subdomain could walk upstream repeatedly. During a failure or attack, that multiplication can burden authorities. Yet a long negative TTL increases recovery time after a missing record is repaired. Neither "cache forever" nor "never cache" is a free choice. Preserve the SOA in your diagnostic capture, and distinguish a cached negative result from an authoritative one before flushing anything.

Do not treat every failure response as a normal negative answer. `SERVFAIL` is not NXDOMAIN, and a timeout is not a statement that the name does not exist. Newer [RFC 9520, December 2023](https://www.rfc-editor.org/rfc/rfc9520.html) addresses caching resolution failures so clients do not pound an already failing dependency, but it does not turn a server failure into proof of absence. If an application retries every `SERVFAIL` immediately and many clients do it together, it can amplify load at the exact moment the resolver or authority is sick. The [retry and backoff post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns that client policy.

## UDP, EDNS(0), truncation, and the TCP path many firewalls forget

A classical DNS query often fits comfortably in UDP. That fact became a brittle assumption: "DNS uses UDP port 53, so TCP port 53 is unnecessary." The base DNS specification constrained DNS messages over UDP without extensions to 512 octets. EDNS(0) lets a client advertise a larger UDP payload size using an OPT pseudo-record. It extends the message format; it does not change the meaning of a delegation or turn an oversized datagram into guaranteed delivery. [RFC 6891, April 2013](https://www.rfc-editor.org/rfc/rfc6891.html) defines EDNS(0) and the advertised payload size.

![A transport decision flow showing the legacy 512-octet UDP limit, EDNS advertised capacity, a truncated TC response, and a TCP retry.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-5.webp)

The capacity in an EDNS OPT record is an advertisement of what the requester can receive in one UDP response, not a promise that the network path will carry a datagram of that size. IP fragmentation and middlebox behavior can still lose large responses. A server may choose to truncate rather than send a large UDP response. The DNS `TC` flag says the answer was truncated. The client then retries using DNS over TCP, which frames messages with a length prefix. Modern DNS implementations are expected to support TCP for ordinary queries, not only zone transfers. [RFC 7766, March 2016](https://www.rfc-editor.org/rfc/rfc7766.html) makes TCP support an implementation requirement and explains the retry after truncation; [RFC 9210, March 2022](https://www.rfc-editor.org/rfc/rfc9210.html) covers operational requirements.

Do not read the 512-octet value as a contemporary recommended EDNS size. It is the legacy UDP limit without EDNS. Likewise, do not read a configured EDNS size as the actual response size. `dig` prints an `EDNS: version: 0` line and often a `udp:` advertised size in its OPT pseudo-section. The reply's `MSG SIZE rcvd` tells you the bytes received for that message. If `tc` appears, look for a TCP retry or test `dig +tcp` explicitly. When a resolver succeeds on a small A response and times out on a large DNSSEC response, a blocked TCP fallback or fragmented UDP path becomes a candidate. It is not the only candidate, so compare with direct authoritative queries and transport-specific tests.

The arithmetic behind the old assumption is straightforward. Suppose an answer would be 900 DNS-message octets and the exchange has no EDNS. The 900-octet answer exceeds the 512-octet UDP payload limit by 388 octets. The server cannot place that full answer in a conforming legacy UDP response, so it truncates and sets `TC=1`; the client needs another exchange over TCP. Those 900 and 388 values are illustrative arithmetic, not an observed production response. With EDNS, an advertised UDP payload of 1,232 octets would be large enough for a 900-octet DNS message in isolation, but path conditions and server policy still matter. This is why a firewall rule that allows only UDP/53 can pass many routine tests and fail a narrower class of real lookups.

A second useful number is the transport overhead. A cold TCP fallback must establish a TCP connection before the query can complete, unless an existing TCP connection is reusable. The handshake adds network round trips, so the impact depends on RTT and server behavior. Avoid asserting a universal millisecond penalty. If a resolver is 2 ms away on a local network, one extra round trip has a different cost from a resolver across a 100 ms path. The actual question in an incident is whether the response was truncated, whether the retry was sent, and whether the retry returned. Packet capture and resolver logs answer that better than a broad claim that "DNS is slow."

For a production-safe probe, use `dig +tcp @RESOLVER target.example. DNSKEY +dnssec` where `RESOLVER` is an approved recursive resolver and `target.example.` is a name you are authorized to query. Compare with the same query over UDP and preserve the `SERVER`, `Query time`, `flags`, and `MSG SIZE rcvd` fields. A query that works with `+tcp` while the default path fails points toward the UDP path or an EDNS interaction; one that works over UDP but fails with `+tcp` points toward TCP/53 reachability or TCP service. DNSSEC can make answers larger, but the exact size depends on the zone and its keys. Do not assert a particular size without recording the answer.

Some modern resolvers deliberately use smaller EDNS UDP sizes to reduce fragmentation risk. That can cause more truncation and therefore more TCP. It is a trade: avoid fragile large datagrams at the cost of TCP fallback for some answers. The correct setting depends on the path, workload, and server support. Benchmark it with the actual resolver and representative record types before changing fleet-wide configuration. The protocol permits different choices; one magic buffer size cannot fix every path.

## A public failure at the DNS boundary: Akamai, July 2021

On July 22, 2021, Akamai reported that a software configuration update at 15:45 UTC triggered a bug in the DNS component of its Secure Edge Content Delivery Network. Some customer websites became unavailable, and the disruption lasted up to an hour. Akamai said service resumed after rollback. Its July 23 update narrowed the scope: the impact was isolated to a DNS component of that secure edge CDN, not Akamai's general DNS service. Those facts come from [Akamai's incident statement, July 22 and updated July 23, 2021](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved). We should preserve that correction rather than repeat the broader first description.

The case is useful because it lands before the application connection. A client cannot use a healthy web server if the name-to-destination step that points the client at it is unavailable or incorrect. The public statement identifies the trigger as a configuration update and the immediate technical condition as a bug in a DNS component. It does not provide packet captures, a detailed root cause inside the component, or a breakdown of resolver cache behavior. We should not invent those details. The documented recovery action was rollback; the stated follow-up was review of the software update process.

Map it onto the path map: the failure entered at the name-resolution boundary for the affected secure edge service. It could make some websites unavailable before the later TCP, TLS, proxy, and application hops mattered. The blast radius depended on which customer sites used that component and what DNS answers were available to clients, but Akamai's short statement does not quantify that distribution. The transferable control is to test the actual DNS path that customers use during configuration rollout, then keep a rollback that restores that path quickly. A synthetic HTTP probe alone can be misleading if it reuses an already established connection or a cached address; include a fresh name lookup from multiple resolver perspectives.

The incident also illustrates honest fault ownership. "DNS outage" can mean authority, delegation, recursive service, local cache, or a DNS-based traffic steering component. Akamai's update specifically corrected which DNS component was affected. In your own incident notes, name the component and evidence: `SERVFAIL from resolver X`, `authoritative server Y returns old A RRset`, `TC set and TCP retry blocked`, or `NXDOMAIN cached with SOA TTL Z`. The phrase "DNS is broken" is too wide to guide a repair.

| Case field | Verified value | Source |
| --- | --- | --- |
| Organization and component | Akamai Secure Edge CDN DNS component | [Akamai statement, July 2021](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved) |
| Event time | July 22, 2021, 15:45 UTC trigger | [Akamai statement, July 2021](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved) |
| User-visible effect | Some customer websites unavailable, up to an hour | [Akamai statement, July 2021](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved) |
| Trigger and recovery | Configuration update triggered bug; rollback restored service | [Akamai statement, July 2021](https://www.akamai.com/blog/news/akamai-summarizes-service-disruption-resolved) |

## Diagnose the boundary, not the name in the error message

A useful DNS runbook starts with the exact question the failing client asked. Record the fully qualified name, record type, resolver address, time, client environment, and whether the operation used a local API or `dig`. A process can append a search suffix, choose AAAA before A, send through a container-local forwarder, or use its own cache. A shell query for a short name is not necessarily the same query. A trailing dot in `dig www.example.com.` makes the name absolute and avoids search-list expansion in that command. It does not force the application to behave the same way.

![A DNS diagnostic decision tree separating local or recursive cache state, authoritative disagreement, negative answers, and UDP truncation with TCP fallback.](/imgs/blogs/dns-the-distributed-database-your-request-starts-in-6.webp)

The decision tree is intentionally ordered by boundaries. First ask the same recursive resolver as the client. Next ask the relevant authoritative servers directly. Then inspect whether answers differ by record type, time, or transport. Only after those comparisons should you flush a cache or change a firewall rule. A cache flush can make a symptom disappear while destroying evidence about which cache held the old value. A broad firewall change can hide a narrower EDNS or TCP failure. Preserve the response and identify the one variable you intend to move.

A compact probe sequence is:

```bash
# Replace RESOLVER and NAME with values observed from the failing client.
dig @RESOLVER NAME. A +comments +noall +answer +authority +additional
dig @RESOLVER NAME. AAAA +comments +noall +answer +authority +additional
dig @RESOLVER NAME. A +tcp +comments +noall +answer +authority
dig NAME. NS +trace
```

The first two queries distinguish record-type results at one recursive server. The third changes only transport for the A question. The trace maps delegation from the vantage point of the diagnostic host. To ask authority directly, select a nameserver learned from the trace and use `dig @AUTHORITY NAME. A +norecurse +comments +noall +answer +authority`. If the authority returns a positive answer while the recursive resolver returns NXDOMAIN with an SOA, check negative-cache TTL. If both return the same answer but the application sees another address, inspect process and operating-system caches. If TCP succeeds and UDP fails, inspect truncation, EDNS size, fragmentation, and path filtering. If UDP succeeds and TCP fails, inspect TCP/53 reachability and server support.

Do not confuse `dig` output with an end-to-end application guarantee. It can prove what one DNS server returned to one client at one moment. It cannot prove that an application used that DNS server, that the returned IP accepted a connection, or that TLS selected the expected certificate. Continue down the [path map and request ladder](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) once the name step is bounded. For Kubernetes clients, search domains and `ndots` can multiply the questions before the absolute name is tried; the [Kubernetes DNS post](/blog/software-development/networking/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout) owns that specific client-side behavior. For service registry and xDS systems, the [service discovery control-plane post](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds) compares the staleness and availability trade-offs above this wire mechanism.

The rule of thumb is to classify the failure by response, not by user-facing text:

| Observed result | What is established | What remains open |
| --- | --- | --- |
| `NXDOMAIN` from recursive server | That server currently reports the name absent | Whether authority has changed, and how long a negative cache may remain |
| `NOERROR` with no requested type | No RRset of that type appeared in this response | Whether another type, CNAME target, or policy answers the client |
| `SERVFAIL` | The server failed to complete this question | Upstream reachability, DNSSEC validation, server state, or policy |
| Timeout | No usable reply reached this client in time | Packet loss, blocked transport, server overload, or wrong destination |
| `TC=1` | The UDP response was truncated | Whether the client retried over TCP and whether that retry succeeded |
| Positive answer | This server supplied an RRset | Whether the client used it and the destination accepted a connection |

The `Source` column rule applies to tables with numerical observations; this table is a qualitative interpretation of DNS response states, grounded in [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035.html), [RFC 2308](https://www.rfc-editor.org/rfc/rfc2308.html), and [RFC 7766](https://www.rfc-editor.org/rfc/rfc7766.html). It is a triage aid, not a substitute for those protocol definitions. In particular, a timeout is no DNS response at all, so it must not be translated into NXDOMAIN. That distinction is a common source of bad retries and bad incident communication.

A production measurement should preserve vantage point. Record whether the probe ran in a pod, on the node, in a workstation, or outside the network. Record its resolver address. Record whether the query was UDP or TCP. If possible, run the same explicit question from two independent networks and two recursive resolvers. A disagreement is evidence that narrows the problem, not proof that the answer you prefer is globally correct. Ask authority and inspect delegation before deciding which answer is stale.

## Delegation mistakes and the shape of partial failure

An operator can update the child zone and still leave the parent pointing at the wrong nameservers. This is a different failure from a stale A record. A resolver learns where to find a zone from the parent delegation. If the parent NS RRset still names an old authority, the new authority's correct record may never be queried by that resolver. The first comparison should therefore be parent versus child, not a dashboard that merely says a record exists in the provider's zone editor. Ask the parent for the delegation, then query every listed child nameserver directly. Observe the `aa` flag, SOA, and target RRset at each.

Consider a controlled illustrative migration. At the parent, `example.com` is delegated to `ns-old.example.net` and `ns-new.example.net`. The operator updates the A RRset only on `ns-new`. One resolver chooses the old server and returns the old address; another chooses the new server and returns the new address. TTL alone does not explain the split, because the authoritative servers themselves disagree. The operative measurement is not "what did my nearest recursive resolver return?" but "what did each authoritative server in the current delegation return when asked the same question?" The remedy is to make the authoritative set consistent and to ensure the parent delegation matches the intended set. Avoid claiming that the first answer you prefer is correct without checking the published control plane.

The same principle applies to a **lame delegation**, where a listed server does not serve the delegated zone authoritatively. A resolver may try another server and still succeed, so a single successful lookup does not prove every authority is healthy. Failure may surface only when the resolver chooses the lame server, when another server is down, or when retries exceed a deadline. `dig +trace` can reveal the referral, and explicit `dig @EACH_NS name. A +norecurse` queries can expose the inconsistent authority set. Record both the parent NS names and the IPs you actually queried. Nameservers can have multiple addresses, so one hostname may hide partial reachability.

Glue deserves its own sanity check. If an in-domain nameserver's glue address at the parent is wrong, cold resolvers may have trouble reaching the child even though the child zone lists the right address for that nameserver. A resolver with cached usable address data might continue to work, creating a warm-versus-cold split. That is why the initial referral section matters. The additional-section address in a referral is a bootstrap aid; compare it with the current nameserver addresses and with actual reachability. [RFC 9471](https://www.rfc-editor.org/rfc/rfc9471.html) specifies when glue should be supplied with referrals. A missing or truncated additional section should be read with the message size and `TC` flag, not treated as proof of a broken zone from one packet.

Delegation changes can also create a double-TTL problem. Parent NS RRsets have cache lifetimes, and child address RRsets have their own cache lifetimes. A resolver can hold an old delegation while another has a new one. The exact interval is bounded by the TTLs each resolver fetched, subject to resolver policy and network conditions. If a migration requires continuity, serve compatible answers from both old and new authorities during the overlap. This advice follows directly from the distributed cache model; it is not a claim that DNS offers an atomic handover. Change management for the wider service belongs in the [service discovery post](/blog/software-development/microservices/service-discovery-and-load-balancing), but the wire-level test is simple: query all authorities before and after changing the parent.

### Why a public resolver is only one witness

Public recursive resolvers are useful for an outside vantage point, but each is its own cache and policy engine. A response from one resolver is not "the internet's answer." Even two queries to the same public service can land on different anycast locations or backend caches. That can alter remaining TTL and transient behavior. When a customer reports a DNS problem, record the actual resolver that customer used if possible. Query authoritative servers to establish current published data. Then compare independent recursors to learn how widely the view has propagated. This is a three-way comparison, not a vote where the majority automatically wins.

Some environments intercept DNS requests or route them through forwarders. A `dig @1.1.1.1` result may not mean the application uses that path. Containers commonly point at a local address that forwards elsewhere. Browser secure-DNS settings may bypass the operating-system resolver. `getent hosts` exercises a host's name-service configuration, which can include `/etc/hosts` and other sources, while `dig` directly constructs DNS questions. If those disagree, inspect the resolver path before touching public authority. The exact cache stack is the subject of the [next DNS article](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers); the diagnostic rule here is to label the query mechanism in every observation.

## DNSSEC changes the failure surface without changing the walk's purpose

DNSSEC adds authenticated records and signatures so a validating resolver can check whether data is consistent with a chain of trust. It does not encrypt DNS questions, and it does not make a service healthy. It affects this post in two concrete ways. First, DNSSEC material can enlarge responses, exercising EDNS, truncation, and TCP fallback. Second, validation failure can appear to a client as `SERVFAIL` even when an authoritative server returns data. A `SERVFAIL` in that case is not evidence that a name is absent. The resolver could not deliver a validated answer under its policy.

The DNSSEC chain follows the same hierarchy in spirit: parent and child records establish a trust relationship, and the resolver validates signed RRsets. We do not need to reimplement DNSSEC to debug the initial symptom. Ask whether the failure is type-specific, resolver-specific, and transport-specific. Compare a validating recursive resolver with authoritative answers. A direct authoritative response tells you what the zone serves, but it does not by itself prove that a validating recursive resolver should accept it. Validation diagnostics need DNSSEC-specific tooling and careful interpretation. [RFC 4033, March 2005](https://www.rfc-editor.org/rfc/rfc4033.html) introduces the security model, and [RFC 4035, March 2005](https://www.rfc-editor.org/rfc/rfc4035.html) defines protocol changes.

This is an important boundary for incident communication. If one resolver says `SERVFAIL` and a direct authoritative query shows an A record, the incident is not "DNS propagated slowly" by default. A stale cache, failed validation, upstream timeout, and transport problem are all still candidates. Capture the resolver's extended error details when available, then test the specific branch. [RFC 8914, September 2020](https://www.rfc-editor.org/rfc/rfc8914.html) defines Extended DNS Errors that can add context without changing the basic response code. Support varies, so the absence of such detail is not proof of anything. The goal is to make the next measurement discriminate between candidate causes.

DNSSEC is also a reason to test TCP/53 in a real deployment. A small unsigned A lookup can succeed over UDP, and a larger signed response can require different packet handling. Passing the first does not certify the second. Record `MSG SIZE rcvd`, the EDNS advertised size, the `TC` flag, and whether an explicit TCP query succeeds. This is a narrower and more useful finding than "DNSSEC broke the network." When the network path is the issue, fix the path; when a signature or chain is wrong, fix the zone's signing and delegation state. The same client symptom can arise at either layer.

## Timing DNS without inventing a universal lookup cost

The latency ladder is a measurement frame, not a benchmark chart. A cold lookup can involve multiple upstream exchanges. A warm lookup can be answered by a local process cache or nearby recursive server. Several upstream queries may overlap, and a resolver may choose a different server after a timeout. Adding "root RTT + TLD RTT + authority RTT" is a useful explanatory approximation for a serial cold walk, but it is not an exact equation for all implementations. Label it as a model and verify with resolver logs or a packet capture before attributing wall-clock time.

For an illustrative serial path, assume the resolver takes 12 ms to get a root referral, 15 ms for a TLD referral, and 18 ms for an authoritative answer, with no retries and negligible local processing. The modeled DNS time is ${12}+15+18=45$ ms. If a warm cache answers in 2 ms, the modeled difference is 43 ms for this one lookup. Those values are fabricated solely as arithmetic inputs, not reported measurements or typical internet times. A real resolver might already have root and TLD delegation data, might query over a different route, or might wait on a timeout that dwarfs these values. Use the model to ask where time could accumulate, not to promise a measured p99.

The practical command is `curl -w` with named fields. For a fresh request, compare `time_namelookup`, `time_connect`, and `time_appconnect`, and inspect whether an address was actually resolved and a connection attempted. Repeated curl invocations may still interact with OS-level caches, while requests within one process can reuse connections. If you need to force a destination to isolate later phases, curl's `--resolve` can supply a hostname-to-address mapping while preserving the hostname for HTTP and TLS. Use that only as a diagnostic experiment; it bypasses the very DNS behavior you are investigating. If a request succeeds with `--resolve` but fails without it, DNS or address selection becomes more plausible. It does not by itself identify which resolver, cache, or delegation failed.

The simplest monitoring split is two independent probes: a direct DNS question to the intended resolver and a full application request. The DNS probe records question, answer type, status, TTL, server, and transport. The application probe records phase timings and destination IP. If the first fails, the later path may never run. If the first succeeds and the second fails, keep moving along the ladder. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) is where to decide alert policy and retention; the networking post gives the exact evidence fields that make the alert actionable.

## Run it yourself

### Question

Can one change in the client's advertised UDP payload size cause the same DNS question to use a different transport path? The experiment compares a normal EDNS-capable UDP query with a legacy-sized UDP query, then explicitly verifies that TCP can carry the full answer. It demonstrates the `TC`/fallback mechanism when the chosen live answer is large enough. Because public DNSSEC RRsets change, the result has a conditional expected state rather than a fabricated fixed byte count.

### Preconditions

Use a Linux host or privileged Linux VM with a recent BIND `dig`. This read-only lab does not need network namespaces or root. The [series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) explains the Linux `netlab` environment used for mutating packet experiments; we need only outbound DNS access here. Confirm your organization's policy permits queries to a public recursive resolver. The commands below use Cloudflare's public resolver at `1.1.1.1`; replace it with an approved resolver if necessary. You need working UDP and TCP port 53 to that resolver. The zone chosen for a large response may change its DNSSEC keys or server behavior, so first inspect the size and status instead of relying on a fixed value.

```bash
set -euo pipefail
command -v dig
DIG_VERSION=$(dig -v 2>&1)
printf 'dig version: %s\n' "$DIG_VERSION"
RESOLVER=1.1.1.1
NAME=org.
dig @"$RESOLVER" "$NAME" DNSKEY +dnssec +time=2 +tries=1 \
  +comments +stats +noall +answer +authority
```

Read the response `status`, `flags`, `MSG SIZE rcvd`, and the presence of `DNSKEY` records. `DNSKEY` with `+dnssec` is chosen because signed key responses can be larger than a small A response, but there is no promised exact size. If this resolver refuses the query or local policy blocks it, use an approved recursive resolver. Do not infer a protocol failure from an unavailable lab destination.

### Baseline

Run the EDNS-capable UDP form and save a copy of the output. `+bufsize=1232` advertises a UDP payload size of 1,232 octets. That value is a lab setting, not a universal path recommendation. The advertised size appears in the OPT pseudo-section when EDNS is negotiated. This command changes no server state.

```bash
set -euo pipefail
RESOLVER=1.1.1.1
NAME=org.
dig @"$RESOLVER" "$NAME" DNSKEY +dnssec +bufsize=1232 \
  +time=2 +tries=1 +comments +stats > /tmp/dns13-edns.txt
rg 'status:|flags:|EDNS:|udp:|MSG SIZE.*rcvd:|SERVER:' /tmp/dns13-edns.txt
```

Read `status`, `flags`, `udp:`, `MSG SIZE rcvd`, and `SERVER`. Expected: `status: NOERROR` and a positive response size in bytes if the resolver and network answer. The `udp:` field should reflect the EDNS negotiation; the actual received size can be smaller than 1,232 and varies with the zone's key material. The command's elapsed query time is an observation, not a benchmark. One public query cannot establish a stable performance difference.

### Apply one change

Change only the advertised UDP payload budget to the legacy 512-octet size while keeping the resolver, name, type, and DNSSEC request fixed. `dig` may automatically retry over TCP after it sees `TC=1`. Preserve the complete output because the diagnostic line about retry is itself evidence. Some resolver implementations may choose a response that fits, or may handle this exact EDNS setting differently; that is why we read fields instead of asserting a guaranteed truncation.

```bash
set -euo pipefail
RESOLVER=1.1.1.1
NAME=org.
dig @"$RESOLVER" "$NAME" DNSKEY +dnssec +bufsize=512 \
  +time=2 +tries=1 +comments +stats > /tmp/dns13-small.txt
rg 'status:|flags:|Truncated|retrying in TCP|EDNS:|udp:|MSG SIZE.*rcvd:|SERVER:' \
  /tmp/dns13-small.txt
```

### Compare

The comparison examines the transport decision, not a single timing sample. If the answer cannot fit the smaller UDP budget, expect a truncated response and a TCP retry, with the full final answer potentially larger than 512 octets. If no truncation occurs, the observed answer fit or the server chose another behavior; pick another DNSSEC-signed name and record it rather than declaring the protocol wrong. Verify TCP explicitly with the same question. `+tcp` sends the DNS question over TCP from the start.

```bash
set -euo pipefail
RESOLVER=1.1.1.1
NAME=org.
dig @"$RESOLVER" "$NAME" DNSKEY +dnssec +tcp \
  +time=2 +tries=1 +comments +stats > /tmp/dns13-tcp.txt
rg 'status:|flags:|MSG SIZE.*rcvd:|SERVER:' /tmp/dns13-tcp.txt
```

Read the `status`, any truncation or retry line, and `MSG SIZE rcvd` across the three files. The controlled variable between baseline and treatment is the advertised UDP capacity. The explicit TCP run checks reachability. Expected qualitative result: all successful full answers have `NOERROR`; if the 512-octet path truncates, TCP obtains the full response. Do not expect a fixed millisecond gap. Public resolver load, route RTT, caches, and key material vary. If `+tcp` times out while UDP works, the experiment reveals a transport reachability problem on your path, not a reason to increase application retries.

### Reset

No resolver, namespace, interface, route, qdisc, or firewall was changed. Remove only this lab's local captures, leaving other `netlab` state untouched.

```bash
rm -f /tmp/dns13-edns.txt /tmp/dns13-small.txt /tmp/dns13-tcp.txt
```

The production translation is read-only: query your approved resolver with the exact failing name and type using default UDP and `+tcp`, then compare `TC`, retry behavior, response status, and answer size. Capture only the records needed for diagnosis. DNS questions can expose internal service names; do not paste them into a public issue or packet trace. A packet capture can contain further sensitive data, so scope and retain it under the same rules as other network evidence.

## What to carry into the next incident

The fastest reliable move is to write the question precisely and locate the responding boundary. The DNS walk is a sequence of authority handoffs. Root and TLD referrals identify where the resolver should ask; the authoritative server supplies zone data; recursive and local caches reuse it while allowed. A positive answer can expire, a negative answer can persist, and a large answer can move from UDP to TCP. Those mechanisms explain why two clients can disagree and why a service can be healthy while a fresh request cannot find it.

For an on-call engineer, the order is practical. Confirm the application asked the name you think it asked. Query its recursive resolver and record status, flags, server, type, and TTL. Walk delegation, then ask authority. Compare UDP with TCP only if response size, truncation, or transport reachability is plausible. Continue to TCP, TLS, proxy, and application measurements when DNS is bounded. The [capstone network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) puts this decision into the full request path, while [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) owns how to make those fields visible during a real incident.

## Further reading

- [RFC 1034: Domain names, concepts and facilities](https://www.rfc-editor.org/rfc/rfc1034.html), November 1987. The delegation and recursive versus iterative model.
- [RFC 1035: Domain names, implementation and specification](https://www.rfc-editor.org/rfc/rfc1035.html), November 1987. Message fields, flags, record format, and classic transport.
- [RFC 2308: Negative caching of DNS queries](https://www.rfc-editor.org/rfc/rfc2308.html), March 1998. Cacheable NXDOMAIN and NODATA semantics.
- [RFC 6891: Extension mechanisms for DNS](https://www.rfc-editor.org/rfc/rfc6891.html), April 2013. EDNS(0) and UDP payload advertisement.
- [RFC 7766: DNS transport over TCP](https://www.rfc-editor.org/rfc/rfc7766.html), March 2016. TCP support and truncation recovery.
- [RFC 9156: QNAME minimisation](https://www.rfc-editor.org/rfc/rfc9156.html), November 2021. Why real upstream questions may differ from the simple full-name walk.
