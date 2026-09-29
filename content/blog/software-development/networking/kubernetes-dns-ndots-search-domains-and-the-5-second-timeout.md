---
title: "Kubernetes DNS: ndots, Search Domains, and the Five-Second Timeout"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Trace a Pod lookup from search suffixes through Service DNAT, distinguish query amplification from a dropped DNS packet, and choose a fix at the right layer."
tags:
  [
    "networking",
    "distributed-systems",
    "kubernetes",
    "dns",
    "coredns",
    "conntrack",
    "nodelocal-dnscache",
    "service-discovery",
    "linux-networking",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-1.webp"
---

An application calls `getaddrinfo("api.example.com")` from a Kubernetes Pod. Most calls finish quickly. A few wait close to five seconds before the HTTP client can even send a SYN. The endpoint is healthy. CoreDNS CPU is quiet. The dashboard says “DNS latency,” but that label hides two different mechanisms: the resolver might be asking several names before the useful one, or the kernel might have dropped one of the UDP questions on the way to the DNS Service.

![The recurring request latency ladder with DNS as the active segment](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-1.webp)

The ladder above is the mental model. If name resolution owns the wall clock, first count the questions the application generates. Then locate the packet that did not receive an answer. A five-second step in a graph is a clue about a resolver retry timer, not proof that CoreDNS spent five seconds computing an answer. The distinction matters because the fixes operate at different layers.

This post follows one Linux Pod with `dnsPolicy: ClusterFirst` and an iptables-based Kubernetes Service path. That scope is deliberate. A Pod using a different resolver library, a local DNS agent, a different kube-proxy mode, or a modern kernel may take a different route. We will read the actual Pod configuration before generalizing. For the recursive and authoritative part of DNS, start with [the DNS walk](/blog/software-development/networking/dns-the-distributed-database-your-request-starts-in). For cache TTLs and stale answers, use [the production DNS companion](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers). This article owns the client search path and the packet's trip through the cluster Service.

## 1. Two clocks inside one lookup

A hostname lookup is not necessarily one DNS packet. The application passes a string and an address-family request to a library. The library reads resolver policy, constructs candidate DNS names, and can request both A and AAAA records. Each candidate may require a round trip. A missing candidate can return NXDOMAIN promptly, which costs traffic but usually little wall time. A query or reply that disappears can force a retry timer, which costs whole seconds. Keep those two clocks separate.

For this article, a *candidate* is one fully expanded DNS owner name, such as `api.example.com.default.svc.cluster.local.`. A *question* is a candidate plus a record type, such as A or AAAA. A *packet* is a transport datagram carrying a question or answer. One `getaddrinfo` call can therefore cause multiple candidates, multiple questions, and retries of a lost question. Counting “DNS lookups” without defining which unit you mean creates misleading incident reports.

A quick diagnostic is to compare `time_namelookup` with `time_connect` in `curl -w`. That output does not expose every DNS question, but it tells you whether the delay occurred before the connection attempt. If the application reuses a connection, `time_namelookup` may be zero or absent as a meaningful cost. If the application caches names itself, a repeated request may never reach the Pod resolver. Measure a cold lookup when testing DNS, and do not infer DNS health from a warm HTTP request.

```bash
curl -sS -o /dev/null \
  -w 'lookup=%{time_namelookup} connect=%{time_connect} total=%{time_total}\n' \
  https://api.example.com/
```

The fields are cumulative seconds from the start of the transfer. `time_connect` includes earlier lookup time. To estimate the TCP connection portion, subtract `time_namelookup` from `time_connect`; do not add those two printed fields. On HTTPS, the analogous TLS interval ends at `time_appconnect`, which also includes earlier phases. This is the same timing discipline introduced in [the request path map](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url).

A five-second step can originate elsewhere: a client timeout, a load balancer idle timer, an upstream retry, or a TCP retransmission. Require packet evidence before naming the DNS conntrack race. The historical XING case below is a useful warning: their one- and three-second spikes were SYN retransmissions, even though DNS was an early suspect.

## 2. Read the Pod resolver before explaining it

Kubernetes documents that a `ClusterFirst` Pod receives a resolver configuration with the Pod namespace, Service zone, and cluster domain on the search path. Its example has `options ndots:5`. The effective file can also inherit or merge deployment-specific settings, so the file inside the affected container is the evidence, not a chart of defaults. The Kubernetes [DNS for Services and Pods documentation](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) gives the file shape and the `dnsConfig` override interface.

```conf
nameserver 10.32.0.10
search default.svc.cluster.local svc.cluster.local cluster.local corp.example
options ndots:5
```

This is an illustrative four-suffix configuration, not a captured production file. The last suffix represents a site-specific search domain. In your cluster, copy the file from the affected Pod and count its actual search entries. The nameserver is the DNS Service ClusterIP in the basic path, or a node-local address if NodeLocal DNSCache is configured. A Pod that uses `dnsPolicy: Default` inherits node behavior instead; a host-networked Pod may need `ClusterFirstWithHostNet`. Those paths cannot be diagnosed by assuming this example.

`ndots` controls the *ordering* of attempts. In the glibc [resolv.conf manual](https://man7.org/linux/man-pages/man5/resolv.conf.5.html), a name containing fewer dots than the threshold is attempted with search suffixes before the initial absolute query. At the default glibc threshold of one, `api.example.com` has enough dots for an initial absolute attempt. With `ndots:5`, its two dots do not, so a search path is tried first. A final dot makes the name explicitly absolute: `api.example.com.`. The final dot is the DNS root label written visibly. It is not an extra character in the intended DNS name.

The threshold is not a promise that five questions will always be sent. The count depends on search-list length, whether an earlier candidate succeeds, resolver implementation, A and AAAA policy, cache hits, retries, and whether the name is already absolute. The useful shorthand is “up to the search candidates plus the bare name in this scenario,” not “Kubernetes sends exactly five packets.” That precision will save you from treating a healthy NXDOMAIN stream as a five-second outage.

The glibc manual also records the default `timeout` value as five seconds and default `attempts` as two. It explicitly warns that one resolver API call need not equal one timeout. A five-second plateau is consistent with a missing response followed by retry, but timing alone cannot identify the dropping component. A library may use its own resolver and timers, and a Pod may set `timeout:` or `attempts:` explicitly. Inspect the container image and effective resolver behavior before using glibc timing as an explanation.

## 3. Why one name can become five candidate names

![Search expansion of a two-dot external name under ndots five](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-2.webp)

Take the illustrative file above. The application asks for `api.example.com`. It has two dots, fewer than `ndots:5`. If no search-expanded candidate succeeds, the resolver can try these names in order:

| Order | Candidate owner name | Why it exists | Source |
| --- | --- | --- | --- |
| 1 | `api.example.com.default.svc.cluster.local.` | Pod namespace search suffix | Derived from the illustrative `search` line and glibc search behavior |
| 2 | `api.example.com.svc.cluster.local.` | Service zone search suffix | Derived here |
| 3 | `api.example.com.cluster.local.` | Cluster domain search suffix | Derived here |
| 4 | `api.example.com.corp.example.` | Site-specific search suffix | Derived here |
| 5 | `api.example.com.` | Final absolute candidate | Derived here |

The table is a candidate-name derivation, not a packet capture. A and AAAA questions may make the packet count larger. Negative answers can stop or continue the search according to resolver behavior. A name collision inside an early search suffix can return a valid but unintended address, which is a correctness problem more serious than a few unnecessary questions. A trailing-dot absolute name avoids suffix expansion for that name, but some application libraries normalize hostnames before calling the resolver. Test the actual client library and URL parser rather than assuming punctuation survives the whole stack.

The inverse matters for cluster Services. A short name such as `orders` in the `default` namespace needs the search path to become `orders.default.svc.cluster.local.`. A name such as `orders.payments` relies on search expansion to find `orders.payments.svc.cluster.local.`. Setting `ndots:1` changes the first query for `orders.payments`: the resolver tries `orders.payments.` before the search list. It can still fall back to the Service name after a negative answer in many resolver implementations, but that costs an extra question and may run through upstream DNS. If a public or private zone unexpectedly owns `orders.payments.`, it may even resolve to the wrong place. This is why a global `ndots:1` recommendation is incomplete.

The arithmetic for the example is simple. Let $S$ be the number of search suffixes, and suppose each candidate is tried, none of the search-expanded candidates exists, there are no cache hits or retries, and the client requests one record type. The candidate count is the explanatory model $Q_{\mathrm{candidates}} = S + 1$. For $S = 4$, $Q_{\mathrm{candidates}} = 5$. If the client requests both A and AAAA for each candidate and sends each question separately, the illustrative upper count is $Q_{\mathrm{questions}} = 2(S+1) = 10$. That is an upper bound for this assumed path, not a protocol rule. The first successful answer, address-family policy, negative caching, or a library's resolver strategy changes the result.

Suppose an application makes 200 uncached external-name resolution calls per second with this exact four-suffix path. Under the one-type assumptions, it could emit $200 \times 5 = 1000$ questions per second instead of 200. Under the two-type assumption, it could emit up to 2000 questions per second. Those are derived workload scenarios, not a CoreDNS benchmark or production measurement. The operational point is that search expansion scales query traffic even when all DNS servers respond promptly. Reducing that traffic can lower load and shrink the number of opportunities for a separate packet-loss mechanism, but it does not repair a kernel race.

The animated sweep below follows those five candidate names. Motion matters here because the resolver advances only after each preceding candidate fails to answer with a usable result. The final absolute name is not the first attempt in this configuration.

<figure class="blog-anim">
<svg viewBox="0 0 920 260" role="img" aria-label="The resolver tries four search-suffixed names, then the absolute name" style="width:100%;height:auto;max-width:920px">
<style>
.dns-a-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.dns-a-txt{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dns-a-sub{font:14px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dns-a-head{font:600 21px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.dns-a-sweep{fill:var(--accent,#6366f1);opacity:.23}
@keyframes dns-a-step{0%,17%{transform:translateX(0)}20%,37%{transform:translateX(176px)}40%,57%{transform:translateX(352px)}60%,77%{transform:translateX(528px)}80%,100%{transform:translateX(704px)}}
.dns-a-moving{animation:dns-a-step 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.dns-a-moving{animation:none;transform:translateX(704px)}}
</style>
<text class="dns-a-head" x="20" y="39">api.example.com with ndots:5</text>
<rect class="dns-a-bg" x="20" y="90" width="160" height="100" rx="12"/>
<rect class="dns-a-bg" x="196" y="90" width="160" height="100" rx="12"/>
<rect class="dns-a-bg" x="372" y="90" width="160" height="100" rx="12"/>
<rect class="dns-a-bg" x="548" y="90" width="160" height="100" rx="12"/>
<rect class="dns-a-bg" x="724" y="90" width="160" height="100" rx="12"/>
<rect class="dns-a-sweep dns-a-moving" x="20" y="90" width="160" height="100" rx="12"/>
<text class="dns-a-txt" x="100" y="130">Search 1</text>
<text class="dns-a-sub" x="100" y="158">+ suffix 1</text>
<text class="dns-a-txt" x="276" y="130">Search 2</text>
<text class="dns-a-sub" x="276" y="158">+ suffix 2</text>
<text class="dns-a-txt" x="452" y="130">Search 3</text>
<text class="dns-a-sub" x="452" y="158">+ suffix 3</text>
<text class="dns-a-txt" x="628" y="130">Search 4</text>
<text class="dns-a-sub" x="628" y="158">+ suffix 4</text>
<text class="dns-a-txt" x="804" y="130">Absolute</text>
<text class="dns-a-sub" x="804" y="158">api.example.com.</text>
<text class="dns-a-sub" x="460" y="226">Configured search suffixes vary by Pod; this sequence assumes four.</text>
</svg>
<figcaption>With four configured suffixes and ndots:5, the resolver walks search names before trying the absolute name.</figcaption>
</figure>

### What `dig` and `getent` prove

A direct `dig api.example.com.` sends an explicit absolute query and says little about how `getaddrinfo("api.example.com")` behaves. `dig +search` can expose suffix handling, but it is still `dig`'s resolver behavior, not your application's library. `getent ahostsv4` uses the operating system's name service path on a glibc image and is closer to many application calls, though it asks only for IPv4 addresses. Python's `socket.getaddrinfo` is useful for exercising the library of the container image. Go binaries may use cgo or the pure Go resolver; JVM and Node.js callers may have their own caching and lookup choices. The packet trace settles what actually left the Pod.

Check `/etc/nsswitch.conf` too. `hosts:` may include `files`, `mdns`, or other modules before `dns`. A matching `/etc/hosts` entry means no DNS query is required. A local cache can answer without an upstream packet. If you count CoreDNS metrics and see fewer queries than the model, that may be a cache doing its job, not a failed experiment.

## 4. The DNS Service is a NAT path

![The Pod to ClusterIP DNS path and the node-local alternative](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-4.webp)

In the historical path analyzed by Weave, the Pod sends a UDP DNS packet to the `kube-dns` Service ClusterIP. `kube-proxy` in iptables mode has installed NAT rules that select a CoreDNS or kube-dns endpoint and replace the packet's destination. This is destination NAT, or DNAT. The DNS response must be translated back so the Pod sees the Service address it queried, rather than a surprising endpoint address. Connection tracking, usually abbreviated `conntrack`, stores the state needed for that reverse mapping.

The Service IP is a virtual destination, not a server process sitting on that address. That distinction explains why a packet can disappear before CoreDNS sees it while the application reports a DNS timeout. If you only look at CoreDNS request duration, the lost packet is invisible: the server never received that question. If you only look at application duration, the kernel drop is invisible: you see the retry gap but not where the first attempt died. Compare packet traces on both sides of the translation and the node's conntrack counters.

The [Weave analysis, originally published in 2018](https://lambda.lt/blog/2018/racy_conntrack.html), uses Kubernetes v1.11.0 and Linux kernel v4.17 examples. It shows an original tuple with Pod source IP and port targeting the DNS ClusterIP, and a reply tuple whose source is the chosen DNS endpoint. A tuple here identifies protocol and two IP:port pairs. It is not a DNS transaction ID. Two DNS questions can have different transaction IDs while sharing the same UDP socket's network tuple. Conntrack is tracking the network flow, not inspecting whether the DNS question asks for A or AAAA.

The same analysis names `nf_conntrack_in`, NAT rule traversal, `get_unique_tuple`, packet mangling, and `__nf_conntrack_confirm` as stages. Treat the diagram in that article as a simplified explanation for its 2018 kernel lineage, not a timeless statement that every current Linux kernel has the identical bug. The Service implementation can also differ. IPVS did not remove conntrack from Weave's tested path, but a current eBPF dataplane or node-local resolver must be inspected on its own terms. The safe diagnostic is to discover the actual nameserver, Service proxy mode, kernel version, and packet path first.

### The tuple in a concrete example

Use addresses only as an explanatory example. A Pod at `10.40.0.17` sends UDP from port `53378` to DNS Service `10.96.0.10:53`. DNAT chooses endpoint `10.32.0.6:53`. The original-direction tuple is `10.40.0.17:53378 -> 10.96.0.10:53`; the reply-direction tuple identifies `10.32.0.6:53 -> 10.40.0.17:53378`. Those addresses and ports come from the Weave article's example, not this cluster. The kernel must associate both directions so a reply from `10.32.0.6` can be presented to the client consistently with its query to `10.96.0.10`.

The setup is fragile when two first packets for the same unconfirmed flow are processed concurrently. UDP has no handshake that pre-creates a confirmed conntrack entry. Each packet can arrive at the lookup stage before the other's new entry is confirmed. Each can then carry its own tentative state into NAT selection. If the tentative entries clash when confirmed, one packet can be dropped. The historical Weave post describes three variants: same tuples, a changed reply tuple during NAT allocation, and different DNS endpoints chosen by Service rules. Do not collapse those into a single “two packets picked one port” story; the exact collision depends on the path through conntrack and NAT.

## 5. The five-second conntrack race, step by step

![Two concurrent DNS questions race through DNAT and conntrack confirmation](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-3.webp)

Imagine a glibc caller that requests both IPv4 and IPv6 addresses. It may issue A and AAAA questions in parallel on the same UDP socket. Because DNS transaction IDs live inside the payload, the outer network tuple can be identical. On a cold flow, packet A and packet AAAA can both reach `nf_conntrack_in` before either newly created conntrack entry is confirmed. Both find no confirmed match. Both progress through NAT. At confirmation, one tentative entry loses a tuple clash and its packet is dropped. The winning question gets a normal response; the losing question appears to the client as silence. After its resolver timeout, the library retries, and the lookup can finish roughly one timeout interval later.

That is the mechanism Weave documented. It is conditional, not inevitable. Parallel A and AAAA alone do not guarantee a clash. The timing window, socket behavior, DNS endpoint choice, kernel fixes, and traffic level matter. Conversely, merely disabling IPv6 is not a satisfying platform fix: it changes application semantics, may not alter a non-glibc library as expected, and gives up IPv6 connectivity. It can be a constrained diagnostic experiment if the application's requirements allow it, but it is a poor blanket recommendation.

A clean way to distinguish the race from a slow DNS server is to pair evidence. Capture the outgoing A and AAAA packets from the Pod side. Capture arrivals at the DNS endpoint or observe server query counters. Watch `conntrack -S` for `insert_failed` movement during the same test interval. The Weave post calls `insert_failed` a useful indicator, not a unique proof. Other conntrack pressure or collisions can move the counter. Likewise, one missing packet in a capture could reflect capture placement, a CNI policy, or a NIC path. Triangulate rather than promote one counter to a root-cause verdict.

```bash
sudo conntrack -S
sudo tcpdump -ni any -tttt -vv 'udp port 53'
```

Run those on the affected node when you have permission, and scope captures by Pod IP and DNS destination in a real incident. `-i any` can show the same packet at multiple interfaces and directions, so count DNS transaction IDs and interface context rather than raw lines. The first command requires the `conntrack` userspace tool; the second requires packet-capture privileges. A CoreDNS Pod capture may need a debug container, and a host capture may see packets both before and after DNAT. Record where each capture was taken.

The five seconds come from the client's retry policy in the historical glibc-shaped path, not from a five-second Kubernetes DNS Service setting. The [glibc resolver manual](https://man7.org/linux/man-pages/man5/resolv.conf.5.html) gives five seconds as the default remote nameserver timeout and cautions that the total API call may differ. If the Pod sets `options timeout:2 attempts:2`, or if the application has its own deadline, the histogram will look different. A five-second plateau can be a strong lead; absence of that plateau does not rule out packet loss.

### What a healthy negative answer looks like

Search expansion commonly generates NXDOMAIN replies for names such as `api.example.com.default.svc.cluster.local.`. Those replies are evidence that the DNS server received and answered a question. A lost question generates no answer for that transaction ID before retry. In a packet trace, compare the question name, type, ID, source port, destination address, and timestamp. A burst of fast NXDOMAIN responses can coexist with one A or AAAA question that has a five-second gap. Calling all NXDOMAINs “DNS failures” obscures the actual missing packet.

CoreDNS's `kubernetes` plugin is authoritative for the cluster zone, while external names may be forwarded upstream by the CoreDNS configuration. A search-expanded external name under `cluster.local` can be rejected locally. The final `api.example.com.` may go through the forwarder. The traffic and timing implications depend on CoreDNS cache settings, upstream latency, and how the resolver treats negative responses. NodeLocal DNSCache can suppress repeated traffic to CoreDNS, but it does not change the application's candidate-name sequence unless the Pod resolver policy also changes.

## 6. Two historical cases, one family of kernel problems

![The Weave DNS race and XING SYN race compared without merging symptoms](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-5.webp)

### Weave: UDP DNS through a Service, 2018

Martynas Pumputis's [August 16, 2018 Weave write-up](https://lambda.lt/blog/2018/racy_conntrack.html) analyzed reports of Pod DNS lookups taking five seconds or more. Its examples used Kubernetes v1.11.0 and Linux v4.17. The relevant path was a UDP question to the kube-dns Service ClusterIP, selected endpoint DNAT, then conntrack confirmation. The post described three race variants and noted that a packet could be dropped when an unconfirmed entry clashed at confirmation. It pointed to `conntrack -S` and `insert_failed` as a useful symptom indicator. It also discussed kernel patches that addressed two variants at the time and a remaining endpoint-selection variant. The transferable lesson is to localize a DNS delay along the path: a server with low processing latency can still be unreachable to one packet because the drop happened on the node before the server.

A [Weave issue opened April 26, 2018](https://github.com/weaveworks/weave/issues/3287) reported intermittent five-second DNS delays and linked to the earlier XING analysis. That issue's user-reported frequency is specific to their environment and should not be treated as an industry failure rate. The engineering mechanism is what matters here. The post was written against a particular kernel and dataplane; use it to design tests, not to assert that every 2026 cluster retains every 2018 race.

### XING: TCP SYN through SNAT, February 2018

Maxime Lagresle's [XING engineering article dated February 22, 2018, preserved as a PDF](https://kangwoo.github.io/assets/pdf/kubernetes/A%20reason%20for%20unexplained%20connection%20timeouts%20on%20Kubernetes:Docker.pdf) investigated connection timeouts after moving applications to Kubernetes. Their published setup used Kubernetes 1.8, Ubuntu Xenial virtual machines, Docker 17.06, and Flannel 1.9.0 in host-gateway mode. Their application saw many slow requests delayed by one or three seconds. The team initially measured kube-dns directly, found it fast, then captured traffic at the container veth, bridge, and host interface. The first SYN appeared on the inner interfaces but not the outgoing host interface. A retransmitted SYN appeared later and left the host after source NAT. That was not the five-second UDP DNS symptom.

XING traced their case to a race in SNAT port allocation for concurrent connections to the same remote endpoint. Their PDF reports that forcing fully randomized NAT port selection dropped errors to zero in their test and nearly zero on live clusters. The precise outcome belongs to their environment, not to arbitrary CNI or current kernel versions. A June 2018 edit to that article notes a similar race can affect DNAT and ClusterIP traffic, including DNS. That edit connects the family of problems, but it does not retroactively turn XING's original SYN trace into a DNS capture. The lesson is methodological: compare the same packet at the inner and outer interface before blaming the server named in an application error.

| Case | Packet and translation | Published symptom | Evidence that located the fault | Source |
| --- | --- | --- | --- | --- |
| Weave, 2018 | UDP DNS to Service ClusterIP, DNAT to DNS endpoint | Some Pod lookups at five seconds or more | Conntrack race analysis and `insert_failed` indicator | [Weave analysis, August 2018](https://lambda.lt/blog/2018/racy_conntrack.html) |
| XING, 2018 | TCP SYN toward external endpoint, SNAT on host | Requests delayed one or three seconds | SYN seen on veth and bridge, absent on host egress until retry | [XING PDF, February 2018](https://kangwoo.github.io/assets/pdf/kubernetes/A%20reason%20for%20unexplained%20connection%20timeouts%20on%20Kubernetes:Docker.pdf) |

The table's dates and delays are source-specific. Neither is a prediction for your workload. They give two falsifiable capture patterns. A dropped DNS question should appear before the DNS server and then reappear as a DNS retry. A dropped SYN should appear before host egress and then reappear as a TCP retransmission. If the packet reaches the remote server and the server replies, you are investigating a different failure.

## 7. NodeLocal DNSCache changes the path

Kubernetes's [NodeLocal DNSCache documentation](https://kubernetes.io/docs/tasks/administer-cluster/nodelocaldns/) describes a node-resident caching agent deployed as a DaemonSet. A Pod sends DNS queries to a local listener. For a cache miss, the agent forwards toward cluster DNS. In the documented design, the local path avoids the usual Pod-to-DNS-Service iptables DNAT and conntrack path, reducing exposure to that race and to UDP DNS entries occupying the conntrack table. The agent can use TCP upstream to CoreDNS, which changes how entries are retired. This is a path change, not a faster DNS record format.

Check the effective Pod `nameserver` after deployment. The documentation notes that iptables-mode deployments can bind both the kube-dns Service IP and a local address in some configurations, while IPVS has different setup requirements. Do not infer installation from the existence of a DaemonSet alone. Query the same Pod before and after, inspect its resolver file, identify the listener, and confirm the cache receives metrics or packets. Managed clusters may provide a different local DNS implementation. Name the implementation you are measuring.

NodeLocal DNSCache has trade-offs. It adds an agent to every node, with its own rollout and health behavior. A node-local failure can affect every Pod on that node. Cached negative answers can reduce repeated misses, but cache TTL choices control how quickly changed records become visible. The local cache can reduce CoreDNS QPS from repeated queries, yet a cold miss still reaches upstream DNS. It cannot stop `getaddrinfo` from constructing four wrong suffix candidates and one right one. It may answer or negatively cache those candidates locally, making the search path cheaper, but the application still asked them. The distinction is visible if you compare application trace or local cache metrics with CoreDNS metrics.

Do not use “NodeLocal fixes DNS” as a substitute for diagnosis. It specifically helps the path documented above. If the five-second delay is an upstream resolver timeout, a broken CoreDNS forwarder, an application-level serial lookup, or an explicit client retry, moving the first hop may have little effect. If the Pod's DNS request is blocked by NetworkPolicy before it reaches the local listener, the local agent cannot answer. If you have confirmed a historical conntrack race on a supported kernel, bringing the kernel and dataplane to a maintained version is also important. A cache is a practical path mitigation, not a reason to leave a known kernel defect unreviewed.

## 8. Three fixes ranked by operational cost

![Three interventions ranked by scope and operational cost](/imgs/blogs/kubernetes-dns-ndots-search-domains-and-the-5-second-timeout-6.webp)

The ranking below is a decision aid, not a universal performance league table. “Cost” means rollout scope, blast radius, and maintenance burden for a team that owns the workload but may not own the cluster. First identify which mechanism your traces show. A change that reduces search expansion does not, by itself, eliminate a DNAT race. A node-local cache does not, by itself, remove the search list.

| Rank | Intervention | When it fits | What changes | Residual risk and rollback | Source |
| --- | --- | --- | --- | --- | --- |
| 1, workload scope | Use an absolute trailing-dot name where the client accepts it; or set Pod `dnsConfig.options` `ndots` after testing all internal names | External-name suffix expansion dominates and one team owns the callers | Candidate ordering or search use for that workload | A global low `ndots` can change short Service-name behavior; revert workload spec or client config | [glibc resolver manual](https://man7.org/linux/man-pages/man5/resolv.conf.5.html), [Kubernetes Pod DNS config](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) |
| 2, node scope | Deploy and verify NodeLocal DNSCache | DNS Service path or repeated queries burden nodes and CoreDNS | Pod reaches local cache; misses forward onward | Per-node agent health and cache staleness become operational concerns; roll back DaemonSet and DNS redirection with platform procedure | [Kubernetes NodeLocal DNSCache](https://kubernetes.io/docs/tasks/administer-cluster/nodelocaldns/) |
| 3, platform scope | Upgrade affected kernel or dataplane and verify conntrack behavior with the vendor's supported path | A packet capture and counters implicate a kernel translation race | Root packet-processing behavior | Needs node rollout and compatibility testing; rollback is a controlled node image or dataplane rollback | [Weave's kernel-fix discussion](https://lambda.lt/blog/2018/racy_conntrack.html) |

The table uses ordinal ranks, not measured dollars or latency. A team with a managed DNS add-on might find NodeLocal cheaper than changing dozens of clients. A cluster-wide kernel update may already be scheduled, reducing marginal cost. Conversely, a trailing dot may be impossible in an HTTP stack that treats it differently for TLS hostname verification or virtual-host routing. Keep the name used for DNS resolution, the HTTP `Host` header, and the TLS server name consistent with your client's documented behavior. Test those semantics before shipping a text replacement.

### Workload fix: explicit names and narrow `dnsConfig`

For an internal Service, prefer a fully specified Service name such as `orders.payments.svc.cluster.local.` when the application supports it. For an external endpoint, an explicit `api.example.com.` can bypass search expansion. This is narrow and easy to compare in a canary. But many application stacks pass the supplied hostname to certificate verification or HTTP authority logic. A trailing dot can affect those comparisons even when DNS resolves the same address. One safe pattern is to retain the application URL and set a tested resolver policy for its Pod rather than rewriting a URL blindly.

Kubernetes supports Pod-level `dnsConfig.options`. This example changes only the resolver threshold while keeping `ClusterFirst`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: resolver-canary
spec:
  dnsPolicy: ClusterFirst
  dnsConfig:
    options:
      - name: ndots
        value: "1"
  containers:
    - name: app
      image: python:3.12-slim
      command: ["sleep", "3600"]
```

This is a canary specification, not a recommendation to blanket-edit every Deployment. Run both external and internal name tests with the same application library. Verify `orders`, `orders.payments`, and a full Service name if your code uses them. A threshold of one can send a partially qualified internal name to the absolute DNS path first; a negative answer may then trigger search expansion. Count questions and inspect returned addresses. If the behavior is not acceptable, restore the original threshold for that workload.

### Node fix: local DNS with measurable health

If multiple workloads show the same DNS Service-path loss or CoreDNS receives more search traffic than it can comfortably handle, NodeLocal DNSCache is a platform-level change worth testing. Verify the agent listens on each node, Pods query the intended address, misses reach CoreDNS, and node-local cache hit and error metrics make sense. Measure the old and new path with the same workload. The claim to test is reduced remote DNS packets and avoidance of the Pod-to-Service DNAT path, not a promise of a particular median latency. The official Kubernetes page describes the architectural benefit; your cluster's CNI and DNS add-on decide the realized result.

A rollout should include cache behavior under record changes and agent restart. Record a baseline for successful resolution, NXDOMAIN volume, timeout count, CoreDNS QPS, node agent restarts, and conntrack counters. Then roll a small node pool. A drop in CoreDNS QPS is expected on cache hits; it does not prove that every application lookup disappeared. If an incident shows the local agent failing, compare a Pod on that node with a Pod on a healthy node before widening a rollout. For higher-level service-discovery choices and load-balancing policy, see [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing).

### Platform fix: maintain the packet path

If captures show packet loss at conntrack confirmation and the environment matches a known affected lineage, review a supported kernel and CNI upgrade. Weave's 2018 article documented fixes for two of its race variants and an unresolved third variant at publication time. That is a historical status report. It is not evidence that the same fixes are absent or present in your 2026 kernel. Identify the exact node kernel, kube-proxy or replacement dataplane, NAT rules, and vendor backports before deciding. Roll out a canary node, repeat the discriminating capture, and compare counters under controlled traffic. Do not transplant a 2018 `iptables` workaround to a current managed cluster without proving it applies.

Changing resolver timeouts can be an emergency latency bound, but it is not one of the three fixes. A shorter timeout makes a lost packet surface sooner and may increase retry traffic. It can also fail a query that would have succeeded under the old deadline. Treat it as a client policy with an application-level error budget, not as a repair for packet loss. [Timeouts and retry budgets](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns the broader policy, while this post tells you which packet disappeared.

## 9. What each metric can and cannot tell you

DNS investigations often fail because several counters with different denominators appear on the same dashboard. `getaddrinfo` calls are application operations. DNS questions are protocol operations. CoreDNS requests are server arrivals. Conntrack insert failures are node-wide kernel events. They should not be expected to match one for one. Build a measurement map before you add an alert or declare that one graph contradicts another.

Suppose an application makes one uncached `getaddrinfo` call for `api.example.com`, using the four-suffix example. It could construct five candidate names if all search-expanded names fail. It could ask A and AAAA for some or all candidates. A NodeLocal DNSCache may answer some of those questions from cache. CoreDNS might therefore receive only the final uncached question, or no question at all, while the application still made one call and the local cache processed several questions. The model `CoreDNS arrivals <= client questions` is useful only when the capture points and retry behavior are specified. A local agent can retry upstream, and other clients can share the same CoreDNS metrics. In production, compare tagged, time-bounded samples rather than global totals.

`coredns_dns_requests_total` is commonly used to count requests entering CoreDNS when the Prometheus plugin is configured. It cannot count packets dropped before CoreDNS. A counter increasing with NXDOMAIN responses can reveal suffix expansion, but you need the `qname` from logs or packets to know whether those negatives were expected search candidates. Query logs can be expensive and may expose sensitive names; enable them narrowly and for a bounded interval. Some CoreDNS configurations expose cache hit metrics and forwarder metrics. Their exact names and labels depend on the CoreDNS version and Corefile, so inspect `/metrics` and the active configuration rather than pasting a dashboard query from another cluster.

`conntrack -S` reports kernel connection-tracking statistics for the node's network namespace. The historical Weave analysis identifies `insert_failed` as a useful indicator for the confirmation clash it describes. Read a *delta* across the reproduction interval. If the count was already high, it could reflect previous workloads and tells you little about this lookup. If the count rises while the DNS question is missing downstream, the evidence becomes stronger. A stable count does not prove there was no packet drop: the packet may have been lost elsewhere, or the current implementation may expose different counters. Verify the local tool and kernel output format.

`nf_conntrack_count` and `nf_conntrack_max` can tell you whether the table is near its configured capacity, but table exhaustion is a different diagnosis from the race Weave analyzed. A full table can drop new tracked flows because there is no room. The race can occur when two unconfirmed entries clash even when there is room. Do not “fix” a correlated `insert_failed` delta by raising `nf_conntrack_max` without first testing whether the count was actually near the limit. That change consumes node memory and can merely postpone table pressure.

For a concrete observation plan, sample the kernel before and after a bounded application request burst, keep a packet capture with timestamps, and retain application durations for those same requests. If you have CoreDNS request metrics, snapshot them over the same interval. The three useful comparisons are: questions emitted versus questions arriving; arrivals versus answers returned; and application retry times versus packet retry times. A large gap in the first comparison points toward the transport path. A gap in the second points toward DNS server or upstream behavior. A gap between server response and application completion points back toward response delivery or application processing. These are directions for investigation, not automatic root-cause labels.

### Capture placement is part of the evidence

A packet captured on a Pod's `eth0` is a pre-Service view of the outgoing destination. A host capture can show a packet before and after NAT, perhaps on multiple interfaces with the same DNS transaction ID. A CoreDNS-side capture shows only what reaches that endpoint. If you capture only at the endpoint and see no packet, you cannot tell whether the application sent nothing or the node dropped it. If you capture only at the Pod and see a question followed by a retry, you cannot tell where the first packet vanished. Pair capture points or use a tracing tool that records the translation boundary.

A practical packet record should include timestamp, interface, Pod IP, UDP source port, DNS Service IP, selected endpoint IP when visible, DNS transaction ID, question name, and type. The DNS transaction ID helps pair a query with its answer, but NAT does not choose a conntrack flow by DNS ID. The source port and addresses identify the network tuple used by conntrack. A and AAAA questions with different IDs but the same UDP source port are exactly the pattern worth examining for the historical race. If they have different source ports, they may not clash in that way. If one is sent after the other answer rather than concurrently, the specific parallel cold-flow window becomes less likely.

In an active outage, a packet capture can itself be intrusive if it runs without a filter or fills a node filesystem. Keep the filter narrow and the interval short. Capture on the affected node because the relevant Service and conntrack state is node-local. A healthy Pod on another node is a useful control: if the same application image and resolver file behave differently, compare kernel, CNI, local cache, and node load. If the problem follows a particular Pod image across nodes, inspect its resolver implementation and application request pattern. This is the kind of differential diagnosis that turns a vague “DNS is flaky” ticket into a falsifiable mechanism.

## 10. Failure modes after the obvious fix

The cheapest fix for external-name expansion may move a problem rather than end it. If you change every Pod to `ndots:1`, fully qualified external names get tried absolute first. But partially qualified internal names such as `orders.payments` also get tried absolute first. If that name is valid in a corporate upstream zone, the application might connect outside the intended cluster Service. Even if it returns NXDOMAIN, the resolver may now send an avoidable external question before the correct cluster suffix. Review the actual hostname inventory. Teams that use only fully specified Service names have a different risk than teams that rely heavily on namespace-relative names.

A trailing dot has its own boundary. At the DNS wire level, it explicitly denotes the absolute name. At the application level, it is still a character in a URL or configuration string until some library parses and normalizes it. HTTP clients may send it in the authority or `Host` value. TLS clients may compare it against certificate names differently depending on the library. Cookie domain matching and redirect logic can also be affected in a browser-facing application. The correct test is an end-to-end canary against the actual HTTPS endpoint, including certificate validation, redirects, and application routing. If that test is awkward, a scoped resolver option may be the safer workload-level intervention.

NodeLocal DNSCache can make the five-second tail vanish in a cluster where the first-hop Service path caused packet loss. Still, a cache hit can mask an upstream outage until an entry expires. Negative caching can mask newly created names for the negative TTL. A node-local DaemonSet can be healthy on most nodes and broken on one node, creating a location-specific symptom that looks like a bad Pod. Add node identity to DNS metrics and canary tests. During rollout, inspect cache hit ratio, error rate, agent restarts, and CoreDNS upstream traffic. After rollout, keep a way to bypass the cache for a controlled diagnostic; otherwise a cache-layer incident can be difficult to locate.

The historical `single-request-reopen` resolver option is sometimes mentioned in discussions of parallel A and AAAA queries. It changes glibc's socket behavior and can alter the race window, but it is a library-specific workaround. Musl, the pure Go resolver, and other runtimes may ignore it or behave differently. Similar cautions apply to disabling AAAA queries. These are useful diagnostic toggles in a narrow test, not general platform guidance. Preserve IPv6 behavior if the service needs it, and ensure an attempted workaround actually changes packets in the capture.

A faster retry timer makes a latency graph look less dramatic while increasing repeated packets. If the server is overloaded or the network is lossy, more retries can make the underlying problem worse. An application deadline shorter than the DNS resolver's timeout can produce an application error before the resolver retries at all. In that case the five-second kernel symptom may be hidden by a one-second client deadline, and the user sees failures instead of slow successes. Set request budgets with the [SRE retry guidance](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right), but locate the packet problem first.

Kernel and dataplane upgrades have a different failure mode: the fix may change the packet path enough that old probes become misleading. A CNI that implements Services without the same iptables rules will not be visible through the same `iptables -t nat` chain. A new kernel may have different conntrack collision handling. A managed platform may backport fixes without changing the headline kernel version in an obvious way. Consult the distribution's build and release notes, then prove behavior with a canary packet trace. The goal is not to find a magic version number from a 2018 blog. The goal is to verify that the current node no longer loses the relevant packet under the workload that previously reproduced it.

## 11. A worked diagnosis from timing to packet

Consider an illustrative incident with one Pod on an iptables-mode cluster. The application logs a hostname lookup duration of approximately five seconds for an external API. Its `/etc/resolv.conf` contains the four-suffix example and `ndots:5`. We first derive five possible candidate names. A packet capture then shows prompt NXDOMAIN replies for the first four names. That observation eliminates slow search expansion as the *direct* explanation for the five-second gap in this call: the suffix questions consumed packets, but they did not wait for a timeout. The final absolute A and AAAA questions leave the Pod nearly together. The A reply returns, but no AAAA reply appears before the resolver retries AAAA around its configured timeout. This is a hypothetical trace pattern for diagnosis, not a production measurement.

At this point, we still do not know whether the DNS server failed to answer AAAA or the packet was lost. A CoreDNS-side capture is the next test. If it shows the first AAAA question arriving and an answer leaving, inspect the return path to the Pod. If it does not show the first question, inspect the Pod-to-Service path. A time-correlated increase in `insert_failed` on the node would strengthen the historical conntrack-race hypothesis, especially if the parallel questions shared the network tuple. If no counter moves and the packet disappears before NAT, inspect CNI policy or local host filtering. The exact commands and capture points matter more than the shape of the latency graph.

Now consider the first change: calling `api.example.com.` yields one candidate instead of five. This reduces DNS traffic and makes the trace easier to read. If the occasional five-second gap remains on a parallel A/AAAA pair, the workload change did not repair the packet loss. Conversely, if the gap disappears, we have two possibilities: reduced traffic lowered the chance of a timing race, or the problematic packet was tied to a particular search candidate. Repeat enough times to separate those explanations, and compare packet loss and conntrack deltas rather than asserting causality from a small latency sample. The public Weave case explains a possible mechanism; it does not assign a probability to this illustrative workload.

A node-local cache is the second controlled change. If the Pod now sends its question to a local listener and the five-second gap disappears while CoreDNS arrivals fall, the first-hop path has changed as intended. The result is operationally useful even if the exact kernel race variant is not proven. Preserve a follow-up for the kernel and dataplane because a different Service may still use the old path. If the gap persists and the local agent sees the question, inspect its forward path and cache behavior. Do not declare “DNS fixed” solely because one external hostname became fast.

A good incident note ends with the evidence chain, not a noun. It names the hostname form, `ndots` and search list, actual candidate sequence, record types, packet capture points, missing transaction ID, retry interval, counter delta, kernel and dataplane versions, controlled treatment, and rollback. That record allows another engineer to retest after an upgrade. It also prevents the next team from applying a global resolver change to a different five-second symptom that belongs to TCP or an upstream timeout.

## 12. A diagnostic ladder for a live incident

Start with a request that shows the symptom. Save the hostname, process, container image, Pod, node, and timestamp. Record `time_namelookup` if the client exposes it. A DNS warning in an application log is not enough: many libraries report a timeout when the underlying problem is a connection establishment deadline or an upstream retry. A request trace with a five-second gap after the TCP SYN belongs in the TCP investigation, not here.

Next read the affected container's `/etc/resolv.conf`, not the node's file. Count search suffixes. Check `ndots`, `timeout`, `attempts`, and nameserver. Determine whether the requested hostname is absolute, how many dots it contains, and whether the application asks for A, AAAA, or both. Compute the candidate list on paper. If the observed DNS questions match the list and return prompt NXDOMAIN followed by a prompt useful answer, you have amplification but not a five-second loss. If an earlier candidate unexpectedly succeeds, you have a possible name collision. Check the returned address before tuning.

Then capture near the Pod and near the DNS target. If the Pod emits a question and CoreDNS sees it, inspect CoreDNS and upstream behavior. If CoreDNS never sees the question, locate the drop between the Pod and server, including CNI policy, the Service translation path, node firewall rules, and local cache. Compare A and AAAA transaction IDs and UDP source ports. A five-second retry of an unanswered question points toward a transport loss or unanswered resolver path. It does not uniquely identify the historical race.

Now read conntrack counters on the same node during a bounded reproduction. `insert_failed` moving with the loss supports the hypothesis in the Weave analysis. `drop` and table-pressure counters add context. A static high total without a time-correlated delta is weak evidence. Consider counter scope: node-wide totals include unrelated workloads. If the cluster uses NodeLocal DNSCache and the Pod points at a local address, the basic Service DNAT race diagram may no longer describe the first hop. Move the capture and counter interpretation accordingly.

Finally, apply one change at a time. A trailing-dot or `ndots` change is a direct test of search expansion. A node-local DNS rollout is a test of the first-hop path and caching. A kernel or dataplane upgrade is a test of packet-processing behavior. Comparing all three in one deployment can improve symptoms, but it does not teach you which failure was responsible. During an incident, that uncertainty can lead to a brittle rollback or an unrelated regression later.

| Observation | Stronger hypothesis | Next discriminating action | Source |
| --- | --- | --- | --- |
| Several fast NXDOMAIN replies precede the right answer | Search expansion | Compare the same name with a trailing dot and capture candidate names | [glibc resolver manual](https://man7.org/linux/man-pages/man5/resolv.conf.5.html) |
| A or AAAA question leaves the Pod but is absent at DNS endpoint; retry appears after resolver timeout | Packet loss before DNS server | Capture on both sides of DNAT and read node conntrack deltas | [Weave analysis](https://lambda.lt/blog/2018/racy_conntrack.html) |
| Both questions reach CoreDNS, but useful answer arrives late | Server, cache, or upstream path | Inspect CoreDNS logs and forwarder timing for that question name | [Kubernetes DNS documentation](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) |
| SYN appears on Pod interface but not host egress until retransmission | SNAT or host path, not a DNS answer delay | Compare veth, bridge, and egress packet captures | [XING PDF](https://kangwoo.github.io/assets/pdf/kubernetes/A%20reason%20for%20unexplained%20connection%20timeouts%20on%20Kubernetes:Docker.pdf) |

The matrix is a starting hypothesis list. It cannot replace a packet trace. One incident may contain both unnecessary suffix queries and occasional packet loss. Search amplification increases query volume, which can make loss more visible, but their causal mechanisms remain separate. Record the packet and time evidence for each before choosing a platform fix.

## Run it yourself

### Question

Does the Pod's `ndots:5` search policy turn an ordinary external hostname into search-expanded candidate names, and does an explicit absolute name avoid those candidates? This short experiment tests the resolver-side amplification. It does not manufacture a conntrack race, whose appearance depends on kernel and concurrent packet timing.

### Preconditions

Use a Linux Kubernetes cluster with `kubectl` access to create a temporary Pod in a namespace where you are allowed to do so. The commands use `python:3.12-slim` so Python's `socket.getaddrinfo` exercises the image's system resolver. A local cluster from [the series setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) is fine if it has Kubernetes DNS. For observing exact question names, you need permission to run `tcpdump` on the Pod's node or an approved packet-capture equivalent. Packet capture may include other workloads' DNS names; use the Pod IP filter and handle traces under your organization's rules. The name `example.com` is a reserved documentation domain; this experiment does not need a successful address to prove expansion.

The capture command below runs on a Linux node shell, not inside the Python container. Install `tcpdump` ahead of the five-minute experiment. If your cluster uses NodeLocal DNSCache, the nameserver may be local; capture at the Pod-facing interface or at the node-local listener. DNS caching can suppress upstream packets, so the experiment reads packets at the Pod side. If your cluster does not permit node capture, use a permitted ephemeral debug container with `NET_RAW` and equivalent interface visibility.

### Baseline

```bash
kubectl run dns-ndots-lab --restart=Never --image=python:3.12-slim \
  --command -- sleep 3600
kubectl wait --for=condition=Ready pod/dns-ndots-lab --timeout=120s
kubectl exec dns-ndots-lab -- cat /etc/resolv.conf
kubectl get pod dns-ndots-lab -o wide
```

Read: `options ndots:5`, the `search` line, `nameserver`, Pod IP, and scheduled node. If your Pod does not have `ndots:5`, write down the actual value and interpret the candidate order using that value. If there are $S$ suffixes and none answers, the simple one-record-type model predicts up to $S+1$ candidates. The example with four suffixes predicts five. This is a derived expected range of one to $S+1$ distinct candidate names, depending on early success, cache and resolver behavior.

On the scheduled node, in a separate shell, capture only that Pod's DNS packets. Replace `POD_IP` with the IP printed above. Stop the capture with Ctrl-C after both lookups. If an overlay hides the Pod IP at `-i any`, capture on the relevant veth or CNI interface using the node's normal debugging procedure.

```bash
sudo tcpdump -ni any -vv -l 'host POD_IP and (udp port 53 or tcp port 53)'
```

Run the unqualified baseline lookup from your first shell. The Python call asks for IPv4 only to keep the trace easier to count. It catches a DNS error so the experiment still prints timing if external resolution is blocked.

```bash
kubectl exec dns-ndots-lab -- python -c 'import socket,time; n="example.com"; t=time.monotonic(); print("name=",n); exec("try:\n print(socket.getaddrinfo(n, None, socket.AF_INET, socket.SOCK_STREAM))\nexcept socket.gaierror as e:\n print(type(e).__name__, e)"); print("elapsed_s=",round(time.monotonic()-t,3))'
```

Read: in the capture, the DNS question owner names, not merely the query count. With `ndots:5`, `example.com` has one dot and is eligible for the search path first. Expect between one and $S+1$ distinct candidate names on an uncached path, with the final `example.com.` candidate if no earlier candidate succeeds. A and AAAA count is constrained here by `AF_INET`, but library behavior can still vary. Do not expect exactly five candidates unless your effective file has four suffixes and none succeeds early. Timing should normally be far below a resolver timeout when all negative answers are prompt; this is a qualitative expectation, not a claimed benchmark.

### Apply one change

Use an explicit absolute DNS name in the application call. This changes only the input name, leaving the Pod, node, resolver file, nameserver, and requested address family untouched.

```bash
kubectl exec dns-ndots-lab -- python -c 'import socket,time; n="example.com."; t=time.monotonic(); print("name=",n); exec("try:\n print(socket.getaddrinfo(n, None, socket.AF_INET, socket.SOCK_STREAM))\nexcept socket.gaierror as e:\n print(type(e).__name__, e)"); print("elapsed_s=",round(time.monotonic()-t,3))'
```

### Compare

```bash
kubectl exec dns-ndots-lab -- cat /etc/resolv.conf
kubectl get pod dns-ndots-lab -o wide
```

Read: the second capture should show `example.com.` without the search-expanded owner names for that call. Expected distinct candidate count is one, subject to the image resolver and local name-service configuration. The Pod IP and resolver file should be unchanged. Compare question names, not elapsed seconds alone: caches, upstream network variance, and name-service behavior make a single timing comparison noisy. If both calls emitted one candidate, inspect the actual `ndots`, search list, and resolver implementation. If the second still emits suffixes, inspect whether the application's library stripped the terminal dot.

This proves the client-side portion of the main claim. It does not prove that `ndots` caused a five-second hang. To investigate a real five-second stall, repeat with the affected application and capture transaction IDs across the wait. Check `conntrack -S` deltas on the scheduled node and verify whether the first question reached the DNS endpoint. Do not intentionally flush the production conntrack table to create a cold-flow race.

### Reset

```bash
kubectl delete pod dns-ndots-lab --ignore-not-found
```

This deletes only the temporary Pod. Stop the scoped `tcpdump` process with Ctrl-C; the experiment created no DNS configuration or cluster-wide mutation.

## Key takeaways

A `ClusterFirst` Pod can put several search suffixes in front of an external name when `ndots:5` makes that name search-first. Count candidate owner names, question types, and retries separately. Search expansion creates extra questions and sometimes correctness risks; it does not by itself explain a five-second silence.

A five-second retry gap can reflect a lost UDP DNS question in a resolver with a five-second timeout. The historical Weave analysis traced such loss to conntrack confirmation races on the Service DNAT path. XING's related SNAT case dropped TCP SYNs and produced one- or three-second delays. Use the packet's location and type to tell them apart. A current cluster may have different kernel fixes or a different dataplane, so verify the live path.

Start with the smallest controlled change that addresses the observed mechanism. Make an external name absolute or test a narrow Pod `ndots` override for suffix expansion. Use NodeLocal DNSCache when the Service path or repeated DNS traffic justifies a node-level agent. Upgrade and validate the kernel or dataplane when captures and counters identify packet-processing loss. For the broader mental model that ties DNS to connection setup and application latency, return to [the series capstone](/blog/software-development/networking/the-senior-engineers-network-mental-model).

## Further reading

- [Kubernetes: DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/), current Pod search and `dnsConfig` behavior.
- [glibc `resolv.conf(5)` manual](https://man7.org/linux/man-pages/man5/resolv.conf.5.html), `ndots`, `timeout`, and search semantics.
- [Kubernetes: Using NodeLocal DNSCache](https://kubernetes.io/docs/tasks/administer-cluster/nodelocaldns/), first-hop architecture and trade-offs.
- [Martynas Pumputis: Racy conntrack and DNS lookup timeouts](https://lambda.lt/blog/2018/racy_conntrack.html), August 16, 2018, originally published by Weave.
- [Maxime Lagresle: A reason for unexplained connection timeouts on Kubernetes/Docker](https://kangwoo.github.io/assets/pdf/kubernetes/A%20reason%20for%20unexplained%20connection%20timeouts%20on%20Kubernetes:Docker.pdf), February 22, 2018, XING engineering article preserved as a PDF.
